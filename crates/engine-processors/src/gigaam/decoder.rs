//! GigaAM-v3-e2e-rnnt decoder — textbook RNN-T greedy search.
//!
//! A direct port of upstream's `RNNTGreedyDecoding._greedy_decode` (iOS
//! `GigaAMDecoder.swift`): walk the encoder frames one at a time, and at each
//! frame keep asking the joint for a symbol until it answers blank or
//! [`MAX_SYMBOLS_PER_STEP`] symbols have come out. There is no duration head —
//! that is a TDT (Parakeet) feature, and the frame-skipping inner loop it needs
//! has no counterpart here.
//!
//! The one departure from the reference is that the prediction network's
//! projection is cached instead of recomputed on every inner iteration. That is
//! exact, not an approximation: the projection is a pure function of (last
//! emitted token, LSTM state), and a blank updates neither. The reference
//! re-runs the LSTM and gets the same tensor back; we run it once per emitted
//! token instead of once per joint call.

use ferment_engine_core::{EngineError, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

use super::config::{
    BLANK_TOKEN_ID, DECODER_HIDDEN, DECODER_LAYERS, ENCODER_HIDDEN, MAX_SYMBOLS_PER_STEP,
    MAX_TOKENS_PER_CHUNK,
};

/// Raw output from a single chunk's RNN-T decode.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkResult {
    pub tokens: Vec<i32>,
    /// Encoder frame indices, file-global.
    pub timestamps: Vec<i32>,
    pub confidences: Vec<f32>,
}

/// Encoder output for one chunk: row-major `[T, hidden]`.
pub struct EncoderFrames {
    pub frames: Vec<Vec<f32>>,
    /// Frames backed by real audio; the zero-padded tail of a short final
    /// chunk is excluded by the caller.
    pub valid_frames: usize,
}

/// LSTM hidden / cell state for the RNN-T prediction network. One layer, 320
/// units — `[1, 1, 320]` for both tensors, matching the exported `h_in` /
/// `c_in` shapes.
#[derive(Debug, Clone)]
pub struct DecoderState {
    pub hidden: Vec<f32>,
    pub cell: Vec<f32>,
}

impl DecoderState {
    #[must_use]
    pub fn fresh() -> Self {
        Self {
            hidden: vec![0.0; DECODER_LAYERS * DECODER_HIDDEN],
            cell: vec![0.0; DECODER_LAYERS * DECODER_HIDDEN],
        }
    }
}

#[derive(Clone)]
pub struct DecoderHandles {
    pub backend: Backend,
    pub version: ModelVersion,
}

/// Decode encoder output into tokens with frame-level timestamps.
///
/// `context_frames` are the leading frames produced from the previous chunk's
/// audio, prepended for mel stability. Decoding starts past them and timestamps
/// are shifted back by the same amount, so a chunk's frame numbering stays
/// aligned to the file. `global_frame_offset` is this chunk's content start, in
/// encoder frames.
///
/// The LSTM state is created here rather than passed in. Chunks are decoded
/// from a zeroed state by design (the seam merge recovers the context), and
/// owning it makes "primed from zero" true by construction instead of a
/// contract a future caller could quietly break.
pub async fn decode(
    handles: &DecoderHandles,
    encoder: &EncoderFrames,
    context_frames: usize,
    global_frame_offset: i32,
) -> Result<ChunkResult> {
    let frame_count = encoder.valid_frames.min(encoder.frames.len());
    if frame_count <= context_frames {
        return Ok(ChunkResult::default());
    }

    let mut state = DecoderState::fresh();
    // Prime the LSTM with blank-as-SOS. The state arrives zeroed, and the
    // blank embedding is `padding_idx` upstream, so this reproduces the
    // reference's `predict(x: nil, ...)` first step exactly.
    let primed = run_decoder(handles, BLANK_TOKEN_ID, &state).await?;
    state.hidden = primed.hidden;
    state.cell = primed.cell;
    let mut projection = primed.projection;

    let mut result = ChunkResult::default();
    let mut tokens_this_chunk = 0usize;

    for frame in context_frames..frame_count {
        for _ in 0..MAX_SYMBOLS_PER_STEP {
            let decision = run_joint(handles, &encoder.frames[frame], &projection).await?;
            if decision.token == BLANK_TOKEN_ID {
                break;
            }

            tokens_this_chunk += 1;
            if tokens_this_chunk > MAX_TOKENS_PER_CHUNK {
                return Ok(result);
            }

            result.tokens.push(decision.token);
            result.timestamps.push(
                i32::try_from(frame - context_frames).unwrap_or(i32::MAX) + global_frame_offset,
            );
            result.confidences.push(decision.confidence);

            let step = run_decoder(handles, decision.token, &state).await?;
            state.hidden = step.hidden;
            state.cell = step.cell;
            projection = step.projection;
        }
    }

    Ok(result)
}

struct DecoderStep {
    projection: Vec<f32>,
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

async fn run_decoder(
    handles: &DecoderHandles,
    token: i32,
    state: &DecoderState,
) -> Result<DecoderStep> {
    let mut inputs = TensorMap::new();
    inputs.insert("x".into(), Tensor::from_i32(vec![1, 1], vec![token])?);
    inputs.insert(
        "h_in".into(),
        Tensor::from_f32(
            vec![DECODER_LAYERS as i64, 1, DECODER_HIDDEN as i64],
            state.hidden.clone(),
        )?,
    );
    inputs.insert(
        "c_in".into(),
        Tensor::from_f32(
            vec![DECODER_LAYERS as i64, 1, DECODER_HIDDEN as i64],
            state.cell.clone(),
        )?,
    );

    let outputs = handles
        .backend
        .infer(ModelId::GigaAmDecoder, &handles.version, inputs)
        .await?;
    let take = |key: &str| -> Result<Vec<f32>> {
        outputs
            .get(key)
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::GigaAmDecoder.name().into(),
                key: key.into(),
            })?
            .to_f32()
    };
    Ok(DecoderStep {
        projection: take("dec")?,
        hidden: take("h_out")?,
        cell: take("c_out")?,
    })
}

#[derive(Clone, Copy)]
struct JointDecision {
    token: i32,
    confidence: f32,
}

async fn run_joint(
    handles: &DecoderHandles,
    encoder_step: &[f32],
    projection: &[f32],
) -> Result<JointDecision> {
    let mut inputs = TensorMap::new();
    inputs.insert(
        "enc".into(),
        Tensor::from_f32(vec![1, ENCODER_HIDDEN as i64, 1], encoder_step.to_vec())?,
    );
    inputs.insert(
        "dec".into(),
        Tensor::from_f32(vec![1, DECODER_HIDDEN as i64, 1], projection.to_vec())?,
    );

    let outputs = handles
        .backend
        .infer(ModelId::GigaAmJoint, &handles.version, inputs)
        .await?;
    let missing = |key: &'static str| EngineError::ModelOutputMissing {
        model: ModelId::GigaAmJoint.name().into(),
        key: key.into(),
    };
    // `to_i64` accepts INT32 and INT64 alike; the export pins INT32 but the
    // ONNX exporter is free to widen a scalar, and a widened argmax is still
    // the same token id.
    let token = outputs
        .get("token_id")
        .ok_or_else(|| missing("token_id"))?
        .to_i64()?
        .first()
        .copied()
        .ok_or_else(|| missing("token_id"))? as i32;
    let probability = outputs
        .get("token_prob")
        .ok_or_else(|| missing("token_prob"))?
        .to_f32()?
        .first()
        .copied()
        .ok_or_else(|| missing("token_prob"))?;

    Ok(JointDecision {
        token,
        confidence: clamp_probability(probability),
    })
}

fn clamp_probability(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_is_zeroed_and_one_layer_wide() {
        let state = DecoderState::fresh();
        assert_eq!(state.hidden.len(), DECODER_HIDDEN);
        assert_eq!(state.cell.len(), DECODER_HIDDEN);
        assert!(state.hidden.iter().all(|value| *value == 0.0));
        assert!(state.cell.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn probabilities_are_clamped_and_nan_reads_as_zero() {
        assert_eq!(clamp_probability(f32::NAN), 0.0);
        assert_eq!(clamp_probability(-0.5), 0.0);
        assert_eq!(clamp_probability(1.5), 1.0);
        assert_eq!(clamp_probability(0.25), 0.25);
    }
}
