//! Top-level MusicHighlighter orchestrator.
//!
//! Performs decode, mel spec, chunking, positional encoding, model call,
//! saliency expansion, and best-window selection. The model invocation
//! uses tiled inputs (chunks/pos_enc replicated to fill 200 slots) to
//! keep the train-mode BatchNorm stats correct — see `MusicHighlighterModel`
//! comment in iOS for the rationale.

use ferment_engine_audio::derived_cache::DerivedFileCache;
use ferment_engine_audio::streaming_resampler::resample_current;
use std::sync::Arc;

use std::collections::BTreeMap;

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};

use super::config::{
    CHUNK_FRAMES, CHUNK_SECONDS, FEATURE_DIM, MAX_CHUNKS, N_MELS, POS_ENC_DIM, SAMPLE_RATE,
};

use super::pos_enc::positional_encoding;
use super::window_search::best_window;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    pub window: usize,
    pub start: f32,
    pub duration: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightResult {
    pub duration: usize,
    pub saliency: Vec<f32>,
    pub highlights: Vec<Highlight>,
}

#[derive(Debug, Clone)]
pub struct MusicHighlighter {
    pub(crate) cache: Option<Arc<DerivedFileCache>>,
    backend: Backend,
    version: ModelVersion,
}

impl MusicHighlighter {
    pub fn with_cache(mut self, cache: Option<Arc<DerivedFileCache>>) -> Self {
        self.cache = cache;
        self
    }

    pub fn new(backend: Backend) -> Self {
        Self {
            cache: None,
            backend,
            version: ModelVersion::default(),
        }
    }

    pub fn with_version(backend: Backend, version: ModelVersion) -> Self {
        Self {
            backend,
            version,
            cache: None,
        }
    }

    #[tracing::instrument(skip_all, fields(processor = "music_highlighter", samples = frames.len(), windows = ?window_lengths))]
    pub async fn detect(
        &self,
        frames: AudioFrames,
        window_lengths: Vec<usize>,
        progress: ProgressSink,
    ) -> Result<HighlightResult> {
        // 1. Resample to 22.05 kHz
        progress.report(0.05, "loading audio").await;
        let identity = (frames.sample_rate == SAMPLE_RATE)
            .then_some(frames.identity)
            .flatten();
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_current(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };

        let duration = audio.len() / SAMPLE_RATE as usize;
        let n_chunks = duration / CHUNK_SECONDS;
        if n_chunks < 2 {
            progress.report(1.0, "done").await;
            return Ok(HighlightResult {
                duration,
                saliency: Vec::new(),
                highlights: Vec::new(),
            });
        }
        if window_lengths.is_empty() {
            return Err(EngineError::invalid_config(
                "window_lengths must not be empty",
            ));
        }
        if n_chunks > MAX_CHUNKS {
            return Err(EngineError::invalid_config(format!(
                "audio too long: {n_chunks} chunks exceeds max {MAX_CHUNKS}"
            )));
        }

        // 2. Mel spectrogram
        progress.report(0.20, "computing mel").await;
        let frames =
            super::config::MEL_SPEC.perform(&audio, identity.as_ref(), self.cache.as_deref())?;
        let mel: Vec<Vec<f32>> = frames
            .values
            .chunks_exact(frames.bands)
            .map(<[f32]>::to_vec)
            .collect();
        if mel.is_empty() {
            return Err(EngineError::AudioTooShort {
                samples: audio.len(),
                min_required: super::config::FFT_SIZE,
            });
        }

        // 3. Chunk into 3-second segments
        let chunk_length_frames = mel.len() / n_chunks;
        let frames_to_copy = chunk_length_frames.min(CHUNK_FRAMES);
        let mut chunked = vec![0.0f32; n_chunks * CHUNK_FRAMES * N_MELS];
        for j in 0..n_chunks {
            let start_frame = j * chunk_length_frames;
            for t in 0..frames_to_copy {
                let dst = (j * CHUNK_FRAMES + t) * N_MELS;
                chunked[dst..dst + N_MELS].copy_from_slice(&mel[start_frame + t]);
            }
        }

        // 4. Positional encoding (real chunks only — model tiles on its side)
        let pe = positional_encoding(n_chunks, POS_ENC_DIM);

        // 5. Model inference
        progress.report(0.50, "running model").await;
        let scores = self.run_model(&chunked, &pe, n_chunks).await?;
        if scores.len() != n_chunks {
            return Err(EngineError::InvalidOutput);
        }

        // 6. Per-second saliency
        let mut saliency = Vec::with_capacity(duration);
        for &score in &scores {
            for _ in 0..CHUNK_SECONDS {
                saliency.push(score);
            }
        }
        let remainder = duration.saturating_sub(n_chunks * CHUNK_SECONDS);
        saliency.extend(std::iter::repeat_n(0.0, remainder));
        let max = saliency.iter().copied().fold(0.0f32, f32::max);
        if max > 0.0 && max.is_finite() {
            for v in &mut saliency {
                *v /= max;
            }
        }

        // 7. Best windows
        let highlights = window_lengths
            .iter()
            .map(|&wl| {
                let clamped = wl.min(duration);
                let (start, end) = best_window(&saliency, clamped as isize);
                Highlight {
                    window: wl,
                    start,
                    duration: end - start,
                }
            })
            .collect();

        progress.report(1.0, "done").await;

        Ok(HighlightResult {
            duration,
            saliency,
            highlights,
        })
    }

    /// Invoke the model with chunks + pos_enc tiled to fill `MAX_CHUNKS` slots.
    /// Returns softmax over the first `n_chunks` logits.
    pub(super) async fn run_model(
        &self,
        chunks: &[f32],
        pos_enc: &[f32],
        n_chunks: usize,
    ) -> Result<Vec<f32>> {
        let chunk_stride = CHUNK_FRAMES * N_MELS;

        let mut tiled_chunks = vec![0.0f32; MAX_CHUNKS * chunk_stride];
        for slot in 0..MAX_CHUNKS {
            let src = slot % n_chunks;
            tiled_chunks[slot * chunk_stride..(slot + 1) * chunk_stride]
                .copy_from_slice(&chunks[src * chunk_stride..(src + 1) * chunk_stride]);
        }

        let mut tiled_pe = vec![0.0f32; MAX_CHUNKS * FEATURE_DIM];
        for slot in 0..MAX_CHUNKS {
            let src = slot % n_chunks;
            tiled_pe[slot * FEATURE_DIM..(slot + 1) * FEATURE_DIM]
                .copy_from_slice(&pos_enc[src * FEATURE_DIM..(src + 1) * FEATURE_DIM]);
        }

        let mut inputs: TensorMap = BTreeMap::new();
        inputs.insert(
            "chunks".into(),
            Tensor::from_f32(
                vec![MAX_CHUNKS as i64, CHUNK_FRAMES as i64, N_MELS as i64],
                tiled_chunks,
            )?,
        );
        inputs.insert(
            "pos_enc".into(),
            Tensor::from_f32(vec![1, MAX_CHUNKS as i64, FEATURE_DIM as i64], tiled_pe)?,
        );

        let outputs = self
            .backend
            .infer(ModelId::MusicHighlighter, &self.version, inputs)
            .await?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: "music_highlighter".into(),
                key: "logits".into(),
            })?
            .to_f32()?;
        if logits.len() < n_chunks {
            return Err(EngineError::InvalidOutput);
        }
        Ok(softmax(&logits[..n_chunks]))
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    let inv = 1.0 / sum;
    for v in &mut exps {
        *v *= inv;
    }
    exps
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let total: f32 = s.iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_handles_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_constant_yields_uniform() {
        let s = softmax(&[1.0, 1.0, 1.0, 1.0]);
        for v in s {
            assert_abs_diff_eq!(v, 0.25, epsilon = 1e-6);
        }
    }
}
