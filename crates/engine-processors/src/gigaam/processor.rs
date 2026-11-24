//! Top-level orchestrator: split the audio into frame-aligned 15-second
//! chunks, run preprocessor → encoder → RNN-T decode on each, stitch the
//! per-chunk token streams with the shared seam merge, and build words.
//!
//! Ported from iOS `GigaAMChunkProcessor.swift` + `GigaAMTranscriber.swift`.
//!
//! **Why streaming reads?** Only the current chunk's window is materialised, so
//! peak RAM is flat in the file's duration.
//!
//! **Deviation from iOS, deliberate:** upstream runs three chunks concurrently
//! against shared Core ML model handles. Here the chunks run in order. The
//! per-chunk work is identical either way — chunks are decoded from a zeroed
//! LSTM state by design and the merge resolves the seam at token level — so the
//! concurrency buys wall time, not output. The ORT backend serialises inference
//! per session anyway, which is where that wall time would have come from.

use crate::audio_source::{ArrayAudioSampleSource, AudioSampleSource};
use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{AudioEventSegment, EngineError, ProgressSink, Result, TranscribedWord};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use std::ops::Range;

use super::config::{
    CHUNK_SAMPLES, ENCODER_HIDDEN, MAX_CHUNK_SAMPLES, MAX_ENCODER_FRAMES, MAX_MEL_FRAMES,
    MEL_CONTEXT_SAMPLES, N_MELS, OVERLAP_FRAMES, SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME,
    SECONDS_PER_ENCODER_FRAME, SEGMENT_PAD_SECONDS, STRIDE_SAMPLES, encoder_frames, encoder_mask,
    mel_frames,
};
use super::decoder::{ChunkResult, DecoderHandles, EncoderFrames, decode};
use super::postprocess::{build_words, merge_chunks};
use super::vocab;
use crate::parakeet::speech_segment_coalescer::coalesce;

/// The only language this model produces. GigaAM-v3-e2e-rnnt is a monolingual
/// Russian acoustic model whose `e2e` head emits punctuated, case-normalised
/// text directly, so nothing downstream has to restore either.
pub const LANGUAGE: &str = "ru";

/// On-device Russian speech transcription using GigaAM-v3-e2e-rnnt.
#[derive(Debug, Clone)]
pub struct GigaAm {
    backend: Backend,
    version: ModelVersion,
}

impl GigaAm {
    /// The ENGINE's knowledge of its own model, for the transcription router.
    pub const SUPPORTED_LANGUAGE_CODES: [&'static str; 1] = ["ru"];

    /// The graphs this transcriber needs on disk. The SentencePiece vocabulary
    /// is compiled into the crate, so it is not a download.
    pub const MODEL_FILES: [ModelId; 4] = [
        ModelId::GigaAmPreprocessor,
        ModelId::GigaAmEncoder,
        ModelId::GigaAmDecoder,
        ModelId::GigaAmJoint,
    ];

    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    /// Transcribe `frames`, restricted to `speech_segments` when supplied.
    ///
    /// Segments are padded and merged before slicing: raw AED boundaries cut
    /// tight against words and split one utterance on a short breath, and
    /// decoding those in isolation clips the seam word and restarts the
    /// sentence. Genuine multi-second gaps stay split, so non-speech stays out
    /// of the decoder.
    #[tracing::instrument(skip_all, fields(processor = "gigaam", samples = frames.len()))]
    pub async fn transcribe(
        &self,
        frames: AudioFrames,
        speech_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<Vec<TranscribedWord>> {
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        if speech_segments.is_some_and(<[AudioEventSegment]>::is_empty) {
            progress.report(1.0, "No speech detected").await;
            return Ok(Vec::new());
        }
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        self.transcribe_source(
            &ArrayAudioSampleSource::new(audio),
            speech_segments,
            progress,
        )
        .await
    }

    /// Transcribe mono 16 kHz samples without materialising the source's full
    /// duration. Exactly the same chunk plan, decoder and token seam merge as
    /// the frame door; each read is bounded by the pinned encoder window.
    pub async fn transcribe_source(
        &self,
        source: &dyn AudioSampleSource,
        speech_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<Vec<TranscribedWord>> {
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        if speech_segments.is_some_and(<[AudioEventSegment]>::is_empty) {
            progress.report(1.0, "No speech detected").await;
            return Ok(Vec::new());
        }

        progress.report(0.02, "loading audio").await;
        if source.sample_count() == 0 {
            return Err(EngineError::NoAudioData);
        }

        let words = match speech_segments {
            None => {
                self.transcribe_span(source, 0..source.sample_count(), 0.0, 0.85, &progress)
                    .await?
            }
            Some(segments) => {
                let duration = source.sample_count() as f64 / f64::from(SAMPLE_RATE);
                let runs = coalesce(segments, SEGMENT_PAD_SECONDS, duration);
                let budget = 0.85 / runs.len().max(1) as f64;
                let mut words = Vec::new();
                for run in runs {
                    let start = seconds_to_sample(run.start_time).min(source.sample_count());
                    let end = seconds_to_sample(run.end_time()).min(source.sample_count());
                    if end <= start {
                        continue;
                    }
                    words.extend(
                        self.transcribe_span(source, start..end, run.start_time, budget, &progress)
                            .await?,
                    );
                }
                words
            }
        };

        progress.report(1.0, "Done").await;
        Ok(words)
    }

    /// Chunked decode over one contiguous span, returning absolute-time words.
    async fn transcribe_span(
        &self,
        source: &dyn AudioSampleSource,
        range: Range<usize>,
        time_offset: f64,
        progress_budget: f64,
        progress: &ProgressSink,
    ) -> Result<Vec<TranscribedWord>> {
        if range.start > range.end || range.end > source.sample_count() {
            return Err(EngineError::decode("GigaAM source range is out of bounds"));
        }
        let plans = chunk_plans(range.len());
        if plans.is_empty() {
            return Ok(Vec::new());
        }
        let handles = DecoderHandles {
            backend: self.backend.clone(),
            version: self.version.clone(),
        };
        let weight = progress_budget / plans.len() as f64;

        let mut merged = ChunkResult::default();
        for (index, plan) in plans.iter().enumerate() {
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            progress
                .report(
                    0.05 + weight * (index + 1) as f64,
                    format!("transcribing ({}/{})", index + 1, plans.len()),
                )
                .await;

            let count = plan.chunk_end - plan.context_start;
            if count > MAX_CHUNK_SAMPLES {
                return Err(EngineError::decode(
                    "GigaAM read exceeds one encoder window",
                ));
            }
            let mut samples = vec![0.; count];
            if source.copy_samples(&mut samples, range.start + plan.context_start, count)? != count
            {
                return Err(EngineError::decode(
                    "GigaAM source ended before its chunk boundary",
                ));
            }
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            let chunk = self.decode_chunk(&handles, &samples, plan).await?;
            merged = merge_into(merged, chunk);
        }

        let words = build_words(
            &merged.tokens,
            &merged.timestamps,
            &merged.confidences,
            vocab::pieces()?,
        );
        if time_offset == 0.0 {
            return Ok(words);
        }
        Ok(words
            .into_iter()
            .map(|word| TranscribedWord {
                start_time: word.start_time + time_offset,
                ..word
            })
            .collect())
    }

    async fn decode_chunk(
        &self,
        handles: &DecoderHandles,
        samples: &[f32],
        plan: &ChunkPlan,
    ) -> Result<ChunkResult> {
        // Mel frames backed by real audio. The preprocessor always sees a full
        // 240k buffer, so without the mask the encoder would read the
        // zero-padded tail of a short final chunk as speech.
        let frames = mel_frames(samples.len());
        if frames <= 0 {
            return Ok(ChunkResult::default());
        }
        let frames = frames as usize;

        let mel = self.run_preprocessor(samples).await?;
        let encoded = self.run_encoder(mel, frames).await?;
        decode(
            handles,
            &EncoderFrames {
                frames: encoded,
                valid_frames: encoder_frames(frames),
            },
            plan.context_samples / SAMPLES_PER_ENCODER_FRAME,
            plan.global_frame_offset(),
        )
        .await
    }

    /// Preprocessor: `audio[1, 240000]` → `mel[1, 64, 1499]`. The chunk is
    /// copied into the fixed window with a zero-padded tail.
    async fn run_preprocessor(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut window = vec![0.0f32; MAX_CHUNK_SAMPLES];
        let count = samples.len().min(MAX_CHUNK_SAMPLES);
        window[..count].copy_from_slice(&samples[..count]);

        let mut inputs = TensorMap::new();
        inputs.insert(
            "audio".into(),
            Tensor::from_f32(vec![1, MAX_CHUNK_SAMPLES as i64], window)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::GigaAmPreprocessor, &self.version, inputs)
            .await?;
        outputs
            .get("mel")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::GigaAmPreprocessor.name().into(),
                key: "mel".into(),
            })?
            .to_f32()
    }

    /// Encoder: `mel[1, 64, 1499]` + `mask[1, 1, 375]` → `encoded[1, 768, 375]`
    /// (`ferment-models-export`, `models/gigaam/model.json`). The mask is 1 on
    /// the encoder frames of the chunk's `mel_frames` real mel frames — the
    /// host counts them; the graph is told nothing else. Returns the frames
    /// laid out as `[T, hidden]` for the decoder's per-frame loop.
    async fn run_encoder(&self, mel: Vec<f32>, mel_frames: usize) -> Result<Vec<Vec<f32>>> {
        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(vec![1, N_MELS as i64, MAX_MEL_FRAMES as i64], mel)?,
        );
        inputs.insert(
            "mask".into(),
            Tensor::from_f32(
                vec![1, 1, MAX_ENCODER_FRAMES as i64],
                encoder_mask(mel_frames),
            )?,
        );

        let outputs = self
            .backend
            .infer(ModelId::GigaAmEncoder, &self.version, inputs)
            .await?;
        let encoded = outputs
            .get("encoded")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::GigaAmEncoder.name().into(),
                key: "encoded".into(),
            })?;
        encoded_rows(encoded.shape.as_slice(), &encoded.to_f32()?)
    }
}

/// Transpose `[1, hidden, T]` into `T` rows of `hidden`.
///
/// The exported graph pins this layout. The Parakeet original probes for the
/// transposed layout too; that probe cannot fire here, and keeping it would
/// only convert a mis-export into wrong numbers instead of a stop.
fn encoded_rows(shape: &[i64], values: &[f32]) -> Result<Vec<Vec<f32>>> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_HIDDEN as i64 {
        return Err(EngineError::invalid_config(format!(
            "gigaam encoder output must be [1, {ENCODER_HIDDEN}, T], got {shape:?}"
        )));
    }
    let time = shape[2].max(0) as usize;
    if values.len() != ENCODER_HIDDEN * time {
        return Err(EngineError::invalid_config(format!(
            "gigaam encoder output has {} values, expected {}",
            values.len(),
            ENCODER_HIDDEN * time
        )));
    }
    Ok((0..time)
        .map(|frame| {
            (0..ENCODER_HIDDEN)
                .map(|hidden| values[hidden * time + frame])
                .collect()
        })
        .collect())
}

fn merge_into(left: ChunkResult, right: ChunkResult) -> ChunkResult {
    if left.tokens.is_empty() || right.tokens.is_empty() {
        let mut out = left;
        out.tokens.extend(right.tokens);
        out.timestamps.extend(right.timestamps);
        out.confidences.extend(right.confidences);
        return out;
    }
    let merged = merge_chunks(
        &left.tokens,
        &left.timestamps,
        &left.confidences,
        &right.tokens,
        &right.timestamps,
        &right.confidences,
        OVERLAP_FRAMES as f64 * SECONDS_PER_ENCODER_FRAME,
        SECONDS_PER_ENCODER_FRAME,
    );
    ChunkResult {
        tokens: merged.tokens,
        timestamps: merged.timestamps,
        confidences: merged.confidences,
    }
}

fn seconds_to_sample(seconds: f64) -> usize {
    (seconds.max(0.0) * f64::from(SAMPLE_RATE)) as usize
}

/// One chunk's read window. `context_start..chunk_end` is what the
/// preprocessor sees; `chunk_start` is where its content begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkPlan {
    pub(super) chunk_start: usize,
    pub(super) context_start: usize,
    pub(super) chunk_end: usize,
    /// 0 for the first chunk.
    pub(super) context_samples: usize,
}

impl ChunkPlan {
    fn global_frame_offset(self) -> i32 {
        i32::try_from(self.chunk_start / SAMPLES_PER_ENCODER_FRAME).unwrap_or(i32::MAX)
    }
}

pub(super) fn chunk_plans(total_samples: usize) -> Vec<ChunkPlan> {
    let mut plans = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while start < total_samples {
        let candidate_end = start + CHUNK_SAMPLES;
        let is_last = candidate_end >= total_samples;
        let chunk_end = if is_last {
            total_samples
        } else {
            candidate_end
        };
        if chunk_end <= start {
            break;
        }
        let context = if index > 0 { MEL_CONTEXT_SAMPLES } else { 0 };
        plans.push(ChunkPlan {
            chunk_start: start,
            context_start: start.saturating_sub(context),
            chunk_end,
            context_samples: context,
        });
        if is_last {
            break;
        }
        start += STRIDE_SAMPLES;
        index += 1;
    }
    plans
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use ferment_engine_infer::InferenceBackend;

    use super::super::config::BLANK_TOKEN_ID;
    use super::*;

    #[derive(Debug)]
    struct NeverBackend;

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl InferenceBackend for NeverBackend {
        async fn infer(
            &self,
            _model: ModelId,
            _version: &ModelVersion,
            _inputs: TensorMap,
        ) -> Result<TensorMap> {
            panic!("empty speech segments must not invoke inference")
        }
    }

    /// The encoder door as the export declares it — `mel` and `mask`, nothing
    /// else — with each encoded frame's every value its own index, and a
    /// joint that answers blank and records the frame it was asked about.
    #[derive(Debug, Default)]
    struct DoorSpy {
        masks: std::sync::Mutex<Vec<Vec<f32>>>,
        jointed: std::sync::Mutex<Vec<f32>>,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl InferenceBackend for DoorSpy {
        async fn infer(
            &self,
            model: ModelId,
            _version: &ModelVersion,
            inputs: TensorMap,
        ) -> Result<TensorMap> {
            let mut outputs = TensorMap::new();
            match model {
                ModelId::GigaAmPreprocessor => {
                    outputs.insert(
                        "mel".into(),
                        Tensor::from_f32(
                            vec![1, N_MELS as i64, MAX_MEL_FRAMES as i64],
                            vec![0.0; N_MELS * MAX_MEL_FRAMES],
                        )?,
                    );
                }
                ModelId::GigaAmEncoder => {
                    assert_eq!(inputs.keys().collect::<Vec<_>>(), ["mask", "mel"]);
                    assert_eq!(
                        inputs["mel"].shape,
                        [1, N_MELS as i64, MAX_MEL_FRAMES as i64]
                    );
                    assert_eq!(inputs["mask"].shape, [1, 1, MAX_ENCODER_FRAMES as i64]);
                    self.masks.lock().unwrap().push(inputs["mask"].to_f32()?);
                    outputs.insert(
                        "encoded".into(),
                        Tensor::from_f32(
                            vec![1, ENCODER_HIDDEN as i64, MAX_ENCODER_FRAMES as i64],
                            (0..ENCODER_HIDDEN)
                                .flat_map(|_| (0..MAX_ENCODER_FRAMES).map(|frame| frame as f32))
                                .collect(),
                        )?,
                    );
                }
                ModelId::GigaAmDecoder => {
                    for key in ["dec", "h_out", "c_out"] {
                        outputs.insert(
                            key.into(),
                            Tensor::from_f32(vec![1, 1, 320], vec![0.0; 320])?,
                        );
                    }
                }
                ModelId::GigaAmJoint => {
                    self.jointed
                        .lock()
                        .unwrap()
                        .push(inputs["enc"].to_f32()?[0]);
                    outputs.insert(
                        "token_id".into(),
                        Tensor::from_i32(vec![1, 1, 1], vec![BLANK_TOKEN_ID])?,
                    );
                    outputs.insert(
                        "token_prob".into(),
                        Tensor::from_f32(vec![1, 1, 1], vec![1.0])?,
                    );
                }
                other => panic!("unexpected model {other}"),
            }
            Ok(outputs)
        }
    }

    /// The mask is load-bearing and the ONNX oracle proves the *graph* honours
    /// it — this proves the *port* builds it from the chunk's real mel frames
    /// rather than the padded window's 1499, and decodes exactly the frames it
    /// masked in: 3 s is 299 mel frames, 75 encoder frames of the door's 375.
    #[test]
    fn the_host_masks_and_decodes_the_chunks_real_frames_not_the_padded_window() {
        let spy = Arc::new(DoorSpy::default());
        let transcriber = GigaAm::new(spy.clone());
        let samples = SAMPLE_RATE as usize * 3;
        futures_executor::block_on(transcriber.transcribe(
            AudioFrames::new(vec![0.1; samples], SAMPLE_RATE),
            None,
            ProgressSink::noop(),
        ))
        .expect("transcribe");

        assert_eq!((mel_frames(samples), encoder_frames(299)), (299, 75));
        let mut expected = vec![1.0; 75];
        expected.resize(MAX_ENCODER_FRAMES, 0.0);
        assert_eq!(*spy.masks.lock().unwrap(), [expected]);
        assert_eq!(
            *spy.jointed.lock().unwrap(),
            (0..75).map(|frame| frame as f32).collect::<Vec<_>>(),
            "one blank per real frame, and not one frame of padding"
        );
    }

    /// Kill: converting audio before the empty-selection guard returns a
    /// sample-rate error instead of the caller's explicit no-speech result.
    #[test]
    fn empty_speech_segments_short_circuit_before_the_model_loads() {
        let transcriber = GigaAm::new(Arc::new(NeverBackend));
        let words = futures_executor::block_on(transcriber.transcribe(
            AudioFrames::new(vec![0.1; 16_000], 0),
            Some(&[]),
            ProgressSink::noop(),
        ))
        .expect("no speech is a result, not a failure");
        assert!(words.is_empty());
    }

    /// Kill: a cancelled frame request resamples/validates audio before
    /// returning Cancelled, unlike the disk/source door.
    #[test]
    fn cancelled_frames_do_not_resample_audio_even_with_an_empty_selection() {
        let transcriber = GigaAm::new(Arc::new(NeverBackend));
        for selection in [None, Some(&[][..])] {
            let progress = ProgressSink::noop();
            progress.cancel();
            let result = futures_executor::block_on(transcriber.transcribe(
                AudioFrames::new(vec![0.1; 16_000], 0),
                selection,
                progress,
            ));
            assert!(matches!(result, Err(EngineError::Cancelled)), "{result:?}");
        }
    }

    #[test]
    fn short_audio_is_one_chunk_with_no_context() {
        let plans = chunk_plans(16_000);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].context_samples, 0);
        assert_eq!(plans[0].context_start, 0);
        assert_eq!(plans[0].chunk_end, 16_000);
        assert_eq!(plans[0].global_frame_offset(), 0);
    }

    #[test]
    fn empty_audio_plans_nothing() {
        assert!(chunk_plans(0).is_empty());
    }

    #[test]
    fn long_audio_overlaps_and_stays_frame_aligned() {
        let plans = chunk_plans(SAMPLE_RATE as usize * 60);
        assert!(plans.len() >= 3, "60 s at a 12.92 s stride");
        for pair in plans.windows(2) {
            assert_eq!(pair[1].chunk_start - pair[0].chunk_start, STRIDE_SAMPLES);
            assert!(
                pair[1].chunk_start < pair[0].chunk_end,
                "consecutive chunks must overlap or the seam merge has nothing to match"
            );
            assert!(pair[1].global_frame_offset() > pair[0].global_frame_offset());
        }
        for plan in &plans {
            assert_eq!(plan.chunk_start % SAMPLES_PER_ENCODER_FRAME, 0);
            assert!(
                plan.chunk_end - plan.context_start <= MAX_CHUNK_SAMPLES,
                "a chunk must never exceed the exported model's fixed input"
            );
        }
        assert_eq!(
            plans.last().unwrap().chunk_end,
            SAMPLE_RATE as usize * 60,
            "the last chunk closes on the end of the audio"
        );
    }

    #[test]
    fn every_chunk_after_the_first_carries_mel_context() {
        let plans = chunk_plans(SAMPLE_RATE as usize * 40);
        assert_eq!(plans[0].context_samples, 0);
        for plan in &plans[1..] {
            assert_eq!(plan.context_samples, MEL_CONTEXT_SAMPLES);
            assert_eq!(plan.context_start, plan.chunk_start - MEL_CONTEXT_SAMPLES);
        }
    }

    /// The graph emits `[1, hidden, T]` hidden-major; the decoder walks
    /// frames. Value `hidden * 1000 + frame` makes a transposed read obvious.
    #[test]
    fn the_encoder_transpose_reads_hidden_major_input() {
        const TIME: usize = 3;
        let values = (0..ENCODER_HIDDEN)
            .flat_map(|hidden| (0..TIME).map(move |frame| (hidden * 1000 + frame) as f32))
            .collect::<Vec<_>>();
        let frames =
            super::encoded_rows(&[1, ENCODER_HIDDEN as i64, TIME as i64], &values).expect("t");
        assert_eq!(frames.len(), TIME);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.len(), ENCODER_HIDDEN);
            assert_eq!(frame[0], index as f32);
            assert_eq!(
                frame[ENCODER_HIDDEN - 1],
                ((ENCODER_HIDDEN - 1) * 1000 + index) as f32
            );
        }
    }

    #[test]
    fn a_mis_exported_encoder_layout_is_a_stop_not_wrong_numbers() {
        let error = super::encoded_rows(&[1, 3, 2], &[0.0; 6]).expect_err("hidden mismatch");
        assert!(
            format!("{error}").contains("must be [1, 768, T]"),
            "{error}"
        );
        assert!(
            super::encoded_rows(&[1, ENCODER_HIDDEN as i64], &[0.0; ENCODER_HIDDEN]).is_err(),
            "a rank-2 output is a mis-export, not a squeezed batch"
        );
        assert!(
            super::encoded_rows(&[1, ENCODER_HIDDEN as i64, 3], &[0.0; 8]).is_err(),
            "a shape the payload cannot fill is a stop"
        );
    }

    #[test]
    fn merging_an_empty_chunk_concatenates_instead_of_matching() {
        let left = ChunkResult {
            tokens: vec![1, 2],
            timestamps: vec![0, 1],
            confidences: vec![0.5, 0.5],
        };
        assert_eq!(merge_into(left.clone(), ChunkResult::default()), left);
        assert_eq!(merge_into(ChunkResult::default(), left.clone()), left);
    }
}
