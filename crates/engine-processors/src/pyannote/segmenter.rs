//! Who spoke, when, and what they sound like — for one element.
//!
//! pyannote/segmentation-3.0 (PyanNet, ~1.5M, MIT) reads 10 s at a time and
//! says which of up to three LOCAL speakers is talking in each 16.9 ms frame.
//! ReDimNet then fingerprints each of them, and the fingerprints are what joins
//! one chunk's "speaker 1" to the next chunk's.
//!
//! Why the two models are one pass rather than two. A segmenter's labels are
//! meaningless outside their own 10 s window, so an element's replies do not
//! EXIST until the embedder has stitched the chunks. Storing the segmenter's
//! output on its own would be storing a number nobody can read.
//!
//! Why it replaces scoring speech spans one window at a time. Measured upstream
//! 2026-08-24 on a cut solo pitch: two 2 s windows of ONE person scored cosine
//! 0.496 against each other — under the 0.60 that says "same voice", so one
//! person became two. That is not a threshold that was set wrong and not a
//! checkpoint that can be swapped: ECAPA-TDNN, eight times the parameters,
//! scored the same two windows at 0.297. 2 s is below the working floor of ANY
//! embedder. The cure the field uses is the one here — segment the replies with
//! a model, then pool a fingerprint over ALL of a speaker's clean speech.
//!
//! Ported from `Processors/Pyannote/PyannoteSpeakerSegmenter.swift`
//! (`d0a690c86`).

use std::path::Path;

use async_trait::async_trait;
use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{
    EngineError, LocalVoice, ProgressSink, Result, SpeakerDiarizing, SpeakerTurn,
};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

use super::powerset::{self, CLASS_COUNT, SPEAKER_COUNT};
use super::stitcher::{self, Candidate};
use super::turns;
use super::windows::{self, WINDOW_FRAMES, WINDOW_SAMPLES};
use crate::audio_source::AudioSource;
use crate::redimnet::{ReDimNetSpeakerEncoder, features, windows as encoder_windows};

/// Cosine at or above this makes two chunk-local speakers one voice INSIDE an
/// element. It is deliberately looser than the 0.60 that makes two elements'
/// voices one project speaker, and the asymmetry is the point: the probes
/// compared here are pooled over at most 10 s of one chunk, so they are noisier
/// than the pooled vectors a project match sees, while the cost of being wrong
/// is smaller — a bad split inside an element is caught at the project match, a
/// bad merge is not.
///
/// Calibrated upstream 2026-08-24 by sweeping it over nine files whose speaker
/// count is known: right on all nine over `[0.20, 0.60]`. The band is wide
/// because the two failure modes are far apart — the closest two DIFFERENT
/// speakers scored 0.072, and the loosest pair that still has to merge sits
/// above 0.60. 0.45 is inside it with six times the margin on the merge side,
/// where the irrecoverable mistake is.
pub const SAME_VOICE_THRESHOLD: f32 = 0.45;

#[derive(Debug, Clone)]
pub struct PyannoteSpeakerSegmenter {
    backend: Backend,
    version: ModelVersion,
    encoder: ReDimNetSpeakerEncoder,
}

impl PyannoteSpeakerSegmenter {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            encoder: ReDimNetSpeakerEncoder::new(backend.clone()),
            backend,
            version: ModelVersion::default(),
        }
    }

    /// The digest of the weights the graph was exported from — the contract's
    /// `source.files` (`ferment-models-export`, `models/pyannote/model.json`),
    /// pinned by `tests/pyannote_contract.rs`; iOS
    /// `PyannoteSpeakerSegmenter.segmenterHash`. A voice is the product of BOTH
    /// graphs, so both are nameable.
    pub const SEGMENTER_HASH: &'static str =
        "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25";

    #[must_use]
    pub fn segmenter_hash() -> &'static str {
        Self::SEGMENTER_HASH
    }

    /// The encoder half's name and digest, for a caller storing a vector.
    #[must_use]
    pub fn encoder(&self) -> &ReDimNetSpeakerEncoder {
        &self.encoder
    }

    /// Every voice in `frames`, in the order they are first heard.
    ///
    /// `frames` is resampled to 16 kHz here; one decode feeds both models.
    ///
    /// # Errors
    /// If the resample, a prediction, or the embedder fails.
    #[tracing::instrument(skip_all, fields(processor = "pyannote", samples = frames.len()))]
    pub async fn diarize(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<Vec<LocalVoice>> {
        progress.report(0.05, "Loading audio...").await;
        let audio = ReDimNetSpeakerEncoder::to_model_rate(frames)?;

        let chunks = windows::chunks(audio.len());
        if chunks.is_empty() {
            progress.report(1.0, "No audio to diarize").await;
            return Ok(Vec::new());
        }
        let frame_count = windows::frame_count(audio.len());

        let mut marginals = Vec::with_capacity(chunks.len());
        let mut confidences = Vec::with_capacity(chunks.len());
        let mut candidates = Vec::new();

        for (index, chunk) in chunks.iter().enumerate() {
            let probabilities = self.predict(&audio, chunk.sample_offset).await?;
            let decoded = powerset::decode(&probabilities, WINDOW_FRAMES);
            let shared = powerset::overlapped(&decoded.activity, chunk.frames);

            marginals.push(decoded.marginals[..chunk.frames * SPEAKER_COUNT].to_vec());
            confidences.push(decoded.confidence[..chunk.frames].to_vec());

            for speaker in 0..SPEAKER_COUNT {
                let active: Vec<bool> = (0..chunk.frames)
                    .map(|frame| decoded.activity[frame * SPEAKER_COUNT + speaker] == 1)
                    .collect();
                let turns = turns::turns(&active, &shared, chunk.frame_offset);
                if turns.is_empty() {
                    continue;
                }
                candidates.push(Candidate {
                    chunk: index,
                    speaker,
                    turns,
                    probe: Vec::new(),
                });
            }
            progress
                .report(
                    0.10 + 0.50 * (index + 1) as f64 / chunks.len() as f64,
                    "Finding voices...",
                )
                .await;
        }

        // The probe. Fingerprinting each chunk-local speaker is what makes the
        // chunk labels comparable at all — it is the join, not a refinement.
        for index in 0..candidates.len() {
            candidates[index].probe = self.fingerprint(&audio, &candidates[index].turns).await?;
            progress
                .report(
                    0.60 + 0.15 * (index + 1) as f64 / candidates.len() as f64,
                    "Telling voices apart...",
                )
                .await;
        }

        let groups = stitcher::cluster(&candidates, SAME_VOICE_THRESHOLD);
        let voices = stitcher::aggregate(
            &candidates,
            &groups,
            &chunks,
            &marginals,
            &confidences,
            frame_count,
        );

        // A group can average away to nothing: if everything it was built from
        // was a seam twitch, the chunks that saw those frames mid-window
        // outvote it and it ends up with no replies at all. It is not a voice
        // then — it is what the averaging removed.
        let heard: Vec<_> = voices
            .into_iter()
            .filter(|voice| !voice.turns.is_empty())
            .collect();

        // The long pooling: one fingerprint per voice, over ALL of its clean
        // speech in the element, from the replies the averaged chunks agreed on
        // — not from the per-chunk probes, which measure the same seconds twice
        // wherever two chunks overlap.
        let mut results = Vec::with_capacity(heard.len());
        for (index, voice) in heard.iter().enumerate() {
            let vector = self.fingerprint(&audio, &voice.turns).await?;
            results.push(LocalVoice {
                turns: voice.turns.clone(),
                vector,
                confidence: voice.confidence,
            });
            progress
                .report(
                    0.75 + 0.20 * (index + 1) as f64 / heard.len() as f64,
                    "Pooling voices...",
                )
                .await;
        }

        tracing::info!(
            chunks = chunks.len(),
            chunk_local_speakers = candidates.len(),
            voices = results.len(),
            "diarized"
        );
        progress.report(1.0, "Done").await;
        Ok(results)
    }

    /// One graph window's `frames × classes` probabilities, row-major.
    ///
    /// The window is zero-padded on the right when the file ends inside it; the
    /// frames that land in the pad are dropped by count downstream, never by
    /// decoding silence the file did not have.
    ///
    /// # Errors
    /// If the prediction fails or the graph is not the 589×7 checkpoint.
    pub async fn predict(&self, audio: &[f32], sample_offset: usize) -> Result<Vec<f32>> {
        let mut window = vec![0.0_f32; WINDOW_SAMPLES];
        let available = audio
            .len()
            .saturating_sub(sample_offset)
            .min(WINDOW_SAMPLES);
        window[..available].copy_from_slice(&audio[sample_offset..sample_offset + available]);

        let mut inputs = TensorMap::new();
        inputs.insert(
            "audio".into(),
            Tensor::from_f32(vec![1, 1, WINDOW_SAMPLES as i64], window)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::PyannoteSegmentation, &self.version, inputs)
            .await?;

        let logits = outputs
            .get("log_probs")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::PyannoteSegmentation.name().into(),
                key: "log_probs".into(),
            })?;

        // The weights' own account of their geometry, checked against the
        // constants the windowing and the decode are written around. A
        // checkpoint with an eighth class is `speaker-diarization-community-1`
        // — a different model that would decode into plausible nonsense here.
        if logits.shape != [1, WINDOW_FRAMES as i64, CLASS_COUNT as i64] {
            return Err(EngineError::invalid_config(format!(
                "{} emits {:?}, expected [1, {WINDOW_FRAMES}, {CLASS_COUNT}]",
                ModelId::PyannoteSegmentation.name(),
                logits.shape
            )));
        }

        // The graph ends in `softmax → log`, so what comes back is
        // log-probabilities. Exponentiating them here rather than at the reader
        // is what keeps "the graph speaks in logs" a fact about this one
        // function: the decoder is a pure function over a distribution and
        // never has to guess at a scale.
        Ok(logits.to_f32()?.iter().map(|value| value.exp()).collect())
    }

    /// Splice this voice's clean replies in time order into full encoder windows.
    /// A window may cross reply boundaries; only the final short tail is dropped.
    /// Keep one audio window, rather than materializing the concatenated speech.
    async fn fingerprint(&self, audio: &[f32], turns: &[SpeakerTurn]) -> Result<Vec<f32>> {
        let size = encoder_windows::WINDOW_SAMPLES;
        let rate = f64::from(encoder_windows::SAMPLE_RATE);
        let mut window = vec![0.0_f32; size];
        let mut filled = 0;
        let mut vectors = Vec::new();
        let mut clean: Vec<_> = turns.iter().filter(|turn| !turn.overlapped).collect();
        clean.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
        for turn in clean {
            let mut from = (turn.start_time * rate).round().max(0.0) as usize;
            let to = (turn.end_time * rate)
                .round()
                .clamp(0.0, audio.len() as f64) as usize;
            while from < to {
                let take = (size - filled).min(to - from);
                window[filled..filled + take].copy_from_slice(&audio[from..from + take]);
                filled += take;
                from += take;
                if filled == size {
                    vectors.push(self.encoder.embedding(&features(&window)?).await?);
                    filled = 0;
                }
            }
        }
        Ok(stitcher::pooled(&vectors))
    }
}

/// The path-taking contract the cook layer wires to. The decode is the engine's
/// own 16 kHz mono ingest, the same one Parakeet and the audio-event detector
/// read through.
#[async_trait(?Send)]
impl SpeakerDiarizing for PyannoteSpeakerSegmenter {
    async fn diarize(&self, audio_path: &Path, progress: ProgressSink) -> Result<Vec<LocalVoice>> {
        let frames = AudioSource::default().decode(audio_path).await?;
        PyannoteSpeakerSegmenter::diarize(self, frames, progress).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ferment_engine_infer::{InferenceBackend, Tensor};

    use super::*;

    /// 0.60 is the cosine that makes two ELEMENTS' voices one project speaker.
    /// Inside an element the probes are pooled over at most 10 s and a bad
    /// merge is the irrecoverable mistake, so this threshold is deliberately
    /// looser — and it must stay a similarity, not drift into being a distance.
    #[test]
    fn the_within_element_threshold_is_looser_than_the_project_match() {
        assert!((0.0..0.60).contains(&SAME_VOICE_THRESHOLD));
    }

    fn assert_speaker_diarizing<T: SpeakerDiarizing>() {}

    #[test]
    fn the_segmenter_is_the_diarization_capability() {
        assert_speaker_diarizing::<PyannoteSpeakerSegmenter>();
    }

    #[derive(Debug)]
    struct RecordingEncoder {
        inputs: Mutex<Vec<Vec<f32>>>,
        answer: Vec<f32>,
    }

    impl RecordingEncoder {
        fn new() -> Self {
            let mut answer = vec![0.0; encoder_windows::EMBEDDING_DIMENSION];
            answer[0] = 1.0;
            Self {
                inputs: Mutex::new(Vec::new()),
                answer,
            }
        }
    }

    #[async_trait]
    impl InferenceBackend for RecordingEncoder {
        async fn infer(
            &self,
            model: ModelId,
            _version: &ModelVersion,
            inputs: TensorMap,
        ) -> Result<TensorMap> {
            assert_eq!(model, ModelId::RedimnetSpeaker);
            self.inputs.lock().unwrap().push(inputs["mel"].to_f32()?);
            let mut outputs = TensorMap::new();
            outputs.insert(
                "embedding".into(),
                Tensor::from_f32(vec![1, self.answer.len() as i64], self.answer.clone())?,
            );
            Ok(outputs)
        }
    }

    fn speech(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index * 37 % 211) as f32 - 105.0) / 106.0)
            .collect()
    }

    fn turn(from: usize, to: usize, overlapped: bool) -> SpeakerTurn {
        let rate = f64::from(encoder_windows::SAMPLE_RATE);
        SpeakerTurn {
            start_time: from as f64 / rate,
            end_time: to as f64 / rate,
            overlapped,
        }
    }

    #[test]
    fn short_clean_replies_are_sorted_and_spliced_without_pauses_or_overlap() {
        let backend = Arc::new(RecordingEncoder::new());
        let segmenter = PyannoteSpeakerSegmenter::new(backend.clone());
        let audio = speech(120_000);
        let turns = [
            turn(90_000, 115_000, false),
            turn(30_000, 48_000, true),
            turn(20_000, 30_000, false),
            turn(100, 10_100, false),
            turn(50_000, 75_000, false),
        ];
        assert!(
            turns
                .iter()
                .all(|turn| turn.duration() < encoder_windows::minimum_span_seconds())
        );
        let pooled = futures_executor::block_on(segmenter.fingerprint(&audio, &turns)).unwrap();

        let mut expected = Vec::new();
        for span in [100..10_100, 20_000..30_000, 50_000..75_000, 90_000..115_000] {
            expected.extend_from_slice(&audio[span]);
        }
        let windows: Vec<_> = expected
            .chunks_exact(encoder_windows::WINDOW_SAMPLES)
            .collect();
        assert_eq!(windows.len(), 2, "the final 5,840 samples are never padded");
        let inputs = backend.inputs.lock().unwrap();
        assert_eq!(inputs.len(), windows.len());
        for (actual, expected) in inputs.iter().zip(windows) {
            assert_eq!(
                *actual,
                features(expected).unwrap(),
                "the graph must see the clean spliced samples"
            );
        }
        assert_eq!(pooled, backend.answer);
    }

    #[test]
    fn total_clean_speech_short_of_one_window_never_calls_the_encoder() {
        let backend = Arc::new(RecordingEncoder::new());
        let segmenter = PyannoteSpeakerSegmenter::new(backend.clone());
        let audio = speech(64_160);
        let turns = [
            turn(0, 16_000, false),
            turn(16_000, 20_000, true),
            turn(20_000, 36_079, false),
        ];
        let pooled = futures_executor::block_on(segmenter.fingerprint(&audio, &turns)).unwrap();
        assert!(pooled.is_empty());
        assert!(backend.inputs.lock().unwrap().is_empty());
    }

    #[test]
    fn clean_reply_bounds_round_and_clamp_to_the_actual_audio() {
        let backend = Arc::new(RecordingEncoder::new());
        let segmenter = PyannoteSpeakerSegmenter::new(backend.clone());
        let audio = speech(37_080);
        let rate = f64::from(encoder_windows::SAMPLE_RATE);
        let turns = [
            SpeakerTurn {
                start_time: 14_999.6 / rate,
                end_time: 100.0,
                overlapped: false,
            },
            SpeakerTurn {
                start_time: -1.0,
                end_time: 10_000.4 / rate,
                overlapped: false,
            },
        ];
        let pooled = futures_executor::block_on(segmenter.fingerprint(&audio, &turns)).unwrap();
        let expected: Vec<_> = audio[..10_000]
            .iter()
            .chain(&audio[15_000..])
            .copied()
            .collect();
        assert_eq!(expected.len(), encoder_windows::WINDOW_SAMPLES);
        assert_eq!(
            *backend.inputs.lock().unwrap(),
            vec![features(&expected).unwrap()]
        );
        assert_eq!(pooled, backend.answer);
    }

    #[test]
    fn spliced_windows_keep_the_encoders_finite_dimension_guard() {
        for answer in [
            vec![1.0; 8],
            vec![f32::NAN; encoder_windows::EMBEDDING_DIMENSION],
        ] {
            let backend = Arc::new(RecordingEncoder {
                inputs: Mutex::new(Vec::new()),
                answer,
            });
            let segmenter = PyannoteSpeakerSegmenter::new(backend);
            let audio = speech(encoder_windows::WINDOW_SAMPLES);
            let turns = [turn(0, 16_000, false), turn(16_000, audio.len(), false)];
            let result = futures_executor::block_on(segmenter.fingerprint(&audio, &turns));
            assert!(
                matches!(result, Err(EngineError::InvalidOutput)),
                "{result:?}"
            );
        }
    }

    /// A backend that answers with whatever shape it was built for, so the
    /// shape guard can be driven without a second checkpoint on disk.
    #[derive(Debug)]
    struct FixedShape(Vec<i64>);

    #[async_trait]
    impl InferenceBackend for FixedShape {
        async fn infer(
            &self,
            _model: ModelId,
            _version: &ModelVersion,
            _inputs: TensorMap,
        ) -> Result<TensorMap> {
            let count = self.0.iter().product::<i64>() as usize;
            let mut outputs = TensorMap::new();
            outputs.insert(
                "log_probs".into(),
                Tensor::from_f32(self.0.clone(), vec![-1.0; count])?,
            );
            Ok(outputs)
        }
    }

    fn predict_against(shape: Vec<i64>) -> Result<Vec<f32>> {
        let segmenter = PyannoteSpeakerSegmenter::new(Arc::new(FixedShape(shape)));
        futures_executor::block_on(segmenter.predict(&[0.0; WINDOW_SAMPLES], 0))
    }

    /// The guard that separates segmentation-3.0 from the model that sits next
    /// to it in the same family. `speaker-diarization-community-1` has an
    /// EIGHT-class powerset and a different frame rate; taking it by accident
    /// would mis-decode every frame while still producing plausible turns, so
    /// the shape has to be refused rather than trusted.
    #[test]
    fn an_eight_class_checkpoint_is_refused_rather_than_mis_decoded() {
        let error = predict_against(vec![1, WINDOW_FRAMES as i64, 8]).expect_err("must refuse");
        assert!(
            format!("{error}").contains("expected [1, 589, 7]"),
            "{error}"
        );
    }

    /// The other half of the same guard: the right class count at the wrong
    /// frame rate is still the wrong checkpoint.
    #[test]
    fn a_different_frame_rate_is_refused_too() {
        assert!(predict_against(vec![1, 293, CLASS_COUNT as i64]).is_err());
    }

    /// The graph speaks in log-probabilities and this function is where they
    /// stop being logs. Everything downstream reads a distribution.
    #[test]
    fn the_graphs_log_probabilities_come_back_exponentiated() {
        let probabilities =
            predict_against(vec![1, WINDOW_FRAMES as i64, CLASS_COUNT as i64]).expect("runs");
        assert_eq!(probabilities.len(), WINDOW_FRAMES * CLASS_COUNT);
        assert!(
            probabilities
                .iter()
                .all(|p| (p - (-1.0_f32).exp()).abs() < 1e-6),
            "log-probabilities must be exponentiated, not passed through"
        );
    }
}
