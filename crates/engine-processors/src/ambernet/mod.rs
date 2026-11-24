//! On-device spoken-language identification using NVIDIA's AmberNet
//! (`langid_ambernet`, NGC v1.12.0) over the VoxLingua107 label set — the
//! `ambernet_langid` contract (`ferment-models-export`,
//! `models/ambernet/model.json`), the same door iOS calls.
//!
//! Pipeline:
//! 1. Gather up to [`SCORING_WINDOW_SECONDS`] of audio: the caller's speech
//!    segments concatenated in order, or the head of the file when no segments
//!    are given.
//! 2. NeMo's front-end on the host ([`features`]) and one inference through the
//!    door — `mel [1, 80, 1, 3008]` and `mask [1, 1, 1, 3008]` in,
//!    `logits [1, 107, 1, 1]` out.
//! 3. Softmax over 107 logits → hypotheses sorted by probability desc.
//!
//! This model has no "unknown", "silence" or "non-speech" class — it always
//! answers with a distribution over its 107 languages, and on music it will
//! answer confidently and wrongly. Gate it upstream by passing the AED's
//! speech segments; an empty slice short-circuits before the model loads.
//! There is deliberately **no coverage threshold**: trust the AED.

pub mod langs;

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, PreEmphasis, Scale, Spectrum,
    TablePrecision, Window as FrameWindow,
};
use ferment_engine_core::{AudioEventSegment, EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};

use crate::audio_source::AudioSampleSource;

/// Number of languages the model scores. Checked against both the committed
/// label list and the graph's logit count.
pub const LANGUAGE_COUNT: usize = 107;

/// How much audio actually reaches the model. AmberNet was trained on
/// VoxLingua107 utterances and its accuracy is flat well below this —
/// anything past the cap is simply not scored.
pub const SCORING_WINDOW_SECONDS: f64 = 30.0;

pub const SAMPLE_RATE: u32 = 16_000;
pub const HOP_LENGTH: usize = 160;
pub const WINDOW_SAMPLES: usize = (SCORING_WINDOW_SECONDS as usize) * SAMPLE_RATE as usize;
pub const MEL_BANDS: usize = 80;
/// The door's width: NeMo pads a full window's 3001 frames to a multiple of 16.
pub const DOOR_FRAMES: usize = 3_008;

/// The per-band normalisation divides by (frames - 1), so a single frame would
/// divide by zero.
pub const MINIMUM_FRAMES: usize = 2;

/// NeMo's `FilterbankFeatures` at the checkpoint's config: pre-emphasis 0.97
/// over the real samples, `torch.stft` centred with zero padding, a 400-sample
/// symmetric Hann in a 512-point transform, power, 80 Slaney mels 0–8000 Hz,
/// area-normalised, `log(x + 2⁻²⁴)` — Parakeet's value at 80 bands. The same
/// value as iOS `AmberNetLanguageID.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: 512,
    hop_length: HOP_LENGTH,
    mel_bands: MEL_BANDS,
    f_min: 0.0,
    f_max: 8_000.0,
    input_scale: 1.0,
    padding: Padding::Constant,
    window: FrameWindow::HannSymmetric,
    window_length: Some(400),
    dc_removal: false,
    pre_emphasis: Some(PreEmphasis::Signal { alpha: 0.97 }),
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Slaney,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::SlaneyArea,
    table_precision: TablePrecision::Float32,
    compression: Compression::Log {
        guard: 5.960_464_5e-8,
    },
};
/// Each band normalised over the window's valid frames: `(x − mean) / (std + ε)`,
/// std with n − 1.
pub const NORMALIZATION_EPSILON: f32 = 1e-5;

/// One detected language hypothesis. `code` is the model's own VoxLingua107
/// tag, verbatim — legacy spellings (`iw`, `jw`) are modernised by
/// [`crate::transcriber_routing`], not here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageHypothesis {
    pub code: String,
    pub probability: f32,
}

/// What the language-ID pass produces: hypotheses sorted by probability desc,
/// plus how many seconds of speech actually reached the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageIdResult {
    pub hypotheses: Vec<LanguageHypothesis>,
    pub speech_seconds: f64,
}

/// The gathered audio — at most [`WINDOW_SAMPLES`], unpadded: the front-end
/// pads — plus the count of frames it backs.
#[derive(Debug, Clone)]
pub struct Window {
    pub samples: Vec<f32>,
    pub frame_count: usize,
}

#[derive(Debug, Clone)]
pub struct AmberNetLanguageId {
    backend: Backend,
    version: ModelVersion,
}

impl AmberNetLanguageId {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    /// Identify the spoken language over `source`, restricted to
    /// `speech_segments` when supplied. An empty slice short-circuits to an
    /// empty result without loading the model.
    #[tracing::instrument(skip_all, fields(processor = "ambernet"))]
    pub async fn identify(
        &self,
        source: &dyn AudioSampleSource,
        speech_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<LanguageIdResult> {
        if speech_segments.is_some_and(<[AudioEventSegment]>::is_empty) {
            progress.report(1.0, "No speech detected").await;
            return Ok(LanguageIdResult {
                hypotheses: Vec::new(),
                speech_seconds: 0.0,
            });
        }
        if source.sample_count() == 0 {
            return Err(EngineError::NoAudioData);
        }

        progress.report(0.05, "loading audio").await;
        let window = build_window(source, speech_segments)?;
        let languages = langs::languages()?;

        progress.report(0.5, "identifying language").await;
        let logits = self.predict(&window).await?;
        if logits.len() != languages.len() || !logits.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidOutput);
        }

        let result = LanguageIdResult {
            hypotheses: hypotheses(&logits, languages),
            speech_seconds: scored_seconds(speech_segments, &window),
        };
        if let Some(top) = result.hypotheses.first() {
            tracing::info!(
                frames = window.frame_count,
                code = %top.code,
                probability = top.probability,
                "language identified"
            );
        }
        progress.report(1.0, "done").await;
        Ok(result)
    }

    async fn predict(&self, window: &Window) -> Result<Vec<f32>> {
        self.logits(&features(window)?, window.frame_count).await
    }

    /// The door: `mel` bands major `[80 × DOOR_FRAMES]`, the first `valid`
    /// frames real.
    pub async fn logits(&self, mel: &[f32], valid: usize) -> Result<Vec<f32>> {
        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(
                vec![1, MEL_BANDS as i64, 1, DOOR_FRAMES as i64],
                mel.to_vec(),
            )?,
        );
        let mask = (0..DOOR_FRAMES)
            .map(|frame| if frame < valid { 1.0 } else { 0.0 })
            .collect();
        inputs.insert(
            "mask".into(),
            Tensor::from_f32(vec![1, 1, 1, DOOR_FRAMES as i64], mask)?,
        );

        let outputs = self
            .backend
            .infer(ModelId::AmberNetLangId, &self.version, inputs)
            .await?;
        outputs
            .get("logits")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::AmberNetLangId.name().into(),
                key: "logits".into(),
            })?
            .to_f32()
    }
}

/// The door's `mel`, bands major `[80 × DOOR_FRAMES]`: the contract's
/// front-end over the window's REAL samples — pre-emphasis runs on them only
/// and everything past them is the constant padding's zeros, which is what
/// NeMo's mask leaves — each band normalised over the valid frames, every
/// frame past them zero.
pub fn features(window: &Window) -> Result<Vec<f32>> {
    let mel = MEL_SPEC
        .compute(&window.samples)?
        .normalized_per_band(window.frame_count, NORMALIZATION_EPSILON);
    Ok(mel.bands_major(Some(DOOR_FRAMES)))
}

/// Fill the model's fixed window.
///
/// With segments, speech is concatenated in start order until the cap is
/// reached — the model pools over time, so a seam between two speech regions
/// costs far less than feeding it the silence that sat between them.
pub fn build_window(
    source: &dyn AudioSampleSource,
    segments: Option<&[AudioEventSegment]>,
) -> Result<Window> {
    let mut samples = vec![0.0f32; WINDOW_SAMPLES];
    let mut filled = 0usize;

    let take = |start: i64, end: i64, samples: &mut Vec<f32>, filled: &mut usize| -> Result<()> {
        let from = start.max(0) as usize;
        let to = (end.max(0) as usize).min(source.sample_count());
        if to <= from {
            return Ok(());
        }
        let count = (to - from).min(WINDOW_SAMPLES - *filled);
        let copied = source.copy_samples(&mut samples[*filled..*filled + count], from, count)?;
        *filled += copied;
        Ok(())
    };

    match segments {
        Some(segments) => {
            let mut ordered = segments.to_vec();
            ordered.sort_by(|left, right| left.start_time.total_cmp(&right.start_time));
            for segment in ordered {
                take(
                    seconds_to_sample(segment.start_time),
                    seconds_to_sample(segment.end_time()),
                    &mut samples,
                    &mut filled,
                )?;
                if filled >= WINDOW_SAMPLES {
                    break;
                }
            }
        }
        None => take(0, WINDOW_SAMPLES as i64, &mut samples, &mut filled)?,
    }

    let frame_count = filled / HOP_LENGTH;
    if frame_count < MINIMUM_FRAMES {
        return Err(EngineError::AudioTooShort {
            samples: filled,
            min_required: MINIMUM_FRAMES * HOP_LENGTH,
        });
    }
    samples.truncate(filled);
    Ok(Window {
        samples,
        frame_count,
    })
}

/// Max-subtracted softmax over the logits, sorted by probability descending.
///
/// The subtraction is the model's own `logits.max()`, not a floor at zero:
/// softmax is shift-invariant, so any constant gives the same probabilities in
/// exact arithmetic — but subtracting zero from a set of large *negative*
/// logits underflows every exponent to 0 and turns the whole distribution into
/// NaN. Only the true maximum guarantees one term of exactly 1.
#[must_use]
pub fn hypotheses(logits: &[f32], languages: &[String]) -> Vec<LanguageHypothesis> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentiated = logits
        .iter()
        .map(|logit| (logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = exponentiated.iter().sum::<f32>();
    let mut out = languages
        .iter()
        .zip(exponentiated)
        .map(|(code, weight)| LanguageHypothesis {
            code: code.clone(),
            probability: weight / total,
        })
        .collect::<Vec<_>>();
    out.sort_by(|left, right| right.probability.total_cmp(&left.probability));
    out
}

/// Speech seconds that reached the model — the total the window gathered,
/// capped at the scoring window. Mirrors iOS
/// `IdentifyLanguageProcessor.scoredSeconds`, except the count comes from the
/// window that was actually built rather than being recomputed from the
/// segments, so a segment past the end of the file cannot inflate it.
fn scored_seconds(segments: Option<&[AudioEventSegment]>, window: &Window) -> f64 {
    let filled_seconds = (window.frame_count * HOP_LENGTH) as f64 / f64::from(SAMPLE_RATE);
    match segments {
        Some(segments) => {
            let declared = segments
                .iter()
                .map(|segment| segment.duration.max(0.0))
                .sum::<f64>();
            declared.min(SCORING_WINDOW_SECONDS).min(filled_seconds)
        }
        None => filled_seconds,
    }
}

fn seconds_to_sample(seconds: f64) -> i64 {
    (seconds * f64::from(SAMPLE_RATE)).trunc() as i64
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use ferment_engine_core::AudioEvent;
    use ferment_engine_infer::InferenceBackend;

    use super::*;
    use crate::audio_source::ArrayAudioSampleSource;

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

    /// "Gate it upstream by passing the AED's speech segments; an empty array
    /// short-circuits before the model loads." There is no coverage threshold
    /// anywhere — this is the whole gate.
    #[test]
    fn empty_speech_segments_short_circuit_before_the_model_loads() {
        let identifier = AmberNetLanguageId::new(Arc::new(NeverBackend));
        let source = ramp(SAMPLE_RATE as usize * 10);
        let stages = Arc::new(Mutex::new(Vec::new()));
        let observed = stages.clone();
        let progress = ProgressSink::from_callback(move |event| {
            observed
                .lock()
                .expect("stage lock")
                .push((event.progress, event.stage));
        });
        let result = futures_executor::block_on(identifier.identify(&source, Some(&[]), progress))
            .expect("no speech is a result, not a failure");
        assert!(result.hypotheses.is_empty());
        assert_eq!(result.speech_seconds, 0.0);
        // Kill: the empty gate returns without reporting current Swift's terminal state.
        assert_eq!(
            stages.lock().expect("stage lock").last(),
            Some(&(1.0, "No speech detected".to_owned()))
        );
    }

    fn ramp(count: usize) -> ArrayAudioSampleSource {
        ArrayAudioSampleSource::new((0..count).map(|index| index as f32).collect())
    }

    fn speech(start: f64, duration: f64) -> AudioEventSegment {
        AudioEventSegment {
            event: AudioEvent::Speech,
            start_time: start,
            duration,
            confidence: 1.0,
        }
    }

    #[test]
    fn the_window_is_the_gathered_audio_unpadded() {
        let window = build_window(&ramp(16_000), None).expect("window");
        assert_eq!(
            window.samples.len(),
            16_000,
            "the front-end pads, not the window"
        );
        assert_eq!(window.frame_count, 100, "16000 / 160");
    }

    /// The door's `mel` is the width the graph was exported at, and every
    /// frame past the valid ones is zero.
    #[test]
    fn the_features_fill_the_door_and_zero_past_the_valid_frames() {
        let samples = (0..16_000)
            .map(|index| (index as f32 * 0.05).sin() * 0.3)
            .collect::<Vec<_>>();
        let window = build_window(&ArrayAudioSampleSource::new(samples), None).expect("window");
        let mel = features(&window).expect("features");
        assert_eq!(mel.len(), MEL_BANDS * DOOR_FRAMES);
        for band in 0..MEL_BANDS {
            let row = &mel[band * DOOR_FRAMES..(band + 1) * DOOR_FRAMES];
            assert!(row[window.frame_count..].iter().all(|value| *value == 0.0));
            assert!(row[..window.frame_count].iter().any(|value| *value != 0.0));
        }
    }

    #[test]
    fn no_segments_takes_the_head_of_the_file() {
        let window = build_window(&ramp(WINDOW_SAMPLES * 2), None).expect("window");
        assert_eq!(window.frame_count, WINDOW_SAMPLES / HOP_LENGTH);
        assert_eq!(window.samples.len(), WINDOW_SAMPLES);
        assert_eq!(window.samples[0], 0.0);
        assert_eq!(
            window.samples[WINDOW_SAMPLES - 1],
            (WINDOW_SAMPLES - 1) as f32
        );
    }

    #[test]
    fn segments_are_concatenated_in_start_order_not_argument_order() {
        // Two one-second regions handed over out of order; the window holds
        // them earliest-first, back to back, with no silence between.
        let source = ramp(SAMPLE_RATE as usize * 10);
        let window = build_window(&source, Some(&[speech(5.0, 1.0), speech(1.0, 1.0)])).expect("w");
        assert_eq!(window.frame_count, 200, "two seconds at 160-sample hops");
        assert_eq!(window.samples.len(), 32_000, "gathered audio is not padded");
        assert_eq!(window.samples[0], 16_000.0, "the 1 s region leads");
        assert_eq!(
            window.samples[16_000], 80_000.0,
            "the 5 s region follows it"
        );
    }

    #[test]
    fn the_window_stops_at_the_cap() {
        let source = ramp(SAMPLE_RATE as usize * 120);
        let window =
            build_window(&source, Some(&[speech(0.0, 40.0), speech(60.0, 20.0)])).expect("window");
        assert_eq!(window.frame_count, WINDOW_SAMPLES / HOP_LENGTH);
        assert_eq!(
            window.samples[WINDOW_SAMPLES - 1],
            (WINDOW_SAMPLES - 1) as f32,
            "the cap cuts inside the first segment, so nothing of the second lands"
        );
    }

    /// Kill: reset the cap for each span instead of capping the concatenated window.
    /// iOS: AmberNetLanguageIDTests.testSegmentsCannotOverrunTheWindow, second branch.
    #[test]
    fn the_window_cap_also_cuts_inside_a_later_segment() {
        let source = ramp(SAMPLE_RATE as usize * 90);
        let window =
            build_window(&source, Some(&[speech(0.0, 20.0), speech(20.0, 25.0)])).expect("window");
        assert_eq!(window.frame_count, 3_000);
        assert_eq!(window.samples.len(), 480_000);
        assert_eq!(window.samples.last(), Some(&479_999.0));
    }

    /// Kill: trust positive out-of-range bounds or score an empty/one-frame selection.
    /// iOS: out-of-range and too-little-audio branches, through the segment path.
    #[test]
    fn segment_bounds_and_the_two_frame_floor_apply_after_gathering() {
        let source = ramp(SAMPLE_RATE as usize * 10);
        let past_end = build_window(&source, Some(&[speech(9.0, 51.0)])).expect("clamped tail");
        assert_eq!(past_end.frame_count, 100);
        let tiny = build_window(&source, Some(&[speech(0.0, 0.01)])).expect_err("160 samples");
        assert!(matches!(
            tiny,
            EngineError::AudioTooShort {
                samples: 160,
                min_required: 320
            }
        ));
        assert!(build_window(&source, Some(&[speech(50.0, 10.0)])).is_err());
        let minimum = build_window(&source, Some(&[speech(0.0, 0.02)])).expect("two frames");
        assert_eq!(minimum.frame_count, 2);
    }

    #[test]
    fn a_segment_past_the_end_of_the_file_contributes_nothing() {
        let source = ramp(SAMPLE_RATE as usize);
        let window =
            build_window(&source, Some(&[speech(0.0, 0.5), speech(90.0, 10.0)])).expect("window");
        assert_eq!(window.frame_count, 50);
    }

    #[test]
    fn below_the_two_frame_floor_is_an_error_not_a_guess() {
        let error = build_window(&ramp(319), None).expect_err("319 samples is one frame");
        assert!(
            matches!(
                error,
                EngineError::AudioTooShort {
                    samples: 319,
                    min_required: 320
                }
            ),
            "{error}"
        );
        assert!(build_window(&ramp(320), None).is_ok(), "two frames pass");
    }

    #[test]
    fn negative_segment_times_clamp_instead_of_wrapping() {
        let source = ramp(SAMPLE_RATE as usize);
        let window = build_window(&source, Some(&[speech(-2.0, 2.5)])).expect("window");
        assert_eq!(window.frame_count, 50, "only the 0..0.5 s overlap is real");
        assert_eq!(window.samples[0], 0.0);
    }

    #[test]
    fn softmax_is_max_subtracted_and_sorted_descending() {
        let languages = ["a", "b", "c"].map(str::to_owned).to_vec();
        let out = hypotheses(&[1.0, 3.0, 2.0], &languages);
        assert_eq!(
            out.iter().map(|h| h.code.as_str()).collect::<Vec<_>>(),
            ["b", "c", "a"]
        );
        let total = out.iter().map(|h| h.probability).sum::<f32>();
        assert!((total - 1.0).abs() < 1e-6, "probabilities sum to one");
        assert!((out[0].probability - 0.665_240_9).abs() < 1e-6);
    }

    /// Without the max subtraction these logits overflow `exp` to `inf` and
    /// every probability comes back NaN.
    #[test]
    fn large_logits_do_not_overflow() {
        let languages = ["a", "b"].map(str::to_owned).to_vec();
        let out = hypotheses(&[200.0, 199.0], &languages);
        assert!(out.iter().all(|h| h.probability.is_finite()));
        assert!((out[0].probability - 0.731_058_6).abs() < 1e-6);
    }

    /// The other end of the same trap: subtracting a zero floor instead of the
    /// model's own maximum underflows every exponent and returns 107 NaNs.
    #[test]
    fn all_negative_logits_do_not_underflow() {
        let languages = ["a", "b"].map(str::to_owned).to_vec();
        let out = hypotheses(&[-200.0, -201.0], &languages);
        assert!(out.iter().all(|h| h.probability.is_finite()), "{out:?}");
        assert!((out[0].probability - 0.731_058_6).abs() < 1e-6);
    }

    #[test]
    fn scored_seconds_follow_the_window_fill() {
        let source = ramp(SAMPLE_RATE as usize * 120);
        let capped = [speech(40.0, 10.0), speech(0.0, 25.0)];
        let window = build_window(&source, Some(&capped)).expect("window");
        assert_eq!(
            scored_seconds(Some(&capped), &window),
            30.0,
            "25 + 10 clamps at the model's 30 s window"
        );

        let short = [speech(1.0, 2.5)];
        let window = build_window(&source, Some(&short)).expect("window");
        assert_eq!(scored_seconds(Some(&short), &window), 2.5);

        let hostile = [speech(5.0, -3.0), speech(0.0, 4.0)];
        let window = build_window(&source, Some(&hostile)).expect("window");
        assert_eq!(
            scored_seconds(Some(&hostile), &window),
            4.0,
            "a negative-duration segment contributes nothing, not a refund"
        );
    }

    /// The declared duration is a claim; the window is the fact. A segment
    /// running past the end of the file must not inflate the reported count.
    #[test]
    fn scored_seconds_cannot_exceed_what_was_actually_gathered() {
        let source = ramp(SAMPLE_RATE as usize);
        let overclaimed = [speech(0.0, 20.0)];
        let window = build_window(&source, Some(&overclaimed)).expect("window");
        assert_eq!(scored_seconds(Some(&overclaimed), &window), 1.0);
    }
}
