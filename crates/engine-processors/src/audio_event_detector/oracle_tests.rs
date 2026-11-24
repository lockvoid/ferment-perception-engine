//! Scenario-for-scenario translations of the Swift AED oracle suite.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{
    AudioEvent, AudioEventDetectionResult, AudioEventFrameProbabilities, AudioEventSegment,
    ProgressSink, Result,
};
use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_test_fixtures as fixtures;

use super::config::{
    FRAME_SHIFT_SECONDS, INPUT_DIM, MUSIC_THRESHOLD, SAMPLE_RATE, SINGING_THRESHOLD,
    SMOOTH_WINDOW_SIZE, SPEECH_THRESHOLD, WINDOW_FRAMES, WINDOW_STRIDE, frames_for_samples,
};
use super::segmenter::Postprocessor;
use super::{AudioEventDetector, AudioEventSegmenter, Cmvn, SegmenterConfig};

fn fixture(stem: &str, file: &str) -> String {
    format!("AudioEventDetector/{stem}/{file}")
}

fn read_f32(stem: &str, file: &str) -> Option<Vec<f32>> {
    fixtures::f32_le(&fixture(stem, file))
}

fn read_i16(stem: &str, file: &str) -> Option<Vec<i16>> {
    Some(
        fixtures::bytes(&fixture(stem, file))?
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect(),
    )
}

fn read_i32_as_u8(stem: &str, file: &str) -> Option<Vec<u8>> {
    Some(
        fixtures::bytes(&fixture(stem, file))?
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u8)
            .collect(),
    )
}

fn class_probabilities(stem: &str, event_index: usize) -> Option<Vec<f32>> {
    Some(
        read_f32(stem, "probs.bin")?
            .chunks_exact(3)
            .map(|row| row[event_index])
            .collect(),
    )
}

fn assert_smoothing_matches(stem: &str, event_index: usize, name: &str) {
    let Some(probabilities) = class_probabilities(stem, event_index) else {
        return;
    };
    let actual = Postprocessor::smooth(&probabilities, SMOOTH_WINDOW_SIZE);
    let expected = read_f32(stem, &format!("smoothed_{name}.bin")).unwrap();
    assert_eq!(actual.len(), expected.len());
    let difference = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(difference < 1.0e-5, "{stem}/{name}: {difference}");
}

fn assert_decisions_match(stem: &str, event_index: usize, name: &str, threshold: f32) {
    let Some(probabilities) = class_probabilities(stem, event_index) else {
        return;
    };
    let actual = Postprocessor::process(&probabilities, SegmenterConfig::new(threshold));
    let expected = read_i32_as_u8(stem, &format!("decisions_{name}.bin")).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual, expected, "{stem}/{name} decisions diverged");
}

#[test]
fn test_smoothing_hairdryer_speech() {
    assert_smoothing_matches("hairdryer", 0, "speech");
}

#[test]
fn test_smoothing_hairdryer_singing() {
    assert_smoothing_matches("hairdryer", 1, "singing");
}

#[test]
fn test_smoothing_hairdryer_music() {
    assert_smoothing_matches("hairdryer", 2, "music");
}

#[test]
fn test_smoothing_sec5_speech() {
    assert_smoothing_matches("sec5", 0, "speech");
}

#[test]
fn test_smoothing_sec5_singing() {
    assert_smoothing_matches("sec5", 1, "singing");
}

#[test]
fn test_smoothing_sec5_music() {
    assert_smoothing_matches("sec5", 2, "music");
}

#[test]
fn test_final_decisions_hairdryer_speech() {
    assert_decisions_match("hairdryer", 0, "speech", SPEECH_THRESHOLD);
}

#[test]
fn test_final_decisions_hairdryer_singing() {
    assert_decisions_match("hairdryer", 1, "singing", SINGING_THRESHOLD);
}

#[test]
fn test_final_decisions_hairdryer_music() {
    assert_decisions_match("hairdryer", 2, "music", MUSIC_THRESHOLD);
}

#[test]
fn test_final_decisions_sec5_speech() {
    assert_decisions_match("sec5", 0, "speech", SPEECH_THRESHOLD);
}

#[test]
fn test_final_decisions_sec5_singing() {
    assert_decisions_match("sec5", 1, "singing", SINGING_THRESHOLD);
}

#[test]
fn test_final_decisions_sec5_music() {
    assert_decisions_match("sec5", 2, "music", MUSIC_THRESHOLD);
}

fn segment(event: AudioEvent, start: f64, end: f64, confidence: f32) -> AudioEventSegment {
    AudioEventSegment {
        event,
        start_time: start,
        duration: end - start,
        confidence,
    }
}

#[test]
fn test_empty_is_zero() {
    let summary = AudioEventSegmenter::summarize(&[], 10.0);
    assert_eq!(summary.speech.coverage, 0.0);
    assert_eq!(summary.musical.coverage, 0.0);
    assert_eq!(summary.musical.confidence, 0.0);
}

#[test]
fn test_disjoint_spans_cover_own_fraction() {
    let summary = AudioEventSegmenter::summarize(
        &[
            segment(AudioEvent::Speech, 0.0, 2.0, 0.8),
            segment(AudioEvent::Music, 4.0, 6.0, 0.6),
        ],
        10.0,
    );
    assert!((summary.speech.coverage - 0.2).abs() < 1.0e-9);
    assert!((summary.music.coverage - 0.2).abs() < 1.0e-9);
    assert_eq!(summary.song.coverage, 0.0);
    assert!((summary.speech.confidence - 0.8).abs() < 1.0e-6);
}

#[test]
fn test_singing_surfaces_as_song() {
    let summary =
        AudioEventSegmenter::summarize(&[segment(AudioEvent::Singing, 1.0, 5.0, 0.7)], 10.0);
    assert!((summary.song.coverage - 0.4).abs() < 1.0e-9);
    assert!((summary.song.confidence - 0.7).abs() < 1.0e-6);
    assert_eq!(summary.speech.coverage, 0.0);
    assert_eq!(summary.music.coverage, 0.0);
}

#[test]
fn test_musical_union_merges_overlap() {
    let summary = AudioEventSegmenter::summarize(
        &[
            segment(AudioEvent::Music, 0.0, 5.0, 0.9),
            segment(AudioEvent::Singing, 3.0, 8.0, 0.7),
        ],
        10.0,
    );
    assert!((summary.music.coverage - 0.5).abs() < 1.0e-9);
    assert!((summary.song.coverage - 0.5).abs() < 1.0e-9);
    assert!((summary.musical.coverage - 0.8).abs() < 1.0e-9);
    assert!((summary.musical.confidence - 0.9).abs() < 1.0e-6);
}

#[test]
fn test_vocal_union_merges_song_and_speech() {
    let summary = AudioEventSegmenter::summarize(
        &[
            segment(AudioEvent::Singing, 0.0, 4.0, 0.6),
            segment(AudioEvent::Speech, 2.0, 6.0, 0.8),
        ],
        10.0,
    );
    assert!((summary.vocal.coverage - 0.6).abs() < 1.0e-9);
    assert!((summary.vocal.confidence - 0.8).abs() < 1.0e-6);
    assert!((summary.musical.coverage - 0.4).abs() < 1.0e-9);
    assert!((summary.musical.confidence - 0.6).abs() < 1.0e-6);
}

#[test]
fn test_touching_spans_of_same_class_merge() {
    let summary = AudioEventSegmenter::summarize(
        &[
            segment(AudioEvent::Speech, 0.0, 2.0, 0.5),
            segment(AudioEvent::Speech, 2.0, 4.0, 0.9),
        ],
        10.0,
    );
    assert!((summary.speech.coverage - 0.4).abs() < 1.0e-9);
    assert!((summary.speech.confidence - 0.9).abs() < 1.0e-6);
}

#[test]
fn test_zero_duration_yields_zero() {
    let summary =
        AudioEventSegmenter::summarize(&[segment(AudioEvent::Speech, 0.0, 0.0, 1.0)], 0.0);
    assert_eq!(summary.speech.coverage, 0.0);
    assert_eq!(summary.vocal.coverage, 0.0);
}

fn fixture_result(stem: &str, duration_seconds: f64) -> Option<AudioEventDetectionResult> {
    let probabilities = read_f32(stem, "probs.bin")?;
    Some(AudioEventDetectionResult {
        duration_seconds,
        frame_shift_seconds: FRAME_SHIFT_SECONDS,
        frame_probs: probabilities
            .chunks_exact(3)
            .map(|row| AudioEventFrameProbabilities {
                speech: row[0],
                singing: row[1],
                music: row[2],
            })
            .collect(),
    })
}

#[test]
fn test_hairdryer_detects_speech() {
    let Some(result) = fixture_result("hairdryer", 20.133_125) else {
        return;
    };
    assert!(result.frame_probs.len() > 1_500);
    assert!((result.frame_shift_seconds - 0.010).abs() < 1.0e-9);
    assert!(result.duration_seconds > 18.0 && result.duration_seconds < 22.0);
    let speech_coverage =
        AudioEventSegmenter::coverage(&result, AudioEvent::Speech, SPEECH_THRESHOLD);
    assert!(speech_coverage > 0.95);
    let segments =
        AudioEventSegmenter::segments(&result, AudioEvent::Speech, SegmenterConfig::SPEECH);
    assert!(!segments.is_empty());
    let total = segments.iter().map(|segment| segment.duration).sum::<f64>();
    assert!(total > result.duration_seconds * 0.9);
}

#[test]
fn test_sec5_detects_music_not_speech() {
    let Some(result) = fixture_result("sec5", 30.998_625) else {
        return;
    };
    assert!(result.frame_probs.len() > 2_500);
    let speech_coverage =
        AudioEventSegmenter::coverage(&result, AudioEvent::Speech, SPEECH_THRESHOLD);
    let music_coverage = AudioEventSegmenter::coverage(&result, AudioEvent::Music, MUSIC_THRESHOLD);
    assert!(music_coverage > 0.95);
    assert!(speech_coverage < 0.20);
    let segments =
        AudioEventSegmenter::segments(&result, AudioEvent::Music, SegmenterConfig::MUSIC);
    let total = segments.iter().map(|segment| segment.duration).sum::<f64>();
    assert!(total > result.duration_seconds * 0.9);
}

/// The door, answered from the fixture: it checks that the `feat` the
/// processor sends is the fixture's fbank through the CMVN, bands major, and
/// answers the fixture's probs — so the processor's host half and its window
/// assembly are graded without a model.
#[derive(Debug)]
struct GoldenBackend {
    feat: Vec<f32>,
    probabilities: Vec<f32>,
    frames: usize,
    worst_feat: std::sync::Mutex<f32>,
    calls: AtomicUsize,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl InferenceBackend for GoldenBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        mut inputs: TensorMap,
    ) -> Result<TensorMap> {
        assert_eq!(model, ModelId::FireRedVadAed);
        let bands = inputs.remove("feat").unwrap().to_f32()?;
        let mask = inputs.remove("mask").unwrap().to_f32()?;
        assert!(mask[..self.frames].iter().all(|value| *value == 1.0));
        assert!(mask[self.frames..].iter().all(|value| *value == 0.0));
        let mut worst = self.worst_feat.lock().unwrap();
        for frame in 0..self.frames {
            for band in 0..INPUT_DIM {
                let delta = (bands[band * WINDOW_FRAMES + frame]
                    - self.feat[frame * INPUT_DIM + band])
                    .abs();
                *worst = worst.max(delta);
            }
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut probs = vec![0.0_f32; 3 * WINDOW_FRAMES];
        for frame in 0..self.frames {
            for class in 0..3 {
                probs[class * WINDOW_FRAMES + frame] = self.probabilities[frame * 3 + class];
            }
        }
        let mut output = TensorMap::new();
        output.insert(
            "probs".into(),
            Tensor::from_f32(vec![1, 3, 1, WINDOW_FRAMES as i64], probs)?,
        );
        Ok(output)
    }
}

fn run_full_pipeline(stem: &str, expected_frames: usize) -> Option<Vec<f32>> {
    let int16_samples = read_i16(stem, "audio_int16.bin")?;
    assert_eq!(frames_for_samples(int16_samples.len()), expected_frames);
    let mut feat = read_f32(stem, "fbank.bin").unwrap();
    assert!(Cmvn::apply(&mut feat, expected_frames));
    let backend = Arc::new(GoldenBackend {
        feat,
        probabilities: read_f32(stem, "probs.bin").unwrap(),
        frames: expected_frames,
        worst_feat: std::sync::Mutex::new(0.0),
        calls: AtomicUsize::new(0),
    });
    let normalized = int16_samples
        .into_iter()
        .map(|sample| f32::from(sample) / 32_768.0)
        .collect();
    let detector = AudioEventDetector::new(backend.clone());
    let result = futures_executor::block_on(detector.detect_audio(
        &AudioFrames::new(normalized, SAMPLE_RATE),
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
    let worst = *backend.worst_feat.lock().unwrap();
    eprintln!(
        "[aed-feat] {stem}: the processor's feat vs kaldi-native-fbank + CMVN, max |Δ| {worst:.3e}"
    );
    assert!(worst < 2e-3, "{stem}: {worst}");
    assert_eq!(result.frame_probs.len(), expected_frames);
    Some(
        result
            .frame_probs
            .into_iter()
            .flat_map(|row| [row.speech, row.singing, row.music])
            .collect(),
    )
}

#[test]
fn test_hairdryer_full_pipeline() {
    let Some(probabilities) = run_full_pipeline("hairdryer", 2_011) else {
        return;
    };
    assert_eq!(probabilities, read_f32("hairdryer", "probs.bin").unwrap());
    let speech_ratio = probabilities
        .chunks_exact(3)
        .filter(|row| row[0] > SPEECH_THRESHOLD)
        .count() as f32
        / 2_011.0;
    assert!(speech_ratio > 0.95);
}

#[test]
fn test_sec5_full_pipeline() {
    let Some(probabilities) = run_full_pipeline("sec5", 3_098) else {
        return;
    };
    assert_eq!(probabilities, read_f32("sec5", "probs.bin").unwrap());
    let speech_ratio = probabilities
        .chunks_exact(3)
        .filter(|row| row[0] > SPEECH_THRESHOLD)
        .count() as f32
        / 3_098.0;
    let music_ratio = probabilities
        .chunks_exact(3)
        .filter(|row| row[2] > MUSIC_THRESHOLD)
        .count() as f32
        / 3_098.0;
    assert!(speech_ratio < 0.20);
    assert!(music_ratio > 0.95);
}

#[derive(Debug, Default)]
struct ConstantBackend {
    calls: AtomicUsize,
    max_input_elements: AtomicUsize,
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl InferenceBackend for ConstantBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        assert_eq!(model, ModelId::FireRedVadAed);
        let elements = inputs
            .values()
            .map(|tensor| tensor.element_count() as usize)
            .sum();
        self.max_input_elements
            .fetch_max(elements, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut probabilities = vec![0.6_f32; WINDOW_FRAMES];
        probabilities.extend(vec![0.1_f32; WINDOW_FRAMES]);
        probabilities.extend(vec![0.7_f32; WINDOW_FRAMES]);
        let mut output = TensorMap::new();
        output.insert(
            "probs".into(),
            Tensor::from_f32(vec![1, 3, 1, WINDOW_FRAMES as i64], probabilities)?,
        );
        Ok(output)
    }
}

fn detect_synthetic(duration_seconds: usize) -> (AudioEventDetectionResult, Arc<ConstantBackend>) {
    let backend = Arc::new(ConstantBackend::default());
    let detector = AudioEventDetector::new(backend.clone());
    let audio = AudioFrames::new(
        vec![0.0; duration_seconds * SAMPLE_RATE as usize],
        SAMPLE_RATE,
    );
    let result =
        futures_executor::block_on(detector.detect_audio(&audio, ProgressSink::noop())).unwrap();
    (result, backend)
}

#[test]
fn test_ten_minute_input_processes_end_to_end() {
    let (result, backend) = detect_synthetic(600);
    assert!((result.duration_seconds - 600.0).abs() <= 1.0);
    assert_eq!(
        result.frame_probs.len(),
        frames_for_samples(600 * SAMPLE_RATE as usize)
    );
    assert_eq!(
        backend.calls.load(Ordering::Relaxed),
        result.frame_probs.len().div_ceil(WINDOW_STRIDE)
    );
    for (event, threshold) in [
        (AudioEvent::Speech, SPEECH_THRESHOLD),
        (AudioEvent::Music, MUSIC_THRESHOLD),
    ] {
        assert!((AudioEventSegmenter::coverage(&result, event, threshold) - 1.0).abs() < 1e-9);
    }
    assert_eq!(
        AudioEventSegmenter::coverage(&result, AudioEvent::Singing, SINGING_THRESHOLD),
        0.0
    );
}

#[test]
fn test_one_second_is_one_window() {
    let (result, backend) = detect_synthetic(1);
    assert_eq!(result.frame_probs.len(), 98);
    assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn test_ten_minute_input_stays_bounded_in_memory() {
    // The backend seam makes the memory invariant deterministic: regardless
    // of input length, only one fixed door can be live per sequential
    // inference call. This is the Rust equivalent of the Swift RSS guard.
    let (_, backend) = detect_synthetic(600);
    assert_eq!(
        backend.max_input_elements.load(Ordering::Relaxed),
        (INPUT_DIM + 1) * WINDOW_FRAMES
    );
}
