//! Scenario-for-scenario translations of the three Swift MusicHighlighter
//! test files.

use std::sync::Arc;

use async_trait::async_trait;
use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{ProgressSink, Result};
use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_test_fixtures as fixtures;

use super::MusicHighlighter;
use super::config::{CHUNK_FRAMES, FEATURE_DIM, MAX_CHUNKS, N_MELS, SAMPLE_RATE};
use super::mel::compute_log_mel;
use super::pos_enc::positional_encoding;
use super::window_search::best_window;
use crate::AudioSource;

fn read_f32(name: &str) -> Option<Vec<f32>> {
    fixtures::f32_le(&format!("MusicHighlighter/{name}.bin"))
}

fn metadata() -> Option<serde_json::Value> {
    serde_json::from_slice(&fixtures::bytes("MusicHighlighter/test_metadata.json")?).ok()
}

fn metadata_usize(key: &str) -> Option<usize> {
    Some(metadata()?.get(key)?.as_u64()? as usize)
}

fn metadata_f32(key: &str) -> Option<f32> {
    Some(metadata()?.get(key)?.as_f64()? as f32)
}

fn max_difference(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
}

fn deterministic_values(count: usize, maximum: f32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 0x00ff_ffff_u32 as f32 * maximum
        })
        .collect()
}

#[derive(Debug, Clone)]
struct LogitBackend {
    logits: Vec<f32>,
}

impl Default for LogitBackend {
    fn default() -> Self {
        Self {
            logits: (0..MAX_CHUNKS)
                .map(|index| (index % 10) as f32 * 0.25)
                .collect(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl InferenceBackend for LogitBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        assert_eq!(model, ModelId::MusicHighlighter);
        assert_eq!(inputs["chunks"].shape, vec![MAX_CHUNKS as i64, 129, 128]);
        assert_eq!(inputs["pos_enc"].shape, vec![1, MAX_CHUNKS as i64, 256]);
        let mut logits = self.logits.clone();
        logits.resize(MAX_CHUNKS, 0.0);
        let mut outputs = TensorMap::new();
        outputs.insert(
            "logits".into(),
            Tensor::from_f32(vec![MAX_CHUNKS as i64], logits)?,
        );
        Ok(outputs)
    }
}

fn highlighter() -> MusicHighlighter {
    MusicHighlighter::new(Arc::new(LogitBackend::default()))
}

fn short_tone(seconds: f64, sample_rate: u32) -> AudioFrames {
    let sample_count = (seconds * f64::from(sample_rate)) as usize;
    AudioFrames::new(
        (0..sample_count)
            .map(|index| 0.01 * (index as f32 * 0.05).sin())
            .collect(),
        sample_rate,
    )
}

/// Prefer decoding the real mp3; fall back to the pre-decoded f32 dump when
/// the platform decoder cannot open it.
fn techno_audio() -> Option<AudioFrames> {
    if let Some(path) = fixtures::path("techno-1-30s.mp3")
        && let Ok(audio) = futures_executor::block_on(AudioSource::platform_default().decode(&path))
    {
        return Some(audio);
    }
    Some(AudioFrames::new(read_f32("test_audio")?, SAMPLE_RATE))
}

fn assert_highlight_detection_on_techno() {
    let Some(audio) = techno_audio() else {
        return;
    };
    let result =
        futures_executor::block_on(highlighter().detect(audio, vec![10, 15], ProgressSink::noop()))
            .unwrap();
    assert!(result.duration >= 28);
    assert!(result.duration <= 31);
    assert_eq!(result.saliency.len(), result.duration);
    assert!(
        result
            .saliency
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );
    assert!((result.saliency.iter().copied().fold(0.0, f32::max) - 1.0).abs() <= 0.001);
    assert_eq!(result.highlights.len(), 2);

    let h10 = &result.highlights[0];
    assert_eq!(h10.window, 10);
    assert!(h10.start >= 0.0);
    assert!(h10.duration > 0.0);
    assert!(h10.start + h10.duration <= result.duration as f32);

    let h15 = &result.highlights[1];
    assert_eq!(h15.window, 15);
    assert!(h15.start >= 0.0);
    assert!(h15.duration > 0.0);
}

fn assert_mel_spectrogram_on_techno() {
    let Some(frames) = techno_audio() else {
        return;
    };
    let audio = if frames.sample_rate == SAMPLE_RATE {
        frames.samples
    } else {
        resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE).unwrap()
    };
    assert!(audio.len() > SAMPLE_RATE as usize);
    let mel = compute_log_mel(&audio).unwrap();
    assert!(!mel.is_empty());
    assert_eq!(mel[0].len(), N_MELS);
    assert!(mel[mel.len() / 2].iter().all(|value| *value >= 0.0));
}

fn assert_model_with_random_input() {
    let n_chunks = 10;
    let chunks = deterministic_values(n_chunks * CHUNK_FRAMES * N_MELS, 10.0);
    let positions = positional_encoding(n_chunks, FEATURE_DIM);
    let scores =
        futures_executor::block_on(highlighter().run_model(&chunks, &positions, n_chunks)).unwrap();
    assert_eq!(scores.len(), n_chunks);
    assert!((scores.iter().sum::<f32>() - 1.0).abs() <= 0.01);
    assert!(scores.iter().all(|score| *score >= 0.0));
}

mod engine_pipeline_tests {
    use super::*;

    #[test]
    fn test_detect_returns_empty_for_short_audio() {
        let result = futures_executor::block_on(highlighter().detect(
            short_tone(2.1, 44_100),
            vec![10, 15],
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(result.highlights.is_empty());
        assert!(result.saliency.is_empty());
    }

    #[test]
    fn test_highlight_detection_on_techno() {
        assert_highlight_detection_on_techno();
    }

    #[test]
    fn test_mel_spectrogram_on_techno() {
        assert_mel_spectrogram_on_techno();
    }

    #[test]
    fn test_model_with_random_input() {
        assert_model_with_random_input();
    }
}

mod integration_pipeline_tests {
    use super::*;

    #[test]
    fn test_highlight_detection_on_techno() {
        assert_highlight_detection_on_techno();
    }

    #[test]
    fn test_mel_spectrogram_on_techno() {
        assert_mel_spectrogram_on_techno();
    }

    #[test]
    fn test_model_with_random_input() {
        assert_model_with_random_input();
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_mel_matches_python() {
        let Some(audio) = read_f32("test_audio") else {
            return;
        };
        let expected = fixtures::require_f32_le("MusicHighlighter/test_mel.bin");
        let expected_frames = metadata_usize("mel_frames").unwrap();
        let mel = compute_log_mel(&audio).unwrap();
        assert_eq!(mel.len(), expected_frames);
        assert!((mel[100][0] - 8.680_281).abs() <= 0.001);
        assert!((mel[100][1] - 8.855_596).abs() <= 0.001);
        assert!((mel[100][4] - 8.564_172).abs() <= 0.001);
        let flat = mel.into_iter().flatten().collect::<Vec<_>>();
        assert!(max_difference(&flat, &expected) < 0.001);
    }

    #[test]
    fn test_mel_empty_audio() {
        assert!(compute_log_mel(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_mel_single_sample() {
        let mel = compute_log_mel(&[0.5]).unwrap();
        assert!(!mel.is_empty());
        assert_eq!(mel[0].len(), N_MELS);
    }

    #[test]
    fn test_positional_encoding_matches_python() {
        let Some(expected) = read_f32("test_pos_enc") else {
            return;
        };
        let n_chunks = metadata_usize("n_chunks").unwrap();
        let dimension = metadata_usize("dim").unwrap();
        let encoding = positional_encoding(n_chunks, dimension);
        assert_eq!(encoding.len() / dimension, n_chunks);
        assert!(encoding[..dimension].iter().all(|value| *value == 0.0));
        assert!((encoding[dimension] - 0.841_470_98).abs() <= 1.0e-5);
        assert!((encoding[dimension + 1] - 0.540_302_31).abs() <= 1.0e-5);
        assert!(max_difference(&encoding, &expected) < 1.0e-5);
    }

    #[test]
    fn test_positional_encoding_single_position() {
        let encoding = positional_encoding(1, 256);
        assert_eq!(encoding.len(), 256);
        assert!(encoding.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_softmax_round_trips_reference_logits() {
        let Some(chunks) = read_f32("test_chunks") else {
            return;
        };
        let expected = fixtures::require_f32_le("MusicHighlighter/test_scores.bin");
        let n_chunks = metadata_usize("n_chunks").unwrap();
        let chunk_length = metadata_usize("chunk_length").unwrap();
        assert_eq!(chunk_length, CHUNK_FRAMES);
        let positions = positional_encoding(n_chunks, FEATURE_DIM);
        let logits = expected
            .iter()
            .map(|score| score.max(f32::MIN_POSITIVE).ln())
            .collect();
        let model = MusicHighlighter::new(Arc::new(LogitBackend { logits }));
        let scores =
            futures_executor::block_on(model.run_model(&chunks, &positions, n_chunks)).unwrap();
        assert_eq!(scores.len(), n_chunks);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() <= 0.01);
        assert!(max_difference(&scores, &expected) < 0.001);
        let peak = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index);
        assert_eq!(peak, Some(5));
    }

    #[test]
    fn test_best_window_matches_python() {
        let Some(saliency) = read_f32("test_saliency") else {
            return;
        };
        let expected_start = metadata_f32("highlight_start").unwrap();
        let expected_end = metadata_f32("highlight_end").unwrap();
        let window = metadata_usize("highlight_window").unwrap() as isize;
        assert!((saliency[15] - 1.0).abs() <= 0.001);
        assert_eq!(
            best_window(&saliency, window),
            (expected_start, expected_end)
        );
    }

    #[test]
    fn test_best_window_picks_highest_region() {
        assert_eq!(
            best_window(&[0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1], 3),
            (2.0, 5.0)
        );
    }

    #[test]
    fn test_best_window_zero_length() {
        assert_eq!(best_window(&[0.5, 0.8], 0), (0.0, 0.0));
    }

    #[test]
    fn test_best_window_exceeds_saliency() {
        assert_eq!(best_window(&[0.5, 0.8, 0.3], 10), (0.0, 3.0));
    }

    #[test]
    fn test_best_window_negative_length() {
        assert_eq!(best_window(&[0.5, 0.8], -5), (0.0, 0.0));
    }

    #[test]
    fn test_best_window_empty_saliency() {
        assert_eq!(best_window(&[], 5), (0.0, 0.0));
    }
}

/// These cases use the published ONNX graph. The controlled LogitBackend
/// above tests postprocessing and cannot establish model-score parity.
#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod real_ort_integration_tests {
    use super::*;
    use ferment_engine_infer::{ModelPaths, OrtBackend, OrtExecutionProvider};

    fn real_backend() -> Arc<OrtBackend> {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "name the pinned ORT runtime"
        );
        let root = std::path::PathBuf::from(
            std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
                .expect("name the frozen published model directory"),
        );
        let graph = root.join("music_highlighter.onnx");
        assert!(graph.is_file(), "required graph: {}", graph.display());
        Arc::new(OrtBackend::with_execution_provider(
            Arc::new(ModelPaths::new().with_path(
                ModelId::MusicHighlighter,
                ModelVersion::Latest,
                graph,
            )),
            OrtExecutionProvider::Cpu,
        ))
    }

    fn required_fixture(name: &str) -> std::path::PathBuf {
        assert!(
            std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").is_some(),
            "name the immutable Swift engine fixture tree; no fallback"
        );
        fixtures::require_path(name)
    }

    fn real_techno() -> AudioFrames {
        futures_executor::block_on(
            AudioSource::platform_default().decode(&required_fixture("techno-1-30s.mp3")),
        )
        .expect("the production platform decoder must open the real MP3")
    }

    #[test]
    #[ignore = "requires explicit pinned ORT, published model and Swift fixtures"]
    fn real_model_scores_match_the_python_fixture() {
        required_fixture("MusicHighlighter/test_metadata.json");
        let chunks = fixtures::require_f32_le("MusicHighlighter/test_chunks.bin");
        let expected = fixtures::require_f32_le("MusicHighlighter/test_scores.bin");
        let n_chunks = metadata_usize("n_chunks").unwrap();
        assert_eq!(metadata_usize("chunk_length").unwrap(), CHUNK_FRAMES);
        assert_eq!(chunks.len(), n_chunks * CHUNK_FRAMES * N_MELS);
        assert_eq!(expected.len(), n_chunks);
        let backend = real_backend();
        let model = MusicHighlighter::new(backend.clone());
        let scores = futures_executor::block_on(model.run_model(
            &chunks,
            &positional_encoding(n_chunks, FEATURE_DIM),
            n_chunks,
        ))
        .unwrap();
        assert_eq!(scores.len(), n_chunks);
        assert!(scores.iter().all(|value| value.is_finite()));
        assert!((scores.iter().sum::<f32>() - 1.0).abs() <= 0.01);
        let delta = max_difference(&scores, &expected);
        assert!(
            delta < 0.001,
            "real ORT scores differ from independent Python fixture: {delta}"
        );
        let peak = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        assert_eq!(peak, Some(5));
        assert_eq!(
            backend.selected_provider_for(ModelId::MusicHighlighter),
            Some(OrtExecutionProvider::Cpu)
        );
    }

    #[test]
    #[ignore = "requires explicit pinned ORT and published model"]
    fn real_model_scores_are_probabilities_for_ten_random_chunks() {
        let n_chunks = 10;
        let model = MusicHighlighter::new(real_backend());
        let scores = futures_executor::block_on(model.run_model(
            &deterministic_values(n_chunks * CHUNK_FRAMES * N_MELS, 10.0),
            &positional_encoding(n_chunks, FEATURE_DIM),
            n_chunks,
        ))
        .unwrap();
        assert_eq!(scores.len(), n_chunks);
        assert!(
            scores
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!((scores.iter().sum::<f32>() - 1.0).abs() <= 0.01);
    }

    #[test]
    #[ignore = "requires explicit pinned ORT, published model and real MP3 fixture"]
    fn real_techno_pipeline_is_bounded_and_normalized() {
        let backend = real_backend();
        let result = futures_executor::block_on(MusicHighlighter::new(backend.clone()).detect(
            real_techno(),
            vec![10, 15],
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!((28..=31).contains(&result.duration));
        assert_eq!(result.saliency.len(), result.duration);
        assert!(
            result
                .saliency
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
        assert!((result.saliency.iter().copied().fold(0.0, f32::max) - 1.0).abs() <= 0.001);
        assert_eq!(result.highlights.len(), 2);
        for (highlight, window) in result.highlights.iter().zip([10, 15]) {
            assert_eq!(highlight.window, window);
            assert!(highlight.start >= 0.0 && highlight.duration > 0.0);
            assert!(highlight.start + highlight.duration <= result.duration as f32);
        }
        assert_eq!(
            backend.selected_provider_for(ModelId::MusicHighlighter),
            Some(OrtExecutionProvider::Cpu)
        );
        assert_eq!(backend.cache_snapshot().resident, 1);
    }

    #[test]
    #[ignore = "requires the explicit Swift fixture tree and real MP3 decoder"]
    fn real_techno_mel_has_the_requested_layout() {
        let frames = real_techno();
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE).unwrap()
        };
        assert!(audio.len() > SAMPLE_RATE as usize);
        let mel = compute_log_mel(&audio).unwrap();
        assert!(!mel.is_empty());
        assert_eq!(mel[0].len(), N_MELS);
        assert!(
            mel[mel.len() / 2]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
    }
    #[test]
    #[ignore = "requires admitted ORT, music_highlighter and both original export golden cases"]
    fn current_scores_and_top_chunk_match_both_independent_model_doors() {
        use ferment_engine_test_fixtures::export;
        let audio = export::require_f32_le("beat_this/goldens/beat_this_audio_in.bin");
        let mel =
            export::require_f32_le("music_highlighter/goldens/music_highlighter_music_mel_out.f32");
        let count = (audio.len() / SAMPLE_RATE as usize) / 3;
        assert_eq!(count, 10);
        let length = (mel.len() / N_MELS) / count;
        let mut chunks = vec![0.0; count * CHUNK_FRAMES * N_MELS];
        for j in 0..count {
            for t in 0..length.min(CHUNK_FRAMES) {
                let from = (j * length + t) * N_MELS;
                let to = (j * CHUNK_FRAMES + t) * N_MELS;
                chunks[to..to + N_MELS].copy_from_slice(&mel[from..from + N_MELS]);
            }
        }
        let track = export::require_f32_le(
            "music_highlighter/goldens/music_highlighter_track_chunks_in.f32",
        );
        assert_eq!(track.len(), 37 * CHUNK_FRAMES * N_MELS);
        let model = MusicHighlighter::new(real_backend());
        for (label, chunks, count) in [("music", chunks, 10), ("track", track, 37)] {
            let logits = export::require_f32_le(&format!(
                "music_highlighter/goldens/music_highlighter_{label}_logits_out.f32"
            ));
            assert_eq!(logits.len(), MAX_CHUNKS);
            let head = &logits[..count];
            let maximum = head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = head
                .iter()
                .map(|value| (value - maximum).exp())
                .collect::<Vec<_>>();
            let sum = exponentials.iter().sum::<f32>();
            let expected = exponentials
                .iter()
                .map(|value| value / sum)
                .collect::<Vec<_>>();
            let actual = futures_executor::block_on(model.run_model(
                &chunks,
                &positional_encoding(count, FEATURE_DIM),
                count,
            ))
            .unwrap();
            assert_eq!(actual.len(), count);
            assert!(actual.iter().all(|value| value.is_finite()));
            let error = max_difference(&actual, &expected);
            assert!(error < 0.01, "{label}: max score delta {error}");
            let top = |values: &[f32]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(index, _)| index)
            };
            assert_eq!(top(&actual), top(&expected), "{label}: top chunk moved");
        }
    }
}
