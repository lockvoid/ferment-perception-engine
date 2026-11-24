//! Current iOS file-level audio assertions using the admitted native decoder
//! and published ONNX models. Explicit resource roots are required; no skips.
//! Host CPU execution is not Android device qualification.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use ferment_engine_core::{
    AudioEvent, AudioEventDetecting, AudioEventSegment, AudioSampleSource, LoudnessNormalizing,
    LoudnessTarget, ProgressSink,
};
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::audio_event_detector::{AudioEventSegmenter, SegmenterConfig};
use ferment_engine_processors::{
    AudioEventDetector, AudioSource, DeepFilterNet, LoudnessNormalizer, MusicHighlighter,
};
use ferment_engine_test_fixtures::{self as fixtures, export};
use futures_executor::block_on;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

fn backend(id: ModelId, name: &str) -> Arc<OrtBackend> {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "required pinned runtime"
    );
    let model = export::artifact(name).expect("required published graph");
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(id, ModelVersion::Latest, model)),
        OrtExecutionProvider::Cpu,
    ))
}
fn fixture(name: &str) -> PathBuf {
    assert!(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").is_some(),
        "required frozen original fixture root"
    );
    fixtures::require_path(name)
}
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "current-audio-files-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn tone(&self, name: &str, seconds: f64, rate: u32, frequency: f64, amplitude: f64) -> PathBuf {
        let path = self.0.join(name);
        let samples = (0..(seconds * rate as f64) as usize)
            .map(|i| {
                (amplitude * (std::f64::consts::TAU * frequency * i as f64 / rate as f64).sin())
                    as f32
            })
            .collect::<Vec<_>>();
        ferment_engine_audio::write_wav_f32(&path, &samples, rate).unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn samples(source: &dyn AudioSampleSource) -> Vec<f32> {
    let mut samples = vec![0.0; source.sample_count()];
    let n = samples.len();
    assert_eq!(source.copy_samples(&mut samples, 0, n).unwrap(), n);
    samples
}
fn rms(samples: &[f32]) -> f64 {
    (samples.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / samples.len().max(1) as f64)
        .sqrt()
}

#[test]
fn current_loudness_sine_files_reach_the_target_and_preserve_report_fields() {
    let scratch = Scratch::new();
    for amplitude in [0.045, 1.0, 0.5, 0.1] {
        let input = scratch.tone(
            &format!("sine-{amplitude}.wav"),
            5.0,
            48_000,
            1000.0,
            amplitude,
        );
        let original = std::fs::read(&input).unwrap();
        let result = block_on(LoudnessNormalizing::normalize(
            &LoudnessNormalizer,
            &input,
            LoudnessTarget::STANDARD,
            ProgressSink::noop(),
        ))
        .unwrap();
        let report = result.report;
        assert_eq!(report.target_lufs, -18.0);
        assert!((report.post_limit_lufs + 18.0).abs() <= 1.0);
        assert!(report.qa_pass);
        assert_eq!(report.spec_version, 1);
        assert_eq!(
            report.peak_ceiling_dbtp,
            LoudnessTarget::STANDARD.peak_ceiling_dbtp
        );
        assert!(report.measured_lufs.is_finite() && report.post_limit_lufs.is_finite());
        assert!(!report.within_clip_leveler_engaged);
        assert!(report.post_limit_true_peak <= report.peak_ceiling_dbtp + 0.1);
        if amplitude == 1.0 {
            assert!(report.gain_applied_db < 0.0);
        }
        assert_eq!(std::fs::read(input).unwrap(), original);
    }
}

#[test]
#[ignore = "requires admitted ORT and dfnet"]
fn current_denoiser_file_duration_bounds_short_rejection_and_resampling() {
    let scratch = Scratch::new();
    let model = DeepFilterNet::new(backend(ModelId::Dfnet, "dfnet.onnx"));
    for (name, seconds, rate, tolerance) in [
        ("length", 5.0, 48_000, 960),
        ("finite", 2.0, 48_000, 960),
        ("resample16", 1.5, 16_000, 3600),
        ("resample22", 1.5, 22_050, 3600),
        ("resample44", 1.5, 44_100, 3600),
    ] {
        let input = scratch.tone(&format!("{name}.wav"), seconds, rate, 440.0, 0.05);
        let original = std::fs::read(&input).unwrap();
        let output =
            block_on(model.denoise_to_file(&input, &scratch.0, None, ProgressSink::noop()))
                .unwrap();
        assert!(
            output
                .sample_count()
                .abs_diff((seconds * 48_000.0) as usize)
                < tolerance
        );
        let values = samples(&output);
        assert!(values.iter().all(|x| x.is_finite()));
        assert!(values.iter().map(|x| x.abs()).fold(0.0_f32, f32::max) < 2.0);
        let path = output.path().to_owned();
        drop(output);
        assert!(!path.exists());
        assert_eq!(std::fs::read(input).unwrap(), original);
    }
    let input = scratch.tone("short.wav", 0.005, 48_000, 440.0, 0.05);
    let before = std::fs::read_dir(&scratch.0).unwrap().count();
    let error = block_on(model.denoise_to_file(&input, &scratch.0, None, ProgressSink::noop()))
        .unwrap_err();
    assert!(matches!(
        error,
        ferment_engine_core::EngineError::AudioTooShort { .. }
    ));
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), before);
}

#[test]
#[ignore = "requires admitted ORT and dfnet"]
fn current_denoiser_file_keep_mask_preserves_the_one_second_speech_interior() {
    let scratch = Scratch::new();
    let input = scratch.tone("masked.wav", 3.0, 48_000, 800.0, 0.2);
    let keep = [AudioEventSegment {
        event: AudioEvent::Speech,
        start_time: 1.0,
        duration: 1.0,
        confidence: 1.0,
    }];
    let output = block_on(
        DeepFilterNet::new(backend(ModelId::Dfnet, "dfnet.onnx")).denoise_to_file(
            &input,
            &scratch.0,
            Some(&keep),
            ProgressSink::noop(),
        ),
    )
    .unwrap();
    let values = samples(&output);
    assert!(values[(0.2 * 48_000.0) as usize].abs() <= 1e-6);
    assert!(values[(2.8 * 48_000.0) as usize].abs() <= 1e-6);
    assert!(rms(&values[72_000..76_800]) > 1e-5);
}

#[test]
#[ignore = "requires admitted ORT, dfnet, original hairdryer MP3 and native decoder"]
fn current_hairdryer_file_chunk_seams_preserve_duration_level_and_progress() {
    let scratch = Scratch::new();
    let input = fixture("hairdryer-product-placement.mp3");
    let before = std::fs::read(&input).unwrap();
    let backend = backend(ModelId::Dfnet, "dfnet.onnx");
    let model = DeepFilterNet::new(backend.clone());
    let stages = Arc::new(Mutex::new(Vec::new()));
    let sink = stages.clone();
    let output = block_on(model.denoise_to_file(
        &input,
        &scratch.0,
        None,
        ProgressSink::from_callback(move |p| sink.lock().unwrap().push(p.stage)),
    ))
    .unwrap();
    let standard = samples(&output);
    let seconds = standard.len() as f64 / 48_000.0;
    assert!(seconds > 15.0 && seconds < 25.0);
    assert!(rms(&standard) > 0.001);
    assert!(!stages.lock().unwrap().is_empty());
    let chunked = block_on(
        DeepFilterNet::new(backend)
            .with_max_frames_per_chunk(300)
            .denoise_to_file(&input, &scratch.0, None, ProgressSink::noop()),
    )
    .unwrap();
    assert!(chunked.sample_count().abs_diff(output.sample_count()) < 960);
    let ratio = rms(&samples(&chunked)) / rms(&standard);
    assert!(ratio > 0.5 && ratio < 1.5, "chunked RMS ratio {ratio}");
    drop(output);
    drop(chunked);
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), 0);
    assert_eq!(std::fs::read(input).unwrap(), before);
}

#[test]
#[ignore = "requires admitted ORT, fireredvad_aed, original hairdryer/sec5 containers and native decoder"]
fn current_real_aed_files_grade_speech_and_music_decisions_after_the_model() {
    let model = AudioEventDetector::new(backend(ModelId::FireRedVadAed, "fireredvad_aed.onnx"));
    for (name, event, config) in [
        (
            "hairdryer-product-placement.mp3",
            AudioEvent::Speech,
            SegmenterConfig::SPEECH,
        ),
        ("sec5.mp4", AudioEvent::Music, SegmenterConfig::MUSIC),
    ] {
        let result = block_on(AudioEventDetecting::detect(
            &model,
            &fixture(name),
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!((result.frame_shift_seconds - 0.010).abs() <= 1e-9);
        let speech = AudioEventSegmenter::coverage(
            &result,
            AudioEvent::Speech,
            SegmenterConfig::SPEECH.threshold,
        );
        let music = AudioEventSegmenter::coverage(
            &result,
            AudioEvent::Music,
            SegmenterConfig::MUSIC.threshold,
        );
        if event == AudioEvent::Speech {
            assert!(result.frame_probs.len() > 1500);
            assert!(result.duration_seconds > 18.0 && result.duration_seconds < 22.0);
            assert!(speech > 0.95, "hairdryer speech coverage {speech}");
        } else {
            assert!(result.frame_probs.len() > 2500);
            assert!(music > 0.95, "sec5 music coverage {music}");
            assert!(speech < 0.20, "sec5 speech coverage {speech}");
        }
        let segments = AudioEventSegmenter::segments(&result, event, config);
        assert!(!segments.is_empty());
        assert!(segments.iter().map(|s| s.duration).sum::<f64>() > result.duration_seconds * 0.9);
    }
}

#[test]
#[ignore = "requires admitted ORT, music_highlighter and pinned original electronic-indie-rock MP3 supplement"]
fn current_full_track_best_highlight_windows_match_the_ios_references() {
    let path = PathBuf::from(
        std::env::var_os("FERMENT_CURRENT_AUDIO_SUPPLEMENT")
            .expect("name the separately pinned original iOS audio supplement"),
    )
    .join("electronic-indie-rock-1.mp3");
    assert!(path.is_file(), "required original full track");
    let audio = block_on(AudioSource::platform_default().decode(Path::new(&path))).unwrap();
    let result =
        block_on(
            MusicHighlighter::new(backend(ModelId::MusicHighlighter, "music_highlighter.onnx"))
                .detect(audio, vec![15, 30, 60], ProgressSink::noop()),
        )
        .unwrap();
    assert_eq!(
        result
            .highlights
            .iter()
            .map(|h| h.start as i64)
            .collect::<Vec<_>>(),
        [204, 186, 159]
    );
}
