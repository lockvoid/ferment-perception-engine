//! End-to-end processor/model contract smoke tests for the real Core ML fleet.
//!
//! The fleet stays outside this repository. These tests are ignored by the
//! regular suite; run them on macOS with `FERMENT_ENGINE_MODELS_DIR` set (or
//! with the sibling Swift checkout in its normal location).

#![cfg(all(feature = "coreml", target_vendor = "apple"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{AudioEvent, AudioEventSegment, ProgressSink, SpeechTranscribing};
use ferment_engine_infer::{Backend, CoreMlBackend, ModelId, ModelPaths, ModelVersion};
use ferment_engine_processors::parakeet::host_decode;
use ferment_engine_processors::{
    AudioEventDetector, AudioSemanticEncoder, AudioSource, BeatThis, DeepFilterNet,
    MusicHighlighter, MusicTagger, Parakeet, VideoSceneDetector,
};

fn model_directory() -> PathBuf {
    std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ferment-cuts-ios/Packages/FermentEngine/LocalModels")
        })
}

fn backend() -> Backend {
    let directory = model_directory();
    assert!(
        directory.is_dir(),
        "Core ML fleet not found at {}; set FERMENT_ENGINE_MODELS_DIR",
        directory.display()
    );
    let mut paths = ModelPaths::new();
    for (id, file_name) in [
        (ModelId::BeatThis1, "beat_this_1.mlmodelc"),
        (ModelId::BeatThis2, "beat_this_2.mlmodelc"),
        (ModelId::BeatThis3, "beat_this_3.mlmodelc"),
        (ModelId::ParakeetEncoder, "parakeet_encoder.mlmodelc"),
        (ModelId::Dfnet, "dfnet.mlmodelc"),
        (ModelId::MusicHighlighter, "music_highlighter.mlmodelc"),
        (ModelId::DashengBase, "dasheng_base_mel.mlmodelc"),
        (ModelId::MusicTaggerHeads, "music_tagger_heads.mlmodelc"),
        (ModelId::Autoshot, "autoshot.mlmodelc"),
        (ModelId::FireRedVadAed, "fireredvad_aed.mlmodelc"),
    ] {
        paths.insert(id, ModelVersion::Latest, directory.join(file_name));
    }
    Arc::new(
        // The oracles must exercise the SHIPPED per-model pins, not All —
        // a wrong pin should fail here, in the suite meant to catch it.
        CoreMlBackend::new(Arc::new(paths)).with_placements(shipped_placements()),
    )
}

/// The transcriber over `backend`, its decode's weights beside the graphs.
fn parakeet(backend: Backend) -> Parakeet {
    Parakeet::new(backend, model_directory().join(host_decode::FILE))
}

fn tone(sample_rate: u32, seconds: f64) -> AudioFrames {
    let count = (f64::from(sample_rate) * seconds) as usize;
    let samples = (0..count)
        .map(|index| {
            let phase = index as f32 * 440.0 * std::f32::consts::TAU / sample_rate as f32;
            phase.sin() * 0.01
        })
        .collect();
    AudioFrames::new(samples, sample_rate)
}

fn techno_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures/Audio/techno-1.wav")
}

fn synthetic_speech_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures/Audio/synthetic-speech-1.wav",
    )
}

fn resident_memory_bytes() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p"])
        .arg(std::process::id().to_string())
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse::<u64>()
        .unwrap()
        * 1024
}

fn parakeet_hairdryer_pcm_fixture() -> PathBuf {
    std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures")
        })
        .join("AudioEventDetector/hairdryer/audio_int16.bin")
}

async fn hairdryer_audio() -> AudioFrames {
    if let Some(fixture) = std::env::var_os("FERMENT_HAIRDRYER_AUDIO") {
        let fixture = PathBuf::from(fixture);
        assert!(
            fixture.is_file(),
            "FERMENT_HAIRDRYER_AUDIO not found at {}",
            fixture.display()
        );
        return AudioSource::platform_default()
            .decode(&fixture)
            .await
            .unwrap();
    }

    let fixture = parakeet_hairdryer_pcm_fixture();
    assert!(
        fixture.is_file(),
        "hairdryer PCM oracle not found at {}; set FERMENT_ENGINE_FIXTURES_DIR",
        fixture.display()
    );
    let samples = std::fs::read(fixture)
        .unwrap()
        .chunks_exact(2)
        .map(|chunk| f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32_768.0)
        .collect();
    AudioFrames::new(samples, 16_000)
}

#[test]
#[ignore = "requires the external 1 GB Core ML model fleet"]
fn audio_processors_accept_real_coreml_artifacts() {
    let backend = backend();
    futures_executor::block_on(async {
        let beats = BeatThis::new(backend.clone())
            .detect(tone(22_050, 2.0), ProgressSink::noop())
            .await
            .unwrap();
        assert!(beats.beats.iter().all(|time| time.is_finite()));

        let embedding = AudioSemanticEncoder::new(backend.clone())
            .encode(tone(16_000, 1.0), false, ProgressSink::noop())
            .await
            .unwrap();
        assert_eq!(embedding.pooled.len(), 768);
        MusicTagger::new(backend.clone())
            .raw_probabilities(&embedding.pooled)
            .await
            .unwrap();

        let highlights = MusicHighlighter::new(backend.clone())
            .detect(tone(22_050, 6.0), vec![3], ProgressSink::noop())
            .await
            .unwrap();
        assert_eq!(highlights.saliency.len(), 6);

        let denoised = DeepFilterNet::new(backend.clone())
            .denoise(tone(48_000, 0.1), ProgressSink::noop())
            .await
            .unwrap();
        assert!(!denoised.samples.is_empty());

        let events = AudioEventDetector::new(backend.clone())
            .detect_audio(&tone(16_000, 0.1), ProgressSink::noop())
            .await
            .unwrap();
        assert!(!events.frame_probs.is_empty());

        let transcription = parakeet(backend)
            .transcribe(tone(16_000, 0.2), ProgressSink::noop())
            .await
            .unwrap();
        assert!(
            transcription
                .captions
                .iter()
                .all(|caption| { caption.start_time.is_finite() && caption.end_time.is_finite() })
        );
    });
}

#[test]
#[ignore = "requires the external Core ML Parakeet fleet plus FERMENT_ENGINE_FIXTURES_DIR (or FERMENT_HAIRDRYER_AUDIO)"]
fn test_transcribe_hairdryer_audio() {
    let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = stages.clone();
    let result = futures_executor::block_on(async {
        let audio = if let Some(fixture) = std::env::var_os("FERMENT_HAIRDRYER_AUDIO") {
            let fixture = PathBuf::from(fixture);
            assert!(
                fixture.is_file(),
                "FERMENT_HAIRDRYER_AUDIO not found at {}",
                fixture.display()
            );
            AudioSource::platform_default()
                .decode(&fixture)
                .await
                .unwrap()
        } else {
            let fixture = parakeet_hairdryer_pcm_fixture();
            assert!(
                fixture.is_file(),
                "hairdryer PCM oracle not found at {}",
                fixture.display()
            );
            let samples = std::fs::read(&fixture)
                .unwrap()
                .chunks_exact(2)
                .map(|chunk| f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32_768.0)
                .collect();
            AudioFrames::new(samples, 16_000)
        };
        parakeet(backend())
            .transcribe(
                audio,
                ProgressSink::from_callback(move |event| {
                    captured.lock().unwrap().push(event.stage);
                }),
            )
            .await
            .unwrap()
    });
    let words = result
        .captions
        .iter()
        .flat_map(|caption| &caption.words)
        .collect::<Vec<_>>();

    assert!(!result.transcription.is_empty());
    assert!(words.len() > 10);
    let average_confidence =
        words.iter().map(|word| word.confidence).sum::<f32>() / words.len() as f32;
    assert!(average_confidence > 0.5);
    for word in &words {
        assert!(word.start_time >= 0.0);
        assert!(word.end_time >= word.start_time);
        assert!(word.confidence > 0.0);
        assert!(word.confidence <= 1.0);
        assert!(!word.text.is_empty());
    }
    for pair in words.windows(2) {
        assert!(pair[1].start_time >= pair[0].start_time);
    }
    assert!(words.last().unwrap().end_time > 10.0);
    let text = result.transcription.to_lowercase();
    assert!(
        ["фен", "волос", "девочки", "дайс"]
            .iter()
            .any(|expected| text.contains(expected))
    );
    assert!(!stages.lock().unwrap().is_empty());
}

#[test]
#[ignore = "requires the external Core ML fleet and a real speech fixture"]
fn test_speech_segment_timestamps_inside_segment() {
    futures_executor::block_on(async {
        let segment_start = 2.0;
        let segment_end = 8.0;
        let segments = [AudioEventSegment {
            event: AudioEvent::Speech,
            start_time: segment_start,
            duration: segment_end - segment_start,
            confidence: 1.0,
        }];
        let result = SpeechTranscribing::transcribe(
            &parakeet(backend()),
            &synthetic_speech_fixture(),
            Some(&segments),
            ProgressSink::noop(),
        )
        .await
        .unwrap();
        assert!(!result.captions.is_empty());
        let words = result
            .captions
            .iter()
            .flat_map(|caption| &caption.words)
            .collect::<Vec<_>>();
        assert!(words.len() > 1);
        let slack = 1.0 + 0.1;
        for word in words {
            assert!(word.start_time >= segment_start - slack);
            assert!(word.end_time() <= segment_end + slack);
        }
    });
}

#[test]
#[ignore = "requires the external Core ML fleet and the long-form speech fixture"]
fn test_transcribe_multi_chunk_audio_end_to_end() {
    futures_executor::block_on(async {
        let source = AudioSource::platform_default()
            .decode(&synthetic_speech_fixture())
            .await
            .unwrap();
        let source_samples = source.samples.clone();
        let mut samples = Vec::with_capacity(source_samples.len() * 6);
        for _ in 0..6 {
            samples.extend_from_slice(&source_samples);
        }
        let audio = AudioFrames::new(samples, source.sample_rate);
        let duration_seconds = audio.duration_secs();
        assert!(duration_seconds > 15.0);

        let result = parakeet(backend())
            .transcribe(audio, ProgressSink::noop())
            .await
            .unwrap();
        assert!(!result.transcription.is_empty());
        assert!(result.captions.len() > 5);
        let words = result
            .captions
            .iter()
            .flat_map(|caption| &caption.words)
            .collect::<Vec<_>>();
        assert!(words.len() > 30);
        for pair in words.windows(2) {
            assert!(pair[1].start_time >= pair[0].start_time);
        }
        if let (Some(first), Some(last)) = (words.first(), words.last()) {
            assert!(last.end_time - first.start_time > duration_seconds * 0.5);
        }
        for pair in result.captions.windows(2) {
            assert_ne!(pair[0].text.trim(), pair[1].text.trim());
        }
        assert!(result.language.is_some());
    });
}

#[test]
#[ignore = "20-minute Core ML stress gate; run manually on Apple Silicon outside CI"]
fn test_twenty_minute_synthetic_audio_completes_within_memory_budget() {
    const DURATION_SECONDS: f64 = 20.0 * 60.0;
    const MEMORY_BUDGET_BYTES: u64 = 600 * 1024 * 1024;

    let baseline = resident_memory_bytes();
    let audio = tone(16_000, DURATION_SECONDS);
    assert!((audio.duration_secs() - DURATION_SECONDS).abs() <= 0.05);
    let started = std::time::Instant::now();
    let result =
        futures_executor::block_on(parakeet(backend()).transcribe(audio, ProgressSink::noop()))
            .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    let growth = resident_memory_bytes().saturating_sub(baseline);

    let _ = result.transcription;
    assert!(growth < MEMORY_BUDGET_BYTES);
    assert!(elapsed < DURATION_SECONDS * 1.5);
}

#[test]
#[ignore = "requires the external 1 GB Core ML model fleet"]
fn scene_detector_accepts_real_coreml_artifact() {
    let result = futures_executor::block_on(VideoSceneDetector::new(backend()).detect(
        &[0; ferment_engine_processors::video_scene_detector::config::PIXELS_PER_FRAME],
        1,
        30.0,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(result.total_frames, 1);
}

#[test]
#[ignore = "requires the external Core ML fleet and techno source fixture"]
fn test_tag_techno_embedding_returns_valid_shape() {
    futures_executor::block_on(async {
        let backend = backend();
        let mut audio = AudioSource::platform_default()
            .decode(&techno_fixture())
            .await
            .unwrap();
        audio.samples.truncate(audio.sample_rate as usize * 30);
        let embedding = AudioSemanticEncoder::new(backend.clone())
            .encode(audio, false, ProgressSink::noop())
            .await
            .unwrap();
        let result = MusicTagger::new(backend)
            .tag(&embedding.pooled, 5)
            .await
            .unwrap();
        for tags in [&result.genre, &result.mood, &result.instrument] {
            assert!(tags.len() <= 5);
            assert!(tags.windows(2).all(|pair| pair[0].score >= pair[1].score));
            assert!(tags.iter().all(|tag| (0.0..=1.0).contains(&tag.score)));
        }
    });
}

#[test]
#[ignore = "requires the external Core ML fleet and techno source fixture"]
fn test_tag_techno_hits_electronic_genre() {
    futures_executor::block_on(async {
        let backend = backend();
        let mut audio = AudioSource::platform_default()
            .decode(&techno_fixture())
            .await
            .unwrap();
        audio.samples.truncate(audio.sample_rate as usize * 30);
        let embedding = AudioSemanticEncoder::new(backend.clone())
            .encode(audio, false, ProgressSink::noop())
            .await
            .unwrap();
        let result = MusicTagger::new(backend)
            .tag(&embedding.pooled, 10)
            .await
            .unwrap();
        let electronic_family = [
            "techno",
            "electronic",
            "electronica",
            "house",
            "edm",
            "dance",
            "club",
            "trance",
            "minimal",
            "downtempo",
            "idm",
            "breakbeat",
            "drumnbass",
        ];
        assert!(
            result
                .genre
                .iter()
                .any(|tag| electronic_family.contains(&tag.name.as_str())),
            "no electronic-family genre fired: {:?}",
            result
                .genre
                .iter()
                .map(|tag| (&tag.name, tag.score))
                .collect::<Vec<_>>()
        );
    });
}

#[test]
#[ignore = "requires Core ML plus FERMENT_ENGINE_FIXTURES_DIR (or FERMENT_HAIRDRYER_AUDIO)"]
fn hairdryer_fixture_produces_reasonable_output() {
    let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&stages);
    let result = futures_executor::block_on(async {
        let audio = hairdryer_audio().await;
        DeepFilterNet::new(backend())
            .denoise(
                audio,
                ProgressSink::from_callback(move |event| {
                    captured.lock().unwrap().push(event.stage);
                }),
            )
            .await
            .unwrap()
    });
    let duration = result.samples.len() as f64 / f64::from(result.sample_rate);
    assert!(
        duration > 15.0 && duration < 25.0,
        "unexpected duration {duration}"
    );
    let rms = (result
        .samples
        .iter()
        .map(|sample| sample * sample)
        .sum::<f32>()
        / result.samples.len() as f32)
        .sqrt();
    assert!(
        rms > 0.001,
        "hairdryer output is suspiciously silent: {rms}"
    );
    assert!(
        !stages.lock().unwrap().is_empty(),
        "expected progress reports"
    );
}

// FermentEngineIntegrationTests/SpeechDenoisingPipelineTests.swift
#[test]
#[ignore = "requires Core ML plus FERMENT_ENGINE_FIXTURES_DIR (or FERMENT_HAIRDRYER_AUDIO)"]
fn denoise_hairdryer_audio() {
    let stages = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&stages);
    let result = futures_executor::block_on(async {
        let audio = hairdryer_audio().await;
        DeepFilterNet::new(backend())
            .denoise(
                audio,
                ProgressSink::from_callback(move |event| {
                    captured.lock().unwrap().push(event.stage);
                }),
            )
            .await
            .unwrap()
    });

    assert!(
        !result.samples.is_empty(),
        "denoised output must not be empty"
    );
    assert!(result.sample_rate > 0, "sample rate must be positive");
    let duration_seconds = result.samples.len() as f64 / f64::from(result.sample_rate);
    assert!(
        duration_seconds > 15.0,
        "output should be ~20s, got {duration_seconds:.1}s"
    );
    assert!(
        duration_seconds < 25.0,
        "output too long: {duration_seconds:.1}s"
    );

    let maximum = result
        .samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    assert!(maximum.is_finite(), "output contains non-finite values");
    assert!(
        maximum < 2.0,
        "denoised samples should be bounded, max={maximum}"
    );
    assert!(
        maximum > 0.01,
        "denoised output is near-silent, max={maximum}"
    );

    let rms = (result
        .samples
        .iter()
        .map(|sample| sample * sample)
        .sum::<f32>()
        / result.samples.len() as f32)
        .sqrt();
    assert!(
        rms > 0.001,
        "RMS too low; denoiser may have zeroed the signal: {rms}"
    );
    assert!(
        !stages.lock().unwrap().is_empty(),
        "expected progress reports"
    );
    eprintln!(
        "denoised {} samples, {duration_seconds:.1}s, rms={rms:.4}, max={maximum:.3}",
        result.samples.len()
    );
}

#[test]
#[ignore = "requires the external Core ML fleet and techno source fixture"]
fn test_highlight_detection_on_techno() {
    futures_executor::block_on(async {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures/Audio/techno-1.wav",
        );
        let mut audio = AudioSource::platform_default()
            .decode(&fixture)
            .await
            .unwrap();
        audio.samples.truncate(audio.sample_rate as usize * 30);
        let result = MusicHighlighter::new(backend())
            .detect(audio, vec![10, 15], ProgressSink::noop())
            .await
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

        let h10 = &result.highlights[0];
        assert_eq!(h10.window, 10);
        assert!(h10.start >= 0.0);
        assert!(h10.duration > 0.0);
        assert!(h10.start + h10.duration <= result.duration as f32);

        let h15 = &result.highlights[1];
        assert_eq!(h15.window, 15);
        assert!(h15.start >= 0.0);
        assert!(h15.duration > 0.0);
    });
}

/// The fleet's Core ML pins, converted exactly the way production does in
/// `ferment-project::runtime::coreml_placements`.
fn shipped_placements() -> Vec<(
    ferment_engine_infer::ModelId,
    ferment_engine_infer::ComputeUnits,
)> {
    use ferment_engine_infer::ComputeUnits;
    use ferment_engine_models::CoreMlComputeUnits;
    ferment_engine_models::default_fleet_manifest()
        .placements(ferment_engine_models::BackendKind::CoreMl)
        .into_iter()
        .filter_map(|(id, placement)| {
            let units = match placement.compute_units()? {
                CoreMlComputeUnits::All => ComputeUnits::All,
                CoreMlComputeUnits::CpuOnly => ComputeUnits::CpuOnly,
                CoreMlComputeUnits::CpuAndGpu => ComputeUnits::CpuAndGpu,
                CoreMlComputeUnits::CpuAndNeuralEngine => ComputeUnits::CpuAndNeuralEngine,
            };
            Some((id, units))
        })
        .collect()
}
