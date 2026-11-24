//! Behavioral cross-EP agreement for the four DirectML-gated families (016
//! §6.2).
//!
//! Raw-logit max-diff tolerances misjudge GPU EPs: fp16 kernel drift can be
//! numerically large yet behaviorally invisible. These tests feed the same
//! golden input from `ferment-models-export` through each processor as the app
//! calls it — once on the CPU EP, once on the selected GPU EP
//! (`FERMENT_ENGINE_ORT_EP`) — and compare the decisions users actually see:
//! beat grids, scene-cut lists, VAD frame decisions, highlight windows.
//!
//! Every test prints its measured agreement so runs record numbers even when
//! the provisional gates pass; final tolerance sign-off is the lead's.

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

#[path = "support/ort_assignment.rs"]
mod ort_assignment;

use std::sync::Arc;

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{AudioEventFrameProbabilities, ProgressSink};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::audio_event_detector::{Postprocessor, SegmenterConfig};
use ferment_engine_processors::beatthis::config::SAMPLE_RATE;
use ferment_engine_processors::musichighlighter::config::CHUNK_SECONDS;
use ferment_engine_processors::{
    AudioEventDetector, BeatThis, MusicHighlighter, VideoSceneDetector,
};
use ferment_engine_test_fixtures::export;

/// BeatThis's golden audio: 30 s of techno at the 22.05 kHz both music
/// families read, which the highlighter reuses for its own golden.
const MUSIC: &str = "beat_this/goldens/beat_this_audio_in.bin";

fn gpu_provider() -> OrtExecutionProvider {
    let provider = match std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() {
        Ok("directml") => OrtExecutionProvider::DirectMl,
        Ok("cuda") => OrtExecutionProvider::Cuda,
        Ok("windows") => OrtExecutionProvider::WindowsMl,
        Ok("cpu") | Err(_) => OrtExecutionProvider::Cpu,
        Ok(other) => panic!("unsupported FERMENT_ENGINE_ORT_EP: {other}"),
    };
    if matches!(provider, OrtExecutionProvider::Cpu) {
        static NOTE: std::sync::Once = std::sync::Once::new();
        NOTE.call_once(|| {
            eprintln!(
                "DEGENERATE ort_behavioral: FERMENT_ENGINE_ORT_EP is unset or `cpu`, so every \
                 comparison below is CPU against CPU and proves nothing about GPU agreement. \
                 Set FERMENT_ENGINE_ORT_EP=windows|directml|cuda on a GPU box for the real gate."
            );
        });
    }
    provider
}

/// The published masters of `models` behind a CPU backend and the
/// EP-under-test backend; `None` (a `SKIP`) when they are not at hand.
fn backend_pair(models: &[(ModelId, &str)]) -> Option<(Arc<OrtBackend>, Arc<OrtBackend>)> {
    let mut paths = ModelPaths::new();
    for (id, file_name) in models {
        paths = paths.with_path(*id, ModelVersion::Latest, export::artifact(file_name)?);
    }
    let paths = Arc::new(paths);
    Some((
        Arc::new(OrtBackend::with_execution_provider(
            paths.clone(),
            OrtExecutionProvider::Cpu,
        )),
        Arc::new(OrtBackend::with_execution_provider(paths, gpu_provider())),
    ))
}

/// A golden clip of 16-bit samples, unit scale.
fn i16_audio(relative: &str) -> Vec<f32> {
    std::fs::read(export::require_path(relative))
        .unwrap()
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect()
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the beat_this masters"]
fn beat_grid_agrees_across_eps() {
    let Some(audio) = export::f32_le(MUSIC) else {
        return;
    };
    let Some((cpu, gpu)) = backend_pair(&[
        (ModelId::BeatThis1, "beat_this_1.onnx"),
        (ModelId::BeatThis2, "beat_this_2.onnx"),
    ]) else {
        return;
    };
    let grid = |backend: Backend| {
        futures_executor::block_on(BeatThis::new(backend).detect(
            AudioFrames::new(audio.clone(), SAMPLE_RATE),
            ProgressSink::noop(),
        ))
        .expect("beat analysis")
    };
    let (cpu_grid, gpu_grid) = (grid(cpu), grid(gpu.clone()));
    for model in [ModelId::BeatThis1, ModelId::BeatThis2] {
        ort_assignment::require_gpu(&gpu, model);
    }

    let worst_delta = cpu_grid
        .beats
        .iter()
        .zip(&gpu_grid.beats)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "beat grid: cpu {} beats, gpu {} beats, worst matched |dt| {worst_delta:.4}s; downbeats {} vs {}",
        cpu_grid.beats.len(),
        gpu_grid.beats.len(),
        cpu_grid.downbeats.len(),
        gpu_grid.downbeats.len(),
    );
    assert_eq!(
        cpu_grid.beats.len(),
        gpu_grid.beats.len(),
        "beat count diverged across EPs"
    );
    // One model frame at 50 fps — a beat moving less than the analysis hop is
    // musically identical.
    assert!(
        worst_delta <= 0.020,
        "beat times drifted {worst_delta:.4}s across EPs"
    );
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the autoshot master"]
fn scene_cuts_agree_across_eps() {
    let Some(path) = export::path("autoshot/model.json") else {
        return;
    };
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let Some((cpu, gpu)) = backend_pair(&[(ModelId::Autoshot, "autoshot.onnx")]) else {
        return;
    };
    let video = &contract["goldens"]["video"];
    let clip = std::fs::read(export::require_path(&format!(
        "autoshot/{}",
        video["clip_frames"].as_str().unwrap()
    )))
    .unwrap();
    let count = video["clip_frame_count"].as_u64().unwrap() as usize;
    let cuts = |backend: Backend| -> Vec<usize> {
        futures_executor::block_on(VideoSceneDetector::new(backend).detect(
            &clip,
            count,
            30.0,
            ProgressSink::noop(),
        ))
        .expect("scene detection")
        .cuts
        .iter()
        .map(|cut| cut.frame)
        .collect()
    };
    let (cpu_cuts, gpu_cuts) = (cuts(cpu), cuts(gpu.clone()));
    ort_assignment::require_gpu(&gpu, ModelId::Autoshot);

    eprintln!("scene cuts: cpu {cpu_cuts:?} vs gpu {gpu_cuts:?}");
    assert_eq!(cpu_cuts, gpu_cuts, "scene-cut lists diverged across EPs");
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the fireredvad_aed master"]
fn vad_decisions_agree_across_eps() {
    if export::path("fireredvad_aed/model.json").is_none() {
        return;
    }
    let Some((cpu, gpu)) = backend_pair(&[(ModelId::FireRedVadAed, "fireredvad_aed.onnx")]) else {
        return;
    };
    let mut worst = ("", "", 1.0f64);
    for clip in ["hairdryer", "sec5", "acapella"] {
        let samples = i16_audio(&format!(
            "fireredvad_aed/goldens/fireredvad_aed_{clip}_audio_in.i16"
        ));
        let probabilities = |backend: Backend| {
            futures_executor::block_on(AudioEventDetector::new(backend).detect_audio(
                &AudioFrames::new(samples.clone(), 16_000),
                ProgressSink::noop(),
            ))
            .expect("detection")
            .frame_probs
        };
        let (cpu_rows, gpu_rows) = (probabilities(cpu.clone()), probabilities(gpu.clone()));
        assert_eq!(cpu_rows.len(), gpu_rows.len());

        for (label, config, class) in [
            (
                "speech",
                SegmenterConfig::SPEECH,
                (|row| row.speech) as fn(&AudioEventFrameProbabilities) -> f32,
            ),
            ("singing", SegmenterConfig::SINGING, |row| row.singing),
            ("music", SegmenterConfig::MUSIC, |row| row.music),
        ] {
            let decisions = |rows: &[AudioEventFrameProbabilities]| {
                Postprocessor::process(&rows.iter().map(class).collect::<Vec<f32>>(), config)
            };
            let (cpu_decisions, gpu_decisions) = (decisions(&cpu_rows), decisions(&gpu_rows));
            let agree = cpu_decisions
                .iter()
                .zip(&gpu_decisions)
                .filter(|(a, b)| a == b)
                .count() as f64
                / cpu_decisions.len().max(1) as f64;
            eprintln!("vad {clip} {label}: frame agreement {agree:.5}");
            if agree < worst.2 {
                worst = (clip, label, agree);
            }
        }
    }
    let (clip, label, agreement) = worst;
    assert!(
        agreement >= 0.995,
        "vad {clip} '{label}' frame agreement {agreement:.5} below 0.995"
    );
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the music_highlighter master"]
fn highlight_window_agrees_across_eps() {
    let Some(audio) = export::f32_le(MUSIC) else {
        return;
    };
    let Some((cpu, gpu)) = backend_pair(&[(ModelId::MusicHighlighter, "music_highlighter.onnx")])
    else {
        return;
    };
    let highlight = |backend: Backend| {
        let result = futures_executor::block_on(MusicHighlighter::new(backend).detect(
            AudioFrames::new(audio.clone(), SAMPLE_RATE),
            vec![9, 15],
            ProgressSink::noop(),
        ))
        .expect("highlight");
        let top_second = result
            .saliency
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(second, _)| second)
            .unwrap();
        let starts: Vec<f32> = result
            .highlights
            .iter()
            .map(|window| window.start)
            .collect();
        (starts, top_second / CHUNK_SECONDS)
    };
    let ((cpu_windows, cpu_top), (gpu_windows, gpu_top)) = (highlight(cpu), highlight(gpu));

    eprintln!(
        "highlight: cpu windows at {cpu_windows:?} top-chunk {cpu_top} vs gpu windows at {gpu_windows:?} top-chunk {gpu_top}"
    );
    assert_eq!(cpu_top, gpu_top, "top highlight chunk diverged across EPs");
    assert_eq!(
        cpu_windows, gpu_windows,
        "best highlight windows diverged across EPs"
    );
}
