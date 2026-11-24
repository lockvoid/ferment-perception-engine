//! Offline behavioral comparison from dumped model outputs (016 §6.2).
//!
//! The CUDA-build onnxruntime DLLs hang under the Rust crate's load-dynamic
//! path, so the CUDA leg of the behavioral gate runs on raw outputs dumped by
//! `fleet_ep_probe.py` (fail-loud python inference, every door fed from
//! `ferment-models-export`'s goldens) and post-processed here with the
//! shipping pure functions. No ONNX Runtime involved — this runs anywhere,
//! including macOS.
//!
//! Point `FERMENT_EP_DUMPS` at the dump directory; each test compares the
//! `cpu` dump against every other EP dump present (`cuda`, `dml`, `trt-rtx`) and
//! prints per-pair behavioral metrics.

use std::path::PathBuf;

use ferment_engine_processors::audio_event_detector::{Postprocessor, SegmenterConfig};
use ferment_engine_processors::beatthis::postprocess;
use ferment_engine_processors::musichighlighter::window_search::best_window;
use ferment_engine_processors::video_scene_detector::scenes::cuts_from_one_hot;
use ferment_engine_test_fixtures::export;

const EPS: &[&str] = &["cuda", "dml", "trt-rtx"];

fn dumps_dir() -> PathBuf {
    PathBuf::from(
        std::env::var_os("FERMENT_EP_DUMPS").expect("FERMENT_EP_DUMPS must point at the dump dir"),
    )
}

fn read_dump(name: &str, ep: &str) -> Option<Vec<f32>> {
    let path = dumps_dir().join(format!("{name}.{ep}.f32"));
    let bytes = std::fs::read(&path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
    )
}

fn require_dump(name: &str, ep: &str) -> Vec<f32> {
    read_dump(name, ep)
        .unwrap_or_else(|| panic!("missing dump {name}.{ep}.f32 in {}", dumps_dir().display()))
}

#[test]
#[ignore = "needs FERMENT_EP_DUMPS (fleet_ep_probe.py output)"]
fn beat_grid_agreement_from_dumps() {
    // The door's `logits [1, 2, 1, 1500]`: the beat row, then the downbeat row.
    let split = |raw: &[f32]| {
        let (beat, down) = raw.split_at(raw.len() / 2);
        (beat.to_vec(), down.to_vec())
    };
    let (cpu_beat, cpu_down) = split(&require_dump("beat_this_1", "cpu"));
    let reference = postprocess::process(&cpu_beat, &cpu_down, 50.0, None);
    for ep in EPS {
        let Some(raw) = read_dump("beat_this_1", ep) else {
            continue;
        };
        let (beat, down) = split(&raw);
        let candidate = postprocess::process(&beat, &down, 50.0, None);
        let worst = reference
            .beats
            .iter()
            .zip(&candidate.beats)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!(
            "beat grid cpu-vs-{ep}: {} vs {} beats, worst matched |dt| {worst:.4}s",
            reference.beats.len(),
            candidate.beats.len()
        );
        assert_eq!(
            reference.beats.len(),
            candidate.beats.len(),
            "{ep}: beat count diverged"
        );
        assert!(worst <= 0.020, "{ep}: beat drift {worst:.4}s");
    }
}

#[test]
#[ignore = "needs FERMENT_EP_DUMPS (fleet_ep_probe.py output)"]
fn scene_cut_agreement_from_dumps() {
    let cuts = |one_hot: &[f32]| -> Vec<usize> {
        cuts_from_one_hot(one_hot, 0.296)
            .into_iter()
            .map(|(frame, _)| frame)
            .collect()
    };
    let reference = cuts(&require_dump("autoshot", "cpu"));
    for ep in EPS {
        let Some(one_hot) = read_dump("autoshot", ep) else {
            continue;
        };
        let candidate = cuts(&one_hot);
        eprintln!("scene cuts cpu-vs-{ep}: {reference:?} vs {candidate:?}");
        assert_eq!(reference, candidate, "{ep}: scene-cut lists diverged");
    }
}

#[test]
#[ignore = "needs FERMENT_EP_DUMPS (fleet_ep_probe.py output)"]
fn vad_agreement_from_dumps() {
    for clip in ["hairdryer", "sec5"] {
        let name = format!("fireredvad_{clip}");
        let reference = require_dump(&name, "cpu");
        for ep in EPS {
            let Some(candidate) = read_dump(&name, ep) else {
                continue;
            };
            let frames = reference.len().min(candidate.len()) / 3;
            for (label, config, channel) in [
                ("speech", SegmenterConfig::SPEECH, 0usize),
                ("singing", SegmenterConfig::SINGING, 1),
                ("music", SegmenterConfig::MUSIC, 2),
            ] {
                let column = |values: &[f32]| -> Vec<f32> {
                    (0..frames).map(|t| values[t * 3 + channel]).collect()
                };
                let cpu_decisions = Postprocessor::process(&column(&reference), config);
                let ep_decisions = Postprocessor::process(&column(&candidate), config);
                let agree = cpu_decisions
                    .iter()
                    .zip(&ep_decisions)
                    .filter(|(a, b)| a == b)
                    .count() as f64
                    / cpu_decisions.len().max(1) as f64;
                eprintln!("vad {clip}/{label} cpu-vs-{ep}: frame agreement {agree:.5}");
                assert!(
                    agree >= 0.995,
                    "{ep}: vad {clip}/{label} agreement {agree:.5} below 0.995"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs FERMENT_EP_DUMPS (fleet_ep_probe.py output)"]
fn highlight_window_agreement_from_dumps() {
    // The golden track's chunks, the ones the probe fed; the rest of the 200
    // slots repeat them.
    let contract: serde_json::Value = serde_json::from_slice(
        &std::fs::read(export::require_path("music_highlighter/model.json")).unwrap(),
    )
    .unwrap();
    let chunks = contract["goldens"]["track"]["count"].as_u64().unwrap() as usize;
    let window_of = |logits: &[f32]| {
        let head = &logits[..chunks];
        let maximum = head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = head.iter().map(|value| (value - maximum).exp()).collect();
        let total: f32 = exps.iter().sum();
        let peak = exps.iter().copied().fold(0.0f32, f32::max);
        let saliency: Vec<f32> = exps
            .iter()
            .flat_map(|&score| std::iter::repeat_n(score / total / (peak / total), 3))
            .collect();
        let argmax = exps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();
        (best_window(&saliency, 9), argmax)
    };
    let (reference_window, reference_top) = window_of(&require_dump("music_highlighter", "cpu"));
    for ep in EPS {
        let Some(logits) = read_dump("music_highlighter", ep) else {
            continue;
        };
        let (window, top) = window_of(&logits);
        eprintln!(
            "highlight cpu-vs-{ep}: window {reference_window:?} top {reference_top} vs window {window:?} top {top}"
        );
        assert_eq!(reference_top, top, "{ep}: top highlight chunk diverged");
        assert_eq!(reference_window, window, "{ep}: best window diverged");
    }
}
