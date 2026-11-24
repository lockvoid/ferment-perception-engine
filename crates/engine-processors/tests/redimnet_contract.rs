//! `redimnet_speaker` against its contract and its golden ladder — both owned
//! by `ferment-models-export` (`models/redimnet/`), the same bytes that grade
//! the iOS engine:
//!
//! * `model.json` — the front-end, the window, the door and the checkpoint
//!   digest this processor holds as constants;
//! * `…_audio_in.i16` → `…_mel_out.f32` — the host's signal steps, `MelSpec`
//!   and band centring against the checkpoint's own `TFMelBanks`, window by
//!   window;
//! * `…_embedding_out.f32` — the checkpoint's L2-normalised embeddings: every
//!   window within `COSINE` through the real backends, and the two speakers
//!   told apart as the checkpoint tells them.
//!
//! The contract and the features run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts, which live in the
//! export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::redimnet::windows::{
    EMBEDDING_DIMENSION, HOP_LENGTH, WINDOW_FRAMES, WINDOW_SAMPLES,
};
use ferment_engine_processors::redimnet::{
    self, MEL_BANDS, MEL_SPEC, PRE_EMPHASIS, ReDimNetSpeakerEncoder, SIGNAL_EPSILON,
};
use ferment_engine_test_fixtures::export;

const CLIPS: [&str; 5] = [
    "speaker-a-1",
    "speaker-a-2",
    "speaker-b-1",
    "speaker-b-2",
    "piano",
];
const COSINE: f32 = 0.999;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("redimnet/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

/// A golden clip's whole windows, unit scale.
fn windows(clip: &str) -> Vec<Vec<f32>> {
    let path = export::require_path(&format!("redimnet/goldens/redimnet_{clip}_audio_in.i16"));
    let samples: Vec<f32> = std::fs::read(path)
        .unwrap()
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect();
    samples
        .chunks_exact(WINDOW_SAMPLES)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The checkpoint's mel per window, frames major `[200 × 72]`.
fn golden_mels(clip: &str) -> Vec<Vec<f32>> {
    let values = export::require_f32_le(&format!("redimnet/goldens/redimnet_{clip}_mel_out.f32"));
    assert_eq!(
        values.len() % (WINDOW_FRAMES * MEL_BANDS),
        0,
        "{clip}: partial golden window"
    );
    values
        .chunks_exact(WINDOW_FRAMES * MEL_BANDS)
        .map(<[f32]>::to_vec)
        .collect()
}

fn golden_embeddings(clip: &str) -> Vec<Vec<f32>> {
    export::require_f32_le(&format!(
        "redimnet/goldens/redimnet_{clip}_embedding_out.f32"
    ))
    .chunks_exact(EMBEDDING_DIMENSION)
    .map(<[f32]>::to_vec)
    .collect()
}

/// Frames major `[200 × 72]` into the door's bands major `[72 × 200]`.
fn bands_major(frames: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; MEL_BANDS * WINDOW_FRAMES];
    for frame in 0..WINDOW_FRAMES {
        for band in 0..MEL_BANDS {
            out[band * WINDOW_FRAMES + frame] = frames[frame * MEL_BANDS + band];
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "redimnet_speaker");
    let files = contract["source"]["files"].as_object().unwrap();
    assert_eq!(
        files.values().next().unwrap(),
        ReDimNetSpeakerEncoder::ENCODER_HASH
    );
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(
        frontend["signal"]["epsilon"].as_f64().unwrap(),
        SIGNAL_EPSILON
    );
    assert_eq!(
        frontend["signal"]["preEmphasis"].as_f64().unwrap() as f32,
        PRE_EMPHASIS
    );
    assert_eq!(frontend["signal"]["padding"], HOP_LENGTH / 2);
    assert_eq!(frontend["window"]["samples"], WINDOW_SAMPLES);
    assert_eq!(frontend["window"]["frames"], WINDOW_FRAMES);

    let main = &contract["functions"][0];
    // Kill: an extra tensor silently passes a check of only tensor zero.
    assert_eq!(main["inputs"].as_array().unwrap().len(), 1);
    assert_eq!(main["outputs"].as_array().unwrap().len(), 1);
    assert_eq!(main["inputs"][0]["name"], "mel");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, 1, MEL_BANDS, WINDOW_FRAMES])
    );
    assert_eq!(main["outputs"][0]["name"], "embedding");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, EMBEDDING_DIMENSION, 1, 1])
    );
    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "redimnet_speaker.onnx"
    );
    assert_eq!(
        contract["artifacts"]["coreml"]["file"],
        "redimnet_speaker.mlmodelc"
    );
}

/// The host half against the checkpoint's `TFMelBanks`, window by window.
#[test]
fn the_features_are_the_checkpoints() {
    if contract().is_none() {
        return;
    }
    for clip in CLIPS {
        let mut worst = 0.0_f32;
        let goldens = golden_mels(clip);
        let windows = windows(clip);
        // Kill: zip silently ignores a missing golden window.
        assert_eq!(goldens.len(), windows.len(), "{clip}: golden/window count");
        assert!(!windows.is_empty(), "{clip}: no golden windows");
        for (window, golden) in windows.iter().zip(&goldens) {
            let ours = redimnet::features(window).unwrap();
            let delta = ours
                .iter()
                .zip(bands_major(golden))
                .map(|(a, e)| (a - e).abs())
                .fold(0.0, f32::max);
            worst = worst.max(delta);
        }
        eprintln!(
            "[redimnet-features] {clip} ({} windows): max |Δ| vs the checkpoint {worst:.3e}",
            goldens.len()
        );
        assert!(worst < 2e-3, "{clip}: {worst}");
    }
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_infer::{Backend, ModelId, ModelPaths, ModelVersion};

    use super::*;

    /// Every window within `COSINE` of the checkpoint — fed the checkpoint's
    /// own mel and fed this host's — and the speakers told apart as the
    /// checkpoint tells them.
    pub(super) fn assert_answers(backend: Backend, label: &str) {
        if contract().is_none() {
            return;
        }
        let encoder = ReDimNetSpeakerEncoder::new(backend);
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
        for clip in CLIPS {
            let mut worst = 1.0_f32;
            let windows = windows(clip);
            let mels = golden_mels(clip);
            let expected_vectors = golden_embeddings(clip);
            // Kill: zip accepts a shortened golden and silently omits the final windows.
            assert!(!windows.is_empty(), "{clip}: missing windows");
            assert_eq!(mels.len(), windows.len(), "{clip}: mel/window count");
            assert_eq!(
                expected_vectors.len(),
                windows.len(),
                "{clip}: embedding/window count"
            );
            for ((window, golden_mel), expected) in windows.iter().zip(mels).zip(expected_vectors) {
                let from_golden =
                    futures_executor::block_on(encoder.embedding(&bands_major(&golden_mel)))
                        .unwrap();
                let from_host = futures_executor::block_on(
                    encoder.embedding(&redimnet::features(window).unwrap()),
                )
                .unwrap();
                worst = worst
                    .min(cosine(&from_golden, &expected))
                    .min(cosine(&from_host, &expected));
                vectors.push((clip.to_owned(), from_host));
            }
            eprintln!("[redimnet] {label} {clip}: worst cosine vs the checkpoint {worst:.6}");
            assert!(worst >= COSINE, "{label} {clip}: {worst}");
        }
        let speaker = |name: &str| name.starts_with("speaker-a");
        let (mut within, mut across) = (1.0_f32, -1.0_f32);
        for (i, (a, x)) in vectors.iter().enumerate() {
            for (b, y) in &vectors[i + 1..] {
                if a == "piano" || b == "piano" {
                    continue;
                }
                let similarity = cosine(x, y);
                if speaker(a) == speaker(b) {
                    within = within.min(similarity);
                } else {
                    across = across.max(similarity);
                }
            }
        }
        eprintln!(
            "[redimnet] {label}: within-speaker min {within:.3}, cross-speaker max {across:.3}"
        );
        assert!(
            within > across,
            "{label}: the speakers are no longer told apart"
        );
    }

    pub(super) fn paths(file_name: &str) -> Option<ModelPaths> {
        let artifact = export::artifact(file_name)?;
        Some(ModelPaths::new().with_path(ModelId::RedimnetSpeaker, ModelVersion::Latest, artifact))
    }

    #[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
    mod ort {
        use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};

        use super::*;

        /// fp32 behind the same door: the export measured cosine 1.000000.
        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_encoder_answers_the_checkpoint() {
            let Some(paths) = paths("redimnet_speaker.onnx") else {
                return;
            };
            let backend =
                OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu);
            assert_answers(Arc::new(backend), "ort");
        }
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    mod coreml {
        use ferment_engine_infer::{ComputeUnits, CoreMlBackend};
        use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

        use super::*;

        /// The fp16 bundle on the CPU, the lane the fleet pins: the export
        /// measured every window within 0.9998.
        #[test]
        #[ignore = "needs ferment-models-export/artifacts/redimnet_speaker.mlmodelc"]
        fn the_encoder_answers_the_checkpoint_on_core_ml() {
            assert_eq!(
                ferment_engine_models::default_fleet_manifest()
                    .model(ModelId::RedimnetSpeaker)
                    .unwrap()
                    .placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuOnly
                })
            );
            let Some(paths) = paths("redimnet_speaker.mlmodelc") else {
                return;
            };
            let backend = CoreMlBackend::with_compute_units(Arc::new(paths), ComputeUnits::CpuOnly);
            assert_answers(Arc::new(backend), "coreml");
        }
    }
}
