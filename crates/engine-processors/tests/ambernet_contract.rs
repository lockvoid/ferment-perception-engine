//! `ambernet_langid` against its contract and its golden ladder — both owned
//! by `ferment-models-export` (`models/ambernet/`), the same bytes that grade
//! the iOS engine:
//!
//! * `model.json` — the front-end, the door and the vocabulary this processor
//!   holds as constants;
//! * `…_audio_in.i16` → `…_mel_out.f32` — the host's `MelSpec` and per-band
//!   normalisation against the frames NeMo's featurizer hands its encoder;
//! * `…_logits_out.f32` — NeMo's logits: the processor must answer its first
//!   language through the real backends, at its probability.
//!
//! The contract and the features run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts, which live in the
//! export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::ambernet::{
    self, DOOR_FRAMES, HOP_LENGTH, LANGUAGE_COUNT, MEL_BANDS, MEL_SPEC, WINDOW_SAMPLES, Window,
    langs,
};
use ferment_engine_test_fixtures::export;

const CLIPS: [&str; 5] = ["ru", "advert", "edu", "piano", "short"];

fn contract() -> Option<serde_json::Value> {
    let path = export::path("ambernet/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

/// A golden window, unit scale.
fn window(clip: &str) -> Window {
    let path = export::require_path(&format!("ambernet/goldens/ambernet_{clip}_audio_in.i16"));
    let samples: Vec<f32> = std::fs::read(path)
        .unwrap()
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect();
    let frame_count = samples.len() / HOP_LENGTH;
    Window {
        samples,
        frame_count,
    }
}

/// NeMo's featurizer frames, time major `[valid × 80]`.
fn golden_mel(clip: &str) -> Vec<f32> {
    export::require_f32_le(&format!("ambernet/goldens/ambernet_{clip}_mel_out.f32"))
}

fn golden_logits(clip: &str) -> Vec<f32> {
    export::require_f32_le(&format!("ambernet/goldens/ambernet_{clip}_logits_out.f32"))
}

/// Time major `[valid × 80]` into the door's bands major `[80 × DOOR_FRAMES]`.
fn bands_major(frames: &[f32]) -> Vec<f32> {
    let valid = frames.len() / MEL_BANDS;
    let mut out = vec![0.0; MEL_BANDS * DOOR_FRAMES];
    for frame in 0..valid {
        for band in 0..MEL_BANDS {
            out[band * DOOR_FRAMES + frame] = frames[frame * MEL_BANDS + band];
        }
    }
    out
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "ambernet_langid");
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(frontend["window"]["samples"], WINDOW_SAMPLES);
    assert_eq!(frontend["window"]["frames"], DOOR_FRAMES);

    let main = &contract["functions"][0];
    // Kill: an extra tensor silently passes a check of only tensor zero.
    assert_eq!(main["inputs"].as_array().unwrap().len(), 2);
    assert_eq!(main["outputs"].as_array().unwrap().len(), 1);
    assert_eq!(main["inputs"][0]["name"], "mel");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, MEL_BANDS, 1, DOOR_FRAMES])
    );
    assert_eq!(main["inputs"][1]["name"], "mask");
    assert_eq!(
        main["inputs"][1]["shape"],
        serde_json::json!([1, 1, 1, DOOR_FRAMES])
    );
    assert_eq!(main["outputs"][0]["name"], "logits");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, LANGUAGE_COUNT, 1, 1])
    );
    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "ambernet_langid.onnx"
    );
    assert_eq!(
        contract["artifacts"]["coreml"]["file"],
        "ambernet_langid.mlmodelc"
    );

    let vocabulary = &contract["vocabulary"];
    assert_eq!(vocabulary["file"], "ambernet_langid_vocab.json");
    assert_eq!(vocabulary["languages"], LANGUAGE_COUNT);
    assert_eq!(vocabulary["bytes"], langs::VOCAB_JSON.len());
    // Kill: bundled bytes and an export copy agree but drift from the signed contract.
    assert_eq!(
        vocabulary["sha256"].as_str().unwrap(),
        ferment_engine_models::sha256_bytes(langs::VOCAB_JSON.as_bytes())
    );
    assert_eq!(langs::languages().unwrap().len(), LANGUAGE_COUNT);
    let exported =
        std::fs::read(export::require_path("ambernet/ambernet_langid_vocab.json")).unwrap();
    assert_eq!(
        langs::VOCAB_JSON.as_bytes(),
        exported.as_slice(),
        "the crate's vocabulary is not the export's bytes"
    );
}

/// The host half against NeMo's featurizer, and zero past the valid frames.
#[test]
fn the_features_are_nemos() {
    if contract().is_none() {
        return;
    }
    for clip in CLIPS {
        let window = window(clip);
        let golden = golden_mel(clip);
        assert_eq!(golden.len(), window.frame_count * MEL_BANDS, "{clip}");
        let ours = ambernet::features(&window).unwrap();
        let expected = bands_major(&golden);
        let delta = ours
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f32::max);
        eprintln!(
            "[ambernet-features] {clip} ({} frames): max |Δ| vs NeMo {delta:.3e}",
            window.frame_count
        );
        assert!(delta < 2e-3, "{clip}: {delta}");
        // Kill: small nonzero padding passes the numeric tolerance but changes the model door.
        assert_eq!(ours.len(), MEL_BANDS * DOOR_FRAMES, "{clip}");
        for band in 0..MEL_BANDS {
            assert!(
                ours[band * DOOR_FRAMES + window.frame_count..(band + 1) * DOOR_FRAMES]
                    .iter()
                    .all(|value| *value == 0.0),
                "{clip}: padded band {band}"
            );
        }
    }
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_infer::{Backend, ModelId, ModelPaths, ModelVersion};
    use ferment_engine_processors::ambernet::{AmberNetLanguageId, hypotheses};

    use super::*;

    /// NeMo's first language on every golden window at its probability, fed
    /// NeMo's own frames and fed this host's; the logits within `bound`.
    pub(super) fn assert_answers(backend: Backend, label: &str, bound: f32) {
        if contract().is_none() {
            return;
        }
        let languages = langs::languages().unwrap();
        let identifier = AmberNetLanguageId::new(backend);
        for clip in CLIPS {
            let window = window(clip);
            let reference = golden_logits(clip);
            let expected = hypotheses(&reference, languages)[0].clone();
            for (source, mel) in [
                ("NeMo's frames", bands_major(&golden_mel(clip))),
                ("the host's frames", ambernet::features(&window).unwrap()),
            ] {
                let logits =
                    futures_executor::block_on(identifier.logits(&mel, window.frame_count))
                        .unwrap();
                let delta = logits
                    .iter()
                    .zip(&reference)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0, f32::max);
                let got = hypotheses(&logits, languages)[0].clone();
                eprintln!(
                    "[ambernet] {label} {clip}, {source}: {} {:.4} (NeMo {} {:.4}), logits max |Δ| {delta:.3e}",
                    got.code, got.probability, expected.code, expected.probability
                );
                assert_eq!(got.code, expected.code, "{label} {clip}, {source}");
                assert!(
                    (got.probability - expected.probability).abs() < 0.05,
                    "{label} {clip}, {source}: {} against {}",
                    got.probability,
                    expected.probability
                );
                assert!(delta < bound, "{label} {clip}, {source}: {delta}");
            }
        }
    }

    pub(super) fn paths(file_name: &str) -> Option<ModelPaths> {
        let artifact = export::artifact(file_name)?;
        Some(ModelPaths::new().with_path(ModelId::AmberNetLangId, ModelVersion::Latest, artifact))
    }

    #[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
    mod ort {
        use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};

        use super::*;

        /// fp32 NeMo layers behind the same door: the export measured the
        /// logits at PSNR ceiling.
        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_processor_answers_nemo() {
            let Some(paths) = paths("ambernet_langid.onnx") else {
                return;
            };
            let backend =
                OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu);
            assert_answers(Arc::new(backend), "ort", 1e-2);
        }
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    mod coreml {
        use ferment_engine_infer::{ComputeUnits, CoreMlBackend};
        use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

        use super::*;

        /// The fp16 bundle on the CPU, the lane the fleet pins: the export
        /// measured the logits 0.175 from NeMo's at worst there
        /// (`parity.coreml.CPU_ONLY`), every first language NeMo's; the bound
        /// keeps twice that.
        #[test]
        #[ignore = "needs ferment-models-export/artifacts/ambernet_langid.mlmodelc"]
        fn the_processor_answers_nemo_on_core_ml() {
            assert_eq!(
                ferment_engine_models::default_fleet_manifest()
                    .model(ModelId::AmberNetLangId)
                    .unwrap()
                    .placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuOnly
                })
            );
            let Some(paths) = paths("ambernet_langid.mlmodelc") else {
                return;
            };
            let backend = CoreMlBackend::with_compute_units(Arc::new(paths), ComputeUnits::CpuOnly);
            assert_answers(Arc::new(backend), "coreml", 0.35);
        }
    }
}
