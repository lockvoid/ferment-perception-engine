//! `fireredvad_aed` against its contract and its golden ladder — both owned by
//! `ferment-models-export` (`models/fireredvad_aed/`), the same bytes that
//! grade the iOS engine:
//!
//! * `model.json` — the front-end, the CMVN, the door and its windows this
//!   processor holds as constants;
//! * `…_audio_in.i16` → `…_fbank_out.f32` — the host's `MelSpec` against
//!   kaldi-native-fbank;
//! * `…_probs_out.f32` — the reference's whole-sequence probs, which the
//!   processor must answer through the real backends, window by window
//!   (`acapella` is two windows).
//!
//! The contract and the fbank run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts, which live in the
//! export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::audio_event_detector::config::{
    INPUT_DIM, MEL_SPEC, NUM_EVENT_CLASSES, WINDOW_FRAMES, WINDOW_MARGIN, WINDOW_STRIDE,
};
use ferment_engine_processors::audio_event_detector::{Cmvn, INVERSE_STDDEV, MEANS};
use ferment_engine_test_fixtures::export;

const CLIPS: [&str; 3] = ["hairdryer", "sec5", "acapella"];

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

fn contract() -> Option<serde_json::Value> {
    let path = export::path("fireredvad_aed/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

/// A golden clip's samples, unit scale.
fn audio(clip: &str) -> Vec<f32> {
    let path = export::require_path(&format!(
        "fireredvad_aed/goldens/fireredvad_aed_{clip}_audio_in.i16"
    ));
    std::fs::read(path)
        .unwrap()
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
        .collect()
}

fn probs(clip: &str) -> Vec<f32> {
    export::require_f32_le(&format!(
        "fireredvad_aed/goldens/fireredvad_aed_{clip}_probs_out.f32"
    ))
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "fireredvad_aed");
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    let means: Vec<f32> = serde_json::from_value(frontend["cmvn"]["means"].clone()).unwrap();
    let inverse_std: Vec<f32> =
        serde_json::from_value(frontend["cmvn"]["inverseStd"].clone()).unwrap();
    assert_eq!(means, MEANS);
    assert_eq!(inverse_std, INVERSE_STDDEV);
    assert_eq!(frontend["windows"]["frames"], WINDOW_FRAMES);
    assert_eq!(frontend["windows"]["margin"], WINDOW_MARGIN);
    assert_eq!(frontend["windows"]["stride"], WINDOW_STRIDE);
    assert_eq!(
        frontend["classes"],
        serde_json::json!(["speech", "singing", "music"])
    );

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "feat");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, INPUT_DIM, 1, WINDOW_FRAMES])
    );
    assert_eq!(main["inputs"][1]["name"], "mask");
    assert_eq!(
        main["inputs"][1]["shape"],
        serde_json::json!([1, 1, 1, WINDOW_FRAMES])
    );
    assert_eq!(main["outputs"][0]["name"], "probs");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, NUM_EVENT_CLASSES, 1, WINDOW_FRAMES])
    );
    assert_eq!(contract["artifacts"]["onnx"]["file"], "fireredvad_aed.onnx");
    assert_eq!(
        contract["artifacts"]["coreml"]["file"],
        "fireredvad_aed.mlmodelc"
    );
}

/// The host half against kaldi-native-fbank, and the CMVN over it.
#[test]
fn the_fbank_is_kaldis() {
    if contract().is_none() {
        return;
    }
    for clip in ["hairdryer", "sec5"] {
        let golden = export::require_f32_le(&format!(
            "fireredvad_aed/goldens/fireredvad_aed_{clip}_fbank_out.f32"
        ));
        let ours = MEL_SPEC.compute(&audio(clip)).unwrap().values;
        let delta = worst(&ours, &golden);
        eprintln!("[aed-fbank] {clip}: max |Δ| vs kaldi-native-fbank {delta:.3e}");
        assert!(delta < 1e-3, "{clip}: {delta}");
        let frames = golden.len() / INPUT_DIM;
        let (mut normalised, mut reference) = (ours, golden);
        assert!(Cmvn::apply(&mut normalised, frames));
        assert!(Cmvn::apply(&mut reference, frames));
        eprintln!(
            "[aed-feat] {clip}: max |Δ| after the CMVN {:.3e}",
            worst(&normalised, &reference)
        );
    }
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_audio::AudioFrames;
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::{Backend, ModelId, ModelPaths, ModelVersion};
    use ferment_engine_processors::audio_event_detector::AudioEventDetector;

    use super::*;

    /// Audio in, probs out — the processor as the app calls it, window by
    /// window, against the reference's whole-sequence probs.
    pub(super) fn assert_processor(backend: Backend, label: &str, bound: f32) {
        if contract().is_none() {
            return;
        }
        for clip in CLIPS {
            let result = futures_executor::block_on(
                AudioEventDetector::new(backend.clone())
                    .detect_audio(&AudioFrames::new(audio(clip), 16_000), ProgressSink::noop()),
            )
            .unwrap();
            let ours: Vec<f32> = result
                .frame_probs
                .iter()
                .flat_map(|row| [row.speech, row.singing, row.music])
                .collect();
            let delta = worst(&ours, &probs(clip));
            eprintln!(
                "[aed-e2e] {label} {clip} ({} frames): max |Δ| vs the reference {delta:.3e}",
                result.frame_probs.len()
            );
            assert!(delta < bound, "{label} {clip}: {delta}");
        }
    }

    pub(super) fn paths(file_name: &str) -> Option<ModelPaths> {
        let artifact = export::artifact(file_name)?;
        Some(ModelPaths::new().with_path(ModelId::FireRedVadAed, ModelVersion::Latest, artifact))
    }

    #[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
    mod ort {
        use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};

        use super::*;

        /// fp32 behind the same door: the export measured 2.0e-6.
        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_processor_answers_the_reference() {
            let Some(paths) = paths("fireredvad_aed.onnx") else {
                return;
            };
            let backend =
                OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu);
            assert_processor(Arc::new(backend), "ort", 1e-4);
        }
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    mod coreml {
        use ferment_engine_infer::{ComputeUnits, CoreMlBackend, ModelId};
        use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

        use super::*;

        /// The fp16 bundle on the CPU, the lane the fleet pins: the export
        /// measured 7.4e-3 there at worst (`parity.coreml.CPU_ONLY`); the bound
        /// keeps twice that.
        #[test]
        #[ignore = "needs ferment-models-export/artifacts/fireredvad_aed.mlmodelc"]
        fn the_processor_answers_the_reference_on_core_ml() {
            assert_eq!(
                ferment_engine_models::default_fleet_manifest()
                    .model(ModelId::FireRedVadAed)
                    .unwrap()
                    .placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuOnly
                })
            );
            let Some(paths) = paths("fireredvad_aed.mlmodelc") else {
                return;
            };
            let backend = CoreMlBackend::with_compute_units(Arc::new(paths), ComputeUnits::CpuOnly);
            assert_processor(Arc::new(backend), "coreml", 1.5e-2);
        }
    }
}
