//! `pyannote_segmentation` against its contract and its golden ladder — both
//! owned by `ferment-models-export` (`models/pyannote/`), the same bytes that
//! grade the iOS engine:
//!
//! * `model.json` — the window, the door, the powerset order and the weights
//!   digest this processor holds as constants;
//! * `…_log_probs_out.f32` — the onnx-community export's powerset
//!   log-probabilities on three windows (one speaker, turn-taking, overlap):
//!   through the real backends current iOS decision, measured-error margin,
//!   and probability limits are enforced together.
//!
//! The contract runs everywhere; the lowered halves are `#[ignore]`d like
//! every oracle and need the artifacts, which live in the export repository
//! only between an export and its upload.

use ferment_engine_processors::pyannote::PyannoteSpeakerSegmenter;
use ferment_engine_processors::pyannote::powerset::CLASS_COUNT;
use ferment_engine_processors::pyannote::windows::{WINDOW_FRAMES, WINDOW_SAMPLES};
use ferment_engine_test_fixtures::export;

const CASES: [&str; 3] = ["one-speaker", "turn-taking", "overlap"];
/// Keep the stricter existing Rust bound (Swift allows 0.05) while also
/// enforcing the current Swift decision-count and measured-margin gates.
const PROBABILITY: f32 = 0.04;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("pyannote/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "pyannote_segmentation");
    let files = contract["source"]["files"].as_object().unwrap();
    assert_eq!(
        files.values().next().unwrap(),
        PyannoteSpeakerSegmenter::SEGMENTER_HASH
    );
    assert_eq!(contract["frontend"]["window"]["samples"], WINDOW_SAMPLES);
    assert_eq!(contract["frontend"]["window"]["frames"], WINDOW_FRAMES);
    let main = &contract["functions"][0];
    // Kill: an extra tensor silently passes a check of only tensor zero.
    assert_eq!(main["inputs"].as_array().unwrap().len(), 1);
    assert_eq!(main["outputs"].as_array().unwrap().len(), 1);
    assert_eq!(main["inputs"][0]["name"], "audio");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, 1, WINDOW_SAMPLES])
    );
    assert_eq!(main["outputs"][0]["name"], "log_probs");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, WINDOW_FRAMES, CLASS_COUNT])
    );
    assert_eq!(contract["powerset"].as_array().unwrap().len(), CLASS_COUNT);
    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "pyannote_segmentation.onnx"
    );
    assert_eq!(
        contract["artifacts"]["coreml"]["file"],
        "pyannote_segmentation.mlmodelc"
    );
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_infer::{Backend, ModelId, ModelPaths, ModelVersion};

    use super::*;

    /// A golden window, unit scale.
    fn window(case: &str) -> Vec<f32> {
        let path = export::require_path(&format!("pyannote/goldens/pyannote_{case}_audio_in.i16"));
        std::fs::read(path)
            .unwrap()
            .chunks_exact(2)
            .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_768.0)
            .collect()
    }

    fn reference(case: &str) -> Vec<f32> {
        export::require_f32_le(&format!(
            "pyannote/goldens/pyannote_{case}_log_probs_out.f32"
        ))
        .into_iter()
        .map(f32::exp)
        .collect()
    }

    fn winner(row: &[f32]) -> usize {
        (0..row.len())
            .max_by(|a, b| row[*a].total_cmp(&row[*b]))
            .unwrap()
    }

    pub(super) fn assert_answers(backend: Backend, label: &str) {
        if contract().is_none() {
            return;
        }
        let segmenter = PyannoteSpeakerSegmenter::new(backend);
        for case in CASES {
            let probabilities =
                futures_executor::block_on(segmenter.predict(&window(case), 0)).unwrap();
            let expected = reference(case);
            assert_eq!(probabilities.len(), expected.len());
            let (mut turned, mut worst) = (0, 0.0_f32);
            let mut turned_margins = Vec::new();
            for (ours, theirs) in probabilities
                .chunks_exact(CLASS_COUNT)
                .zip(expected.chunks_exact(CLASS_COUNT))
            {
                for (a, e) in ours.iter().zip(theirs) {
                    worst = worst.max((a - e).abs());
                }
                if winner(ours) != winner(theirs) {
                    turned += 1;
                    let mut sorted = theirs.to_vec();
                    sorted.sort_by(|a, b| b.total_cmp(a));
                    turned_margins.push(sorted[0] - sorted[1]);
                }
            }
            // Kill: accept widespread near-tie changes, or a decision outside this lane's error.
            let decided = turned_margins
                .iter()
                .filter(|margin| **margin >= 0.02 || **margin > 2.0 * worst)
                .count();
            assert!(
                (turned as f64) / (WINDOW_FRAMES as f64) < 0.02,
                "{label} {case}: {turned}/{WINDOW_FRAMES} frames decide differently"
            );
            eprintln!(
                "[pyannote] {label} {case}: {turned}/{WINDOW_FRAMES} frames turned, {decided} of them decided; probabilities max |Δ| {worst:.3e}"
            );
            assert_eq!(
                decided, 0,
                "{label} {case}: frames the export decides turned"
            );
            assert!(worst < PROBABILITY, "{label} {case}: {worst}");
        }
    }

    pub(super) fn paths(file_name: &str) -> Option<ModelPaths> {
        let artifact = export::artifact(file_name)?;
        Some(ModelPaths::new().with_path(
            ModelId::PyannoteSegmentation,
            ModelVersion::Latest,
            artifact,
        ))
    }

    #[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
    mod ort {
        use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};

        use super::*;

        /// fp32 behind the same door: the export measured every decision the
        /// reference's.
        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_segmenter_answers_the_export() {
            let Some(paths) = paths("pyannote_segmentation.onnx") else {
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
        /// measured 7 of 1767 frames turned, every one within 0.016 of a tie.
        #[test]
        #[ignore = "needs ferment-models-export/artifacts/pyannote_segmentation.mlmodelc"]
        fn the_segmenter_answers_the_export_on_core_ml() {
            assert_eq!(
                ferment_engine_models::default_fleet_manifest()
                    .model(ModelId::PyannoteSegmentation)
                    .unwrap()
                    .placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuOnly
                })
            );
            let Some(paths) = paths("pyannote_segmentation.mlmodelc") else {
                return;
            };
            let backend = CoreMlBackend::with_compute_units(Arc::new(paths), ComputeUnits::CpuOnly);
            assert_answers(Arc::new(backend), "coreml");
        }
    }
}
