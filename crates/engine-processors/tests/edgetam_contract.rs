//! `edgetam_v1_*` against its contract and its golden chain — both owned by
//! `ferment-models-export` (`models/edgetam/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — the four doors, their shapes, and the ONNX decoder's own
//!   form: ONE file whose point axis is dynamic and whose previous mask is
//!   always bound behind `has_mask_input`, which is what this engine calls
//!   (`edgetam/model.rs`) where Apple's back-ends carry sixteen functions;
//! * the chain — a click on a real frame, a correction on it, that frame's
//!   memory, and the frame tracked behind it: every door fed the reference's
//!   own tensors, and what is asserted is what the product reads — the
//!   candidate the host selects and its mask over the same pixels.
//!
//! The contract runs everywhere; the lowered half is `#[ignore]`d like every
//! oracle and needs the artifacts.

use ferment_engine_test_fixtures::export;

const IMAGE_SIZE: usize = 1024;
const VISION: [usize; 4] = [1, 256, 64, 64];
const HIGH_0: [usize; 4] = [1, 32, 256, 256];
const HIGH_1: [usize; 4] = [1, 64, 128, 128];
const LOW_RES: usize = 256;
const CANDIDATES: usize = 4;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("edgetam/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn door<'a>(contract: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    contract["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|function| function["name"] == name)
        .unwrap_or_else(|| panic!("{name} is not a door of this contract"))
}

#[test]
fn the_contract_is_the_four_doors() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "edgetam_v1");

    let encoder = door(&contract, "edgetam_v1_image_encoder");
    assert_eq!(
        encoder["inputs"][0]["shape"],
        serde_json::json!([1, 3, IMAGE_SIZE, IMAGE_SIZE])
    );
    assert_eq!(encoder["outputs"][0]["shape"], serde_json::json!(VISION));
    assert_eq!(encoder["outputs"][1]["shape"], serde_json::json!(HIGH_0));
    assert_eq!(encoder["outputs"][2]["shape"], serde_json::json!(HIGH_1));

    let decoder = door(&contract, "edgetam_v1_prompt_mask_decoder");
    assert_eq!(
        decoder["outputs"][0]["shape"],
        serde_json::json!([1, CANDIDATES, LOW_RES, LOW_RES])
    );
    // The ONNX door's own inputs: the mask is bound on every call and gated.
    let onnx = &contract["members"]["edgetam_v1_prompt_mask_decoder"]["artifacts"]["onnx"];
    let inputs: Vec<&str> = onnx["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert!(
        inputs.contains(&"mask_input") && inputs.contains(&"has_mask_input"),
        "{inputs:?}"
    );
    assert_eq!(onnx["dynamic_axes"]["point_coordinates"]["1"], "points");

    let memory = door(&contract, "edgetam_v1_memory_encoder");
    assert_eq!(
        memory["outputs"][0]["shape"],
        serde_json::json!([1, 512, 64])
    );
    let attention = door(&contract, "edgetam_v1_memory_attention");
    assert_eq!(
        attention["inputs"][1]["shape"],
        serde_json::json!([1, 7, 512, 64])
    );
    assert_eq!(attention["outputs"][0]["shape"], serde_json::json!(VISION));
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend,
        OrtExecutionProvider, Tensor, TensorMap,
    };

    use super::*;

    /// fp32 behind the same doors: the export measured every mask within
    /// 0.9999 of the reference's.
    const MASK_IOU: f64 = 0.99;

    fn goldens_f32(case: &str, tensor: &str) -> Vec<f32> {
        export::require_f32_le(&format!("edgetam/goldens/edgetam_{case}_{tensor}.f32"))
    }

    /// A door input: the goldens keep them in fp16, the reference's outputs in
    /// fp32 — either way the ONNX door takes fp32.
    fn feed(case: &str, tensor: &str) -> Vec<f32> {
        let half_precision = export::require_path("edgetam/model.json")
            .parent()
            .unwrap()
            .join("goldens")
            .join(format!("edgetam_{case}_{tensor}.f16"));
        if half_precision.is_file() {
            let bytes = std::fs::read(half_precision).unwrap();
            return bytes
                .chunks_exact(2)
                .map(|pair| f32::from(half::f16::from_le_bytes([pair[0], pair[1]])))
                .collect();
        }
        goldens_f32(case, tensor)
    }

    fn tensor(shape: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::from_f32(shape.iter().map(|value| *value as i64).collect(), values).unwrap()
    }

    fn run(backend: &Backend, model: ModelId, inputs: TensorMap) -> TensorMap {
        futures_executor::block_on(backend.infer(model, &ModelVersion::default(), inputs)).unwrap()
    }

    /// The stock multimask rule the host uses: the highest IoU of candidates 1…3.
    fn selected(iou: &[f32]) -> usize {
        1 + iou[1..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    fn mask_iou(a: &[f32], b: &[f32]) -> f64 {
        let (mut intersection, mut union) = (0_usize, 0_usize);
        for (x, y) in a.iter().zip(b) {
            if *x > 0.0 || *y > 0.0 {
                union += 1;
                if *x > 0.0 && *y > 0.0 {
                    intersection += 1;
                }
            }
        }
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn paths() -> Option<ModelPaths> {
        let mut paths = ModelPaths::new();
        for (model, file) in [
            (
                ModelId::EdgetamImageEncoder,
                "edgetam_v1_image_encoder.onnx",
            ),
            (
                ModelId::EdgetamPromptMaskDecoder,
                "edgetam_v1_prompt_mask_decoder.onnx",
            ),
            (
                ModelId::EdgetamMemoryEncoder,
                "edgetam_v1_memory_encoder.onnx",
            ),
            (
                ModelId::EdgetamMemoryAttention,
                "edgetam_v1_memory_attention.onnx",
            ),
        ] {
            paths = paths.with_path(model, ModelVersion::Latest, export::artifact(file)?);
        }
        Some(paths)
    }

    fn decode(
        backend: &Backend,
        case: &str,
        features: (&str, Vec<f32>),
        points: usize,
        has_mask: bool,
    ) -> (usize, Vec<f32>) {
        let (frame, vision) = features;
        let mut inputs = TensorMap::new();
        inputs.insert("vision_features".into(), tensor(&VISION, vision));
        inputs.insert(
            "high_resolution_features_0".into(),
            tensor(&HIGH_0, feed(frame, "high_resolution_features_0")),
        );
        inputs.insert(
            "high_resolution_features_1".into(),
            tensor(&HIGH_1, feed(frame, "high_resolution_features_1")),
        );
        inputs.insert(
            "point_coordinates".into(),
            tensor(&[1, points, 2], goldens_f32(case, "point_coordinates")),
        );
        let labels =
            export::require_path(&format!("edgetam/goldens/edgetam_{case}_point_labels.i32"));
        let labels: Vec<f32> = std::fs::read(labels)
            .unwrap()
            .chunks_exact(4)
            .map(|four| i32::from_le_bytes([four[0], four[1], four[2], four[3]]) as f32)
            .collect();
        inputs.insert("point_labels".into(), tensor(&[1, points], labels));
        inputs.insert(
            "add_no_memory_embedding".into(),
            tensor(&[1], vec![if case == "track" { 0.0 } else { 1.0 }]),
        );
        inputs.insert(
            "apply_object_score_mask_gate".into(),
            tensor(
                &[1],
                vec![if matches!(case, "click" | "refine") {
                    0.0
                } else {
                    1.0
                }],
            ),
        );
        inputs.insert(
            "mask_input".into(),
            tensor(
                &[1, 1, LOW_RES, LOW_RES],
                if has_mask {
                    feed(case, "mask_input")
                } else {
                    vec![0.0; LOW_RES * LOW_RES]
                },
            ),
        );
        inputs.insert(
            "has_mask_input".into(),
            tensor(&[1], vec![if has_mask { 1.0 } else { 0.0 }]),
        );

        let outputs = run(backend, ModelId::EdgetamPromptMaskDecoder, inputs);
        let iou = outputs["iou_predictions"].to_f32().unwrap();
        (selected(&iou), outputs["candidate_masks"].to_f32().unwrap())
    }

    fn assert_case(backend: &Backend, case: &str, frame: &str, points: usize, has_mask: bool) {
        let vision = if case == "track" {
            feed("track", "conditioned_vision_features")
        } else {
            feed(frame, "vision_features")
        };
        let (chosen, masks) = decode(backend, case, (frame, vision), points, has_mask);
        let reference_masks = goldens_f32(case, "candidate_masks");
        let reference_iou = goldens_f32(case, "iou_predictions");
        let expected = selected(&reference_iou);
        let size = LOW_RES * LOW_RES;
        let overlap = mask_iou(
            &masks[expected * size..(expected + 1) * size],
            &reference_masks[expected * size..(expected + 1) * size],
        );
        eprintln!(
            "[edgetam] ort {case}: candidate {chosen} (reference {expected}), mask IoU {overlap:.4}"
        );
        assert_eq!(
            chosen, expected,
            "{case}: the host would select another candidate"
        );
        assert!(overlap >= MASK_IOU, "{case}: mask IoU {overlap}");
    }

    /// Current CoreMLCacheEvictionTests' EdgeTAM ownership regression, using
    /// Android's actual four ONNX graphs and its single dynamic decoder.
    /// The retained model owns a backend, not a hidden process-wide session
    /// registry. Generic SessionCache weak-reference tests additionally prove
    /// that dropping the registry reference actually releases the value.
    #[test]
    #[ignore = "requires explicit frozen ONNX models/exporter goldens and ORT_DYLIB_PATH"]
    fn retained_edgetam_runtime_does_not_pin_evicted_sessions_or_duplicate_point_variants() {
        use ferment_engine_processors::edgetam::EdgeTamModel;
        for variable in [
            "FERMENT_ENGINE_MODELS_DIR",
            "FERMENT_MODELS_EXPORT_DIR",
            "ORT_DYLIB_PATH",
        ] {
            assert!(
                std::env::var_os(variable).is_some(),
                "{variable} must name the frozen input"
            );
        }
        let backend = Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths().expect("all four published EdgeTAM files are required")),
            OrtExecutionProvider::Cpu,
        ));
        let inference: Backend = backend.clone();
        let retained = EdgeTamModel::new(inference.clone());
        futures_executor::block_on(backend.warmup(&EdgeTamModel::MODEL_FILES)).unwrap();
        assert_eq!(backend.cache_snapshot().resident, 4);
        let evicted = backend.evict_idle_models(&[]);
        assert_eq!(evicted.len(), 4);
        for id in EdgeTamModel::MODEL_FILES {
            assert!(evicted.contains(&id));
            assert_eq!(backend.selected_provider_for(id), None);
        }
        assert_eq!(backend.cache_snapshot().resident, 0);
        // One-point click and two-point correction are two shapes of ONE
        // dynamic ONNX session, not Apple's separately bound functions.
        assert_case(&inference, "click", "click", 1, false);
        assert_case(&inference, "refine", "click", 2, true);
        assert_eq!(backend.cache_snapshot().resident, 1);
        assert_eq!(
            backend.evict_idle_models(&[]),
            [ModelId::EdgetamPromptMaskDecoder]
        );
        assert_case(&inference, "refine", "click", 2, true);
        assert_eq!(backend.cache_snapshot().resident, 1);
        // The actual retained product model remains callable after the sweep.
        let image: Vec<_> = feed("click", "normalized_image")
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let features = futures_executor::block_on(retained.encode_image(&image)).unwrap();
        assert_eq!(
            features.vision_features.len(),
            VISION.iter().product::<usize>()
        );
        assert_eq!(backend.cache_snapshot().resident, 2);
        drop(retained);
        assert_eq!(backend.evict_idle_models(&[]).len(), 2);
        assert_eq!(backend.cache_snapshot().resident, 0);
    }

    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_four_doors_answer_the_checkpoint() {
        if contract().is_none() {
            return;
        }
        let Some(paths) = paths() else { return };
        let backend: Backend = Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ));

        // The image encoder, on the frame the click lands in.
        let mut inputs = TensorMap::new();
        inputs.insert(
            "normalized_image".into(),
            tensor(
                &[1, 3, IMAGE_SIZE, IMAGE_SIZE],
                feed("click", "normalized_image"),
            ),
        );
        let encoded = run(&backend, ModelId::EdgetamImageEncoder, inputs);
        let vision = encoded["vision_features"].to_f32().unwrap();
        let reference = goldens_f32("click", "vision_features");
        let worst = vision
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[edgetam] ort image encoder: max |Δ| vs the reference {worst:.3e}");
        assert!(worst < 1e-2, "image encoder: {worst}");

        assert_case(&backend, "click", "click", 1, false);
        assert_case(&backend, "refine", "click", 2, true);
        assert_case(&backend, "conditioning", "click", 1, false);
        assert_case(&backend, "track", "track", 1, false);

        // The memory encoder and the attention, fed the reference's own tensors.
        let mut inputs = TensorMap::new();
        inputs.insert(
            "raw_vision_features".into(),
            tensor(&VISION, feed("click", "vision_features")),
        );
        inputs.insert(
            "selected_low_resolution_mask".into(),
            tensor(
                &[1, 1, LOW_RES, LOW_RES],
                feed("conditioning", "selected_low_resolution_mask"),
            ),
        );
        let memory = run(&backend, ModelId::EdgetamMemoryEncoder, inputs)["memory_features"]
            .to_f32()
            .unwrap();
        let reference = goldens_f32("conditioning", "memory_features");
        let worst = memory
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[edgetam] ort memory encoder: max |Δ| vs the reference {worst:.3e}");
        assert!(worst < 1e-1, "memory encoder: {worst}");

        let mut inputs = TensorMap::new();
        inputs.insert(
            "raw_vision_features".into(),
            tensor(&VISION, feed("track", "vision_features")),
        );
        inputs.insert(
            "spatial_memory_features".into(),
            tensor(&[1, 7, 512, 64], feed("track", "spatial_memory_features")),
        );
        inputs.insert(
            "object_pointers".into(),
            tensor(&[1, 16, 256], feed("track", "object_pointers")),
        );
        inputs.insert(
            "spatial_memory_validity".into(),
            tensor(&[1, 7], feed("track", "spatial_memory_validity")),
        );
        inputs.insert(
            "object_pointer_validity".into(),
            tensor(&[1, 16], feed("track", "object_pointer_validity")),
        );
        let conditioned =
            run(&backend, ModelId::EdgetamMemoryAttention, inputs)["conditioned_vision_features"]
                .to_f32()
                .unwrap();
        let reference = goldens_f32("track", "conditioned_vision_features");
        let worst = conditioned
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[edgetam] ort memory attention: max |Δ| vs the reference {worst:.3e}");
        assert!(worst < 1e-2, "memory attention: {worst}");
    }
}
