//! `autoshot` against its contract and its golden windows — both owned by
//! `ferment-models-export` (`models/autoshot/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — the host's frame, histogram, window and cut rules this
//!   processor holds as constants, and the door every lowering shares;
//! * `autoshot_<case>_frames_in` → `autoshot_<case>_color_hist_out` — the host
//!   half, a synthetic window and a real one;
//! * `autoshot_<case>_one_hot_out` — the stock network over both, through the
//!   real backends;
//! * `autoshot_clip_frames_in` — `hairdryer-product-placement.mp4`, all 604
//!   frames at 48 × 27: the processor's own batching over it must fire the
//!   stock network's cuts.
//!
//! The contract and the histograms run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts, which live in the
//! export repository only between an export and its upload.

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
use ferment_engine_core::ProgressSink;
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
use ferment_engine_processors::VideoSceneDetector;
use ferment_engine_processors::video_scene_detector::config::{
    BORDER_SIZE, CENTER_SIZE, DEFAULT_THRESHOLD, FRAME_HEIGHT, FRAME_WIDTH, N_COLOR_BINS,
    N_FRAMES_PER_BATCH,
};
use ferment_engine_processors::video_scene_detector::histograms::color_histograms;
use ferment_engine_test_fixtures::export;

const CASES: [&str; 2] = ["synthetic", "video"];

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

fn contract() -> Option<serde_json::Value> {
    let path = export::path("autoshot/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn frames(case: &str) -> Option<Vec<u8>> {
    let path = export::path(&format!("autoshot/goldens/autoshot_{case}_frames_in.u8"))?;
    Some(std::fs::read(path).unwrap())
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "autoshot");
    let frontend = &contract["frontend"];
    assert_eq!(frontend["frame"]["width"], FRAME_WIDTH);
    assert_eq!(frontend["frame"]["height"], FRAME_HEIGHT);
    assert_eq!(frontend["histogram"]["bins"], N_COLOR_BINS);
    assert_eq!(frontend["window"]["frames"], N_FRAMES_PER_BATCH);
    assert_eq!(frontend["window"]["center"], CENTER_SIZE);
    assert_eq!(frontend["window"]["border"], BORDER_SIZE);
    assert_eq!(frontend["cut"]["sigmoid"], ">");
    assert_eq!(
        frontend["cut"]["threshold"].as_f64().unwrap() as f32,
        DEFAULT_THRESHOLD
    );

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "frames");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, 3, N_FRAMES_PER_BATCH, FRAME_HEIGHT, FRAME_WIDTH])
    );
    assert_eq!(main["inputs"][1]["name"], "color_hist");
    assert_eq!(
        main["inputs"][1]["shape"],
        serde_json::json!([1, N_FRAMES_PER_BATCH, N_COLOR_BINS])
    );
    assert_eq!(main["outputs"][0]["name"], "one_hot");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, N_FRAMES_PER_BATCH])
    );
    assert_eq!(contract["artifacts"]["onnx"]["file"], "autoshot.onnx");
    assert_eq!(contract["artifacts"]["coreml"]["file"], "autoshot.mlmodelc");
}

/// The host half: 3 bits a channel, each frame's histogram over its L2 norm —
/// the authors' `compute_color_histograms` in numpy, to float32 rounding.
#[test]
fn the_histograms_are_the_authors() {
    for case in CASES {
        let Some(frames) = frames(case) else { return };
        let golden = export::require_f32_le(&format!(
            "autoshot/goldens/autoshot_{case}_color_hist_out.f32"
        ));
        let ours = color_histograms(&frames, N_FRAMES_PER_BATCH);
        let delta = worst(&ours, &golden);
        eprintln!("[autoshot-hist] {case}: max |Δ| vs numpy {delta:.3e}");
        assert!(delta < 1e-6, "{case}: {delta}");
    }
}

/// All 604 frames of the clip through `detector` as the app calls it — its
/// padding, its windows, its cut rule — must fire the stock network's cuts.
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
fn assert_clip_cuts(contract: &serde_json::Value, detector: &VideoSceneDetector, label: &str) {
    let video = &contract["goldens"]["video"];
    let clip = std::fs::read(export::require_path(&format!(
        "autoshot/{}",
        video["clip_frames"].as_str().unwrap()
    )))
    .unwrap();
    let count = video["clip_frame_count"].as_u64().unwrap() as usize;
    let result =
        futures_executor::block_on(detector.detect(&clip, count, 30.0, ProgressSink::noop()))
            .unwrap();
    let fired: Vec<u64> = result.cuts.iter().map(|cut| cut.frame as u64).collect();
    let expected: Vec<u64> = video["clip_cuts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| frame.as_u64().unwrap())
        .collect();
    eprintln!("[autoshot-clip] {label}: fired {fired:?}");
    assert_eq!(fired, expected);
}

/// The graph alone on both golden windows, against the stock network's
/// `one_hot`: each case's max |Δ| under its bound (`[synthetic, video]`).
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
fn assert_graph(
    backend: &dyn ferment_engine_infer::InferenceBackend,
    label: &str,
    bounds: [f32; 2],
) {
    use ferment_engine_infer::{ModelId, ModelVersion, Tensor, TensorMap};

    for (case, bound) in CASES.into_iter().zip(bounds) {
        let frames = frames(case).unwrap();
        let pixels = FRAME_HEIGHT * FRAME_WIDTH;
        let stride = N_FRAMES_PER_BATCH * pixels;
        let mut planar = vec![0.0_f32; 3 * stride];
        for (index, value) in frames.iter().enumerate() {
            let (pixel, channel) = (index / 3, index % 3);
            planar[channel * stride + pixel] = f32::from(*value);
        }
        let mut inputs = TensorMap::new();
        inputs.insert(
            "frames".into(),
            Tensor::from_f32(
                vec![
                    1,
                    3,
                    N_FRAMES_PER_BATCH as i64,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64,
                ],
                planar,
            )
            .unwrap(),
        );
        inputs.insert(
            "color_hist".into(),
            Tensor::from_f32(
                vec![1, N_FRAMES_PER_BATCH as i64, N_COLOR_BINS as i64],
                color_histograms(&frames, N_FRAMES_PER_BATCH),
            )
            .unwrap(),
        );
        let outputs = futures_executor::block_on(backend.infer(
            ModelId::Autoshot,
            &ModelVersion::Latest,
            inputs,
        ))
        .unwrap();
        let actual = outputs["one_hot"].to_f32().unwrap();
        let golden =
            export::require_f32_le(&format!("autoshot/goldens/autoshot_{case}_one_hot_out.f32"));
        let delta = worst(&actual, &golden);
        eprintln!("[autoshot-graph] {label} {case}: max |Δ| vs the stock network {delta:.3e}");
        assert!(delta < bound, "{label} {case}: {delta}");
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };

    use super::*;

    fn backend() -> Option<OrtBackend> {
        let master = export::artifact("autoshot.onnx")?;
        let paths = ModelPaths::new().with_path(ModelId::Autoshot, ModelVersion::Latest, master);
        Some(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ))
    }

    /// fp32 arithmetic over fp16-stored weights, which the export measured
    /// 7.0e-3 (synthetic) and 2.1e-3 (video) from the stock network.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_master_matches_the_stock_network() {
        let Some(backend) = backend() else { return };
        assert_graph(&backend, "ort", [2e-2, 2e-2]);
    }

    /// All 604 frames of the clip through the processor as the app calls it —
    /// its padding, its windows, its cut rule — fire the stock network's cuts.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_processor_fires_the_stock_networks_cuts() {
        let (Some(contract), Some(backend)) = (contract(), backend()) else {
            return;
        };
        let backend: Backend = Arc::new(backend);
        assert_clip_cuts(&contract, &VideoSceneDetector::new(backend), "ort");
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion,
    };
    use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

    use super::*;

    /// The device bundle on the units both runtimes pin (`CpuAndNeuralEngine`):
    /// float16 at both doors, the processor speaks float32, Core ML converts.
    fn backend() -> Option<CoreMlBackend> {
        assert_eq!(
            ferment_engine_models::default_fleet_manifest()
                .model(ModelId::Autoshot)
                .unwrap()
                .placement_for(BackendKind::CoreMl),
            Some(Placement::CoreMl {
                units: CoreMlComputeUnits::CpuAndNeuralEngine
            })
        );
        let bundle = export::artifact("autoshot.mlmodelc")?;
        let paths = ModelPaths::new().with_path(ModelId::Autoshot, ModelVersion::Latest, bundle);
        Some(CoreMlBackend::with_compute_units(
            Arc::new(paths),
            ComputeUnits::CpuAndNeuralEngine,
        ))
    }

    /// The export measured 0.030 (synthetic) and 0.018 (video) there
    /// (`parity.coreml.autoshot.CPU_AND_NE`); the bounds keep twice that.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/autoshot.mlmodelc"]
    fn the_bundle_matches_the_stock_network() {
        let Some(backend) = backend() else { return };
        assert_graph(&backend, "coreml", [6.1e-2, 3.7e-2]);
    }

    #[test]
    #[ignore = "needs ferment-models-export/artifacts/autoshot.mlmodelc"]
    fn the_processor_fires_the_stock_networks_cuts_on_core_ml() {
        let (Some(contract), Some(backend)) = (contract(), backend()) else {
            return;
        };
        let backend: Backend = Arc::new(backend);
        assert_clip_cuts(&contract, &VideoSceneDetector::new(backend), "coreml");
    }
}
