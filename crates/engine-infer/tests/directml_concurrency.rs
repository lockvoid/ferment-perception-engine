//! Two models inferring at once on DirectML, through the backend the app uses.
//!
//! ONNX Runtime's DirectML execution provider shares unlocked state between
//! the sessions of one process (onnxruntime #20713, PR #28007): two sessions
//! working at once crash the process with an access violation or "0x8000FFFF
//! Catastrophic failure". Separate processes are unaffected. Run explicitly on
//! a Windows GPU:
//!
//! ```text
//! ORT_DYLIB_PATH             the ONNX Runtime with DirectML
//! FERMENT_ENGINE_MODELS_DIR  the three realesr_*.onnx graphs
//! ```
#![cfg(all(windows, feature = "ort"))]

use std::path::PathBuf;
use std::sync::Arc;

use ferment_engine_infer::{
    InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
    TensorMap,
};
use futures_executor::block_on;

const MODELS: [(ModelId, &str); 3] = [
    (ModelId::RealesrGeneralX4, "realesr_general_x4v3.onnx"),
    (
        ModelId::RealesrGeneralWdnX4,
        "realesr_general_wdn_x4v3.onnx",
    ),
    (ModelId::RealesrAnimeVideo3, "realesr_animevideov3.onnx"),
];

fn graph(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR").expect("set FERMENT_ENGINE_MODELS_DIR"),
    )
    .join(name);
    assert!(path.is_file(), "required graph: {}", path.display());
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH"
    );
    path
}

/// One thread per model infers 200 times and drops its session every 10
/// rounds, so the creation, inference and release of different sessions
/// overlap: every pattern seen crashing the provider.
///
/// Kill: let DirectML sessions work at the same time → the process dies
/// before the threads finish (2 of 5 runs at 100 rounds with Windows ML's ORT
/// 1.28.3, 1 of 5 with ORT 1.24.4; access violation, heap corruption).
#[test]
#[ignore = "needs a DirectML GPU, ORT_DYLIB_PATH and the RealESR graphs"]
fn models_infer_at_once_on_directml() {
    let paths = MODELS
        .iter()
        .fold(ModelPaths::new(), |paths, &(model, file)| {
            paths.with_path(model, ModelVersion::Latest, graph(file))
        });
    let backend = Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::DirectMl,
    ));
    let workers: Vec<_> = MODELS
        .iter()
        .map(|&(model, _)| {
            let backend = backend.clone();
            std::thread::spawn(move || {
                for round in 0..200 {
                    if round % 10 == 0 {
                        backend.invalidate_model(model);
                    }
                    let mut input = TensorMap::new();
                    input.insert(
                        "lr".into(),
                        Tensor::from_f32(
                            vec![1, 3, 96, 96],
                            vec![round as f32 / 200.; 3 * 96 * 96],
                        )
                        .unwrap(),
                    );
                    let output =
                        block_on(backend.infer(model, &ModelVersion::Latest, input)).unwrap();
                    assert_eq!(output.values().next().unwrap().shape, [1, 3, 384, 384]);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    for (model, _) in MODELS {
        assert_eq!(
            backend.selected_provider_for(model),
            Some(OrtExecutionProvider::DirectMl)
        );
    }
}
