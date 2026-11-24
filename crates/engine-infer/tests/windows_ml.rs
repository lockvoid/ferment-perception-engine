//! Regression for the Windows default bypassing the Windows ML catalog.
#![cfg(feature = "ort")]

use ferment_engine_infer::OrtExecutionProvider;

/// Kill: the previous hard-coded DirectML → CPU chain bypasses catalog
/// discovery, installation and automatic device selection.
#[test]
fn windows_default_enters_windows_ml() {
    assert_eq!(OrtExecutionProvider::windows_chain()[0].name(), "WindowsML");
}

#[cfg(windows)]
mod native {
    use super::*;
    use ferment_engine_infer::{
        InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, Tensor, TensorMap,
    };
    use futures_executor::block_on;
    use std::sync::{Arc, Barrier};

    const MODELS: [(ModelId, &str); 3] = [
        (ModelId::RealesrGeneralX4, "realesr_general_x4v3.onnx"),
        (
            ModelId::RealesrGeneralWdnX4,
            "realesr_general_wdn_x4v3.onnx",
        ),
        (ModelId::RealesrAnimeVideo3, "realesr_animevideov3.onnx"),
    ];

    fn paths() -> Arc<ModelPaths> {
        let root = std::path::PathBuf::from(
            std::env::var_os("FERMENT_ENGINE_MODELS_DIR").expect("model root"),
        );
        let mut paths = ModelPaths::new();
        for (model, file) in MODELS {
            let path = root.join(file);
            assert!(path.is_file(), "missing {}", path.display());
            paths.insert(model, ModelVersion::Latest, path);
        }
        Arc::new(paths)
    }

    fn predict(backend: &OrtBackend, model: ModelId) {
        let input = Tensor::from_f32(vec![1, 3, 32, 32], vec![0.; 3 * 32 * 32]).unwrap();
        let outputs = block_on(backend.infer(
            model,
            &ModelVersion::Latest,
            TensorMap::from([("lr".into(), input)]),
        ))
        .unwrap();
        assert_eq!(outputs.len(), 1);
        let tensor = outputs.values().next().unwrap();
        assert_eq!(tensor.shape, [1, 3, 128, 128]);
        assert!(tensor.to_f32().unwrap().iter().all(|v| v.is_finite()));
    }

    /// Kill: racing process-wide catalog registration/session creation, losing
    /// registration ownership on eviction, or a silent CPU fallback.
    #[test]
    #[ignore = "requires Windows ML GPU and three RealESR graphs"]
    fn concurrent_windows_ml_load_infer_and_reload() {
        let backend = Arc::new(OrtBackend::windows_default(paths()));
        let barrier = Arc::new(Barrier::new(MODELS.len()));
        let workers: Vec<_> =
            MODELS
                .into_iter()
                .map(|(model, _)| {
                    let backend = backend.clone();
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        for round in 0..12 {
                            if round % 6 == 0 {
                                backend.invalidate_model(model);
                            }
                            predict(&backend, model);
                            assert_eq!(
                                backend.selected_provider_for(model),
                                Some(OrtExecutionProvider::WindowsMl)
                            );
                            let assignment = backend
                                .graph_assignment_for(model)
                                .expect("actual assignment");
                            assert!(assignment.iter().any(|g| g.provider
                                != "CPUExecutionProvider"
                                && !g.nodes.is_empty()));
                        }
                    })
                })
                .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(backend.evict_idle_models(&[]).len(), 3);
        assert_eq!(backend.cache_snapshot().resident, 0);
    }

    /// Kill: dropping CPU from the default chain makes a missing catalog DLL
    /// fatal. Run in a fresh process with ORT_DYLIB_PATH pointing to the same
    /// pinned runtime copied WITHOUT its catalog DLL; never alter the install.
    #[test]
    #[ignore = "requires isolated runtime WITHOUT catalog DLL, plus RealESR graphs"]
    fn missing_catalog_keeps_explicit_cpu_fallback() {
        let backend = OrtBackend::windows_default(paths());
        predict(&backend, ModelId::RealesrGeneralX4);
        assert_eq!(
            backend.selected_provider_for(ModelId::RealesrGeneralX4),
            Some(OrtExecutionProvider::Cpu)
        );
        let assignment = backend
            .graph_assignment_for(ModelId::RealesrGeneralX4)
            .expect("CPU assignment");
        assert!(!assignment.is_empty());
        assert!(
            assignment
                .iter()
                .all(|g| g.provider == "CPUExecutionProvider")
        );
    }
}
