//! Regression: Android's facade smoke lost the missing EdgeTAM filename when
//! physical-identity inspection returned a bare filesystem error before loading.
#![cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]

use std::{path::PathBuf, sync::Arc};

use ferment_engine_core::EngineError;
use ferment_engine_infer::{InferenceBackend, ModelId, ModelPaths, ModelVersion};

fn missing_artifact(extension: &str) -> (PathBuf, Arc<ModelPaths>) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir()
        .join(format!("missing-model-{}-{nonce}", std::process::id()))
        .join(format!("edgetam_v1_image_encoder.{extension}"));
    assert!(!path.exists());
    let paths =
        ModelPaths::new().with_path(ModelId::EdgetamImageEncoder, ModelVersion::Latest, &path);
    (path, Arc::new(paths))
}

fn check_missing(backend: &dyn InferenceBackend, path: &std::path::Path) {
    let error = futures_executor::block_on(backend.warmup(&[ModelId::EdgetamImageEncoder]))
        .expect_err("a missing artifact must not warm a native session");
    match error {
        EngineError::ModelNotFound(actual) => assert_eq!(actual, path.display().to_string()),
        error => panic!("expected the exact missing artifact path, got {error:?}"),
    }
}

/// Kill: propagate canonicalize's raw I/O error and the diagnostic loses its path.
#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
#[test]
fn ort_warmup_names_the_missing_graph_before_native_load() {
    let (path, locator) = missing_artifact("onnx");
    let backend = ferment_engine_infer::OrtBackend::new(locator);
    check_missing(&backend, &path);
    assert_eq!(backend.cache_snapshot().resident, 0);
}

/// Kill: return an unqualified bundle-inspection error instead of the artifact.
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
#[test]
fn coreml_warmup_names_the_missing_bundle_before_native_load() {
    let (path, locator) = missing_artifact("mlmodelc");
    let backend = ferment_engine_infer::CoreMlBackend::new(locator);
    check_missing(&backend, &path);
    assert_eq!(backend.cache_snapshot().resident, 0);
}
