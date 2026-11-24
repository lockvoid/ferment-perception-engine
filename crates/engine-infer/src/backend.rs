use std::sync::Arc;

use async_trait::async_trait;
use ferment_engine_core::{EngineError, EngineService, Result};

use crate::{ModelId, ModelVersion, TensorMap};

/// Platform-neutral, named-tensor inference seam.
///
/// Native backends are CoreML and ONNX Runtime. Browser builds implement this
/// trait in their wasm bridge and forward to onnxruntime-web.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait InferenceBackend: EngineService + std::fmt::Debug {
    async fn infer(
        &self,
        model: ModelId,
        version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap>;

    async fn warmup(&self, _models: &[ModelId]) -> Result<()> {
        Ok(())
    }

    /// One value out of the model's own creator-defined metadata.
    ///
    /// iOS reads `model.modelDescription.metadata[.creatorDefinedKey]`; ONNX's
    /// twin is `metadata_props`. `Ok(None)` means the artifact carries no such
    /// key — which is a fact a caller must be able to see, not an error: a
    /// speaker vector whose encoder cannot be named must not be stored, and
    /// that decision belongs to the caller.
    ///
    /// The default answers `None`, so a backend that has no metadata surface
    /// does not have to lie about one.
    async fn metadata(
        &self,
        _model: ModelId,
        _version: &ModelVersion,
        _key: &str,
    ) -> Result<Option<String>> {
        Ok(None)
    }
}

pub type Backend = Arc<dyn InferenceBackend>;

#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableBackend;

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl InferenceBackend for UnavailableBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        _inputs: TensorMap,
    ) -> Result<TensorMap> {
        Err(EngineError::Unsupported(format!(
            "no inference backend configured for {model}"
        )))
    }
}
