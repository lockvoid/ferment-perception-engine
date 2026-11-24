//! Core AI execution on macOS 27+, behind the same inference contract as Core ML.
//! The consumer selects and verifies a compiled artifact for the current chip.
//! Native work owns its model, inputs and admission permit until it completes;
//! cancellation detaches the reply and never frees buffers still in use.
//! Final macOS binaries using this feature must add `/usr/lib/swift` to rpath
//! in their build script; Cargo does not propagate a dependency's link arguments.
use std::sync::Arc;

use async_trait::async_trait;
use coreai_native::{LoadOptions, Model};
use ferment_engine_core::{EngineError, Result};

use crate::{
    DType, InferenceBackend, ModelId, ModelVersion, SharedModelLocator, Tensor, TensorMap,
};

type ModelKey = (ModelId, ModelVersion);

struct LoadedModel {
    model: Arc<Model>,
    predictions: crate::prediction_permit::PredictionPermitQueue,
    worker: crate::native_worker::NativeWorker,
}

pub struct CoreAiBackend {
    locator: SharedModelLocator,
    models: crate::session_cache::SessionCache<ModelKey, LoadedModel>,
}

impl std::fmt::Debug for CoreAiBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoreAiBackend")
            .field("locator", &self.locator)
            .field("cache", &self.models.snapshot())
            .finish_non_exhaustive()
    }
}

impl CoreAiBackend {
    pub fn new(locator: SharedModelLocator) -> Self {
        Self {
            locator,
            models: crate::session_cache::SessionCache::new(),
        }
    }

    async fn model(&self, id: ModelId, version: &ModelVersion) -> Result<Arc<LoadedModel>> {
        let key = (id, version.clone());
        let resolved = self.locator.locate(id, version).and_then(|path| {
            if path.extension().and_then(|s| s.to_str()) != Some("aimodelc") {
                return Err(EngineError::invalid_config(
                    "Core AI requires a compiled .aimodelc bundle",
                ));
            }
            let (identity, weight) = crate::coreml_identity::bundle_identity(&path)?;
            Ok((path, identity, weight))
        });
        let (path, identity, weight) = match resolved {
            Ok(value) => value,
            Err(error) => {
                self.models.invalidate(|candidate| candidate == &key);
                return Err(error);
            }
        };
        let expected = identity.clone();
        self.models
            .load(key, identity, weight, move || {
                crate::coreml_disk::check(weight)?;
                // Measured per encoder: the generic dynamic path passes GLAP text,
                // but corrupts audio on the GPU. Never apply this fleet-wide.
                let options = LoadOptions {
                    expect_frequent_reshapes: id == ModelId::GlapText,
                };
                let model =
                    Model::load_with_options(&path, options).map_err(EngineError::inference)?;
                if crate::coreml_identity::bundle_identity(&path)?.0 != expected {
                    return Err(EngineError::invalid_config(
                        "Core AI bundle changed during native load",
                    ));
                }
                Ok(LoadedModel {
                    model: Arc::new(model),
                    predictions: crate::prediction_permit::PredictionPermitQueue::default(),
                    worker: crate::native_worker::NativeWorker::new(format!(
                        "coreai-{}",
                        id.name()
                    ))?,
                })
            })
            .await
    }

    pub fn cache_snapshot(&self) -> crate::ModelCacheSnapshot {
        self.models.snapshot()
    }

    pub fn invalidate_model(&self, model: ModelId) -> bool {
        self.models.invalidate(|(id, _)| *id == model) != 0
    }

    pub fn evict_idle_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.models
            .evict_idle(|(id, _)| except.contains(id))
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn evict_inactive_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.models
            .evict_inactive(|(id, _)| except.contains(id))
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub async fn prediction_snapshot(
        &self,
        model: ModelId,
        version: &ModelVersion,
    ) -> Result<crate::prediction_permit::PredictionSnapshot> {
        Ok(self.model(model, version).await?.predictions.snapshot())
    }
}

#[async_trait]
impl InferenceBackend for CoreAiBackend {
    async fn infer(
        &self,
        id: ModelId,
        version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        let (function, inputs) = prepare_inputs(id, inputs)?;
        let loaded = self.model(id, version).await?;
        let permit = loaded.predictions.acquire().await;
        self.models
            .evict_inactive_if_pressure(|(model, v)| *model == id && v == version);
        let model = Arc::clone(&loaded.model);
        let keep_instance = Arc::clone(&loaded);
        loaded
            .worker
            .run(move || {
                let result = model
                    .predict(function, &inputs)
                    .map_err(EngineError::inference)
                    .and_then(outputs);
                permit.finish(result.is_ok());
                drop(keep_instance);
                result
            })
            .await
    }

    async fn warmup(&self, models: &[ModelId]) -> Result<()> {
        for &id in models {
            self.model(id, &ModelVersion::Latest).await?;
        }
        Ok(())
    }
}

fn prepare_inputs(
    id: ModelId,
    mut inputs: TensorMap,
) -> Result<(&'static str, Vec<coreai_native::Tensor>)> {
    let function = if id == ModelId::GlapAudio {
        // The upstream 4-frame patch convolution discards the last incomplete
        // patch. Core AI exports encode this as two functions with bounded,
        // divisible lengths. Other backends keep the unchanged host mel door.
        let mel = inputs
            .get_mut("mel")
            .ok_or_else(|| EngineError::inference("GLAP requires mel"))?;
        let [1, 64, frames @ 4..=1001] = mel.shape.as_slice() else {
            return Err(EngineError::inference(
                "GLAP mel must have shape [1,64,4..1001]",
            ));
        };
        let frames = *frames as usize;
        let width = match mel.dtype {
            DType::Fp16 => 2,
            DType::Fp32 => 4,
            _ => return Err(EngineError::inference("GLAP mel must be floating point")),
        };
        if mel.data.len() != 64 * frames * width {
            return Err(EngineError::inference(
                "GLAP mel byte length differs from shape",
            ));
        }
        let trimmed = frames / 4 * 4;
        if trimmed != frames {
            let mut data = Vec::with_capacity(64 * trimmed * width);
            for row in mel.data.chunks_exact(frames * width) {
                data.extend_from_slice(&row[..trimmed * width]);
            }
            *mel = Tensor::new(vec![1, 64, trimmed as i64], mel.dtype, data.into())?;
        }
        if trimmed == 4 { "short" } else { "main" }
    } else {
        "main"
    };
    let inputs = inputs
        .into_iter()
        .map(|(name, tensor)| {
            let dtype = match tensor.dtype {
                DType::Fp32 => coreai_native::DType::F32,
                DType::Fp16 => coreai_native::DType::F16,
                DType::Int32 => coreai_native::DType::I32,
                DType::Int64 => coreai_native::DType::I64,
                dtype => {
                    return Err(EngineError::Unsupported(format!(
                        "Core AI input {name} uses {}",
                        dtype.as_str()
                    )));
                }
            };
            Ok(coreai_native::Tensor {
                name,
                shape: tensor.shape,
                dtype,
                data: tensor.data.to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((function, inputs))
}

fn outputs(values: Vec<coreai_native::Tensor>) -> Result<TensorMap> {
    values
        .into_iter()
        .map(|tensor| {
            let dtype = match tensor.dtype {
                coreai_native::DType::F32 => DType::Fp32,
                coreai_native::DType::F16 => DType::Fp16,
                coreai_native::DType::I32 => DType::Int32,
                coreai_native::DType::I64 => DType::Int64,
            };
            Ok((
                tensor.name,
                Tensor::new(tensor.shape, dtype, tensor.data.into())?,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: truncating the flat buffer instead of each mel band shifts band 1.
    #[test]
    fn incomplete_audio_patches_are_trimmed_per_band_and_select_short_function() {
        let values = (0..64)
            .flat_map(|band| (0..5).map(move |frame| (10 * band + frame) as f32))
            .collect();
        let inputs = TensorMap::from([(
            "mel".into(),
            Tensor::from_f32(vec![1, 64, 5], values).unwrap(),
        )]);
        let (function, tensors) = prepare_inputs(ModelId::GlapAudio, inputs).unwrap();
        assert_eq!(function, "short");
        assert_eq!(tensors[0].shape, [1, 64, 4]);
        let prefix = tensors[0].data[..32]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(prefix, [0., 1., 2., 3., 10., 11., 12., 13.]);
    }
}
