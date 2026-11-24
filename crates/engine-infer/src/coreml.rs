use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use coreml_native::{AsMultiArray, BorrowedTensor, ComputeUnits, DataType, Model};
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

/// CoreML backend for compiled `.mlmodelc` bundles.
///
/// `coreml-native` is a safe facade over the requested `objc2-core-ml`
/// bindings, keeping unsafe Objective-C buffer access out of this workspace.
pub struct CoreMlBackend {
    locator: SharedModelLocator,
    /// Units for a model with no pin of its own. Not a policy — a floor.
    default_compute_units: ComputeUnits,
    /// The placement axis: one compute-unit pin per model, read from the fleet
    /// manifest by whoever constructs the backend. Fixed for the backend's
    /// lifetime except for the consuming builders, which invalidate changed
    /// rows before returning. Old native loads cannot republish stale rows.
    compute_units: HashMap<ModelId, ComputeUnits>,
    models: crate::session_cache::SessionCache<ModelKey, LoadedModel>,
}

impl std::fmt::Debug for CoreMlBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CoreMlBackend")
            .field("locator", &self.locator)
            .field("default_compute_units", &self.default_compute_units)
            .field("compute_units", &self.compute_units)
            .field("cache", &self.models.snapshot())
            .finish_non_exhaustive()
    }
}

impl CoreMlBackend {
    pub fn new(locator: SharedModelLocator) -> Self {
        Self::with_compute_units(locator, ComputeUnits::All)
    }

    /// One set of units for every model. Still the right constructor for a
    /// benchmark that steers the whole fleet onto one compute unit; production
    /// pins per model on top of it.
    pub fn with_compute_units(locator: SharedModelLocator, compute_units: ComputeUnits) -> Self {
        Self {
            locator,
            default_compute_units: compute_units,
            compute_units: HashMap::new(),
            models: crate::session_cache::SessionCache::new(),
        }
    }

    /// Pin one model's compute units, overriding the backend-wide default.
    #[must_use]
    pub fn with_model_compute_units(mut self, model: ModelId, units: ComputeUnits) -> Self {
        if self.compute_units_for(model) != units {
            self.invalidate_model(model);
        }
        self.compute_units.insert(model, units);
        self
    }

    /// Pin a whole placement table at once — the fleet manifest's Core ML
    /// column, as `engine-models` hands it over.
    #[must_use]
    pub fn with_placements(
        mut self,
        placements: impl IntoIterator<Item = (ModelId, ComputeUnits)>,
    ) -> Self {
        for (model, units) in placements {
            self = self.with_model_compute_units(model, units);
        }
        self
    }

    /// Where this model's graph will run: its pin, or the backend default.
    pub fn compute_units_for(&self, model: ModelId) -> ComputeUnits {
        self.compute_units
            .get(&model)
            .copied()
            .unwrap_or(self.default_compute_units)
    }

    async fn model(&self, id: ModelId, version: &ModelVersion) -> Result<Arc<LoadedModel>> {
        let key = (id, version.clone());
        let resolved = self.locator.locate(id, version).and_then(|path| {
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
        let units = self.compute_units_for(id);
        self.models
            .load(key, identity, weight, move || {
                // Only a real cold load needs compiler scratch space. A cached
                // instance remains usable if disk space later becomes scarce.
                #[cfg(target_os = "macos")]
                crate::coreml_disk::check(weight)?;
                let model = Model::load(&path, units)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                if crate::coreml_identity::bundle_identity(&path)?.0 != expected {
                    return Err(EngineError::invalid_config(
                        "Core ML bundle changed during native load",
                    ));
                }
                Ok(LoadedModel {
                    model: Arc::new(model),
                    predictions: crate::prediction_permit::PredictionPermitQueue::default(),
                    worker: crate::native_worker::NativeWorker::new(format!(
                        "coreml-{}",
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
        self.models.invalidate(|(candidate, _)| *candidate == model) != 0
    }

    pub fn evict_idle_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.models
            .evict_idle(|(model, _)| except.contains(model))
            .into_iter()
            .map(|(model, _)| model)
            .collect()
    }

    /// Drop only unleased models, heaviest-first, regardless of memory pressure.
    /// Active/queued predictions and native calls draining after cancellation
    /// stay registered and joinable. Pending loads and `except` IDs also stay.
    /// Prefer this to `evict_idle_models` when moving between analysis stages.
    pub fn evict_inactive_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.models
            .evict_inactive(|(model, _)| except.contains(model))
            .into_iter()
            .map(|(model, _)| model)
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
impl InferenceBackend for CoreMlBackend {
    async fn infer(
        &self,
        model_id: ModelId,
        version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        let loaded = self.model(model_id, version).await?;
        let permit = loaded.predictions.acquire().await;
        self.models
            .evict_inactive_if_pressure(|(id, v)| *id == model_id && v == version);
        // The native call owns its model, buffers and permit through completion
        // even if this future is dropped. BorrowedTensor never escapes it.
        let model = Arc::clone(&loaded.model);
        let keep_instance = Arc::clone(&loaded);
        loaded
            .worker
            .run(move || {
                let result = predict(&model, inputs);
                permit.finish(result.is_ok());
                drop(keep_instance);
                result
            })
            .await
    }

    async fn warmup(&self, models: &[ModelId]) -> Result<()> {
        for &model in models {
            self.model(model, &ModelVersion::Latest).await?;
        }
        Ok(())
    }
}

fn predict(model: &Model, inputs: TensorMap) -> Result<TensorMap> {
    let storage = inputs
        .into_iter()
        .map(InputStorage::from_tensor)
        .collect::<Result<Vec<_>>>()?;
    let borrowed = storage
        .iter()
        .map(InputStorage::borrow)
        .collect::<Result<Vec<_>>>()?;
    let named_inputs = storage
        .iter()
        .zip(&borrowed)
        .map(|(storage, tensor)| (storage.name.as_str(), tensor as &dyn AsMultiArray))
        .collect::<Vec<_>>();
    let prediction = model
        .predict(&named_inputs)
        .map_err(|error| EngineError::inference(error.to_string()))?;

    let mut outputs = TensorMap::new();
    for description in model.outputs() {
        let name = description.name();
        let data_type = description.data_type().ok_or_else(|| {
            EngineError::inference(format!("CoreML output '{name}' has no tensor datatype"))
        })?;
        let tensor = match data_type {
            // `MLMultiArray` results produced by GPU/ANE execution are not
            // necessarily contiguous. `get_f32` follows the array's
            // logical strides and also gives every floating backend the
            // same FP32 boundary representation.
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                let (values, shape) = prediction
                    .get_f32(name)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                Tensor::from_f32(coreml_shape(shape), values)?
            }
            DataType::Int32 => {
                let (values, shape) = prediction
                    .get_i32(name)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                Tensor::from_i32(coreml_shape(shape), values)?
            }
            _ => {
                let (bytes, shape, raw_type) = prediction
                    .get_raw(name)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                let dtype = map_coreml_dtype(raw_type.ok_or_else(|| {
                    EngineError::inference(format!("CoreML output '{name}' has no tensor datatype"))
                })?)?;
                Tensor::new(coreml_shape(shape), dtype, bytes.into())?
            }
        };
        outputs.insert(name.to_owned(), tensor);
    }
    Ok(outputs)
}

struct InputStorage {
    name: String,
    shape: Vec<usize>,
    data: InputData,
}

enum InputData {
    F16(Vec<u16>),
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl InputStorage {
    fn from_tensor((name, tensor): (String, Tensor)) -> Result<Self> {
        let shape = tensor
            .shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).map_err(|_| {
                    EngineError::inference(format!(
                        "CoreML input '{name}' has negative dimension {dimension}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let data = match tensor.dtype {
            DType::Fp16 => InputData::F16(
                tensor
                    .to_f16()?
                    .into_iter()
                    .map(half::f16::to_bits)
                    .collect(),
            ),
            DType::Fp32 => InputData::F32(tensor.to_f32()?),
            DType::Int32 => InputData::I32(tensor.to_i32()?),
            dtype => {
                return Err(EngineError::Unsupported(format!(
                    "CoreML input '{name}' uses unsupported dtype {}",
                    dtype.as_str()
                )));
            }
        };
        Ok(Self { name, shape, data })
    }

    fn borrow(&self) -> Result<BorrowedTensor<'_>> {
        match &self.data {
            InputData::F16(values) => BorrowedTensor::from_f16_bits(values, &self.shape),
            InputData::F32(values) => BorrowedTensor::from_f32(values, &self.shape),
            InputData::I32(values) => BorrowedTensor::from_i32(values, &self.shape),
        }
        .map_err(|error| EngineError::inference(error.to_string()))
    }
}

fn coreml_shape(shape: Vec<usize>) -> Vec<i64> {
    shape
        .into_iter()
        .map(|dimension| dimension as i64)
        .collect()
}

fn map_coreml_dtype(dtype: DataType) -> Result<DType> {
    Ok(match dtype {
        DataType::Float16 => DType::Fp16,
        DataType::Float32 => DType::Fp32,
        DataType::Float64 => DType::Fp64,
        DataType::Int8 => DType::Int8,
        DataType::Int16 => DType::Int16,
        DataType::Int32 => DType::Int32,
        DataType::UInt8 => DType::UInt8,
        DataType::UInt16 => DType::UInt16,
        DataType::UInt32 => DType::UInt32,
    })
}

#[cfg(test)]
mod tests {
    use crate::ModelPaths;

    use super::*;

    #[test]
    fn a_half_input_keeps_its_dtype_and_exact_bits_at_the_native_door() {
        let bits = [0x8000, 0x3c00, 0x3555, 0x0001];
        let tensor = Tensor::from_f16(vec![1, 4], bits.map(half::f16::from_bits).to_vec()).unwrap();
        let input = InputStorage::from_tensor(("input".into(), tensor)).unwrap();
        assert!(matches!(&input.data, InputData::F16(values) if values == &bits));
        assert_eq!(input.borrow().unwrap().data_type(), DataType::Float16);
    }

    #[test]
    fn placements_pin_per_model_and_leave_the_rest_on_the_default() {
        let backend = CoreMlBackend::new(Arc::new(ModelPaths::new())).with_placements([
            (ModelId::Autoshot, ComputeUnits::CpuAndNeuralEngine),
            (ModelId::MusicHighlighter, ComputeUnits::CpuOnly),
        ]);
        assert_eq!(
            backend.compute_units_for(ModelId::Autoshot),
            ComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(
            backend.compute_units_for(ModelId::MusicHighlighter),
            ComputeUnits::CpuOnly
        );
        // Unpinned models keep the backend-wide floor rather than inheriting a
        // neighbour's pin.
        assert_eq!(
            backend.compute_units_for(ModelId::DashengBase),
            ComputeUnits::All
        );
    }

    #[test]
    fn a_steered_backend_moves_only_its_unpinned_models() {
        let backend =
            CoreMlBackend::with_compute_units(Arc::new(ModelPaths::new()), ComputeUnits::CpuAndGpu)
                .with_model_compute_units(ModelId::Autoshot, ComputeUnits::CpuAndNeuralEngine);
        assert_eq!(
            backend.compute_units_for(ModelId::Dfnet),
            ComputeUnits::CpuAndGpu
        );
        assert_eq!(
            backend.compute_units_for(ModelId::Autoshot),
            ComputeUnits::CpuAndNeuralEngine
        );
    }
}
