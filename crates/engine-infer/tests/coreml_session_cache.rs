//! Real native load/infer through the lifecycle facade; no production weights.
#![cfg(all(feature = "coreml", target_vendor = "apple"))]

use ferment_engine_infer::{
    ComputeUnits, CoreMlBackend, InferenceBackend, ModelId, ModelPaths, ModelVersion, Tensor,
    TensorMap,
};
use futures_executor::block_on;
use std::sync::Arc;

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../vendor/coreml-native/tests/fixtures/test_linear.mlmodelc"
);
const ID: ModelId = ModelId::DashengBase;

fn backend() -> CoreMlBackend {
    CoreMlBackend::with_compute_units(
        Arc::new(ModelPaths::new().with_path(ID, ModelVersion::Latest, MODEL)),
        ComputeUnits::CpuOnly,
    )
}

fn infer(backend: &CoreMlBackend) {
    let inputs = TensorMap::from([(
        "input".into(),
        Tensor::from_f32(vec![1, 4], vec![1., 2., 3., 4.]).unwrap(),
    )]);
    let output = block_on(backend.infer(ID, &ModelVersion::Latest, inputs)).unwrap();
    assert_eq!(output["output"].to_f32().unwrap(), [3., 5., 7., 9.]);
}

#[test]
fn warmup_and_predictions_share_one_instance_and_release_waiters() {
    let backend = Arc::new(backend());
    block_on(backend.warmup(&[ID])).unwrap();
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let backend = Arc::clone(&backend);
            std::thread::spawn(move || infer(&backend))
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    let cache = backend.cache_snapshot();
    assert_eq!((cache.resident, cache.loading, cache.waiters), (1, 0, 0));
    let predictions = block_on(backend.prediction_snapshot(ID, &ModelVersion::Latest)).unwrap();
    assert_eq!(
        (
            predictions.active,
            predictions.waiting,
            predictions.completed,
            predictions.failed
        ),
        (0, 0, 8, 0)
    );
    assert_eq!(predictions.peak_active, 1);
    assert!(backend.evict_idle_models(&[ID]).is_empty());
    assert_eq!(backend.evict_idle_models(&[]), [ID]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    infer(&backend);
    assert_eq!(
        block_on(backend.prediction_snapshot(ID, &ModelVersion::Latest))
            .unwrap()
            .completed,
        1
    );
}

#[test]
fn changing_placement_invalidates_loaded_specialization_but_same_units_reuse_it() {
    let backend = backend();
    infer(&backend);
    let backend = backend.with_model_compute_units(ID, ComputeUnits::CpuOnly);
    assert_eq!(backend.cache_snapshot().resident, 1);
    let backend = backend.with_model_compute_units(ID, ComputeUnits::CpuAndNeuralEngine);
    assert_eq!(backend.cache_snapshot().resident, 0);
    infer(&backend);
    assert_eq!(
        block_on(backend.prediction_snapshot(ID, &ModelVersion::Latest))
            .unwrap()
            .completed,
        1
    );
    assert!(backend.invalidate_model(ID));
    assert!(!backend.invalidate_model(ID));
}

#[test]
fn missing_model_does_not_leave_loading_slots_or_waiters() {
    let backend = CoreMlBackend::new(Arc::new(ModelPaths::new()));
    for _ in 0..3 {
        assert!(block_on(backend.warmup(&[ID])).is_err());
    }
    let cache = backend.cache_snapshot();
    assert_eq!((cache.resident, cache.loading, cache.waiters), (0, 0, 0));
}
