//! Real CPU ORT sessions, in addition to the deterministic loader races.
//! Requires explicit immutable model/runtime roots; no sibling fallback.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_infer::{
    InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
    TensorMap,
};
use futures_executor::block_on;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferment-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn master_file(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
            .expect("set explicit FERMENT_ENGINE_MODELS_DIR to the frozen model capture"),
    )
    .join(name);
    assert!(path.is_file(), "required frozen graph: {}", path.display());
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set explicit pinned ORT_DYLIB_PATH"
    );
    path
}

fn master() -> PathBuf {
    master_file("yunet_face.onnx")
}

fn backend(path: &Path) -> OrtBackend {
    OrtBackend::new(Arc::new(ModelPaths::new().with_path(
        ModelId::YuNetFace,
        ModelVersion::Latest,
        path.to_owned(),
    )))
}

fn infer(backend: &OrtBackend) -> TensorMap {
    let mut input = TensorMap::new();
    input.insert(
        "input".into(),
        Tensor::from_f32(vec![1, 3, 32, 32], vec![0.25; 3 * 32 * 32]).unwrap(),
    );
    block_on(backend.infer(ModelId::YuNetFace, &ModelVersion::Latest, input)).unwrap()
}

fn assert_outputs(actual: &TensorMap, reference: &TensorMap) {
    assert_eq!(
        actual.len(),
        12,
        "all three strides and four YuNet output families"
    );
    assert_eq!(actual.len(), reference.len());
    for (name, tensor) in reference {
        assert_eq!(actual[name].shape, tensor.shape);
        assert_eq!(
            actual[name].to_f32().unwrap(),
            tensor.to_f32().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn missing_model_fails_repeatedly_without_a_phantom_resident_or_waiter() {
    let root = Directory::new();
    let backend = backend(&root.0.join("missing.onnx"));
    for _ in 0..2 {
        assert!(block_on(backend.warmup(&[ModelId::YuNetFace])).is_err());
        assert_eq!(backend.cache_snapshot().resident, 0);
        assert_eq!(backend.cache_snapshot().loading, 0);
        assert_eq!(backend.cache_snapshot().waiters, 0);
        assert_eq!(backend.selected_provider_for(ModelId::YuNetFace), None);
    }
}

#[test]
#[ignore = "requires explicit pinned ORT and YuNet"]
fn real_graph_evicts_reloads_and_preserves_prediction_outputs() {
    let backend = backend(&master());
    let reference = infer(&backend);
    assert_eq!(backend.cache_snapshot().resident, 1);
    assert_eq!(
        backend.selected_provider_for(ModelId::YuNetFace),
        Some(OrtExecutionProvider::Cpu)
    );
    assert!(backend.evict_idle_models(&[ModelId::YuNetFace]).is_empty());
    assert_eq!(backend.evict_idle_models(&[]), [ModelId::YuNetFace]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    assert_eq!(backend.selected_provider_for(ModelId::YuNetFace), None);
    assert_outputs(&infer(&backend), &reference);
    assert_eq!(backend.cache_snapshot().resident, 1);
}

#[test]
#[ignore = "requires explicit pinned ORT and YuNet"]
fn same_path_equal_size_replacement_is_reloaded_and_bad_bytes_do_not_poison_retry() {
    let root = Directory::new();
    let installed = root.0.join("installed.onnx");
    let master = master();
    std::fs::copy(&master, &installed).unwrap();
    let backend = backend(&installed);
    block_on(backend.warmup(&[ModelId::YuNetFace])).unwrap();
    // The replacement has the same path and byte length. A key made from
    // only the logical model/version or path/length would return old success.
    let bad = root.0.join("replacement.onnx");
    std::fs::write(
        &bad,
        vec![0; std::fs::metadata(&installed).unwrap().len() as usize],
    )
    .unwrap();
    std::fs::rename(&bad, &installed).unwrap();
    assert!(block_on(backend.warmup(&[ModelId::YuNetFace])).is_err());
    assert_eq!(backend.cache_snapshot().resident, 0);
    assert_eq!(backend.cache_snapshot().loading, 0);
    assert_eq!(backend.selected_provider_for(ModelId::YuNetFace), None);
    std::fs::copy(&master, &installed).unwrap();
    block_on(backend.warmup(&[ModelId::YuNetFace])).unwrap();
    assert_eq!(backend.cache_snapshot().resident, 1);
}

#[test]
#[ignore = "requires explicit pinned ORT and YuNet"]
fn targeted_invalidation_preserves_other_models_and_provider_reconfiguration_retires_its_session() {
    let path = master();
    let paths = ModelPaths::new()
        .with_path(ModelId::YuNetFace, ModelVersion::Latest, path.clone())
        .with_path(ModelId::U2Netp, ModelVersion::Latest, path);
    // Only session loading is exercised for the second logical model: no
    // claim that YuNet supplies a U2Netp processor's tensor contract.
    let backend = OrtBackend::new(Arc::new(paths));
    block_on(backend.warmup(&[ModelId::YuNetFace, ModelId::U2Netp])).unwrap();
    assert_eq!(backend.cache_snapshot().resident, 2);
    assert!(backend.invalidate_model(ModelId::YuNetFace));
    assert!(!backend.invalidate_model(ModelId::YuNetFace));
    assert_eq!(backend.cache_snapshot().resident, 1);
    assert_eq!(
        backend.selected_provider_for(ModelId::U2Netp),
        Some(OrtExecutionProvider::Cpu)
    );
    let backend = backend.with_model_provider_chain(
        ModelId::U2Netp,
        [OrtExecutionProvider::Cpu, OrtExecutionProvider::Xnnpack],
    );
    assert_eq!(
        backend.cache_snapshot().resident,
        0,
        "configuration is part of session identity"
    );
    block_on(backend.warmup(&[ModelId::U2Netp])).unwrap();
    assert_eq!(
        backend.selected_provider_for(ModelId::U2Netp),
        Some(OrtExecutionProvider::Cpu)
    );
}

#[test]
#[ignore = "requires explicit pinned ORT and YuNet"]
fn memory_sweeps_racing_real_predictions_leave_every_accepted_call_safe() {
    let backend = Arc::new(backend(&master()));
    let reference = infer(&backend);
    let barrier = Arc::new(std::sync::Barrier::new(5));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let backend = Arc::clone(&backend);
            let barrier = Arc::clone(&barrier);
            let reference = &reference;
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..4 {
                    assert_outputs(&infer(&backend), reference);
                }
            });
        }
        barrier.wait();
        for _ in 0..32 {
            backend.evict_idle_models(&[]);
            std::thread::yield_now();
        }
    });
    assert_outputs(&infer(&backend), &reference);
    assert_eq!(backend.cache_snapshot().loading, 0);
    assert_eq!(backend.cache_snapshot().waiters, 0);
}

/// This grades the path→identity→cache-weight plumbing too: passing each
/// logical cache key to a file stat, or forwarding zero weights, cannot
/// reproduce the order of two differently-sized real published graphs.
#[test]
#[ignore = "requires explicit pinned ORT, YuNet and U2Netp"]
fn eviction_uses_actual_graph_file_weights_for_its_order() {
    let face = master_file("yunet_face.onnx");
    let saliency = master_file("u2netp.onnx");
    let mut expected = [
        (ModelId::YuNetFace, face.metadata().unwrap().len()),
        (ModelId::U2Netp, saliency.metadata().unwrap().len()),
    ];
    assert_ne!(
        expected[0].1, expected[1].1,
        "independent fixtures must distinguish the order"
    );
    expected.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
    let paths = ModelPaths::new()
        .with_path(ModelId::YuNetFace, ModelVersion::Latest, face)
        .with_path(ModelId::U2Netp, ModelVersion::Latest, saliency);
    let backend = OrtBackend::new(Arc::new(paths));
    block_on(backend.warmup(&[ModelId::YuNetFace, ModelId::U2Netp])).unwrap();
    assert_eq!(backend.cache_snapshot().resident, 2);
    assert_eq!(backend.evict_idle_models(&[]), expected.map(|(id, _)| id));
    assert_eq!(backend.cache_snapshot().resident, 0);
}
