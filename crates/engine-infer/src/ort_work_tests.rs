use super::*;
use ferment_engine_core::{WorkActivity, WorkEvent, WorkPhase};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

fn graph() -> PathBuf {
    assert!(std::env::var_os("ORT_DYLIB_PATH").is_some());
    let path = PathBuf::from(
        std::env::var_os("FERMENT_WORK_YUNET_GRAPH").expect("name the pinned public YuNet graph"),
    );
    assert!(path.is_file(), "required graph: {}", path.display());
    path
}

fn backend(path: &Path) -> OrtBackend {
    OrtBackend::with_execution_provider(
        Arc::new(crate::ModelPaths::new().with_path(
            ModelId::YuNetFace,
            ModelVersion::Latest,
            path.to_owned(),
        )),
        OrtExecutionProvider::Cpu,
    )
}

fn inputs() -> TensorMap {
    TensorMap::from([(
        "input".into(),
        Tensor::from_f32(vec![1, 3, 640, 640], vec![0.0; 3 * 640 * 640]).unwrap(),
    )])
}

fn lane(backend: &OrtBackend) -> Arc<LoadedSession> {
    backend
        .sessions
        .peek(&(ModelId::YuNetFace, ModelVersion::Latest))
        .expect("the real graph must be resident")
}

/// Kill: remove the successful native forward's checkpoint; three completed ORT calls report zero activity.
#[test]
#[ignore = "requires pinned ORT and FERMENT_WORK_YUNET_GRAPH"]
fn real_ort_predictions_report_completed_work_without_ui_progress() {
    let backend = backend(&graph());
    let checkpoints = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&checkpoints);
    let work = WorkActivity::from_callback(move |event| {
        if event == WorkEvent::Checkpoint {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    });
    let _scope = work.enter();
    for _ in 0..3 {
        let output = futures_executor::block_on(backend.infer(
            ModelId::YuNetFace,
            &ModelVersion::Latest,
            inputs(),
        ))
        .unwrap();
        assert!(output.contains_key("cls_8") && output.contains_key("bbox_32"));
    }
    assert_eq!(lane(&backend).predictions.snapshot().completed, 3);
    assert_eq!(checkpoints.load(Ordering::SeqCst), 3);
}

/// Kill: report a checkpoint for a failed native forward; a refused input is not completed work.
#[test]
#[ignore = "requires pinned ORT and FERMENT_WORK_YUNET_GRAPH"]
fn a_failed_real_ort_prediction_does_not_report_completed_work() {
    let backend = backend(&graph());
    let checkpoints = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&checkpoints);
    let work = WorkActivity::from_callback(move |event| {
        if event == WorkEvent::Checkpoint {
            observed.fetch_add(1, Ordering::SeqCst);
        }
    });
    let mut invalid = inputs();
    let image = invalid.remove("input").unwrap();
    invalid.insert("not_a_yunet_input".into(), image);
    let _scope = work.enter();
    let result = futures_executor::block_on(backend.infer(
        ModelId::YuNetFace,
        &ModelVersion::Latest,
        invalid,
    ));
    assert!(matches!(result, Err(EngineError::Inference(_))));
    assert_eq!(lane(&backend).predictions.snapshot().failed, 1);
    assert_eq!(checkpoints.load(Ordering::SeqCst), 0);
}

/// Kill: omit the post-load cancellation check; the actual graph records a prediction after its scope closes.
#[test]
#[ignore = "requires pinned ORT and FERMENT_WORK_YUNET_GRAPH"]
fn cancellation_at_real_load_completion_cannot_start_a_prediction() {
    let backend = backend(&graph());
    let cancelled = Arc::new(AtomicBool::new(false));
    let close = Arc::clone(&cancelled);
    let loading = Arc::new(AtomicU64::new(u64::MAX));
    let phase = Arc::clone(&loading);
    let work = WorkActivity::from_callback(move |event| match event {
        WorkEvent::Started {
            id,
            phase: Some(WorkPhase::LoadingModel),
        } => phase.store(id, Ordering::SeqCst),
        WorkEvent::Finished { id } if id == phase.load(Ordering::SeqCst) => {
            close.store(true, Ordering::SeqCst);
        }
        _ => {}
    })
    .with_cancellation(Arc::clone(&cancelled));
    let result = {
        let _scope = work.enter();
        futures_executor::block_on(backend.infer(
            ModelId::YuNetFace,
            &ModelVersion::Latest,
            inputs(),
        ))
    };
    assert!(
        cancelled.load(Ordering::SeqCst),
        "the real load phase never ended"
    );
    assert_eq!(lane(&backend).predictions.snapshot().peak_active, 0);
    assert!(matches!(result, Err(EngineError::Cancelled)));
}

/// Kill: use thread-local cancellation after the permit wait; the queued ORT call runs after an executor hop.
#[test]
#[ignore = "requires pinned ORT and FERMENT_WORK_YUNET_GRAPH"]
fn a_cancelled_prediction_waiter_cannot_run_when_its_permit_arrives() {
    let backend = backend(&graph());
    futures_executor::block_on(backend.warmup(&[ModelId::YuNetFace])).unwrap();
    let loaded = lane(&backend);
    let held = futures_executor::block_on(loaded.predictions.acquire());
    let cancelled = Arc::new(AtomicBool::new(false));
    let work = WorkActivity::from_callback(|_| {}).with_cancellation(Arc::clone(&cancelled));
    let scope = work.enter();
    let mut pending = backend.infer(ModelId::YuNetFace, &ModelVersion::Latest, inputs());
    let mut context = Context::from_waker(Waker::noop());
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(loaded.predictions.snapshot().waiting, 1);
    cancelled.store(true, Ordering::SeqCst);
    drop(scope);
    held.finish(false);
    let result = std::thread::scope(|threads| {
        threads
            .spawn(move || {
                assert!(WorkActivity::current().is_none());
                futures_executor::block_on(pending)
            })
            .join()
            .unwrap()
    });
    let snapshot = loaded.predictions.snapshot();
    assert_eq!(snapshot.active, 0);
    assert_eq!(snapshot.waiting, 0);
    assert_eq!(snapshot.completed, 0);
    assert!(matches!(result, Err(EngineError::Cancelled)));
}

/// Kill: resolve/invalidate graph identity before checking the closed scope; its resident graph disappears.
#[test]
#[ignore = "requires pinned ORT and FERMENT_WORK_YUNET_GRAPH"]
fn a_closed_scope_cannot_invalidate_a_resident_graph_before_lookup() {
    let directory = std::env::temp_dir().join(format!(
        "ferment-ort-closed-scope-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let copied = directory.join("yunet.onnx");
    std::fs::copy(graph(), &copied).unwrap();
    let backend = backend(&copied);
    futures_executor::block_on(backend.warmup(&[ModelId::YuNetFace])).unwrap();
    std::fs::remove_file(&copied).unwrap();
    let work =
        WorkActivity::from_callback(|_| {}).with_cancellation(Arc::new(AtomicBool::new(true)));
    let result = {
        let _scope = work.enter();
        futures_executor::block_on(backend.infer(
            ModelId::YuNetFace,
            &ModelVersion::Latest,
            inputs(),
        ))
    };
    let snapshot = backend.cache_snapshot();
    std::fs::remove_dir(directory).unwrap();
    assert_eq!(snapshot.resident, 1);
    assert!(matches!(result, Err(EngineError::Cancelled)));
}
