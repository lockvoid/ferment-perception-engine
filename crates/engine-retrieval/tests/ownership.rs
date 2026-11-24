use async_trait::async_trait;
use ferment_engine_core::{AudioSampleSource, EngineError, ProgressSink, Result};
use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_retrieval::{GLAP_MAX_SAMPLES, Glap};
use futures_executor::block_on;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

#[derive(Debug, Default)]
struct UnitBackend(AtomicUsize);
#[async_trait]
impl InferenceBackend for UnitBackend {
    async fn infer(&self, model: ModelId, _: &ModelVersion, _: TensorMap) -> Result<TensorMap> {
        assert_eq!(model, ModelId::GlapAudio);
        self.0.fetch_add(1, Ordering::SeqCst);
        let mut values = vec![0.0; 1024];
        values[0] = 1.0;
        Ok(TensorMap::from([(
            "embedding".into(),
            Tensor::from_f32(vec![1, 1024], values)?,
        )]))
    }
}

struct SyntheticSource(usize);
impl AudioSampleSource for SyntheticSource {
    fn sample_count(&self) -> usize {
        self.0
    }
    fn copy_samples(&self, destination: &mut [f32], offset: usize, count: usize) -> Result<usize> {
        assert!(destination.len() <= GLAP_MAX_SAMPLES);
        let copied = (self.0 - offset).min(count).min(1043);
        destination[..copied].fill(0.0);
        Ok(copied)
    }
}

#[test]
fn ten_minutes_are_streamed_in_bounded_windows_and_tail_time_is_exact() {
    let backend = Arc::new(UnitBackend::default());
    let encoder = Glap::new(backend.clone(), ModelVersion::Latest).unwrap();
    let source = SyntheticSource(60 * GLAP_MAX_SAMPLES + 17);
    let mut ends = vec![];
    block_on(
        encoder.audio_source_16k(&source, &ProgressSink::noop(), |window| {
            ends.push((window.start_sample, window.sample_count));
            Ok(())
        }),
    )
    .unwrap();
    assert_eq!(ends.len(), 61);
    assert_eq!(backend.0.load(Ordering::SeqCst), 61);
    assert_eq!(ends[60], ((60 * GLAP_MAX_SAMPLES) as u64, 17));
    assert!(
        ends[..60]
            .iter()
            .enumerate()
            .all(|(i, &(start, len))| start == (i * GLAP_MAX_SAMPLES) as u64
                && len == GLAP_MAX_SAMPLES as u32)
    );
}

#[test]
fn cancellation_stops_before_the_next_window_and_sink_errors_propagate() {
    let backend = Arc::new(UnitBackend::default());
    let encoder = Glap::new(backend.clone(), ModelVersion::Latest).unwrap();
    let source = SyntheticSource(GLAP_MAX_SAMPLES * 3);
    let progress = ProgressSink::noop();
    let result = block_on(encoder.audio_source_16k(&source, &progress, |_| {
        progress.cancel();
        Ok(())
    }));
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(backend.0.load(Ordering::SeqCst), 1);
    let result = block_on(
        encoder.audio_source_16k(&source, &ProgressSink::noop(), |_| {
            Err(EngineError::internal("sink failed"))
        }),
    );
    assert!(result.unwrap_err().to_string().contains("sink failed"));
    assert_eq!(backend.0.load(Ordering::SeqCst), 2);
}

#[derive(Debug)]
struct PendingBackend(mpsc::Sender<()>);
#[async_trait]
impl InferenceBackend for PendingBackend {
    async fn infer(&self, _: ModelId, _: &ModelVersion, _: TensorMap) -> Result<TensorMap> {
        self.0.send(()).unwrap();
        std::future::pending().await
    }
}

#[test]
fn cancelling_during_prediction_detaches_the_retrieval_caller() {
    let (started, wait) = mpsc::channel();
    let encoder = Glap::new(Arc::new(PendingBackend(started)), ModelVersion::Latest).unwrap();
    let progress = ProgressSink::noop();
    let cancel = progress.clone();
    let worker = std::thread::spawn(move || {
        wait.recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        cancel.cancel();
    });
    let result = block_on(encoder.audio_window(&[0.0; 512], &progress));
    worker.join().unwrap();
    assert!(matches!(result, Err(EngineError::Cancelled)));
}
