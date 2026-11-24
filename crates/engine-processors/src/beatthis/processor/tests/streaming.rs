use super::*;
use ferment_engine_core::AudioSampleSource;
use ferment_engine_infer::InferenceBackend;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
struct Oracle {
    calls: Mutex<Vec<Vec<f32>>>,
}
#[async_trait::async_trait]
impl InferenceBackend for Oracle {
    async fn infer(&self, _: ModelId, _: &ModelVersion, inputs: TensorMap) -> Result<TensorMap> {
        let mel = inputs["mel"].to_f32()?;
        self.calls.lock().unwrap().push(mel);
        let mut logits = vec![-8.; 2 * CHUNK_SIZE];
        for at in (10..CHUNK_SIZE).step_by(25) {
            logits[at] = 8.;
        }
        for at in (10..CHUNK_SIZE).step_by(100) {
            logits[CHUNK_SIZE + at] = 8.;
        }
        Ok(TensorMap::from([(
            "logits".into(),
            Tensor::from_f32(vec![1, 2, 1, CHUNK_SIZE as i64], logits)?,
        )]))
    }
}
struct Source {
    samples: Vec<f32>,
    maximum: AtomicUsize,
}
impl AudioSampleSource for Source {
    fn sample_count(&self) -> usize {
        self.samples.len()
    }
    fn copy_samples(&self, out: &mut [f32], at: usize, count: usize) -> Result<usize> {
        self.maximum.fetch_max(count, Ordering::Relaxed);
        out[..count].copy_from_slice(&self.samples[at..at + count]);
        Ok(count)
    }
}

#[test]
fn bounded_pipeline_preserves_every_model_window_and_final_grid() {
    let source = Source {
        samples: (0..1_345_137)
            .map(|i| (i as f32 * 0.061).sin() * 0.2)
            .collect(),
        maximum: AtomicUsize::new(0),
    };
    for post in [PostProcessing::Minimal, PostProcessing::Dbn] {
        let backend = Arc::new(Oracle::default());
        let detector = BeatThis::new(backend.clone()).with_post_processing(post);
        let expected = futures_executor::block_on(detector.detect(
            AudioFrames {
                samples: source.samples.clone(),
                sample_rate: SAMPLE_RATE,
                identity: None,
            },
            ProgressSink::noop(),
        ))
        .unwrap();
        let mut calls = std::mem::take(&mut *backend.calls.lock().unwrap());
        let mut scratch = || tempfile::tempfile().map_err(|e| EngineError::internal(e.to_string()));
        let actual = futures_executor::block_on(detector.detect_source(
            &source,
            &mut scratch,
            32768,
            ProgressSink::noop(),
        ))
        .unwrap();
        let mut paged_calls = std::mem::take(&mut *backend.calls.lock().unwrap());
        let order = |a: &Vec<f32>, b: &Vec<f32>| {
            a.iter()
                .zip(b)
                .map(|(a, b)| a.total_cmp(b))
                .find(|order| !order.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        calls.sort_by(order);
        paged_calls.sort_by(order);
        assert_eq!(
            paged_calls, calls,
            "every tensor must match, including overlap and final tail"
        );
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert!(source.maximum.load(Ordering::Relaxed) <= CHUNK_SIZE * 441 + 1024);
    }
}
