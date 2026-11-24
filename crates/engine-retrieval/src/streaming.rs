use crate::{Embedding, GLAP_MAX_SAMPLES, Glap, check_cancelled};
use ferment_engine_core::{AudioSampleSource, EngineError, ProgressSink, Result};
use serde::{Deserialize, Serialize};

/// Time is explicit and lossless at 16 kHz; the last window's validity is not
/// enlarged by the minimum 512-sample pad required by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioWindowEmbedding {
    pub start_sample: u64,
    pub sample_count: u32,
    pub embedding: Embedding,
}

impl Glap {
    /// Consume a mono 16 kHz source using one reusable <=640 KB PCM window.
    /// Source decoding/resampling is the existing audio frontend's job.
    /// `on_window` runs synchronously on the caller's analysis worker, after
    /// each prediction. It supplies backpressure and can fail/cancel without
    /// retaining every embedding of a long recording in memory.
    pub async fn audio_source_16k(
        &self,
        source: &dyn AudioSampleSource,
        progress: &ProgressSink,
        mut on_window: impl FnMut(AudioWindowEmbedding) -> Result<()>,
    ) -> Result<()> {
        check_cancelled(progress)?;
        let count = source.sample_count();
        if count == 0 {
            return Err(EngineError::NoAudioData);
        }
        let mut samples = vec![0.0; count.min(GLAP_MAX_SAMPLES)];
        let mut offset = 0;
        while offset < count {
            check_cancelled(progress)?;
            let required = samples.len().min(count - offset);
            let mut copied = 0;
            while copied < required {
                check_cancelled(progress)?;
                let read = source.copy_samples(
                    &mut samples[copied..required],
                    offset + copied,
                    required - copied,
                )?;
                if read == 0 || read > required - copied {
                    return Err(EngineError::Decode(
                        "retrieval source returned an invalid or premature short read".into(),
                    ));
                }
                copied += read;
            }
            let embedding = self.audio_window(&samples[..required], progress).await?;
            check_cancelled(progress)?;
            on_window(AudioWindowEmbedding {
                start_sample: offset as u64,
                sample_count: required as u32,
                embedding,
            })?;
            offset += required;
            progress
                .report(offset as f64 / count as f64, "audio retrieval")
                .await;
        }
        check_cancelled(progress)
    }
}
