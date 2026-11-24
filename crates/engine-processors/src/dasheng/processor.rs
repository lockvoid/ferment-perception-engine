//! Orchestrator: decode → resample → chunk → mel → encode each window → mean-pool.

use ferment_engine_audio::derived_cache::DerivedFileCache;
use ferment_engine_audio::streaming_resampler::resample_current;
use std::sync::Arc;

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};

use super::chunking::chunk_into_windows;
use super::config::{EMBEDDING_DIM, PATCH_CHANNELS, PATCH_COUNT, SAMPLE_RATE, WINDOW_SECONDS};
use super::frontend::{mel, patches};
use super::pool::mean_pool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEmbedding {
    /// Track-level mean-pooled embedding. Always `EMBEDDING_DIM` long.
    pub pooled: Vec<f32>,
    /// Per-window embeddings, present only when `return_chunks=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<Vec<f32>>>,
    pub duration_seconds: f64,
    pub window_seconds: usize,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct AudioSemanticEncoder {
    pub(crate) cache: Option<Arc<DerivedFileCache>>,
    backend: Backend,
    version: ModelVersion,
}

impl AudioSemanticEncoder {
    pub fn with_cache(mut self, cache: Option<Arc<DerivedFileCache>>) -> Self {
        self.cache = cache;
        self
    }

    pub fn new(backend: Backend) -> Self {
        Self {
            cache: None,
            backend,
            version: ModelVersion::default(),
        }
    }

    pub fn with_version(backend: Backend, version: ModelVersion) -> Self {
        Self {
            backend,
            version,
            cache: None,
        }
    }

    #[tracing::instrument(skip_all, fields(processor = "dasheng", samples = frames.len(), return_chunks))]
    pub async fn encode(
        &self,
        frames: AudioFrames,
        return_chunks: bool,
        progress: ProgressSink,
    ) -> Result<AudioEmbedding> {
        progress.report(0.05, "loading audio").await;
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_current(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        if audio.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        let duration_seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);

        let windows = chunk_into_windows(&audio);
        let n_windows = windows.len();
        progress.report(0.10, "encoding").await;

        let mut chunk_embeddings = Vec::with_capacity(n_windows);
        for (i, window) in windows.into_iter().enumerate() {
            let emb = self.encode_window(window).await?;
            chunk_embeddings.push(emb);
            // Progress span 0.10 → 0.95 across windows
            let frac = 0.10 + 0.85 * (i + 1) as f64 / n_windows as f64;
            progress
                .report(frac, format!("encoding ({}/{})", i + 1, n_windows))
                .await;
        }

        let pooled = mean_pool(&chunk_embeddings);
        progress.report(1.0, "done").await;

        Ok(AudioEmbedding {
            pooled,
            chunks: if return_chunks {
                Some(chunk_embeddings)
            } else {
                None
            },
            duration_seconds,
            window_seconds: WINDOW_SECONDS,
            dim: EMBEDDING_DIM,
        })
    }

    async fn encode_window(&self, window: Vec<f32>) -> Result<Vec<f32>> {
        let mut inputs: TensorMap = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(
                vec![1, PATCH_CHANNELS as i64, 1, PATCH_COUNT as i64],
                patches(&mel(&window)?),
            )?,
        );
        let outputs = self
            .backend
            .infer(ModelId::DashengBase, &self.version, inputs)
            .await?;

        let tensor = outputs
            .get("embedding")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::DashengBase.name().into(),
                key: "embedding".into(),
            })?;
        let emb = tensor.to_f32()?;
        if emb.len() != EMBEDDING_DIM {
            return Err(EngineError::InvalidOutput);
        }
        Ok(emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_embedding_honors_return_chunks_flag() {
        let zero = vec![0.0; EMBEDDING_DIM];
        let with_chunks = AudioEmbedding {
            pooled: zero.clone(),
            chunks: Some(vec![zero.clone()]),
            duration_seconds: 10.0,
            window_seconds: WINDOW_SECONDS,
            dim: EMBEDDING_DIM,
        };
        assert_eq!(with_chunks.chunks.as_ref().unwrap().len(), 1);

        let without_chunks = AudioEmbedding {
            pooled: zero,
            chunks: None,
            duration_seconds: 10.0,
            window_seconds: WINDOW_SECONDS,
            dim: EMBEDDING_DIM,
        };
        assert!(without_chunks.chunks.is_none());
    }
}
