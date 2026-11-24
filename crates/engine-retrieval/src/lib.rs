//! Text-aligned audio/image retrieval, independent of any database or UI.
//!
//! Construct encoders on a job worker (tokenizer loading is synchronous), then
//! reuse them. Supply model paths from the verified fleet. Native prediction
//! ownership, concurrency and cancellation remain the inference backend's.
//! No library scanning, cache policy or implicit whole-file pooling lives here.

mod embedding;
mod glap;
mod resize;
mod siglip2;
mod streaming;
mod tokenization;

pub use embedding::{Embedding, EmbeddingSpace};
pub use glap::{GLAP_MAX_SAMPLES, GLAP_SAMPLE_RATE, Glap, GlapFrontend};
pub use siglip2::{SIGLIP2_IMAGE_SIZE, Siglip2, siglip2_pixels};
pub use streaming::AudioWindowEmbedding;
pub use tokenization::{GlapLanguage, GlapTokenizer, Siglip2Tokenizer};

use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

async fn encode(
    backend: &Backend,
    model: ModelId,
    version: &ModelVersion,
    inputs: TensorMap,
    space: EmbeddingSpace,
    progress: &ProgressSink,
) -> Result<Embedding> {
    let outputs = progress
        .until_cancelled(backend.infer(model, version, inputs))
        .await?;
    let output = outputs
        .get("embedding")
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: model.name().into(),
            key: "embedding".into(),
        })?;
    if output.shape != [1, space.dimensions() as i64] {
        return Err(EngineError::InvalidOutput);
    }
    Embedding::from_unit_values(space, output.to_f32()?)
}

fn token_input(ids: Vec<i32>) -> Result<TensorMap> {
    Ok(TensorMap::from([(
        "token_ids".into(),
        Tensor::from_i32(vec![1, ids.len() as i64], ids)?,
    )]))
}

fn check_cancelled(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}
