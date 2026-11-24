//! Ported from `Processors/EdgeTAM/EdgeTAMError.swift`.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EdgeTamError {
    #[error("Frame {index} is outside the video range 0..<{frame_count}.")]
    InvalidFrameIndex { index: usize, frame_count: usize },
    #[error("Apply at least one EdgeTAM prompt before propagation.")]
    NoPromptedObjects,
    #[error("EdgeTAM received {actual} prompt tokens; this build supports at most {maximum}.")]
    TooManyPromptTokens { actual: usize, maximum: usize },
    #[error(
        "EdgeTAM image session received {actual} active object IDs; this build supports at most \
         {maximum}. Remove an object or reset the session first."
    )]
    TooManyImageObjects { actual: usize, maximum: usize },
    #[error(
        "EdgeTAM received prompts on {actual} video frames; the V1 memory budget supports at most \
         {maximum}."
    )]
    TooManyPromptFrames { actual: usize, maximum: usize },
    #[error("A box must start a prompt because EdgeTAM requires box-corner tokens to come first.")]
    CannotAppendBox,
    #[error("The EdgeTAM operation was superseded by reset.")]
    StaleOperation,
    #[error("Invalid EdgeTAM model boundary: {0}")]
    InvalidModelBoundary(String),
    #[error("Unsupported EdgeTAM V1 reference profile: {0}")]
    UnsupportedReferenceProfile(String),
}

impl From<EdgeTamError> for ferment_engine_core::EngineError {
    fn from(error: EdgeTamError) -> Self {
        Self::Inference(error.to_string())
    }
}
