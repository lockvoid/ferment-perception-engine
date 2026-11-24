use std::io;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("model {model} returned unexpected output: missing key {key}")]
    ModelOutputMissing { model: String, key: String },
    #[error("model returned unexpected output format")]
    InvalidOutput,
    #[error("no audio data in input")]
    NoAudioData,
    #[error("audio too short: {samples} samples, need at least {min_required}")]
    AudioTooShort { samples: usize, min_required: usize },
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("STFT failed: {0}")]
    StftFailed(String),
    #[error("audio decode failed: {0}")]
    Decode(String),
    #[error("audio encode failed: {0}")]
    Encode(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("video I/O failed: {0}")]
    VideoIo(String),
    #[error("model fleet error: {0}")]
    Models(String),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("operation cancelled")]
    Cancelled,
    #[error("unsupported on this platform: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl EngineError {
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn decode(message: impl Into<String>) -> Self {
        Self::Decode(message.into())
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference(message.into())
    }
}
