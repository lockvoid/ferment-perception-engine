use std::io;

/// Errors surfaced by demuxers and platform codec sessions.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    #[error("media I/O failed: {0}")]
    Io(#[from] io::Error),

    #[error("invalid media data: {0}")]
    InvalidData(String),

    #[error("unsupported container: {0}")]
    UnsupportedContainer(String),

    #[error("unsupported codec: {0}")]
    UnsupportedCodec(String),

    #[error("demux failed: {0}")]
    Demux(String),

    #[error("platform media backend is unavailable: {0}")]
    BackendUnavailable(String),

    #[error("platform media operation failed: {operation} returned status {status}")]
    Platform {
        operation: &'static str,
        status: i32,
    },

    #[error("external developer tool failed: {0}")]
    DevTool(String),

    /// A libav call failed. `call` names the function so the message reads as
    /// a place rather than a number, and `message` is `av_strerror`'s own
    /// text. Every libav status crosses through
    /// `platform::libav::check_av`, so this is the only shape a libav failure
    /// ever takes.
    #[error("{call} failed: {message} (libav {code})")]
    Libav {
        code: i32,
        call: &'static str,
        message: String,
    },
}

impl MediaError {
    #[cfg(any(target_os = "macos", target_os = "android"))]
    pub(crate) fn platform(operation: &'static str, status: i32) -> Self {
        Self::Platform { operation, status }
    }
}

/// Compatibility name retained for the original video-only API.
pub type VideoError = MediaError;

pub type Result<T> = std::result::Result<T, MediaError>;
