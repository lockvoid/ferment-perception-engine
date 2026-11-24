//! Bulk PCM boundary matching the Android client's incremental capture API.
//! The processor still reads one growing source; only new bytes cross UniFFI.
use std::sync::Mutex;

use ferment_engine_core::{self as core, AudioSampleSource};

use crate::{EngineFfiError, FfiResult, LiveTranscript, TranscriptionResult};

struct Take {
    session: Box<dyn core::LiveTranscriptionSession>,
    samples: Vec<f32>,
}

impl Take {
    fn append(&mut self, bytes: &[u8]) -> FfiResult<()> {
        if !bytes.len().is_multiple_of(4) {
            return Err(EngineFfiError::new(
                "live PCM must contain whole little-endian f32 samples",
            ));
        }
        let values = || {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        if values().any(|value| !value.is_finite()) {
            return Err(EngineFfiError::new("live PCM contains a non-finite sample"));
        }
        self.samples
            .try_reserve(bytes.len() / 4)
            .map_err(|error| EngineFfiError::new(format!("cannot extend live PCM: {error}")))?;
        self.samples.extend(values());
        Ok(())
    }
}

// Borrow the take while its session runs, rather than cloning the whole take
// into ArrayAudioSampleSource on every incremental update.
struct BorrowedTake<'a>(&'a [f32]);

impl AudioSampleSource for BorrowedTake<'_> {
    fn sample_count(&self) -> usize {
        self.0.len()
    }

    fn copy_samples(&self, output: &mut [f32], offset: usize, count: usize) -> core::Result<usize> {
        let count = self
            .0
            .len()
            .saturating_sub(offset)
            .min(count)
            .min(output.len());
        if count != 0 {
            output[..count].copy_from_slice(&self.0[offset..offset + count]);
        }
        Ok(count)
    }
}

fn closed() -> EngineFfiError {
    EngineFfiError::new("live transcription session is closed")
}

/// One take's worth of live transcription over a buffer that is still being
/// written. The take accumulates here: each call crosses only what the
/// capture added since the previous one, so a minute of dictation is not
/// re-sent ten times a second.
#[derive(uniffi::Object)]
pub struct LiveTranscriptionSession(Mutex<Option<Take>>);

impl LiveTranscriptionSession {
    pub(crate) fn new(session: Box<dyn core::LiveTranscriptionSession>) -> Self {
        Self(Mutex::new(Some(Take {
            session,
            samples: Vec::new(),
        })))
    }
}

#[uniffi::export]
impl LiveTranscriptionSession {
    /// Append what the capture added since the previous call and answer the
    /// current hypothesis; the session reads the prefix of the take it has
    /// not consumed yet.
    pub fn update(&self, appended: Vec<u8>) -> FfiResult<LiveTranscript> {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let take = state.as_mut().ok_or_else(closed)?;
        take.append(&appended)?;
        // Accepted samples remain owned even if inference fails. A retry sends
        // an empty delta, as the Android capture has already advanced its cursor.
        let transcript =
            futures_executor::block_on(take.session.update(&BorrowedTake(&take.samples)))
                .map_err(EngineFfiError::from)?;
        Ok(LiveTranscript {
            committed: transcript.committed.into_iter().map(Into::into).collect(),
            volatile: transcript.volatile.into_iter().map(Into::into).collect(),
        })
    }

    /// Append the take's last samples and run once over all of it. Its result
    /// is the same one the file path would produce for that audio.
    pub fn finish(&self, appended: Vec<u8>) -> FfiResult<TranscriptionResult> {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.as_mut().ok_or_else(closed)?.append(&appended)?;
        let mut take = state.take().ok_or_else(closed)?;
        drop(state);
        // Closing releases both audio and processor ownership even on failure.
        futures_executor::block_on(take.session.finish(&BorrowedTake(&take.samples)))
            .map(Into::into)
            .map_err(EngineFfiError::from)
    }

    pub fn cancel(&self) {
        let take = self
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(mut take) = take {
            futures_executor::block_on(take.session.cancel());
        }
    }
}

#[cfg(test)]
mod tests;
