use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    AudioTiming, DecoderConfiguration, EncodedPacket, MediaInfo, MediaTime, Result, VideoError,
};

/// Interleaved native-endian `f32` PCM at the source sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct PcmAudioFrame {
    pub sample_rate: u32,
    pub channels: u16,
    /// Presentation time of the first sample frame.
    pub start: MediaTime,
    pub samples: Arc<[f32]>,
}

impl PcmAudioFrame {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        start: MediaTime,
        samples: impl Into<Arc<[f32]>>,
    ) -> Result<Self> {
        let samples = samples.into();
        if sample_rate == 0 || channels == 0 {
            return Err(VideoError::InvalidData(
                "PCM sample rate and channel count must be non-zero".into(),
            ));
        }
        if samples.len() % usize::from(channels) != 0 {
            return Err(VideoError::InvalidData(format!(
                "PCM sample count {} is not divisible by {channels} channels",
                samples.len()
            )));
        }
        Ok(Self {
            sample_rate,
            channels,
            start,
            samples,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioEncoderSettings {
    pub bit_rate: u32,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone)]
pub struct EncodedAudio {
    pub configuration: DecoderConfiguration,
    pub packets: Vec<EncodedPacket>,
    pub timing: AudioTiming,
}

#[async_trait(?Send)]
pub trait AudioDecodeSession: Debug {
    /// Reset codec state, decode sufficient pre-roll, then discard samples so
    /// the next output starts at exactly `target` on the audible timeline.
    async fn seek(&mut self, target: MediaTime) -> Result<()>;

    async fn decode_next(&mut self) -> Result<Option<PcmAudioFrame>>;
}

#[async_trait(?Send)]
pub trait AudioEncodeSession: Debug {
    async fn push_pcm(&mut self, frame: PcmAudioFrame) -> Result<()>;
    async fn finalize(self: Box<Self>) -> Result<EncodedAudio>;
}

#[async_trait(?Send)]
pub trait AudioIo: Debug + Send + Sync {
    async fn probe(&self, source: &Path) -> Result<MediaInfo>;

    async fn open_audio_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn AudioDecodeSession>>;

    async fn open_audio_encoder(
        &self,
        settings: AudioEncoderSettings,
    ) -> Result<Box<dyn AudioEncodeSession>>;
}

pub type AudioBackend = Arc<dyn AudioIo>;

#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableAudioIo;

#[async_trait(?Send)]
impl AudioIo for UnavailableAudioIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        crate::demux::probe(source)
    }

    async fn open_audio_decoder(
        &self,
        _source: &Path,
        _stream_index: u32,
    ) -> Result<Box<dyn AudioDecodeSession>> {
        Err(unavailable())
    }

    async fn open_audio_encoder(
        &self,
        _settings: AudioEncoderSettings,
    ) -> Result<Box<dyn AudioEncodeSession>> {
        Err(unavailable())
    }
}

fn unavailable() -> VideoError {
    VideoError::BackendUnavailable("no platform compressed-audio backend configured".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeBase;

    #[test]
    fn validates_interleaved_pcm_shape() {
        assert!(PcmAudioFrame::new(48_000, 2, MediaTime::ZERO, vec![0.0, 1.0, 2.0]).is_err());
        let frame = PcmAudioFrame::new(
            48_000,
            2,
            MediaTime::new(0, TimeBase::new_unchecked(1, 48_000)),
            vec![0.0; 8],
        )
        .unwrap();
        assert_eq!(frame.frame_count(), 4);
    }

    #[test]
    fn unavailable_backend_still_probes_and_rejects_codec_sessions() {
        futures_executor::block_on(async {
            let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/generated/aac-lc-48000-stereo.m4a");
            let backend = UnavailableAudioIo;
            assert_eq!(
                backend
                    .probe(&source)
                    .await
                    .unwrap()
                    .audio_streams()
                    .count(),
                1
            );

            let decoder_error = backend.open_audio_decoder(&source, 0).await.unwrap_err();
            assert!(matches!(decoder_error, VideoError::BackendUnavailable(_)));

            let encoder_error = backend
                .open_audio_encoder(AudioEncoderSettings {
                    bit_rate: 128_000,
                    sample_rate: 48_000,
                    channels: 2,
                })
                .await
                .unwrap_err();
            assert!(matches!(encoder_error, VideoError::BackendUnavailable(_)));
        });
    }
}
