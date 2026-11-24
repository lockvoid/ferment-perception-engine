//! Sequential platform video decode for processor path-based capabilities.

use std::path::Path;
use std::sync::Arc;

use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_media::{DecodeOptions, DecodeSession, MediaError, VideoBackend};

#[derive(Debug, Clone)]
pub(crate) struct RgbFrame {
    pub width: u32,
    pub height: u32,
    pub presentation_time: f64,
    pub data: Vec<u8>,
}

/// The platform decoder this build talks to — VideoToolbox on macOS, libav on
/// Windows, MediaCodec on Android. Public so the façade can hand it to
/// [`crate::EdgeTam`], which is the one processor whose SESSION owns a decoder
/// rather than decoding inside one call.
#[must_use]
pub fn video_backend() -> VideoBackend {
    VideoSource::platform_default().backend
}

#[derive(Debug, Clone)]
pub(crate) struct VideoSource {
    backend: VideoBackend,
}

impl VideoSource {
    pub fn new(backend: VideoBackend) -> Self {
        Self { backend }
    }

    pub fn platform_default() -> Self {
        #[cfg(target_os = "macos")]
        let backend: VideoBackend =
            Arc::new(ferment_engine_media::platform::macos::VideoToolboxVideoIo);
        #[cfg(target_os = "windows")]
        let backend: VideoBackend = Arc::new(ferment_engine_media::platform::libav::LibavVideoIo);
        #[cfg(target_os = "android")]
        let backend: VideoBackend =
            Arc::new(ferment_engine_media::platform::android::AndroidVideoIo);
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "android")))]
        let backend: VideoBackend = Arc::new(ferment_engine_media::UnavailableVideoIo);
        Self::new(backend)
    }

    /// One exclusively owned decoder. Neither dropping the consumer nor cancelling
    /// its sink races another task inside the platform decoder.
    #[cfg(test)]
    pub async fn open_rgb(&self, source: &Path, progress: &ProgressSink) -> Result<RgbStream> {
        self.open_rgb_with_policy(source, progress, false, false)
            .await
    }

    /// Scene analysis follows the current iOS reader: positive declared cadence,
    /// a 256-wide even picture when the source is wider, otherwise source size.
    /// Other processors keep their original full-resolution/fallback behavior.
    pub async fn open_scene_rgb(
        &self,
        source: &Path,
        progress: &ProgressSink,
    ) -> Result<RgbStream> {
        self.open_rgb_with_policy(source, progress, true, false)
            .await
    }

    /// Minimap gets explicitly coded pictures plus the container matrix. Its
    /// selected-thumbnail sampler applies that matrix once at bounded size,
    /// including when Android surface decode would normally auto-transform.
    pub async fn open_minimap_rgb(
        &self,
        source: &Path,
        progress: &ProgressSink,
    ) -> Result<RgbStream> {
        self.open_rgb_with_policy(source, progress, false, true)
            .await
    }

    async fn open_rgb_with_policy(
        &self,
        source: &Path,
        progress: &ProgressSink,
        scene_analysis: bool,
        display_oriented: bool,
    ) -> Result<RgbStream> {
        check_cancellation(progress)?;
        let info = self.backend.probe(source).await.map_err(map_media_error)?;
        check_cancellation(progress)?;
        let stream = info
            .video_streams()
            .next()
            .cloned()
            .ok_or_else(|| EngineError::Decode("source has no video stream".into()))?;
        let declared_rate = stream
            .frame_rate
            .filter(|(numerator, denominator)| *numerator > 0 && *denominator > 0)
            .map(|(numerator, denominator)| f64::from(numerator) / f64::from(denominator));
        if scene_analysis && declared_rate.is_none() {
            return Err(EngineError::invalid_config(
                "Video has zero/unknown frame rate",
            ));
        }
        let output_size = if scene_analysis {
            stream
                .width
                .zip(stream.height)
                .and_then(|(width, height)| scene_analysis_size(width, height))
        } else {
            None
        };
        let expected_frame_count = stream
            .duration
            .or(info.duration)
            .map(|duration| duration.seconds() * declared_rate.unwrap_or(30.0))
            .filter(|count| count.is_finite() && *count > 0.0)
            .map(|count| (count.round() as usize).max(1));
        let display_matrix = display_oriented.then_some(stream.display_matrix).flatten();
        let decoder = if display_oriented {
            self.backend.open_coded_decoder(source, stream.index).await
        } else if scene_analysis {
            // Swift's AVAssetReaderTrackOutput scales naturalSize's coded
            // raster. Suppress surface transforms before the analysis resize.
            self.backend
                .open_coded_decoder_with(
                    source,
                    stream.index,
                    DecodeOptions {
                        output_size,
                        acceleration: ferment_engine_media::DecodeAcceleration::Software,
                        output: ferment_engine_media::DecodeOutput::Native,
                    },
                )
                .await
        } else {
            self.backend.open_decoder(source, stream.index).await
        }
        .map_err(map_media_error)?;
        check_cancellation(progress)?;
        Ok(RgbStream {
            decoder: Some(decoder),
            scene_analysis,
            display_matrix,
            scene_output_size: output_size,
            declared_rate,
            expected_frame_count,
            first_time: None,
            last_time: None,
            frame_count: 0,
        })
    }
}

impl Default for VideoSource {
    fn default() -> Self {
        Self::platform_default()
    }
}

fn map_media_error(error: MediaError) -> EngineError {
    match error {
        MediaError::BackendUnavailable(message) => EngineError::Unsupported(message),
        other => EngineError::Decode(other.to_string()),
    }
}

/// Sequential RGB frames with constant decoder ownership and no retained pictures.
/// Only timestamps and a count survive a call to `next`.
#[derive(Debug)]
pub(crate) struct RgbStream {
    decoder: Option<Box<dyn DecodeSession>>,
    scene_analysis: bool,
    display_matrix: Option<[i32; 9]>,
    scene_output_size: Option<(u32, u32)>,
    declared_rate: Option<f64>,
    expected_frame_count: Option<usize>,
    first_time: Option<f64>,
    last_time: Option<f64>,
    frame_count: usize,
}

impl RgbStream {
    /// Only the explicit minimap coded-reader policy carries a display matrix.
    /// Pixels remain coded here; selected thumbnails apply it once at their
    /// bounded output size instead of allocating a transformed movie frame.
    pub fn display_matrix(&self) -> Option<[i32; 9]> {
        self.display_matrix
    }
    pub fn frame_rate(&self) -> f64 {
        let measured = self
            .first_time
            .zip(self.last_time)
            .filter(|(first, last)| self.frame_count > 1 && last > first)
            .map(|(first, last)| (self.frame_count - 1) as f64 / (last - first));
        self.declared_rate.or(measured).unwrap_or(30.0)
    }

    pub fn expected_frame_count(&self) -> Option<usize> {
        self.expected_frame_count
    }

    pub async fn next(&mut self, progress: &ProgressSink) -> Result<Option<RgbFrame>> {
        let result = self.next_live(progress).await;
        if result.is_err() || matches!(&result, Ok(None)) {
            // Success at EOF and every error/cancellation release the native reader
            // before returning. No second task can still be using it.
            self.decoder = None;
        }
        result
    }

    async fn next_live(&mut self, progress: &ProgressSink) -> Result<Option<RgbFrame>> {
        check_cancellation(progress)?;
        let Some(decoder) = self.decoder.as_mut() else {
            return Ok(None);
        };
        let frame = decoder.decode_next().await.map_err(map_media_error)?;
        check_cancellation(progress)?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        let presentation_time = frame.pts.seconds();
        // A backend may decline the decoder-size request. Enforce the size
        // before any RGB frame can escape to a retained snapshot or model tap.
        let cpu = if self.scene_analysis {
            match self
                .scene_output_size
                .or_else(|| scene_analysis_size(frame.width, frame.height))
            {
                Some((width, height)) => frame.scaled_cpu_rgb(width, height),
                None => frame.to_cpu_rgb(),
            }
        } else {
            frame.to_cpu_rgb()
        }
        .map_err(map_media_error)?;
        check_cancellation(progress)?;
        let result = RgbFrame {
            width: cpu.width,
            height: cpu.height,
            presentation_time,
            data: cpu.data.as_ref().to_vec(),
        };
        check_cancellation(progress)?;
        self.first_time.get_or_insert(presentation_time);
        self.last_time = Some(presentation_time);
        self.frame_count += 1;
        Ok(Some(result))
    }
}

/// Swift `analysisSize(natural:)`: do not upscale small sources; for larger
/// pictures round first and then clear the low bit for 4:2:0-compatible dimensions.
pub(crate) fn scene_analysis_size(width: u32, height: u32) -> Option<(u32, u32)> {
    const ANALYSIS_WIDTH: u32 = 256;
    if width <= ANALYSIS_WIDTH {
        return None;
    }
    let scale = f64::from(ANALYSIS_WIDTH) / f64::from(width);
    let scaled_width = ((f64::from(width) * scale).round() as u32 & !1).max(2);
    let scaled_height = ((f64::from(height) * scale).round() as u32 & !1).max(2);
    Some((scaled_width, scaled_height))
}

pub(crate) fn check_cancellation(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}
