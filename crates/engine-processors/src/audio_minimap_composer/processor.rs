//! Top-level orchestrator. Decode → mel + RMS → render → encode PNG/JPEG.

use ferment_engine_audio::derived_cache::DerivedFileCache;
use ferment_engine_audio::streaming_resampler::resample_current;
use std::sync::Arc;

use std::io::Cursor;

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{EngineError, ProgressSink, ProgressTracker, Result};
use serde::{Deserialize, Serialize};

use super::layout::{Layout, LayoutGeometry};
use super::mel::SAMPLE_RATE;
use super::renderer::{RenderInput, render};
use super::{mel, rms};

/// Anthropic vision API rejects images larger than 5 MB; throw early so
/// the caller doesn't ship a payload that gets rejected on upload.
const MAX_OUTPUT_BYTES: usize = 5 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg { quality: u8 },
}

impl ImageFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpg",
        }
    }
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg { .. } => "image/jpeg",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioMinimapComposerConfig {
    pub max_width: usize,
    pub format: ImageFormat,
}

impl Default for AudioMinimapComposerConfig {
    fn default() -> Self {
        Self {
            max_width: 2048,
            format: ImageFormat::Jpeg { quality: 90 },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioMinimapResult {
    pub image_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub layout: LayoutGeometry,
}

pub struct AudioMinimapComposer {
    pub(crate) cache: Option<Arc<DerivedFileCache>>,
    config: AudioMinimapComposerConfig,
}

impl AudioMinimapComposer {
    pub fn with_cache(mut self, cache: Option<Arc<DerivedFileCache>>) -> Self {
        self.cache = cache;
        self
    }

    pub fn new(config: AudioMinimapComposerConfig) -> Self {
        Self {
            config,
            cache: None,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(AudioMinimapComposerConfig::default())
    }

    pub async fn compose(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<AudioMinimapResult> {
        self.compose_with_bar_ruler(frames, &[], progress).await
    }

    /// `bar_ruler` = downbeat times in track seconds. Non-empty, the image
    /// gains a BAR RULER — a downbeat vertical through mel + RMS — the axis a
    /// vision LLM reads structure against ("the drop is bar 35"). An axis, not
    /// metadata: the numbers themselves ride in `layout.bar_ruler`.
    pub async fn compose_with_bar_ruler(
        &self,
        frames: AudioFrames,
        bar_ruler: &[f64],
        progress: ProgressSink,
    ) -> Result<AudioMinimapResult> {
        let tracker = ProgressTracker::new(progress.clone());
        check_cancelled(&progress)?;
        tracker.advance(0.05, "Loading audio...").await;
        check_cancelled(&progress)?;
        let identity = (frames.sample_rate == SAMPLE_RATE)
            .then_some(frames.identity)
            .flatten();
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_current(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        if audio.is_empty() {
            return Err(EngineError::NoAudioData);
        }

        let duration = audio.len() as f64 / f64::from(SAMPLE_RATE);

        tracker.advance(0.40, "Computing mel spectrogram...").await;
        check_cancelled(&progress)?;
        let frames = mel::MEL_SPEC.perform(&audio, identity.as_ref(), self.cache.as_deref())?;
        let mel_db: Vec<Vec<f32>> = frames
            .values
            .chunks_exact(frames.bands)
            .map(<[f32]>::to_vec)
            .collect();
        if mel_db.is_empty() {
            return Err(EngineError::AudioTooShort {
                samples: audio.len(),
                min_required: mel::FFT_SIZE,
            });
        }

        tracker.advance(0.10, "Computing RMS envelope...").await;
        check_cancelled(&progress)?;
        let rms_db = rms::compute(&audio);

        tracker.advance(0.05, "Resolving layout...").await;
        check_cancelled(&progress)?;
        let layout = Layout::resolve(duration, self.config.max_width);

        tracker.advance(0.30, "Rendering minimap...").await;
        check_cancelled(&progress)?;
        let rendered = render(&RenderInput {
            mel_db: &mel_db,
            rms_db: &rms_db,
            bar_ruler,
            duration_secs: duration,
            layout,
        });

        tracker.advance(0.05, "Encoding...").await;
        check_cancelled(&progress)?;
        let image_data = encode(&rendered.pixmap, self.config.format)?;
        if image_data.len() > MAX_OUTPUT_BYTES {
            return Err(EngineError::invalid_config(format!(
                "audio minimap is {} KB, exceeds {} KB limit",
                image_data.len() / 1024,
                MAX_OUTPUT_BYTES / 1024
            )));
        }

        check_cancelled(&progress)?;
        tracker.advance(0.05, "Done").await;
        Ok(AudioMinimapResult {
            image_data,
            width: layout.total_width as u32,
            height: layout.total_height as u32,
            format: self.config.format,
            layout: layout.geometry(duration, bar_ruler),
        })
    }
}

fn check_cancelled(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn encode(pixmap: &tiny_skia::Pixmap, format: ImageFormat) -> Result<Vec<u8>> {
    let w = pixmap.width();
    let h = pixmap.height();
    // tiny-skia pixels are premultiplied RGBA; un-premultiply for image crate.
    let rgba: Vec<u8> = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let c = p.demultiply();
            [c.red(), c.green(), c.blue(), c.alpha()]
        })
        .collect();
    let img = image::RgbaImage::from_raw(w, h, rgba)
        .ok_or_else(|| EngineError::internal("rgba buffer size mismatch"))?;

    let mut out = Cursor::new(Vec::new());
    match format {
        ImageFormat::Png => img
            .write_to(&mut out, image::ImageFormat::Png)
            .map_err(|e| EngineError::internal(format!("png encode: {e}")))?,
        ImageFormat::Jpeg { quality } => {
            // JPEG doesn't support alpha — drop it.
            let rgb = image::DynamicImage::ImageRgba8(img).into_rgb8();
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.min(100));
            encoder
                .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
                .map_err(|e| EngineError::internal(format!("jpeg encode: {e}")))?;
        }
    }
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Current Swift advances [5,40,10,5,30,5,5] percent, cumulatively.
    /// Kill: pass each stage's weight to the absolute callback instead.
    #[test]
    fn actual_composition_reports_the_current_cumulative_stages() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&events);
        let progress = ProgressSink::from_callback(move |event| seen.lock().unwrap().push(event));
        let audio = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let result = futures_executor::block_on(
            AudioMinimapComposer::with_defaults()
                .compose(AudioFrames::new(audio, SAMPLE_RATE), progress),
        )
        .unwrap();
        assert!(!result.image_data.is_empty());
        let actual = events.lock().unwrap();
        let expected = [
            (0.05, "Loading audio..."),
            (0.45, "Computing mel spectrogram..."),
            (0.55, "Computing RMS envelope..."),
            (0.60, "Resolving layout..."),
            (0.90, "Rendering minimap..."),
            (0.95, "Encoding..."),
            (1.0, "Done"),
        ];
        assert_eq!(actual.len(), expected.len());
        for (event, (fraction, stage)) in actual.iter().zip(expected) {
            assert!(
                (event.progress - fraction).abs() < 1e-12,
                "{}: {}",
                event.stage,
                event.progress
            );
            assert_eq!(event.stage, stage);
        }
    }

    /// A closed native call returns cancellation, not a completed image.
    /// Kill: omit admission cancellation and render the full supplied input.
    #[test]
    fn a_cancelled_composition_never_returns_an_image() {
        let progress = ProgressSink::noop();
        progress.cancel();
        let error = futures_executor::block_on(AudioMinimapComposer::with_defaults().compose(
            AudioFrames::new(vec![0.0; SAMPLE_RATE as usize], SAMPLE_RATE),
            progress,
        ))
        .unwrap_err();
        assert!(matches!(error, EngineError::Cancelled));
    }
}
