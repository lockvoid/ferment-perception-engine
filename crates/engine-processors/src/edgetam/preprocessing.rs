//! Deterministic decoded-video RGB preprocessing for EdgeTAM.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMPreprocessing.swift`.
//!
//! Stage 1 stretches the complete oriented source frame to 1024x1024. This is
//! deliberately not an aspect-fit operation: upstream EdgeTAM video loading
//! requests a square decode and does not letterbox.
//!
//! Stage 2 converts RGB8 to planar NCHW `f16` and applies ImageNet
//! normalization in `f32` before the final `f16` rounding:
//!
//! ```text
//! output = f16((f32(pixel) / 255 - mean) / std)
//! ```
//!
//! The split stage API lets video parity fixtures compare the square RGB bytes
//! and normalized FP16 bit patterns independently. Still images use
//! [`super::image_preprocessing`], which matches torchvision's `f32`
//! antialiased resize without an intermediate RGB8 quantization.

use ferment_engine_media::{MediaTime, TimeBase};
use half::f16;
use thiserror::Error;

use super::processing::MODEL_IMAGE_SIZE;

pub const IMAGE_SIZE: usize = MODEL_IMAGE_SIZE;
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];

/// One decoded source frame at EdgeTAM's canonical boundary: tightly packed
/// sRGB RGB8, row-major, `3 * width * height` bytes.
///
/// The Swift twin is `EdgeTAMRGBFrame`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeTamRgbFrame {
    pub frame_index: usize,
    pub presentation_timestamp: MediaTime,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl EdgeTamRgbFrame {
    pub fn new(
        frame_index: usize,
        presentation_timestamp: MediaTime,
        width: usize,
        height: usize,
        pixels: Vec<u8>,
    ) -> Result<Self, PreprocessingError> {
        let frame = Self {
            frame_index,
            presentation_timestamp,
            width,
            height,
            pixels,
        };
        validate(&frame)?;
        Ok(frame)
    }

    /// Convert only at the public delivery boundary; retain the decoder's exact
    /// rational clock while frames are indexed, cached and tracked.
    #[must_use]
    pub fn presentation_time(&self) -> f64 {
        self.presentation_timestamp.seconds()
    }

    /// Diagnostic projection only. It is never used to identify a picture.
    #[must_use]
    pub fn presentation_time_micros(&self) -> i64 {
        self.presentation_timestamp
            .rescale(TimeBase::MICROSECONDS)
            .value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PreprocessingError {
    #[error("Invalid RGB frame dimensions {width}x{height}.")]
    InvalidDimensions { width: usize, height: usize },
    #[error("RGB frame has {actual} bytes; expected {expected}.")]
    InvalidRgbByteCount { expected: usize, actual: usize },
    #[error(
        "Video preprocessing workspace is {expected_width}x{expected_height}, but received \
         {actual_width}x{actual_height}."
    )]
    WorkspaceDimensionsChanged {
        expected_width: usize,
        expected_height: usize,
        actual_width: usize,
        actual_height: usize,
    },
}

impl From<PreprocessingError> for ferment_engine_core::EngineError {
    fn from(error: PreprocessingError) -> Self {
        Self::Inference(error.to_string())
    }
}

/// Reusable video-only buffers. A session has fixed display dimensions, so
/// retaining these allocations removes two large heap buffers from every
/// decoded frame.
#[derive(Debug)]
pub struct Workspace {
    source_width: usize,
    source_height: usize,
    square_rgb: Vec<u8>,
    resampler: Option<crate::rgb_hq::Workspace>,
}

impl Workspace {
    pub fn new(source_width: usize, source_height: usize) -> Result<Self, PreprocessingError> {
        if source_width == 0 || source_height == 0 {
            return Err(PreprocessingError::InvalidDimensions {
                width: source_width,
                height: source_height,
            });
        }
        source_width
            .checked_mul(source_height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(PreprocessingError::InvalidDimensions {
                width: source_width,
                height: source_height,
            })?;
        source_width
            .checked_mul(IMAGE_SIZE)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(PreprocessingError::InvalidDimensions {
                width: source_width,
                height: source_height,
            })?;
        let square_rgb = if source_width == IMAGE_SIZE && source_height == IMAGE_SIZE {
            Vec::new()
        } else {
            vec![0_u8; IMAGE_SIZE * IMAGE_SIZE * 3]
        };
        Ok(Self {
            source_width,
            source_height,
            square_rgb,
            resampler: if source_width == IMAGE_SIZE && source_height == IMAGE_SIZE {
                None
            } else {
                Some(crate::rgb_hq::Workspace::new(
                    source_width,
                    source_height,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ))
            },
        })
    }

    pub fn normalized_image(
        &mut self,
        source: &EdgeTamRgbFrame,
    ) -> Result<Vec<f16>, PreprocessingError> {
        let square = self.square_model_frame(source)?;
        normalized_image_from_square_model_frame(square.width, square.height, square.pixels)
    }

    /// Stretch the full source image to EdgeTAM's fixed square input.
    ///
    /// No crop and no letterbox are applied. A 1024x1024 source takes an
    /// identity fast path, preserving every RGB byte exactly.
    pub fn square_model_frame<'frame>(
        &'frame mut self,
        source: &'frame EdgeTamRgbFrame,
    ) -> Result<SquareFrame<'frame>, PreprocessingError> {
        validate(source)?;
        if source.width != self.source_width || source.height != self.source_height {
            return Err(PreprocessingError::WorkspaceDimensionsChanged {
                expected_width: self.source_width,
                expected_height: self.source_height,
                actual_width: source.width,
                actual_height: source.height,
            });
        }
        if self.source_width == IMAGE_SIZE && self.source_height == IMAGE_SIZE {
            return Ok(SquareFrame {
                width: IMAGE_SIZE,
                height: IMAGE_SIZE,
                pixels: &source.pixels,
            });
        }

        self.resampler
            .as_mut()
            .expect("nonidentity dimensions own HQ workspace")
            .resize_into(&source.pixels, &mut self.square_rgb);
        Ok(SquareFrame {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            pixels: &self.square_rgb,
        })
    }

    #[must_use]
    pub fn retained_byte_count(&self) -> usize {
        self.square_rgb.capacity()
            + self
                .resampler
                .as_ref()
                .map_or(0, crate::rgb_hq::Workspace::retained_byte_count)
    }
}

/// A borrowed square RGB8 stage output — the workspace's buffer, or the source
/// frame itself on the 1024x1024 identity path.
#[derive(Debug, Clone, Copy)]
pub struct SquareFrame<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a [u8],
}

/// Stretch the full source image to EdgeTAM's fixed square input, allocating
/// its own workspace. The session path reuses one [`Workspace`] instead.
pub fn square_model_frame(source: &EdgeTamRgbFrame) -> Result<Vec<u8>, PreprocessingError> {
    let mut workspace = Workspace::new(source.width, source.height)?;
    Ok(workspace.square_model_frame(source)?.pixels.to_vec())
}

/// Produce the `[1, 3, 1024, 1024]` `f16` image input consumed by the image
/// encoder.
pub fn normalized_image(source: &EdgeTamRgbFrame) -> Result<Vec<f16>, PreprocessingError> {
    let mut workspace = Workspace::new(source.width, source.height)?;
    workspace.normalized_image(source)
}

/// Normalize an already-square stage output without another render.
pub fn normalized_image_from_square_model_frame(
    width: usize,
    height: usize,
    pixels: &[u8],
) -> Result<Vec<f16>, PreprocessingError> {
    if width != IMAGE_SIZE || height != IMAGE_SIZE {
        return Err(PreprocessingError::InvalidDimensions { width, height });
    }
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(PreprocessingError::InvalidRgbByteCount {
            expected,
            actual: pixels.len(),
        });
    }

    let lookup = normalized_lookup();
    let pixels_per_channel = IMAGE_SIZE * IMAGE_SIZE;
    let mut output = vec![f16::ZERO; pixels_per_channel * 3];
    for pixel in 0..pixels_per_channel {
        let source_offset = pixel * 3;
        output[pixel] = lookup[0][usize::from(pixels[source_offset])];
        output[pixels_per_channel + pixel] = lookup[1][usize::from(pixels[source_offset + 1])];
        output[pixels_per_channel * 2 + pixel] = lookup[2][usize::from(pixels[source_offset + 2])];
    }
    Ok(output)
}

/// 256 normalized `f16` values per channel — the whole domain of an RGB8
/// input, so the per-pixel path is a table read rather than three flops.
fn normalized_lookup() -> &'static [[f16; 256]; 3] {
    use std::sync::OnceLock;
    static LOOKUP: OnceLock<[[f16; 256]; 3]> = OnceLock::new();
    LOOKUP.get_or_init(|| {
        let mut table = [[f16::ZERO; 256]; 3];
        for (channel, row) in table.iter_mut().enumerate() {
            for (byte, value) in row.iter_mut().enumerate() {
                let unit = byte as f32 / 255.0;
                let normalized = (unit - IMAGE_MEAN[channel]) / IMAGE_STANDARD_DEVIATION[channel];
                *value = f16::from_f32(normalized);
            }
        }
        table
    })
}

fn validate(frame: &EdgeTamRgbFrame) -> Result<(), PreprocessingError> {
    if frame.width == 0 || frame.height == 0 {
        return Err(PreprocessingError::InvalidDimensions {
            width: frame.width,
            height: frame.height,
        });
    }
    let expected = frame.width * frame.height * 3;
    if frame.pixels.len() != expected {
        return Err(PreprocessingError::InvalidRgbByteCount {
            expected,
            actual: frame.pixels.len(),
        });
    }
    Ok(())
}
