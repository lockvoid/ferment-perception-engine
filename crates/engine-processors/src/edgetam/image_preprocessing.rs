//! Exact still-image preprocessing for the pinned `SAM2ImagePredictor`.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMImagePreprocessing.swift`.
//!
//! The strict parity boundary is an orientation-corrected, opaque, tightly
//! packed sRGB RGB8 image. From that boundary, torchvision 0.22 performs a
//! planar `f32` conversion, an antialiased bilinear resize, ImageNet
//! normalization in `f32`, and one final `f16` rounding. Video decoding
//! intentionally keeps its separate RGB8 resize path.

use half::f16;
use thiserror::Error;

use super::preprocessing::{EdgeTamRgbFrame, IMAGE_MEAN, IMAGE_STANDARD_DEVIATION};
use super::processing::MODEL_IMAGE_SIZE;

pub const IMAGE_SIZE: usize = MODEL_IMAGE_SIZE;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ImagePreprocessingError {
    #[error("Invalid image preprocessing destination {width}x{height}.")]
    InvalidDestinationDimensions { width: usize, height: usize },
    #[error("Image preprocessing destination {width}x{height} is too large.")]
    DestinationSizeOverflow { width: usize, height: usize },
}

impl From<ImagePreprocessingError> for ferment_engine_core::EngineError {
    fn from(error: ImagePreprocessingError) -> Self {
        Self::Inference(error.to_string())
    }
}

/// Produce the `[1, 3, 1024, 1024]` `f16` image-encoder input.
pub fn normalized_image(source: &EdgeTamRgbFrame) -> Result<Vec<f16>, ImagePreprocessingError> {
    image_predictor_normalized_fp16(source, IMAGE_SIZE, IMAGE_SIZE)
}

/// The exact torchvision resize stage, before normalization.
///
/// The result is planar CHW `f32` in `[0, 1]`.
pub fn image_predictor_resize_unit_rgb(
    source: &EdgeTamRgbFrame,
    destination_width: usize,
    destination_height: usize,
) -> Result<Vec<f32>, ImagePreprocessingError> {
    let pixel_count = validated_destination_pixel_count(destination_width, destination_height)?;
    let mut result = vec![0.0_f32; pixel_count * 3];
    for_each_resized_unit_channel(
        source,
        destination_width,
        destination_height,
        |channel, resized| {
            result[channel * pixel_count..(channel + 1) * pixel_count].copy_from_slice(resized);
        },
    )?;
    Ok(result)
}

/// The complete image-predictor preprocessing arithmetic.
///
/// The result is planar CHW and has already undergone the final `f16` cast.
pub fn image_predictor_normalized_fp16(
    source: &EdgeTamRgbFrame,
    destination_width: usize,
    destination_height: usize,
) -> Result<Vec<f16>, ImagePreprocessingError> {
    let resized = image_predictor_resize_unit_rgb(source, destination_width, destination_height)?;
    let pixel_count = validated_destination_pixel_count(destination_width, destination_height)?;
    let mut result = vec![f16::ZERO; resized.len()];
    for channel in 0..3 {
        let channel_offset = channel * pixel_count;
        for pixel in 0..pixel_count {
            result[channel_offset + pixel] =
                normalized_fp16(resized[channel_offset + pixel], channel);
        }
    }
    Ok(result)
}

/// Keep these as two distinct `f32` operations. The pinned PyTorch graph
/// subtracts, then divides; a precomputed affine transform can change the final
/// half-precision bit pattern.
#[inline(always)]
fn normalized_fp16(unit_value: f32, channel: usize) -> f16 {
    let shifted = unit_value - IMAGE_MEAN[channel];
    let normalized = shifted / IMAGE_STANDARD_DEVIATION[channel];
    f16::from_f32(normalized)
}

/// One destination sample's taps into the source axis.
///
/// Shared with [`super::preprocessing`]'s square resize, which resamples RGB8
/// with the same table.
pub(super) struct AxisContribution {
    pub(super) first_source_index: usize,
    pub(super) weights: Vec<f32>,
}

/// Match ATen's antialiased linear coefficient construction. Several
/// apparently equivalent formulas round differently in `f32`.
pub(super) fn axis_contributions(
    source_count: usize,
    destination_count: usize,
) -> Vec<AxisContribution> {
    if source_count == destination_count {
        return (0..destination_count)
            .map(|index| AxisContribution {
                first_source_index: index,
                weights: vec![1.0],
            })
            .collect();
    }

    // ATen rounds the source/destination scale to opmath `f32` before
    // promoting that rounded value for center and reciprocal arithmetic.
    let scale = source_count as f32 / destination_count as f32;
    let scale_double = f64::from(scale);
    let support: f32 = if scale >= 1.0 { scale } else { 1.0 };
    let inverse_scale: f32 = if scale >= 1.0 {
        (1.0_f64 / scale_double) as f32
    } else {
        1.0
    };
    let maximum_span = (2.0 * f64::from(support).ceil() + 1.0) as usize;

    (0..destination_count)
        .map(|destination_index| {
            let center = (scale_double * (destination_index as f64 + 0.5)) as f32;
            let lower = (f64::from(center - support) + 0.5).max(0.0) as usize;
            let upper = ((f64::from(center + support) + 0.5) as usize).min(source_count);
            assert!(upper > lower);
            assert!(upper - lower <= maximum_span);

            let mut weights = Vec::with_capacity(upper - lower);
            let mut total = 0.0_f32;
            for source_index in lower..upper {
                let delta = source_index as f32 - center;
                let unit_distance = ((f64::from(delta) + 0.5) * f64::from(inverse_scale)) as f32;
                let absolute_distance = unit_distance.abs();
                let weight: f32 = if absolute_distance < 1.0 {
                    (1.0_f64 - f64::from(absolute_distance)) as f32
                } else {
                    0.0
                };
                weights.push(weight);
                total += weight;
            }
            assert!(total > 0.0);
            for weight in &mut weights {
                *weight /= total;
            }
            AxisContribution {
                first_source_index: lower,
                weights,
            }
        })
        .collect()
}

fn for_each_resized_unit_channel(
    source: &EdgeTamRgbFrame,
    destination_width: usize,
    destination_height: usize,
    mut body: impl FnMut(usize, &[f32]),
) -> Result<(), ImagePreprocessingError> {
    validated_destination_pixel_count(destination_width, destination_height)?;
    let horizontal = axis_contributions(source.width, destination_width);
    let vertical = axis_contributions(source.height, destination_height);

    for channel in 0..3 {
        let horizontally_resized: Vec<f32> = if source.width == destination_width {
            (0..source.width * source.height)
                .map(|pixel| f32::from(source.pixels[pixel * 3 + channel]) / 255.0)
                .collect()
        } else {
            let mut source_row_unit = vec![0.0_f32; source.width];
            let mut intermediate = vec![0.0_f32; destination_width * source.height];
            for row in 0..source.height {
                let source_byte_row = row * source.width * 3;
                for (column, unit) in source_row_unit.iter_mut().enumerate() {
                    *unit =
                        f32::from(source.pixels[source_byte_row + column * 3 + channel]) / 255.0;
                }
                let destination_row = row * destination_width;
                for column in 0..destination_width {
                    intermediate[destination_row + column] =
                        weighted_sum(&source_row_unit, 0, 1, &horizontal[column]);
                }
            }
            intermediate
        };

        let resized: Vec<f32> = if source.height == destination_height {
            horizontally_resized
        } else {
            let mut destination = vec![0.0_f32; destination_width * destination_height];
            for (row, contribution) in vertical.iter().enumerate().take(destination_height) {
                let destination_row = row * destination_width;
                for column in 0..destination_width {
                    destination[destination_row + column] = weighted_sum(
                        &horizontally_resized,
                        column,
                        destination_width,
                        contribution,
                    );
                }
            }
            destination
        };

        body(channel, &resized);
    }
    Ok(())
}

/// ATen accumulates each tap into an opmath `f32` scalar. Explicit FMA is
/// required here; an ordinary `sum += value * weight` leaks through the final
/// `f16` rounding for valid images.
#[inline(always)]
fn weighted_sum(
    values: &[f32],
    base_offset: usize,
    stride: usize,
    contribution: &AxisContribution,
) -> f32 {
    let first_index = base_offset + contribution.first_source_index * stride;
    let mut sum = values[first_index] * contribution.weights[0];
    for (tap, weight) in contribution.weights.iter().copied().enumerate().skip(1) {
        let source_index = first_index + tap * stride;
        sum = values[source_index].mul_add(weight, sum);
    }
    sum
}

fn validated_destination_pixel_count(
    width: usize,
    height: usize,
) -> Result<usize, ImagePreprocessingError> {
    if width == 0 || height == 0 {
        return Err(ImagePreprocessingError::InvalidDestinationDimensions { width, height });
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or(ImagePreprocessingError::DestinationSizeOverflow { width, height })?;
    if pixel_count > usize::MAX / 3 {
        return Err(ImagePreprocessingError::DestinationSizeOverflow { width, height });
    }
    Ok(pixel_count)
}
