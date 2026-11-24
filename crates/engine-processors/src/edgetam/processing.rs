//! Pure deterministic stages shared by the EdgeTAM runtime and parity tests.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMProcessing.swift`. No inference and
//! no platform API belongs here. Keeping prompt mapping and mask
//! postprocessing scalar and explicit gives the Python fixture generator stable
//! stage boundaries to compare.

use ferment_engine_core::{
    NormalizedVideoBox, NormalizedVideoPoint, VideoObjectMask, VideoObjectPrompt,
};
use thiserror::Error;

pub const MODEL_IMAGE_SIZE: usize = 1024;
pub const LOW_RESOLUTION_MASK_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptTensor {
    /// Flattened `(x, y)` model coordinates.
    pub coordinates: Vec<f32>,
    /// EdgeTAM labels: background=0, foreground=1, box corners=2 and 3.
    pub labels: Vec<i32>,
}

impl PromptTensor {
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.labels.len()
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ProcessingError {
    #[error("EdgeTAM point prompt is empty.")]
    EmptyPointPrompt,
    #[error("Invalid normalized point ({}, {}).", .0.x, .0.y)]
    InvalidNormalizedPoint(NormalizedVideoPoint),
    #[error(
        "Invalid normalized box ({}, {})-({}, {}).",
        .0.top_left.x, .0.top_left.y, .0.bottom_right.x, .0.bottom_right.y
    )]
    InvalidBox(NormalizedVideoBox),
    #[error("Invalid dimensions {width}x{height}.")]
    InvalidDimensions { width: usize, height: usize },
    #[error("Element count is {actual}; expected {expected}.")]
    InvalidElementCount { expected: usize, actual: usize },
}

impl From<ProcessingError> for ferment_engine_core::EngineError {
    fn from(error: ProcessingError) -> Self {
        Self::Inference(error.to_string())
    }
}

#[derive(Debug, Clone, Copy)]
struct AxisSample {
    lower_index: usize,
    upper_index: usize,
    lower_weight: f32,
    upper_weight: f32,
}

/// Immutable interpolation coordinates for one fixed input/output size.
/// Video and image sessions retain the display plan instead of rebuilding
/// the same x/y arrays for every emitted frame.
#[derive(Debug, Clone)]
pub struct BilinearResizePlan {
    source_width: usize,
    source_height: usize,
    destination_width: usize,
    destination_height: usize,
    source_count: usize,
    destination_count: usize,
    x_samples: Vec<AxisSample>,
    y_samples: Vec<AxisSample>,
}

impl BilinearResizePlan {
    pub fn new(
        source_width: usize,
        source_height: usize,
        destination_width: usize,
        destination_height: usize,
    ) -> Result<Self, ProcessingError> {
        let source_count = element_count(source_width, source_height)?;
        let destination_count = element_count(destination_width, destination_height)?;
        Ok(Self {
            source_width,
            source_height,
            destination_width,
            destination_height,
            source_count,
            destination_count,
            x_samples: axis_samples(source_width, destination_width),
            y_samples: axis_samples(source_height, destination_height),
        })
    }

    #[must_use]
    pub fn retained_byte_count(&self) -> usize {
        (self.x_samples.len() + self.y_samples.len()) * std::mem::size_of::<AxisSample>()
    }
}

// MARK: - Prompt preprocessing

/// Convert a public normalized prompt to the exact token order expected by
/// `SAM2VideoPredictor.add_new_points_or_box`.
pub fn prompt_tensor(prompt: &VideoObjectPrompt) -> Result<PromptTensor, ProcessingError> {
    match prompt {
        VideoObjectPrompt::Points(points) => {
            if points.is_empty() {
                return Err(ProcessingError::EmptyPointPrompt);
            }
            point_tensor(points)
        }
        VideoObjectPrompt::Box(bounds) => box_tensor(*bounds, &[]),
        VideoObjectPrompt::BoxAndPoints { r#box, points } => box_tensor(*r#box, points),
    }
}

pub fn model_point(normalized_point: NormalizedVideoPoint) -> Result<ModelPoint, ProcessingError> {
    if !is_valid(normalized_point) {
        return Err(ProcessingError::InvalidNormalizedPoint(normalized_point));
    }
    let size = MODEL_IMAGE_SIZE as f32;
    Ok(ModelPoint {
        x: normalized_point.x * size,
        y: normalized_point.y * size,
    })
}

fn point_tensor(
    points: &[ferment_engine_core::VideoObjectPointPrompt],
) -> Result<PromptTensor, ProcessingError> {
    let mut coordinates = Vec::with_capacity(points.len() * 2);
    let mut labels = Vec::with_capacity(points.len());
    for prompt in points {
        let point = model_point(prompt.point)?;
        coordinates.push(point.x);
        coordinates.push(point.y);
        labels.push(prompt.label.raw_value());
    }
    Ok(PromptTensor {
        coordinates,
        labels,
    })
}

fn box_tensor(
    bounds: NormalizedVideoBox,
    points: &[ferment_engine_core::VideoObjectPointPrompt],
) -> Result<PromptTensor, ProcessingError> {
    // A ported Swift `guard` is the negation of the WHOLE predicate: with a
    // NaN corner every comparison below is false, so the guard bails and this
    // `if` must fall through to the error, not past it.
    if !(is_valid(bounds.top_left)
        && is_valid(bounds.bottom_right)
        && bounds.top_left.x <= bounds.bottom_right.x
        && bounds.top_left.y <= bounds.bottom_right.y)
    {
        return Err(ProcessingError::InvalidBox(bounds));
    }

    let top_left = model_point(bounds.top_left)?;
    let bottom_right = model_point(bounds.bottom_right)?;
    let mut coordinates = Vec::with_capacity((2 + points.len()) * 2);
    coordinates.extend_from_slice(&[top_left.x, top_left.y, bottom_right.x, bottom_right.y]);
    let mut labels = Vec::with_capacity(2 + points.len());
    labels.extend_from_slice(&[2, 3]);

    for prompt in points {
        let point = model_point(prompt.point)?;
        coordinates.push(point.x);
        coordinates.push(point.y);
        labels.push(prompt.label.raw_value());
    }
    Ok(PromptTensor {
        coordinates,
        labels,
    })
}

fn is_valid(point: NormalizedVideoPoint) -> bool {
    point.x.is_finite()
        && point.y.is_finite()
        && (0.0..=1.0).contains(&point.x)
        && (0.0..=1.0).contains(&point.y)
}

// MARK: - Bilinear mask-logit resizing

/// Stock video-output path: resize selected 256x256 logits directly to
/// source-video resolution using PyTorch `align_corners=False` semantics.
pub fn display_logits(
    logits: &[f32],
    source_width: usize,
    source_height: usize,
) -> Result<Vec<f32>, ProcessingError> {
    resize_bilinear_align_corners_false(
        logits,
        LOW_RESOLUTION_MASK_SIZE,
        LOW_RESOLUTION_MASK_SIZE,
        source_width,
        source_height,
    )
}

/// Stock memory-encoder path: resize selected 256x256 logits to the model's
/// 1024x1024 image size.
pub fn memory_encoder_logits(logits: &[f32]) -> Result<Vec<f32>, ProcessingError> {
    resize_bilinear_align_corners_false(
        logits,
        LOW_RESOLUTION_MASK_SIZE,
        LOW_RESOLUTION_MASK_SIZE,
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
    )
}

/// Explicit two-stage parity probe: 256→1024→source. Useful for testing both
/// interpolation boundaries together, but stock EdgeTAM uses the direct
/// [`display_logits`] path for its emitted video masks.
pub fn two_stage_display_logits(
    logits: &[f32],
    source_width: usize,
    source_height: usize,
) -> Result<Vec<f32>, ProcessingError> {
    let model_resolution = memory_encoder_logits(logits)?;
    resize_bilinear_align_corners_false(
        &model_resolution,
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
        source_width,
        source_height,
    )
}

/// Scalar reference implementation of PyTorch bilinear interpolation with
/// `align_corners=False` and `antialias=False`.
pub fn resize_bilinear_align_corners_false(
    input: &[f32],
    source_width: usize,
    source_height: usize,
    destination_width: usize,
    destination_height: usize,
) -> Result<Vec<f32>, ProcessingError> {
    let plan = BilinearResizePlan::new(
        source_width,
        source_height,
        destination_width,
        destination_height,
    )?;
    let mut output = vec![0.0_f32; plan.destination_count];
    for_each_resized_bilinear_value(input, &plan, |index, value| output[index] = value)?;
    Ok(output)
}

/// Production mask path: preserve the exact interpolation arithmetic but
/// threshold each result immediately. This avoids retaining a source-sized
/// `f32` array solely to turn it into binary bytes.
pub fn thresholded_display_mask(
    logits: &[f32],
    source_width: usize,
    source_height: usize,
) -> Result<VideoObjectMask, ProcessingError> {
    let plan = display_mask_plan(source_width, source_height)?;
    thresholded_display_mask_with_plan(logits, &plan)
}

pub fn display_mask_plan(
    source_width: usize,
    source_height: usize,
) -> Result<BilinearResizePlan, ProcessingError> {
    BilinearResizePlan::new(
        LOW_RESOLUTION_MASK_SIZE,
        LOW_RESOLUTION_MASK_SIZE,
        source_width,
        source_height,
    )
}

pub fn thresholded_display_mask_with_plan(
    logits: &[f32],
    plan: &BilinearResizePlan,
) -> Result<VideoObjectMask, ProcessingError> {
    let mut bytes = vec![0_u8; plan.destination_count];
    for_each_resized_bilinear_value(logits, plan, |index, value| {
        bytes[index] = if value > 0.0 { 255 } else { 0 };
    })?;
    Ok(VideoObjectMask::from_validated_bytes(
        plan.destination_width,
        plan.destination_height,
        bytes,
    ))
}

fn for_each_resized_bilinear_value(
    input: &[f32],
    plan: &BilinearResizePlan,
    mut body: impl FnMut(usize, f32),
) -> Result<(), ProcessingError> {
    if input.len() != plan.source_count {
        return Err(ProcessingError::InvalidElementCount {
            expected: plan.source_count,
            actual: input.len(),
        });
    }
    if plan.source_width == plan.destination_width && plan.source_height == plan.destination_height
    {
        for (index, value) in input.iter().copied().enumerate() {
            body(index, value);
        }
        return Ok(());
    }

    for destination_y in 0..plan.destination_height {
        let y = plan.y_samples[destination_y];
        let top_row = y.lower_index * plan.source_width;
        let bottom_row = y.upper_index * plan.source_width;
        let output_row = destination_y * plan.destination_width;

        for destination_x in 0..plan.destination_width {
            let x = plan.x_samples[destination_x];
            let top = weighted_pair(
                input[top_row + x.lower_index],
                input[top_row + x.upper_index],
                x.lower_weight,
                x.upper_weight,
            );
            let bottom = weighted_pair(
                input[bottom_row + x.lower_index],
                input[bottom_row + x.upper_index],
                x.lower_weight,
                x.upper_weight,
            );
            let value = weighted_pair(top, bottom, y.lower_weight, y.upper_weight);
            body(output_row + destination_x, value);
        }
    }
    Ok(())
}

/// Mirrors ATen's `t0 * w0 + t1 * w1`. The release C++ kernel contracts the
/// addition and the second multiplication on arm64, which `mul_add` is.
#[inline(always)]
fn weighted_pair(lower: f32, upper: f32, lower_weight: f32, upper_weight: f32) -> f32 {
    upper.mul_add(upper_weight, lower * lower_weight)
}

fn axis_samples(source_count: usize, destination_count: usize) -> Vec<AxisSample> {
    let scale = source_count as f32 / destination_count as f32;
    (0..destination_count)
        .map(|destination_index| {
            // `area_pixel_compute_source_index(..., align_corners=false)`.
            let unclamped = (destination_index as f32 + 0.5).mul_add(scale, -0.5);
            let source_position = unclamped.max(0.0);
            let lower = (source_position.floor() as usize).min(source_count - 1);
            let upper = (lower + 1).min(source_count - 1);
            let upper_weight = source_position - lower as f32;
            AxisSample {
                lower_index: lower,
                upper_index: upper,
                lower_weight: 1.0 - upper_weight,
                upper_weight,
            }
        })
        .collect()
}

fn element_count(width: usize, height: usize) -> Result<usize, ProcessingError> {
    if width == 0 || height == 0 {
        return Err(ProcessingError::InvalidDimensions { width, height });
    }
    width
        .checked_mul(height)
        .ok_or(ProcessingError::InvalidDimensions { width, height })
}

// MARK: - Thresholding

/// EdgeTAM emits foreground where the selected raw mask logit is strictly
/// greater than zero. Positive infinity is foreground; zero, negative infinity
/// and NaN are background, matching PyTorch comparison semantics.
pub fn threshold(
    logits: &[f32],
    width: usize,
    height: usize,
) -> Result<VideoObjectMask, ProcessingError> {
    let count = element_count(width, height)?;
    if logits.len() != count {
        return Err(ProcessingError::InvalidElementCount {
            expected: count,
            actual: logits.len(),
        });
    }
    let bytes = logits
        .iter()
        .map(|value| if *value > 0.0 { 255 } else { 0 })
        .collect();
    Ok(VideoObjectMask::from_validated_bytes(width, height, bytes))
}
