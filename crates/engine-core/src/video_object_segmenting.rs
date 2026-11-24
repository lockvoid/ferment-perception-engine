//! Ported from `Protocols/VideoObjectSegmenting.swift`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{EngineService, ProgressSink, Result};

/// A stable caller-owned identity for one object in an interactive
/// segmentation session.
///
/// EdgeTAM assigns compact model indices internally. Keeping that detail out of
/// the public contract lets callers use the same identity before and after a
/// session is recreated.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VideoObjectId(pub String);

impl VideoObjectId {
    pub fn new(raw_value: impl Into<String>) -> Self {
        Self(raw_value.into())
    }

    #[must_use]
    pub fn raw_value(&self) -> &str {
        &self.0
    }
}

/// A point in source coordinates, normalized independently by the source width
/// and height.
///
/// Valid coordinates are finite and in the closed interval `0...1`. Validation
/// happens when a prompt is applied, so constructing UI state never traps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedVideoPoint {
    pub x: f32,
    pub y: f32,
}

impl NormalizedVideoPoint {
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoObjectPointLabel {
    Background = 0,
    Foreground = 1,
}

impl VideoObjectPointLabel {
    #[must_use]
    pub fn raw_value(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoObjectPointPrompt {
    pub point: NormalizedVideoPoint,
    pub label: VideoObjectPointLabel,
}

impl VideoObjectPointPrompt {
    #[must_use]
    pub fn new(point: NormalizedVideoPoint, label: VideoObjectPointLabel) -> Self {
        Self { point, label }
    }
}

/// An axis-aligned source box in normalized coordinates.
///
/// `top_left.x <= bottom_right.x` and `top_left.y <= bottom_right.y` are
/// validated when the prompt is applied.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedVideoBox {
    pub top_left: NormalizedVideoPoint,
    pub bottom_right: NormalizedVideoPoint,
}

impl NormalizedVideoBox {
    #[must_use]
    pub fn new(top_left: NormalizedVideoPoint, bottom_right: NormalizedVideoPoint) -> Self {
        Self {
            top_left,
            bottom_right,
        }
    }

    #[must_use]
    pub fn from_extents(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self::new(
            NormalizedVideoPoint::new(min_x, min_y),
            NormalizedVideoPoint::new(max_x, max_y),
        )
    }
}

/// A prompt accepted by an interactive object segmentation session.
///
/// EdgeTAM encodes a box as two prompt tokens before any correction points.
/// The combined case preserves that exact ordering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoObjectPrompt {
    Points(Vec<VideoObjectPointPrompt>),
    Box(NormalizedVideoBox),
    BoxAndPoints {
        r#box: NormalizedVideoBox,
        points: Vec<VideoObjectPointPrompt>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoObjectPropagationDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoObjectSegmentationSourceInfo {
    pub frame_count: usize,
    pub frame_rate: f64,
    pub width: usize,
    pub height: usize,
}

impl VideoObjectSegmentationSourceInfo {
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.frame_rate > 0.0 {
            self.frame_count as f64 / self.frame_rate
        } else {
            0.0
        }
    }
}

/// Uncompressed COCO mask RLE.
///
/// Counts traverse the image in column-major order, alternate background and
/// foreground runs, and always begin with a background count. Consequently a
/// mask whose first pixel is foreground starts with a zero count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoObjectMaskRle {
    pub width: usize,
    pub height: usize,
    pub counts: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VideoObjectMaskError {
    #[error("Invalid mask dimensions {width}x{height}.")]
    InvalidDimensions { width: usize, height: usize },
    #[error("Mask has {actual} bytes; expected {expected}.")]
    InvalidByteCount { expected: usize, actual: usize },
    #[error("Mask byte {index} is {value}; OneComponent8 binary masks use only 0 or 255.")]
    NonBinaryByte { index: usize, value: u8 },
    #[error("Malformed mask RLE: {0}")]
    MalformedRle(String),
}

/// An immutable, tightly packed binary mask.
///
/// `bytes` is row-major with exactly `width` bytes per row. Background is `0`
/// and foreground is `255`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoObjectMask {
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}

impl VideoObjectMask {
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn new(
        width: usize,
        height: usize,
        bytes: Vec<u8>,
    ) -> std::result::Result<Self, VideoObjectMaskError> {
        let expected = Self::validated_pixel_count(width, height)?;
        if bytes.len() != expected {
            return Err(VideoObjectMaskError::InvalidByteCount {
                expected,
                actual: bytes.len(),
            });
        }
        if let Some((index, value)) = bytes
            .iter()
            .enumerate()
            .find(|(_, byte)| **byte != 0 && **byte != 255)
        {
            return Err(VideoObjectMaskError::NonBinaryByte {
                index,
                value: *value,
            });
        }
        Ok(Self {
            width,
            height,
            bytes,
        })
    }

    /// Decode an uncompressed COCO RLE into tightly packed row-major bytes.
    pub fn from_uncompressed_coco_rle(
        rle: &VideoObjectMaskRle,
    ) -> std::result::Result<Self, VideoObjectMaskError> {
        let pixel_count = Self::validated_pixel_count(rle.width, rle.height)?;
        if rle.counts.is_empty() {
            return Err(VideoObjectMaskError::MalformedRle(
                "counts are empty".into(),
            ));
        }

        let mut total: usize = 0;
        for (index, count) in rle.counts.iter().copied().enumerate() {
            if count < 0 {
                return Err(VideoObjectMaskError::MalformedRle(format!(
                    "count {index} is negative"
                )));
            }
            if index != 0 && count == 0 {
                return Err(VideoObjectMaskError::MalformedRle(
                    "only the initial background count may be zero".into(),
                ));
            }
            let count = usize::try_from(count).map_err(|_| {
                VideoObjectMaskError::MalformedRle(format!(
                    "counts exceed the {pixel_count}-pixel mask"
                ))
            })?;
            let new_total = total.checked_add(count).ok_or_else(|| {
                VideoObjectMaskError::MalformedRle(format!(
                    "counts exceed the {pixel_count}-pixel mask"
                ))
            })?;
            if new_total > pixel_count {
                return Err(VideoObjectMaskError::MalformedRle(format!(
                    "counts exceed the {pixel_count}-pixel mask"
                )));
            }
            total = new_total;
        }
        if total != pixel_count {
            return Err(VideoObjectMaskError::MalformedRle(format!(
                "counts cover {total} pixels; expected {pixel_count}"
            )));
        }

        let mut bytes = vec![0_u8; pixel_count];
        let mut column_major_index = 0_usize;
        let mut foreground = false;
        for count in rle.counts.iter().copied() {
            let count = count as usize;
            if foreground {
                for index in column_major_index..(column_major_index + count) {
                    let x = index / rle.height;
                    let y = index % rle.height;
                    bytes[y * rle.width + x] = 255;
                }
            }
            column_major_index += count;
            foreground = !foreground;
        }

        Ok(Self::from_validated_bytes(rle.width, rle.height, bytes))
    }

    /// Encode this mask as canonical uncompressed COCO RLE.
    #[must_use]
    pub fn uncompressed_coco_rle(&self) -> VideoObjectMaskRle {
        let mut counts: Vec<i64> = Vec::with_capacity(1.max(self.bytes.len() / 8));
        let mut foreground = false;
        let mut run_length: i64 = 0;

        // COCO uses Fortran/column-major traversal even though the mask bytes
        // themselves are row-major.
        for x in 0..self.width {
            for y in 0..self.height {
                let pixel_is_foreground = self.bytes[y * self.width + x] != 0;
                if pixel_is_foreground == foreground {
                    run_length += 1;
                } else {
                    counts.push(run_length);
                    run_length = 1;
                    foreground = !foreground;
                }
            }
        }
        counts.push(run_length);

        VideoObjectMaskRle {
            width: self.width,
            height: self.height,
            counts,
        }
    }

    /// Internal fast path for deterministic processing stages that have
    /// already produced and validated one byte per pixel.
    #[must_use]
    pub fn from_validated_bytes(width: usize, height: usize, bytes: Vec<u8>) -> Self {
        Self {
            width,
            height,
            bytes,
        }
    }

    pub fn validated_pixel_count(
        width: usize,
        height: usize,
    ) -> std::result::Result<usize, VideoObjectMaskError> {
        if width == 0 || height == 0 {
            return Err(VideoObjectMaskError::InvalidDimensions { width, height });
        }
        width
            .checked_mul(height)
            .ok_or(VideoObjectMaskError::InvalidDimensions { width, height })
    }
}

/// One object's mask at one source-video frame.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoObjectMaskFrame {
    pub frame_index: usize,
    pub presentation_time: f64,
    pub object_id: VideoObjectId,
    pub mask: VideoObjectMask,
    pub predicted_iou: Option<f32>,
}

/// Where propagated masks land, one at a time, as they are produced.
///
/// The iOS twin is an `AsyncThrowingStream` with `bufferingOldest(0)`: the
/// producer computes at most one frame ahead and waits for a consumer. A sink
/// that answers `false` for "stop" reproduces that exactly and matches the
/// crate's existing streaming seam ([`crate::ThumbnailSink`]).
pub trait VideoObjectMaskSink: EngineService {
    /// `false` ends the propagation — the twin of the iOS continuation's
    /// `.terminated` result.
    fn emit(&self, frame: VideoObjectMaskFrame) -> bool;
}

/// Mutable inference state is exclusive because prompts, temporal memory,
/// and propagation order must never race.
/// NOT `Send`: the platform decoders this session owns hold raw codec pointers
/// (libav's `AVCodecContext`, VideoToolbox's session), so the session never
/// leaves the thread that created it. A caller that needs it behind a `Send`
/// handle owns it on one thread and marshals calls, which is also what iOS's
/// actor executor does.
#[async_trait(?Send)]
pub trait VideoObjectSegmentationSession {
    fn source_info(&self) -> VideoObjectSegmentationSourceInfo;

    /// Apply an initial prompt or a correction and return the newly segmented
    /// prompt frame. Set `replacing_existing_prompts` to false to append clicks
    /// on that same object and frame. EdgeTAM V1 retains at most 32 distinct
    /// prompted frames per session; an existing frame may be corrected without
    /// consuming another slot, and `reset` restores the complete budget.
    async fn apply(
        &mut self,
        prompt: &VideoObjectPrompt,
        object_id: &VideoObjectId,
        frame_index: usize,
        replacing_existing_prompts: bool,
    ) -> Result<VideoObjectMaskFrame> {
        self.apply_with_cancellation(
            prompt,
            object_id,
            frame_index,
            replacing_existing_prompts,
            ProgressSink::noop(),
        )
        .await
    }

    /// Cancellation must not replace an existing prompt or commit a new one.
    async fn apply_with_cancellation(
        &mut self,
        prompt: &VideoObjectPrompt,
        object_id: &VideoObjectId,
        frame_index: usize,
        replacing_existing_prompts: bool,
        cancellation: ProgressSink,
    ) -> Result<VideoObjectMaskFrame>;

    /// Propagate every registered object from `frame_index` to the end in the
    /// selected direction. The prompt frame itself is included.
    async fn propagate(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        sink: &dyn VideoObjectMaskSink,
    ) -> Result<()> {
        self.propagate_with_cancellation(frame_index, direction, sink, ProgressSink::noop())
            .await
    }

    async fn propagate_with_cancellation(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        sink: &dyn VideoObjectMaskSink,
        cancellation: ProgressSink,
    ) -> Result<()>;

    /// Remove every prompt, object mapping, cached feature, and memory frame.
    async fn reset(&mut self);
}

/// Creates isolated interactive sessions for promptable video segmentation.
#[async_trait(?Send)]
pub trait VideoObjectSegmenting: EngineService {
    async fn make_session(
        &self,
        video_path: &std::path::Path,
    ) -> Result<Box<dyn VideoObjectSegmentationSession>> {
        self.make_session_with_cancellation(video_path, ProgressSink::noop())
            .await
    }

    async fn make_session_with_cancellation(
        &self,
        video_path: &std::path::Path,
        cancellation: ProgressSink,
    ) -> Result<Box<dyn VideoObjectSegmentationSession>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask contract refuses every malformed shape rather than producing a
    /// mask nobody can trust. Ported from `VideoObjectMask`'s own guards in
    /// `Protocols/VideoObjectSegmenting.swift`; the RLE rules are COCO's, and
    /// the doc comment there is the spec: counts are column-major, alternate
    /// background-first, and only the initial background count may be zero.
    ///
    /// kill: drop any one arm of `from_uncompressed_coco_rle`'s validation and
    /// the matching case below decodes into a mask of the wrong length or with
    /// runs that overrun the buffer.
    #[test]
    fn a_malformed_rle_is_refused_rather_than_decoded() {
        let rle = |width, height, counts: &[i64]| VideoObjectMaskRle {
            width,
            height,
            counts: counts.to_vec(),
        };
        // The honest case first, so the refusals below are not vacuous.
        let good = rle(3, 2, &[2, 2, 2]);
        let mask = VideoObjectMask::from_uncompressed_coco_rle(&good).expect("a valid RLE");
        assert_eq!(mask.bytes(), [0, 255, 0, 0, 255, 0]);

        for (name, bad) in [
            ("empty counts", rle(3, 2, &[])),
            ("a negative count", rle(3, 2, &[2, -1, 5])),
            ("a zero after the first", rle(3, 2, &[2, 0, 4])),
            ("counts short of the pixel count", rle(3, 2, &[2, 2])),
            ("counts past the pixel count", rle(3, 2, &[2, 2, 2, 2])),
            ("a zero dimension", rle(0, 2, &[1])),
            ("a count that overflows usize", rle(3, 2, &[i64::MAX, 1])),
        ] {
            assert!(
                VideoObjectMask::from_uncompressed_coco_rle(&bad).is_err(),
                "{name} must be refused"
            );
        }
    }

    /// `new` is the checked constructor; `from_validated_bytes` is the internal
    /// fast path. Only the first may be handed caller bytes.
    ///
    /// kill: drop the `firstIndex(where:)` non-binary scan and a mask carrying
    /// 128 is accepted, which then renders as neither foreground nor background.
    #[test]
    fn a_mask_must_be_binary_and_the_right_length() {
        assert!(VideoObjectMask::new(2, 2, vec![0, 255, 0, 255]).is_ok());
        assert!(
            VideoObjectMask::new(2, 2, vec![0, 128, 0, 255]).is_err(),
            "a non-binary byte must be refused"
        );
        assert!(
            VideoObjectMask::new(2, 2, vec![0, 255, 0]).is_err(),
            "a short buffer must be refused"
        );
        assert!(VideoObjectMask::new(0, 2, vec![]).is_err());
        assert!(
            VideoObjectMask::validated_pixel_count(usize::MAX, 2).is_err(),
            "an overflowing pixel count must be refused, not wrapped"
        );
    }

    /// A mask whose FIRST pixel is foreground starts with a zero count — the
    /// one place COCO's "alternate, background first" rule is visible, and the
    /// one the decoder's "only the initial count may be zero" rule exists for.
    ///
    /// kill: start the encoder with `foreground = true` and this mask encodes
    /// to `[2, 4]`, which decodes to its inverse.
    #[test]
    fn a_foreground_first_mask_opens_with_a_zero_count() {
        // Column-major: column 0 is both-foreground, then four background.
        let mask = VideoObjectMask::new(3, 2, vec![255, 0, 0, 255, 0, 0]).expect("mask");
        let rle = mask.uncompressed_coco_rle();
        assert_eq!(rle.counts, [0, 2, 4]);
        let round = VideoObjectMask::from_uncompressed_coco_rle(&rle).expect("decode");
        assert_eq!(round.bytes(), mask.bytes());
    }
}
