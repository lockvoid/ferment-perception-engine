//! Ported from `Protocols/ImageObjectSegmenting.swift`.

use async_trait::async_trait;

use crate::{
    EngineService, NormalizedVideoBox, NormalizedVideoPoint, ProgressSink, Result, RgbFrameRef,
    VideoObjectId, VideoObjectMask, VideoObjectMaskError, VideoObjectMaskRle,
    VideoObjectPointLabel, VideoObjectPointPrompt, VideoObjectPrompt,
};

/// Image aliases intentionally share the exact prompt and mask value
/// representations used by video segmentation. A prompt can therefore move
/// between a still-image editor and a video-frame editor without conversion.
pub type ImageObjectId = VideoObjectId;
pub type NormalizedImagePoint = NormalizedVideoPoint;
pub type ImageObjectPointLabel = VideoObjectPointLabel;
pub type ImageObjectPointPrompt = VideoObjectPointPrompt;
pub type NormalizedImageBox = NormalizedVideoBox;
pub type ImageObjectPrompt = VideoObjectPrompt;
pub type ImageObjectMaskRle = VideoObjectMaskRle;
pub type ImageObjectMaskError = VideoObjectMaskError;
pub type ImageObjectMask = VideoObjectMask;

/// Immutable dimensions of the image bound to an interactive session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageObjectSegmentationSourceInfo {
    pub width: usize,
    pub height: usize,
}

/// One object's most recently decoded still-image mask.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageObjectSegmentationResult {
    pub object_id: ImageObjectId,
    pub mask: ImageObjectMask,
    pub predicted_iou: Option<f32>,
}

/// Mutable prompt state for one fixed image is exclusive so corrections and
/// prior-mask reuse can never race.
/// `Send` for the same reason [`crate::VideoObjectSegmentationSession`] is.
#[async_trait(?Send)]
pub trait ImageObjectSegmentationSession: Send {
    fn source_info(&self) -> ImageObjectSegmentationSourceInfo;

    /// Maximum number of object IDs whose correction masks may be retained at
    /// once. Callers can remove finished objects to release their state.
    fn maximum_active_object_count(&self) -> usize;

    /// Apply an initial prompt or correction for `object_id`.
    ///
    /// Set `replacing_existing_prompts` to false to append point clicks to the
    /// object's existing prompt. A box can only start or replace a prompt.
    async fn apply(
        &mut self,
        prompt: &ImageObjectPrompt,
        object_id: &ImageObjectId,
        replacing_existing_prompts: bool,
    ) -> Result<ImageObjectSegmentationResult> {
        self.apply_with_cancellation(
            prompt,
            object_id,
            replacing_existing_prompts,
            ProgressSink::noop(),
        )
        .await
    }

    /// A cancelled decode must not commit a new object or replace its previous
    /// prompt/mask. The sink is checked before inference and before commit.
    async fn apply_with_cancellation(
        &mut self,
        prompt: &ImageObjectPrompt,
        object_id: &ImageObjectId,
        replacing_existing_prompts: bool,
        cancellation: ProgressSink,
    ) -> Result<ImageObjectSegmentationResult>;

    /// Release one object's prompts and previous correction mask without
    /// discarding the image embedding or any other object.
    async fn remove(&mut self, object_id: &ImageObjectId);

    /// Clear every object's prompts and previous mask while retaining the
    /// immutable image embedding owned by this session.
    async fn reset(&mut self);
}

/// Creates isolated interactive sessions for promptable image segmentation.
///
/// [`RgbFrameRef`] is the canonical boundary — packed RGB24 with explicit pixel
/// dimensions, already in the intended display orientation. It is the twin of
/// iOS's `CGImage`, which likewise carries no EXIF orientation.
#[async_trait(?Send)]
pub trait ImageObjectSegmenting: EngineService {
    /// Create a session bound to one immutable image.
    ///
    /// Implementations may perform and cache expensive image encoding here so
    /// subsequent clicks only execute prompt-dependent inference.
    async fn make_session(
        &self,
        image: RgbFrameRef<'_>,
    ) -> Result<Box<dyn ImageObjectSegmentationSession>>;
}
