//! Stock EdgeTAM interactive image and video segmentation.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAM.swift`. All neural tensor boundaries
//! are FP16, exactly as the iOS Core ML assets are.
//!
//! Pipeline, per prompted frame:
//! 1. decode → the source RGB8 frame (video) or the caller's image
//! 2. preprocess → `[1, 3, 1024, 1024]` FP16
//! 3. image encoder → vision features + two high-resolution maps
//! 4. memory attention (tracking frames only) → conditioned vision features
//! 5. prompt/mask decoder → four candidates; host policy selects one
//! 6. memory encoder → the memory this frame contributes to later ones
//! 7. bilinear resize + threshold → the display mask

pub mod error;
mod frame_orientation;
pub mod frame_source;
mod frame_time;
pub mod image_preprocessing;
pub mod image_session;
pub mod model;
pub mod preprocessing;
pub mod processing;
pub mod prompt_inference;
pub mod session;
pub mod temporal_state;

#[cfg(test)]
mod oracle_tests;

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ferment_engine_core::{
    ImageObjectSegmentationSession, ImageObjectSegmentationSourceInfo, ImageObjectSegmenting,
    ProgressSink, Result, RgbFrameRef, VideoObjectSegmentationSession, VideoObjectSegmenting,
};
use ferment_engine_infer::{Backend, ModelId};
use ferment_engine_media::VideoBackend;

pub use error::EdgeTamError;
pub use image_session::EdgeTamImageSession;
pub use model::EdgeTamModel;
pub use session::EdgeTamSession;

use preprocessing::EdgeTamRgbFrame;

#[derive(Debug, Clone)]
pub struct EdgeTam {
    model: EdgeTamModel,
    video: VideoBackend,
    temporary_directory: PathBuf,
}

impl EdgeTam {
    /// The graphs this segmenter needs on disk.
    pub const MODEL_FILES: [ModelId; 4] = EdgeTamModel::MODEL_FILES;

    #[must_use]
    pub fn new(backend: Backend, video: VideoBackend) -> Self {
        Self {
            model: EdgeTamModel::new(backend),
            video,
            temporary_directory: std::env::temp_dir(),
        }
    }

    #[must_use]
    pub fn with_temporary_directory(mut self, directory: PathBuf) -> Self {
        self.temporary_directory = directory;
        self
    }
}

#[async_trait(?Send)]
impl ImageObjectSegmenting for EdgeTam {
    /// Encode one immutable still image and retain its features for every
    /// prompt/correction made through the returned session.
    async fn make_session(
        &self,
        image: RgbFrameRef<'_>,
    ) -> Result<Box<dyn ImageObjectSegmentationSession>> {
        let source = EdgeTamRgbFrame::new(
            0,
            ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
            image.width as usize,
            image.height as usize,
            image.data.to_vec(),
        )?;
        let normalized = image_preprocessing::normalized_image(&source)?;
        let features = self.model.encode_image(&normalized).await?;
        Ok(Box::new(EdgeTamImageSession::new(
            self.model.clone(),
            features,
            ImageObjectSegmentationSourceInfo {
                width: source.width,
                height: source.height,
            },
        )))
    }
}

#[async_trait(?Send)]
impl VideoObjectSegmenting for EdgeTam {
    async fn make_session_with_cancellation(
        &self,
        video_path: &Path,
        cancellation: ProgressSink,
    ) -> Result<Box<dyn VideoObjectSegmentationSession>> {
        let frame_source = frame_source::EdgeTamFrameSource::new_in_with_cancellation(
            self.video.clone(),
            video_path,
            &self.temporary_directory,
            cancellation,
        )
        .await?;
        Ok(Box::new(EdgeTamSession::new(
            self.model.clone(),
            frame_source,
        )))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod current_cpu_hot_paths_tests;
