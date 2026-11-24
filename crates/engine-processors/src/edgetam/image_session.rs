//! Interactive state for one fixed image.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMImageSession.swift`.
//!
//! The immutable image features are computed before this session is created.
//! Each object then owns only its prompt tokens and selected 256×256 mask, so
//! corrections never rerun the image encoder or any video-memory model.

use std::collections::BTreeMap;

use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, ImageObjectId, ImageObjectPrompt, ImageObjectSegmentationResult,
    ImageObjectSegmentationSession, ImageObjectSegmentationSourceInfo, ProgressSink, Result,
};
use half::f16;

use super::error::EdgeTamError;
use super::model::{EdgeTamImageFeatures, EdgeTamPromptDecoding};
use super::processing::PromptTensor;
use super::prompt_inference::{self, MaskWorkspace};
use super::temporal_state::LOW_RESOLUTION_MASK_COUNT;

/// One retained FP16 correction mask is 256×256×2 bytes. Thirty-two active IDs
/// therefore have a fixed 4 MiB logical mask budget.
pub const MAXIMUM_ACTIVE_OBJECTS: usize = 32;

#[derive(Debug, Clone)]
struct ObjectState {
    prompt: PromptTensor,
    previous_mask: Vec<f16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleSnapshot {
    pub active_object_count: usize,
    pub retained_mask_byte_count: usize,
}

pub struct EdgeTamImageSession<D: EdgeTamPromptDecoding> {
    model: D,
    features: EdgeTamImageFeatures,
    source_info: ImageObjectSegmentationSourceInfo,
    objects: BTreeMap<ImageObjectId, ObjectState>,
    mask_workspace: Option<MaskWorkspace>,
}

impl<D: EdgeTamPromptDecoding> EdgeTamImageSession<D> {
    #[must_use]
    pub fn new(
        model: D,
        features: EdgeTamImageFeatures,
        source_info: ImageObjectSegmentationSourceInfo,
    ) -> Self {
        Self {
            model,
            features,
            source_info,
            objects: BTreeMap::new(),
            mask_workspace: None,
        }
    }

    #[must_use]
    pub fn lifecycle_snapshot(&self) -> LifecycleSnapshot {
        LifecycleSnapshot {
            active_object_count: self.objects.len(),
            retained_mask_byte_count: self.objects.len()
                * LOW_RESOLUTION_MASK_COUNT
                * std::mem::size_of::<f16>(),
        }
    }
}

#[async_trait(?Send)]
impl<D: EdgeTamPromptDecoding + Send> ImageObjectSegmentationSession for EdgeTamImageSession<D> {
    fn source_info(&self) -> ImageObjectSegmentationSourceInfo {
        self.source_info
    }

    fn maximum_active_object_count(&self) -> usize {
        MAXIMUM_ACTIVE_OBJECTS
    }

    async fn apply_with_cancellation(
        &mut self,
        prompt: &ImageObjectPrompt,
        object_id: &ImageObjectId,
        replacing_existing_prompts: bool,
        cancellation: ProgressSink,
    ) -> Result<ImageObjectSegmentationResult> {
        if cancellation.is_closed() {
            return Err(EngineError::Cancelled);
        }
        if !self.objects.contains_key(object_id)
            && self.objects.len() >= self.maximum_active_object_count()
        {
            return Err(EdgeTamError::TooManyImageObjects {
                actual: self.objects.len() + 1,
                maximum: MAXIMUM_ACTIVE_OBJECTS,
            }
            .into());
        }

        let previous_state = self.objects.get(object_id);
        let prompt_tensor = prompt_inference::prompt_tensor(
            prompt,
            replacing_existing_prompts,
            previous_state.map(|state| &state.prompt),
        )?;
        let previous_mask = previous_state.map(|state| state.previous_mask.clone());

        let decoded = cancellation
            .until_cancelled(self.model.decode(
                &self.features,
                &prompt_tensor,
                // Pinned `SAM2ImagePredictor.set_image` adds `no_mem_embed`
                // once to the lowest-resolution image feature.
                true,
                // Unlike the video predictor, `SAM2ImagePredictor` returns the
                // raw mask decoder logits when the object score is negative.
                false,
                previous_mask.as_deref(),
            ))
            .await?;

        if cancellation.is_closed() {
            return Err(EngineError::Cancelled);
        }
        let candidate_index =
            prompt_inference::selected_candidate_index(&decoded, prompt_tensor.point_count() <= 1)?;
        let candidate = decoded.select_candidate(candidate_index)?;

        let workspace = match &mut self.mask_workspace {
            Some(workspace) => workspace,
            slot => slot.insert(MaskWorkspace::new(
                self.source_info.width,
                self.source_info.height,
            )?),
        };
        let mask = workspace.mask(&candidate.mask)?;

        // Compute the retained correction input before the final cancellation
        // check: clamping traverses all 65,536 cells and must not create a
        // post-check window in which a cancelled decode can commit.
        let previous_mask = prompt_inference::previous_mask_input(&candidate.mask);
        if cancellation.is_closed() {
            return Err(EngineError::Cancelled);
        }
        self.objects.insert(
            object_id.clone(),
            ObjectState {
                prompt: prompt_tensor,
                // Clamping preserves the exact binary result already read out
                // of `candidate.mask` and is what the next correction feeds
                // back in.
                previous_mask,
            },
        );
        Ok(ImageObjectSegmentationResult {
            object_id: object_id.clone(),
            mask,
            predicted_iou: Some(candidate.predicted_iou),
        })
    }

    async fn remove(&mut self, object_id: &ImageObjectId) {
        self.objects.remove(object_id);
    }

    async fn reset(&mut self) {
        self.objects.clear();
        // `features` deliberately remains cached for this fixed image.
    }
}
