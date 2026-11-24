//! Ported from `Processors/EdgeTAM/EdgeTAMSession.swift`.

use std::collections::BTreeMap;

use async_trait::async_trait;
use ferment_engine_core::{
    ProgressSink, Result, VideoObjectId, VideoObjectMaskFrame, VideoObjectMaskSink,
    VideoObjectPrompt, VideoObjectPropagationDirection, VideoObjectSegmentationSession,
    VideoObjectSegmentationSourceInfo,
};

use super::error::EdgeTamError;
use super::frame_source::{EdgeTamFrameSource, check_cancellation};
use super::model::{EdgeTamImageFeatures, EdgeTamModel, EdgeTamPromptDecoding};
use super::preprocessing::Workspace;
use super::processing::PromptTensor;
use super::prompt_inference::{self, MaskWorkspace};
use super::temporal_state::{self, EdgeTamStoredFrame, MAXIMUM_OBJECT_POINTERS, PackedBank};

/// One retained prompt anchor is about 193 KiB of FP16 temporal state. Capping
/// anchors keeps steady propagation below an approximately 10 MiB
/// temporal-state budget, including its fifteen ordinary rolling frames.
pub const MAXIMUM_PROMPTED_FRAME_COUNT: usize = 32;

/// Internal for deterministic regression tests; session mutation stays
/// exclusive because the map holding these values is private.
#[derive(Debug, Clone)]
pub struct ObjectState {
    pub conditioning: EdgeTamStoredFrame,
    pub non_conditioning: BTreeMap<usize, EdgeTamStoredFrame>,
    pub prompts: BTreeMap<usize, PromptTensor>,
    pub conditioning_was_tracked: bool,
    pub directions: BTreeMap<usize, VideoObjectPropagationDirection>,
}

impl ObjectState {
    pub fn store_prompt(
        &mut self,
        prompt: PromptTensor,
        frame_index: usize,
    ) -> std::result::Result<(), EdgeTamError> {
        self.validate_prompt_frame_budget(frame_index)?;
        self.prompts.insert(frame_index, prompt);
        Ok(())
    }

    pub fn validate_prompt_frame_budget(
        &self,
        frame_index: usize,
    ) -> std::result::Result<(), EdgeTamError> {
        if self.prompts.contains_key(&frame_index) {
            return Ok(());
        }
        let requested_count = self.prompts.len() + 1;
        if requested_count > MAXIMUM_PROMPTED_FRAME_COUNT {
            return Err(EdgeTamError::TooManyPromptFrames {
                actual: requested_count,
                maximum: MAXIMUM_PROMPTED_FRAME_COUNT,
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn output(&self, frame_index: usize) -> Option<&EdgeTamStoredFrame> {
        if self.conditioning.frame_index == frame_index {
            return Some(&self.conditioning);
        }
        self.non_conditioning.get(&frame_index)
    }

    /// A second correction before propagation uses the most recent temporary
    /// correction mask. Pinned EdgeTAM stores a correction made to an
    /// already-tracked conditioning frame in the non-conditioning temporary map
    /// even though the original conditioning output still wins during
    /// propagation preflight.
    #[must_use]
    pub fn correction_input(&self, frame_index: usize) -> Option<&EdgeTamStoredFrame> {
        self.non_conditioning
            .get(&frame_index)
            .or_else(|| self.output(frame_index))
    }

    pub fn store(&mut self, output: EdgeTamStoredFrame, as_conditioning: bool) {
        if as_conditioning {
            self.non_conditioning.remove(&output.frame_index);
            self.conditioning = output;
        } else {
            self.non_conditioning.insert(output.frame_index, output);
        }
    }

    pub fn prepare_for_propagation(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
    ) {
        // Stock preflight consolidates conditioning outputs first, then drops a
        // same-frame non-conditioning temporary correction.
        self.non_conditioning.remove(&self.conditioning.frame_index);
        let prompts = &self.prompts;
        self.non_conditioning.retain(|index, _| {
            if prompts.contains_key(index) {
                return true;
            }
            match direction {
                VideoObjectPropagationDirection::Forward => *index < frame_index,
                VideoObjectPropagationDirection::Reverse => *index > frame_index,
            }
        });
        self.prune_direction_metadata();
    }

    pub fn prune_tracking_history(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
    ) {
        // Slot zero is the conditioning pointer. The remaining 15 slots consume
        // the current output plus fourteen predecessors on the next frame, so
        // exactly fifteen ordinary outputs are sufficient.
        let maximum_distance = MAXIMUM_OBJECT_POINTERS - 2;
        let prompts = &self.prompts;
        self.non_conditioning.retain(|index, _| {
            if prompts.contains_key(index) {
                return true;
            }
            match direction {
                VideoObjectPropagationDirection::Forward => {
                    *index <= frame_index && *index + maximum_distance >= frame_index
                }
                VideoObjectPropagationDirection::Reverse => {
                    *index >= frame_index && *index <= frame_index + maximum_distance
                }
            }
        });
        self.prune_direction_metadata();
    }

    fn prune_direction_metadata(&mut self) {
        let conditioning_frame = self.conditioning.frame_index;
        let prompts = &self.prompts;
        let non_conditioning = &self.non_conditioning;
        self.directions.retain(|index, _| {
            *index == conditioning_frame
                || prompts.contains_key(index)
                || non_conditioning.contains_key(index)
        });
    }
}

struct EncodedFrame {
    frame_index: usize,
    presentation_timestamp: ferment_engine_media::MediaTime,
    features: EdgeTamImageFeatures,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub prompted_frame_count: usize,
    pub stored_frame_count: usize,
    pub prompt_frame_budget_remaining: usize,
    pub has_cached_encoded_features: bool,
    pub has_temporal_tensor_bank: bool,
    pub preprocessing_workspace_byte_count: usize,
    pub has_decoder_resources: bool,
}

pub struct EdgeTamSession {
    model: EdgeTamModel,
    frame_source: EdgeTamFrameSource,
    source_info: VideoObjectSegmentationSourceInfo,
    objects: BTreeMap<VideoObjectId, ObjectState>,
    cached_encoded_frame: Option<EncodedFrame>,
    temporal_bank: Option<PackedBank>,
    preprocessing_workspace: Option<Workspace>,
    mask_workspace: Option<MaskWorkspace>,
}

impl EdgeTamSession {
    #[must_use]
    pub fn new(model: EdgeTamModel, frame_source: EdgeTamFrameSource) -> Self {
        let source_info = frame_source.source_info();
        Self {
            model,
            frame_source,
            source_info,
            objects: BTreeMap::new(),
            cached_encoded_frame: None,
            temporal_bank: None,
            preprocessing_workspace: None,
            mask_workspace: None,
        }
    }

    /// Runtime evidence for memory-bound tests. Counts are derived from the
    /// session's actual owners rather than a parallel test model.
    #[must_use]
    pub fn memory_snapshot(&self) -> MemorySnapshot {
        let state = self.objects.values().next();
        let prompted_frame_count = state.map_or(0, |state| state.prompts.len());
        MemorySnapshot {
            prompted_frame_count,
            stored_frame_count: state.map_or(0, |state| 1 + state.non_conditioning.len()),
            prompt_frame_budget_remaining: MAXIMUM_PROMPTED_FRAME_COUNT - prompted_frame_count,
            has_cached_encoded_features: self.cached_encoded_frame.is_some(),
            has_temporal_tensor_bank: self.temporal_bank.is_some(),
            preprocessing_workspace_byte_count: self
                .preprocessing_workspace
                .as_ref()
                .map_or(0, Workspace::retained_byte_count),
            has_decoder_resources: self.frame_source.has_decoder_resources(),
        }
    }

    // MARK: - Neural stages

    async fn encoded_frame(
        &mut self,
        frame_index: usize,
        cancellation: &ProgressSink,
    ) -> Result<()> {
        check_cancellation(cancellation)?;
        if self
            .cached_encoded_frame
            .as_ref()
            .is_some_and(|cached| cached.frame_index == frame_index)
        {
            return Ok(());
        }
        let source = self
            .frame_source
            .frame_with_cancellation(frame_index, cancellation.clone())
            .await?;
        self.encode(source, cancellation).await
    }

    async fn sequential_encoded_frame(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        cancellation: &ProgressSink,
    ) -> Result<()> {
        check_cancellation(cancellation)?;
        if self
            .cached_encoded_frame
            .as_ref()
            .is_some_and(|cached| cached.frame_index == frame_index)
        {
            return Ok(());
        }
        let source = self
            .frame_source
            .sequential_frame_with_cancellation(frame_index, direction, cancellation.clone())
            .await?;
        self.encode(source, cancellation).await
    }

    async fn encode(
        &mut self,
        source: super::preprocessing::EdgeTamRgbFrame,
        cancellation: &ProgressSink,
    ) -> Result<()> {
        let workspace = match &mut self.preprocessing_workspace {
            Some(workspace) => workspace,
            slot => slot.insert(Workspace::new(
                self.source_info.width,
                self.source_info.height,
            )?),
        };
        let normalized = workspace.normalized_image(&source)?;
        check_cancellation(cancellation)?;
        let features = cancellation
            .until_cancelled(self.model.encode_image(&normalized))
            .await?;
        check_cancellation(cancellation)?;
        // Source RGB pixels are no longer needed after the image encoder.
        // Retaining only metadata avoids pinning width × height × 3 bytes
        // (about 24 MiB for 4K) between session operations.
        self.cached_encoded_frame = Some(EncodedFrame {
            frame_index: source.frame_index,
            presentation_timestamp: source.presentation_timestamp,
            features,
        });
        Ok(())
    }

    async fn memory_conditioned_features(
        &mut self,
        raw: &EdgeTamImageFeatures,
        object_id: &VideoObjectId,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamImageFeatures> {
        let state = self
            .objects
            .get(object_id)
            .ok_or(EdgeTamError::StaleOperation)?;
        let bank = self
            .temporal_bank
            .get_or_insert_with(temporal_state::make_bank);
        temporal_state::pack_into(
            bank,
            frame_index,
            direction,
            &state.conditioning,
            &state.non_conditioning,
        );
        check_cancellation(cancellation)?;
        let conditioned = cancellation
            .until_cancelled(self.model.condition(
                &raw.vision_features,
                &bank.spatial_memory_features,
                &bank.object_pointers,
                &bank.spatial_validity,
                &bank.pointer_validity,
            ))
            .await?;
        check_cancellation(cancellation)?;
        raw.replacing_vision_features(conditioned)
    }

    #[allow(clippy::too_many_arguments)]
    async fn predict_prompted_frame(
        &mut self,
        frame_index: usize,
        raw_features: &EdgeTamImageFeatures,
        decoder_features: &EdgeTamImageFeatures,
        prompt: &PromptTensor,
        add_no_memory_embedding: bool,
        previous_mask: Option<&[half::f16]>,
        presentation_timestamp: ferment_engine_media::MediaTime,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamStoredFrame> {
        check_cancellation(cancellation)?;
        let decoded = cancellation
            .until_cancelled(self.model.decode(
                decoder_features,
                prompt,
                add_no_memory_embedding,
                true,
                previous_mask,
            ))
            .await?;
        check_cancellation(cancellation)?;
        let candidate_index =
            prompt_inference::selected_candidate_index(&decoded, prompt.point_count() <= 1)?;
        self.store_candidate(
            &decoded,
            candidate_index,
            frame_index,
            presentation_timestamp,
            raw_features,
            cancellation,
        )
        .await
    }

    async fn predict_tracking_frame(
        &mut self,
        frame_index: usize,
        raw_features: &EdgeTamImageFeatures,
        decoder_features: &EdgeTamImageFeatures,
        presentation_timestamp: ferment_engine_media::MediaTime,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamStoredFrame> {
        // The exact stock representation for "no prompt": one not-a-point
        // token, after which PromptEncoder appends its own padding token.
        // Tracking uses the three multimask candidates.
        let tracking_prompt = PromptTensor {
            coordinates: vec![0.0, 0.0],
            labels: vec![-1],
        };
        check_cancellation(cancellation)?;
        let decoded = cancellation
            .until_cancelled(self.model.decode(
                decoder_features,
                &tracking_prompt,
                false,
                true,
                None,
            ))
            .await?;
        check_cancellation(cancellation)?;
        let candidate_index = prompt_inference::selected_candidate_index(&decoded, true)?;
        self.store_candidate(
            &decoded,
            candidate_index,
            frame_index,
            presentation_timestamp,
            raw_features,
            cancellation,
        )
        .await
    }

    async fn store_candidate(
        &mut self,
        decoded: &super::model::EdgeTamDecodeOutput,
        candidate_index: usize,
        frame_index: usize,
        presentation_timestamp: ferment_engine_media::MediaTime,
        raw_features: &EdgeTamImageFeatures,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamStoredFrame> {
        let candidate = decoded.select_candidate(candidate_index)?;
        check_cancellation(cancellation)?;
        let memory = cancellation
            .until_cancelled(
                self.model
                    .encode_memory(&raw_features.vision_features, &candidate.mask),
            )
            .await?;
        check_cancellation(cancellation)?;
        EdgeTamStoredFrame::new(
            frame_index,
            presentation_timestamp,
            candidate.mask,
            memory,
            candidate.pointer,
            candidate.predicted_iou,
            candidate.object_score,
        )
        .map_err(Into::into)
    }

    // MARK: - Deterministic host stages

    fn make_mask_frame(
        &mut self,
        object_id: &VideoObjectId,
        output: &EdgeTamStoredFrame,
    ) -> Result<VideoObjectMaskFrame> {
        let workspace = match &mut self.mask_workspace {
            Some(workspace) => workspace,
            slot => slot.insert(MaskWorkspace::new(
                self.source_info.width,
                self.source_info.height,
            )?),
        };
        let mask = workspace.mask(&output.low_resolution_mask)?;
        Ok(VideoObjectMaskFrame {
            frame_index: output.frame_index,
            presentation_time: output.presentation_time(),
            object_id: object_id.clone(),
            mask,
            predicted_iou: Some(output.predicted_iou),
        })
    }

    fn inferred_direction(
        frame_index: usize,
        conditioning_frame: usize,
    ) -> VideoObjectPropagationDirection {
        if frame_index >= conditioning_frame {
            VideoObjectPropagationDirection::Forward
        } else {
            VideoObjectPropagationDirection::Reverse
        }
    }

    fn validate(&self, frame_index: usize) -> Result<()> {
        if frame_index >= self.source_info.frame_count {
            return Err(EdgeTamError::InvalidFrameIndex {
                index: frame_index,
                frame_count: self.source_info.frame_count,
            }
            .into());
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl VideoObjectSegmentationSession for EdgeTamSession {
    fn source_info(&self) -> VideoObjectSegmentationSourceInfo {
        self.source_info
    }

    async fn apply_with_cancellation(
        &mut self,
        prompt: &VideoObjectPrompt,
        object_id: &VideoObjectId,
        frame_index: usize,
        replacing_existing_prompts: bool,
        cancellation: ProgressSink,
    ) -> Result<VideoObjectMaskFrame> {
        let result = self
            .apply_prompt(
                prompt,
                object_id,
                frame_index,
                replacing_existing_prompts,
                &cancellation,
            )
            .await;
        if result.is_err() {
            self.frame_source.cancel_decoder_resources();
        }
        result
    }

    async fn propagate_with_cancellation(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        sink: &dyn VideoObjectMaskSink,
        cancellation: ProgressSink,
    ) -> Result<()> {
        let outcome = self
            .run_propagation(frame_index, direction, sink, &cancellation)
            .await;
        self.frame_source.cancel_decoder_resources();
        outcome
    }

    async fn reset(&mut self) {
        self.frame_source.cancel_decoder_resources();
        self.objects.clear();
        self.cached_encoded_frame = None;
        self.temporal_bank = None;
    }
}

impl EdgeTamSession {
    async fn apply_prompt(
        &mut self,
        prompt: &VideoObjectPrompt,
        object_id: &VideoObjectId,
        frame_index: usize,
        replacing_existing_prompts: bool,
        cancellation: &ProgressSink,
    ) -> Result<VideoObjectMaskFrame> {
        check_cancellation(cancellation)?;
        self.validate(frame_index)?;
        if self
            .objects
            .keys()
            .next()
            .is_some_and(|existing| existing != object_id)
        {
            return Err(EdgeTamError::UnsupportedReferenceProfile(format!(
                "V1 supports one tracked object per session; reset before adding '{}'.",
                object_id.raw_value()
            ))
            .into());
        }
        if let Some(state) = self.objects.get(object_id) {
            state.validate_prompt_frame_budget(frame_index)?;
        }

        let prompt_tensor = prompt_inference::prompt_tensor(
            prompt,
            replacing_existing_prompts,
            self.objects
                .get(object_id)
                .and_then(|state| state.prompts.get(&frame_index)),
        )?;

        self.encoded_frame(frame_index, cancellation).await?;
        let (raw_features, presentation_timestamp) = {
            let encoded = self
                .cached_encoded_frame
                .as_ref()
                .ok_or(EdgeTamError::StaleOperation)?;
            (encoded.features.clone(), encoded.presentation_timestamp)
        };

        if self.objects.contains_key(object_id) {
            let (previous_mask, is_initial_conditioning, direction) = {
                let state = &self.objects[object_id];
                let Some(previous_output) = state.correction_input(frame_index) else {
                    return Err(EdgeTamError::UnsupportedReferenceProfile(format!(
                        "V1 uses one conditioning frame. Propagate to frame {frame_index} before \
                         correcting it, or reset before moving the initial prompt."
                    ))
                    .into());
                };
                let is_initial_conditioning = state.conditioning.frame_index == frame_index
                    && !state.conditioning_was_tracked;
                let direction = state
                    .directions
                    .get(&frame_index)
                    .copied()
                    .unwrap_or_else(|| {
                        Self::inferred_direction(frame_index, state.conditioning.frame_index)
                    });
                (
                    prompt_inference::previous_mask_input(&previous_output.low_resolution_mask),
                    is_initial_conditioning,
                    direction,
                )
            };

            let decoder_features = if is_initial_conditioning {
                raw_features.clone()
            } else {
                self.memory_conditioned_features(
                    &raw_features,
                    object_id,
                    frame_index,
                    direction,
                    cancellation,
                )
                .await?
            };

            check_cancellation(cancellation)?;
            let output = self
                .predict_prompted_frame(
                    frame_index,
                    &raw_features,
                    &decoder_features,
                    &prompt_tensor,
                    is_initial_conditioning,
                    Some(&previous_mask),
                    presentation_timestamp,
                    cancellation,
                )
                .await?;
            let frame = self.make_mask_frame(object_id, &output)?;
            // Pinned V1 has `add_all_frames_to_correct_as_cond=false`. Before
            // the initial frame has been tracked, corrections replace its
            // conditioning output. After tracking, stock puts the correction in
            // the temporary non-conditioning map; propagation preflight then
            // retains the original condition.
            let mut state = self
                .objects
                .get(object_id)
                .ok_or(EdgeTamError::StaleOperation)?
                .clone();
            let is_conditioning =
                state.conditioning.frame_index == frame_index && !state.conditioning_was_tracked;
            state.store(output, is_conditioning);
            state.store_prompt(prompt_tensor, frame_index)?;
            check_cancellation(cancellation)?;
            self.objects.insert(object_id.clone(), state);
            return Ok(frame);
        }

        // The first prompt creates the session's single conditioning frame.
        let output = self
            .predict_prompted_frame(
                frame_index,
                &raw_features,
                &raw_features,
                &prompt_tensor,
                true,
                None,
                presentation_timestamp,
                cancellation,
            )
            .await?;
        let frame = self.make_mask_frame(object_id, &output)?;
        let state = ObjectState {
            conditioning: output,
            non_conditioning: BTreeMap::new(),
            prompts: BTreeMap::from([(frame_index, prompt_tensor)]),
            conditioning_was_tracked: false,
            directions: BTreeMap::new(),
        };
        check_cancellation(cancellation)?;
        self.objects.insert(object_id.clone(), state);
        Ok(frame)
    }

    async fn run_propagation(
        &mut self,
        frame_index: usize,
        direction: VideoObjectPropagationDirection,
        sink: &dyn VideoObjectMaskSink,
        cancellation: &ProgressSink,
    ) -> Result<()> {
        check_cancellation(cancellation)?;
        self.validate(frame_index)?;
        if self.objects.is_empty() {
            return Err(EdgeTamError::NoPromptedObjects.into());
        }
        // Prompted outputs are persistent anchors. Ordinary tracking outputs on
        // or beyond the propagation start are invalidated so a correction or a
        // direction change can never reuse stale masks.
        for state in self.objects.values_mut() {
            state.prepare_for_propagation(frame_index, direction);
        }

        let object_ids: Vec<VideoObjectId> = self.objects.keys().cloned().collect();
        let mut current_frame = frame_index;
        loop {
            check_cancellation(cancellation)?;
            if current_frame >= self.source_info.frame_count {
                break;
            }
            let mut encoded_this_frame = false;

            for object_id in &object_ids {
                let prompted = self.objects[object_id].prompts.contains_key(&current_frame);
                let output = if prompted {
                    self.objects[object_id]
                        .output(current_frame)
                        .cloned()
                        .ok_or_else(|| {
                            EdgeTamError::InvalidModelBoundary(format!(
                                "prompted frame {current_frame} has no retained output"
                            ))
                        })?
                } else {
                    if !encoded_this_frame {
                        self.sequential_encoded_frame(current_frame, direction, cancellation)
                            .await?;
                        encoded_this_frame = true;
                    }
                    let (raw_features, presentation_timestamp) = {
                        let encoded = self
                            .cached_encoded_frame
                            .as_ref()
                            .ok_or(EdgeTamError::StaleOperation)?;
                        (encoded.features.clone(), encoded.presentation_timestamp)
                    };
                    check_cancellation(cancellation)?;
                    let conditioned = self
                        .memory_conditioned_features(
                            &raw_features,
                            object_id,
                            current_frame,
                            direction,
                            cancellation,
                        )
                        .await?;
                    check_cancellation(cancellation)?;

                    self.predict_tracking_frame(
                        current_frame,
                        &raw_features,
                        &conditioned,
                        presentation_timestamp,
                        cancellation,
                    )
                    .await?
                };

                let frame = self.make_mask_frame(object_id, &output)?;
                check_cancellation(cancellation)?;
                {
                    let state = self
                        .objects
                        .get_mut(object_id)
                        .ok_or(EdgeTamError::StaleOperation)?;
                    if !prompted {
                        state.non_conditioning.insert(current_frame, output);
                    }
                    if current_frame == state.conditioning.frame_index {
                        state.conditioning_was_tracked = true;
                    }
                    state.directions.insert(current_frame, direction);
                    state.prune_tracking_history(current_frame, direction);
                }

                if !sink.emit(frame) {
                    return Ok(());
                }
            }

            match direction {
                VideoObjectPropagationDirection::Forward => current_frame += 1,
                VideoObjectPropagationDirection::Reverse => {
                    let Some(previous) = current_frame.checked_sub(1) else {
                        break;
                    };
                    current_frame = previous;
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "ort", not(target_arch = "wasm32")))]
#[path = "current_lifecycle_tests.rs"]
mod current_lifecycle_tests;
