//! Compact FP16 state retained between EdgeTAM video frames.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMTemporalState.swift`.
//!
//! The upstream predictor stores spatial memory as BF16. The exported graphs
//! have no BF16 tensor boundary, so this port stores the already-FP16
//! memory-encoder output without a promote/demote round trip.

use std::collections::BTreeMap;

use ferment_engine_core::VideoObjectPropagationDirection;
use ferment_engine_media::MediaTime;
use half::f16;
use thiserror::Error;

pub const MAXIMUM_SPATIAL_MEMORIES: usize = 7;
pub const SPATIAL_TOKENS_PER_MEMORY: usize = 512;
pub const MEMORY_CHANNEL_COUNT: usize = 64;
pub const MAXIMUM_OBJECT_POINTERS: usize = 16;
pub const OBJECT_POINTER_COUNT: usize = 256;
pub const LOW_RESOLUTION_MASK_COUNT: usize = 256 * 256;
pub const MEMORY_FEATURE_COUNT: usize = SPATIAL_TOKENS_PER_MEMORY * MEMORY_CHANNEL_COUNT;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateError {
    #[error("EdgeTAM frame has invalid presentation time {0}.")]
    InvalidPresentationTime(i64),
    #[error("EdgeTAM {field} has {actual} elements; expected {expected}.")]
    InvalidElementCount {
        field: String,
        expected: usize,
        actual: usize,
    },
}

impl From<StateError> for ferment_engine_core::EngineError {
    fn from(error: StateError) -> Self {
        Self::Inference(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeTamStoredFrame {
    pub frame_index: usize,
    pub presentation_timestamp: MediaTime,
    pub low_resolution_mask: Vec<f16>,
    pub memory_features: Vec<f16>,
    pub object_pointer: Vec<f16>,
    pub predicted_iou: f32,
    pub object_score: f32,
}

impl EdgeTamStoredFrame {
    pub fn new(
        frame_index: usize,
        presentation_timestamp: MediaTime,
        low_resolution_mask: Vec<f16>,
        memory_features: Vec<f16>,
        object_pointer: Vec<f16>,
        predicted_iou: f32,
        object_score: f32,
    ) -> Result<Self, StateError> {
        if presentation_timestamp.value < 0
            || super::frame_time::validate(presentation_timestamp).is_err()
        {
            return Err(StateError::InvalidPresentationTime(
                presentation_timestamp.value,
            ));
        }
        check(
            "lowResolutionMask",
            LOW_RESOLUTION_MASK_COUNT,
            low_resolution_mask.len(),
        )?;
        check(
            "memoryFeatures",
            MEMORY_FEATURE_COUNT,
            memory_features.len(),
        )?;
        check("objectPointer", OBJECT_POINTER_COUNT, object_pointer.len())?;
        Ok(Self {
            frame_index,
            presentation_timestamp,
            low_resolution_mask,
            memory_features,
            object_pointer,
            predicted_iou,
            object_score,
        })
    }

    #[must_use]
    pub fn presentation_time(&self) -> f64 {
        self.presentation_timestamp.seconds()
    }
}

fn check(field: &str, expected: usize, actual: usize) -> Result<(), StateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(StateError::InvalidElementCount {
            field: field.to_owned(),
            expected,
            actual,
        })
    }
}

/// The fixed tensor contract the memory-attention graph reads.
///
/// Stock EdgeTAM's dynamic Python memory selection is expressed here as four
/// fixed-size tensors plus two validity vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedBank {
    /// Logical row-major `[1, 7, 512, 64]`.
    pub spatial_memory_features: Vec<f16>,
    /// Logical row-major `[1, 16, 256]`.
    pub object_pointers: Vec<f16>,
    /// Logical row-major `[1, 7]`, values exactly zero or one.
    pub spatial_validity: Vec<f16>,
    /// Logical row-major `[1, 16]`, values exactly zero or one.
    pub pointer_validity: Vec<f16>,
}

impl PackedBank {
    #[must_use]
    fn zeroed() -> Self {
        Self {
            spatial_memory_features: vec![
                f16::ZERO;
                MAXIMUM_SPATIAL_MEMORIES * MEMORY_FEATURE_COUNT
            ],
            object_pointers: vec![f16::ZERO; MAXIMUM_OBJECT_POINTERS * OBJECT_POINTER_COUNT],
            spatial_validity: vec![f16::ZERO; MAXIMUM_SPATIAL_MEMORIES],
            pointer_validity: vec![f16::ZERO; MAXIMUM_OBJECT_POINTERS],
        }
    }

    /// Reuse this bank for the next frame. The session owns one bank because
    /// one session never overlaps temporal predictions.
    fn clear(&mut self) {
        self.spatial_memory_features.fill(f16::ZERO);
        self.object_pointers.fill(f16::ZERO);
        self.spatial_validity.fill(f16::ZERO);
        self.pointer_validity.fill(f16::ZERO);
    }
}

/// Allocate the fixed memory bank once per video session.
#[must_use]
pub fn make_bank() -> PackedBank {
    PackedBank::zeroed()
}

/// Pack the exact one-conditioning-frame EdgeTAM memory order.
///
/// Spatial slots:
/// - slot 0: conditioning frame (temporal position 6 is baked into the model);
/// - slots 1...6: distances 6...1 from the current frame.
///
/// Pointer slots:
/// - slot 0: conditioning pointer;
/// - slots 1...15: immediately preceding frames at distances 1...15.
///
/// Missing frames stay all-zero and are excluded by their validity value.
pub fn pack(
    current_frame: usize,
    direction: VideoObjectPropagationDirection,
    conditioning: &EdgeTamStoredFrame,
    non_conditioning: &BTreeMap<usize, EdgeTamStoredFrame>,
) -> PackedBank {
    let mut bank = PackedBank::zeroed();
    pack_into(
        &mut bank,
        current_frame,
        direction,
        conditioning,
        non_conditioning,
    );
    bank
}

/// Repack directly into the session-owned bank.
pub fn pack_into(
    bank: &mut PackedBank,
    current_frame: usize,
    direction: VideoObjectPropagationDirection,
    conditioning: &EdgeTamStoredFrame,
    non_conditioning: &BTreeMap<usize, EdgeTamStoredFrame>,
) {
    bank.clear();

    bank.spatial_memory_features[..MEMORY_FEATURE_COUNT]
        .copy_from_slice(&conditioning.memory_features);
    bank.spatial_validity[0] = f16::ONE;

    // Stock eval filters conditioning pointers to the already-traversed side
    // of time (`only_obj_ptrs_in_the_past_for_eval=true`). Spatial conditioning
    // memory is always present, but a conditioning frame on the not-yet-
    // traversed side contributes no pointer.
    if is_on_traversed_side(conditioning.frame_index, current_frame, direction) {
        bank.object_pointers[..OBJECT_POINTER_COUNT].copy_from_slice(&conditioning.object_pointer);
        bank.pointer_validity[0] = f16::ONE;
    }

    // Original `_prepare_memory_conditioned_features` loops t_pos 1...6, whose
    // corresponding frame distances are 6...1.
    for slot in 1..MAXIMUM_SPATIAL_MEMORIES {
        let distance = MAXIMUM_SPATIAL_MEMORIES - slot;
        let Some(frame_index) = offset(current_frame, distance, direction) else {
            continue;
        };
        let Some(frame) = non_conditioning.get(&frame_index) else {
            continue;
        };
        let start = slot * MEMORY_FEATURE_COUNT;
        bank.spatial_memory_features[start..start + MEMORY_FEATURE_COUNT]
            .copy_from_slice(&frame.memory_features);
        bank.spatial_validity[slot] = f16::ONE;
    }

    for slot in 1..MAXIMUM_OBJECT_POINTERS {
        let Some(frame_index) = offset(current_frame, slot, direction) else {
            continue;
        };
        let Some(frame) = non_conditioning.get(&frame_index) else {
            continue;
        };
        let start = slot * OBJECT_POINTER_COUNT;
        bank.object_pointers[start..start + OBJECT_POINTER_COUNT]
            .copy_from_slice(&frame.object_pointer);
        bank.pointer_validity[slot] = f16::ONE;
    }
}

/// `None` where iOS's `Int` arithmetic would have gone negative — a frame index
/// below zero names no stored frame, so the slot stays invalid either way.
#[inline(always)]
fn offset(
    current_frame: usize,
    distance: usize,
    direction: VideoObjectPropagationDirection,
) -> Option<usize> {
    match direction {
        VideoObjectPropagationDirection::Forward => current_frame.checked_sub(distance),
        VideoObjectPropagationDirection::Reverse => current_frame.checked_add(distance),
    }
}

#[inline(always)]
fn is_on_traversed_side(
    candidate_frame: usize,
    current_frame: usize,
    direction: VideoObjectPropagationDirection,
) -> bool {
    match direction {
        VideoObjectPropagationDirection::Forward => candidate_frame <= current_frame,
        VideoObjectPropagationDirection::Reverse => candidate_frame >= current_frame,
    }
}
