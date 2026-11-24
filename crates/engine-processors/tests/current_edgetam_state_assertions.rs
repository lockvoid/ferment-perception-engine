//! Current iOS temporal, prompt, candidate and session-state assertions.
use ferment_engine_core::{
    NormalizedVideoBox, NormalizedVideoPoint, VideoObjectId, VideoObjectMask, VideoObjectMaskRle,
    VideoObjectPointLabel, VideoObjectPointPrompt, VideoObjectPrompt,
    VideoObjectPropagationDirection::{self, Forward, Reverse},
};
use ferment_engine_processors::edgetam::{
    model::EdgeTamDecodeOutput,
    processing::{self, PromptTensor},
    prompt_inference as prompt,
    session::ObjectState,
    temporal_state::{
        self as temporal, EdgeTamStoredFrame, LOW_RESOLUTION_MASK_COUNT, MEMORY_FEATURE_COUNT,
        OBJECT_POINTER_COUNT,
    },
};
use half::f16;
use std::collections::BTreeMap;

fn frame(index: usize, iou: f32) -> EdgeTamStoredFrame {
    EdgeTamStoredFrame::new(
        index,
        ferment_engine_media::MediaTime::new(
            index as i64 * 33_333,
            ferment_engine_media::TimeBase::MICROSECONDS,
        ),
        vec![f16::ZERO; LOW_RESOLUTION_MASK_COUNT],
        vec![f16::from_f32(index as f32); MEMORY_FEATURE_COUNT],
        vec![f16::from_f32(index as f32); OBJECT_POINTER_COUNT],
        iou,
        1.0,
    )
    .unwrap()
}

fn history(range: std::ops::RangeInclusive<usize>) -> BTreeMap<usize, EdgeTamStoredFrame> {
    range.map(|index| (index, frame(index, 0.5))).collect()
}

fn tokens() -> PromptTensor {
    PromptTensor {
        coordinates: vec![512.0, 512.0],
        labels: vec![1],
    }
}

fn state(
    conditioning: usize,
    ordinary: std::ops::RangeInclusive<usize>,
    anchors: &[usize],
    direction: VideoObjectPropagationDirection,
) -> ObjectState {
    let mut directions = ordinary
        .clone()
        .map(|i| (i, direction))
        .collect::<BTreeMap<_, _>>();
    directions.insert(conditioning, direction);
    ObjectState {
        conditioning: frame(conditioning, 0.875),
        non_conditioning: history(ordinary),
        prompts: anchors.iter().map(|i| (*i, tokens())).collect(),
        conditioning_was_tracked: true,
        directions,
    }
}

fn keys<V>(map: &BTreeMap<usize, V>) -> Vec<usize> {
    map.keys().copied().collect()
}

// KILL: retain future ordinary frames or drop a prompt anchor at propagation preflight.
#[test]
fn forward_and_reverse_preflight_keep_only_anchors_and_valid_history() {
    let mut forward = state(0, 1..=20, &[0, 10], Forward);
    forward.prepare_for_propagation(5, Forward);
    assert_eq!(keys(&forward.non_conditioning), [1, 2, 3, 4, 10]);
    assert_eq!(keys(&forward.directions), [0, 1, 2, 3, 4, 10]);
    let mut reverse = state(20, 0..=19, &[10, 20], Reverse);
    reverse.prepare_for_propagation(15, Reverse);
    assert_eq!(keys(&reverse.non_conditioning), [10, 16, 17, 18, 19]);
    assert_eq!(keys(&reverse.directions), [10, 16, 17, 18, 19, 20]);
}

// KILL: let ordinary tracking state grow beyond fifteen frames or evict an older prompt anchor.
#[test]
fn tracking_history_keeps_fifteen_ordinary_frames_and_the_prompt_anchor() {
    let mut state = state(0, 1..=20, &[0, 2], Forward);
    state.prune_tracking_history(20, Forward);
    let ordinary = state
        .non_conditioning
        .keys()
        .filter(|i| !state.prompts.contains_key(*i))
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(ordinary, (6..=20).collect::<Vec<_>>());
    assert_eq!(ordinary.len(), 15);
    assert!(state.non_conditioning.contains_key(&2));
    assert_eq!(
        keys(&state.directions),
        [vec![0, 2], (6..=20).collect()].concat()
    );
}

// KILL: admit a thirty-third distinct prompt or retain temporary corrections after preflight.
#[test]
fn thirty_two_prompt_anchors_bound_the_steady_and_temporary_state() {
    let mut state = state(0, 1..=100, &[0], Forward);
    for index in 1..32 {
        state.store_prompt(tokens(), index).unwrap();
    }
    assert_eq!(state.prompts.len(), 32);
    state.store_prompt(tokens(), 31).unwrap();
    assert_eq!(state.prompts.len(), 32);
    let error = state.store_prompt(tokens(), 32).unwrap_err();
    assert!(matches!(
        error,
        ferment_engine_processors::edgetam::error::EdgeTamError::TooManyPromptFrames {
            actual: 33,
            maximum: 32
        }
    ));
    state.prune_tracking_history(100, Forward);
    assert_eq!(state.non_conditioning.len() + 1, 47);
    let bytes = |f: &EdgeTamStoredFrame| {
        2 * (f.low_resolution_mask.len() + f.memory_features.len() + f.object_pointer.len())
    };
    let retained =
        bytes(&state.conditioning) + state.non_conditioning.values().map(bytes).sum::<usize>();
    assert_eq!(retained, 47 * 197_120);
    let correction = frame(0, 0.25);
    let correction_bytes = bytes(&correction);
    state.store(correction, false);
    assert_eq!(state.non_conditioning.len() + 1, 48);
    assert_eq!(retained + correction_bytes, 48 * bytes(&state.conditioning));
    state.prepare_for_propagation(0, Forward);
    assert!(!state.non_conditioning.contains_key(&0));
    assert_eq!(state.non_conditioning.len() + 1, 32);
}

// KILL: replace the original conditioning output with a temporary correction during propagation.
#[test]
fn a_temporary_conditioning_correction_is_only_the_next_corrections_input() {
    let mut state = state(0, 1..=2, &[0], Forward);
    state.store(frame(0, 0.25), false);
    assert_eq!(state.correction_input(0).unwrap().predicted_iou, 0.25);
    assert_eq!(state.output(0).unwrap().predicted_iou, 0.875);
    state.prepare_for_propagation(0, Forward);
    assert!(!state.non_conditioning.contains_key(&0));
    assert_eq!(state.conditioning.predicted_iou, 0.875);
    assert_eq!(state.output(0).unwrap().predicted_iou, 0.875);
}

// KILL: reverse the spatial or pointer slot order used by the fixed memory-attention graph.
#[test]
fn temporal_banks_follow_the_exact_forward_and_reverse_slot_order() {
    for (current, direction, conditioning, history, spatial, pointers) in [
        (
            17,
            Forward,
            0,
            history(1..=16),
            vec![0, 11, 12, 13, 14, 15, 16],
            vec![0, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
        ),
        (
            3,
            Reverse,
            20,
            history(4..=19),
            vec![20, 9, 8, 7, 6, 5, 4],
            vec![20, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ),
    ] {
        let bank = temporal::pack(current, direction, &frame(conditioning, 0.5), &history);
        assert_eq!(bank.spatial_validity, vec![f16::ONE; 7]);
        assert_eq!(bank.pointer_validity, vec![f16::ONE; 16]);
        for (slot, marker) in bank
            .spatial_memory_features
            .chunks_exact(MEMORY_FEATURE_COUNT)
            .zip(spatial)
        {
            assert!(slot.iter().all(|v| *v == f16::from_f32(marker as f32)));
        }
        for (slot, marker) in bank
            .object_pointers
            .chunks_exact(OBJECT_POINTER_COUNT)
            .zip(pointers)
        {
            assert!(slot.iter().all(|v| *v == f16::from_f32(marker as f32)));
        }
    }
}

// KILL: expose missing or future pointer state as valid memory.
#[test]
fn missing_and_future_temporal_slots_are_byte_zero_and_invalid() {
    let bank = temporal::pack(9, Forward, &frame(4, 0.5), &history(8..=8));
    assert_eq!(
        bank.spatial_validity,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0].map(f16::from_f32)
    );
    assert_eq!(&bank.pointer_validity[..2], &[f16::ONE; 2]);
    assert!(bank.pointer_validity[2..].iter().all(|v| v.to_bits() == 0));
    assert_eq!(bank.spatial_memory_features[0].to_f32(), 4.0);
    assert_eq!(
        bank.spatial_memory_features[6 * MEMORY_FEATURE_COUNT].to_f32(),
        8.0
    );
    assert_eq!(bank.object_pointers[0].to_f32(), 4.0);
    assert_eq!(bank.object_pointers[OBJECT_POINTER_COUNT].to_f32(), 8.0);
    assert!(
        bank.spatial_memory_features[MEMORY_FEATURE_COUNT..6 * MEMORY_FEATURE_COUNT]
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    assert!(
        bank.object_pointers[2 * OBJECT_POINTER_COUNT..]
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    let future = temporal::pack(4, Forward, &frame(10, 0.5), &BTreeMap::new());
    assert_eq!(future.spatial_validity[0], f16::ONE);
    assert_eq!(future.pointer_validity[0], f16::ZERO);
    assert!(
        future.object_pointers[..OBJECT_POINTER_COUNT]
            .iter()
            .all(|v| v.to_bits() == 0)
    );
}

// KILL: keep poisoned values from the preceding packed bank in invalid slots.
#[test]
fn reusing_a_full_temporal_bank_for_sparse_reverse_clears_every_stale_slot() {
    let mut bank = temporal::make_bank();
    temporal::pack_into(&mut bank, 17, Forward, &frame(0, 0.5), &history(1..=16));
    assert_eq!(
        bank,
        temporal::pack(17, Forward, &frame(0, 0.5), &history(1..=16))
    );
    temporal::pack_into(&mut bank, 3, Reverse, &frame(20, 0.5), &history(4..=4));
    assert_eq!(
        bank,
        temporal::pack(3, Reverse, &frame(20, 0.5), &history(4..=4))
    );
}

// KILL: allow any malformed low-resolution mask, memory feature or object pointer into retained state.
#[test]
fn each_stored_frame_boundary_rejects_the_wrong_shape() {
    for missing in 0..3 {
        let result = EdgeTamStoredFrame::new(
            0,
            ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
            vec![
                f16::ZERO;
                if missing == 0 {
                    0
                } else {
                    LOW_RESOLUTION_MASK_COUNT
                }
            ],
            vec![
                f16::ZERO;
                if missing == 1 {
                    0
                } else {
                    MEMORY_FEATURE_COUNT
                }
            ],
            vec![
                f16::ZERO;
                if missing == 2 {
                    0
                } else {
                    OBJECT_POINTER_COUNT
                }
            ],
            0.0,
            0.0,
        );
        assert!(result.is_err(), "boundary {missing}");
    }
}

fn point(x: f32, y: f32, label: VideoObjectPointLabel) -> VideoObjectPointPrompt {
    VideoObjectPointPrompt {
        point: NormalizedVideoPoint { x, y },
        label,
    }
}

// KILL: mis-scale appended coordinates or preserve old box tokens when replacing the prompt.
#[test]
fn append_and_replace_use_the_exact_current_prompt_coordinates() {
    let first = prompt::prompt_tensor(
        &VideoObjectPrompt::Points(vec![
            point(0.125, 0.25, VideoObjectPointLabel::Foreground),
            point(0.5, 0.75, VideoObjectPointLabel::Background),
        ]),
        true,
        None,
    )
    .unwrap();
    let appended = prompt::prompt_tensor(
        &VideoObjectPrompt::Points(vec![point(1.0, 0.0, VideoObjectPointLabel::Foreground)]),
        false,
        Some(&first),
    )
    .unwrap();
    assert_eq!(
        appended.coordinates,
        [128.0, 256.0, 512.0, 768.0, 1024.0, 0.0]
    );
    assert_eq!(appended.labels, [1, 0, 1]);
    assert_eq!(appended.point_count(), 3);
    let bounds = NormalizedVideoBox::from_extents(0.1, 0.2, 0.9, 0.8);
    let previous = processing::prompt_tensor(&VideoObjectPrompt::Box(bounds)).unwrap();
    let replaced = prompt::prompt_tensor(
        &VideoObjectPrompt::Points(vec![point(0.25, 0.75, VideoObjectPointLabel::Background)]),
        true,
        Some(&previous),
    )
    .unwrap();
    assert_eq!(replaced.coordinates, [256.0, 768.0]);
    assert_eq!(replaced.labels, [0]);
    assert_eq!(replaced.point_count(), 1);
    let initial = prompt::prompt_tensor(
        &VideoObjectPrompt::Points(vec![point(0.5, 0.5, VideoObjectPointLabel::Foreground)]),
        false,
        None,
    )
    .unwrap();
    assert_eq!(initial.coordinates, [512.0, 512.0]);
    assert_eq!(initial.labels, [1]);
}

// KILL: count a box as one token and admit nine decoder tokens.
#[test]
fn box_corners_consume_two_of_the_eight_prompt_tokens() {
    let bounds = NormalizedVideoBox::from_extents(0.0, 0.0, 1.0, 1.0);
    let mut points = (0..6)
        .map(|i| point(i as f32 / 5.0, 0.5, VideoObjectPointLabel::Foreground))
        .collect::<Vec<_>>();
    let accepted = prompt::prompt_tensor(
        &VideoObjectPrompt::BoxAndPoints {
            r#box: bounds,
            points: points.clone(),
        },
        true,
        None,
    )
    .unwrap();
    assert_eq!(accepted.point_count(), 8);
    points.push(point(1.0, 1.0, VideoObjectPointLabel::Background));
    assert!(matches!(
        prompt::prompt_tensor(
            &VideoObjectPrompt::BoxAndPoints {
                r#box: bounds,
                points
            },
            true,
            None
        ),
        Err(
            ferment_engine_processors::edgetam::error::EdgeTamError::TooManyPromptTokens {
                actual: 9,
                maximum: 8
            }
        )
    ));
}

// KILL: feed unclamped infinite logits into the next correction or alter untouched cells.
#[test]
fn previous_mask_clamp_preserves_nan_and_every_untouched_zero() {
    let mut values = vec![f16::ZERO; LOW_RESOLUTION_MASK_COUNT];
    values[..7].copy_from_slice(
        &[
            f32::NEG_INFINITY,
            -33.0,
            -32.0,
            31.0,
            33.0,
            f32::INFINITY,
            f32::NAN,
        ]
        .map(f16::from_f32),
    );
    let result = prompt::previous_mask_input(&values);
    assert_eq!(
        &result[..6],
        &[-32.0, -32.0, -32.0, 31.0, 32.0, 32.0].map(f16::from_f32)
    );
    assert!(result[6].is_nan());
    assert!(result[7..].iter().all(|v| v.to_bits() == 0));
}

fn decoded(ious: [f32; 4]) -> EdgeTamDecodeOutput {
    EdgeTamDecodeOutput::new(
        vec![f16::ZERO; 4 * 256 * 256],
        ious.map(f16::from_f32).to_vec(),
        vec![f16::ZERO; 4 * 256],
        vec![f16::from_f32(1.5)],
    )
    .unwrap()
}

// KILL: slice the wrong candidate plane or return mismatched mask/pointer metadata.
#[test]
fn candidate_selection_copies_exact_masks_pointers_and_metadata() {
    let expected_mask = (0..256 * 256)
        .map(|i| f16::from_f32(((i % 31) - 15) as f32 / 16.0))
        .collect::<Vec<_>>();
    let expected_pointer = (0..256)
        .map(|i| f16::from_f32(((i % 17) - 8) as f32 / 8.0))
        .collect::<Vec<_>>();
    let mut output = decoded([0.125, 0.25, 0.75, 0.5]);
    output.candidate_masks[2 * 256 * 256..3 * 256 * 256].copy_from_slice(&expected_mask);
    output.candidate_object_pointers[2 * 256..3 * 256].copy_from_slice(&expected_pointer);
    let selected = output.select_candidate(2).unwrap();
    assert_eq!(selected.mask, expected_mask);
    assert_eq!(selected.pointer, expected_pointer);
    assert_eq!(selected.predicted_iou, 0.75);
    assert_eq!(selected.object_score, 1.5);
    assert!(output.select_candidate(4).is_err());
    assert!(output.iou_prediction(4).is_err());
}

// KILL: choose candidate zero in multimask mode or resolve tied IOU to the last candidate.
#[test]
fn candidate_ties_keep_the_first_eligible_candidate() {
    let best = decoded([0.99, 0.5, 0.75, 0.75]);
    assert_eq!(prompt::selected_candidate_index(&best, false).unwrap(), 0);
    assert_eq!(prompt::selected_candidate_index(&best, true).unwrap(), 2);
    assert_eq!(
        prompt::selected_candidate_index(&decoded([0.0, 0.75, 0.75, 0.75]), true).unwrap(),
        1
    );
}

// KILL: encode row-major COCO runs or make image aliases diverge from the public video values.
#[test]
fn canonical_coco_runs_and_image_aliases_preserve_the_video_values() {
    for (width, height, pixels, counts) in [
        (3, 2, vec![0; 6], vec![6]),
        (3, 2, vec![255; 6], vec![0, 6]),
        (3, 2, vec![0, 255, 0, 255, 255, 0], vec![1, 3, 2]),
        (2, 2, vec![255, 0, 0, 0], vec![0, 1, 3]),
    ] {
        let mask = VideoObjectMask::new(width, height, pixels).unwrap();
        let rle = mask.uncompressed_coco_rle();
        assert_eq!((rle.width, rle.height), (width, height));
        assert_eq!(rle.counts, counts);
        assert_eq!(
            VideoObjectMask::from_uncompressed_coco_rle(&rle).unwrap(),
            mask
        );
    }
    let image_id = ferment_engine_core::ImageObjectId::new("subject");
    assert_eq!(image_id, VideoObjectId::new("subject"));
    let image_prompt: ferment_engine_core::ImageObjectPrompt =
        VideoObjectPrompt::Points(vec![point(0.25, 0.75, VideoObjectPointLabel::Foreground)]);
    let video_prompt: VideoObjectPrompt = image_prompt;
    assert_eq!(
        video_prompt,
        VideoObjectPrompt::Points(vec![point(0.25, 0.75, VideoObjectPointLabel::Foreground)])
    );
    let mask = ferment_engine_core::ImageObjectMask::new(2, 2, vec![0, 255, 255, 0]).unwrap();
    let image_rle: ferment_engine_core::ImageObjectMaskRle = mask.uncompressed_coco_rle();
    let video_rle: VideoObjectMaskRle = image_rle;
    assert_eq!(
        ferment_engine_core::ImageObjectMask::from_uncompressed_coco_rle(&video_rle).unwrap(),
        mask
    );
}
