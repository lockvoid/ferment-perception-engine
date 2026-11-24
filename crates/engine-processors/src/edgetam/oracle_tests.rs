//! EdgeTAM deterministic-stage parity against the PyTorch origin.
//!
//! Every expected value here is bytes written by `generate_swift_fixtures.py`
//! (EdgeTAM upstream `7711e012a`, torch 2.7.0 / torchvision 0.22.0), staged and
//! sha256-verified into `scripts/exports/rust_goldens/edgetam_v1/` by
//! `scripts/exports/edgetam/gen_rust_goldens.py`. Nothing is derived through the
//! code under test, and the iOS suite that grades against the SAME bytes is
//! `EdgeTAMGoldenFixtureTests`.

use std::path::{Path, PathBuf};

use ferment_engine_core::{
    NormalizedVideoBox, NormalizedVideoPoint, VideoObjectMask, VideoObjectPointLabel,
    VideoObjectPointPrompt, VideoObjectPrompt,
};
use half::f16;
use serde_json::Value;

use super::preprocessing::{self, EdgeTamRgbFrame, IMAGE_SIZE};
use super::processing;

fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/exports/rust_goldens/edgetam_v1")
}

fn golden(name: &str) -> Vec<u8> {
    let path = goldens_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing EdgeTAM golden {}: {error}. Run `uv run --project scripts/exports/edgetam \
             python scripts/exports/edgetam/gen_rust_goldens.py`.",
            path.display()
        )
    })
}

fn index() -> Value {
    serde_json::from_slice(&golden("INDEX.json")).expect("INDEX.json parses")
}

fn u16_values(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn i64_values(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut value = [0_u8; 8];
            value.copy_from_slice(chunk);
            i64::from_le_bytes(value)
        })
        .collect()
}

/// The video lane's normalization, graded on FP16 BIT PATTERNS over the whole
/// 256-value RGB domain.
///
/// kill: swap the green and blue entries of `IMAGE_STANDARD_DEVIATION` (0.224
/// and 0.225) and 512 of the 768 bit patterns move.
///
/// NOT a kill, measured: folding the subtract and the divide into one affine
/// constant. iOS's comment warns that a precomputed transform "can change the
/// final half-precision bit pattern", and for the IMAGE lane's `f32` pipeline
/// it does — but the video lane rounds a 256-entry LOOKUP once, and over that
/// whole domain both forms land on the same halves. The claim is true where
/// iOS makes it and false here; the two-operation form is kept because the
/// pinned graph has it, not because this test could tell.
#[test]
fn the_video_normalization_matches_every_python_fp16_byte() {
    let rgb_probe = golden("pre_rgb_probe_u8.bin");
    let expected_bits = u16_values(&golden("pre_normalized_probe_fp16.bin"));
    assert_eq!(rgb_probe.len(), 256 * 3);
    assert_eq!(expected_bits.len(), 3 * 256);

    let mut pixels = vec![0_u8; IMAGE_SIZE * IMAGE_SIZE * 3];
    pixels[..rgb_probe.len()].copy_from_slice(&rgb_probe);
    let normalized =
        preprocessing::normalized_image_from_square_model_frame(IMAGE_SIZE, IMAGE_SIZE, &pixels)
            .expect("square normalization");

    let pixels_per_channel = IMAGE_SIZE * IMAGE_SIZE;
    for channel in 0..3 {
        for probe_index in 0..256 {
            assert_eq!(
                normalized[channel * pixels_per_channel + probe_index].to_bits(),
                expected_bits[channel * 256 + probe_index],
                "FP16 preprocessing drift at channel {channel}, RGB probe {probe_index}"
            );
        }
    }
}

/// The image lane's torchvision resize + normalization, both stages, on a
/// 31×7 → 19×19 case whose axis contributions are all short.
///
/// kill: replace the `mul_add` accumulation in `weighted_sum` with
/// `sum += value * weight` and the `f32` stage stops being byte-exact.
#[test]
fn the_image_preprocessing_matches_the_torchvision_origin_byte_for_byte() {
    let index = index();
    let spec = &index["image_preprocessing"];
    let source_width = spec["source_width"].as_u64().unwrap() as usize;
    let source_height = spec["source_height"].as_u64().unwrap() as usize;
    let destination_width = spec["destination_width"].as_u64().unwrap() as usize;
    let destination_height = spec["destination_height"].as_u64().unwrap() as usize;

    let rgb = golden("pre_image_rgb_u8.bin");
    assert_eq!(rgb.len(), source_width * source_height * 3);
    let frame = EdgeTamRgbFrame::new(
        0,
        ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
        source_width,
        source_height,
        rgb,
    )
    .expect("source frame");

    let resized = super::image_preprocessing::image_predictor_resize_unit_rgb(
        &frame,
        destination_width,
        destination_height,
    )
    .expect("resize");
    let expected_resized = f32_values(&golden("pre_image_resized_unit_fp32.bin"));
    assert_eq!(resized.len(), expected_resized.len());
    for (index, (actual, expected)) in resized.iter().zip(&expected_resized).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "resize drift at planar index {index}"
        );
    }

    let normalized = super::image_preprocessing::image_predictor_normalized_fp16(
        &frame,
        destination_width,
        destination_height,
    )
    .expect("normalize");
    let expected_normalized = u16_values(&golden("pre_image_normalized_fp16.bin"));
    assert_eq!(normalized.len(), expected_normalized.len());
    for (index, (actual, expected)) in normalized.iter().zip(&expected_normalized).enumerate() {
        assert_eq!(
            actual.to_bits(),
            *expected,
            "normalization drift at planar index {index}"
        );
    }
}

/// The prompt token order and the model-space coordinates, graded on the
/// origin's own float bits and int labels.
///
/// kill: emit the correction points before the two box-corner tokens and the
/// label vector stops being `[2, 3, …]` — the order EdgeTAM's prompt encoder
/// requires.
#[test]
fn the_prompt_tensor_matches_the_python_float_and_label_bytes() {
    let index = index();
    let prompt = &index["prompt"];
    let box_points: Vec<Vec<f64>> =
        serde_json::from_value(prompt["normalized_box"].clone()).expect("normalized_box");
    let points: Vec<Vec<f64>> =
        serde_json::from_value(prompt["normalized_points"].clone()).expect("normalized_points");
    let labels: Vec<i64> =
        serde_json::from_value(prompt["point_labels"].clone()).expect("point_labels");
    assert_eq!(box_points.len(), 2);
    assert_eq!(points.len(), labels.len());

    let bounds = NormalizedVideoBox::from_extents(
        box_points[0][0] as f32,
        box_points[0][1] as f32,
        box_points[1][0] as f32,
        box_points[1][1] as f32,
    );
    let corrections: Vec<VideoObjectPointPrompt> = points
        .iter()
        .zip(&labels)
        .map(|(point, label)| {
            VideoObjectPointPrompt::new(
                NormalizedVideoPoint::new(point[0] as f32, point[1] as f32),
                if *label == 0 {
                    VideoObjectPointLabel::Background
                } else {
                    VideoObjectPointLabel::Foreground
                },
            )
        })
        .collect();

    let tensor = processing::prompt_tensor(&VideoObjectPrompt::BoxAndPoints {
        r#box: bounds,
        points: corrections,
    })
    .expect("prompt tensor");

    let expected_coordinates = f32_values(&golden("prompt_coordinates_fp32.bin"));
    let expected_labels = i32_values(&golden("prompt_labels_int32.bin"));
    assert_eq!(
        tensor
            .coordinates
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected_coordinates
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(tensor.labels, expected_labels);
}

/// The mask postprocessing chain — bilinear resize, threshold, COCO RLE —
/// graded at three separate boundaries against the origin.
///
/// kill: drop the `-0.5` in `axis_samples` (the `align_corners=false` shift)
/// and the ULP distance explodes; traverse the RLE row-major and the counts
/// change.
///
/// NOT a kill, measured: flipping the threshold from `>` to `>=`. The origin's
/// own manifest records `minimum_absolute_reference_logit` = 1.36e-3 — the
/// fixture was BUILT with no logit near zero so the interpolation comparison
/// would be well-conditioned, which is exactly what makes it unable to grade
/// the boundary. `the_threshold_boundary_is_pytorchs_own` covers that.
#[test]
fn the_mask_postprocessing_matches_the_origin_within_four_ulp_then_byte_for_byte() {
    let index = index();
    let post = &index["postprocessing"];
    let source_width = post["low_width"].as_u64().unwrap() as usize;
    let source_height = post["low_height"].as_u64().unwrap() as usize;
    let display_width = post["display_width"].as_u64().unwrap() as usize;
    let display_height = post["display_height"].as_u64().unwrap() as usize;
    assert_eq!((source_width, source_height), (256, 256));

    let low_logits = f32_values(&golden("post_low_logits_fp32.bin"));
    let expected_display = f32_values(&golden("post_display_logits_fp32.bin"));
    let display = processing::resize_bilinear_align_corners_false(
        &low_logits,
        source_width,
        source_height,
        display_width,
        display_height,
    )
    .expect("display resize");
    assert_eq!(display.len(), expected_display.len());

    let mut maximum_ulp: i64 = 0;
    let mut maximum_index = 0;
    for (index, (actual, expected)) in display.iter().zip(&expected_display).enumerate() {
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "interpolation sign drift at {index}"
        );
        let distance = (i64::from(actual.to_bits()) - i64::from(expected.to_bits())).abs();
        if distance > maximum_ulp {
            maximum_ulp = distance;
            maximum_index = index;
        }
    }
    assert!(
        maximum_ulp <= 4,
        "interpolation drift at {maximum_index}: {maximum_ulp} ULP"
    );

    let mask = processing::threshold(&display, display_width, display_height).expect("threshold");
    assert_eq!(
        mask.bytes(),
        golden("post_mask_u8.bin").as_slice(),
        "the final binary mask must be byte-identical to pinned PyTorch"
    );

    let fused = processing::thresholded_display_mask(&low_logits, display_width, display_height)
        .expect("fused threshold");
    assert_eq!(
        fused.bytes(),
        mask.bytes(),
        "the allocation-free production path must match the staged reference"
    );

    let expected_rle = i64_values(&golden("post_rle_counts_int64.bin"));
    assert_eq!(
        mask.uncompressed_coco_rle().counts,
        expected_rle,
        "COCO RLE must be byte-for-byte semantically identical"
    );
}

/// The RLE decoder is the inverse of the encoder the origin pinned — but the
/// bytes it must produce are the ORIGIN's mask, not a round trip of our own.
///
/// kill: decode the counts in row-major order and the reconstructed mask stops
/// matching `post_mask_u8`.
#[test]
fn decoding_the_origins_rle_reproduces_the_origins_mask() {
    let index = index();
    let post = &index["postprocessing"];
    let width = post["display_width"].as_u64().unwrap() as usize;
    let height = post["display_height"].as_u64().unwrap() as usize;

    let rle = ferment_engine_core::VideoObjectMaskRle {
        width,
        height,
        counts: i64_values(&golden("post_rle_counts_int64.bin")),
    };
    let mask = VideoObjectMask::from_uncompressed_coco_rle(&rle).expect("decode");
    assert_eq!(mask.bytes(), golden("post_mask_u8.bin").as_slice());
}

/// A box whose corners are out of ORDER is refused — the one thing the box
/// guard alone decides.
///
/// kill: delete the ordering half of `box_tensor`'s guard (`top_left.x <=
/// bottom_right.x && top_left.y <= bottom_right.y`) and the reversed boxes
/// below are accepted, so EdgeTAM receives corner tokens describing a negative
/// rectangle.
///
/// NOT a kill, measured: writing the guard as a flipped operator
/// (`if top_left.x > bottom_right.x`) instead of a negated whole predicate. A
/// NaN corner IS still refused — by `model_point`, which validates each corner
/// independently and runs after the guard. The Swift-`guard` class of bug is
/// therefore not observable through `prompt_tensor`; the non-finite cases below
/// pin that no such corner reaches the prompt encoder BY WHICHEVER GATE, which
/// is the property that matters, rather than grading the box guard.
#[test]
fn a_box_with_reversed_or_non_finite_corners_is_refused() {
    for (name, bounds) in [
        (
            "x reversed",
            NormalizedVideoBox::from_extents(0.9, 0.1, 0.1, 0.9),
        ),
        (
            "y reversed",
            NormalizedVideoBox::from_extents(0.1, 0.9, 0.9, 0.1),
        ),
        (
            "both reversed",
            NormalizedVideoBox::from_extents(0.9, 0.9, 0.1, 0.1),
        ),
    ] {
        assert!(
            processing::prompt_tensor(&VideoObjectPrompt::Box(bounds)).is_err(),
            "a box with {name} must be refused"
        );
    }
    // A degenerate but ORDERED box is legal — the guard is `<=`, not `<`.
    for bounds in [
        NormalizedVideoBox::from_extents(0.5, 0.5, 0.5, 0.5),
        NormalizedVideoBox::from_extents(0.1, 0.1, 0.9, 0.9),
    ] {
        assert!(processing::prompt_tensor(&VideoObjectPrompt::Box(bounds)).is_ok());
    }
    for corner in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            processing::prompt_tensor(&VideoObjectPrompt::Box(NormalizedVideoBox::from_extents(
                0.1, 0.1, corner, 0.9
            )))
            .is_err(),
            "a box with a {corner} corner must be refused"
        );
        assert!(
            processing::prompt_tensor(&VideoObjectPrompt::Points(vec![
                VideoObjectPointPrompt::new(
                    NormalizedVideoPoint::new(corner, 0.5),
                    VideoObjectPointLabel::Foreground,
                )
            ]))
            .is_err(),
            "a point at {corner} must be refused"
        );
    }
}

/// The temporal bank's slot arithmetic, graded from the iOS doc's own contract:
/// slot k of seven carries the frame at distance `7 - k`, and a conditioning
/// frame on the not-yet-traversed side contributes spatial memory but NO
/// pointer (`only_obj_ptrs_in_the_past_for_eval`).
///
/// kill: drop the `isOnTraversedSide` check and slot 0's pointer validity turns
/// on for a reverse pass whose conditioning frame is behind the cursor.
#[test]
fn the_memory_bank_fills_the_slots_the_upstream_loop_names() {
    use std::collections::BTreeMap;

    use ferment_engine_core::VideoObjectPropagationDirection::{Forward, Reverse};

    use super::temporal_state::{
        self, EdgeTamStoredFrame, LOW_RESOLUTION_MASK_COUNT, MEMORY_FEATURE_COUNT,
        OBJECT_POINTER_COUNT,
    };

    let frame = |index: usize, fill: f32| {
        EdgeTamStoredFrame::new(
            index,
            ferment_engine_media::MediaTime::new(
                index as i64 * 33_333,
                ferment_engine_media::TimeBase::MICROSECONDS,
            ),
            vec![f16::ZERO; LOW_RESOLUTION_MASK_COUNT],
            vec![f16::from_f32(fill); MEMORY_FEATURE_COUNT],
            vec![f16::from_f32(fill); OBJECT_POINTER_COUNT],
            0.5,
            1.0,
        )
        .expect("stored frame")
    };

    let conditioning = frame(0, 1.0);
    let mut non_conditioning = BTreeMap::new();
    for index in 1..=9 {
        non_conditioning.insert(index, frame(index, index as f32));
    }

    let bank = temporal_state::pack(10, Forward, &conditioning, &non_conditioning);
    // Slot 0 is the conditioning frame; slots 1..6 are distances 6..1, i.e.
    // frames 4..9. Frame 3 (distance 7) is out of the seven-slot window.
    assert_eq!(bank.spatial_validity[0], f16::ONE);
    for (slot, expected_frame) in (1..7).zip(4..10) {
        assert_eq!(
            bank.spatial_validity[slot],
            f16::ONE,
            "slot {slot} must carry frame {expected_frame}"
        );
        assert_eq!(
            bank.spatial_memory_features[slot * MEMORY_FEATURE_COUNT].to_f32(),
            expected_frame as f32
        );
    }
    // Forward: the conditioning frame is behind the cursor, so its pointer counts.
    assert_eq!(bank.pointer_validity[0], f16::ONE);

    // Reverse walks from high indices to low, so the traversed side is ABOVE
    // the cursor. A conditioning frame BELOW it has not been reached yet:
    // spatial memory is still present, the pointer is not.
    let reverse = temporal_state::pack(10, Reverse, &conditioning, &non_conditioning);
    assert_eq!(reverse.spatial_validity[0], f16::ONE);
    assert_eq!(
        reverse.pointer_validity[0],
        f16::ZERO,
        "a conditioning frame ahead of a reverse cursor contributes no pointer"
    );
    // …and one the reverse cursor HAS passed does contribute, so the check is
    // not simply off in reverse.
    let passed = temporal_state::pack(10, Reverse, &frame(20, 1.0), &non_conditioning);
    assert_eq!(passed.pointer_validity[0], f16::ONE);
}

/// `pack` and `pack_into` are the same packer — the session reuses one bank
/// across a whole propagation, and a stale slot left behind by the reuse path
/// would be memory from another frame.
///
/// kill: drop the `clear()` in `pack_into` and the second call keeps the first
/// frame's slots wherever the second has none.
#[test]
fn reusing_a_bank_leaves_nothing_of_the_previous_frame_behind() {
    use std::collections::BTreeMap;

    use ferment_engine_core::VideoObjectPropagationDirection::Forward;

    use super::temporal_state::{
        self, EdgeTamStoredFrame, LOW_RESOLUTION_MASK_COUNT, MEMORY_FEATURE_COUNT,
        OBJECT_POINTER_COUNT,
    };

    let frame = |index: usize, fill: f32| {
        EdgeTamStoredFrame::new(
            index,
            ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
            vec![f16::ZERO; LOW_RESOLUTION_MASK_COUNT],
            vec![f16::from_f32(fill); MEMORY_FEATURE_COUNT],
            vec![f16::from_f32(fill); OBJECT_POINTER_COUNT],
            0.5,
            1.0,
        )
        .expect("stored frame")
    };
    let conditioning = frame(0, 1.0);
    let mut crowded = BTreeMap::new();
    for index in 1..=9 {
        crowded.insert(index, frame(index, index as f32));
    }

    let mut bank = temporal_state::make_bank();
    temporal_state::pack_into(&mut bank, 10, Forward, &conditioning, &crowded);
    temporal_state::pack_into(&mut bank, 10, Forward, &conditioning, &BTreeMap::new());

    let fresh = temporal_state::pack(10, Forward, &conditioning, &BTreeMap::new());
    assert_eq!(bank, fresh, "a reused bank must equal a fresh one");
    for slot in 1..7 {
        assert_eq!(bank.spatial_validity[slot], f16::ZERO);
    }
}

/// EdgeTAM emits foreground where the selected raw logit is STRICTLY greater
/// than zero: `+inf` is foreground; zero, `-inf` and NaN are background. That
/// is PyTorch's comparison semantics, stated in the iOS doc comment, and the
/// origin fixture cannot grade it — its own manifest pins every reference logit
/// at least 1.36e-3 away from zero.
///
/// kill: flip the comparison to `>=` and the zero column turns foreground;
/// write it as `!(value <= 0.0)` and the NaN column does.
#[test]
fn the_threshold_boundary_is_pytorchs_own() {
    let logits = [
        f32::INFINITY,
        1.0,
        f32::MIN_POSITIVE,
        0.0,
        -0.0,
        -f32::MIN_POSITIVE,
        -1.0,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let expected = [255_u8, 255, 255, 0, 0, 0, 0, 0, 0];

    let mask = processing::threshold(&logits, logits.len(), 1).expect("threshold");
    assert_eq!(mask.bytes(), expected);

    // The production path fuses the resize with the threshold, so it has to
    // answer the same at the boundary. A 256×256 plane of one value each.
    for (value, expected) in logits.iter().copied().zip(expected) {
        let plane = vec![value; 256 * 256];
        let fused = processing::thresholded_display_mask(&plane, 4, 4).expect("fused");
        assert!(
            fused.bytes().iter().all(|byte| *byte == expected),
            "a plane of {value} must threshold to {expected}"
        );
    }
}
