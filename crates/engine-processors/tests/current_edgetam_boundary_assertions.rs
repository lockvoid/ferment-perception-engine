//! Portable current iOS EdgeTAM boundary assertions, independent of model files.
use ferment_engine_core::{
    NormalizedVideoBox, NormalizedVideoPoint, VideoObjectPointLabel, VideoObjectPointPrompt,
    VideoObjectPrompt,
};
use ferment_engine_processors::edgetam::{
    image_preprocessing as image, preprocessing as pre, processing as mask,
    prompt_inference as prompt,
};
use half::f16;

fn frame(width: usize, height: usize, pixels: Vec<u8>) -> pre::EdgeTamRgbFrame {
    pre::EdgeTamRgbFrame::new(
        0,
        ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
        width,
        height,
        pixels,
    )
    .unwrap()
}

fn point(x: f32, y: f32, label: VideoObjectPointLabel) -> VideoObjectPointPrompt {
    VideoObjectPointPrompt {
        point: NormalizedVideoPoint { x, y },
        label,
    }
}

#[test]
fn one_pixel_resize_is_constant_in_each_planar_channel() {
    let result =
        image::image_predictor_resize_unit_rgb(&frame(1, 1, vec![17, 101, 233]), 7, 5).unwrap();
    assert_eq!(&result[..35], &[17.0 / 255.0; 35]);
    assert_eq!(&result[35..70], &[101.0 / 255.0; 35]);
    assert_eq!(&result[70..], &[233.0 / 255.0; 35]);
}

#[test]
fn invalid_and_overflowing_destinations_and_short_rgb_are_refused() {
    let source = frame(1, 1, vec![0, 0, 0]);
    assert!(image::image_predictor_resize_unit_rgb(&source, 0, 1).is_err());
    assert!(image::image_predictor_resize_unit_rgb(&source, usize::MAX, 2).is_err());
    assert_eq!(
        pre::EdgeTamRgbFrame::new(
            0,
            ferment_engine_media::MediaTime::new(0, ferment_engine_media::TimeBase::MICROSECONDS),
            2,
            2,
            vec![0; 11]
        ),
        Err(pre::PreprocessingError::InvalidRgbByteCount {
            expected: 12,
            actual: 11
        })
    );
}

#[test]
fn asymmetric_rgb_matches_the_independent_pytorch_half_bits() {
    let size = pre::IMAGE_SIZE;
    let mut rgb = vec![0; size * size * 3];
    rgb[..12].copy_from_slice(&[0, 255, 17, 101, 233, 128, 64, 192, 255, 255, 0, 233]);
    let result = pre::normalized_image(&frame(size, size, rgb)).unwrap();
    assert_eq!(result.len(), 3 * 1024 * 1024);
    for (channel, expected) in [
        [0xc03c, 0xb636, 0xbc16, 0x407f],
        [0x40db, 0x4016, 0x3d4d, 0xc012],
        [0xbe08, 0x36d3, 0x4148, 0x4083],
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            result[channel * size * size..channel * size * size + 4]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn square_identity_preserves_bytes_and_non_square_sources_never_letterbox() {
    let size = pre::IMAGE_SIZE;
    let rgb = (0..size * size * 3)
        .map(|index| index.wrapping_mul(37) as u8)
        .collect::<Vec<_>>();
    assert_eq!(
        pre::square_model_frame(&frame(size, size, rgb.clone())).unwrap(),
        rgb
    );
    let result = pre::square_model_frame(&frame(3, 2, [19, 87, 203].repeat(6))).unwrap();
    assert_eq!(result.len(), 1024 * 1024 * 3);
    assert!(result.chunks_exact(3).all(|pixel| pixel == [19, 87, 203]));
}

#[test]
fn video_workspace_matches_static_path_and_retains_its_fixed_storage() {
    let mut workspace = pre::Workspace::new(3, 2).unwrap();
    let retained = workspace.retained_byte_count();
    assert!(retained > 0);
    for seed in [0, 17, 255] {
        let source = frame(
            3,
            2,
            (0..18).map(|index| (seed + index * 37) as u8).collect(),
        );
        let expected = pre::square_model_frame(&source).unwrap();
        assert_eq!(
            workspace.square_model_frame(&source).unwrap().pixels,
            expected
        );
        assert_eq!(workspace.retained_byte_count(), retained);
    }
    assert_eq!(
        workspace
            .square_model_frame(&frame(1, 1, vec![0; 3]))
            .unwrap_err(),
        pre::PreprocessingError::WorkspaceDimensionsChanged {
            expected_width: 3,
            expected_height: 2,
            actual_width: 1,
            actual_height: 1
        }
    );
}

#[test]
fn retained_normalized_output_survives_workspace_reuse() {
    let mut workspace = pre::Workspace::new(2, 1).unwrap();
    let first = workspace
        .normalized_image(&frame(2, 1, vec![0; 6]))
        .unwrap();
    let saved = first.clone();
    let second = workspace
        .normalized_image(&frame(2, 1, vec![255; 6]))
        .unwrap();
    assert_eq!(first, saved);
    assert_eq!(
        first,
        pre::normalized_image(&frame(2, 1, vec![0; 6])).unwrap()
    );
    assert_eq!(
        second,
        pre::normalized_image(&frame(2, 1, vec![255; 6])).unwrap()
    );
    assert_ne!(first, second);
}

#[test]
fn bilinear_downsample_and_upsample_match_independent_known_values() {
    assert_eq!(
        mask::resize_bilinear_align_corners_false(
            &(0..16).map(|x| x as f32).collect::<Vec<_>>(),
            4,
            4,
            2,
            2
        )
        .unwrap(),
        [2.5, 4.5, 10.5, 12.5]
    );
    assert_eq!(
        mask::resize_bilinear_align_corners_false(&[1.0, 2.0, 3.0, 4.0], 2, 2, 4, 4).unwrap(),
        [
            1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0
        ]
    );
}

#[test]
fn bilinear_identity_preserves_signed_zero_and_infinity_bits() {
    let input = [f32::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f32::INFINITY];
    let output = mask::resize_bilinear_align_corners_false(&input, 3, 2, 3, 2).unwrap();
    assert_eq!(
        output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        input.map(f32::to_bits)
    );
}

#[test]
fn bilinear_resize_stays_within_two_ulp_of_pinned_pytorch() {
    let result = mask::resize_bilinear_align_corners_false(
        &[0.125, -2.75, 7.0, 1.5, 0.25, -0.875],
        3,
        2,
        7,
        5,
    )
    .unwrap();
    let expected: [u32; 35] = [
        1040187392, 3197258021, 3217180965, 3224371200, 1068948336, 1085500856, 1088421888,
        1048995431, 3187671039, 3215248589, 3223112908, 1067555226, 1084175156, 1086770380,
        1062207488, 1057264201, 3199954361, 3214934016, 1058612372, 1075614282, 1078198272,
        1068394087, 1066701386, 1057893348, 3175926964, 3179761792, 3181918887, 3182637913,
        1069547520, 1068049555, 1061758098, 1048576000, 3194861276, 3208043375, 3210739712,
    ];
    assert_eq!(result.len(), expected.len());
    for (index, (actual, expected)) in result.iter().zip(expected).enumerate() {
        assert!(
            actual.to_bits().abs_diff(expected) <= 2,
            "drift at {index}: {} vs {expected}",
            actual.to_bits()
        );
    }
}

#[test]
fn malformed_bilinear_and_fused_masks_are_refused() {
    assert!(mask::resize_bilinear_align_corners_false(&[1.0, 2.0, 3.0], 2, 2, 4, 4).is_err());
    assert!(mask::resize_bilinear_align_corners_false(&[1.0], 1, 1, 0, 1).is_err());
    assert!(mask::thresholded_display_mask(&[0.0], 1, 1).is_err());
    assert!(mask::thresholded_display_mask(&vec![0.0; 256 * 256], 0, 1).is_err());
}

#[test]
fn fused_display_masks_match_the_staged_path_across_edge_shapes() {
    let mut logits = (0..256 * 256)
        .map(|i| (((i * 73) % 509) - 254) as f32 / 31.0)
        .collect::<Vec<_>>();
    logits[..6].copy_from_slice(&[
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::NAN,
    ]);
    for (width, height) in [
        (1, 1),
        (1, 257),
        (257, 1),
        (2, 2),
        (17, 19),
        (256, 256),
        (1024, 1024),
        (1920, 1080),
    ] {
        let staged = mask::threshold(
            &mask::display_logits(&logits, width, height).unwrap(),
            width,
            height,
        )
        .unwrap();
        assert_eq!(
            mask::thresholded_display_mask(&logits, width, height).unwrap(),
            staged,
            "{width}x{height}"
        );
    }
}

#[test]
fn two_stage_constant_logits_are_exact() {
    assert_eq!(
        mask::two_stage_display_logits(&vec![2.25; 256 * 256], 7, 5).unwrap(),
        vec![2.25; 35]
    );
}

#[test]
fn normalized_points_and_box_tokens_keep_model_space_order() {
    let points = VideoObjectPrompt::Points(vec![
        point(0.25, 0.75, VideoObjectPointLabel::Foreground),
        point(1.0, 0.0, VideoObjectPointLabel::Background),
    ]);
    let tensor = mask::prompt_tensor(&points).unwrap();
    assert_eq!(tensor.coordinates, [256.0, 768.0, 1024.0, 0.0]);
    assert_eq!(tensor.labels, [1, 0]);
    assert_eq!(tensor.point_count(), 2);
    let with_box = VideoObjectPrompt::BoxAndPoints {
        r#box: NormalizedVideoBox {
            top_left: NormalizedVideoPoint { x: 0.125, y: 0.25 },
            bottom_right: NormalizedVideoPoint { x: 0.875, y: 0.75 },
        },
        points: vec![point(0.5, 0.625, VideoObjectPointLabel::Background)],
    };
    let tensor = mask::prompt_tensor(&with_box).unwrap();
    assert_eq!(
        tensor.coordinates,
        [128.0, 256.0, 896.0, 768.0, 512.0, 640.0]
    );
    assert_eq!(tensor.labels, [2, 3, 0]);
    assert!(mask::prompt_tensor(&VideoObjectPrompt::Points(vec![])).is_err());
    for x in [-0.001, f32::NAN] {
        assert!(
            mask::prompt_tensor(&VideoObjectPrompt::Points(vec![point(
                x,
                0.5,
                VideoObjectPointLabel::Foreground
            )]))
            .is_err()
        );
    }
}

#[test]
fn fp16_workspace_matches_promoted_edges_and_retains_its_storage() {
    let mut values = (0..256 * 256)
        .map(|i| f16::from_f32((((i * 73) % 509) - 254) as f32 / 31.0))
        .collect::<Vec<_>>();
    values[..9].copy_from_slice(&[
        f16::NEG_INFINITY,
        f16::from_f32(-1.0),
        f16::from_bits(0x8001),
        f16::from_bits(0x8000),
        f16::ZERO,
        f16::from_bits(1),
        f16::ONE,
        f16::INFINITY,
        f16::NAN,
    ]);
    for (width, height) in [(1, 1), (17, 19), (257, 129)] {
        let promoted = values.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
        let expected = mask::thresholded_display_mask(&promoted, width, height).unwrap();
        let plan = mask::display_mask_plan(width, height).unwrap();
        assert_eq!(
            mask::thresholded_display_mask_with_plan(&promoted, &plan).unwrap(),
            expected
        );
        let mut workspace = prompt::MaskWorkspace::new(width, height).unwrap();
        assert_eq!(workspace.mask(&values).unwrap(), expected);
    }
    let mut workspace = prompt::MaskWorkspace::new(257, 129).unwrap();
    let retained = workspace.retained_byte_count();
    let first = workspace
        .mask(&vec![f16::from_f32(-1.0); 256 * 256])
        .unwrap();
    let saved = first.clone();
    let second = workspace.mask(&vec![f16::ONE; 256 * 256]).unwrap();
    assert_eq!(first, saved);
    assert!(first.bytes().iter().all(|b| *b == 0));
    assert!(second.bytes().iter().all(|b| *b == 255));
    assert_eq!(workspace.retained_byte_count(), retained);
    assert_eq!(
        workspace.mask(&[]),
        Err(mask::ProcessingError::InvalidElementCount {
            expected: 256 * 256,
            actual: 0
        })
    );
}
