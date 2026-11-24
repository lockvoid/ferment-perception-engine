use std::time::{Duration, Instant};

use super::*;

fn texture(width: usize, height: usize, seed: u64) -> LumaPlane {
    let mut state = seed;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8
    };
    let mut data = vec![0_u8; width * height];
    for value in &mut data {
        *value = next();
    }

    for _ in 0..2 {
        let mut output = data.clone();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0_u32;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let sample_y = y.wrapping_add_signed(dy);
                        let sample_x = x.wrapping_add_signed(dx);
                        sum += u32::from(data[sample_y * width + sample_x]);
                    }
                }
                output[y * width + x] = (sum / 9) as u8;
            }
        }
        data = output;
    }
    LumaPlane {
        width,
        height,
        bytes_per_row: width,
        data,
    }
}

fn shifted(plane: &LumaPlane, dx: isize, dy: isize) -> LumaPlane {
    let mut data = vec![0_u8; plane.width * plane.height];
    for y in 0..plane.height {
        let source_y = (y as isize - dy).clamp(0, plane.height as isize - 1) as usize;
        for x in 0..plane.width {
            let source_x = (x as isize - dx).clamp(0, plane.width as isize - 1) as usize;
            data[y * plane.width + x] = plane.data[source_y * plane.bytes_per_row + source_x];
        }
    }
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data,
    }
}

fn motion_blurred(plane: &LumaPlane, ux: f64, uy: f64, length: f64) -> LumaPlane {
    if length <= 0.0 {
        return plane.clone();
    }
    let taps = (length.ceil() as usize).max(3) | 1;
    let mut data = vec![0_u8; plane.width * plane.height];
    for y in 0..plane.height {
        for x in 0..plane.width {
            let mut sum = 0.0;
            for tap in 0..taps {
                let offset = (tap as f64 / (taps - 1) as f64 - 0.5) * length;
                sum += synthetic_bilinear(plane, x as f64 + ux * offset, y as f64 + uy * offset);
            }
            data[y * plane.width + x] = (sum / taps as f64).round().clamp(0.0, 255.0) as u8;
        }
    }
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data,
    }
}

fn synthetic_bilinear(plane: &LumaPlane, x: f64, y: f64) -> f64 {
    let clamped_x = x.clamp(0.0, (plane.width - 1) as f64);
    let clamped_y = y.clamp(0.0, (plane.height - 1) as f64);
    let x0 = clamped_x as usize;
    let y0 = clamped_y as usize;
    let x1 = (x0 + 1).min(plane.width - 1);
    let y1 = (y0 + 1).min(plane.height - 1);
    let fraction_x = clamped_x - x0 as f64;
    let fraction_y = clamped_y - y0 as f64;
    let p00 = f64::from(plane.data[y0 * plane.bytes_per_row + x0]);
    let p10 = f64::from(plane.data[y0 * plane.bytes_per_row + x1]);
    let p01 = f64::from(plane.data[y1 * plane.bytes_per_row + x0]);
    let p11 = f64::from(plane.data[y1 * plane.bytes_per_row + x1]);
    p00 * (1.0 - fraction_x) * (1.0 - fraction_y)
        + p10 * fraction_x * (1.0 - fraction_y)
        + p01 * (1.0 - fraction_x) * fraction_y
        + p11 * fraction_x * fraction_y
}

fn median_vector(field: &MotionField) -> (f32, f32) {
    let mut x = field
        .vectors
        .iter()
        .map(|vector| vector.dx)
        .collect::<Vec<_>>();
    let mut y = field
        .vectors
        .iter()
        .map(|vector| vector.dy)
        .collect::<Vec<_>>();
    x.sort_by(f32::total_cmp);
    y.sort_by(f32::total_cmp);
    (x[x.len() / 2], y[y.len() / 2])
}

fn analysis(shutter: Option<f64>, median: f64) -> SlowMotionAnalysis {
    SlowMotionAnalysis {
        source_shutter_angle: shutter,
        median_displacement: median,
        p95_displacement: 0.08,
        pairs_sampled: 12,
    }
}

fn pan(displacement: isize, trail: f64, seed: u64) -> (LumaPlane, MotionField) {
    let sharp = texture(256, 144, seed);
    let first = motion_blurred(&sharp, 1.0, 0.0, trail);
    let second = shifted(&first, displacement, 0);
    let field = BlockMotionEstimator::estimate(&first, &second);
    (first, field)
}

fn pan_pairs(
    displacement: isize,
    trail: f64,
    seed: u64,
    count: usize,
) -> Vec<(LumaPlane, LumaPlane)> {
    let sharp = texture(256, 144, seed);
    let first = motion_blurred(&sharp, 1.0, 0.0, trail);
    let second = shifted(&first, displacement, 0);
    (0..count)
        .map(|_| (first.clone(), second.clone()))
        .collect()
}

// BlockMotionEstimatorTests.swift

#[test]
fn recovers_global_shift() {
    let first = texture(256, 144, 7);
    let second = shifted(&first, 5, -3);
    let field = BlockMotionEstimator::estimate(&first, &second);
    assert!(field.vectors.len() > 20);
    let median = median_vector(&field);
    assert!((median.0 - 5.0).abs() <= 1.0);
    assert!((median.1 - -3.0).abs() <= 1.0);
}

#[test]
fn recovers_shift_near_search_radius() {
    let first = texture(256, 144, 11);
    let second = shifted(&first, -10, 0);
    let median = median_vector(&BlockMotionEstimator::estimate(&first, &second));
    assert!((median.0 - -10.0).abs() <= 1.0);
    assert!(median.1.abs() <= 1.0);
}

#[test]
fn flat_frame_yields_no_vectors() {
    let flat = LumaPlane {
        width: 256,
        height: 144,
        bytes_per_row: 256,
        data: vec![128; 256 * 144],
    };
    assert!(
        BlockMotionEstimator::estimate(&flat, &flat)
            .vectors
            .is_empty()
    );
}

#[test]
fn flat_region_is_skipped_textured_region_matched() {
    let mut plane = texture(256, 144, 3);
    for y in 0..plane.height {
        for x in 0..128 {
            plane.data[y * plane.width + x] = 100;
        }
    }
    let moved = shifted(&plane, 4, 2);
    let field = BlockMotionEstimator::estimate(&plane, &moved);
    assert!(!field.vectors.is_empty());
    assert!(field.vectors.iter().all(|vector| vector.cx >= 120));
}

#[test]
fn static_scene_yields_zero_vectors() {
    let plane = texture(256, 144, 5);
    let median = median_vector(&BlockMotionEstimator::estimate(&plane, &plane));
    assert!(median.0.abs() <= 0.5);
    assert!(median.1.abs() <= 0.5);
}

// MotionBlurPolicyTests.swift

#[test]
fn output_smear_angle_multiplies_by_factor() {
    assert_eq!(MotionBlurPolicy::output_smear_angle(180.0, 2), 360.0);
    assert_eq!(MotionBlurPolicy::output_smear_angle(10.0, 4), 40.0);
    assert_eq!(MotionBlurPolicy::output_smear_angle(180.0, 8), 1440.0);
}

#[test]
fn strength_fills_toward_capped_target() {
    assert_eq!(MotionBlurPolicy::strength(120.0, 40.0, 4), 6);
    assert_eq!(MotionBlurPolicy::strength(120.0, 10.0, 2), 15);
    assert_eq!(MotionBlurPolicy::strength(120.0, 119.0, 8), 1);
}

#[test]
fn fill_target_is_capped_pre_calibration() {
    let source = analysis(Some(10.0), 0.05);
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 2, 180.0, true);
    assert!(decision.applied);
    assert_eq!(decision.strength, 14);
    let softer = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 2, 90.0, true);
    assert_eq!(softer.strength, 10);
}

#[test]
fn normal_shutter_footage_stays_crisp() {
    let source = analysis(Some(180.0), 0.05);
    for factor in [2, 4, 8] {
        let decision =
            MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), factor, 180.0, true);
        assert!(!decision.applied);
        assert_eq!(decision.reason, MotionBlurReason::AlreadyBlurred);
        assert_eq!(decision.output_smear_angle, 180.0 * f64::from(factor));
    }
}

#[test]
fn fast_shutter_source_gets_blur() {
    let source = analysis(Some(10.0), 0.05);
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 4, 180.0, true);
    assert!(decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::FastShutterSource);
    assert_eq!(decision.source_shutter_angle, 10.0);
    assert_eq!(decision.output_smear_angle, 40.0);
    assert!(decision.strength >= 1);
}

#[test]
fn factor_itself_can_resolve_the_strobe() {
    let source = analysis(Some(40.0), 0.05);
    let at_two = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 2, 180.0, true);
    assert!(at_two.applied);
    let at_four = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 4, 180.0, true);
    assert!(!at_four.applied);
    assert_eq!(at_four.reason, MotionBlurReason::AlreadyBlurred);
}

#[test]
fn between_thresholds_biases_to_crisp() {
    let source = analysis(Some(50.0), 0.05);
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 2, 180.0, true);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::CrispPreferred);
}

#[test]
fn target_below_current_smear_stays_off() {
    let source = analysis(Some(20.0), 0.05);
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 4, 60.0, true);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::CrispPreferred);
}

#[test]
fn tiny_residual_motion_cannot_strobe() {
    let source = analysis(Some(10.0), 0.02);
    let still = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 8, 180.0, true);
    assert!(!still.applied);
    assert_eq!(still.reason, MotionBlurReason::MotionTooSmall);
    let moving = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 4, 180.0, true);
    assert!(moving.applied);
}

#[test]
fn explicit_modes_bypass_analysis() {
    let off = MotionBlurPolicy::decide(MotionBlurMode::Off, None, 2, 180.0, true);
    assert!(!off.applied);
    assert_eq!(off.reason, MotionBlurReason::ExplicitOff);
    let on = MotionBlurPolicy::decide(MotionBlurMode::On { strength: 42 }, None, 2, 180.0, true);
    assert!(on.applied);
    assert_eq!(on.strength, 42);
    assert_eq!(on.reason, MotionBlurReason::ExplicitOn);
    let clamped =
        MotionBlurPolicy::decide(MotionBlurMode::On { strength: 400 }, None, 2, 180.0, true);
    assert_eq!(clamped.strength, 100);
}

#[test]
fn auto_without_blur_processor_reports_unsupported() {
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, None, 2, 180.0, false);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::Unsupported);
}

#[test]
fn explicit_on_without_blur_processor_reports_unsupported() {
    let decision =
        MotionBlurPolicy::decide(MotionBlurMode::On { strength: 42 }, None, 2, 180.0, false);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::Unsupported);
}

#[test]
fn inconclusive_analysis_assumes_180() {
    let source = analysis(None, 0.05);
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, Some(&source), 2, 180.0, true);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::Indeterminate);
    assert_eq!(decision.source_shutter_angle, 180.0);
    assert_eq!(decision.output_smear_angle, 360.0);
}

#[test]
fn missing_analysis_is_indeterminate() {
    let decision = MotionBlurPolicy::decide(MotionBlurMode::Auto, None, 2, 180.0, true);
    assert!(!decision.applied);
    assert_eq!(decision.reason, MotionBlurReason::Indeterminate);
}

// ShutterAngleEstimatorTests.swift

#[test]
fn crisp_pan_reads_fast_shutter() {
    let (plane, field) = pan(16, 0.0, 21);
    let angle = ShutterAngleEstimator::estimate(&plane, &field).unwrap();
    assert!(angle < 45.0, "angle: {angle}");
}

#[test]
fn half_interval_trail_reads_normal() {
    let (plane, field) = pan(16, 8.0, 21);
    let angle = ShutterAngleEstimator::estimate(&plane, &field).unwrap();
    assert!(angle > 120.0, "angle: {angle}");
    assert!(angle < 300.0, "angle: {angle}");
}

#[test]
fn full_interval_trail_reads_wide() {
    let (plane, field) = pan(16, 16.0, 21);
    let angle = ShutterAngleEstimator::estimate(&plane, &field).unwrap();
    assert!(angle > 150.0, "angle: {angle}");
}

#[test]
fn estimate_is_monotone_in_trail() {
    let (short_plane, short_field) = pan(16, 2.0, 21);
    let (medium_plane, medium_field) = pan(16, 6.0, 21);
    let (long_plane, long_field) = pan(16, 12.0, 21);
    let short = ShutterAngleEstimator::estimate(&short_plane, &short_field).unwrap();
    let medium = ShutterAngleEstimator::estimate(&medium_plane, &medium_field).unwrap();
    let long = ShutterAngleEstimator::estimate(&long_plane, &long_field).unwrap();
    assert!(short < medium, "{short} !< {medium}");
    assert!(medium < long, "{medium} !< {long}");
}

#[test]
fn buckets_hold_at_smaller_displacement() {
    let (crisp_plane, crisp_field) = pan(8, 0.0, 33);
    let crisp = ShutterAngleEstimator::estimate(&crisp_plane, &crisp_field).unwrap();
    assert!(crisp < 45.0, "angle: {crisp}");
    let (normal_plane, normal_field) = pan(8, 4.0, 33);
    let normal = ShutterAngleEstimator::estimate(&normal_plane, &normal_field).unwrap();
    assert!(normal > 100.0, "angle: {normal}");
}

#[test]
fn static_field_returns_none() {
    let plane = texture(256, 144, 9);
    let field = BlockMotionEstimator::estimate(&plane, &plane);
    assert!(ShutterAngleEstimator::estimate(&plane, &field).is_none());
}

#[test]
fn aggregate_takes_median_of_valid_pairs() {
    assert_eq!(
        ShutterAngleEstimator::aggregate(&[Some(90.0), Some(110.0), Some(100.0), None]),
        Some(100.0)
    );
}

#[test]
fn aggregate_needs_three_valid_pairs() {
    assert!(ShutterAngleEstimator::aggregate(&[Some(100.0), None, Some(120.0), None]).is_none());
    assert!(ShutterAngleEstimator::aggregate(&[]).is_none());
}

// SlowMotionAnalyzerTests.swift — codec I/O is replaced by decoded luma pairs.

#[test]
fn crisp_pan_reads_fast_shutter_and_displacement() {
    let pairs = pan_pairs(6, 0.0, 51, 12);
    let result = SlowMotionAnalyzer::analyze_pairs(&pairs);
    assert!(result.source_shutter_angle.unwrap() < 45.0);
    assert!((result.median_displacement - 0.044).abs() <= 0.02);
    assert!(result.pairs_sampled >= 6);
}

#[test]
fn normal_shutter_pan_reads_wide() {
    let pairs = pan_pairs(8, 4.0, 52, 12);
    let result = SlowMotionAnalyzer::analyze_pairs(&pairs);
    assert!(result.source_shutter_angle.unwrap() > 100.0);
}

#[test]
fn static_clip_is_inconclusive() {
    let plane = texture(256, 144, 53);
    let pairs = (0..12)
        .map(|_| (plane.clone(), plane.clone()))
        .collect::<Vec<_>>();
    let result = SlowMotionAnalyzer::analyze_pairs(&pairs);
    assert!(result.source_shutter_angle.is_none());
    assert!(result.median_displacement < 0.005);
}

#[test]
fn analyzes_only_the_requested_window() {
    let times = SlowMotionAnalyzer::sample_times(0.4, 0.8, 30.0);
    assert!(times.len() >= 4);
    assert!(times.iter().all(|time| *time >= 0.4 && *time < 1.2));
    assert!(times.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn pre_pass_stays_inside_its_time_budget() {
    let pairs = pan_pairs(6, 0.0, 54, 12);
    let start = Instant::now();
    let _ = SlowMotionAnalyzer::analyze_pairs(&pairs);
    assert!(start.elapsed() < Duration::from_secs(3));
}
