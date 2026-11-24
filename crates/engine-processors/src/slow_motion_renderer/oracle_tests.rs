use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MockFrame {
    source_index: usize,
    blurred: bool,
}

#[derive(Debug, Clone)]
struct InterpolationCall {
    from_seconds: f64,
    phases: Vec<f32>,
    starts_new_sequence: bool,
}

#[derive(Default)]
struct SpyInterpolator {
    calls: Vec<InterpolationCall>,
}

impl FrameInterpolator<MockFrame> for SpyInterpolator {
    fn interpolate(
        &mut self,
        from: &TimedFrame<MockFrame>,
        _to: &TimedFrame<MockFrame>,
        phases: &[f32],
        starts_new_sequence: bool,
    ) -> Result<Vec<MockFrame>, String> {
        self.calls.push(InterpolationCall {
            from_seconds: from.pts.seconds(),
            phases: phases.to_vec(),
            starts_new_sequence,
        });
        Ok(vec![from.frame.clone(); phases.len()])
    }
}

impl SpyInterpolator {
    fn recorded_from_seconds(&self) -> Vec<f64> {
        self.calls.iter().map(|call| call.from_seconds).collect()
    }

    fn sequence_restarts(&self) -> Vec<f64> {
        self.calls
            .iter()
            .filter(|call| call.starts_new_sequence)
            .map(|call| call.from_seconds)
            .collect()
    }

    fn phases(&self, from_seconds: f64) -> Option<&[f32]> {
        self.calls
            .iter()
            .find(|call| (call.from_seconds - from_seconds).abs() < 0.001)
            .map(|call| call.phases.as_slice())
    }
}

#[derive(Debug, Clone)]
struct BlurCall {
    pts_seconds: f64,
    has_previous: bool,
    has_next: bool,
    strength: i32,
}

#[derive(Default)]
struct SpyBlurrer {
    calls: Vec<BlurCall>,
}

impl FrameBlurrer<MockFrame> for SpyBlurrer {
    fn blur(
        &mut self,
        frame: &TimedFrame<MockFrame>,
        previous: Option<&TimedFrame<MockFrame>>,
        next: Option<&TimedFrame<MockFrame>>,
        strength: i32,
    ) -> Result<MockFrame, String> {
        self.calls.push(BlurCall {
            pts_seconds: frame.pts.seconds(),
            has_previous: previous.is_some(),
            has_next: next.is_some(),
            strength,
        });
        Ok(MockFrame {
            source_index: frame.frame.source_index,
            blurred: true,
        })
    }
}

#[derive(Default)]
struct MockOutput {
    frames: Vec<TimedFrame<MockFrame>>,
    partial_active: bool,
    delivered: bool,
    committed: bool,
    aborted: bool,
}

impl FrameOutput<MockFrame> for MockOutput {
    fn begin(&mut self) -> Result<(), String> {
        self.partial_active = true;
        Ok(())
    }

    fn append(&mut self, frame: MockFrame, pts: MediaTime) -> Result<(), String> {
        self.frames.push(TimedFrame { frame, pts });
        Ok(())
    }

    fn commit(&mut self) -> Result<(), String> {
        self.partial_active = false;
        self.delivered = true;
        self.committed = true;
        Ok(())
    }

    fn abort(&mut self) {
        self.frames.clear();
        self.partial_active = false;
        self.delivered = false;
        self.aborted = true;
    }
}

fn source_frames(first: usize, count: usize) -> Vec<TimedFrame<MockFrame>> {
    (first..first + count)
        .map(|index| TimedFrame {
            frame: MockFrame {
                source_index: index,
                blurred: false,
            },
            pts: MediaTime::new(index as i64, 30),
        })
        .collect()
}

fn config(factor: u8, frame_count: usize) -> RenderConfig {
    let mut config = RenderConfig::new(factor, MediaTime::new(frame_count as i64, 30));
    config.motion_blur = MotionBlurMode::Off;
    config
}

fn render(
    source: &[TimedFrame<MockFrame>],
    config: &RenderConfig,
    interpolator: &mut SpyInterpolator,
    blurrer: Option<&mut SpyBlurrer>,
    output: &mut MockOutput,
) -> Result<SlowMotionRenderResult, SlowMotionRenderError> {
    render_frames(
        source,
        config,
        interpolator,
        blurrer.map(|value| value as &mut dyn FrameBlurrer<MockFrame>),
        output,
        || false,
        |_| {},
    )
}

fn contains_time(values: &[f64], expected: f64) -> bool {
    values.iter().any(|value| (*value - expected).abs() < 0.001)
}

// SlowMotionRendererBlurTests.swift

#[test]
fn test_explicit_on_blurs_every_anchor_with_neighbor_contract() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.motion_blur = MotionBlurMode::On { strength: 42 };
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();

    let result = render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert!(result.motion_blur.applied);
    assert_eq!(result.motion_blur.reason, MotionBlurReason::ExplicitOn);
    assert_eq!(result.motion_blur.strength, 42);
    assert_eq!(blurrer.calls.len(), 24);
    assert!(
        blurrer
            .calls
            .windows(2)
            .all(|calls| calls[0].pts_seconds <= calls[1].pts_seconds)
    );
    assert!(blurrer.calls.iter().all(|call| call.strength == 42));
    assert!(!blurrer.calls.first().unwrap().has_previous);
    assert!(blurrer.calls.first().unwrap().has_next);
    assert!(blurrer.calls.last().unwrap().has_previous);
    assert!(!blurrer.calls.last().unwrap().has_next);
    assert!(
        blurrer.calls[1..23]
            .iter()
            .all(|call| call.has_previous && call.has_next)
    );
    assert_eq!(result.frames_written, 2 * 24 - 1);
    assert!(
        output.frames.iter().all(|frame| frame.frame.blurred),
        "the blurred buffers, not the sharp source buffers, must reach output"
    );
}

#[test]
fn test_off_never_touches_the_blur_seam() {
    let source = source_frames(0, 24);
    let config = config(2, 24);
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();
    let result = render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert!(blurrer.calls.is_empty());
    assert!(!result.motion_blur.applied);
    assert_eq!(result.motion_blur.reason, MotionBlurReason::ExplicitOff);
    assert_eq!(result.frames_written, 2 * 24 - 1);
}

#[test]
fn test_single_frame_window_blurs_solo_frame() {
    let source = source_frames(0, 1);
    let mut config = config(2, 1);
    config.motion_blur = MotionBlurMode::On { strength: 30 };
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();
    let result = render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert_eq!(blurrer.calls.len(), 1);
    assert!(!blurrer.calls[0].has_previous);
    assert!(!blurrer.calls[0].has_next);
    assert_eq!(result.frames_written, 1);
}

#[test]
fn test_format_compatible_matches_against_blurrer_formats() {
    let matching = [PIXEL_FORMAT_420V, PIXEL_FORMAT_X420];
    let disjoint = [PIXEL_FORMAT_X420];
    assert!(format_compatible(Some(PIXEL_FORMAT_420V), Some(&matching)));
    assert!(!format_compatible(Some(PIXEL_FORMAT_420V), Some(&disjoint)));
    assert!(
        format_compatible(Some(PIXEL_FORMAT_420V), None),
        "nil attributes = accepts anything"
    );
    assert!(
        format_compatible(None, Some(&disjoint)),
        "no chosen format = nothing to contradict"
    );
}

#[test]
fn test_auto_applies_on_crisp_pan() {
    let source = source_frames(0, 40);
    let mut config = config(2, 40);
    config.motion_blur = MotionBlurMode::Auto;
    config.analysis = Some(SlowMotionAnalysis {
        source_shutter_angle: Some(10.0),
        median_displacement: 0.05,
        p95_displacement: 0.08,
        pairs_sampled: 12,
    });
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();
    let result = render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert!(result.motion_blur.applied);
    assert_eq!(
        result.motion_blur.reason,
        MotionBlurReason::FastShutterSource
    );
    assert!(result.motion_blur.source_shutter_angle < 45.0);
    assert!(result.motion_blur.strength >= 1);
    assert!(!blurrer.calls.is_empty());
    assert!(
        blurrer
            .calls
            .iter()
            .all(|call| call.strength == result.motion_blur.strength)
    );
}

#[test]
fn test_auto_stays_crisp_on_normal_shutter_pan() {
    let source = source_frames(0, 40);
    let mut config = config(2, 40);
    config.motion_blur = MotionBlurMode::Auto;
    config.analysis = Some(SlowMotionAnalysis {
        source_shutter_angle: Some(180.0),
        median_displacement: 0.05,
        p95_displacement: 0.08,
        pairs_sampled: 12,
    });
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();
    let result = render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert!(!result.motion_blur.applied);
    assert!(blurrer.calls.is_empty());
    assert_ne!(result.motion_blur.reason, MotionBlurReason::Unsupported);
}

// SlowMotionRendererSceneCutTests.swift

#[test]
fn test_pair_crosses_cut_boundaries() {
    let f11 = MediaTime::new(11, 30);
    let f12 = MediaTime::new(12, 30);
    let f13 = MediaTime::new(13, 30);
    let cut = MediaTime::new(12, 30);
    assert!(pair_crosses_cut(f11, f12, &[cut]));
    assert!(!pair_crosses_cut(f12, f13, &[cut]));
    assert!(pair_crosses_cut(f11, f13, &[MediaTime::new(246, 600)]));
    assert!(!pair_crosses_cut(f11, f12, &[]));
    assert!(!pair_crosses_cut(f11, f12, &[MediaTime::new(20, 30)]));
}

#[test]
fn test_scene_pair_is_held_not_interpolated() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 2 * 24 - 1);
    let from_seconds = interpolator.recorded_from_seconds();
    assert_eq!(from_seconds.len(), 22);
    assert!(!contains_time(&from_seconds, 11.0 / 30.0));
    assert!(contains_time(&from_seconds, 10.0 / 30.0));
    assert!(contains_time(&from_seconds, 12.0 / 30.0));
    let mut restarts = interpolator.sequence_restarts();
    restarts.sort_by(f64::total_cmp);
    assert_eq!(restarts.len(), 2, "restarts: {restarts:?}");
    assert!((restarts[0] - 0.0).abs() < 0.001);
    assert!((restarts[1] - 12.0 / 30.0).abs() < 0.001);
}

#[test]
fn test_incoming_pair_stretches_across_the_cut_slots() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 2 * 24 - 1);
    let stretched = interpolator.phases(12.0 / 30.0).unwrap();
    assert_eq!(stretched.len(), 2);
    for (got, want) in stretched.iter().zip([1.0 / 3.0, 2.0 / 3.0]) {
        assert!((*got - want).abs() < 1e-4);
    }
    assert_eq!(interpolator.phases(6.0 / 30.0).unwrap(), &[0.5]);
}

#[test]
fn test_cut_is_immediate_not_frozen() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(output.frames[22].frame.source_index, 11);
    for slot in 23..=25 {
        assert_ne!(
            output.frames[slot].frame.source_index, 11,
            "slot {slot} still shows the outgoing scene"
        );
        assert_eq!(output.frames[slot].frame.source_index, 12);
    }
}

#[test]
fn test_cut_on_final_pair_fills_with_incoming_frame() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(23, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 2 * 24 - 1);
    assert!(!contains_time(
        &interpolator.recorded_from_seconds(),
        22.0 / 30.0
    ));
    assert_eq!(output.frames.last().unwrap().frame.source_index, 23);
}

#[test]
fn test_single_frame_scene_falls_back_to_copies_then_stretches() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30), MediaTime::new(13, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 2 * 24 - 1);
    let from_seconds = interpolator.recorded_from_seconds();
    assert!(!contains_time(&from_seconds, 11.0 / 30.0));
    assert!(!contains_time(&from_seconds, 12.0 / 30.0));
    assert_eq!(interpolator.phases(13.0 / 30.0).unwrap().len(), 2);
    assert!(contains_time(
        &interpolator.sequence_restarts(),
        13.0 / 30.0
    ));
}

#[test]
fn test_window_rebases_scene_boundaries() {
    let source = source_frames(6, 18);
    let mut config = config(2, 18);
    config.window_start = MediaTime::new(6, 30);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 2 * 18 - 1);
    let from_seconds = interpolator.recorded_from_seconds();
    assert_eq!(from_seconds.len(), 16);
    assert!(!contains_time(&from_seconds, 5.0 / 30.0));
}

#[test]
fn test_deficit_spreads_across_incoming_pairs_at_factor4() {
    let source = source_frames(0, 24);
    let mut config = config(4, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let result = render(&source, &config, &mut interpolator, None, &mut output).unwrap();

    assert_eq!(result.frames_written, 4 * 24 - 3);
    for frame in [12, 13, 14] {
        let phases = interpolator.phases(f64::from(frame) / 30.0).unwrap();
        assert_eq!(phases.len(), 4, "pair {frame} should absorb one tick");
        assert!((phases[0] - 0.2).abs() < 1e-4);
    }
    assert_eq!(interpolator.phases(15.0 / 30.0).unwrap().len(), 3);
    assert_eq!(interpolator.sequence_restarts().len(), 2);
}

#[test]
fn test_blur_neighbor_contract_treats_cut_as_clip_edge() {
    let source = source_frames(0, 24);
    let mut config = config(2, 24);
    config.scene_cuts = vec![MediaTime::new(12, 30)];
    config.motion_blur = MotionBlurMode::On { strength: 30 };
    let mut interpolator = SpyInterpolator::default();
    let mut blurrer = SpyBlurrer::default();
    let mut output = MockOutput::default();
    render(
        &source,
        &config,
        &mut interpolator,
        Some(&mut blurrer),
        &mut output,
    )
    .unwrap();

    assert_eq!(blurrer.calls.len(), 24);
    let call = |frame: usize| {
        blurrer
            .calls
            .iter()
            .find(|call| (call.pts_seconds - frame as f64 / 30.0).abs() < 0.001)
            .unwrap()
    };
    let before_cut = call(11);
    assert!(before_cut.has_previous);
    assert!(!before_cut.has_next);
    let after_cut = call(12);
    assert!(!after_cut.has_previous);
    assert!(after_cut.has_next);
    let interior = call(6);
    assert!(interior.has_previous);
    assert!(interior.has_next);
}

// SlowMotionRendererTests.swift

#[test]
fn test_interpolation_phases() {
    assert_eq!(interpolation_phases(2), [0.5]);
    assert_eq!(interpolation_phases(4), [0.25, 0.5, 0.75]);
}

#[test]
fn test_output_pts_scales_by_factor_preserving_timescale() {
    let source = MediaTime::new(300, 600);
    let output = output_pts(source, 2);
    assert!((output.seconds() - 1.0).abs() < 0.0001);
    assert_eq!(output.timescale, 600);
}

#[test]
fn test_reader_attributes_picks_single_format_from_vt_style_array() {
    let formats = [PIXEL_FORMAT_420V, PIXEL_FORMAT_X420];
    assert_eq!(
        reader_pixel_format(Some(&formats), false),
        PIXEL_FORMAT_420V
    );
    assert_eq!(reader_pixel_format(Some(&formats), true), PIXEL_FORMAT_X420);
    assert_eq!(reader_pixel_format(None, false), PIXEL_FORMAT_420V);
}

#[test]
fn test_hdr_transfer_detection() {
    assert!(is_hdr_transfer(Some(TRANSFER_HLG)));
    assert!(is_hdr_transfer(Some(TRANSFER_PQ)));
    assert!(!is_hdr_transfer(Some("ITU_R_709_2")));
    assert!(!is_hdr_transfer(None));
}

#[test]
fn test_invalid_factor_throws() {
    let source = source_frames(0, 2);
    let config = config(1, 2);
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let error = render(&source, &config, &mut interpolator, None, &mut output).unwrap_err();
    assert_eq!(error, SlowMotionRenderError::InvalidFactor(1));
    assert!(!output.partial_active);
}

#[test]
fn test_render_timeline_doubles_duration_and_preserves_frame_interval() {
    let source = source_frames(0, 16);
    let mut config = config(2, 16);
    config.window_duration = MediaTime::new(15, 30);
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let mut progress = Vec::new();
    let result = render_frames(
        &source,
        &config,
        &mut interpolator,
        None,
        &mut output,
        || false,
        |value| progress.push(value),
    )
    .unwrap();

    assert!(output.delivered);
    assert!(!output.partial_active);
    assert!((result.output_pts.seconds() - 1.0).abs() < 0.0001);
    assert!(
        output.frames.windows(2).all(|pair| (pair[1].pts.seconds()
            - pair[0].pts.seconds()
            - 1.0 / 30.0)
            .abs()
            < 0.0001),
        "source fps must be preserved"
    );
    assert_eq!(result.frames_written % 2, 1);
    assert!(result.frames_written > 10);
    assert_eq!(progress.last(), Some(&1.0));
    assert!(progress.windows(2).all(|values| values[0] <= values[1]));
}

#[test]
#[ignore = "blocked: requires VideoIO decode/encode and inspection of a playable muxed output"]
fn test_render_doubles_duration_and_preserves_frame_interval() {
    require_render_capabilities(&[
        RenderCapability::VideoDecode,
        RenderCapability::VideoEncode,
        RenderCapability::PlayableMuxedOutput,
    ])
    .unwrap();
}

#[test]
fn test_cancellation_removes_partial_and_throws() {
    let source = source_frames(0, 24);
    let config = config(4, 24);
    let mut interpolator = SpyInterpolator::default();
    let mut output = MockOutput::default();
    let mut checks = 0;
    let error = render_frames(
        &source,
        &config,
        &mut interpolator,
        None,
        &mut output,
        || {
            checks += 1;
            checks > 5
        },
        |_| {},
    )
    .unwrap_err();

    assert_eq!(error, SlowMotionRenderError::Cancelled);
    assert!(!output.delivered, "no output on cancel");
    assert!(!output.partial_active, "cancel must clean the partial");
    assert!(output.aborted);
}

#[test]
#[ignore = "blocked: requires platform VideoIO and a VideoToolbox frame-interpolation backend"]
fn test_vt_interpolation_on_hardware() {
    require_render_capabilities(&[
        RenderCapability::VideoDecode,
        RenderCapability::VideoToolboxFrameInterpolation,
        RenderCapability::VideoEncode,
        RenderCapability::PlayableMuxedOutput,
    ])
    .unwrap();
}

// VTMotionBlurrerTests.swift

#[test]
#[ignore = "blocked: requires a platform VideoToolbox capability factory"]
fn test_make_blurrer_matches_platform_gate() {
    require_render_capabilities(&[RenderCapability::VideoToolboxCapabilityFactory]).unwrap();
}

#[test]
#[ignore = "blocked: requires VideoToolbox blur/interpolation and decoded-pixel verification"]
fn test_vt_blur_bakes_a_trail_on_hardware() {
    require_render_capabilities(&[
        RenderCapability::VideoDecode,
        RenderCapability::VideoToolboxMotionBlur,
        RenderCapability::VideoToolboxFrameInterpolation,
        RenderCapability::VideoEncode,
        RenderCapability::PrivatePixelEffectVerification,
    ])
    .unwrap();
}

#[test]
#[ignore = "blocked: requires VideoToolbox blur/interpolation and playable muxed output"]
fn test_auto_decision_drives_vt_blur_on_hardware() {
    require_render_capabilities(&[
        RenderCapability::VideoDecode,
        RenderCapability::VideoToolboxMotionBlur,
        RenderCapability::VideoToolboxFrameInterpolation,
        RenderCapability::VideoEncode,
        RenderCapability::PlayableMuxedOutput,
    ])
    .unwrap();
}

// VTSceneCutFlushTests.swift

#[test]
#[ignore = "blocked: requires VideoToolbox's private flow cache and decoded-pixel verification"]
fn test_first_pair_after_cut_is_not_ghosted_by_stale_flow_state() {
    require_render_capabilities(&[
        RenderCapability::VideoDecode,
        RenderCapability::VideoToolboxFrameInterpolation,
        RenderCapability::VideoEncode,
        RenderCapability::PrivatePixelEffectVerification,
    ])
    .unwrap();
}
