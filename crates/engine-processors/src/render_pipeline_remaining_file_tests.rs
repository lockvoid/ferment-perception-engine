//! Remaining current portrait and upscale file assertions. All source pixels
//! use the same SyntheticVideoFixture generator as the frozen slow-motion16.
use super::*;
use crate::video_upscaler::Dimensions;

fn assert_file_geometry(path: &Path, width: u32, height: u32, frames: usize) {
    let backends = crate::platform_encode_backends();
    futures_executor::block_on(async {
        let info = backends.video.probe(path).await.unwrap();
        let stream = info.video_streams().next().unwrap();
        assert_eq!((stream.width, stream.height), (Some(width), Some(height)));
        assert!(
            stream
                .display_matrix
                .is_none_or(|matrix| matrix == [65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30]),
            "delivered pixels must have identity presentation transform"
        );
        let mut decoder = backends
            .video
            .open_decoder(path, stream.index)
            .await
            .unwrap();
        let mut count = 0;
        while let Some(frame) = decoder.decode_next().await.unwrap() {
            assert_eq!((frame.width, frame.height), (width, height));
            count += 1;
        }
        assert_eq!(count, frames);
    });
}

#[derive(Default)]
struct SizedDuplicate {
    calls: Vec<(u32, u32)>,
}
impl FrameInterpolator<VideoFrame> for SizedDuplicate {
    fn interpolate(
        &mut self,
        from: &TimedFrame<VideoFrame>,
        to: &TimedFrame<VideoFrame>,
        phases: &[f32],
        _restart: bool,
    ) -> std::result::Result<Vec<VideoFrame>, String> {
        let size = (from.frame.width, from.frame.height);
        assert_eq!(size, (to.frame.width, to.frame.height));
        assert!(
            size.0 <= 2560 && size.1 <= 1440,
            "current iOS landscape processing envelope"
        );
        self.calls.push(size);
        Ok(vec![from.frame.clone(); phases.len()])
    }
}

/// Kill: send a portrait extent through the landscape processing envelope or rotate an already-landscape clip.
#[test]
fn portrait_processing_uses_the_original_landscape_dimensions() {
    assert_eq!(slow_processing_size(1440, 2560), (2560, 1440));
    assert_eq!(slow_processing_size(2160, 3840), (3840, 2160));
    assert_eq!(slow_processing_size(2560, 1440), (2560, 1440));
}

/// Kill: omit portrait preparation/restoration or change the original three source frames into anything other than five delivered portrait pictures.
#[test]
#[ignore = "actual 1440x2560 three-frame source and real platform codecs/ffmpeg; fresh process"]
fn actual_portrait_2k_render_preserves_dimensions_through_the_processing_limit() {
    let f = fixture_at_size(3, 0.0, None, 1440, 2560);
    let before = std::fs::read(&f.source).unwrap();
    let output = f.output("portrait-duplicate");
    let mut spy = SizedDuplicate::default();
    let result = slow_motion_video_with_processors(
        &f.source,
        &output,
        2,
        None,
        &[],
        off(),
        180.0,
        ProgressSink::noop(),
        &mut spy,
        &mut SpyBlurrer::default(),
    )
    .unwrap();
    assert_eq!(result.frames_written, 5);
    assert_eq!(spy.calls, [(2560, 1440), (2560, 1440)]);
    assert_file_geometry(&output, 1440, 2560, 5);
    assert_eq!(std::fs::read(&f.source).unwrap(), before);
    f.no_partials();
}

/// Kill: substitute a duplicate/model-free processor for the actual portrait RIFE route or lose the original size, identity transform or five-frame result.
#[cfg(feature = "ort")]
#[test]
#[ignore = "actual pinned RIFE and 1440x2560 three-frame source; real codecs; fresh process"]
fn actual_rife_portrait_2k_interpolation_preserves_the_original_file_contract() {
    let f = fixture_at_size(3, 0.0, None, 1440, 2560);
    let result = real_render(&f, "portrait-rife", 2, &[], off());
    assert_eq!(result.frames_written, 5);
    assert_file_geometry(&f.output("portrait-rife"), 1440, 2560, 5);
}

#[derive(Default)]
struct LanczosSpy {
    started: Vec<Dimensions>,
    inputs: Vec<Dimensions>,
    outputs: Vec<Dimensions>,
    ended: usize,
}
#[async_trait::async_trait(?Send)]
impl FrameUpscaling for LanczosSpy {
    fn native_scale_factor(&self) -> u32 {
        2
    }
    fn start_session(&mut self, input: Dimensions, _prefer_hdr: bool) -> Result<()> {
        self.started.push(input);
        Ok(())
    }
    async fn upscale(&mut self, input: &VideoFrame) -> Result<FrameBuffer> {
        let size = Dimensions::new(input.width, input.height);
        self.inputs.push(size);
        assert_eq!(self.started.last(), Some(&size));
        let output = Dimensions::new(input.width * 2, input.height * 2);
        self.outputs.push(output);
        Ok(resized(input, output)?.buffer)
    }
    fn end_session(&mut self) {
        self.ended += 1;
    }
}

fn upscale_file(
    f: &Fixture,
    name: &str,
    mode: UpscaleMode,
    cap: Option<Dimensions>,
    spy: &mut LanczosSpy,
) -> core::VideoRenderResult {
    let before = std::fs::read(&f.source).unwrap();
    let result = futures_executor::block_on(upscale_video_with_processor(
        crate::platform_encode_backends(),
        &f.source,
        &f.output(name),
        mode,
        None,
        ProgressSink::noop(),
        spy,
        cap,
    ))
    .unwrap();
    assert_eq!(std::fs::read(&f.source).unwrap(), before);
    assert_eq!(spy.ended, 1);
    f.no_partials();
    result
}

/// Kill: skip the post-resample after the native x2 frame processor or drop any of the original sixteen encoded pictures.
#[test]
#[ignore = "original 480x272 sixteen-frame synthetic source; real codecs/ffmpeg"]
fn enhance_post_resamples_the_actual_x2_frames_back_to_the_original_size() {
    let f = fixture(16, 0.0, None);
    let mut spy = LanczosSpy::default();
    let result = upscale_file(&f, "enhance", UpscaleMode::Enhance, None, &mut spy);
    assert_eq!(result.frames_written, 16);
    assert_eq!(
        (result.output_width, result.output_height),
        (Some(480), Some(272))
    );
    assert_eq!(spy.outputs, vec![Dimensions::new(960, 544); 16]);
    assert_file_geometry(&f.output("enhance"), 480, 272, 16);
}

/// Kill: deliver source size after a real x2 upscale, report dimensions without writing them, or change the original sixteen-frame count.
#[test]
#[ignore = "original 480x272 sixteen-frame synthetic source; real codecs/ffmpeg"]
fn actual_lanczos_file_upscale_doubles_both_original_dimensions() {
    let f = fixture(16, 0.0, None);
    let result = upscale_file(
        &f,
        "double",
        UpscaleMode::Upscale { longest_side: 960 },
        None,
        &mut LanczosSpy::default(),
    );
    assert_eq!(result.frames_written, 16);
    assert_eq!(
        (result.output_width, result.output_height),
        (Some(960), Some(544))
    );
    assert_file_geometry(&f.output("double"), 960, 544, 16);
}

/// Kill: calculate a pre-cap plan without actually downscaling source pixels before the frame processor starts and consumes them.
#[test]
#[ignore = "original 480x272 sixteen-frame synthetic source; real codecs/ffmpeg"]
fn oversize_actual_source_reaches_the_processor_at_the_original_240x136_cap() {
    let f = fixture(16, 0.0, None);
    let mut spy = LanczosSpy::default();
    upscale_file(
        &f,
        "cap",
        UpscaleMode::Enhance,
        Some(Dimensions::new(240, 136)),
        &mut spy,
    );
    assert_eq!(spy.started, [Dimensions::new(240, 136)]);
    assert_eq!(spy.inputs, vec![Dimensions::new(240, 136); 16]);
    assert_file_geometry(&f.output("cap"), 480, 272, 16);
}
