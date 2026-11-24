//! Independent current Swift thumbnail geometry and exact pixel controls.
use super::*;
use ferment_engine_media::{ColorInfo, CpuPixelFormat, FrameBuffer, MediaTime, TimeBase};

fn matrix(turn: u8) -> [i32; 9] {
    let (a, b, c, d) = match turn {
        0 => (65536, 0, 0, 65536),
        1 => (0, 65536, -65536, 0),
        2 => (-65536, 0, 0, -65536),
        3 => (0, -65536, 65536, 0),
        _ => unreachable!(),
    };
    [a, b, 0, c, d, 0, 123 * 65536, 17 * 65536, 1073741824]
}
fn options(width: u32, height: u32) -> ThumbStripOptions {
    ThumbStripOptions {
        interval: 0.25,
        max_width: width,
        max_height: height,
    }
}
// Kill: cap rather than upscale the small source, floor rather than round to
// the nearest even cell, or use the encoded landscape dimensions for a strip.
#[test]
fn current_thumbnail_fit_uses_display_geometry_rounding_and_small_source_upscaling() {
    assert_eq!(
        ThumbnailGeometry::new(100, 50, None, options(192, 192)),
        ThumbnailGeometry {
            width: 192,
            height: 96,
            clockwise_turns: 0
        }
    );
    assert_eq!(
        ThumbnailGeometry::new(100, 51, None, options(192, 100)),
        ThumbnailGeometry {
            width: 192,
            height: 98,
            clockwise_turns: 0
        }
    );
    for turn in 0..4 {
        let result = ThumbnailGeometry::new(100, 50, Some(matrix(turn)), options(192, 192));
        let expected = if turn % 2 == 1 { (96, 192) } else { (192, 96) };
        assert_eq!(
            (result.width, result.height, result.clockwise_turns),
            (expected.0, expected.1, turn)
        );
    }
}
// Kill: rotate the encoded picture, use the wrong quarter-turn direction, or
// apply the source transform twice to an already displayed thumbnail.
#[test]
fn thumbnail_pixels_apply_one_quarter_turn_and_keep_coded_input_and_callback_time() {
    let bytes: Vec<u8> = (0..8u8)
        .flat_map(|value| [value * 20, value * 20 + 1, value * 20 + 2])
        .collect();
    let frame = ferment_engine_media::VideoFrame {
        width: 4,
        height: 2,
        pts: MediaTime::new(83_333, TimeBase::MICROSECONDS),
        duration: Some(MediaTime::new(50_000, TimeBase::MICROSECONDS)),
        color: ColorInfo::default(),
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(4, 2, 12, CpuPixelFormat::Rgb24, bytes.clone()).unwrap(),
        ),
    };
    for (turn, order) in [
        (0, [0, 1, 2, 3, 4, 5, 6, 7]),
        (1, [4, 0, 5, 1, 6, 2, 7, 3]),
        (2, [7, 6, 5, 4, 3, 2, 1, 0]),
        (3, [3, 7, 2, 6, 1, 5, 0, 4]),
    ] {
        let geometry = ThumbnailGeometry::new(4, 2, Some(matrix(turn)), options(4, 4));
        let small = geometry.picture(&frame).unwrap();
        let expected: Vec<u8> = order
            .into_iter()
            .flat_map(|value| [value * 20, value * 20 + 1, value * 20 + 2])
            .collect();
        assert_eq!(&*small.to_rgb24().data, &expected);
        let callback = thumbnail_frame(&small, 7, frame.pts.seconds());
        assert_eq!(
            (callback.frame.width, callback.frame.height),
            (geometry.width, geometry.height)
        );
        assert_eq!(callback.index, 7);
        assert_eq!(callback.presentation_time, 0.083333);
        for (bgra, rgb) in callback
            .frame
            .data
            .chunks_exact(4)
            .zip(expected.chunks_exact(3))
        {
            assert_eq!(bgra, [rgb[2], rgb[1], rgb[0], 255]);
        }
        assert_eq!(&*frame.to_cpu_rgb().unwrap().data, &bytes);
        assert_eq!((frame.width, frame.height), (4, 2));
        assert_eq!(
            frame.duration,
            Some(MediaTime::new(50_000, TimeBase::MICROSECONDS))
        );
    }
}

#[derive(Debug)]
struct CapturedInput(std::sync::Arc<std::sync::Mutex<Option<ferment_engine_media::VideoFrame>>>);
#[async_trait::async_trait(?Send)]
impl ferment_engine_media::EncodeSession for CapturedInput {
    async fn push_frame(
        &mut self,
        frame: ferment_engine_media::VideoFrame,
    ) -> ferment_engine_media::Result<()> {
        *self.0.lock().unwrap() = Some(frame);
        Ok(())
    }
    async fn finalize(
        self: Box<Self>,
    ) -> ferment_engine_media::Result<ferment_engine_media::EncodedVideo> {
        panic!("input observation does not finalize");
    }
}

#[test]
fn prepared_cpu_bgra_reuses_its_samples_without_an_rgb_round_trip() {
    let preset = VideoEncoder::ai_preset();
    let settings = VideoEncoder::plan(
        SourceVideo {
            width: 8,
            height: 4,
            frames_per_second: 30.0,
            color: ColorInfo::default(),
        },
        &preset,
    );
    let data: std::sync::Arc<[u8]> = (0..32u8)
        .flat_map(|n| [n, n + 1, n + 2, 255])
        .collect::<Vec<_>>()
        .into();
    let frame = ferment_engine_media::VideoFrame {
        width: 8,
        height: 4,
        pts: MediaTime::new(7, TimeBase::MICROSECONDS),
        duration: Some(MediaTime::new(33_333, TimeBase::MICROSECONDS)),
        color: settings.color,
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(8, 4, 32, CpuPixelFormat::Bgra8, data.clone()).unwrap(),
        ),
    };
    let capture = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut lane = VariantLane {
        id: "ai".into(),
        output: "unused.mp4".into(),
        preset,
        settings,
        encoder: Box::new(CapturedInput(capture.clone())),
        thumb_strip: None,
        thumbnails_emitted: 0,
        next_thumbnail_at: 0.0,
        display_matrix: None,
        thumbnail_geometry: None,
    };
    futures_executor::block_on(lane.push(&frame)).unwrap();
    let prepared = capture.lock().unwrap().take().unwrap();
    assert_eq!(prepared.pts, frame.pts);
    assert_eq!(prepared.duration, frame.duration);
    assert_eq!(prepared.color, lane.settings.color);
    assert_eq!(
        prepared.to_cpu_rgb().unwrap().data,
        frame.to_cpu_rgb().unwrap().data
    );
    // CPU-only encoder backends can retain the original Arc. A native-buffer
    // backend still uses the shared pixel-buffer allocator.
    #[cfg(not(target_os = "macos"))]
    {
        let FrameBuffer::Cpu(cpu) = prepared.buffer else {
            panic!("CPU encoder input expected")
        };
        assert_eq!(cpu.format, CpuPixelFormat::Bgra8);
        assert!(std::sync::Arc::ptr_eq(&cpu.data, &data));
    }
}
