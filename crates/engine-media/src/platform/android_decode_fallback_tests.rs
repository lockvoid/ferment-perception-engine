use super::*;
use std::future::Future;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

use crate::{
    ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, EncodeSession, EncoderSettings,
    FrameBuffer, MediaInfo, TransferCharacteristic, Yuv420Format, Yuv420Frame,
};
use futures_executor::block_on;

type Events = Arc<Mutex<Vec<String>>>;

#[derive(Debug, Clone, Copy)]
enum Step {
    Picture {
        pts: MediaTime,
        code: u16,
    },
    Rectangle {
        pts: MediaTime,
        code: u16,
        width: u32,
        height: u32,
        turns: u8,
    },
    Refusal,
    ImageTimeout,
    FileError,
}
fn picture(micros: i64, code: u16) -> Step {
    Step::Picture {
        pts: MediaTime::new(micros, TimeBase::MICROSECONDS),
        code,
    }
}
fn color() -> ColorInfo {
    ColorInfo {
        range: ColorRange::Limited,
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferCharacteristic::Pq,
        matrix: ColorMatrix::Bt2020NonConstant,
        bits_per_channel: Some(10),
    }
}
fn frame(pts: MediaTime, code: u16) -> VideoFrame {
    sized_frame(pts, code, 2, 2)
}
fn sized_frame(pts: MediaTime, code: u16, width: u32, height: u32) -> VideoFrame {
    let words = |values: &[u16]| {
        values
            .iter()
            .flat_map(|value| (value << 6).to_le_bytes())
            .collect()
    };
    VideoFrame {
        width,
        height,
        pts,
        duration: Some(MediaTime::new(17, TimeBase::MICROSECONDS)),
        color: color(),
        buffer: FrameBuffer::Yuv420(
            Yuv420Frame::new(
                width,
                height,
                Yuv420Format::P010Video,
                words(
                    &(0..width * height)
                        .map(|index| code + index as u16)
                        .collect::<Vec<_>>(),
                ),
                words(&[501, 513].repeat((width * height / 4) as usize)),
            )
            .unwrap(),
        ),
    }
}
fn code(frame: &VideoFrame) -> u16 {
    assert_eq!(
        frame.color,
        color(),
        "fallback must retain original PQ/BT.2020/10-bit metadata"
    );
    assert_eq!(
        frame.duration,
        Some(MediaTime::new(17, TimeBase::MICROSECONDS))
    );
    let FrameBuffer::Yuv420(yuv) = &frame.buffer else {
        panic!("HDR was quantized to RGB8");
    };
    assert_eq!(yuv.format, Yuv420Format::P010Video);
    u16::from_le_bytes([yuv.y[0], yuv.y[1]]) >> 6
}

#[derive(Debug)]
struct Script {
    name: &'static str,
    steps: Vec<Step>,
    at: usize,
    fail_seek: bool,
    // A software time base may round its seek threshold down. The wrapper's
    // final microsecond filter must still honor the public MediaCodec target.
    coarse_seek: bool,
    events: Events,
}
impl Drop for Script {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("drop:{}", self.name));
    }
}
#[async_trait(?Send)]
impl DecodeSession for Script {
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        let micros = target.rescale(TimeBase::MICROSECONDS).value;
        self.events
            .lock()
            .unwrap()
            .push(format!("seek:{}:{micros}", self.name));
        if self.fail_seek {
            return Err(VideoError::Platform {
                operation: "AMediaCodec_start",
                status: -10000,
            });
        }
        let threshold = if self.coarse_seek {
            micros / 10 * 10
        } else {
            micros
        };
        self.at = self
            .steps
            .iter()
            .position(|step| {
                matches!(step,
            Step::Picture { pts, .. } | Step::Rectangle { pts, .. }
                if pts.rescale(TimeBase::MICROSECONDS).value >= threshold)
            })
            .unwrap_or(self.steps.len());
        Ok(())
    }
    async fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        let step = self.steps.get(self.at).copied();
        self.at += usize::from(step.is_some());
        match step {
            Some(Step::Picture { pts, code }) => Ok(Some(frame(pts, code))),
            Some(Step::Rectangle {
                pts,
                code,
                width,
                height,
                turns,
            }) => sized_frame(pts, code, width, height)
                .rotated_clockwise(turns)
                .map(Some),
            Some(Step::Refusal) => Err(VideoError::UnsupportedCodec(
                "P010 ImageReader refused output".into(),
            )),
            Some(Step::ImageTimeout) => Err(VideoError::BackendUnavailable(
                "rendered decoder image did not arrive within one second".into(),
            )),
            Some(Step::FileError) => Err(VideoError::Io(std::io::Error::other("source removed"))),
            None => Ok(None),
        }
    }
}

#[derive(Debug)]
struct Software {
    steps: Vec<Step>,
    expected_size: Option<(u32, u32)>,
    coarse_seek: bool,
    events: Events,
}
impl Software {
    fn open(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
        coded: bool,
    ) -> Box<dyn DecodeSession> {
        assert_eq!(source, Path::new("original-pq.mov"));
        assert_eq!(stream_index, 3);
        assert_eq!(options.output_size, self.expected_size);
        self.events
            .lock()
            .unwrap()
            .push(format!("open:software:coded={coded}"));
        Box::new(Script {
            name: "software",
            steps: self.steps.clone(),
            at: 0,
            fail_seek: false,
            coarse_seek: self.coarse_seek,
            events: self.events.clone(),
        })
    }
}
#[async_trait(?Send)]
impl VideoIo for Software {
    async fn probe(&self, _: &Path) -> Result<MediaInfo> {
        unreachable!()
    }
    async fn open_decoder(&self, _: &Path, _: u32) -> Result<Box<dyn DecodeSession>> {
        unreachable!()
    }
    async fn open_decoder_with(
        &self,
        source: &Path,
        stream: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        Ok(self.open(source, stream, options, false))
    }
    async fn open_coded_decoder_with(
        &self,
        source: &Path,
        stream: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        Ok(self.open(source, stream, options, true))
    }
    async fn open_encoder(&self, _: EncoderSettings) -> Result<Box<dyn EncodeSession>> {
        unreachable!()
    }
}
fn decoder(
    primary: Vec<Step>,
    software: Vec<Step>,
    fail_seek: bool,
    coarse_seek: bool,
    coded: bool,
) -> (RecoveringDecoder, Events) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let primary = Script {
        name: "hardware",
        steps: primary,
        at: 0,
        fail_seek,
        coarse_seek: false,
        events: events.clone(),
    };
    let software = Software {
        steps: software,
        expected_size: Some((2, 2)),
        coarse_seek,
        events: events.clone(),
    };
    (
        RecoveringDecoder::new(
            Path::new("original-pq.mov"),
            3,
            DecodeOptions {
                output_size: Some((2, 2)),
                ..Default::default()
            },
            SoftwareOrientation::from_track(coded, 0),
            Some(Box::new(primary)),
            Arc::new(software),
        ),
        events,
    )
}
fn next(decoder: &mut RecoveringDecoder) -> VideoFrame {
    block_on(decoder.decode_next()).unwrap().unwrap()
}

#[test]
fn codec_start_and_image_refusals_are_recoverable_but_source_failures_are_not() {
    for operation in [
        "AMediaCodec_configure",
        "AMediaCodec_start",
        "AImageReader_acquireNextImage",
        "AImage_getPlaneData",
    ] {
        assert!(decoder_refusal(&VideoError::Platform {
            operation,
            status: -10000
        }));
    }
    assert!(!decoder_refusal(&VideoError::Platform {
        operation: "AMediaExtractor_setDataSourceFd",
        status: -10000
    }));
    assert!(!decoder_refusal(&VideoError::Io(std::io::Error::other(
        "closed source"
    ))));
    assert!(!decoder_refusal(&VideoError::InvalidData(
        "invalid requested size".into()
    )));
}

#[test]
fn first_image_refusal_reopens_original_hdr_without_seeking_or_quantizing() {
    let (mut reader, events) = decoder(
        vec![Step::Refusal],
        vec![picture(0, 65), picture(31, 69)],
        false,
        false,
        true,
    );
    block_on(reader.seek(MediaTime::ZERO)).unwrap();
    assert_eq!(code(&next(&mut reader)), 65);
    assert_eq!(code(&next(&mut reader)), 69);
    assert!(block_on(reader.decode_next()).unwrap().is_none());
    drop(reader);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true", "drop:software"]
    );
}

#[test]
fn late_refusal_counts_equal_pts_and_keeps_the_next_original_picture() {
    let (mut reader, events) = decoder(
        vec![picture(10, 65), picture(10, 66), Step::Refusal],
        vec![
            picture(0, 64),
            picture(10, 65),
            picture(10, 66),
            picture(10, 67),
            picture(31, 68),
        ],
        false,
        false,
        false,
    );
    assert_eq!(code(&next(&mut reader)), 65);
    assert_eq!(code(&next(&mut reader)), 66);
    assert_eq!(code(&next(&mut reader)), 67);
    assert_eq!(code(&next(&mut reader)), 68);
    assert!(block_on(reader.decode_next()).unwrap().is_none());
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true"]
    );
}

#[test]
fn fallback_matches_the_delivered_microsecond_clock_without_retiming_frames() {
    let (mut reader, events) = decoder(
        vec![picture(16_666, 81), Step::Refusal],
        vec![
            Step::Picture {
                pts: MediaTime::new(1_500, TimeBase::new_unchecked(1, 90_000)),
                code: 81,
            },
            Step::Picture {
                pts: MediaTime::new(3_000, TimeBase::new_unchecked(1, 90_000)),
                code: 85,
            },
        ],
        false,
        false,
        true,
    );
    next(&mut reader);
    let recovered = next(&mut reader);
    assert_eq!(code(&recovered), 85);
    assert_eq!(
        recovered.pts,
        MediaTime::new(3_000, TimeBase::new_unchecked(1, 90_000))
    );
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .all(|event| !event.starts_with("seek:")),
        "sequential recovery must open at the original unread source, never seek near the boundary"
    );
}

#[test]
fn explicit_seek_switches_before_hardware_seek_and_filters_the_software_clock() {
    let (mut reader, events) = decoder(
        vec![picture(0, 64), picture(20, 70)],
        vec![picture(0, 64), picture(10, 65), picture(20, 70)],
        true,
        true,
        true,
    );
    next(&mut reader);
    block_on(reader.seek(MediaTime::new(12, TimeBase::MICROSECONDS))).unwrap();
    assert_eq!(
        code(&next(&mut reader)),
        70,
        "floor-rounded software seek must not expose picture 10"
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "drop:hardware",
            "open:software:coded=true",
            "seek:software:12"
        ]
    );
}

#[test]
fn backward_seek_after_handoff_stays_on_software_and_discards_old_resume_state() {
    let (mut reader, events) = decoder(
        vec![picture(10, 65), Step::Refusal],
        vec![picture(0, 64), picture(10, 65), picture(20, 66)],
        false,
        false,
        true,
    );
    next(&mut reader);
    assert_eq!(code(&next(&mut reader)), 66);
    block_on(reader.seek(MediaTime::ZERO)).unwrap();
    assert_eq!(code(&next(&mut reader)), 64);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.starts_with("open:"))
            .count(),
        1
    );
}

#[test]
fn a_missing_replay_boundary_refuses_instead_of_silently_losing_a_frame() {
    for software in [vec![picture(20, 66)], vec![]] {
        let (mut reader, events) = decoder(
            vec![picture(10, 65), Step::Refusal],
            software,
            false,
            false,
            true,
        );
        next(&mut reader);
        assert!(
            matches!(block_on(reader.decode_next()), Err(VideoError::InvalidData(message)) if message.contains("did not reproduce"))
        );
        assert!(
            block_on(reader.decode_next()).is_err(),
            "failed recovery cannot later expose an unchecked suffix"
        );
        assert_eq!(
            *events.lock().unwrap(),
            vec!["drop:hardware", "open:software:coded=true", "drop:software"]
        );
    }
}

#[test]
fn cancelling_replay_keeps_its_cursor_and_closes_each_owner_once() {
    let (mut reader, events) = decoder(
        vec![picture(10, 65), picture(10, 66), Step::Refusal],
        vec![
            picture(10, 65),
            picture(10, 66),
            picture(10, 67),
            picture(20, 68),
        ],
        false,
        false,
        true,
    );
    next(&mut reader);
    next(&mut reader);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut pending = Box::pin(reader.decode_next());
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    drop(pending);
    assert_eq!(
        code(&next(&mut reader)),
        67,
        "a cancelled replay must not skip the third equal-PTS picture"
    );
    drop(reader);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true", "drop:software"]
    );
}

#[test]
fn abandoning_pending_recovery_releases_both_owners_without_another_poll() {
    let (mut reader, events) = decoder(
        vec![picture(10, 65), Step::Refusal],
        vec![picture(10, 65), picture(20, 66)],
        false,
        false,
        true,
    );
    next(&mut reader);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut pending = Box::pin(reader.decode_next());
    assert!(matches!(pending.as_mut().poll(&mut context), Poll::Pending));
    drop(pending);
    drop(reader);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true", "drop:software"]
    );
}

#[test]
fn file_errors_and_successful_eof_do_not_open_another_backend() {
    for steps in [vec![Step::FileError], vec![]] {
        let (mut reader, events) = decoder(steps.clone(), vec![picture(0, 64)], false, false, true);
        let result = block_on(reader.decode_next());
        if steps.is_empty() {
            assert!(result.unwrap().is_none());
        } else {
            assert!(matches!(result, Err(VideoError::Io(_))));
        }
        drop(reader);
        assert_eq!(*events.lock().unwrap(), vec!["drop:hardware"]);
    }
}

#[test]
fn software_failure_is_terminal_and_never_retries_hardware() {
    let (mut reader, events) = decoder(
        vec![Step::Refusal],
        vec![Step::FileError],
        false,
        false,
        true,
    );
    assert!(matches!(
        block_on(reader.decode_next()),
        Err(VideoError::Io(_))
    ));
    assert!(block_on(reader.decode_next()).is_err());
    drop(reader);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true", "drop:software"]
    );
}

#[test]
fn out_of_order_hardware_output_is_never_guessed_into_a_software_cursor() {
    let (mut reader, events) = decoder(
        vec![picture(20, 66), picture(10, 65), Step::Refusal],
        vec![picture(10, 65), picture(20, 66)],
        false,
        false,
        true,
    );
    next(&mut reader);
    next(&mut reader);
    assert!(
        matches!(block_on(reader.decode_next()), Err(VideoError::InvalidData(message)) if message.contains("out-of-order"))
    );
    assert_eq!(*events.lock().unwrap(), vec!["drop:hardware"]);
}

#[test]
fn zero_boundary_duplicates_replay_from_origin_without_a_minus_one_seek() {
    let (mut reader, events) = decoder(
        vec![picture(0, 64), Step::Refusal],
        vec![
            picture(-10, 61),
            picture(0, 64),
            picture(0, 65),
            picture(10, 70),
        ],
        false,
        true,
        true,
    );
    assert_eq!(code(&next(&mut reader)), 64);
    assert_eq!(code(&next(&mut reader)), 65);
    assert_eq!(code(&next(&mut reader)), 70);
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .all(|event| !event.starts_with("seek:"))
    );
}

#[test]
fn initial_seek_keeps_leading_negative_pts_on_first_hardware_refusal() {
    for target in [0, -100] {
        let (mut reader, events) = decoder(
            vec![Step::Refusal],
            vec![picture(-30, 61), picture(-10, 62), picture(0, 64)],
            false,
            true,
            true,
        );
        block_on(reader.seek(MediaTime::new(target, TimeBase::MICROSECONDS))).unwrap();
        for (pts, sample) in [(-30, 61), (-10, 62), (0, 64)] {
            let frame = next(&mut reader);
            assert_eq!(frame.pts.value, pts);
            assert_eq!(code(&frame), sample);
        }
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .all(|event| !event.starts_with("seek:"))
        );
    }
}

#[test]
fn negative_boundary_duplicates_keep_the_first_unread_equal_pts_picture() {
    let (mut reader, events) = decoder(
        vec![picture(-20, 61), picture(-10, 62), Step::Refusal],
        vec![
            picture(-20, 61),
            picture(-10, 62),
            picture(-10, 63),
            picture(0, 64),
        ],
        false,
        true,
        true,
    );
    for sample in [61, 62, 63, 64] {
        assert_eq!(code(&next(&mut reader)), sample);
    }
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .all(|event| !event.starts_with("seek:"))
    );
}

#[test]
fn explicit_seek_never_enters_a_hardware_duplicate_group() {
    let (mut reader, events) = decoder(
        vec![picture(0, 61), picture(10, 63), Step::Refusal],
        vec![
            picture(0, 61),
            picture(10, 62),
            picture(10, 63),
            picture(20, 64),
        ],
        false,
        false,
        true,
    );
    assert_eq!(code(&next(&mut reader)), 61);
    block_on(reader.seek(MediaTime::new(10, TimeBase::MICROSECONDS))).unwrap();
    for sample in [62, 63, 64] {
        assert_eq!(code(&next(&mut reader)), sample);
    }
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "drop:hardware",
            "open:software:coded=true",
            "seek:software:10",
        ]
    );
}

#[test]
fn software_only_initial_zero_seek_also_preserves_negative_leading_pictures() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut reader = block_on(RecoveringDecoder::software(
        Path::new("original-pq.mov"),
        3,
        DecodeOptions {
            output_size: Some((2, 2)),
            ..Default::default()
        },
        SoftwareOrientation::CODED,
        Arc::new(Software {
            steps: vec![picture(-10, 61), picture(0, 64)],
            expected_size: Some((2, 2)),
            coarse_seek: true,
            events: events.clone(),
        }),
    ))
    .unwrap();
    block_on(reader.seek(MediaTime::ZERO)).unwrap();
    assert_eq!(next(&mut reader).pts.value, -10);
    assert_eq!(next(&mut reader).pts.value, 0);
    assert_eq!(*events.lock().unwrap(), vec!["open:software:coded=true"]);
}

#[test]
fn sdr_requires_real_image_planes_and_hdr_buffers_keep_strict_source_depth() {
    let mut sdr = color();
    sdr.transfer = TransferCharacteristic::Bt709;
    sdr.bits_per_channel = Some(8);
    assert!(require_supported_sink(true, sdr, false).is_ok());
    assert!(matches!(
        require_supported_sink(false, sdr, false),
        Err(VideoError::UnsupportedCodec(_))
    ));
    assert!(matches!(
        require_supported_sink(false, color(), true),
        Err(VideoError::UnsupportedCodec(_))
    ));
    assert!(require_supported_sink(true, color(), true).is_ok());
    assert!(require_supported_sink(false, color(), false).is_ok());
    let mut eight_bit_hdr = color();
    eight_bit_hdr.bits_per_channel = Some(8);
    assert!(require_supported_sink(false, eight_bit_hdr, false).is_ok());
}

#[test]
fn image_crop_uses_exclusive_edges_and_refuses_any_hidden_pixels() {
    assert!(require_full_image_crop(192, 108, 0, 0, 192, 108).is_ok());
    for crop in [
        (0, 0, 191, 107),
        (2, 0, 192, 108),
        (0, 2, 192, 108),
        (-1, 0, 192, 108),
        (0, 0, 194, 108),
        (0, 0, 192, 110),
    ] {
        assert!(
            matches!(
                require_full_image_crop(192, 108, crop.0, crop.1, crop.2, crop.3),
                Err(VideoError::UnsupportedCodec(_))
            ),
            "crop {crop:?}"
        );
    }
    assert!(require_full_image_crop(0, 0, 0, 0, 0, 0).is_err());
}

#[test]
fn buffer_crop_uses_inclusive_edges_and_refuses_partial_or_non_full_metadata() {
    assert!(require_full_buffer_crop(192, 108, None, None, None, None).is_ok());
    assert!(require_full_buffer_crop(192, 108, Some(0), Some(0), Some(191), Some(107)).is_ok());
    for crop in [
        (Some(0), Some(0), Some(192), Some(108)),
        (Some(2), Some(0), Some(191), Some(107)),
        (Some(0), Some(0), Some(189), Some(107)),
        (None, Some(0), Some(191), Some(107)),
        (Some(0), None, None, None),
    ] {
        assert!(
            matches!(
                require_full_buffer_crop(192, 108, crop.0, crop.1, crop.2, crop.3),
                Err(VideoError::UnsupportedCodec(_))
            ),
            "crop {crop:?}"
        );
    }
    assert!(require_full_buffer_crop(0, 108, None, None, None, None).is_err());
}

#[test]
fn display_rotation_swaps_requested_axes_and_permutes_original_hdr_samples_once() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let pts = MediaTime::new(23, TimeBase::MICROSECONDS);
    let primary = Script {
        name: "hardware",
        steps: vec![
            Step::Rectangle {
                pts: MediaTime::ZERO,
                code: 65,
                width: 2,
                height: 4,
                turns: 1,
            },
            Step::Refusal,
        ],
        at: 0,
        fail_seek: false,
        coarse_seek: false,
        events: events.clone(),
    };
    let mut reader = RecoveringDecoder::new(
        Path::new("original-pq.mov"),
        3,
        DecodeOptions {
            output_size: Some((4, 2)),
            ..Default::default()
        },
        SoftwareOrientation::from_track(false, 90),
        Some(Box::new(primary)),
        Arc::new(Software {
            steps: vec![
                Step::Rectangle {
                    pts: MediaTime::ZERO,
                    code: 65,
                    width: 2,
                    height: 4,
                    turns: 0,
                },
                Step::Rectangle {
                    pts,
                    code: 81,
                    width: 2,
                    height: 4,
                    turns: 0,
                },
            ],
            expected_size: Some((2, 4)),
            coarse_seek: false,
            events: events.clone(),
        }),
    );
    let first = next(&mut reader);
    assert_eq!((first.width, first.height), (4, 2));
    assert_eq!(
        code(&first),
        71,
        "hardware already applied display rotation"
    );
    let frame = next(&mut reader);
    assert_eq!((frame.width, frame.height), (4, 2));
    assert_eq!(frame.pts, pts);
    assert_eq!(frame.color, color());
    assert_eq!(
        frame.duration,
        Some(MediaTime::new(17, TimeBase::MICROSECONDS))
    );
    let FrameBuffer::Yuv420(yuv) = frame.buffer else {
        panic!("lost source-depth planes");
    };
    assert_eq!(yuv.format, Yuv420Format::P010Video);
    let words = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes([word[0], word[1]]) >> 6)
            .collect::<Vec<_>>()
    };
    assert_eq!(words(&yuv.y), vec![87, 85, 83, 81, 88, 86, 84, 82]);
    assert_eq!(words(&yuv.uv), vec![501, 513, 501, 513]);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true"]
    );
}

#[test]
fn coded_and_ambiguous_display_rotations_remain_consistent_from_open() {
    let options = DecodeOptions {
        output_size: Some((192, 108)),
        ..Default::default()
    };
    for degrees in [0, 90, 180, 270, -90, 45] {
        let coded = SoftwareOrientation::from_track(true, degrees);
        assert_eq!(coded, SoftwareOrientation::CODED);
        assert_eq!(coded.options(options), options);
        assert_eq!(code(&coded.frame(frame(MediaTime::ZERO, 65)).unwrap()), 65);
    }
    let ambiguous = SoftwareOrientation::from_track(false, 45);
    assert!(!ambiguous.hardware_allowed);
    assert_eq!(ambiguous.options(options), options);
    assert_eq!(
        code(&ambiguous.frame(frame(MediaTime::ZERO, 65)).unwrap()),
        65
    );
    for (degrees, turns, swap) in [
        (0, 0, false),
        (90, 1, true),
        (180, 2, false),
        (270, 3, true),
        (-90, 3, true),
    ] {
        let display = SoftwareOrientation::from_track(false, degrees);
        assert_eq!(display.quarter_turns, turns);
        assert_eq!(
            display.options(options).output_size,
            Some(if swap { (108, 192) } else { (192, 108) })
        );
        assert_eq!(display.hardware_allowed, degrees >= 0);
    }
}

#[test]
fn expected_image_timeout_recovers_the_unpublished_tail_instead_of_claiming_eof() {
    let (mut reader, events) = decoder(
        vec![picture(0, 65), Step::ImageTimeout],
        vec![picture(0, 65), picture(10, 66), picture(20, 67)],
        false,
        false,
        true,
    );
    for sample in [65, 66, 67] {
        assert_eq!(code(&next(&mut reader)), sample);
    }
    assert!(block_on(reader.decode_next()).unwrap().is_none());
    drop(reader);
    assert_eq!(
        *events.lock().unwrap(),
        vec!["drop:hardware", "open:software:coded=true", "drop:software",]
    );
}
