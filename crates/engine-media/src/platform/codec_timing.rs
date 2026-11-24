//! Timing which MediaCodec's buffer-info structure does not carry.
//!
//! Keep one decoded picture for the next presentation boundary, and preserve
//! submitted encoder timing independently of codec delivery order. No pixel
//! copies, assumed constant frame rate, or whole-file sample table are needed.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use crate::{MediaTime, Result, TimeBase, VideoError, VideoFrame};

/// Encoding is the explicit clock boundary: its caller supplied the output
/// time base. Public decoded timestamps remain the exact NDK µs observations.
fn nearest_ratio(numerator: i128, denominator: i128, base: TimeBase) -> Result<MediaTime> {
    if denominator <= 0 {
        return Err(VideoError::InvalidData(
            "video timing has an invalid clock".into(),
        ));
    }
    let whole = numerator / denominator;
    let remainder = numerator % denominator;
    let rounded = whole
        + if remainder.abs() >= denominator / 2 + denominator % 2 {
            remainder.signum()
        } else {
            0
        };
    let value = i64::try_from(rounded).map_err(|_| timing_overflow())?;
    Ok(MediaTime::new(value, base))
}

fn timing_overflow() -> VideoError {
    VideoError::InvalidData("video timing exceeds its declared clock".into())
}

fn nearest(time: MediaTime, base: TimeBase) -> Result<MediaTime> {
    if time.time_base.numerator <= 0
        || time.time_base.denominator <= 0
        || base.numerator <= 0
        || base.denominator <= 0
    {
        return Err(timing_overflow());
    }
    let numerator = i128::from(time.value)
        .checked_mul(i128::from(time.time_base.numerator))
        .and_then(|value| value.checked_mul(i128::from(base.denominator)))
        .ok_or_else(timing_overflow)?;
    let denominator = i128::from(time.time_base.denominator)
        .checked_mul(i128::from(base.numerator))
        .ok_or_else(timing_overflow)?;
    nearest_ratio(numerator, denominator, base)
}

/// Round the sum once, including when PTS and duration use different clocks.
/// Rounding the duration alone loses a terminal tick for fractional-rate input.
fn interval_end(pts: MediaTime, duration: MediaTime, base: TimeBase) -> Result<MediaTime> {
    let a = pts.time_base;
    let b = duration.time_base;
    if [
        a.numerator,
        a.denominator,
        b.numerator,
        b.denominator,
        base.numerator,
        base.denominator,
    ]
    .iter()
    .any(|value| *value <= 0)
    {
        return Err(timing_overflow());
    }
    let first = i128::from(pts.value)
        .checked_mul(i128::from(a.numerator))
        .and_then(|value| value.checked_mul(i128::from(b.denominator)))
        .ok_or_else(timing_overflow)?;
    let second = i128::from(duration.value)
        .checked_mul(i128::from(b.numerator))
        .and_then(|value| value.checked_mul(i128::from(a.denominator)))
        .ok_or_else(timing_overflow)?;
    let numerator = first
        .checked_add(second)
        .and_then(|value| value.checked_mul(i128::from(base.denominator)))
        .ok_or_else(timing_overflow)?;
    let denominator = i128::from(a.denominator)
        .checked_mul(i128::from(b.denominator))
        .and_then(|value| value.checked_mul(i128::from(base.numerator)))
        .ok_or_else(timing_overflow)?;
    nearest_ratio(numerator, denominator, base)
}

#[derive(Debug)]
pub(super) struct DecodedTimeline {
    base: TimeBase,
    end: Option<MediaTime>,
    pending: Option<VideoFrame>,
}

impl DecodedTimeline {
    /// Read raw VIDEO stream timing. MediaInfo deliberately substitutes the
    /// container duration and a zero origin when metadata is missing; neither
    /// fallback proves the last picture's end. This context closes immediately.
    pub(super) fn open(source: &Path, stream_index: u32) -> Result<Self> {
        crate::platform::libav::init()?;
        let input = ffmpeg_next::format::input(source)
            .map_err(|error| VideoError::Demux(format!("{}: {error}", source.display())))?;
        let stream = input
            .stream(stream_index as usize)
            .ok_or_else(|| VideoError::InvalidData(format!("no video stream {stream_index}")))?;
        if stream.parameters().medium() != ffmpeg_next::media::Type::Video {
            return Err(VideoError::InvalidData(format!(
                "stream {stream_index} is not video"
            )));
        }
        let rational = stream.time_base();
        let base = TimeBase::new(
            i64::from(rational.numerator()),
            i64::from(rational.denominator()),
        )?;
        let at = |value| {
            (value != ffmpeg_next::ffi::AV_NOPTS_VALUE).then(|| MediaTime::new(value, base))
        };
        Self::new(
            TimeBase::MICROSECONDS,
            at(stream.start_time()),
            at(stream.duration()),
        )
    }
    pub(super) fn new(
        base: TimeBase,
        start: Option<MediaTime>,
        duration: Option<MediaTime>,
    ) -> Result<Self> {
        // A duration is not an absolute end when its origin is unknown.
        // In particular a seek must not establish a new origin for the track.
        nearest(MediaTime::ZERO, base)?;
        let end = start
            .zip(duration.filter(|duration| duration.value > 0))
            .map(|(start, duration)| -> Result<MediaTime> { interval_end(start, duration, base) })
            .transpose()?;
        Ok(Self {
            base,
            end,
            pending: None,
        })
    }

    /// Retains exactly the newly decoded picture. The previous picture is
    /// returned with its actual next presentation boundary, including VFR.
    pub(super) fn push(&mut self, frame: VideoFrame) -> Option<VideoFrame> {
        let next = frame.pts;
        self.pending
            .replace(frame)
            .map(|frame| self.finish_frame(frame, Some(next)))
    }

    /// The declared VIDEO track end supplies only the terminal boundary.
    /// Container duration could include later audio and must not be used here.
    pub(super) fn finish(&mut self) -> Option<VideoFrame> {
        self.pending
            .take()
            .map(|frame| self.finish_frame(frame, None))
    }

    fn finish_frame(&self, mut frame: VideoFrame, next: Option<MediaTime>) -> VideoFrame {
        let boundary = match (next, self.end) {
            (Some(next), Some(end)) if next.value > end.value => Some(end),
            (Some(next), _) => Some(next),
            (None, end) => end,
        };
        if frame.duration.is_none() {
            frame.duration = boundary
                .and_then(|end| end.value.checked_sub(frame.pts.value))
                .filter(|value| *value > 0)
                .map(|value| MediaTime::new(value, self.base));
        }
        frame
    }

    /// Seek invalidates and releases the held picture. Dropping this owner
    /// likewise releases it when the caller cancels or abandons a decode.
    pub(super) fn clear(&mut self) {
        self.pending = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SubmittedTiming {
    pub(super) pts: MediaTime,
    pub(super) duration: Option<MediaTime>,
}

#[derive(Debug, Default)]
pub(super) struct SubmittedTimings {
    outstanding: usize,
    by_codec_pts: BTreeMap<i64, VecDeque<SubmittedTiming>>,
}

impl SubmittedTimings {
    // A dropped-output encoder must fail while timing ownership is bounded,
    // rather than accepting a whole movie and discovering missing frames at
    // EOS. This permits 256 in-flight/reordered pictures, retaining no pixels.
    const MAX_OUTSTANDING: usize = 256;

    pub(super) fn record(
        &mut self,
        codec_pts: i64,
        frame: &VideoFrame,
        output_clock: TimeBase,
    ) -> Result<()> {
        if self.outstanding == Self::MAX_OUTSTANDING {
            return Err(VideoError::BackendUnavailable(format!(
                "MediaCodec retained {} video submissions without returning their pictures",
                Self::MAX_OUTSTANDING
            )));
        }
        let pts = nearest(frame.pts, output_clock)?;
        let duration = frame
            .duration
            .map(|duration| -> Result<MediaTime> {
                let end = interval_end(frame.pts, duration, output_clock)?;
                let value = end
                    .value
                    .checked_sub(pts.value)
                    .ok_or_else(timing_overflow)?;
                Ok(MediaTime::new(value, output_clock))
            })
            .transpose()?;
        self.by_codec_pts
            .entry(codec_pts)
            .or_default()
            .push_back(SubmittedTiming { pts, duration });
        self.outstanding += 1;
        Ok(())
    }

    pub(super) fn take(&mut self, codec_pts: i64) -> Result<SubmittedTiming> {
        let entries = self.by_codec_pts.get_mut(&codec_pts).ok_or_else(|| {
            VideoError::InvalidData(format!(
                "MediaCodec produced an unsubmitted video timestamp {codec_pts}"
            ))
        })?;
        let timing = entries.pop_front().expect("a timing entry is never empty");
        self.outstanding -= 1;
        if entries.is_empty() {
            self.by_codec_pts.remove(&codec_pts);
        }
        Ok(timing)
    }

    pub(super) fn finish(&self) -> Result<()> {
        if self.by_codec_pts.is_empty() {
            return Ok(());
        }
        let remaining = self.outstanding;
        Err(VideoError::InvalidData(format!(
            "MediaCodec omitted {remaining} submitted video frames at end of stream"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorInfo, CpuFrame, CpuPixelFormat, FrameBuffer};
    use std::sync::Arc;

    fn frame(value: i64, base: TimeBase) -> VideoFrame {
        VideoFrame {
            width: 1,
            height: 1,
            pts: MediaTime::new(value, base),
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(
                CpuFrame::new(1, 1, 3, CpuPixelFormat::Rgb24, vec![1, 2, 3]).unwrap(),
            ),
        }
    }

    /// Kill: truncate observed µs into the caller's explicit encode clock a
    /// second time, duplicating frames; do not alter public decode timestamps.
    #[test]
    fn encoding_clock_rounding_preserves_observed_decode_pts_and_distinct_frames() {
        let output_clock = TimeBase::new_unchecked(1, 12);
        let mut ledger = SubmittedTimings::default();
        for index in 0..12 {
            let observed = index * 1_000_000 / 12;
            let input = frame(observed, TimeBase::MICROSECONDS);
            ledger.record(observed, &input, output_clock).unwrap();
            assert_eq!(input.pts, MediaTime::new(observed, TimeBase::MICROSECONDS));
            let encoded = ledger.take(observed).unwrap();
            assert_eq!(encoded.pts, MediaTime::new(index, output_clock));
            assert!((encoded.pts.seconds() * 1e6 - observed as f64).abs() < 1.0);
        }
        assert_eq!(
            nearest(
                MediaTime::new(83_333, TimeBase::MICROSECONDS),
                TimeBase::NANOSECONDS
            )
            .unwrap()
            .value,
            83_333_000,
            "a finer output clock does not invent the lost fractional microsecond"
        );
        assert_eq!(
            nearest(
                MediaTime::new(-83_333, TimeBase::MICROSECONDS),
                output_clock
            )
            .unwrap()
            .value,
            -1
        );
    }

    /// Kill: derive every frame's duration from average frame rate or use
    /// duration as an absolute end without the positive stream origin.
    #[test]
    fn variable_intervals_and_nonzero_origin_reach_the_exact_video_track_end() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline = DecodedTimeline::new(
            base,
            Some(MediaTime::new(500, base)),
            Some(MediaTime::new(140, base)),
        )
        .unwrap();
        assert!(timeline.push(frame(500, base)).is_none());
        assert_eq!(
            timeline
                .push(frame(520, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            20
        );
        assert_eq!(
            timeline
                .push(frame(575, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            55
        );
        let last = timeline.finish().unwrap();
        assert_eq!((last.pts.value, last.duration.unwrap().value), (575, 65));
        assert!(timeline.finish().is_none());
    }

    /// Kill: use the final picture PTS in place of the raw MP4 track end.
    #[test]
    fn raw_mp4_track_metadata_supplies_the_terminal_boundary() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated/h264-gop4.mp4");
        let timeline = DecodedTimeline::open(&path, 0).unwrap();
        assert_eq!(
            timeline.end,
            Some(MediaTime::new(1_000_000, TimeBase::MICROSECONDS))
        );
        assert!(DecodedTimeline::open(&path, u32::MAX).is_err());
    }

    /// Kill: borrow a Matroska container duration when the raw video stream
    /// has no own end; later audio must not lengthen the terminal picture.
    #[test]
    fn container_only_duration_is_not_admitted_as_the_video_track_end() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated/h264-gop6.mkv");
        assert!(crate::demux::probe(&path).unwrap().duration.is_some());
        let timeline = DecodedTimeline::open(&path, 0).unwrap();
        assert!(timeline.end.is_none());
    }

    /// Kill: invent zero as an unknown source origin and clamp observed
    /// 500→520 ms presentation intervals to an unrelated 140 ms boundary.
    #[test]
    fn absent_origin_never_turns_a_duration_into_an_absolute_end() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline =
            DecodedTimeline::new(base, None, Some(MediaTime::new(140, base))).unwrap();
        timeline.push(frame(500, base));
        assert_eq!(
            timeline
                .push(frame(520, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            20
        );
        assert!(timeline.finish().unwrap().duration.is_none());
        timeline.clear();
        timeline.push(frame(900, base));
        assert!(
            timeline.finish().unwrap().duration.is_none(),
            "seek must not rebase the source duration"
        );
    }

    /// Kill: clamp a negative timeline origin to zero and extend the tail.
    #[test]
    fn negative_origin_and_a_short_final_interval_are_preserved() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline = DecodedTimeline::new(
            base,
            Some(MediaTime::new(-100, base)),
            Some(MediaTime::new(150, base)),
        )
        .unwrap();
        timeline.push(frame(-100, base));
        assert_eq!(
            timeline
                .push(frame(-30, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            70
        );
        assert_eq!(
            timeline
                .push(frame(40, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            70
        );
        assert_eq!(timeline.finish().unwrap().duration.unwrap().value, 10);
    }

    /// Kill: return EOF as soon as the one submitted picture has an EOS flag,
    /// or use an average cadence in place of this source's terminal boundary.
    #[test]
    fn a_single_picture_has_a_terminal_interval_and_only_one_eos_delivery() {
        let base = TimeBase::MICROSECONDS;
        let mut timeline = DecodedTimeline::new(
            base,
            Some(MediaTime::new(750_000, base)),
            Some(MediaTime::new(17_123, base)),
        )
        .unwrap();
        assert!(timeline.push(frame(750_000, base)).is_none());
        let single = timeline.finish().unwrap();
        assert_eq!(single.duration, Some(MediaTime::new(17_123, base)));
        assert!(timeline.finish().is_none());
    }

    /// Kill: infer the cropped terminal interval from the selected duration,
    /// discard it at the encoder, or lose one source tick via µs truncation.
    #[test]
    fn cropped_thirty_fps_picture_retains_the_last_interval_and_exact_end() {
        let source_clock = TimeBase::new_unchecked(1, 15_360);
        for (first, count) in [(0, 45), (9, 36)] {
            let mut timeline = DecodedTimeline::new(
                TimeBase::MICROSECONDS,
                Some(MediaTime::ZERO),
                Some(MediaTime::new(30_720, source_clock)),
            )
            .unwrap();
            let mut ledger = SubmittedTimings::default();
            let mut selected = Vec::new();
            for index in first..=first + count {
                let observed = index * 1_000_000 / 30;
                if let Some(mut ready) = timeline.push(frame(observed, TimeBase::MICROSECONDS)) {
                    assert_eq!(
                        ready.pts.value,
                        (index - 1) * 1_000_000 / 30,
                        "public decode PTS remains the exact MediaExtractor observation"
                    );
                    ready.pts.value -= first * 1_000_000 / 30;
                    ledger
                        .record(ready.pts.value, &ready, source_clock)
                        .unwrap();
                    selected.push(ledger.take(ready.pts.value).unwrap());
                }
            }
            assert_eq!(selected.len(), count as usize);
            assert!(
                selected
                    .iter()
                    .all(|timing| timing.duration == Some(MediaTime::new(512, source_clock)))
            );
            let last = selected.last().unwrap();
            assert_eq!(last.pts.value + last.duration.unwrap().value, count * 512);
            assert_eq!(
                (last.pts.value + last.duration.unwrap().value) * 48_000 / 15_360,
                count * 1_600
            );
        }
    }

    /// Kill: manufacture a final frame duration from prior cadence when the
    /// source has no declared video end, or stretch an existing duration.
    #[test]
    fn absent_track_end_does_not_guess_and_explicit_duration_is_retained() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline = DecodedTimeline::new(base, None, None).unwrap();
        timeline.push(frame(0, base));
        assert_eq!(
            timeline
                .push(frame(70, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            70
        );
        assert!(timeline.finish().unwrap().duration.is_none());
        let mut explicit = frame(0, base);
        explicit.duration = Some(MediaTime::new(10, base));
        timeline.push(explicit);
        assert_eq!(
            timeline
                .push(frame(70, base))
                .unwrap()
                .duration
                .unwrap()
                .value,
            10
        );
    }

    /// Kill: preserve a pre-seek picture and deliver it after resetting the codec.
    #[test]
    fn seek_reset_releases_pending_pixels_and_never_flushes_the_old_picture() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline =
            DecodedTimeline::new(base, Some(MediaTime::ZERO), Some(MediaTime::new(200, base)))
                .unwrap();
        let pending = frame(0, base);
        let FrameBuffer::Cpu(cpu) = &pending.buffer else {
            unreachable!()
        };
        let pixels = Arc::downgrade(&cpu.data);
        timeline.push(pending);
        assert!(pixels.upgrade().is_some());
        timeline.clear();
        assert!(pixels.upgrade().is_none());
        assert!(timeline.finish().is_none());
        assert!(timeline.push(frame(170, base)).is_none());
        let next = timeline.finish().unwrap();
        assert_eq!((next.pts.value, next.duration.unwrap().value), (170, 30));
    }

    /// Kill: retain the whole movie or leak the pending hardware picture when
    /// the owning decode future/session is abandoned.
    #[test]
    fn one_frame_ownership_is_bounded_and_drop_releases_the_cancelled_tail() {
        let base = TimeBase::MILLISECONDS;
        let mut timeline = DecodedTimeline::new(base, None, None).unwrap();
        let mut previous: Option<std::sync::Weak<[u8]>> = None;
        for index in 0..1_000 {
            let next = frame(index, base);
            let FrameBuffer::Cpu(cpu) = &next.buffer else {
                unreachable!()
            };
            let current = Arc::downgrade(&cpu.data);
            drop(timeline.push(next));
            if let Some(previous) = previous {
                assert!(std::sync::Weak::upgrade(&previous).is_none());
            }
            previous = Some(current);
        }
        let retained = previous.unwrap();
        assert!(retained.upgrade().is_some());
        drop(timeline);
        assert!(retained.upgrade().is_none());
    }

    /// Kill: derive the output PTS clock from a duration's units, or round
    /// duration independently and lose a terminal tick on a mixed-clock frame.
    #[test]
    fn the_supplied_encode_clock_controls_mixed_clock_interval_endpoints() {
        let mut ledger = SubmittedTimings::default();
        let mut input = frame(1_466_666, TimeBase::MICROSECONDS);
        input.duration = Some(MediaTime::new(1, TimeBase::new_unchecked(1, 30)));
        let output_clock = TimeBase::new_unchecked(1, 90_000);
        ledger.record(1_466_666, &input, output_clock).unwrap();
        let output = ledger.take(1_466_666).unwrap();
        assert_eq!(input.pts, MediaTime::new(1_466_666, TimeBase::MICROSECONDS));
        assert_eq!(output.pts, MediaTime::new(132_000, output_clock));
        assert_eq!(output.duration, Some(MediaTime::new(3_000, output_clock)));
        assert_eq!(output.pts.value + output.duration.unwrap().value, 135_000);
    }

    /// Kill: pair encoder output with submission order instead of the
    /// returned timestamp, or discard exact submitted durations at collect.
    #[test]
    fn reordered_encoder_output_retains_its_own_exact_submitted_timing() {
        let base = TimeBase::new_unchecked(1, 15_360);
        let mut ledger = SubmittedTimings::default();
        let mut expected = Vec::new();
        for index in 0..4 {
            let mut frame = frame(index * 512, base);
            frame.duration = Some(MediaTime::new(512 + index, base));
            let codec_pts = frame.pts.rescale(TimeBase::MICROSECONDS).value;
            ledger.record(codec_pts, &frame, base).unwrap();
            expected.push((
                codec_pts,
                SubmittedTiming {
                    pts: frame.pts,
                    duration: frame.duration,
                },
            ));
        }
        assert!(ledger.finish().is_err());
        for index in [0, 3, 1, 2] {
            assert_eq!(ledger.take(expected[index].0).unwrap(), expected[index].1);
        }
        ledger.finish().unwrap();
        assert!(ledger.take(expected[0].0).is_err());
    }

    /// Kill: let a codec that accepts and drops input accumulate metadata for
    /// the whole movie, failing only when finalize eventually notices it.
    #[test]
    fn dropped_encoder_output_fails_at_a_finite_outstanding_bound() {
        let base = TimeBase::MICROSECONDS;
        let mut ledger = SubmittedTimings::default();
        for value in 0..SubmittedTimings::MAX_OUTSTANDING as i64 {
            ledger.record(value, &frame(value, base), base).unwrap();
        }
        assert!(ledger.record(999, &frame(999, base), base).is_err());
        assert_eq!(ledger.outstanding, SubmittedTimings::MAX_OUTSTANDING);
        assert_eq!(ledger.by_codec_pts.len(), SubmittedTimings::MAX_OUTSTANDING);
        ledger.take(0).unwrap();
        ledger.record(999, &frame(999, base), base).unwrap();
        assert_eq!(ledger.outstanding, SubmittedTimings::MAX_OUTSTANDING);
        assert!(ledger.finish().is_err());
    }

    /// Kill: overwrite two submitted instants that quantize to one codec µs,
    /// or retain already-consumed metadata for the rest of a long encode.
    #[test]
    fn equal_codec_keys_do_not_overwrite_inputs_or_retain_consumed_timing() {
        let mut ledger = SubmittedTimings::default();
        for value in [1_000, 1_400] {
            ledger
                .record(
                    1,
                    &frame(value, TimeBase::NANOSECONDS),
                    TimeBase::NANOSECONDS,
                )
                .unwrap();
        }
        assert_eq!(ledger.take(1).unwrap().pts.value, 1_000);
        assert_eq!(ledger.take(1).unwrap().pts.value, 1_400);
        assert!(ledger.by_codec_pts.is_empty());
        assert!(ledger.take(99).is_err());
        ledger.finish().unwrap();
    }
}
