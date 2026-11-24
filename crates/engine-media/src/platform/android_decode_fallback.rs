//! One-way hardware decoder recovery. Only caller-delivered pictures advance
//! the cursor. Recovery starts at the original unread source; explicit seeks
//! switch to software before seeking, so no hardware seek origin is inferred.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    ColorInfo, DecodeOptions, DecodeSession, MediaTime, Result, TimeBase, TransferCharacteristic,
    VideoError, VideoFrame, VideoIo,
};

pub(super) fn decoder_refusal(error: &VideoError) -> bool {
    match error {
        VideoError::UnsupportedCodec(_) | VideoError::BackendUnavailable(_) => true,
        VideoError::Platform { operation, .. } => {
            operation.starts_with("AMediaCodec") || operation.starts_with("AImage")
        }
        _ => false,
    }
}

/// Flexible/vendor ByteBuffer tags do not describe a plane layout. SDR uses
/// real AImage planes, or FFmpeg when the decoder cannot provide that surface.
pub(super) fn require_supported_sink(
    has_surface: bool,
    source_color: ColorInfo,
    tone_mapped: bool,
) -> Result<()> {
    let hdr = matches!(
        source_color.transfer,
        TransferCharacteristic::Hlg | TransferCharacteristic::Pq
    );
    if !has_surface && (tone_mapped || !hdr) {
        return Err(VideoError::UnsupportedCodec(
            "SDR decoder has no image surface; use libav instead of guessing ByteBuffer planes"
                .into(),
        ));
    }
    Ok(())
}

/// NDK AImageCropRect right/bottom are EXCLUSIVE. Only a complete image can
/// enter the existing plane/GPU path; FFmpeg owns non-full visible rectangles.
pub(super) fn require_full_image_crop(
    width: u32,
    height: u32,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> Result<()> {
    if width == 0
        || height == 0
        || left != 0
        || top != 0
        || i64::from(right) != i64::from(width)
        || i64::from(bottom) != i64::from(height)
    {
        return Err(VideoError::UnsupportedCodec(format!(
            "AImage crop ({left},{top})..({right},{bottom}) is not the full {width}x{height} image",
        )));
    }
    Ok(())
}

/// MediaFormat uses INCLUSIVE right/bottom, unlike AImageCropRect. Absent
/// keys mean the full coded picture; partially specified crops are refused.
pub(super) fn require_full_buffer_crop(
    width: i32,
    height: i32,
    left: Option<i32>,
    top: Option<i32>,
    right: Option<i32>,
    bottom: Option<i32>,
) -> Result<()> {
    let absent = [left, top, right, bottom].iter().all(Option::is_none);
    if width <= 0
        || height <= 0
        || (!absent
            && (left != Some(0)
                || top != Some(0)
                || right != Some(width - 1)
                || bottom != Some(height - 1)))
    {
        return Err(VideoError::UnsupportedCodec(format!(
            "MediaFormat crop {left:?},{top:?}..{right:?},{bottom:?} is not the full {width}x{height} coded picture",
        )));
    }
    Ok(())
}

/// Rotation comes from the exact Extractor format passed to MediaCodec, not
/// a separately interpreted container matrix. Coded doors never rotate. A
/// non-quarter-turn cannot be reproduced by the exact pixel permutation, so
/// that source stays on software from open, with its existing coded behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SoftwareOrientation {
    quarter_turns: u8,
    pub(super) hardware_allowed: bool,
}
impl SoftwareOrientation {
    pub(super) const CODED: Self = Self {
        quarter_turns: 0,
        hardware_allowed: true,
    };

    pub(super) fn from_track(coded: bool, rotation_degrees: i32) -> Self {
        if coded {
            return Self::CODED;
        }
        if rotation_degrees % 90 != 0 {
            return Self {
                quarter_turns: 0,
                hardware_allowed: false,
            };
        }
        Self {
            quarter_turns: (rotation_degrees.rem_euclid(360) / 90) as u8,
            // These are the four documented MediaFormat KEY_ROTATION values.
            hardware_allowed: matches!(rotation_degrees, 0 | 90 | 180 | 270),
        }
    }

    fn options(self, mut options: DecodeOptions) -> DecodeOptions {
        if !self.quarter_turns.is_multiple_of(2) {
            options.output_size = options.output_size.map(|(width, height)| (height, width));
        }
        options
    }

    fn frame(self, frame: VideoFrame) -> Result<VideoFrame> {
        frame.rotated_clockwise(self.quarter_turns)
    }
}

#[derive(Debug, Default)]
struct Delivered {
    last: Option<(i64, u64)>,
    out_of_order: bool,
}

impl Delivered {
    fn record(&mut self, frame: &VideoFrame) {
        let pts = frame.pts.rescale(TimeBase::MICROSECONDS).value;
        self.last = Some(match self.last {
            Some((previous, count)) if previous == pts => (pts, count + 1),
            Some((previous, _)) => {
                self.out_of_order |= pts < previous;
                (pts, 1)
            }
            None => (pts, 1),
        });
    }

    fn resume(&self) -> Result<Resume> {
        if self.out_of_order {
            return Err(VideoError::InvalidData(
                "cannot resume a decoder that delivered out-of-order timestamps".into(),
            ));
        }
        Ok(Resume {
            boundary: self.last.map(|(pts, _)| pts),
            remaining: self.last.map_or(0, |(_, count)| count),
        })
    }
}

#[derive(Debug)]
struct Resume {
    boundary: Option<i64>,
    remaining: u64,
}

impl Resume {
    fn discard(&mut self, frame: &VideoFrame) -> Result<bool> {
        let pts = frame.pts.rescale(TimeBase::MICROSECONDS).value;
        if let Some(boundary) = self.boundary {
            if pts < boundary {
                return Ok(true);
            }
            if self.remaining > 0 {
                if pts != boundary {
                    return Err(self.missing_boundary());
                }
                self.remaining -= 1;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn missing_boundary(&self) -> VideoError {
        VideoError::InvalidData(format!(
            "software recovery did not reproduce the delivered timestamp {:?} ({} equal-PTS pictures remaining)",
            self.boundary, self.remaining,
        ))
    }
}

/// Holds one decoder at a time. Software reopens the ORIGINAL source, with the
/// matching orientation/size request; it never reuses tone-mapped metadata or a
/// hardware extractor position. Explicit seeks use software exclusively.
/// Android display rotation uses the existing exact sample permutation only.
#[derive(Debug)]
pub(super) struct RecoveringDecoder {
    source: PathBuf,
    stream_index: u32,
    options: DecodeOptions,
    orientation: SoftwareOrientation,
    software_io: Arc<dyn VideoIo>,
    primary: Option<Box<dyn DecodeSession>>,
    software: Option<Box<dyn DecodeSession>>,
    delivered: Delivered,
    resume: Option<Resume>,
    pending_seek: Option<MediaTime>,
    minimum: Option<i64>,
    at_unread_start: bool,
    terminal: bool,
}

impl RecoveringDecoder {
    pub(super) fn new(
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
        orientation: SoftwareOrientation,
        primary: Option<Box<dyn DecodeSession>>,
        software_io: Arc<dyn VideoIo>,
    ) -> Self {
        Self {
            source: source.to_owned(),
            stream_index,
            options,
            orientation,
            software_io,
            primary,
            software: None,
            delivered: Delivered::default(),
            resume: None,
            pending_seek: None,
            minimum: None,
            at_unread_start: true,
            terminal: false,
        }
    }

    pub(super) async fn software(
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
        orientation: SoftwareOrientation,
        software_io: Arc<dyn VideoIo>,
    ) -> Result<Self> {
        let mut decoder = Self::new(
            source,
            stream_index,
            options,
            orientation,
            None,
            software_io,
        );
        decoder.prepare_software().await?;
        Ok(decoder)
    }

    fn begin_recovery(&mut self, error: &VideoError) -> Result<()> {
        // Close codec/reader/lookahead before allocating software state. A
        // caller-held AImage retains its own reader Arc until that frame drops.
        self.primary = None;
        log::debug!(target: "ferment_engine_media::android", "{}: {error}; replaying original source through libavcodec", self.source.display());
        let resume = self.delivered.resume().inspect_err(|_| {
            self.terminal = true;
        })?;
        // No timestamp seek: even -1us rounds to zero in a coarser source
        // clock and can land after the first equal-PTS picture. Start unread.
        self.pending_seek = None;
        self.minimum = None;
        self.resume = Some(resume);
        Ok(())
    }

    async fn prepare_software(&mut self) -> Result<()> {
        if self.software.is_none() {
            // Libav supplies coded pixels. Apply Android's display request
            // after decoding, keeping this policy out of the generic backend.
            let decoder = self
                .software_io
                .open_coded_decoder_with(
                    &self.source,
                    self.stream_index,
                    self.orientation.options(self.options),
                )
                .await?;
            self.software = Some(decoder);
        }
        if let Some(target) = self.pending_seek {
            self.software
                .as_mut()
                .expect("opened decoder")
                .seek(target)
                .await?;
            self.pending_seek = None;
        }
        Ok(())
    }

    async fn software_next(&mut self) -> Result<Option<VideoFrame>> {
        self.prepare_software().await?;
        loop {
            let next = self
                .software
                .as_mut()
                .expect("opened decoder")
                .decode_next()
                .await?;
            let Some(frame) = next else {
                if let Some(resume) = &self.resume
                    && resume.remaining > 0
                {
                    return Err(resume.missing_boundary());
                }
                self.resume = None;
                self.minimum = None;
                return Ok(None);
            };
            if self
                .resume
                .as_mut()
                .map(|resume| resume.discard(&frame))
                .transpose()?
                .unwrap_or(false)
                || self.minimum.is_some_and(|minimum| {
                    frame.pts.rescale(TimeBase::MICROSECONDS).value < minimum
                })
            {
                drop(frame);
                // Keep replay progress on self before yielding: cancelling this
                // future neither leaks the reader nor loses the replay cursor.
                let mut yielded = false;
                std::future::poll_fn(|context| {
                    if yielded {
                        std::task::Poll::Ready(())
                    } else {
                        yielded = true;
                        context.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
                continue;
            }
            self.resume = None;
            self.minimum = None;
            return self.orientation.frame(frame).map(Some);
        }
    }

    fn finish_call<T>(&mut self, result: Result<T>) -> Result<T> {
        if result.is_err() {
            self.terminal = true;
            self.primary = None;
            self.software = None;
        }
        result
    }

    fn check_live(&self) -> Result<()> {
        if self.terminal {
            Err(VideoError::BackendUnavailable(
                "decoder closed after a decode/recovery failure".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[async_trait(?Send)]
impl DecodeSession for RecoveringDecoder {
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        self.check_live()?;
        let target = target.rescale(TimeBase::MICROSECONDS);
        if self.at_unread_start && target.value <= 0 {
            // Keep the original unread start, including any leading negative PTS.
            return Ok(());
        }
        self.at_unread_start = false;
        // A hardware seek can land partway through an equal-PTS group. Do not
        // infer that group's identity later: move to software before seeking.
        self.primary = None;
        self.delivered = Delivered::default();
        self.resume = None;
        self.minimum = Some(target.value);
        self.pending_seek = Some(target);
        let result = self.prepare_software().await;
        self.finish_call(result)
    }

    async fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        self.check_live()?;
        self.at_unread_start = false;
        if let Some(primary) = &mut self.primary {
            match primary.decode_next().await {
                Ok(frame) => {
                    if let Some(frame) = &frame {
                        self.delivered.record(frame);
                    }
                    return Ok(frame);
                }
                // Invalid decoded plane/format data is also a platform output
                // refusal. File I/O/demux errors are not retried here.
                Err(error)
                    if decoder_refusal(&error) || matches!(error, VideoError::InvalidData(_)) =>
                {
                    self.begin_recovery(&error)?
                }
                Err(error) => return self.finish_call(Err(error)),
            }
        }
        let result = self.software_next().await;
        self.finish_call(result)
    }
}

#[cfg(test)]
#[path = "android_decode_fallback_tests.rs"]
mod tests;
