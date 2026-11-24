//! Display-order frame access for one video, over the platform decoder.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMFrameSource.swift` and
//! `EdgeTAMFrameTimelineIndex.swift`, including its anonymous temporary-file
//! timestamp index. The platform decoder delivers
//! display-order samples; their exact rational MediaTime timestamps stream
//! into a bounded writer. Reverse walks retain one RGB byte-budgeted batch.

use std::path::Path;

use super::frame_time;
use ferment_engine_core::{EngineError, ProgressSink, Result, VideoObjectSegmentationSourceInfo};
use ferment_engine_media::{DecodeSession, MediaError, MediaTime, TimeBase, VideoBackend};

use super::preprocessing::EdgeTamRgbFrame;
#[path = "frame_timeline.rs"]
mod timeline;
use timeline::{FrameTimeline, FrameTimelineWriter};

/// The decoded timeline of one video, in presentation order.
///
/// Exact rational timestamps are preserved on disk and through delivery. Neither
/// integer microsecond quantization nor a Double round trip identifies a sample.
#[derive(Debug)]
pub struct EdgeTamFrameSource {
    backend: VideoBackend,
    path: std::path::PathBuf,
    stream_index: u32,
    source_info: VideoObjectSegmentationSourceInfo,
    timeline: FrameTimeline,
    orientation: super::frame_orientation::DisplayOrientation,
    reverse_frames: Vec<EdgeTamRgbFrame>,
    decoder: Option<Box<dyn DecodeSession>>,
    /// The frame index the open decoder will deliver next, when it is
    /// positioned. `None` means "not positioned".
    cursor: Option<usize>,
    /// The first frame's timestamp, subtracted from every frame this source
    /// emits.
    ///
    /// iOS does the same (`EdgeTAMFrameSource.swift:268, 962`) and says why:
    /// a container's first retained sample is video time zero to a decoder even
    /// when an edit list places it after the asset timeline's zero. MPEG-TS
    /// conventionally starts near 1.4 s. Without this every emitted mask is
    /// late by the origin.
    origin: MediaTime,
    /// How many times the decoder has been seeked. Runtime evidence, in the
    /// spirit of iOS's `EdgeTAMSession.MemorySnapshot`: the sequential fast
    /// path is an OPTIMIZATION, so its output is identical whether it works or
    /// not, and this is the only thing that can tell a forward walk that reuses
    /// the decoder from one that seeks 18 000 times.
    seeks: usize,
}

impl EdgeTamFrameSource {
    /// Probe the source, decode its display-order timestamps, and answer the
    /// session's immutable geometry.
    ///
    /// The timeline pass writes twenty-four bytes per sample to an anonymous file;
    /// its resident buffer does not grow with the video's duration.
    pub async fn new(backend: VideoBackend, path: &Path) -> Result<Self> {
        Self::new_in(backend, path, &std::env::temp_dir()).await
    }

    /// App callers name their writable directory; Android's process default
    /// temporary directory is not writable from the application sandbox.
    pub async fn new_in(
        backend: VideoBackend,
        path: &Path,
        temporary_directory: &Path,
    ) -> Result<Self> {
        Self::new_in_with_cancellation(backend, path, temporary_directory, ProgressSink::noop())
            .await
    }

    pub async fn new_in_with_cancellation(
        backend: VideoBackend,
        path: &Path,
        temporary_directory: &Path,
        cancellation: ProgressSink,
    ) -> Result<Self> {
        check_cancellation(&cancellation)?;
        let info = backend.probe(path).await.map_err(map_media_error)?;
        check_cancellation(&cancellation)?;
        let stream = info
            .video_streams()
            .next()
            .cloned()
            .ok_or_else(|| EngineError::Decode("source has no video stream".into()))?;
        let mut decoder = backend
            .open_coded_decoder(path, stream.index)
            .await
            .map_err(map_media_error)?;
        check_cancellation(&cancellation)?;
        let mut timeline = FrameTimelineWriter::new(temporary_directory)?;
        let mut width = 0_usize;
        let mut height = 0_usize;
        while let Some(frame) = next_frame(decoder.as_mut(), &cancellation).await? {
            width = frame.width as usize;
            height = frame.height as usize;
            timeline.append(frame.pts)?;
        }
        check_cancellation(&cancellation)?;
        let timeline = timeline.finish()?;

        // iOS's rule, verbatim (`EdgeTAMFrameSource.swift:241-244`): the AVERAGE
        // rate over the indexed frames, and a refusal if it is not finite and
        // positive. The container's declared/nominal rate is never read — it is
        // wrong for every variable-frame-rate source, which is most phone
        // footage, and a one-frame source with no declared rate would otherwise
        // be handed a magic 30.
        let duration_seconds = info
            .duration
            .map(ferment_engine_media::MediaTime::seconds)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .ok_or_else(|| EngineError::Decode("Video has an invalid track duration.".into()))?;
        let frame_rate = timeline.len() as f64 / duration_seconds;
        if !frame_rate.is_finite() || frame_rate <= 0.0 {
            return Err(EngineError::Decode(format!(
                "Video has an invalid frame rate: {frame_rate}."
            )));
        }
        let origin = timeline.first();
        let orientation = super::frame_orientation::DisplayOrientation::new(
            width,
            height,
            stream.display_matrix,
        )?;
        let (width, height) = (orientation.width, orientation.height);

        Ok(Self {
            backend,
            path: path.to_owned(),
            stream_index: stream.index,
            source_info: VideoObjectSegmentationSourceInfo {
                frame_count: timeline.len(),
                frame_rate,
                width,
                height,
            },
            timeline,
            orientation,
            reverse_frames: Vec::new(),
            origin,
            decoder: None,
            cursor: None,
            seeks: 0,
        })
    }

    #[must_use]
    pub fn source_info(&self) -> VideoObjectSegmentationSourceInfo {
        self.source_info
    }

    /// The presentation time of one frame, from the timeline alone, relative
    /// to the source's own start — iOS's `EdgeTAMFrameTimelineIndex.time(at:)`
    /// less `trackStart`.
    ///
    /// The link between "the frame you asked for" and "the frame you got":
    /// [`Self::frame`] reads its timestamp back off the DECODER, so comparing
    /// the two is a real check and not a restatement.
    pub fn presentation_timestamp(&self, frame_index: usize) -> Result<MediaTime> {
        self.validate(frame_index)?;
        frame_time::subtract(self.timeline.time(frame_index)?, self.origin)
    }

    pub fn presentation_time(&self, frame_index: usize) -> Result<f64> {
        Ok(self.presentation_timestamp(frame_index)?.seconds())
    }

    /// Diagnostic projection; seek and exact sample matching use MediaTime.
    pub fn presentation_time_micros(&self, frame_index: usize) -> Result<i64> {
        Ok(self
            .presentation_timestamp(frame_index)?
            .rescale(TimeBase::MICROSECONDS)
            .value)
    }

    /// Seeks performed so far. See [`Self::seeks`].
    #[must_use]
    pub fn seek_count(&self) -> usize {
        self.seeks
    }

    #[must_use]
    pub fn has_decoder_resources(&self) -> bool {
        self.decoder.is_some() || !self.reverse_frames.is_empty()
    }

    /// Random access: seek to the frame's own timestamp, then take the first
    /// picture at or after it.
    pub async fn frame(&mut self, frame_index: usize) -> Result<EdgeTamRgbFrame> {
        self.frame_with_cancellation(frame_index, ProgressSink::noop())
            .await
    }

    pub async fn frame_with_cancellation(
        &mut self,
        frame_index: usize,
        cancellation: ProgressSink,
    ) -> Result<EdgeTamRgbFrame> {
        let result = self.read_frame(frame_index, &cancellation).await;
        if result.is_err() {
            self.cancel_decoder_resources();
        }
        result
    }

    async fn read_frame(
        &mut self,
        frame_index: usize,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamRgbFrame> {
        check_cancellation(cancellation)?;
        self.validate(frame_index)?;
        self.reverse_frames.clear();
        self.ensure_decoder(cancellation).await?;
        let target = self.timeline.time(frame_index)?;
        let decoder = self.decoder.as_mut().expect("decoder opened above");
        decoder.seek(target).await.map_err(map_media_error)?;
        check_cancellation(cancellation)?;
        self.seeks += 1;
        self.cursor = None;
        self.take_frame_at(frame_index, cancellation).await
    }

    /// Forward access reuses the next decoded picture; reverse access fills
    /// one forward-decoded, exact-PTS batch and consumes it in reverse order.
    pub async fn sequential_frame(
        &mut self,
        frame_index: usize,
        direction: ferment_engine_core::VideoObjectPropagationDirection,
    ) -> Result<EdgeTamRgbFrame> {
        self.sequential_frame_with_cancellation(frame_index, direction, ProgressSink::noop())
            .await
    }

    pub async fn sequential_frame_with_cancellation(
        &mut self,
        frame_index: usize,
        direction: ferment_engine_core::VideoObjectPropagationDirection,
        cancellation: ProgressSink,
    ) -> Result<EdgeTamRgbFrame> {
        let result = self
            .read_sequential_frame(frame_index, direction, &cancellation)
            .await;
        if result.is_err() {
            self.cancel_decoder_resources();
        }
        result
    }

    async fn read_sequential_frame(
        &mut self,
        frame_index: usize,
        direction: ferment_engine_core::VideoObjectPropagationDirection,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamRgbFrame> {
        use ferment_engine_core::VideoObjectPropagationDirection::Forward;
        check_cancellation(cancellation)?;
        self.validate(frame_index)?;
        if direction == Forward {
            self.reverse_frames.clear();
            if self.cursor == Some(frame_index) && self.decoder.is_some() {
                return self.take_frame_at(frame_index, cancellation).await;
            }
            return self.read_frame(frame_index, cancellation).await;
        }
        if self
            .reverse_frames
            .last()
            .is_none_or(|frame| frame.frame_index != frame_index)
        {
            self.cancel_decoder_resources();
            let count =
                Self::reverse_reader_batch_size(self.source_info.width, self.source_info.height);
            let first = frame_index.saturating_sub(count - 1);
            let batch = async {
                let mut frames = Vec::with_capacity(frame_index - first + 1);
                frames.push(self.read_frame(first, cancellation).await?);
                for index in first + 1..=frame_index {
                    frames.push(self.take_frame_at(index, cancellation).await?);
                }
                Ok::<_, EngineError>(frames)
            }
            .await;
            // Keep owned RGB only; native codec surfaces do not live for the
            // consumer's pause. Any failure drops the whole partial batch.
            self.decoder = None;
            self.cursor = None;
            check_cancellation(cancellation)?;
            self.reverse_frames = batch?;
        }
        check_cancellation(cancellation)?;
        self.reverse_frames
            .pop()
            .ok_or_else(|| EngineError::Decode("reverse reader produced no requested frame".into()))
    }

    /// Same 64 MiB RGB / 1,024-frame bounds as the current iOS reader.
    #[must_use]
    pub fn reverse_reader_batch_size(width: usize, height: usize) -> usize {
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            .filter(|bytes| *bytes > 0)
            .map_or(1, |bytes| ((64 * 1_048_576) / bytes).clamp(1, 1_024))
    }

    #[must_use]
    pub fn cached_reverse_byte_count(&self) -> usize {
        self.reverse_frames
            .iter()
            .map(|frame| frame.pixels.len())
            .sum()
    }

    /// Close the decoder. iOS calls this at reset and at the end of a
    /// propagation; the next access reopens.
    pub fn cancel_decoder_resources(&mut self) {
        self.decoder = None;
        self.cursor = None;
        self.reverse_frames.clear();
    }

    async fn ensure_decoder(&mut self, cancellation: &ProgressSink) -> Result<()> {
        check_cancellation(cancellation)?;
        if self.decoder.is_none() {
            self.decoder = Some(
                self.backend
                    .open_coded_decoder(&self.path, self.stream_index)
                    .await
                    .map_err(map_media_error)?,
            );
        }
        check_cancellation(cancellation)
    }

    async fn take_frame_at(
        &mut self,
        frame_index: usize,
        cancellation: &ProgressSink,
    ) -> Result<EdgeTamRgbFrame> {
        // ABSOLUTE: the timeline and the decoder both speak the container's own
        // clock. Only what leaves this function is origin-relative.
        let target = self.timeline.time(frame_index)?;
        let decoder = self.decoder.as_mut().ok_or_else(|| {
            EngineError::Decode("EdgeTAM frame source has no open decoder".into())
        })?;
        while let Some(frame) = next_frame(decoder.as_mut(), cancellation).await? {
            // `DecodeSession::seek`'s contract already says it discards
            // leading pictures until the target, and on VideoToolbox it does —
            // deleting this skip changes nothing there, measured. It stays for
            // the backends this lane cannot run (MediaCodec on a phone), where
            // a looser seek would otherwise return the previous sync sample.
            let comparison = frame_time::compare(frame.pts, target)?;
            if comparison == std::cmp::Ordering::Less {
                continue;
            }
            // OVERSHOOT is the case that corrupts silently, and iOS refuses it
            // in all three of its reader paths ("Random reader skipped the
            // indexed timestamp …"). Accepting a later picture under the
            // requested index would stamp the wrong frame AND leave the cursor
            // claiming a position the decoder is past, so every later frame of
            // the walk is wrong too.
            if comparison != std::cmp::Ordering::Equal {
                return Err(EngineError::Decode(format!(
                    "The decoder skipped the indexed timestamp {target:?} for frame \
                     {frame_index} and returned {:?}.",
                    frame.pts
                )));
            }
            let cpu = frame.to_cpu_rgb().map_err(map_media_error)?;
            let cpu = self.orientation.apply(&cpu)?;
            check_cancellation(cancellation)?;
            self.cursor = Some(frame_index + 1);
            return EdgeTamRgbFrame::new(
                frame_index,
                frame_time::subtract(frame.pts, self.origin)?,
                cpu.width as usize,
                cpu.height as usize,
                cpu.data.as_ref().to_vec(),
            )
            .map_err(Into::into);
        }
        self.cursor = None;
        Err(EngineError::Decode(format!(
            "the decoder delivered no picture at frame {frame_index}"
        )))
    }

    fn validate(&self, frame_index: usize) -> Result<()> {
        if frame_index >= self.source_info.frame_count {
            return Err(super::error::EdgeTamError::InvalidFrameIndex {
                index: frame_index,
                frame_count: self.source_info.frame_count,
            }
            .into());
        }
        Ok(())
    }
}

fn map_media_error(error: MediaError) -> EngineError {
    EngineError::VideoIo(error.to_string())
}

/// Native codec calls cannot be preempted safely; drain one call and discard
/// its result before another read, RGB allocation, or state commit.
async fn next_frame(
    decoder: &mut dyn DecodeSession,
    cancellation: &ProgressSink,
) -> Result<Option<ferment_engine_media::VideoFrame>> {
    check_cancellation(cancellation)?;
    let frame = decoder.decode_next().await.map_err(map_media_error)?;
    check_cancellation(cancellation)?;
    Ok(frame)
}

pub(super) fn check_cancellation(cancellation: &ProgressSink) -> Result<()> {
    if cancellation.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "frame_source_cancellation_tests.rs"]
mod cancellation_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "frame_source_boundaries_tests.rs"]
mod current_boundaries_tests;
