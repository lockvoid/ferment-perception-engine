//! Source audio packets for a picture-only render. The packet bytes and codec
//! parameters stay in their original format; only container time/trim changes.
//! PCM may be cut at complete sample-frame byte boundaries, without conversion.
use super::{check_av, init};
use crate::{MediaTime, Result, StreamInfo, TimeBase, VideoError};
use ffmpeg_next::{
    codec::{
        Parameters,
        packet::{
            Packet,
            traits::{Mut as PacketMut, Ref as PacketRef},
        },
    },
    ffi,
};
use std::{ffi::CStr, path::Path};

pub struct AudioPassthroughTrack {
    pub(crate) parameters: Parameters,
    pub(crate) time_base: TimeBase,
    pub(crate) presentation_end: MediaTime,
    pub(crate) metadata: Vec<(String, String)>,
    pub(crate) disposition: i32,
    stream_index: u32,
}
impl std::fmt::Debug for AudioPassthroughTrack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPassthroughTrack")
            .field("stream_index", &self.stream_index)
            .field("time_base", &self.time_base)
            .finish_non_exhaustive()
    }
}
impl AudioPassthroughTrack {
    pub fn stream_index(&self) -> u32 {
        self.stream_index
    }
}

/// Owns the native packet, including codec/container side data. No ffmpeg
/// pointer escapes engine-media; callers inspect bytes and time through values.
pub struct AudioPassthroughPacket {
    pub(crate) packet: Packet,
    pub(crate) time_base: TimeBase,
}
impl std::fmt::Debug for AudioPassthroughPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPassthroughPacket")
            .field("pts", &self.pts())
            .field("bytes", &self.data().len())
            .finish()
    }
}
impl AudioPassthroughPacket {
    pub fn data(&self) -> &[u8] {
        self.packet.data().unwrap_or_default()
    }
    pub fn pts(&self) -> Option<MediaTime> {
        self.packet.pts().map(|v| MediaTime::new(v, self.time_base))
    }
    pub fn dts(&self) -> Option<MediaTime> {
        self.packet.dts().map(|v| MediaTime::new(v, self.time_base))
    }
    pub fn duration(&self) -> MediaTime {
        MediaTime::new(self.packet.duration(), self.time_base)
    }
}

pub struct AudioPassthroughReader {
    input: ffmpeg_next::format::context::Input,
    track: AudioPassthroughTrack,
    window_start: i64,
    selected_start: i64,
    selected_end: i64,
    sample_rate: u32,
    pcm_bytes_per_frame: Option<usize>,
    finished: bool,
}
impl std::fmt::Debug for AudioPassthroughReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPassthroughReader")
            .field("track", &self.track)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}
impl AudioPassthroughReader {
    /// Intersect the current rendered window with this source track's own
    /// presentation range. A late soundtrack keeps its positive relative PTS.
    pub fn open(
        source: &Path,
        stream: &StreamInfo,
        start: MediaTime,
        duration: MediaTime,
    ) -> Result<Option<Self>> {
        init()?;
        if stream.kind != crate::MediaKind::Audio {
            return Err(VideoError::InvalidData(
                "passthrough source is not an audio track".into(),
            ));
        }
        let mut input =
            ffmpeg_next::format::input(source).map_err(|e| VideoError::Demux(e.to_string()))?;
        let raw_stream = input
            .stream(stream.index as usize)
            .ok_or_else(|| VideoError::InvalidData("source audio track vanished".into()))?;
        let base = TimeBase::new(
            i64::from(raw_stream.time_base().numerator()),
            i64::from(raw_stream.time_base().denominator()),
        )?;
        let window_start = ticks(start, base)?;
        let window_length = ticks(duration, base)?;
        let window_end = window_start
            .checked_add(window_length)
            .ok_or_else(overflow)?;
        if window_length <= 0 {
            return Err(VideoError::InvalidData(
                "audio source window must be positive".into(),
            ));
        }
        let source_start = ticks(stream.start_time.unwrap_or(MediaTime::ZERO), base)?;
        let source_end = match stream.duration {
            Some(length) => source_start
                .checked_add(ticks(length, base)?)
                .ok_or_else(overflow)?,
            None => window_end,
        };
        let selected_start = window_start.max(source_start);
        let selected_end = window_end.min(source_end);
        if selected_start >= selected_end {
            return Ok(None);
        }
        let original = raw_stream.parameters();
        let mut parameters = Parameters::new();
        // SAFETY: owned destination and borrowed source both live for this call.
        unsafe {
            if parameters.as_ptr().is_null() {
                return Err(VideoError::BackendUnavailable(
                    "audio codec parameter allocation failed".into(),
                ));
            }
            check_av(
                "avcodec_parameters_copy",
                ffi::avcodec_parameters_copy(parameters.as_mut_ptr(), original.as_ptr()),
            )?;
        }
        // SAFETY: the copied parameters are owned and valid.
        let raw = unsafe { &*parameters.as_ptr() };
        let sample_rate = u32::try_from(raw.sample_rate)
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| {
                VideoError::InvalidData("audio passthrough sample rate is missing".into())
            })?;
        let name = unsafe { CStr::from_ptr(ffi::avcodec_get_name(raw.codec_id)) }.to_string_lossy();
        let pcm_bytes_per_frame = if name.starts_with("pcm_") {
            let bytes = if raw.block_align > 0 {
                raw.block_align as usize
            } else {
                let bits = unsafe { ffi::av_get_bits_per_sample(raw.codec_id) };
                usize::try_from(bits).unwrap_or(0)
                    * usize::try_from(raw.ch_layout.nb_channels).unwrap_or(0)
                    / 8
            };
            if bytes == 0 {
                return Err(VideoError::InvalidData(
                    "PCM sample-frame size is missing".into(),
                ));
            }
            Some(bytes)
        } else {
            None
        };
        let metadata = raw_stream
            .metadata()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        let disposition = unsafe { (*raw_stream.as_ptr()).disposition };
        let track = AudioPassthroughTrack {
            parameters,
            time_base: base,
            presentation_end: MediaTime::new(selected_end - window_start, base),
            metadata,
            disposition,
            stream_index: stream.index,
        };
        // Seek only when cutting inside a source track. Its initial negative
        // priming packet is kept when rendering from the original beginning.
        if selected_start > source_start {
            check_av("av_seek_frame", unsafe {
                ffi::av_seek_frame(
                    input.as_mut_ptr(),
                    stream.index as i32,
                    selected_start,
                    ffi::AVSEEK_FLAG_BACKWARD,
                )
            })?;
        }
        Ok(Some(Self {
            input,
            track,
            window_start,
            selected_start,
            selected_end,
            sample_rate,
            pcm_bytes_per_frame,
            finished: false,
        }))
    }
    pub fn track(&self) -> &AudioPassthroughTrack {
        &self.track
    }
    pub fn next_packet(&mut self) -> Result<Option<AudioPassthroughPacket>> {
        self.next_packet_checked(|| Ok(()))
    }
    /// The processor checks cancellation around every native packet read,
    /// including skipped packets belonging to other source tracks.
    pub fn next_packet_checked(
        &mut self,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Option<AudioPassthroughPacket>> {
        check()?;
        if self.finished {
            return Ok(None);
        }
        loop {
            check()?;
            // One owner per read also releases packets from unselected streams.
            let mut packet = Packet::empty();
            let status =
                unsafe { ffi::av_read_frame(self.input.as_mut_ptr(), packet.as_mut_ptr()) };
            check()?;
            if status == ffi::AVERROR_EOF {
                self.finished = true;
                return Ok(None);
            }
            check_av("av_read_frame", status)?;
            if packet.stream() != self.track.stream_index as usize {
                continue;
            }
            let pts = packet
                .pts()
                .or(packet.dts())
                .ok_or_else(|| VideoError::InvalidData("audio packet has no timestamp".into()))?;
            let coded_duration = if packet.duration() > 0 {
                packet.duration()
            } else {
                let samples = unsafe {
                    ffi::av_get_audio_frame_duration2(
                        self.track.parameters.as_mut_ptr(),
                        i32::try_from(packet.size()).map_err(|_| overflow())?,
                    )
                };
                if samples <= 0 {
                    return Err(VideoError::InvalidData(
                        "audio packet has no sample duration".into(),
                    ));
                }
                ticks(
                    MediaTime::new(
                        i64::from(samples),
                        TimeBase::new(1, i64::from(self.sample_rate))?,
                    ),
                    self.track.time_base,
                )?
            };
            let end = pts.checked_add(coded_duration).ok_or_else(overflow)?;
            // Original priming remains a decode reference when the selected
            // range begins with the source. Its existing skip metadata, not a
            // guessed AAC frame size, identifies this packet.
            let (old_front, old_back) = skip_samples(&packet);
            let original_priming = old_front > 0 && end == self.selected_start;
            if end <= self.selected_start && !original_priming {
                continue;
            }
            if pts >= self.selected_end {
                self.finished = true;
                return Ok(None);
            }
            let clip_start = pts.max(self.selected_start);
            let clip_end = end.min(self.selected_end);
            let sample_base = TimeBase::new(1, i64::from(self.sample_rate))?;
            let front = ticks(
                MediaTime::new(clip_start - pts, self.track.time_base),
                sample_base,
            )?;
            let back = ticks(
                MediaTime::new(end - clip_end, self.track.time_base),
                sample_base,
            )?;
            if let Some(bytes_per_frame) = self.pcm_bytes_per_frame {
                let payload = packet.data().unwrap_or_default();
                if !payload.len().is_multiple_of(bytes_per_frame) {
                    return Err(VideoError::InvalidData(
                        "PCM packet is not whole sample frames".into(),
                    ));
                }
                let frames = payload.len() / bytes_per_frame;
                let from = usize::try_from(front).map_err(|_| overflow())?.min(frames);
                let to = frames.saturating_sub(usize::try_from(back).map_err(|_| overflow())?);
                if from >= to {
                    continue;
                }
                let mut clipped =
                    Packet::copy(&payload[from * bytes_per_frame..to * bytes_per_frame]);
                check_av("av_packet_copy_props", unsafe {
                    ffi::av_packet_copy_props(clipped.as_mut_ptr(), packet.as_ptr())
                })?;
                let shift = ticks(
                    MediaTime::new(from as i64, sample_base),
                    self.track.time_base,
                )?;
                clipped.set_pts(Some(pts.checked_add(shift).ok_or_else(overflow)?));
                clipped.set_dts(Some(
                    packet
                        .dts()
                        .unwrap_or(pts)
                        .checked_add(shift)
                        .ok_or_else(overflow)?,
                ));
                clipped.set_duration(ticks(
                    MediaTime::new((to - from) as i64, sample_base),
                    self.track.time_base,
                )?);
                packet = clipped;
            } else {
                // Preserve the complete encoded access unit. A negative first
                // timestamp becomes the MOV edit-list head trim; a shortened
                // terminal duration bounds its audible tail without re-encoding.
                packet.set_duration(clip_end - pts);
                set_skip_samples(
                    &mut packet,
                    old_front.max(u32::try_from(front).map_err(|_| overflow())?),
                    old_back.max(u32::try_from(back).map_err(|_| overflow())?),
                )?;
            }
            packet.set_pts(
                packet
                    .pts()
                    .or(Some(pts))
                    .map(|v| v.checked_sub(self.window_start).ok_or_else(overflow))
                    .transpose()?,
            );
            packet.set_dts(
                packet
                    .dts()
                    .or(Some(pts))
                    .map(|v| v.checked_sub(self.window_start).ok_or_else(overflow))
                    .transpose()?,
            );
            packet.set_position(-1);
            return Ok(Some(AudioPassthroughPacket {
                packet,
                time_base: self.track.time_base,
            }));
        }
    }
}
fn overflow() -> VideoError {
    VideoError::InvalidData("audio passthrough timestamp overflow".into())
}
fn ticks(time: MediaTime, base: TimeBase) -> Result<i64> {
    if time.time_base.numerator <= 0
        || time.time_base.denominator <= 0
        || base.numerator <= 0
        || base.denominator <= 0
    {
        return Err(overflow());
    }
    let numerator = i128::from(time.value)
        .checked_mul(i128::from(time.time_base.numerator))
        .and_then(|v| v.checked_mul(i128::from(base.denominator)))
        .ok_or_else(overflow)?;
    let denominator = i128::from(time.time_base.denominator)
        .checked_mul(i128::from(base.numerator))
        .ok_or_else(overflow)?;
    i64::try_from(numerator / denominator).map_err(|_| overflow())
}
fn skip_samples(packet: &Packet) -> (u32, u32) {
    let mut size = 0;
    let data = unsafe {
        ffi::av_packet_get_side_data(
            packet.as_ptr(),
            ffi::AVPacketSideDataType::AV_PKT_DATA_SKIP_SAMPLES,
            &mut size,
        )
    };
    if data.is_null() || size < 8 {
        return (0, 0);
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, 8) };
    (
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    )
}
fn set_skip_samples(packet: &mut Packet, start: u32, end: u32) -> Result<()> {
    if start == 0 && end == 0 {
        return Ok(());
    }
    let data = unsafe {
        ffi::av_packet_new_side_data(
            packet.as_mut_ptr(),
            ffi::AVPacketSideDataType::AV_PKT_DATA_SKIP_SAMPLES,
            10,
        )
    };
    if data.is_null() {
        return Err(VideoError::BackendUnavailable(
            "audio trim side-data allocation failed".into(),
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts_mut(data, 10) };
    bytes[..4].copy_from_slice(&start.to_le_bytes());
    bytes[4..8].copy_from_slice(&end.to_le_bytes());
    bytes[8..].fill(0);
    Ok(())
}
