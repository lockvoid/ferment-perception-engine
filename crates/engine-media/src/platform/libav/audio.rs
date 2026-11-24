//! Compressed audio through `libavcodec` and `libswresample`.
//!
//! Ruling 10. Decode is libavcodec's for every codec in the build; encode is
//! the first of `aac_at` (macOS), `aac_mf` (Windows) and libavcodec's own `aac`
//! that opens and reports its priming.
//!
//! The one thing this module is really about is sample counts.
//! [`PcmAudioFrame`] is interleaved native-endian `f32` at the SOURCE rate,
//! and that layout is its law rather than libswresample's default — a decoder
//! that hands back planar, or resamples to 48 kHz because that is what the
//! output device wants, is wrong in a way that only shows up as drift halfway
//! through a long file. Encoder priming is libavcodec's to apply: it consumes
//! `AV_PKT_DATA_SKIP_SAMPLES` itself unless asked not to, so the frames that
//! come out of here are already the audible program.

use std::path::Path;

use async_trait::async_trait;
use ffmpeg_next::codec::packet::traits::{Mut as PacketMut, Ref as PacketRef};
use ffmpeg_next::ffi;

use super::{check_av, init, probe};
use crate::{
    AudioDecodeSession, AudioEncodeSession, AudioEncoderSettings, AudioIo, AudioTiming, Codec,
    DecoderConfiguration, EncodedAudio, EncodedPacket, MediaInfo, MediaTime, Packetization,
    PcmAudioFrame, Result, TimeBase, VideoError,
};

/// The audio backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct LibavAudioIo;

impl LibavAudioIo {
    /// Open a worker-owned verified stream. No path/URL, whole-file copy or
    /// secondary resource is opened; the existing sample-accurate codec stays.
    pub fn open_stream(
        &self,
        source: Box<dyn crate::MediaInput>,
        stream_index: u32,
    ) -> Result<Box<dyn AudioDecodeSession>> {
        let mut input = super::input::Input::stream(source)?;
        let metadata = probe::describe(&mut input);
        input.check()?;
        let metadata = metadata?;
        Ok(Box::new(LibavAudioDecoder::from_input(
            input.reopen()?,
            stream_index,
            metadata,
        )?))
    }
}

#[async_trait(?Send)]
impl AudioIo for LibavAudioIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        probe::probe(source)
    }

    async fn open_audio_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn AudioDecodeSession>> {
        Ok(Box::new(LibavAudioDecoder::open(source, stream_index)?))
    }

    async fn open_audio_encoder(
        &self,
        settings: AudioEncoderSettings,
    ) -> Result<Box<dyn AudioEncodeSession>> {
        Ok(Box::new(LibavAudioEncoder::open(settings)?))
    }
}

struct CodecContext(*mut ffi::AVCodecContext);

impl CodecContext {
    fn audio_decoder(
        parameters: &ffmpeg_next::codec::Parameters,
        time_base: ffmpeg_next::Rational,
    ) -> Result<Self> {
        // SAFETY: parameters owns a live codecpar; decoder is static or null.
        let codec_id = unsafe { (*parameters.as_ptr()).codec_id };
        let decoder = unsafe { ffi::avcodec_find_decoder(codec_id) };
        if decoder.is_null() {
            return Err(VideoError::UnsupportedCodec(format!(
                "no audio decoder for {codec_id:?}"
            )));
        }
        // SAFETY: allocates a new context; RAII releases it on every error.
        let raw = unsafe { ffi::avcodec_alloc_context3(decoder) };
        if raw.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an audio decoder context".into(),
            ));
        }
        let context = Self(raw);
        // SAFETY: copy valid stream parameters into our unopened context.
        check_av("avcodec_parameters_to_context", unsafe {
            ffi::avcodec_parameters_to_context(raw, parameters.as_ptr())
        })?;
        // Packets keep the demuxer's timestamps. Partial AAC skip-samples must
        // advance PTS in that unit, not a default codec time base.
        unsafe {
            (*raw).pkt_timebase = ffi::AVRational {
                num: time_base.numerator(),
                den: time_base.denominator(),
            };
        }
        // SAFETY: matching decoder and fresh context; no dictionary ownership.
        check_av("avcodec_open2", unsafe {
            ffi::avcodec_open2(raw, decoder, std::ptr::null_mut())
        })?;
        Ok(context)
    }
}

impl Drop for CodecContext {
    fn drop(&mut self) {
        // SAFETY: this type is the sole owner.
        unsafe { ffi::avcodec_free_context(&mut self.0) };
    }
}

/// A resampler into interleaved `f32`, which is the only layout
/// [`PcmAudioFrame`] has.
struct Resampler(*mut ffi::SwrContext);

impl Drop for Resampler {
    fn drop(&mut self) {
        // SAFETY: sole owner.
        unsafe { ffi::swr_free(&mut self.0) };
    }
}

impl Resampler {
    /// # Safety
    ///
    /// `frame` must be a live decoded audio frame.
    unsafe fn convert_to_interleaved_f32(
        &mut self,
        frame: *const ffi::AVFrame,
        channels: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: the caller guarantees a live frame.
        let samples = unsafe { (*frame).nb_samples };
        if samples <= 0 {
            return Ok(Vec::new());
        }
        let mut output = vec![0f32; samples as usize * channels];
        let pointer = output.as_mut_ptr().cast::<u8>();
        // SAFETY: the context is configured for this frame's layout; the
        // output buffer holds exactly `samples * channels` f32.
        let written = unsafe {
            ffi::swr_convert(
                self.0,
                (&raw const pointer).cast_mut(),
                samples,
                (*frame).extended_data as *const *const u8,
                samples,
            )
        };
        let written = check_av("swr_convert", written)?;
        output.truncate(written as usize * channels);
        Ok(output)
    }
}

// -------------------------------------------------------------------------
// Decode

struct LibavAudioDecoder {
    input: super::input::Input,
    codec: CodecContext,
    /// Built from the first decoded FRAME, not from the codec context at open
    /// time. Several decoders — AAC among them — do not settle `sample_fmt`
    /// until they have decoded something, so a resampler configured at open
    /// can be reading a format the frames are not in, and silently converts
    /// noise.
    resampler: Option<Resampler>,
    stream_index: usize,
    time_base: TimeBase,
    sample_rate: u32,
    channels: u16,
    /// Where the next frame starts, counted in samples — `None` until the
    /// first frame after opening or seeking establishes it.
    ///
    /// The container's own timestamps set the origin and nothing else. A
    /// Matroska track is timed in milliseconds, so converting each frame's
    /// timestamp to samples rounds, and consecutive frames stop meeting: a
    /// caller stitching PCM into a buffer gets a gap or an overlap every few
    /// frames. Counting samples cannot drift, because samples are the unit.
    next_start: Option<i64>,
    /// Where the audible program ends, in samples — the container's own
    /// declaration.
    ///
    /// libavcodec applies the LEADING trim itself (it consumes
    /// `AV_PKT_DATA_SKIP_SAMPLES`), but the MOV demuxer never emits an end
    /// discard, so the encoder's padding would otherwise arrive as real
    /// samples: 128 of near-silence after a one-second tone, enough to shift
    /// every window of a waveform envelope. Trimmed here, as ruling 10 asks.
    audible_limit: Option<i64>,
    /// Audible origin; AAC priming may precede it, including below PTS zero.
    origin: i64,
    seek_target: Option<i64>,
    draining: bool,
    finished: bool,
}

impl std::fmt::Debug for LibavAudioDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibavAudioDecoder")
            .field("stream_index", &self.stream_index)
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl LibavAudioDecoder {
    fn open(source: &Path, stream_index: u32) -> Result<Self> {
        let input = super::input::Input::path(source)?;
        Self::from_input(input, stream_index, probe::probe(source)?)
    }

    fn from_input(
        input: super::input::Input,
        stream_index: u32,
        metadata: MediaInfo,
    ) -> Result<Self> {
        let stream = input
            .streams()
            .find(|stream| stream.index() == stream_index as usize)
            .ok_or_else(|| VideoError::InvalidData(format!("no audio stream {stream_index}")))?;
        let parameters = stream.parameters();
        // SAFETY: `codecpar` lives as long as the open context.
        let raw = unsafe { &*parameters.as_ptr() };
        if raw.codec_type != ffi::AVMediaType::AVMEDIA_TYPE_AUDIO {
            return Err(VideoError::InvalidData(format!(
                "stream {stream_index} is not audio"
            )));
        }
        let time_base = TimeBase::new(
            i64::from(stream.time_base().numerator()),
            i64::from(stream.time_base().denominator()),
        )?;

        let context = CodecContext::audio_decoder(&parameters, stream.time_base())?;

        // SAFETY: the context is open, so its layout fields are settled.
        let (sample_rate, channels) = unsafe {
            let raw = &*context.0;
            (
                raw.sample_rate.max(0) as u32,
                u16::try_from(raw.ch_layout.nb_channels).unwrap_or(0),
            )
        };
        if sample_rate == 0 || channels == 0 {
            return Err(VideoError::InvalidData(
                "audio stream declares no rate or channels".into(),
            ));
        }

        // The probe declares a LENGTH, while decoded PTS and seek targets
        // use the container's presentation timeline. Include a delayed
        // track's origin before treating the length as an absolute endpoint.
        // libav's MOV decoder drops leading priming but does not apply the
        // terminal edit, so this endpoint still owns the final sample trim.
        let stream = metadata.stream(stream_index);
        let audible_length = stream
            .and_then(|stream| stream.audio_timing)
            .and_then(|timing| timing.audible_samples)
            .and_then(|samples| i64::try_from(samples).ok());
        let origin = stream
            .and_then(|stream| stream.start_time)
            .unwrap_or(MediaTime::ZERO)
            .rescale(TimeBase::new_unchecked(1, i64::from(sample_rate)))
            .value;
        let audible_limit = audible_length
            .map(|length| {
                origin.checked_add(length).ok_or_else(|| {
                    VideoError::InvalidData("audio presentation endpoint overflow".into())
                })
            })
            .transpose()?;

        Ok(Self {
            input,
            codec: context,
            resampler: None,
            stream_index: stream_index as usize,
            time_base,
            sample_rate,
            channels,
            next_start: None,
            audible_limit,
            origin,
            seek_target: None,
            draining: false,
            finished: false,
        })
    }

    /// How far before a seek target to start decoding.
    ///
    /// `seek_preroll` when the codec declares one (Opus does). AAC needs window
    /// overlap; MP3 also needs its bit reservoir, which spans more than two
    /// low-bitrate VBR packets. Keep that warm-up in the decoder, not its clients.
    fn seek_preroll_samples(&self) -> i64 {
        // SAFETY: the context is open; both are plain fields.
        let (declared, frame_size, codec_id) = unsafe {
            let raw = &*self.codec.0;
            (
                i64::from(raw.seek_preroll),
                i64::from(raw.frame_size),
                raw.codec_id,
            )
        };
        let frames = if frame_size > 0 { frame_size } else { 1_152 };
        let warmup = if codec_id == ffi::AVCodecID::AV_CODEC_ID_MP3 {
            16
        } else {
            2
        };
        declared.max(frames * warmup)
    }

    /// A resampler matching this frame's own layout, built once.
    ///
    /// # Safety
    ///
    /// `frame` must be a live decoded audio frame.
    unsafe fn resampler_for(&mut self, frame: *const ffi::AVFrame) -> Result<&mut Resampler> {
        if self.resampler.is_none() {
            // SAFETY: the caller guarantees a live frame.
            let (layout, format, rate) =
                unsafe { ((*frame).ch_layout, (*frame).format, (*frame).sample_rate) };
            let format = unsafe { std::mem::transmute::<i32, ffi::AVSampleFormat>(format) };
            let mut raw: *mut ffi::SwrContext = std::ptr::null_mut();
            // Rate and layout unchanged, format forced to interleaved f32:
            // this is a format conversion, NOT a resample. `PcmAudioFrame` is
            // at the source rate by definition.
            // SAFETY: a valid out-pointer and a layout taken from the frame.
            check_av("swr_alloc_set_opts2", unsafe {
                ffi::swr_alloc_set_opts2(
                    &mut raw,
                    &layout,
                    ffi::AVSampleFormat::AV_SAMPLE_FMT_FLT,
                    rate,
                    &layout,
                    format,
                    rate,
                    0,
                    std::ptr::null_mut(),
                )
            })?;
            let resampler = Resampler(raw);
            // SAFETY: just configured.
            check_av("swr_init", unsafe { ffi::swr_init(resampler.0) })?;
            self.resampler = Some(resampler);
        }
        Ok(self.resampler.as_mut().expect("just stored"))
    }

    fn feed(&mut self) -> Result<()> {
        if self.draining {
            self.finished = true;
            return Ok(());
        }
        loop {
            // Also release packets from other interleaved streams before the
            // next demux read; selected/EOF/error paths use the same RAII owner.
            let mut packet = ffmpeg_next::codec::packet::Packet::empty();
            // SAFETY: both pointers live.
            let status =
                unsafe { ffi::av_read_frame(self.input.as_mut_ptr(), packet.as_mut_ptr()) };
            self.input.check()?;
            if status == ffi::AVERROR_EOF {
                self.draining = true;
                // SAFETY: a null packet is the documented flush.
                check_av("avcodec_send_packet(flush)", unsafe {
                    ffi::avcodec_send_packet(self.codec.0, std::ptr::null())
                })?;
                return Ok(());
            }
            check_av("av_read_frame", status)?;
            if packet.stream() == self.stream_index {
                // SAFETY: both live; the codec retains needed buffer refs,
                // while this caller still owns and drops the Packet.
                check_av("avcodec_send_packet", unsafe {
                    ffi::avcodec_send_packet(self.codec.0, packet.as_ptr())
                })?;
                return Ok(());
            }
        }
    }
}

#[async_trait(?Send)]
impl AudioDecodeSession for LibavAudioDecoder {
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        self.input.check()?;
        let rate_base = TimeBase::new_unchecked(1, i64::from(self.sample_rate));
        let target_sample = target.rescale(rate_base).value;
        // Land BEFORE the target and decode into it. A compressed format's
        // first frame after a flush is not yet right — AAC's window overlaps
        // its predecessor, MP3's polyphase filter carries state — so seeking
        // straight to the target returns samples that differ from what a
        // sequential decode produced at the same instant. Discarding a couple
        // of coded frames' worth is what makes a scrub match playback.
        let preroll = self.seek_preroll_samples();
        // AAC's overlap/encoder priming packet can have negative PTS. Clamping
        // that seek to zero loses the beginning even when the target is exact.
        // SAFETY: immutable codec ID from our live context.
        let aac = unsafe { (*self.codec.0).codec_id } == ffi::AVCodecID::AV_CODEC_ID_AAC;
        let minimum = if aac {
            self.origin.saturating_sub(preroll)
        } else {
            0
        };
        let landing = MediaTime::new(
            target_sample.saturating_sub(preroll).max(minimum),
            rate_base,
        )
        .rescale(self.time_base);
        // SAFETY: the context is live and the index is one of its own.
        check_av("av_seek_frame", unsafe {
            ffi::av_seek_frame(
                self.input.as_mut_ptr(),
                self.stream_index as i32,
                landing.value,
                ffi::AVSEEK_FLAG_BACKWARD,
            )
        })?;
        if aac {
            // AAC flush preserves state (notably noise synthesis) from earlier
            // playback. Reset the codec on this worker so repeated seeks have
            // the same initial state as a fresh decode, including at frame zero.
            let stream = self
                .input
                .streams()
                .find(|s| s.index() == self.stream_index)
                .ok_or_else(|| VideoError::InvalidData("Audio stream disappeared".into()))?;
            self.codec = CodecContext::audio_decoder(&stream.parameters(), stream.time_base())?;
            self.resampler = None;
        } else {
            // SAFETY: live and open.
            unsafe { ffi::avcodec_flush_buffers(self.codec.0) };
        }
        self.seek_target = Some(target_sample);
        self.next_start = None;
        self.draining = false;
        self.finished = false;
        self.input.check()
    }

    async fn decode_next(&mut self) -> Result<Option<PcmAudioFrame>> {
        let channels = usize::from(self.channels);
        let rate_base = TimeBase::new_unchecked(1, i64::from(self.sample_rate));
        loop {
            self.input.check()?;
            let mut frame = ffmpeg_next::frame::Audio::empty();
            // SAFETY: both live; the frame is unreferenced.
            let status = unsafe { ffi::avcodec_receive_frame(self.codec.0, frame.as_mut_ptr()) };
            if status == ffi::AVERROR(ffi::EAGAIN) {
                if self.finished {
                    return Ok(None);
                }
                self.feed()?;
                continue;
            }
            if status == ffi::AVERROR_EOF {
                self.finished = true;
                return Ok(None);
            }
            check_av("avcodec_receive_frame", status)?;

            // SAFETY: a live decoded frame.
            let samples = unsafe {
                let raw = frame.as_ptr();
                self.resampler_for(raw)?
                    .convert_to_interleaved_f32(raw, channels)?
            };
            if samples.is_empty() {
                continue;
            }
            let frame_samples = (samples.len() / channels) as i64;

            // SAFETY: a plain field of a live frame.
            let pts = unsafe { (*frame.as_ptr()).pts };
            let start = match self.next_start {
                Some(running) => running,
                None if pts == ffi::AV_NOPTS_VALUE => 0,
                None => MediaTime::new(pts, self.time_base).rescale(rate_base).value,
            };
            self.next_start = Some(start + frame_samples);

            let mut from = 0;
            let mut presentation_start = start;
            if let Some(target) = self.seek_target {
                let end = start + frame_samples;
                if end <= target {
                    continue;
                }
                self.seek_target = None;
                if start < target {
                    // Land exactly on the requested sample, then apply the
                    // same terminal trim as a sequential read. Returning here
                    // used to leak padding after a seek into the final frame.
                    from = ((target - start) as usize) * channels;
                    presentation_start = target;
                }
            }
            // Stop at the declared presentation endpoint. This applies even
            // when the same coded frame straddles both the seek and the end.
            let to = match self.audible_limit {
                Some(limit) if presentation_start >= limit => return Ok(None),
                Some(limit) if start + frame_samples > limit => {
                    ((limit - start) as usize) * channels
                }
                _ => samples.len(),
            };
            let samples = if from == 0 && to == samples.len() {
                samples
            } else {
                samples[from..to].to_vec()
            };
            if samples.is_empty() {
                return Ok(None);
            }
            self.input.check()?;
            return PcmAudioFrame::new(
                self.sample_rate,
                self.channels,
                MediaTime::new(presentation_start, rate_base),
                samples,
            )
            .map(Some);
        }
    }
}

// -------------------------------------------------------------------------
// Encode

struct LibavAudioEncoder {
    codec: CodecContext,
    resampler: Resampler,
    settings: AudioEncoderSettings,
    /// Interleaved f32 waiting for a full coded frame.
    pending: Vec<f32>,
    packets: Vec<EncodedPacket>,
    frame_size: usize,
    next_pts: i64,
    /// Samples the caller handed in. NOT what was submitted: the last coded
    /// frame is zero-padded to `frame_size`, and that padding is
    /// `trailing_samples`, not program.
    pushed: u64,
    submitted: u64,
}

impl std::fmt::Debug for LibavAudioEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibavAudioEncoder")
            .field("settings", &self.settings)
            .field("packets", &self.packets.len())
            .finish_non_exhaustive()
    }
}

impl LibavAudioEncoder {
    /// Platform first, libavcodec's own last. `aac_at` and `aac_mf` are the
    /// system encoders the app shipped with; the built-in serves where neither
    /// opens, and for one that does not say how many priming samples it adds:
    /// sample counts are this module's law, and an unknown priming would shift
    /// every encode by it.
    fn open(settings: AudioEncoderSettings) -> Result<Self> {
        init()?;
        let mut refusals = Vec::new();
        for name in ["aac_at", "aac_mf", "aac"] {
            let c_name = std::ffi::CString::new(name).expect("a literal without NUL");
            // SAFETY: a live NUL-terminated name; returns a static descriptor
            // or null.
            let encoder = unsafe { ffi::avcodec_find_encoder_by_name(c_name.as_ptr()) };
            if encoder.is_null() {
                continue;
            }
            match Self::open_with(encoder, name, settings) {
                Err(VideoError::BackendUnavailable(refusal)) => refusals.push(refusal),
                opened => return opened,
            }
        }
        Err(if refusals.is_empty() {
            VideoError::UnsupportedCodec("no AAC encoder in this build".into())
        } else {
            VideoError::BackendUnavailable(refusals.join("; "))
        })
    }

    fn open_with(
        encoder: *const ffi::AVCodec,
        name: &'static str,
        settings: AudioEncoderSettings,
    ) -> Result<Self> {
        // SAFETY: a static descriptor.
        let context = unsafe { ffi::avcodec_alloc_context3(encoder) };
        if context.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an audio encoder context".into(),
            ));
        }
        let context = CodecContext(context);

        // Whatever sample format this encoder takes. `aac_at` wants
        // interleaved 16-bit, libavcodec's `aac` wants planar float — asking
        // the encoder is the only way that works for both.
        // SAFETY: the descriptor's format list is NULL-terminated or absent.
        let target_format = unsafe {
            let mut formats: *const ffi::AVSampleFormat = std::ptr::null();
            ffi::avcodec_get_supported_config(
                context.0,
                encoder,
                ffi::AVCodecConfig::AV_CODEC_CONFIG_SAMPLE_FORMAT,
                0,
                &mut formats as *mut _ as *mut *const std::ffi::c_void,
                std::ptr::null_mut(),
            );
            if formats.is_null() {
                ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP
            } else {
                *formats
            }
        };

        let mut layout: ffi::AVChannelLayout = unsafe { std::mem::zeroed() };
        // SAFETY: a zeroed layout is the documented input to the default
        // initializer.
        unsafe { ffi::av_channel_layout_default(&mut layout, i32::from(settings.channels)) };

        // SAFETY: the context is ours and unopened.
        unsafe {
            let raw = context.0;
            (*raw).sample_rate = settings.sample_rate as i32;
            (*raw).sample_fmt = target_format;
            (*raw).bit_rate = i64::from(settings.bit_rate);
            (*raw).ch_layout = layout;
            (*raw).time_base = ffi::AVRational {
                num: 1,
                den: settings.sample_rate as i32,
            };
            (*raw).flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
        }
        // SAFETY: context and encoder match; unopened.
        if let Err(error) = check_av("avcodec_open2", unsafe {
            ffi::avcodec_open2(context.0, encoder, std::ptr::null_mut())
        }) {
            return Err(VideoError::BackendUnavailable(format!(
                "{name} refused these settings: {error}"
            )));
        }
        // SAFETY: open, so the encoder has said what priming it adds. AAC
        // always has some; zero means this encoder does not report it.
        if unsafe { (*context.0).initial_padding } <= 0 {
            return Err(VideoError::BackendUnavailable(format!(
                "{name} does not report its priming"
            )));
        }

        // SAFETY: open, so `frame_size` is settled.
        let frame_size = unsafe { (*context.0).frame_size.max(0) as usize };
        let frame_size = if frame_size == 0 { 1024 } else { frame_size };

        let source_layout = layout;
        let mut resampler: *mut ffi::SwrContext = std::ptr::null_mut();
        // SAFETY: valid out-pointer, both layouts settled.
        check_av("swr_alloc_set_opts2", unsafe {
            ffi::swr_alloc_set_opts2(
                &mut resampler,
                &layout,
                target_format,
                settings.sample_rate as i32,
                &source_layout,
                ffi::AVSampleFormat::AV_SAMPLE_FMT_FLT,
                settings.sample_rate as i32,
                0,
                std::ptr::null_mut(),
            )
        })?;
        let resampler = Resampler(resampler);
        // SAFETY: just configured.
        check_av("swr_init", unsafe { ffi::swr_init(resampler.0) })?;

        log::debug!(target: "ferment_engine_media::libav", "encoding audio with {name}");
        Ok(Self {
            codec: context,
            resampler,
            settings,
            pending: Vec::new(),
            packets: Vec::new(),
            frame_size,
            next_pts: 0,
            pushed: 0,
            submitted: 0,
        })
    }

    fn collect(&mut self) -> Result<()> {
        let base = TimeBase::new_unchecked(1, i64::from(self.settings.sample_rate));
        loop {
            let mut packet = ffmpeg_next::codec::packet::Packet::empty();
            // SAFETY: both live.
            let status = unsafe { ffi::avcodec_receive_packet(self.codec.0, packet.as_mut_ptr()) };
            if status == ffi::AVERROR(ffi::EAGAIN) || status == ffi::AVERROR_EOF {
                return Ok(());
            }
            check_av("avcodec_receive_packet", status)?;
            // SAFETY: a live packet.
            let raw = unsafe { &*packet.as_ptr() };
            let at =
                |value: i64| (value != ffi::AV_NOPTS_VALUE).then(|| MediaTime::new(value, base));
            self.packets.push(EncodedPacket {
                stream_index: 0,
                pts: at(raw.pts),
                dts: at(raw.dts),
                duration: (raw.duration > 0).then(|| MediaTime::new(raw.duration, base)),
                is_sync: true,
                data: packet.data().unwrap_or_default().to_vec(),
                packetization: Packetization::Elementary,
            });
        }
    }

    /// Hand the encoder one coded frame's worth, converted to its format.
    fn submit(&mut self, samples: &[f32], count: usize) -> Result<()> {
        let channels = usize::from(self.settings.channels);
        let mut frame = ffmpeg_next::frame::Audio::empty();
        // SAFETY: a fresh frame; these are the documented fields to set
        // before `av_frame_get_buffer`.
        unsafe {
            let raw = frame.as_mut_ptr();
            (*raw).nb_samples = count as i32;
            (*raw).format = (*self.codec.0).sample_fmt as i32;
            ffi::av_channel_layout_copy(&mut (*raw).ch_layout, &(*self.codec.0).ch_layout);
            (*raw).sample_rate = self.settings.sample_rate as i32;
            check_av("av_frame_get_buffer", ffi::av_frame_get_buffer(raw, 0))?;
            (*raw).pts = self.next_pts;

            let mut input = samples.as_ptr().cast::<u8>();
            let written = ffi::swr_convert(
                self.resampler.0,
                (*raw).extended_data,
                count as i32,
                &mut input as *mut _ as *const *const u8,
                count as i32,
            );
            check_av("swr_convert", written)?;
        }
        self.next_pts += count as i64;
        self.submitted += count as u64;
        let _ = channels;
        // SAFETY: both live; the encoder references the frame's buffers.
        check_av("avcodec_send_frame", unsafe {
            ffi::avcodec_send_frame(self.codec.0, frame.as_ptr())
        })?;
        self.collect()
    }
}

#[async_trait(?Send)]
impl AudioEncodeSession for LibavAudioEncoder {
    async fn push_pcm(&mut self, frame: PcmAudioFrame) -> Result<()> {
        if frame.sample_rate != self.settings.sample_rate
            || frame.channels != self.settings.channels
        {
            return Err(VideoError::InvalidData(format!(
                "encoder takes {} Hz / {} channels, got {} Hz / {}",
                self.settings.sample_rate,
                self.settings.channels,
                frame.sample_rate,
                frame.channels,
            )));
        }
        self.pushed += frame.frame_count() as u64;
        self.pending.extend_from_slice(&frame.samples);
        let channels = usize::from(self.settings.channels);
        let chunk = self.frame_size * channels;
        while self.pending.len() >= chunk {
            let block: Vec<f32> = self.pending.drain(..chunk).collect();
            self.submit(&block, self.frame_size)?;
        }
        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<EncodedAudio> {
        // The remainder is a partial coded frame; AAC pads it, and that
        // padding is what `trailing_samples` accounts for.
        let channels = usize::from(self.settings.channels);
        let remainder = self.pending.len() / channels;
        if remainder > 0 {
            let mut block: Vec<f32> = std::mem::take(&mut self.pending);
            block.resize(self.frame_size * channels, 0.0);
            self.submit(&block, self.frame_size)?;
        }
        // SAFETY: live and open; a null frame is the flush.
        check_av("avcodec_send_frame(flush)", unsafe {
            ffi::avcodec_send_frame(self.codec.0, std::ptr::null())
        })?;
        self.collect()?;

        // SAFETY: open, so extradata is the AudioSpecificConfig.
        let (description, leading) = unsafe {
            let raw = &*self.codec.0;
            let description = if raw.extradata.is_null() || raw.extradata_size <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(raw.extradata, raw.extradata_size as usize).to_vec()
            };
            (description, raw.initial_padding.max(0) as u32)
        };
        let configuration = DecoderConfiguration::from_codec_private(Codec::Aac, description)?;

        // The priming/remainder law: everything submitted is priming plus the
        // audible program plus whatever padded the last coded frame.
        let coded = self.packets.len() as u64 * self.frame_size as u64;
        let audible = self.pushed;
        let trailing = coded
            .saturating_sub(u64::from(leading))
            .saturating_sub(audible);
        Ok(EncodedAudio {
            configuration,
            packets: std::mem::take(&mut self.packets),
            timing: AudioTiming {
                leading_samples: leading,
                trailing_samples: u32::try_from(trailing).unwrap_or(0),
                audible_samples: Some(audible),
            },
        })
    }
}
