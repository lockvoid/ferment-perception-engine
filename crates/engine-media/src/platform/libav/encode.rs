//! Video encode through `libavcodec`.
//!
//! Ruling 9. This is the Windows encode path — NVENC first, Media Foundation
//! behind it — and it is what the Media Foundation backend was replaced by.
//! macOS
//! encodes through its own VideoToolbox session (amendment A1) and does not
//! come here, but the code is not `cfg`-gated to Windows: it works wherever
//! libavcodec has an encoder, which is what lets it be tested on a Mac against
//! `h264_videotoolbox` rather than only compiled there.
//!
//! Every output is `AV_CODEC_FLAG_GLOBAL_HEADER`, so extradata is an
//! avcC/hvcC record and packets are length-prefixed. That is the
//! [`EncodedVideo`] contract the muxer and the sequencer already expect, and
//! it is not negotiable per codec.

use async_trait::async_trait;
use ffmpeg_next::codec::packet::traits::{Mut as PacketMut, Ref as PacketRef};
use ffmpeg_next::ffi;

use super::{check_av, init};
use crate::{
    Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, DecoderConfiguration, EncodeSession,
    EncodedPacket, EncodedVideo, EncoderSettings, FrameBuffer, MediaTime, Packetization, Result,
    TimeBase, TransferCharacteristic, VideoError, VideoFrame,
};

/// Encoders to try for a codec, best first.
///
/// NVIDIA leads on Windows because that is the shipping target; Media
/// Foundation is what an Intel or AMD machine falls back to. The names that do
/// not exist in a given build are simply not found, so one list serves every
/// platform, and one that does not open with the settings is passed over.
fn candidates(codec: &Codec) -> &'static [&'static str] {
    match codec {
        Codec::H264 => &["h264_nvenc", "h264_videotoolbox", "h264_mf"],
        Codec::Hevc => &["hevc_nvenc", "hevc_videotoolbox", "hevc_mf"],
        Codec::Av1 => &["av1_nvenc"],
        // ProRes has no hardware encoder outside VideoToolbox; `prores_ks` is
        // libavcodec's own and is what Windows uses.
        Codec::ProRes(_) => &["prores_videotoolbox", "prores_ks"],
        _ => &[],
    }
}

struct EncoderContext(*mut ffi::AVCodecContext);

impl Drop for EncoderContext {
    fn drop(&mut self) {
        // SAFETY: this type is the sole owner.
        unsafe { ffi::avcodec_free_context(&mut self.0) };
    }
}

pub(super) struct LibavEncoder {
    codec: EncoderContext,
    settings: EncoderSettings,
    packets: Vec<EncodedPacket>,
    time_base: TimeBase,
    /// What the encoder actually emits, read off its extradata once it is
    /// open rather than assumed.
    ///
    /// `AV_CODEC_FLAG_GLOBAL_HEADER` asks for an avcC/hvcC record, and most
    /// encoders oblige — but not all: `h264_videotoolbox` returns Annex-B
    /// start codes through libavcodec even with the flag set, and its packets
    /// match. Both forms are carried end to end (the mov muxer converts
    /// Annex-B extradata itself, and `EncodedPacket::data_in_format` converts
    /// the payloads), so the honest answer is the one the encoder gave.
    packetization: Packetization,
    configuration: DecoderConfiguration,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
    /// Each pushed frame's explicit duration by its PTS, encoder clock. Not
    /// every encoder copies `AVFrame.duration` to its packets, even with
    /// `AV_CODEC_FLAG_FRAME_DURATION` (`h264_videotoolbox` through libavcodec
    /// leaves it zero); a packet carries its frame's PTS, so it gets it here.
    durations: std::collections::BTreeMap<i64, i64>,
}

impl std::fmt::Debug for LibavEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibavEncoder")
            .field("codec", &self.settings.codec)
            .field("size", &(self.settings.width, self.settings.height))
            .field("packets", &self.packets.len())
            .finish_non_exhaustive()
    }
}

impl LibavEncoder {
    /// The first encoder in [`candidates`] that opens with these settings. One
    /// that the build has can still refuse them: NVENC takes no frame below its
    /// minimum size and does not open on a machine without an NVIDIA GPU, and
    /// then the next candidate serves.
    pub(super) fn open(settings: EncoderSettings) -> Result<Self> {
        init()?;
        let encoders = available_encoders(&settings.codec);
        if encoders.is_empty() {
            return Err(VideoError::UnsupportedCodec(format!(
                "no encoder for {:?} in this build (tried {:?})",
                settings.codec,
                candidates(&settings.codec)
            )));
        }
        let mut refusals = Vec::new();
        for (encoder, name) in encoders {
            match Self::open_with(encoder, name, settings.clone()) {
                Err(VideoError::BackendUnavailable(refusal)) => refusals.push(refusal),
                opened => return opened,
            }
        }
        Err(VideoError::BackendUnavailable(refusals.join("; ")))
    }

    fn open_with(
        encoder: *const ffi::AVCodec,
        name: &'static str,
        settings: EncoderSettings,
    ) -> Result<Self> {
        // SAFETY: `encoder` is a static descriptor from the registry.
        let context = unsafe { ffi::avcodec_alloc_context3(encoder) };
        if context.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an encoder context".into(),
            ));
        }
        let context = EncoderContext(context);

        // SAFETY: the context is ours and unopened; every write is a field.
        unsafe {
            let raw = context.0;
            (*raw).width = settings.width as i32;
            (*raw).height = settings.height as i32;
            (*raw).time_base = ffi::AVRational {
                num: settings.time_base.numerator as i32,
                den: settings.time_base.denominator as i32,
            };
            // NV12 is what every hardware encoder here ingests without an
            // internal conversion.
            (*raw).pix_fmt = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
            if let Some(bit_rate) = settings.bit_rate {
                (*raw).bit_rate = bit_rate as i64;
                // `rc_max_rate` deliberately unset: VBR, as today.
            }
            if let Some(interval) = settings.max_keyframe_interval {
                (*raw).gop_size = interval as i32;
            }
            if !settings.allow_frame_reordering {
                // Delivery order equals presentation order. The scene
                // detectors index frames positionally and would silently
                // mis-associate every result without this.
                (*raw).max_b_frames = 0;
            }
            if let Some(color) = settings.color {
                apply_color(raw, color);
            }
            // avcC/hvcC extradata and length-prefixed packets, always. Explicit
            // frame durations must survive encoding into the packet timeline.
            (*raw).flags |=
                (ffi::AV_CODEC_FLAG_GLOBAL_HEADER | ffi::AV_CODEC_FLAG_FRAME_DURATION) as i32;
        }

        let mut options: *mut ffi::AVDictionary = std::ptr::null_mut();
        if let Some(profile) = &settings.profile {
            set_option(&mut options, "profile", profile)?;
        }
        // SAFETY: context and encoder match; the context is unopened.
        let status = unsafe { ffi::avcodec_open2(context.0, encoder, &mut options) };
        // SAFETY: ours to free whatever the open left behind.
        unsafe { ffi::av_dict_free(&mut options) };
        if let Err(error) = check_av("avcodec_open2", status) {
            return Err(VideoError::BackendUnavailable(format!(
                "{name} refused these settings: {error}"
            )));
        }

        // SAFETY: the context is open, so extradata is whatever the encoder
        // produced for its global header.
        let description = unsafe {
            let raw = &*context.0;
            if raw.extradata.is_null() || raw.extradata_size <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(raw.extradata, raw.extradata_size as usize).to_vec()
            }
        };
        let configuration =
            DecoderConfiguration::from_codec_private(settings.codec.clone(), description)?;
        let packetization = configuration.packetization;

        log::debug!(
            target: "ferment_engine_media::libav",
            "encoding with {name} ({packetization:?})",
        );
        Ok(Self {
            codec: context,
            time_base: settings.time_base,
            settings,
            packets: Vec::new(),
            packetization,
            configuration,
            scaler: None,
            durations: std::collections::BTreeMap::new(),
        })
    }

    /// Drain whatever the encoder has ready. Called after every frame and
    /// once more at the end, when the flush has been sent.
    fn collect(&mut self) -> Result<()> {
        loop {
            let mut packet = ffmpeg_next::codec::packet::Packet::empty();
            // SAFETY: both pointers are live.
            let status = unsafe { ffi::avcodec_receive_packet(self.codec.0, packet.as_mut_ptr()) };
            if status == ffi::AVERROR(ffi::EAGAIN) || status == ffi::AVERROR_EOF {
                return Ok(());
            }
            check_av("avcodec_receive_packet", status)?;

            // SAFETY: a live packet just received.
            let raw = unsafe { &*(packet.as_ptr()) };
            let at = |value: i64| {
                (value != ffi::AV_NOPTS_VALUE).then(|| MediaTime::new(value, self.time_base))
            };
            let pushed = self.durations.remove(&raw.pts);
            let duration = (raw.duration > 0).then_some(raw.duration).or(pushed);
            self.packets.push(EncodedPacket {
                stream_index: 0,
                pts: at(raw.pts),
                dts: at(raw.dts),
                duration: duration.map(|duration| MediaTime::new(duration, self.time_base)),
                is_sync: raw.flags & ffi::AV_PKT_FLAG_KEY != 0,
                data: packet.data().unwrap_or_default().to_vec(),
                packetization: self.packetization,
            });
        }
    }

    /// The frame an encoder takes: NV12 at the configured size.
    fn scale_to_nv12(&mut self, frame: &VideoFrame) -> Result<ffmpeg_next::frame::Video> {
        use ffmpeg_next::format::Pixel;
        use ffmpeg_next::software::scaling::{Context, Flags};

        // A platform-native frame has to come down before libavcodec's
        // software path can read it. macOS never reaches here (A1 keeps its own
        // encoder) and Android reaches it only as the fallback its own
        // MediaCodec encoder refused, which is why this is a conversion and not
        // a zero-copy wrap.
        let cpu = match &frame.buffer {
            FrameBuffer::Cpu(cpu) => cpu.to_bgra8(),
            FrameBuffer::Yuv420(_) | FrameBuffer::Ycbcr(_) => frame.to_cpu_rgb()?.to_bgra8(),
            #[cfg(target_os = "macos")]
            FrameBuffer::MacOS(_) => frame.to_cpu_rgb()?.to_bgra8(),
            #[cfg(target_os = "android")]
            FrameBuffer::Android(_) => frame.to_cpu_rgb()?.to_bgra8(),
        };

        let mut source = ffmpeg_next::frame::Video::new(Pixel::BGRA, cpu.width, cpu.height);
        let stride = source.stride(0);
        let row = cpu.width as usize * 4;
        for y in 0..cpu.height as usize {
            let from = &cpu.data[y * cpu.stride..y * cpu.stride + row];
            source.data_mut(0)[y * stride..y * stride + row].copy_from_slice(from);
        }

        let (width, height) = (self.settings.width, self.settings.height);
        let scaler = match self.scaler.as_mut() {
            Some(scaler) => {
                scaler.cached(
                    Pixel::BGRA,
                    cpu.width,
                    cpu.height,
                    Pixel::NV12,
                    width,
                    height,
                    Flags::BILINEAR,
                );
                scaler
            }
            None => self.scaler.insert(
                Context::get(
                    Pixel::BGRA,
                    cpu.width,
                    cpu.height,
                    Pixel::NV12,
                    width,
                    height,
                    Flags::BILINEAR,
                )
                .map_err(|error| VideoError::Demux(format!("swscale: {error}")))?,
            ),
        };
        let mut output = ffmpeg_next::frame::Video::empty();
        scaler
            .run(&source, &mut output)
            .map_err(|error| VideoError::Demux(format!("sws_scale: {error}")))?;
        output.set_pts(Some(frame.pts.rescale(self.time_base).value));
        // SAFETY: output owns this live frame. Like PTS, duration uses the
        // encoder clock; zero preserves an unknown duration without guessing.
        unsafe {
            (*output.as_mut_ptr()).duration = frame
                .duration
                .map_or(0, |duration| duration.rescale(self.time_base).value);
        }
        Ok(output)
    }
}

#[async_trait(?Send)]
impl EncodeSession for LibavEncoder {
    async fn push_frame(&mut self, frame: VideoFrame) -> Result<()> {
        let converted = self.scale_to_nv12(&frame)?;
        if let Some(duration) = frame.duration {
            self.durations.insert(
                frame.pts.rescale(self.time_base).value,
                duration.rescale(self.time_base).value,
            );
        }
        // SAFETY: both pointers are live; the encoder takes a reference to
        // the frame's buffers rather than owning them.
        check_av("avcodec_send_frame", unsafe {
            ffi::avcodec_send_frame(self.codec.0, converted.as_ptr())
        })?;
        self.collect()
    }

    async fn finalize(mut self: Box<Self>) -> Result<EncodedVideo> {
        // A null frame is the flush. Without it the encoder keeps whatever is
        // in its lookahead and the tail of the clip is missing.
        // SAFETY: the context is live and open.
        check_av("avcodec_send_frame(flush)", unsafe {
            ffi::avcodec_send_frame(self.codec.0, std::ptr::null())
        })?;
        self.collect()?;

        Ok(EncodedVideo {
            configuration: self.configuration.clone(),
            packets: std::mem::take(&mut self.packets),
        })
    }
}

/// The encoders of [`candidates`] that this build has, in its order.
fn available_encoders(codec: &Codec) -> Vec<(*const ffi::AVCodec, &'static str)> {
    candidates(codec)
        .iter()
        .filter_map(|name| {
            let c_name = std::ffi::CString::new(*name).expect("a literal without NUL");
            // SAFETY: the name is a live NUL-terminated string; the call returns
            // a static descriptor or null.
            let encoder = unsafe { ffi::avcodec_find_encoder_by_name(c_name.as_ptr()) };
            (!encoder.is_null()).then_some((encoder, *name))
        })
        .collect()
}

/// # Safety
///
/// `context` must be an allocated, unopened `AVCodecContext`.
unsafe fn apply_color(context: *mut ffi::AVCodecContext, color: ColorInfo) {
    // SAFETY: the caller guarantees a live unopened context.
    unsafe {
        (*context).color_range = match color.range {
            ColorRange::Limited => ffi::AVColorRange::AVCOL_RANGE_MPEG,
            ColorRange::Full => ffi::AVColorRange::AVCOL_RANGE_JPEG,
            _ => ffi::AVColorRange::AVCOL_RANGE_UNSPECIFIED,
        };
        (*context).color_primaries = match color.primaries {
            ColorPrimaries::Bt709 => ffi::AVColorPrimaries::AVCOL_PRI_BT709,
            ColorPrimaries::Bt2020 => ffi::AVColorPrimaries::AVCOL_PRI_BT2020,
            ColorPrimaries::DisplayP3 => ffi::AVColorPrimaries::AVCOL_PRI_SMPTE432,
            ColorPrimaries::Smpte170M => ffi::AVColorPrimaries::AVCOL_PRI_SMPTE170M,
            _ => ffi::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED,
        };
        (*context).color_trc = match color.transfer {
            TransferCharacteristic::Bt709 => ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709,
            TransferCharacteristic::Pq => ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE2084,
            TransferCharacteristic::Hlg => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_ARIB_STD_B67
            }
            TransferCharacteristic::Smpte170M => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE170M
            }
            _ => ffi::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED,
        };
        (*context).colorspace = match color.matrix {
            ColorMatrix::Bt709 => ffi::AVColorSpace::AVCOL_SPC_BT709,
            ColorMatrix::Bt2020NonConstant => ffi::AVColorSpace::AVCOL_SPC_BT2020_NCL,
            ColorMatrix::Bt2020Constant => ffi::AVColorSpace::AVCOL_SPC_BT2020_CL,
            ColorMatrix::Smpte170M => ffi::AVColorSpace::AVCOL_SPC_SMPTE170M,
            _ => ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED,
        };
    }
}

fn set_option(options: *mut *mut ffi::AVDictionary, key: &str, value: &str) -> Result<()> {
    let key = std::ffi::CString::new(key).expect("a literal without NUL");
    let value = std::ffi::CString::new(value)
        .map_err(|_| VideoError::InvalidData("encoder option contains an interior NUL".into()))?;
    // SAFETY: a live out-pointer and two strings the call copies.
    check_av("av_dict_set", unsafe {
        ffi::av_dict_set(options, key.as_ptr(), value.as_ptr(), 0)
    })
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuFrame, CpuPixelFormat};

    /// A 64x48 frame with a moving bright block, so successive frames are not
    /// identical — an encoder handed twelve copies of one picture produces a
    /// stream that tells you nothing about whether it works.
    fn frame(index: i64, time_base: TimeBase) -> VideoFrame {
        let (width, height) = (64usize, 48usize);
        let mut pixels = vec![16u8; width * height * 3];
        let x0 = (index as usize * 4) % (width - 8);
        for y in 8..24 {
            for x in x0..x0 + 8 {
                let at = (y * width + x) * 3;
                pixels[at..at + 3].copy_from_slice(&[220, 180, 90]);
            }
        }
        let cpu = CpuFrame::new(
            width as u32,
            height as u32,
            width * 3,
            CpuPixelFormat::Rgb24,
            pixels,
        )
        .unwrap();
        VideoFrame::encodable_from_cpu(
            &cpu,
            width as u32,
            height as u32,
            MediaTime::new(index, time_base),
            Some(MediaTime::new(1, time_base)),
            ColorInfo::default(),
        )
        .unwrap()
    }

    fn encode(codec: Codec) -> EncodedVideo {
        let time_base = TimeBase::new_unchecked(1, 12);
        let settings = EncoderSettings {
            frame_rate: Some((12, 1)),
            bit_rate: Some(400_000),
            max_keyframe_interval: Some(4),
            allow_frame_reordering: false,
            ..EncoderSettings::new(codec, 64, 48, time_base)
        };
        let mut encoder = Box::new(LibavEncoder::open(settings).expect("open the encoder"));
        for index in 0..12 {
            futures_executor::block_on(encoder.push_frame(frame(index, time_base)))
                .expect("push a frame");
        }
        futures_executor::block_on(encoder.finalize()).expect("finalize")
    }

    /// Twelve frames in, twelve packets out — which only holds if `finalize`
    /// flushes the encoder. A lookahead that is never drained silently drops
    /// the tail of every clip.
    #[test]
    fn every_frame_comes_back_as_a_packet_after_the_flush() {
        let encoded = encode(Codec::H264);
        assert_eq!(encoded.packets.len(), 12);
        assert!(encoded.packets.iter().all(|packet| !packet.data.is_empty()));
        assert!(
            encoded.packets.iter().any(|packet| packet.is_sync),
            "no keyframe: nothing could start decoding this",
        );
    }

    #[test]
    fn explicit_vfr_frame_durations_reach_packets_in_the_encoder_clock() {
        let clock = TimeBase::new_unchecked(1, 90_000);
        let duration_clock = TimeBase::new_unchecked(1, 30_000);
        let settings = EncoderSettings {
            frame_rate: Some((30, 1)),
            bit_rate: Some(400_000),
            allow_frame_reordering: false,
            ..EncoderSettings::new(Codec::H264, 320, 180, clock)
        };
        let mut encoder = Box::new(LibavEncoder::open(settings).unwrap());
        let mut at = 0;
        for (index, (duration, ticks)) in [(1001, 3003), (2002, 6006), (500, 1500)]
            .into_iter()
            .enumerate()
        {
            let mut picture = frame(index as i64, clock);
            picture.pts = MediaTime::new(at, clock);
            picture.duration = Some(MediaTime::new(duration, duration_clock));
            futures_executor::block_on(encoder.push_frame(picture)).unwrap();
            at += ticks;
        }
        let encoded = futures_executor::block_on(encoder.finalize()).unwrap();
        let durations: Vec<_> = encoded
            .packets
            .iter()
            .map(|packet| {
                packet
                    .duration
                    .map(|duration| duration.rescale(clock).value)
            })
            .collect();
        assert_eq!(durations, vec![Some(3003), Some(6006), Some(1500)]);
        let pts: Vec<_> = encoded
            .packets
            .iter()
            .map(|packet| packet.pts.unwrap().rescale(clock).value)
            .collect();
        assert_eq!(pts, vec![0, 3003, 9009]);
    }

    /// The `EncodedVideo` contract: a global header, and packets whose
    /// declared form matches it.
    ///
    /// `AV_CODEC_FLAG_GLOBAL_HEADER` asks for an avcC record and most
    /// encoders give one — `h264_videotoolbox` through libavcodec gives
    /// Annex-B instead. What must hold either way is that the configuration
    /// and the packets agree, because a consumer converts payloads using the
    /// form the configuration declares.
    #[test]
    fn the_configuration_and_the_packets_declare_the_same_form() {
        let encoded = encode(Codec::H264);
        assert_eq!(encoded.configuration.codec, Codec::H264);
        assert!(
            !encoded.configuration.description.is_empty(),
            "GLOBAL_HEADER should have produced extradata",
        );
        assert!(
            encoded
                .packets
                .iter()
                .all(|packet| packet.packetization == encoded.configuration.packetization),
            "packets declare a different form than the configuration ({:?})",
            encoded.configuration.packetization,
        );
        // Whichever form it is, a decoder can be built from it: an avcC
        // record yields parameter sets, and Annex-B carries them inline.
        match encoded.configuration.packetization {
            Packetization::LengthPrefixed { .. } => assert!(
                !encoded.configuration.parameter_sets.is_empty(),
                "an avcC record with no SPS/PPS",
            ),
            Packetization::AnnexB => assert!(
                encoded.configuration.description.starts_with(&[0, 0, 0, 1])
                    || encoded.configuration.description.starts_with(&[0, 0, 1]),
                "Annex-B extradata without a start code",
            ),
            Packetization::Elementary => panic!("H.264 is never elementary"),
        }
    }

    /// Whatever form the encoder chose, the muxer accepts it — the path a
    /// real export takes. This is the assertion that makes the tolerance
    /// above safe rather than merely honest.
    #[test]
    fn the_encoders_output_muxes() {
        use crate::{MuxFormat, MuxSettings, Muxer, VideoMuxTrack};

        let encoded = encode(Codec::H264);
        let path =
            std::env::temp_dir().join(format!("ferment-libav-encode-{}.mp4", std::process::id()));
        let mut session = super::super::LibavMuxer
            .open(
                &path,
                MuxSettings {
                    format: MuxFormat::Mp4,
                    ..MuxSettings::default()
                },
            )
            .expect("open the muxer");
        let track = session
            .add_video_track(VideoMuxTrack {
                display_matrix: None,
                configuration: encoded.configuration.clone(),
                width: 64,
                height: 48,
                time_base: TimeBase::new_unchecked(1, 12),
                color: ColorInfo::default(),
                duration: None,
            })
            .expect("add the track");
        for packet in encoded.packets {
            session.write_packet(track, packet).expect("write a packet");
        }
        session.finalize().expect("finalize");

        let info = super::super::probe(&path).expect("the muxed output probes");
        let video = info.video_streams().next().expect("a video stream");
        assert_eq!(video.codec, Codec::H264);
        assert_eq!((video.width, video.height), (Some(64), Some(48)));
        let _ = std::fs::remove_file(&path);
    }

    /// Reordering off means delivery order IS presentation order. The scene
    /// detectors index frames positionally, so a stream that reorders would
    /// mis-associate every result rather than fail.
    #[test]
    fn reordering_off_delivers_presentation_order() {
        let encoded = encode(Codec::H264);
        let pts: Vec<i64> = encoded
            .packets
            .iter()
            .filter_map(|packet| packet.pts.map(|pts| pts.value))
            .collect();
        assert_eq!(pts.len(), 12);
        assert!(
            pts.windows(2).all(|pair| pair[0] < pair[1]),
            "delivery order is not presentation order: {pts:?}",
        );
    }

    /// A codec with no encoder in this build is refused by name, not by a
    /// null-pointer surprise somewhere later.
    #[test]
    fn a_codec_without_an_encoder_is_refused() {
        let error = LibavEncoder::open(EncoderSettings::new(
            Codec::Vp8,
            64,
            48,
            TimeBase::new_unchecked(1, 12),
        ))
        .unwrap_err();
        assert!(
            error.to_string().contains("no encoder for"),
            "unexpected error: {error}",
        );
    }
}
