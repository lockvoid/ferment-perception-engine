//! Video decode: `libavcodec` driving the platform's hardware decoder.
//!
//! Ruling 8. The shape is `doc/examples/hw_decode.c`'s, with our own seams:
//! a hardware device context, a `get_format` that picks the hardware pixel
//! format when it is offered, and a frame conversion that keeps a macOS
//! picture on the GPU as the `CVPixelBuffer` it already is.
//!
//! The three behaviours a caller depends on, all of them tested:
//!
//! * `seek` lands on or after the target, never before — the demuxer goes to
//!   the previous sync sample and the decoded leading pictures are discarded.
//! * end of stream drains the decoder, so the last frames of a GOP are
//!   delivered rather than lost with the file handle.
//! * `DecodeOptions::output_size` delivers that exact size in BGRA.

use std::path::Path;

use async_trait::async_trait;
use ffmpeg_next::codec::packet::traits::{Mut as PacketMut, Ref as PacketRef};
use ffmpeg_next::ffi;

use super::{check_av, hw::HwDevice, probe};
use crate::{
    ColorInfo, CpuFrame, CpuPixelFormat, DecodeOptions, DecodeSession, EncodeSession,
    EncoderSettings, FrameBuffer, MediaInfo, MediaTime, Result, TimeBase, VideoError, VideoFrame,
    VideoIo,
};

/// The media backend. One type for both platforms; what differs is which
/// hardware device `libavcodec` opens underneath.
#[derive(Debug, Clone, Copy, Default)]
pub struct LibavVideoIo;

impl LibavVideoIo {
    /// Decode a verified seekable stream on a media worker. Full-size macOS
    /// frames retain the native asynchronous VideoToolbox path.
    pub fn open_stream(
        &self,
        source: Box<dyn crate::MediaInput>,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        Self::open_input(super::input::Input::stream(source)?, stream_index, options)
    }

    fn open_input(
        input: super::input::Input,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        options.validate()?;
        #[cfg(target_os = "macos")]
        let mut input = input;
        #[cfg(target_os = "macos")]
        if options.acceleration == crate::DecodeAcceleration::Automatic
            && !matches!(
                options.output,
                crate::DecodeOutput::CpuRgba32Float | crate::DecodeOutput::CpuPlanes
            )
        {
            let info = probe::describe(&mut input)?;
            input.check()?;
            let stream = info.stream(stream_index).ok_or_else(|| {
                VideoError::InvalidData(format!("no video stream {stream_index}"))
            })?;
            if matches!(
                stream.codec,
                crate::Codec::H264 | crate::Codec::Hevc | crate::Codec::ProRes(_)
            ) {
                let reopen = input.reopener();
                let packets = super::PacketReader::from_input(input.reopen()?, info, stream_index)?;
                match crate::platform::macos::VideoToolboxDecoder::from_packets(
                    packets,
                    stream_index,
                    options,
                ) {
                    Ok(decoder) => return Ok(Box::new(decoder)),
                    Err(VideoError::UnsupportedCodec(_) | VideoError::BackendUnavailable(_)) => {}
                    Err(VideoError::Platform {
                        operation: "VTDecompressionSessionCreate",
                        ..
                    }) => {}
                    Err(error) => return Err(error),
                }
                // A refused native session has dropped its demuxer now.
                input = reopen()?;
            } else {
                input = input.reopen()?;
            }
            // Retry only native capability refusal. The same checked source
            // retains its authority; corruption/revocation errors propagate.
        }
        if options.output == crate::DecodeOutput::GpuPlanes {
            // libavcodec decodes into CPU memory: no GPU planes to hand over.
            return Err(VideoError::BackendUnavailable(
                "GPU planes need the platform's hardware decoder for this stream".into(),
            ));
        }
        Ok(Box::new(LibavDecoder::from_input(
            input,
            stream_index,
            options,
        )?))
    }
}

#[async_trait(?Send)]
impl VideoIo for LibavVideoIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        probe::probe(source)
    }

    async fn open_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        self.open_decoder_with(source, stream_index, DecodeOptions::default())
            .await
    }

    /// macOS uses its asynchronous VideoToolbox session. A consumer can
    /// explicitly choose software for its measured small CPU-thumbnail path.
    ///
    /// Both halves are measured, over a 19.4 s 1080x1350 clip, release
    /// profile. Full size, frames staying on the GPU: 0.49 s through the
    /// VideoToolbox session against 1.58 s through libavcodec's hwaccel,
    /// which waits for every frame it submits. Scaled to 192x192 for a CPU
    /// read: 0.30 s through libavcodec's software decoder and swscale against
    /// 1.35 s through VideoToolbox plus a pixel-transfer session — a hardware
    /// decode would scale on the GPU and read back, two transfers to reach
    /// where the software decoder writes directly.
    async fn open_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        Self::open_input(super::input::Input::path(source)?, stream_index, options)
    }

    /// macOS encodes through its own VideoToolbox session (amendment A1) and
    /// hands the packets to the libavformat muxer. Everywhere else this is
    /// libavcodec's encoder, which lands in stage 4.
    async fn open_encoder(&self, settings: EncoderSettings) -> Result<Box<dyn EncodeSession>> {
        #[cfg(target_os = "macos")]
        {
            crate::platform::macos::VideoToolboxVideoIo
                .open_encoder(settings)
                .await
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Box::new(super::encode::LibavEncoder::open(settings)?))
        }
    }
}

/// An owned `AVCodecContext`.
struct DecoderContext(*mut ffi::AVCodecContext);

impl Drop for DecoderContext {
    fn drop(&mut self) {
        // SAFETY: this type is the sole owner; freeing also releases the
        // hardware device reference `HwDevice::attach` handed over.
        unsafe { ffi::avcodec_free_context(&mut self.0) };
    }
}

struct LibavDecoder {
    input: super::input::Input,
    codec: DecoderContext,
    /// Held for the decoder's lifetime. The codec context takes its own
    /// reference, but every frame that came off the device outlives the
    /// decode call that produced it, so the device outlives the session.
    /// Read only where a frame has to be brought down off it.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    device: Option<HwDevice>,
    hardware: bool,
    stream_index: usize,
    time_base: TimeBase,
    color: ColorInfo,
    options: DecodeOptions,
    /// Set by `seek`, cleared by the first frame at or after it. Discarding
    /// lazily here rather than eagerly in `seek` keeps one definition of
    /// "the next picture" instead of two.
    seek_target: Option<MediaTime>,
    /// `avcodec_send_packet(NULL)` has been issued; only receives remain.
    draining: bool,
    finished: bool,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
    /// NV12 apart into planar 4:2:0 ahead of `scaler`; see `deinterleave`.
    deinterleaver: Option<ffmpeg_next::software::scaling::Context>,
}

impl std::fmt::Debug for LibavDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibavDecoder")
            .field("stream_index", &self.stream_index)
            .field("hardware", &self.hardware)
            .field("output_size", &self.options.output_size)
            .finish_non_exhaustive()
    }
}

impl LibavDecoder {
    fn from_input(
        input: super::input::Input,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Self> {
        options.validate()?;
        let stream = input
            .streams()
            .find(|stream| stream.index() == stream_index as usize)
            .ok_or_else(|| VideoError::InvalidData(format!("no video stream {stream_index}")))?;
        let parameters = stream.parameters();
        // SAFETY: `codecpar` lives as long as the open input context.
        let raw = unsafe { &*parameters.as_ptr() };
        if raw.codec_type != ffi::AVMediaType::AVMEDIA_TYPE_VIDEO {
            return Err(VideoError::InvalidData(format!(
                "stream {stream_index} is not video"
            )));
        }
        let time_base = TimeBase::new(
            i64::from(stream.time_base().numerator()),
            i64::from(stream.time_base().denominator()),
        )?;
        // Read the colorimetry off the stream in hand. Calling `probe` here
        // would open and analyse the file a second time — including its
        // seek-to-the-end scan for audio priming — for two fields already
        // sitting in this `codecpar`.
        let color = probe::color_info(raw);
        options.validate_color(color, true)?;

        // SAFETY: `codec_id` is this stream's; the finder returns a static
        // descriptor or null.
        let codec = unsafe { ffi::avcodec_find_decoder(raw.codec_id) };
        if codec.is_null() {
            return Err(VideoError::UnsupportedCodec(format!(
                "no decoder for {:?}",
                raw.codec_id
            )));
        }
        // SAFETY: allocates an owned context or returns null.
        let context = unsafe { ffi::avcodec_alloc_context3(codec) };
        if context.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate a decoder context".into(),
            ));
        }
        let context = DecoderContext(context);
        // SAFETY: both pointers are live; this copies the stream's parameters.
        check_av("avcodec_parameters_to_context", unsafe {
            ffi::avcodec_parameters_to_context(context.0, parameters.as_ptr())
        })?;

        // A resize or CPU output alone must not disable hardware decoding.
        // A macOS hardware frame reaches the CPU only as BGRA8, past any HDR
        // conversion, so CPU pixels from an HDR source decode in software.
        let hdr_on_cpu = cfg!(target_os = "macos")
            && matches!(
                options.output,
                crate::DecodeOutput::CpuBgra8 | crate::DecodeOutput::CpuSceneLinear
            )
            && crate::hdr::is_hdr(color.transfer);
        // Planes come from software frames: a hardware download would be
        // eight-bit NV12 (Windows) or a format the planes do not read.
        let device = (options.acceleration == crate::DecodeAcceleration::Automatic
            && !matches!(
                options.output,
                crate::DecodeOutput::CpuRgba32Float | crate::DecodeOutput::CpuPlanes
            )
            && !hdr_on_cpu)
            .then(|| HwDevice::open(codec))
            .flatten();
        if let Some(device) = &device {
            // SAFETY: the context is allocated and not yet opened.
            unsafe { device.attach(context.0)? };
        }
        let hardware = device.is_some();
        // A hardware decoder is one engine, and extra threads around it only
        // add latency. Software decode takes 0, which is libavcodec's "choose
        // for me" and means one thread per core.
        // SAFETY: the context is ours and unopened.
        unsafe {
            (*context.0).thread_count = if hardware { 1 } else { 0 };
        }
        // SAFETY: the context and codec match and the context is unopened.
        check_av("avcodec_open2", unsafe {
            ffi::avcodec_open2(context.0, codec, std::ptr::null_mut())
        })?;

        if !hardware {
            log::debug!(
                target: "ferment_engine_media::libav",
                "decoding stream {stream_index} in software",
            );
        }

        Ok(Self {
            input,
            codec: context,
            device,
            hardware,
            stream_index: stream_index as usize,
            time_base,
            color,
            options,
            seek_target: None,
            draining: false,
            finished: false,
            scaler: None,
            deinterleaver: None,
        })
    }

    /// Pull one decoded picture, feeding packets until libavcodec produces one.
    fn next_frame(&mut self) -> Result<Option<ffmpeg_next::frame::Video>> {
        loop {
            let mut frame = ffmpeg_next::frame::Video::empty();
            // SAFETY: both pointers are live; the frame is unreferenced.
            let status = unsafe { ffi::avcodec_receive_frame(self.codec.0, frame.as_mut_ptr()) };
            if status == 0 {
                return Ok(Some(frame));
            }
            if status == eagain() {
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
        }
    }

    /// Send the next packet for our stream, or start the drain at end of file.
    fn feed(&mut self) -> Result<()> {
        if self.draining {
            // Already flushed; the receive loop is only waiting for EOF.
            self.finished = true;
            return Ok(());
        }
        loop {
            // av_read_frame hands us an owned packet, including streams we
            // skip. Drop it before the next read instead of overwriting refs.
            let mut packet = ffmpeg_next::codec::packet::Packet::empty();
            // SAFETY: both pointers are live for the call.
            let status =
                unsafe { ffi::av_read_frame(self.input.as_mut_ptr(), packet.as_mut_ptr()) };
            self.input.check()?;
            if status == ffi::AVERROR_EOF {
                // The tail of a GOP is still inside the decoder. Flushing is
                // what gets it out; without this the last frames of every
                // file are silently missing.
                self.draining = true;
                // SAFETY: a null packet is the documented flush request.
                check_av("avcodec_send_packet(flush)", unsafe {
                    ffi::avcodec_send_packet(self.codec.0, std::ptr::null())
                })?;
                return Ok(());
            }
            check_av("av_read_frame", status)?;
            if packet.stream() == self.stream_index {
                // SAFETY: both pointers are live. The decoder retains any
                // needed buffer reference; the Packet stays caller-owned.
                check_av("avcodec_send_packet", unsafe {
                    ffi::avcodec_send_packet(self.codec.0, packet.as_ptr())
                })?;
                return Ok(());
            }
        }
    }

    /// One decoded `AVFrame` as a [`VideoFrame`], honouring `output_size`.
    fn convert(&mut self, frame: &ffmpeg_next::frame::Video) -> Result<VideoFrame> {
        let pts = MediaTime::new(frame.pts().unwrap_or(0), self.time_base);
        // SAFETY: `duration` is a plain field of a live frame.
        let duration = unsafe { (*frame.as_ptr()).duration };
        let duration = (duration > 0).then(|| MediaTime::new(duration, self.time_base));

        #[cfg(target_os = "macos")]
        if frame.format() == ffmpeg_next::format::Pixel::VIDEOTOOLBOX {
            return self.convert_videotoolbox(frame, pts, duration);
        }

        if self.options.output == crate::DecodeOutput::CpuPlanes {
            let planes = crate::YcbcrFrame::pack(&picture(frame)?)?;
            let mut color = self.color;
            color.bits_per_channel = Some(planes.depth as u8);
            return Ok(VideoFrame {
                width: planes.width,
                height: planes.height,
                pts,
                duration,
                color,
                buffer: FrameBuffer::Ycbcr(planes),
            });
        }
        let float = super::float::is_float(frame.format());
        if self.options.output == crate::DecodeOutput::CpuRgba32Float
            || float && self.options.output == crate::DecodeOutput::Native
        {
            if self
                .options
                .output_size
                .is_some_and(|size| size != (frame.width(), frame.height()))
            {
                return Err(VideoError::UnsupportedCodec(
                    "Float decode preserves its source extent; resize in the linear graph".into(),
                ));
            }
            let cpu = super::float::decode(frame)?;
            let mut color = self.color;
            color.bits_per_channel = Some(32);
            return Ok(VideoFrame {
                width: frame.width(),
                height: frame.height(),
                pts,
                duration,
                color,
                buffer: FrameBuffer::Cpu(cpu),
            });
        }

        let (width, height) = self
            .options
            .output_size
            .unwrap_or((frame.width(), frame.height()));
        let hdr = crate::hdr::is_hdr(self.color.transfer);
        // Native keeps a deep or HDR source's own samples, for an encoder.
        #[cfg(not(target_os = "macos"))]
        if self.options.output == crate::DecodeOutput::Native
            && (hdr || self.color.bits_per_channel.is_some_and(|bits| bits > 8))
        {
            let yuv = self.scale_to_yuv420(frame, width, height)?;
            let mut color = self.color;
            color.bits_per_channel = Some(if yuv.format.bytes_per_sample() == 2 {
                10
            } else {
                8
            });
            return Ok(VideoFrame {
                width,
                height,
                pts,
                duration,
                color,
                buffer: FrameBuffer::Yuv420(yuv),
            });
        }
        if hdr {
            let linear = self.options.output == crate::DecodeOutput::CpuSceneLinear;
            return Ok(VideoFrame {
                width,
                height,
                pts,
                duration,
                color: if linear {
                    crate::hdr::SCENE_LINEAR
                } else {
                    crate::hdr::SDR
                },
                buffer: FrameBuffer::Cpu(self.hdr_to_cpu(frame, width, height, linear)?),
            });
        }
        let cpu = self.scale_to_bgra(frame, width, height)?;
        Ok(VideoFrame {
            width,
            height,
            pts,
            duration,
            color: self.color,
            buffer: FrameBuffer::Cpu(cpu),
        })
    }

    /// An HDR frame at `width`×`height` (`crate::hdr`): SDR BT.709 BGRA, or
    /// the scene light of an HDR project when `linear`. A resize scales the
    /// YUV samples to 16-bit 4:4:4 first, so the conversion never reads
    /// eight-bit or interpolated RGB.
    fn hdr_to_cpu(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
        width: u32,
        height: u32,
        linear: bool,
    ) -> Result<CpuFrame> {
        use ffmpeg_next::software::scaling::{Context, Flags};
        let color = self.color;
        let convert = |picture: &crate::hdr::Picture| {
            if linear {
                crate::hdr::to_scene_linear(picture, color)
            } else {
                crate::hdr::to_sdr(picture, color, CpuPixelFormat::Bgra8)
            }
        };
        if (width, height) == (frame.width(), frame.height()) {
            return convert(&picture(frame)?);
        }
        let target = ffmpeg_next::format::Pixel::YUV444P16LE;
        let scaler = match self.scaler.as_mut() {
            Some(scaler) => {
                scaler.cached(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    target,
                    width,
                    height,
                    Flags::BILINEAR,
                );
                scaler
            }
            None => self.scaler.insert(
                Context::get(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    target,
                    width,
                    height,
                    Flags::BILINEAR,
                )
                .map_err(|error| VideoError::Demux(format!("swscale HDR samples: {error}")))?,
            ),
        };
        let mut scaled = ffmpeg_next::frame::Video::empty();
        scaler
            .run(frame, &mut scaled)
            .map_err(|error| VideoError::Demux(format!("sws_scale HDR samples: {error}")))?;
        convert(&picture(&scaled)?)
    }

    /// A VideoToolbox frame's `data[3]` IS the `CVPixelBuffer`. Retaining it
    /// and dropping the `AVFrame` is the whole conversion — no copy, and the
    /// buffer goes straight into wgpu through the existing IOSurface path.
    #[cfg(target_os = "macos")]
    fn convert_videotoolbox(
        &self,
        frame: &ffmpeg_next::frame::Video,
        pts: MediaTime,
        duration: Option<MediaTime>,
    ) -> Result<VideoFrame> {
        use std::ptr::NonNull;

        use objc2_core_foundation::CFRetained;
        use objc2_core_video::CVPixelBuffer;

        // SAFETY: for AV_PIX_FMT_VIDEOTOOLBOX, libavcodec documents data[3] as
        // the CVPixelBufferRef.
        let raw = unsafe { (*frame.as_ptr()).data[3] }.cast::<CVPixelBuffer>();
        let buffer = NonNull::new(raw).ok_or_else(|| {
            VideoError::InvalidData("VideoToolbox frame carries no pixel buffer".into())
        })?;
        // SAFETY: the buffer is owned by the AVFrame, which is released when
        // this function returns — so take a reference of our own first.
        let retained = unsafe { CFRetained::retain(buffer) };

        let (width, height) = (frame.width(), frame.height());
        let Some((target_width, target_height)) = self.options.output_size else {
            let buffer = crate::MacPixelBuffer::new(retained);
            return Ok(VideoFrame {
                width,
                height,
                pts,
                duration,
                color: self.color,
                buffer: if self.options.output == crate::DecodeOutput::CpuBgra8 {
                    FrameBuffer::Cpu(buffer.copy_bgra8()?)
                } else {
                    FrameBuffer::MacOS(buffer)
                },
            });
        };

        // `output_size` promises that exact size in 32BGRA, and the
        // pixel-transfer session does both in one hardware pass — so a
        // consumer that wants a small picture never pays for a big one.
        let scaled = crate::platform::macos::transfer_scaled(
            &retained,
            target_width,
            target_height,
            objc2_core_video::kCVPixelFormatType_32BGRA,
        )?;
        let buffer = crate::MacPixelBuffer::new(scaled);
        Ok(VideoFrame {
            width: target_width,
            height: target_height,
            pts,
            duration,
            color: self.color,
            buffer: if self.options.output == crate::DecodeOutput::CpuBgra8 {
                FrameBuffer::Cpu(buffer.copy_bgra8()?)
            } else {
                FrameBuffer::MacOS(buffer)
            },
        })
    }

    #[cfg(not(target_os = "macos"))]
    fn scale_to_yuv420(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
        width: u32,
        height: u32,
    ) -> Result<crate::Yuv420Frame> {
        use ffmpeg_next::software::scaling::{Context, Flags};
        let depth = self.color.bits_per_channel.ok_or_else(|| {
            VideoError::InvalidData(
                "source-depth YUV conversion requires a declared bit depth".into(),
            )
        })?;
        let ten_bit = depth > 8;
        let target = if ten_bit {
            ffmpeg_next::format::Pixel::P010LE
        } else {
            ffmpeg_next::format::Pixel::NV12
        };
        let (range, format) = match (self.color.range, ten_bit) {
            (crate::ColorRange::Full, false) => (1, crate::Yuv420Format::Nv12Full),
            (crate::ColorRange::Limited, false) => (0, crate::Yuv420Format::Nv12Video),
            (crate::ColorRange::Full, true) => (1, crate::Yuv420Format::P010Full),
            (crate::ColorRange::Limited, true) => (0, crate::Yuv420Format::P010Video),
            _ => {
                return Err(VideoError::InvalidData(
                    "source-depth YUV range is not declared".into(),
                ));
            }
        };
        let matrix = match self.color.matrix {
            crate::ColorMatrix::Bt2020NonConstant => 9, // SWS_CS_BT2020
            crate::ColorMatrix::Bt709 => 1,             // SWS_CS_ITU709
            _ => {
                return Err(VideoError::UnsupportedCodec(
                    "unsupported source-depth YUV matrix".into(),
                ));
            }
        };
        let scaler = match self.scaler.as_mut() {
            Some(scaler) => {
                scaler.cached(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    target,
                    width,
                    height,
                    Flags::BILINEAR,
                );
                scaler
            }
            None => self.scaler.insert(
                Context::get(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    target,
                    width,
                    height,
                    Flags::BILINEAR,
                )
                .map_err(|error| VideoError::Demux(format!("swscale source-depth YUV: {error}")))?,
            ),
        };
        // Keep source and destination YUV matrix/range identical. swscale's
        // default is limited BT.601; carrying HDR tags over that would be wrong.
        // SAFETY: this owned context is live and both tables are immutable
        // library coefficients. Contrast/saturation are identity16.16 values.
        let configured = unsafe {
            let coefficients = ffi::sws_getCoefficients(matrix);
            ffi::sws_setColorspaceDetails(
                scaler.as_mut_ptr(),
                coefficients,
                range,
                coefficients,
                range,
                0,
                1 << 16,
                1 << 16,
            )
        };
        if configured < 0 {
            return Err(VideoError::UnsupportedCodec(
                "swscale refused source-depth color preservation".into(),
            ));
        }
        let mut output = ffmpeg_next::frame::Video::empty();
        scaler
            .run(frame, &mut output)
            .map_err(|error| VideoError::Demux(format!("sws_scale source-depth YUV: {error}")))?;
        let row_bytes = width as usize * format.bytes_per_sample();
        let copy = |plane, rows| -> Result<Vec<u8>> {
            let stride = output.stride(plane);
            let data = output.data(plane);
            if stride < row_bytes || data.len() < stride * rows {
                return Err(VideoError::InvalidData(
                    "short libav source-depth YUV plane".into(),
                ));
            }
            let mut owned = Vec::with_capacity(row_bytes * rows);
            for row in 0..rows {
                owned.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
            }
            Ok(owned)
        };
        crate::Yuv420Frame::new(
            width,
            height,
            format,
            copy(0, height as usize)?,
            copy(1, height as usize / 2)?,
        )
    }

    /// `swscale` to tightly-packed BGRA at `width`×`height`.
    ///
    /// This is every non-macOS path: a software-decoded frame anywhere, and a
    /// Windows hardware frame after `av_hwframe_transfer_data` has brought it
    /// down to NV12 (ruling 13 — the D3D11 texture does not reach wgpu in v1).
    fn scale_to_bgra(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
        width: u32,
        height: u32,
    ) -> Result<CpuFrame> {
        use ffmpeg_next::software::scaling::{Context, Flags};

        let planar;
        let frame = if frame.format() == ffmpeg_next::format::Pixel::NV12 {
            planar = self.deinterleave(frame)?;
            &planar
        } else {
            frame
        };
        let source = frame.format();
        let scaler = match self.scaler.as_mut() {
            // `cached` keeps the context when the shape is unchanged and
            // rebuilds it when it is not, which is what a stream that changes
            // resolution mid-file needs.
            Some(scaler) => {
                scaler.cached(
                    source,
                    frame.width(),
                    frame.height(),
                    ffmpeg_next::format::Pixel::BGRA,
                    width,
                    height,
                    Flags::BILINEAR,
                );
                scaler
            }
            None => self.scaler.insert(
                Context::get(
                    source,
                    frame.width(),
                    frame.height(),
                    ffmpeg_next::format::Pixel::BGRA,
                    width,
                    height,
                    Flags::BILINEAR,
                )
                .map_err(|error| VideoError::Demux(format!("swscale: {error}")))?,
            ),
        };

        rgb_conversion(scaler, self.color)?;
        let mut output = ffmpeg_next::frame::Video::empty();
        scaler
            .run(frame, &mut output)
            .map_err(|error| VideoError::Demux(format!("sws_scale: {error}")))?;

        let stride = output.stride(0);
        let rows = usize::try_from(height).unwrap_or(0);
        let data = output.data(0);
        if data.len() < stride.saturating_mul(rows) {
            return Err(VideoError::InvalidData(
                "swscale produced a short BGRA buffer".into(),
            ));
        }
        CpuFrame::new(
            width,
            height,
            stride,
            CpuPixelFormat::Bgra8,
            &data[..stride * rows],
        )
    }

    /// The same 4:2:0 samples with the two chroma planes apart: swscale's
    /// unscaled NV12-to-planar copy, no resampling.
    ///
    /// A hardware frame comes down as NV12; software decode hands
    /// `scale_to_bgra` planar 4:2:0. swscale turns planar 4:2:0 into BGRA
    /// with its unscaled converter on every architecture, but on x86 it takes
    /// NV12 through its general scaler, which rounds the last two rows
    /// differently and takes four times as long. Measured on the Windows
    /// server: a flat grey field came back one level darker in its bottom two
    /// rows, which U2Netp took for a subject; 1080p took 4.25 ms, against
    /// 0.42 ms for this copy plus 1.05 ms for the planar conversion. A
    /// hardware frame now gets its software decode's conversion.
    fn deinterleave(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
    ) -> Result<ffmpeg_next::frame::Video> {
        use ffmpeg_next::format::Pixel;
        use ffmpeg_next::software::scaling::{Context, Flags};

        let (width, height) = (frame.width(), frame.height());
        let deinterleaver = match self.deinterleaver.as_mut() {
            Some(deinterleaver) => {
                deinterleaver.cached(
                    Pixel::NV12,
                    width,
                    height,
                    Pixel::YUV420P,
                    width,
                    height,
                    Flags::POINT,
                );
                deinterleaver
            }
            None => self.deinterleaver.insert(
                Context::get(
                    Pixel::NV12,
                    width,
                    height,
                    Pixel::YUV420P,
                    width,
                    height,
                    Flags::POINT,
                )
                .map_err(|error| VideoError::Demux(format!("swscale: {error}")))?,
            ),
        };
        let mut planar = ffmpeg_next::frame::Video::empty();
        deinterleaver
            .run(frame, &mut planar)
            .map_err(|error| VideoError::Demux(format!("sws_scale NV12: {error}")))?;
        Ok(planar)
    }
}

/// Set `scaler` to read YUV by `color`'s matrix and range into full-range RGB.
pub(super) fn rgb_conversion(
    scaler: &mut ffmpeg_next::software::scaling::Context,
    color: crate::ColorInfo,
) -> Result<()> {
    let matrix = match color.matrix.for_rgb() {
        crate::ColorMatrix::Bt709 => ffi::SWS_CS_ITU709,
        crate::ColorMatrix::Fcc => ffi::SWS_CS_FCC,
        crate::ColorMatrix::Smpte240M => ffi::SWS_CS_SMPTE240M,
        crate::ColorMatrix::Bt2020NonConstant => ffi::SWS_CS_BT2020,
        crate::ColorMatrix::Bt470Bg
        | crate::ColorMatrix::Smpte170M
        | crate::ColorMatrix::Identity => ffi::SWS_CS_ITU601,
        other => {
            return Err(VideoError::UnsupportedCodec(format!(
                "RGB conversion matrix {other:?}"
            )));
        }
    };
    let range = i32::from(color.range == crate::ColorRange::Full);
    // SAFETY: live scaler, static coefficient tables, identity 16.16
    // controls; RGB is full-range regardless of the source YUV range.
    let configured = unsafe {
        let coefficients = ffi::sws_getCoefficients(matrix);
        ffi::sws_setColorspaceDetails(
            scaler.as_mut_ptr(),
            coefficients,
            range,
            coefficients,
            1,
            0,
            1 << 16,
            1 << 16,
        )
    };
    if configured < 0 {
        return Err(VideoError::UnsupportedCodec(
            "swscale refused RGB color conversion".into(),
        ));
    }
    Ok(())
}

/// A decoded YUV frame as `crate::hdr` reads it: planes, steps and shifts
/// from its pixel format's own descriptor.
pub(super) fn picture(frame: &ffmpeg_next::frame::Video) -> Result<crate::hdr::Picture<'_>> {
    let format = frame.format();
    // SAFETY: libavutil returns a pointer into its static table, or null.
    let descriptor = unsafe { ffi::av_pix_fmt_desc_get(format.into()).as_ref() }
        .ok_or_else(|| VideoError::UnsupportedCodec(format!("no descriptor for {format:?}")))?;
    let refused = ffi::AV_PIX_FMT_FLAG_BE
        | ffi::AV_PIX_FMT_FLAG_PAL
        | ffi::AV_PIX_FMT_FLAG_BITSTREAM
        | ffi::AV_PIX_FMT_FLAG_HWACCEL
        | ffi::AV_PIX_FMT_FLAG_RGB
        | ffi::AV_PIX_FMT_FLAG_FLOAT
        | ffi::AV_PIX_FMT_FLAG_BAYER
        | ffi::AV_PIX_FMT_FLAG_XYZ;
    let components = &descriptor.comp[..3];
    let depth = components[0].depth;
    if descriptor.nb_components < 3
        || descriptor.flags & refused as u64 != 0
        || components.iter().any(|component| {
            component.depth != depth
                || component.step <= 0
                || component.offset < 0
                || component.shift < 0
                || component.plane < 0
                || component.plane as usize >= frame.planes()
        })
    {
        return Err(VideoError::UnsupportedCodec(format!(
            "HDR to SDR cannot read {format:?} samples"
        )));
    }
    let planes = [0, 1, 2].map(|index| {
        let component = components[index];
        let plane = component.plane as usize;
        crate::hdr::Plane {
            data: frame.data(plane),
            stride: frame.stride(plane),
            step: component.step as usize,
            offset: component.offset as usize,
            shift: component.shift as u32,
            subsampling: if index == 0 {
                (0, 0)
            } else {
                (
                    u32::from(descriptor.log2_chroma_w),
                    u32::from(descriptor.log2_chroma_h),
                )
            },
        }
    });
    Ok(crate::hdr::Picture {
        width: frame.width(),
        height: frame.height(),
        depth: depth as u32,
        planes,
    })
}

#[async_trait(?Send)]
impl DecodeSession for LibavDecoder {
    async fn seek_sync(&mut self, target: MediaTime) -> Result<()> {
        self.seek(target).await?;
        self.seek_target = None;
        Ok(())
    }
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        self.input.check()?;
        let target = target.rescale(self.time_base);
        // BACKWARD lands on the sync sample at or before the target, which is
        // the only place a decoder can start. `decode_next` throws away what
        // that produces until the target is reached.
        // SAFETY: the context is live and the stream index is one of its own.
        check_av("av_seek_frame", unsafe {
            ffi::av_seek_frame(
                self.input.as_mut_ptr(),
                self.stream_index as i32,
                target.value,
                ffi::AVSEEK_FLAG_BACKWARD,
            )
        })?;
        // SAFETY: the context is live and open.
        unsafe { ffi::avcodec_flush_buffers(self.codec.0) };
        self.seek_target = Some(target);
        self.draining = false;
        self.finished = false;
        self.input.check()
    }

    async fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            self.input.check()?;
            let Some(frame) = self.next_frame()? else {
                return Ok(None);
            };
            #[cfg(not(target_os = "macos"))]
            let frame = self.download_if_hardware(frame)?;

            if let Some(target) = self.seek_target {
                let pts = frame.pts().unwrap_or(0);
                if pts < target.value {
                    continue;
                }
                self.seek_target = None;
            }
            let converted = self.convert(&frame)?;
            self.input.check()?;
            return Ok(Some(converted));
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl LibavDecoder {
    /// Windows hardware frames arrive as a D3D11 texture; wgpu has no import
    /// path for one yet, so they come down to NV12 here and go on as CPU
    /// frames. Ruling 13 names the zero-copy version as a follow-up.
    fn download_if_hardware(
        &self,
        frame: ffmpeg_next::frame::Video,
    ) -> Result<ffmpeg_next::frame::Video> {
        // Any format that is still on the device — D3D11 or CUDA — has to come
        // down before it can become a `CpuFrame`.
        let on_device = self
            .device
            .as_ref()
            .is_some_and(|device| frame.format() == device.frame_format().into());
        if !on_device {
            return Ok(frame);
        }
        // SAFETY: `frame` is a live decoded frame; `transfer_to_cpu` only
        // reads from it and returns a new owned frame.
        let downloaded =
            unsafe { super::hw::transfer_to_cpu(frame.as_ptr() as *mut ffi::AVFrame) }?;
        // SAFETY: `transfer_to_cpu` returns an owned frame; `wrap` adopts it.
        Ok(unsafe { ffmpeg_next::frame::Video::wrap(downloaded) })
    }
}

/// `AVERROR(EAGAIN)`, which libavcodec returns when it wants more input.
///
/// `EAGAIN` differs by platform (35 on the BSDs, 11 on Linux and the Windows
/// CRT), so it comes from the generated bindings rather than a constant here —
/// the same headers everything else in this module is bound against.
fn eagain() -> i32 {
    ffi::AVERROR(ffi::EAGAIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg_next::format::Pixel;

    /// A hardware frame comes down as NV12 and its software decode is planar
    /// 4:2:0: the same samples, so the same BGRA. Red on x86 before
    /// `deinterleave`: swscale's general NV12 path made the last two rows of
    /// this flat field (the saliency fixture's) one level darker.
    #[test]
    fn nv12_and_planar_420_of_one_picture_become_the_same_bgra() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated/h264-gop4.mp4");
        let input = super::super::input::Input::path(&fixture).unwrap();
        let options = DecodeOptions {
            acceleration: crate::DecodeAcceleration::Software,
            ..DecodeOptions::default()
        };
        let mut decoder = LibavDecoder::from_input(input, 0, options).unwrap();

        let (width, height) = (480, 270);
        let mut planar = ffmpeg_next::frame::Video::new(Pixel::YUV420P, width, height);
        for (plane, sample) in [(0, 119), (1, 128), (2, 128)] {
            planar.data_mut(plane).fill(sample);
        }
        let mut nv12 = ffmpeg_next::frame::Video::empty();
        ffmpeg_next::software::scaling::Context::get(
            Pixel::YUV420P,
            width,
            height,
            Pixel::NV12,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::POINT,
        )
        .unwrap()
        .run(&planar, &mut nv12)
        .unwrap();

        let from_planar = decoder.scale_to_bgra(&planar, width, height).unwrap();
        let from_nv12 = decoder.scale_to_bgra(&nv12, width, height).unwrap();
        fn row(frame: &CpuFrame, y: usize) -> &[u8] {
            &frame.data[y * frame.stride..][..frame.width as usize * 4]
        }
        let differing: Vec<usize> = (0..height as usize)
            .filter(|&y| row(&from_nv12, y) != row(&from_planar, y))
            .collect();
        assert!(differing.is_empty(), "rows {differing:?} differ");
    }
}
