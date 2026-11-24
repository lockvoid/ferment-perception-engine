//! Editing proxies: every picture a ProRes 422 Proxy key frame at a capped
//! short edge, with the source's own timestamps, colour and orientation.
//!
//! A jump into a long-GOP original decodes from its sync sample: a 4K HLG
//! HEVC with 8 s GOPs took 0.54 s median per jump even after the leading
//! pictures skip conversion, against 16 ms on an intra-only 1080p proxy
//! (release, M3 Max). ProRes 422 Proxy is the proxy codec of Final Cut,
//! Premiere and Resolve: 10-bit 4:2:2, every frame a key frame, the source's
//! colour tags (HLG/PQ included) carried as they are. It was also smaller and
//! twice as quick to jump in as an all-intra HEVC of the same footage.
//! macOS encodes with VideoToolbox, other systems with libavcodec's
//! `prores_ks`; the samples are rescaled once, YUV to YUV, without RGB.

use std::path::Path;

use ffmpeg_next::codec::packet::traits::Mut as PacketMut;
use ffmpeg_next::{Dictionary, ffi, format::Pixel};

use super::{check_av, init, input::Input, probe};
use crate::{
    Codec, ColorInfo, CpuFrame, CpuPixelFormat, DecoderConfiguration, EncodedPacket, MediaInput,
    MediaTime, MuxFormat, MuxSettings, Muxer, Packetization, ProResProfile, Result, TimeBase,
    VideoError, VideoMuxTrack,
};

/// What [`write_proxy`] made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyReport {
    pub width: u32,
    pub height: u32,
    pub pictures: u64,
    pub encoder: &'static str,
}

/// Thumbnails taken from a proxy's own decode, as Cuts on iOS takes them:
/// every `interval` seconds of the stream (from its first picture), the
/// proxy's picture fitted into `edge` pixels, as SDR RGB — an HDR source
/// through the engine's chain — and not yet turned by the display matrix.
/// `take` answers whether to go on.
pub struct ProxyThumbs<'a> {
    pub interval: f64,
    pub edge: u32,
    pub take: &'a mut dyn FnMut(ProxyThumb) -> bool,
}

/// One thumbnail of [`ProxyThumbs`]: its time in the stream and its pixels.
pub struct ProxyThumb {
    pub seconds: f64,
    /// `Rgb24`.
    pub image: CpuFrame,
}

/// A thumbnail's extent: `extent` fitted into `edge` pixels, the aspect kept.
pub fn thumb_extent(extent: (u32, u32), edge: u32) -> (u32, u32) {
    let (width, height) = extent;
    let scale = (f64::from(edge) / f64::from(width.max(height).max(1))).min(1.);
    let side = |side: u32| ((f64::from(side) * scale).round() as u32).max(1);
    (side(width), side(height))
}

/// The proxy extent: the short edge at most `short_edge` and never larger
/// than the source's, both sides even, the aspect kept.
pub fn proxy_extent(width: u32, height: u32, short_edge: u32) -> (u32, u32) {
    let short = width.min(height);
    let scale = f64::from(short_edge.min(short)) / f64::from(short.max(1));
    let even = |side: u32| ((f64::from(side) * scale / 2.).round() as u32 * 2).max(2);
    (even(width), even(height))
}

/// Writes `output` (MOV) from the video `stream` of `source`. `progress`
/// receives the share done, 0..=1, and answers whether to go on; `false`
/// stops with an error and leaves `output` for the caller to discard.
pub fn write_proxy(
    source: Box<dyn MediaInput>,
    stream_index: u32,
    output: &Path,
    short_edge: u32,
    progress: &mut dyn FnMut(f64) -> bool,
) -> Result<ProxyReport> {
    write_proxy_with_thumbs(source, stream_index, output, short_edge, None, progress)
}

/// [`write_proxy`], handing `thumbs` the thumbnails of the same decode.
pub fn write_proxy_with_thumbs(
    source: Box<dyn MediaInput>,
    stream_index: u32,
    output: &Path,
    short_edge: u32,
    thumbs: Option<ProxyThumbs<'_>>,
    progress: &mut dyn FnMut(f64) -> bool,
) -> Result<ProxyReport> {
    init()?;
    let mut input = Input::stream(source)?;
    let info = probe::describe(&mut input)?;
    input.check()?;
    let stream = info
        .stream(stream_index)
        .filter(|stream| stream.codec.is_video())
        .ok_or_else(|| VideoError::InvalidData(format!("no video stream {stream_index}")))?
        .clone();
    let mut input = input.reopen()?;
    let (source_width, source_height) = stream
        .width
        .zip(stream.height)
        .ok_or_else(|| VideoError::InvalidData("a proxy source has no extent".into()))?;
    let (width, height) = proxy_extent(source_width, source_height, short_edge);
    let time_base = stream.time_base;
    let start = stream
        .start_time
        .map_or(0, |time| time.rescale(time_base).value);
    let length = stream
        .duration
        .map(|duration| duration.rescale(time_base).value)
        .filter(|ticks| *ticks > 0);

    let parameters = input
        .stream(stream_index as usize)
        .ok_or_else(|| VideoError::InvalidData(format!("no stream {stream_index}")))?
        .parameters();
    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(parameters)
        .and_then(|mut context| {
            // Frame threads on every core: a single-threaded 4K HEVC decode
            // made a 30 s proxy take 31 s instead of 5 s.
            context.set_threading(ffmpeg_next::threading::Config {
                kind: ffmpeg_next::threading::Type::Frame,
                count: 0,
            });
            context.decoder().video()
        })
        .map_err(|error| VideoError::UnsupportedCodec(format!("proxy decoder: {error}")))?;

    let (codec, encoder_name, format) = encoder()?;
    let mut encoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .map_err(|error| VideoError::BackendUnavailable(format!("{encoder_name}: {error}")))?;
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(format);
    encoder.set_time_base(ffmpeg_next::Rational::new(
        time_base.numerator as i32,
        time_base.denominator as i32,
    ));
    if let Some((numerator, denominator)) = stream.frame_rate {
        encoder.set_frame_rate(Some(ffmpeg_next::Rational::new(
            numerator as i32,
            denominator as i32,
        )));
    }
    // SAFETY: the encoder context is ours and unopened; these are fields.
    // The colour is the source's: a proxy is the same picture, smaller.
    unsafe {
        let raw = encoder.as_mut_ptr();
        let source = parameters_color(&input, stream_index)?;
        (*raw).color_range = source.0;
        (*raw).color_primaries = source.1;
        (*raw).color_trc = source.2;
        (*raw).colorspace = source.3;
        (*raw).flags |= ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
        (*raw).thread_count = 0;
    }
    let mut options = Dictionary::new();
    options.set("profile", "proxy");
    let encoder = encoder
        .open_with(options)
        .map_err(|error| VideoError::BackendUnavailable(format!("{encoder_name}: {error}")))?;

    let mut session = super::LibavMuxer.open(
        output,
        MuxSettings {
            format: MuxFormat::Mov,
            fast_start: false,
            movie_timescale: None,
        },
    )?;
    let track = session.add_video_track(VideoMuxTrack {
        display_matrix: stream.display_matrix,
        configuration: DecoderConfiguration::from_codec_private(
            Codec::ProRes(ProResProfile::Proxy),
            Vec::new(),
        )?,
        width,
        height,
        time_base,
        color: stream.color,
        // From the packets: a probed stream duration can end at the last
        // picture's start, and an edit list that short drops that picture.
        duration: None,
    })?;

    let nominal = stream
        .frame_rate
        .filter(|(numerator, denominator)| *numerator > 0 && *denominator > 0)
        .map_or(0, |(numerator, denominator)| {
            MediaTime::new(
                i64::from(denominator),
                TimeBase::new_unchecked(1, numerator.into()),
            )
            .rescale(time_base)
            .value
        });
    let mut pipeline = Pipeline {
        encoder,
        encoder_name,
        session,
        track,
        time_base,
        format,
        extent: (width, height),
        scaler: Scaler::new()?,
        durations: std::collections::BTreeMap::new(),
        nominal,
        thumbs: thumbs.map(|thumbs| Thumbs {
            extent: thumb_extent((width, height), thumbs.edge),
            interval: thumbs.interval.max(1e-3),
            take: thumbs.take,
            next: 0.,
            start,
            color: stream.color,
            scaler: None,
        }),
    };
    let mut decoded = ffmpeg_next::frame::Video::empty();
    let (mut packets, mut pictures) = (0u64, 0u64);
    let mut finished = false;
    while !finished {
        let mut packet = ffmpeg_next::Packet::empty();
        // SAFETY: both pointers are live; the packet is unreferenced.
        let status = unsafe { ffi::av_read_frame(input.as_mut_ptr(), packet.as_mut_ptr()) };
        input.check()?;
        if status == ffi::AVERROR_EOF {
            decoder
                .send_eof()
                .map_err(|error| VideoError::Demux(format!("proxy decode: {error}")))?;
            finished = true;
        } else {
            check_av("av_read_frame", status)?;
            if packet.stream() != stream_index as usize {
                continue;
            }
            packets += 1;
            decoder
                .send_packet(&packet)
                .map_err(|error| VideoError::Demux(format!("proxy decode: {error}")))?;
        }
        while decoder.receive_frame(&mut decoded).is_ok() {
            pipeline.encode(&decoded)?;
            pictures += 1;
            let at = decoded.timestamp().or(decoded.pts());
            let done = at.zip(length).map_or(0., |(at, length)| {
                ((at - start) as f64 / length as f64).clamp(0., 1.)
            });
            if !progress(done) {
                return Err(VideoError::InvalidData("proxy cancelled".into()));
            }
        }
    }
    pipeline
        .encoder
        .send_eof()
        .map_err(|error| VideoError::Demux(format!("{encoder_name}: {error}")))?;
    pipeline.write()?;
    let Pipeline { session, .. } = pipeline;
    session.finalize()?;
    if packets == 0 || pictures == 0 {
        return Err(VideoError::InvalidData(
            "a proxy source has no pictures".into(),
        ));
    }
    Ok(ProxyReport {
        width,
        height,
        pictures,
        encoder: encoder_name,
    })
}

/// Decoded pictures in, muxed ProRes packets out (and thumbnails).
struct Pipeline<'a> {
    encoder: ffmpeg_next::encoder::video::Encoder,
    encoder_name: &'static str,
    session: Box<dyn crate::MuxSession>,
    track: crate::MuxTrackId,
    time_base: crate::TimeBase,
    format: Pixel,
    extent: (u32, u32),
    scaler: Scaler,
    /// Each picture's duration by its PTS. Not every encoder hands it on:
    /// VideoToolbox leaves the flushed last packet without one, and an edit
    /// list that ends at that packet's start hides the picture.
    durations: std::collections::BTreeMap<i64, i64>,
    /// One frame at the stream's nominal rate, for a picture without a duration.
    nominal: i64,
    thumbs: Option<Thumbs<'a>>,
}

/// The thumbnail side of a pipeline: the next due time, and its own scaler.
struct Thumbs<'a> {
    extent: (u32, u32),
    interval: f64,
    take: &'a mut dyn FnMut(ProxyThumb) -> bool,
    next: f64,
    /// The stream's first time, in its time base.
    start: i64,
    color: ColorInfo,
    scaler: Option<ffmpeg_next::software::scaling::Context>,
}

impl Thumbs<'_> {
    /// Hand on `frame` (the proxy's samples at `pts`) when a thumbnail is due.
    fn offer(
        &mut self,
        frame: &ffmpeg_next::frame::Video,
        pts: Option<i64>,
        time_base: TimeBase,
    ) -> Result<()> {
        let Some(pts) = pts else {
            return Ok(());
        };
        let seconds = ((pts - self.start) as f64 * time_base.numerator as f64
            / time_base.denominator as f64)
            .max(0.);
        if seconds + 1e-6 < self.next {
            return Ok(());
        }
        let image = thumbnail(&mut self.scaler, frame, self.extent, self.color)?;
        if !(self.take)(ProxyThumb { seconds, image }) {
            return Err(VideoError::InvalidData("proxy cancelled".into()));
        }
        self.next = ((seconds / self.interval).floor() + 1.) * self.interval;
        Ok(())
    }
}

/// `frame` (YUV) fitted into `extent` as SDR RGB24: an HDR source scaled as
/// 16-bit 4:4:4 samples and taken through the engine's chain, an SDR one by
/// its own matrix and range.
fn thumbnail(
    scaler: &mut Option<ffmpeg_next::software::scaling::Context>,
    frame: &ffmpeg_next::frame::Video,
    (width, height): (u32, u32),
    color: ColorInfo,
) -> Result<CpuFrame> {
    use ffmpeg_next::software::scaling::{Context, Flags};
    let hdr = crate::hdr::is_hdr(color.transfer);
    let target = if hdr {
        Pixel::YUV444P16LE
    } else {
        Pixel::RGB24
    };
    let scaler = match scaler {
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
        None => scaler.insert(
            Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                target,
                width,
                height,
                Flags::BILINEAR,
            )
            .map_err(|error| VideoError::Demux(format!("swscale thumbnail: {error}")))?,
        ),
    };
    if !hdr {
        super::decode::rgb_conversion(scaler, color)?;
    }
    let mut scaled = ffmpeg_next::frame::Video::empty();
    scaler
        .run(frame, &mut scaled)
        .map_err(|error| VideoError::Demux(format!("sws_scale thumbnail: {error}")))?;
    if hdr {
        return crate::hdr::to_sdr(
            &super::decode::picture(&scaled)?,
            color,
            CpuPixelFormat::Rgb24,
        );
    }
    let stride = scaled.stride(0);
    let rows = height as usize;
    let data = scaled.data(0);
    if data.len() < stride.saturating_mul(rows) {
        return Err(VideoError::InvalidData(
            "swscale produced a short thumbnail".into(),
        ));
    }
    CpuFrame::new(
        width,
        height,
        stride,
        CpuPixelFormat::Rgb24,
        &data[..stride * rows],
    )
}

impl Pipeline<'_> {
    fn encode(&mut self, frame: &ffmpeg_next::frame::Video) -> Result<()> {
        let (width, height) = self.extent;
        let mut scaled = ffmpeg_next::frame::Video::empty();
        // SAFETY: both frames are live; the empty one gets its buffers from
        // the scaler. Same colour on both sides: swscale rescales YUV only.
        unsafe {
            let (source, target) = (frame.as_ptr(), scaled.as_mut_ptr());
            (*target).width = width as i32;
            (*target).height = height as i32;
            (*target).format = ffi::AVPixelFormat::from(self.format) as i32;
            (*target).color_range = (*source).color_range;
            (*target).colorspace = (*source).colorspace;
            (*target).color_primaries = (*source).color_primaries;
            (*target).color_trc = (*source).color_trc;
            (*target).chroma_location = (*source).chroma_location;
            check_av(
                "sws_scale_frame",
                ffi::sws_scale_frame(self.scaler.0, target, source),
            )?;
        }
        let pts = frame.timestamp().or(frame.pts());
        scaled.set_pts(pts);
        if let Some(thumbs) = &mut self.thumbs {
            thumbs.offer(&scaled, pts, self.time_base)?;
        }
        if let Some(pts) = pts {
            // SAFETY: a field of the live decoded frame.
            let duration = unsafe { (*frame.as_ptr()).duration };
            self.durations
                .insert(pts, if duration > 0 { duration } else { self.nominal });
        }
        self.encoder
            .send_frame(&scaled)
            .map_err(|error| VideoError::Demux(format!("{}: {error}", self.encoder_name)))?;
        self.write()
    }

    fn write(&mut self) -> Result<()> {
        let mut packet = ffmpeg_next::Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            let at = |value: Option<i64>| value.map(|value| MediaTime::new(value, self.time_base));
            let pushed = packet.pts().and_then(|pts| self.durations.remove(&pts));
            let duration = Some(packet.duration())
                .filter(|duration| *duration > 0)
                .or(pushed)
                .filter(|duration| *duration > 0);
            self.session.write_packet(
                self.track,
                EncodedPacket {
                    stream_index: 0,
                    pts: at(packet.pts()),
                    dts: at(packet.dts()),
                    duration: duration.map(|duration| MediaTime::new(duration, self.time_base)),
                    is_sync: packet.is_key(),
                    data: packet.data().unwrap_or_default().to_vec(),
                    packetization: Packetization::Elementary,
                },
            )?;
        }
        Ok(())
    }
}

/// libswscale's frame API on every core: a 4K-to-1080p bicubic pass ran on
/// the encoding thread and held the whole pipeline back.
struct Scaler(*mut ffi::SwsContext);

// SAFETY: the context is used by one pipeline at a time and never shared.
unsafe impl Send for Scaler {}

impl Scaler {
    fn new() -> Result<Self> {
        // SAFETY: a fresh context; the fields are documented public options.
        unsafe {
            let context = ffi::sws_alloc_context();
            if context.is_null() {
                return Err(VideoError::BackendUnavailable(
                    "could not allocate a scaler".into(),
                ));
            }
            (*context).threads = 0;
            (*context).flags = ffi::SwsFlags::SWS_BICUBIC as u32;
            Ok(Self(context))
        }
    }
}

impl Drop for Scaler {
    fn drop(&mut self) {
        // SAFETY: this type owns the context; freeing nulls the pointer.
        unsafe { ffi::sws_free_context(&mut self.0) };
    }
}

/// VideoToolbox where the build has it (macOS), else `prores_ks`, with the
/// 10-bit 4:2:2 layout each takes.
fn encoder() -> Result<(ffmpeg_next::Codec, &'static str, Pixel)> {
    for (name, format) in [
        ("prores_videotoolbox", Pixel::P210LE),
        ("prores_ks", Pixel::YUV422P10LE),
    ] {
        if let Some(codec) = ffmpeg_next::encoder::find_by_name(name) {
            return Ok((codec, name, format));
        }
    }
    Err(VideoError::UnsupportedCodec(
        "no ProRes encoder in this build".into(),
    ))
}

/// The stream's colour as libav writes it: range, primaries, transfer, matrix.
fn parameters_color(
    input: &Input,
    stream_index: u32,
) -> Result<(
    ffi::AVColorRange,
    ffi::AVColorPrimaries,
    ffi::AVColorTransferCharacteristic,
    ffi::AVColorSpace,
)> {
    let stream = input
        .stream(stream_index as usize)
        .ok_or_else(|| VideoError::InvalidData(format!("no stream {stream_index}")))?;
    // SAFETY: `codecpar` lives as long as the open input.
    let raw = unsafe { &*stream.parameters().as_ptr() };
    Ok((
        raw.color_range,
        raw.color_primaries,
        raw.color_trc,
        raw.color_space,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_short_edge_is_capped_never_raised_and_both_sides_stay_even() {
        assert_eq!(proxy_extent(3840, 2160, 1080), (1920, 1080));
        assert_eq!(proxy_extent(2160, 3840, 1080), (1080, 1920));
        assert_eq!(proxy_extent(1280, 720, 1080), (1280, 720));
        assert_eq!(proxy_extent(1921, 1081, 1080), (1920, 1080));
        assert_eq!(proxy_extent(4096, 1716, 1080), (2578, 1080));
    }
}
