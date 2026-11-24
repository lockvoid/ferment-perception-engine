//! `MediaInfo` from `libavformat`, field by field.
//!
//! This is ruling 7 of the libav track, and the parity suite
//! (`tests/libav_parity.rs`) is what says it is right: every field below is
//! compared against the hand-built demuxer's answer for every fixture in the
//! corpus, with a written tolerance for the few that legitimately differ.
//!
//! `ffmpeg-next` wraps very little of `AVCodecParameters`, so most of this
//! reads the struct directly. Each block is small, reads one field, and says
//! which one.

use std::ffi::CStr;
use std::path::Path;
mod nut;

use ffmpeg_next::ffi;

use super::{check_av, init};
use crate::{
    AudioTiming, Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, ContainerFormat,
    DecoderConfiguration, MediaInfo, MediaKind, MediaTime, ProResProfile, Result, StreamInfo,
    TimeBase, TransferCharacteristic, VideoError,
};

/// Open `source` with libavformat and describe it.
///
/// `avformat_find_stream_info` is part of opening: without it a Matroska
/// stream has no frame rate and an MP4's `codecpar` may be missing the
/// extradata every decoder configuration is parsed from.
pub fn probe(source: &Path) -> Result<MediaInfo> {
    init()?;
    let mut input =
        ffmpeg_next::format::input(source).map_err(|error| map_open_error(source, error))?;
    describe(&mut input)
}

/// Probe a verified seekable stream on a worker, without opening any pathname.
pub fn probe_stream(source: Box<dyn crate::MediaInput>) -> Result<MediaInfo> {
    let mut input = super::input::Input::stream(source)?;
    let info = describe(&mut input);
    input.check()?;
    info
}

pub(super) fn describe(input: &mut ffmpeg_next::format::context::Input) -> Result<MediaInfo> {
    let priming = priming_per_stream(input);

    // SAFETY: `iformat` is set for the lifetime of an opened input context.
    let format_name = unsafe {
        let format = (*input.as_ptr()).iformat;
        if format.is_null() {
            return Err(VideoError::UnsupportedContainer(
                "libavformat opened the file without naming a demuxer".into(),
            ));
        }
        cstr_to_string((*format).name).unwrap_or_default()
    };

    let mut container_duration = duration_of(input.duration(), AV_TIME_BASE_Q);
    let extents = if format_name == "nut" {
        nut::durations(input)?
    } else {
        Default::default()
    };
    // An untagged CBR MP3 has no declared length: libavformat divides file
    // size by bit rate and says so. That number is fine as a duration and
    // wrong as `AudioTiming::audible_samples`, which means "after applying
    // container timing metadata" — of which there is none.
    // SAFETY: a plain field of the open context.
    let duration_is_estimated = unsafe {
        (*input.as_ptr()).duration_estimation_method
            == ffi::AVDurationEstimationMethod::AVFMT_DURATION_FROM_BITRATE
    };
    let streams = input
        .streams()
        .map(|stream| {
            let priming = priming.get(&stream.index()).copied().unwrap_or_default();
            let mut info =
                stream_info(&stream, container_duration, duration_is_estimated, priming)?;
            if let Some(duration) = extents.get(&stream.index()) {
                info.duration = Some(*duration);
            }
            Ok(info)
        })
        .collect::<Result<Vec<_>>>()?;
    // The one-stream recording retains its exact native rational extent. A
    // multi-stream container keeps libav's global offset/interleave duration;
    // each observed video stream still reports its own complete frame span.
    if streams.len() == 1
        && let Some(duration) = extents.values().next()
    {
        container_duration = Some(*duration);
    }

    Ok(MediaInfo {
        container: container_format(&format_name),
        duration: container_duration,
        streams,
    })
}

/// libavformat's microsecond container time base, as a [`TimeBase`].
const AV_TIME_BASE_Q: TimeBase = TimeBase::MICROSECONDS;

/// A demuxer's short name is a comma-separated family — the `mov` demuxer
/// answers to `"mov,mp4,m4a,3gp,3g2,mj2"` — so the whole string is the key,
/// not one element of it.
fn container_format(name: &str) -> ContainerFormat {
    match name {
        "mov,mp4,m4a,3gp,3g2,mj2" | "mov" | "mp4" | "m4a" => ContainerFormat::Mp4,
        "matroska,webm" | "matroska" | "webm" => ContainerFormat::Matroska,
        "mp3" => ContainerFormat::Mp3,
        other => ContainerFormat::Other(other.to_owned()),
    }
}

fn stream_info(
    stream: &ffmpeg_next::format::stream::Stream<'_>,
    container_duration: Option<MediaTime>,
    duration_is_estimated: bool,
    priming: Priming,
) -> Result<StreamInfo> {
    let parameters = stream.parameters();
    // SAFETY: `parameters` borrows the open context's `codecpar`, which lives
    // as long as the stream. Every read below is of a plain field.
    let raw = unsafe { &*parameters.as_ptr() };

    let time_base = rational_time_base(stream.time_base())?;
    let kind = media_kind(raw.codec_type);
    let codec = codec_of(raw.codec_id, raw.profile);
    let is_video = kind == MediaKind::Video;
    let is_audio = kind == MediaKind::Audio;

    // SAFETY: `extradata` is `extradata_size` readable bytes or null.
    let extradata = unsafe {
        if raw.extradata.is_null() || raw.extradata_size <= 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(raw.extradata, raw.extradata_size as usize).to_vec()
        }
    };

    // A stream with no duration of its own borrows the container's — a
    // Matroska track carries neither `duration` nor `nb_frames`.
    let duration = duration_of(stream.duration(), time_base)
        .or_else(|| container_duration.map(|total| total.rescale(time_base)));

    Ok(StreamInfo {
        index: u32::try_from(stream.index()).unwrap_or(u32::MAX),
        kind,
        codec: codec.clone(),
        time_base,
        start_time: duration_of(stream.start_time(), time_base)
            .or(Some(MediaTime::new(0, time_base))),
        duration,
        width: is_video.then(|| raw.width.max(0) as u32),
        height: is_video.then(|| raw.height.max(0) as u32),
        sample_rate: is_audio.then(|| raw.sample_rate.max(0) as u32),
        channels: is_audio.then(|| u16::try_from(raw.ch_layout.nb_channels).unwrap_or(0)),
        frame_rate: is_video.then(|| frame_rate(stream)).flatten(),
        display_matrix: display_matrix(stream),
        bit_rate: (raw.bit_rate > 0).then_some(raw.bit_rate as u64),
        language: language(stream),
        color: color_info(raw),
        audio_timing: is_audio
            .then(|| audio_timing(raw, stream, duration, duration_is_estimated, priming)),
        decoder_configuration: DecoderConfiguration::from_codec_private(codec, extradata)?,
    })
}

/// `AV_NOPTS_VALUE` is libav's "no timestamp", and it is `i64::MIN` — read as
/// a number it would be a duration of minus three hundred thousand years.
fn duration_of(ticks: i64, time_base: TimeBase) -> Option<MediaTime> {
    (ticks != ffi::AV_NOPTS_VALUE).then(|| MediaTime::new(ticks, time_base))
}

fn rational_time_base(rational: ffmpeg_next::Rational) -> Result<TimeBase> {
    TimeBase::new(
        i64::from(rational.numerator()),
        i64::from(rational.denominator()),
    )
}

fn media_kind(kind: ffi::AVMediaType) -> MediaKind {
    match kind {
        ffi::AVMediaType::AVMEDIA_TYPE_VIDEO => MediaKind::Video,
        ffi::AVMediaType::AVMEDIA_TYPE_AUDIO => MediaKind::Audio,
        ffi::AVMediaType::AVMEDIA_TYPE_SUBTITLE => MediaKind::Subtitle,
        ffi::AVMediaType::AVMEDIA_TYPE_DATA | ffi::AVMediaType::AVMEDIA_TYPE_ATTACHMENT => {
            MediaKind::Data
        }
        _ => MediaKind::Unknown,
    }
}

fn codec_of(id: ffi::AVCodecID, profile: i32) -> Codec {
    use ffi::AVCodecID::*;
    match id {
        AV_CODEC_ID_H264 => Codec::H264,
        AV_CODEC_ID_HEVC => Codec::Hevc,
        AV_CODEC_ID_PRORES => Codec::ProRes(prores_profile(profile)),
        AV_CODEC_ID_AAC => Codec::Aac,
        AV_CODEC_ID_MP3 => Codec::Mp3,
        AV_CODEC_ID_VP8 => Codec::Vp8,
        AV_CODEC_ID_VP9 => Codec::Vp9,
        AV_CODEC_ID_AV1 => Codec::Av1,
        other => Codec::Other(codec_name(other)),
    }
}

/// `libavcodec/profiles.h`'s ProRes ladder. `FF_PROFILE_UNKNOWN` is -99, and
/// a Matroska ProRes track routinely has no profile at all.
fn prores_profile(profile: i32) -> ProResProfile {
    match profile {
        0 => ProResProfile::Proxy,
        1 => ProResProfile::Lt,
        2 => ProResProfile::Standard,
        3 => ProResProfile::Hq,
        4 => ProResProfile::P4444,
        5 => ProResProfile::P4444Xq,
        _ => ProResProfile::Unknown,
    }
}

fn codec_name(id: ffi::AVCodecID) -> String {
    // SAFETY: `avcodec_get_name` returns a static string for every id,
    // including unknown ones ("none").
    unsafe { cstr_to_string(ffi::avcodec_get_name(id)) }.unwrap_or_else(|| "unknown".into())
}

/// `avg_frame_rate` is the container's own average; `r_frame_rate` is
/// libavformat's guess at the lowest rate every timestamp is a multiple of.
/// The average is the honest answer for a variable-rate file, so it wins, and
/// the guess is only the fallback when the container declared nothing.
fn frame_rate(stream: &ffmpeg_next::format::stream::Stream<'_>) -> Option<(u32, u32)> {
    let positive = |rate: ffmpeg_next::Rational| {
        (rate.numerator() > 0 && rate.denominator() > 0)
            .then(|| (rate.numerator() as u32, rate.denominator() as u32))
    };
    positive(stream.avg_frame_rate()).or_else(|| positive(stream.rate()))
}

/// The ISO-BMFF track matrix, from the stream's display-matrix side data.
/// libav hands it over as nine `int32` in the same 16.16/2.30 fixed-point row
/// order `StreamInfo::display_matrix` documents.
fn display_matrix(stream: &ffmpeg_next::format::stream::Stream<'_>) -> Option<[i32; 9]> {
    for side_data in stream.side_data() {
        if side_data.kind() != ffmpeg_next::codec::packet::side_data::Type::DisplayMatrix {
            continue;
        }
        let bytes = side_data.data();
        if bytes.len() < 36 {
            continue;
        }
        let mut matrix = [0i32; 9];
        for (slot, chunk) in matrix.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
            *slot = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        return Some(matrix);
    }
    None
}

fn language(stream: &ffmpeg_next::format::stream::Stream<'_>) -> Option<String> {
    stream.metadata().get("language").map(str::to_owned)
}

pub(super) fn color_info(raw: &ffi::AVCodecParameters) -> ColorInfo {
    ColorInfo {
        range: match raw.color_range {
            ffi::AVColorRange::AVCOL_RANGE_MPEG => ColorRange::Limited,
            ffi::AVColorRange::AVCOL_RANGE_JPEG => ColorRange::Full,
            ffi::AVColorRange::AVCOL_RANGE_UNSPECIFIED => ColorRange::Unspecified,
            other => ColorRange::Other(other as u16),
        },
        primaries: match raw.color_primaries {
            ffi::AVColorPrimaries::AVCOL_PRI_BT709 => ColorPrimaries::Bt709,
            ffi::AVColorPrimaries::AVCOL_PRI_BT470M => ColorPrimaries::Bt470M,
            ffi::AVColorPrimaries::AVCOL_PRI_BT470BG => ColorPrimaries::Bt470Bg,
            ffi::AVColorPrimaries::AVCOL_PRI_SMPTE170M => ColorPrimaries::Smpte170M,
            ffi::AVColorPrimaries::AVCOL_PRI_SMPTE240M => ColorPrimaries::Smpte240M,
            ffi::AVColorPrimaries::AVCOL_PRI_FILM => ColorPrimaries::Film,
            ffi::AVColorPrimaries::AVCOL_PRI_BT2020 => ColorPrimaries::Bt2020,
            ffi::AVColorPrimaries::AVCOL_PRI_SMPTE432 => ColorPrimaries::DisplayP3,
            ffi::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED => ColorPrimaries::Unspecified,
            other => ColorPrimaries::Other(other as u16),
        },
        transfer: match raw.color_trc {
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709 => TransferCharacteristic::Bt709,
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE170M => {
                TransferCharacteristic::Smpte170M
            }
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_LINEAR => TransferCharacteristic::Linear,
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_IEC61966_2_1 => {
                TransferCharacteristic::Srgb
            }
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT2020_10 => {
                TransferCharacteristic::Bt2020TenBit
            }
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT2020_12 => {
                TransferCharacteristic::Bt2020TwelveBit
            }
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE2084 => TransferCharacteristic::Pq,
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_ARIB_STD_B67 => {
                TransferCharacteristic::Hlg
            }
            ffi::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED => {
                TransferCharacteristic::Unspecified
            }
            other => TransferCharacteristic::Other(other as u16),
        },
        matrix: match raw.color_space {
            ffi::AVColorSpace::AVCOL_SPC_RGB => ColorMatrix::Identity,
            ffi::AVColorSpace::AVCOL_SPC_BT709 => ColorMatrix::Bt709,
            ffi::AVColorSpace::AVCOL_SPC_FCC => ColorMatrix::Fcc,
            ffi::AVColorSpace::AVCOL_SPC_BT470BG => ColorMatrix::Bt470Bg,
            ffi::AVColorSpace::AVCOL_SPC_SMPTE170M => ColorMatrix::Smpte170M,
            ffi::AVColorSpace::AVCOL_SPC_SMPTE240M => ColorMatrix::Smpte240M,
            ffi::AVColorSpace::AVCOL_SPC_BT2020_NCL => ColorMatrix::Bt2020NonConstant,
            ffi::AVColorSpace::AVCOL_SPC_BT2020_CL => ColorMatrix::Bt2020Constant,
            ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED => ColorMatrix::Unspecified,
            other => ColorMatrix::Other(other as u16),
        },
        bits_per_channel: (raw.bits_per_raw_sample > 0)
            .then(|| u8::try_from(raw.bits_per_raw_sample).unwrap_or(u8::MAX))
            .or_else(|| {
                (raw.codec_type == ffi::AVMediaType::AVMEDIA_TYPE_VIDEO)
                    .then(|| pixel_depth(raw.format))
                    .flatten()
            }),
    }
}

/// Many compressed codecs publish their depth through the decoded pixel
/// format, leaving bits_per_raw_sample unset. Do not cast an arbitrary C int
/// to a Rust enum; enumerate libav's static descriptors at probe time instead.
fn pixel_depth(format: i32) -> Option<u8> {
    let mut descriptor = std::ptr::null();
    loop {
        // SAFETY: null starts the static table; subsequent pointers came from
        // this same iterator. The library retains all descriptors for life.
        descriptor = unsafe { ffi::av_pix_fmt_desc_next(descriptor) };
        if descriptor.is_null() {
            return None;
        }
        // SAFETY: a live library descriptor returned above.
        if unsafe { ffi::av_pix_fmt_desc_get_id(descriptor) } as i32 == format {
            let descriptor = unsafe { &*descriptor };
            return descriptor
                .comp
                .iter()
                .take(usize::from(descriptor.nb_components))
                .filter_map(|component| u8::try_from(component.depth).ok())
                .filter(|depth| *depth > 0)
                .max();
        }
    }
}

/// What a stream's own packets say about its length, which `codecpar` does
/// not: the trims at each end, and how many samples one coded frame holds.
#[derive(Debug, Clone, Copy, Default)]
struct Priming {
    leading: u32,
    trailing: u32,
    /// Samples per coded frame, taken from the first packet's duration.
    /// `codecpar->frame_size` is the obvious source and is the one used when
    /// it is set, but it is filled by DECODING a frame — so it stays 0 for
    /// any stream libavcodec could not decode during `find_stream_info`,
    /// while the container's own packet duration is right there.
    frame_length: u32,
}

/// The priming and padding libavformat attaches to an audio stream's first
/// and last packets.
///
/// This is what `codecpar` does not carry. An AAC track's 1024 encoder-delay
/// samples, an HE-AAC track's 2112, and a LAME-tagged MP3's Xing delay all
/// arrive as `AV_PKT_DATA_SKIP_SAMPLES` on the FIRST packet, with the end trim
/// as the same side data's `discard_padding` on the LAST one;
/// `codecpar->initial_padding` stays zero for every one of them.
/// Sample-accurate audio depends on both numbers, so the probe pays for a
/// bounded look at each end of the file.
fn priming_per_stream(
    input: &mut ffmpeg_next::format::context::Input,
) -> std::collections::HashMap<usize, Priming> {
    let audio: std::collections::HashSet<usize> = input
        .streams()
        .filter(|stream| stream.parameters().medium() == ffmpeg_next::media::Type::Audio)
        .map(|stream| stream.index())
        .collect();
    if audio.is_empty() {
        return std::collections::HashMap::new();
    }

    let mut priming: std::collections::HashMap<usize, Priming> = std::collections::HashMap::new();
    for (index, leading, frame_length) in leading_from_head(input, &audio) {
        let entry = priming.entry(index).or_default();
        entry.leading = leading;
        entry.frame_length = frame_length;
    }
    for (index, trailing) in trailing_from_tail(input, &audio) {
        priming.entry(index).or_default().trailing = trailing;
    }
    // `packets()` left the read head wherever it stopped. Nothing below reads
    // packets, and the context is dropped at the end of `probe`.
    priming
}

/// One packet per audio stream is all the leading trim needs. The ceiling is
/// for an interleaving that puts a track's first packet unusually late; a
/// stream never seen keeps the default, which is what `codecpar` already said.
fn leading_from_head(
    input: &mut ffmpeg_next::format::context::Input,
    audio: &std::collections::HashSet<usize>,
) -> Vec<(usize, u32, u32)> {
    const MAX_PACKETS: usize = 256;
    let mut found = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (stream, packet) in input.packets().take(MAX_PACKETS) {
        let index = stream.index();
        if !audio.contains(&index) || !seen.insert(index) {
            continue;
        }
        let leading = skip_samples(&packet).map_or(0, |(leading, _)| leading);
        // The packet's duration is in the stream's time base, which for an
        // audio stream is 1/sample_rate — so the tick count IS the sample
        // count. Rescaling anyway keeps it honest for a demuxer that chose
        // some other base.
        let frame_length = u32::try_from(
            MediaTime::new(packet.duration(), stream_time_base(&stream))
                .rescale(sample_time_base(&stream))
                .value,
        )
        .unwrap_or(0);
        found.push((index, leading, frame_length));
        if seen.len() == audio.len() {
            break;
        }
    }
    found
}

fn stream_time_base(stream: &ffmpeg_next::format::stream::Stream<'_>) -> TimeBase {
    rational_time_base(stream.time_base()).unwrap_or(TimeBase::MICROSECONDS)
}

fn sample_time_base(stream: &ffmpeg_next::format::stream::Stream<'_>) -> TimeBase {
    // SAFETY: `codecpar` lives as long as the stream; this reads one field.
    let rate = unsafe { (*stream.parameters().as_ptr()).sample_rate };
    TimeBase::new(1, i64::from(rate)).unwrap_or(TimeBase::MICROSECONDS)
}

/// The end trim rides the LAST packet, so reading the head cannot find it and
/// reading to EOF would make every probe proportional to file length. Seek to
/// a second before the end instead and read out from there: bounded work, and
/// the last packet of each stream is necessarily inside that window.
///
/// A file libavformat cannot seek, or one with no duration to seek relative
/// to, simply reports no padding — the same answer as before this existed.
fn trailing_from_tail(
    input: &mut ffmpeg_next::format::context::Input,
    audio: &std::collections::HashSet<usize>,
) -> Vec<(usize, u32)> {
    const TAIL_WINDOW: i64 = ffi::AV_TIME_BASE as i64;
    let duration = input.duration();
    if duration == ffi::AV_NOPTS_VALUE || duration <= 0 {
        return Vec::new();
    }
    let target = (duration - TAIL_WINDOW).max(0);
    if input.seek(target, target..duration).is_err() {
        return Vec::new();
    }

    let mut last: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for (stream, packet) in input.packets() {
        let index = stream.index();
        if !audio.contains(&index) {
            continue;
        }
        if let Some((_, trailing)) = skip_samples(&packet) {
            // Only the final packet carries a non-zero padding; keeping the
            // last non-zero one seen is the same thing without depending on
            // which packet EOF lands after.
            if trailing > 0 {
                last.insert(index, trailing);
            }
        }
    }
    last.into_iter().collect()
}

/// `(skip from start, discard from end)` out of one packet's skip-samples
/// side data, if it carries any.
fn skip_samples(packet: &ffmpeg_next::codec::packet::Packet) -> Option<(u32, u32)> {
    use ffmpeg_next::codec::packet::side_data::Type;
    packet.side_data().find_map(|side_data| {
        if side_data.kind() != Type::SkipSamples {
            return None;
        }
        // `libavcodec/packet.h`: u32 samples to skip from the start, u32 to
        // discard from the end, then two 8-bit reasons. Little-endian, always.
        let bytes = side_data.data();
        (bytes.len() >= 8).then(|| {
            (
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            )
        })
    })
}

/// Encoder priming and padding.
///
/// `audible_samples` is libav's stream duration: every demuxer here reports
/// the PRESENTATION length, with the edit list already applied for MP4 and
/// the Xing/LAME header already applied for MP3. It is the audible program by
/// definition, so it is read rather than derived — EXCEPT when libavformat
/// estimated it from the bit rate, which is not metadata and is not exact.
///
/// `trailing_samples` is the one number nothing hands over directly for MP4.
/// libavformat turns a MOV edit list's HEAD trim into skip-samples side data
/// but drops its TAIL trim, so the remainder is recovered by arithmetic:
/// the coded length is `nb_frames` whole frames, and what is neither priming
/// nor audible is padding.
fn audio_timing(
    raw: &ffi::AVCodecParameters,
    stream: &ffmpeg_next::format::stream::Stream<'_>,
    duration: Option<MediaTime>,
    duration_is_estimated: bool,
    priming: Priming,
) -> AudioTiming {
    // Three places say "priming", and they do not overlap: the packet's
    // skip-samples side data (AAC delay, LAME gapless), `initial_padding`
    // (Matroska's CodecDelay), and `seek_preroll` (Opus's decoded lead-in).
    // Whichever is set is the answer; taking the max avoids ordering them by
    // guesswork.
    let leading_samples = priming
        .leading
        .max(raw.initial_padding.max(0) as u32)
        .max(raw.seek_preroll.max(0) as u32);

    let rate = i64::from(raw.sample_rate);
    let audible_samples = duration
        .filter(|_| !duration_is_estimated)
        .and_then(|duration| {
            (rate > 0)
                .then(|| duration.rescale(TimeBase::new_unchecked(1, rate)).value)
                .and_then(|samples| u64::try_from(samples).ok())
        });

    let trailing_samples = priming
        .trailing
        .max(raw.trailing_padding.max(0) as u32)
        .max(derived_trailing(
            frame_length(raw, priming),
            stream,
            leading_samples,
            audible_samples,
        ));

    AudioTiming {
        leading_samples,
        trailing_samples,
        audible_samples,
    }
}

/// The padding a MOV edit list trimmed off the end, recovered as
/// `coded - priming - audible`.
///
/// Only sound when the demuxer counted the frames (`nb_frames`) and the codec
/// has a fixed frame length — an MP4's sample table gives both, an MP3 or
/// Matroska stream gives neither, and those carry the number in side data
/// instead. The result is refused unless it is under one coded frame, which
/// is what padding IS: the encoder filling out its last frame. A larger
/// answer means one of the two inputs was not what it looked like.
fn frame_length(raw: &ffi::AVCodecParameters, priming: Priming) -> u32 {
    match u32::try_from(raw.frame_size) {
        Ok(size) if size > 0 => size,
        _ => priming.frame_length,
    }
}

fn derived_trailing(
    frame_size: u32,
    stream: &ffmpeg_next::format::stream::Stream<'_>,
    leading: u32,
    audible: Option<u64>,
) -> u32 {
    let (Ok(frames), frame_size) = (u64::try_from(stream.frames()), u64::from(frame_size)) else {
        return 0;
    };
    let (Some(audible), true) = (audible, frames > 0 && frame_size > 0) else {
        return 0;
    };
    let coded = frames.saturating_mul(frame_size);
    let trailing = coded
        .saturating_sub(u64::from(leading))
        .saturating_sub(audible);
    if trailing > frame_size {
        return 0;
    }
    u32::try_from(trailing).unwrap_or(0)
}

/// # Safety
///
/// `pointer` is either null or a NUL-terminated C string that outlives the
/// call.
unsafe fn cstr_to_string(pointer: *const std::ffi::c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    // SAFETY: the caller guarantees a live NUL-terminated string.
    let text = unsafe { CStr::from_ptr(pointer) };
    Some(String::from_utf8_lossy(text.to_bytes()).into_owned())
}

/// A file libavformat refuses is not always a broken file — it is just as
/// often a path that is not there. Keep the distinction the callers already
/// make, and let everything else carry `av_strerror`'s words.
fn map_open_error(source: &Path, error: ffmpeg_next::Error) -> VideoError {
    if let ffmpeg_next::Error::Other { errno } = error
        && errno == libc_enoent()
    {
        return VideoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} could not be opened", source.display()),
        ));
    }
    match check_av("avformat_open_input", raw_code(error)) {
        Ok(_) => VideoError::Demux(error.to_string()),
        Err(mapped) => mapped,
    }
}

fn raw_code(error: ffmpeg_next::Error) -> i32 {
    // `Error`'s own `Into<c_int>` is the AVERROR code it was built from.
    error.into()
}

fn libc_enoent() -> i32 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demuxer_family_names_collapse_to_the_three_we_name() {
        assert_eq!(
            container_format("mov,mp4,m4a,3gp,3g2,mj2"),
            ContainerFormat::Mp4
        );
        assert_eq!(container_format("matroska,webm"), ContainerFormat::Matroska);
        assert_eq!(container_format("mp3"), ContainerFormat::Mp3);
        assert_eq!(
            container_format("wav"),
            ContainerFormat::Other("wav".into()),
            "a container we do not name keeps libav's spelling"
        );
    }

    /// `AV_NOPTS_VALUE` is `i64::MIN`; treating it as a timestamp is the
    /// classic libav bug, and it reads as a duration of -292471 years.
    #[test]
    fn the_no_timestamp_sentinel_is_absence_not_a_huge_negative_time() {
        assert_eq!(
            duration_of(ffi::AV_NOPTS_VALUE, TimeBase::MILLISECONDS),
            None
        );
        assert_eq!(
            duration_of(1_500, TimeBase::MILLISECONDS),
            Some(MediaTime::new(1_500, TimeBase::MILLISECONDS))
        );
    }

    #[test]
    fn prores_profiles_follow_libavcodecs_ladder_and_unknown_stays_unknown() {
        assert_eq!(prores_profile(0), ProResProfile::Proxy);
        assert_eq!(prores_profile(3), ProResProfile::Hq);
        assert_eq!(prores_profile(5), ProResProfile::P4444Xq);
        assert_eq!(prores_profile(-99), ProResProfile::Unknown);
    }
}
