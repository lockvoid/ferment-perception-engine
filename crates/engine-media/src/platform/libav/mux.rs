//! MP4 and QuickTime output through `libavformat`'s `mov` muxer.
//!
//! Ruling 11. This replaces a 2 300-line hand-written ISO-BMFF writer with the
//! one that has been writing these files for twenty years — including the
//! parts that are easy to get subtly wrong and hard to notice: the edit list
//! that carries encoder priming, `iTunSMPB`, `colr`, chunk interleaving, and
//! `faststart`'s second pass that moves `moov` to the front.
//!
//! The session is a thin state machine over one `AVFormatContext`, because
//! libavformat's own ordering rules are strict: every stream must exist before
//! the header is written, and no packet may be written before it. Tracks are
//! declared, the header goes out lazily on the first packet, and `finalize`
//! writes the trailer.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use ffmpeg_next::codec::packet::traits::Mut as PacketMut;
use ffmpeg_next::ffi;

use super::{check_av, init};
use crate::{
    AudioMuxTrack, Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, DecoderConfiguration,
    EncodedPacket, MediaTime, MuxFormat, MuxOutput, MuxSession, MuxSettings, MuxTrackId, Muxer,
    Packetization, Result, TimeBase, TransferCharacteristic, VideoError, VideoMuxTrack,
};

/// The container writer. One type for both formats; `MuxSettings::format`
/// picks which the `mov` muxer family writes.
#[derive(Debug, Clone, Copy, Default)]
pub struct LibavMuxer;

impl LibavMuxer {
    /// A session that writes into memory instead of a file, returning the
    /// bytes in [`MuxOutput::bytes`].
    ///
    /// libavformat's `avio_open_dyn_buf` — the same muxer, a different sink.
    /// `faststart` is refused here rather than silently ignored: its second
    /// pass reopens the output for reading, which a dynamic buffer cannot do,
    /// and an export that quietly lost `faststart` would be discovered by a
    /// player seeking to the end of a file it was meant to stream.
    pub fn memory(settings: MuxSettings) -> Result<Box<dyn MuxSession>> {
        if settings.fast_start {
            return Err(VideoError::InvalidData(
                "an in-memory mux cannot fast-start: the second pass needs a re-readable output"
                    .into(),
            ));
        }
        Ok(Box::new(LibavMuxSession::open_memory(settings)?))
    }
}

impl Muxer for LibavMuxer {
    fn open(&self, output: &Path, settings: MuxSettings) -> Result<Box<dyn MuxSession>> {
        Ok(Box::new(LibavMuxSession::open(output, settings)?))
    }
}

struct Track {
    stream_index: i32,
    nal: Option<NalMuxConfiguration>,
    /// The presentation extent the CALLER declared: a video track's exact
    /// visible duration after its edit list, or an audio track's audible
    /// sample count. Both are documented on their `*MuxTrack` types as the
    /// authority, and both differ from the packets' coded extent — an audio
    /// track's packets carry the encoder's priming and padding, which are
    /// coded audio but not program.
    declared: Option<MediaTime>,
    /// The furthest packet end seen, in the packets' own base. Used only when
    /// the caller declared nothing.
    packet_end: Option<MediaTime>,
    /// The time base libavformat settled on for this stream, which is NOT
    /// necessarily the one asked for — the `mov` muxer rounds a video track's
    /// timescale and rewrites the field. Packets rescale into this.
    time_base: TimeBase,
}

/// Keep packet framing consistent with the codec-private bytes given to mov.
/// HEVC's hvc1 contract also fixes every parameter set before the first sample.
struct NalMuxConfiguration {
    packetization: Packetization,
    hevc_parameter_sets: Option<Vec<Vec<u8>>>,
}

impl NalMuxConfiguration {
    fn new(configuration: &DecoderConfiguration) -> Result<Option<Self>> {
        if !matches!(configuration.codec, Codec::H264 | Codec::Hevc) {
            return Ok(None);
        }
        // Public fields can be assembled by a caller; derive framing and sets
        // from the bytes the muxer will actually receive, not cached metadata.
        let parsed = DecoderConfiguration::from_codec_private(
            configuration.codec.clone(),
            configuration.description.clone(),
        )?;
        let hevc_parameter_sets = if parsed.codec == Codec::Hevc {
            let units = if parsed.packetization == Packetization::AnnexB {
                let prefixed = crate::mux::annex_b_to_length_prefixed(&parsed.description)?;
                nal_units(&prefixed, 4)?
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect::<Vec<_>>()
            } else {
                let mut units = Vec::new();
                for set in &parsed.parameter_sets {
                    if hevc_kind(&set.bytes)? != set.kind {
                        return Err(VideoError::InvalidData(
                            "HEVC array and NAL header disagree".into(),
                        ));
                    }
                    units.push(set.bytes.clone());
                }
                units
            };
            let mut sets = Vec::new();
            let mut kinds = [false; 3];
            for unit in units {
                let kind = hevc_kind(&unit)?;
                if (32..=34).contains(&kind) {
                    kinds[(kind - 32) as usize] = true;
                    sets.push(unit);
                }
            }
            if kinds != [true; 3] {
                return Err(VideoError::InvalidData(
                    "hvc1 requires out-of-band VPS, SPS and PPS".into(),
                ));
            }
            Some(sets)
        } else {
            None
        };
        Ok(Some(Self {
            packetization: parsed.packetization,
            hevc_parameter_sets,
        }))
    }

    fn packet_data(&self, packet: &EncodedPacket) -> Result<Vec<u8>> {
        let converted;
        let units = match packet.packetization {
            Packetization::AnnexB => {
                converted = crate::mux::annex_b_to_length_prefixed(&packet.data)?;
                nal_units(&converted, 4)?
            }
            Packetization::LengthPrefixed { nal_length_size } => {
                nal_units(&packet.data, nal_length_size)?
            }
            Packetization::Elementary => {
                return Err(VideoError::InvalidData(
                    "NAL codec packet has no framing".into(),
                ));
            }
        };
        let mut output = Vec::with_capacity(packet.data.len());
        for unit in units {
            if let Some(sets) = &self.hevc_parameter_sets
                && (32..=34).contains(&hevc_kind(unit)?)
            {
                if !sets.iter().any(|set| set.as_slice() == unit) {
                    return Err(VideoError::InvalidData(
                        "hvc1 sample changes an out-of-band parameter set".into(),
                    ));
                }
                continue;
            }
            match self.packetization {
                Packetization::AnnexB => output.extend_from_slice(&[0, 0, 0, 1]),
                Packetization::LengthPrefixed { nal_length_size } => {
                    let length = u32::try_from(unit.len())
                        .map_err(|_| VideoError::InvalidData("NAL unit exceeds MP4 u32".into()))?;
                    let width = usize::from(nal_length_size);
                    if width < 4 && u64::from(length) >= (1_u64 << (8 * width)) {
                        return Err(VideoError::InvalidData(
                            "NAL unit exceeds the configured length prefix".into(),
                        ));
                    }
                    output.extend_from_slice(&length.to_be_bytes()[4 - width..]);
                }
                Packetization::Elementary => unreachable!("parsed NAL configuration"),
            }
            output.extend_from_slice(unit);
        }
        Ok(output)
    }
}

fn hevc_kind(unit: &[u8]) -> Result<u8> {
    if unit.len() < 2 || unit[0] & 0x80 != 0 || unit[1] & 7 == 0 {
        return Err(VideoError::InvalidData("invalid HEVC NAL header".into()));
    }
    Ok((unit[0] >> 1) & 0x3f)
}

fn nal_units(data: &[u8], width: u8) -> Result<Vec<&[u8]>> {
    if !(1..=4).contains(&width) {
        return Err(VideoError::InvalidData(
            "invalid NAL length prefix width".into(),
        ));
    }
    let width = usize::from(width);
    let mut cursor = 0;
    let mut units = Vec::new();
    while cursor < data.len() {
        let prefix = data
            .get(cursor..cursor + width)
            .ok_or_else(|| VideoError::InvalidData("truncated NAL length prefix".into()))?;
        let length = prefix
            .iter()
            .fold(0_usize, |n, byte| n * 256 + usize::from(*byte));
        cursor += width;
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| VideoError::InvalidData("NAL extent overflow".into()))?;
        let unit = data
            .get(cursor..end)
            .filter(|unit| !unit.is_empty())
            .ok_or_else(|| VideoError::InvalidData("truncated or empty NAL payload".into()))?;
        units.push(unit);
        cursor = end;
    }
    if units.is_empty() {
        return Err(VideoError::InvalidData(
            "packet contains no NAL units".into(),
        ));
    }
    Ok(units)
}

/// Where the muxed bytes go.
enum Destination {
    Path(PathBuf),
    /// A dynamic buffer, drained at `finalize`.
    Memory,
}

struct LibavMuxSession {
    context: *mut ffi::AVFormatContext,
    destination: Destination,
    settings: MuxSettings,
    tracks: Vec<Track>,
    header_written: bool,
    finalized: bool,
}

// SAFETY: the context is owned solely by this session and never shared; the
// `MuxSession` trait requires `Send` so a session can be moved onto the
// blocking thread that writes the file.
unsafe impl Send for LibavMuxSession {}

impl std::fmt::Debug for LibavMuxSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibavMuxSession")
            .field("path", &self.path())
            .field("format", &self.settings.format)
            .field("tracks", &self.tracks.len())
            .finish_non_exhaustive()
    }
}

impl LibavMuxSession {
    fn open(output: &Path, settings: MuxSettings) -> Result<Self> {
        init()?;
        // `mp4` and `mov` are separate muxer names over the same writer; the
        // brand and a few defaults differ, which is the whole of `MuxFormat`.
        let name = CString::new(match settings.format {
            MuxFormat::Mp4 => "mp4",
            MuxFormat::Mov => "mov",
        })
        .expect("a literal without NUL");
        let path = CString::new(output.as_os_str().as_encoded_bytes())
            .map_err(|_| VideoError::InvalidData("output path contains an interior NUL".into()))?;

        let mut context: *mut ffi::AVFormatContext = std::ptr::null_mut();
        // SAFETY: all three pointers are valid for the call; `context` is an
        // out-parameter this function takes ownership of on success.
        check_av("avformat_alloc_output_context2", unsafe {
            ffi::avformat_alloc_output_context2(
                &mut context,
                std::ptr::null(),
                name.as_ptr(),
                path.as_ptr(),
            )
        })?;
        if context.is_null() {
            return Err(VideoError::UnsupportedContainer(format!(
                "libavformat has no {:?} muxer",
                settings.format
            )));
        }

        let session = Self {
            context,
            destination: Destination::Path(output.to_owned()),
            settings,
            tracks: Vec::new(),
            header_written: false,
            finalized: false,
        };

        // SAFETY: the context is live and not yet writing.
        let status =
            unsafe { ffi::avio_open(&mut (*context).pb, path.as_ptr(), ffi::AVIO_FLAG_WRITE) };
        check_av("avio_open", status)?;
        Ok(session)
    }

    /// The in-memory variant of [`Self::open`].
    fn open_memory(settings: MuxSettings) -> Result<Self> {
        init()?;
        let name = CString::new(match settings.format {
            MuxFormat::Mp4 => "mp4",
            MuxFormat::Mov => "mov",
        })
        .expect("a literal without NUL");

        let mut context: *mut ffi::AVFormatContext = std::ptr::null_mut();
        // SAFETY: valid out-pointer; no filename, because there is no file.
        check_av("avformat_alloc_output_context2", unsafe {
            ffi::avformat_alloc_output_context2(
                &mut context,
                std::ptr::null(),
                name.as_ptr(),
                std::ptr::null(),
            )
        })?;
        if context.is_null() {
            return Err(VideoError::UnsupportedContainer(format!(
                "libavformat has no {:?} muxer",
                settings.format
            )));
        }
        let session = Self {
            context,
            destination: Destination::Memory,
            settings,
            tracks: Vec::new(),
            header_written: false,
            finalized: false,
        };
        // SAFETY: the context is live and not yet writing.
        check_av("avio_open_dyn_buf", unsafe {
            ffi::avio_open_dyn_buf(&mut (*context).pb)
        })?;
        Ok(session)
    }

    fn path(&self) -> Option<&Path> {
        match &self.destination {
            Destination::Path(path) => Some(path),
            Destination::Memory => None,
        }
    }

    /// A new stream carrying `codecpar` filled from a track description.
    fn add_stream(
        &mut self,
        time_base: TimeBase,
        declared: Option<MediaTime>,
    ) -> Result<(*mut ffi::AVStream, MuxTrackId)> {
        if self.header_written {
            return Err(VideoError::InvalidData(
                "cannot add a track after the first packet was written".into(),
            ));
        }
        // SAFETY: the context is live; a null codec means "describe it myself",
        // which is what a remuxer does.
        let stream = unsafe { ffi::avformat_new_stream(self.context, std::ptr::null()) };
        if stream.is_null() {
            return Err(VideoError::BackendUnavailable(
                "libavformat could not allocate a stream".into(),
            ));
        }
        // SAFETY: a freshly allocated stream owned by the context.
        unsafe {
            (*stream).time_base = ffi::AVRational {
                num: time_base.numerator as i32,
                den: time_base.denominator as i32,
            };
        }
        // SAFETY: as above.
        let stream_index = unsafe { (*stream).index };
        let id = MuxTrackId::new(u32::try_from(self.tracks.len()).unwrap_or(u32::MAX));
        self.tracks.push(Track {
            stream_index,
            nal: None,
            declared,
            packet_end: None,
            time_base,
        });
        Ok((stream, id))
    }

    /// Write the header once every track exists.
    ///
    /// Deferred to the first packet because [`MuxSession`] lets a caller add
    /// tracks in any order before writing, and libavformat requires every
    /// stream to exist before `avformat_write_header`.
    fn ensure_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }
        if self.tracks.is_empty() {
            return Err(VideoError::InvalidData(
                "cannot write a container with no tracks".into(),
            ));
        }

        let mut options: *mut ffi::AVDictionary = std::ptr::null_mut();
        // `faststart` is a second pass that relocates `moov` ahead of `mdat`,
        // so a player can start without seeking to the end of the file.
        if self.settings.fast_start {
            set_option(&mut options, "movflags", "+faststart")?;
        }
        if let Some(timescale) = self
            .settings
            .movie_timescale
            .or_else(|| exact_movie_timescale(self.tracks.iter().map(|track| track.time_base)))
        {
            set_option(&mut options, "movie_timescale", &timescale.to_string())?;
        }

        // SAFETY: the context is live with all its streams declared;
        // `options` is a dictionary this call consumes into.
        let status = unsafe { ffi::avformat_write_header(self.context, &mut options) };
        // SAFETY: whatever the header call left unconsumed is ours to free.
        unsafe { ffi::av_dict_free(&mut options) };
        check_av("avformat_write_header", status)?;

        // The muxer may have rewritten each stream's time base — the `mov`
        // muxer picks a timescale of its own for video. Packets have to
        // rescale into what it chose, not what we asked for.
        for track in &mut self.tracks {
            // SAFETY: the stream index came from this context.
            let actual =
                unsafe { (**(*self.context).streams.add(track.stream_index as usize)).time_base };
            if actual.num > 0 && actual.den > 0 {
                track.time_base =
                    TimeBase::new_unchecked(i64::from(actual.num), i64::from(actual.den));
            }
        }
        self.header_written = true;
        Ok(())
    }
}

/// The movie clock carries edit-list delays. FFmpeg's default 1000 Hz rounds
/// a delayed AAC start down by as many as 47 samples. Honor MuxSettings' exact
/// default whenever the reduced track clocks have a representable common rate.
/// FFmpeg stores this option in an int, so an unrepresentable LCM retains its
/// existing default; an explicit caller override is passed through unchanged.
fn exact_movie_timescale(bases: impl Iterator<Item = TimeBase>) -> Option<u32> {
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let mut common = 1_u64;
    for base in bases {
        let numerator = u64::try_from(base.numerator).ok().filter(|n| *n > 0)?;
        let denominator = u64::try_from(base.denominator).ok().filter(|n| *n > 0)?;
        let rate = denominator / gcd(numerator, denominator);
        common = (common / gcd(common, rate)).checked_mul(rate)?;
        if common > i32::MAX as u64 {
            return None;
        }
    }
    Some(common as u32)
}

/// Fill a stream's `codecpar` from a decoder configuration.
///
/// # Safety
///
/// `stream` must be a live stream of an output context whose header has not
/// been written.
unsafe fn describe_codec(
    stream: *mut ffi::AVStream,
    codec: &Codec,
    description: &[u8],
) -> Result<()> {
    // SAFETY: the caller guarantees a live stream.
    let parameters = unsafe { (*stream).codecpar };
    // SAFETY: `codecpar` is allocated with the stream.
    unsafe {
        (*parameters).codec_id = codec_id(codec);
        (*parameters).codec_type = match codec {
            Codec::Aac | Codec::Mp3 => ffi::AVMediaType::AVMEDIA_TYPE_AUDIO,
            _ => ffi::AVMediaType::AVMEDIA_TYPE_VIDEO,
        };
    }
    // ProRes is one codec id with six fourccs, and the container carries the
    // fourcc — `apcn` and `apch` are different files. libavformat picks a
    // default when `codec_tag` is 0, which silently relabels a Standard clip
    // as HQ; the profile is right there in `Codec`, so state it.
    if let Codec::ProRes(profile) = codec {
        // SAFETY: `parameters` is this stream's, allocated with it.
        unsafe { (*parameters).codec_tag = prores_tag(*profile) };
    }
    if *codec == Codec::Hevc {
        // Admission validates out-of-band VPS/SPS/PPS; write_packet refuses
        // new in-band sets and removes duplicates. FFmpeg then writes complete
        // hvcC arrays for the hvc1 sample entry, as the current iOS writer does.
        unsafe { (*parameters).codec_tag = u32::from_le_bytes(*b"hvc1") };
    }
    if description.is_empty() {
        return Ok(());
    }
    // libavcodec reads past the end of extradata when parsing, so the buffer
    // it owns must carry the padding it expects — allocate, do not borrow.
    let size = description.len();
    // SAFETY: a plain allocation of the documented size.
    let buffer =
        unsafe { ffi::av_mallocz(size + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize).cast::<u8>() };
    if buffer.is_null() {
        return Err(VideoError::BackendUnavailable(
            "could not allocate codec extradata".into(),
        ));
    }
    // SAFETY: `buffer` has at least `size` writable bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(description.as_ptr(), buffer, size);
        (*parameters).extradata = buffer;
        (*parameters).extradata_size = size as i32;
    }
    Ok(())
}

/// The ProRes fourcc for a profile, in libavutil's `MKTAG` byte order.
fn prores_tag(profile: crate::ProResProfile) -> u32 {
    use crate::ProResProfile::*;
    let tag: &[u8; 4] = match profile {
        Proxy => b"apco",
        Lt => b"apcs",
        Standard => b"apcn",
        Hq => b"apch",
        P4444 => b"ap4h",
        P4444Xq => b"ap4x",
        // Nothing said which; `apcn` is the middle of the ladder and what a
        // ProRes file without a readable profile most often is.
        Unknown => b"apcn",
    };
    u32::from_le_bytes(*tag)
}

fn codec_id(codec: &Codec) -> ffi::AVCodecID {
    use ffi::AVCodecID::*;
    match codec {
        Codec::H264 => AV_CODEC_ID_H264,
        Codec::Hevc => AV_CODEC_ID_HEVC,
        Codec::ProRes(_) => AV_CODEC_ID_PRORES,
        Codec::Aac => AV_CODEC_ID_AAC,
        Codec::Mp3 => AV_CODEC_ID_MP3,
        Codec::Vp8 => AV_CODEC_ID_VP8,
        Codec::Vp9 => AV_CODEC_ID_VP9,
        Codec::Av1 => AV_CODEC_ID_AV1,
        Codec::Other(_) => AV_CODEC_ID_NONE,
    }
}

/// The colour tags that become the `colr` atom. Unspecified stays
/// unspecified: writing a guess is worse than writing nothing, because a
/// player trusts what the container says.
///
/// # Safety
///
/// `stream` must be a live stream whose header has not been written.
unsafe fn describe_color(stream: *mut ffi::AVStream, color: ColorInfo) {
    // SAFETY: the caller guarantees a live stream.
    let parameters = unsafe { (*stream).codecpar };
    // SAFETY: as above; every write is a plain enum field.
    unsafe {
        (*parameters).color_range = match color.range {
            ColorRange::Limited => ffi::AVColorRange::AVCOL_RANGE_MPEG,
            ColorRange::Full => ffi::AVColorRange::AVCOL_RANGE_JPEG,
            _ => ffi::AVColorRange::AVCOL_RANGE_UNSPECIFIED,
        };
        (*parameters).color_primaries = match color.primaries {
            ColorPrimaries::Bt709 => ffi::AVColorPrimaries::AVCOL_PRI_BT709,
            ColorPrimaries::Bt470M => ffi::AVColorPrimaries::AVCOL_PRI_BT470M,
            ColorPrimaries::Bt470Bg => ffi::AVColorPrimaries::AVCOL_PRI_BT470BG,
            ColorPrimaries::Smpte170M => ffi::AVColorPrimaries::AVCOL_PRI_SMPTE170M,
            ColorPrimaries::Smpte240M => ffi::AVColorPrimaries::AVCOL_PRI_SMPTE240M,
            ColorPrimaries::Film => ffi::AVColorPrimaries::AVCOL_PRI_FILM,
            ColorPrimaries::Bt2020 => ffi::AVColorPrimaries::AVCOL_PRI_BT2020,
            ColorPrimaries::DisplayP3 => ffi::AVColorPrimaries::AVCOL_PRI_SMPTE432,
            _ => ffi::AVColorPrimaries::AVCOL_PRI_UNSPECIFIED,
        };
        (*parameters).color_trc = match color.transfer {
            TransferCharacteristic::Bt709 => ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709,
            TransferCharacteristic::Smpte170M => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE170M
            }
            TransferCharacteristic::Linear => ffi::AVColorTransferCharacteristic::AVCOL_TRC_LINEAR,
            TransferCharacteristic::Srgb => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_IEC61966_2_1
            }
            TransferCharacteristic::Bt2020TenBit => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT2020_10
            }
            TransferCharacteristic::Bt2020TwelveBit => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT2020_12
            }
            TransferCharacteristic::Pq => ffi::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE2084,
            TransferCharacteristic::Hlg => {
                ffi::AVColorTransferCharacteristic::AVCOL_TRC_ARIB_STD_B67
            }
            _ => ffi::AVColorTransferCharacteristic::AVCOL_TRC_UNSPECIFIED,
        };
        (*parameters).color_space = match color.matrix {
            ColorMatrix::Identity => ffi::AVColorSpace::AVCOL_SPC_RGB,
            ColorMatrix::Bt709 => ffi::AVColorSpace::AVCOL_SPC_BT709,
            ColorMatrix::Fcc => ffi::AVColorSpace::AVCOL_SPC_FCC,
            ColorMatrix::Bt470Bg => ffi::AVColorSpace::AVCOL_SPC_BT470BG,
            ColorMatrix::Smpte170M => ffi::AVColorSpace::AVCOL_SPC_SMPTE170M,
            ColorMatrix::Smpte240M => ffi::AVColorSpace::AVCOL_SPC_SMPTE240M,
            ColorMatrix::Bt2020NonConstant => ffi::AVColorSpace::AVCOL_SPC_BT2020_NCL,
            ColorMatrix::Bt2020Constant => ffi::AVColorSpace::AVCOL_SPC_BT2020_CL,
            _ => ffi::AVColorSpace::AVCOL_SPC_UNSPECIFIED,
        };
    }
}

/// Preserve the caller's complete ISO-BMFF transform, including translation.
/// FFmpeg8's mov writer reads global codec-parameter DISPLAYMATRIX side data
/// and writes all nine int32 values verbatim into tkhd. Dimensions do not
/// rescale this metadata: Swift AVAssetWriterInput.transform retains the source
/// transform even when its coded output dimensions shrink.
///
/// # Safety
/// `stream` must be a live context-owned stream before its header is written.
unsafe fn describe_display_matrix(
    stream: *mut ffi::AVStream,
    matrix: Option<[i32; 9]>,
) -> Result<()> {
    let Some(matrix) = matrix else {
        return Ok(());
    };
    // SAFETY: the caller guarantees a live stream; the codec parameters own
    // the allocated side-data array and free it with the format context.
    let parameters = unsafe { (*stream).codecpar };
    let side_data = unsafe {
        ffi::av_packet_side_data_new(
            &mut (*parameters).coded_side_data,
            &mut (*parameters).nb_coded_side_data,
            ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            std::mem::size_of_val(&matrix),
            0,
        )
    };
    if side_data.is_null() {
        return Err(VideoError::BackendUnavailable(
            "could not allocate video display matrix".into(),
        ));
    }
    // SAFETY: av_packet_side_data_new allocated exactly36 writable bytes.
    // The documented representation is nine native-endian int32 values.
    // Copying ends the borrow before the caller's stack matrix expires.
    unsafe {
        std::ptr::copy_nonoverlapping(
            matrix.as_ptr().cast::<u8>(),
            (*side_data).data,
            std::mem::size_of_val(&matrix),
        );
    }
    Ok(())
}

fn set_option(options: *mut *mut ffi::AVDictionary, key: &str, value: &str) -> Result<()> {
    let key = CString::new(key).expect("a literal without NUL");
    let value = CString::new(value)
        .map_err(|_| VideoError::InvalidData("mux option value contains an interior NUL".into()))?;
    // SAFETY: `options` is a live out-pointer; both strings outlive the call,
    // which copies them.
    check_av("av_dict_set", unsafe {
        ffi::av_dict_set(options, key.as_ptr(), value.as_ptr(), 0)
    })
    .map(|_| ())
}

impl MuxSession for LibavMuxSession {
    fn add_video_track(&mut self, track: VideoMuxTrack) -> Result<MuxTrackId> {
        let nal = NalMuxConfiguration::new(&track.configuration)?;
        let (stream, id) = self.add_stream(track.time_base, track.duration)?;
        self.tracks[id.get() as usize].nal = nal;
        // SAFETY: a live stream of this context, header not yet written.
        unsafe {
            describe_codec(
                stream,
                &track.configuration.codec,
                &track.configuration.description,
            )?;
            let parameters = (*stream).codecpar;
            (*parameters).width = track.width as i32;
            (*parameters).height = track.height as i32;
            describe_color(stream, track.color);
            describe_display_matrix(stream, track.display_matrix)?;
        }
        Ok(id)
    }

    fn add_audio_track(&mut self, track: AudioMuxTrack) -> Result<MuxTrackId> {
        let time_base = TimeBase::new(1, i64::from(track.sample_rate))?;
        let declared = track
            .timing
            .audible_samples
            .and_then(|samples| i64::try_from(samples).ok())
            .map(|samples| MediaTime::new(samples, time_base));
        let (stream, id) = self.add_stream(time_base, declared)?;
        // SAFETY: a live stream of this context, header not yet written.
        unsafe {
            describe_codec(
                stream,
                &track.configuration.codec,
                &track.configuration.description,
            )?;
            let parameters = (*stream).codecpar;
            (*parameters).sample_rate = track.sample_rate as i32;
            ffi::av_channel_layout_default(&mut (*parameters).ch_layout, i32::from(track.channels));
            // The encoder delay. libavformat turns this into the edit list and
            // the `iTunSMPB` tag that make gapless playback work — the same
            // two things the hand-written muxer wrote by hand.
            (*parameters).initial_padding = track.timing.leading_samples as i32;
            (*parameters).trailing_padding = track.timing.trailing_samples as i32;
        }
        Ok(id)
    }

    fn add_passthrough_audio_track(
        &mut self,
        track: &crate::AudioPassthroughTrack,
    ) -> Result<MuxTrackId> {
        let (stream, id) = self.add_stream(track.time_base, Some(track.presentation_end))?;
        // SAFETY: both codec parameter sets are live, independently owned;
        // libav copies channel layout, sample format, extradata and padding.
        unsafe {
            check_av(
                "avcodec_parameters_copy",
                ffi::avcodec_parameters_copy((*stream).codecpar, track.parameters.as_ptr()),
            )?;
            // A source MOV/WAV tag need not be legal in MP4. Let the muxer
            // choose the destination tag for the unchanged codec ID.
            (*(*stream).codecpar).codec_tag = 0;
            (*stream).disposition = track.disposition;
            for (key, value) in &track.metadata {
                set_option(&mut (*stream).metadata, key, value)?;
            }
        }
        Ok(id)
    }

    fn write_passthrough_audio_packet(
        &mut self,
        track: MuxTrackId,
        mut packet: crate::AudioPassthroughPacket,
    ) -> Result<()> {
        self.ensure_header()?;
        let entry = self
            .tracks
            .get_mut(track.get() as usize)
            .ok_or_else(|| VideoError::InvalidData(format!("unknown mux track {}", track.get())))?;
        let rational = |base: TimeBase| -> Result<ffi::AVRational> {
            Ok(ffi::AVRational {
                num: i32::try_from(base.numerator).map_err(|_| {
                    VideoError::InvalidData("audio time base numerator overflow".into())
                })?,
                den: i32::try_from(base.denominator).map_err(|_| {
                    VideoError::InvalidData("audio time base denominator overflow".into())
                })?,
            })
        };
        if let Some(pts) = packet.packet.pts() {
            let end = MediaTime::new(
                pts.checked_add(packet.packet.duration())
                    .ok_or_else(|| VideoError::InvalidData("audio packet end overflow".into()))?,
                packet.time_base,
            );
            if entry
                .packet_end
                .is_none_or(|current| end.seconds() > current.seconds())
            {
                entry.packet_end = Some(end);
            }
        }
        // SAFETY: the packet remains uniquely owned until the muxer consumes
        // it. Rescaling leaves its payload and all codec side data intact.
        unsafe {
            ffi::av_packet_rescale_ts(
                packet.packet.as_mut_ptr(),
                rational(packet.time_base)?,
                rational(entry.time_base)?,
            );
            (*packet.packet.as_mut_ptr()).stream_index = entry.stream_index;
            (*packet.packet.as_mut_ptr()).pos = -1;
            check_av(
                "av_interleaved_write_frame",
                ffi::av_interleaved_write_frame(self.context, packet.packet.as_mut_ptr()),
            )?;
        }
        Ok(())
    }

    fn write_packet(&mut self, track: MuxTrackId, packet: EncodedPacket) -> Result<()> {
        self.ensure_header()?;
        let index = track.get() as usize;
        let entry = self
            .tracks
            .get(index)
            .ok_or_else(|| VideoError::InvalidData(format!("unknown mux track {}", track.get())))?;
        let stream_index = entry.stream_index;
        let target = entry.time_base;

        // FFmpeg derives input framing from extradata. Android's Annex-B CSD
        // must reach it with Annex-B pictures, or it converts a length-prefixed
        // packet a second time. hvcC/avcC instead retain their declared width.
        let data = match &entry.nal {
            Some(nal) => nal.packet_data(&packet)?,
            None => packet.data.clone(),
        };
        // A redundant parameter-set-only packet carries no displayed picture.
        if entry.nal.is_some() && data.is_empty() {
            return Ok(());
        }

        let mut out = ffmpeg_next::codec::packet::Packet::copy(&data);
        let rescale = |time: Option<MediaTime>| {
            time.map_or(ffi::AV_NOPTS_VALUE, |time| time.rescale(target).value)
        };
        // SAFETY: `out` owns a live packet allocated by `copy`.
        unsafe {
            let raw = out.as_mut_ptr();
            (*raw).stream_index = stream_index;
            (*raw).pts = rescale(packet.pts);
            (*raw).dts = rescale(packet.dts.or(packet.pts));
            (*raw).duration = packet
                .duration
                .map_or(0, |duration| duration.rescale(target).value);
            if packet.is_sync {
                (*raw).flags |= ffi::AV_PKT_FLAG_KEY;
            }
        }

        // The packets' own extent, per track, for tracks that declared none.
        //
        // In the PACKET's time base, not the stream's: the `mov` muxer picks
        // its own timescale for a video track and rewrites `time_base`, so
        // rescaling first would round the answer.
        if let Some(pts) = packet.pts {
            let base = pts.time_base;
            let end = MediaTime::new(
                pts.value
                    + packet
                        .duration
                        .map_or(0, |duration| duration.rescale(base).value),
                base,
            );
            let track = &mut self.tracks[index];
            if track
                .packet_end
                .is_none_or(|current| end.seconds() > current.seconds())
            {
                track.packet_end = Some(end);
            }
        }

        // Interleaved, not direct: this is what buffers across tracks so the
        // chunks land in playback order rather than in the order we happened
        // to encode them.
        // SAFETY: the context is live and the packet belongs to one of its
        // streams; the call takes ownership of the packet's contents.
        check_av("av_interleaved_write_frame", unsafe {
            ffi::av_interleaved_write_frame(self.context, out.as_mut_ptr())
        })
        .map(|_| ())
    }

    fn finalize(mut self: Box<Self>) -> Result<MuxOutput> {
        if self.finalized {
            return Err(VideoError::InvalidData(
                "mux session was already finalized".into(),
            ));
        }
        self.ensure_header()?;
        self.finalized = true;
        // SAFETY: the context is live and its header was written.
        check_av("av_write_trailer", unsafe {
            ffi::av_write_trailer(self.context)
        })?;

        // A path-backed session wrote the file and has no bytes to hand back;
        // an in-memory one has nothing else.
        let bytes = match self.destination {
            Destination::Path(_) => None,
            Destination::Memory => {
                let mut buffer: *mut u8 = std::ptr::null_mut();
                // SAFETY: the context's `pb` is the dynamic buffer opened in
                // `open_memory`; this closes it and hands over the allocation.
                let size = unsafe { ffi::avio_close_dyn_buf((*self.context).pb, &mut buffer) };
                // SAFETY: `pb` is now closed; clearing it stops `Drop` from
                // closing it a second time.
                unsafe { (*self.context).pb = std::ptr::null_mut() };
                let size = check_av("avio_close_dyn_buf", size)? as usize;
                // SAFETY: `buffer` holds `size` bytes, allocated by libavutil.
                let bytes = unsafe { std::slice::from_raw_parts(buffer, size).to_vec() };
                // SAFETY: the allocation is ours to release.
                unsafe { ffi::av_free(buffer.cast()) };
                Some(bytes)
            }
        };
        // The longest track wins, and each track answers with what its
        // caller declared before what its packets happen to span.
        let duration = self
            .tracks
            .iter()
            .filter_map(|track| track.declared.or(track.packet_end))
            .max_by(|left, right| {
                left.seconds()
                    .partial_cmp(&right.seconds())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(MediaTime::ZERO);
        Ok(MuxOutput {
            path: self.path().map(Path::to_path_buf),
            bytes,
            duration,
            track_count: self.tracks.len(),
        })
    }
}

impl Drop for LibavMuxSession {
    fn drop(&mut self) {
        if self.context.is_null() {
            return;
        }
        // SAFETY: the context is ours; closing the IO and freeing it is the
        // documented teardown. A session dropped without `finalize` leaves a
        // file with no trailer, which is what an abandoned export should be.
        //
        // The two sinks close differently, and using the wrong one ABORTS the
        // process rather than failing: a dynamic buffer owns an allocation
        // that only `avio_close_dyn_buf` hands back, so `avio_closep` on one
        // is not a leak, it is a crash. A caller that builds a session and
        // drops it without finalizing — asking a mux job for its track
        // descriptors, say — takes exactly this path.
        unsafe {
            if !(*self.context).pb.is_null() {
                match self.destination {
                    Destination::Path(_) => {
                        ffi::avio_closep(&mut (*self.context).pb);
                    }
                    Destination::Memory => {
                        let mut buffer: *mut u8 = std::ptr::null_mut();
                        ffi::avio_close_dyn_buf((*self.context).pb, &mut buffer);
                        (*self.context).pb = std::ptr::null_mut();
                        if !buffer.is_null() {
                            ffi::av_free(buffer.cast());
                        }
                    }
                }
            }
            ffi::avformat_free_context(self.context);
        }
        self.context = std::ptr::null_mut();
    }
}
