use crate::{MediaTime, Result, TimeBase, VideoError};

/// Not `Copy`: `Other` carries libavformat's own name for a container none of
/// the three named variants covers, and libavformat opens far more than three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerFormat {
    Mp4,
    Matroska,
    /// A bare MPEG audio elementary stream (Layer III frames, no container).
    Mp3,
    /// The input format's short name, as libavformat spells it — `"wav"`,
    /// `"flac"`, `"ogg"`, `"image2"`. A consumer that only knows the three
    /// above matches this with a `_` arm.
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Video,
    Audio,
    Subtitle,
    Data,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProResProfile {
    Proxy,
    Lt,
    Standard,
    Hq,
    P4444,
    P4444Xq,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
    ProRes(ProResProfile),
    Aac,
    Mp3,
    Vp8,
    Vp9,
    Av1,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    LowComplexity,
    HighEfficiency,
    HighEfficiencyV2,
    Other(u8),
}

impl Codec {
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::H264 | Self::Hevc | Self::ProRes(_) | Self::Vp8 | Self::Vp9 | Self::Av1
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorRange {
    Limited,
    Full,
    #[default]
    Unspecified,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPrimaries {
    Bt709,
    Bt470M,
    Bt470Bg,
    Smpte170M,
    Smpte240M,
    Film,
    Bt2020,
    DisplayP3,
    #[default]
    Unspecified,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferCharacteristic {
    Bt709,
    Smpte170M,
    Linear,
    Srgb,
    Bt2020TenBit,
    Bt2020TwelveBit,
    Pq,
    Hlg,
    #[default]
    Unspecified,
    Other(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMatrix {
    Identity,
    Bt709,
    Fcc,
    Bt470Bg,
    Smpte170M,
    Smpte240M,
    Bt2020NonConstant,
    Bt2020Constant,
    #[default]
    Unspecified,
    Other(u16),
}

impl ColorMatrix {
    /// Stable RGB conversion fallback, shared with the existing CPU YUV path.
    /// Source metadata remains Unspecified; image dimensions do not invent tags.
    pub(crate) fn for_rgb(self) -> Self {
        if self == Self::Unspecified {
            Self::Smpte170M
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorInfo {
    pub range: ColorRange,
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristic,
    pub matrix: ColorMatrix,
    pub bits_per_channel: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AudioTiming {
    /// Encoded samples before presentation time zero (encoder priming).
    pub leading_samples: u32,
    /// Encoded samples after the audible program ends (padding).
    pub trailing_samples: u32,
    /// Audible program length after applying container timing metadata.
    pub audible_samples: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packetization {
    /// One complete compressed picture per packet (for example ProRes).
    Elementary,
    /// NAL units prefixed with an N-byte big-endian size (avcC/hvcC form).
    LengthPrefixed { nal_length_size: u8 },
    /// NAL units delimited by three- or four-byte start codes.
    AnnexB,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalUnit {
    pub kind: u8,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfiguration {
    pub codec: Codec,
    /// Container codec-private bytes (AVCDecoderConfigurationRecord,
    /// HEVCDecoderConfigurationRecord, or another codec-defined payload).
    pub description: Vec<u8>,
    pub packetization: Packetization,
    pub parameter_sets: Vec<NalUnit>,
}

impl DecoderConfiguration {
    pub fn from_codec_private(codec: Codec, description: Vec<u8>) -> Result<Self> {
        let (packetization, parameter_sets) = match codec {
            Codec::H264 if description.is_empty() => (Packetization::AnnexB, Vec::new()),
            Codec::Hevc if description.is_empty() => (Packetization::AnnexB, Vec::new()),
            Codec::H264 => parse_avcc(&description)?,
            Codec::Hevc => parse_hvcc(&description)?,
            _ => (Packetization::Elementary, Vec::new()),
        };
        Ok(Self {
            codec,
            description,
            packetization,
            parameter_sets,
        })
    }

    pub fn aac_profile(&self) -> Result<Option<AacProfile>> {
        if self.codec != Codec::Aac {
            return Ok(None);
        }
        let object_type = read_audio_object_type(&self.description)?;
        let profile = match object_type {
            2 => match sbr_sync_extension_offset(&self.description) {
                Some(offset) if has_ps_sync_extension(&self.description, offset) => {
                    AacProfile::HighEfficiencyV2
                }
                Some(_) => AacProfile::HighEfficiency,
                None => AacProfile::LowComplexity,
            },
            5 => AacProfile::HighEfficiency,
            29 => AacProfile::HighEfficiencyV2,
            other => AacProfile::Other(other),
        };
        Ok(Some(profile))
    }

    /// The channel count decoded PCM actually carries, from the
    /// AudioSpecificConfig. Container sample entries routinely lie here —
    /// QuickTime-family muxers write `channelcount = 2` for mono AAC — so
    /// the ASC wins. HE-AACv2 decodes its mono core to parametric stereo.
    /// `None` for non-AAC streams or when the ASC defers to an in-band PCE
    /// (channel configuration 0).
    pub fn aac_output_channels(&self) -> Result<Option<u16>> {
        let profile = match self.aac_profile()? {
            Some(profile) => profile,
            None => return Ok(None),
        };
        if profile == AacProfile::HighEfficiencyV2 {
            return Ok(Some(2));
        }
        let mut bits = BitReader::new(&self.description);
        if bits.read(5)? == 31 {
            let _extended_object_type = bits.read(6)?;
        }
        let _core_rate = read_sampling_frequency(&mut bits)?;
        Ok(match bits.read(4)? {
            configuration @ 1..=6 => Some(configuration as u16),
            7 => Some(8),
            _ => None,
        })
    }

    /// The SBR output sample rate for an HE-AAC stream — the rate the decoded
    /// PCM actually plays at, as opposed to the core AAC rate the
    /// AudioSpecificConfig leads with. `None` for non-HE streams.
    ///
    /// Explicit signaling (object type 5/29) carries the extension frequency
    /// right after the channel configuration; implicit signaling (object
    /// type 2 plus a trailing SBR sync extension) carries it after the
    /// `sbrPresentFlag`.
    pub fn aac_effective_sample_rate(&self) -> Result<Option<u32>> {
        match self.aac_profile()? {
            Some(AacProfile::HighEfficiency | AacProfile::HighEfficiencyV2) => {}
            _ => return Ok(None),
        }
        let data = &self.description;
        let mut bits = BitReader::new(data);
        let mut object_type = bits.read(5)?;
        if object_type == 31 {
            object_type = 32u32.saturating_add(bits.read(6)?);
        }
        let _core_rate = read_sampling_frequency(&mut bits)?;
        let _channel_configuration = bits.read(4)?;
        if object_type == 5 || object_type == 29 {
            return read_sampling_frequency(&mut bits).map(Some);
        }
        let offset = sbr_sync_extension_offset(data).ok_or_else(|| {
            VideoError::InvalidData("HE-AAC profile without an SBR sync extension".into())
        })?;
        let mut bits = BitReader::at(data, offset + 17);
        read_sampling_frequency(&mut bits).map(Some)
    }
}

/// ISO/IEC 14496-3 sampling frequency: a 4-bit table index, or the escape
/// index 15 followed by a 24-bit explicit value.
fn read_sampling_frequency(bits: &mut BitReader<'_>) -> Result<u32> {
    const AAC_SAMPLE_RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    let index = bits.read(4)? as usize;
    if index == 15 {
        return bits.read(24);
    }
    AAC_SAMPLE_RATES.get(index).copied().ok_or_else(|| {
        VideoError::InvalidData(format!("reserved AAC sampling frequency index {index}"))
    })
}

fn read_audio_object_type(data: &[u8]) -> Result<u8> {
    let mut bits = BitReader::new(data);
    let mut object_type = bits.read(5)? as u8;
    if object_type == 31 {
        object_type = 32u8.saturating_add(bits.read(6)? as u8);
    }
    Ok(object_type)
}

/// Bit offset of the `0x2b7` SBR sync extension (with `sbrPresentFlag` set),
/// or `None` when the AudioSpecificConfig carries no implicit SBR signal.
fn sbr_sync_extension_offset(data: &[u8]) -> Option<usize> {
    let bit_length = data.len().saturating_mul(8);
    (0..bit_length.saturating_sub(16)).find(|&offset| {
        read_bits_at(data, offset, 11) == Some(0x2b7)
            && read_bits_at(data, offset + 11, 5) == Some(5)
            && read_bits_at(data, offset + 16, 1) == Some(1)
    })
}

/// Whether the `0x548` parametric-stereo sync extension (with
/// `psPresentFlag` set) follows the SBR extension frequency — the implicit
/// HE-AACv2 signal: a mono AAC core that decodes to stereo.
fn has_ps_sync_extension(data: &[u8], sbr_offset: usize) -> bool {
    // sbr_offset + 11 (sync) + 5 (extension object) + 1 (sbrPresent) lands on
    // the extension frequency: a 4-bit index, or 4 + 24 escape bits.
    let frequency_start = sbr_offset + 17;
    let after_frequency = match read_bits_at(data, frequency_start, 4) {
        Some(15) => frequency_start + 28,
        Some(_) => frequency_start + 4,
        None => return false,
    };
    read_bits_at(data, after_frequency, 11) == Some(0x548)
        && read_bits_at(data, after_frequency + 11, 1) == Some(1)
}

fn read_bits_at(data: &[u8], offset: usize, count: usize) -> Option<u32> {
    let end = offset.checked_add(count)?;
    if end > data.len().checked_mul(8)? || count > 32 {
        return None;
    }
    let mut value = 0u32;
    for bit in offset..end {
        value = (value << 1) | u32::from((data[bit / 8] >> (7 - bit % 8)) & 1);
    }
    Some(value)
}

struct BitReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn at(data: &'a [u8], offset: usize) -> Self {
        Self { data, offset }
    }

    fn read(&mut self, count: usize) -> Result<u32> {
        let value = read_bits_at(self.data, self.offset, count)
            .ok_or_else(|| VideoError::InvalidData("truncated AAC AudioSpecificConfig".into()))?;
        self.offset += count;
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamInfo {
    pub index: u32,
    pub kind: MediaKind,
    pub codec: Codec,
    pub time_base: TimeBase,
    pub start_time: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub frame_rate: Option<(u32, u32)>,
    /// ISO-BMFF track display matrix in 16.16/2.30 fixed-point form.
    /// `None` means the container did not provide a transform.
    pub display_matrix: Option<[i32; 9]>,
    pub bit_rate: Option<u64>,
    pub language: Option<String>,
    pub color: ColorInfo,
    pub audio_timing: Option<AudioTiming>,
    pub decoder_configuration: DecoderConfiguration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaInfo {
    pub container: ContainerFormat,
    pub duration: Option<MediaTime>,
    pub streams: Vec<StreamInfo>,
}

impl MediaInfo {
    pub fn video_streams(&self) -> impl Iterator<Item = &StreamInfo> {
        self.streams
            .iter()
            .filter(|stream| stream.kind == MediaKind::Video)
    }

    pub fn audio_streams(&self) -> impl Iterator<Item = &StreamInfo> {
        self.streams
            .iter()
            .filter(|stream| stream.kind == MediaKind::Audio)
    }

    pub fn stream(&self, index: u32) -> Option<&StreamInfo> {
        self.streams.iter().find(|stream| stream.index == index)
    }
}

fn parse_avcc(data: &[u8]) -> Result<(Packetization, Vec<NalUnit>)> {
    if data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1]) {
        return Ok((Packetization::AnnexB, Vec::new()));
    }
    if data.len() < 7 || data[0] != 1 {
        return Err(VideoError::InvalidData(
            "invalid AVC decoder configuration record".into(),
        ));
    }
    let nal_length_size = (data[4] & 0x03) + 1;
    let mut cursor = 6usize;
    let mut parameter_sets = Vec::new();
    let sps_count = usize::from(data[5] & 0x1f);
    for _ in 0..sps_count {
        parameter_sets.push(read_parameter_set(data, &mut cursor, 7)?);
    }
    let pps_count = *data.get(cursor).ok_or_else(|| {
        VideoError::InvalidData("AVC configuration is missing its PPS count".into())
    })?;
    cursor += 1;
    for _ in 0..pps_count {
        parameter_sets.push(read_parameter_set(data, &mut cursor, 8)?);
    }
    Ok((
        Packetization::LengthPrefixed { nal_length_size },
        parameter_sets,
    ))
}

fn parse_hvcc(data: &[u8]) -> Result<(Packetization, Vec<NalUnit>)> {
    if data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1]) {
        return Ok((Packetization::AnnexB, Vec::new()));
    }
    if data.len() < 23 || data[0] != 1 {
        return Err(VideoError::InvalidData(
            "invalid HEVC decoder configuration record".into(),
        ));
    }
    let nal_length_size = (data[21] & 0x03) + 1;
    let array_count = usize::from(data[22]);
    let mut cursor = 23usize;
    let mut parameter_sets = Vec::new();
    for _ in 0..array_count {
        let header = *data
            .get(cursor)
            .ok_or_else(|| VideoError::InvalidData("truncated HEVC parameter-set array".into()))?;
        cursor += 1;
        let kind = header & 0x3f;
        let count = read_u16(data, &mut cursor)?;
        for _ in 0..count {
            parameter_sets.push(read_parameter_set(data, &mut cursor, kind)?);
        }
    }
    Ok((
        Packetization::LengthPrefixed { nal_length_size },
        parameter_sets,
    ))
}

fn read_parameter_set(data: &[u8], cursor: &mut usize, kind: u8) -> Result<NalUnit> {
    let length = usize::from(read_u16(data, cursor)?);
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| VideoError::InvalidData("parameter-set length overflow".into()))?;
    let bytes = data
        .get(*cursor..end)
        .ok_or_else(|| VideoError::InvalidData("truncated codec parameter set".into()))?
        .to_vec();
    *cursor = end;
    Ok(NalUnit { kind, bytes })
}

fn read_u16(data: &[u8], cursor: &mut usize) -> Result<u16> {
    let bytes: [u8; 2] = data
        .get(*cursor..*cursor + 2)
        .ok_or_else(|| VideoError::InvalidData("truncated 16-bit field".into()))?
        .try_into()
        .expect("slice length was checked");
    *cursor += 2;
    Ok(u16::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_avcc_parameter_sets_and_nal_width() {
        let avcc = vec![
            1, 100, 0, 31, 0xff, 0xe1, 0, 3, 0x67, 0x64, 0x00, 1, 0, 2, 0x68, 0xee,
        ];
        let config = DecoderConfiguration::from_codec_private(Codec::H264, avcc).unwrap();
        assert_eq!(
            config.packetization,
            Packetization::LengthPrefixed { nal_length_size: 4 }
        );
        assert_eq!(config.parameter_sets.len(), 2);
        assert_eq!(config.parameter_sets[0].kind, 7);
        assert_eq!(config.parameter_sets[1].kind, 8);
    }

    #[test]
    fn rejects_truncated_hvcc() {
        assert!(DecoderConfiguration::from_codec_private(Codec::Hevc, vec![1, 2, 3]).is_err());
    }

    #[test]
    fn distinguishes_aac_lc_and_he_profiles() {
        let lc = DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x12, 0x10]).unwrap();
        assert_eq!(lc.aac_profile().unwrap(), Some(AacProfile::LowComplexity));

        // Explicit object type 5 (HE-AAC/SBR).
        let he = DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x2a, 0x10]).unwrap();
        assert_eq!(he.aac_profile().unwrap(), Some(AacProfile::HighEfficiency));
    }

    #[test]
    fn reads_he_extension_rate_from_explicit_signaling() {
        // Real-world ASC: object 5, core 22050 Hz, stereo, extension 44100 Hz.
        let he = DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x2b, 0x92, 0x08, 0x00])
            .unwrap();
        assert_eq!(he.aac_profile().unwrap(), Some(AacProfile::HighEfficiency));
        assert_eq!(he.aac_effective_sample_rate().unwrap(), Some(44_100));
    }

    #[test]
    fn reads_he_extension_rate_from_implicit_sync_extension() {
        // LC 22050 Hz stereo ASC followed by the 0x2b7 sync extension with
        // sbrPresent set and extension frequency index 4 (44100 Hz).
        let he = DecoderConfiguration::from_codec_private(
            Codec::Aac,
            vec![0x13, 0x90, 0x56, 0xE5, 0xA0],
        )
        .unwrap();
        assert_eq!(he.aac_profile().unwrap(), Some(AacProfile::HighEfficiency));
        assert_eq!(he.aac_effective_sample_rate().unwrap(), Some(44_100));
    }

    #[test]
    fn lc_streams_have_no_extension_rate() {
        let lc = DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x12, 0x10]).unwrap();
        assert_eq!(lc.aac_effective_sample_rate().unwrap(), None);
    }

    #[test]
    fn asc_channel_configuration_beats_container_claims() {
        let mono = DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x13, 0x88]).unwrap();
        assert_eq!(mono.aac_output_channels().unwrap(), Some(1));
        let stereo =
            DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x13, 0x90]).unwrap();
        assert_eq!(stereo.aac_output_channels().unwrap(), Some(2));
        // HE-AACv2: mono core, parametric stereo output.
        let v2 = DecoderConfiguration::from_codec_private(
            Codec::Aac,
            vec![0x13, 0x88, 0x56, 0xE5, 0xA5, 0x48, 0x80],
        )
        .unwrap();
        assert_eq!(v2.aac_output_channels().unwrap(), Some(2));
    }

    #[test]
    fn detects_implicit_parametric_stereo_as_he_v2() {
        // afconvert `aacp` ASC: mono LC core at 22050 Hz, implicit SBR to
        // 44100 Hz, then the 0x548 PS sync extension with psPresent set.
        let v2 = DecoderConfiguration::from_codec_private(
            Codec::Aac,
            vec![0x13, 0x88, 0x56, 0xE5, 0xA5, 0x48, 0x80],
        )
        .unwrap();
        assert_eq!(
            v2.aac_profile().unwrap(),
            Some(AacProfile::HighEfficiencyV2)
        );
        assert_eq!(v2.aac_effective_sample_rate().unwrap(), Some(44_100));
    }
}
