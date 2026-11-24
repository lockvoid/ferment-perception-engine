//! Android API33 wire values and pure validation/packing shared by the real
//! MediaCodec path and host tests. No codec capability is inferred here.
use crate::{
    Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, EncoderSettings, Result,
    TransferCharacteristic, VideoError, Yuv420Format, Yuv420Frame,
};

pub(crate) const P010: i32 = 54;
pub(crate) const HEVC_MAIN10: i32 = 2;
pub(crate) const HEVC_MAIN10_HDR10: i32 = 4096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct InputPolicy {
    pub profile: Option<i32>,
    pub p010: bool,
    pub hdr: bool,
}
impl InputPolicy {
    pub fn for_settings(settings: &EncoderSettings) -> Result<Self> {
        let color = settings.color;
        let hdr = color.is_some_and(|c| is_hdr(c.transfer));
        // Preserve the existing codec-selected SDR profile. Requesting H.264
        // High here would newly exclude Android's Baseline/Main software
        // encoders; strict profile negotiation belongs to the HDR path.
        let profile =
            if settings.codec == Codec::Hevc && settings.profile.as_deref() == Some("main10") {
                Some(
                    if color.is_some_and(|c| c.transfer == TransferCharacteristic::Pq)
                        && settings.hdr_static_metadata.is_some()
                    {
                        HEVC_MAIN10_HDR10
                    } else {
                        HEVC_MAIN10
                    },
                )
            } else {
                None
            };
        let p010 = settings.codec == Codec::Hevc
            && matches!(profile, Some(HEVC_MAIN10 | HEVC_MAIN10_HDR10));
        if hdr
            || color.is_some_and(|c| c.bits_per_channel.is_some_and(|b| b > 8))
            || settings.insert_hdr_metadata
        {
            let c = color.ok_or_else(|| {
                VideoError::InvalidData("HDR encoding requires color information".into())
            })?;
            if !p010
                || c.bits_per_channel != Some(10)
                || c.primaries != ColorPrimaries::Bt2020
                || c.matrix != ColorMatrix::Bt2020NonConstant
                || !is_hdr(c.transfer)
                || !matches!(c.range, ColorRange::Full | ColorRange::Limited)
            {
                return Err(VideoError::UnsupportedCodec("HDR encoding requires HEVC Main10, actual10-bit BT.2020 HLG/PQ and an explicit range".into()));
            }
            // Swift VT Auto may be a no-op. PQ without source mastering data
            // stays Main10/PQ without an invented HDR10 mastering descriptor.
        }
        if p010 && !hdr {
            return Err(VideoError::UnsupportedCodec(
                "this Main10 path requires a declared HDR transfer".into(),
            ));
        }
        Ok(Self { profile, p010, hdr })
    }

    pub fn validate_output(
        &self,
        settings: &EncoderSettings,
        profile: Option<i32>,
        standard: Option<i32>,
        transfer: Option<i32>,
        range: Option<i32>,
    ) -> Result<()> {
        if self.p010 {
            let actual = profile.ok_or_else(|| {
                VideoError::UnsupportedCodec("encoder did not confirm its Main10 profile".into())
            })?;
            if !matches!(actual, HEVC_MAIN10 | HEVC_MAIN10_HDR10 | 8192) {
                return Err(VideoError::UnsupportedCodec(format!(
                    "encoder changed Main10 to profile {actual}"
                )));
            }
        } else if let (Some(requested), Some(actual)) = (self.profile, profile)
            && actual != requested
        {
            return Err(VideoError::UnsupportedCodec(format!(
                "encoder changed profile {requested} to {actual}"
            )));
        }
        if self.hdr {
            let c = settings.color.expect("validated HDR color");
            let expected_transfer = if c.transfer == TransferCharacteristic::Hlg {
                7
            } else {
                6
            };
            let expected_range = if c.range == ColorRange::Full { 1 } else { 2 };
            if (standard, transfer, range)
                != (Some(6), Some(expected_transfer), Some(expected_range))
            {
                return Err(VideoError::UnsupportedCodec(format!(
                    "encoder did not preserve HDR color: standard={standard:?} transfer={transfer:?} range={range:?}"
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn is_hdr(transfer: TransferCharacteristic) -> bool {
    matches!(
        transfer,
        TransferCharacteristic::Hlg | TransferCharacteristic::Pq
    )
}

pub(crate) fn sdr_request(source: ColorInfo, target: Option<ColorInfo>) -> Result<bool> {
    let Some(target) = target.filter(|_| is_hdr(source.transfer)) else {
        return Ok(false);
    };
    if is_hdr(target.transfer) {
        if source.primaries != target.primaries
            || source.transfer != target.transfer
            || source.matrix != target.matrix
        {
            return Err(VideoError::UnsupportedCodec(
                "HDR decoder cannot relabel another HDR transfer or gamut".into(),
            ));
        }
        return Ok(false);
    }
    if (
        target.primaries,
        target.matrix,
        target.transfer,
        target.range,
        target.bits_per_channel,
    ) != (
        ColorPrimaries::Bt709,
        ColorMatrix::Bt709,
        TransferCharacteristic::Bt709,
        ColorRange::Limited,
        Some(8),
    ) {
        return Err(VideoError::UnsupportedCodec(
            "Android tone mapping supports an explicit BT.709 limited SDR derivative".into(),
        ));
    }
    Ok(true)
}

/// Preserve declared source depth in the decoder's no-Surface fallback.
/// An explicit SDR conversion uses the decoder's ordinary 8-bit output; an
/// ImageReader already declares its own format and needs no ByteBuffer request.
pub(crate) fn decoder_buffer_color_format(
    source: ColorInfo,
    target: Option<ColorInfo>,
    has_surface: bool,
) -> Result<Option<i32>> {
    let tone_map = sdr_request(source, target)?;
    Ok(
        (!has_surface && !tone_map && source.bits_per_channel.is_some_and(|bits| bits > 8))
            .then_some(P010),
    )
}

/// KEY_STRIDE is BYTES, including for P010. Fill padding with legal neutral
/// black and copy only active samples. Never interpret byte pitch as pixels.
pub(crate) fn write_p010(
    frame: &Yuv420Frame,
    buffer: &mut [u8],
    stride: usize,
    slice_height: usize,
) -> Result<usize> {
    frame.validate_owned_layout()?;
    if !matches!(
        frame.format,
        Yuv420Format::P010Video | Yuv420Format::P010Full
    ) {
        return Err(VideoError::InvalidData(
            "P010 packing requires ten-bit planes".into(),
        ));
    }
    let row_bytes = (frame.width as usize)
        .checked_mul(2)
        .ok_or_else(|| VideoError::InvalidData("P010 row overflow".into()))?;
    let height = frame.height as usize;
    if stride < row_bytes
        || !stride.is_multiple_of(4)
        || slice_height < height
        || !slice_height.is_multiple_of(2)
    {
        return Err(VideoError::UnsupportedCodec(format!(
            "invalid P010 byte layout: stride={stride} slice-height={slice_height} size={}x{}",
            frame.width, frame.height
        )));
    }
    let luma = stride
        .checked_mul(slice_height)
        .ok_or_else(|| VideoError::InvalidData("P010 extent overflow".into()))?;
    let needed = luma
        .checked_add(luma / 2)
        .ok_or_else(|| VideoError::InvalidData("P010 extent overflow".into()))?;
    let destination = buffer.get_mut(..needed).ok_or_else(|| {
        VideoError::InvalidData(format!("short P010 codec input; need {needed} bytes"))
    })?;
    let black = if frame.format == Yuv420Format::P010Video {
        64u16 << 6
    } else {
        0
    };
    for word in destination[..luma].chunks_exact_mut(2) {
        word.copy_from_slice(&black.to_le_bytes());
    }
    for word in destination[luma..].chunks_exact_mut(2) {
        word.copy_from_slice(&(512u16 << 6).to_le_bytes());
    }
    for (source, offset, rows) in [(&frame.y, 0, height), (&frame.uv, luma, height / 2)] {
        for row in 0..rows {
            destination[offset + row * stride..offset + row * stride + row_bytes]
                .copy_from_slice(&source[row * row_bytes..(row + 1) * row_bytes]);
        }
    }
    Ok(needed)
}

pub(crate) fn read_p010(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    slice_height: usize,
    range: ColorRange,
) -> Result<Yuv420Frame> {
    let row_bytes = (width as usize)
        .checked_mul(2)
        .ok_or_else(|| VideoError::InvalidData("P010 row overflow".into()))?;
    if width == 0
        || height == 0
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
        || stride < row_bytes
        || !stride.is_multiple_of(4)
        || slice_height < height as usize
        || !slice_height.is_multiple_of(2)
    {
        return Err(VideoError::InvalidData(
            "invalid decoded P010 layout".into(),
        ));
    }
    let luma = stride
        .checked_mul(slice_height)
        .ok_or_else(|| VideoError::InvalidData("P010 extent overflow".into()))?;
    let last = stride
        .checked_mul(height as usize / 2 - 1)
        .and_then(|offset| luma.checked_add(offset))
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or_else(|| VideoError::InvalidData("P010 extent overflow".into()))?;
    if data.len() < last {
        return Err(VideoError::InvalidData("short decoded P010 buffer".into()));
    }
    let mut y = Vec::with_capacity(row_bytes * height as usize);
    let mut uv = Vec::with_capacity(row_bytes * height as usize / 2);
    for (out, offset, rows) in [
        (&mut y, 0, height as usize),
        (&mut uv, luma, height as usize / 2),
    ] {
        for row in 0..rows {
            out.extend_from_slice(&data[offset + row * stride..offset + row * stride + row_bytes]);
        }
    }
    let format = match range {
        ColorRange::Full => Yuv420Format::P010Full,
        ColorRange::Limited => Yuv420Format::P010Video,
        _ => {
            return Err(VideoError::InvalidData(
                "decoded P010 range is not declared".into(),
            ));
        }
    };
    Yuv420Frame::new(width, height, format, y, uv)
}

/// Preserve legitimate 8-bit HDR source YUV before Main10 promotion or tone
/// mapping. Unknown/flexible ByteBuffer layouts are refused instead of guessed.
pub(crate) fn read_yuv420_8(
    data: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    slice_height: usize,
    pixel_format: i32,
    range: ColorRange,
) -> Result<Yuv420Frame> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
        || stride < width as usize
        || !stride.is_multiple_of(2)
        || slice_height < height as usize
        || !slice_height.is_multiple_of(2)
        || !matches!(pixel_format, 19 | 21 | 39)
    {
        return Err(VideoError::UnsupportedCodec(
            "unrecognized source-depth 8-bit YUV layout".into(),
        ));
    }
    let luma = stride
        .checked_mul(slice_height)
        .ok_or_else(|| VideoError::InvalidData("YUV extent overflow".into()))?;
    let length = luma
        .checked_add(luma / 2)
        .ok_or_else(|| VideoError::InvalidData("YUV extent overflow".into()))?;
    if data.len() < length {
        return Err(VideoError::InvalidData(
            "short source-depth YUV buffer".into(),
        ));
    }
    let mut y = Vec::with_capacity(width as usize * height as usize);
    for row in 0..height as usize {
        y.extend_from_slice(&data[row * stride..row * stride + width as usize]);
    }
    let mut uv = Vec::with_capacity(y.len() / 2);
    for row in 0..height as usize / 2 {
        for column in 0..width as usize / 2 {
            let (u, v) = if pixel_format == 19 {
                let at = row * (stride / 2) + column;
                (luma + at, luma + luma / 4 + at)
            } else {
                let at = luma + row * stride + column * 2;
                (at, at + 1)
            };
            uv.extend_from_slice(&[data[u], data[v]]);
        }
    }
    let format = match range {
        ColorRange::Full => Yuv420Format::Nv12Full,
        ColorRange::Limited => Yuv420Format::Nv12Video,
        _ => {
            return Err(VideoError::InvalidData(
                "source YUV range is not declared".into(),
            ));
        }
    };
    Yuv420Frame::new(width, height, format, y, uv)
}

/// Validate actual HEVC SPS bit depth, not merely the codec's requested or
/// returned profile key (Main10 can legally carry an8-bit picture). Annex-B
/// codec-private data is what MediaCodec returns for HEVC.
pub(crate) fn require_main10_sps(description: &[u8]) -> Result<()> {
    let mut starts = Vec::new();
    let mut at = 0;
    while at + 3 <= description.len() {
        let prefix = if description[at..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if description[at..].starts_with(&[0, 0, 1]) {
            3
        } else {
            at += 1;
            continue;
        };
        starts.push((at, at + prefix));
        at += prefix;
    }
    let mut found = false;
    for (index, (_, start)) in starts.iter().copied().enumerate() {
        let end = starts
            .get(index + 1)
            .map_or(description.len(), |(at, _)| *at);
        let nal = &description[start..end];
        if nal.len() < 2 || (nal[0] >> 1) & 63 != 33 {
            continue;
        }
        found = true;
        let mut rbsp = Vec::with_capacity(nal.len() - 2);
        let mut zeros = 0;
        for &byte in &nal[2..] {
            if zeros >= 2 && byte == 3 {
                zeros = 0;
                continue;
            }
            rbsp.push(byte);
            zeros = if byte == 0 { zeros + 1 } else { 0 };
        }
        let mut bits = SpsBits {
            bytes: &rbsp,
            bit: 0,
        };
        bits.read(4)?;
        let sublayers = bits.read(3)? as usize;
        if sublayers > 6 {
            return Err(VideoError::InvalidData(
                "invalid HEVC sub-layer count".into(),
            ));
        }
        bits.read(1)?;
        bits.read(3)?; // profile_space, tier_flag
        let profile = bits.read(5)?;
        let compatibility = bits.read(32)?;
        bits.skip(48 + 8)?; // constraint flags and general_level_idc
        let mut sub = Vec::with_capacity(sublayers);
        for _ in 0..sublayers {
            sub.push((bits.read(1)? != 0, bits.read(1)? != 0));
        }
        if sublayers > 0 {
            bits.skip((8 - sublayers) * 2)?;
        }
        for (profile, level) in sub {
            if profile {
                bits.skip(88)?;
            }
            if level {
                bits.skip(8)?;
            }
        }
        bits.ue()?; // sps_seq_parameter_set_id
        let chroma = bits.ue()?;
        if chroma == 3 {
            bits.read(1)?;
        }
        let width = bits.ue()?;
        let height = bits.ue()?;
        if bits.read(1)? != 0 {
            for _ in 0..4 {
                bits.ue()?;
            }
        }
        let luma_depth = bits.ue()?;
        let chroma_depth = bits.ue()?;
        if (profile != 2 && compatibility & (1 << 29) == 0)
            || chroma != 1
            || luma_depth != 2
            || chroma_depth != 2
            || width == 0
            || height == 0
        {
            return Err(VideoError::UnsupportedCodec(format!(
                "encoded HEVC SPS is not actual Main10 4:2:0: profile={profile} chroma={chroma} bit-depth-minus8={luma_depth}/{chroma_depth}"
            )));
        }
    }
    if !found {
        return Err(VideoError::InvalidData(
            "HEVC encoder supplied no SPS for bit-depth verification".into(),
        ));
    }
    Ok(())
}
struct SpsBits<'a> {
    bytes: &'a [u8],
    bit: usize,
}
impl SpsBits<'_> {
    fn read(&mut self, count: usize) -> Result<u32> {
        if count > 32
            || self
                .bit
                .checked_add(count)
                .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err(VideoError::InvalidData("truncated HEVC SPS".into()));
        }
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u32::from((self.bytes[self.bit / 8] >> (7 - self.bit % 8)) & 1);
            self.bit += 1;
        }
        Ok(value)
    }
    fn skip(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            self.read(1)?;
        }
        Ok(())
    }
    fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while self.read(1)? == 0 {
            zeros += 1;
            if zeros >= 32 {
                return Err(VideoError::InvalidData(
                    "HEVC SPS exp-Golomb overflow".into(),
                ));
            }
        }
        Ok(((1u32 << zeros) - 1) + self.read(zeros)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CpuFrame, CpuPixelFormat, FrameBuffer, HdrStaticMetadata, MediaTime, TimeBase, VideoFrame,
    };

    fn words(codes: &[u16]) -> Vec<u8> {
        codes
            .iter()
            .flat_map(|value| (value << 6).to_le_bytes())
            .collect()
    }
    fn codes(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) >> 6)
            .collect()
    }
    fn color(transfer: TransferCharacteristic) -> ColorInfo {
        ColorInfo {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt2020,
            transfer,
            matrix: ColorMatrix::Bt2020NonConstant,
            bits_per_channel: Some(10),
        }
    }
    fn sdr() -> ColorInfo {
        ColorInfo {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            matrix: ColorMatrix::Bt709,
            bits_per_channel: Some(8),
        }
    }
    fn settings(transfer: TransferCharacteristic) -> EncoderSettings {
        let mut settings = EncoderSettings::new(Codec::Hevc, 4, 4, TimeBase::new_unchecked(1, 30));
        settings.profile = Some("main10".into());
        settings.color = Some(color(transfer));
        settings.insert_hdr_metadata = true;
        settings
    }
    fn frame(y: Yuv420Frame, transfer: TransferCharacteristic) -> VideoFrame {
        VideoFrame {
            width: y.width,
            height: y.height,
            pts: MediaTime::new(-17, TimeBase::new_unchecked(1, 90_000)),
            duration: Some(MediaTime::new(3003, TimeBase::new_unchecked(1, 90_000))),
            color: color(transfer),
            buffer: FrameBuffer::Yuv420(y),
        }
    }

    #[test]
    fn p010_packing_uses_byte_pitch_and_preserves_low_bits_and_padding() {
        let source = Yuv420Frame::new(
            4,
            2,
            Yuv420Format::P010Video,
            words(&[64, 65, 66, 67, 937, 938, 939, 940]),
            words(&[511, 513, 509, 515]),
        )
        .unwrap();
        let mut out = [0xaau8; 104];
        assert_eq!(write_p010(&source, &mut out, 16, 4).unwrap(), 96);
        assert_eq!(&out[..8], &source.y[..8]);
        assert_eq!(&out[16..24], &source.y[8..]);
        assert_eq!(&out[64..72], source.uv.as_ref());
        assert_eq!(codes(&out[8..16]), vec![64; 4]);
        assert_eq!(codes(&out[32..64]), vec![64; 16]);
        assert_eq!(codes(&out[72..96]), vec![512; 12]);
        assert_eq!(&out[96..], &[0xaa; 8]);
        assert_eq!(
            read_p010(&out, 4, 2, 16, 4, ColorRange::Limited).unwrap(),
            source
        );
    }

    #[test]
    fn p010_rejects_pixel_pitch_short_planes_and_eight_bit_input() {
        let source = Yuv420Frame::new(
            4,
            2,
            Yuv420Format::P010Video,
            words(&[65; 8]),
            words(&[513; 4]),
        )
        .unwrap();
        for (stride, slice, length) in [
            (4, 2, 100),
            (8, 1, 100),
            (8, 2, 23),
            (10, 2, 100),
            (8, 3, 100),
        ] {
            assert!(write_p010(&source, &mut vec![0; length], stride, slice).is_err());
        }
        assert!(read_p010(&[0; 23], 4, 2, 8, 2, ColorRange::Limited).is_err());
        let eight =
            Yuv420Frame::new(4, 2, Yuv420Format::Nv12Video, vec![16; 8], vec![128; 4]).unwrap();
        assert!(write_p010(&eight, &mut [0; 24], 8, 2).is_err());
        let malformed = Yuv420Frame {
            y: Vec::<u8>::new().into(),
            ..source.clone()
        };
        assert!(write_p010(&malformed, &mut [0; 24], 8, 2).is_err());
        assert!(malformed.scaled_to(2, 2).is_err());
    }

    #[test]
    fn ten_bit_resizing_retains_sub_eight_bit_steps_and_timing() {
        let source = Yuv420Frame::new(
            4,
            4,
            Yuv420Format::P010Video,
            words(&(64..80).collect::<Vec<_>>()),
            words(&[500, 600, 501, 601, 502, 602, 503, 603]),
        )
        .unwrap();
        let original = frame(source.clone(), TransferCharacteristic::Hlg);
        let output = original.encodable_yuv420(2, 2, original.color).unwrap();
        let samples = output.owned_yuv420().unwrap();
        // Independent 2x2 cell averages, rounded at10-bit precision.
        assert_eq!(codes(&samples.y), [67, 69, 75, 77]);
        assert_eq!(codes(&samples.uv), [502, 602]);
        assert_eq!(output.pts, original.pts);
        assert_eq!(output.duration, original.duration);
        assert_eq!(output.color, original.color);
        let unchanged = source.scaled_to(4, 4).unwrap();
        assert!(std::sync::Arc::ptr_eq(&source.y, &unchanged.y));
        assert!(std::sync::Arc::ptr_eq(&source.uv, &unchanged.uv));
        assert_eq!(codes(&source.y), (64..80).collect::<Vec<_>>());
        let mut original_eight = frame(
            Yuv420Frame::new(
                4,
                4,
                Yuv420Format::Nv12Video,
                (16u8..32).collect(),
                vec![128; 8],
            )
            .unwrap(),
            TransferCharacteristic::Hlg,
        );
        original_eight.color.bits_per_channel = Some(8);
        let promoted_before_scale = original_eight
            .encodable_yuv420(2, 2, color(TransferCharacteristic::Hlg))
            .unwrap();
        assert_eq!(
            codes(&promoted_before_scale.owned_yuv420().unwrap().y),
            [74, 82, 106, 114],
            "8-bit source must be promoted before interpolated samples are rounded"
        );
    }

    #[test]
    fn bt2100_table9_range_anchors_include_full_chroma_minimum_one() {
        // ITU-R BT.2100-3 Table9: narrow Y64/940, C64/512/960;
        // full Y0/1023, C1/512/1023 (the negative nominal peak is1).
        for (limited_chroma, full_chroma) in [(64, 1), (512, 512), (960, 1023)] {
            let limited = Yuv420Frame::new(
                2,
                2,
                Yuv420Format::P010Video,
                words(&[64, 940, 64, 940]),
                words(&[limited_chroma; 2]),
            )
            .unwrap();
            let full = limited.with_range(ColorRange::Full).unwrap();
            assert_eq!(codes(&full.y), [0, 1023, 0, 1023]);
            assert_eq!(codes(&full.uv), [full_chroma; 2]);
            let restored = full.with_range(ColorRange::Limited).unwrap();
            assert_eq!(restored, limited);
        }
    }

    #[test]
    fn actual_source_samples_are_tone_mapped_before_eight_bit_quantization() {
        // BT.2408 diffuse white: HLG signal.75 (10-bit limited code721),
        // PQ203cd/m² (nearest source code573). Code64 is black for both.
        for (transfer, white) in [
            (TransferCharacteristic::Hlg, 721),
            (TransferCharacteristic::Pq, 573),
        ] {
            let source = frame(
                Yuv420Frame::new(
                    2,
                    2,
                    Yuv420Format::P010Video,
                    words(&[64, white, 64, 940]),
                    words(&[512; 2]),
                )
                .unwrap(),
                transfer,
            );
            let converted = source.encodable_sdr709(2, 2, sdr()).unwrap();
            let rgb = converted.to_cpu_rgb().unwrap();
            assert_eq!(&rgb.data[..3], &[0, 0, 0]);
            assert!(rgb.data[3..6].iter().all(|v| *v >= 254));
            assert_eq!(&rgb.data[9..12], &[255, 255, 255]);
            assert_eq!(converted.color, sdr());
            assert_eq!(converted.pts, source.pts);
            assert_eq!(converted.duration, source.duration);
            assert_eq!(
                codes(&source.owned_yuv420().unwrap().y),
                [64, white, 64, 940]
            );
        }
    }

    #[test]
    fn main10_refuses_rgb8_and_color_relabeling() {
        let rgb = CpuFrame::new(4, 4, 12, CpuPixelFormat::Rgb24, vec![127; 48]).unwrap();
        let source = VideoFrame {
            width: 4,
            height: 4,
            pts: MediaTime::new(0, TimeBase::new_unchecked(1, 30)),
            duration: None,
            color: color(TransferCharacteristic::Hlg),
            buffer: FrameBuffer::Cpu(rgb),
        };
        assert!(source.encodable_yuv420(4, 4, source.color).is_err());
        let mut eight = frame(
            Yuv420Frame::new(
                2,
                2,
                Yuv420Format::Nv12Video,
                vec![16, 17, 234, 235],
                vec![127, 129],
            )
            .unwrap(),
            TransferCharacteristic::Pq,
        );
        assert!(
            eight.encodable_yuv420(2, 2, eight.color).is_err(),
            "declared 10-bit source must not accept an 8-bit decoder downgrade"
        );
        eight.color.bits_per_channel = Some(8);
        let promoted = eight
            .encodable_yuv420(2, 2, color(TransferCharacteristic::Pq))
            .unwrap();
        let promoted_yuv = promoted.owned_yuv420().unwrap();
        assert_eq!(codes(&promoted_yuv.y), [64, 68, 936, 940]);
        assert_eq!(codes(&promoted_yuv.uv), [508, 516]);
        assert_eq!(promoted.pts, eight.pts);
        assert_eq!(promoted.duration, eight.duration);
        assert_eq!(
            eight.owned_yuv420().unwrap().y.as_ref(),
            &[16, 17, 234, 235]
        );
        let full = Yuv420Frame::new(
            2,
            2,
            Yuv420Format::Nv12Full,
            vec![0, 255, 0, 255],
            vec![128, 128],
        )
        .unwrap();
        let full_promoted = full.to_p010(ColorRange::Full).unwrap();
        assert_eq!(codes(&full_promoted.y), [0, 1023, 0, 1023]);
        assert_eq!(codes(&full_promoted.uv), [512, 512]);
        let planar = read_yuv420_8(
            &[16, 17, 18, 19, 20, 21, 22, 23, 127, 125, 129, 131],
            4,
            2,
            4,
            2,
            19,
            ColorRange::Limited,
        )
        .unwrap();
        assert_eq!(planar.y.as_ref(), &[16, 17, 18, 19, 20, 21, 22, 23]);
        assert_eq!(planar.uv.as_ref(), &[127, 129, 125, 131]);
        let semiplanar = read_yuv420_8(
            &[16, 17, 18, 19, 20, 21, 22, 23, 127, 129, 125, 131],
            4,
            2,
            4,
            2,
            21,
            ColorRange::Limited,
        )
        .unwrap();
        assert_eq!(planar, semiplanar);
        assert!(read_yuv420_8(&[0; 12], 4, 2, 4, 2, 0x7f420888, ColorRange::Limited).is_err());
        let original = frame(
            Yuv420Frame::new(
                2,
                2,
                Yuv420Format::P010Video,
                words(&[65; 4]),
                words(&[513; 2]),
            )
            .unwrap(),
            TransferCharacteristic::Hlg,
        );
        assert!(
            original
                .encodable_yuv420(2, 2, color(TransferCharacteristic::Pq))
                .is_err()
        );
        assert!(
            original
                .encodable_sdr709(2, 2, color(TransferCharacteristic::Hlg))
                .is_err()
        );
    }

    #[test]
    fn auto_metadata_preserves_hlg_and_pq_without_fabrication() {
        let hlg = settings(TransferCharacteristic::Hlg);
        let mut pq = settings(TransferCharacteristic::Pq);
        assert_eq!(
            InputPolicy::for_settings(&hlg).unwrap().profile,
            Some(HEVC_MAIN10)
        );
        assert_eq!(
            InputPolicy::for_settings(&pq).unwrap().profile,
            Some(HEVC_MAIN10)
        );
        let mut descriptor = [0u8; 25];
        descriptor[1..].copy_from_slice(&(1u8..=24).collect::<Vec<_>>());
        pq.hdr_static_metadata = Some(HdrStaticMetadata::from_cta861_3(&descriptor).unwrap());
        assert_eq!(
            InputPolicy::for_settings(&pq).unwrap().profile,
            Some(HEVC_MAIN10_HDR10)
        );
        assert_eq!(pq.hdr_static_metadata.unwrap().as_bytes(), &descriptor);
        assert!(HdrStaticMetadata::from_cta861_3(&descriptor[..24]).is_err());
        descriptor[0] = 1;
        assert!(HdrStaticMetadata::from_cta861_3(&descriptor).is_err());
    }

    #[test]
    fn codec_negotiation_rejects_depth_transfer_and_range_downgrades() {
        let mut legacy_sdr =
            EncoderSettings::new(Codec::H264, 640, 360, TimeBase::new_unchecked(1, 30));
        legacy_sdr.profile = Some("high".into());
        legacy_sdr.color = Some(sdr());
        let sdr_policy = InputPolicy::for_settings(&legacy_sdr).unwrap();
        assert_eq!(
            sdr_policy.profile, None,
            "preserve existing encoder-selected SDR profile"
        );
        sdr_policy
            .validate_output(&legacy_sdr, Some(1), Some(1), Some(3), Some(2))
            .unwrap();
        let settings = settings(TransferCharacteristic::Hlg);
        let policy = InputPolicy::for_settings(&settings).unwrap();
        policy
            .validate_output(&settings, Some(2), Some(6), Some(7), Some(2))
            .unwrap();
        for (profile, standard, transfer, range) in [
            (Some(1), Some(6), Some(7), Some(2)),
            (None, Some(6), Some(7), Some(2)),
            (Some(2), Some(1), Some(7), Some(2)),
            (Some(2), Some(6), Some(3), Some(2)),
            (Some(2), Some(6), Some(7), Some(1)),
        ] {
            assert!(
                policy
                    .validate_output(&settings, profile, standard, transfer, range)
                    .is_err()
            );
        }
    }

    #[test]
    fn real_hevc_sps_proves_ten_bits_instead_of_trusting_main10_tag() {
        // Extracted verbatim from pinned original HLG and original synthetic
        // 8-bit/PQ hvcC records; exact extraction/hash receipt is in the batch.
        let ten = hex(
            "00000001420101022000000300900000030000030078a003c08010e4d96566924caf016a12241208000003000800000300f040",
        );
        let eight = hex(
            "0000000142010101600000030090000003000003001ea020831659592e4caf016a122012080000030008000003006040",
        );
        require_main10_sps(&ten).unwrap();
        assert!(require_main10_sps(&eight).is_err());
        let mut eight_main10 = eight.clone();
        eight_main10[7] = (eight_main10[7] & !31) | 2;
        assert!(
            require_main10_sps(&eight_main10).is_err(),
            "a Main10 profile tag cannot turn an 8-bit SPS into ten-bit samples"
        );
        assert!(require_main10_sps(&ten[..12]).is_err());
        assert!(require_main10_sps(&[]).is_err());
    }
    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&s[at..at + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn only_sdr_derivative_requests_decoder_transfer_conversion() {
        let hlg = color(TransferCharacteristic::Hlg);
        let pq = color(TransferCharacteristic::Pq);
        // Actual configure policy: raw 10-bit output must request Android's
        // P010 wire value 54; requesting SDR or using a Surface must not do so.
        assert_eq!(
            decoder_buffer_color_format(hlg, None, false).unwrap(),
            Some(54)
        );
        assert_eq!(
            decoder_buffer_color_format(pq, Some(pq), false).unwrap(),
            Some(54)
        );
        assert_eq!(
            decoder_buffer_color_format(hlg, Some(sdr()), false).unwrap(),
            None
        );
        assert_eq!(
            decoder_buffer_color_format(pq, Some(sdr()), false).unwrap(),
            None
        );
        assert_eq!(decoder_buffer_color_format(hlg, None, true).unwrap(), None);
        assert_eq!(
            decoder_buffer_color_format(sdr(), Some(sdr()), false).unwrap(),
            None
        );
        // A genuinely 8-bit PQ source remains 8-bit at decode, before its later
        // Main10 promotion; the encoder's target depth cannot change this.
        let pq8 = ColorInfo {
            bits_per_channel: Some(8),
            ..pq
        };
        assert_eq!(
            decoder_buffer_color_format(pq8, Some(pq), false).unwrap(),
            None
        );
        assert!(decoder_buffer_color_format(hlg, Some(pq), false).is_err());
        assert!(
            !sdr_request(
                color(TransferCharacteristic::Hlg),
                Some(color(TransferCharacteristic::Hlg))
            )
            .unwrap()
        );
        assert!(sdr_request(color(TransferCharacteristic::Hlg), Some(sdr())).unwrap());
        assert!(!sdr_request(sdr(), Some(sdr())).unwrap());
        assert!(
            sdr_request(
                color(TransferCharacteristic::Hlg),
                Some(color(TransferCharacteristic::Pq))
            )
            .is_err()
        );
    }
}
