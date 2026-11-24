//! Scaling owned 4:2:0 samples without an RGB8 intermediate.
use super::*;
#[cfg(any(not(target_os = "macos"), test))]
use crate::ColorRange;

impl Yuv420Frame {
    pub(crate) fn validate_owned_layout(&self) -> Result<()> {
        let luma = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(self.format.bytes_per_sample()));
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
            || luma != Some(self.y.len())
            || luma.map(|n| n / 2) != Some(self.uv.len())
        {
            return Err(VideoError::InvalidData(
                "invalid owned YUV420 plane extent".into(),
            ));
        }
        Ok(())
    }
    /// Center-aligned bilinear resampling at the source bit depth. A no-op
    /// preserves both plane allocations and every original sample bit.
    pub fn scaled_to(&self, width: u32, height: u32) -> Result<Self> {
        self.validate_owned_layout()?;
        if width == self.width && height == self.height {
            return Ok(self.clone());
        }
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(VideoError::InvalidData(
                "YUV420 scaling requires positive even dimensions".into(),
            ));
        }
        let sample_bytes = self.format.bytes_per_sample();
        let y = scale_plane(
            &self.y,
            self.width,
            self.height,
            width,
            height,
            1,
            sample_bytes,
        )?;
        let uv = scale_plane(
            &self.uv,
            self.width / 2,
            self.height / 2,
            width / 2,
            height / 2,
            2,
            sample_bytes,
        )?;
        Self::new(width, height, self.format, y, uv)
    }

    /// Promote original NV12 source codes without RGB conversion. Limited
    /// 8-bit code16/235/128 becomes 10-bit64/940/512; full-range luma spans
    /// 0..1023 and chroma remains centered at512. This does not authorize an
    /// NV12 frame that was declared as a10-bit decoded source.
    #[cfg(any(not(target_os = "macos"), test))]
    pub(crate) fn to_p010(&self, range: ColorRange) -> Result<Self> {
        self.validate_owned_layout()?;
        if self.format.bytes_per_sample() == 2 {
            return self.with_range(range);
        }
        let target_full = match range {
            ColorRange::Full => true,
            ColorRange::Limited => false,
            _ => {
                return Err(VideoError::InvalidData(
                    "P010 output range is not declared".into(),
                ));
            }
        };
        let source_full = self.format == Yuv420Format::Nv12Full;
        let promote = |data: &[u8], chroma: bool| -> Vec<u8> {
            let (source_center, source_span, target_center, target_span) = if chroma {
                (
                    128.0,
                    if source_full { 255.0 } else { 224.0 },
                    512.0,
                    if target_full { 1023.0 } else { 896.0 },
                )
            } else {
                (
                    if source_full { 0.0 } else { 16.0 },
                    if source_full { 255.0 } else { 219.0 },
                    if target_full { 0.0 } else { 64.0 },
                    if target_full { 1023.0 } else { 876.0 },
                )
            };
            let mut out = Vec::with_capacity(data.len() * 2);
            for code in data {
                let value =
                    (f64::from(*code) - source_center) / source_span * target_span + target_center;
                append_code(&mut out, value.round().clamp(0.0, 1023.0) as u16, 2);
            }
            out
        };
        Self::new(
            self.width,
            self.height,
            if target_full {
                Yuv420Format::P010Full
            } else {
                Yuv420Format::P010Video
            },
            promote(&self.y, false),
            promote(&self.uv, true),
        )
    }

    /// Convert only code range; primaries/transfer/matrix conversion is never
    /// hidden in this operation. Chroma remains centered at code512 in P010.
    #[cfg(any(not(target_os = "macos"), test))]
    pub(crate) fn with_range(&self, range: ColorRange) -> Result<Self> {
        self.validate_owned_layout()?;
        let target_full = match range {
            ColorRange::Full => true,
            ColorRange::Limited => false,
            _ => {
                return Err(VideoError::InvalidData(
                    "encoder input must state its YUV range".into(),
                ));
            }
        };
        let source_full = matches!(self.format, Yuv420Format::Nv12Full | Yuv420Format::P010Full);
        if source_full == target_full {
            return Ok(self.clone());
        }
        let bytes = self.format.bytes_per_sample();
        let max = if bytes == 2 { 1023.0 } else { 255.0 };
        let scale = if bytes == 2 { 4.0 } else { 1.0 };
        let convert = |data: &[u8], chroma: bool| -> Vec<u8> {
            let mut out = Vec::with_capacity(data.len());
            for at in (0..data.len()).step_by(bytes) {
                let code = read_code(data, at, bytes);
                let (center, span) = if chroma {
                    (128.0 * scale, 224.0 * scale)
                } else {
                    (16.0 * scale, 219.0 * scale)
                };
                let value = if target_full {
                    if chroma {
                        (code - center) * max / span + 128.0 * scale
                    } else {
                        (code - center) * max / span
                    }
                } else if chroma {
                    (code - 128.0 * scale) * span / max + center
                } else {
                    code * span / max + center
                };
                append_code(&mut out, value.round().clamp(0.0, max) as u16, bytes);
            }
            out
        };
        let format = match (bytes, target_full) {
            (1, true) => Yuv420Format::Nv12Full,
            (1, false) => Yuv420Format::Nv12Video,
            (_, true) => Yuv420Format::P010Full,
            (_, false) => Yuv420Format::P010Video,
        };
        Self::new(
            self.width,
            self.height,
            format,
            convert(&self.y, false),
            convert(&self.uv, true),
        )
    }
}

fn read_code(data: &[u8], at: usize, bytes: usize) -> f64 {
    if bytes == 1 {
        f64::from(data[at])
    } else {
        f64::from(u16::from_le_bytes([data[at], data[at + 1]]) >> 6)
    }
}
fn append_code(data: &mut Vec<u8>, code: u16, bytes: usize) {
    if bytes == 1 {
        data.push(code as u8);
    } else {
        data.extend_from_slice(&(code << 6).to_le_bytes());
    }
}
fn scale_plane(
    data: &[u8],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    channels: usize,
    bytes: usize,
) -> Result<Vec<u8>> {
    let length = (dw as usize)
        .checked_mul(dh as usize)
        .and_then(|n| n.checked_mul(channels * bytes))
        .ok_or_else(|| VideoError::InvalidData("YUV scaling extent overflow".into()))?;
    let mut out = Vec::with_capacity(length);
    let max = if bytes == 1 { 255.0 } else { 1023.0 };
    for row in 0..dh {
        let sy = ((f64::from(row) + 0.5) * f64::from(sh) / f64::from(dh) - 0.5).max(0.0);
        let y0 = (sy.floor() as u32).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - f64::from(y0);
        for col in 0..dw {
            let sx = ((f64::from(col) + 0.5) * f64::from(sw) / f64::from(dw) - 0.5).max(0.0);
            let x0 = (sx.floor() as u32).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - f64::from(x0);
            for channel in 0..channels {
                let sample = |x, y| {
                    read_code(
                        data,
                        ((y as usize * sw as usize + x as usize) * channels + channel) * bytes,
                        bytes,
                    )
                };
                let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
                let bottom = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
                append_code(
                    &mut out,
                    (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, max) as u16,
                    bytes,
                );
            }
        }
    }
    Ok(out)
}

impl VideoFrame {
    /// Owned source samples for a ten-bit encoder. This is also used by the
    /// Android pool-pressure spill: no RGB8 conversion may precede encoding.
    pub fn owned_yuv420(&self) -> Result<Yuv420Frame> {
        match &self.buffer {
            FrameBuffer::Yuv420(frame) => Ok(frame.clone()),
            #[cfg(target_os = "android")]
            FrameBuffer::Android(image) => match image.rotation_pixels(self.color)? {
                FrameBuffer::Yuv420(frame) => Ok(frame),
                _ => Err(VideoError::UnsupportedCodec(
                    "the decoder did not deliver YUV420 samples".into(),
                )),
            },
            _ => Err(VideoError::UnsupportedCodec(
                "10-bit encoding requires original YUV420 samples, not RGB8".into(),
            )),
        }
    }

    #[cfg(any(not(target_os = "macos"), test))]
    pub(crate) fn encodable_yuv420(
        &self,
        width: u32,
        height: u32,
        color: ColorInfo,
    ) -> Result<Self> {
        if self.color.primaries != color.primaries
            || self.color.transfer != color.transfer
            || self.color.matrix != color.matrix
        {
            return Err(VideoError::UnsupportedCodec(
                "YUV scaling cannot relabel a different transfer function or gamut".into(),
            ));
        }
        let source = self.owned_yuv420()?;
        if color.bits_per_channel != Some(10)
            || !matches!(
                (
                    self.color.bits_per_channel,
                    source.format.bytes_per_sample()
                ),
                (Some(8), 1) | (Some(10), 2)
            )
        {
            return Err(VideoError::UnsupportedCodec(
                "Main10 requires original source-depth YUV; a10-bit source cannot arrive as NV12"
                    .into(),
            ));
        }
        if (source.width, source.height) != (self.width, self.height) {
            return Err(VideoError::InvalidData(
                "frame and YUV plane dimensions disagree".into(),
            ));
        }
        let scaled = source.to_p010(color.range)?.scaled_to(width, height)?;
        Ok(Self {
            width,
            height,
            color,
            buffer: FrameBuffer::Yuv420(scaled),
            ..self.clone()
        })
    }
}

#[cfg(any(target_os = "android", test))]
impl VideoFrame {
    /// Same bounded diffuse-white HDR→SDR conversion already used by the
    /// Android VideoBulkReader.Pixels path. Read/scale original YUV codes at
    /// source depth, then apply transfer and primaries conversion in f64;
    /// quantization to8-bit happens only after the BT.709 OETF.
    pub(crate) fn encodable_sdr709(
        &self,
        width: u32,
        height: u32,
        color: ColorInfo,
    ) -> Result<Self> {
        use crate::{ColorMatrix, ColorPrimaries, TransferCharacteristic};
        if (
            color.primaries,
            color.matrix,
            color.transfer,
            color.range,
            color.bits_per_channel,
        ) != (
            ColorPrimaries::Bt709,
            ColorMatrix::Bt709,
            TransferCharacteristic::Bt709,
            ColorRange::Limited,
            Some(8),
        ) || self.color.primaries != ColorPrimaries::Bt2020
            || self.color.matrix != ColorMatrix::Bt2020NonConstant
            || !matches!(
                self.color.transfer,
                TransferCharacteristic::Hlg | TransferCharacteristic::Pq
            )
        {
            return Err(VideoError::UnsupportedCodec(
                "HDR conversion requires BT.2020 HLG/PQ to BT.709 SDR".into(),
            ));
        }
        let yuv = self.owned_yuv420()?.scaled_to(width, height)?;
        let bytes = yuv.format.bytes_per_sample();
        let max = if bytes == 2 { 1023.0 } else { 255.0 };
        let factor = (max + 1.0) / 256.0;
        let full = matches!(yuv.format, Yuv420Format::Nv12Full | Yuv420Format::P010Full);
        let transfer = self.color.transfer;
        let white = if transfer == TransferCharacteristic::Pq {
            203.0 / 10_000.0
        } else {
            hdr_light(0.75, transfer)
        };
        let length = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| VideoError::InvalidData("tone-map extent overflow".into()))?;
        let mut rgb = Vec::with_capacity(length);
        for row in 0..height as usize {
            for column in 0..width as usize {
                let yy = read_code(&yuv.y, (row * width as usize + column) * bytes, bytes);
                let uv = ((row / 2) * width as usize + (column / 2) * 2) * bytes;
                let y = if full {
                    yy / max
                } else {
                    (yy - 16.0 * factor) / (219.0 * factor)
                }
                .clamp(0.0, 1.0);
                let span = if full { max } else { 224.0 * factor };
                let cb = ((read_code(&yuv.uv, uv, bytes) - 128.0 * factor) / span).clamp(-0.5, 0.5);
                let cr = ((read_code(&yuv.uv, uv + bytes, bytes) - 128.0 * factor) / span)
                    .clamp(-0.5, 0.5);
                let r = hdr_light(y + 1.4746 * cr, transfer) / white;
                let g = hdr_light(y - 0.16455 * cb - 0.57135 * cr, transfer) / white;
                let b = hdr_light(y + 1.8814 * cb, transfer) / white;
                for linear in [
                    1.6605 * r - 0.5876 * g - 0.0728 * b,
                    -0.1246 * r + 1.1329 * g - 0.0083 * b,
                    -0.0182 * r - 0.1006 * g + 1.1187 * b,
                ] {
                    let linear = linear.clamp(0.0, 1.0);
                    let encoded = if linear < 0.018 {
                        4.5 * linear
                    } else {
                        1.099 * linear.powf(0.45) - 0.099
                    };
                    rgb.push((encoded * 255.0).round().clamp(0.0, 255.0) as u8);
                }
            }
        }
        let cpu = CpuFrame::new(
            width,
            height,
            width as usize * 3,
            CpuPixelFormat::Rgb24,
            rgb,
        )?;
        Ok(Self {
            width,
            height,
            color,
            buffer: FrameBuffer::Cpu(cpu),
            ..self.clone()
        })
    }
}

#[cfg(any(target_os = "android", test))]
fn hdr_light(signal: f64, transfer: crate::TransferCharacteristic) -> f64 {
    let value = signal.clamp(0.0, 1.0);
    if transfer == crate::TransferCharacteristic::Pq {
        let power = value.powf(1.0 / (2523.0 / 4096.0 * 128.0));
        ((power - 3424.0 / 4096.0).max(0.0)
            / (2413.0 / 4096.0 * 32.0 - 2392.0 / 4096.0 * 32.0 * power))
            .powf(1.0 / (2610.0 / 16384.0))
    } else {
        let a = 0.17883277f64;
        let b = 1.0 - 4.0 * a;
        let c = 0.5 - a * (4.0 * a).ln();
        let scene = if value <= 0.5 {
            value * value / 3.0
        } else {
            (((value - c) / a).exp() + b) / 12.0
        };
        scene.clamp(0.0, 1.0).powf(1.2)
    }
}
