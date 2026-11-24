//! Explicit float source preservation. No swscale/BGRA8, transfer curve,
//! premultiplication, clipping, scaling or application-specific metadata.
use crate::{CpuFrame, CpuPixelFormat, Result, VideoError};
use ffmpeg_next::{format::Pixel, frame::Video};

pub(super) fn is_float(pixel: Pixel) -> bool {
    matches!(
        pixel,
        Pixel::GBRAPF32LE
            | Pixel::GBRAPF32BE
            | Pixel::GBRPF32LE
            | Pixel::GBRPF32BE
            | Pixel::RGBAF32LE
            | Pixel::RGBAF32BE
    )
}
pub(super) fn decode(frame: &Video) -> Result<CpuFrame> {
    let format = frame.format();
    let (width, height) = (frame.width(), frame.height());
    let bytes = u64::from(width) * u64::from(height) * 16;
    if !is_float(format)
        || width == 0
        || height == 0
        || width > 8192
        || height > 8192
        || bytes > 512 * 1024 * 1024
    {
        return Err(VideoError::UnsupportedCodec(
            "Source is not a bounded float32 RGB(A) frame".into(),
        ));
    }
    let planar = !matches!(format, Pixel::RGBAF32LE | Pixel::RGBAF32BE);
    let alpha = !matches!(format, Pixel::GBRPF32LE | Pixel::GBRPF32BE);
    let big = matches!(
        format,
        Pixel::GBRAPF32BE | Pixel::GBRPF32BE | Pixel::RGBAF32BE
    );
    let mut output = vec![0u8; bytes as usize];
    for channel in 0..4 {
        if channel == 3 && !alpha {
            for pixel in output.as_chunks_mut::<16>().0 {
                pixel[12..16].copy_from_slice(&1f32.to_le_bytes());
            }
            continue;
        }
        let plane = if planar { [2, 0, 1, 3][channel] } else { 0 };
        let stride = frame.stride(plane);
        let row_bytes = width as usize * if planar { 4 } else { 16 };
        let data = frame.data(plane);
        if stride < row_bytes || data.len() < stride * height as usize {
            return Err(VideoError::InvalidData(
                "Truncated float frame plane".into(),
            ));
        }
        for y in 0..height as usize {
            for x in 0..width as usize {
                let from = y * stride
                    + x * if planar { 4 } else { 16 }
                    + if planar { 0 } else { channel * 4 };
                let to = (y * width as usize + x) * 16 + channel * 4;
                let word: [u8; 4] = data[from..from + 4].try_into().expect("four bytes");
                let bits = if big {
                    u32::from_be_bytes(word)
                } else {
                    u32::from_le_bytes(word)
                };
                output[to..to + 4].copy_from_slice(&bits.to_le_bytes());
            }
        }
    }
    CpuFrame::new(
        width,
        height,
        width as usize * 16,
        CpuPixelFormat::Rgba32Float,
        output,
    )
}
