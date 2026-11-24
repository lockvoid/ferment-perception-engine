//! A software decoder's Y'CbCr picture kept as its codes, for a GPU to read
//! and convert (`DecodeOutput::CpuPlanes`): the HDR chain then runs in a
//! shader, as for a hardware decoder's planes, instead of per pixel on the
//! CPU. Any depth from 8 to 16 bits and 4:2:0, 4:2:2 or 4:4:4 chroma stay as
//! the stream has them.
use std::sync::Arc;

use crate::{ColorInfo, CpuFrame, CpuPixelFormat, Result, VideoError};

/// Luma rows of one code, then Cb Cr interleaved rows at the chroma's own
/// resolution, both tightly packed. A code is a byte at eight bits, a
/// little-endian word (in its low bits) past that. The colour — matrix,
/// range, transfer, primaries — is the frame's `ColorInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YcbcrFrame {
    pub width: u32,
    pub height: u32,
    /// Bits per code, 8–16.
    pub depth: u32,
    /// log2 of the chroma's horizontal and vertical subsampling: 4:2:0 is
    /// [1, 1], 4:2:2 [1, 0], 4:4:4 [0, 0].
    pub subsampling: [u32; 2],
    pub luma: Arc<[u8]>,
    pub chroma: Arc<[u8]>,
}

impl YcbcrFrame {
    /// Bytes per code: one at eight bits, two past that.
    pub fn bytes_per_code(&self) -> usize {
        if self.depth > 8 { 2 } else { 1 }
    }

    /// The chroma planes' width and height in samples.
    pub fn chroma_size(&self) -> [u32; 2] {
        [
            self.width.div_ceil(1 << self.subsampling[0]),
            self.height.div_ceil(1 << self.subsampling[1]),
        ]
    }

    /// Pack a decoded picture's codes (`crate::hdr::Picture` reads any
    /// planar, semi-planar or packed layout with its steps and shifts).
    pub(crate) fn pack(picture: &crate::hdr::Picture) -> Result<Self> {
        let depth = picture.depth;
        if !(8..=16).contains(&depth) {
            return Err(VideoError::UnsupportedCodec(format!(
                "Y'CbCr planes of {depth} bits"
            )));
        }
        let [y_plane, cb_plane, cr_plane] = picture.planes;
        if cb_plane.subsampling != cr_plane.subsampling || y_plane.subsampling != (0, 0) {
            return Err(VideoError::UnsupportedCodec(
                "Y'CbCr planes with unequal chroma".into(),
            ));
        }
        let (width, height) = (picture.width as usize, picture.height as usize);
        let wide = depth > 8;
        for plane in &picture.planes {
            plane.check(width, height, wide)?;
        }
        let subsampling = [cb_plane.subsampling.0, cb_plane.subsampling.1];
        let mask = ((1u32 << depth) - 1) as u16;
        let bytes = if wide { 2 } else { 1 };
        let mut luma = vec![0u8; width * height * bytes];
        for (y, row) in luma.chunks_exact_mut(width * bytes).enumerate() {
            y_plane.copy_row(y, width, wide, mask, row);
        }
        let chroma_width = width.div_ceil(1 << subsampling[0]);
        let chroma_height = height.div_ceil(1 << subsampling[1]);
        let mut chroma = vec![0u8; chroma_width * chroma_height * 2 * bytes];
        let mut cb = vec![0u8; chroma_width * bytes];
        let mut cr = vec![0u8; chroma_width * bytes];
        for (y, row) in chroma
            .chunks_exact_mut(chroma_width * 2 * bytes)
            .enumerate()
        {
            // Chroma rows of the subsampled plane: row `y` of luma row y << sy.
            let source = y << subsampling[1];
            cb_plane.copy_row(source, chroma_width, wide, mask, &mut cb);
            cr_plane.copy_row(source, chroma_width, wide, mask, &mut cr);
            for ((pair, cb), cr) in row
                .chunks_exact_mut(2 * bytes)
                .zip(cb.chunks_exact(bytes))
                .zip(cr.chunks_exact(bytes))
            {
                pair[..bytes].copy_from_slice(cb);
                pair[bytes..].copy_from_slice(cr);
            }
        }
        Ok(Self {
            width: picture.width,
            height: picture.height,
            depth,
            subsampling,
            luma: luma.into(),
            chroma: chroma.into(),
        })
    }

    /// Read back as an HDR picture for the CPU chain (`crate::hdr`).
    fn picture(&self) -> crate::hdr::Picture<'_> {
        let bytes = self.bytes_per_code();
        let [chroma_width, _] = self.chroma_size();
        let chroma_stride = chroma_width as usize * 2 * bytes;
        let subsampling = (self.subsampling[0], self.subsampling[1]);
        let chroma = |offset| crate::hdr::Plane {
            data: &self.chroma,
            stride: chroma_stride,
            step: 2 * bytes,
            offset,
            shift: 0,
            subsampling,
        };
        crate::hdr::Picture {
            width: self.width,
            height: self.height,
            depth: self.depth,
            planes: [
                crate::hdr::Plane {
                    data: &self.luma,
                    stride: self.width as usize * bytes,
                    step: bytes,
                    offset: 0,
                    shift: 0,
                    subsampling: (0, 0),
                },
                chroma(0),
                chroma(bytes),
            ],
        }
    }

    /// RGB on the CPU, for an explicit readback: an HDR picture through the
    /// engine's SDR chain. Planes are made only for HDR sources.
    pub fn to_rgb24(&self, color: ColorInfo) -> Result<CpuFrame> {
        crate::hdr::to_sdr(&self.picture(), color, CpuPixelFormat::Rgb24)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdr::{Picture, Plane};

    /// A 4×2 picture of planar 10-bit codes in the low bits (yuv422p10le).
    fn yuv422p10() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let words = |codes: &[u16]| {
            codes
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect::<Vec<_>>()
        };
        (
            words(&[64, 397, 721, 940, 100, 200, 300, 400]),
            words(&[512, 600, 520, 610]),
            words(&[480, 490, 500, 510]),
        )
    }

    #[test]
    fn planes_keep_the_codes_their_depth_and_chroma_as_the_stream_has_them() {
        let (y, cb, cr) = yuv422p10();
        let plane = |data: &'static [u8], stride, subsampling| Plane {
            data,
            stride,
            step: 2,
            offset: 0,
            shift: 0,
            subsampling,
        };
        let (y, cb, cr): (&'static [u8], &'static [u8], &'static [u8]) = (
            Box::leak(y.into_boxed_slice()),
            Box::leak(cb.into_boxed_slice()),
            Box::leak(cr.into_boxed_slice()),
        );
        let frame = YcbcrFrame::pack(&Picture {
            width: 4,
            height: 2,
            depth: 10,
            planes: [
                plane(y, 8, (0, 0)),
                plane(cb, 4, (1, 0)),
                plane(cr, 4, (1, 0)),
            ],
        })
        .unwrap();
        assert_eq!((frame.depth, frame.subsampling), (10, [1, 0]));
        assert_eq!(frame.chroma_size(), [2, 2]);
        let words = |bytes: &[u8]| {
            bytes
                .chunks_exact(2)
                .map(|w| u16::from_le_bytes([w[0], w[1]]))
                .collect::<Vec<_>>()
        };
        assert_eq!(words(&frame.luma), [64, 397, 721, 940, 100, 200, 300, 400]);
        // Cb Cr interleaved, row by row.
        assert_eq!(
            words(&frame.chroma),
            [512, 480, 600, 490, 520, 500, 610, 510]
        );
    }

    #[test]
    fn p010_codes_come_down_from_the_high_bits() {
        // One 2×2 P010 picture: luma words, then one interleaved CbCr pair.
        let luma: Vec<u8> = [940u16, 64, 512, 721]
            .iter()
            .flat_map(|c| (c << 6).to_le_bytes())
            .collect();
        let chroma: Vec<u8> = [600u16, 400]
            .iter()
            .flat_map(|c| (c << 6).to_le_bytes())
            .collect();
        let (luma, chroma): (&'static [u8], &'static [u8]) = (
            Box::leak(luma.into_boxed_slice()),
            Box::leak(chroma.into_boxed_slice()),
        );
        let frame = YcbcrFrame::pack(&Picture {
            width: 2,
            height: 2,
            depth: 10,
            planes: [
                Plane {
                    data: luma,
                    stride: 4,
                    step: 2,
                    offset: 0,
                    shift: 6,
                    subsampling: (0, 0),
                },
                Plane {
                    data: chroma,
                    stride: 4,
                    step: 4,
                    offset: 0,
                    shift: 6,
                    subsampling: (1, 1),
                },
                Plane {
                    data: chroma,
                    stride: 4,
                    step: 4,
                    offset: 2,
                    shift: 6,
                    subsampling: (1, 1),
                },
            ],
        })
        .unwrap();
        let words: Vec<u16> = frame
            .luma
            .chunks_exact(2)
            .chain(frame.chroma.chunks_exact(2))
            .map(|w| u16::from_le_bytes([w[0], w[1]]))
            .collect();
        assert_eq!(words, [940, 64, 512, 721, 600, 400]);
    }
}
