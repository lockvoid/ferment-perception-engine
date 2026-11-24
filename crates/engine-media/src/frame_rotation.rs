//! Exact sample permutations for display orientation and portrait processing.
//! Bi-planar 8/10-bit pictures stay bi-planar until a consumer explicitly asks
//! for RGB; rotating an HDR picture must not silently quantize its samples.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Yuv420Format {
    Nv12Video,
    Nv12Full,
    P010Video,
    P010Full,
}
impl Yuv420Format {
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12Video | Self::Nv12Full => 1,
            Self::P010Video | Self::P010Full => 2,
        }
    }
    fn full_range(self) -> bool {
        matches!(self, Self::Nv12Full | Self::P010Full)
    }
}

/// Tightly packed luma and interleaved CbCr. P010 stores the original 10-bit
/// code in the high ten bits of each little-endian word, never narrowed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Frame {
    pub width: u32,
    pub height: u32,
    pub format: Yuv420Format,
    pub y: Arc<[u8]>,
    pub uv: Arc<[u8]>,
}
impl Yuv420Frame {
    pub fn new(
        width: u32,
        height: u32,
        format: Yuv420Format,
        y: Vec<u8>,
        uv: Vec<u8>,
    ) -> Result<Self> {
        let bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(format.bytes_per_sample()));
        if width == 0
            || height == 0
            || !width.is_multiple_of(2)
            || !height.is_multiple_of(2)
            || bytes != Some(y.len())
            || bytes.map(|n| n / 2) != Some(uv.len())
        {
            return Err(VideoError::InvalidData(
                "invalid owned YUV420 plane extent".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            format,
            y: y.into(),
            uv: uv.into(),
        })
    }

    fn rotated(&self, turns: u8) -> Result<Self> {
        let sample = self.format.bytes_per_sample();
        let (width, height) = rotated_size(self.width, self.height, turns);
        Self::new(
            width,
            height,
            self.format,
            rotate_plane(
                &self.y,
                self.width as usize,
                self.height as usize,
                self.width as usize * sample,
                sample,
                turns,
            )?,
            rotate_plane(
                &self.uv,
                self.width as usize / 2,
                self.height as usize / 2,
                self.width as usize * sample,
                sample * 2,
                turns,
            )?,
        )
    }

    pub(crate) fn to_rgb24(&self, color: ColorInfo) -> Result<CpuFrame> {
        let sample = self.format.bytes_per_sample();
        if crate::hdr::is_hdr(color.transfer) {
            // P010 keeps its ten bits high in each little-endian word.
            let shift = if sample == 2 { 6 } else { 0 };
            let stride = self.width as usize * sample;
            let plane = |data, step, offset, subsampling| crate::hdr::Plane {
                data,
                stride,
                step,
                offset,
                shift,
                subsampling,
            };
            let picture = crate::hdr::Picture {
                width: self.width,
                height: self.height,
                depth: if sample == 2 { 10 } else { 8 },
                planes: [
                    plane(&self.y, sample, 0, (0, 0)),
                    plane(&self.uv, 2 * sample, 0, (1, 1)),
                    plane(&self.uv, 2 * sample, sample, (1, 1)),
                ],
            };
            let range = if self.format.full_range() {
                crate::ColorRange::Full
            } else {
                crate::ColorRange::Limited
            };
            return crate::hdr::to_sdr(
                &picture,
                ColorInfo { range, ..color },
                CpuPixelFormat::Rgb24,
            );
        }
        let read = |data: &[u8], offset: usize| -> f32 {
            if sample == 1 {
                f32::from(data[offset])
            } else {
                f32::from(u16::from_le_bytes([data[offset], data[offset + 1]]) >> 6) / 4.0
            }
        };
        let mut rgb = vec![0; self.width as usize * self.height as usize * 3];
        for row in 0..self.height as usize {
            for column in 0..self.width as usize {
                let y = read(&self.y, (row * self.width as usize + column) * sample);
                let uv = ((row / 2) * self.width as usize + (column / 2) * 2) * sample;
                let pixel = ycbcr_to_rgb(
                    y,
                    read(&self.uv, uv),
                    read(&self.uv, uv + sample),
                    self.format.full_range(),
                    color,
                );
                let offset = (row * self.width as usize + column) * 3;
                rgb[offset..offset + 3].copy_from_slice(&pixel);
            }
        }
        CpuFrame::new(
            self.width,
            self.height,
            self.width as usize * 3,
            CpuPixelFormat::Rgb24,
            rgb,
        )
    }
}

fn rotated_size(width: u32, height: u32, turns: u8) -> (u32, u32) {
    if turns.is_multiple_of(2) {
        (width, height)
    } else {
        (height, width)
    }
}

// Rotate whole samples (including an interleaved chroma pair), not bytes
// independently. That distinction is what preserves Cb/Cr and 16-bit words.
fn rotate_plane(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    sample: usize,
    turns: u8,
) -> Result<Vec<u8>> {
    let row_bytes = width
        .checked_mul(sample)
        .ok_or_else(|| VideoError::InvalidData("rotation extent overflow".into()))?;
    let extent = stride
        .checked_mul(height)
        .ok_or_else(|| VideoError::InvalidData("rotation extent overflow".into()))?;
    if width == 0 || height == 0 || sample == 0 || stride < row_bytes || data.len() < extent {
        return Err(VideoError::InvalidData(
            "rotation plane is too short".into(),
        ));
    }
    let turns = turns % 4;
    let out_width = if turns.is_multiple_of(2) {
        width
    } else {
        height
    };
    let mut out =
        vec![
            0;
            row_bytes
                .checked_mul(height)
                .ok_or_else(|| VideoError::InvalidData("rotation extent overflow".into()))?
        ];
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = match turns {
                0 => (x, y),
                1 => (height - 1 - y, x),
                2 => (width - 1 - x, height - 1 - y),
                _ => (y, width - 1 - x),
            };
            let source = y * stride + x * sample;
            let destination = (dy * out_width + dx) * sample;
            out[destination..destination + sample].copy_from_slice(&data[source..source + sample]);
        }
    }
    Ok(out)
}

impl VideoFrame {
    /// A clockwise quarter-turn permutation, retaining exact timing and color.
    /// Hardware buffers are copied only because this operation requests pixels.
    pub fn rotated_clockwise(&self, quarter_turns: u8) -> Result<Self> {
        let turns = quarter_turns % 4;
        if turns == 0 {
            return Ok(self.clone());
        }
        let (width, height) = rotated_size(self.width, self.height, turns);
        let rotate_owned = |buffer: FrameBuffer| -> Result<FrameBuffer> {
            let extent = match &buffer {
                FrameBuffer::Cpu(cpu) => (cpu.width, cpu.height),
                FrameBuffer::Yuv420(cpu) => (cpu.width, cpu.height),
                _ => {
                    return Err(VideoError::InvalidData(
                        "rotation requires owned pixels".into(),
                    ));
                }
            };
            if extent != (self.width, self.height) {
                return Err(VideoError::InvalidData(
                    "rotation frame/buffer dimensions disagree".into(),
                ));
            }
            match buffer {
                FrameBuffer::Cpu(cpu) => Ok(FrameBuffer::Cpu(CpuFrame::new(
                    width,
                    height,
                    width as usize * cpu.format.bytes_per_pixel(),
                    cpu.format,
                    rotate_plane(
                        &cpu.data,
                        cpu.width as usize,
                        cpu.height as usize,
                        cpu.stride,
                        cpu.format.bytes_per_pixel(),
                        turns,
                    )?,
                )?)),
                FrameBuffer::Yuv420(cpu) => Ok(FrameBuffer::Yuv420(cpu.rotated(turns)?)),
                _ => Err(VideoError::InvalidData(
                    "rotation readback did not produce owned pixels".into(),
                )),
            }
        };
        let buffer = match &self.buffer {
            FrameBuffer::Cpu(_) | FrameBuffer::Yuv420(_) => rotate_owned(self.buffer.clone())?,
            // A GPU applies the display matrix as it converts the planes.
            FrameBuffer::Ycbcr(_) => {
                return Err(VideoError::UnsupportedCodec(
                    "Y'CbCr planes turn where a GPU converts them".into(),
                ));
            }
            #[cfg(target_os = "macos")]
            FrameBuffer::MacOS(source) => {
                let owned = rotate_owned(read_macos(source)?)?;
                let native = match owned {
                    FrameBuffer::Cpu(cpu) => MacPixelBuffer::from_cpu_bgra(&cpu)?,
                    FrameBuffer::Yuv420(cpu) => cpu.to_macos()?,
                    _ => unreachable!(),
                };
                source
                    .as_cv_pixel_buffer()
                    .propagate_attachments(native.as_cv_pixel_buffer());
                FrameBuffer::MacOS(native)
            }
            #[cfg(target_os = "android")]
            FrameBuffer::Android(source) => rotate_owned(source.rotation_pixels(self.color)?)?,
        };
        Ok(Self {
            width,
            height,
            buffer,
            ..self.clone()
        })
    }
}

#[cfg(target_os = "macos")]
struct Locked<'a> {
    buffer: &'a objc2_core_video::CVPixelBuffer,
    flags: objc2_core_video::CVPixelBufferLockFlags,
}
#[cfg(target_os = "macos")]
impl<'a> Locked<'a> {
    fn new(
        buffer: &'a objc2_core_video::CVPixelBuffer,
        flags: objc2_core_video::CVPixelBufferLockFlags,
    ) -> Result<Self> {
        // SAFETY: the caller retains the pixel buffer until this guard drops.
        let status = unsafe { objc2_core_video::CVPixelBufferLockBaseAddress(buffer, flags) };
        if status != 0 {
            return Err(VideoError::platform("CVPixelBufferLockBaseAddress", status));
        }
        Ok(Self { buffer, flags })
    }
}
#[cfg(target_os = "macos")]
impl Drop for Locked<'_> {
    fn drop(&mut self) {
        // SAFETY: paired with the successful lock and the still-retained buffer.
        unsafe {
            objc2_core_video::CVPixelBufferUnlockBaseAddress(self.buffer, self.flags);
        }
    }
}

#[cfg(target_os = "macos")]
fn read_macos(source: &MacPixelBuffer) -> Result<FrameBuffer> {
    use objc2_core_video::*;
    let buffer = source.as_cv_pixel_buffer();
    let (width, height) = (
        CVPixelBufferGetWidth(buffer) as u32,
        CVPixelBufferGetHeight(buffer) as u32,
    );
    let native = CVPixelBufferGetPixelFormatType(buffer);
    let format = match native {
        value if value == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange => {
            Some(Yuv420Format::Nv12Video)
        }
        value if value == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => {
            Some(Yuv420Format::Nv12Full)
        }
        value if value == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange => {
            Some(Yuv420Format::P010Video)
        }
        value if value == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange => {
            Some(Yuv420Format::P010Full)
        }
        _ => None,
    };
    if native != kCVPixelFormatType_32BGRA && format.is_none() {
        let converted = MacPixelBuffer::new(crate::platform::macos::transfer_scaled(
            buffer,
            width,
            height,
            encoder_input_format(buffer),
        )?);
        return read_macos(&converted);
    }
    let _lock = Locked::new(buffer, CVPixelBufferLockFlags::ReadOnly)?;
    let copy = |pointer: *mut std::ffi::c_void,
                stride: usize,
                rows: usize,
                row_bytes: usize|
     -> Result<Vec<u8>> {
        if pointer.is_null() || stride < row_bytes {
            return Err(VideoError::InvalidData(
                "invalid native rotation plane".into(),
            ));
        }
        let mut result = vec![0; rows * row_bytes];
        for row in 0..rows {
            // SAFETY: CoreVideo's lock exposes each documented row; copy only its active samples.
            let bytes = unsafe {
                std::slice::from_raw_parts(pointer.cast::<u8>().add(row * stride), row_bytes)
            };
            result[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(bytes);
        }
        Ok(result)
    };
    if let Some(format) = format {
        if CVPixelBufferGetPlaneCount(buffer) != 2 {
            return Err(VideoError::InvalidData(
                "YUV420 rotation requires two planes".into(),
            ));
        }
        let row = width as usize * format.bytes_per_sample();
        let y = copy(
            CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
            CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
            height as usize,
            row,
        )?;
        let uv = copy(
            CVPixelBufferGetBaseAddressOfPlane(buffer, 1),
            CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
            height as usize / 2,
            row,
        )?;
        Ok(FrameBuffer::Yuv420(Yuv420Frame::new(
            width, height, format, y, uv,
        )?))
    } else {
        let bytes = copy(
            CVPixelBufferGetBaseAddress(buffer),
            CVPixelBufferGetBytesPerRow(buffer),
            height as usize,
            width as usize * 4,
        )?;
        Ok(FrameBuffer::Cpu(CpuFrame::new(
            width,
            height,
            width as usize * 4,
            CpuPixelFormat::Bgra8,
            bytes,
        )?))
    }
}

#[cfg(target_os = "macos")]
impl Yuv420Frame {
    pub(crate) fn to_macos(&self) -> Result<MacPixelBuffer> {
        use objc2_core_video::*;
        let format = match self.format {
            Yuv420Format::Nv12Video => kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            Yuv420Format::Nv12Full => kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            Yuv420Format::P010Video => kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            Yuv420Format::P010Full => kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        };
        let inner = allocate_pixel_buffer(self.width, self.height, format)?;
        let buffer = inner.as_ref();
        let lock = Locked::new(buffer, CVPixelBufferLockFlags::empty())?;
        let row_bytes = self.width as usize * self.format.bytes_per_sample();
        for (plane, data, rows) in [
            (0, self.y.as_ref(), self.height as usize),
            (1, self.uv.as_ref(), self.height as usize / 2),
        ] {
            let pointer = CVPixelBufferGetBaseAddressOfPlane(buffer, plane).cast::<u8>();
            let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
            if pointer.is_null() || stride < row_bytes {
                return Err(VideoError::InvalidData(
                    "invalid writable rotation plane".into(),
                ));
            }
            for row in 0..rows {
                // SAFETY: a fresh retained and write-locked destination owns every copied row.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr().add(row * row_bytes),
                        pointer.add(row * stride),
                        row_bytes,
                    );
                }
            }
        }
        drop(lock);
        Ok(MacPixelBuffer::new(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorMatrix, ColorPrimaries, ColorRange, TimeBase, TransferCharacteristic};

    fn color() -> ColorInfo {
        ColorInfo {
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Hlg,
            matrix: ColorMatrix::Bt2020NonConstant,
            range: ColorRange::Limited,
            bits_per_channel: Some(10),
        }
    }
    fn frame(buffer: FrameBuffer) -> VideoFrame {
        VideoFrame {
            width: 32,
            height: 48,
            pts: MediaTime::new(7, TimeBase::new_unchecked(1, 30)),
            duration: Some(MediaTime::new(1, TimeBase::new_unchecked(1, 30))),
            color: color(),
            buffer,
        }
    }
    fn buffers() -> Vec<FrameBuffer> {
        let sample = |x: usize, y: usize| (16 + (x * 5 + y * 3) % 200) as u8;
        let mut bgra = Vec::new();
        let mut y8 = Vec::new();
        let mut y10 = Vec::new();
        for y in 0..48 {
            for x in 0..32 {
                let v = sample(x, y);
                bgra.extend_from_slice(&[v, 235 - v, 80, 255]);
                y8.push(v);
                y10.extend_from_slice(&((u16::from(v) * 4) << 6).to_le_bytes());
            }
        }
        let mut uv10 = Vec::new();
        for _ in 0..32 * 24 {
            uv10.extend_from_slice(&(512_u16 << 6).to_le_bytes());
        }
        vec![
            FrameBuffer::Cpu(CpuFrame::new(32, 48, 32 * 4, CpuPixelFormat::Bgra8, bgra).unwrap()),
            FrameBuffer::Yuv420(
                Yuv420Frame::new(32, 48, Yuv420Format::Nv12Video, y8, vec![128; 32 * 24]).unwrap(),
            ),
            FrameBuffer::Yuv420(
                Yuv420Frame::new(32, 48, Yuv420Format::P010Video, y10, uv10).unwrap(),
            ),
        ]
    }
    fn assert_same(a: &FrameBuffer, b: &FrameBuffer) {
        match (a, b) {
            (FrameBuffer::Cpu(a), FrameBuffer::Cpu(b)) => assert_eq!(a, b),
            (FrameBuffer::Yuv420(a), FrameBuffer::Yuv420(b)) => assert_eq!(a, b),
            _ => panic!("quarter turn changed owned sample layout"),
        }
    }
    /// Kill: narrow P010 through RGB8 or rotate Cb/Cr as independent words.
    /// The source's neutral chroma and multiples-of-four luma cannot see these
    /// defects, so retain that fixture below and add discriminating samples.
    #[test]
    fn quarter_turns_preserve_low_ten_bit_codes_and_asymmetric_chroma_pairs() {
        for format in [Yuv420Format::Nv12Video, Yuv420Format::P010Video] {
            let word = format.bytes_per_sample();
            let encode = |codes: Vec<u16>| -> Vec<u8> {
                codes
                    .into_iter()
                    .flat_map(|code| {
                        if word == 1 {
                            vec![code as u8]
                        } else {
                            (code << 6).to_le_bytes().to_vec()
                        }
                    })
                    .collect()
            };
            let ceiling = if word == 1 { 256 } else { 1024 };
            let y = encode((0..24).map(|i| ((i * 53 + 3) % ceiling) as u16).collect());
            let uv = encode(
                (0..6)
                    .flat_map(|i| {
                        [
                            ((i * 97 + 11) % ceiling) as u16,
                            ((i * 37 + 173) % ceiling) as u16,
                        ]
                    })
                    .collect(),
            );
            let original = Yuv420Frame::new(4, 6, format, y, uv).unwrap();
            let rotated = original.rotated(1).unwrap();
            assert_eq!((rotated.width, rotated.height), (6, 4));
            // The top-right output chroma pair is the original top-left pair.
            assert_eq!(&rotated.uv[4 * word..6 * word], &original.uv[..2 * word]);
            assert_eq!(rotated.rotated(3).unwrap(), original);
        }
    }
    /// Kill: quantize the original BGRA/NV12/P010 gradient, swap chroma bytes, or lose HLG/BT2020/timing on either turn.
    #[test]
    fn the_original_portrait_formats_round_trip_exact_samples_and_hdr_metadata() {
        for buffer in buffers() {
            let source = frame(buffer.clone());
            let prepared = source.rotated_clockwise(1).unwrap();
            assert_eq!((prepared.width, prepared.height), (48, 32));
            assert_eq!(prepared.color, source.color);
            let restored = prepared.rotated_clockwise(3).unwrap();
            assert_eq!((restored.width, restored.height), (32, 48));
            assert_eq!(restored.color, source.color);
            assert_eq!(restored.pts, source.pts);
            assert_eq!(restored.duration, source.duration);
            assert_same(&restored.buffer, &buffer);
            #[cfg(target_os = "macos")]
            {
                let native = match &buffer {
                    FrameBuffer::Cpu(cpu) => MacPixelBuffer::from_cpu_bgra(cpu).unwrap(),
                    FrameBuffer::Yuv420(cpu) => cpu.to_macos().unwrap(),
                    _ => unreachable!(),
                };
                use objc2_core_video::*;
                // SAFETY: these framework-owned CFString constants live for the process.
                let attachments = unsafe {
                    [
                        (
                            kCVImageBufferTransferFunctionKey,
                            kCVImageBufferTransferFunction_ITU_R_2100_HLG,
                        ),
                        (
                            kCVImageBufferYCbCrMatrixKey,
                            kCVImageBufferYCbCrMatrix_ITU_R_2020,
                        ),
                        (
                            kCVImageBufferColorPrimariesKey,
                            kCVImageBufferColorPrimaries_ITU_R_2020,
                        ),
                    ]
                };
                for (key, value) in attachments {
                    // SAFETY: each documented color key takes the paired CFString value.
                    unsafe {
                        native.as_cv_pixel_buffer().set_attachment(
                            key,
                            value,
                            CVAttachmentMode::ShouldPropagate,
                        );
                    }
                }
                let assert_attachments = |frame: &VideoFrame| {
                    let FrameBuffer::MacOS(native) = &frame.buffer else {
                        panic!("lost native color attachments");
                    };
                    for (key, value) in attachments {
                        // SAFETY: no attachment-mode output is requested; the copy owns its CFType.
                        let actual = unsafe {
                            native
                                .as_cv_pixel_buffer()
                                .attachment(key, std::ptr::null_mut())
                        }
                        .unwrap();
                        assert!(objc2_core_foundation::CFEqual(Some(&actual), Some(value)));
                    }
                };
                let source = frame(FrameBuffer::MacOS(native));
                let prepared = source.rotated_clockwise(1).unwrap();
                assert_attachments(&prepared);
                let restored = prepared.rotated_clockwise(3).unwrap();
                assert_attachments(&restored);
                assert_eq!(prepared.color, source.color);
                assert_eq!(restored.color, source.color);
                let FrameBuffer::MacOS(restored) = restored.buffer else {
                    panic!("lost native buffer");
                };
                assert_same(&read_macos(&restored).unwrap(), &buffer);
            }
        }
    }
}
