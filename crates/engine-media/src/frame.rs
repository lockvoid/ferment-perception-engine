use std::sync::Arc;

use crate::{ColorInfo, MediaTime, Result, VideoError};

#[path = "frame_rotation.rs"]
mod rotation;
#[path = "frame_yuv_scale.rs"]
mod yuv_scale;
pub use rotation::{Yuv420Format, Yuv420Frame};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPixelFormat {
    Rgb24,
    Rgba8,
    Bgra8,
    /// Four packed little-endian float32 components; no color/alpha conversion.
    Rgba32Float,
}

impl CpuPixelFormat {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb24 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Rgba32Float => 16,
        }
    }

    /// Copy borrowed rows to tightly packed RGBA, preserving alpha when present.
    /// Size/stride validation precedes allocation; row padding is not copied.
    pub fn pack_rgba8(
        self,
        width: u32,
        height: u32,
        stride: usize,
        data: &[u8],
    ) -> Result<Vec<u8>> {
        if self == Self::Rgba32Float {
            return Err(VideoError::UnsupportedCodec(
                "Float source requires an explicit display conversion".into(),
            ));
        }
        let invalid =
            || VideoError::InvalidData("Invalid CPU frame size, stride or byte length".into());
        let width = usize::try_from(width).map_err(|_| invalid())?;
        let height = usize::try_from(height).map_err(|_| invalid())?;
        let row = width
            .checked_mul(self.bytes_per_pixel())
            .ok_or_else(invalid)?;
        let output_row = width.checked_mul(4).ok_or_else(invalid)?;
        let required = stride.checked_mul(height).ok_or_else(invalid)?;
        let output_bytes = output_row.checked_mul(height).ok_or_else(invalid)?;
        if width == 0 || height == 0 || stride < row || data.len() < required {
            return Err(invalid());
        }
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(output_bytes)
            .map_err(|e| VideoError::InvalidData(format!("CPU frame allocation failed: {e}")))?;
        rgba.resize(output_bytes, 0);
        for (source, target) in data[..required]
            .chunks_exact(stride)
            .zip(rgba.chunks_exact_mut(output_row))
        {
            let source = &source[..row];
            match self {
                Self::Rgba32Float => unreachable!("float output is explicit"),
                Self::Rgba8 => target.copy_from_slice(source),
                Self::Bgra8 => {
                    for (pixel, target) in source
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(target.as_chunks_mut::<4>().0)
                    {
                        *target = [pixel[2], pixel[1], pixel[0], pixel[3]];
                    }
                }
                Self::Rgb24 => {
                    for (pixel, target) in source
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .zip(target.as_chunks_mut::<4>().0)
                    {
                        *target = [pixel[0], pixel[1], pixel[2], 255];
                    }
                }
            }
        }
        Ok(rgba)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: CpuPixelFormat,
    pub data: Arc<[u8]>,
}

impl CpuFrame {
    /// Checked packed copy for CPU consumers; decoder storage remains immutable.
    pub fn packed_rgba8(&self) -> Result<Vec<u8>> {
        self.format
            .pack_rgba8(self.width, self.height, self.stride, &self.data)
    }

    /// Preserve source bits while discarding only row padding.
    pub fn packed_rgba32f(&self) -> Result<Vec<[f32; 4]>> {
        if self.format != CpuPixelFormat::Rgba32Float {
            return Err(VideoError::InvalidData(
                "CPU frame is not float32 RGBA".into(),
            ));
        }
        let invalid = || VideoError::InvalidData("Invalid float CPU frame geometry/stride".into());
        let count = (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or_else(invalid)?;
        let row_bytes = (self.width as usize).checked_mul(16).ok_or_else(invalid)?;
        let required = self
            .stride
            .checked_mul(self.height as usize)
            .ok_or_else(invalid)?;
        if count == 0 || self.stride < row_bytes || self.data.len() < required {
            return Err(invalid());
        }
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(count)
            .map_err(|e| VideoError::InvalidData(format!("Float frame allocation failed: {e}")))?;
        for row in self
            .data
            .chunks_exact(self.stride)
            .take(self.height as usize)
        {
            for bytes in row[..self.width as usize * 16].as_chunks::<16>().0 {
                pixels.push(std::array::from_fn(|c| {
                    f32::from_le_bytes(bytes[c * 4..c * 4 + 4].try_into().expect("four bytes"))
                }));
            }
        }
        Ok(pixels)
    }

    pub fn new(
        width: u32,
        height: u32,
        stride: usize,
        format: CpuPixelFormat,
        data: impl Into<Arc<[u8]>>,
    ) -> Result<Self> {
        let data = data.into();
        let row_bytes = usize::try_from(width)
            .unwrap_or(usize::MAX)
            .saturating_mul(format.bytes_per_pixel());
        let required = stride.saturating_mul(usize::try_from(height).unwrap_or(usize::MAX));
        if stride < row_bytes || data.len() < required {
            return Err(VideoError::InvalidData(format!(
                "CPU frame buffer is too small: {} bytes for {width}x{height}, stride {stride}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            stride,
            format,
            data,
        })
    }

    /// Tightly-packed BGRA copy — the layout platform encoders and pixel
    /// buffer allocators consume. Alpha is forced opaque.
    pub fn to_bgra8(&self) -> Self {
        if self.format == CpuPixelFormat::Bgra8 && self.stride == self.width as usize * 4 {
            return self.clone();
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let mut bgra = vec![0u8; width.saturating_mul(height).saturating_mul(4)];
        for y in 0..height {
            let source = &self.data[y * self.stride..];
            let destination = &mut bgra[y * width * 4..(y + 1) * width * 4];
            match self.format {
                CpuPixelFormat::Rgba32Float => {
                    for (pixel, target) in source
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<4>().0)
                        .take(width)
                    {
                        *target = [2, 1, 0, 3].map(|c| {
                            (f32::from_le_bytes(
                                pixel[c * 4..c * 4 + 4].try_into().expect("four bytes"),
                            )
                            .clamp(0., 1.)
                                * 255.)
                                .round() as u8
                        });
                        target[3] = 255;
                    }
                }
                CpuPixelFormat::Bgra8 => destination.copy_from_slice(&source[..width * 4]),
                CpuPixelFormat::Rgba8 => {
                    for (source, destination) in source
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<4>().0.iter_mut())
                        .take(width)
                    {
                        destination.copy_from_slice(&[source[2], source[1], source[0], 0xff]);
                    }
                }
                CpuPixelFormat::Rgb24 => {
                    for (source, destination) in source
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<4>().0.iter_mut())
                        .take(width)
                    {
                        destination.copy_from_slice(&[source[2], source[1], source[0], 0xff]);
                    }
                }
            }
        }
        Self {
            width: self.width,
            height: self.height,
            stride: width * 4,
            format: CpuPixelFormat::Bgra8,
            data: bgra.into(),
        }
    }

    /// Bilinear resample preserving the pixel format. Intended for encode
    /// pipelines (proxy/AI variants, thumbnails); quality-critical scaling
    /// (viewer zoom, ML input) stays on the GPU/compositor path.
    pub fn scaled_to(&self, width: u32, height: u32) -> Result<Self> {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return Ok(self.clone());
        }
        if self.format == CpuPixelFormat::Rgba32Float {
            return Err(VideoError::UnsupportedCodec(
                "Resize float sources through an explicit linear image operation".into(),
            ));
        }
        let channels = self.format.bytes_per_pixel();
        let source_width = self.width as usize;
        let source_height = self.height as usize;
        if source_width == 0 || source_height == 0 {
            return Err(VideoError::InvalidData(
                "cannot scale an empty frame".into(),
            ));
        }
        let target_width = width as usize;
        let target_height = height as usize;
        let mut output = vec![0u8; target_width * target_height * channels];
        let x_ratio = source_width as f32 / target_width as f32;
        let y_ratio = source_height as f32 / target_height as f32;
        for y in 0..target_height {
            let source_y = ((y as f32 + 0.5) * y_ratio - 0.5).max(0.0);
            let y0 = (source_y as usize).min(source_height - 1);
            let y1 = (y0 + 1).min(source_height - 1);
            let y_weight = source_y - y0 as f32;
            let row0 = &self.data[y0 * self.stride..];
            let row1 = &self.data[y1 * self.stride..];
            let destination = &mut output[y * target_width * channels..];
            for x in 0..target_width {
                let source_x = ((x as f32 + 0.5) * x_ratio - 0.5).max(0.0);
                let x0 = (source_x as usize).min(source_width - 1);
                let x1 = (x0 + 1).min(source_width - 1);
                let x_weight = source_x - x0 as f32;
                for channel in 0..channels {
                    let top = f32::from(row0[x0 * channels + channel]) * (1.0 - x_weight)
                        + f32::from(row0[x1 * channels + channel]) * x_weight;
                    let bottom = f32::from(row1[x0 * channels + channel]) * (1.0 - x_weight)
                        + f32::from(row1[x1 * channels + channel]) * x_weight;
                    destination[x * channels + channel] =
                        (top * (1.0 - y_weight) + bottom * y_weight).round() as u8;
                }
            }
        }
        Self::new(width, height, target_width * channels, self.format, output)
    }

    pub fn to_rgb24(&self) -> Self {
        if self.format == CpuPixelFormat::Rgb24 && self.stride == self.width as usize * 3 {
            return self.clone();
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let mut rgb = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
        for y in 0..height {
            let source = &self.data[y * self.stride..];
            let destination = &mut rgb[y * width * 3..(y + 1) * width * 3];
            match self.format {
                CpuPixelFormat::Rgba32Float => {
                    for (pixel, target) in source
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<3>().0)
                        .take(width)
                    {
                        *target = [0, 1, 2].map(|c| {
                            (f32::from_le_bytes(
                                pixel[c * 4..c * 4 + 4].try_into().expect("four bytes"),
                            )
                            .clamp(0., 1.)
                                * 255.)
                                .round() as u8
                        });
                    }
                }
                CpuPixelFormat::Rgb24 => destination.copy_from_slice(&source[..width * 3]),
                CpuPixelFormat::Rgba8 => {
                    for (source, destination) in source
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<3>().0.iter_mut())
                        .take(width)
                    {
                        destination.copy_from_slice(&source[..3]);
                    }
                }
                CpuPixelFormat::Bgra8 => {
                    for (source, destination) in source
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(destination.as_chunks_mut::<3>().0.iter_mut())
                        .take(width)
                    {
                        destination.copy_from_slice(&[source[2], source[1], source[0]]);
                    }
                }
            }
        }
        Self {
            width: self.width,
            height: self.height,
            stride: width * 3,
            format: CpuPixelFormat::Rgb24,
            data: rgb.into(),
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct MacPixelBuffer {
    inner: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

// SAFETY: a CVPixelBuffer is a CoreFoundation object whose retain and release
// are thread-safe; VideoToolbox hands it over from its own callback thread. A
// decoded picture is not written after that (`from_cpu_bgra` fills a new one
// before anyone else holds it), and CPU readers take its base-address lock.
#[cfg(target_os = "macos")]
unsafe impl Send for MacPixelBuffer {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MacPixelBuffer {}

/// An IOSurface-backed `CVPixelBuffer` of one size and format, with default
/// surface properties — what VideoToolbox wants at both its inputs and what
/// the pixel-transfer scaler writes into.
#[cfg(target_os = "macos")]
pub(crate) fn allocate_pixel_buffer(
    width: u32,
    height: u32,
    format: u32,
) -> Result<objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>> {
    use std::ptr::NonNull;

    use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
    use objc2_core_video::{
        CVPixelBuffer, CVPixelBufferCreate, kCVPixelBufferIOSurfacePropertiesKey,
    };

    if width == 0 || height == 0 {
        return Err(VideoError::InvalidData(
            "cannot allocate a pixel buffer for an empty frame".into(),
        ));
    }
    // Request IOSurface backing with default properties (empty dict). The
    // inner dictionary's key/value types are immaterial (it stays empty);
    // concrete `CFString` types satisfy the `Type + Hash` bounds.
    let io_surface_properties: CFRetained<CFDictionary<CFString, CFString>> = CFDictionary::empty();
    // SAFETY: the CoreVideo key statics are immutable, initialized strings.
    let io_surface_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
    let attributes = CFDictionary::from_slices(&[io_surface_key], &[&*io_surface_properties]);
    let mut raw: *mut CVPixelBuffer = std::ptr::null_mut();
    // SAFETY: `attributes` outlives the call; `raw` is a valid out-pointer.
    let status = unsafe {
        CVPixelBufferCreate(
            None,
            width as usize,
            height as usize,
            format,
            Some(attributes.as_opaque()),
            NonNull::from(&mut raw),
        )
    };
    if status != 0 || raw.is_null() {
        return Err(VideoError::platform("CVPixelBufferCreate", status));
    }
    // SAFETY: CVPixelBufferCreate returns a +1 retained buffer; adopt it.
    Ok(unsafe { CFRetained::from_raw(NonNull::new_unchecked(raw)) })
}

/// The format a scaled ENCODER input takes: a 4:2:0 biplanar source keeps
/// its own layout and bit depth (a 10-bit HDR frame stays 10-bit for the
/// Main10 encoder); anything else lands as 8-bit NV12, which every
/// VideoToolbox encoder ingests without an internal conversion.
#[cfg(target_os = "macos")]
fn encoder_input_format(source: &objc2_core_video::CVPixelBuffer) -> u32 {
    use objc2_core_video::{
        CVPixelBufferGetPixelFormatType, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
    };
    let format = CVPixelBufferGetPixelFormatType(source);
    if [
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
    ]
    .contains(&format)
    {
        format
    } else {
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    }
}

#[cfg(target_os = "macos")]
impl MacPixelBuffer {
    pub(crate) fn new(
        inner: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
    ) -> Self {
        Self { inner }
    }

    /// Borrow the retained native buffer for IOSurface/Metal interop.
    pub fn as_cv_pixel_buffer(&self) -> &objc2_core_video::CVPixelBuffer {
        &self.inner
    }

    /// Allocate an IOSurface-backed 32BGRA `CVPixelBuffer` and copy a CPU frame
    /// into it. This is the macOS pixel-buffer allocator the export/encode path
    /// needs: VideoToolbox requires a `CVPixelBuffer` at its input, and the
    /// IOSurface backing lets the buffer round-trip zero-copy into the hardware
    /// encoder (and, later, be rendered into by wgpu on the no-readback export
    /// path). The source is converted to tightly-packed BGRA first.
    pub fn from_cpu_bgra(source: &CpuFrame) -> Result<Self> {
        use objc2_core_video::{
            CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferLockBaseAddress,
            CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelFormatType_32BGRA,
        };

        let bgra = source.to_bgra8();
        let width = bgra.width;
        let height = bgra.height;
        let row_bytes = width as usize * 4;
        let inner = allocate_pixel_buffer(width, height, kCVPixelFormatType_32BGRA)?;
        let buffer = inner.as_ref();

        let flags = CVPixelBufferLockFlags::empty();
        // SAFETY: `buffer` is retained for the lock interval; the write-lock is
        // released by the guard before `inner` is returned.
        let lock_status = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
        if lock_status != 0 {
            return Err(VideoError::platform(
                "CVPixelBufferLockBaseAddress",
                lock_status,
            ));
        }
        struct Unlock<'a> {
            buffer: &'a objc2_core_video::CVPixelBuffer,
            flags: CVPixelBufferLockFlags,
        }
        impl Drop for Unlock<'_> {
            fn drop(&mut self) {
                // SAFETY: Paired with the successful lock above.
                let _ = unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
            }
        }
        let _unlock = Unlock { buffer, flags };

        let destination_stride = CVPixelBufferGetBytesPerRow(buffer);
        let base = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
        if base.is_null() {
            return Err(VideoError::InvalidData(
                "allocated CVPixelBuffer has no base address".into(),
            ));
        }
        if destination_stride < row_bytes {
            return Err(VideoError::InvalidData(format!(
                "CVPixelBuffer stride {destination_stride} is smaller than {row_bytes}"
            )));
        }
        // SAFETY: CoreVideo guarantees `destination_stride * height` writable
        // bytes while the non-planar buffer is locked; each source row copies
        // exactly `row_bytes` into the leading bytes of the destination row.
        unsafe {
            for row in 0..height as usize {
                let source_row = &bgra.data[row * bgra.stride..row * bgra.stride + row_bytes];
                let destination = base.add(row * destination_stride);
                std::ptr::copy_nonoverlapping(source_row.as_ptr(), destination, row_bytes);
            }
        }

        drop(_unlock);
        Ok(Self { inner })
    }

    /// Explicit readback boundary. The native retained pixel buffer stays
    /// alive through the read lock; the returned CPU bytes own their storage.
    pub(crate) fn copy_bgra8(&self) -> Result<CpuFrame> {
        use objc2_core_video::{
            CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
            CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockFlags,
            kCVPixelFormatType_32BGRA,
        };
        let original = self.as_cv_pixel_buffer();
        let converted;
        let buffer = if CVPixelBufferGetPixelFormatType(original) == kCVPixelFormatType_32BGRA {
            original
        } else {
            converted = crate::platform::macos::transfer_scaled(
                original,
                CVPixelBufferGetWidth(original) as u32,
                CVPixelBufferGetHeight(original) as u32,
                kCVPixelFormatType_32BGRA,
            )?;
            converted.as_ref()
        };
        if CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_32BGRA {
            return Err(VideoError::InvalidData(
                "CPU decode transfer did not return BGRA".into(),
            ));
        }
        let flags = CVPixelBufferLockFlags::ReadOnly;
        // SAFETY: retained buffer outlives the lock and all reads below.
        let status = unsafe { objc2_core_video::CVPixelBufferLockBaseAddress(buffer, flags) };
        if status != 0 {
            return Err(VideoError::platform("CVPixelBufferLockBaseAddress", status));
        }
        struct Unlock<'a>(&'a objc2_core_video::CVPixelBuffer, CVPixelBufferLockFlags);
        impl Drop for Unlock<'_> {
            fn drop(&mut self) {
                // SAFETY: paired with the successful read lock above.
                let _ = unsafe { objc2_core_video::CVPixelBufferUnlockBaseAddress(self.0, self.1) };
            }
        }
        let _unlock = Unlock(buffer, flags);
        let width = CVPixelBufferGetWidth(buffer);
        let height = CVPixelBufferGetHeight(buffer);
        let stride = CVPixelBufferGetBytesPerRow(buffer);
        let pointer = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
        let length = stride
            .checked_mul(height)
            .filter(|length| *length <= isize::MAX as usize)
            .ok_or_else(|| VideoError::InvalidData("CPU pixel buffer length overflow".into()))?;
        if pointer.is_null()
            || width == 0
            || height == 0
            || width > u32::MAX as usize
            || height > u32::MAX as usize
            || width.checked_mul(4).is_none_or(|row| stride < row)
        {
            return Err(VideoError::InvalidData(
                "Invalid BGRA pixel buffer extent, stride or address".into(),
            ));
        }
        // SAFETY: CoreVideo exposes stride*height readable bytes for this
        // non-planar BGRA buffer until _unlock runs; Arc copies the bytes.
        let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
        CpuFrame::new(
            width as u32,
            height as u32,
            stride,
            CpuPixelFormat::Bgra8,
            Arc::<[u8]>::from(bytes),
        )
    }

    fn to_rgb24(&self, color: ColorInfo) -> Result<CpuFrame> {
        use objc2_core_video::{
            CVPixelBufferGetBaseAddress, CVPixelBufferGetBaseAddressOfPlane,
            CVPixelBufferGetBytesPerRow, CVPixelBufferGetBytesPerRowOfPlane,
            CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount,
            CVPixelBufferGetWidth, CVPixelBufferLockFlags, kCVPixelFormatType_32BGRA,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_422YpCbCr16,
        };

        let buffer = self.as_cv_pixel_buffer();
        let format = CVPixelBufferGetPixelFormatType(buffer);
        if ![
            kCVPixelFormatType_32BGRA,
            kCVPixelFormatType_422YpCbCr16,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        ]
        .contains(&format)
        {
            // Newer VideoToolbox decoders can return compressed or packed IOSurface
            // layouts (&xv0/p420) with no usable CPU planes. Keep native frames
            // native until explicit readback, then convert through the existing
            // hardware transfer session before taking a CPU lock.
            let converted = crate::platform::macos::transfer_scaled(
                buffer,
                CVPixelBufferGetWidth(buffer) as u32,
                CVPixelBufferGetHeight(buffer) as u32,
                kCVPixelFormatType_32BGRA,
            )?;
            return MacPixelBuffer::new(converted).to_rgb24(color);
        }
        let flags = CVPixelBufferLockFlags::ReadOnly;
        // SAFETY: `buffer` is retained for the entire lock interval, and every
        // returned base pointer is only read before the matching unlock.
        let status = unsafe { objc2_core_video::CVPixelBufferLockBaseAddress(buffer, flags) };
        if status != 0 {
            return Err(VideoError::platform("CVPixelBufferLockBaseAddress", status));
        }
        struct Unlock<'a> {
            buffer: &'a objc2_core_video::CVPixelBuffer,
            flags: CVPixelBufferLockFlags,
        }
        impl Drop for Unlock<'_> {
            fn drop(&mut self) {
                // SAFETY: Paired with the successful lock above.
                let _ = unsafe {
                    objc2_core_video::CVPixelBufferUnlockBaseAddress(self.buffer, self.flags)
                };
            }
        }
        let _unlock = Unlock { buffer, flags };

        let width = CVPixelBufferGetWidth(buffer);
        let height = CVPixelBufferGetHeight(buffer);
        let format = CVPixelBufferGetPixelFormatType(buffer);
        if format == kCVPixelFormatType_32BGRA {
            let stride = CVPixelBufferGetBytesPerRow(buffer);
            let pointer = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
            if pointer.is_null() {
                return Err(VideoError::InvalidData(
                    "CVPixelBuffer has no BGRA base address".into(),
                ));
            }
            let length = stride.saturating_mul(height);
            // SAFETY: CoreVideo guarantees `stride * height` readable bytes
            // while a non-planar pixel buffer is locked.
            let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
            return CpuFrame::new(
                width as u32,
                height as u32,
                stride,
                CpuPixelFormat::Bgra8,
                Arc::<[u8]>::from(bytes),
            )
            .map(|frame| frame.to_rgb24());
        }

        if format == kCVPixelFormatType_422YpCbCr16 {
            let stride = CVPixelBufferGetBytesPerRow(buffer);
            let pointer = CVPixelBufferGetBaseAddress(buffer).cast::<u8>();
            if pointer.is_null() || stride < width.div_ceil(2) * 8 {
                return Err(VideoError::InvalidData(
                    "CVPixelBuffer has an invalid v216 base address or stride".into(),
                ));
            }
            let length = stride.saturating_mul(height);
            // SAFETY: CoreVideo guarantees `stride * height` readable bytes
            // while the non-planar pixel buffer is locked.
            let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
            let full_range = color.range == crate::ColorRange::Full;
            let mut rgb = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
            for row in 0..height {
                let source = &bytes[row * stride..(row + 1) * stride];
                for pair in 0..width.div_ceil(2) {
                    let source_offset = pair * 8;
                    // v216 stores little-endian, left-justified Cb Y0 Cr Y1.
                    // Taking the high byte maps video-range 10-bit codes to
                    // the corresponding 8-bit codes without a color shift.
                    let cb = f32::from(
                        u16::from_le_bytes([source[source_offset], source[source_offset + 1]]) >> 8,
                    );
                    let y0 = f32::from(
                        u16::from_le_bytes([source[source_offset + 2], source[source_offset + 3]])
                            >> 8,
                    );
                    let cr = f32::from(
                        u16::from_le_bytes([source[source_offset + 4], source[source_offset + 5]])
                            >> 8,
                    );
                    let y1 = f32::from(
                        u16::from_le_bytes([source[source_offset + 6], source[source_offset + 7]])
                            >> 8,
                    );
                    for (within_pair, y) in [y0, y1].into_iter().enumerate() {
                        let column = pair * 2 + within_pair;
                        if column >= width {
                            break;
                        }
                        let pixel = ycbcr_to_rgb(y, cb, cr, full_range, color);
                        let destination = (row * width + column) * 3;
                        rgb[destination..destination + 3].copy_from_slice(&pixel);
                    }
                }
            }
            return CpuFrame::new(
                width as u32,
                height as u32,
                width * 3,
                CpuPixelFormat::Rgb24,
                rgb,
            );
        }

        // Ten-bit 4:2:0 is what VideoToolbox hands back for HLG/PQ sources,
        // even when the coded stream is 8-bit — so without this case every HDR
        // import fails to decode at all. Samples are 10-bit codes in the top
        // ten bits of little-endian 16-bit words.
        let ten_bit = format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
        if !ten_bit
            && format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
            && format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        {
            return Err(VideoError::UnsupportedCodec(format!(
                "CPU conversion for CV pixel format 0x{format:08x}"
            )));
        }
        if CVPixelBufferGetPlaneCount(buffer) < 2 {
            return Err(VideoError::InvalidData(
                "bi-planar CVPixelBuffer has fewer than two planes".into(),
            ));
        }
        let y_stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
        let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1);
        let y_pointer = CVPixelBufferGetBaseAddressOfPlane(buffer, 0).cast::<u8>();
        let uv_pointer = CVPixelBufferGetBaseAddressOfPlane(buffer, 1).cast::<u8>();
        if y_pointer.is_null() || uv_pointer.is_null() {
            return Err(VideoError::InvalidData(
                "CVPixelBuffer has a null Y or CbCr plane".into(),
            ));
        }
        // SAFETY: Both plane extents are readable while the buffer is locked.
        let y_plane = unsafe { std::slice::from_raw_parts(y_pointer, y_stride * height) };
        let uv_plane =
            unsafe { std::slice::from_raw_parts(uv_pointer, uv_stride * height.div_ceil(2)) };
        let full_range = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            || format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
        let narrowed;
        let planes = if ten_bit {
            narrowed = narrow_ten_bit_planes(y_plane, uv_plane, y_stride, uv_stride, width, height);
            Nv12Planes {
                y: &narrowed.y,
                uv: &narrowed.uv,
                y_stride: width,
                uv_stride: width.div_ceil(2) * 2,
            }
        } else {
            Nv12Planes {
                y: y_plane,
                uv: uv_plane,
                y_stride,
                uv_stride,
            }
        };
        let rgb = nv12_to_rgb(planes, width, height, full_range, color);
        CpuFrame::new(
            width as u32,
            height as u32,
            width * 3,
            CpuPixelFormat::Rgb24,
            rgb,
        )
    }
}

/// Eight-bit copies of a ten-bit bi-planar buffer.
#[cfg(target_os = "macos")]
struct NarrowedPlanes {
    y: Vec<u8>,
    uv: Vec<u8>,
}

/// Narrow `x420`/`xf20` planes to the 8-bit codes the shared YCbCr math takes.
///
/// The 10-bit code sits in the top ten bits of each little-endian 16-bit word
/// (VideoToolbox writes `code << 6`, the low six bits zero), so the 8-bit
/// equivalent is the code divided by four with rounding.
#[cfg(target_os = "macos")]
fn narrow_ten_bit_planes(
    y_plane: &[u8],
    uv_plane: &[u8],
    y_stride: usize,
    uv_stride: usize,
    width: usize,
    height: usize,
) -> NarrowedPlanes {
    fn narrow(source: &[u8], stride: usize, samples_per_row: usize, rows: usize) -> Vec<u8> {
        let mut output = vec![0u8; samples_per_row * rows];
        for row in 0..rows {
            let line = &source[row * stride..];
            for sample in 0..samples_per_row {
                let word = u16::from_le_bytes([line[sample * 2], line[sample * 2 + 1]]) >> 6;
                output[row * samples_per_row + sample] = ((u32::from(word) + 2) / 4).min(255) as u8;
            }
        }
        output
    }

    let chroma_samples = width.div_ceil(2) * 2;
    NarrowedPlanes {
        y: narrow(y_plane, y_stride, width, height),
        uv: narrow(uv_plane, uv_stride, chroma_samples, height.div_ceil(2)),
    }
}

#[cfg(target_os = "macos")]
struct Nv12Planes<'a> {
    y: &'a [u8],
    uv: &'a [u8],
    y_stride: usize,
    uv_stride: usize,
}

#[cfg(target_os = "macos")]
fn nv12_to_rgb(
    planes: Nv12Planes<'_>,
    width: usize,
    height: usize,
    full_range: bool,
    color: ColorInfo,
) -> Vec<u8> {
    let mut output = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
    for row in 0..height {
        for column in 0..width {
            let y_code = f32::from(planes.y[row * planes.y_stride + column]);
            let uv_offset = (row / 2) * planes.uv_stride + (column / 2) * 2;
            let cb_code = f32::from(planes.uv[uv_offset]);
            let cr_code = f32::from(planes.uv[uv_offset + 1]);
            let offset = (row * width + column) * 3;
            output[offset..offset + 3]
                .copy_from_slice(&ycbcr_to_rgb(y_code, cb_code, cr_code, full_range, color));
        }
    }
    output
}

pub(crate) fn ycbcr_to_rgb(
    y_code: f32,
    cb_code: f32,
    cr_code: f32,
    full_range: bool,
    color: ColorInfo,
) -> [u8; 3] {
    use crate::ColorMatrix;

    let (cr_r, cb_g, cr_g, cb_b) = match color.matrix.for_rgb() {
        ColorMatrix::Bt709 | ColorMatrix::Bt2020Constant | ColorMatrix::Bt2020NonConstant => {
            (1.5748, -0.187_324, -0.468_124, 1.8556)
        }
        _ => (1.402, -0.344_136, -0.714_136, 1.772),
    };
    let (y, cb, cr) = if full_range {
        (y_code, cb_code - 128.0, cr_code - 128.0)
    } else {
        (
            (y_code - 16.0) * (255.0 / 219.0),
            (cb_code - 128.0) * (255.0 / 224.0),
            (cr_code - 128.0) * (255.0 / 224.0),
        )
    };
    [
        (y + cr_r * cr).round().clamp(0.0, 255.0) as u8,
        (y + cb_g * cb + cr_g * cr).round().clamp(0.0, 255.0) as u8,
        (y + cb_b * cb).round().clamp(0.0, 255.0) as u8,
    ]
}

#[derive(Debug, Clone)]
pub enum FrameBuffer {
    Cpu(CpuFrame),
    Yuv420(Yuv420Frame),
    /// A software decoder's own Y'CbCr codes, for a GPU to convert
    /// (`DecodeOutput::CpuPlanes`).
    Ycbcr(crate::YcbcrFrame),
    #[cfg(target_os = "macos")]
    MacOS(MacPixelBuffer),
    /// A decoded picture still on the phone's graphics memory: the `AImage`
    /// MediaCodec rendered into an `AImageReader`, holding an
    /// `AHardwareBuffer`. The analogue of [`MacPixelBuffer`], and the reason
    /// an Android decode does not pay a full-resolution readback per frame.
    #[cfg(target_os = "android")]
    Android(crate::platform::android::AndroidImage),
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub pts: MediaTime,
    pub duration: Option<MediaTime>,
    pub color: ColorInfo,
    pub buffer: FrameBuffer,
}

impl VideoFrame {
    /// Explicit CPU download/conversion. Platform decode does not invoke this
    /// method in the hot path.
    pub fn to_cpu_rgb(&self) -> Result<CpuFrame> {
        match &self.buffer {
            FrameBuffer::Cpu(frame) => Ok(frame.to_rgb24()),
            FrameBuffer::Yuv420(frame) => frame.to_rgb24(self.color),
            FrameBuffer::Ycbcr(frame) => frame.to_rgb24(self.color),
            #[cfg(target_os = "macos")]
            FrameBuffer::MacOS(buffer) => buffer.to_rgb24(self.color),
            #[cfg(target_os = "android")]
            FrameBuffer::Android(image) => image.to_rgb24(self.color),
        }
    }

    /// Build a platform-native, encoder-ready frame from a CPU buffer at the
    /// requested timestamps and dimensions. On macOS this allocates an
    /// IOSurface-backed `CVPixelBuffer` (VideoToolbox requires one); on Windows
    /// and other targets the frame stays CPU-resident, since the Media
    /// Foundation encoder converts CPU RGB to NV12 and pure-Rust paths keep
    /// bytes on the CPU. `source` is resampled to `width`×`height` first when
    /// they differ.
    pub fn encodable_from_cpu(
        source: &CpuFrame,
        width: u32,
        height: u32,
        pts: MediaTime,
        duration: Option<MediaTime>,
        color: ColorInfo,
    ) -> Result<Self> {
        let scaled = source.scaled_to(width, height)?;
        #[cfg(target_os = "macos")]
        let buffer = FrameBuffer::MacOS(MacPixelBuffer::from_cpu_bgra(&scaled)?);
        #[cfg(not(target_os = "macos"))]
        let buffer = FrameBuffer::Cpu(scaled);
        Ok(Self {
            width,
            height,
            pts,
            duration,
            color,
            buffer,
        })
    }

    /// This frame as an encoder input of `width`×`height`, timestamps kept.
    ///
    /// On macOS a decoded frame never leaves the GPU: a frame already at
    /// size is handed over as it is (the encoder retains the decoder's
    /// buffer), any other size goes through the VideoToolbox pixel-transfer
    /// scaler into a fresh IOSurface. Only a CPU-resident frame (or another
    /// platform) takes the CPU route: RGB download, resample, upload.
    /// `color` is what the CPU route stamps on the frame; the native route
    /// keeps the source's, which the encoder ignores in favour of its own
    /// session settings.
    pub fn encodable_scaled(&self, width: u32, height: u32, color: ColorInfo) -> Result<Self> {
        #[cfg(target_os = "macos")]
        if let FrameBuffer::Yuv420(buffer) = &self.buffer {
            let native = Self {
                buffer: FrameBuffer::MacOS(buffer.to_macos()?),
                ..self.clone()
            };
            return native.encodable_scaled(width, height, color);
        }
        #[cfg(target_os = "macos")]
        if let FrameBuffer::MacOS(buffer) = &self.buffer {
            let scaled = if self.width == width && self.height == height {
                buffer.clone()
            } else {
                let format = encoder_input_format(buffer.as_cv_pixel_buffer());
                MacPixelBuffer::new(crate::platform::macos::transfer_scaled(
                    buffer.as_cv_pixel_buffer(),
                    width,
                    height,
                    format,
                )?)
            };
            return Ok(Self {
                width,
                height,
                pts: self.pts,
                duration: self.duration,
                color: self.color,
                buffer: FrameBuffer::MacOS(scaled),
            });
        }
        // Preserve original samples through Android Main10 encoding.
        #[cfg(not(target_os = "macos"))]
        if color.bits_per_channel.is_some_and(|bits| bits > 8) {
            return self.encodable_yuv420(width, height, color);
        }
        // Native decoder tone mapping is optional. If unavailable, transform
        // original samples in floating point before any8-bit conversion.
        #[cfg(target_os = "android")]
        if matches!(
            self.color.transfer,
            crate::TransferCharacteristic::Hlg | crate::TransferCharacteristic::Pq
        ) && self.color.transfer != color.transfer
        {
            return self.encodable_sdr709(width, height, color);
        }
        let cpu = self.to_cpu_rgb()?;
        Self::encodable_from_cpu(&cpu, width, height, self.pts, self.duration, color)
    }

    /// A small RGB copy of this frame at `width`×`height` — thumbnails,
    /// analysis inputs. On macOS the scale happens in the pixel-transfer
    /// session and only the small picture is read back; elsewhere the full
    /// frame is downloaded and resampled on the CPU.
    pub fn scaled_cpu_rgb(&self, width: u32, height: u32) -> Result<CpuFrame> {
        #[cfg(target_os = "macos")]
        if let FrameBuffer::MacOS(buffer) = &self.buffer
            && (self.width != width || self.height != height)
        {
            let small = MacPixelBuffer::new(crate::platform::macos::transfer_scaled(
                buffer.as_cv_pixel_buffer(),
                width,
                height,
                objc2_core_video::kCVPixelFormatType_32BGRA,
            )?);
            return small.to_rgb24(self.color);
        }
        let cpu = self.to_cpu_rgb()?;
        if cpu.width == width && cpu.height == height {
            return Ok(cpu);
        }
        cpu.scaled_to(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeBase;

    fn rgb_frame() -> CpuFrame {
        // 2x2 RGB: red, green / blue, white.
        CpuFrame::new(
            2,
            2,
            6,
            CpuPixelFormat::Rgb24,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        )
        .unwrap()
    }

    #[test]
    fn bgra_conversion_swaps_channels_and_forces_opaque_alpha() {
        let bgra = rgb_frame().to_bgra8();
        assert_eq!(bgra.format, CpuPixelFormat::Bgra8);
        assert_eq!(bgra.stride, 8);
        assert_eq!(&bgra.data[..8], &[0, 0, 255, 255, 0, 255, 0, 255]);
        // Round-trip through the existing RGB path preserves pixels.
        assert_eq!(bgra.to_rgb24().data, rgb_frame().to_rgb24().data);
    }

    #[test]
    fn bilinear_scale_preserves_solid_colors_and_dimensions() {
        let solid = CpuFrame::new(
            4,
            4,
            16,
            CpuPixelFormat::Bgra8,
            vec![10u8; 4 * 4 * 4].into_iter().collect::<Vec<_>>(),
        )
        .unwrap();
        let scaled = solid.scaled_to(2, 2).unwrap();
        assert_eq!((scaled.width, scaled.height), (2, 2));
        assert!(scaled.data.iter().all(|byte| *byte == 10));
        // Identity scale is a clone, not a resample.
        let identity = solid.scaled_to(4, 4).unwrap();
        assert_eq!(identity.data, solid.data);
    }

    #[test]
    fn scaling_up_interpolates_between_neighbors() {
        // 1x2 grayscale-ish gradient in BGRA: 0 and 200.
        let frame = CpuFrame::new(
            2,
            1,
            8,
            CpuPixelFormat::Bgra8,
            vec![0, 0, 0, 255, 200, 200, 200, 255],
        )
        .unwrap();
        let scaled = frame.scaled_to(4, 1).unwrap();
        let blue: Vec<u8> = scaled.data.chunks_exact(4).map(|px| px[0]).collect();
        assert_eq!(blue.first().copied(), Some(0));
        assert_eq!(blue.last().copied(), Some(200));
        assert!(
            blue[1] > 0 && blue[1] < 200,
            "interior interpolates: {blue:?}"
        );
        assert!(
            blue.windows(2).all(|pair| pair[0] <= pair[1]),
            "monotonic: {blue:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mac_pixel_buffer_allocator_round_trips_cpu_pixels() {
        let source = rgb_frame();
        let buffer = MacPixelBuffer::from_cpu_bgra(&source).unwrap();
        let read_back = buffer.to_rgb24(ColorInfo::default()).unwrap();
        assert_eq!(read_back.width, 2);
        assert_eq!(read_back.height, 2);
        assert_eq!(read_back.data, source.to_rgb24().data);
    }

    #[test]
    fn encodable_frames_scale_and_land_on_the_platform_buffer() {
        let source = rgb_frame();
        let frame = VideoFrame::encodable_from_cpu(
            &source,
            4,
            4,
            MediaTime::new(0, TimeBase::new_unchecked(1, 600)),
            None,
            ColorInfo::default(),
        )
        .unwrap();
        assert_eq!((frame.width, frame.height), (4, 4));
        #[cfg(target_os = "macos")]
        assert!(matches!(frame.buffer, FrameBuffer::MacOS(_)));
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(frame.buffer, FrameBuffer::Cpu(_)));
    }

    /// The ten-bit code sits in the TOP ten bits of each little-endian word,
    /// as VideoToolbox writes `x420` (`libav_decode`'s GPU-planes test reads
    /// it from a real decode). Reading the low ten bits takes padding and the
    /// code's last four bits; truncating instead of rounding is a level off.
    #[cfg(target_os = "macos")]
    #[test]
    fn ten_bit_planes_narrow_the_code_in_the_top_ten_bits_by_four_with_rounding() {
        // A 2x4 frame: four luma rows of two samples, two interleaved chroma
        // rows. Each row carries a trailing word beyond the used samples, so a
        // stride that is ignored shows up as shifted output.
        fn plane(rows: &[[u16; 3]]) -> Vec<u8> {
            rows.iter()
                .flat_map(|row| {
                    let [first, second, trailing] = *row;
                    [first << 6, second << 6, trailing]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                })
                .collect()
        }
        let luma = plane(&[
            [0, 4, 0xDEAD],
            [511, 1023, 0xDEAD],
            [64, 940, 0xDEAD],
            [512, 256, 0xDEAD],
        ]);
        let chroma = plane(&[[512, 512, 0xBEEF], [64, 940, 0xBEEF]]);
        let narrowed = narrow_ten_bit_planes(&luma, &chroma, 6, 6, 2, 4);
        assert_eq!(narrowed.y, vec![0, 1, 128, 255, 16, 235, 128, 64]);
        assert_eq!(narrowed.uv, vec![128, 128, 16, 235]);
    }

    /// The six bits below the code belong to no sample and are ignored.
    #[cfg(target_os = "macos")]
    #[test]
    fn ten_bit_narrowing_ignores_the_padding_below_the_code() {
        let luma: Vec<u8> = [(500u16 << 6) | 0x3F, 0xFFFF]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let chroma: Vec<u8> = [512u16 << 6, 512 << 6]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let narrowed = narrow_ten_bit_planes(&luma, &chroma, 4, 4, 2, 1);
        assert_eq!(narrowed.y, vec![125, 255]);
    }
}
