//! The hardware decode path: a device context, and the negotiation that picks
//! it.
//!
//! `libavcodec` will decode in software without being asked twice, so the only
//! thing standing between us and a CPU-bound decode is `get_format` — the
//! callback libavcodec invokes with the pixel formats a stream can be decoded
//! into. Returning the hardware format there is what keeps frames on the GPU;
//! returning anything else silently costs a full-resolution readback per frame.
//!
//! This is also where a chip's limits show up honestly. An M1 has no AV1
//! decoder; `avcodec_get_hw_config` simply will not offer `AV_PIX_FMT_*` for
//! it, and the software path takes over with a debug line rather than an error.

use std::ffi::c_void;

use ffmpeg_next::ffi;

#[cfg(not(target_os = "macos"))]
use super::check_av;
use crate::Result;

/// The hardware devices this platform can decode with, in preference order.
///
/// Windows tries CUDA before D3D11VA deliberately: the shipping target is
/// NVIDIA, `libavcodec/nvdec.c` pipelines properly (no per-frame wait, unlike
/// the VideoToolbox hwaccel), and D3D11VA remains the answer on an Intel or
/// AMD machine. Both are opened by dlopen — no NVIDIA library is linked, and
/// `ffnvcodec` is MIT-licensed headers only, so the LGPL posture is unchanged.
///
/// The first entry that BOTH opens on this machine and has a config for the
/// stream's codec wins; a box with no device decodes in software.
#[cfg(target_os = "macos")]
pub(super) const DEVICE_TYPES: &[ffi::AVHWDeviceType] =
    &[ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX];
#[cfg(target_os = "windows")]
pub(super) const DEVICE_TYPES: &[ffi::AVHWDeviceType] = &[
    ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
    ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub(super) const DEVICE_TYPES: &[ffi::AVHWDeviceType] = &[];

/// An `AVBufferRef` holding a hardware device context, freed on drop.
pub(super) struct HwDevice {
    inner: *mut ffi::AVBufferRef,
    kind: ffi::AVHWDeviceType,
    /// The pixel format frames arrive in when this device decodes — what
    /// `get_format` has to return, and what marks a frame as still being on
    /// the device afterwards.
    pixel_format: ffi::AVPixelFormat,
}

// SAFETY: an `AVBufferRef` is reference-counted and internally synchronized;
// the decode session that owns one never shares it across threads anyway.
unsafe impl Send for HwDevice {}

impl HwDevice {
    /// Open the best hardware device for `codec`, or `None` when this machine
    /// has none that can decode it.
    ///
    /// Software decode is not an error — an M1 has no AV1 decoder, and a
    /// Windows box may have no NVIDIA card. It is a debug line and a slower
    /// decode, which is what the caller asked for by opening the file.
    pub(super) fn open(codec: *const ffi::AVCodec) -> Option<Self> {
        if std::env::var_os("FERMENT_LIBAV_SOFTWARE").is_some() {
            log::debug!(
                target: "ferment_engine_media::libav",
                "FERMENT_LIBAV_SOFTWARE is set; skipping the hardware device",
            );
            return None;
        }
        for &kind in DEVICE_TYPES {
            let Some(pixel_format) = Self::pixel_format(codec, kind) else {
                continue;
            };
            let mut inner: *mut ffi::AVBufferRef = std::ptr::null_mut();
            // SAFETY: `inner` is a valid out-pointer; the rest is the
            // documented "no device name, no options" form.
            let status = unsafe {
                ffi::av_hwdevice_ctx_create(
                    &mut inner,
                    kind,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                )
            };
            if status < 0 || inner.is_null() {
                continue;
            }
            log::debug!(
                target: "ferment_engine_media::libav",
                "hardware decode on {kind:?} ({pixel_format:?})",
            );
            return Some(Self {
                inner,
                kind,
                pixel_format,
            });
        }
        log::debug!(
            target: "ferment_engine_media::libav",
            "no hardware decoder for this stream; decoding in software",
        );
        None
    }

    /// The pixel format `kind` decodes `codec` into, if it can at all.
    fn pixel_format(
        codec: *const ffi::AVCodec,
        kind: ffi::AVHWDeviceType,
    ) -> Option<ffi::AVPixelFormat> {
        for index in 0.. {
            // SAFETY: takes any codec pointer, returns null past the end.
            let config = unsafe { ffi::avcodec_get_hw_config(codec, index) };
            if config.is_null() {
                return None;
            }
            // SAFETY: non-null means a valid config for this codec.
            let config = unsafe { &*config };
            let usable = config.methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0;
            if usable && config.device_type == kind {
                return Some(config.pix_fmt);
            }
        }
        None
    }

    /// The format a frame still on the device carries. Only the platforms
    /// that have to download one ask (macOS never does — a VideoToolbox frame
    /// IS the buffer wgpu wants).
    #[cfg(not(target_os = "macos"))]
    pub(super) fn frame_format(&self) -> ffi::AVPixelFormat {
        self.pixel_format
    }

    /// Which device opened — for the log line and for tests.
    #[allow(dead_code)]
    pub(super) fn kind(&self) -> ffi::AVHWDeviceType {
        self.kind
    }

    /// Attach this device to a decoder context, and point its format
    /// negotiation at `hardware_format`.
    ///
    /// # Safety
    ///
    /// `context` must be an allocated, not-yet-opened `AVCodecContext`.
    pub(super) unsafe fn attach(&self, context: *mut ffi::AVCodecContext) -> Result<()> {
        let hardware_format = self.pixel_format;
        // SAFETY: the caller guarantees a live unopened context; `av_buffer_ref`
        // hands it its own reference, which `avcodec_free_context` releases.
        unsafe {
            (*context).hw_device_ctx = ffi::av_buffer_ref(self.inner);
            if (*context).hw_device_ctx.is_null() {
                return Err(crate::VideoError::BackendUnavailable(
                    "could not reference the hardware device context".into(),
                ));
            }
            // `opaque` is how the C callback learns which format we want; it is
            // ours to use and libavcodec never reads it.
            (*context).opaque = format_as_opaque(hardware_format);
            (*context).get_format = Some(select_format);
        }
        Ok(())
    }
}

impl Drop for HwDevice {
    fn drop(&mut self) {
        // SAFETY: `inner` is a reference this type owns; unref it once.
        unsafe { ffi::av_buffer_unref(&mut self.inner) };
    }
}

fn format_as_opaque(format: ffi::AVPixelFormat) -> *mut c_void {
    (format as i64 as isize) as *mut c_void
}

/// libavcodec's format negotiation: take the hardware format when it is
/// offered, and the first software format otherwise.
///
/// # Safety
///
/// Called by libavcodec with its own context and a `AV_PIX_FMT_NONE`-terminated
/// list of candidates.
unsafe extern "C" fn select_format(
    context: *mut ffi::AVCodecContext,
    mut formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: libavcodec passes back the `opaque` we set in `attach`.
    let wanted = unsafe { (*context).opaque } as isize as i64;
    let mut first = ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    // SAFETY: the list is non-null and terminated by AV_PIX_FMT_NONE.
    unsafe {
        while *formats != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if first == ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                first = *formats;
            }
            if (*formats as i64) == wanted {
                return *formats;
            }
            formats = formats.add(1);
        }
    }
    // The chip cannot decode this codec. Software is the answer, and it is a
    // fact about the hardware rather than a failure.
    log::debug!(
        target: "ferment_engine_media::libav",
        "hardware decode unavailable for this stream; falling back to software",
    );
    first
}

/// Copy a hardware frame down to system memory, into `AV_PIX_FMT_NV12`.
///
/// Only Windows needs this: a VideoToolbox frame IS a `CVPixelBuffer` that
/// wgpu imports directly, while a D3D11 texture has no such path yet
/// (ruling 13's named follow-up).
/// # Safety
///
/// `source` must be a live hardware-backed `AVFrame`.
#[cfg(not(target_os = "macos"))]
pub(super) unsafe fn transfer_to_cpu(source: *mut ffi::AVFrame) -> Result<*mut ffi::AVFrame> {
    // SAFETY: `av_frame_alloc` returns an owned frame or null.
    let destination = unsafe { ffi::av_frame_alloc() };
    if destination.is_null() {
        return Err(crate::VideoError::BackendUnavailable(
            "could not allocate a frame for the hardware transfer".into(),
        ));
    }
    // Asking for NV12 explicitly rather than letting libav choose: it is what
    // every D3D11 decoder produces, so the transfer stays a copy instead of
    // becoming a conversion.
    // SAFETY: `destination` is a fresh frame; `source` is a live hardware one.
    unsafe {
        (*destination).format = ffi::AVPixelFormat::AV_PIX_FMT_NV12 as i32;
    }
    // SAFETY: both frames are live; 0 is the documented flags value.
    let status = unsafe { ffi::av_hwframe_transfer_data(destination, source, 0) };
    if let Err(error) = check_av("av_hwframe_transfer_data", status) {
        // SAFETY: `destination` is owned here and not yet handed out.
        unsafe { ffi::av_frame_free(&mut { destination }) };
        return Err(error);
    }
    // Timestamps do not ride along with the pixels.
    // SAFETY: both frames are live and distinct.
    let status = unsafe { ffi::av_frame_copy_props(destination, source) };
    if let Err(error) = check_av("av_frame_copy_props", status) {
        // SAFETY: as above.
        unsafe { ffi::av_frame_free(&mut { destination }) };
        return Err(error);
    }
    Ok(destination)
}
