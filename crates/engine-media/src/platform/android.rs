//! MediaCodec, MediaExtractor and MediaMuxer — the NDK media backend.
//!
//! The phone's own silicon decodes and encodes; `libmediandk.so` is the door.
//! The shape is `macos.rs`'s, because the two platforms answer the same
//! question the same way: a decoded picture stays in graphics memory (there an
//! IOSurface-backed `CVPixelBuffer`, here an `AHardwareBuffer` behind an
//! `AImage`) and only a caller who asks for pixels pays for the download.
//!
//! Three things are deliberately NOT MediaCodec's:
//!
//! * **`probe` is libavformat's**, on this platform as on every other. The
//!   answer a caller gets — `AudioTiming`'s priming and padding, the display
//!   matrix, the ProRes profile, the colour tags — is recovered by 600 lines
//!   of `platform/libav/probe.rs` that no NDK call reproduces, and one probe
//!   means one `MediaInfo` on every platform rather than two that agree until
//!   they do not. `AMediaExtractor` is used for what it IS the authority on:
//!   the samples.
//! * **The libav decoder remains the fallback.** `AMediaCodec` refuses codecs
//!   the device has no decoder for — ProRes on every Android device shipped so
//!   far — and refusing the file is not an answer when six `libav*.so` are
//!   already in the APK. [`AndroidVideoIo`] hands those to [`LibavVideoIo`].
//! * **Decoded timestamps remain observed microseconds.** MediaExtractor and
//!   MediaCodec speak µs; public frame timelines preserve those exact values.
//!   One picture of lookahead recovers VFR durations. Only encoding normalizes
//!   interval endpoints to its caller's explicitly supplied output clock,
//!   avoiding a second downward conversion and a lost terminal duration.
//!
//! What the platform cannot do is refused rather than approximated:
//! [`AndroidMuxer`] writes MPEG-4 only, without `faststart`, and says so.

use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Condvar, Mutex};

use async_trait::async_trait;
use ndk_sys as ndk;

use super::android_codec_selection::{EncoderQuery, encoder_names};
use super::android_decode_fallback::{
    RecoveringDecoder, SoftwareOrientation, decoder_refusal, require_full_buffer_crop,
    require_full_image_crop, require_supported_sink,
};
use super::android_hdr::{self, InputPolicy};
use super::codec_timing::{DecodedTimeline, SubmittedTimings};
use crate::platform::libav::{LibavAudioIo, LibavVideoIo};
use crate::{
    AudioDecodeSession, AudioEncodeSession, AudioEncoderSettings, AudioIo, AudioMuxTrack,
    AudioTiming, Codec, ColorInfo, ColorRange, CpuFrame, CpuPixelFormat, DecodeOptions,
    DecodeSession, DecoderConfiguration, EncodeSession, EncodedAudio, EncodedPacket, EncodedVideo,
    EncoderSettings, FrameBuffer, MediaInfo, MediaKind, MediaTime, MuxFormat, MuxOutput,
    MuxSession, MuxSettings, MuxTrackId, Muxer, PacketFormat, Packetization, PcmAudioFrame, Result,
    StreamInfo, TimeBase, VideoError, VideoFrame, VideoIo, VideoMuxTrack,
};

// -------------------------------------------------------------------------
// The NDK, minimally wrapped

/// `AMEDIA_OK`, or a [`MediaError::Platform`](crate::MediaError::Platform)
/// naming the call. The NDK's media status codes are a flat negative space
/// (`NdkMediaError.h`: -10000 for the codec family, -30000 for the image
/// reader), so the number alone says nothing and the call name is the message.
fn check_media(operation: &'static str, status: ndk::media_status_t) -> Result<()> {
    if status == ndk::media_status_t::AMEDIA_OK {
        return Ok(());
    }
    Err(VideoError::platform(operation, status.0))
}

/// `AIMAGE_FORMAT_YCBCR_P010`.
///
/// Not in NDK 28's `media/NdkImage.h` — `AHARDWAREBUFFER_FORMAT_YCbCr_P010`
/// (`android/hardware_buffer.h:175`) is the same 0x36, and it is what
/// `AImageReader_newWithUsage` takes for a 10-bit reader: the value is the
/// graphics format, and `ImageFormat.YCBCR_P010` on the Java side is 0x36 too.
const AIMAGE_FORMAT_YCBCR_P010: i32 = 0x36;

/// Ten-bit 4:2:0, two bytes a sample, the ten bits LEFT-justified in each
/// little-endian word — the opposite of CoreVideo's `x420`, which
/// right-justifies. Reading one layout with the other's rule is four stops out.
const P010_SAMPLE_SHIFT: u32 = 6;

/// How many decoded pictures the reader may hold at once.
///
/// Four, not one: MediaCodec renders asynchronously, so a reader that can hold
/// a single image forces the decoder to stall behind every acquire. More than a
/// handful only grows the working set — a 4K P010 image is 12 MB.
///
/// This is a hard limit of the platform, and [`DecodeSession`] has no matching
/// one. `Sink::Surface`'s `outstanding` is where the two are reconciled.
const READER_MAX_IMAGES: i32 = 4;

/// `AMediaCodec_dequeueOutputBuffer`'s timeout.
///
/// MediaCodec's dequeue IS the platform's blocking primitive; this bounds it so
/// a wedged codec becomes an error at [`DECODE_STALL_LIMIT`] rather than a
/// process that never returns (FLEET_LAW 5 — a bounded poll with a named
/// reason).
const DEQUEUE_TIMEOUT_US: i64 = 10_000;

/// A rendered picture is asynchronous, not another codec dequeue poll. Give
/// the listener one bounded second before recovering the original source.
const EXPECTED_IMAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

/// Consecutive dequeue timeouts that mean the codec has stopped producing.
/// 500 × 10 ms = 5 s, two orders above the worst frame interval any of this
/// decodes.
///
/// Counted on EVERY timeout, not only after the input is drained: a codec that
/// wedges with buffers still in flight is exactly the case a bound exists for,
/// and gating on `input_done` left that path unbounded with no timeout above
/// it.
const DECODE_STALL_LIMIT: u32 = 500;

/// An owned `AMediaFormat`. Both `AMediaExtractor_getTrackFormat` and
/// `AMediaCodec_getOutputFormat` hand over a format the caller deletes.
struct Format(*mut ndk::AMediaFormat);

impl Drop for Format {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: sole owner of a format the NDK handed over.
            unsafe { ndk::AMediaFormat_delete(self.0) };
        }
    }
}

impl Format {
    fn new() -> Result<Self> {
        // SAFETY: allocates an owned format or returns null.
        let raw = unsafe { ndk::AMediaFormat_new() };
        if raw.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an AMediaFormat".into(),
            ));
        }
        Ok(Self(raw))
    }

    fn adopt(raw: *mut ndk::AMediaFormat, what: &'static str) -> Result<Self> {
        if raw.is_null() {
            return Err(VideoError::BackendUnavailable(format!(
                "{what} returned no format"
            )));
        }
        Ok(Self(raw))
    }

    fn string(&self, key: *const c_char) -> Option<String> {
        let mut value: *const c_char = ptr::null();
        // SAFETY: the format is live; the out-pointer is valid. On success the
        // string belongs to the format and outlives this borrow.
        let found = unsafe { ndk::AMediaFormat_getString(self.0, key, &mut value) };
        if !found || value.is_null() {
            return None;
        }
        // SAFETY: the NDK guarantees a NUL-terminated string on success.
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn int32(&self, key: *const c_char) -> Option<i32> {
        let mut value = 0i32;
        // SAFETY: live format, valid out-pointer.
        unsafe { ndk::AMediaFormat_getInt32(self.0, key, &mut value) }.then_some(value)
    }

    fn buffer(&self, key: *const c_char, max_len: usize) -> Result<Option<Vec<u8>>> {
        let mut data: *mut c_void = ptr::null_mut();
        let mut length = 0usize;
        // SAFETY: live format; successful bytes are owned by it until copied.
        if !unsafe { ndk::AMediaFormat_getBuffer(self.0, key, &mut data, &mut length) } {
            return Ok(None);
        }
        if data.is_null() || length > max_len {
            return Err(VideoError::InvalidData(
                "invalid bounded MediaFormat data".into(),
            ));
        }
        Ok(Some(
            unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) }.to_vec(),
        ))
    }

    fn set_int32(&self, key: *const c_char, value: i32) {
        // SAFETY: live format; the NDK copies the value.
        unsafe { ndk::AMediaFormat_setInt32(self.0, key, value) };
    }

    fn set_float(&self, key: *const c_char, value: f32) {
        // SAFETY: live format; the NDK copies the value.
        unsafe { ndk::AMediaFormat_setFloat(self.0, key, value) };
    }

    fn set_buffer(&self, key: *const c_char, bytes: &[u8]) {
        // SAFETY: live format; the NDK copies `bytes.len()` bytes out of a
        // slice that outlives the call.
        unsafe {
            ndk::AMediaFormat_setBuffer(self.0, key, bytes.as_ptr().cast::<c_void>(), bytes.len())
        };
    }

    /// `set_int32` for a key the NDK's own constant cannot be named at this
    /// API level. The key is a string either way; only the way it is spelled
    /// differs.
    fn set_int32_by_name(&self, key: &str, value: i32) -> Result<()> {
        let key = CString::new(key)
            .map_err(|_| VideoError::InvalidData("format key contains an interior NUL".into()))?;
        // SAFETY: live format and a NUL-terminated key the NDK copies.
        unsafe { ndk::AMediaFormat_setInt32(self.0, key.as_ptr(), value) };
        Ok(())
    }

    fn set_string(&self, key: *const c_char, value: &str) -> Result<()> {
        let value = CString::new(value)
            .map_err(|_| VideoError::InvalidData("format value contains an interior NUL".into()))?;
        // SAFETY: live format and a NUL-terminated string the NDK copies.
        unsafe { ndk::AMediaFormat_setString(self.0, key, value.as_ptr()) };
        Ok(())
    }

    fn describe(&self) -> String {
        // SAFETY: the format is live; the returned string belongs to it.
        let text = unsafe { ndk::AMediaFormat_toString(self.0) };
        if text.is_null() {
            return "<no description>".into();
        }
        // SAFETY: NUL-terminated for as long as the format lives.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    }
}

/// A read-only file descriptor, closed on drop.
///
/// The NDK takes descriptors where every other backend here takes a path:
/// `AMediaExtractor_setDataSourceFd` avoids the URI parsing `setDataSource`
/// does, and `AMediaMuxer_new` has no path form at all.
struct Fd(i32);

impl Drop for Fd {
    fn drop(&mut self) {
        if self.0 >= 0 {
            // SAFETY: the descriptor is ours and closed exactly once.
            unsafe { libc::close(self.0) };
        }
    }
}

impl Fd {
    fn open_read(path: &Path) -> Result<(Self, i64)> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| VideoError::InvalidData("path contains an interior NUL".into()))?;
        // SAFETY: a live NUL-terminated path; O_RDONLY takes no mode.
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(VideoError::Io(std::io::Error::last_os_error()));
        }
        let fd = Self(fd);
        // SAFETY: a zeroed stat is the documented input; the descriptor is live.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.0, &mut stat) } != 0 {
            return Err(VideoError::Io(std::io::Error::last_os_error()));
        }
        Ok((fd, stat.st_size))
    }

    fn create(path: &Path) -> Result<Self> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| VideoError::InvalidData("path contains an interior NUL".into()))?;
        // SAFETY: a live NUL-terminated path; the mode is read by O_CREAT.
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC | libc::O_CLOEXEC,
                0o644,
            )
        };
        if fd < 0 {
            return Err(VideoError::Io(std::io::Error::last_os_error()));
        }
        Ok(Self(fd))
    }
}

/// The MIME type MediaCodec and MediaMuxer name a codec by.
fn mime_for(codec: &Codec) -> Result<&'static str> {
    match codec {
        Codec::H264 => Ok("video/avc"),
        Codec::Hevc => Ok("video/hevc"),
        Codec::Vp8 => Ok("video/x-vnd.on2.vp8"),
        Codec::Vp9 => Ok("video/x-vnd.on2.vp9"),
        Codec::Av1 => Ok("video/av01"),
        Codec::Aac => Ok("audio/mp4a-latm"),
        Codec::Mp3 => Ok("audio/mpeg"),
        // Android has never shipped a ProRes codec; `Other` is by definition
        // something libavformat named and we did not.
        codec => Err(VideoError::UnsupportedCodec(format!(
            "no Android MIME type for {codec:?}"
        ))),
    }
}

/// The extractor track that carries `stream_index`'s samples.
///
/// libavformat and `AMediaExtractor` number tracks independently, and neither
/// exposes the other's index. What both preserve is ORDER within a kind: the
/// nth video stream libavformat reports is the nth video track the extractor
/// reports, for ISO-BMFF and Matroska alike, because both walk the container's
/// own track list. Matching on that is exact where matching on the raw index
/// would silently select the wrong track in any file whose first stream is not
/// video.
fn track_for_stream(
    extractor: *mut ndk::AMediaExtractor,
    stream: &StreamInfo,
    info: &MediaInfo,
) -> Result<usize> {
    let ordinal = info
        .streams
        .iter()
        .filter(|candidate| candidate.kind == stream.kind)
        .position(|candidate| candidate.index == stream.index)
        .ok_or_else(|| {
            VideoError::InvalidData(format!("stream {} is not in this file", stream.index))
        })?;

    // SAFETY: the extractor is live and has a data source.
    let count = unsafe { ndk::AMediaExtractor_getTrackCount(extractor) };
    let mut seen = 0usize;
    for track in 0..count {
        // SAFETY: `track` is below the count the extractor just reported.
        let format = Format::adopt(
            unsafe { ndk::AMediaExtractor_getTrackFormat(extractor, track) },
            "AMediaExtractor_getTrackFormat",
        )?;
        // SAFETY: the key statics are immutable NUL-terminated names.
        let mime = format
            .string(unsafe { ndk::AMEDIAFORMAT_KEY_MIME })
            .unwrap_or_default();
        let kind = if mime.starts_with("video/") {
            MediaKind::Video
        } else if mime.starts_with("audio/") {
            MediaKind::Audio
        } else {
            MediaKind::Unknown
        };
        if kind != stream.kind {
            continue;
        }
        if seen == ordinal {
            return Ok(track);
        }
        seen += 1;
    }
    Err(VideoError::InvalidData(format!(
        "the extractor has no {:?} track {ordinal} for stream {}",
        stream.kind, stream.index
    )))
}

/// The stream a caller named, from the one probe.
fn stream_of(info: &MediaInfo, stream_index: u32) -> Result<StreamInfo> {
    info.stream(stream_index)
        .cloned()
        .ok_or_else(|| VideoError::InvalidData(format!("no stream {stream_index} in this file")))
}

/// An extractor with its data source set and one track selected.
struct Extractor {
    inner: *mut ndk::AMediaExtractor,
    /// Held for the extractor's life: `setDataSourceFd` does not dup.
    _source: Fd,
    track: usize,
}

impl Drop for Extractor {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            // SAFETY: sole owner.
            unsafe { ndk::AMediaExtractor_delete(self.inner) };
        }
    }
}

impl Extractor {
    fn open(path: &Path, stream: &StreamInfo, info: &MediaInfo) -> Result<(Self, Format)> {
        let (source, length) = Fd::open_read(path)?;
        // SAFETY: allocates an extractor or returns null.
        let inner = unsafe { ndk::AMediaExtractor_new() };
        if inner.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an AMediaExtractor".into(),
            ));
        }
        let mut extractor = Self {
            inner,
            _source: source,
            track: 0,
        };
        // SAFETY: the extractor is live and the descriptor is open for the
        // whole of its life (it is owned by the struct being built).
        check_media("AMediaExtractor_setDataSourceFd", unsafe {
            ndk::AMediaExtractor_setDataSourceFd(extractor.inner, extractor._source.0, 0, length)
        })?;
        let track = track_for_stream(extractor.inner, stream, info)?;
        // SAFETY: the track index came from the extractor's own count.
        check_media("AMediaExtractor_selectTrack", unsafe {
            ndk::AMediaExtractor_selectTrack(extractor.inner, track)
        })?;
        extractor.track = track;
        let format = Format::adopt(
            // SAFETY: as above.
            unsafe { ndk::AMediaExtractor_getTrackFormat(extractor.inner, track) },
            "AMediaExtractor_getTrackFormat",
        )?;
        Ok((extractor, format))
    }

    /// Seek to the sync sample at or before `target`, the only place a decoder
    /// can start. Microseconds, which is the extractor's own unit.
    fn seek_backward(&mut self, target_us: i64) -> Result<()> {
        // SAFETY: the extractor is live with a selected track.
        check_media("AMediaExtractor_seekTo", unsafe {
            ndk::AMediaExtractor_seekTo(
                self.inner,
                target_us.max(0),
                ndk::SeekMode::AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC,
            )
        })?;
        Ok(())
    }
}

// -------------------------------------------------------------------------
// A decoded picture on the phone's graphics memory

/// The `AImage` MediaCodec rendered, and the reader that owns its buffer pool.
///
/// Cheap to clone (`Arc`) because [`VideoFrame`] is `Clone` and a caller may
/// hold one frame while decoding the next; the image is released exactly once,
/// when the last clone drops, and the reader outlives every image it issued.
#[derive(Clone)]
pub struct AndroidImage {
    inner: Arc<OwnedImage>,
}

struct OwnedImage {
    image: *mut ndk::AImage,
    /// The image's buffer belongs to this reader's pool. Deleting the reader
    /// first would free the memory the image still points at.
    _reader: Arc<ImageReader>,
    /// The decoder's live-image count, decremented when this one is released.
    outstanding: Arc<std::sync::atomic::AtomicUsize>,
}

// SAFETY: an `AImage` is an owned handle with no thread affinity; the pool it
// borrows from is kept alive by the `Arc` beside it, and nothing here is
// mutated after construction.
unsafe impl Send for OwnedImage {}
// SAFETY: as above — every method is a read through a `&self`.
unsafe impl Sync for OwnedImage {}

impl Drop for OwnedImage {
    fn drop(&mut self) {
        if !self.image.is_null() {
            // SAFETY: sole owner; returns the buffer to the reader's pool.
            unsafe { ndk::AImage_delete(self.image) };
        }
        self.outstanding
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl std::fmt::Debug for AndroidImage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AndroidImage")
            .field("size", &(self.width(), self.height()))
            .field("format", &self.format())
            .finish_non_exhaustive()
    }
}

/// One plane of a decoded image, as the NDK describes it.
struct Plane<'a> {
    data: &'a [u8],
    row_stride: usize,
    pixel_stride: usize,
}

impl AndroidImage {
    fn new(
        image: *mut ndk::AImage,
        reader: Arc<ImageReader>,
        outstanding: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        outstanding.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self {
            inner: Arc::new(OwnedImage {
                image,
                _reader: reader,
                outstanding,
            }),
        }
    }

    /// Images the decoder has issued and nobody has dropped, this one
    /// included. `AImageReader`'s pool is [`READER_MAX_IMAGES`] deep and
    /// refuses to issue past it, so this is what the decoder reads to decide
    /// whether the last slot is about to be given away.
    fn outstanding(&self) -> usize {
        self.inner
            .outstanding
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn width(&self) -> u32 {
        let mut value = 0i32;
        // SAFETY: the image is live for the whole of `self`.
        unsafe { ndk::AImage_getWidth(self.inner.image, &mut value) };
        value.max(0) as u32
    }

    fn height(&self) -> u32 {
        let mut value = 0i32;
        // SAFETY: as above.
        unsafe { ndk::AImage_getHeight(self.inner.image, &mut value) };
        value.max(0) as u32
    }

    fn require_full_crop(&self) -> Result<()> {
        let mut crop = ndk::AImageCropRect {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        // SAFETY: the AImage is owned by self and crop is a valid out-pointer.
        check_media("AImage_getCropRect", unsafe {
            ndk::AImage_getCropRect(self.inner.image, &mut crop)
        })?;
        require_full_image_crop(
            self.width(),
            self.height(),
            crop.left,
            crop.top,
            crop.right,
            crop.bottom,
        )
    }

    /// The graphics format this picture arrived in — `0x23` for 8-bit
    /// `YUV_420_888`, `0x36` for `YCbCr_P010`. Public because it is the only
    /// way a caller can tell a 10-bit decode from an 8-bit one, and a
    /// silently-downgraded HDR import is the defect that question exists to
    /// catch.
    pub fn format(&self) -> i32 {
        let mut value = 0i32;
        // SAFETY: as above.
        unsafe { ndk::AImage_getFormat(self.inner.image, &mut value) };
        value
    }

    /// The `AHardwareBuffer` behind this image — what a GPU consumer imports,
    /// and the reason nothing here reads pixels unless a caller asks.
    ///
    /// Borrowed, not retained: the buffer belongs to the image.
    pub fn hardware_buffer(&self) -> Option<NonNull<ndk::AHardwareBuffer>> {
        let mut raw: *mut ndk::AHardwareBuffer = ptr::null_mut();
        // SAFETY: live image, valid out-pointer. The NDK does not retain.
        let status = unsafe { ndk::AImage_getHardwareBuffer(self.inner.image, &mut raw) };
        (status == ndk::media_status_t::AMEDIA_OK)
            .then(|| NonNull::new(raw))
            .flatten()
    }

    fn plane(&self, index: i32) -> Result<Plane<'_>> {
        let mut data: *mut u8 = ptr::null_mut();
        let mut length = 0i32;
        // SAFETY: live image; both out-pointers are valid. The returned buffer
        // is the image's and lives as long as it does.
        check_media("AImage_getPlaneData", unsafe {
            ndk::AImage_getPlaneData(self.inner.image, index, &mut data, &mut length)
        })?;
        if data.is_null() || length <= 0 {
            return Err(VideoError::InvalidData(format!(
                "AImage plane {index} has no data"
            )));
        }
        let mut row_stride = 0i32;
        // SAFETY: as above.
        check_media("AImage_getPlaneRowStride", unsafe {
            ndk::AImage_getPlaneRowStride(self.inner.image, index, &mut row_stride)
        })?;
        let mut pixel_stride = 0i32;
        // SAFETY: as above.
        check_media("AImage_getPlanePixelStride", unsafe {
            ndk::AImage_getPlanePixelStride(self.inner.image, index, &mut pixel_stride)
        })?;
        Ok(Plane {
            // SAFETY: the NDK reported `length` readable bytes at `data`.
            data: unsafe { std::slice::from_raw_parts(data, length as usize) },
            row_stride: row_stride.max(0) as usize,
            pixel_stride: pixel_stride.max(1) as usize,
        })
    }

    /// Owned rotation input. Keep original P010 words and 4:2:0 chroma instead
    /// of passing a geometric transform through the explicit RGB8 model door.
    pub(crate) fn rotation_pixels(&self, color: ColorInfo) -> Result<FrameBuffer> {
        use crate::{Yuv420Format, Yuv420Frame};
        let (width, height) = (self.width(), self.height());
        let format = self.format();
        if format == ndk::AIMAGE_FORMATS::AIMAGE_FORMAT_RGBA_8888.0 as i32 {
            let plane = self.plane(0)?;
            let mut packed = Vec::with_capacity(width as usize * height as usize * 4);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let offset = y * plane.row_stride + x * plane.pixel_stride;
                    packed.extend_from_slice(plane.data.get(offset..offset + 4).ok_or_else(
                        || VideoError::InvalidData("short RGBA rotation plane".into()),
                    )?);
                }
            }
            return Ok(FrameBuffer::Cpu(CpuFrame::new(
                width,
                height,
                width as usize * 4,
                CpuPixelFormat::Rgba8,
                packed,
            )?));
        }
        let ten_bit = format == AIMAGE_FORMAT_YCBCR_P010;
        if !ten_bit && format != ndk::AIMAGE_FORMATS::AIMAGE_FORMAT_YUV_420_888.0 as i32 {
            return Err(VideoError::UnsupportedCodec(format!(
                "rotation for AImage format 0x{format:08x}"
            )));
        }
        let (y, cb, cr) = (self.plane(0)?, self.plane(1)?, self.plane(2)?);
        let bytes = if ten_bit { 2 } else { 1 };
        fn read<'a>(plane: &Plane<'a>, x: usize, y: usize, bytes: usize) -> Result<&'a [u8]> {
            let offset = y * plane.row_stride + x * plane.pixel_stride;
            plane
                .data
                .get(offset..offset + bytes)
                .ok_or_else(|| VideoError::InvalidData("short YUV rotation plane".into()))
        }
        let mut luma = Vec::with_capacity(width as usize * height as usize * bytes);
        let mut chroma = Vec::with_capacity(width as usize * height as usize * bytes / 2);
        for row in 0..height as usize {
            for column in 0..width as usize {
                luma.extend_from_slice(read(&y, column, row, bytes)?);
            }
        }
        for row in 0..height as usize / 2 {
            for column in 0..width as usize / 2 {
                chroma.extend_from_slice(read(&cb, column, row, bytes)?);
                chroma.extend_from_slice(read(&cr, column, row, bytes)?);
            }
        }
        let format = match (ten_bit, color.range == ColorRange::Full) {
            (true, true) => Yuv420Format::P010Full,
            (true, false) => Yuv420Format::P010Video,
            (false, true) => Yuv420Format::Nv12Full,
            (false, false) => Yuv420Format::Nv12Video,
        };
        Ok(FrameBuffer::Yuv420(Yuv420Frame::new(
            width, height, format, luma, chroma,
        )?))
    }

    /// Download this picture as packed RGB24 — the explicit readback, never
    /// taken by the decode path itself.
    pub(crate) fn to_rgb24(&self, color: ColorInfo) -> Result<CpuFrame> {
        let width = self.width() as usize;
        let height = self.height() as usize;
        if width == 0 || height == 0 {
            return Err(VideoError::InvalidData("AImage has no extent".into()));
        }
        let format = self.format();
        if format == ndk::AIMAGE_FORMATS::AIMAGE_FORMAT_RGBA_8888.0 as i32 {
            let plane = self.plane(0)?;
            let frame = CpuFrame::new(
                width as u32,
                height as u32,
                plane.row_stride,
                CpuPixelFormat::Rgba8,
                plane.data,
            )?;
            return Ok(frame.to_rgb24());
        }
        let ten_bit = format == AIMAGE_FORMAT_YCBCR_P010;
        if !ten_bit && format != ndk::AIMAGE_FORMATS::AIMAGE_FORMAT_YUV_420_888.0 as i32 {
            return Err(VideoError::UnsupportedCodec(format!(
                "CPU conversion for AImage format 0x{format:08x}"
            )));
        }
        let (luma, cb, cr) = (self.plane(0)?, self.plane(1)?, self.plane(2)?);
        let full_range = color.range == ColorRange::Full;
        let rgb = planar_420_to_rgb(&luma, &cb, &cr, width, height, ten_bit, full_range, color);
        CpuFrame::new(
            width as u32,
            height as u32,
            width * 3,
            CpuPixelFormat::Rgb24,
            rgb,
        )
    }
}

/// A 4:2:0 image with arbitrary plane strides, as RGB24.
///
/// `AImage` describes each plane with a row stride AND a pixel stride, and the
/// pixel stride is what makes one walker serve every layout the device hands
/// back: 1 for planar I420, 2 for semi-planar NV12/NV21 (where the Cb and Cr
/// planes are the same allocation one byte apart), 2 again for P010's 16-bit
/// samples. Assuming any one of them is how a backend reads green as blue on
/// the one device that differs.
#[allow(clippy::too_many_arguments)]
fn planar_420_to_rgb(
    luma: &Plane<'_>,
    cb: &Plane<'_>,
    cr: &Plane<'_>,
    width: usize,
    height: usize,
    ten_bit: bool,
    full_range: bool,
    color: ColorInfo,
) -> Vec<u8> {
    let sample = |plane: &Plane<'_>, x: usize, y: usize| -> f32 {
        let offset = y * plane.row_stride + x * plane.pixel_stride;
        match plane.data.get(offset) {
            None => 0.0,
            Some(&byte) if !ten_bit => f32::from(byte),
            Some(&low) => {
                let high = plane.data.get(offset + 1).copied().unwrap_or(0);
                // Ten bits, left-justified in a little-endian word: the value
                // is `word >> 6`, and the 8-bit equivalent is that over four.
                let word = (u16::from(low) | (u16::from(high) << 8)) >> P010_SAMPLE_SHIFT;
                f32::from(((u32::from(word) + 2) / 4).min(255) as u8)
            }
        }
    };

    let mut output = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
    for row in 0..height {
        for column in 0..width {
            let y = sample(luma, column, row);
            let u = sample(cb, column / 2, row / 2);
            let v = sample(cr, column / 2, row / 2);
            let offset = (row * width + column) * 3;
            output[offset..offset + 3]
                .copy_from_slice(&crate::frame::ycbcr_to_rgb(y, u, v, full_range, color));
        }
    }
    output
}

/// An `AImageReader` and the surface MediaCodec renders into.
struct ImageReader {
    reader: *mut ndk::AImageReader,
    window: *mut ndk::ANativeWindow,
    /// Raised by the reader's own callback when a picture lands. The wait is
    /// on this, not on a sleep: `AImageReader_setImageListener` is the
    /// platform's own "a frame is ready" edge.
    arrived: Arc<(Mutex<usize>, Condvar)>,
}

// SAFETY: the reader is owned by one decode session and the listener only
// touches the `Arc`'d condition pair, which is itself `Send + Sync`.
unsafe impl Send for ImageReader {}
// SAFETY: as above — the image path reads through `&self`.
unsafe impl Sync for ImageReader {}

impl Drop for ImageReader {
    fn drop(&mut self) {
        if !self.reader.is_null() {
            // SAFETY: sole owner. Every image issued from this pool holds an
            // `Arc` to this reader, so none is outstanding here.
            unsafe { ndk::AImageReader_delete(self.reader) };
        }
    }
}

/// `AImageReader_setImageListener`'s callback.
///
/// # Safety
///
/// Called by the NDK with the `context` handed to `setImageListener`, which is
/// a `*const (Mutex<usize>, Condvar)` kept alive by the reader.
unsafe extern "C" fn image_available(context: *mut c_void, _reader: *mut ndk::AImageReader) {
    let Some(pointer) = NonNull::new(context.cast::<(Mutex<usize>, Condvar)>()) else {
        return;
    };
    // SAFETY: the `Arc` inside the reader outlives the listener, which the
    // NDK stops calling when the reader is deleted.
    let (pending, ready) = unsafe { pointer.as_ref() };
    *pending.lock().unwrap_or_else(|error| error.into_inner()) += 1;
    ready.notify_all();
}

impl ImageReader {
    fn open(width: u32, height: u32, format: i32) -> Result<Arc<Self>> {
        let mut reader: *mut ndk::AImageReader = ptr::null_mut();
        // CPU_READ_OFTEN because `to_cpu_rgb` is a real caller: without it a
        // hardware buffer may be allocated in memory `AImage_getPlaneData`
        // cannot map, and the download fails on the device that took the hint.
        // SAFETY: valid out-pointer; the rest are plain values.
        check_media("AImageReader_newWithUsage", unsafe {
            ndk::AImageReader_newWithUsage(
                width as i32,
                height as i32,
                format,
                ndk::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN.0,
                READER_MAX_IMAGES,
                &mut reader,
            )
        })?;
        if reader.is_null() {
            return Err(VideoError::BackendUnavailable(
                "AImageReader_newWithUsage returned no reader".into(),
            ));
        }
        let mut window: *mut ndk::ANativeWindow = ptr::null_mut();
        // SAFETY: the reader is live; the window belongs to it.
        let status = unsafe { ndk::AImageReader_getWindow(reader, &mut window) };
        if status != ndk::media_status_t::AMEDIA_OK || window.is_null() {
            // SAFETY: nothing else took ownership yet.
            unsafe { ndk::AImageReader_delete(reader) };
            return Err(VideoError::platform("AImageReader_getWindow", status.0));
        }
        let owned = Arc::new(Self {
            reader,
            window,
            arrived: Arc::new((Mutex::new(0), Condvar::new())),
        });
        let mut listener = ndk::AImageReader_ImageListener {
            context: Arc::as_ptr(&owned.arrived).cast::<c_void>().cast_mut(),
            onImageAvailable: Some(image_available),
        };
        // SAFETY: the reader is live and the context outlives it (the `Arc`
        // is a field of the same object).
        check_media("AImageReader_setImageListener", unsafe {
            ndk::AImageReader_setImageListener(owned.reader, &mut listener)
        })?;
        Ok(owned)
    }

    /// The next rendered picture, waiting up to `timeout` for the listener.
    ///
    /// `AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE` is not an error — a buffer
    /// released to the surface arrives a moment later, and this is that
    /// moment.
    fn acquire(
        self: &Arc<Self>,
        timeout: std::time::Duration,
        outstanding: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<Option<AndroidImage>> {
        let deadline = std::time::Instant::now() + timeout;
        let (pending, ready) = &*self.arrived;
        let mut count = pending.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            if *count == 0 {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Ok(None);
                }
                // Spurious notifications do not turn a short scheduling delay
                // into decoder failure; wait against the same fixed deadline.
                let (guard, _) = ready
                    .wait_timeout_while(count, remaining, |count| *count == 0)
                    .unwrap_or_else(|error| error.into_inner());
                count = guard;
                if *count == 0 {
                    return Ok(None);
                }
            }
            let mut image: *mut ndk::AImage = ptr::null_mut();
            // SAFETY: live reader and valid out-pointer. The listener can have
            // a stale notification, which returns to waiting instead of polling.
            let status = unsafe { ndk::AImageReader_acquireNextImage(self.reader, &mut image) };
            if status == ndk::media_status_t::AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE {
                *count = 0;
                continue;
            }
            check_media("AImageReader_acquireNextImage", status)?;
            *count = count.saturating_sub(1);
            drop(count);
            return match NonNull::new(image) {
                Some(_) => Ok(Some(AndroidImage::new(
                    image,
                    Arc::clone(self),
                    Arc::clone(outstanding),
                ))),
                None => Err(VideoError::BackendUnavailable(
                    "AImageReader returned success without its expected image".into(),
                )),
            };
        }
    }
}

/// An owned `AMediaCodec`, stopped and deleted on drop.
struct MediaCodec {
    inner: *mut ndk::AMediaCodec,
    started: bool,
}

impl Drop for MediaCodec {
    fn drop(&mut self) {
        self.close();
    }
}

impl MediaCodec {
    fn decoder(mime: &str) -> Result<Self> {
        let mime = CString::new(mime).expect("a MIME name without NUL");
        // SAFETY: a live NUL-terminated MIME name; returns a codec or null.
        let inner = unsafe { ndk::AMediaCodec_createDecoderByType(mime.as_ptr()) };
        if inner.is_null() {
            return Err(VideoError::UnsupportedCodec(format!(
                "this device has no decoder for {}",
                mime.to_string_lossy()
            )));
        }
        Ok(Self {
            inner,
            started: false,
        })
    }

    fn encoder(mime: &str) -> Result<Self> {
        let mime = CString::new(mime).expect("a MIME name without NUL");
        // SAFETY: as above.
        let inner = unsafe { ndk::AMediaCodec_createEncoderByType(mime.as_ptr()) };
        if inner.is_null() {
            return Err(VideoError::UnsupportedCodec(format!(
                "this device has no encoder for {}",
                mime.to_string_lossy()
            )));
        }
        Ok(Self {
            inner,
            started: false,
        })
    }

    fn by_name(name: &str) -> Result<Self> {
        let name = CString::new(name)
            .map_err(|_| VideoError::InvalidData("codec name contains NUL".into()))?;
        // SAFETY: the codec name comes from the platform catalog; owned NUL-terminated bytes.
        let inner = unsafe { ndk::AMediaCodec_createCodecByName(name.as_ptr()) };
        if inner.is_null() {
            return Err(VideoError::UnsupportedCodec(format!(
                "cannot create codec {}",
                name.to_string_lossy()
            )));
        }
        Ok(Self {
            inner,
            started: false,
        })
    }

    fn name(&self) -> String {
        let mut raw: *mut c_char = ptr::null_mut();
        // SAFETY: the codec is live; the name is released below.
        let status = unsafe { ndk::AMediaCodec_getName(self.inner, &mut raw) };
        if status != ndk::media_status_t::AMEDIA_OK || raw.is_null() {
            return "<unnamed>".into();
        }
        // SAFETY: NUL-terminated until released.
        let name = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: the NDK allocated it and takes it back.
        unsafe { ndk::AMediaCodec_releaseName(self.inner, raw) };
        name
    }

    fn configure(
        &mut self,
        format: &Format,
        surface: *mut ndk::ANativeWindow,
        encode: bool,
    ) -> Result<()> {
        // SAFETY: the codec is live and not started; the format and surface
        // outlive the call.
        check_media("AMediaCodec_configure", unsafe {
            ndk::AMediaCodec_configure(
                self.inner,
                format.0,
                surface,
                ptr::null_mut(),
                if encode {
                    ndk::AMEDIACODEC_CONFIGURE_FLAG_ENCODE as u32
                } else {
                    0
                },
            )
        })
    }

    fn start(&mut self) -> Result<()> {
        // SAFETY: the codec is live and configured.
        check_media("AMediaCodec_start", unsafe {
            ndk::AMediaCodec_start(self.inner)
        })?;
        self.started = true;
        Ok(())
    }

    /// A decoder for `mime`, configured for this track and started.
    ///
    /// Both `open` and `seek` come through here, because a seek REBUILDS the
    /// codec rather than flushing it — which is the macOS twin's answer too
    /// (`VideoToolboxDecoder::recreate_session`). `AMediaCodec_flush` is the
    /// obvious cheaper move and it is wrong here, in two ways both measured on
    /// `cuts-36`:
    ///
    /// * flushing before the codec has produced any output wedges it. A seek
    ///   issued as the FIRST operation on a decoder — exactly what a thumbnail
    ///   or a scrub does — then swallowed all twelve samples of the fixture and
    ///   emitted nothing but end-of-stream, so `decode_next` returned `None`
    ///   for the whole clip.
    /// * an audio decoder discards its first coded frame after every flush,
    ///   warm or cold. Mid-stream that frame is pre-roll and costs nothing; at
    ///   the start of the program it is program, and a seek to sample 0 came
    ///   back at 1 024 with the drain 1 024 samples short of 48 000.
    ///
    /// `AMediaCodec_start` after `AMediaCodec_flush` is not the fix either: the
    /// platform refuses it with `AMEDIA_ERROR_UNKNOWN`.
    /// Stop and release this codec, leaving the handle empty.
    ///
    /// A seek calls it BEFORE building the replacement, and the order is not a
    /// nicety: an `ANativeWindow` takes one producer at a time, so configuring
    /// a second codec against a surface the first is still connected to fails
    /// with `AMEDIA_ERROR_UNKNOWN` — measured on `cuts-36`.
    fn close(&mut self) {
        if self.inner.is_null() {
            return;
        }
        // SAFETY: sole owner; stop before delete is the documented teardown.
        unsafe {
            if self.started {
                ndk::AMediaCodec_stop(self.inner);
            }
            ndk::AMediaCodec_delete(self.inner);
        }
        self.inner = ptr::null_mut();
        self.started = false;
    }

    fn open_decoder_codec(
        mime: &str,
        format: &Format,
        surface: *mut ndk::ANativeWindow,
    ) -> Result<Self> {
        let mut codec = Self::decoder(mime)?;
        codec.configure(format, surface, false)?;
        codec.start()?;
        Ok(codec)
    }

    /// An input buffer index, or `None` when the codec has none free.
    fn dequeue_input(&self, timeout_us: i64) -> Option<usize> {
        // SAFETY: the codec is live and started.
        let index = unsafe { ndk::AMediaCodec_dequeueInputBuffer(self.inner, timeout_us) };
        (index >= 0).then_some(index as usize)
    }

    fn input_buffer(&mut self, index: usize) -> Result<&mut [u8]> {
        let mut capacity = 0usize;
        // SAFETY: the index came from `dequeue_input`, so the buffer is ours
        // until it is queued; the NDK reports its capacity.
        let pointer = unsafe { ndk::AMediaCodec_getInputBuffer(self.inner, index, &mut capacity) };
        if pointer.is_null() {
            return Err(VideoError::BackendUnavailable(
                "AMediaCodec_getInputBuffer returned null".into(),
            ));
        }
        // SAFETY: `capacity` writable bytes, exclusively ours until queued.
        Ok(unsafe { std::slice::from_raw_parts_mut(pointer, capacity) })
    }

    fn output_buffer(&self, index: usize) -> Result<&[u8]> {
        let mut size = 0usize;
        // SAFETY: the index came from a successful dequeue.
        let pointer = unsafe { ndk::AMediaCodec_getOutputBuffer(self.inner, index, &mut size) };
        if pointer.is_null() {
            return Err(VideoError::BackendUnavailable(
                "AMediaCodec_getOutputBuffer returned null".into(),
            ));
        }
        // SAFETY: `size` readable bytes until the buffer is released.
        Ok(unsafe { std::slice::from_raw_parts(pointer, size) })
    }

    fn queue_input(
        &self,
        index: usize,
        size: usize,
        presentation_time_us: i64,
        flags: u32,
    ) -> Result<()> {
        // SAFETY: the index came from `dequeue_input` and the codec takes the
        // buffer back here.
        check_media("AMediaCodec_queueInputBuffer", unsafe {
            ndk::AMediaCodec_queueInputBuffer(
                self.inner,
                index,
                0,
                size,
                presentation_time_us as u64,
                flags,
            )
        })
    }

    fn output_format(&self) -> Result<Format> {
        Format::adopt(
            // SAFETY: the codec is live; the format is the caller's to free.
            unsafe { ndk::AMediaCodec_getOutputFormat(self.inner) },
            "AMediaCodec_getOutputFormat",
        )
    }

    fn input_format(&self) -> Result<Format> {
        Format::adopt(
            // SAFETY: as above.
            unsafe { ndk::AMediaCodec_getInputFormat(self.inner) },
            "AMediaCodec_getInputFormat",
        )
    }

    fn release_output(&self, index: usize, render: bool) -> Result<()> {
        // SAFETY: the index came from a successful dequeue and is released
        // exactly once.
        check_media("AMediaCodec_releaseOutputBuffer", unsafe {
            ndk::AMediaCodec_releaseOutputBuffer(self.inner, index, render)
        })
    }
}

/// What one `dequeueOutputBuffer` said.
enum Output {
    Buffer {
        index: usize,
        info: ndk::AMediaCodecBufferInfo,
    },
    FormatChanged,
    TryAgain,
}

impl MediaCodec {
    fn dequeue_output(&self, timeout_us: i64) -> Result<Output> {
        // SAFETY: a zeroed BufferInfo is the documented out-parameter.
        let mut info: ndk::AMediaCodecBufferInfo = unsafe { std::mem::zeroed() };
        // SAFETY: the codec is live and started; `info` is a valid out-pointer.
        let index =
            unsafe { ndk::AMediaCodec_dequeueOutputBuffer(self.inner, &mut info, timeout_us) };
        if index >= 0 {
            return Ok(Output::Buffer {
                index: index as usize,
                info,
            });
        }
        match index as i32 {
            ndk::AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED => Ok(Output::FormatChanged),
            // The legacy "buffers changed" signal; the NDK's getOutputBuffer
            // re-reads the pool every call, so there is nothing to do.
            ndk::AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED
            | ndk::AMEDIACODEC_INFO_TRY_AGAIN_LATER => Ok(Output::TryAgain),
            status => Err(VideoError::platform(
                "AMediaCodec_dequeueOutputBuffer",
                status,
            )),
        }
    }
}

// -------------------------------------------------------------------------
// Video

/// The Android video backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct AndroidVideoIo;

#[async_trait(?Send)]
impl VideoIo for AndroidVideoIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        crate::demux::probe(source)
    }

    async fn open_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        self.open_decoder_with(source, stream_index, DecodeOptions::default())
            .await
    }

    async fn open_coded_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        open_original_decoder(source, stream_index, DecodeOptions::default(), true).await
    }

    async fn open_coded_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        open_original_decoder(source, stream_index, options, true).await
    }

    /// MediaCodec first, libavcodec for what it refuses.
    ///
    /// The refusal is a real one and not a fallback for convenience: no
    /// Android device has ever shipped a ProRes decoder, and the six
    /// `libav*.so` are already in the APK because `engine-media` links them.
    /// A file the phone cannot decode in hardware is decoded in software
    /// rather than rejected.
    async fn open_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        open_original_decoder(source, stream_index, options, false).await
    }

    async fn hdr_static_metadata(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Option<crate::HdrStaticMetadata>> {
        let info = crate::demux::probe(source)?;
        let stream = stream_of(&info, stream_index)?;
        let (_extractor, format) = Extractor::open(source, &stream, &info)?;
        format
            .buffer(c"hdr-static-info".as_ptr(), 25)?
            .map(|bytes| crate::HdrStaticMetadata::from_cta861_3(&bytes))
            .transpose()
    }

    async fn open_coded_decoder_for_color(
        &self,
        source: &Path,
        stream_index: u32,
        color: ColorInfo,
    ) -> Result<Box<dyn DecodeSession>> {
        let info = crate::demux::probe(source)?;
        let source_color = stream_of(&info, stream_index)?.color;
        if android_hdr::sdr_request(source_color, Some(color))? {
            match MediaCodecDecoder::open_for_color(source, stream_index, color) {
                Ok(decoder) => Ok(Box::new(RecoveringDecoder::new(
                    source,
                    stream_index,
                    DecodeOptions::default(),
                    SoftwareOrientation::CODED,
                    Some(Box::new(decoder)),
                    Arc::new(LibavVideoIo),
                ))),
                Err(error) => {
                    // Preserve original HDR samples; encodable_scaled performs
                    // the reviewed per-frame transfer/gamut conversion before
                    // quantization. No relabeling or whole-clip buffering.
                    log::info!(target: "ferment_engine_media::android", "decoder tone-map unavailable ({error}); using source-depth SDR conversion");
                    self.open_coded_decoder(source, stream_index).await
                }
            }
        } else {
            self.open_coded_decoder(source, stream_index).await
        }
    }

    async fn open_encoder(&self, settings: EncoderSettings) -> Result<Box<dyn EncodeSession>> {
        Ok(Box::new(MediaCodecEncoder::open(settings)?))
    }
}

/// A one-way recovery door for unsupported platform output, including refusals
/// discovered after the first picture. File/demux failures are not reclassified.
async fn open_original_decoder(
    source: &Path,
    stream_index: u32,
    options: DecodeOptions,
    coded_orientation: bool,
) -> Result<Box<dyn DecodeSession>> {
    // Prepare once so software recovery sees the very rotation metadata sent
    // to MediaCodec, including when configure/start refuses the source.
    options.validate()?;
    let input = match DecoderInput::open(source, stream_index) {
        Ok(input) => input,
        Err(error) if decoder_refusal(&error) => {
            log::debug!(target: "ferment_engine_media::android", "{}: {error}; decoding through libavcodec instead", source.display());
            // No Android track format was admitted; preserve the existing
            // software-only coded contract rather than guessing a rotation.
            return Ok(Box::new(
                RecoveringDecoder::software(
                    source,
                    stream_index,
                    options,
                    SoftwareOrientation::CODED,
                    Arc::new(LibavVideoIo),
                )
                .await?,
            ));
        }
        Err(error) => return Err(error),
    };
    let degrees = input
        .format
        .int32(c"rotation-degrees".as_ptr())
        .unwrap_or(0);
    let orientation = SoftwareOrientation::from_track(coded_orientation, degrees);
    // Android's current native path exposes surfaces/YUV. Explicit BGRA CPU
    // output uses the existing libav owner until that readback path is added.
    if !orientation.hardware_allowed
        || options.acceleration == crate::DecodeAcceleration::Software
        || options.output == crate::DecodeOutput::CpuBgra8
    {
        drop(input);
        return Ok(Box::new(
            RecoveringDecoder::software(
                source,
                stream_index,
                options,
                orientation,
                Arc::new(LibavVideoIo),
            )
            .await?,
        ));
    }
    match MediaCodecDecoder::open_input(
        source,
        stream_index,
        options,
        coded_orientation,
        None,
        input,
    ) {
        Ok(decoder) => Ok(Box::new(RecoveringDecoder::new(
            source,
            stream_index,
            options,
            orientation,
            Some(Box::new(decoder)),
            Arc::new(LibavVideoIo),
        ))),
        Err(error) if decoder_refusal(&error) => {
            log::debug!(target: "ferment_engine_media::android", "{}: {error}; decoding through libavcodec instead", source.display());
            Ok(Box::new(
                RecoveringDecoder::software(
                    source,
                    stream_index,
                    options,
                    orientation,
                    Arc::new(LibavVideoIo),
                )
                .await?,
            ))
        }
        Err(error) => Err(error),
    }
}

/// The presentation timestamps of buffers released to the reader, in the order
/// they were released — which is the order the reader issues images.
#[derive(Clone, Copy)]
struct Timing {
    pts: MediaTime,
    duration: Option<MediaTime>,
}

/// Where a decoded picture comes from.
enum Sink {
    /// MediaCodec renders into an `AImageReader`; frames stay in graphics
    /// memory as `AHardwareBuffer`s.
    Surface {
        reader: Arc<ImageReader>,
        released: VecDeque<Timing>,
        /// Images handed to callers and not yet dropped.
        ///
        /// `AImageReader`'s pool is [`READER_MAX_IMAGES`] deep and
        /// `acquireNextImage` fails with `MAX_IMAGES_ACQUIRED` once a caller
        /// holds them all — while `decode_next` promises an owned frame with
        /// no lifetime attached to it. A consumer that converts each frame and
        /// drops it (`VideoSource::decode_rgb`, the only one in this
        /// workspace) never comes near the limit; one that collects a whole
        /// clip would hit it on the fifth frame, and it would read as a decode
        /// failure rather than as the pool being full.
        ///
        /// So the last slot is never given away: when handing this picture
        /// over would fill the pool, it is downloaded and the image released
        /// instead. The pixels are the same either way — only the residency
        /// differs, which is exactly what `FrameBuffer`'s two variants model,
        /// and what `DecodeOptions` already tells consumers to read rather
        /// than assume.
        outstanding: Arc<std::sync::atomic::AtomicUsize>,
    },
    /// MediaCodec writes YUV into a CPU buffer. Taken when the caller asked
    /// for a size the reader could not be created at, or when the device
    /// refuses the reader's format.
    Buffers { output: Option<Format> },
}

/// Source ownership prepared before codec negotiation. The selected track's
/// format is the single authority for both hardware and recovery orientation.
struct DecoderInput {
    stream: StreamInfo,
    extractor: Extractor,
    format: Format,
    mime: &'static str,
}
impl DecoderInput {
    fn open(source: &Path, stream_index: u32) -> Result<Self> {
        let info = crate::demux::probe(source)?;
        let stream = stream_of(&info, stream_index)?;
        if stream.kind != MediaKind::Video {
            return Err(VideoError::InvalidData(format!(
                "stream {stream_index} is not video"
            )));
        }
        let mime = mime_for(&stream.codec)?;
        let (extractor, format) = Extractor::open(source, &stream, &info)?;
        Ok(Self {
            stream,
            extractor,
            format,
            mime,
        })
    }
}

struct MediaCodecDecoder {
    extractor: Extractor,
    codec: MediaCodec,
    /// The track description and its MIME, kept because a seek rebuilds the
    /// codec from them — see [`MediaCodec::open_decoder_codec`].
    format: Format,
    mime: &'static str,
    sink: Sink,
    color: ColorInfo,
    tone_mapped_color: Option<ColorInfo>,
    /// The first output has already confirmed an optional tone-map request.
    /// Its pixels and timing are delivered once, through the ordinary lookahead.
    prefetched: Option<VideoFrame>,
    options: DecodeOptions,
    /// The unit `AMediaExtractor` and `AMediaCodec` both speak.
    time_base: TimeBase,
    /// One pending hardware picture, released on seek or session drop.
    frame_timing: DecodedTimeline,
    /// Set by `seek`, cleared by the first picture at or after it — the same
    /// lazy discard the libav decoder does, for the same reason: one
    /// definition of "the next picture" instead of two.
    seek_target: Option<i64>,
    input_done: bool,
    output_done: bool,
    stalls: u32,
    /// Has anything been fed to this codec since the session was opened?
    ///
    /// A seek costs a coded frame twice over, and both were measured on
    /// `cuts-36`. MediaCodec consumes one frame before it emits anything, so
    /// rebuilding the codec throws that frame away; and
    /// `AMediaExtractor_seekTo(0)` lands on the first sample whose
    /// PRESENTATION time is at or after zero, which on an MP4 with an edit list
    /// skips the priming sample the decoder would otherwise have burned. A
    /// seek to zero therefore came back at sample 1 024 with the program's
    /// first 1 024 samples gone.
    ///
    /// An untouched session is already at the beginning, so seeking it there is
    /// a no-op — and saying so makes `open` followed by `seek(0)` behave
    /// exactly like `open`, which is the only behaviour a caller can reason
    /// about.
    touched: bool,
}

impl std::fmt::Debug for MediaCodecDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaCodecDecoder")
            .field("track", &self.extractor.track)
            .field(
                "sink",
                &match self.sink {
                    Sink::Surface { .. } => "surface",
                    Sink::Buffers { .. } => "buffers",
                },
            )
            .field("output_size", &self.options.output_size)
            .finish_non_exhaustive()
    }
}

impl MediaCodecDecoder {
    fn open_for_color(source: &Path, stream_index: u32, color: ColorInfo) -> Result<Self> {
        let mut decoder = Self::open_with_color(
            source,
            stream_index,
            DecodeOptions::default(),
            true,
            Some(color),
        )?;
        // Some codecs acknowledge the input request but decline conversion in
        // the first output format. Validate before exposing the session: an
        // error drops this codec/reader before the caller opens source-depth
        // decode. A successful first picture is retained, never discarded or
        // decoded twice, and enters the usual timing lookahead on the next call.
        let first = decoder.next_frame()?.ok_or_else(|| {
            VideoError::InvalidData("tone-map decoder produced no video picture".into())
        })?;
        if first.color != color {
            return Err(VideoError::UnsupportedCodec(
                "tone-map decoder did not confirm the requested SDR picture color".into(),
            ));
        }
        decoder.prefetched = Some(first);
        Ok(decoder)
    }

    fn open_with_color(
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
        coded_orientation: bool,
        target_color: Option<ColorInfo>,
    ) -> Result<Self> {
        Self::open_input(
            source,
            stream_index,
            options,
            coded_orientation,
            target_color,
            DecoderInput::open(source, stream_index)?,
        )
    }

    fn open_input(
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
        coded_orientation: bool,
        target_color: Option<ColorInfo>,
        input: DecoderInput,
    ) -> Result<Self> {
        let DecoderInput {
            stream,
            extractor,
            format,
            mime,
        } = input;
        let tone_map = android_hdr::sdr_request(stream.color, target_color)?;
        if tone_map {
            // API31: the decoder must acknowledge this exact request in its
            // input format. The source color properties remain unchanged.
            format.set_int32_by_name("color-transfer-request", 3)?;
        }
        if coded_orientation {
            // KEY_ROTATION is automatically applied only in surface mode.
            // Clearing it on the OWNED format makes surface and buffer output
            // coded alike. This very format is retained and reused on seek.
            format.set_int32_by_name("rotation-degrees", 0)?;
        }
        // The reader's size is the caller's when they named one. `AImageReader`
        // does not itself scale — the producer does, when the consumer's buffer
        // geometry differs — so this is a request the device may decline, and
        // `VideoFrame::width` reports what actually arrived, exactly as
        // `DecodeOptions::output_size` documents.
        if options
            .output_size
            .is_some_and(|(width, height)| width == 0 || height == 0)
        {
            return Err(VideoError::InvalidData(
                "decoder output size must be non-zero".into(),
            ));
        }
        let (width, height) = options
            .output_size
            .unwrap_or((stream.width.unwrap_or(0), stream.height.unwrap_or(0)));
        // A 10-bit reader for a stream that DECLARES ten bits, and for no
        // other. The transfer curve is not the signal, and that was measured:
        // the corpus's `synthetic-hdr-nclx.mp4` is tagged BT.2020/PQ and is
        // 8-bit, and asking for a P010 reader for it made
        // `AImageReader_acquireNextImage` fail with AMEDIA_ERROR_UNKNOWN on
        // `cuts-36` — a reader the device will create and the decoder cannot
        // render into. libav fills `bits_per_raw_sample` from the SPS during
        // `find_stream_info`, so a real Main10 stream arrives here declaring
        // ten, and a stream that declares nothing is 8-bit far more often than
        // it is 10. Getting it wrong the other way costs two bits; getting it
        // wrong this way costs the decode.
        let ten_bit = !tone_map && stream.color.bits_per_channel.is_some_and(|bits| bits > 8);
        let reader_format = if ten_bit {
            AIMAGE_FORMAT_YCBCR_P010
        } else {
            ndk::AIMAGE_FORMATS::AIMAGE_FORMAT_YUV_420_888.0 as i32
        };
        let sink = match (width, height) {
            (0, _) | (_, 0) => None,
            (width, height) => ImageReader::open(width, height, reader_format)
                .inspect_err(|error| {
                    log::debug!(
                        target: "ferment_engine_media::android",
                        "no {width}x{height} image reader (format 0x{reader_format:x}): {error}; \
                         decoding into CPU buffers",
                    );
                })
                .ok(),
        };

        let surface = sink
            .as_ref()
            .map_or(ptr::null_mut(), |reader| reader.window);
        require_supported_sink(!surface.is_null(), stream.color, tone_map)?;
        if !surface.is_null() {
            // Offline/non-View surfaces must not drop pictures (API29+).
            // Retain this owned format for any direct decoder seek rebuild.
            format.set_int32_by_name("allow-frame-drop", 0)?;
        }
        if surface.is_null()
            && !coded_orientation
            && format.int32(c"rotation-degrees".as_ptr()).unwrap_or(0) != 0
        {
            // MediaCodec applies KEY_ROTATION only to Surface output. Keep
            // buffer fallback on the same display-oriented software path.
            return Err(VideoError::UnsupportedCodec(
                "rotated non-coded output needs a surface or software rotation".into(),
            ));
        }
        if let Some(pixel_format) = android_hdr::decoder_buffer_color_format(
            stream.color,
            target_color,
            !surface.is_null(),
        )? {
            // AOSP defaults an unspecified ByteBuffer decoder to 8-bit planar
            // output. Request P010 explicitly before configure; the returned
            // layout/depth is still validated when the first buffer arrives.
            format.set_int32_by_name("color-format", pixel_format)?;
        }
        let codec = MediaCodec::open_decoder_codec(mime, &format, surface)?;
        if tone_map
            && codec
                .input_format()?
                .int32(c"color-transfer-request".as_ptr())
                != Some(3)
        {
            return Err(VideoError::UnsupportedCodec(format!(
                "{} does not support HDR-to-SDR tone mapping",
                codec.name()
            )));
        }
        log::debug!(
            target: "ferment_engine_media::android",
            "{}: {mime} on {} into {}",
            source.display(),
            codec.name(),
            if surface.is_null() { "CPU buffers" } else { "an image reader" },
        );

        Ok(Self {
            extractor,
            codec,
            format,
            mime,
            sink: match sink {
                Some(reader) => Sink::Surface {
                    reader,
                    released: VecDeque::new(),
                    outstanding: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                },
                None => Sink::Buffers { output: None },
            },
            color: stream.color,
            tone_mapped_color: target_color.filter(|_| tone_map),
            prefetched: None,
            options,
            time_base: TimeBase::MICROSECONDS,
            frame_timing: DecodedTimeline::open(source, stream_index)?,
            seek_target: None,
            input_done: false,
            output_done: false,
            stalls: 0,
            touched: false,
        })
    }

    /// Hand the codec the next coded sample, or the end-of-stream flag.
    fn feed(&mut self) -> Result<()> {
        if self.input_done {
            return Ok(());
        }
        let Some(index) = self.codec.dequeue_input(0) else {
            return Ok(());
        };
        let buffer = self.codec.input_buffer(index)?;
        // SAFETY: the extractor is live with a selected track; the buffer is
        // ours for exactly `buffer.len()` bytes until it is queued.
        let read = unsafe {
            ndk::AMediaExtractor_readSampleData(
                self.extractor.inner,
                buffer.as_mut_ptr(),
                buffer.len(),
            )
        };
        if read < 0 {
            self.input_done = true;
            return self
                .codec
                .queue_input(index, 0, 0, ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
        }
        // SAFETY: the extractor is positioned on the sample just read.
        let (time_us, flags) = unsafe {
            (
                ndk::AMediaExtractor_getSampleTime(self.extractor.inner),
                ndk::AMediaExtractor_getSampleFlags(self.extractor.inner),
            )
        };
        if time_us < 0 {
            // Do not collapse leading negative PTS into zero: recovery must
            // replay the original timeline, including equal-PTS identities.
            return Err(VideoError::UnsupportedCodec(
                "negative video PTS requires the original software timeline".into(),
            ));
        }
        self.touched = true;
        self.codec
            .queue_input(index, read as usize, time_us, flags)?;
        // SAFETY: the extractor is live; advancing past the last sample is
        // what makes the next read report end of stream.
        unsafe { ndk::AMediaExtractor_advance(self.extractor.inner) };
        Ok(())
    }

    /// One decoded picture, or `None` when the stream is exhausted.
    fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        if let Some(frame) = self.prefetched.take() {
            return Ok(Some(frame));
        }
        loop {
            if self.output_done {
                return Ok(None);
            }
            self.feed()?;
            match self.codec.dequeue_output(DEQUEUE_TIMEOUT_US)? {
                Output::TryAgain => {
                    self.stalls += 1;
                    if self.stalls > DECODE_STALL_LIMIT {
                        return Err(VideoError::BackendUnavailable(format!(
                            "the decoder produced nothing for {} ms",
                            DECODE_STALL_LIMIT as i64 * DEQUEUE_TIMEOUT_US / 1_000,
                        )));
                    }
                }
                Output::FormatChanged => {
                    let format = self.codec.output_format()?;
                    if let Some(target) = self.tone_mapped_color {
                        // Acknowledging the request alone is not enough: the
                        // decoded pictures must actually declare SDR BT.709.
                        let observed = unsafe {
                            (
                                format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_STANDARD),
                                format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_TRANSFER),
                                format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_RANGE),
                            )
                        };
                        if observed != (Some(1), Some(3), Some(2)) {
                            return Err(VideoError::UnsupportedCodec(format!(
                                "decoder did not deliver BT.709 SDR after tone-map request: {observed:?}"
                            )));
                        }
                        self.color = target;
                    }
                    log::debug!(
                        target: "ferment_engine_media::android",
                        "decoder output format: {}",
                        format.describe(),
                    );
                    if let Sink::Buffers { output } = &mut self.sink {
                        *output = Some(format);
                    }
                }
                Output::Buffer { index, info } => {
                    self.stalls = 0;
                    let end = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
                    let frame = if info.size > 0 {
                        self.take(index, &info)?
                    } else {
                        self.codec.release_output(index, false)?;
                        None
                    };
                    if end {
                        if matches!(&self.sink, Sink::Surface { released, .. } if !released.is_empty())
                        {
                            return Err(VideoError::BackendUnavailable(
                                "codec EOS arrived with undelivered surface images".into(),
                            ));
                        }
                        self.output_done = true;
                    }
                    if frame.is_some() {
                        return Ok(frame);
                    }
                    if end {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Turn one output buffer into a frame, by whichever route this session
    /// was opened on.
    fn take(
        &mut self,
        index: usize,
        info: &ndk::AMediaCodecBufferInfo,
    ) -> Result<Option<VideoFrame>> {
        let timing = Timing {
            pts: MediaTime::new(info.presentationTimeUs, self.time_base),
            duration: None,
        };
        let color = self.color;
        match &mut self.sink {
            Sink::Surface {
                reader,
                released,
                outstanding,
            } => {
                released.push_back(timing);
                // Render, then acquire. The reader issues images in the order
                // buffers were released to it, so the queue above pairs each
                // picture with the timestamp of the buffer that produced it —
                // `AImage_getTimestamp` reports the surface's clock, which is
                // not the stream's.
                let reader = Arc::clone(reader);
                let outstanding = Arc::clone(outstanding);
                self.codec.release_output(index, true)?;
                let image = reader
                    .acquire(EXPECTED_IMAGE_TIMEOUT, &outstanding)?
                    .ok_or_else(|| {
                        VideoError::BackendUnavailable(
                            "rendered decoder image did not arrive within one second".into(),
                        )
                    })?;
                let Sink::Surface { released, .. } = &mut self.sink else {
                    unreachable!("the sink cannot change under a decode");
                };
                let timing = released.pop_front().unwrap_or(timing);
                image.require_full_crop()?;
                let (width, height) = (image.width(), image.height());
                if color.bits_per_channel.is_some_and(|bits| bits > 8)
                    && image.format() != AIMAGE_FORMAT_YCBCR_P010
                {
                    return Err(VideoError::UnsupportedCodec(
                        "ten-bit decoder returned an8-bit image".into(),
                    ));
                }
                // The last slot is never handed out: see `Sink::Surface`.
                let buffer = if image.outstanding() >= READER_MAX_IMAGES as usize {
                    image.rotation_pixels(color)?
                } else {
                    FrameBuffer::Android(image)
                };
                Ok(Some(VideoFrame {
                    width,
                    height,
                    pts: timing.pts,
                    duration: timing.duration,
                    color,
                    buffer,
                }))
            }
            Sink::Buffers { output } => {
                let format = output.as_ref().ok_or_else(|| {
                    VideoError::InvalidData(
                        "a decoded buffer arrived before the output format".into(),
                    )
                })?;
                require_full_buffer_crop(
                    format.int32(c"width".as_ptr()).unwrap_or(0),
                    format.int32(c"height".as_ptr()).unwrap_or(0),
                    format.int32(c"crop-left".as_ptr()),
                    format.int32(c"crop-top".as_ptr()),
                    format.int32(c"crop-right".as_ptr()),
                    format.int32(c"crop-bottom".as_ptr()),
                )?;
                let frame = if color.bits_per_channel.is_some_and(|bits| bits > 8) {
                    let (width, height, stride, slice_height, pixel_format) = unsafe {
                        (
                            format.int32(ndk::AMEDIAFORMAT_KEY_WIDTH),
                            format.int32(ndk::AMEDIAFORMAT_KEY_HEIGHT),
                            format.int32(ndk::AMEDIAFORMAT_KEY_STRIDE),
                            format.int32(ndk::AMEDIAFORMAT_KEY_SLICE_HEIGHT),
                            format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_FORMAT),
                        )
                    };
                    if pixel_format != Some(android_hdr::P010) {
                        return Err(VideoError::UnsupportedCodec(
                            "ten-bit decoder did not expose P010 ByteBuffers".into(),
                        ));
                    }
                    let (width, height) = (
                        width.unwrap_or(0).max(0) as u32,
                        height.unwrap_or(0).max(0) as u32,
                    );
                    let data = self.codec.output_buffer(index)?;
                    let offset = usize::try_from(info.offset).map_err(|_| {
                        VideoError::InvalidData("negative codec buffer offset".into())
                    })?;
                    let size = usize::try_from(info.size).map_err(|_| {
                        VideoError::InvalidData("negative codec buffer size".into())
                    })?;
                    let end = offset.checked_add(size).ok_or_else(|| {
                        VideoError::InvalidData("codec buffer extent overflow".into())
                    })?;
                    let data = data
                        .get(offset..end)
                        .ok_or_else(|| VideoError::InvalidData("short codec output".into()))?;
                    FrameBuffer::Yuv420(android_hdr::read_p010(
                        data,
                        width,
                        height,
                        stride.unwrap_or(0).max(0) as usize,
                        slice_height.unwrap_or(0).max(0) as usize,
                        color.range,
                    )?)
                } else if android_hdr::is_hdr(color.transfer) {
                    // A genuinely 8-bit PQ/HLG source is legal. Retain its
                    // original YUV codes for later promotion, rather than
                    // applying an RGB8 matrix before transfer conversion.
                    let (width, height, stride, slice_height, pixel_format) = unsafe {
                        (
                            format.int32(ndk::AMEDIAFORMAT_KEY_WIDTH),
                            format.int32(ndk::AMEDIAFORMAT_KEY_HEIGHT),
                            format.int32(ndk::AMEDIAFORMAT_KEY_STRIDE),
                            format.int32(ndk::AMEDIAFORMAT_KEY_SLICE_HEIGHT),
                            format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_FORMAT),
                        )
                    };
                    let data = self.codec.output_buffer(index)?;
                    let offset = usize::try_from(info.offset).map_err(|_| {
                        VideoError::InvalidData("negative codec buffer offset".into())
                    })?;
                    let size = usize::try_from(info.size).map_err(|_| {
                        VideoError::InvalidData("negative codec buffer size".into())
                    })?;
                    let end = offset.checked_add(size).ok_or_else(|| {
                        VideoError::InvalidData("codec buffer extent overflow".into())
                    })?;
                    let data = data
                        .get(offset..end)
                        .ok_or_else(|| VideoError::InvalidData("short codec output".into()))?;
                    FrameBuffer::Yuv420(android_hdr::read_yuv420_8(
                        data,
                        width.unwrap_or(0).max(0) as u32,
                        height.unwrap_or(0).max(0) as u32,
                        stride.unwrap_or(0).max(0) as usize,
                        slice_height.unwrap_or(0).max(0) as usize,
                        pixel_format.unwrap_or(0),
                        color.range,
                    )?)
                } else {
                    return Err(VideoError::UnsupportedCodec(
                        "SDR ByteBuffer output requires libav decoding".into(),
                    ));
                };
                let (width, height) = match &frame {
                    FrameBuffer::Cpu(f) => (f.width, f.height),
                    FrameBuffer::Yuv420(f) => (f.width, f.height),
                    _ => unreachable!(),
                };
                self.codec.release_output(index, false)?;
                Ok(Some(VideoFrame {
                    width,
                    height,
                    pts: timing.pts,
                    duration: timing.duration,
                    color,
                    buffer: frame,
                }))
            }
        }
    }
}

#[async_trait(?Send)]
impl DecodeSession for MediaCodecDecoder {
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        let target = target.rescale(self.time_base);
        if self.seek_target.is_none()
            && (!self.touched || self.prefetched.is_some())
            && target.value <= 0
        {
            return Ok(());
        }
        // Release lookahead before rebuilding the codec or draining its reader.
        self.prefetched = None;
        self.frame_timing.clear();
        self.extractor.seek_backward(target.value)?;
        let surface = match &self.sink {
            Sink::Surface { reader, .. } => reader.window,
            Sink::Buffers { .. } => ptr::null_mut(),
        };
        self.codec.close();
        self.codec = MediaCodec::open_decoder_codec(self.mime, &self.format, surface)?;
        self.touched = false;
        // A flushed codec is stopped until it is started again — not by
        // `AMediaCodec_start`, which would refuse, but by the next queued
        // input. What DOES have to be dropped is everything the old position
        // produced: the timings of buffers released before the flush, whose
        // images the reader may still be holding.
        if let Sink::Surface {
            reader,
            released,
            outstanding,
        } = &mut self.sink
        {
            released.clear();
            let reader = Arc::clone(reader);
            let outstanding = Arc::clone(outstanding);
            while reader
                .acquire(std::time::Duration::from_millis(0), &outstanding)?
                .is_some()
            {}
        }
        self.seek_target = Some(target.value);
        self.input_done = false;
        self.output_done = false;
        self.stalls = 0;
        Ok(())
    }

    async fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        loop {
            let frame = match self.next_frame()? {
                Some(frame) => {
                    let Some(ready) = self.frame_timing.push(frame) else {
                        continue;
                    };
                    ready
                }
                None => {
                    let Some(last) = self.frame_timing.finish() else {
                        return Ok(None);
                    };
                    last
                }
            };
            if let Some(target) = self.seek_target {
                if frame.pts.rescale(self.time_base).value < target {
                    continue;
                }
                self.seek_target = None;
            }
            return Ok(Some(frame));
        }
    }
}

// -------------------------------------------------------------------------
// Video encode

struct MediaCodecEncoder {
    codec: MediaCodec,
    settings: EncoderSettings,
    packets: Vec<EncodedPacket>,
    /// Only unconsumed submissions; codec-config buffers do not consume it.
    submitted_timing: SubmittedTimings,
    /// The `csd-0`/`csd-1` the codec emitted with `BUFFER_FLAG_CODEC_CONFIG`.
    /// MediaCodec hands parameter sets over as Annex-B, so the configuration
    /// this session reports is Annex-B and its packets are too — the same
    /// tolerance `h264_videotoolbox` needs from the libav encoder, and the
    /// muxer converts either.
    description: Vec<u8>,
    /// The encoder's negotiated input geometry, read back after `start` rather
    /// than assumed: `stride` and `slice-height` are the codec's, not ours.
    stride: usize,
    slice_height: usize,
    semi_planar: bool,
    policy: InputPolicy,
}

impl std::fmt::Debug for MediaCodecEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaCodecEncoder")
            .field("codec", &self.settings.codec)
            .field("size", &(self.settings.width, self.settings.height))
            .field("packets", &self.packets.len())
            .finish_non_exhaustive()
    }
}

/// `MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar` — I420, the
/// layout every Android encoder accepts and the one this writes.
const COLOR_FORMAT_YUV420_PLANAR: i32 = 19;

/// `android.media.MediaFormat.KEY_MAX_B_FRAMES`, spelled out.
///
/// Read off the platform jar, not from memory — it is `max-bframes`, with no
/// hyphen before the `b`, and this constant said `max-b-frames` until a review
/// checked it:
/// `javap -constants -classpath $ANDROID_HOME/platforms/*/android.jar
/// android.media.MediaFormat | grep KEY_MAX_B_FRAMES`.
/// MediaCodec ignores a key it does not know, silently, so the wrong spelling
/// made `allow_frame_reordering: false` do nothing at all — which is the exact
/// failure the setting exists to prevent.
///
/// `AMEDIAFORMAT_KEY_MAX_B_FRAMES` is `__INTRODUCED_IN(34)`
/// (`media/NdkMediaFormat.h:187`) and this project's floor is 33, so naming the
/// symbol makes the library fail to LINK — which `cargo ndk check` never sees,
/// because it does not link. A MediaFormat key is a plain string, and this is
/// the string the platform's own constant holds; it is a wire value between us
/// and MediaCodec, which is why it is pinned by a test rather than trusted.
const KEY_MAX_B_FRAMES: &str = "max-bframes";

impl MediaCodecEncoder {
    fn open(settings: EncoderSettings) -> Result<Self> {
        let mime = mime_for(&settings.codec)?;
        let policy = InputPolicy::for_settings(&settings)?;
        let format = Format::new()?;
        // SAFETY: the key statics are immutable NUL-terminated names.
        unsafe {
            format.set_string(ndk::AMEDIAFORMAT_KEY_MIME, mime)?;
            format.set_int32(ndk::AMEDIAFORMAT_KEY_WIDTH, settings.width as i32);
            format.set_int32(ndk::AMEDIAFORMAT_KEY_HEIGHT, settings.height as i32);
            format.set_int32(
                ndk::AMEDIAFORMAT_KEY_COLOR_FORMAT,
                if policy.p010 {
                    android_hdr::P010
                } else {
                    COLOR_FORMAT_YUV420_PLANAR
                },
            );
            // A rate is not optional for an Android encoder: configure without
            // one and `AMediaCodec_configure` refuses with INVALID_PARAMETER.
            format.set_int32(
                ndk::AMEDIAFORMAT_KEY_BIT_RATE,
                settings.bit_rate.unwrap_or(2_000_000).min(i32::MAX as u64) as i32,
            );
            // This is an initial rate-control hint, not a timestamp clock.
            // Keep fractional source cadence (e.g. 30000/1001); MediaCodec
            // follows the actual buffer timestamps for variable-rate input.
            format.set_float(
                ndk::AMEDIAFORMAT_KEY_FRAME_RATE,
                settings.frame_rate_hint() as f32,
            );
            format.set_float(
                ndk::AMEDIAFORMAT_KEY_I_FRAME_INTERVAL,
                settings.keyframe_interval_seconds() as f32,
            );
        }
        if let Some(profile) = policy.profile {
            format.set_int32_by_name("profile", profile)?;
        }
        if !settings.allow_frame_reordering {
            // Delivery order equals presentation order. The scene detectors
            // index frames positionally and would silently mis-associate every
            // result without this.
            format.set_int32_by_name(KEY_MAX_B_FRAMES, 0)?;
        }
        // SAFETY: the key statics are immutable NUL-terminated names.
        unsafe {
            if let Some(color) = settings.color {
                if let Some(standard) = color_standard(color) {
                    format.set_int32(ndk::AMEDIAFORMAT_KEY_COLOR_STANDARD, standard);
                }
                if let Some(transfer) = color_transfer(color) {
                    format.set_int32(ndk::AMEDIAFORMAT_KEY_COLOR_TRANSFER, transfer);
                }
                format.set_int32(
                    ndk::AMEDIAFORMAT_KEY_COLOR_RANGE,
                    if color.range == ColorRange::Full {
                        1
                    } else {
                        2
                    },
                );
            }
        }
        if let Some(metadata) = settings.hdr_static_metadata {
            format.set_buffer(c"hdr-static-info".as_ptr(), metadata.as_bytes());
        }
        let mut failures = Vec::new();
        let codecs = if policy.p010 {
            let color = settings.color.expect("validated HDR color");
            encoder_names(EncoderQuery {
                mime: mime.to_owned(), width: settings.width, height: settings.height,
                bit_rate: settings.bit_rate.unwrap_or(2_000_000).min(i32::MAX as u64) as u32,
                frame_rate: settings.frame_rate_hint(), profile: policy.profile, color_format: android_hdr::P010,
                color_standard: color_standard(color), color_transfer: color_transfer(color),
                color_range: Some(if color.range == ColorRange::Full { 1 } else { 2 }),
            })?.ok_or_else(|| VideoError::UnsupportedCodec("Main10 requires the Android MediaCodecList resolver installed by FermentEngine.start".into()))?
                .into_iter().map(Some).collect::<Vec<_>>()
        } else {
            vec![None]
        };
        for name in codecs {
            // One codec at a time. Rejected candidates release their native
            // session before the next attempt, including start failures.
            let codec = match &name {
                Some(name) => MediaCodec::by_name(name),
                None => MediaCodec::encoder(mime),
            };
            match codec
                .and_then(|codec| Self::configured(settings.clone(), policy, &format, codec))
                .map_err(|error| format!("{}: {error}", name.as_deref().unwrap_or(mime)))
            {
                Ok(encoder) => return Ok(encoder),
                Err(error) => failures.push(error),
            }
        }
        Err(VideoError::UnsupportedCodec(format!(
            "no encoder accepted the required {:?} profile/P010/color: {}",
            policy.profile,
            failures.join("; ")
        )))
    }

    fn configured(
        settings: EncoderSettings,
        policy: InputPolicy,
        format: &Format,
        mut codec: MediaCodec,
    ) -> Result<Self> {
        codec.configure(format, ptr::null_mut(), true)?;
        validate_encoder_output(&codec.output_format()?, policy, &settings)?;
        codec.start()?;
        let input = codec.input_format()?;
        // SAFETY: the key statics are immutable NUL-terminated names.
        let (stride, slice_height, color_format) = unsafe {
            (
                input.int32(ndk::AMEDIAFORMAT_KEY_STRIDE),
                input.int32(ndk::AMEDIAFORMAT_KEY_SLICE_HEIGHT),
                input.int32(ndk::AMEDIAFORMAT_KEY_COLOR_FORMAT),
            )
        };
        // Only the two layouts this writes are accepted. `COLOR_FormatYUV420Flexible`
        // (2 135 033 992) is what many hardware encoders report, and its real
        // layout is only readable through the Java `getInputImage`, which the
        // NDK has no counterpart for — writing I420 into it is a colour-garbage
        // export rather than an error, so it is refused here instead.
        const COLOR_FORMAT_YUV420_SEMI_PLANAR: i32 = 21;
        const COLOR_FORMAT_YUV420_PACKED_SEMI_PLANAR: i32 = 39;
        let semi_planar = match color_format {
            Some(android_hdr::P010) if policy.p010 => true,
            other if policy.p010 => {
                return Err(VideoError::UnsupportedCodec(format!(
                    "Main10 encoder did not negotiate P010 input: {other:?}"
                )));
            }
            Some(COLOR_FORMAT_YUV420_PLANAR) | None => false,
            Some(COLOR_FORMAT_YUV420_SEMI_PLANAR | COLOR_FORMAT_YUV420_PACKED_SEMI_PLANAR) => true,
            Some(other) => {
                return Err(VideoError::UnsupportedCodec(format!(
                    "the encoder wants colour format {other}, which this backend cannot write"
                )));
            }
        };
        log::debug!(
            target: "ferment_engine_media::android",
            "encoding {:?} with {} ({})",
            settings.codec, codec.name(),
            input.describe(),
        );
        Ok(Self {
            settings,
            stride: stride.unwrap_or(0).max(0) as usize,
            slice_height: slice_height.unwrap_or(0).max(0) as usize,
            semi_planar,
            policy,
            codec,
            packets: Vec::new(),
            submitted_timing: SubmittedTimings::default(),
            description: Vec::new(),
        })
    }

    /// Drain whatever the encoder has ready.
    fn collect(&mut self, drain_to_end: bool) -> Result<()> {
        let mut stalls = 0u32;
        loop {
            match self.codec.dequeue_output(DEQUEUE_TIMEOUT_US)? {
                Output::TryAgain => {
                    if !drain_to_end {
                        return Ok(());
                    }
                    stalls += 1;
                    if stalls > DECODE_STALL_LIMIT {
                        return Err(VideoError::BackendUnavailable(
                            "the encoder stopped producing packets after the flush".into(),
                        ));
                    }
                }
                Output::FormatChanged => {
                    // csd-0/csd-1 also arrive here, as the format's buffers,
                    // on the devices that do not send a CODEC_CONFIG buffer.
                    let format = self.codec.output_format()?;
                    validate_encoder_output(&format, self.policy, &self.settings)?;
                    if self.description.is_empty() {
                        self.description = codec_specific_data(&format);
                    }
                }
                Output::Buffer { index, info } => {
                    stalls = 0;
                    let end = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
                    let config = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG != 0;
                    if info.size > 0 {
                        let data = {
                            let buffer = self.codec.output_buffer(index)?;
                            let start = info.offset.max(0) as usize;
                            let end = start.saturating_add(info.size.max(0) as usize);
                            buffer
                                .get(start..end)
                                .ok_or_else(|| {
                                    VideoError::InvalidData(
                                        "the encoder declared more bytes than it produced".into(),
                                    )
                                })?
                                .to_vec()
                        };
                        if config {
                            self.description = data;
                        } else {
                            // Buffer-info has no duration. Match the returned
                            // codec timestamp to its original submitted timing,
                            // including when the encoder reorders output. Do not
                            // truncate µs into the caller's clock a second time.
                            let timing = self.submitted_timing.take(info.presentationTimeUs)?;
                            self.packets.push(EncodedPacket {
                                stream_index: 0,
                                pts: Some(timing.pts),
                                dts: None,
                                duration: timing.duration,
                                is_sync: info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_KEY_FRAME != 0,
                                data,
                                packetization: Packetization::AnnexB,
                            });
                        }
                    }
                    self.codec.release_output(index, false)?;
                    if end {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Write one frame into an input buffer in the encoder's own layout.
    fn submit(&mut self, frame: &VideoFrame) -> Result<()> {
        // Prepare an owned picture before taking a codec input buffer. Failure
        // never queues8-bit data with HDR tags, and each call owns at most one
        // source/scaled frame, not a clip-sized decoded-frame collection.
        let p010 = if self.policy.p010 {
            let color = self.settings.color.expect("validated HDR color");
            Some(
                frame
                    .encodable_yuv420(self.settings.width, self.settings.height, color)?
                    .owned_yuv420()?,
            )
        } else {
            None
        };
        if !self.policy.p010 && android_hdr::is_hdr(frame.color.transfer) {
            return Err(VideoError::UnsupportedCodec(
                "SDR encoder requires tone-mapped SDR pictures".into(),
            ));
        }
        let rgb = if p010.is_none() {
            let rgb = frame.to_cpu_rgb()?;
            Some(
                if (rgb.width, rgb.height) != (self.settings.width, self.settings.height) {
                    rgb.scaled_to(self.settings.width, self.settings.height)?
                } else {
                    rgb
                },
            )
        } else {
            None
        };
        let width = self.settings.width as usize;
        let height = self.settings.height as usize;
        let index = self
            .codec
            .dequeue_input(DEQUEUE_TIMEOUT_US * 100)
            .ok_or_else(|| {
                VideoError::BackendUnavailable("the encoder took no input for a second".into())
            })?;
        // On Codec2, allocation of the first buffer publishes its real byte
        // pitch. Read it after dequeue, not before buffer allocation.
        let (stride, slice_height) = if self.policy.p010 {
            let input = self.codec.input_format()?;
            let (format, stride, slice) = unsafe {
                (
                    input.int32(ndk::AMEDIAFORMAT_KEY_COLOR_FORMAT),
                    input.int32(ndk::AMEDIAFORMAT_KEY_STRIDE),
                    input.int32(ndk::AMEDIAFORMAT_KEY_SLICE_HEIGHT),
                )
            };
            if format != Some(android_hdr::P010) {
                return Err(VideoError::UnsupportedCodec(
                    "encoder changed P010 input format".into(),
                ));
            }
            (
                stride.unwrap_or(0).max(0) as usize,
                slice.unwrap_or(0).max(0) as usize,
            )
        } else {
            (
                if self.stride == 0 { width } else { self.stride },
                if self.slice_height == 0 {
                    height
                } else {
                    self.slice_height
                },
            )
        };
        let buffer = self.codec.input_buffer(index)?;
        let needed = if let Some(frame) = p010 {
            android_hdr::write_p010(&frame, buffer, stride, slice_height)?
        } else {
            if [width, height, stride, slice_height]
                .iter()
                .any(|extent| !extent.is_multiple_of(2))
                || stride < width
                || slice_height < height
            {
                return Err(VideoError::InvalidData(
                    "4:2:0 encode needs valid even extents/stride/slice-height".into(),
                ));
            }
            let needed = stride
                .checked_mul(slice_height)
                .and_then(|n| n.checked_add(n / 2))
                .ok_or_else(|| VideoError::InvalidData("encoder input extent overflow".into()))?;
            if buffer.len() < needed {
                return Err(VideoError::InvalidData(format!(
                    "encoder input is shorter than {needed} bytes"
                )));
            }
            write_i420(
                rgb.as_ref().expect("SDR picture"),
                buffer,
                stride,
                slice_height,
                self.semi_planar,
            );
            needed
        };
        let presentation_time_us = frame.pts.rescale(TimeBase::MICROSECONDS).value.max(0);
        self.codec
            .queue_input(index, needed, presentation_time_us, 0)?;
        self.submitted_timing
            .record(presentation_time_us, frame, self.settings.time_base)?;
        self.collect(false)
    }
}

fn validate_encoder_output(
    format: &Format,
    policy: InputPolicy,
    settings: &EncoderSettings,
) -> Result<()> {
    policy.validate_output(
        settings,
        format.int32(c"profile".as_ptr()),
        unsafe { format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_STANDARD) },
        unsafe { format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_TRANSFER) },
        unsafe { format.int32(ndk::AMEDIAFORMAT_KEY_COLOR_RANGE) },
    )?;
    if let Some(expected) = settings.hdr_static_metadata {
        let actual = format.buffer(c"hdr-static-info".as_ptr(), 25)?;
        if actual.as_deref() != Some(expected.as_bytes().as_slice()) {
            return Err(VideoError::UnsupportedCodec(
                "encoder did not preserve the source HDR static descriptor".into(),
            ));
        }
    }
    Ok(())
}

/// Pack an RGB frame into an encoder input buffer as I420 (or NV12).
///
/// BT.709 limited range, which is what `EncoderSettings::color` defaults to on
/// every path that reaches here and what an untagged H.264 stream is read as.
fn write_i420(
    rgb: &CpuFrame,
    buffer: &mut [u8],
    stride: usize,
    slice_height: usize,
    semi_planar: bool,
) {
    let width = rgb.width as usize;
    let height = rgb.height as usize;
    let luma_len = stride * slice_height;
    let chroma_stride = if semi_planar { stride } else { stride / 2 };
    let chroma_len = chroma_stride * slice_height.div_ceil(2);

    for row in 0..height {
        for column in 0..width {
            let at = row * rgb.stride + column * 3;
            let r = f32::from(rgb.data[at]);
            let g = f32::from(rgb.data[at + 1]);
            let b = f32::from(rgb.data[at + 2]);
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            buffer[row * stride + column] =
                (16.0 + y * 219.0 / 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    for row in 0..height.div_ceil(2) {
        for column in 0..width.div_ceil(2) {
            // One 2x2 block averaged, which is what a 4:2:0 downsample is.
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut taken = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let (y, x) = (row * 2 + dy, column * 2 + dx);
                    if y >= height || x >= width {
                        continue;
                    }
                    let at = y * rgb.stride + x * 3;
                    r += f32::from(rgb.data[at]);
                    g += f32::from(rgb.data[at + 1]);
                    b += f32::from(rgb.data[at + 2]);
                    taken += 1.0;
                }
            }
            let taken = taken.max(1.0);
            let (r, g, b) = (r / taken, g / taken, b / taken);
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let cb = (128.0 + (b - y) * 0.5389 * 224.0 / 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let cr = (128.0 + (r - y) * 0.6350 * 224.0 / 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            if semi_planar {
                let at = luma_len + row * chroma_stride + column * 2;
                buffer[at] = cb;
                buffer[at + 1] = cr;
            } else {
                buffer[luma_len + row * chroma_stride + column] = cb;
                buffer[luma_len + chroma_len + row * chroma_stride + column] = cr;
            }
        }
    }
}

/// `csd-0` and `csd-1` concatenated — the parameter sets, in the Annex-B form
/// MediaCodec states them in.
fn codec_specific_data(format: &Format) -> Vec<u8> {
    let mut description = Vec::new();
    // SAFETY: the key statics are immutable NUL-terminated names.
    for key in unsafe { [ndk::AMEDIAFORMAT_KEY_CSD_0, ndk::AMEDIAFORMAT_KEY_CSD_1] } {
        let mut data: *mut c_void = ptr::null_mut();
        let mut length = 0usize;
        // SAFETY: live format; both out-pointers are valid. The buffer belongs
        // to the format and outlives this read.
        let found = unsafe { ndk::AMediaFormat_getBuffer(format.0, key, &mut data, &mut length) };
        if !found || data.is_null() || length == 0 {
            continue;
        }
        // SAFETY: the NDK reported `length` readable bytes.
        description
            .extend_from_slice(unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) });
    }
    description
}

/// `MediaFormat.COLOR_STANDARD_*`, for the colorimetry a caller stated.
fn color_standard(color: ColorInfo) -> Option<i32> {
    use crate::ColorMatrix::*;
    match color.matrix {
        Bt709 => Some(1),
        // 2 is BT601_PAL and 4 is BT601_NTSC: same primaries family, different
        // primaries. Read off the platform jar.
        Bt470Bg => Some(2),
        Smpte170M => Some(4),
        Bt2020NonConstant | Bt2020Constant => Some(6),
        _ => None,
    }
}

/// `MediaFormat.COLOR_TRANSFER_*`.
fn color_transfer(color: ColorInfo) -> Option<i32> {
    use crate::TransferCharacteristic::*;
    match color.transfer {
        Linear => Some(1),
        Bt709 | Smpte170M | Bt2020TenBit | Bt2020TwelveBit => Some(3),
        Srgb => Some(2),
        Pq => Some(6),
        Hlg => Some(7),
        _ => None,
    }
}

#[async_trait(?Send)]
impl EncodeSession for MediaCodecEncoder {
    async fn push_frame(&mut self, frame: VideoFrame) -> Result<()> {
        self.submit(&frame)
    }

    /// `finalize` consumes `Box<Self>`, so there is no "already finalized"
    /// state a caller can reach and no flag to keep.
    async fn finalize(mut self: Box<Self>) -> Result<EncodedVideo> {
        // The end-of-stream flag on an empty input buffer is the flush. Without
        // it the encoder keeps its lookahead and the tail of every clip is
        // missing.
        let index = self
            .codec
            .dequeue_input(DEQUEUE_TIMEOUT_US * 100)
            .ok_or_else(|| {
                VideoError::BackendUnavailable("the encoder took no end-of-stream buffer".into())
            })?;
        self.codec
            .queue_input(index, 0, 0, ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM)?;
        self.collect(true)?;
        self.submitted_timing.finish()?;
        if self.policy.p010 {
            android_hdr::require_main10_sps(&self.description)?;
        }
        let configuration = DecoderConfiguration::from_codec_private(
            self.settings.codec.clone(),
            std::mem::take(&mut self.description),
        )?;
        Ok(EncodedVideo {
            configuration,
            packets: std::mem::take(&mut self.packets),
        })
    }
}

// -------------------------------------------------------------------------
// Audio

/// The Android compressed-audio backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct AndroidAudioIo;

#[async_trait(?Send)]
impl AudioIo for AndroidAudioIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        crate::demux::probe(source)
    }

    async fn open_audio_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn AudioDecodeSession>> {
        // Use the shipped CAF/AIFF demuxers for the iOS music picker's formats
        // instead of depending on a platform extractor. This preserves PCM
        // layout and CAF packet-table timing. Inspect bytes through the probe:
        // imported blob paths have no suffix.
        let info = crate::demux::probe(source)?;
        if matches!(&info.container, crate::ContainerFormat::Other(name) if name == "caf" || name == "aiff")
        {
            return LibavAudioIo.open_audio_decoder(source, stream_index).await;
        }
        match MediaCodecAudioDecoder::open(source, stream_index) {
            Ok(decoder) => Ok(Box::new(decoder)),
            Err(error @ VideoError::UnsupportedCodec(_)) => {
                log::debug!(
                    target: "ferment_engine_media::android",
                    "{}: {error}; decoding through libavcodec instead",
                    source.display(),
                );
                LibavAudioIo.open_audio_decoder(source, stream_index).await
            }
            Err(error) => Err(error),
        }
    }

    /// MediaCodec first, libavcodec for what it refuses — the same fallback the
    /// decoder has, and for the same reason. Without it, taking the platform
    /// default on Android NARROWED what could be encoded: `LibavAudioIo` was
    /// the backend before this module existed, and it reaches whatever AAC
    /// encoder the FFmpeg build carries.
    async fn open_audio_encoder(
        &self,
        settings: AudioEncoderSettings,
    ) -> Result<Box<dyn AudioEncodeSession>> {
        match MediaCodecAudioEncoder::open(settings) {
            Ok(encoder) => Ok(Box::new(encoder)),
            Err(error @ VideoError::UnsupportedCodec(_)) => {
                log::debug!(
                    target: "ferment_engine_media::android",
                    "{error}; encoding audio through libavcodec instead",
                );
                LibavAudioIo.open_audio_encoder(settings).await
            }
            Err(error) => Err(error),
        }
    }
}

/// `AudioFormat.ENCODING_PCM_16BIT` and `ENCODING_PCM_FLOAT`.
const ENCODING_PCM_16BIT: i32 = 2;
const ENCODING_PCM_FLOAT: i32 = 4;

struct MediaCodecAudioDecoder {
    extractor: Extractor,
    codec: MediaCodec,
    /// As on the video side: a seek rebuilds the codec from these.
    format: Format,
    mime: &'static str,
    sample_rate: u32,
    channels: u16,
    /// Whatever the codec settled on, read off the OUTPUT format rather than
    /// asked for: a decoder that ignores `KEY_PCM_ENCODING` hands back 16-bit
    /// PCM, and reading it as float is noise.
    pcm_encoding: i32,
    /// Where the next frame starts, in samples. The container's timestamps set
    /// the origin and nothing else — counting samples cannot drift, because
    /// samples are the unit.
    next_start: Option<i64>,
    /// Where the audible program ends, in samples — the same declaration the
    /// libav backend truncates to.
    ///
    /// The LEADING trim is not applied here, and that was measured rather than
    /// assumed: dropping the container's `leading_samples` as well left
    /// `aac-lc-48000-stereo.m4a` 896 samples short of the Mac's 48 000 —
    /// 49 152 coded, less 1 024 of priming MediaExtractor/MediaCodec had
    /// ALREADY dropped, less another 1 024 dropped here. So this backend is in
    /// exactly the libav decoder's position: the platform consumes the head
    /// trim itself and reports no end discard, and the only trim left to do is
    /// the one below.
    audible_limit: Option<i64>,
    seek_target: Option<i64>,
    input_done: bool,
    output_done: bool,
    stalls: u32,
    /// As on the video side: an untouched session is already at the beginning.
    touched: bool,
}

impl std::fmt::Debug for MediaCodecAudioDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaCodecAudioDecoder")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("pcm_encoding", &self.pcm_encoding)
            .finish_non_exhaustive()
    }
}

impl MediaCodecAudioDecoder {
    fn open(source: &Path, stream_index: u32) -> Result<Self> {
        let info = crate::demux::probe(source)?;
        let stream = stream_of(&info, stream_index)?;
        if stream.kind != MediaKind::Audio {
            return Err(VideoError::InvalidData(format!(
                "stream {stream_index} is not audio"
            )));
        }
        let mime = mime_for(&stream.codec)?;
        let (extractor, format) = Extractor::open(source, &stream, &info)?;
        // Ask for float. A decoder that cannot is not an error — the output
        // format below says which arrived.
        // SAFETY: the key static is an immutable NUL-terminated name.
        format.set_int32(
            unsafe { ndk::AMEDIAFORMAT_KEY_PCM_ENCODING },
            ENCODING_PCM_FLOAT,
        );
        let codec = MediaCodec::open_decoder_codec(mime, &format, ptr::null_mut())?;

        let timing = stream.audio_timing.unwrap_or_default();
        Ok(Self {
            extractor,
            codec,
            format,
            mime,
            sample_rate: stream.sample_rate.unwrap_or(0),
            channels: stream.channels.unwrap_or(0),
            pcm_encoding: ENCODING_PCM_16BIT,
            next_start: None,
            audible_limit: timing
                .audible_samples
                .and_then(|samples| i64::try_from(samples).ok()),
            seek_target: None,
            input_done: false,
            output_done: false,
            stalls: 0,
            touched: false,
        })
        .and_then(|decoder| decoder.validated())
    }

    fn validated(self) -> Result<Self> {
        if self.sample_rate == 0 || self.channels == 0 {
            return Err(VideoError::InvalidData(
                "audio stream declares no rate or channels".into(),
            ));
        }
        Ok(self)
    }

    fn rate_base(&self) -> TimeBase {
        TimeBase::new_unchecked(1, i64::from(self.sample_rate))
    }

    fn feed(&mut self) -> Result<()> {
        if self.input_done {
            return Ok(());
        }
        let Some(index) = self.codec.dequeue_input(0) else {
            return Ok(());
        };
        let buffer = self.codec.input_buffer(index)?;
        // SAFETY: the extractor is live with a selected track.
        let read = unsafe {
            ndk::AMediaExtractor_readSampleData(
                self.extractor.inner,
                buffer.as_mut_ptr(),
                buffer.len(),
            )
        };
        if read < 0 {
            self.input_done = true;
            return self
                .codec
                .queue_input(index, 0, 0, ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
        }
        // SAFETY: as above.
        let (time_us, flags) = unsafe {
            (
                ndk::AMediaExtractor_getSampleTime(self.extractor.inner),
                ndk::AMediaExtractor_getSampleFlags(self.extractor.inner),
            )
        };
        self.touched = true;
        self.codec
            .queue_input(index, read as usize, time_us.max(0), flags)?;
        // SAFETY: as above.
        unsafe { ndk::AMediaExtractor_advance(self.extractor.inner) };
        Ok(())
    }

    /// Interleaved native-endian f32 out of one output buffer, whichever PCM
    /// encoding the codec settled on.
    fn interleaved(&self, index: usize, info: &ndk::AMediaCodecBufferInfo) -> Result<Vec<f32>> {
        let buffer = self.codec.output_buffer(index)?;
        let start = info.offset.max(0) as usize;
        let end = start.saturating_add(info.size.max(0) as usize);
        let bytes = buffer.get(start..end).ok_or_else(|| {
            VideoError::InvalidData("the decoder declared more bytes than it produced".into())
        })?;
        Ok(match self.pcm_encoding {
            ENCODING_PCM_FLOAT => bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect(),
            // 16-bit signed, native-endian. The scale is 1/32768 rather than
            // 1/32767: it is the one that maps the full negative code exactly,
            // and it is what every decoder here divides by.
            _ => bytes
                .chunks_exact(2)
                .map(|word| f32::from(i16::from_le_bytes([word[0], word[1]])) / 32_768.0)
                .collect(),
        })
    }
}

#[async_trait(?Send)]
impl AudioDecodeSession for MediaCodecAudioDecoder {
    async fn seek(&mut self, target: MediaTime) -> Result<()> {
        let target_sample = target.rescale(self.rate_base()).value;
        // Land BEFORE the target and decode into it: a compressed format's
        // first frame after a flush is not yet right — AAC's window overlaps
        // its predecessor, MP3's polyphase filter carries state — so seeking
        // straight to the target returns samples a sequential decode did not
        // produce there.
        //
        // Four MP3 granule pairs — 1 152 samples is the longer of the two coded
        // frame lengths this decodes (AAC's is 1 024), so four of them is at
        // least four coded frames either way. Two was measured to be too few on
        // `cuts-36`: an AAC seek with two frames of pre-roll came back 0.049
        // from a sequential decode, because the target still landed inside the
        // first frame the rebuilt codec produced — the one it consumes before
        // it emits anything.
        let preroll = 4 * MP3_GRANULE_PAIR_SAMPLES;
        let landing = MediaTime::new((target_sample - preroll).max(0), self.rate_base())
            .rescale(TimeBase::MICROSECONDS);
        if !self.touched && landing.value <= 0 && target_sample <= 0 {
            return Ok(());
        }
        self.extractor.seek_backward(landing.value)?;
        self.codec.close();
        self.codec = MediaCodec::open_decoder_codec(self.mime, &self.format, ptr::null_mut())?;
        self.touched = false;
        self.seek_target = Some(target_sample);
        self.next_start = None;
        self.input_done = false;
        self.output_done = false;
        self.stalls = 0;
        Ok(())
    }

    async fn decode_next(&mut self) -> Result<Option<PcmAudioFrame>> {
        loop {
            if self.output_done {
                return Ok(None);
            }
            self.feed()?;
            let (index, info) = match self.codec.dequeue_output(DEQUEUE_TIMEOUT_US)? {
                Output::TryAgain => {
                    self.stalls += 1;
                    if self.stalls > DECODE_STALL_LIMIT {
                        return Err(VideoError::BackendUnavailable(format!(
                            "the audio decoder produced nothing for {} ms",
                            DECODE_STALL_LIMIT as i64 * DEQUEUE_TIMEOUT_US / 1_000,
                        )));
                    }
                    continue;
                }
                Output::FormatChanged => {
                    let format = self.codec.output_format()?;
                    // SAFETY: the key statics are immutable NUL-terminated names.
                    unsafe {
                        self.pcm_encoding = format
                            .int32(ndk::AMEDIAFORMAT_KEY_PCM_ENCODING)
                            .unwrap_or(ENCODING_PCM_16BIT);
                        // HE-AAC decodes at the SBR output rate, which is not
                        // the core rate the container leads with — the output
                        // format is the only place that number appears.
                        if let Some(rate) = format.int32(ndk::AMEDIAFORMAT_KEY_SAMPLE_RATE)
                            && rate > 0
                        {
                            self.sample_rate = rate as u32;
                        }
                        if let Some(channels) = format.int32(ndk::AMEDIAFORMAT_KEY_CHANNEL_COUNT)
                            && channels > 0
                        {
                            self.channels = channels as u16;
                        }
                    }
                    continue;
                }
                Output::Buffer { index, info } => (index, info),
            };
            self.stalls = 0;
            self.touched = true;
            // The decoder's own channel count, read after every format change
            // rather than once at open: an HE-AAC v2 stream is a mono core the
            // decoder hands back as parametric stereo, and a 5.1 track may be
            // downmixed. Hoisting the probe's number made every later
            // `next_start` advance by the wrong stride.
            let channels = usize::from(self.channels);
            let end = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
            if info.size <= 0 {
                self.codec.release_output(index, false)?;
                if end {
                    self.output_done = true;
                    return Ok(None);
                }
                continue;
            }
            let samples = self.interleaved(index, &info)?;
            self.codec.release_output(index, false)?;
            if end {
                self.output_done = true;
            }
            if samples.is_empty() || channels == 0 {
                continue;
            }

            let frame_samples = (samples.len() / channels) as i64;
            // Where this decode sits on the audible timeline. From the first
            // frame onwards samples are counted rather than converted, because
            // counting cannot drift — a Matroska track timed in milliseconds
            // rounds every conversion, and consecutive frames stop meeting.
            //
            // The origin differs by how the decode began, and both halves were
            // measured on `cuts-36`:
            //
            // * after an OPEN it is zero. The program starts at its start, and
            //   that is the timeline `AudioTiming::audible_samples` is
            //   expressed in. Taking the first buffer's timestamp instead cost
            //   `mp3-vbr-44100-stereo.mp3` 1 774 samples off its end — the
            //   extractor reports that file's first sample at 40 ms, and the
            //   audible limit then truncated 40 ms early.
            // * after a SEEK it is the buffer's own timestamp, because only the
            //   container knows where `AMediaExtractor_seekTo` landed.
            //   Assuming zero there made the discard below count from the
            //   landing point rather than from the start of the file, and the
            //   audio came back a whole pre-roll late: samples differing from a
            //   sequential decode by 0.72, against a tolerance of 0.01.
            let start = self.next_start.unwrap_or_else(|| {
                if self.seek_target.is_none() {
                    return 0;
                }
                // ROUNDED, not truncated. `MediaTime::rescale` divides, and a
                // sample position that went out as microseconds comes back one
                // sample short whenever the division had a remainder — at
                // 48 kHz one sample is 20.83 µs, so most positions do. The
                // frame then reports the sample it was asked for while
                // carrying the audio of its neighbour: measured on `cuts-36`,
                // a seek a quarter-second into a 440 Hz tone came back 0.0488
                // away from a sequential decode, which is exactly one sample
                // of phase at that frequency.
                let rate = i64::from(self.sample_rate);
                (info.presentationTimeUs.max(0) * rate + 500_000) / 1_000_000
            });
            self.next_start = Some(start + frame_samples);

            if let Some(target) = self.seek_target {
                let end = start + frame_samples;
                if end <= target {
                    continue;
                }
                self.seek_target = None;
                if start < target {
                    let drop = ((target - start) as usize) * channels;
                    self.next_start = Some(end);
                    return PcmAudioFrame::new(
                        self.sample_rate,
                        self.channels,
                        MediaTime::new(target, self.rate_base()),
                        samples[drop..].to_vec(),
                    )
                    .map(Some);
                }
            }

            // Stop at the declared end of the program: the encoder's padding
            // is coded audio, but it is not the recording.
            let samples = match self.audible_limit {
                Some(limit) if start >= limit => return Ok(None),
                Some(limit) if start + frame_samples > limit => {
                    samples[..((limit - start) as usize) * channels].to_vec()
                }
                _ => samples,
            };
            if samples.is_empty() {
                return Ok(None);
            }
            return PcmAudioFrame::new(
                self.sample_rate,
                self.channels,
                MediaTime::new(start, self.rate_base()),
                samples,
            )
            .map(Some);
        }
    }
}

struct MediaCodecAudioEncoder {
    codec: MediaCodec,
    settings: AudioEncoderSettings,
    /// Interleaved f32 waiting for a full coded frame.
    pending: Vec<f32>,
    packets: Vec<EncodedPacket>,
    description: Vec<u8>,
    pcm_encoding: i32,
    next_pts: i64,
    pushed: u64,
}

impl std::fmt::Debug for MediaCodecAudioEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaCodecAudioEncoder")
            .field("settings", &self.settings)
            .field("packets", &self.packets.len())
            .finish_non_exhaustive()
    }
}

/// AAC-LC's coded frame, and its encoder delay: one frame, always.
const AAC_FRAME_SAMPLES: usize = 1_024;
/// MP3's coded frame at every rate this decodes — the longer of the two, and
/// so the unit a seek's pre-roll is counted in.
const MP3_GRANULE_PAIR_SAMPLES: i64 = 1_152;
/// `MediaCodecInfo.CodecProfileLevel.AACObjectLC`.
const AAC_OBJECT_LC: i32 = 2;

impl MediaCodecAudioEncoder {
    fn open(settings: AudioEncoderSettings) -> Result<Self> {
        let mime = mime_for(&Codec::Aac)?;
        let mut codec = MediaCodec::encoder(mime)?;
        let format = Format::new()?;
        // SAFETY: the key statics are immutable NUL-terminated names.
        unsafe {
            format.set_string(ndk::AMEDIAFORMAT_KEY_MIME, mime)?;
            format.set_int32(
                ndk::AMEDIAFORMAT_KEY_SAMPLE_RATE,
                settings.sample_rate as i32,
            );
            format.set_int32(
                ndk::AMEDIAFORMAT_KEY_CHANNEL_COUNT,
                i32::from(settings.channels),
            );
            format.set_int32(ndk::AMEDIAFORMAT_KEY_BIT_RATE, settings.bit_rate as i32);
            format.set_int32(ndk::AMEDIAFORMAT_KEY_AAC_PROFILE, AAC_OBJECT_LC);
            format.set_int32(ndk::AMEDIAFORMAT_KEY_PCM_ENCODING, ENCODING_PCM_16BIT);
        }
        codec.configure(&format, ptr::null_mut(), true)?;
        codec.start()?;
        log::debug!(
            target: "ferment_engine_media::android",
            "encoding audio with {}",
            codec.name(),
        );
        Ok(Self {
            codec,
            settings,
            pending: Vec::new(),
            packets: Vec::new(),
            description: Vec::new(),
            pcm_encoding: ENCODING_PCM_16BIT,
            next_pts: 0,
            pushed: 0,
        })
    }

    fn collect(&mut self, drain_to_end: bool) -> Result<()> {
        let base = TimeBase::new_unchecked(1, i64::from(self.settings.sample_rate));
        let mut stalls = 0u32;
        loop {
            match self.codec.dequeue_output(DEQUEUE_TIMEOUT_US)? {
                Output::TryAgain => {
                    if !drain_to_end {
                        return Ok(());
                    }
                    stalls += 1;
                    if stalls > DECODE_STALL_LIMIT {
                        return Err(VideoError::BackendUnavailable(
                            "the audio encoder stopped producing packets after the flush".into(),
                        ));
                    }
                }
                Output::FormatChanged => {
                    let format = self.codec.output_format()?;
                    if self.description.is_empty() {
                        self.description = codec_specific_data(&format);
                    }
                }
                Output::Buffer { index, info } => {
                    stalls = 0;
                    let end = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0;
                    let config = info.flags & ndk::AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG != 0;
                    if info.size > 0 {
                        let data = {
                            let buffer = self.codec.output_buffer(index)?;
                            let start = info.offset.max(0) as usize;
                            let end = start.saturating_add(info.size.max(0) as usize);
                            buffer
                                .get(start..end)
                                .ok_or_else(|| {
                                    VideoError::InvalidData(
                                        "the encoder declared more bytes than it produced".into(),
                                    )
                                })?
                                .to_vec()
                        };
                        if config {
                            self.description = data;
                        } else {
                            self.packets.push(EncodedPacket {
                                stream_index: 0,
                                pts: Some(
                                    MediaTime::new(info.presentationTimeUs, TimeBase::MICROSECONDS)
                                        .rescale(base),
                                ),
                                dts: None,
                                duration: Some(MediaTime::new(AAC_FRAME_SAMPLES as i64, base)),
                                is_sync: true,
                                data,
                                packetization: Packetization::Elementary,
                            });
                        }
                    }
                    self.codec.release_output(index, false)?;
                    if end {
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Hand the encoder one coded frame's worth, in the PCM layout it takes.
    fn submit(&mut self, block: &[f32]) -> Result<()> {
        let index = self
            .codec
            .dequeue_input(DEQUEUE_TIMEOUT_US * 100)
            .ok_or_else(|| {
                VideoError::BackendUnavailable("the audio encoder took no input".into())
            })?;
        let bytes_per_sample = if self.pcm_encoding == ENCODING_PCM_FLOAT {
            4
        } else {
            2
        };
        let needed = block.len() * bytes_per_sample;
        let buffer = self.codec.input_buffer(index)?;
        if buffer.len() < needed {
            return Err(VideoError::InvalidData(format!(
                "the audio encoder's input buffer holds {} bytes, short of {needed}",
                buffer.len()
            )));
        }
        for (slot, sample) in buffer.chunks_exact_mut(bytes_per_sample).zip(block) {
            if bytes_per_sample == 4 {
                slot.copy_from_slice(&sample.to_le_bytes());
            } else {
                let value = (sample.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
                slot.copy_from_slice(&value.to_le_bytes());
            }
        }
        let presentation_time_us = MediaTime::new(
            self.next_pts,
            TimeBase::new_unchecked(1, i64::from(self.settings.sample_rate)),
        )
        .rescale(TimeBase::MICROSECONDS)
        .value;
        self.codec
            .queue_input(index, needed, presentation_time_us, 0)?;
        self.next_pts += (block.len() / usize::from(self.settings.channels)) as i64;
        self.collect(false)
    }
}

#[async_trait(?Send)]
impl AudioEncodeSession for MediaCodecAudioEncoder {
    async fn push_pcm(&mut self, frame: PcmAudioFrame) -> Result<()> {
        if frame.sample_rate != self.settings.sample_rate
            || frame.channels != self.settings.channels
        {
            return Err(VideoError::InvalidData(format!(
                "encoder takes {} Hz / {} channels, got {} Hz / {}",
                self.settings.sample_rate,
                self.settings.channels,
                frame.sample_rate,
                frame.channels,
            )));
        }
        self.pushed += frame.frame_count() as u64;
        self.pending.extend_from_slice(&frame.samples);
        let chunk = AAC_FRAME_SAMPLES * usize::from(self.settings.channels);
        while self.pending.len() >= chunk {
            let block: Vec<f32> = self.pending.drain(..chunk).collect();
            self.submit(&block)?;
        }
        Ok(())
    }

    async fn finalize(mut self: Box<Self>) -> Result<EncodedAudio> {
        let channels = usize::from(self.settings.channels);
        let chunk = AAC_FRAME_SAMPLES * channels;
        if !self.pending.is_empty() {
            // The remainder is a partial coded frame; AAC pads it, and that
            // padding is what `trailing_samples` accounts for.
            let mut block: Vec<f32> = std::mem::take(&mut self.pending);
            block.resize(chunk, 0.0);
            self.submit(&block)?;
        }
        let index = self
            .codec
            .dequeue_input(DEQUEUE_TIMEOUT_US * 100)
            .ok_or_else(|| {
                VideoError::BackendUnavailable(
                    "the audio encoder took no end-of-stream buffer".into(),
                )
            })?;
        self.codec
            .queue_input(index, 0, 0, ndk::AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM)?;
        self.collect(true)?;

        let configuration = DecoderConfiguration::from_codec_private(
            Codec::Aac,
            std::mem::take(&mut self.description),
        )?;
        // AAC-LC's encoder delay is one coded frame. MediaCodec reports it
        // nowhere — there is no `initial_padding` on this API — so it is the
        // codec's own constant rather than a number read back, and the
        // priming/remainder law is the same: everything coded is priming plus
        // the audible program plus whatever padded the last frame.
        let leading = AAC_FRAME_SAMPLES as u64;
        let coded = self.packets.len() as u64 * AAC_FRAME_SAMPLES as u64;
        let audible = self.pushed;
        let trailing = coded.saturating_sub(leading).saturating_sub(audible);
        Ok(EncodedAudio {
            configuration,
            packets: std::mem::take(&mut self.packets),
            timing: AudioTiming {
                leading_samples: leading as u32,
                trailing_samples: u32::try_from(trailing).unwrap_or(0),
                audible_samples: Some(audible),
            },
        })
    }
}

// -------------------------------------------------------------------------
// Mux

/// MPEG-4 output through `AMediaMuxer`.
///
/// Three things this writer cannot do, all refused rather than approximated:
///
/// * **QuickTime.** `AMediaMuxer_new` takes `AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4`,
///   `_WEBM`, `_THREE_GPP`, `_HEIF` or `_OGG`. There is no `mov`, and writing
///   an MP4 when a caller asked for a MOV is the silent relabel the libav
///   muxer refuses for ProRes.
/// * **`faststart`.** MediaMuxer writes `moov` when `stop()` is called and
///   offers no second pass. An export that quietly lost fast-start is found by
///   a player seeking to the end of a file meant to stream, not by us.
/// * **A movie timescale.** There is no key for it.
///
/// [`LibavMuxer`](crate::platform::libav::LibavMuxer) does all three and is
/// linked into this same binary; this writer is the platform container for the
/// callers that need MediaCodec's own output written without a second library
/// in the path.
#[derive(Debug, Clone, Copy, Default)]
pub struct AndroidMuxer;

impl Muxer for AndroidMuxer {
    fn open(&self, output: &Path, settings: MuxSettings) -> Result<Box<dyn MuxSession>> {
        Ok(Box::new(AndroidMuxSession::open(output, settings)?))
    }
}

struct MuxTrack {
    index: usize,
    declared: Option<MediaTime>,
    packet_end: Option<MediaTime>,
}

struct AndroidMuxSession {
    muxer: *mut ndk::AMediaMuxer,
    /// Held for the muxer's life: `AMediaMuxer_new` does not dup.
    _sink: Fd,
    path: PathBuf,
    tracks: Vec<MuxTrack>,
    started: bool,
    finalized: bool,
}

// SAFETY: the muxer is owned solely by this session and never shared; the
// `MuxSession` trait requires `Send` so a session can be moved onto the
// blocking thread that writes the file.
unsafe impl Send for AndroidMuxSession {}

impl std::fmt::Debug for AndroidMuxSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AndroidMuxSession")
            .field("path", &self.path)
            .field("tracks", &self.tracks.len())
            .finish_non_exhaustive()
    }
}

impl Drop for AndroidMuxSession {
    fn drop(&mut self) {
        if self.muxer.is_null() {
            return;
        }
        // SAFETY: sole owner. A session dropped without `finalize` leaves a
        // file with no `moov`, which is what an abandoned export should be.
        unsafe {
            if self.started && !self.finalized {
                ndk::AMediaMuxer_stop(self.muxer);
            }
            ndk::AMediaMuxer_delete(self.muxer);
        }
        self.muxer = ptr::null_mut();
    }
}

impl AndroidMuxSession {
    fn open(output: &Path, settings: MuxSettings) -> Result<Self> {
        if settings.format != MuxFormat::Mp4 {
            return Err(VideoError::UnsupportedContainer(
                "AMediaMuxer writes MPEG-4; QuickTime output is libav's".into(),
            ));
        }
        if settings.fast_start {
            return Err(VideoError::InvalidData(
                "AMediaMuxer cannot fast-start: it writes moov at stop() and has no second pass"
                    .into(),
            ));
        }
        if settings.movie_timescale.is_some() {
            return Err(VideoError::InvalidData(
                "AMediaMuxer has no movie-timescale control".into(),
            ));
        }
        let sink = Fd::create(output)?;
        // SAFETY: the descriptor is open for the whole of the muxer's life
        // (it is owned by the struct built below).
        let muxer = unsafe {
            ndk::AMediaMuxer_new(sink.0, ndk::OutputFormat::AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4)
        };
        if muxer.is_null() {
            return Err(VideoError::BackendUnavailable(
                "could not allocate an AMediaMuxer".into(),
            ));
        }
        Ok(Self {
            muxer,
            _sink: sink,
            path: output.to_owned(),
            tracks: Vec::new(),
            started: false,
            finalized: false,
        })
    }

    fn add_track(&mut self, format: &Format, declared: Option<MediaTime>) -> Result<MuxTrackId> {
        if self.started {
            return Err(VideoError::InvalidData(
                "cannot add a track after the first packet was written".into(),
            ));
        }
        // SAFETY: the muxer is live and not started; the format outlives the
        // call, which copies what it needs.
        let index = unsafe { ndk::AMediaMuxer_addTrack(self.muxer, format.0) };
        if index < 0 {
            return Err(VideoError::platform("AMediaMuxer_addTrack", index as i32));
        }
        let id = MuxTrackId::new(u32::try_from(self.tracks.len()).unwrap_or(u32::MAX));
        self.tracks.push(MuxTrack {
            index: index as usize,
            declared,
            packet_end: None,
        });
        Ok(id)
    }

    /// Start once every track exists. MediaMuxer, like libavformat, requires
    /// every track to be declared before the first sample.
    fn ensure_started(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        if self.tracks.is_empty() {
            return Err(VideoError::InvalidData(
                "cannot write a container with no tracks".into(),
            ));
        }
        // SAFETY: the muxer is live with all its tracks declared.
        check_media("AMediaMuxer_start", unsafe {
            ndk::AMediaMuxer_start(self.muxer)
        })?;
        self.started = true;
        Ok(())
    }
}

/// The `AMediaFormat` a track description becomes.
fn mux_video_format(track: &VideoMuxTrack) -> Result<Format> {
    let format = Format::new()?;
    let mime = mime_for(&track.configuration.codec)?;
    // SAFETY: the key statics are immutable NUL-terminated names.
    unsafe {
        format.set_string(ndk::AMEDIAFORMAT_KEY_MIME, mime)?;
        format.set_int32(ndk::AMEDIAFORMAT_KEY_WIDTH, track.width as i32);
        format.set_int32(ndk::AMEDIAFORMAT_KEY_HEIGHT, track.height as i32);
        if let Some(standard) = color_standard(track.color) {
            format.set_int32(ndk::AMEDIAFORMAT_KEY_COLOR_STANDARD, standard);
        }
        if let Some(transfer) = color_transfer(track.color) {
            format.set_int32(ndk::AMEDIAFORMAT_KEY_COLOR_TRANSFER, transfer);
        }
        if track.color.range != ColorRange::Unspecified {
            format.set_int32(
                ndk::AMEDIAFORMAT_KEY_COLOR_RANGE,
                if track.color.range == ColorRange::Full {
                    1
                } else {
                    2
                },
            );
        }
        // csd-0/csd-1 are Annex-B parameter sets, which is what the muxer
        // writes an `avcC` from. An avcC record handed over as csd is not
        // understood; the parameter sets it holds are.
        let sets = annex_b_parameter_sets(&track.configuration);
        match track.configuration.codec {
            // H.264 wants SPS in csd-0 and PPS in csd-1.
            Codec::H264 if sets.len() >= 2 => {
                format.set_buffer(ndk::AMEDIAFORMAT_KEY_CSD_0, &sets[0]);
                format.set_buffer(ndk::AMEDIAFORMAT_KEY_CSD_1, &sets[1]);
            }
            // HEVC wants VPS+SPS+PPS concatenated in csd-0.
            _ if !sets.is_empty() => {
                format.set_buffer(ndk::AMEDIAFORMAT_KEY_CSD_0, &sets.concat());
            }
            _ => {}
        }
    }
    Ok(format)
}

fn mux_audio_format(track: &AudioMuxTrack) -> Result<Format> {
    let format = Format::new()?;
    let mime = mime_for(&track.configuration.codec)?;
    // SAFETY: the key statics are immutable NUL-terminated names.
    unsafe {
        format.set_string(ndk::AMEDIAFORMAT_KEY_MIME, mime)?;
        format.set_int32(ndk::AMEDIAFORMAT_KEY_SAMPLE_RATE, track.sample_rate as i32);
        format.set_int32(
            ndk::AMEDIAFORMAT_KEY_CHANNEL_COUNT,
            i32::from(track.channels),
        );
        if !track.configuration.description.is_empty() {
            // For AAC csd-0 IS the AudioSpecificConfig, verbatim.
            format.set_buffer(
                ndk::AMEDIAFORMAT_KEY_CSD_0,
                &track.configuration.description,
            );
        }
    }
    Ok(format)
}

/// The parameter sets of a decoder configuration, each as its own buffer with
/// a four-byte start code — the form MediaMuxer's `csd-*` keys take.
fn annex_b_parameter_sets(configuration: &DecoderConfiguration) -> Vec<Vec<u8>> {
    if !configuration.parameter_sets.is_empty() {
        return configuration
            .parameter_sets
            .iter()
            .map(|set| {
                let mut unit = vec![0, 0, 0, 1];
                unit.extend_from_slice(&set.bytes);
                unit
            })
            .collect();
    }
    // Already Annex-B: hand it over whole rather than re-splitting it.
    if configuration.description.starts_with(&[0, 0, 0, 1])
        || configuration.description.starts_with(&[0, 0, 1])
    {
        return vec![configuration.description.clone()];
    }
    Vec::new()
}

impl MuxSession for AndroidMuxSession {
    fn add_video_track(&mut self, track: VideoMuxTrack) -> Result<MuxTrackId> {
        // AMediaMuxer cannot represent an arbitrary preferred transform.
        // The production export backend is LibavMuxer; this limited backend
        // must refuse metadata it cannot preserve before adding any track.
        crate::mux::require_identity_display_matrix(track.display_matrix)?;
        let format = mux_video_format(&track)?;
        self.add_track(&format, track.duration)
    }

    fn add_audio_track(&mut self, track: AudioMuxTrack) -> Result<MuxTrackId> {
        let time_base = TimeBase::new(1, i64::from(track.sample_rate))?;
        let declared = track
            .timing
            .audible_samples
            .and_then(|samples| i64::try_from(samples).ok())
            .map(|samples| MediaTime::new(samples, time_base));
        let format = mux_audio_format(&track)?;
        self.add_track(&format, declared)
    }

    fn write_packet(&mut self, track: MuxTrackId, packet: EncodedPacket) -> Result<()> {
        self.ensure_started()?;
        let index = track.get() as usize;
        let entry = self
            .tracks
            .get(index)
            .ok_or_else(|| VideoError::InvalidData(format!("unknown mux track {}", track.get())))?;
        let muxer_track = entry.index;
        // MediaMuxer takes Annex-B and writes the length prefixes itself — the
        // mirror of the libav muxer, which takes length-prefixed NAL units and
        // converts Annex-B on the way in.
        let data = match packet.packetization {
            Packetization::LengthPrefixed { .. } => packet.data_in_format(PacketFormat::AnnexB)?,
            _ => packet.data,
        };
        if data.is_empty() {
            return Err(VideoError::InvalidData(
                "cannot write an empty sample".into(),
            ));
        }

        // SAFETY: a zeroed BufferInfo is the documented shape.
        let mut info: ndk::AMediaCodecBufferInfo = unsafe { std::mem::zeroed() };
        info.offset = 0;
        info.size = i32::try_from(data.len())
            .map_err(|_| VideoError::InvalidData("sample exceeds the muxer's i32 size".into()))?;
        info.presentationTimeUs = packet
            .pts
            .map_or(0, |pts| pts.rescale(TimeBase::MICROSECONDS).value);
        info.flags = if packet.is_sync {
            ndk::AMEDIACODEC_BUFFER_FLAG_KEY_FRAME
        } else {
            0
        };

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

        // SAFETY: the muxer is started, the track index is one of its own, and
        // `data` outlives the call, which copies it.
        check_media("AMediaMuxer_writeSampleData", unsafe {
            ndk::AMediaMuxer_writeSampleData(self.muxer, muxer_track, data.as_ptr(), &info)
        })
    }

    fn finalize(mut self: Box<Self>) -> Result<MuxOutput> {
        self.ensure_started()?;
        // SAFETY: the muxer is live and started; this writes the moov.
        check_media("AMediaMuxer_stop", unsafe {
            ndk::AMediaMuxer_stop(self.muxer)
        })?;
        self.finalized = true;
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
            path: Some(self.path.clone()),
            // MediaMuxer writes to a descriptor and hands nothing back; an
            // in-memory session is `LibavMuxer::memory`'s.
            bytes: None,
            duration,
            track_count: self.tracks.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: preserve container rotation in MediaCodec's surface format, or
    /// recreate seek from the extractor's original rotated format.
    #[test]
    #[ignore = "requires the pinned rotated fixture on an actual Android device; run exact with --include-ignored"]
    fn coded_decoder_retains_cleared_rotation_across_actual_seek() {
        let source=std::env::var_os("FERMENT_ROTATED_TEST_VIDEO").map(PathBuf::from)
            .expect("pin FERMENT_ROTATED_TEST_VIDEO to the original rotated48x32 fixture on this Android device");
        let info = crate::demux::probe(&source).unwrap();
        let stream = info.video_streams().next().unwrap();
        assert!(
            stream.display_matrix.is_some(),
            "requires the real container matrix"
        );
        let mut decoder = MediaCodecDecoder::open_with_color(
            &source,
            stream.index,
            DecodeOptions::default(),
            true,
            None,
        )
        .unwrap();
        let key = CString::new("rotation-degrees").unwrap();
        assert_eq!(decoder.format.int32(key.as_ptr()), Some(0));
        futures_executor::block_on(async {
            let first = decoder.decode_next().await.unwrap().unwrap();
            assert_eq!(
                (first.width, first.height),
                (stream.width.unwrap(), stream.height.unwrap())
            );
            decoder.seek(first.pts).await.unwrap();
            assert_eq!(decoder.format.int32(key.as_ptr()), Some(0));
            let again = decoder.decode_next().await.unwrap().unwrap();
            assert_eq!((again.width, again.height), (first.width, first.height));
            assert_eq!(again.pts, first.pts);
        });
    }

    /// Every codec this platform names, and every one it does not. A MIME
    /// typo is a codec the device silently cannot create, so the mapping is
    /// pinned rather than trusted.
    ///
    /// KILL: spell `video/hevc` as `video/x-hevc` (the name the WebM family
    /// uses) — `AMediaCodec_createDecoderByType` returns null on every device
    /// and every HEVC file falls through to the software decoder.
    #[test]
    fn every_named_codec_has_the_platforms_own_mime_type() {
        assert_eq!(mime_for(&Codec::H264).unwrap(), "video/avc");
        assert_eq!(mime_for(&Codec::Hevc).unwrap(), "video/hevc");
        assert_eq!(mime_for(&Codec::Aac).unwrap(), "audio/mp4a-latm");
        assert_eq!(mime_for(&Codec::Mp3).unwrap(), "audio/mpeg");
        assert_eq!(mime_for(&Codec::Vp9).unwrap(), "video/x-vnd.on2.vp9");
        // No Android device has ever shipped a ProRes codec, and a MIME type
        // for one would send the file to a codec that cannot be created
        // instead of to the libav fallback.
        assert!(mime_for(&Codec::ProRes(crate::ProResProfile::Hq)).is_err());
        assert!(mime_for(&Codec::Other("dnxhd".into())).is_err());
    }

    /// The `csd-*` form: parameter sets, each with its own start code — not
    /// the avcC record the container carries them in.
    ///
    /// KILL: hand `configuration.description` straight to `csd-0` for an avcC
    /// input; MediaMuxer writes a track whose `avcC` is an avcC nested inside
    /// an avcC, and nothing decodes it.
    #[test]
    fn parameter_sets_reach_the_muxer_as_annex_b_units() {
        let avcc = vec![
            1, 100, 0, 31, 0xff, 0xe1, 0, 3, 0x67, 0x64, 0x00, 1, 0, 2, 0x68, 0xee,
        ];
        let configuration = DecoderConfiguration::from_codec_private(Codec::H264, avcc).unwrap();
        let sets = annex_b_parameter_sets(&configuration);
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0], vec![0, 0, 0, 1, 0x67, 0x64, 0x00]);
        assert_eq!(sets[1], vec![0, 0, 0, 1, 0x68, 0xee]);
    }

    /// An encoder's own Annex-B extradata is handed over whole. Splitting it
    /// into units and re-prefixing them would be the same bytes; passing it
    /// through says so without depending on that being true.
    #[test]
    fn annex_b_extradata_is_passed_through_unchanged() {
        let annex_b = vec![0, 0, 0, 1, 0x67, 0xaa, 0, 0, 0, 1, 0x68, 0xbb];
        let configuration =
            DecoderConfiguration::from_codec_private(Codec::H264, annex_b.clone()).unwrap();
        assert_eq!(annex_b_parameter_sets(&configuration), vec![annex_b]);
    }

    /// The colour tags a caller states become `MediaFormat`'s integers, and
    /// the ones the platform has no number for become nothing rather than a
    /// wrong number.
    ///
    /// KILL: map `Pq` and `Hlg` to the same value — an HDR10 export is tagged
    /// HLG and every player tone-maps it with the wrong curve.
    #[test]
    fn colour_tags_map_to_the_platforms_own_numbers() {
        use crate::{ColorMatrix, TransferCharacteristic};
        let hlg = ColorInfo {
            matrix: ColorMatrix::Bt2020NonConstant,
            transfer: TransferCharacteristic::Hlg,
            ..ColorInfo::default()
        };
        assert_eq!(color_standard(hlg), Some(6));
        assert_eq!(color_transfer(hlg), Some(7));
        let pq = ColorInfo {
            transfer: TransferCharacteristic::Pq,
            ..hlg
        };
        assert_eq!(color_transfer(pq), Some(6));
        // Unspecified stays unspecified: a guess written into the container is
        // worse than nothing, because a player trusts what it reads.
        assert_eq!(color_standard(ColorInfo::default()), None);
        assert_eq!(color_transfer(ColorInfo::default()), None);
    }
}
