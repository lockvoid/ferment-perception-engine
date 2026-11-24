//! Codec backends.
//!
//! `libav` is the backend: it opens and writes every container, and decodes
//! and encodes everything the platform does not do better itself. `macos`
//! holds the two things it does — the VideoToolbox decompression and
//! compression sessions, which pipeline where libavcodec's hwaccel waits for
//! every frame (amendment A1). `android` is the same shape over the NDK media
//! APIs: MediaCodec decodes and encodes on the phone's own silicon,
//! MediaExtractor feeds it and MediaMuxer writes containers. Both read their
//! stream description from libavformat, which stays the one probe.
pub mod libav;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "android")]
pub mod android;

// The metadata seam is compiled on hosts so its timing/ownership laws run
// without an Android codec. Production use remains Android-only.
#[cfg(any(target_os = "android", test))]
mod codec_timing;

#[cfg(any(target_os = "android", test))]
mod android_hdr;

#[cfg(target_os = "android")]
pub mod android_codec_selection;

#[cfg(any(target_os = "android", test))]
mod android_decode_fallback;
