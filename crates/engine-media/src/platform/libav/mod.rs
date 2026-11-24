//! The FFmpeg libav* backend.
//!
//! One demuxer, one muxer, one decode path and one encode path for every
//! container and codec this app opens — behind the traits in [`crate::io`],
//! [`crate::audio`] and [`crate::mux`], which do not change. Consumers keep
//! constructing a backend and calling `probe`/`open_decoder`/`open_encoder`;
//! what changed underneath is that `libavformat` reads the container and
//! `libavcodec` drives VideoToolbox or Media Foundation.
//!
//! The libraries are LGPL-2.1-or-later shared objects we build, pin and bundle
//! (`scripts/provision-ffmpeg`); nothing here links a GPL component and
//! nothing shells out to an `ffmpeg` executable.

mod audio;
mod decode;
mod float;
// Dead on macOS, which encodes through its own VideoToolbox session
// (amendment A1) — but compiled and TESTED there against
// `h264_videotoolbox`, so the Windows encode path is not shipped on the
// strength of a cross-compile alone.
#[cfg_attr(target_os = "macos", allow(dead_code))]
mod encode;
mod error;
mod hw;
mod input;
mod log;
mod mux;
mod packets;
mod probe;
mod proxy;
mod remux_audio;

pub use audio::LibavAudioIo;
pub use decode::LibavVideoIo;
pub use error::check_av;
pub use mux::LibavMuxer;
pub use packets::PacketReader;
pub use probe::{probe, probe_stream};
pub use proxy::{
    ProxyReport, ProxyThumb, ProxyThumbs, proxy_extent, thumb_extent, write_proxy,
    write_proxy_with_thumbs,
};
pub use remux_audio::{AudioPassthroughPacket, AudioPassthroughReader, AudioPassthroughTrack};

use std::sync::OnceLock;

/// Initialize libav exactly once per process: register everything
/// `ffmpeg-next` wants registered, then take libav's log stream away from
/// stderr and put it behind the `log` facade.
///
/// Every entry point into this backend calls it, so a consumer normally does
/// not have to. It is public for the one case that benefits: a host that
/// wants libav's lines in its log before the first media call. Cheap after
/// the first call and safe from any thread.
pub fn init() -> crate::Result<()> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        ffmpeg_next::init().map_err(|error| error.to_string())?;
        log::install();
        Ok(())
    })
    .clone()
    .map_err(|message| {
        crate::VideoError::BackendUnavailable(format!("libav initialization failed: {message}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin, asserted rather than assumed. `scripts/provision-ffmpeg`
    /// vendors FFmpeg 8.1.x, whose libavformat is major 62 — and the whole
    /// backend is written against that ABI. A box that provisioned something
    /// else fails here, in one line, instead of somewhere inside a decode.
    #[test]
    fn links_the_pinned_libavformat_major() {
        init().unwrap();
        // SAFETY: a pure version accessor; it reads a compile-time constant.
        let packed = unsafe { ffmpeg_next::ffi::avformat_version() };
        let (major, minor, micro) = (packed >> 16, (packed >> 8) & 0xff, packed & 0xff);
        println!("libavformat {major}.{minor}.{micro}");
        assert_eq!(major, 62, "expected the FFmpeg 8.1 ABI (libavformat 62)");
    }
}
