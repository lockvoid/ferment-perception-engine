//! Frame- and sample-accurate media I/O.
//!
//! `libavformat` reads and writes every container; `libavcodec` decodes and
//! encodes, driving the operating system's licensed codec APIs where they are
//! faster than its own — VideoToolbox's decompression and compression
//! sessions on macOS, NVENC/NVDEC then Media Foundation on Windows. The
//! libraries are LGPL-2.1-or-later shared objects the app bundles
//! (`scripts/provision-ffmpeg`).

mod audio;
mod error;
mod frame;
mod frame_ycbcr;
mod hdr;
mod input;
mod io;
mod media;
mod mux;
mod packet;
mod time;

pub mod demux;
pub mod platform;

pub use audio::{
    AudioBackend, AudioDecodeSession, AudioEncodeSession, AudioEncoderSettings, AudioIo,
    EncodedAudio, PcmAudioFrame, UnavailableAudioIo,
};
pub use error::{MediaError, Result, VideoError};
#[cfg(target_os = "macos")]
pub use frame::MacPixelBuffer;
pub use frame::{CpuFrame, CpuPixelFormat, FrameBuffer, VideoFrame, Yuv420Format, Yuv420Frame};
pub use frame_ycbcr::YcbcrFrame;
pub use input::MediaInput;
pub use io::{
    DecodeAcceleration, DecodeOptions, DecodeOutput, DecodeSession, EncodeSession, EncodedVideo,
    EncoderSettings, HdrStaticMetadata, UnavailableVideoIo, VideoBackend, VideoIo,
};
pub use media::{
    AacProfile, AudioTiming, Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange,
    ContainerFormat, DecoderConfiguration, MediaInfo, MediaKind, NalUnit, Packetization,
    ProResProfile, StreamInfo, TransferCharacteristic,
};
pub use mux::{
    AudioMuxTrack, MuxBackend, MuxFormat, MuxOutput, MuxSession, MuxSettings, MuxTrackId, Muxer,
    VideoMuxTrack,
};
pub use packet::{EncodedPacket, PacketFormat, SampleInfo, SampleTable};
pub use time::{MediaTime, TimeBase};

// Native packet-preserving soundtrack lane; no decoded audio crosses this door.
pub use platform::libav::{AudioPassthroughPacket, AudioPassthroughReader, AudioPassthroughTrack};
