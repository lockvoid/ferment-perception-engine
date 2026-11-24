//! Audio minimap composer — single image summary of an audio track for
//! vision-LLM consumption. Three stacked panels (mel spectrogram, RMS
//! envelope, time axis) sharing a common x-extent so any vertical band
//! lines up across panels.
//!
//! Layout matches iOS pixel-for-pixel where possible. The renderer uses
//! tiny-skia (CPU rasterizer with SIMD on x86/aarch64); rendering output
//! won't be byte-identical to CoreGraphics but is structurally equivalent.

pub mod colormap;
pub mod layout;
pub mod mel;
pub mod renderer;
pub mod rms;
pub(crate) mod text;

#[cfg(test)]
mod oracle_tests;
mod processor;

pub use processor::{
    AudioMinimapComposer, AudioMinimapComposerConfig, AudioMinimapResult, ImageFormat,
};
