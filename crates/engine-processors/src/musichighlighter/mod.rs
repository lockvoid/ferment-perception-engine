//! Music Highlighter — finds the most highlight-worthy windows in a track.
//!
//! Pipeline (mirrors `MusicHighlighter.swift`):
//! 1. Decode audio → resample 22.05 kHz mono
//! 2. Mel spectrogram (128 bins, log(1 + 1e4·mel))
//! 3. Chunk into 3-second segments (129 mel frames each)
//! 4. Build positional encoding (256-dim)
//! 5. CNN+attention model → softmax → per-chunk attention scores
//! 6. Expand to per-second saliency, normalize 0..1
//! 7. Best-window search per requested length

pub mod config;
pub mod mel;
pub mod pos_enc;
pub mod window_search;

#[cfg(test)]
mod oracle_tests;
mod processor;

pub use processor::{Highlight, HighlightResult, MusicHighlighter};
