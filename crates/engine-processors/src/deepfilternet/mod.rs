//! DeepFilterNet3 speech denoiser.
//!
//! Pipeline (matches `DeepFilterNet.swift` from iOS):
//! 1. Load audio → resample to 48 kHz mono
//! 2. STFT (Vorbis window, fft=960, hop=480), streamed across the whole input
//! 3. Per 1500-frame chunk, from that chunk alone: ERB and spec features with
//!    fresh running normalisations, the lookahead shift, the streaming door
//!    (`ferment-models-export`, `models/dfnet`, 100 frames a call, its state
//!    handed from call to call) → 32-band mask, 5×96 complex coefficients,
//!    lsnr; then the ERB mask, the deep filter and the low/high merge
//! 4. ISTFT, streamed, front delay trimmed

pub mod config;
pub mod enhance;
pub mod events_mask;
pub mod features;

mod processor;

pub use events_mask::apply_events_mask;
pub use processor::{ChunkOutputs, DeepFilterNet, DenoiseOutput};

#[cfg(test)]
mod oracle_tests;

#[cfg(test)]
mod trace_tests;
