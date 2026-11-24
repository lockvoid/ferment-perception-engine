//! On-device speech transcription via Parakeet-TDT-0.6B-v3 — the `parakeet`
//! contract (`ferment-models-export`, `models/parakeet/model.json`).
//!
//! Pipeline:
//! 1. Decode → resample to 16 kHz mono
//! 2. Overlapping 15 s windows, 25 encoder frames of overlap
//! 3. Per window:
//!    - the host's mel, each band normalised over the valid frames
//!    - `parakeet_encoder` — `encoded [1, 640, 1, 188]`
//!    - the greedy TDT decode on the host over `parakeet_decode.bin`, from
//!      zero states
//! 4. Merge the windows at their seams (the shared token seam merge)
//! 5. Build words from tokens (SentencePiece word boundaries)

pub mod config;
pub mod decoder;
pub mod host_decode;
pub mod postprocess;
pub mod speech_segment_coalescer;
pub mod vocab;

#[cfg(test)]
mod oracle_tests;
mod processor;

pub use processor::{Caption, Parakeet, TranscribedWord, TranscriptionResult, features};
