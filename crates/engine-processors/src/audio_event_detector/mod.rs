//! FireRedVAD-AED audio-event detection.
//!
//! The model is the `fireredvad_aed` door (`ferment-models-export`): the host
//! computes kaldi fbank ([`config::MEL_SPEC`]) and the checkpoint's [`Cmvn`],
//! the graph the DFSMN and its sigmoid.

mod cmvn;
pub mod config;
#[cfg(test)]
mod oracle_tests;
mod processor;
mod segmenter;

pub use cmvn::{Cmvn, INVERSE_STDDEV, MEANS};
pub use processor::AudioEventDetector;
pub use segmenter::{AudioEventSegmenter, AudioEventSegments, Postprocessor, SegmenterConfig};
