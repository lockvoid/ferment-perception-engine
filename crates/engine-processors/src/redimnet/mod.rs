//! On-device speaker embedding using ReDimNet2-B2 (PalabraAI, MIT), the `vox2`
//! large-margin checkpoint — 192-d, L2-normalised, one prediction per fixed
//! 2.005 s window of 16 kHz mono.
//!
//! The `redimnet_speaker` contract (`ferment-models-export`,
//! `models/redimnet/model.json`), the same door iOS calls.

mod encoder;
pub mod windows;

pub use encoder::{
    MEL_BANDS, MEL_SPEC, PRE_EMPHASIS, ReDimNetSpeakerEncoder, SIGNAL_EPSILON,
    SpeakerSegmentEmbedding, features, signal,
};
