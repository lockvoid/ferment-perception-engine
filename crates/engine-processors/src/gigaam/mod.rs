//! On-device Russian speech transcription using GigaAM-v3-e2e-rnnt
//! (SberDevices / salute-developers, MIT).
//!
//! Four graphs plus a 1024-piece SentencePiece vocabulary:
//!
//! ```text
//! preprocessor  audio [1, 240000] f32        -> mel [1, 64, 1499] f32
//! encoder       mel [1, 64, 1499] f32,
//!               mask [1, 1, 375] f32         -> encoded [1, 768, 375] f32
//! decoder       x [1, 1] i32,
//!               h_in / c_in [1, 1, 320] f32  -> dec [1, 1, 320] f32,
//!                                               h_out / c_out f32
//! joint         enc [1, 768, 1] f32,
//!               dec [1, 320, 1] f32          -> token_id [1,1,1] i32,
//!                                               token_prob [1,1,1] f32
//! ```
//!
//! The encoder's `mask` is 1 on the chunk's real encoder frames and 0 after;
//! the host counts them from the chunk's real mel frames
//! (`config::encoder_mask`).
//!
//! Nothing here probes the model description: `ferment-models-export`
//! (`models/gigaam/model.json`) pins every feature name, shape and dtype.

pub mod config;
pub mod decoder;
pub mod postprocess;
mod processor;
pub mod vocab;

pub use processor::{GigaAm, LANGUAGE};
