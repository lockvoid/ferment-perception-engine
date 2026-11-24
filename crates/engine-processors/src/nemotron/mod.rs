//! Streaming dictation on Nemotron 3.5 ASR.
//!
//! Ported from `Processors/Nemotron/`.

pub mod config;
pub mod live_session;
pub mod models;
pub mod processor;
pub mod stream;

#[cfg(test)]
mod oracle_tests;

pub use live_session::NemotronLiveSession;
pub use models::NemotronModels;
pub use processor::NemotronTranscriber;
pub use stream::{NemotronStream, NemotronToken, split_transcript, words_of};
