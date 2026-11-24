//! Audio semantic encoder backed by Dasheng-Base (Apache 2.0, 86M params,
//! 768-dim embeddings), cut at its front-end: the host computes each
//! 10-second window's log-mel (`config::MEL_SPEC`) and hands it over in patch
//! layout; the graph — `dasheng_base_mel` — keeps the transformer and the
//! time-pool. One contract with iOS, exported by `ferment-models-export`.
//!
//! Pairs naturally with `MusicTagger` (genre/mood/instrument heads on top
//! of the same embedding).

pub mod chunking;
pub mod config;
pub mod frontend;
pub mod pool;

mod processor;

pub use processor::{AudioEmbedding, AudioSemanticEncoder};
