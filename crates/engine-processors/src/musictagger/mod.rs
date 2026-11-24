//! Music Tagger — three trained MLP heads (genre/mood/instrument) on top
//! of a 768-dim Dasheng-Base embedding from `AudioSemanticEncoder`.
//!
//! `music_tagger_heads`, exported by `ferment-models-export`
//! (`models/music_tagger/`): `embedding [1, 768]` → `genre_probs [1, 87]`,
//! `mood_probs [1, 8]`, `instrument_probs [1, 13]`, sigmoid applied, the same
//! door on every back-end. Vocab + per-tag thresholds are the contract's
//! other half and are bundled at compile time.

pub mod vocab;

mod processor;

pub use processor::{
    AxisProbabilities, MusicTagger, MusicTagsResult, RawTagProbabilities, ScoredTag,
};
