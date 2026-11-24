//! Speaker diarization: pyannote/segmentation-3.0 (MIT) for who speaks when,
//! ReDimNet2-B2 for who they are.
//!
//! Ported from `Processors/Pyannote/` (`d0a690c86`). The chunk geometry, the
//! powerset table and the agglomerative-clustering shape are adapted from
//! FluidAudio (Apache-2.0) — see `notices/THIRD_PARTY_NOTICES.md` beside this
//! crate's manifest. The whole-frame chunk step, the global frame grid, the
//! marginals, the per-frame confidence, the turn assembler and the two-phase
//! fingerprinting are Ferment's.

pub mod powerset;
mod segmenter;
pub mod stitcher;
pub mod turns;
pub mod windows;

#[cfg(test)]
mod oracle_tests;

pub use segmenter::{PyannoteSpeakerSegmenter, SAME_VOICE_THRESHOLD};
