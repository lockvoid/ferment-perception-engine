//! Shot-boundary detection — port of AutoShot (CVPR NAS 2023). Renamed
//! from iOS `SceneDetector` to `VideoSceneDetector` for naming clarity
//! (audio-side equivalents would also benefit from the prefix).
//!
//! Pipeline:
//! 1. Pre-extracted 27×48 RGB frames in (T, H, W, RGB) uint8 layout
//! 2. Pad: 25 prefix + N + (suffix to align with 50-stride) + 25
//! 3. For each 100-frame batch with stride=50:
//!    - 9-bit color histogram (3 bits per channel), L2-normalized
//!    - AutoShot supernet (Inference) → one_hot + many_hot logits
//!    - Keep middle 50 outputs (Python `predictions[25:75]`)
//! 4. Sigmoid + threshold one_hot → cut frames
//! 5. Build scenes (cut frame belongs to outgoing scene; next starts at +1)

pub mod config;
pub mod embedding_tap;
pub mod histograms;
pub mod jump_cut;
pub mod photometric;
pub mod scenes;
pub mod shot_identity;
pub mod transient;
pub mod transient_verdict;

mod processor;

#[cfg(test)]
pub(crate) mod synthetic;

pub use embedding_tap::{Sample, SceneEmbeddingTap};
pub use jump_cut::{Detection as JumpCut, JumpCutDetector};
pub use processor::{
    Cut, Decoded, Scene, VideoSceneDetectionResult, VideoSceneDetector, apply_transients,
};
pub use shot_identity::{EncoderShotIdentity, ShotIdentifying, TransientCollector};
pub use transient::Run as TransientRun;
pub use transient_verdict::Verdict as TransientVerdict;
