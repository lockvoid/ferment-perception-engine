//! Sparse subject measurements over a video's scenes: plan -> grab -> measure
//! -> reduce. Port of the Swift `VideoSaliencyAnalyzer` family.
//!
//! Upstream runs Apple Vision, which is unreachable off Apple platforms. The
//! structure, coordinates and reduction math are ported exactly; the heat map
//! and the face boxes come from ONNX backings behind
//! [`ferment_engine_core::SaliencyHeatSource`] and
//! [`ferment_engine_core::FaceRectSource`]. Every constant that was calibrated
//! against Vision's response is re-measured against the substitute — see
//! `config.rs` and `scripts/exports/video_saliency/measure_pedestal.py`.
//!
//! Coordinates are normalized to the display-oriented frame with a TOP-LEFT
//! origin: x right, y down.

pub mod config;
pub mod measure;
pub mod reduce;
pub mod sample_plan;

mod analyzer;
mod grabber;
pub(crate) use grabber::rotation_degrees;
pub mod u2netp;
pub mod yunet;

pub use analyzer::VideoSaliencyAnalyzer;
pub use config::VideoSaliencyConfig;
pub use grabber::{display_size, maximum_size};
pub use sample_plan::SampleEntry;
pub use u2netp::U2NetpHeatSource;
pub use yunet::YuNetFaceSource;

use ferment_engine_core::SaliencyMode;
use serde::{Deserialize, Serialize};

/// What produced a sample's numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleSource {
    Face,
    Saliency,
}

/// One grabbed frame's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoSaliencySample {
    pub scene_index: usize,
    /// The plan's requested time, not the decoded frame's presentation time.
    pub time: f64,
    /// Subject centroid.
    pub cx: f64,
    pub cy: f64,
    /// Subject extent.
    pub w: f64,
    pub h: f64,
    /// Faces: the dominant face's detector score, 0.75..1.0 by construction
    /// ([`yunet::SCORE_THRESHOLD`]).
    ///
    /// Saliency: how concentrated the floored heat map is — the fraction of its
    /// mass inside a +-[`measure::CONFIDENCE_WINDOW`] box on the centroid. It
    /// is NOT a subject-present probability, and **its scale is not iOS's**.
    /// Vision answered a featureless field with a compact centred blob, so on
    /// iOS a no-subject frame read ~0.97; U^2-Netp answers one with noise, so
    /// here it reads ~0.03. Measured through this backing (heat floor 0.3):
    /// an isolated synthetic subject 1.00, a talking head 0.27-0.30, a street
    /// scene 0.24, a product shot 0.18-0.21, a landscape 0.63, a frame the
    /// model found nothing in 0.03. Consumers must not carry iOS thresholds
    /// across.
    pub confidence: f64,
    pub source: SampleSource,
}

/// A scene's verdict — component-wise medians over its samples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoSaliencySceneVerdict {
    pub index: usize,
    /// 0 => every grab in this scene failed; the verdict is the centred
    /// full-frame default and carries no signal.
    pub sample_count: usize,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    /// Fraction of the scene's samples that came from a face.
    pub face_share: f64,
    /// RMS distance of the sample centroids from the median centroid, in
    /// normalized units — a still subject is ~0, a travelling one is large.
    pub spread: f64,
}

/// Raw, policy-free subject MEASUREMENTS for a video. The engine measures; the
/// app decides — there is no crop, aspect or keep-out margin here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoSaliencyResult {
    /// Display-oriented source dimensions, NOT the downscaled size the heat
    /// source saw.
    pub frame_width: u32,
    pub frame_height: u32,
    /// Which kind of map the heat source produced — the shipped ONNX backing
    /// is a salient-object model, so this reads `objectness` where iOS's
    /// Vision default read `attention`.
    pub mode: SaliencyMode,
    /// Exactly one per input scene, same order and index.
    pub scenes: Vec<VideoSaliencySceneVerdict>,
    /// Every sample that measured, ascending time.
    pub samples: Vec<VideoSaliencySample>,
}
