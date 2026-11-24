//! Knobs for a saliency pass, ported from Swift `VideoSaliencyConfig`
//! (`Protocols/VideoSaliencyAnalyzing.swift`).
//!
//! Sparse by construction: saliency is a per-still verdict, so a handful of
//! samples per scene is the whole signal.

/// Heat-map background floor as a fraction of the map's peak, as iOS ships it
/// against Apple Vision. Vision's attention map carries a wide low pedestal
/// (mean ~0.06 against a 0.85 peak on a talking head); left in, it drags every
/// mass statistic to the frame centre and flattens the confidence range.
pub const VISION_HEAT_FLOOR: f64 = 0.3;

/// The same number, kept for the U^2-Netp backing — after measuring, not on
/// faith.
///
/// `scripts/exports/video_saliency/measure_pedestal.py` ran the substitute over
/// nine frames of real footage (talking head, street, product, cockpit,
/// landscape, crowd). **U^2-Netp has no pedestal**: its background sits at
/// 3e-6..3e-5 under a ~1.0 peak, four orders of magnitude below Vision's 0.07
/// ratio, because it is a salient-object segmenter rather than a fixation
/// model. Placing the cut at the same height between background and peak that
/// iOS chose (24.68 % of the way up) yields an equivalent floor of **0.2468**,
/// stable to +-0.0002 across all nine frames.
///
/// That is not a material difference, and the sweep proves it: from floor 0.0
/// to floor 0.5 the measured centroid moves by less than 0.005 on every frame
/// with a real subject. The pedestal subtraction simply has no work to do here.
/// So the iOS constant stays — one number across both platforms is worth more
/// than 0.05 of an inert one — and the measurement, not the inheritance, is
/// what justifies it. Re-run the script if the heat source is ever swapped.
pub const U2NETP_HEAT_FLOOR: f64 = VISION_HEAT_FLOOR;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoSaliencyConfig {
    /// Grabs per second of scene duration, before clamping.
    pub samples_per_second: f64,
    pub min_samples_per_scene: usize,
    pub max_samples_per_scene: usize,
    /// Samples span `[interior, 1 - interior]` of the scene.
    pub interior: f64,
    /// Long edge of the grabbed frame handed to the heat source.
    pub max_long_edge: u32,
    /// Fraction of the map's peak subtracted from every cell before the
    /// moments are taken.
    pub heat_floor: f64,
}

impl Default for VideoSaliencyConfig {
    fn default() -> Self {
        Self {
            samples_per_second: 0.5,
            min_samples_per_scene: 3,
            max_samples_per_scene: 32,
            interior: 0.1,
            max_long_edge: 512,
            heat_floor: U2NETP_HEAT_FLOOR,
        }
    }
}
