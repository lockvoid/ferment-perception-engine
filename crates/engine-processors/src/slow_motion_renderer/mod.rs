//! Pure CPU slow-motion pre-pass and motion-blur decision layer.
//!
//! Media decoding, VideoToolbox interpolation/blur, and output muxing stay in
//! platform I/O adapters. This module operates only on owned 8-bit luma
//! planes, so it is deterministic and `wasm32-unknown-unknown` compatible.

/// Public engine service for path-based slow-motion rendering.
///
/// The portable scheduler is available through [`render_frames`]. The
/// path-based capability remains an explicit unsupported boundary until a
/// platform frame processor and playable video encode/mux path are injected.
#[derive(Default, Clone)]
pub struct SlowMotionRenderer {
    pub(crate) backend: Option<ferment_engine_infer::Backend>,
}

impl std::fmt::Debug for SlowMotionRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlowMotionRenderer")
            .field("configured", &self.backend.is_some())
            .finish()
    }
}

impl SlowMotionRenderer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(mut self, backend: ferment_engine_infer::Backend) -> Self {
        self.backend = Some(backend);
        self
    }
}

mod analyzer;
mod block_motion;
mod motion_blur;
pub(crate) mod orchestrator;
mod rife;
mod shutter_angle;

pub use analyzer::SlowMotionAnalyzer;
pub use block_motion::{BlockMotionEstimator, LumaPlane, MotionField, MotionVector};
pub use motion_blur::{
    MotionBlurDecision, MotionBlurMode, MotionBlurPolicy, MotionBlurReason, SlowMotionAnalysis,
};
pub use orchestrator::{
    FrameBlurrer, FrameInterpolator, FrameOutput, MediaTime, PIXEL_FORMAT_420V, PIXEL_FORMAT_X420,
    RenderCapability, RenderConfig, SlowMotionRenderError, SlowMotionRenderResult, TRANSFER_HLG,
    TRANSFER_PQ, TimedFrame, format_compatible, interpolation_phases, is_hdr_transfer, output_pts,
    pair_crosses_cut, reader_pixel_format, render_frames, require_render_capabilities,
};
pub use rife::RifeFrameInterpolator;
pub use shutter_angle::ShutterAngleEstimator;

#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod tests;
