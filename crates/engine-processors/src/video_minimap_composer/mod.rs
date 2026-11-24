//! Video minimap composer — single image summary of a video for vision
//! LLM consumption. One row per scene, each with N keyframe thumbnails;
//! a "barcode" strip below rolls up mean color over time; a shared time
//! axis at the bottom.
//!
//! Video frame extraction is **not** in this crate — callers (CLI, server)
//! pass in already-decoded keyframes. Same pattern as VideoSceneDetector
//! for symmetry and to keep ffmpeg out of the processors crate.

pub mod allocate;
pub mod layout;
pub mod renderer;
pub mod sampler;

mod processor;

pub use allocate::Scene;
pub use processor::{
    Cell, ImageFormat, KeyframeBuffer, PoolEntry, VideoMinimapComposer, VideoMinimapComposerConfig,
    VideoMinimapResult,
};
pub use sampler::SamplingStrategy;

mod transform;

/// A single display-oriented preview using the same affine geometry as video
/// minimaps. Callers can pass a decoder-scaled picture to bound CPU readback;
/// the container matrix is applied exactly once here, including reflections.
pub fn display_thumbnail(
    frame: &ferment_engine_media::CpuFrame,
    matrix: Option<[i32; 9]>,
    maximum: (u32, u32),
) -> ferment_engine_core::Result<KeyframeBuffer> {
    if frame.width == 0
        || frame.height == 0
        || u64::from(frame.width) * u64::from(frame.height) > 16_777_216
        || !(1..=4096).contains(&maximum.0)
        || !(1..=4096).contains(&maximum.1)
    {
        return Err(ferment_engine_core::EngineError::invalid_config(
            "display thumbnail requires a bounded decoded frame and output size",
        ));
    }
    ferment_engine_media::CpuFrame::new(
        frame.width,
        frame.height,
        frame.stride,
        frame.format,
        frame.data.clone(),
    )
    .map_err(|e| ferment_engine_core::EngineError::Decode(e.to_string()))?;
    let frame = frame.to_rgb24();
    let mut data = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
    for row in frame.data.chunks(frame.stride).take(frame.height as usize) {
        data.extend_from_slice(&row[..frame.width as usize * 3]);
    }
    transform::DisplayTransform::from_matrix(matrix)?.thumbnail(
        &crate::video_source::RgbFrame {
            width: frame.width,
            height: frame.height,
            presentation_time: 0.,
            data,
        },
        maximum,
    )
}

#[cfg(test)]
mod display_tests {
    use super::*;
    use ferment_engine_media::{CpuFrame, CpuPixelFormat};
    #[test]
    fn display_thumbnail_checks_untrusted_frame_and_preserves_padded_rows() {
        let frame = CpuFrame::new(
            2,
            1,
            8,
            CpuPixelFormat::Rgb24,
            vec![255, 0, 0, 0, 255, 0, 99, 99],
        )
        .unwrap();
        let out = display_thumbnail(&frame, None, (2, 1)).unwrap();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 1);
        assert_eq!(out.data, [255, 0, 0, 0, 255, 0]);
        let mut broken = frame;
        broken.stride = 0;
        assert!(display_thumbnail(&broken, None, (2, 1)).is_err());
        broken.width = u32::MAX;
        assert!(display_thumbnail(&broken, None, (2, 1)).is_err());
    }
}
