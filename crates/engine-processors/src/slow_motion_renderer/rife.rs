//! Practical-RIFE v4.25 frame interpolation behind [`FrameInterpolator`] —
//! the off-macOS `SlowMotionRendering` backing (migration 016).
//!
//! Wired against the vs-mlrt **v2** interface, whose channel semantics were
//! empirically verified during the 016 spikes (identity and endpoint checks in
//! `016-notes.md`): `input [1, 7, H, W]` = `img0(3) + img1(3) + timestep(1)`
//! → `output [1, 3, H, W]`, all values in `[0, 1]`. Spatial dims must be
//! multiples of 32; this adapter edge-pads and crops the result back.
//!
//! The model is stateless per frame pair, so `starts_new_sequence` needs no
//! action — there is no flow cache to reset (unlike VideoToolbox).

use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_media::{CpuFrame, FrameBuffer, VideoFrame};

use super::orchestrator::{FrameInterpolator, TimedFrame};
use crate::frame_tensors::{append_planar_rgb_padded, planar_rgb_frame};

/// RIFE v4.x requires spatial dims aligned to 32 at scale 1.
const RIFE_ALIGNMENT: usize = 32;

pub struct RifeFrameInterpolator {
    backend: Backend,
}

impl std::fmt::Debug for RifeFrameInterpolator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RifeFrameInterpolator")
            .finish_non_exhaustive()
    }
}

impl RifeFrameInterpolator {
    pub fn new(backend: Backend) -> Self {
        Self { backend }
    }

    fn rgb(frame: &VideoFrame) -> Result<CpuFrame, String> {
        frame
            .to_cpu_rgb()
            .map_err(|error| format!("RGB download failed: {error}"))
    }
}

impl FrameInterpolator<VideoFrame> for RifeFrameInterpolator {
    fn interpolate(
        &mut self,
        from: &TimedFrame<VideoFrame>,
        to: &TimedFrame<VideoFrame>,
        phases: &[f32],
        _starts_new_sequence: bool,
    ) -> Result<Vec<VideoFrame>, String> {
        let first = Self::rgb(&from.frame)?;
        let second = Self::rgb(&to.frame)?;
        if first.width != second.width || first.height != second.height {
            return Err(format!(
                "interpolation pair mixes frame sizes: {}x{} vs {}x{}",
                first.width, first.height, second.width, second.height
            ));
        }
        if first.width == 0 || first.height == 0 {
            return Err("cannot interpolate zero-sized frames".into());
        }
        let (width, height) = (first.width as usize, first.height as usize);
        let padded_w = width.div_ceil(RIFE_ALIGNMENT) * RIFE_ALIGNMENT;
        let padded_h = height.div_ceil(RIFE_ALIGNMENT) * RIFE_ALIGNMENT;
        let plane = padded_w * padded_h;

        let mut synthesized = Vec::with_capacity(phases.len());
        for &phase in phases {
            if !(0.0..=1.0).contains(&phase) {
                return Err(format!("interpolation phase {phase} is outside [0, 1]"));
            }
            let mut values = Vec::with_capacity(7 * plane);
            append_planar_rgb_padded(&first, padded_w, padded_h, &mut values);
            append_planar_rgb_padded(&second, padded_w, padded_h, &mut values);
            values.resize(7 * plane, phase);
            let mut inputs = TensorMap::new();
            inputs.insert(
                "input".into(),
                Tensor::from_f32(vec![1, 7, padded_h as i64, padded_w as i64], values)
                    .map_err(|error| error.to_string())?,
            );
            let outputs = futures_executor::block_on(self.backend.infer(
                ModelId::RifeV425,
                &ModelVersion::Latest,
                inputs,
            ))
            .map_err(|error| error.to_string())?;
            let tensor = outputs
                .get("output")
                .ok_or_else(|| "RIFE output missing 'output'".to_owned())?;
            let cpu = planar_rgb_frame(
                &tensor.to_f32().map_err(|error| error.to_string())?,
                padded_w,
                padded_h,
                first.width,
                first.height,
            )
            .ok_or_else(|| "RIFE returned a malformed tensor".to_owned())?;
            // Timing is owned by the orchestrator (`FrameOutput::append` takes
            // the presentation timestamp); the synthesized frame inherits the
            // source frame's metadata as a placeholder.
            synthesized.push(VideoFrame {
                width: from.frame.width,
                height: from.frame.height,
                pts: from.frame.pts,
                duration: from.frame.duration,
                color: from.frame.color,
                buffer: FrameBuffer::Cpu(cpu),
            });
        }
        Ok(synthesized)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ferment_engine_media::{ColorInfo, CpuPixelFormat, MediaTime, TimeBase};

    use super::super::orchestrator::MediaTime as RenderTime;
    use super::*;
    use crate::frame_tensors::testing::MockBackend;

    fn timed_frame(size: u32, value: u8, index: i64) -> TimedFrame<VideoFrame> {
        let time_base = TimeBase::new_unchecked(1, 30);
        let bytes = vec![value; (size * size * 3) as usize];
        TimedFrame {
            frame: VideoFrame {
                width: size,
                height: size,
                pts: MediaTime::new(index, time_base),
                duration: None,
                color: ColorInfo::default(),
                buffer: FrameBuffer::Cpu(
                    CpuFrame::new(size, size, size as usize * 3, CpuPixelFormat::Rgb24, bytes)
                        .unwrap(),
                ),
            },
            pts: RenderTime::new(index, 30),
        }
    }

    #[test]
    fn rife_pads_to_32_packs_timestep_and_crops_back() {
        let backend = Arc::new(MockBackend::constant("output", vec![1, 3, 64, 64], 0.5));
        let mut interpolator = RifeFrameInterpolator::new(backend.clone());

        let from = timed_frame(48, 100, 0);
        let to = timed_frame(48, 200, 1);
        let frames = interpolator
            .interpolate(&from, &to, &[0.25, 0.75], false)
            .unwrap();

        assert_eq!(frames.len(), 2);
        for frame in &frames {
            assert_eq!((frame.width, frame.height), (48, 48));
            // FrameBuffer collapses to only Cpu when the macOS/Windows GPU
            // variants are cfg'd out, making this irrefutable off those OSes.
            #[allow(irrefutable_let_patterns)]
            let FrameBuffer::Cpu(cpu) = &frame.buffer else {
                panic!("expected a CPU buffer");
            };
            assert_eq!(cpu.data.len(), 48 * 48 * 3);
            assert!(cpu.data.iter().all(|&value| value == 128));
        }

        let captured = backend.captured.lock().unwrap();
        assert_eq!(captured.len(), 2, "one inference per phase");
        for (call, expected_phase) in captured.iter().zip([0.25f32, 0.75]) {
            assert_eq!(call.0, ModelId::RifeV425);
            let input = &call.1["input"];
            assert_eq!(input.shape, vec![1, 7, 64, 64], "48 pads to 64");
            let values = input.to_f32().unwrap();
            let plane = 64 * 64;
            // img0 red plane leads; img1 follows after three planes.
            assert!((values[0] - 100.0 / 255.0).abs() < 1e-6);
            assert!((values[3 * plane] - 200.0 / 255.0).abs() < 1e-6);
            // The trailing plane is the constant timestep.
            assert!(
                values[6 * plane..]
                    .iter()
                    .all(|&value| (value - expected_phase).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn rife_rejects_mismatched_pair_and_bad_phase() {
        let backend = Arc::new(MockBackend::constant("output", vec![1, 3, 64, 64], 0.5));
        let mut interpolator = RifeFrameInterpolator::new(backend);

        let small = timed_frame(32, 0, 0);
        let large = timed_frame(64, 0, 1);
        assert!(
            interpolator
                .interpolate(&small, &large, &[0.5], false)
                .unwrap_err()
                .contains("mixes frame sizes")
        );

        let from = timed_frame(32, 0, 0);
        let to = timed_frame(32, 0, 1);
        assert!(
            interpolator
                .interpolate(&from, &to, &[1.5], false)
                .unwrap_err()
                .contains("outside [0, 1]")
        );
    }
}
