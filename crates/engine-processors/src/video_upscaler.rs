//! Backend-neutral planning contracts for video super-resolution.
//!
//! Pixel enhancement and video output remain platform capabilities. This
//! module resolves the reusable geometry and bitrate decisions without
//! pretending that a resize is equivalent to super-resolution.

use async_trait::async_trait;
use ferment_engine_core::{EngineError, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_media::{FrameBuffer, VideoFrame};

use crate::frame_tensors::{append_planar_rgb, planar_rgb_frame};

pub const FOUR_K_LONG_SIDE: u32 = 3_840;
pub const MINIMUM_OUTPUT_BITRATE: u64 = 8_000_000;

/// One render service over the host's installed inference runtime.
#[derive(Clone, Default)]
pub struct VideoUpscaler {
    pub(crate) backend: Option<Backend>,
}

impl std::fmt::Debug for VideoUpscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoUpscaler")
            .field("configured", &self.backend.is_some())
            .finish()
    }
}

impl VideoUpscaler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

impl Dimensions {
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[must_use]
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    #[must_use]
    pub const fn longest_side(self) -> u32 {
        if self.width > self.height {
            self.width
        } else {
            self.height
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleMode {
    Enhance,
    Upscale { longest_side: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscalePlan {
    pub input_size: Dimensions,
    pub scale_factor: u32,
    pub output_size: Dimensions,
}

impl UpscalePlan {
    #[must_use]
    pub const fn ml_size(self) -> Dimensions {
        Dimensions {
            width: self.input_size.width.saturating_mul(self.scale_factor),
            height: self.input_size.height.saturating_mul(self.scale_factor),
        }
    }
}

impl UpscaleMode {
    #[must_use]
    pub fn plan(
        self,
        source: Dimensions,
        native_factor: u32,
        input_cap: Option<Dimensions>,
    ) -> UpscalePlan {
        let input_size = clamp_to_cap(source, input_cap);
        let output_size = match self {
            Self::Enhance => even(source.width as f64, source.height as f64),
            Self::Upscale { longest_side } => {
                let source_longest = source.longest_side();
                let scale = if source_longest > 0 {
                    f64::from(longest_side) / f64::from(source_longest)
                } else {
                    1.0
                };
                even(
                    f64::from(source.width) * scale,
                    f64::from(source.height) * scale,
                )
            }
        };
        UpscalePlan {
            input_size,
            scale_factor: native_factor,
            output_size,
        }
    }

    #[must_use]
    pub fn reachable_target(source: Dimensions, ceiling: u32) -> Option<u32> {
        (source.longest_side() < ceiling).then_some(ceiling)
    }
}

#[must_use]
pub fn scaled_bitrate(source: u64, source_area: u64, output_area: u64) -> u64 {
    if source_area == 0 {
        return source.max(MINIMUM_OUTPUT_BITRATE);
    }
    ((source as f64 * output_area as f64 / source_area as f64) as u64).max(MINIMUM_OUTPUT_BITRATE)
}

fn clamp_to_cap(source: Dimensions, cap: Option<Dimensions>) -> Dimensions {
    let Some(cap) = cap else {
        return even(source.width as f64, source.height as f64);
    };
    if source.width == 0 || source.height == 0 {
        return even(source.width as f64, source.height as f64);
    }
    let scale = (f64::from(cap.width) / f64::from(source.width))
        .min(f64::from(cap.height) / f64::from(source.height))
        .min(1.0);
    even(
        f64::from(source.width) * scale,
        f64::from(source.height) * scale,
    )
}

fn even(width: f64, height: f64) -> Dimensions {
    let even_dimension = |value: f64| {
        let rounded = value.round().max(0.0) as u32;
        rounded / 2 * 2
    };
    Dimensions::new(even_dimension(width), even_dimension(height))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderCapability {
    VideoDecode,
    VideoEncode,
    LanczosFrameUpscaler,
    StatefulRecurrence,
    SceneBoundaryRestart,
    PreResampler,
    PostResampler,
}

/// Render operations are deliberately unavailable until platform VideoIO and
/// frame-upscaling backends land. Exact-name blocked oracle tests call this so
/// forcing them fails loudly with the missing capabilities.
pub fn require_render_capabilities(required: &[RenderCapability]) -> Result<()> {
    Err(EngineError::Unsupported(format!(
        "video upscaler render unavailable; missing capabilities: {required:?}"
    )))
}

/// Backend-neutral single-frame super-resolution seam — the spatial-domain
/// counterpart to [`crate::video_noise_filter::TemporalNoiseFiltering`].
///
/// Super-resolution is stateless per frame: SRVGGNetCompact (Real-ESRGAN
/// Compact and its `-dn`/animevideo variants) takes one frame and emits one
/// enlarged frame, so — unlike temporal denoising — this seam carries no
/// neighbor window. Frame timing, color metadata, and the final resample to
/// [`UpscalePlan::output_size`] stay owned by [`VideoUpscaler`]; a backend only
/// returns the enlarged pixels, and so can never accidentally retime output.
///
/// This is the seam the off-macOS `VideoUpscaling` backing (an ORT session
/// running the Compact ONNX master, per migration 016) plugs into, exactly as
/// FastDVDnet plugs into `TemporalNoiseFiltering` and RIFE into
/// `FrameInterpolator`.
#[async_trait(?Send)]
pub trait FrameUpscaling {
    /// The backend model's native integer scale (e.g. `4` for
    /// `realesr-general-x4v3`, `2` for an ×2 export). Drives
    /// [`UpscalePlan::scale_factor`], so a backend and its plan agree on
    /// [`UpscalePlan::ml_size`].
    fn native_scale_factor(&self) -> u32;

    /// Prepare the backend for a run of frames decoded at `input` size.
    /// `prefer_hdr` mirrors the temporal seam's HDR opt-in. Called once before
    /// a batch; tiling/allocation setup belongs here, not in [`Self::upscale`].
    fn start_session(&mut self, input: Dimensions, prefer_hdr: bool) -> Result<()>;

    /// Upscale one decoded frame by [`Self::native_scale_factor`], returning the
    /// enlarged pixel buffer only. The output is `input * scale` in size; the
    /// resample down to `UpscalePlan::output_size` (the `PostResampler` /
    /// `PreResampler` stages) remains the orchestrator's responsibility.
    async fn upscale(&mut self, frame: &VideoFrame) -> Result<FrameBuffer>;

    /// Release any per-session backend resources.
    fn end_session(&mut self);
}

/// Identity upscaler for tests and platforms without an SR backend: it reports
/// unit scale and returns the source frame unchanged. Mirrors
/// [`crate::video_noise_filter::PassthroughTemporalNoiseFilter`] so orchestrator
/// tests can exercise the frame path without a model.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughFrameUpscaler;

#[async_trait(?Send)]
impl FrameUpscaling for PassthroughFrameUpscaler {
    fn native_scale_factor(&self) -> u32 {
        1
    }

    fn start_session(&mut self, _input: Dimensions, _prefer_hdr: bool) -> Result<()> {
        Ok(())
    }

    async fn upscale(&mut self, frame: &VideoFrame) -> Result<FrameBuffer> {
        Ok(frame.buffer.clone())
    }

    fn end_session(&mut self) {}
}

/// Real-ESRGAN Compact (SRVGGNetCompact) super-resolution behind
/// [`FrameUpscaling`] — the off-macOS `VideoUpscaling` backing (migration 016).
///
/// All three masters (`realesr_general_x4v3`, `realesr_general_wdn_x4v3`,
/// `realesr_animevideov3`) share the ×4 architecture and the
/// `lr [1,3,H,W] -> sr [1,3,4H,4W]` contract with dynamic spatial dims.
///
/// `wdn` is the WEAK-DENOISE end, not the denoiser: upstream reads a denoise
/// strength by interpolating the two general checkpoints' weights
/// (`inference_realesrgan.py`, `dni_weight [strength, 1 - strength]`), where
/// strength 0 — "keep the noise" — IS the plain wdn checkpoint and strength 1
/// is the plain general one. This backing loads one master or the other, so
/// only the two ends are reachable; the middle would need the blend at install
/// time.
pub struct CompactFrameUpscaler {
    backend: Backend,
    model: ModelId,
    progress: ferment_engine_core::ProgressSink,
}

impl std::fmt::Debug for CompactFrameUpscaler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompactFrameUpscaler")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl CompactFrameUpscaler {
    const SCALE: u32 = 4;

    /// The default photographic checkpoint (`realesr-general-x4v3`).
    pub fn general(backend: Backend) -> Self {
        Self::for_model(backend, ModelId::RealesrGeneralX4)
    }

    /// The weak-denoise end of the general pair (`realesr_general_wdn_x4v3`) —
    /// upstream's strength 0, which keeps more of the source's noise than
    /// [`Self::general`] does.
    pub fn general_denoise(backend: Backend) -> Self {
        Self::for_model(backend, ModelId::RealesrGeneralWdnX4)
    }

    /// The animation/video-tuned checkpoint (`realesr-animevideov3`).
    pub fn anime_video(backend: Backend) -> Self {
        Self::for_model(backend, ModelId::RealesrAnimeVideo3)
    }

    /// The outer render's cancellation token follows every bounded inference tile.
    pub fn with_progress(mut self, progress: ferment_engine_core::ProgressSink) -> Self {
        self.progress = progress;
        self
    }

    fn for_model(backend: Backend, model: ModelId) -> Self {
        Self {
            backend,
            model,
            progress: ferment_engine_core::ProgressSink::noop(),
        }
    }
}

#[async_trait(?Send)]
impl FrameUpscaling for CompactFrameUpscaler {
    fn native_scale_factor(&self) -> u32 {
        Self::SCALE
    }

    fn start_session(&mut self, input: Dimensions, _prefer_hdr: bool) -> Result<()> {
        if input.width == 0 || input.height == 0 {
            return Err(EngineError::InvalidConfiguration(format!(
                "cannot upscale a zero-sized frame ({}x{})",
                input.width, input.height
            )));
        }
        Ok(())
    }

    async fn upscale(&mut self, frame: &VideoFrame) -> Result<FrameBuffer> {
        let rgb = frame
            .to_cpu_rgb()
            .map_err(|error| EngineError::inference(format!("RGB download failed: {error}")))?
            .to_rgb24();
        const TILE: usize = 192;
        const HALO: usize = 40;
        if rgb.width <= (TILE + HALO * 2) as u32 && rgb.height <= (TILE + HALO * 2) as u32 {
            return Ok(FrameBuffer::Cpu(self.infer_rgb(&rgb).await?));
        }
        let (width, height) = (rgb.width as usize, rgb.height as usize);
        let out_stride = width * 4 * 3;
        let mut output = vec![0u8; out_stride * height * 4];
        for y in (0..height).step_by(TILE) {
            for x in (0..width).step_by(TILE) {
                let (left, top) = (x.saturating_sub(HALO), y.saturating_sub(HALO));
                let (right, bottom) = ((x + TILE + HALO).min(width), (y + TILE + HALO).min(height));
                let (tw, th) = (right - left, bottom - top);
                let mut tile = Vec::with_capacity(tw * th * 3);
                for row in top..bottom {
                    tile.extend_from_slice(
                        &rgb.data[row * rgb.stride + left * 3..row * rgb.stride + right * 3],
                    );
                }
                let tile = ferment_engine_media::CpuFrame::new(
                    tw as u32,
                    th as u32,
                    tw * 3,
                    ferment_engine_media::CpuPixelFormat::Rgb24,
                    tile,
                )
                .map_err(|error| EngineError::inference(error.to_string()))?;
                let scaled = self.infer_rgb(&tile).await?;
                let row_bytes = (width - x).min(TILE) * 4 * 3;
                for row in 0..(height - y).min(TILE) * 4 {
                    let source = ((y - top) * 4 + row) * scaled.stride + (x - left) * 4 * 3;
                    let destination = (y * 4 + row) * out_stride + x * 4 * 3;
                    output[destination..destination + row_bytes]
                        .copy_from_slice(&scaled.data[source..source + row_bytes]);
                }
            }
        }
        Ok(FrameBuffer::Cpu(
            ferment_engine_media::CpuFrame::new(
                rgb.width * 4,
                rgb.height * 4,
                out_stride,
                ferment_engine_media::CpuPixelFormat::Rgb24,
                output,
            )
            .map_err(|error| EngineError::inference(error.to_string()))?,
        ))
    }

    fn end_session(&mut self) {}
}

impl CompactFrameUpscaler {
    async fn infer_rgb(
        &self,
        rgb: &ferment_engine_media::CpuFrame,
    ) -> Result<ferment_engine_media::CpuFrame> {
        if self.progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        let mut values = Vec::new();
        append_planar_rgb(rgb, &mut values);
        let mut inputs = TensorMap::new();
        inputs.insert(
            "lr".into(),
            Tensor::from_f32(
                vec![1, 3, i64::from(rgb.height), i64::from(rgb.width)],
                values,
            )?,
        );
        let outputs = self
            .backend
            .infer(self.model, &ModelVersion::Latest, inputs)
            .await?;
        let tensor = outputs
            .get("sr")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: self.model.name().into(),
                key: "sr".into(),
            })?;
        let out_w = rgb.width * Self::SCALE;
        let out_h = rgb.height * Self::SCALE;
        let cpu = planar_rgb_frame(
            &tensor.to_f32()?,
            out_w as usize,
            out_h as usize,
            out_w,
            out_h,
        )
        .ok_or(EngineError::InvalidOutput)?;
        Ok(cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_media::{ColorInfo, CpuFrame, CpuPixelFormat, MediaTime, TimeBase};

    const MAC_CAP: Dimensions = Dimensions::new(1_920, 1_080);
    const IOS_CAP: Dimensions = Dimensions::new(1_440, 1_080);

    fn sample_frame(value: u8) -> VideoFrame {
        let time_base = TimeBase::new_unchecked(1, 30);
        VideoFrame {
            width: 1,
            height: 1,
            pts: MediaTime::new(0, time_base),
            duration: Some(MediaTime::new(1, time_base)),
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(
                CpuFrame::new(1, 1, 3, CpuPixelFormat::Rgb24, vec![value; 3]).unwrap(),
            ),
        }
    }

    #[test]
    fn passthrough_upscaler_reports_unit_scale() {
        assert_eq!(PassthroughFrameUpscaler.native_scale_factor(), 1);
    }

    #[test]
    fn compact_upscaler_packs_lr_and_unpacks_sr() {
        use std::sync::Arc;

        use crate::frame_tensors::testing::MockBackend;

        futures_executor::block_on(async {
            let backend = Arc::new(MockBackend::constant("sr", vec![1, 3, 8, 8], 0.5));
            let mut upscaler = CompactFrameUpscaler::general(backend.clone());
            assert_eq!(upscaler.native_scale_factor(), 4);
            upscaler
                .start_session(Dimensions::new(2, 2), false)
                .unwrap();

            let bytes: Vec<u8> = (10..22).collect();
            let source = VideoFrame {
                width: 2,
                height: 2,
                pts: MediaTime::new(0, TimeBase::new_unchecked(1, 30)),
                duration: None,
                color: ColorInfo::default(),
                buffer: FrameBuffer::Cpu(
                    CpuFrame::new(2, 2, 6, CpuPixelFormat::Rgb24, bytes.clone()).unwrap(),
                ),
            };
            let output = upscaler.upscale(&source).await.unwrap();
            upscaler.end_session();

            // Irrefutable once FrameBuffer's macOS/Windows GPU variants cfg out.
            #[allow(irrefutable_let_patterns)]
            let FrameBuffer::Cpu(cpu) = output else {
                panic!("expected a CPU output buffer");
            };
            assert_eq!((cpu.width, cpu.height), (8, 8));
            assert!(cpu.data.iter().all(|&value| value == 128));

            let captured = backend.captured.lock().unwrap();
            let (model, inputs) = &captured[0];
            assert_eq!(*model, ferment_engine_infer::ModelId::RealesrGeneralX4);
            let lr = &inputs["lr"];
            assert_eq!(lr.shape, vec![1, 3, 2, 2]);
            // Planar layout: reds first, and the first red is pixel (0,0).
            let values = lr.to_f32().unwrap();
            assert!((values[0] - f32::from(bytes[0]) / 255.0).abs() < 1e-6);
            assert!((values[4] - f32::from(bytes[1]) / 255.0).abs() < 1e-6);
        });
    }

    #[test]
    fn compact_upscaler_rejects_zero_sized_session() {
        use std::sync::Arc;

        use crate::frame_tensors::testing::MockBackend;

        let backend = Arc::new(MockBackend::constant("sr", vec![1, 3, 4, 4], 0.0));
        let mut upscaler = CompactFrameUpscaler::anime_video(backend);
        assert!(
            upscaler
                .start_session(Dimensions::new(0, 4), false)
                .is_err()
        );
    }

    #[test]
    fn passthrough_upscaler_returns_source_pixels_unchanged() {
        futures_executor::block_on(async {
            let mut upscaler = PassthroughFrameUpscaler;
            upscaler
                .start_session(Dimensions::new(1, 1), false)
                .unwrap();
            let source = sample_frame(7);
            let output = upscaler.upscale(&source).await.unwrap();
            upscaler.end_session();

            // Irrefutable once FrameBuffer's macOS/Windows GPU variants cfg out.
            #[allow(irrefutable_let_patterns)]
            let FrameBuffer::Cpu(output) = output else {
                panic!("passthrough must keep the CPU buffer");
            };
            #[allow(irrefutable_let_patterns)]
            let FrameBuffer::Cpu(expected) = source.buffer else {
                unreachable!("constructed as a CPU frame");
            };
            assert_eq!(output, expected);
        });
    }

    #[test]
    fn test_scaled_bitrate_scales_by_area() {
        assert_eq!(scaled_bitrate(5_000_000, 100, 400), 20_000_000);
        assert_eq!(scaled_bitrate(100, 100, 400), 8_000_000);
    }

    #[test]
    fn test_enhance_keeps_source_size_when_source_fits_cap() {
        let plan = UpscaleMode::Enhance.plan(Dimensions::new(1_280, 720), 4, Some(MAC_CAP));
        assert_eq!(plan.input_size, Dimensions::new(1_280, 720));
        assert_eq!(plan.output_size, Dimensions::new(1_280, 720));
        assert_eq!(plan.ml_size(), Dimensions::new(5_120, 2_880));
    }

    #[test]
    fn test_enhance_downscales_oversize_input_but_keeps_output_at_source() {
        let plan = UpscaleMode::Enhance.plan(Dimensions::new(3_840, 2_160), 4, Some(MAC_CAP));
        assert_eq!(plan.input_size, Dimensions::new(1_920, 1_080));
        assert_eq!(plan.output_size, Dimensions::new(3_840, 2_160));
    }

    #[test]
    fn test_enhance_input_cap_is_aspect_preserved_on_ios() {
        let plan = UpscaleMode::Enhance.plan(Dimensions::new(3_840, 2_160), 4, Some(IOS_CAP));
        assert_eq!(plan.input_size, Dimensions::new(1_440, 810));
        assert_eq!(plan.output_size, Dimensions::new(3_840, 2_160));
    }

    #[test]
    fn test_upscale_720p_to_four_k_lands_at_four_k() {
        let plan = UpscaleMode::Upscale {
            longest_side: 3_840,
        }
        .plan(Dimensions::new(1_280, 720), 4, Some(MAC_CAP));
        assert_eq!(plan.input_size, Dimensions::new(1_280, 720));
        assert_eq!(plan.output_size, Dimensions::new(3_840, 2_160));
    }

    #[test]
    fn test_upscale_portrait_targets_long_side_and_clamps_input() {
        let plan = UpscaleMode::Upscale {
            longest_side: 3_840,
        }
        .plan(Dimensions::new(1_080, 1_920), 4, Some(MAC_CAP));
        assert_eq!(plan.output_size, Dimensions::new(2_160, 3_840));
        assert_eq!(plan.input_size, Dimensions::new(608, 1_080));
    }

    #[test]
    fn test_nil_cap_feeds_source_whole() {
        let plan = UpscaleMode::Enhance.plan(Dimensions::new(3_840, 2_160), 2, None);
        assert_eq!(plan.input_size, Dimensions::new(3_840, 2_160));
    }

    #[test]
    fn test_dimensions_are_forced_even() {
        let plan =
            UpscaleMode::Upscale { longest_side: 999 }.plan(Dimensions::new(641, 361), 4, None);
        assert!(plan.output_size.width.is_multiple_of(2));
        assert!(plan.output_size.height.is_multiple_of(2));
        assert!(plan.input_size.width.is_multiple_of(2));
        assert!(plan.input_size.height.is_multiple_of(2));
    }

    #[test]
    fn test_reachable_target_offers_four_k_below_four_k() {
        assert_eq!(
            UpscaleMode::reachable_target(Dimensions::new(1_280, 720), FOUR_K_LONG_SIDE),
            Some(3_840)
        );
        assert_eq!(
            UpscaleMode::reachable_target(Dimensions::new(2_560, 1_440), FOUR_K_LONG_SIDE),
            Some(3_840)
        );
        assert_eq!(
            UpscaleMode::reachable_target(Dimensions::new(1_920, 1_080), FOUR_K_LONG_SIDE),
            Some(3_840)
        );
    }

    #[test]
    fn test_reachable_target_hidden_at_or_above_four_k() {
        assert_eq!(
            UpscaleMode::reachable_target(Dimensions::new(3_840, 2_160), FOUR_K_LONG_SIDE),
            None
        );
        assert_eq!(
            UpscaleMode::reachable_target(Dimensions::new(2_160, 3_840), FOUR_K_LONG_SIDE),
            None
        );
    }

    #[test]
    #[ignore = "blocked: requires VideoIO decode/encode plus a Lanczos frame-upscaling backend"]
    fn test_lanczos_upscale_doubles_dimensions() {
        require_render_capabilities(&[
            RenderCapability::VideoDecode,
            RenderCapability::LanczosFrameUpscaler,
            RenderCapability::VideoEncode,
        ])
        .unwrap();
    }

    #[test]
    #[ignore = "blocked: requires stateful frame recurrence, cut restarts, and VideoIO"]
    fn test_recurrence_threads_and_sequence_restarts_at_cuts() {
        require_render_capabilities(&[
            RenderCapability::VideoDecode,
            RenderCapability::StatefulRecurrence,
            RenderCapability::SceneBoundaryRestart,
            RenderCapability::VideoEncode,
        ])
        .unwrap();
    }

    #[test]
    #[ignore = "blocked: requires frame enhancement, post-resampling, and VideoIO"]
    fn test_enhance_runs_post_resampler_back_to_source_size() {
        require_render_capabilities(&[
            RenderCapability::VideoDecode,
            RenderCapability::LanczosFrameUpscaler,
            RenderCapability::PostResampler,
            RenderCapability::VideoEncode,
        ])
        .unwrap();
    }

    #[test]
    #[ignore = "blocked: requires capped frame enhancement, pre-resampling, and VideoIO"]
    fn test_oversize_source_pre_downscales_to_cap() {
        require_render_capabilities(&[
            RenderCapability::VideoDecode,
            RenderCapability::PreResampler,
            RenderCapability::LanczosFrameUpscaler,
            RenderCapability::VideoEncode,
        ])
        .unwrap();
    }
}
