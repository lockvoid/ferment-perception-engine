//! Scene-aware temporal video-noise-filter contract.
//!
//! This module owns the backend-neutral part of the Swift `VideoNoiseFilter`:
//! temporal-window assembly, scene discontinuities, strength validation, and
//! the lone-frame pass-through guard. Decoding, encoding, and playable output
//! muxing remain platform media capabilities and are deliberately not faked
//! here.

use std::fmt;

use async_trait::async_trait;
use ferment_engine_core::{EngineError, ProgressSink};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_media::{CpuFrame, FrameBuffer, MediaTime, TransferCharacteristic, VideoFrame};

use crate::frame_tensors::{append_planar_rgb_padded, planar_rgb_frame};

pub const RENDER_UNAVAILABLE_REASON: &str = "video noise filter render unavailable: the decoded-frame adapter and container mux/finalize-to-path boundary are not implemented";

/// The side of the current frame for which temporal neighbors are requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSide {
    Previous,
    Next,
}

/// Errors from the backend-neutral temporal-filter contract.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoNoiseFilterError {
    InvalidStrength(f32),
    EmptyWindow,
    Backend(String),
}

impl fmt::Display for VideoNoiseFilterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStrength(strength) => write!(
                formatter,
                "noise-filter strength {strength} is out of range (0-1)"
            ),
            Self::EmptyWindow => formatter.write_str("the requested time window is empty"),
            Self::Backend(reason) => write!(formatter, "temporal noise filter failed: {reason}"),
        }
    }
}

impl std::error::Error for VideoNoiseFilterError {}

pub type Result<T> = std::result::Result<T, VideoNoiseFilterError>;

/// One-frame temporal denoising seam, mirroring Swift's
/// `TemporalNoiseFiltering` protocol.
///
/// The returned value is only the filtered pixel buffer. Frame timing and
/// metadata remain owned by [`VideoNoiseFilter`], so a backend cannot
/// accidentally retime output.
#[async_trait(?Send)]
pub trait TemporalNoiseFiltering {
    fn previous_frame_count(&self) -> usize;
    fn next_frame_count(&self) -> usize;

    fn start_session(&mut self, width: u32, height: u32, prefer_hdr: bool) -> Result<()>;

    async fn filter(
        &mut self,
        frame: &VideoFrame,
        previous: &[VideoFrame],
        next: &[VideoFrame],
        strength: f32,
        discontinuity: bool,
    ) -> Result<FrameBuffer>;

    fn end_session(&mut self);
}

/// Identity temporal filter for tests and unsupported platform environments.
#[derive(Debug, Clone)]
pub struct PassthroughTemporalNoiseFilter {
    previous_frame_count: usize,
    next_frame_count: usize,
}

impl Default for PassthroughTemporalNoiseFilter {
    fn default() -> Self {
        Self::new(2, 2)
    }
}

impl PassthroughTemporalNoiseFilter {
    #[must_use]
    pub const fn new(previous_frame_count: usize, next_frame_count: usize) -> Self {
        Self {
            previous_frame_count,
            next_frame_count,
        }
    }
}

#[async_trait(?Send)]
impl TemporalNoiseFiltering for PassthroughTemporalNoiseFilter {
    fn previous_frame_count(&self) -> usize {
        self.previous_frame_count
    }

    fn next_frame_count(&self) -> usize {
        self.next_frame_count
    }

    fn start_session(&mut self, _width: u32, _height: u32, _prefer_hdr: bool) -> Result<()> {
        Ok(())
    }

    async fn filter(
        &mut self,
        frame: &VideoFrame,
        _previous: &[VideoFrame],
        _next: &[VideoFrame],
        _strength: f32,
        _discontinuity: bool,
    ) -> Result<FrameBuffer> {
        Ok(frame.buffer.clone())
    }

    fn end_session(&mut self) {}
}

/// Sigma ceiling of the released FastDVDnet checkpoint: it was trained on
/// AWGN with sigma in `[5, 55]` (of 255). Strength 1.0 maps to the top of the
/// trained range; pushing beyond it produces artifacts, not more denoising.
pub const FASTDVDNET_MAX_SIGMA: f32 = 55.0 / 255.0;

/// FastDVDnet's block downsamples twice and shuffles back up twice, so a side
/// that is not a multiple of four makes the graph's own skip addition mismatch
/// and the session fails outright — measured 2026-09-22 on the export's master:
/// 854 × 480 and 1280 × 718 both die in `/temp1_2/Add`, 856 × 480 and
/// 1920 × 1080 run. Upstream pads to the multiple and crops the answer back
/// (`fastdvdnet.temp_denoise`); so does this filter, the same way RIFE does.
const FASTDVDNET_ALIGNMENT: usize = 4;

/// FastDVDnet (MIT) behind [`TemporalNoiseFiltering`] — the off-macOS
/// `VideoNoiseFiltering` backing (migration 016).
///
/// Stateless five-frame window (`prev2 prev1 cur next1 next2` → the denoised
/// centre frame). The orchestrator already clips windows at scene cuts; when a
/// neighbor is missing (clip edge or cut) the nearest available frame is
/// REPLICATED — `[0, 0, 0, 1, 2]` at the first frame. Upstream REFLECTS there
/// (`[2, 1, 0, 1, 2]`, `fastdvdnet.denoise_seq_fastdvdnet`), so the two differ
/// on the two frames at each end of a clip and nowhere else.
pub struct FastDvdnetTemporalFilter {
    backend: Backend,
}

impl fmt::Debug for FastDvdnetTemporalFilter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FastDvdnetTemporalFilter")
            .finish_non_exhaustive()
    }
}

impl FastDvdnetTemporalFilter {
    pub fn new(backend: Backend) -> Self {
        Self { backend }
    }

    fn rgb(frame: &VideoFrame) -> Result<CpuFrame> {
        frame.to_cpu_rgb().map_err(|error| {
            VideoNoiseFilterError::Backend(format!("RGB download failed: {error}"))
        })
    }
}

/// `steps_back` = 1 selects the nearest previous frame, 2 the second-nearest;
/// missing neighbors replicate the closest available frame (or the fallback).
fn nth_previous<'a>(
    frames: &'a [VideoFrame],
    fallback: &'a VideoFrame,
    steps_back: usize,
) -> &'a VideoFrame {
    match frames.len() {
        0 => fallback,
        length => &frames[length.saturating_sub(steps_back)],
    }
}

fn nth_next<'a>(
    frames: &'a [VideoFrame],
    fallback: &'a VideoFrame,
    steps_forward: usize,
) -> &'a VideoFrame {
    match frames.len() {
        0 => fallback,
        length => &frames[(steps_forward - 1).min(length - 1)],
    }
}

#[async_trait(?Send)]
impl TemporalNoiseFiltering for FastDvdnetTemporalFilter {
    fn previous_frame_count(&self) -> usize {
        2
    }

    fn next_frame_count(&self) -> usize {
        2
    }

    fn start_session(&mut self, width: u32, height: u32, _prefer_hdr: bool) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(VideoNoiseFilterError::Backend(format!(
                "cannot denoise a zero-sized frame ({width}x{height})"
            )));
        }
        Ok(())
    }

    async fn filter(
        &mut self,
        frame: &VideoFrame,
        previous: &[VideoFrame],
        next: &[VideoFrame],
        strength: f32,
        _discontinuity: bool,
    ) -> Result<FrameBuffer> {
        VideoNoiseFilter::validate_strength(strength)?;
        let current = Self::rgb(frame)?;
        let window = [
            Self::rgb(nth_previous(previous, frame, 2))?,
            Self::rgb(nth_previous(previous, frame, 1))?,
            current,
            Self::rgb(nth_next(next, frame, 1))?,
            Self::rgb(nth_next(next, frame, 2))?,
        ];
        let (width, height) = (window[2].width, window[2].height);
        let padded_w = (width as usize).div_ceil(FASTDVDNET_ALIGNMENT) * FASTDVDNET_ALIGNMENT;
        let padded_h = (height as usize).div_ceil(FASTDVDNET_ALIGNMENT) * FASTDVDNET_ALIGNMENT;
        let plane = padded_w * padded_h;
        let mut values = Vec::with_capacity(15 * plane);
        for member in &window {
            if member.width != width || member.height != height {
                return Err(VideoNoiseFilterError::Backend(format!(
                    "temporal window mixes frame sizes: {width}x{height} vs {}x{}",
                    member.width, member.height
                )));
            }
            append_planar_rgb_padded(member, padded_w, padded_h, &mut values);
        }
        let mut inputs = TensorMap::new();
        let shape = |channels: i64| vec![1, channels, padded_h as i64, padded_w as i64];
        inputs.insert(
            "frames".into(),
            Tensor::from_f32(shape(15), values)
                .map_err(|error| VideoNoiseFilterError::Backend(error.to_string()))?,
        );
        inputs.insert(
            "noise_map".into(),
            Tensor::from_f32(shape(1), vec![strength * FASTDVDNET_MAX_SIGMA; plane])
                .map_err(|error| VideoNoiseFilterError::Backend(error.to_string()))?,
        );
        let outputs = self
            .backend
            .infer(ModelId::Fastdvdnet, &ModelVersion::Latest, inputs)
            .await
            .map_err(|error| VideoNoiseFilterError::Backend(error.to_string()))?;
        let tensor = outputs.get("denoised").ok_or_else(|| {
            VideoNoiseFilterError::Backend("FastDVDnet output missing 'denoised'".into())
        })?;
        let denoised = tensor
            .to_f32()
            .map_err(|error| VideoNoiseFilterError::Backend(error.to_string()))?;
        let cpu =
            planar_rgb_frame(&denoised, padded_w, padded_h, width, height).ok_or_else(|| {
                VideoNoiseFilterError::Backend("FastDVDnet returned a malformed tensor".into())
            })?;
        Ok(FrameBuffer::Cpu(cpu))
    }

    fn end_session(&mut self) {}
}

/// Result of the in-memory temporal-filter stage.
///
/// These frames are not a playable media artifact. A platform render adapter
/// must encode and mux them before satisfying the path-based engine capability.
#[derive(Debug, Clone)]
pub struct VideoNoiseFilterResult {
    pub frames: Vec<VideoFrame>,
    pub frames_written: usize,
    pub output_seconds: f64,
}

#[derive(Default, Clone)]
pub struct VideoNoiseFilter {
    pub(crate) backend: Option<Backend>,
}

impl std::fmt::Debug for VideoNoiseFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoNoiseFilter")
            .field("configured", &self.backend.is_some())
            .finish()
    }
}

impl VideoNoiseFilter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Neighbor indices for `index`, clipped to the containing scene and
    /// returned in ascending presentation order on both sides.
    #[must_use]
    pub fn neighbor_indices(
        index: usize,
        count: usize,
        radius: usize,
        scene_starts: &[usize],
        side: NeighborSide,
    ) -> Vec<usize> {
        if radius == 0 || index >= count {
            return Vec::new();
        }

        let scene_start = scene_starts
            .iter()
            .copied()
            .filter(|&start| start <= index)
            .max()
            .unwrap_or(0);
        let scene_end_exclusive = scene_starts
            .iter()
            .copied()
            .filter(|&start| start > index)
            .min()
            .unwrap_or(count)
            .min(count);

        match side {
            NeighborSide::Previous => {
                let first = index.saturating_sub(radius).max(scene_start);
                (first..index).collect()
            }
            NeighborSide::Next => {
                let first = index.saturating_add(1);
                let end = first.saturating_add(radius).min(scene_end_exclusive);
                (first..end).collect()
            }
        }
    }

    /// Validate the VideoToolbox-compatible strength range.
    pub fn validate_strength(strength: f32) -> Result<()> {
        if (0.0..=1.0).contains(&strength) {
            Ok(())
        } else {
            Err(VideoNoiseFilterError::InvalidStrength(strength))
        }
    }

    /// Apply the temporal-filter contract to already-decoded frames.
    ///
    /// `scene_starts` are source-timeline timestamps. A cut in
    /// `(previous_pts, current_pts]` starts a new scene at the current frame.
    /// No decode, encode, output path, or container operation occurs here.
    pub async fn filter_frames<F: TemporalNoiseFiltering + ?Sized>(
        &self,
        frames: &[VideoFrame],
        strength: f32,
        scene_starts: &[MediaTime],
        filter: &mut F,
    ) -> Result<VideoNoiseFilterResult> {
        Self::validate_strength(strength)?;
        let Some(first_frame) = frames.first() else {
            return Err(VideoNoiseFilterError::EmptyWindow);
        };

        let prefer_hdr = matches!(
            first_frame.color.transfer,
            TransferCharacteristic::Pq | TransferCharacteristic::Hlg
        );
        filter.start_session(first_frame.width, first_frame.height, prefer_hdr)?;

        let result = async {
            let scene_start_indices = scene_start_indices(frames, scene_starts);
            let mut output = Vec::with_capacity(frames.len());

            for (index, current) in frames.iter().enumerate() {
                let previous_indices = Self::neighbor_indices(
                    index,
                    frames.len(),
                    filter.previous_frame_count(),
                    &scene_start_indices,
                    NeighborSide::Previous,
                );
                let next_indices = Self::neighbor_indices(
                    index,
                    frames.len(),
                    filter.next_frame_count(),
                    &scene_start_indices,
                    NeighborSide::Next,
                );
                let discontinuity = scene_start_indices.binary_search(&index).is_ok();

                // VideoToolbox rejects a request without any reference frame.
                // Preserve a lone-frame scene unchanged instead of rejecting
                // the entire operation.
                let buffer = if previous_indices.is_empty() && next_indices.is_empty() {
                    current.buffer.clone()
                } else {
                    let previous = previous_indices
                        .iter()
                        .map(|&neighbor| frames[neighbor].clone())
                        .collect::<Vec<_>>();
                    let next = next_indices
                        .iter()
                        .map(|&neighbor| frames[neighbor].clone())
                        .collect::<Vec<_>>();
                    filter
                        .filter(current, &previous, &next, strength, discontinuity)
                        .await?
                };

                let mut filtered = current.clone();
                filtered.buffer = buffer;
                output.push(filtered);
            }

            Ok(VideoNoiseFilterResult {
                frames_written: output.len(),
                output_seconds: output.last().map_or(0.0, |frame| frame.pts.seconds()),
                frames: output,
            })
        }
        .await;

        filter.end_session();
        result
    }
}

#[async_trait(?Send)]
impl ferment_engine_core::VideoNoiseFiltering for VideoNoiseFilter {
    async fn render(
        &self,
        source_path: &std::path::Path,
        output_path: &std::path::Path,
        strength: f32,
        window: Option<ferment_engine_core::MediaTimeRange>,
        scene_starts: &[ferment_engine_core::MediaTime],
        progress: ProgressSink,
    ) -> ferment_engine_core::Result<ferment_engine_core::VideoRenderResult> {
        Self::validate_strength(strength)
            .map_err(|error| EngineError::InvalidConfiguration(error.to_string()))?;
        let backend = self
            .backend
            .clone()
            .ok_or_else(|| EngineError::Unsupported(RENDER_UNAVAILABLE_REASON.into()))?;
        crate::render_pipeline::denoise_video(
            backend,
            source_path,
            output_path,
            strength,
            window,
            scene_starts,
            progress,
        )
        .await
    }
}

fn scene_start_indices(frames: &[VideoFrame], scene_starts: &[MediaTime]) -> Vec<usize> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut indices = vec![0];
    for index in 1..frames.len() {
        let previous_seconds = frames[index - 1].pts.seconds();
        let current_seconds = frames[index].pts.seconds();
        if scene_starts.iter().any(|cut| {
            let cut_seconds = cut.seconds();
            cut_seconds > previous_seconds && cut_seconds <= current_seconds
        }) {
            indices.push(index);
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_core::VideoNoiseFiltering;
    use ferment_engine_media::{ColorInfo, CpuFrame, CpuPixelFormat, TimeBase};

    fn synthetic_frames(count: usize) -> Vec<VideoFrame> {
        let time_base = TimeBase::new_unchecked(1, 30);
        (0..count)
            .map(|index| {
                let value = u8::try_from(index).unwrap();
                VideoFrame {
                    width: 1,
                    height: 1,
                    pts: MediaTime::new(index as i64, time_base),
                    duration: Some(MediaTime::new(1, time_base)),
                    color: ColorInfo::default(),
                    buffer: FrameBuffer::Cpu(
                        CpuFrame::new(1, 1, 3, CpuPixelFormat::Rgb24, vec![value; 3]).unwrap(),
                    ),
                }
            })
            .collect()
    }

    #[test]
    fn fastdvdnet_filter_replicates_edges_and_packs_noise_map() {
        use std::sync::Arc;

        use crate::frame_tensors::testing::MockBackend;

        futures_executor::block_on(async {
            // A 1 × 1 frame pads to the graph's 4 × 4, and the answer is
            // cropped back to the caller's size.
            let backend = Arc::new(MockBackend::constant("denoised", vec![1, 3, 4, 4], 0.25));
            let mut filter = FastDvdnetTemporalFilter::new(backend.clone());
            assert_eq!(filter.previous_frame_count(), 2);
            assert_eq!(filter.next_frame_count(), 2);
            filter.start_session(1, 1, false).unwrap();

            // One previous neighbor, none forward: window must replicate to
            // [p, p, cur, cur, cur] — upstream FastDVDnet's edge handling.
            let frames = synthetic_frames(2);
            let (previous, current) = (&frames[..1], &frames[1]);
            let output = filter
                .filter(current, previous, &[], 0.5, false)
                .await
                .unwrap();
            filter.end_session();

            // Irrefutable once FrameBuffer's macOS/Windows GPU variants cfg out.
            #[allow(irrefutable_let_patterns)]
            let FrameBuffer::Cpu(cpu) = output else {
                panic!("expected a CPU output buffer");
            };
            assert_eq!(&cpu.data[..], &[64, 64, 64], "0.25 quantizes to 64");

            let captured = backend.captured.lock().unwrap();
            let (model, inputs) = &captured[0];
            assert_eq!(*model, ModelId::Fastdvdnet);
            let frames_tensor = &inputs["frames"];
            assert_eq!(frames_tensor.shape, vec![1, 15, 4, 4]);
            let values = frames_tensor.to_f32().unwrap();
            let previous_value = 0.0 / 255.0;
            let current_value = 1.0 / 255.0;
            // Fifteen padded planes: the window's five frames × RGB, each one
            // edge-replicated out of a single pixel, so every value in a plane
            // is that frame's.
            let expected: Vec<f32> = [
                previous_value,
                previous_value,
                current_value,
                current_value,
                current_value,
            ]
            .iter()
            .flat_map(|&value| [value; 3])
            .flat_map(|value| [value; 16])
            .collect();
            assert_eq!(values.len(), expected.len());
            for (actual, expected) in values.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
            let noise = inputs["noise_map"].to_f32().unwrap();
            assert_eq!(noise.len(), 16, "the noise plane is the padded one");
            assert!((noise[0] - 0.5 * FASTDVDNET_MAX_SIGMA).abs() < 1e-6);
        });
    }

    #[test]
    fn fastdvdnet_filter_rejects_invalid_strength() {
        use std::sync::Arc;

        use crate::frame_tensors::testing::MockBackend;

        futures_executor::block_on(async {
            let backend = Arc::new(MockBackend::constant("denoised", vec![1, 3, 1, 1], 0.0));
            let mut filter = FastDvdnetTemporalFilter::new(backend);
            let frames = synthetic_frames(1);
            let error = filter
                .filter(&frames[0], &[], &[], 1.5, false)
                .await
                .unwrap_err();
            assert!(matches!(error, VideoNoiseFilterError::InvalidStrength(_)));
        });
    }

    #[test]
    fn test_interior_frame_gets_full_windows() {
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(6, 24, 2, &[0], NeighborSide::Previous),
            vec![4, 5]
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(6, 24, 2, &[0], NeighborSide::Next),
            vec![7, 8]
        );
    }

    #[test]
    fn test_windows_trim_at_clip_edges() {
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(0, 24, 2, &[0], NeighborSide::Previous),
            Vec::<usize>::new()
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(23, 24, 2, &[0], NeighborSide::Next),
            Vec::<usize>::new()
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(1, 24, 2, &[0], NeighborSide::Previous),
            vec![0]
        );
    }

    #[test]
    fn test_windows_never_cross_a_scene_boundary() {
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(11, 24, 2, &[0, 12], NeighborSide::Next),
            Vec::<usize>::new()
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(12, 24, 2, &[0, 12], NeighborSide::Previous),
            Vec::<usize>::new()
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(13, 24, 2, &[0, 12], NeighborSide::Previous),
            vec![12]
        );
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(10, 24, 2, &[0, 12], NeighborSide::Next),
            vec![11]
        );
    }

    #[test]
    fn test_zero_radius_yields_empty_windows() {
        assert_eq!(
            VideoNoiseFilter::neighbor_indices(6, 24, 0, &[0], NeighborSide::Previous),
            Vec::<usize>::new()
        );
    }

    #[derive(Debug)]
    struct Call {
        pts_seconds: f64,
        previous_count: usize,
        next_count: usize,
        previous_pts: Vec<f64>,
        strength: f32,
        discontinuity: bool,
    }

    #[derive(Debug)]
    struct SpyNoiseFilter {
        previous_frame_count: usize,
        next_frame_count: usize,
        calls: Vec<Call>,
    }

    impl SpyNoiseFilter {
        fn new(previous_frame_count: usize, next_frame_count: usize) -> Self {
            Self {
                previous_frame_count,
                next_frame_count,
                calls: Vec::new(),
            }
        }
    }

    #[async_trait(?Send)]
    impl TemporalNoiseFiltering for SpyNoiseFilter {
        fn previous_frame_count(&self) -> usize {
            self.previous_frame_count
        }

        fn next_frame_count(&self) -> usize {
            self.next_frame_count
        }

        fn start_session(&mut self, _width: u32, _height: u32, _prefer_hdr: bool) -> Result<()> {
            Ok(())
        }

        async fn filter(
            &mut self,
            frame: &VideoFrame,
            previous: &[VideoFrame],
            next: &[VideoFrame],
            strength: f32,
            discontinuity: bool,
        ) -> Result<FrameBuffer> {
            self.calls.push(Call {
                pts_seconds: frame.pts.seconds(),
                previous_count: previous.len(),
                next_count: next.len(),
                previous_pts: previous.iter().map(|frame| frame.pts.seconds()).collect(),
                strength,
                discontinuity,
            });
            Ok(frame.buffer.clone())
        }

        fn end_session(&mut self) {}
    }

    #[test]
    fn test_filter_sees_scene_aware_windows_and_discontinuities() {
        futures_executor::block_on(async {
            let frames = synthetic_frames(24);
            let mut spy = SpyNoiseFilter::new(2, 2);
            let result = VideoNoiseFilter::new()
                .filter_frames(
                    &frames,
                    0.7,
                    &[MediaTime::new(12, TimeBase::new_unchecked(1, 30))],
                    &mut spy,
                )
                .await
                .unwrap();

            assert_eq!(
                result.frames_written, 24,
                "timing untouched - one output per source frame"
            );
            let calls = &spy.calls;
            assert_eq!(calls.len(), 24);
            assert!(calls.iter().all(|call| (call.strength - 0.7).abs() < 1e-5));

            let call = |frame: usize| {
                calls
                    .iter()
                    .find(|call| (call.pts_seconds - frame as f64 / 30.0).abs() < 0.001)
                    .unwrap()
            };
            let mut discontinuities = calls
                .iter()
                .filter(|call| call.discontinuity)
                .map(|call| call.pts_seconds)
                .collect::<Vec<_>>();
            discontinuities.sort_by(f64::total_cmp);
            assert_eq!(discontinuities, vec![0.0, 12.0 / 30.0]);

            assert_eq!(call(6).previous_count, 2);
            assert_eq!(call(6).next_count, 2);
            let previous_six = &call(6).previous_pts;
            let mut sorted_previous_six = previous_six.clone();
            sorted_previous_six.sort_by(f64::total_cmp);
            assert_eq!(
                previous_six, &sorted_previous_six,
                "previous frames must be ascending PTS (VT contract)"
            );
            assert_eq!(call(11).next_count, 0);
            assert_eq!(call(12).previous_count, 0);
            assert_eq!(call(13).previous_count, 1);
            assert_eq!(call(0).previous_count, 0);
            assert_eq!(call(23).next_count, 0);
        });
    }

    #[derive(Debug)]
    struct RejectingNoiseFilter {
        previous_frame_count: usize,
        next_frame_count: usize,
        filtered_pts: Vec<f64>,
    }

    #[async_trait(?Send)]
    impl TemporalNoiseFiltering for RejectingNoiseFilter {
        fn previous_frame_count(&self) -> usize {
            self.previous_frame_count
        }

        fn next_frame_count(&self) -> usize {
            self.next_frame_count
        }

        fn start_session(&mut self, _width: u32, _height: u32, _prefer_hdr: bool) -> Result<()> {
            Ok(())
        }

        async fn filter(
            &mut self,
            frame: &VideoFrame,
            previous: &[VideoFrame],
            next: &[VideoFrame],
            _strength: f32,
            _discontinuity: bool,
        ) -> Result<FrameBuffer> {
            if previous.is_empty() && next.is_empty() {
                return Err(VideoNoiseFilterError::EmptyWindow);
            }
            self.filtered_pts.push(frame.pts.seconds());
            Ok(frame.buffer.clone())
        }

        fn end_session(&mut self) {}
    }

    #[test]
    fn test_lone_frame_scene_is_passed_through_not_rejected() {
        futures_executor::block_on(async {
            let frames = synthetic_frames(4);
            let mut filter = RejectingNoiseFilter {
                previous_frame_count: 2,
                next_frame_count: 2,
                filtered_pts: Vec::new(),
            };
            let result = VideoNoiseFilter::new()
                .filter_frames(
                    &frames,
                    0.7,
                    &[MediaTime::new(3, TimeBase::new_unchecked(1, 30))],
                    &mut filter,
                )
                .await
                .unwrap();

            assert_eq!(
                result.frames_written, 4,
                "all frames emitted, including the passed-through lone frame"
            );
            assert!(
                !filter
                    .filtered_pts
                    .iter()
                    .any(|pts| (*pts - 3.0 / 30.0).abs() < 0.001),
                "the lone 1-frame scene must be passed through, not filtered"
            );
        });
    }

    #[test]
    fn test_invalid_strength_throws() {
        futures_executor::block_on(async {
            let mut filter = SpyNoiseFilter::new(0, 0);
            let error = VideoNoiseFilter::new()
                .filter_frames(&[], 1.5, &[], &mut filter)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                VideoNoiseFilterError::InvalidStrength(strength)
                    if (strength - 1.5).abs() < f32::EPSILON
            ));
        });
    }

    #[test]
    fn trait_render_validates_strength_then_gates_missing_output_boundary() {
        let processor = VideoNoiseFilter::new();
        let source = std::path::Path::new("unused-source.mov");
        let output = std::path::Path::new("unused-output.mov");

        let invalid = futures_executor::block_on(VideoNoiseFiltering::render(
            &processor,
            source,
            output,
            1.5,
            None,
            &[],
            ProgressSink::noop(),
        ))
        .unwrap_err();
        assert!(matches!(
            invalid,
            EngineError::InvalidConfiguration(reason)
                if reason == "noise-filter strength 1.5 is out of range (0-1)"
        ));

        let unavailable = futures_executor::block_on(VideoNoiseFiltering::render(
            &processor,
            source,
            output,
            0.7,
            None,
            &[],
            ProgressSink::noop(),
        ))
        .unwrap_err();
        assert!(matches!(
            unavailable,
            EngineError::Unsupported(reason) if reason == RENDER_UNAVAILABLE_REASON
        ));
    }
}
