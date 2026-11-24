//! Plan -> grab -> measure -> reduce. Port of the Swift `VideoSaliencyAnalyzer`
//! orchestration, with the Vision requests replaced by the two injected
//! sources.

use std::path::Path;
use std::sync::Arc;

use ferment_engine_core::{
    EngineError, FaceRectSource, ProgressSink, Result, RgbFrameRef, SaliencyHeatSource, VideoScene,
};
use ferment_engine_media::VideoBackend;

use super::config::VideoSaliencyConfig;
use super::grabber::{SparseGrabber, frame_ref};
use super::sample_plan::{SampleEntry, plan};
use super::{VideoSaliencyResult, VideoSaliencySample, reduce};

pub struct VideoSaliencyAnalyzer {
    heat: Arc<dyn SaliencyHeatSource>,
    faces: Arc<dyn FaceRectSource>,
    video: VideoBackend,
    config: VideoSaliencyConfig,
}

impl std::fmt::Debug for VideoSaliencyAnalyzer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VideoSaliencyAnalyzer")
            .field("mode", &self.heat.mode())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl VideoSaliencyAnalyzer {
    pub fn new(heat: Arc<dyn SaliencyHeatSource>, faces: Arc<dyn FaceRectSource>) -> Self {
        Self {
            heat,
            faces,
            video: platform_video_backend(),
            config: VideoSaliencyConfig::default(),
        }
    }

    #[must_use]
    pub fn with_video_backend(mut self, backend: VideoBackend) -> Self {
        self.video = backend;
        self
    }

    /// The grabs this analyzer wants for `scenes`, ascending in time.
    pub fn plan(&self, scenes: &[VideoScene]) -> Vec<SampleEntry> {
        plan(scenes, &self.config)
    }

    /// Measure one already-grabbed frame against its plan entry. Callers that
    /// own decode drive [`Self::plan`] + this + [`Self::assemble`] directly.
    pub async fn measure(
        &self,
        entry: &SampleEntry,
        frame: RgbFrameRef<'_>,
    ) -> Result<VideoSaliencySample> {
        super::measure::measure(
            self.heat.as_ref(),
            self.faces.as_ref(),
            frame,
            entry.scene_index,
            entry.time,
            &self.config,
        )
        .await
    }

    /// Samples -> the result: sorted by time, one verdict per input scene.
    pub fn assemble(
        &self,
        frame_size: (u32, u32),
        scene_count: usize,
        mut samples: Vec<VideoSaliencySample>,
    ) -> VideoSaliencyResult {
        samples.sort_by(|a, b| a.time.total_cmp(&b.time));
        VideoSaliencyResult {
            frame_width: frame_size.0,
            frame_height: frame_size.1,
            mode: self.heat.mode(),
            scenes: reduce::verdicts(&samples, scene_count),
            samples,
        }
    }

    /// Decode `video_path` sparsely and measure every planned grab. A frame
    /// that fails to decode is logged and skipped — one bad seek is not a
    /// failed analysis.
    pub async fn analyze(
        &self,
        video_path: &Path,
        scenes: &[VideoScene],
        progress: ProgressSink,
    ) -> Result<VideoSaliencyResult> {
        if scenes.is_empty() {
            return Err(EngineError::invalid_config("empty scene list"));
        }
        let entries = self.plan(scenes);
        let (mut grabber, frame_size) =
            SparseGrabber::open(&self.video, video_path, self.config.max_long_edge).await?;

        let total = entries.len().max(1);
        let mut samples = Vec::with_capacity(entries.len());
        for (done, entry) in entries.iter().enumerate() {
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            match grabber.grab(entry).await {
                Ok(Some(image)) => samples.push(self.measure(entry, frame_ref(&image)?).await?),
                Ok(None) => tracing::warn!(
                    time = entry.time,
                    "saliency: no frame decoded at or after the planned time"
                ),
                Err(error) => {
                    tracing::warn!(time = entry.time, %error, "saliency: grab failed");
                }
            }
            progress
                .report(
                    (done + 1) as f64 / total as f64,
                    format!("saliency {}/{total}", done + 1),
                )
                .await;
        }
        progress.report(1.0, "saliency done").await;
        Ok(self.assemble(frame_size, scenes.len(), samples))
    }
}

fn platform_video_backend() -> VideoBackend {
    #[cfg(target_os = "macos")]
    {
        Arc::new(ferment_engine_media::platform::macos::VideoToolboxVideoIo)
    }
    #[cfg(target_os = "windows")]
    {
        Arc::new(ferment_engine_media::platform::libav::LibavVideoIo)
    }
    #[cfg(target_os = "android")]
    {
        Arc::new(ferment_engine_media::platform::android::AndroidVideoIo)
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "android")))]
    {
        Arc::new(ferment_engine_media::UnavailableVideoIo)
    }
}

#[cfg(test)]
mod tests {
    //! The orchestration, driven through `plan` + `measure` + `assemble` with
    //! stand-in sources: no decode, no inference, every platform. The
    //! model-backed end-to-end lives in `tests/ort_saliency.rs`.

    use ferment_engine_core::{FaceRect, SaliencyHeatMap, SaliencyMode};

    use super::*;
    use crate::video_saliency::{SampleSource, VideoSaliencySceneVerdict};

    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 288;
    const BACKGROUND: u8 = 58;
    const SUBJECT: u8 = 238;

    /// Downsampled luma. Trivial, but a real heat source: it turns a bright
    /// subject on a dark field into exactly the map the moments consume.
    #[derive(Debug)]
    struct LumaHeat {
        mode: SaliencyMode,
    }

    #[async_trait::async_trait(?Send)]
    impl SaliencyHeatSource for LumaHeat {
        fn mode(&self) -> SaliencyMode {
            self.mode
        }

        async fn heat(&self, frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap> {
            let (grid_w, grid_h) = (64usize, 64usize);
            let mut values = vec![0.0f32; grid_w * grid_h];
            for (index, value) in values.iter_mut().enumerate() {
                let x = (index % grid_w) * frame.width as usize / grid_w;
                let y = (index / grid_w) * frame.height as usize / grid_h;
                let pixel = (y * frame.width as usize + x) * 3;
                *value = f32::from(frame.data[pixel]) / 255.0;
            }
            Ok(SaliencyHeatMap {
                values,
                width: grid_w,
                height: grid_h,
            })
        }
    }

    #[derive(Debug, Default)]
    struct FixedFaces(Vec<FaceRect>);

    #[async_trait::async_trait(?Send)]
    impl FaceRectSource for FixedFaces {
        async fn faces(&self, _frame: RgbFrameRef<'_>) -> Result<Vec<FaceRect>> {
            Ok(self.0.clone())
        }
    }

    fn disc(cx: f64, cy: f64) -> Vec<u8> {
        let radius = 0.12 * f64::from(HEIGHT);
        let mut data = vec![BACKGROUND; (3 * WIDTH * HEIGHT) as usize];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = f64::from(x) + 0.5 - cx * f64::from(WIDTH);
                let dy = f64::from(y) + 0.5 - cy * f64::from(HEIGHT);
                if dx * dx + dy * dy <= radius * radius {
                    let offset = ((y * WIDTH + x) * 3) as usize;
                    data[offset..offset + 3].fill(SUBJECT);
                }
            }
        }
        data
    }

    fn analyzer(mode: SaliencyMode, faces: Vec<FaceRect>) -> VideoSaliencyAnalyzer {
        VideoSaliencyAnalyzer::new(Arc::new(LumaHeat { mode }), Arc::new(FixedFaces(faces)))
    }

    fn scene(index: usize, start: f64, duration: f64) -> VideoScene {
        VideoScene {
            index,
            start_frame: (start * 30.0).round() as usize,
            frame_count: (duration * 30.0).round() as usize,
            start_time: start,
            duration,
            start_confidence: None,
            embedding: None,
        }
    }

    /// The same two-scene shape as the committed end-to-end clip: 4 s parked at
    /// (0.30, 0.50), then 6 s crossing x 0.15 -> 0.85.
    fn two_scene_run(analyzer: &VideoSaliencyAnalyzer) -> VideoSaliencyResult {
        let scenes = [scene(0, 0.0, 4.0), scene(1, 4.0, 6.0)];
        let entries = analyzer.plan(&scenes);
        let samples = entries
            .iter()
            .map(|entry| {
                let cx = if entry.scene_index == 0 {
                    0.30
                } else {
                    let progress = (entry.time - 4.6) / 4.8;
                    0.15 + 0.70 * progress
                };
                let pixels = disc(cx, 0.50);
                let frame = RgbFrameRef::new(&pixels, WIDTH, HEIGHT).unwrap();
                futures_executor::block_on(analyzer.measure(entry, frame)).unwrap()
            })
            .collect();
        analyzer.assemble((WIDTH, HEIGHT), scenes.len(), samples)
    }

    #[test]
    fn a_still_scene_gives_a_still_verdict_on_the_subject() {
        let result = two_scene_run(&analyzer(SaliencyMode::Objectness, Vec::new()));
        let verdict = result.scenes[0];
        assert_eq!(verdict.sample_count, 3); // 4 s x 0.5/s = 2 -> min 3
        assert!((verdict.cx - 0.30).abs() < 0.02, "cx {}", verdict.cx);
        assert!((verdict.cy - 0.50).abs() < 0.02, "cy {}", verdict.cy);
        assert!(verdict.spread < 0.05, "spread {}", verdict.spread);
        assert_eq!(verdict.face_share, 0.0);
        assert!(
            result
                .samples
                .iter()
                .all(|sample| sample.source == SampleSource::Saliency)
        );
    }

    #[test]
    fn a_travelling_scene_keeps_its_curve() {
        let result = two_scene_run(&analyzer(SaliencyMode::Objectness, Vec::new()));
        let verdict = result.scenes[1];
        assert_eq!(verdict.sample_count, 3); // 6 s x 0.5/s = 3
        assert!(verdict.spread > 0.15, "spread {}", verdict.spread);
        let travelling: Vec<f64> = result
            .samples
            .iter()
            .filter(|sample| sample.scene_index == 1)
            .map(|sample| sample.cx)
            .collect();
        assert!(
            travelling.windows(2).all(|pair| pair[0] < pair[1]),
            "cx must rise with time: {travelling:?}"
        );
        assert!(travelling[0] < 0.35);
        assert!(*travelling.last().unwrap() > 0.65);
    }

    /// A face wins outright, and its box is the sample — the heat map under it
    /// is not consulted.
    #[test]
    fn a_face_beats_the_heat_map() {
        let face = FaceRect {
            x: 0.60,
            y: 0.10,
            width: 0.10,
            height: 0.20,
            confidence: 0.93,
        };
        let result = two_scene_run(&analyzer(SaliencyMode::Objectness, vec![face]));
        assert!(
            result
                .samples
                .iter()
                .all(|sample| sample.source == SampleSource::Face)
        );
        let sample = result.samples[0];
        assert!((sample.cx - 0.65).abs() < 1e-12);
        assert!((sample.cy - 0.20).abs() < 1e-12);
        assert!((sample.w - 0.10).abs() < 1e-12);
        assert!((sample.confidence - 0.93).abs() < 1e-12);
        assert_eq!(result.scenes[0].face_share, 1.0);
    }

    /// The result reports the heat source's OWN mode. Hard-coding `objectness`
    /// at this layer would pass every other test in this file.
    #[test]
    fn the_result_reports_the_heat_sources_own_mode() {
        for mode in [SaliencyMode::Attention, SaliencyMode::Objectness] {
            let result = two_scene_run(&analyzer(mode, Vec::new()));
            assert_eq!(result.mode, mode);
        }
    }

    #[test]
    fn assemble_sorts_by_time_and_answers_every_scene() {
        let analyzer = analyzer(SaliencyMode::Objectness, Vec::new());
        let sample = |scene_index: usize, time: f64| VideoSaliencySample {
            scene_index,
            time,
            cx: 0.5,
            cy: 0.5,
            w: 0.2,
            h: 0.2,
            confidence: 0.5,
            source: SampleSource::Saliency,
        };
        let result = analyzer.assemble(
            (WIDTH, HEIGHT),
            3,
            vec![sample(2, 9.0), sample(0, 1.0), sample(0, 5.0)],
        );
        assert_eq!(
            result.samples.iter().map(|s| s.time).collect::<Vec<_>>(),
            [1.0, 5.0, 9.0]
        );
        assert_eq!(
            result
                .scenes
                .iter()
                .map(|v| v.sample_count)
                .collect::<Vec<_>>(),
            [2, 0, 1]
        );
        assert_eq!(result.scenes[1], VideoSaliencySceneVerdict::no_samples(1));
        assert_eq!(result.frame_width, WIDTH);
        assert_eq!(result.frame_height, HEIGHT);
    }
}
