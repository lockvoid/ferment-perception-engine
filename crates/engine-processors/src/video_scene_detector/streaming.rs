//! One decoder, a 100-frame RGB window, and the same central 50 predictions as
//! the buffered oracle. No producer task or channel can retain the rest of a clip.

use super::*;
use crate::video_scene_detector::shot_identity::ShotIdentifying;
use async_trait::async_trait;
use std::path::Path;

/// Prefix the first picture 25 times, slide by 50 after each inference, and
/// replicate only the last picture at EOF. Capacity does not depend on duration.
struct ModelWindow {
    pixels: Vec<u8>,
    last_frame: Vec<u8>,
    frame_count: usize,
    batches: usize,
}

impl ModelWindow {
    fn new() -> Self {
        Self {
            pixels: Vec::with_capacity(BATCH_SIZE * PIXELS_PER_FRAME),
            last_frame: Vec::with_capacity(PIXELS_PER_FRAME),
            frame_count: 0,
            batches: 0,
        }
    }

    fn push(&mut self, frame: &[u8]) {
        assert_eq!(frame.len(), PIXELS_PER_FRAME);
        assert!(
            !self.ready(),
            "consume a full model window before decoding again"
        );
        if self.frame_count == 0 {
            for _ in 0..BORDER_SIZE {
                self.pixels.extend_from_slice(frame);
            }
        }
        self.pixels.extend_from_slice(frame);
        self.last_frame.clear();
        self.last_frame.extend_from_slice(frame);
        self.frame_count += 1;
    }

    fn ready(&self) -> bool {
        self.pixels.len() == BATCH_SIZE * PIXELS_PER_FRAME
    }

    fn consume(&mut self) {
        assert!(self.ready());
        self.pixels.drain(..CENTER_SIZE * PIXELS_PER_FRAME);
        self.batches += 1;
    }

    fn finished(&self) -> bool {
        self.batches * CENTER_SIZE >= self.frame_count
    }

    fn pad_tail(&mut self) {
        assert!(self.frame_count > 0);
        while !self.ready() {
            self.pixels.extend_from_slice(&self.last_frame);
        }
    }
}

/// A cancelled transient judge can ask for additional run frames. They abstain
/// without entering another DINO graph; the outer pipeline then returns Cancelled.
#[derive(Clone)]
struct CancellableIdentity {
    inner: EncoderShotIdentity,
    progress: ProgressSink,
}

#[async_trait(?Send)]
impl ShotIdentifying for CancellableIdentity {
    async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32> {
        if self.progress.is_closed() {
            return Vec::new();
        }
        let vector = self.inner.embedding(frame).await;
        if self.progress.is_closed() {
            Vec::new()
        } else {
            vector
        }
    }
    fn same_shot_distance(&self) -> f32 {
        self.inner.same_shot_distance()
    }
    fn similar_shot_distance(&self) -> f32 {
        self.inner.similar_shot_distance()
    }
}

impl VideoSceneDetector {
    /// The production path: model, jump cuts, transient veto and scene medoids
    /// all consume the same sequential decode. Only result-sized logits/samples
    /// and the transient scanner's fixed look-behind survive a frame.
    pub async fn detect_streaming(
        &self,
        video_path: &Path,
        progress: ProgressSink,
    ) -> Result<VideoSceneDetectionResult> {
        self.detect_streaming_with_options(video_path, true, true, progress)
            .await
    }

    /// Diagnostic switches matching iOS's `jumpCuts` / `transientVeto` options.
    /// With both off, compare precisely the same model and threshold against
    /// `detect` without confusing a deliberate flash veto with lost predictions.
    pub async fn detect_streaming_with_options(
        &self,
        video_path: &Path,
        include_jump_cuts: bool,
        transient_veto: bool,
        progress: ProgressSink,
    ) -> Result<VideoSceneDetectionResult> {
        check_cancellation(&progress)?;
        progress.report(0.02, "decoding video").await;
        let mut stream = self.video.open_scene_rgb(video_path, &progress).await?;
        let expected_frames = stream.expected_frame_count();
        let identity = CancellableIdentity {
            inner: EncoderShotIdentity::new(Dinov2SemanticEncoder::new(self.backend.clone())),
            progress: progress.clone(),
        };
        let mut transients = transient_veto.then(|| TransientCollector::new(identity.clone()));
        let mut embeddings =
            transient_veto.then(|| SceneEmbeddingTap::new(identity, stream.frame_rate()));
        let mut jump_scanner = jump_cut::Scanner::default();
        let mut jump_cuts = Vec::new();
        let mut window = ModelWindow::new();
        let mut one_hot_all = Vec::new();
        while let Some(frame) = stream.next(&progress).await? {
            check_cancellation(&progress)?;
            if (include_jump_cuts || transient_veto)
                && let Some(luma) = jump_cut::analysis_luma(&frame)
            {
                if include_jump_cuts {
                    jump_cuts.extend(jump_scanner.push(&luma));
                }
                if transient_veto {
                    let snapshot = FrameSnapshot::packed(
                        frame.width as usize,
                        frame.height as usize,
                        frame.data.clone(),
                    );
                    if let Some(tap) = embeddings.as_mut() {
                        tap.push(&luma, &snapshot).await;
                        check_cancellation(&progress)?;
                    }
                    if let Some(collector) = transients.as_mut() {
                        collector.push(&luma, snapshot).await;
                        check_cancellation(&progress)?;
                    }
                }
            }
            let image = image::RgbImage::from_raw(frame.width, frame.height, frame.data)
                .ok_or_else(|| EngineError::Decode("invalid decoded RGB frame".into()))?;
            let small = image::imageops::resize(
                &image,
                FRAME_WIDTH as u32,
                FRAME_HEIGHT as u32,
                image::imageops::FilterType::Triangle,
            );
            check_cancellation(&progress)?;
            window.push(small.as_raw());
            if window.ready() {
                self.infer_streaming_window(
                    &mut window,
                    &mut one_hot_all,
                    expected_frames,
                    &progress,
                )
                .await?;
            }
        }
        let frame_rate = stream.frame_rate();
        drop(stream);
        if window.frame_count == 0 {
            return Err(EngineError::Decode(
                "video decoder produced no frames".into(),
            ));
        }
        while !window.finished() {
            check_cancellation(&progress)?;
            window.pad_tail();
            self.infer_streaming_window(&mut window, &mut one_hot_all, expected_frames, &progress)
                .await?;
        }
        one_hot_all.truncate(window.frame_count);
        let boundaries = cuts_from_one_hot(&one_hot_all, self.threshold);
        let model = VideoSceneDetectionResult {
            cuts: boundaries
                .iter()
                .map(|&(frame, confidence)| Cut {
                    frame,
                    time: frame as f64 / frame_rate,
                    confidence,
                })
                .collect(),
            scenes: scenes_from_boundaries(&boundaries, window.frame_count, frame_rate),
            total_frames: window.frame_count,
            frame_rate,
        };
        check_cancellation(&progress)?;
        progress.report(0.95, "building scenes").await;
        let verdicts = transients
            .map(TransientCollector::into_verdicts)
            .unwrap_or_default();
        let samples = embeddings
            .map(SceneEmbeddingTap::into_samples)
            .unwrap_or_default();
        let merged = Self::merge_boundaries(model, &jump_cuts, &verdicts);
        let result = Self::attach_embeddings(merged, &samples);
        check_cancellation(&progress)?;
        progress.report(1.0, "done").await;
        check_cancellation(&progress)?;
        Ok(result)
    }

    async fn infer_streaming_window(
        &self,
        window: &mut ModelWindow,
        one_hot_all: &mut Vec<f32>,
        expected_frames: Option<usize>,
        progress: &ProgressSink,
    ) -> Result<()> {
        check_cancellation(progress)?;
        let histograms = color_histograms(&window.pixels, BATCH_SIZE);
        check_cancellation(progress)?;
        let one_hot = self.run_autoshot(&window.pixels, &histograms).await?;
        check_cancellation(progress)?;
        one_hot_all.extend_from_slice(&one_hot[BORDER_SIZE..BORDER_SIZE + CENTER_SIZE]);
        window.consume();
        let completed = one_hot_all.len() as f64;
        let fraction = match expected_frames {
            Some(expected) => completed / expected as f64,
            // Without trustworthy metadata, approach the decode ceiling and
            // reserve the final 10% for EOF and scene assembly.
            None => completed / (completed + CENTER_SIZE as f64),
        };
        progress
            .report(
                0.05 + 0.85 * fraction.clamp(0.0, 1.0),
                format!("batch {}", window.batches),
            )
            .await;
        check_cancellation(progress)
    }
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
