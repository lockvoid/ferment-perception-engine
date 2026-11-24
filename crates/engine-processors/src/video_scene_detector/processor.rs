//! AutoShot scene detection from either a bounded sequential decoder window or
//! pre-extracted RGB frames. The buffered entry point is retained as an
//! independent comparison oracle for the production streaming path.

use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_media::VideoBackend;
use serde::{Deserialize, Serialize};

use super::config::{
    BATCH_SIZE, BORDER_SIZE, CENTER_SIZE, DEFAULT_THRESHOLD, FRAME_HEIGHT, FRAME_WIDTH,
    N_COLOR_BINS, PIXELS_PER_FRAME,
};
use super::embedding_tap::{Sample, SceneEmbeddingTap, scene_embeddings};
use super::histograms::color_histograms;
use super::jump_cut::{self, Detection as JumpCut};
use super::scenes::{cuts_from_one_hot, pad_for_batching, scenes_from_boundaries};
use super::shot_identity::{EncoderShotIdentity, TransientCollector};
use super::transient_verdict::Verdict;
use crate::dinov2::{Dinov2SemanticEncoder, FrameSnapshot};
use crate::video_source::{VideoSource, check_cancellation};

#[path = "streaming.rs"]
mod streaming;

/// Everything one decode pass yields: the model's frames plus the three
/// producer-side taps.
#[derive(Debug)]
pub struct Decoded {
    pub frames: Vec<u8>,
    pub frame_count: usize,
    pub frame_rate: f64,
    pub jump_cuts: Vec<JumpCut>,
    pub verdicts: Vec<Verdict>,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    pub frame: usize,
    pub time: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub index: usize,
    pub start_frame: usize,
    pub frame_count: usize,
    pub start_time: f64,
    pub duration: f64,
    pub start_confidence: Option<f32>,
    /// The scene's DINOv2 identity: the medoid of the samples the embedding tap
    /// took inside it. `None` on every scene or none of them — see
    /// [`super::embedding_tap::scene_embeddings`] for why it is all-or-nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSceneDetectionResult {
    pub cuts: Vec<Cut>,
    pub scenes: Vec<Scene>,
    pub total_frames: usize,
    pub frame_rate: f64,
}

#[derive(Debug, Clone)]
pub struct VideoSceneDetector {
    backend: Backend,
    version: ModelVersion,
    threshold: f32,
    video: VideoSource,
}

impl VideoSceneDetector {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
            threshold: DEFAULT_THRESHOLD,
            video: VideoSource::default(),
        }
    }

    #[must_use]
    pub fn with_video_backend(mut self, backend: VideoBackend) -> Self {
        self.video = VideoSource::new(backend);
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// One decode, every consumer. The model sees 27x48 RGB; the three
    /// producer-side passes share the current iOS analysis-size picture:
    /// up to 256 wide, with even dimensions when downscaled. The buffered
    /// profiler retains only model tiles; its snapshots stay analysis-sized.
    pub async fn decode_for_detection(&self, video_path: &std::path::Path) -> Result<Decoded> {
        let progress = ProgressSink::noop();
        let mut stream = self.video.open_scene_rgb(video_path, &progress).await?;
        let identity = EncoderShotIdentity::new(Dinov2SemanticEncoder::new(self.backend.clone()));
        let mut jump_cuts = jump_cut::Scanner::default();
        let mut transients = TransientCollector::new(identity.clone());
        let mut embeddings = SceneEmbeddingTap::new(identity, stream.frame_rate());

        let mut detections = Vec::new();
        let mut resized = Vec::new();
        while let Some(frame) = stream.next(&progress).await? {
            if let Some(luma) = jump_cut::analysis_luma(&frame) {
                detections.extend(jump_cuts.push(&luma));
                // The identity reads the same analysis-size pixels as the
                // scanner, matching the iOS single-reader snapshot geometry.
                let snapshot = FrameSnapshot::packed(
                    frame.width as usize,
                    frame.height as usize,
                    frame.data.clone(),
                );
                embeddings.push(&luma, &snapshot).await;
                transients.push(&luma, snapshot).await;
            }
            let image = image::RgbImage::from_raw(frame.width, frame.height, frame.data)
                .ok_or_else(|| EngineError::Decode("invalid decoded RGB frame".into()))?;
            resized.extend_from_slice(
                image::imageops::resize(
                    &image,
                    FRAME_WIDTH as u32,
                    FRAME_HEIGHT as u32,
                    image::imageops::FilterType::Triangle,
                )
                .as_raw(),
            );
        }
        if resized.is_empty() {
            return Err(EngineError::Decode(
                "video decoder produced no frames".into(),
            ));
        }
        Ok(Decoded {
            frame_count: resized.len() / PIXELS_PER_FRAME,
            frames: resized,
            frame_rate: stream.frame_rate(),
            jump_cuts: detections,
            verdicts: transients.into_verdicts(),
            samples: embeddings.into_samples(),
        })
    }

    /// The model's boundaries, corrected by the two post-passes, in the order
    /// that matters:
    ///
    /// 1. the **flash veto** first — a model boundary sitting on a photometric
    ///    transient is the light, not an edit, unless the shot changed across
    ///    it, and then it is ONE boundary at the frame the new shot starts;
    /// 2. the **jump-cut merge** second — same-camera edits AutoShot is blind
    ///    to. Model boundaries win collisions within three frames; a jump-cut
    ///    boundary carries its measured tear as confidence.
    ///
    /// The order is not cosmetic: running the merge first would let a jump cut
    /// land inside a transient the veto is about to collapse, and the veto only
    /// ever looks at boundaries, so it would not see it.
    #[must_use]
    pub fn merge_boundaries(
        result: VideoSceneDetectionResult,
        jump_cuts: &[JumpCut],
        verdicts: &[Verdict],
    ) -> VideoSceneDetectionResult {
        let model_boundaries: Vec<(usize, f32)> = result
            .cuts
            .iter()
            .map(|cut| (cut.frame, cut.confidence))
            .collect();
        let mut boundaries = apply_transients(verdicts, &model_boundaries);

        let kept: Vec<usize> = boundaries.iter().map(|(frame, _)| *frame).collect();
        for jump in jump_cuts {
            if jump.frame >= result.total_frames.saturating_sub(1)
                || kept.iter().any(|frame| frame.abs_diff(jump.frame) <= 3)
            {
                continue;
            }
            boundaries.push((jump.frame, jump.confidence));
        }
        boundaries.sort_by_key(|(frame, _)| *frame);

        VideoSceneDetectionResult {
            cuts: boundaries
                .iter()
                .map(|&(frame, confidence)| Cut {
                    frame,
                    time: frame as f64 / result.frame_rate,
                    confidence,
                })
                .collect(),
            scenes: scenes_from_boundaries(&boundaries, result.total_frames, result.frame_rate),
            total_frames: result.total_frames,
            frame_rate: result.frame_rate,
        }
    }

    /// The pass's identity samples folded onto the FINAL scene list (medoid per
    /// scene, nearest-sample fallback for slivers). No samples — no encoder in
    /// the fleet — leaves every `Scene::embedding` `None`.
    ///
    /// Must run last: the samples carry frame stamps, not scene indices, so
    /// they can only be assigned once the boundaries have settled.
    #[must_use]
    pub fn attach_embeddings(
        mut result: VideoSceneDetectionResult,
        samples: &[Sample],
    ) -> VideoSceneDetectionResult {
        let Some(per_scene) = scene_embeddings(samples, &result.scenes) else {
            return result;
        };
        for (scene, embedding) in result.scenes.iter_mut().zip(per_scene) {
            scene.embedding = Some(embedding);
        }
        result
    }

    /// Detect scenes from pre-extracted (T, 27, 48, RGB) uint8 frames.
    /// `frame_rate` is used only for absolute timestamps in the result.
    #[tracing::instrument(skip_all, fields(processor = "video_scene_detector", frame_count, frame_rate, threshold = self.threshold))]
    pub async fn detect(
        &self,
        rgb_frames: &[u8],
        frame_count: usize,
        frame_rate: f64,
        progress: ProgressSink,
    ) -> Result<VideoSceneDetectionResult> {
        check_cancellation(&progress)?;
        if !frame_rate.is_finite() || frame_rate <= 0.0 {
            return Err(EngineError::invalid_config(
                "Video has zero/unknown frame rate",
            ));
        }
        if frame_count == 0 {
            return Err(EngineError::invalid_config("frame_count must be > 0"));
        }
        let expected_bytes = frame_count * PIXELS_PER_FRAME;
        if rgb_frames.len() != expected_bytes {
            return Err(EngineError::invalid_config(format!(
                "rgb_frames size mismatch: got {}, expected {expected_bytes}",
                rgb_frames.len()
            )));
        }

        progress.report(0.05, "padding frames").await;
        let padded = pad_for_batching(rgb_frames, frame_count);
        let total_padded_frames = padded.len() / PIXELS_PER_FRAME;
        let n_batches = (total_padded_frames - CENTER_SIZE) / CENTER_SIZE;
        if n_batches == 0 {
            return Err(EngineError::invalid_config(
                "pipeline produced 0 batches — frame count too small",
            ));
        }

        progress.report(0.10, "detecting boundaries").await;
        let mut one_hot_all = Vec::<f32>::with_capacity(n_batches * CENTER_SIZE);
        let progress_share = 0.80 / n_batches as f64;

        for b in 0..n_batches {
            check_cancellation(&progress)?;
            let start_frame = b * CENTER_SIZE;
            let batch_bytes = BATCH_SIZE * PIXELS_PER_FRAME;
            let batch = &padded
                [start_frame * PIXELS_PER_FRAME..start_frame * PIXELS_PER_FRAME + batch_bytes];
            let hist = color_histograms(batch, BATCH_SIZE);
            check_cancellation(&progress)?;
            let one_hot = self.run_autoshot(batch, &hist).await?;
            check_cancellation(&progress)?;
            // Keep middle 50 outputs (predictions[25:75])
            one_hot_all.extend_from_slice(&one_hot[BORDER_SIZE..BORDER_SIZE + CENTER_SIZE]);
            progress
                .report(
                    0.10 + progress_share * (b + 1) as f64,
                    format!("batch {}/{n_batches}", b + 1),
                )
                .await;
        }

        // Trim to original frame count, sigmoid + threshold → cut frames.
        let one_hot_trimmed = &one_hot_all[..frame_count.min(one_hot_all.len())];
        let raw_cuts = cuts_from_one_hot(one_hot_trimmed, self.threshold);

        progress.report(0.95, "building scenes").await;
        let cuts: Vec<Cut> = raw_cuts
            .iter()
            .map(|&(frame, conf)| Cut {
                frame,
                time: frame as f64 / frame_rate,
                confidence: conf,
            })
            .collect();
        let scenes = scenes_from_boundaries(&raw_cuts, frame_count, frame_rate);

        check_cancellation(&progress)?;
        progress.report(1.0, "done").await;
        check_cancellation(&progress)?;
        Ok(VideoSceneDetectionResult {
            cuts,
            scenes,
            total_frames: frame_count,
            frame_rate,
        })
    }

    async fn run_autoshot(&self, frames: &[u8], color_hist: &[f32]) -> Result<Vec<f32>> {
        debug_assert_eq!(frames.len(), BATCH_SIZE * PIXELS_PER_FRAME);
        debug_assert_eq!(color_hist.len(), BATCH_SIZE * N_COLOR_BINS);

        // Build (1, 3, T, H, W) float frames from (T, H, W, RGB) uint8 source.
        // Source layout: [t][y][x][c]; dest layout: [c][t][y][x]
        let pixels_per_frame = FRAME_HEIGHT * FRAME_WIDTH;
        let stride_per_channel = BATCH_SIZE * pixels_per_frame;
        let mut frame_floats = vec![0.0f32; 3 * stride_per_channel];
        for t in 0..BATCH_SIZE {
            for y in 0..FRAME_HEIGHT {
                for x in 0..FRAME_WIDTH {
                    let src_idx = ((t * FRAME_HEIGHT + y) * FRAME_WIDTH + x) * 3;
                    let dst_idx = (t * FRAME_HEIGHT + y) * FRAME_WIDTH + x;
                    frame_floats[dst_idx] = f32::from(frames[src_idx]);
                    frame_floats[stride_per_channel + dst_idx] = f32::from(frames[src_idx + 1]);
                    frame_floats[2 * stride_per_channel + dst_idx] = f32::from(frames[src_idx + 2]);
                }
            }
        }

        let mut inputs = TensorMap::new();
        inputs.insert(
            "frames".into(),
            Tensor::from_f32(
                vec![
                    1,
                    3,
                    BATCH_SIZE as i64,
                    FRAME_HEIGHT as i64,
                    FRAME_WIDTH as i64,
                ],
                frame_floats,
            )?,
        );
        inputs.insert(
            "color_hist".into(),
            Tensor::from_f32(
                vec![1, BATCH_SIZE as i64, N_COLOR_BINS as i64],
                color_hist.to_vec(),
            )?,
        );

        let out = self
            .backend
            .infer(ModelId::Autoshot, &self.version, inputs)
            .await?;

        let one_hot = out
            .get("one_hot")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: "autoshot".into(),
                key: "one_hot".into(),
            })?
            .to_f32()?;
        if one_hot.len() < BATCH_SIZE {
            return Err(EngineError::InvalidOutput);
        }
        Ok(one_hot)
    }
}

/// The transient law applied to the model's boundaries: boundaries touching a
/// same-shot run are dropped; boundaries touching a shot-change run collapse
/// into ONE — at `cut_start − 1` when the verdict places the cut, else the most
/// confident of them; the rest stand.
///
/// Boundaries are attributed to the FIRST verdict that touches them, and the
/// verdicts are then walked in index order. Upstream walks a dictionary, whose
/// order Swift does not define; deciding in run order costs nothing and makes
/// the output reproducible.
#[must_use]
pub fn apply_transients(verdicts: &[Verdict], boundaries: &[(usize, f32)]) -> Vec<(usize, f32)> {
    if verdicts.is_empty() {
        return boundaries.to_vec();
    }
    let mut kept: Vec<(usize, f32)> = Vec::new();
    let mut touching: Vec<Vec<(usize, f32)>> = vec![Vec::new(); verdicts.len()];
    for &boundary in boundaries {
        match verdicts
            .iter()
            .position(|verdict| verdict.touches(boundary.0))
        {
            Some(index) => touching[index].push(boundary),
            None => kept.push(boundary),
        }
    }
    for (verdict, group) in verdicts.iter().zip(&touching) {
        if verdict.same_shot || group.is_empty() {
            continue;
        }
        // The first maximum, as upstream's `max(by:)` returns.
        let best = group
            .iter()
            .copied()
            .reduce(|best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .expect("the group is not empty");
        // `cut_start` is the first frame of the NEW shot; a boundary is the
        // last frame of the outgoing one.
        let frame = verdict.cut_start.map_or(best.0, |start| start - 1);
        if kept.iter().any(|(existing, _)| *existing == frame) {
            continue;
        }
        kept.push((frame, best.1));
    }
    kept.sort_by_key(|(frame, _)| *frame);
    kept
}

#[cfg(test)]
mod post_pass_tests {
    use super::*;
    use crate::video_scene_detector::transient::Run;

    fn result(boundaries: &[(usize, f32)], total_frames: usize) -> VideoSceneDetectionResult {
        VideoSceneDetectionResult {
            cuts: boundaries
                .iter()
                .map(|&(frame, confidence)| Cut {
                    frame,
                    time: frame as f64 / 10.0,
                    confidence,
                })
                .collect(),
            scenes: scenes_from_boundaries(boundaries, total_frames, 10.0),
            total_frames,
            frame_rate: 10.0,
        }
    }

    #[test]
    fn jump_postpass_preserves_model_boundaries_and_adds_distant_hits() {
        let merged = VideoSceneDetector::merge_boundaries(
            result(&[(10, 0.9)], 30),
            &[
                JumpCut {
                    frame: 12,
                    confidence: 0.4,
                },
                JumpCut {
                    frame: 20,
                    confidence: 0.5,
                },
                JumpCut {
                    frame: 29,
                    confidence: 0.8,
                },
            ],
            &[],
        );

        assert_eq!(
            merged
                .cuts
                .iter()
                .map(|cut| (cut.frame, cut.confidence))
                .collect::<Vec<_>>(),
            vec![(10, 0.9), (20, 0.5)]
        );
        assert_eq!(
            merged
                .scenes
                .iter()
                .map(|scene| scene.frame_count)
                .collect::<Vec<_>>(),
            vec![11, 10, 9]
        );
        assert_eq!(merged.scenes[1].start_confidence, Some(0.9));
        assert_eq!(merged.scenes[2].start_confidence, Some(0.5));
    }

    /// The veto runs first, so a jump cut three frames from a boundary the
    /// veto is about to move still loses to it.
    #[test]
    fn the_veto_runs_before_the_jump_cut_merge() {
        let verdict = Verdict {
            run: Run {
                first: 20,
                last: 21,
            },
            same_shot: false,
            cut_start: Some(21),
        };
        let merged = VideoSceneDetector::merge_boundaries(
            result(&[(19, 0.9), (22, 0.5)], 60),
            &[JumpCut {
                frame: 22,
                confidence: 0.4,
            }],
            &[verdict],
        );
        assert_eq!(
            merged.cuts.iter().map(|cut| cut.frame).collect::<Vec<_>>(),
            vec![20],
            "the two model boundaries collapse onto the placed cut and the jump \
             cut is inside the model's three-frame shadow"
        );
    }

    // TransientVerdictTests.swift — applied to the model's boundaries.

    const RUN: Run = Run {
        first: 20,
        last: 21,
    };

    fn verdict(same_shot: bool, cut_start: Option<usize>) -> Verdict {
        Verdict {
            run: RUN,
            same_shot,
            cut_start,
        }
    }

    #[test]
    fn boundaries_on_a_same_shot_run_are_dropped() {
        let out = apply_transients(
            &[verdict(true, None)],
            &[(5, 0.9), (19, 0.4), (22, 0.5), (60, 0.8)],
        );
        assert_eq!(
            out.iter().map(|(frame, _)| *frame).collect::<Vec<_>>(),
            vec![5, 60]
        );
    }

    #[test]
    fn boundaries_on_a_shot_change_run_become_one_at_the_placed_cut() {
        let out = apply_transients(
            &[verdict(false, Some(21))],
            &[(19, 0.9), (22, 0.5), (60, 0.8)],
        );
        assert_eq!(
            out.iter().map(|(frame, _)| *frame).collect::<Vec<_>>(),
            vec![20, 60]
        );
        assert!((out[0].1 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn unplaced_shot_change_keeps_the_models_best_boundary() {
        let out = apply_transients(&[verdict(false, None)], &[(19, 0.4), (22, 0.7)]);
        assert_eq!(
            out.iter().map(|(frame, _)| *frame).collect::<Vec<_>>(),
            vec![22]
        );
    }

    #[test]
    fn boundaries_away_from_runs_stand() {
        let out = apply_transients(&[verdict(true, None)], &[(3, 0.5), (40, 0.5)]);
        assert_eq!(
            out.iter().map(|(frame, _)| *frame).collect::<Vec<_>>(),
            vec![3, 40]
        );
    }

    #[test]
    fn no_verdicts_leaves_every_boundary_untouched() {
        let boundaries = [(3, 0.5), (19, 0.4), (40, 0.5)];
        assert_eq!(apply_transients(&[], &boundaries), boundaries.to_vec());
    }

    /// A shot-change run with no boundary of its own invents none: the model
    /// never saw a cut there, and the veto only ever removes or moves.
    #[test]
    fn a_shot_change_run_the_model_missed_adds_no_boundary() {
        assert_eq!(
            apply_transients(&[verdict(false, Some(21))], &[(60, 0.8)]),
            vec![(60, 0.8)]
        );
    }

    // MARK: - Embeddings on the final scenes

    #[test]
    fn embeddings_are_attached_all_or_nothing() {
        let detected = result(&[(29, 0.9)], 100);
        let untouched = VideoSceneDetector::attach_embeddings(detected.clone(), &[]);
        assert!(
            untouched
                .scenes
                .iter()
                .all(|scene| scene.embedding.is_none())
        );

        let attached = VideoSceneDetector::attach_embeddings(
            detected,
            &[
                Sample {
                    frame: 5,
                    vector: vec![1.0],
                },
                Sample {
                    frame: 50,
                    vector: vec![2.0],
                },
            ],
        );
        assert_eq!(
            attached
                .scenes
                .iter()
                .map(|scene| scene.embedding.clone())
                .collect::<Vec<_>>(),
            vec![Some(vec![1.0]), Some(vec![2.0])]
        );
    }
}

#[cfg(all(
    test,
    any(feature = "ort", all(feature = "coreml", target_vendor = "apple"))
))]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    #[cfg(all(feature = "coreml", not(feature = "ort"), target_vendor = "apple"))]
    use ferment_engine_infer::CoreMlBackend;
    use ferment_engine_infer::ModelPaths;

    use super::*;

    fn fixture_directory() -> PathBuf {
        ferment_engine_test_fixtures::path("VideoSceneDetector/test_frames.bin")
            .expect("original AutoShot synthetic frames are required")
            .parent()
            .unwrap()
            .to_owned()
    }

    #[cfg(all(not(feature = "ort"), feature = "coreml", target_vendor = "apple"))]
    fn model_directory() -> PathBuf {
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../ferment-cuts-ios/Packages/FermentEngine/LocalModels")
            })
    }

    fn current_detector() -> VideoSceneDetector {
        #[cfg(feature = "ort")]
        {
            use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};
            let model = ferment_engine_test_fixtures::export::artifact("autoshot.onnx")
                .expect("approved AutoShot ONNX master is required");
            let paths = ModelPaths::new().with_path(ModelId::Autoshot, ModelVersion::Latest, model);
            VideoSceneDetector::new(Arc::new(OrtBackend::with_execution_provider(
                Arc::new(paths),
                OrtExecutionProvider::Cpu,
            )))
        }
        #[cfg(all(not(feature = "ort"), feature = "coreml", target_vendor = "apple"))]
        {
            let model = model_directory().join("autoshot.mlmodelc");
            assert!(
                model.is_dir(),
                "AutoShot Core ML model not found at {}; set FERMENT_ENGINE_MODELS_DIR",
                model.display()
            );
            let mut paths = ModelPaths::new();
            paths.insert(ModelId::Autoshot, ModelVersion::Latest, model);
            VideoSceneDetector::new(Arc::new(CoreMlBackend::new(Arc::new(paths))))
        }
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    // VideoSceneDetectorPipelineTests.swift — real artifact parity gates.

    // Kill: change logits or sigmoid beyond the original checkpoint bounds.
    #[test]
    #[ignore = "requires original Swift AutoShot fixtures and selected actual model fleet"]
    fn model_matches_pytorch_reference() {
        let directory = fixture_directory();
        let frames = std::fs::read(directory.join("test_frames.bin")).unwrap();
        let reference = read_f32(&directory.join("test_one_hot.bin"));
        assert_eq!(reference.len(), BATCH_SIZE);
        let histogram = color_histograms(&frames, BATCH_SIZE);
        let actual =
            futures_executor::block_on(current_detector().run_autoshot(&frames, &histogram))
                .unwrap();
        let mut max_logit_difference = 0.0_f32;
        let mut max_sigmoid_difference = 0.0_f32;
        for (actual, reference) in actual.iter().zip(reference) {
            max_logit_difference = max_logit_difference.max((actual - reference).abs());
            max_sigmoid_difference = max_sigmoid_difference.max(
                (super::super::scenes::sigmoid(*actual) - super::super::scenes::sigmoid(reference))
                    .abs(),
            );
        }
        assert!(
            max_logit_difference < 3.5e-2,
            "one_hot logit drift: {max_logit_difference}"
        );
        assert!(
            max_sigmoid_difference < 5e-3,
            "sigmoid drift: {max_sigmoid_difference}"
        );
    }

    // Kill: return wrong scene boundaries, fields or confidence for original frames.
    #[test]
    #[ignore = "requires original Swift AutoShot fixtures and selected actual model fleet"]
    fn end_to_end_on_synthetic_three_shot_clip() {
        let frames = std::fs::read(fixture_directory().join("test_frames.bin")).unwrap();
        let result = futures_executor::block_on(current_detector().detect(
            &frames,
            100,
            30.0,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert_eq!(result.total_frames, 100);
        assert!((result.frame_rate - 30.0).abs() < 0.001);
        assert!((result.total_frames as f64 / result.frame_rate - 100.0 / 30.0).abs() < 1e-6);
        assert_eq!(result.scenes.len(), 3);
        assert_eq!(
            (result.scenes[0].start_frame, result.scenes[0].frame_count),
            (0, 30)
        );
        assert_eq!(
            (result.scenes[1].start_frame, result.scenes[1].frame_count),
            (30, 40)
        );
        assert_eq!(
            (result.scenes[2].start_frame, result.scenes[2].frame_count),
            (70, 30)
        );
        assert_eq!(result.scenes[0].start_confidence, None);
        for scene in &result.scenes[1..] {
            let confidence = scene.start_confidence.expect("detected scene confidence");
            assert!(confidence > DEFAULT_THRESHOLD && confidence <= 1.0);
            assert!((scene.start_time - scene.start_frame as f64 / 30.0).abs() < 1e-6);
        }
        assert_eq!(
            result
                .scenes
                .iter()
                .map(|scene| scene.frame_count)
                .sum::<usize>(),
            100
        );
        assert!((result.scenes[0].start_time - 0.0).abs() < 1e-6);
        assert!((result.scenes[0].duration - 1.0).abs() < 1e-6);
    }
}
