//! Frame batching/padding + cut→scene conversion. All pure functions.

use super::config::{BORDER_SIZE, CENTER_SIZE, PIXELS_PER_FRAME};
use super::processor::Scene;

/// Pad raw RGB frames to a length divisible into 100-frame batches with
/// 50-frame stride. Mirrors AutoShot's `utils.get_batches`:
///   - 25 copies of the first frame at the start
///   - copies of the last frame at the end so total length is a multiple
///     of 50, then +25
pub fn pad_for_batching(frames: &[u8], n_frames: usize) -> Vec<u8> {
    debug_assert_eq!(frames.len(), n_frames * PIXELS_PER_FRAME);
    let mut reminder = CENTER_SIZE - n_frames % CENTER_SIZE;
    if reminder == CENTER_SIZE {
        reminder = 0;
    }
    let prefix = BORDER_SIZE;
    let suffix = reminder + BORDER_SIZE;
    let total = prefix + n_frames + suffix;
    let mut out = vec![0u8; total * PIXELS_PER_FRAME];

    if n_frames == 0 {
        return out;
    }

    // Prefix: 25 copies of frame[0]
    for i in 0..prefix {
        let dst_off = i * PIXELS_PER_FRAME;
        out[dst_off..dst_off + PIXELS_PER_FRAME].copy_from_slice(&frames[..PIXELS_PER_FRAME]);
    }
    // Original frames
    let mid_off = prefix * PIXELS_PER_FRAME;
    out[mid_off..mid_off + n_frames * PIXELS_PER_FRAME]
        .copy_from_slice(&frames[..n_frames * PIXELS_PER_FRAME]);
    // Suffix: copies of frame[N-1]
    let last_off = (n_frames - 1) * PIXELS_PER_FRAME;
    let last_frame = &frames[last_off..last_off + PIXELS_PER_FRAME];
    for i in 0..suffix {
        let dst_off = (prefix + n_frames + i) * PIXELS_PER_FRAME;
        out[dst_off..dst_off + PIXELS_PER_FRAME].copy_from_slice(last_frame);
    }
    out
}

/// Apply sigmoid to one-hot logits, threshold, and emit cut frame indices
/// (with their probabilities).
pub fn cuts_from_one_hot(one_hot: &[f32], threshold: f32) -> Vec<(usize, f32)> {
    let mut cuts = Vec::new();
    let mut index = 0;
    while index < one_hot.len() {
        let probability = sigmoid(one_hot[index]);
        if probability > threshold {
            let mut peak_frame = index;
            let mut peak_probability = probability;
            let mut end = index + 1;
            while end < one_hot.len() {
                let next_probability = sigmoid(one_hot[end]);
                if next_probability <= threshold {
                    break;
                }
                if next_probability > peak_probability {
                    peak_probability = next_probability;
                    peak_frame = end;
                }
                end += 1;
            }
            cuts.push((peak_frame, peak_probability));
            index = end;
        } else {
            index += 1;
        }
    }
    cuts
}

pub(super) fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// Build fully attributed scene records from boundary frame/confidence pairs.
/// A boundary belongs to the outgoing scene; its confidence belongs to the
/// scene starting on the following frame.
pub fn scenes_from_boundaries(
    boundaries: &[(usize, f32)],
    total_frames: usize,
    frame_rate: f64,
) -> Vec<Scene> {
    let mut scenes = Vec::with_capacity(boundaries.len() + 1);
    let mut start_frame = 0;
    let mut start_confidence = None;
    for &(boundary_frame, confidence) in boundaries {
        let frame_count = boundary_frame + 1 - start_frame;
        scenes.push(Scene {
            index: scenes.len(),
            start_frame,
            frame_count,
            start_time: start_frame as f64 / frame_rate,
            duration: frame_count as f64 / frame_rate,
            start_confidence,
            embedding: None,
        });
        start_frame = boundary_frame + 1;
        start_confidence = Some(confidence);
    }
    if start_frame < total_frames {
        let frame_count = total_frames - start_frame;
        scenes.push(Scene {
            index: scenes.len(),
            start_frame,
            frame_count,
            start_time: start_frame as f64 / frame_rate,
            duration: frame_count as f64 / frame_rate,
            start_confidence,
            embedding: None,
        });
    }
    scenes
}

/// Build scenes from sorted cut frame indices.
///
/// Convention: a cut frame is the LAST frame of the OUTGOING scene; the
/// next scene begins at `cut+1`. Adjacent scenes touch with no gap; the
/// frame counts always sum to `total_frames`.
pub fn scenes_from_cuts(cuts: &[usize], total_frames: usize) -> Vec<(usize, usize)> {
    let mut scenes = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0usize;
    for &cut in cuts {
        let count = cut + 1 - start;
        scenes.push((start, count));
        start = cut + 1;
    }
    if start < total_frames {
        scenes.push((start, total_frames - start));
    }
    scenes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_zero_frames_yields_zero_buffer_of_borders() {
        let out = pad_for_batching(&[], 0);
        // 25 prefix + 0 + 25 suffix = 50 frames worth of zeros
        assert_eq!(out.len(), 50 * PIXELS_PER_FRAME);
    }

    #[test]
    fn pad_50_frames_yields_100_total() {
        let frames = vec![1u8; 50 * PIXELS_PER_FRAME];
        let out = pad_for_batching(&frames, 50);
        // 25 + 50 + 25 = 100 frames
        assert_eq!(out.len(), 100 * PIXELS_PER_FRAME);
    }

    #[test]
    fn pad_aligns_to_50_stride() {
        // 60 frames → reminder = 50 - 10 = 40 → prefix 25 + 60 + 40 + 25 suffix = 150
        let frames = vec![0u8; 60 * PIXELS_PER_FRAME];
        let out = pad_for_batching(&frames, 60);
        assert_eq!(out.len(), 150 * PIXELS_PER_FRAME);
    }

    #[test]
    fn scenes_no_cuts_yields_one_scene() {
        let s = scenes_from_cuts(&[], 100);
        assert_eq!(s, vec![(0, 100)]);
    }

    #[test]
    fn scenes_single_cut_splits_two() {
        // Cut at frame 49 → scenes [0..49] (50 frames) + [50..99] (50 frames)
        let s = scenes_from_cuts(&[49], 100);
        assert_eq!(s, vec![(0, 50), (50, 50)]);
    }

    #[test]
    fn scenes_multiple_cuts_match_fixture() {
        // Mirrors test_metadata.json: cuts at 29, 69 → 3 scenes [0..29], [30..69], [70..99]
        let s = scenes_from_cuts(&[29, 69], 100);
        assert_eq!(s, vec![(0, 30), (30, 40), (70, 30)]);
    }

    fn logit(probability: f32) -> f32 {
        let probability = probability.clamp(1e-6, 1.0 - 1e-6);
        (probability / (1.0 - probability)).ln()
    }

    fn logits(frame_count: usize, peaks: &[(usize, f32)]) -> Vec<f32> {
        let mut logits = vec![-10.0; frame_count];
        for &(frame, probability) in peaks {
            logits[frame] = logit(probability);
        }
        logits
    }

    // VideoSceneDetectorProcessingTests.swift

    #[test]
    fn scenes_from_boundaries_builds_contiguous_non_overlapping_scenes() {
        let scenes = scenes_from_boundaries(&[(29, 0.9), (69, 0.7)], 100, 30.0);
        assert_eq!(scenes.len(), 3);
        assert_eq!((scenes[0].start_frame, scenes[0].frame_count), (0, 30));
        assert_eq!((scenes[1].start_frame, scenes[1].frame_count), (30, 40));
        assert_eq!((scenes[2].start_frame, scenes[2].frame_count), (70, 30));
        assert_eq!(scenes[0].start_confidence, None);
        assert!((scenes[1].start_confidence.unwrap() - 0.9).abs() < 1e-3);
        assert!((scenes[2].start_confidence.unwrap() - 0.7).abs() < 1e-3);
        assert_eq!(
            scenes.iter().map(|scene| scene.index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!((scenes[1].start_time - 1.0).abs() < 1e-6);
        assert_eq!(
            scenes.iter().map(|scene| scene.frame_count).sum::<usize>(),
            100
        );
        assert!(
            scenes
                .windows(2)
                .all(|pair| { pair[0].start_frame + pair[0].frame_count == pair[1].start_frame })
        );
    }

    #[test]
    fn scenes_from_boundaries_no_boundaries_yields_single_scene_covering_whole_video() {
        let scenes = scenes_from_boundaries(&[], 100, 30.0);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].index, 0);
        assert_eq!(scenes[0].start_frame, 0);
        assert_eq!(scenes[0].frame_count, 100);
        assert_eq!(scenes[0].start_confidence, None);
    }

    #[test]
    fn scenes_from_boundaries_boundary_on_last_frame_omits_trailing_scene() {
        let scenes = scenes_from_boundaries(&[(99, 0.9)], 100, 30.0);
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].frame_count, 100);
    }

    #[test]
    fn boundaries_from_logits_empty_on_all_sub_threshold() {
        assert!(cuts_from_one_hot(&[-5.0; 100], 0.296).is_empty());
    }

    #[test]
    fn boundaries_from_logits_isolated_peak_produces_one_boundary() {
        let boundaries = cuts_from_one_hot(&logits(100, &[(50, 0.9)]), 0.296);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].0, 50);
        assert!((boundaries[0].1 - 0.9).abs() < 1e-3);
    }

    #[test]
    fn boundaries_from_logits_run_collapses_to_peak_frame() {
        let boundaries =
            cuts_from_one_hot(&logits(100, &[(49, 0.40), (50, 0.90), (51, 0.55)]), 0.296);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].0, 50);
        assert!((boundaries[0].1 - 0.90).abs() < 1e-3);
    }

    #[test]
    fn boundaries_from_logits_separate_runs_become_separate_boundaries() {
        let boundaries = cuts_from_one_hot(&logits(100, &[(10, 0.7), (11, 0.5), (30, 0.8)]), 0.296);
        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0].0, 10);
        assert_eq!(boundaries[1].0, 30);
    }

    #[test]
    fn boundaries_from_logits_single_frame_run() {
        let boundaries = cuts_from_one_hot(&logits(100, &[(50, 0.4)]), 0.296);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].0, 50);
    }

    #[test]
    fn boundaries_from_logits_run_at_end_of_array() {
        let boundaries = cuts_from_one_hot(&logits(100, &[(98, 0.5), (99, 0.7)]), 0.296);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].0, 99);
    }

    #[test]
    fn boundaries_from_logits_threshold_equality_is_exclusive() {
        let mut logits = vec![-10.0; 10];
        logits[5] = 0.0;
        assert!(cuts_from_one_hot(&logits, 0.5).is_empty());
    }

    // VideoSceneDetectorPipelineTests.swift

    #[test]
    fn pad_for_batching_matches_python_get_batches() {
        let mut frames = vec![0_u8; 100 * PIXELS_PER_FRAME];
        for frame in 0..100 {
            frames[frame * PIXELS_PER_FRAME] = frame as u8;
        }
        let padded = pad_for_batching(&frames, 100);
        assert_eq!(padded.len() / PIXELS_PER_FRAME, 150);
        for frame in 0..25 {
            assert_eq!(padded[frame * PIXELS_PER_FRAME], 0, "prefix slot {frame}");
        }
        assert_eq!(padded[25 * PIXELS_PER_FRAME], 0);
        assert_eq!(padded[(25 + 50) * PIXELS_PER_FRAME], 50);
        for frame in 125..150 {
            assert_eq!(padded[frame * PIXELS_PER_FRAME], 99, "suffix slot {frame}");
        }
    }

    #[test]
    fn pad_for_batching_uneven_frame_count() {
        let mut frames = vec![0_u8; 37 * PIXELS_PER_FRAME];
        for frame in 0..37 {
            frames[frame * PIXELS_PER_FRAME] = frame as u8;
        }
        assert_eq!(pad_for_batching(&frames, 37).len() / PIXELS_PER_FRAME, 100);
    }
}
