//! The scene-identity tap on the detector's ONE decode pass. A scene is by
//! definition one shot — visually uniform — so it needs a HANDFUL of DINOv2
//! samples, not a cadence: the producer embeds a frame only when the luma plane
//! (already computed for the jump-cut and transient scanners) says the picture
//! CHANGED since the last embedded frame, plus a slow drift refresh. A static
//! minute costs ~6 samples; a real cut costs one, immediately; a strobe may
//! cost an extra that the per-scene MEDOID discards later.
//!
//! Boundaries are not known here and are not needed: samples carry frame
//! stamps, and the detector assigns them to the FINAL scenes (after the
//! jump-cut merge and the flash veto).
//!
//! Port of iOS `SceneEmbeddingTap.swift`.

use super::photometric::mean_absolute_difference;
use super::processor::Scene;
use super::shot_identity::ShotIdentifying;
use super::transient::{DIFF_FLOOR, IMPULSE_FACTOR};
use crate::dinov2::FrameSnapshot;
use crate::slow_motion_renderer::LumaPlane;

/// Refresh an unchanged shot this often (slow pans drift under the gate).
const REFRESH_SECONDS: f64 = 10.0;
/// Never embed more often than this (a strobe run fires the gate every frame;
/// one sample of it is plenty for the medoid to discard).
const MIN_GAP_SECONDS: f64 = 0.3;
/// Weight of the newest pair in the running baseline.
const BASELINE_WEIGHT: f64 = 0.1;

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub frame: usize,
    pub vector: Vec<f32>,
}

pub struct SceneEmbeddingTap<I: ShotIdentifying> {
    identity: I,
    refresh_frames: usize,
    min_gap_frames: usize,
    last_embedded: Option<LumaPlane>,
    previous: Option<LumaPlane>,
    baseline: Option<f64>,
    last_embed_frame: Option<usize>,
    frame: usize,
    collected: Vec<Sample>,
}

impl<I: ShotIdentifying> SceneEmbeddingTap<I> {
    #[must_use]
    pub fn new(identity: I, frame_rate: f64) -> Self {
        let rate = frame_rate.max(1.0);
        Self {
            identity,
            refresh_frames: (REFRESH_SECONDS * rate) as usize,
            min_gap_frames: ((MIN_GAP_SECONDS * rate) as usize).max(1),
            last_embedded: None,
            previous: None,
            baseline: None,
            last_embed_frame: None,
            frame: 0,
            collected: Vec::new(),
        }
    }

    pub async fn push(&mut self, luma: &LumaPlane, snapshot: &FrameSnapshot) {
        if let Some(previous) = &self.previous {
            let pair_diff = mean_absolute_difference(previous, luma);
            self.baseline = Some(match self.baseline {
                Some(baseline) => baseline * (1.0 - BASELINE_WEIGHT) + pair_diff * BASELINE_WEIGHT,
                None => pair_diff,
            });
        }
        self.previous = Some(luma.clone());
        let frame = self.frame;
        self.frame += 1;

        // The first frame always embeds; afterwards the gate and the minimum
        // gap decide.
        let since_embed = self.last_embed_frame.map(|last| frame - last);
        if since_embed.is_some_and(|gap| gap < self.min_gap_frames) {
            return;
        }
        let changed = match &self.last_embedded {
            Some(last) => {
                mean_absolute_difference(last, luma)
                    >= DIFF_FLOOR.max(IMPULSE_FACTOR * self.baseline.unwrap_or(0.0))
            }
            None => true,
        };
        if !changed && since_embed.is_none_or(|gap| gap < self.refresh_frames) {
            return;
        }

        let vector = self.identity.embedding(snapshot).await;
        if vector.is_empty() {
            return;
        }
        self.collected.push(Sample { frame, vector });
        self.last_embedded = Some(luma.clone());
        self.last_embed_frame = Some(frame);
    }

    #[must_use]
    pub fn samples(&self) -> &[Sample] {
        &self.collected
    }

    #[must_use]
    pub fn into_samples(self) -> Vec<Sample> {
        self.collected
    }
}

/// One vector per FINAL scene: the medoid of the samples inside it; a scene the
/// gate never sampled (a sliver between two cuts) borrows the nearest sample —
/// same shot by construction of the gate.
///
/// `None` when the pass produced no samples at all. That is the all-or-nothing
/// rule: either every scene carries an embedding or none does, so a partially
/// filled column can never reach the server.
#[must_use]
pub fn scene_embeddings(samples: &[Sample], scenes: &[Scene]) -> Option<Vec<Vec<f32>>> {
    if samples.is_empty() || scenes.is_empty() {
        return None;
    }
    let mut assigned: Vec<Vec<&[f32]>> = vec![Vec::new(); scenes.len()];
    for sample in samples {
        let index = scenes
            .iter()
            .rposition(|scene| scene.start_frame <= sample.frame)
            .unwrap_or(0);
        assigned[index].push(&sample.vector);
    }
    Some(
        scenes
            .iter()
            .zip(&assigned)
            .map(|(scene, candidates)| {
                medoid(candidates).unwrap_or_else(|| {
                    let low = scene.start_frame;
                    let high = low + scene.frame_count;
                    nearest(samples, low, high)
                })
            })
            .collect(),
    )
}

/// The member with the smallest summed distance to the rest — never a synthetic
/// mean, so a single poisoned sample (a strobe frame) loses instead of pulling.
fn medoid(candidates: &[&[f32]]) -> Option<Vec<f32>> {
    if candidates.len() <= 1 {
        return candidates.first().map(|vector| vector.to_vec());
    }
    // A candidate's distance to itself is 0, so it needs no special case.
    let mut best = 0;
    let mut best_sum = f32::INFINITY;
    for (index, candidate) in candidates.iter().enumerate() {
        let sum: f32 = candidates
            .iter()
            .map(|other| super::transient_verdict::distance(candidate, other))
            .sum();
        if sum < best_sum {
            best_sum = sum;
            best = index;
        }
    }
    Some(candidates[best].to_vec())
}

/// The sample whose frame sits closest to `low..high`; ties go to the earlier
/// sample.
fn nearest(samples: &[Sample], low: usize, high: usize) -> Vec<f32> {
    samples
        .iter()
        .min_by_key(|sample| distance_to_span(sample.frame, low, high))
        .map(|sample| sample.vector.clone())
        .unwrap_or_default()
}

fn distance_to_span(frame: usize, low: usize, high: usize) -> usize {
    if frame < low {
        low - frame
    } else if frame >= high {
        frame - high + 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_scene_detector::scenes::scenes_from_boundaries;
    use crate::video_scene_detector::synthetic::{flashed, static_sequence, texture};

    const W: usize = 256;
    const H: usize = 144;

    /// An identity that reads the snapshot's first byte as a shot label, so a
    /// sample's provenance is visible in its vector.
    struct LabelIdentity;

    #[async_trait::async_trait(?Send)]
    impl ShotIdentifying for LabelIdentity {
        async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32> {
            vec![f32::from(frame.rgb[0])]
        }
        fn same_shot_distance(&self) -> f32 {
            0.5
        }
        fn similar_shot_distance(&self) -> f32 {
            0.7
        }
    }

    struct NoIdentity;

    #[async_trait::async_trait(?Send)]
    impl ShotIdentifying for NoIdentity {
        async fn embedding(&self, _frame: &FrameSnapshot) -> Vec<f32> {
            Vec::new()
        }
        fn same_shot_distance(&self) -> f32 {
            0.5
        }
        fn similar_shot_distance(&self) -> f32 {
            0.7
        }
    }

    fn label(value: u8) -> FrameSnapshot {
        FrameSnapshot::packed(1, 1, vec![value, 0, 0])
    }

    fn run_tap(planes: &[LumaPlane], labels: impl Fn(usize) -> u8) -> Vec<Sample> {
        let mut tap = SceneEmbeddingTap::new(LabelIdentity, 30.0);
        for (index, plane) in planes.iter().enumerate() {
            futures_executor::block_on(tap.push(plane, &label(labels(index))));
        }
        tap.into_samples()
    }

    #[test]
    fn a_static_shot_costs_a_handful_of_samples() {
        // 10 s at 30 fps of one still shot: the first frame, then the drift
        // refresh only.
        let samples = run_tap(&static_sequence(W, H, 7, 300), |_| 0);
        assert!(
            (1..=4).contains(&samples.len()),
            "a still minute must be cheap: {} samples",
            samples.len()
        );
        assert_eq!(samples[0].frame, 0, "the first frame always embeds");
    }

    #[test]
    fn a_cut_costs_one_sample_immediately() {
        let mut planes = static_sequence(W, H, 7, 40);
        planes.extend(static_sequence(W, H, 99, 40));
        let samples = run_tap(&planes, |index| u8::from(index >= 40));
        let after_cut = samples
            .iter()
            .find(|sample| sample.frame >= 40)
            .expect("the new shot must be sampled");
        assert_eq!(after_cut.frame, 40, "{samples:?}");
    }

    #[test]
    fn a_strobe_cannot_fire_the_gate_every_frame() {
        let mut planes = static_sequence(W, H, 7, 60);
        for index in (20..40).step_by(2) {
            planes[index] = flashed(&planes[index], 120);
        }
        let samples = run_tap(&planes, |_| 0);
        let strobe: Vec<_> = samples
            .iter()
            .filter(|sample| (20..40).contains(&sample.frame))
            .collect();
        // min_gap_frames = 0.3 * 30 = 9, so at most three samples fit.
        assert!(strobe.len() <= 3, "{strobe:?}");
    }

    #[test]
    fn an_identity_that_cannot_answer_collects_nothing() {
        let mut tap = SceneEmbeddingTap::new(NoIdentity, 30.0);
        for plane in &static_sequence(W, H, 7, 40) {
            futures_executor::block_on(tap.push(plane, &label(0)));
        }
        assert_eq!(tap.into_samples(), []);
    }

    // MARK: - Assignment

    fn sample(frame: usize, value: f32) -> Sample {
        Sample {
            frame,
            vector: vec![value],
        }
    }

    #[test]
    fn no_samples_means_no_embeddings_anywhere() {
        let scenes = scenes_from_boundaries(&[(29, 0.9)], 100, 30.0);
        assert_eq!(scene_embeddings(&[], &scenes), None);
    }

    #[test]
    fn every_scene_gets_the_medoid_of_its_own_samples() {
        let scenes = scenes_from_boundaries(&[(29, 0.9), (69, 0.8)], 100, 30.0);
        let samples = vec![
            sample(0, 0.0),
            sample(10, 0.1),
            sample(20, 9.0), // the poisoned strobe frame
            sample(30, 5.0),
            sample(40, 5.1),
            sample(70, 2.0),
        ];
        let embeddings = scene_embeddings(&samples, &scenes).expect("samples exist");
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0], vec![0.1], "the medoid discards the outlier");
        assert_eq!(embeddings[1], vec![5.0]);
        assert_eq!(embeddings[2], vec![2.0]);
    }

    #[test]
    fn a_scene_the_gate_never_sampled_borrows_the_nearest_sample() {
        // Boundaries at 29 and 33 leave scene 1 (frames 30..33) unsampled.
        let scenes = scenes_from_boundaries(&[(29, 0.9), (33, 0.8)], 100, 30.0);
        let samples = vec![sample(0, 0.0), sample(40, 7.0)];
        let embeddings = scene_embeddings(&samples, &scenes).expect("samples exist");
        assert_eq!(embeddings.len(), 3);
        assert_eq!(
            embeddings[1],
            vec![7.0],
            "frame 40 is 7 away, frame 0 is 30"
        );
    }

    #[test]
    fn samples_before_the_first_scene_start_land_in_the_first_scene() {
        let scenes = scenes_from_boundaries(&[(29, 0.9)], 100, 30.0);
        let embeddings =
            scene_embeddings(&[sample(0, 4.0), sample(50, 6.0)], &scenes).expect("samples exist");
        assert_eq!(embeddings[0], vec![4.0]);
        assert_eq!(embeddings[1], vec![6.0]);
    }

    #[test]
    fn the_medoid_of_one_candidate_is_that_candidate() {
        assert_eq!(medoid(&[&[1.0, 2.0][..]]), Some(vec![1.0, 2.0]));
        assert_eq!(medoid(&[]), None);
    }

    #[test]
    fn the_medoid_is_a_member_not_a_mean() {
        let candidates: Vec<&[f32]> = vec![&[0.0], &[1.0], &[10.0]];
        assert_eq!(medoid(&candidates), Some(vec![1.0]));
    }

    #[test]
    fn distance_to_span_measures_frames_outside_the_scene() {
        assert_eq!(distance_to_span(5, 10, 20), 5);
        assert_eq!(distance_to_span(15, 10, 20), 0);
        assert_eq!(distance_to_span(20, 10, 20), 1);
        assert_eq!(distance_to_span(25, 10, 20), 6);
    }

    /// A frame rate under 1 fps would otherwise make the minimum gap zero and
    /// the tap would embed on every frame.
    #[test]
    fn a_degenerate_frame_rate_still_bounds_the_sample_gap() {
        let tap = SceneEmbeddingTap::new(LabelIdentity, 0.0);
        assert_eq!(tap.min_gap_frames, 1);
        assert_eq!(tap.refresh_frames, 10);
        let _ = texture(8, 8, 1);
    }
    // Exact current SceneEmbeddingTapTests inputs, kept in addition to the
    // broader textured-shot, strobe and nearest-span regressions above.
    #[test]
    fn current_gate_embeds_on_change_not_on_cadence() {
        let flat = |value| LumaPlane {
            width: 8,
            height: 8,
            bytes_per_row: 8,
            data: vec![value; 64],
        };
        let mut tap = SceneEmbeddingTap::new(LabelIdentity, 25.0);
        for _ in 0..250 {
            futures_executor::block_on(tap.push(&flat(40), &label(40)));
        }
        let static_calls = tap.samples().len();
        assert!(static_calls <= 3);
        futures_executor::block_on(tap.push(&flat(220), &label(220)));
        assert_eq!(tap.samples().len(), static_calls + 1);
        assert_eq!(tap.samples().last().unwrap().frame, 250);
        for _ in 0..100 {
            futures_executor::block_on(tap.push(&flat(220), &label(220)));
        }
        assert_eq!(tap.samples().len(), static_calls + 1);
    }

    #[test]
    fn current_medoid_degenerate_and_duplicate_majority() {
        assert_eq!(medoid(&[]), None);
        assert_eq!(medoid(&[&[0.5]]), Some(vec![0.5]));
        assert_eq!(
            medoid(&[&[1.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]]),
            Some(vec![1.0, 0.0])
        );
    }

    #[test]
    fn current_medoid_discards_the_flash_in_every_source_order() {
        let normal: &[f32] = &[1.0, 0.0, 0.0];
        let similar: &[f32] = &[0.98, 0.02, 0.0];
        let flash: &[f32] = &[0.0, 0.0, 1.0];
        for order in [
            [flash, normal, similar],
            [normal, flash, similar],
            [similar, normal, flash],
        ] {
            let picked = medoid(&order).unwrap();
            assert_ne!(picked, flash);
            assert!(picked == normal || picked == similar);
        }
    }

    #[test]
    fn current_scene_assignment_and_four_frame_sliver_fallback() {
        let scenes = scenes_from_boundaries(&[(99, 0.9), (103, 0.9)], 204, 25.0);
        let samples = vec![
            Sample {
                frame: 10,
                vector: vec![1.0, 0.0],
            },
            Sample {
                frame: 60,
                vector: vec![0.9, 0.1],
            },
            Sample {
                frame: 150,
                vector: vec![0.0, 1.0],
            },
        ];
        let vectors = scene_embeddings(&samples, &scenes).unwrap();
        assert_eq!(vectors.len(), 3);
        assert!(vectors[0] == [1.0, 0.0] || vectors[0] == [0.9, 0.1]);
        assert_eq!(vectors[1], [0.9, 0.1]); // frame 60 is 40 away; frame 150 is 47.
        assert_eq!(vectors[2], [0.0, 1.0]);
        assert_eq!(scene_embeddings(&[], &scenes), None);
    }
}
