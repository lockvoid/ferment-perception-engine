//! Pure scene-row sampling for videos that exceed the minimap row budget.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::Scene;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    AllScenes,
    UniformByTime,
    LongestFirst,
    StratifiedLongestPerBin,
}

impl SamplingStrategy {
    pub const ALL: [Self; 4] = [
        Self::AllScenes,
        Self::UniformByTime,
        Self::LongestFirst,
        Self::StratifiedLongestPerBin,
    ];
}

/// Return original scene indices in temporal order.
#[must_use]
pub fn sample_indices(
    scenes: &[Scene],
    max_count: usize,
    strategy: SamplingStrategy,
) -> Vec<usize> {
    if scenes.is_empty() {
        return Vec::new();
    }
    if strategy != SamplingStrategy::AllScenes && max_count == 0 {
        return Vec::new();
    }
    if strategy != SamplingStrategy::AllScenes && scenes.len() <= max_count {
        return (0..scenes.len()).collect();
    }

    match strategy {
        SamplingStrategy::AllScenes => (0..scenes.len()).collect(),
        SamplingStrategy::UniformByTime => {
            let stride = scenes.len() as f64 / max_count as f64;
            let mut picks = Vec::with_capacity(max_count);
            for index in 0..max_count {
                let scene_index = (index as f64 * stride).floor() as usize;
                if picks.last() != Some(&scene_index) {
                    picks.push(scene_index);
                }
            }
            picks
        }
        SamplingStrategy::LongestFirst => {
            let mut indices = (0..scenes.len()).collect::<Vec<_>>();
            indices.sort_by(|left, right| {
                scenes[*right]
                    .duration
                    .partial_cmp(&scenes[*left].duration)
                    .unwrap_or(Ordering::Equal)
            });
            indices.truncate(max_count);
            indices.sort_unstable();
            indices
        }
        SamplingStrategy::StratifiedLongestPerBin => stratified_longest_per_bin(scenes, max_count),
    }
}

fn stratified_longest_per_bin(scenes: &[Scene], max_count: usize) -> Vec<usize> {
    let Some(last) = scenes.last() else {
        return Vec::new();
    };
    let base_time = scenes[0].start_time;
    let total_duration = last.start_time + last.duration - base_time;
    if total_duration <= 0.0 {
        return (0..scenes.len().min(max_count)).collect();
    }

    let bin_width = total_duration / max_count as f64;
    let mut best_per_bin: Vec<Option<usize>> = vec![None; max_count];
    for (index, scene) in scenes.iter().enumerate() {
        let midpoint = scene.start_time + scene.duration / 2.0 - base_time;
        let bin = ((midpoint / bin_width) as isize).clamp(0, max_count as isize - 1) as usize;
        match best_per_bin[bin] {
            Some(existing) if scene.duration <= scenes[existing].duration => {}
            _ => best_per_bin[bin] = Some(index),
        }
    }
    let mut indices = best_per_bin.into_iter().flatten().collect::<Vec<_>>();
    indices.sort_unstable();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenes(durations: &[f64]) -> Vec<Scene> {
        let mut time = 0.0;
        durations
            .iter()
            .map(|duration| {
                let scene = Scene {
                    start_frame: (time * 30.0) as usize,
                    frame_count: ((*duration * 30.0) as usize).max(1),
                    start_time: time,
                    duration: *duration,
                };
                time += duration;
                scene
            })
            .collect()
    }

    #[test]
    fn all_scenes_under_cap_returns_everything() {
        assert_eq!(
            sample_indices(&scenes(&[1.0; 5]), 10, SamplingStrategy::AllScenes),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn all_scenes_at_cap_returns_everything() {
        assert_eq!(
            sample_indices(&scenes(&[1.0; 10]), 10, SamplingStrategy::AllScenes).len(),
            10
        );
    }

    #[test]
    fn all_scenes_over_cap_still_returns_all() {
        assert_eq!(
            sample_indices(&scenes(&[1.0; 30]), 10, SamplingStrategy::AllScenes).len(),
            30
        );
    }

    #[test]
    fn uniform_below_cap_returns_all() {
        assert_eq!(
            sample_indices(&scenes(&[1.0; 5]), 10, SamplingStrategy::UniformByTime),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn uniform_at_cap_returns_all() {
        assert_eq!(
            sample_indices(&scenes(&[1.0; 10]), 10, SamplingStrategy::UniformByTime),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn uniform_over_cap_picks_even_stride() {
        let indices = sample_indices(&scenes(&[1.0; 100]), 10, SamplingStrategy::UniformByTime);
        assert_eq!(indices.len(), 10);
        assert_eq!(indices.first(), Some(&0));
        assert!(indices.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(indices.last().is_some_and(|index| *index > 80));
    }

    #[test]
    fn uniform_preserves_temporal_order() {
        let indices = sample_indices(&scenes(&[0.5; 50]), 7, SamplingStrategy::UniformByTime);
        assert!(indices.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn longest_first_picks_top_n_by_duration() {
        let indices = sample_indices(
            &scenes(&[1.0, 5.0, 3.0, 2.0, 4.0, 1.0, 6.0, 2.0, 3.0, 4.0]),
            5,
            SamplingStrategy::LongestFirst,
        );
        assert_eq!(indices, [1, 2, 4, 6, 9]);
    }

    #[test]
    fn longest_first_returns_temporally_ordered() {
        let indices = sample_indices(
            &scenes(&[1.0, 5.0, 3.0, 2.0, 4.0, 1.0, 6.0, 2.0, 3.0, 4.0]),
            5,
            SamplingStrategy::LongestFirst,
        );
        assert!(indices.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn longest_first_under_cap_returns_all() {
        assert_eq!(
            sample_indices(
                &scenes(&[1.0, 2.0, 3.0]),
                10,
                SamplingStrategy::LongestFirst
            ),
            [0, 1, 2]
        );
    }

    #[test]
    fn stratified_divides_into_time_bins_and_picks_longest() {
        assert_eq!(
            sample_indices(
                &scenes(&[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]),
                2,
                SamplingStrategy::StratifiedLongestPerBin,
            ),
            [1, 6]
        );
    }

    #[test]
    fn stratified_returns_temporally_ordered() {
        let indices = sample_indices(
            &scenes(&[1.0; 20]),
            5,
            SamplingStrategy::StratifiedLongestPerBin,
        );
        assert!(indices.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn stratified_at_cap_returns_all() {
        assert_eq!(
            sample_indices(
                &scenes(&[1.0; 5]),
                5,
                SamplingStrategy::StratifiedLongestPerBin
            ),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn stratified_skips_empty_bins() {
        let indices = sample_indices(
            &scenes(&[1.0; 10]),
            5,
            SamplingStrategy::StratifiedLongestPerBin,
        );
        assert!(!indices.is_empty());
        assert!(indices.len() <= 5);
    }

    #[test]
    fn stratified_max_count_one_picks_longest_overall() {
        assert_eq!(
            sample_indices(
                &scenes(&[1.0, 5.0, 3.0, 2.0, 4.0]),
                1,
                SamplingStrategy::StratifiedLongestPerBin,
            ),
            [1]
        );
    }

    #[test]
    fn empty_scenes_returns_empty() {
        for strategy in SamplingStrategy::ALL {
            assert!(sample_indices(&[], 10, strategy).is_empty());
        }
    }

    #[test]
    fn zero_max_count_returns_empty() {
        let input = scenes(&[1.0, 2.0, 3.0]);
        for strategy in [
            SamplingStrategy::UniformByTime,
            SamplingStrategy::LongestFirst,
            SamplingStrategy::StratifiedLongestPerBin,
        ] {
            assert!(sample_indices(&input, 0, strategy).is_empty());
        }
    }
}
