//! Per-scene cell-count + frame-index allocation. Pure logic — no I/O.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub start_frame: usize,
    pub frame_count: usize,
    pub start_time: f64,
    pub duration: f64,
}

#[derive(Debug, Clone)]
pub struct SceneAllocation {
    pub scene_index: usize,
    pub scene: Scene,
    pub cell_frames: Vec<usize>,
    pub cell_times: Vec<f64>,
}

/// Cell count by scene duration. Mirrors iOS `cellCount(forDuration:)`.
pub fn cell_count(duration: f64) -> usize {
    if duration < 1.0 {
        1
    } else if duration < 3.0 {
        2
    } else if duration < 8.0 {
        3
    } else if duration < 20.0 {
        4
    } else {
        5
    }
}

/// Linearly distribute `n` frame indices across the scene from
/// `start_frame` to `last_frame`. Mirrors iOS `pickFrameIndices`.
pub fn pick_frame_indices(scene: &Scene, n: usize) -> Vec<usize> {
    if n == 0 || scene.frame_count == 0 {
        return Vec::new();
    }
    let first = scene.start_frame;
    let last = scene.start_frame + scene.frame_count - 1;
    if n == 1 || scene.frame_count == 1 {
        return vec![first];
    }
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            first + ((last - first) as f64 * t).round() as usize
        })
        .collect()
}

pub fn allocate(scenes: &[Scene]) -> Vec<SceneAllocation> {
    allocate_indexed(scenes, None)
}

/// Allocate a sampled scene list while retaining indices into the original
/// full scene list for the JSON/image sidecar.
pub fn allocate_indexed(
    scenes: &[Scene],
    original_indices: Option<&[usize]>,
) -> Vec<SceneAllocation> {
    assert!(
        original_indices.is_none_or(|indices| indices.len() == scenes.len()),
        "original_indices must match scenes count"
    );
    scenes
        .iter()
        .enumerate()
        .map(|(i, sc)| {
            let n = cell_count(sc.duration);
            let frames = pick_frame_indices(sc, n);
            // Derive time from the inclusive frame index. Interpolating over
            // the full scene duration put the final cell exactly on the next
            // scene's first frame.
            let frame_duration = if sc.frame_count == 0 {
                0.0
            } else {
                sc.duration / sc.frame_count as f64
            };
            let times = frames
                .iter()
                .map(|frame| {
                    sc.start_time + frame.saturating_sub(sc.start_frame) as f64 * frame_duration
                })
                .collect::<Vec<_>>();
            SceneAllocation {
                scene_index: original_indices.map_or(i, |indices| indices[i]),
                scene: *sc,
                cell_frames: frames,
                cell_times: times,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(start: usize, count: usize, fps: f64) -> Scene {
        Scene {
            start_frame: start,
            frame_count: count,
            start_time: start as f64 / fps,
            duration: count as f64 / fps,
        }
    }

    #[test]
    fn cell_count_table() {
        assert_eq!(cell_count(0.5), 1);
        assert_eq!(cell_count(1.5), 2);
        assert_eq!(cell_count(5.0), 3);
        assert_eq!(cell_count(12.0), 4);
        assert_eq!(cell_count(30.0), 5);
    }

    #[test]
    fn first_and_last_frame_inclusive() {
        assert_eq!(pick_frame_indices(&scene(100, 60, 30.0), 2), [100, 159]);
    }

    #[test]
    fn three_cells_hits_midpoint() {
        let frames = pick_frame_indices(&scene(0, 100, 30.0), 3);
        assert_eq!(frames.first(), Some(&0));
        assert_eq!(frames.last(), Some(&99));
        assert_eq!(frames[1], 50);
    }

    #[test]
    fn five_cells_evenly_spaced() {
        let frames = pick_frame_indices(&scene(0, 1_000, 30.0), 5);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames.first(), Some(&0));
        assert_eq!(frames.last(), Some(&999));
        for difference in frames.windows(2).map(|pair| pair[1] - pair[0]) {
            assert!(
                difference.abs_diff(250) <= 1,
                "frames not equispaced: {frames:?}"
            );
        }
    }

    #[test]
    fn single_frame_scene_collapses_to_one_cell() {
        assert_eq!(pick_frame_indices(&scene(5, 1, 30.0), 5), [5]);
    }

    #[test]
    fn allocate_attaches_times_via_scene_times() {
        let allocations = allocate(&[scene(60, 60, 30.0)]);
        assert_eq!(allocations.len(), 1);
        assert_eq!(allocations[0].cell_times.len(), 2);
        assert!((allocations[0].cell_times[0] - 60.0 / 30.0).abs() <= 0.0001);
        assert!((allocations[0].cell_times[1] - 119.0 / 30.0).abs() <= 0.0001);
    }

    #[test]
    fn last_cell_time_maps_to_inclusive_last_frame_not_next_scene() {
        let source = scene(100, 60, 30.0);
        let allocation = allocate(&[source]).remove(0);
        let last_frame_time = *allocation.cell_frames.last().unwrap() as f64 / 30.0;
        let next_scene_start = source.start_time + source.duration;
        assert!((allocation.cell_times.last().unwrap() - last_frame_time).abs() <= 0.0001);
        assert!(*allocation.cell_times.last().unwrap() < next_scene_start);
    }

    #[test]
    fn all_cell_times_match_their_cell_frames() {
        let allocation = allocate(&[scene(0, 1_000, 25.0)]).remove(0);
        assert_eq!(allocation.cell_times.len(), allocation.cell_frames.len());
        for (frame, time) in allocation.cell_frames.iter().zip(&allocation.cell_times) {
            assert!((*time - *frame as f64 / 25.0).abs() <= 0.0001);
        }
    }

    #[test]
    fn allocate_on_hairdryer_like_scenes() {
        let durations = [
            2.07, 1.10, 1.47, 1.27, 1.03, 0.87, 1.87, 3.87, 2.60, 0.83, 0.73, 1.40, 0.60,
        ];
        let mut time = 0.0;
        let scenes = durations
            .into_iter()
            .map(|duration| {
                let value = Scene {
                    start_frame: (time * 30.0) as usize,
                    frame_count: ((duration * 30.0) as usize).max(1),
                    start_time: time,
                    duration,
                };
                time += duration;
                value
            })
            .collect::<Vec<_>>();
        let allocations = allocate(&scenes);
        assert_eq!(allocations.len(), 13);
        let counts = allocations
            .iter()
            .map(|allocation| allocation.cell_frames.len())
            .collect::<Vec<_>>();
        assert_eq!(counts, [2, 2, 2, 2, 2, 1, 2, 3, 2, 1, 1, 2, 1]);
        assert_eq!(counts.into_iter().max(), Some(3));
    }
}
