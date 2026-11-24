//! Pure: scenes -> the flat list of frames to grab, ascending in time.
//! Port of Swift `VideoSaliencySamplePlan`.
//!
//! Samples sit in the scene's interior and the seek slack is capped at the
//! same interior fraction of the scene, so a grab can never land in a
//! neighbouring scene.

use ferment_engine_core::VideoScene;

use super::config::VideoSaliencyConfig;

/// Ceiling on the seek slack, in seconds: past a ten-second scene the interior
/// fraction would hand the generator more slack than a grab is worth.
const MAX_TOLERANCE_SECONDS: f64 = 0.25;

/// A scene shorter than this carries no curve — one midpoint grab is the whole
/// measurement.
const MINIMUM_CURVE_SECONDS: f64 = 0.4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleEntry {
    pub scene_index: usize,
    pub time: f64,
    /// Seek slack the grabber may use around `time`, either direction.
    pub tolerance: f64,
}

/// One grab for a scene too short to carry a curve; otherwise the rate,
/// clamped both ways.
fn sample_count(duration: f64, config: &VideoSaliencyConfig) -> usize {
    if duration < MINIMUM_CURVE_SECONDS {
        return 1;
    }
    let rate = (duration * config.samples_per_second).ceil().max(0.0) as usize;
    rate.clamp(config.min_samples_per_scene, config.max_samples_per_scene)
}

pub fn plan(scenes: &[VideoScene], config: &VideoSaliencyConfig) -> Vec<SampleEntry> {
    let mut entries = Vec::new();
    for scene in scenes {
        let count = sample_count(scene.duration, config);
        let tolerance = MAX_TOLERANCE_SECONDS.min(config.interior * scene.duration);
        if count <= 1 {
            entries.push(SampleEntry {
                scene_index: scene.index,
                time: scene.start_time + scene.duration / 2.0,
                tolerance,
            });
            continue;
        }
        let span = 1.0 - 2.0 * config.interior;
        for i in 0..count {
            let fraction = config.interior + span * i as f64 / (count - 1) as f64;
            entries.push(SampleEntry {
                scene_index: scene.index,
                time: scene.start_time + scene.duration * fraction,
                tolerance,
            });
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    //! Ported from `VideoSaliencySamplePlanTests.swift`. The plan is the whole
    //! safety story for grabs: a sample must never land in a neighbouring
    //! scene, so both the times AND the seek slack are pinned here.

    use super::*;

    fn config() -> VideoSaliencyConfig {
        VideoSaliencyConfig::default()
    }

    fn scene(index: usize, start: f64, duration: f64) -> VideoScene {
        let fps = 30.0;
        VideoScene {
            index,
            start_frame: (start * fps).round() as usize,
            frame_count: (duration * fps).round() as usize,
            start_time: start,
            duration,
            start_confidence: (index > 0).then_some(0.9),
            embedding: None,
        }
    }

    #[test]
    fn scene_shorter_than_four_tenths_gets_one_sample_at_its_midpoint() {
        let entries = plan(&[scene(0, 1.0, 0.3)], &config());
        assert_eq!(entries.len(), 1);
        assert!((entries[0].time - 1.15).abs() < 1e-9);
    }

    /// 4 s x 0.5/s = 2 -> clamped up to the 3 minimum; 10 s -> 5; 100 s -> the
    /// 32 cap.
    #[test]
    fn sample_count_clamps_at_both_ends() {
        assert_eq!(plan(&[scene(0, 0.0, 4.0)], &config()).len(), 3);
        assert_eq!(plan(&[scene(0, 0.0, 10.0)], &config()).len(), 5);
        assert_eq!(plan(&[scene(0, 0.0, 100.0)], &config()).len(), 32);
    }

    #[test]
    fn first_and_last_sample_sit_at_ten_and_ninety_percent_of_the_scene() {
        let entries = plan(&[scene(0, 2.0, 10.0)], &config());
        assert!((entries.first().unwrap().time - 3.0).abs() < 1e-9);
        assert!((entries.last().unwrap().time - 11.0).abs() < 1e-9);
    }

    fn assert_windows_stay_inside(scenes: &[VideoScene], entries: &[SampleEntry]) {
        for entry in entries {
            let scene = &scenes[entry.scene_index];
            assert!(
                entry.time - entry.tolerance >= scene.start_time,
                "sample {} +-{} starts before scene {}",
                entry.time,
                entry.tolerance,
                entry.scene_index
            );
            assert!(
                entry.time + entry.tolerance <= scene.start_time + scene.duration,
                "sample {} +-{} runs past scene {}",
                entry.time,
                entry.tolerance,
                entry.scene_index
            );
        }
    }

    #[test]
    fn every_seek_window_stays_inside_its_own_scene() {
        let scenes = [scene(0, 0.0, 0.3), scene(1, 0.3, 4.0), scene(2, 4.3, 60.0)];
        let entries = plan(&scenes, &config());
        assert!(!entries.is_empty());
        assert_windows_stay_inside(&scenes, &entries);
    }

    #[test]
    fn times_are_ascending_across_the_whole_plan() {
        let scenes = [scene(0, 0.0, 6.0), scene(1, 6.0, 0.2), scene(2, 6.2, 12.0)];
        let times: Vec<f64> = plan(&scenes, &config()).iter().map(|e| e.time).collect();
        assert_eq!(times.len(), 10); // 3 + 1 + 6
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(times, sorted);
        let mut unique = sorted.clone();
        unique.dedup();
        assert_eq!(unique.len(), times.len());
    }

    #[test]
    fn each_entry_carries_its_own_scene_index() {
        let scenes = [
            scene(0, 0.0, 4.0),  // 3 samples
            scene(1, 4.0, 0.2),  // 1 sample
            scene(2, 4.2, 10.0), // 5 samples
        ];
        let entries = plan(&scenes, &config());
        let count = |index: usize| entries.iter().filter(|e| e.scene_index == index).count();
        assert_eq!(count(0), 3);
        assert_eq!(count(1), 1);
        assert_eq!(count(2), 5);
        assert_eq!(entries.len(), 9);
    }

    /// The slack follows the interior, not a constant: with samples at 5 % of
    /// the scene the seek window must shrink to 5 % too, or the first grab of
    /// a scene can land in the previous one.
    #[test]
    fn seek_window_stays_inside_the_scene_for_a_narrower_interior() {
        let narrow = VideoSaliencyConfig {
            interior: 0.05,
            ..VideoSaliencyConfig::default()
        };
        let scenes = [scene(0, 0.0, 2.0), scene(1, 2.0, 2.0)];
        let entries = plan(&scenes, &narrow);
        assert_eq!(entries.len(), 6);
        assert_windows_stay_inside(&scenes, &entries);
    }

    /// interior x duration until the 0.25 s ceiling.
    #[test]
    fn tolerance_is_the_tighter_of_a_tenth_of_the_scene_and_a_quarter_second() {
        let short = plan(&[scene(0, 0.0, 1.0)], &config());
        let long = plan(&[scene(0, 0.0, 60.0)], &config());
        assert!((short[0].tolerance - 0.1).abs() < 1e-9);
        assert!((long[0].tolerance - 0.25).abs() < 1e-9);
    }
}
