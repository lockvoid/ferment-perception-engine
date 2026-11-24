//! Photometric TRANSIENTS — the light changed and came back: a strobe hit, a
//! camera flash, a rolling-shutter half-frame, a strobe train with dark gaps.
//! AutoShot fires on them (58 of the 67 boundaries in one festival minute were
//! flashes), because on its 48×27 input a whited-out frame is as big a jump as
//! a cut and it never learned that "came back" is not an edit. This scanner
//! finds the transients **on the RAW luma stream** — the light is precisely
//! what it is looking for, so normalizing it away would blind it — and the
//! detector then asks, per model boundary, whether the SHOT changed across the
//! transient or only the light did.
//!
//! A transient is a RUN of variable length, judged relatively — no per-video
//! knobs:
//!   * it opens on an impulse (a pair's change ≥ [`IMPULSE_FACTOR`] × the
//!     rolling baseline of calm pairs) that does NOT settle at once — a change
//!     that settles immediately is a step (a cut), not a transient;
//!   * it stays open while the frames have not settled — settled = the frame's
//!     next [`SETTLE_FRAMES`] pairs carry no impulse; so a single white frame,
//!     a two-frame flash and a strobe train with gaps under three frames are
//!     each ONE run;
//!   * it is abandoned past [`MAX_RUN_FRAMES`] (a second) — that is a change of
//!     light or scene, or a pan, not a flash, and the model keeps the say.
//!
//! `before` is the last stable frame before the run, `after` the first stable
//! frame after it — the two frames whose shot identity decides.
//!
//! Port of iOS `PhotometricTransientScanner.swift`.

use super::photometric::mean_absolute_difference;
use crate::slow_motion_renderer::LumaPlane;

const BASELINE_WINDOW: usize = 24;
const MIN_BASELINE_SAMPLES: usize = 8;
pub(super) const IMPULSE_FACTOR: f64 = 2.5;
/// Absolute floor in luma units: below this nothing photometric happened.
pub(super) const DIFF_FLOOR: f64 = 4.0;
const SETTLE_FRAMES: usize = 2;
pub(super) const MAX_RUN_FRAMES: usize = 25;

/// First and last frame of a transient (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub first: usize,
    pub last: usize,
}

impl Run {
    /// The last stable frame before the run.
    #[must_use]
    pub fn before(&self) -> usize {
        self.first - 1
    }

    /// The first stable frame after the run.
    #[must_use]
    pub fn after(&self) -> usize {
        self.last + 1
    }
}

/// Streaming form: feed RAW luma planes in decode order; a run is reported
/// when its `after` frame is confirmed stable (two frames of look-ahead).
#[derive(Debug, Default)]
pub struct Scanner {
    last: Option<LumaPlane>,
    /// `diffs[k]` = change of pair (k−1, k); `diffs[0]` = 0.
    diffs: Vec<f64>,
    /// The baseline ring: pairs judged calm.
    calm: Vec<f64>,
    /// Index of the NEXT frame to be pushed.
    frame: usize,
    /// First frame of the run being watched.
    open: Option<usize>,
}

impl Scanner {
    pub fn push(&mut self, plane: &LumaPlane) -> Option<Run> {
        self.diffs.push(match &self.last {
            Some(last) => mean_absolute_difference(last, plane),
            None => 0.0,
        });
        self.last = Some(plane.clone());
        self.frame += 1;
        // Frame k is judged once d[k+1 … k+SETTLE_FRAMES] exist.
        let k = self.frame.checked_sub(1 + SETTLE_FRAMES)?;
        if k < 1 {
            return None;
        }
        self.judge(k)
    }

    fn judge(&mut self, k: usize) -> Option<Run> {
        let baseline = (self.calm.len() >= MIN_BASELINE_SAMPLES).then(|| median(&self.calm));
        let impulse = DIFF_FLOOR.max(IMPULSE_FACTOR * baseline.unwrap_or(0.0));
        let settled = self.diffs[k + 1..=k + SETTLE_FRAMES]
            .iter()
            .all(|&diff| diff < impulse);

        if let Some(first) = self.open {
            if settled {
                self.open = None;
                return Some(Run { first, last: k - 1 });
            }
            if k - first + 1 > MAX_RUN_FRAMES {
                self.open = None;
                for index in first..=k {
                    self.remember(self.diffs[index]);
                }
            }
            return None;
        }
        if baseline.is_some() && self.diffs[k] >= impulse && !settled {
            self.open = Some(k);
            return None;
        }
        self.remember(self.diffs[k]);
        None
    }

    fn remember(&mut self, diff: f64) {
        self.calm.push(diff);
        if self.calm.len() > BASELINE_WINDOW {
            self.calm.remove(0);
        }
    }
}

/// Whole-sequence form (tests).
#[must_use]
pub fn scan(planes: &[LumaPlane]) -> Vec<Run> {
    let mut scanner = Scanner::default();
    planes
        .iter()
        .filter_map(|plane| scanner.push(plane))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_scene_detector::synthetic::{flashed, static_sequence};

    const W: usize = 256;
    const H: usize = 144;

    fn sequence(length: usize) -> Vec<LumaPlane> {
        static_sequence(W, H, 7, length)
    }

    fn run(first: usize, last: usize) -> Run {
        Run { first, last }
    }

    // PhotometricTransientTests.swift

    // MARK: - Flashes are runs

    #[test]
    fn one_white_frame_is_a_one_frame_run() {
        let mut frames = sequence(40);
        frames[20] = flashed(&frames[20], 120);
        assert_eq!(scan(&frames), [run(20, 20)]);
    }

    #[test]
    fn two_frame_flash_is_one_run() {
        let mut frames = sequence(40);
        for frame in frames.iter_mut().take(22).skip(20) {
            *frame = flashed(frame, 120);
        }
        assert_eq!(scan(&frames), [run(20, 21)]);
    }

    #[test]
    fn decaying_flash_is_one_run() {
        let mut frames = sequence(40);
        for (index, delta) in [(20, 120), (21, 70), (22, 30)] {
            frames[index] = flashed(&frames[index], delta);
        }
        assert_eq!(scan(&frames), [run(20, 22)]);
    }

    #[test]
    fn strobe_train_with_short_gaps_is_one_run() {
        let mut frames = sequence(60);
        // gaps of 1, 1, 2 calm frames
        for index in [20, 22, 24, 27] {
            frames[index] = flashed(&frames[index], 120);
        }
        assert_eq!(scan(&frames), [run(20, 27)]);
    }

    #[test]
    fn flashes_three_calm_frames_apart_are_two_runs() {
        let mut frames = sequence(60);
        frames[20] = flashed(&frames[20], 120);
        frames[24] = flashed(&frames[24], 120); // 21, 22, 23 calm
        assert_eq!(scan(&frames), [run(20, 20), run(24, 24)]);
    }

    #[test]
    fn two_cuts_three_frames_apart_are_not_a_run() {
        let mut frames = sequence(20);
        frames.extend(static_sequence(W, H, 99, 3));
        frames.extend(static_sequence(W, H, 5, 20));
        assert_eq!(scan(&frames), []);
    }

    // MARK: - Steps and sustained change are not

    #[test]
    fn a_hard_cut_is_not_a_run() {
        let mut frames = sequence(20);
        frames.extend(static_sequence(W, H, 99, 20));
        assert_eq!(scan(&frames), []);
    }

    #[test]
    fn a_light_that_stays_on_is_not_a_run() {
        let mut frames = sequence(40);
        for frame in frames.iter_mut().skip(20) {
            *frame = flashed(frame, 80);
        }
        assert_eq!(scan(&frames), []);
    }

    #[test]
    fn a_change_that_never_settles_is_abandoned() {
        let mut frames = sequence(80);
        // 25 flashes, 1-frame gaps
        for index in (20..70).step_by(2) {
            frames[index] = flashed(&frames[index], 120);
        }
        assert_eq!(scan(&frames), []);
    }

    // MARK: - A cut inside a flash

    #[test]
    fn cut_under_a_flash_is_one_run_ending_on_the_new_shot() {
        let mut frames = sequence(20);
        frames.extend(static_sequence(W, H, 99, 20));
        frames[19] = flashed(&frames[19], 120); // outgoing shot's last frame is white
        frames[20] = flashed(&frames[20], 120); // new shot's first frame is white
        assert_eq!(scan(&frames), [run(19, 20)]);
    }

    #[test]
    fn no_verdicts_before_a_baseline_exists() {
        let mut frames = sequence(12);
        frames[3] = flashed(&frames[3], 120);
        assert_eq!(scan(&frames), []);
    }

    // MARK: - Run geometry

    #[test]
    fn a_run_names_the_stable_frames_around_it() {
        let run = run(20, 21);
        assert_eq!(run.before(), 19);
        assert_eq!(run.after(), 22);
    }

    /// The 640-wide ai feed: the flash law is scale-free, so it must read the
    /// same on the geometry the lead is rewiring onto.
    #[test]
    fn the_flash_law_holds_on_a_640_wide_feed() {
        let mut frames = static_sequence(640, 360, 7, 40);
        frames[20] = flashed(&frames[20], 120);
        assert_eq!(scan(&frames), [run(20, 20)]);
    }
}
