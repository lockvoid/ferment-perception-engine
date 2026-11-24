//! Same-camera JUMP CUTS — AutoShot's documented blind spot: an edit where
//! the subject teleports but the set stays put. Every shot-boundary network
//! calls that "no shot change".
//!
//! The old test here was bimodality alone — subject blocks tear while
//! background blocks lock — and it fired on everything that LOOKS like that
//! for one frame: a strobe hit or a sweeping beam (brightness tears blocks
//! judged by absolute residual), a head whip or dancing (fast motion tears the
//! subject on EVERY pair, the background locks on every pair).
//!
//! A jump cut is now four things at once, all judged relatively — no
//! per-video knobs:
//!   1. structure, not light: matching runs on photometrically normalized
//!      planes ([`super::photometric`]), so illumination changes lock;
//!   2. an IMPULSE in time: the pair's change clears the rolling baseline
//!      while the pairs before/after sit at it — a run of high pairs is
//!      motion, and a change that reverts within two frames is a flash;
//!   3. a COMPACT tear: the torn blocks have a body (a 4-connected component
//!      of ≥ 4 blocks holding ≥ 20 % of them) — beams, dust and blur scatter
//!      into singletons;
//!   4. a locked background under it — the bimodality, kept.
//!
//! One boundary per minimum scene ([`MIN_GAP_FRAMES`]).
//!
//! Port of iOS `JumpCutDetector.swift`.

use std::collections::HashSet;

use super::photometric::{mean_absolute_difference, normalize};
use crate::slow_motion_renderer::{BlockMotionEstimator, LumaPlane};
use crate::video_source::RgbFrame;

/// Pairs behind the candidate whose median is "what this footage does".
const BASELINE_WINDOW: usize = 24;
/// Fewer than this and the baseline is not a baseline yet — no verdicts.
const MIN_BASELINE_SAMPLES: usize = 8;
/// The candidate pair's mean change vs the baseline.
const IMPULSE_FACTOR: f64 = 2.5;
/// … while its neighbours stay near it (a run = motion, not an edit).
const NEIGHBOUR_FACTOR: f64 = 1.6;
/// Absolute floor for the impulse, in normalized units (32 per σ): 4 = 0.125 σ
/// — below this nothing structural happened however quiet the baseline is.
const DIFF_FLOOR: f64 = 4.0;
/// A change that has reverted (frame t−1 ≈ t+1 or t+2 at half the impulse) is
/// a flash, whatever the block matcher would say.
const REVERT_RATIO: f64 = 0.5;
/// Verdict: at least this fraction of textured blocks torn …
const TORN_FRACTION_FLOOR: f64 = 0.12;
/// … at least this many locked (the static background) …
const MIN_LOCKED_BLOCKS: usize = 8;
/// … and the tear has a BODY: its largest 4-connected component holds at least
/// this many blocks and this share of the torn blocks. A subject is a few
/// bodies (head, hands, a prop); beams, dust and blur are singletons.
const COMPACT_BODY_BLOCKS: usize = 4;
const COMPACT_SHARE: f64 = 0.2;
/// Match gate on the normalized planes: 12/32 = 0.375 σ mean residual.
pub(super) const NORMALIZED_RESIDUAL: f64 = 12.0;
/// One boundary per minimum scene — an edit every 5 frames is not editing.
const MIN_GAP_FRAMES: usize = 12;
/// Frames the scanner holds: t−1, t, t+1, t+2. The pair (t−1, t) is decided
/// once t+2 has arrived — that is the flash-revert look-ahead.
const WINDOW_FRAMES: usize = 4;

/// A verified jump cut: `frame` is the LAST frame of the outgoing scene (the
/// detector-internal boundary convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub frame: usize,
    /// Fraction of textured blocks with no acceptable match — the measured
    /// tear, an honest confidence in (0, 1].
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JumpCutDetector;

impl JumpCutDetector {
    /// Whole-sequence form (tests, and any caller that already holds the
    /// frames). Frame indices are positions in `planes`.
    #[must_use]
    pub fn scan(planes: &[LumaPlane]) -> Vec<Detection> {
        let mut scanner = Scanner::default();
        planes
            .iter()
            .filter_map(|plane| scanner.push(plane))
            .collect()
    }

    /// Block-level verdict for a nominated pair (normalized planes; texture
    /// judged on the raw outgoing frame): bimodal AND compact. The measured
    /// tear fraction is the confidence.
    #[must_use]
    pub fn verdict(from: &LumaPlane, to: &LumaPlane, texture: &LumaPlane) -> Option<f32> {
        let field = BlockMotionEstimator::estimate_matching(from, to, texture, NORMALIZED_RESIDUAL);
        let locked = field.vectors.len();
        let torn = field.torn_blocks.len();
        if field.textured_blocks == 0 || locked < MIN_LOCKED_BLOCKS {
            return None;
        }
        let torn_fraction = torn as f64 / field.textured_blocks as f64;
        if torn_fraction < TORN_FRACTION_FLOOR {
            return None;
        }
        let body = largest_body(&field.torn_blocks);
        if body < COMPACT_BODY_BLOCKS || (body as f64) < COMPACT_SHARE * torn as f64 {
            return None;
        }
        Some(torn_fraction as f32)
    }
}

/// The scene reader already enforces the iOS analysis geometry. Convert those
/// exact pixels to luma without a second resize (or upscaling a tiny source).
pub(crate) fn analysis_luma(frame: &RgbFrame) -> Option<LumaPlane> {
    rgb_luma(frame)
}

/// Streaming form: feed frames in decode order, collect verdicts as they
/// finalize.
#[derive(Debug, Default)]
pub struct Scanner {
    /// Last 4 raw planes: t−1 … t+2 (the texture reference).
    raw: Vec<LumaPlane>,
    /// Their normalized twins.
    normalized: Vec<LumaPlane>,
    /// `diffs[k]` = change of pair (k−1, k); index = frame k.
    diffs: Vec<f64>,
    /// Index of the NEXT frame to be pushed.
    frame: usize,
    last_accepted: Option<usize>,
}

impl Scanner {
    pub fn push(&mut self, plane: &LumaPlane) -> Option<Detection> {
        let normalized = normalize(plane);
        self.diffs.push(match self.normalized.last() {
            Some(last) => mean_absolute_difference(last, &normalized),
            None => 0.0,
        });
        self.raw.push(plane.clone());
        self.normalized.push(normalized);
        if self.raw.len() > WINDOW_FRAMES {
            self.raw.remove(0);
            self.normalized.remove(0);
        }
        self.frame += 1;
        if self.raw.len() < WINDOW_FRAMES {
            return None;
        }
        // Frames held: t−1, t, t+1, t+2 ⇒ candidate pair index t = frame − 3.
        let t = self.frame - (WINDOW_FRAMES - 1);
        let hit = self.judge(t)?;
        self.last_accepted = Some(hit.frame);
        Some(hit)
    }

    /// The candidate is the pair (t−1, t); `raw[0..4]` = frames t−1 … t+2.
    fn judge(&self, t: usize) -> Option<Detection> {
        if let Some(last) = self.last_accepted
            && (t - 1) - last < MIN_GAP_FRAMES
        {
            return None;
        }
        let d = self.diffs[t];
        // Baseline: the pairs strictly before the candidate's neighbourhood.
        let end = t - 1;
        let start = end.saturating_sub(BASELINE_WINDOW).max(1);
        if end < start + MIN_BASELINE_SAMPLES {
            return None;
        }
        let baseline = median(&self.diffs[start..end]);
        let impulse = DIFF_FLOOR.max(IMPULSE_FACTOR * baseline);
        let calm = DIFF_FLOOR.max(NEIGHBOUR_FACTOR * baseline);
        if d < impulse || self.diffs[t - 1] > calm || self.diffs[t + 1] > calm {
            return None;
        }
        // Flash: t−1 meets t+1 or t+2 again.
        let ahead1 = mean_absolute_difference(&self.normalized[0], &self.normalized[2]);
        let ahead2 = mean_absolute_difference(&self.normalized[0], &self.normalized[3]);
        if ahead1.min(ahead2) < REVERT_RATIO * d {
            return None;
        }
        let confidence =
            JumpCutDetector::verdict(&self.normalized[0], &self.normalized[1], &self.raw[0])?;
        Some(Detection {
            frame: t - 1,
            confidence,
        })
    }
}

/// Size of the largest 4-connected component of the torn blocks.
fn largest_body(blocks: &[(usize, usize)]) -> usize {
    let mut remaining: HashSet<(usize, usize)> = blocks.iter().copied().collect();
    let mut largest = 0;
    while let Some(&seed) = remaining.iter().next() {
        remaining.remove(&seed);
        let mut stack = vec![seed];
        let mut size = 0;
        while let Some((bx, by)) = stack.pop() {
            size += 1;
            let neighbours = [
                Some((bx + 1, by)),
                bx.checked_sub(1).map(|x| (x, by)),
                Some((bx, by + 1)),
                by.checked_sub(1).map(|y| (bx, y)),
            ];
            for neighbour in neighbours.into_iter().flatten() {
                if remaining.remove(&neighbour) {
                    stack.push(neighbour);
                }
            }
        }
        largest = largest.max(size);
    }
    largest
}

fn rgb_luma(frame: &RgbFrame) -> Option<LumaPlane> {
    let width = usize::try_from(frame.width).ok()?;
    let height = usize::try_from(frame.height).ok()?;
    if frame.data.len() != width.checked_mul(height)?.checked_mul(3)? {
        return None;
    }
    let data = frame
        .data
        .chunks_exact(3)
        .map(|pixel| {
            let value = 77 * u32::from(pixel[0])
                + 150 * u32::from(pixel[1])
                + 29 * u32::from(pixel[2])
                + 128;
            (value >> 8) as u8
        })
        .collect();
    Some(LumaPlane {
        width,
        height,
        bytes_per_row: width,
        data,
    })
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_scene_detector::synthetic::{
        flashed, jittered, static_sequence, teleported, texture,
    };

    const W: usize = 256;
    const H: usize = 144;

    /// Half real texture, half a flat wall carrying only sensor noise — the
    /// shape `texture_reference` exists for. `frame` reseeds the wall's noise.
    fn half_flat_shot(frame: usize) -> LumaPlane {
        let textured = texture(W, H, 7);
        let mut data = vec![128_u8; W * H];
        let mut state = (frame as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        for y in 0..H {
            for x in 0..W {
                data[y * W + x] = if x < W / 2 {
                    textured.data[y * W + x]
                } else {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let noise = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 62) as i32 - 1;
                    (128 + noise * 2) as u8
                };
            }
        }
        LumaPlane {
            width: W,
            height: H,
            bytes_per_row: W,
            data,
        }
    }

    fn base() -> LumaPlane {
        texture(W, H, 7)
    }

    // JumpCutDetectorTests.swift

    // MARK: - The one shape that IS a jump cut

    #[test]
    fn a_teleported_subject_in_a_static_shot_is_one_jump_cut_at_its_frame() {
        let mut planes = static_sequence(W, H, 7, 40);
        let after = teleported(&base(), 99);
        for (index, plane) in planes.iter_mut().enumerate().skip(20) {
            *plane = jittered(&after, index);
        }

        let hits = JumpCutDetector::scan(&planes);

        assert_eq!(
            hits.iter().map(|hit| hit.frame).collect::<Vec<_>>(),
            vec![19],
            "one boundary, on the LAST frame of the outgoing shot"
        );
        assert!(hits[0].confidence > 0.25, "{}", hits[0].confidence);
        assert!(hits[0].confidence <= 1.0);
    }

    /// The ai feed the lead is rewiring the detector onto is 640 wide; the
    /// analysis downscale is the detector's business, so the law has to hold
    /// at that geometry too.
    #[test]
    fn the_same_jump_cut_is_found_on_a_640_wide_feed() {
        let (width, height) = (640, 360);
        let mut planes = static_sequence(width, height, 7, 40);
        let after = teleported(&texture(width, height, 7), 99);
        for (index, plane) in planes.iter_mut().enumerate().skip(20) {
            *plane = jittered(&after, index);
        }

        let hits = JumpCutDetector::scan(&planes);

        assert_eq!(
            hits.iter().map(|hit| hit.frame).collect::<Vec<_>>(),
            vec![19]
        );
        assert!(hits[0].confidence > 0.25, "{}", hits[0].confidence);
    }

    // MARK: - What must NOT be a jump cut

    #[test]
    fn a_strobe_flash_is_not_a_jump_cut() {
        let mut planes = static_sequence(W, H, 7, 40);
        planes[20] = flashed(&planes[20], 70); // one-frame flash
        planes[30] = flashed(&planes[30], 90); // two-frame flash
        planes[31] = flashed(&planes[31], 90);

        assert_eq!(
            JumpCutDetector::scan(&planes),
            [],
            "brightness is not structure"
        );
    }

    #[test]
    fn a_partial_flash_light_beam_is_not_a_jump_cut() {
        // A beam sweeps the right half only: half the frame jumps in
        // brightness, the other half is untouched — bimodal by luma, but every
        // block still has its structure.
        let mut planes = static_sequence(W, H, 7, 40);
        let mut data = planes[20].data.clone();
        for y in 0..H {
            for x in W / 2..W {
                data[y * W + x] = (i32::from(data[y * W + x]) + 80).clamp(0, 255) as u8;
            }
        }
        planes[20] = LumaPlane {
            width: W,
            height: H,
            bytes_per_row: W,
            data,
        };

        assert_eq!(JumpCutDetector::scan(&planes), []);
    }

    #[test]
    fn sustained_fast_motion_is_not_a_jump_cut() {
        // A subject region flying 30 px per frame (past the 20 px search
        // radius — a head whip, motion blur) for 6 frames while the background
        // holds: every pair tears the subject and locks the rest — the old
        // bimodal test's exact false positive.
        let mut planes = static_sequence(W, H, 7, 50);
        let subject = texture(W, H, 33);
        for (index, plane) in planes.iter_mut().enumerate().take(21).skip(15) {
            let mut data = plane.data.clone();
            let ox = 20 + (index - 15) * 30;
            for y in 20..124 {
                for x in ox..ox + 80 {
                    data[y * W + x] = subject.data[y * W + x - ox];
                }
            }
            *plane = LumaPlane {
                width: W,
                height: H,
                bytes_per_row: W,
                data,
            };
        }

        assert_eq!(
            JumpCutDetector::scan(&planes),
            [],
            "a run of high pairs is motion, not an edit"
        );
    }

    #[test]
    fn a_whip_pan_is_not_a_jump_cut() {
        let mut planes = static_sequence(W, H, 7, 40);
        planes[20] = crate::video_scene_detector::synthetic::shifted(&base(), 60, 0);
        for index in 21..40 {
            planes[index] = jittered(&planes[20].clone(), index);
        }

        assert_eq!(
            JumpCutDetector::scan(&planes),
            [],
            "diffuse failure — nothing locks"
        );
    }

    #[test]
    fn scattered_torn_blocks_are_not_a_jump_cut() {
        // A checkerboard of 16×16 blocks (no two 4-adjacent) swapped for
        // unrelated texture, the rest untouched: half the textured blocks torn
        // — far over the tear floor — the other half locked: the old signature
        // at full strength, but the tear has no body.
        let mut planes = static_sequence(W, H, 7, 40);
        let replacement = texture(W, H, 5);
        let mut data = planes[20].data.clone();
        // Block grid starts at the search margin (20 px).
        for row in 0..6 {
            for column in (0..13).filter(|column| (row + column) % 2 == 0) {
                let (bx, by) = (20 + column * 16, 20 + row * 16);
                for y in by..by + 16 {
                    for x in bx..bx + 16 {
                        data[y * W + x] = replacement.data[y * W + x];
                    }
                }
            }
        }
        planes[20] = LumaPlane {
            width: W,
            height: H,
            bytes_per_row: W,
            data,
        };
        for index in 21..40 {
            planes[index] = jittered(&planes[20].clone(), index);
        }

        assert_eq!(JumpCutDetector::scan(&planes), []);
    }

    #[test]
    fn two_jump_cuts_inside_the_minimum_gap_collapse_to_the_first() {
        let mut planes = static_sequence(W, H, 7, 40);
        let (a, b) = (teleported(&base(), 99), teleported(&base(), 123));
        for (index, plane) in planes.iter_mut().enumerate().take(25).skip(20) {
            *plane = jittered(&a, index);
        }
        for (index, plane) in planes.iter_mut().enumerate().skip(25) {
            *plane = jittered(&b, index);
        }

        assert_eq!(
            JumpCutDetector::scan(&planes)
                .iter()
                .map(|hit| hit.frame)
                .collect::<Vec<_>>(),
            vec![19],
            "an edit every 5 frames is not editing — one boundary per minimum scene"
        );
    }

    #[test]
    fn flat_content_is_inconclusive() {
        let flat = LumaPlane {
            width: W,
            height: H,
            bytes_per_row: W,
            data: vec![128; W * H],
        };
        assert_eq!(JumpCutDetector::scan(&vec![flat; 30]), []);
    }

    #[test]
    fn no_verdicts_before_a_baseline_exists() {
        let mut planes = static_sequence(W, H, 7, 14);
        let after = teleported(&base(), 99);
        for (index, plane) in planes.iter_mut().enumerate().skip(5) {
            *plane = jittered(&after, index);
        }
        assert_eq!(JumpCutDetector::scan(&planes), []);
    }

    /// The flash-revert rule, which nothing else covers: photometric
    /// normalization kills a *brightness* flash before the impulse test, and
    /// the neighbour rule kills a one-frame interruption. What is left for the
    /// revert rule is a STRUCTURAL interruption two frames long — the subject
    /// is replaced at 20, holds at 21, and the picture is back at 22. The
    /// opening pair (19, 20) looks exactly like a jump cut to the block matcher
    /// and passes both the impulse and the neighbour test; only "frame 19 meets
    /// frame 22 again" rejects it. Delete the rule and the boundary moves to
    /// 19, where nothing was edited.
    #[test]
    fn a_structural_interruption_that_reverts_is_not_an_edit_at_its_start() {
        let mut planes = static_sequence(W, H, 7, 40);
        let interrupted = teleported(&base(), 99);
        for (index, plane) in planes.iter_mut().enumerate().take(22).skip(20) {
            *plane = jittered(&interrupted, index);
        }

        assert_eq!(
            JumpCutDetector::scan(&planes)
                .iter()
                .map(|hit| hit.frame)
                .collect::<Vec<_>>(),
            vec![21],
            "the opening pair reverts; only the pair where the picture stops \
             reverting can be a boundary"
        );
    }

    /// Texture is judged on the RAW plane, matching on the NORMALIZED one.
    /// Judge texture on the normalized plane instead and a flat wall's sensor
    /// noise — z-scored up to full contrast — becomes "textured" blocks that
    /// match nothing, and a static shot invents a jump cut out of them.
    #[test]
    fn texture_is_judged_on_the_raw_plane_not_the_normalized_one() {
        let (a, b) = (half_flat_shot(0), half_flat_shot(1));
        let (normalized_a, normalized_b) = (normalize(&a), normalize(&b));

        let honest = BlockMotionEstimator::estimate_matching(
            &normalized_a,
            &normalized_b,
            &a,
            NORMALIZED_RESIDUAL,
        );
        assert_eq!(honest.torn_blocks.len(), 0, "nothing moved");
        assert!(honest.textured_blocks >= MIN_LOCKED_BLOCKS);
        assert_eq!(
            JumpCutDetector::verdict(&normalized_a, &normalized_b, &a),
            None,
            "a static half-flat shot is not an edit"
        );

        let confused = BlockMotionEstimator::estimate_matching(
            &normalized_a,
            &normalized_b,
            &normalized_a,
            NORMALIZED_RESIDUAL,
        );
        assert!(
            confused.textured_blocks > honest.textured_blocks,
            "the wall's noise must look textured once normalized, or this test \
             proves nothing: {} vs {}",
            confused.textured_blocks,
            honest.textured_blocks
        );
        assert!(
            JumpCutDetector::verdict(&normalized_a, &normalized_b, &normalized_a).is_some(),
            "and it must be enough to invent a boundary — that is the trap"
        );
    }

    // MARK: - The verdict on its own

    #[test]
    fn the_verdict_reads_the_tear_fraction_as_confidence() {
        let before = base();
        let after = teleported(&before, 99);
        let confidence = JumpCutDetector::verdict(&normalize(&before), &normalize(&after), &before)
            .expect("subject swap must be detected");
        assert!(
            confidence > 0.25,
            "an obvious subject swap should tear far above the 0.12 floor: {confidence}"
        );
        assert!(confidence <= 1.0);
    }

    #[test]
    fn the_verdict_needs_a_locked_background() {
        let before = base();
        let after = texture(W, H, 99);
        assert_eq!(
            JumpCutDetector::verdict(&normalize(&before), &normalize(&after), &before),
            None,
            "everything torn is a cut the model already sees, not a jump cut"
        );
    }

    // MARK: - The compact-body rule

    #[test]
    fn largest_body_counts_the_biggest_four_connected_component() {
        // An L of four, a diagonal pair (never 4-connected), a singleton.
        let blocks = [(1, 1), (2, 1), (2, 2), (2, 3), (5, 5), (6, 6), (9, 0)];
        assert_eq!(largest_body(&blocks), 4);
        assert_eq!(largest_body(&[]), 0);
        assert_eq!(largest_body(&[(0, 0)]), 1);
    }

    #[test]
    fn largest_body_does_not_wrap_at_the_grid_edge() {
        // Column 0 of row 1 and column 0 of row 0 ARE adjacent; the underflow
        // guard must not silently join (0, 0) to some (usize::MAX, 0).
        assert_eq!(largest_body(&[(0, 0), (0, 1)]), 2);
        assert_eq!(largest_body(&[(0, 0), (2, 0)]), 1);
    }
}
