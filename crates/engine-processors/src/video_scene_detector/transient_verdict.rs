//! What a photometric transient meant for the shot: the same shot with a flash
//! inside it (`same_shot` — every model boundary touching the run is the light,
//! not an edit), or a shot change hidden in the flash — ONE boundary, at
//! `cut_start` when the identity places it decisively, else where the model's
//! most confident touching boundary put it.
//!
//! "Same shot" is decided by two keys, both measured upstream on strobe footage
//! (a festival minute: 9 real cuts, 58 flashes; a light-show reel; a film scene
//! with lightning):
//!   * the shot identity ([`ShotIdentifying`] — DINOv2): the same set across a
//!     flash and a different set across a flash separate, but a cut to another
//!     angle of the SAME set can sit in between — so below the identity's
//!     `same_shot_distance` it decides alone, up to `similar_shot_distance` it
//!     needs the second key;
//!   * the structure ([`BlockMotionEstimator`] on photometrically normalized
//!     planes, the jump-cut matcher): the same camera set-up locks most of its
//!     textured blocks across the transient (flashes: median 75 %), a real cut
//!     locks few (17–54 % measured) — [`STRUCTURE_LOCK_SHARE`] sits above the
//!     measured cuts.
//!
//! A boundary is never dropped on one thin key: identity alone only when it is
//! unambiguous, otherwise identity AND structure together. **No verdict is a
//! verdict too** — without an identity the model's boundaries stand untouched,
//! so the veto fails towards keeping cuts.
//!
//! Port of iOS `TransientVerdict.swift`.

use super::jump_cut::NORMALIZED_RESIDUAL;
use super::photometric::normalize;
use super::transient::Run;
use crate::slow_motion_renderer::{BlockMotionEstimator, LumaPlane};

/// Share of textured blocks that must still lock across the run for the
/// structure key to call it the same camera set-up.
pub const STRUCTURE_LOCK_SHARE: f64 = 0.6;
/// A run frame is the new shot's first when it is closer to `after` than to
/// `before` by at least this (real cuts measured 0.18–0.55; a tie is not a
/// placement).
pub const CUT_MARGIN: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    pub run: Run,
    pub same_shot: bool,
    /// First frame of the new shot, when the identity is decisive.
    pub cut_start: Option<usize>,
}

impl Verdict {
    /// Does a boundary (last frame of the outgoing scene, i.e. the pair
    /// `(frame, frame+1)`) belong to this transient? The run's own pairs are
    /// `(before, first)` … `(last, after)`; one pair of slack each side for the
    /// model's peak drift.
    #[must_use]
    pub fn touches(&self, boundary_frame: usize) -> bool {
        boundary_frame >= self.run.first.saturating_sub(2) && boundary_frame <= self.run.last + 1
    }
}

/// What a judge needs: the identity's two distances, and for one frame index in
/// `run.before()..=run.after()` its embedding and raw luma.
///
/// Embeddings are already computed — the encoder is async and this law is not.
/// [`TransientCollector`](super::shot_identity::TransientCollector) fills them
/// in two rounds so a same-shot run still costs exactly two encoder passes.
pub trait ShotFrames {
    fn same_shot_distance(&self) -> f32;
    fn similar_shot_distance(&self) -> f32;
    /// The frame's embedding, or empty when it was not embedded.
    fn embedding(&self, frame: usize) -> &[f32];
    fn luma(&self, frame: usize) -> Option<&LumaPlane>;
}

/// `None` = no verdict: without an identity (no encoder, a frame gone) the
/// model's boundaries stand untouched.
#[must_use]
pub fn judge(run: Run, frames: &impl ShotFrames) -> Option<Verdict> {
    let before = frames.embedding(run.before());
    let after = frames.embedding(run.after());
    if before.is_empty() || after.is_empty() {
        return None;
    }
    let across = distance(before, after);
    let mut lock = -1.0;
    if across >= frames.same_shot_distance()
        && across < frames.similar_shot_distance()
        && let (Some(a), Some(b)) = (frames.luma(run.before()), frames.luma(run.after()))
    {
        lock = structure_lock(a, b);
    }
    let same = across < frames.same_shot_distance()
        || (across < frames.similar_shot_distance() && lock >= STRUCTURE_LOCK_SHARE);
    if same {
        return Some(Verdict {
            run,
            same_shot: true,
            cut_start: None,
        });
    }
    for frame in run.first..=run.last {
        let embedding = frames.embedding(frame);
        if embedding.is_empty() {
            // No embedding for this frame: either the identity could not answer
            // (upstream lands here too — its distance to an empty vector is 0,
            // so neither margin is met and the placement is left to the model),
            // or this is the collector's first round and the run's own frames
            // are not embedded yet. Both mean "not placed".
            return Some(Verdict {
                run,
                same_shot: false,
                cut_start: None,
            });
        }
        let to_before = distance(embedding, before);
        let to_after = distance(embedding, after);
        if to_after <= to_before - CUT_MARGIN {
            return Some(Verdict {
                run,
                same_shot: false,
                cut_start: Some(frame),
            });
        }
        if to_before <= to_after - CUT_MARGIN {
            continue;
        }
        // A tie is not a placement: the model's best boundary keeps the say.
        return Some(Verdict {
            run,
            same_shot: false,
            cut_start: None,
        });
    }
    Some(Verdict {
        run,
        same_shot: false,
        cut_start: Some(run.after()),
    })
}

/// Euclidean distance over the shared prefix of two embeddings.
#[must_use]
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt()
}

/// Share of the textured blocks of `from` that find their match in `to` on the
/// normalized planes (structure, not light).
#[must_use]
pub fn structure_lock(from: &LumaPlane, to: &LumaPlane) -> f64 {
    let field = BlockMotionEstimator::estimate_matching(
        &normalize(from),
        &normalize(to),
        from,
        NORMALIZED_RESIDUAL,
    );
    if field.textured_blocks == 0 {
        return 0.0;
    }
    field.vectors.len() as f64 / field.textured_blocks as f64
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::video_scene_detector::synthetic::texture;

    const W: usize = 256;
    const H: usize = 144;

    /// Embeddings as points on a line: distance = |a − b|. Mirrors the Swift
    /// suite's stub, so each rule is pinned without a model.
    struct Points {
        points: HashMap<usize, Vec<f32>>,
        luma: HashMap<usize, LumaPlane>,
    }

    impl Points {
        /// Every listed frame embeds to its point; anything unlisted is a frame
        /// the identity could not answer for.
        fn new(points: &[(usize, f32)]) -> Self {
            Self {
                points: points
                    .iter()
                    .map(|&(frame, value)| (frame, vec![value]))
                    .collect(),
                luma: HashMap::new(),
            }
        }

        fn with_luma(mut self, luma: &[(usize, LumaPlane)]) -> Self {
            self.luma = luma.iter().cloned().collect();
            self
        }
    }

    impl ShotFrames for Points {
        fn same_shot_distance(&self) -> f32 {
            0.5
        }
        fn similar_shot_distance(&self) -> f32 {
            0.7
        }
        fn embedding(&self, frame: usize) -> &[f32] {
            self.points.get(&frame).map_or(&[], Vec::as_slice)
        }
        fn luma(&self, frame: usize) -> Option<&LumaPlane> {
            self.luma.get(&frame)
        }
    }

    const RUN: Run = Run {
        first: 20,
        last: 21,
    }; // before 19, after 22

    fn same() -> LumaPlane {
        texture(W, H, 3)
    }
    fn other() -> LumaPlane {
        texture(W, H, 44)
    }

    // TransientVerdictTests.swift

    // MARK: - Same shot / shot change

    #[test]
    fn unambiguous_identity_alone_means_same_shot() {
        let verdict = judge(RUN, &Points::new(&[(19, 0.0), (22, 0.3)])).expect("a verdict");
        assert!(verdict.same_shot);
    }

    #[test]
    fn far_identity_means_shot_change() {
        let verdict = judge(
            RUN,
            &Points::new(&[(19, 0.0), (20, 1.0), (21, 1.0), (22, 1.0)]),
        )
        .expect("a verdict");
        assert!(!verdict.same_shot);
    }

    #[test]
    fn band_identity_needs_the_structure_to_lock() {
        let locked = judge(
            RUN,
            &Points::new(&[(19, 0.0), (22, 0.6)]).with_luma(&[(19, same()), (22, same())]),
        )
        .expect("a verdict");
        assert!(locked.same_shot, "the same set under a flash locks");

        let torn = judge(
            RUN,
            &Points::new(&[(19, 0.0), (22, 0.6)]).with_luma(&[(19, same()), (22, other())]),
        )
        .expect("a verdict");
        assert!(!torn.same_shot);

        let unknown = judge(RUN, &Points::new(&[(19, 0.0), (22, 0.6)])).expect("a verdict");
        assert!(!unknown.same_shot, "no luma is no second key");
    }

    #[test]
    fn no_identity_is_no_verdict() {
        assert_eq!(judge(RUN, &Points::new(&[])), None);
    }

    // MARK: - Placing the cut

    #[test]
    fn cut_lands_on_the_first_run_frame_that_looks_like_after() {
        let verdict = judge(
            RUN,
            &Points::new(&[(19, 0.0), (20, 0.1), (21, 0.9), (22, 1.0)]),
        )
        .expect("a verdict");
        assert_eq!(verdict.cut_start, Some(21));
    }

    #[test]
    fn cut_on_the_first_run_frame() {
        let verdict = judge(
            RUN,
            &Points::new(&[(19, 0.0), (20, 0.95), (21, 1.0), (22, 1.0)]),
        )
        .expect("a verdict");
        assert_eq!(verdict.cut_start, Some(20));
    }

    #[test]
    fn cut_after_the_run_when_every_run_frame_looks_like_before() {
        let verdict = judge(
            RUN,
            &Points::new(&[(19, 0.0), (20, 0.05), (21, 0.05), (22, 1.0)]),
        )
        .expect("a verdict");
        assert_eq!(verdict.cut_start, Some(22));
    }

    #[test]
    fn a_tie_is_not_a_placement() {
        let verdict = judge(
            RUN,
            &Points::new(&[(19, 0.0), (20, 0.5), (21, 0.9), (22, 1.0)]),
        )
        .expect("a verdict");
        assert!(!verdict.same_shot);
        assert_eq!(verdict.cut_start, None);
    }

    // MARK: - Touching

    #[test]
    fn touching_covers_the_run_pairs_plus_one_of_slack() {
        let verdict = Verdict {
            run: RUN,
            same_shot: true,
            cut_start: None,
        };
        assert!(!verdict.touches(17));
        assert!(verdict.touches(18));
        assert!(verdict.touches(19));
        assert!(verdict.touches(21));
        assert!(verdict.touches(22));
        assert!(!verdict.touches(23));
    }

    /// A run opening on frame 1 has no frame −1; the slack window must clamp
    /// rather than wrap into the top of the address space.
    #[test]
    fn touching_clamps_at_the_start_of_the_video() {
        let verdict = Verdict {
            run: Run { first: 1, last: 2 },
            same_shot: true,
            cut_start: None,
        };
        assert!(verdict.touches(0));
        assert!(verdict.touches(3));
        assert!(!verdict.touches(4));
    }

    // MARK: - The structure key on its own

    #[test]
    fn structure_locks_across_a_flash_and_tears_across_a_cut() {
        let plane = texture(W, H, 3);
        let lit = crate::video_scene_detector::synthetic::flashed(&plane, 70);
        assert!(
            structure_lock(&plane, &lit) >= STRUCTURE_LOCK_SHARE,
            "lock across a flash: {}",
            structure_lock(&plane, &lit)
        );
        assert!(
            structure_lock(&plane, &other()) < STRUCTURE_LOCK_SHARE,
            "lock across a cut: {}",
            structure_lock(&plane, &other())
        );
    }

    #[test]
    fn structure_lock_of_a_flat_plane_is_zero() {
        let flat = LumaPlane {
            width: W,
            height: H,
            bytes_per_row: W,
            data: vec![128; W * H],
        };
        assert!(structure_lock(&flat, &flat).abs() < f64::EPSILON);
    }
}
