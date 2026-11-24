//! "Is this the same shot?" — a frame embedding whose distance separates the
//! same set under different light from a different set, and the two distances
//! that read it: below `same_shot_distance` the identity decides alone, up to
//! `similar_shot_distance` it needs the structure key.
//!
//! Port of iOS `ShotIdentity.swift` + `EncoderShotIdentity.swift`.
//!
//! **Deviation from iOS, recorded:** upstream has two identities (DINOv2 and
//! Apple's Vision feature print) and therefore a `providesSceneEmbeddings` flag
//! that keeps the two vector spaces from mixing in the server's column. Vision
//! is Apple-only and is not ported (locked decision 9), so there is exactly one
//! identity here and the flag has no case to distinguish — it is gone. The
//! trait stays because the veto must work with no identity at all, and because
//! the verdict suite pins its rules against a stub.

use std::collections::HashMap;

use async_trait::async_trait;

use super::transient::{self, Run};
use super::transient_verdict::{ShotFrames, Verdict, judge};
use crate::dinov2::{Dinov2SemanticEncoder, FrameSnapshot};
use crate::slow_motion_renderer::LumaPlane;

/// Frames retained behind the scanner: `before` + the longest possible run +
/// `after` + the settle look-ahead.
const KEEP_FRAMES: usize = transient::MAX_RUN_FRAMES + 6;

/// The encoder is async, so this is too. Everything downstream of it — the
/// verdict law, the placement, the medoid — stays synchronous and pure.
#[async_trait(?Send)]
pub trait ShotIdentifying {
    /// The frame's embedding, or empty when this identity cannot answer.
    async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32>;
    fn same_shot_distance(&self) -> f32;
    fn similar_shot_distance(&self) -> f32;
}

/// The flash veto's identity: `DINOv2SemanticEncoder` with the two distances
/// measured upstream on the labelled festival minute — a real cut across a
/// flash sits at ≥ 0.82 Euclidean on unit vectors (cosine 0.34), the same shot
/// across a flash at ≤ 0.67 (0.22), a ×1.5 gap where Vision's feature print had
/// ×1.03.
#[derive(Debug, Clone)]
pub struct EncoderShotIdentity {
    encoder: Dinov2SemanticEncoder,
}

impl EncoderShotIdentity {
    pub const SAME_SHOT_DISTANCE: f32 = 0.72;
    pub const SIMILAR_SHOT_DISTANCE: f32 = 0.80;

    #[must_use]
    pub fn new(encoder: Dinov2SemanticEncoder) -> Self {
        Self { encoder }
    }
}

#[async_trait(?Send)]
impl ShotIdentifying for EncoderShotIdentity {
    async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32> {
        self.encoder.embedding(frame).await
    }

    fn same_shot_distance(&self) -> f32 {
        Self::SAME_SHOT_DISTANCE
    }

    fn similar_shot_distance(&self) -> f32 {
        Self::SIMILAR_SHOT_DISTANCE
    }
}

/// The transient pass over one decode: push every decoded frame (raw luma for
/// the scanner and the structure key, a snapshot for the identity), read the
/// verdicts after the stream ends. Frames are kept only as long as an
/// unfinished run can still need them — a 20-minute clip holds 31 of them, not
/// 36 000.
pub struct TransientCollector<I: ShotIdentifying> {
    scanner: transient::Scanner,
    identity: I,
    retained: Vec<(usize, LumaPlane, FrameSnapshot)>,
    verdicts: Vec<Verdict>,
    frame: usize,
}

impl<I: ShotIdentifying> TransientCollector<I> {
    #[must_use]
    pub fn new(identity: I) -> Self {
        Self {
            scanner: transient::Scanner::default(),
            identity,
            retained: Vec::new(),
            verdicts: Vec::new(),
            frame: 0,
        }
    }

    pub async fn push(&mut self, luma: &LumaPlane, snapshot: FrameSnapshot) {
        self.retained.push((self.frame, luma.clone(), snapshot));
        // Swift removes frame - keepFrames after inserting the current one:
        // keep exactly that many snapshots, including the first/current frame.
        let horizon = self.frame.saturating_add(1).saturating_sub(KEEP_FRAMES);
        self.retained.retain(|(frame, _, _)| *frame >= horizon);
        self.frame += 1;

        let Some(run) = self.scanner.push(luma) else {
            return;
        };
        if let Some(verdict) = self.judge_run(run).await {
            self.verdicts.push(verdict);
        }
    }

    /// Two rounds, so a same-shot run — the common case, a flash — costs
    /// exactly two encoder passes instead of one per run frame. `judge` is a
    /// pure function of the embeddings it is given, so asking it twice is free
    /// apart from one repeated block match on the rare shot-change run.
    async fn judge_run(&self, run: Run) -> Option<Verdict> {
        let mut embeddings = HashMap::new();
        for frame in [run.before(), run.after()] {
            self.embed(&mut embeddings, frame).await;
        }
        let verdict = judge(run, &self.frames(&embeddings))?;
        if verdict.same_shot {
            return Some(verdict);
        }
        for frame in run.first..=run.last {
            self.embed(&mut embeddings, frame).await;
        }
        judge(run, &self.frames(&embeddings))
    }

    async fn embed(&self, embeddings: &mut HashMap<usize, Vec<f32>>, frame: usize) {
        let Some((_, _, snapshot)) = self.retained.iter().find(|(index, _, _)| *index == frame)
        else {
            return;
        };
        embeddings.insert(frame, self.identity.embedding(snapshot).await);
    }

    fn frames<'a>(&'a self, embeddings: &'a HashMap<usize, Vec<f32>>) -> RunFrames<'a> {
        RunFrames {
            same_shot_distance: self.identity.same_shot_distance(),
            similar_shot_distance: self.identity.similar_shot_distance(),
            embeddings,
            retained: &self.retained,
        }
    }

    /// Test-only accounting of actual retained buffers, not an allocator/RSS estimate.
    #[cfg(test)]
    pub(crate) fn retained_pixel_storage(&self) -> (usize, usize, usize) {
        (
            self.retained.len(),
            self.retained
                .iter()
                .map(|(_, _, snapshot)| snapshot.rgb.len())
                .sum(),
            self.retained
                .iter()
                .map(|(_, luma, _)| luma.data.len())
                .sum(),
        )
    }

    #[must_use]
    pub fn verdicts(&self) -> &[Verdict] {
        &self.verdicts
    }

    #[must_use]
    pub fn into_verdicts(self) -> Vec<Verdict> {
        self.verdicts
    }
}

/// One run's view of the retained frames.
struct RunFrames<'a> {
    same_shot_distance: f32,
    similar_shot_distance: f32,
    embeddings: &'a HashMap<usize, Vec<f32>>,
    retained: &'a [(usize, LumaPlane, FrameSnapshot)],
}

impl ShotFrames for RunFrames<'_> {
    fn same_shot_distance(&self) -> f32 {
        self.same_shot_distance
    }

    fn similar_shot_distance(&self) -> f32 {
        self.similar_shot_distance
    }

    fn embedding(&self, frame: usize) -> &[f32] {
        self.embeddings.get(&frame).map_or(&[], Vec::as_slice)
    }

    fn luma(&self, frame: usize) -> Option<&LumaPlane> {
        self.retained
            .iter()
            .find(|(index, _, _)| *index == frame)
            .map(|(_, plane, _)| plane)
    }
}

/// Whole-sequence form (tests, and any caller that already holds the frames).
pub async fn collect_verdicts<I: ShotIdentifying>(
    identity: I,
    frames: &[(LumaPlane, FrameSnapshot)],
) -> Vec<Verdict> {
    let mut collector = TransientCollector::new(identity);
    for (luma, snapshot) in frames {
        collector.push(luma, snapshot.clone()).await;
    }
    collector.into_verdicts()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_scene_detector::synthetic::{flashed, static_sequence, texture};

    const W: usize = 256;
    const H: usize = 144;

    /// An identity that reads the snapshot's first byte as a shot label — a
    /// stub with real per-frame behaviour, so the collector's frame retention
    /// and two-round fetch are exercised rather than stubbed out.
    struct LabelIdentity;

    #[async_trait(?Send)]
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

    #[async_trait(?Send)]
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

    fn collect<I: ShotIdentifying>(
        identity: I,
        frames: &[(LumaPlane, FrameSnapshot)],
    ) -> Vec<Verdict> {
        futures_executor::block_on(collect_verdicts(identity, frames))
    }

    /// One flash inside one shot: the collector must find the run and call it
    /// the same shot from the identity alone.
    #[test]
    fn a_flash_inside_one_shot_collects_a_same_shot_verdict() {
        let mut planes = static_sequence(W, H, 7, 40);
        planes[20] = flashed(&planes[20], 120);
        let frames: Vec<_> = planes.into_iter().map(|plane| (plane, label(0))).collect();

        let verdicts = collect(LabelIdentity, &frames);

        assert_eq!(verdicts.len(), 1, "{verdicts:?}");
        assert_eq!(
            verdicts[0].run,
            Run {
                first: 20,
                last: 20
            }
        );
        assert!(verdicts[0].same_shot);
    }

    /// A cut hidden in a flash: the identity separates before from after, and
    /// the run frame that already looks like `after` places the cut. This is
    /// the case the second fetch round exists for.
    #[test]
    fn a_cut_hidden_in_a_flash_collects_a_placed_shot_change() {
        let mut planes = static_sequence(W, H, 7, 20);
        planes.extend(static_sequence(W, H, 99, 20));
        planes[19] = flashed(&planes[19], 120);
        planes[20] = flashed(&planes[20], 120);
        let frames: Vec<_> = planes
            .into_iter()
            .enumerate()
            .map(|(index, plane)| (plane, label(u8::from(index >= 20))))
            .collect();

        let verdicts = collect(LabelIdentity, &frames);

        assert_eq!(verdicts.len(), 1, "{verdicts:?}");
        assert!(!verdicts[0].same_shot);
        assert_eq!(verdicts[0].cut_start, Some(20));
    }

    /// The veto abstains without an identity — the model's boundaries stand.
    #[test]
    fn no_identity_yields_no_verdicts() {
        let mut planes = static_sequence(W, H, 7, 40);
        planes[20] = flashed(&planes[20], 120);
        let frames: Vec<_> = planes.into_iter().map(|plane| (plane, label(0))).collect();
        assert_eq!(collect(NoIdentity, &frames), []);
    }

    /// Frames older than the longest possible run are dropped, so a 20-minute
    /// clip does not hold 36 000 snapshots.
    #[test]
    fn the_collector_retains_only_what_an_open_run_can_still_need() {
        let mut collector = TransientCollector::new(LabelIdentity);
        let plane = texture(64, 64, 3);
        for index in 0..200 {
            futures_executor::block_on(collector.push(&plane, label(0)));
            assert_eq!(collector.retained.len(), (index + 1).min(KEEP_FRAMES));
        }
        assert_eq!(collector.retained.len(), 31, "the current iOS look-behind");
    }
    struct YieldingIdentity(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);
    #[async_trait(?Send)]
    impl ShotIdentifying for YieldingIdentity {
        async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32> {
            let mut yielded = false;
            std::future::poll_fn(|cx| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
            self.0.borrow_mut().push(frame.rgb[0]);
            vec![f32::from(frame.rgb[0]) / 255.0]
        }
        fn same_shot_distance(&self) -> f32 {
            0.5
        }
        fn similar_shot_distance(&self) -> f32 {
            0.7
        }
    }
    fn original_one_frame_flash(change: bool) -> (Vec<Verdict>, Vec<u8>) {
        let calls = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut collector = TransientCollector::new(YieldingIdentity(calls.clone()));
        let mut planes = static_sequence(64, 36, 7, 40);
        planes[20] = flashed(&planes[20], 120);
        futures_executor::block_on(async {
            for (index, plane) in planes.iter().enumerate() {
                let value = if change && index >= 21 { 250 } else { 10 };
                collector
                    .push(plane, FrameSnapshot::packed(4, 4, vec![value; 4 * 4 * 3]))
                    .await;
            }
        });
        let labels = calls.borrow().clone();
        (collector.into_verdicts(), labels)
    }
    /// Kill: eagerly embed every frame or repeat the memoized before/after fetch.
    #[test]
    fn the_original_flash_uses_exactly_two_suspending_identity_calls() {
        let (verdicts, calls) = original_one_frame_flash(false);
        assert_eq!(verdicts.len(), 1);
        assert!(verdicts[0].same_shot);
        assert_eq!(
            verdicts[0].run,
            Run {
                first: 20,
                last: 20
            }
        );
        assert_eq!(calls, vec![10, 10]);
    }
    /// Kill: place a hidden change on the white frame, or fetch boundary identities twice.
    #[test]
    fn the_original_hidden_cut_is_21_after_three_suspending_identity_calls() {
        let (verdicts, calls) = original_one_frame_flash(true);
        assert_eq!(verdicts.len(), 1);
        assert!(!verdicts[0].same_shot);
        assert_eq!(
            verdicts[0].run,
            Run {
                first: 20,
                last: 20
            }
        );
        assert_eq!(verdicts[0].cut_start, Some(21));
        assert_eq!(calls, vec![10, 250, 10]);
    }
}
