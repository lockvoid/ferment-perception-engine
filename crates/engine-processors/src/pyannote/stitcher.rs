//! Chunk-local speakers → the element's voices.
//!
//! The segmenter's labels only mean anything inside one 10 s window: "speaker
//! 1" in the third chunk and "speaker 1" in the fourth are two different facts.
//! Joining them is not a bookkeeping step — it is a MEASUREMENT, and the only
//! instrument that can make it is the embedder. That is why segmentation and
//! embedding are one capability and not two.
//!
//! Two joins happen here, in this order and for different reasons:
//!
//!   1. by VECTOR — agglomerative clustering, average linkage on cosine over
//!      the probe vector each chunk-local speaker was fingerprinted with.
//!      Average linkage rather than single: single linkage chains, and one
//!      ambiguous probe sitting between two people would weld them.
//!   2. by TIME — a chunk-local speaker whose clean speech never filled a whole
//!      2.005 s encoder window has no vector to be joined by. Chunks overlap by
//!      half a window, so the same moment is almost always also described by a
//!      neighbour that DID fingerprint it; the fallback joins it to whichever
//!      voice shares the most seconds with it. A stretch no voice shares at all
//!      becomes a voice with no vector — an honest "we heard someone here and
//!      cannot say who".
//!
//! Once the voices are known, the chunks are AVERAGED rather than unioned: each
//! voice's per-frame probability is the mean over the chunks that saw that
//! frame. This is the whole reason the chunks overlap. A model twitching at a
//! window seam — where SincNet runs out of context and the LSTM has not warmed
//! up — is outvoted by the neighbour that saw the same moment mid-window, where
//! it is at its best.
//!
//! Ported from `Processors/Pyannote/LocalSpeakerStitcher.swift`, itself adapted
//! from FluidAudio (Apache-2.0), `Sources/FluidAudio/Diarizer/Offline/` — the
//! agglomerative-clustering and cross-chunk-reconstruction shape. Their
//! fastcluster FFI, PLDA and VBx are not taken: a handful of chunk-local
//! speakers does not need a C linkage kernel, and scoring stays plain cosine so
//! one threshold means one thing everywhere.

use ferment_engine_core::SpeakerTurn;

use super::powerset::SPEAKER_COUNT;
use super::turns;
use super::windows::Chunk;

/// One chunk's account of one local speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub chunk: usize,
    pub speaker: usize,
    pub turns: Vec<SpeakerTurn>,
    /// Mean of the unit vectors of this speaker's clean encoder windows inside
    /// this chunk, renormalised. EMPTY when no whole window fit.
    pub probe: Vec<f32>,
}

/// One voice of the element.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    pub turns: Vec<SpeakerTurn>,
    /// The model's own certainty, over the frames this voice owns.
    pub confidence: f64,
}

/// Agglomerative clustering over the probes, average linkage, stopping when the
/// closest two groups are further apart than `threshold`.
///
/// Returns groups of indices into `candidates`, ordered by when the voice is
/// first heard — so the same audio always numbers its voices the same way,
/// whatever order the chunks were measured in.
#[must_use]
pub fn cluster(candidates: &[Candidate], threshold: f32) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !candidate.probe.is_empty())
        .map(|(index, _)| vec![index])
        .collect();

    while groups.len() > 1 {
        let mut best: Option<(usize, usize, f32)> = None;
        for left in 0..groups.len() - 1 {
            for right in left + 1..groups.len() {
                let score = linkage(&groups[left], &groups[right], candidates);
                // Strictly greater: on a tie the first pair found wins, which
                // is what makes the merge order a function of the input alone.
                if score > best.map_or(f32::NEG_INFINITY, |(_, _, score)| score) {
                    best = Some((left, right, score));
                }
            }
        }
        let Some((left, right, score)) = best else {
            break;
        };
        if score < threshold {
            break;
        }
        let merged = groups.remove(right);
        groups[left].extend(merged);
    }

    attach_vectorless(candidates, &mut groups);
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_by(|a, b| {
        start(a, candidates)
            .partial_cmp(&start(b, candidates))
            .expect("turn start times are finite")
    });
    groups
}

/// Mean cosine over every cross pair — the average linkage.
fn linkage(left: &[usize], right: &[usize], candidates: &[Candidate]) -> f32 {
    let mut total = 0.0;
    for &a in left {
        for &b in right {
            total += cosine(&candidates[a].probe, &candidates[b].probe);
        }
    }
    total / (left.len() * right.len()) as f32
}

/// Chunk-local speakers with no probe, joined by shared seconds. Walked in time
/// order so a run of them along a quiet stretch chains into ONE voice rather
/// than one voice each: a group minted by an earlier candidate is a candidate
/// group for the next.
fn attach_vectorless(candidates: &[Candidate], groups: &mut Vec<Vec<usize>>) {
    let mut orphans: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.probe.is_empty())
        .map(|(index, _)| index)
        .collect();
    orphans.sort_by(|&a, &b| {
        start(&[a], candidates)
            .partial_cmp(&start(&[b], candidates))
            .expect("turn start times are finite")
    });

    for orphan in orphans {
        let mut best: Option<(usize, f64)> = None;
        for (index, group) in groups.iter().enumerate() {
            let shared: f64 = group
                .iter()
                .map(|&member| {
                    turns::overlap_duration(&candidates[orphan].turns, &candidates[member].turns)
                })
                .sum();
            // Strictly greater than zero: sharing nothing with anyone is a
            // voice of its own, not a nearest neighbour.
            if shared > best.map_or(0.0, |(_, shared)| shared) {
                best = Some((index, shared));
            }
        }
        match best {
            Some((group, _)) => groups[group].push(orphan),
            None => groups.push(vec![orphan]),
        }
    }
}

fn start(group: &[usize], candidates: &[Candidate]) -> f64 {
    group
        .iter()
        .filter_map(|&index| candidates[index].turns.first().map(|turn| turn.start_time))
        .fold(f64::INFINITY, f64::min)
}

/// Every voice's replies, from the mean of what the chunks that saw each frame
/// said about it.
///
/// `marginals[chunk]` is that chunk's `frames × SPEAKER_COUNT` probability
/// block, as the decoder produced it.
#[must_use]
pub fn aggregate(
    candidates: &[Candidate],
    groups: &[Vec<usize>],
    chunks: &[Chunk],
    marginals: &[Vec<f32>],
    confidences: &[Vec<f32>],
    frame_count: usize,
) -> Vec<Voice> {
    if frame_count == 0 {
        return Vec::new();
    }

    let mut coverage = vec![0.0_f32; frame_count];
    for chunk in chunks {
        for seen in &mut coverage[chunk.frame_offset..chunk.frame_offset + chunk.frames] {
            *seen += 1.0;
        }
    }

    groups
        .iter()
        .map(|group| {
            let mut probability = vec![0.0_f32; frame_count];
            let mut certainty = vec![0.0_f32; frame_count];
            for &index in group {
                let candidate = &candidates[index];
                let chunk = &chunks[candidate.chunk];
                let block = &marginals[candidate.chunk];
                let confidence = &confidences[candidate.chunk];
                for frame in 0..chunk.frames {
                    let global = chunk.frame_offset + frame;
                    probability[global] += block[frame * SPEAKER_COUNT + candidate.speaker];
                    certainty[global] = certainty[global].max(confidence[frame]);
                }
            }

            // A frame with no chunk-local speaker for this voice contributed a
            // zero, which is exactly the vote it should cast.
            let active: Vec<bool> = (0..frame_count)
                .map(|frame| probability[frame] / coverage[frame] >= 0.5)
                .collect();
            let shared =
                shared_frames(candidates, group, chunks, marginals, &coverage, frame_count);
            let turns = turns::turns(&active, &shared, 0);

            let mut weight = 0.0;
            let mut total = 0.0;
            for frame in 0..frame_count {
                if active[frame] {
                    weight += 1.0;
                    total += f64::from(certainty[frame]);
                }
            }
            Voice {
                turns,
                confidence: if weight > 0.0 { total / weight } else { 0.0 },
            }
        })
        .collect()
}

/// Frames where SOME OTHER local speaker of the same chunk was also speaking.
/// Crosstalk is a per-chunk observation, so it is read off the chunks rather
/// than off the aggregated voices — two voices that happen to be active in the
/// same frame after averaging may simply be a seam disagreement, which is not
/// the same claim as "the model saw two people at once".
fn shared_frames(
    candidates: &[Candidate],
    group: &[usize],
    chunks: &[Chunk],
    marginals: &[Vec<f32>],
    coverage: &[f32],
    frame_count: usize,
) -> Vec<bool> {
    let mut votes = vec![0.0_f32; frame_count];
    for &index in group {
        let candidate = &candidates[index];
        let chunk = &chunks[candidate.chunk];
        let block = &marginals[candidate.chunk];
        for frame in 0..chunk.frames {
            let mut others = 0.0_f32;
            for speaker in 0..SPEAKER_COUNT {
                if speaker != candidate.speaker {
                    others = others.max(block[frame * SPEAKER_COUNT + speaker]);
                }
            }
            votes[chunk.frame_offset + frame] += others;
        }
    }
    (0..frame_count)
        .map(|frame| votes[frame] / coverage[frame] >= 0.5)
        .collect()
}

/// Mean of unit vectors, renormalised — the aggregation the embedder documents
/// for a span it had to tile. Empty in, empty out: a voice that filled no whole
/// window has no direction, and inventing one would be worse than saying so.
#[must_use]
pub fn pooled(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(width) = vectors.first().map(Vec::len).filter(|width| *width > 0) else {
        return Vec::new();
    };
    let mut sum = vec![0.0_f32; width];
    for vector in vectors.iter().filter(|vector| vector.len() == width) {
        for (accumulated, unit) in sum.iter_mut().zip(normalized(vector)) {
            *accumulated += unit;
        }
    }
    normalized(&sum)
}

#[must_use]
pub fn normalized(vector: &[f32]) -> Vec<f32> {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length > 0.0 {
        vector.iter().map(|value| value / length).collect()
    } else {
        vector.to_vec()
    }
}

/// Cosine similarity, or `-1` for anything that cannot honestly be compared —
/// mismatched widths, an empty vector, a zero vector.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    let mut dot = 0.0;
    let mut length_a = 0.0;
    let mut length_b = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        dot += left * right;
        length_a += left * left;
        length_b += right * right;
    }
    if length_a > 0.0 && length_b > 0.0 {
        dot / (length_a.sqrt() * length_b.sqrt())
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::super::windows;
    use super::*;

    fn turn(start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            start_time: start,
            end_time: end,
            overlapped: false,
        }
    }

    fn candidate(
        chunk: usize,
        speaker: usize,
        turns: Vec<SpeakerTurn>,
        probe: &[f32],
    ) -> Candidate {
        Candidate {
            chunk,
            speaker,
            turns,
            probe: probe.to_vec(),
        }
    }

    #[test]
    fn the_same_voice_in_two_chunks_is_one_voice() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 5.0)], &[1.0, 0.0, 0.0]),
                candidate(1, 2, vec![turn(5.0, 10.0)], &[0.99, 0.14, 0.0]),
            ],
            0.45,
        );

        assert_eq!(groups, vec![vec![0, 1]]);
    }

    #[test]
    fn two_voices_in_one_chunk_stay_two_voices() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 5.0)], &[1.0, 0.0, 0.0]),
                candidate(0, 1, vec![turn(0.0, 5.0)], &[0.0, 1.0, 0.0]),
            ],
            0.45,
        );

        assert_eq!(groups.len(), 2);
    }

    /// Average linkage, not single. With A·B = 0.6, B·C = 0.8 and A·C = 0 at a
    /// threshold of 0.5, single linkage welds all three through B — one
    /// ambiguous probe sitting between two people is exactly the situation a
    /// diarizer meets, and chaining through it merges two speakers into one
    /// unrecoverably.
    #[test]
    fn an_ambiguous_probe_does_not_chain_two_voices_together() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 5.0)], &[1.0, 0.0, 0.0]),
                candidate(1, 0, vec![turn(5.0, 10.0)], &[0.6, 0.8, 0.0]),
                candidate(2, 0, vec![turn(10.0, 15.0)], &[0.0, 1.0, 0.0]),
            ],
            0.5,
        );

        assert_eq!(
            groups,
            vec![vec![0], vec![1, 2]],
            "single linkage would chain A-B-C into one voice"
        );
    }

    /// Voices come back in the order they are first heard, whatever order the
    /// chunks were measured in — otherwise the same audio would number its
    /// speakers differently between two runs.
    #[test]
    fn voices_come_back_in_the_order_they_are_first_heard() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(30.0, 35.0)], &[0.0, 1.0, 0.0]),
                candidate(1, 0, vec![turn(0.0, 5.0)], &[1.0, 0.0, 0.0]),
            ],
            0.45,
        );

        assert_eq!(groups, vec![vec![1], vec![0]]);
    }

    /// A chunk-local speaker whose clean speech never filled a whole encoder
    /// window has no vector to be joined by. Chunks overlap by half a window,
    /// so the same moment is almost always described by a neighbour that DID
    /// fingerprint it.
    #[test]
    fn a_speaker_too_short_to_fingerprint_joins_the_voice_it_shares_time_with() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 8.0)], &[1.0, 0.0, 0.0]),
                candidate(1, 0, vec![turn(20.0, 28.0)], &[0.0, 1.0, 0.0]),
                candidate(2, 0, vec![turn(21.0, 22.0)], &[]),
            ],
            0.45,
        );

        assert_eq!(groups.len(), 2);
        assert!(
            groups
                .iter()
                .any(|group| group.contains(&1) && group.contains(&2)),
            "the vectorless sliver belongs to the voice it overlaps"
        );
    }

    /// Speech no fingerprinted voice shares a second with is still speech. It
    /// becomes a voice with no vector — an honest "someone spoke here and we
    /// cannot say who" — rather than being folded into whoever was nearest.
    #[test]
    fn speech_no_voice_shares_becomes_a_voice_with_no_vector() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 8.0)], &[1.0, 0.0, 0.0]),
                candidate(3, 0, vec![turn(40.0, 41.0)], &[]),
            ],
            0.45,
        );

        assert_eq!(groups, vec![vec![0], vec![1]]);
    }

    /// A run of slivers along a quiet stretch is ONE voice, not one voice each
    /// — which is what walking them in time order buys.
    #[test]
    fn a_chain_of_slivers_is_one_voice() {
        let groups = cluster(
            &[
                candidate(0, 0, vec![turn(0.0, 1.0)], &[]),
                candidate(1, 0, vec![turn(0.5, 1.5)], &[]),
                candidate(2, 0, vec![turn(1.2, 2.0)], &[]),
            ],
            0.45,
        );

        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    struct Seam {
        chunks: Vec<Chunk>,
        candidates: Vec<Candidate>,
        marginals: Vec<Vec<f32>>,
        confidences: Vec<Vec<f32>>,
        frame_count: usize,
    }

    /// Two overlapping chunks describing one person. Chunk 0 also twitches at
    /// its own tail — the seam, where SincNet has run out of context — over
    /// frames chunk 1 sees mid-window and calls silence.
    fn seam() -> Seam {
        let width = windows::WINDOW_FRAMES;
        let step = windows::STEP_FRAMES;

        // Both chunks agree the person speaks over global frames 0..400; only
        // chunk 0 claims global 560..589, which is its last 29 frames.
        let mut first = vec![0.0_f32; width * SPEAKER_COUNT];
        for frame in (0..400).chain(560..width) {
            first[frame * SPEAKER_COUNT] = 0.9;
        }
        let mut second = vec![0.0_f32; width * SPEAKER_COUNT];
        for frame in 0..400 - step {
            second[frame * SPEAKER_COUNT + 1] = 0.9;
        }

        Seam {
            chunks: vec![
                Chunk {
                    sample_offset: 0,
                    frame_offset: 0,
                    frames: width,
                },
                Chunk {
                    sample_offset: step * windows::FRAME_STRIDE,
                    frame_offset: step,
                    frames: width,
                },
            ],
            candidates: vec![
                candidate(
                    0,
                    0,
                    vec![turn(windows::start_time(0), windows::end_time(399))],
                    &[1.0, 0.0, 0.0],
                ),
                candidate(
                    1,
                    1,
                    vec![turn(windows::start_time(step), windows::end_time(399))],
                    &[1.0, 0.0, 0.0],
                ),
            ],
            marginals: vec![first, second],
            confidences: vec![vec![0.8; width], vec![0.9; width]],
            frame_count: step + width,
        }
    }

    fn aggregate_seam(seam: &Seam) -> Vec<Voice> {
        let groups = cluster(&seam.candidates, 0.45);
        aggregate(
            &seam.candidates,
            &groups,
            &seam.chunks,
            &seam.marginals,
            &seam.confidences,
            seam.frame_count,
        )
    }

    #[test]
    fn a_seam_twitch_is_outvoted_by_the_chunk_that_saw_it_mid_window() {
        let seam = seam();
        let voices = aggregate_seam(&seam);

        assert_eq!(voices.len(), 1, "one person, two chunks");
        assert_eq!(
            voices[0].turns.len(),
            1,
            "the tail twitch must not become a second reply"
        );
        assert!((voices[0].turns[0].start_time - windows::start_time(0)).abs() < 1e-9);
        assert!((voices[0].turns[0].end_time - windows::end_time(399)).abs() < 0.05);
        assert!(
            voices[0].turns[0].end_time < windows::start_time(560),
            "nothing may survive from the seam"
        );
    }

    /// The frames only ONE chunk ever saw — the head of the file and its tail —
    /// are that chunk's word against nothing, so they stand. Averaging must
    /// divide by the chunks that were there, not by the chunks that exist.
    #[test]
    fn frames_only_one_chunk_saw_are_still_heard() {
        let voices = aggregate_seam(&seam());

        assert!(
            voices[0].turns[0].start_time.abs() < 1e-9,
            "the file's first frames are covered by chunk 0 alone"
        );
    }

    #[test]
    fn confidence_is_the_models_certainty_over_the_frames_the_voice_owns() {
        let voices = aggregate_seam(&seam());

        assert!(voices[0].confidence >= 0.8);
        assert!(voices[0].confidence <= 0.9);
    }

    /// A group whose every frame was outvoted averages away to no replies. It
    /// is not a voice then — it is what the averaging removed.
    #[test]
    fn a_group_that_averages_away_has_no_replies() {
        let width = windows::WINDOW_FRAMES;
        let step = windows::STEP_FRAMES;

        // Chunk 0 claims its last 29 frames; chunk 1 sees the same frames
        // mid-window and says silence. Nothing else is claimed at all.
        let mut first = vec![0.0_f32; width * SPEAKER_COUNT];
        for frame in 560..width {
            first[frame * SPEAKER_COUNT] = 0.9;
        }
        let second = vec![0.0_f32; width * SPEAKER_COUNT];

        let candidates = vec![candidate(
            0,
            0,
            vec![turn(windows::start_time(560), windows::end_time(width - 1))],
            &[1.0, 0.0, 0.0],
        )];
        let voices = aggregate(
            &candidates,
            &[vec![0]],
            &[
                Chunk {
                    sample_offset: 0,
                    frame_offset: 0,
                    frames: width,
                },
                Chunk {
                    sample_offset: step * windows::FRAME_STRIDE,
                    frame_offset: step,
                    frames: width,
                },
            ],
            &[first, second],
            &[vec![0.8; width], vec![0.9; width]],
            step + width,
        );

        assert_eq!(voices.len(), 1, "aggregate reports what it was asked about");
        assert!(
            voices[0].turns.is_empty(),
            "and it is a voice with nothing to say"
        );
    }

    #[test]
    fn no_frames_is_no_voices() {
        assert!(aggregate(&[], &[], &[], &[], &[], 0).is_empty());
    }

    #[test]
    fn pooling_averages_unit_vectors_and_renormalises() {
        let pooled = pooled(&[vec![3.0, 0.0, 0.0], vec![0.0, 4.0, 0.0]]);

        assert_eq!(pooled.len(), 3);
        assert!(
            (cosine(&pooled, &[1.0, 1.0, 0.0]) - 1.0).abs() < 1e-6,
            "length must not weigh a window more than another"
        );
        assert!((pooled.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn no_windows_is_no_vector() {
        assert!(pooled(&[]).is_empty());
    }

    #[test]
    fn cosine_refuses_to_compare_widths_it_cannot() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]), -1.0);
        assert_eq!(cosine(&[], &[]), -1.0);
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]), -1.0);
    }
}
