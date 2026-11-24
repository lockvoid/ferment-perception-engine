//! The 7-class POWERSET pyannote/segmentation-3.0 emits, turned into "who is
//! speaking in this frame".
//!
//! The network does not score three speakers independently. It scores the
//! seven ways up to three local speakers can be simultaneously active when at
//! most two of them overlap — silence, each speaker alone, each pair — as ONE
//! multiclass distribution. That is what makes overlap a first-class answer
//! rather than the accident of two independent sigmoids both firing, and it is
//! why the decode is an argmax over a lookup table and not a threshold.
//!
//! The labels are LOCAL to the chunk: speaker 1 in one 10 s window and speaker
//! 1 in the next are not the same person, and nothing here pretends otherwise.
//! Joining them is [`super::stitcher`]'s job, with embeddings.
//!
//! Ported from `Processors/Pyannote/PowersetDecoder.swift`, itself adapted from
//! FluidAudio (Apache-2.0), `Sources/FluidAudio/Diarizer/Segmentation/
//! SegmentationProcessor.swift` — the powerset table and the argmax decode. The
//! marginals and the per-frame confidence are ours.

pub const SPEAKER_COUNT: usize = 3;
pub const CLASS_COUNT: usize = 7;

/// Class index → the local speakers it says are active, in the order the
/// checkpoint's own `config.json` declares: `NO_SPEAKER, SPEAKER_1..3,
/// SPEAKERS_1_AND_2, _1_AND_3, _2_AND_3`.
pub const TABLE: [&[usize]; CLASS_COUNT] = [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

/// One chunk's frames, decoded three ways because three different readers want
/// three different things from the same numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// `frames × SPEAKER_COUNT`, 0 or 1 — the argmax decision. What turns and
    /// embedding windows are cut from: a window is only worth scoring if the
    /// model actually committed to one speaker over that stretch.
    pub activity: Vec<u8>,

    /// `frames × SPEAKER_COUNT` — p(this speaker is active), the sum of every
    /// class that contains them. What chunks are AVERAGED on where they
    /// overlap: averaging two hard 0/1 answers ties at exactly 0.5 on every
    /// frame the two chunks disagree about, and a tie needs a tiebreak that
    /// means nothing. Averaging the probabilities is the same arithmetic the
    /// model already did to produce them.
    pub marginals: Vec<f32>,

    /// `frames` — the winning class's probability. The model's own certainty
    /// about the frame, which is what a speaker's confidence is averaged from.
    pub confidence: Vec<f32>,
}

/// `probabilities` is `frames × CLASS_COUNT`, row-major, already softmaxed —
/// the graph bakes the normalisation in, so this function never sees a logit
/// and never has to guess at a scale.
///
/// # Panics
/// If `probabilities` is not exactly `frames * CLASS_COUNT` long.
#[must_use]
pub fn decode(probabilities: &[f32], frames: usize) -> Decoded {
    assert_eq!(
        probabilities.len(),
        frames * CLASS_COUNT,
        "powerset decode wants frames × {CLASS_COUNT} probabilities"
    );

    let mut activity = vec![0_u8; frames * SPEAKER_COUNT];
    let mut marginals = vec![0.0_f32; frames * SPEAKER_COUNT];
    let mut confidence = vec![0.0_f32; frames];

    let rows = probabilities.chunks_exact(CLASS_COUNT);
    let lit = activity.chunks_exact_mut(SPEAKER_COUNT);
    let sums = marginals.chunks_exact_mut(SPEAKER_COUNT);
    for (((row, lit), sums), confidence) in rows.zip(lit).zip(sums).zip(&mut confidence) {
        let mut best = 0;
        for klass in 1..CLASS_COUNT {
            if row[klass] > row[best] {
                best = klass;
            }
        }
        *confidence = row[best];

        for &speaker in TABLE[best] {
            lit[speaker] = 1;
        }
        for (probability, speakers) in row.iter().zip(TABLE) {
            for &speaker in speakers {
                sums[speaker] += probability;
            }
        }
    }

    Decoded {
        activity,
        marginals,
        confidence,
    }
}

/// Frames where the argmax named two speakers. Those frames are excluded from
/// every embedding window: two voices mixed into one 2 s window produce a
/// vector that belongs to neither, and one such window poisons the mean a whole
/// speaker is pooled from.
#[must_use]
pub fn overlapped(activity: &[u8], frames: usize) -> Vec<bool> {
    (0..frames)
        .map(|frame| {
            let base = frame * SPEAKER_COUNT;
            activity[base..base + SPEAKER_COUNT]
                .iter()
                .filter(|&&lit| lit == 1)
                .count()
                > 1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is the checkpoint's `config.json`, not a preference. A row in
    /// the wrong place decodes every frame of that class into the wrong person,
    /// which is invisible until real audio lights that row.
    #[test]
    fn table_is_the_powerset_of_three_with_at_most_two_at_once() {
        assert_eq!(TABLE.len(), CLASS_COUNT);
        assert_eq!(
            TABLE,
            [&[] as &[usize], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]]
        );
        let mut sets: Vec<Vec<usize>> = TABLE.iter().map(|row| row.to_vec()).collect();
        sets.sort_unstable();
        sets.dedup();
        assert_eq!(
            sets.len(),
            CLASS_COUNT,
            "no two classes may name the same set of speakers"
        );
        assert!(
            TABLE.iter().all(|row| row.len() <= 2),
            "this checkpoint scores at most two simultaneous speakers"
        );
    }

    /// The decoder is asked about a prefix of a padded chunk — the last chunk
    /// of a file reads past its end — so it must never look past `frames`.
    #[test]
    fn overlapped_reads_only_the_frames_it_was_given() {
        let frames = 4;
        let mut activity = vec![0_u8; 8 * SPEAKER_COUNT];
        activity[7 * SPEAKER_COUNT] = 1;
        activity[7 * SPEAKER_COUNT + 1] = 1;

        let shared = overlapped(&activity, frames);
        assert_eq!(shared.len(), frames);
        assert!(!shared.contains(&true));
    }

    #[test]
    fn a_two_speaker_class_lights_both_of_its_speakers() {
        // One frame, class 4 (SPEAKERS_1_AND_2) winning.
        let probabilities = [0.05, 0.05, 0.05, 0.05, 0.7, 0.05, 0.05];
        let decoded = decode(&probabilities, 1);

        assert_eq!(decoded.activity, vec![1, 1, 0]);
        assert!((decoded.confidence[0] - 0.7).abs() < 1e-6);
        assert!(overlapped(&decoded.activity, 1)[0]);
        // marginals[0] = classes 1, 4, 5; marginals[1] = 2, 4, 6; [2] = 3, 5, 6.
        assert!((decoded.marginals[0] - 0.8).abs() < 1e-6);
        assert!((decoded.marginals[1] - 0.8).abs() < 1e-6);
        assert!((decoded.marginals[2] - 0.15).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "powerset decode wants")]
    fn a_ragged_probability_block_is_rejected() {
        let _ = decode(&[0.0; 6], 1);
    }
}
