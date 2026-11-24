//! The powerset decode against the reference export's own answer.
//!
//! `Tests/Fixtures/Pyannote/*/reference_*.bin` are onnxruntime fp32 outputs of
//! `onnx-community/pyannote-segmentation-3.0` — the same ONNX master this crate
//! ships — written by the iOS package's
//! `Scripts/speaker_segmenter/make_fixtures.py`. A golden taken from the
//! artifact under test could only prove it has not changed; one taken from an
//! independent run of the checkpoint proves the decode is right.
//!
//! The three cases light DIFFERENT rows of the powerset table on purpose: one
//! speaker, two speakers in turn, two speakers at once. A single-speaker read
//! only ever lights classes 0 and 1, so a table row in the wrong place would be
//! invisible on it.
//!
//! The model itself is exercised by `tests/ort_diarization.rs`; nothing here
//! needs a backend.

use ferment_engine_test_fixtures as fixtures;

use super::powerset::{self, CLASS_COUNT, SPEAKER_COUNT, TABLE};
use super::windows::WINDOW_FRAMES;

const CASES: [&str; 3] = ["one-speaker", "turn-taking", "overlap"];

fn probabilities(case: &str) -> Option<Vec<f32>> {
    fixtures::f32_le(&format!("Pyannote/{case}/reference_probabilities.bin"))
}

fn reference(case: &str, name: &str) -> Vec<u8> {
    fixtures::require_bytes(&format!("Pyannote/{case}/{name}.bin"))
}

#[test]
fn every_reference_frame_decodes_to_the_reference_speakers() {
    for case in CASES {
        let Some(probabilities) = probabilities(case) else {
            return;
        };
        let expected = reference(case, "reference_activity");
        let frames = probabilities.len() / CLASS_COUNT;

        let decoded = powerset::decode(&probabilities, frames);

        assert_eq!(frames, WINDOW_FRAMES, "{case}");
        assert_eq!(
            decoded.activity, expected,
            "{case}: decoded speakers differ"
        );
    }
}

#[test]
fn confidence_is_the_winning_class_probability() {
    for case in CASES {
        let Some(probabilities) = probabilities(case) else {
            return;
        };
        let classes = reference(case, "reference_classes");
        let frames = probabilities.len() / CLASS_COUNT;

        let decoded = powerset::decode(&probabilities, frames);

        for frame in 0..frames {
            let winner = probabilities[frame * CLASS_COUNT + usize::from(classes[frame])];
            assert!(
                (decoded.confidence[frame] - winner).abs() < 1e-6,
                "{case} frame {frame}: {} vs {winner}",
                decoded.confidence[frame]
            );
        }
    }
}

/// The marginals are what OVERLAPPING CHUNKS are averaged on, so they have to
/// be a distribution rather than a rescoring: a speaker's probability is the
/// sum of every class that names them.
#[test]
fn marginals_are_the_sum_of_every_class_naming_the_speaker() {
    let Some(probabilities) = probabilities("overlap") else {
        return;
    };
    let frames = probabilities.len() / CLASS_COUNT;

    let decoded = powerset::decode(&probabilities, frames);

    for frame in (0..frames).step_by(37) {
        let row = frame * CLASS_COUNT;
        let base = frame * SPEAKER_COUNT;
        for speaker in 0..SPEAKER_COUNT {
            let expected: f32 = (0..CLASS_COUNT)
                .filter(|klass| TABLE[*klass].contains(&speaker))
                .map(|klass| probabilities[row + klass])
                .sum();
            assert!((decoded.marginals[base + speaker] - expected).abs() < 1e-6);
            assert!(decoded.marginals[base + speaker] >= 0.0);
            assert!(decoded.marginals[base + speaker] <= 1.0001);
        }
    }
}

/// A speaker the argmax chose is always at least as likely as the class it was
/// chosen from — the decision and the marginal cannot contradict.
#[test]
fn the_chosen_speakers_are_never_less_likely_than_the_chosen_class() {
    let Some(probabilities) = probabilities("turn-taking") else {
        return;
    };
    let frames = probabilities.len() / CLASS_COUNT;

    let decoded = powerset::decode(&probabilities, frames);

    for frame in 0..frames {
        let base = frame * SPEAKER_COUNT;
        for speaker in 0..SPEAKER_COUNT {
            if decoded.activity[base + speaker] == 1 {
                assert!(decoded.marginals[base + speaker] >= decoded.confidence[frame] - 1e-6);
            }
        }
    }
}

#[test]
fn overlapped_is_exactly_the_frames_where_two_speakers_were_chosen() {
    let Some(probabilities) = probabilities("overlap") else {
        return;
    };
    let classes = reference("overlap", "reference_classes");
    let frames = probabilities.len() / CLASS_COUNT;

    let decoded = powerset::decode(&probabilities, frames);
    let shared = powerset::overlapped(&decoded.activity, frames);

    for frame in 0..frames {
        assert_eq!(
            shared[frame],
            TABLE[usize::from(classes[frame])].len() > 1,
            "frame {frame}, class {}",
            classes[frame]
        );
    }
    assert!(
        shared.contains(&true),
        "the overlap fixture must contain crosstalk at all"
    );
}
