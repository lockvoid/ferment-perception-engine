//! Frames → replies. One voice's per-frame decision becomes the stretches of
//! the file it was heard in, in seconds.
//!
//! Two kinds of smoothing, and they are not the same rule twice:
//!
//!   - a GAP shorter than [`MINIMUM_GAP_SECONDS`] is closed. Continuous speech
//!     is not continuously voiced — a stop consonant, a breath between clauses
//!     — and a decoder that splits on every one of them reports a hundred
//!     replies where a person spoke once.
//!   - a RUN shorter than [`MINIMUM_TURN_SECONDS`] is dropped. Below a fifth of
//!     a second there is no reply to hear, and what is there is usually the
//!     model twitching at a chunk seam.
//!
//! Both thresholds are pyannote's own `min_duration_off` / `min_duration_on`
//! defaults for this checkpoint. Order matters and is not symmetric: gaps close
//! first, so a real reply broken by one breath is healed rather than deleted
//! twice.
//!
//! A reply is uniformly overlapped or uniformly clean, never half of each —
//! runs are split where the flag changes. That is what lets the pooling read
//! "the clean replies" without re-deriving the boundary, and the flag is
//! smoothed on its own before the split so a single frame of crosstalk cannot
//! shatter a long reply into six.
//!
//! Ported from `Processors/Pyannote/TurnAssembler.swift`.

use ferment_engine_core::SpeakerTurn;

use super::windows;

pub const MINIMUM_TURN_SECONDS: f64 = 0.20;
pub const MINIMUM_GAP_SECONDS: f64 = 0.20;

/// `active` and `overlapped` are one speaker's decision over the frames
/// `frame_offset .. frame_offset + active.len()` of the file's own grid.
///
/// # Panics
/// If `active` and `overlapped` differ in length.
#[must_use]
pub fn turns(active: &[bool], overlapped: &[bool], frame_offset: usize) -> Vec<SpeakerTurn> {
    assert_eq!(
        active.len(),
        overlapped.len(),
        "a speaker's activity and crosstalk cover the same frames"
    );
    if active.is_empty() {
        return Vec::new();
    }

    let minimum_turn = frames_for(MINIMUM_TURN_SECONDS);
    let mut speaking = closing_gaps(active, frames_for(MINIMUM_GAP_SECONDS));
    speaking = dropping_runs(&speaking, minimum_turn);

    // Crosstalk only counts where this voice is speaking at all, and only where
    // it lasted long enough to be worth carving a reply around.
    let shared: Vec<bool> = overlapped
        .iter()
        .zip(&speaking)
        .map(|(&shared, &speaking)| shared && speaking)
        .collect();
    let shared = dropping_runs(&shared, minimum_turn);

    let mut turns = Vec::new();
    let mut index = 0;
    while index < speaking.len() {
        if !speaking[index] {
            index += 1;
            continue;
        }
        let flag = shared[index];
        let mut end = index;
        while end + 1 < speaking.len() && speaking[end + 1] && shared[end + 1] == flag {
            end += 1;
        }
        turns.push(SpeakerTurn {
            start_time: windows::start_time(frame_offset + index),
            end_time: windows::end_time(frame_offset + end),
            overlapped: flag,
        });
        index = end + 1;
    }
    turns
}

/// Total seconds of speech, crosstalk included — what a voice is ranked and
/// thresholded by.
#[must_use]
pub fn duration(turns: &[SpeakerTurn]) -> f64 {
    turns.iter().map(SpeakerTurn::duration).sum()
}

/// Seconds the two sets of replies were both heard in. The stitcher's fallback
/// for a chunk-local voice too short to fingerprint: two chunks that describe
/// the same instants are describing the same person.
#[must_use]
pub fn overlap_duration(a: &[SpeakerTurn], b: &[SpeakerTurn]) -> f64 {
    let mut total = 0.0;
    for left in a {
        for right in b {
            total += (left.end_time.min(right.end_time) - left.start_time.max(right.start_time))
                .max(0.0);
        }
    }
    total
}

/// Seconds as a whole number of frames, never zero.
#[must_use]
pub fn frames_for(seconds: f64) -> usize {
    ((seconds / windows::frame_seconds()).round() as usize).max(1)
}

/// Runs of `false` shorter than `limit` become `true`. The leading and trailing
/// runs are not gaps between anything — closing them would invent speech before
/// the first word.
#[must_use]
pub fn closing_gaps(mask: &[bool], limit: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    let mut index = 0;
    while index < out.len() {
        if out[index] {
            index += 1;
            continue;
        }
        let mut end = index;
        while end + 1 < out.len() && !out[end + 1] {
            end += 1;
        }
        if index > 0 && end + 1 < out.len() && end - index + 1 < limit {
            out[index..=end].fill(true);
        }
        index = end + 1;
    }
    out
}

/// Runs of `true` shorter than `limit` become `false`.
#[must_use]
pub fn dropping_runs(mask: &[bool], limit: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    let mut index = 0;
    while index < out.len() {
        if !out[index] {
            index += 1;
            continue;
        }
        let mut end = index;
        while end + 1 < out.len() && out[end + 1] {
            end += 1;
        }
        if end - index + 1 < limit {
            out[index..=end].fill(false);
        }
        index = end + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mask from run lengths in FRAMES: `[10, 5, 10]` is 10 on, 5 off, 10 on.
    fn mask(runs: &[usize], starting_on: bool) -> Vec<bool> {
        let mut out = Vec::new();
        let mut on = starting_on;
        for &run in runs {
            out.extend(std::iter::repeat_n(on, run));
            on = !on;
        }
        out
    }

    fn quiet(count: usize) -> Vec<bool> {
        vec![false; count]
    }

    fn minimum_frames() -> usize {
        frames_for(MINIMUM_TURN_SECONDS)
    }

    /// Continuous speech is not continuously voiced. A stop consonant or a
    /// breath between clauses puts a handful of quiet frames inside one reply,
    /// and a decoder that splits on each of them reports a hundred replies
    /// where a person spoke once.
    #[test]
    fn a_breath_inside_a_reply_does_not_split_it() {
        let minimum = minimum_frames();
        let long = minimum * 4;
        let turns = turns(
            &mask(&[long, minimum - 2, long], true),
            &quiet(long * 2 + minimum - 2),
            0,
        );

        assert_eq!(turns.len(), 1);
        assert!(turns[0].start_time.abs() < 1e-9);
    }

    /// A gap long enough to be a pause is a pause.
    #[test]
    fn a_pause_splits_the_reply() {
        let minimum = minimum_frames();
        let long = minimum * 4;
        let turns = turns(
            &mask(&[long, minimum * 2, long], true),
            &quiet(long * 2 + minimum * 2),
            0,
        );

        assert_eq!(turns.len(), 2);
        assert!(turns[0].end_time < turns[1].start_time);
    }

    /// Below a fifth of a second there is no reply to hear.
    #[test]
    fn a_twitch_is_not_a_reply() {
        let minimum = minimum_frames();
        let turns = turns(
            &mask(&[minimum * 3, minimum - 1], false),
            &quiet(minimum * 4 - 1),
            0,
        );

        assert!(turns.is_empty());
    }

    /// The leading and trailing quiet are not gaps BETWEEN anything. Closing
    /// them would invent speech before the first word — the one direction a
    /// smoothing rule must never move.
    #[test]
    fn the_quiet_before_the_first_word_is_never_closed() {
        let minimum = minimum_frames();
        let long = minimum * 4;
        let turns = turns(
            &mask(&[minimum - 2, long], false),
            &quiet(long + minimum - 2),
            0,
        );

        assert_eq!(turns.len(), 1);
        assert!((turns[0].start_time - windows::start_time(minimum - 2)).abs() < 1e-9);
    }

    /// A reply's audio runs to the END of its last frame's receptive field —
    /// 61.9 ms, not the 16.9 ms step. Truncating to the step would clip 45 ms
    /// off every reply, which is a syllable.
    #[test]
    fn a_reply_runs_to_the_end_of_its_last_frames_audio() {
        let long = minimum_frames() * 3;
        let turns = turns(&mask(&[long], true), &quiet(long), 0);

        assert_eq!(turns.len(), 1);
        assert!((turns[0].end_time - windows::end_time(long - 1)).abs() < 1e-9);
    }

    #[test]
    fn the_frame_offset_places_replies_in_the_files_own_time() {
        let long = minimum_frames() * 3;
        let offset = 1_000;
        let turns = turns(&mask(&[long], true), &quiet(long), offset);

        assert!((turns[0].start_time - windows::start_time(offset)).abs() < 1e-9);
    }

    /// A reply is uniformly overlapped or uniformly clean, never half of each.
    #[test]
    fn crosstalk_splits_a_reply_rather_than_flagging_the_whole_of_it() {
        let long = minimum_frames() * 3;
        let turns = turns(&mask(&[long * 2], true), &mask(&[long, long], false), 0);

        assert_eq!(turns.len(), 2);
        assert!(!turns[0].overlapped);
        assert!(turns[1].overlapped);
        assert!((turns[0].end_time - windows::end_time(long - 1)).abs() < 1e-9);
        assert!((turns[1].start_time - windows::start_time(long)).abs() < 1e-9);
    }

    /// One frame of crosstalk must not shatter a long reply into three.
    #[test]
    fn a_single_frame_of_crosstalk_is_not_a_crosstalk_reply() {
        let long = minimum_frames() * 6;
        let mut shared = quiet(long);
        shared[long / 2] = true;
        let turns = turns(&mask(&[long], true), &shared, 0);

        assert_eq!(turns.len(), 1);
        assert!(!turns[0].overlapped);
    }

    /// Crosstalk is only crosstalk where this voice is speaking at all.
    #[test]
    fn crosstalk_outside_the_reply_is_ignored() {
        let long = minimum_frames() * 3;
        let turns = turns(&mask(&[long, long], true), &mask(&[long, long], false), 0);

        assert_eq!(turns.len(), 1);
        assert!(!turns[0].overlapped);
    }

    #[test]
    fn overlap_duration_is_the_seconds_two_voices_share() {
        let turn = |start, end| SpeakerTurn {
            start_time: start,
            end_time: end,
            overlapped: false,
        };
        let a = [turn(0.0, 10.0)];
        let b = [turn(5.0, 8.0), turn(9.0, 20.0)];

        assert!((overlap_duration(&a, &b) - 4.0).abs() < 1e-9);
        assert!(overlap_duration(&a, &[]).abs() < 1e-12);
        assert!(
            overlap_duration(&a, &[turn(20.0, 30.0)]).abs() < 1e-12,
            "disjoint replies share nothing, never a negative"
        );
    }

    #[test]
    fn duration_is_total_speech_crosstalk_included() {
        let total = duration(&[
            SpeakerTurn {
                start_time: 0.0,
                end_time: 2.0,
                overlapped: false,
            },
            SpeakerTurn {
                start_time: 3.0,
                end_time: 4.5,
                overlapped: true,
            },
        ]);
        assert!((total - 3.5).abs() < 1e-9);
    }

    #[test]
    fn empty_frames_are_empty_replies() {
        assert!(turns(&[], &[], 0).is_empty());
        assert!(turns(&quiet(500), &quiet(500), 0).is_empty());
    }
}
