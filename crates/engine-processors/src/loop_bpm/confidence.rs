//! Font & Serra (ISMIR 2016) loop tempo confidence.

use ferment_engine_audio::AudioFrames;

/// The longest loop, in beats, a tempo can be confirmed against: 32 bars of
/// 4/4, the paper's bound. A 32-bar loop cut exactly on the grid is included.
pub const MAX_LOOP_BEATS: usize = 128;

/// Beat count and score for one explicit length. A score for a silence-trimmed
/// length must never validate a different range during playback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopBpmMatch {
    pub beats: usize,
    pub confidence: f64,
}

/// Font & Serra's integer-beat length fit, without guessing cut points from
/// amplitude. The caller supplies the exact interval it intends to repeat.
pub fn match_loop_length(
    length: usize,
    sample_rate: u32,
    bpm: f64,
    max_beats: usize,
) -> Option<LoopBpmMatch> {
    let bpm = bpm.round();
    if !bpm.is_finite() || bpm <= 0. || length == 0 || sample_rate == 0 || max_beats == 0 {
        return None;
    }
    let beat = 60. * f64::from(sample_rate) / bpm;
    let beats = ((length as f64 / beat).round() as usize).clamp(1, max_beats);
    let distance = (length as f64 - (beat * beats as f64).round()).abs();
    Some(LoopBpmMatch {
        beats,
        confidence: (1. - distance / (beat * 0.5)).max(0.),
    })
}

/// How exactly `bpm`, rounded to a whole tempo, divides the loop into whole
/// beats: 1 when its length is a multiple of the beat, falling to 0 at half a
/// beat away. The length is also tried without leading and/or trailing
/// silence (envelope below 5 % of its peak) and the best match wins.
///
/// Only meaningful for audio cut as a loop with a steady tempo, and a file of
/// one beat matches any tempo whose beat it spans; bounding lengths is the
/// caller's policy. A file shorter than half a beat scores 0.
pub fn loop_bpm_confidence(frames: &AudioFrames, bpm: f64) -> f64 {
    loop_bpm_confidence_with_max_beats(frames, bpm, MAX_LOOP_BEATS)
}

/// The same confidence with an explicit caller-owned duration/beat bound.
/// The default API keeps its historical 128-beat policy.
pub fn loop_bpm_confidence_with_max_beats(frames: &AudioFrames, bpm: f64, max_beats: usize) -> f64 {
    let length = frames.samples.len();
    let (start, end) = audible_span(&frames.samples, frames.sample_rate);
    [length, length - start, end, end - start]
        .into_iter()
        .filter_map(|length| match_loop_length(length, frames.sample_rate, bpm, max_beats))
        .map(|matched| matched.confidence)
        .fold(0.0, f64::max)
}

/// First and last samples where a 10 ms attack/release envelope reaches 5 %
/// of its peak; the whole signal when it never rises above zero.
fn audible_span(samples: &[f32], sample_rate: u32) -> (usize, usize) {
    let smoothing = (-1.0 / (f64::from(sample_rate) * 0.010)).exp();
    let mut level = 0.0f64;
    let envelope: Vec<f64> = samples
        .iter()
        .map(|sample| {
            level = (1.0 - smoothing) * f64::from(sample.abs()) + smoothing * level;
            level
        })
        .collect();
    let threshold = envelope.iter().copied().fold(0.0, f64::max) * 0.05;
    let start = envelope.iter().position(|value| *value >= threshold);
    let end = envelope.iter().rposition(|value| *value >= threshold);
    (start.unwrap_or(0), end.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_bpm::test_signal::click_loop;

    const RATE: u32 = 44_100;

    fn frames(samples: Vec<f32>) -> AudioFrames {
        AudioFrames::new(samples, RATE)
    }

    // Fractional expectations are reference values computed outside this
    // crate on the same samples.

    #[test]
    fn explicit_duration_policy_can_confirm_more_than_128_beats() {
        let loop_180 = frames(click_loop(180.0, 180, RATE));
        assert_eq!(loop_bpm_confidence(&loop_180, 180.0), 0.0);
        assert_eq!(
            loop_bpm_confidence_with_max_beats(&loop_180, 180.0, 300),
            1.0
        );
        assert_eq!(loop_bpm_confidence_with_max_beats(&loop_180, 180.0, 0), 0.0);
    }

    #[test]
    fn a_loop_cut_on_the_grid_confirms_its_tempo_but_not_a_neighbour() {
        let loop_16 = frames(click_loop(120.0, 16, RATE));
        assert_eq!(loop_bpm_confidence(&loop_16, 120.0), 1.0);
        assert_eq!(loop_bpm_confidence(&loop_16, 120.4), 1.0);
        let neighbour = loop_bpm_confidence(&loop_16, 121.0);
        assert!((neighbour - 0.733855).abs() < 1e-5, "{neighbour}");
    }

    #[test]
    fn leading_silence_is_trimmed_before_matching() {
        let mut samples = vec![0.0; RATE as usize / 4];
        samples.extend(click_loop(120.0, 16, RATE));
        let confidence = loop_bpm_confidence(&frames(samples), 120.0);
        assert!((confidence - 0.999456).abs() < 2e-4, "{confidence}");
    }

    #[test]
    fn thirty_two_bars_are_the_longest_confirmable_loop() {
        let bars_32 = frames(click_loop(120.0, 128, RATE));
        let bars_34 = frames(click_loop(120.0, 136, RATE));
        assert_eq!(loop_bpm_confidence(&bars_32, 120.0), 1.0);
        assert_eq!(loop_bpm_confidence(&bars_34, 120.0), 0.0);
    }

    #[test]
    fn audio_shorter_than_half_a_beat_is_not_matched_against_zero_beats() {
        let click = frames(click_loop(120.0, 16, RATE)[..RATE as usize / 5].to_vec());
        assert_eq!(loop_bpm_confidence(&click, 120.0), 0.0);
    }

    #[test]
    fn no_tempo_or_no_audio_is_never_confident() {
        let loop_16 = frames(click_loop(120.0, 16, RATE));
        assert_eq!(loop_bpm_confidence(&loop_16, 0.0), 0.0);
        assert_eq!(loop_bpm_confidence(&frames(Vec::new()), 120.0), 0.0);
    }
}
