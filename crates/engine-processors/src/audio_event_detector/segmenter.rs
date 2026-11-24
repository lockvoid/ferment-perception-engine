use ferment_engine_core::{
    AudioEvent, AudioEventAnalysis, AudioEventClassStats, AudioEventDetectionResult,
    AudioEventFrameProbabilities, AudioEventSegment, AudioEventSummary,
};

use super::config::{
    EXTEND_SPEECH_FRAME, FRAME_LENGTH_SAMPLES, MAX_EVENT_FRAME, MERGE_SILENCE_FRAME,
    MIN_EVENT_FRAME, MIN_SILENCE_FRAME, MUSIC_THRESHOLD, SAMPLE_RATE, SINGING_THRESHOLD,
    SMOOTH_WINDOW_SIZE, SPEECH_THRESHOLD,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmenterConfig {
    pub threshold: f32,
    pub smooth_window_size: usize,
    pub min_event_frame: usize,
    pub max_event_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
    pub min_confidence: f32,
}

impl SegmenterConfig {
    #[must_use]
    pub const fn new(threshold: f32) -> Self {
        Self {
            threshold,
            smooth_window_size: SMOOTH_WINDOW_SIZE,
            min_event_frame: MIN_EVENT_FRAME,
            max_event_frame: MAX_EVENT_FRAME,
            min_silence_frame: MIN_SILENCE_FRAME,
            merge_silence_frame: MERGE_SILENCE_FRAME,
            extend_speech_frame: EXTEND_SPEECH_FRAME,
            min_confidence: 0.0,
        }
    }

    pub const SPEECH: Self = Self::new(SPEECH_THRESHOLD);
    pub const SINGING: Self = Self::new(SINGING_THRESHOLD);
    pub const MUSIC: Self = Self::new(MUSIC_THRESHOLD);
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioEventSegments {
    pub speech: Vec<AudioEventSegment>,
    pub singing: Vec<AudioEventSegment>,
    pub music: Vec<AudioEventSegment>,
}

impl AudioEventSegments {
    pub fn get(&self, event: AudioEvent) -> &[AudioEventSegment] {
        match event {
            AudioEvent::Speech => &self.speech,
            AudioEvent::Singing => &self.singing,
            AudioEvent::Music => &self.music,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioEventSegmenter;

impl AudioEventSegmenter {
    #[must_use]
    pub fn segments(
        result: &AudioEventDetectionResult,
        event: AudioEvent,
        config: SegmenterConfig,
    ) -> Vec<AudioEventSegment> {
        let probabilities = class_probabilities(&result.frame_probs, event);
        Self::segments_from_probabilities(
            &probabilities,
            event,
            result.duration_seconds,
            result.frame_shift_seconds,
            config,
        )
    }

    #[must_use]
    pub fn segments_from_probabilities(
        frame_probabilities: &[f32],
        event: AudioEvent,
        duration_seconds: f64,
        frame_shift_seconds: f64,
        config: SegmenterConfig,
    ) -> Vec<AudioEventSegment> {
        if frame_probabilities.is_empty() || frame_shift_seconds <= 0.0 {
            return Vec::new();
        }
        let decisions = Postprocessor::process(frame_probabilities, config);
        let ranges = Postprocessor::decisions_to_segments(
            &decisions,
            Some(duration_seconds),
            frame_shift_seconds,
        );
        let frame_count = frame_probabilities.len();
        ranges
            .into_iter()
            .filter_map(|(start, end)| {
                let start_frame = ((start / frame_shift_seconds) as usize).min(frame_count - 1);
                let end_frame =
                    ((end / frame_shift_seconds) as usize).clamp(start_frame + 1, frame_count);
                let confidence = frame_probabilities[start_frame..end_frame]
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / (end_frame - start_frame) as f32;
                (confidence >= config.min_confidence).then_some(AudioEventSegment {
                    event,
                    start_time: start,
                    duration: end - start,
                    confidence,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn all_segments(
        result: &AudioEventDetectionResult,
        speech: SegmenterConfig,
        singing: SegmenterConfig,
        music: SegmenterConfig,
    ) -> AudioEventSegments {
        AudioEventSegments {
            speech: Self::segments(result, AudioEvent::Speech, speech),
            singing: Self::segments(result, AudioEvent::Singing, singing),
            music: Self::segments(result, AudioEvent::Music, music),
        }
    }

    #[must_use]
    pub fn analyze(
        result: &AudioEventDetectionResult,
        speech: SegmenterConfig,
        singing: SegmenterConfig,
        music: SegmenterConfig,
    ) -> AudioEventAnalysis {
        let by_class = Self::all_segments(result, speech, singing, music);
        let mut segments = Vec::with_capacity(
            by_class.speech.len() + by_class.singing.len() + by_class.music.len(),
        );
        segments.extend(by_class.speech);
        segments.extend(by_class.singing);
        segments.extend(by_class.music);
        segments.sort_by(|left, right| {
            event_name(left.event)
                .cmp(event_name(right.event))
                .then_with(|| left.start_time.total_cmp(&right.start_time))
        });
        let summary = Self::summarize(&segments, result.duration_seconds);
        AudioEventAnalysis { segments, summary }
    }

    #[must_use]
    pub fn summarize(segments: &[AudioEventSegment], duration: f64) -> AudioEventSummary {
        let stat = |events: &[AudioEvent]| AudioEventClassStats {
            coverage: union_coverage(segments, events, duration),
            confidence: peak_confidence(segments, events),
        };
        AudioEventSummary {
            speech: stat(&[AudioEvent::Speech]),
            song: stat(&[AudioEvent::Singing]),
            music: stat(&[AudioEvent::Music]),
            musical: stat(&[AudioEvent::Music, AudioEvent::Singing]),
            vocal: stat(&[AudioEvent::Singing, AudioEvent::Speech]),
        }
    }

    #[must_use]
    pub fn coverage(result: &AudioEventDetectionResult, event: AudioEvent, threshold: f32) -> f32 {
        let probabilities = class_probabilities(&result.frame_probs, event);
        if probabilities.is_empty() {
            return 0.0;
        }
        let above = probabilities
            .iter()
            .filter(|probability| **probability >= threshold)
            .count();
        above as f32 / probabilities.len() as f32
    }
}

fn class_probabilities(
    probabilities: &[AudioEventFrameProbabilities],
    event: AudioEvent,
) -> Vec<f32> {
    probabilities
        .iter()
        .map(|probability| match event {
            AudioEvent::Speech => probability.speech,
            AudioEvent::Singing => probability.singing,
            AudioEvent::Music => probability.music,
        })
        .collect()
}

fn event_name(event: AudioEvent) -> &'static str {
    match event {
        AudioEvent::Speech => "speech",
        AudioEvent::Singing => "singing",
        AudioEvent::Music => "music",
    }
}

fn union_coverage(segments: &[AudioEventSegment], events: &[AudioEvent], duration: f64) -> f64 {
    if duration <= 0.0 {
        return 0.0;
    }
    let mut spans = segments
        .iter()
        .filter(|segment| events.contains(&segment.event))
        .map(|segment| (segment.start_time, segment.end_time()))
        .collect::<Vec<_>>();
    spans.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut merged = 0.0;
    let mut current_start = -1.0;
    let mut current_end = -1.0;
    for (start, end) in spans {
        if start > current_end {
            if current_end > current_start {
                merged += current_end - current_start;
            }
            current_start = start;
            current_end = end;
        } else {
            current_end = current_end.max(end);
        }
    }
    if current_end > current_start {
        merged += current_end - current_start;
    }
    (merged / duration).min(1.0)
}

fn peak_confidence(segments: &[AudioEventSegment], events: &[AudioEvent]) -> f64 {
    segments
        .iter()
        .filter(|segment| events.contains(&segment.event))
        .map(|segment| f64::from(segment.confidence))
        .reduce(f64::max)
        .unwrap_or(0.0)
}

/// Seven-stage FireRed VAD postprocessor.
#[derive(Debug, Clone, Copy, Default)]
pub struct Postprocessor;

impl Postprocessor {
    #[must_use]
    pub fn smooth(probabilities: &[f32], window_size: usize) -> Vec<f32> {
        if window_size <= 1 {
            return probabilities.to_vec();
        }
        if probabilities.is_empty() {
            return Vec::new();
        }
        let mut output = vec![0.0; probabilities.len()];
        for index in 0..probabilities.len() {
            let first = index.saturating_sub(window_size - 1);
            output[index] =
                probabilities[first..=index].iter().copied().sum::<f32>() / window_size as f32;
        }
        let edge = (window_size - 1).min(probabilities.len());
        let mut cumulative = 0.0;
        for index in 0..edge {
            cumulative += probabilities[index];
            output[index] = cumulative / (index + 1) as f32;
        }
        output
    }

    #[must_use]
    pub fn threshold(probabilities: &[f32], threshold: f32) -> Vec<u8> {
        probabilities
            .iter()
            .map(|probability| u8::from(*probability >= threshold))
            .collect()
    }

    #[must_use]
    pub fn state_machine(
        binary: &[u8],
        min_speech_frame: usize,
        min_silence_frame: usize,
    ) -> Vec<u8> {
        if min_speech_frame == 0 && min_silence_frame == 0 {
            return binary.to_vec();
        }
        #[derive(Clone, Copy)]
        enum State {
            Silence,
            PossibleSpeech,
            Speech,
            PossibleSilence,
        }

        let mut output = vec![0; binary.len()];
        let mut state = State::Silence;
        let mut speech_start = 0;
        let mut silence_start = 0;
        for (index, value) in binary.iter().copied().enumerate() {
            let is_speech = value == 1;
            match state {
                State::Silence if is_speech => {
                    state = State::PossibleSpeech;
                    speech_start = index;
                }
                State::PossibleSpeech if is_speech => {
                    if index - speech_start >= min_speech_frame {
                        state = State::Speech;
                        output[speech_start..index].fill(1);
                    }
                }
                State::PossibleSpeech => state = State::Silence,
                State::Speech if !is_speech => {
                    state = State::PossibleSilence;
                    silence_start = index;
                }
                State::PossibleSilence if !is_speech => {
                    if index - silence_start >= min_silence_frame {
                        state = State::Silence;
                    }
                }
                State::PossibleSilence => state = State::Speech,
                State::Silence | State::Speech => {}
            }
            output[index] = match state {
                State::Speech | State::PossibleSilence => 1,
                State::Silence | State::PossibleSpeech => 0,
            };
        }
        output
    }

    #[must_use]
    pub fn fix_smooth_window_start(decisions: &[u8], smooth_window_size: usize) -> Vec<u8> {
        let mut output = decisions.to_vec();
        for index in 1..decisions.len() {
            if decisions[index - 1] == 0 && decisions[index] == 1 {
                output[index.saturating_sub(smooth_window_size)..index].fill(1);
            }
        }
        output
    }

    #[must_use]
    pub fn merge_short_silences(decisions: &[u8], merge_silence_frame: usize) -> Vec<u8> {
        if merge_silence_frame == 0 {
            return decisions.to_vec();
        }
        let mut output = decisions.to_vec();
        let mut silence_start = None;
        for index in 1..decisions.len() {
            if decisions[index - 1] == 1 && decisions[index] == 0 && silence_start.is_none() {
                silence_start = Some(index);
            } else if decisions[index - 1] == 0
                && decisions[index] == 1
                && let Some(start) = silence_start.take()
                && index - start < merge_silence_frame
            {
                output[start..index].fill(1);
            }
        }
        output
    }

    #[must_use]
    pub fn extend_speech(decisions: &[u8], extend_frame: usize) -> Vec<u8> {
        if extend_frame == 0 {
            return decisions.to_vec();
        }
        let mut output = vec![0; decisions.len()];
        for (index, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 {
                let start = index.saturating_sub(extend_frame);
                let end = (index + extend_frame + 1).min(decisions.len());
                output[start..end].fill(1);
            }
        }
        output
    }

    #[must_use]
    pub fn split_long_segments(
        decisions: &[u8],
        raw_probabilities: &[f32],
        max_speech_frame: usize,
    ) -> Vec<u8> {
        if max_speech_frame == 0 || raw_probabilities.len() < decisions.len() {
            return decisions.to_vec();
        }
        let mut output = decisions.to_vec();
        let mut segments = Vec::new();
        let mut start = None;
        for (index, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && start.is_none() {
                start = Some(index);
            } else if decision == 0
                && let Some(segment_start) = start.take()
            {
                segments.push((segment_start, index));
            }
        }
        if let Some(segment_start) = start {
            segments.push((segment_start, decisions.len()));
        }

        for (segment_start, segment_end) in segments {
            if segment_end - segment_start > max_speech_frame {
                for offset in find_split_points(
                    &raw_probabilities[segment_start..segment_end],
                    max_speech_frame,
                ) {
                    output[segment_start + offset] = 0;
                }
            }
        }
        output
    }

    #[must_use]
    pub fn decisions_to_segments(
        decisions: &[u8],
        duration_seconds: Option<f64>,
        frame_shift_seconds: f64,
    ) -> Vec<(f64, f64)> {
        let mut output = Vec::new();
        let mut start = None;
        for (index, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && start.is_none() {
                start = Some(index);
            } else if decision == 0
                && let Some(segment_start) = start.take()
            {
                output.push((
                    round3(segment_start as f64 * frame_shift_seconds),
                    round3(index as f64 * frame_shift_seconds),
                ));
            }
        }
        if let Some(segment_start) = start {
            let frame_length_seconds = FRAME_LENGTH_SAMPLES as f64 / f64::from(SAMPLE_RATE);
            let mut end = decisions.len() as f64 * frame_shift_seconds + frame_length_seconds;
            if let Some(duration) = duration_seconds {
                end = end.min(duration);
            }
            let start_seconds = (segment_start as f64 * frame_shift_seconds).min(end);
            output.push((round3(start_seconds), round3(end)));
        }
        output
    }

    #[must_use]
    pub fn process(raw_probabilities: &[f32], config: SegmenterConfig) -> Vec<u8> {
        let smoothed = Self::smooth(raw_probabilities, config.smooth_window_size);
        let binary = Self::threshold(&smoothed, config.threshold);
        let stable = Self::state_machine(&binary, config.min_event_frame, config.min_silence_frame);
        let fixed = Self::fix_smooth_window_start(&stable, config.smooth_window_size);
        let merged = Self::merge_short_silences(&fixed, config.merge_silence_frame);
        let extended = Self::extend_speech(&merged, config.extend_speech_frame);
        Self::split_long_segments(&extended, raw_probabilities, config.max_event_frame)
    }
}

fn find_split_points(probabilities: &[f32], max_speech_frame: usize) -> Vec<usize> {
    let mut splits = Vec::new();
    let mut start = 0;
    while start < probabilities.len() {
        if probabilities.len() - start <= max_speech_frame {
            break;
        }
        let window_start = start + max_speech_frame / 2;
        let window_end = (start + max_speech_frame).min(probabilities.len());
        if window_start >= window_end {
            break;
        }
        let mut minimum_index = window_start;
        let mut minimum_value = probabilities[window_start];
        for (index, value) in probabilities
            .iter()
            .copied()
            .enumerate()
            .take(window_end)
            .skip(window_start + 1)
        {
            if value < minimum_value {
                minimum_value = value;
                minimum_index = index;
            }
        }
        splits.push(minimum_index);
        start = minimum_index + 1;
    }
    splits
}

fn round3(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    fn segment(event: AudioEvent, start: f64, end: f64, confidence: f32) -> AudioEventSegment {
        AudioEventSegment {
            event,
            start_time: start,
            duration: end - start,
            confidence,
        }
    }

    fn fixture(stem: &str, file: &str) -> String {
        format!("AudioEventDetector/{stem}/{file}")
    }

    fn read_i32_as_u8(relative: &str) -> Vec<u8> {
        fixtures::require_bytes(relative)
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u8)
            .collect()
    }

    #[test]
    fn smoothing_matches_numpy_full_convolution_edge_correction() {
        let actual = Postprocessor::smooth(&[1.0, 2.0, 3.0, 4.0], 3);
        assert_eq!(actual, vec![1.0, 1.5, 2.0, 3.0]);
    }

    #[test]
    fn state_machine_backfills_confirmed_speech_and_hangs_over_silence() {
        let actual = Postprocessor::state_machine(&[1, 1, 1, 0, 0, 1, 0, 0, 0], 2, 2);
        assert_eq!(actual, vec![1, 1, 1, 1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn long_segment_splits_at_lowest_probability_in_latter_half() {
        let decisions = vec![1; 8];
        let probabilities = vec![0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.4, 0.3];
        let actual = Postprocessor::split_long_segments(&decisions, &probabilities, 6);
        assert_eq!(actual, vec![1, 1, 1, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn trailing_segment_is_capped_to_audio_duration() {
        let actual = Postprocessor::decisions_to_segments(&[0, 1, 1], Some(0.03), 0.01);
        assert_eq!(actual, vec![(0.01, 0.03)]);
    }

    #[test]
    fn oracle_smoothing_and_decisions_match_when_fixtures_are_available() {
        for stem in ["hairdryer", "sec5"] {
            let Some(flat_probabilities) = fixtures::f32_le(&fixture(stem, "probs.bin")) else {
                return;
            };
            for (event_index, name, config) in [
                (0, "speech", SegmenterConfig::SPEECH),
                (1, "singing", SegmenterConfig::SINGING),
                (2, "music", SegmenterConfig::MUSIC),
            ] {
                let probabilities = flat_probabilities
                    .chunks_exact(3)
                    .map(|row| row[event_index])
                    .collect::<Vec<_>>();
                let expected_smoothed =
                    fixtures::require_f32_le(&fixture(stem, &format!("smoothed_{name}.bin")));
                let smoothed = Postprocessor::smooth(&probabilities, SMOOTH_WINDOW_SIZE);
                let max_difference = smoothed
                    .iter()
                    .zip(expected_smoothed)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0_f32, f32::max);
                assert!(max_difference < 1.0e-5, "{stem}/{name}: {max_difference}");

                let expected_decisions =
                    read_i32_as_u8(&fixture(stem, &format!("decisions_{name}.bin")));
                assert_eq!(
                    Postprocessor::process(&probabilities, config),
                    expected_decisions
                );
            }
        }
    }

    #[test]
    fn empty_summary_is_zero() {
        assert_eq!(
            AudioEventSegmenter::summarize(&[], 10.0),
            AudioEventSummary::default()
        );
    }

    #[test]
    fn disjoint_spans_cover_their_own_fraction() {
        let summary = AudioEventSegmenter::summarize(
            &[
                segment(AudioEvent::Speech, 0.0, 2.0, 0.8),
                segment(AudioEvent::Music, 4.0, 6.0, 0.6),
            ],
            10.0,
        );
        assert!((summary.speech.coverage - 0.2).abs() < 1.0e-9);
        assert!((summary.music.coverage - 0.2).abs() < 1.0e-9);
        assert_eq!(summary.song.coverage, 0.0);
        assert!((summary.speech.confidence - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn singing_surfaces_as_song() {
        let summary =
            AudioEventSegmenter::summarize(&[segment(AudioEvent::Singing, 1.0, 5.0, 0.7)], 10.0);
        assert!((summary.song.coverage - 0.4).abs() < 1.0e-9);
        assert!((summary.song.confidence - 0.7).abs() < 1.0e-6);
        assert_eq!(summary.speech.coverage, 0.0);
        assert_eq!(summary.music.coverage, 0.0);
    }

    #[test]
    fn musical_union_merges_overlap() {
        let summary = AudioEventSegmenter::summarize(
            &[
                segment(AudioEvent::Music, 0.0, 5.0, 0.9),
                segment(AudioEvent::Singing, 3.0, 8.0, 0.7),
            ],
            10.0,
        );
        assert!((summary.musical.coverage - 0.8).abs() < 1.0e-9);
        assert!((summary.musical.confidence - 0.9).abs() < 1.0e-6);
    }

    #[test]
    fn vocal_union_merges_song_and_speech() {
        let summary = AudioEventSegmenter::summarize(
            &[
                segment(AudioEvent::Singing, 0.0, 4.0, 0.6),
                segment(AudioEvent::Speech, 2.0, 6.0, 0.8),
            ],
            10.0,
        );
        assert!((summary.vocal.coverage - 0.6).abs() < 1.0e-9);
        assert!((summary.musical.coverage - 0.4).abs() < 1.0e-9);
    }

    #[test]
    fn touching_spans_merge_and_zero_duration_is_safe() {
        let segments = [
            segment(AudioEvent::Speech, 0.0, 2.0, 0.5),
            segment(AudioEvent::Speech, 2.0, 4.0, 0.9),
        ];
        let summary = AudioEventSegmenter::summarize(&segments, 10.0);
        assert!((summary.speech.coverage - 0.4).abs() < 1.0e-9);
        assert!((summary.speech.confidence - 0.9).abs() < 1.0e-6);
        assert_eq!(
            AudioEventSegmenter::summarize(&segments, 0.0)
                .speech
                .coverage,
            0.0
        );
    }
}
