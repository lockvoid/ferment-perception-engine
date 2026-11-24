//! Scenario-for-scenario translations of the portable Swift Parakeet tests.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use ferment_engine_core::{
    AudioEvent, AudioEventSegment, ProgressSink, Result, SpeechTranscribing,
};
use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, TensorMap};

use super::Parakeet;
use super::config::{
    BLANK_TOKEN_ID, DURATION_BINS, OVERLAP_FRAMES, SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME,
    SECONDS_PER_ENCODER_FRAME, WINDOW_SAMPLES,
};
use super::postprocess::{build_words, merge_chunks};
use super::processor::{CHUNK_SAMPLES, windows};
use super::speech_segment_coalescer::coalesce;
use super::vocab::Vocab;
use crate::token_seam_merge::sequence_matcher::{
    find_contiguous_matches, find_longest_common_subsequence,
};

#[derive(Debug)]
struct NeverBackend;

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl InferenceBackend for NeverBackend {
    async fn infer(
        &self,
        _model: ModelId,
        _version: &ModelVersion,
        _inputs: TensorMap,
    ) -> Result<TensorMap> {
        panic!("empty speech segments must not invoke inference")
    }
}

mod parakeet_events_tests {
    use super::*;

    #[test]
    fn test_empty_speech_events_short_circuits_to_empty_result() {
        let parakeet = Parakeet::new(Arc::new(NeverBackend), "must-not-be-read.bin");
        let segments = Vec::<AudioEventSegment>::new();
        let started = Instant::now();
        let result = futures_executor::block_on(SpeechTranscribing::transcribe(
            &parakeet,
            Path::new("must-not-be-read.wav"),
            Some(&segments),
            ProgressSink::noop(),
        ))
        .unwrap();
        assert_eq!(result.transcription, "");
        assert!(result.captions.is_empty());
        assert!(result.language.is_none());
        assert!(started.elapsed().as_secs_f64() < 1.0);
    }
}

mod parakeet_config_tests {
    use super::*;

    #[test]
    fn test_samples_per_encoder_frame() {
        assert_eq!(SAMPLES_PER_ENCODER_FRAME, 1280);
        assert!((SECONDS_PER_ENCODER_FRAME - 0.08).abs() <= 1.0e-10);
    }

    #[test]
    fn test_the_window_is15_seconds() {
        let seconds = WINDOW_SAMPLES as f64 / f64::from(SAMPLE_RATE);
        assert!((seconds - 15.0).abs() <= 1.0e-10);
    }

    #[test]
    fn test_overlap_is2_seconds() {
        let seconds = (OVERLAP_FRAMES * SAMPLES_PER_ENCODER_FRAME) as f64 / f64::from(SAMPLE_RATE);
        assert!((seconds - 2.0).abs() <= 1.0e-10);
    }

    #[test]
    fn test_duration_bins_match_v3() {
        assert_eq!(DURATION_BINS, [0, 1, 2, 3, 4]);
        assert_eq!(BLANK_TOKEN_ID, 8192);
    }
}

fn vocab(entries: &[(i32, &'static str)]) -> Vocab {
    Vocab::from_entries(entries.iter().copied())
}

mod parakeet_word_building_tests {
    use super::*;

    #[test]
    fn test_two_words() {
        let words = build_words(
            &[0, 1],
            &[0, 10],
            &[0.9, 0.8],
            &[2, 2],
            &vocab(&[(0, " hello"), (1, " world")]),
        );
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn test_subwords_merge() {
        let words = build_words(
            &[0, 1],
            &[0, 5],
            &[0.9, 0.85],
            &[2, 2],
            &vocab(&[(0, " run"), (1, "ning")]),
        );
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "running");
        assert!((words[0].confidence - 0.875).abs() <= 0.001);
    }

    #[test]
    fn test_multiple_subwords() {
        let words = build_words(
            &[0, 1, 2],
            &[0, 2, 4],
            &[0.8, 0.9, 0.85],
            &[2, 2, 2],
            &vocab(&[(0, " un"), (1, "believ"), (2, "able")]),
        );
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "unbelievable");
    }

    #[test]
    fn test_timestamps_convert_to_seconds() {
        let words = build_words(&[0], &[25], &[0.95], &[3], &vocab(&[(0, " hi")]));
        assert!((words[0].start_time - 2.0).abs() <= 0.001);
        assert!((words[0].end_time - 2.24).abs() <= 0.001);
    }

    #[test]
    fn test_sentence_piece_underscore() {
        let words = build_words(&[0], &[0], &[0.9], &[1], &vocab(&[(0, "▁test")]));
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "test");
    }

    #[test]
    fn test_empty_input() {
        assert!(build_words(&[], &[], &[], &[], &vocab(&[])).is_empty());
    }

    #[test]
    fn test_mismatched_lengths_returns_empty() {
        assert!(
            build_words(&[1, 2], &[0], &[0.9], &[1], &vocab(&[(1, " a"), (2, " b")]),).is_empty()
        );
    }

    #[test]
    fn test_unknown_token_id_produces_empty_text() {
        let words = build_words(&[999], &[0], &[0.5], &[1], &vocab(&[]));
        assert!(words.is_empty() || words[0].text.is_empty());
    }

    #[test]
    fn test_missing_durations_fallback_to_one() {
        let words = build_words(&[0], &[10], &[0.9], &[], &vocab(&[(0, " ok")]));
        assert_eq!(words.len(), 1);
        assert!((words[0].end_time - words[0].start_time - 0.08).abs() <= 0.001);
    }

    #[test]
    fn test_punctuation_stays_attached() {
        let words = build_words(
            &[0, 1],
            &[0, 5],
            &[0.9, 0.8],
            &[2, 1],
            &vocab(&[(0, " hello"), (1, ",")]),
        );
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello,");
    }

    #[test]
    fn test_real_token_stream_with_punctuation() {
        let tokens = [
            7718, 512, 2933, 7877, 391, 386, 613, 7973, 7877, 550, 391, 1120, 495,
        ];
        let timestamps = [0, 2, 3, 5, 6, 7, 8, 9, 9, 10, 12, 13, 14];
        let confidences = vec![0.99; tokens.len()];
        let durations = vec![1; tokens.len()];
        let words = build_words(
            &tokens,
            &timestamps,
            &confidences,
            &durations,
            &vocab(&[
                (7718, " Де"),
                (512, "во"),
                (2933, "чки"),
                (7877, ","),
                (391, " на"),
                (386, "ко"),
                (613, "не"),
                (7973, "ц"),
                (550, " я"),
                (1120, "ш"),
                (495, "ла"),
            ]),
        );
        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "Девочки,");
        assert_eq!(words[1].text, "наконец,");
        assert_eq!(words[2].text, "я");
        assert_eq!(words[3].text, "нашла");
        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            "Девочки, наконец, я нашла"
        );
    }
}

fn caption_word(text: &str, start: f64, end: f64) -> ferment_engine_core::TranscribedWord {
    ferment_engine_core::TranscribedWord {
        id: format!("{text}-{start}"),
        text: text.to_owned(),
        start_time: start,
        duration: end - start,
        confidence: 0.9,
    }
}

mod parakeet_segment_tests {
    use super::*;
    use crate::CaptionSegmenter;

    #[test]
    fn test_split_on_period() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("Hello.", 0.0, 1.0),
            caption_word("World.", 2.0, 3.0),
        ]);
        assert_eq!(captions.len(), 2);
        assert_eq!(captions[0].text, "Hello.");
        assert_eq!(captions[1].text, "World.");
    }

    #[test]
    fn test_split_on_question_mark() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("How?", 0.0, 1.0),
            caption_word("Good.", 2.0, 3.0),
        ]);
        assert_eq!(captions.len(), 2);
    }

    #[test]
    fn test_split_on_exclamation() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("Wow!", 0.0, 1.0),
            caption_word("Nice.", 2.0, 3.0),
        ]);
        assert_eq!(captions.len(), 2);
    }

    #[test]
    fn test_split_on_long_pause() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("first", 0.0, 1.0),
            caption_word("second", 5.0, 6.0),
        ]);
        assert_eq!(captions.len(), 2);
        assert_eq!(captions[0].text, "first");
    }

    #[test]
    fn test_no_split_on_short_pause() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("hello", 0.0, 0.5),
            caption_word("world", 1.0, 1.5),
        ]);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].text, "hello world");
    }

    #[test]
    fn test_overlong_segment_splits_at_comma() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("one", 0.0, 1.0),
            caption_word("two", 1.0, 2.0),
            caption_word("three,", 2.0, 3.0),
            caption_word("four", 3.0, 4.0),
            caption_word("five", 4.0, 5.0),
            caption_word("six,", 5.0, 7.0),
            caption_word("seven.", 7.0, 8.0),
        ]);
        assert!(captions.len() >= 2);
    }

    #[test]
    fn test_segment_timings_match_words() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("hello", 1.0, 1.5),
            caption_word("world.", 2.0, 2.5),
        ]);
        let caption = &captions[0];
        assert_eq!(caption.start_time, 1.0);
        assert_eq!(caption.end_time(), 2.5);
        assert_eq!(caption.words.len(), 2);
    }

    #[test]
    fn test_segment_contains_its_words() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("a.", 0.0, 1.0),
            caption_word("b.", 2.0, 3.0),
        ]);
        assert_eq!(captions[0].words.len(), 1);
        assert_eq!(captions[0].words[0].text, "a.");
        assert_eq!(captions[1].words[0].text, "b.");
    }

    #[test]
    fn test_empty_words_no_segments() {
        assert!(CaptionSegmenter::captions(&[]).is_empty());
    }

    #[test]
    fn test_single_word_one_segment() {
        let captions = CaptionSegmenter::captions(&[caption_word("hello", 0.0, 1.0)]);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].text, "hello");
    }

    #[test]
    fn test_segment_text_has_spaces_between_words() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("Девочки,", 0.0, 0.4),
            caption_word("наконец", 0.5, 1.0),
            caption_word("я", 1.0, 1.2),
            caption_word("нашла.", 1.3, 1.8),
        ]);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].text, "Девочки, наконец я нашла.");
    }

    #[test]
    fn test_no_punctuation_single_segment() {
        let captions = CaptionSegmenter::captions(&[
            caption_word("hello", 0.0, 0.5),
            caption_word("world", 0.5, 1.0),
            caption_word("test", 1.0, 1.5),
        ]);
        assert_eq!(captions.len(), 1);
    }
}

mod parakeet_chunking_tests {
    use super::*;

    #[test]
    fn test_short_audio_single_window() {
        let windows = windows(80_000);
        assert_eq!(windows.len(), 1);
        assert_eq!((windows[0].from, windows[0].to), (0, 80_000));
    }

    #[test]
    fn test_exact_chunk_single_window() {
        assert_eq!(windows(CHUNK_SAMPLES).len(), 1);
    }

    #[test]
    fn test_long_audio_multiple_windows() {
        let windows = windows(720_000);
        assert!(windows.len() > 1);
        assert_eq!(windows.first().map(|window| window.from), Some(0));
    }

    #[test]
    fn test_windows_overlap() {
        let windows = windows(720_000);
        for pair in windows.windows(2) {
            assert!(pair[0].to > pair[1].from);
        }
    }

    #[test]
    fn test_windows_frame_aligned() {
        for window in windows(500_000) {
            assert_eq!(window.from % SAMPLES_PER_ENCODER_FRAME, 0);
            assert!(window.to - window.from <= WINDOW_SAMPLES);
        }
    }

    #[test]
    fn test_covers_all_audio() {
        assert_eq!(windows(720_000).last().unwrap().to, 720_000);
    }

    #[test]
    fn test_one_sample_single_window() {
        assert_eq!(windows(1).len(), 1);
    }

    #[test]
    fn test_one_more_than_a_chunk_produces_two_windows() {
        assert_eq!(windows(CHUNK_SAMPLES + 1).len(), 2);
    }
}

mod sequence_matcher_tests {
    use super::*;

    #[test]
    fn test_contiguous_returns_longest_run() {
        let matches = find_contiguous_matches(&[9, 1, 2, 3, 5], &[0, 1, 2, 3, 8], |a, b| a == b);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].left_start_index, 1);
        assert_eq!(matches[0].right_start_index, 1);
        assert_eq!(matches[1].left_start_index, 2);
        assert_eq!(matches[2].left_start_index, 3);
    }

    #[test]
    fn test_contiguous_empty_on_no_match() {
        assert!(find_contiguous_matches(&[1, 2], &[3, 4], |a, b| a == b).is_empty());
    }

    #[test]
    fn test_contiguous_single_match() {
        let matches = find_contiguous_matches(&[1, 7, 3], &[9, 7, 5], |a, b| a == b);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].left_start_index, 1);
        assert_eq!(matches[0].right_start_index, 1);
    }

    #[test]
    fn test_lcs_allows_gaps() {
        let matches =
            find_longest_common_subsequence(&[1, 2, 3, 4], &[1, 99, 3, 88, 4], |a, b| a == b);
        assert_eq!(matches.len(), 3);
        assert_eq!(
            matches
                .iter()
                .map(|matched| matched.left_start_index)
                .collect::<Vec<_>>(),
            [0, 2, 3]
        );
        assert_eq!(
            matches
                .iter()
                .map(|matched| matched.right_start_index)
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );
    }

    #[test]
    fn test_lcs_empty_inputs() {
        assert!(
            find_longest_common_subsequence(&[] as &[i32], &[1, 2, 3], |a, b| a == b).is_empty()
        );
        assert!(find_longest_common_subsequence(&[1, 2, 3], &[], |a, b| a == b).is_empty());
    }

    #[test]
    fn test_lcs_ordered_ascending() {
        let matches =
            find_longest_common_subsequence(&[5, 1, 5, 2, 5, 3], &[1, 2, 3], |a, b| a == b);
        for pair in matches.windows(2) {
            assert!(pair[1].left_start_index > pair[0].left_start_index);
            assert!(pair[1].right_start_index > pair[0].right_start_index);
        }
    }

    #[test]
    fn test_custom_matcher_with_time_tolerance() {
        #[derive(Clone, Copy)]
        struct Row {
            token: i32,
            time: i32,
        }
        let left = [Row { token: 1, time: 10 }, Row { token: 2, time: 12 }];
        let right = [Row { token: 1, time: 11 }, Row { token: 2, time: 12 }];
        let matches = find_contiguous_matches(&left, &right, |left, right| {
            left.token == right.token && (left.time - right.time).abs() <= 1
        });
        assert_eq!(matches.len(), 2);
    }
}

mod parakeet_merge_chunks_tests {
    use super::*;

    fn merge(
        previous: &[i32],
        previous_times: &[i32],
        current: &[i32],
        current_times: &[i32],
    ) -> super::super::postprocess::MergeResult {
        merge_chunks(
            previous,
            previous_times,
            &vec![1; previous.len()],
            &vec![0.9; previous.len()],
            current,
            current_times,
            &vec![1; current.len()],
            &vec![0.9; current.len()],
            0.4,
            0.08,
        )
    }

    #[test]
    fn test_contiguous_overlap_deduplicates() {
        let result = merge(
            &[1, 2, 3, 4],
            &[0, 5, 8, 10],
            &[3, 4, 5, 6],
            &[8, 10, 12, 14],
        );
        assert_eq!(result.tokens, [1, 2, 3, 4, 5, 6]);
        assert_eq!(result.removed_count, 2);
    }

    #[test]
    fn test_lcs_fallback_handles_punctuation_rewrite() {
        let result = merge(
            &[1, 100, 200, 300],
            &[0, 5, 8, 10],
            &[100, 999, 300, 400],
            &[5, 7, 10, 12],
        );
        assert_eq!(
            result.tokens.iter().filter(|token| **token == 100).count(),
            1
        );
        assert_eq!(
            result.tokens.iter().filter(|token| **token == 300).count(),
            1
        );
        assert!(result.tokens.contains(&400));
    }

    #[test]
    fn test_midpoint_cut_when_no_matches() {
        let result = merge(&[1, 2, 3], &[0, 4, 8], &[9, 8, 7], &[6, 9, 12]);
        for pair in result.timestamps.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
    }

    #[test]
    fn test_empty_previous_returns_current() {
        let result = merge(&[], &[], &[1, 2], &[0, 5]);
        assert_eq!(result.tokens, [1, 2]);
        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn test_empty_current_returns_previous() {
        let result = merge(&[1, 2], &[0, 5], &[], &[]);
        assert_eq!(result.tokens, [1, 2]);
        assert_eq!(result.removed_count, 0);
    }

    #[test]
    fn test_no_overlap_in_time_is_concatenated() {
        let result = merge(&[1, 2], &[0, 5], &[3, 4], &[100, 105]);
        assert_eq!(result.tokens, [1, 2, 3, 4]);
        assert_eq!(result.removed_count, 0);
    }
}

fn segment(start: f64, end: f64, confidence: f32) -> AudioEventSegment {
    AudioEventSegment {
        event: AudioEvent::Speech,
        start_time: start,
        duration: end - start,
        confidence,
    }
}

fn spans(segments: &[AudioEventSegment]) -> Vec<[f64; 2]> {
    segments
        .iter()
        .map(|segment| [segment.start_time, segment.end_time()])
        .collect()
}

mod speech_segment_coalescer_tests {
    use super::*;

    #[test]
    fn test_empty_input_returns_empty() {
        assert!(coalesce(&[], 1.0, 30.0).is_empty());
    }

    #[test]
    fn test_single_segment_padded_both_sides() {
        assert_eq!(
            spans(&coalesce(&[segment(5.0, 6.0, 1.0)], 1.0, 30.0)),
            [[4.0, 7.0]]
        );
    }

    #[test]
    fn test_single_segment_clamped_at_edges() {
        assert_eq!(
            spans(&coalesce(&[segment(0.5, 9.5, 1.0)], 1.0, 10.0)),
            [[0.0, 10.0]]
        );
    }

    #[test]
    fn test_sub_two_pad_gap_merges() {
        assert_eq!(
            spans(&coalesce(
                &[segment(1.0, 2.0, 1.0), segment(3.0, 4.0, 1.0)],
                1.0,
                100.0,
            )),
            [[0.0, 5.0]]
        );
    }

    #[test]
    fn test_gap_beyond_two_pad_stays_split() {
        assert_eq!(
            spans(&coalesce(
                &[segment(1.0, 2.0, 1.0), segment(6.0, 7.0, 1.0)],
                1.0,
                100.0,
            )),
            [[0.0, 3.0], [5.0, 8.0]]
        );
    }

    #[test]
    fn test_chain_merge_across_three() {
        assert_eq!(
            spans(&coalesce(
                &[
                    segment(1.0, 2.0, 1.0),
                    segment(3.0, 4.0, 1.0),
                    segment(5.0, 6.0, 1.0),
                ],
                1.0,
                100.0,
            )),
            [[0.0, 7.0]]
        );
    }

    #[test]
    fn test_unsorted_input_is_sorted_then_merged() {
        assert_eq!(
            spans(&coalesce(
                &[segment(6.0, 7.0, 1.0), segment(1.0, 2.0, 1.0)],
                1.0,
                100.0,
            )),
            [[0.0, 3.0], [5.0, 8.0]]
        );
    }

    #[test]
    fn test_zero_pad_merges_touching_but_not_gapped() {
        assert_eq!(
            spans(&coalesce(
                &[
                    segment(1.0, 2.0, 1.0),
                    segment(2.0, 3.0, 1.0),
                    segment(4.0, 5.0, 1.0),
                ],
                0.0,
                100.0,
            )),
            [[1.0, 3.0], [4.0, 5.0]]
        );
    }

    #[test]
    fn test_output_is_non_overlapping_and_ascending() {
        let output = coalesce(
            &[
                segment(8.0, 9.0, 1.0),
                segment(0.2, 0.4, 1.0),
                segment(8.5, 9.2, 1.0),
                segment(3.0, 3.1, 1.0),
                segment(3.4, 3.6, 1.0),
            ],
            0.5,
            20.0,
        );
        for pair in output.windows(2) {
            assert!(pair[1].start_time > pair[0].end_time());
        }
        for segment in output {
            assert!(segment.start_time >= 0.0);
            assert!(segment.end_time() <= 20.0);
            assert!(segment.start_time < segment.end_time());
        }
    }

    #[test]
    fn test_merged_run_keeps_max_confidence() {
        let output = coalesce(
            &[segment(1.0, 2.0, 0.4), segment(3.0, 4.0, 0.9)],
            1.0,
            100.0,
        );
        assert_eq!(output.len(), 1);
        assert!((output[0].confidence - 0.9).abs() <= 1.0e-6);
    }
}
