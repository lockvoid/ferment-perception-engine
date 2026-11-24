//! Pure post-processing for GigaAM token output: SentencePiece pieces to
//! timed words, and chunk-boundary reconciliation.
//!
//! Ported from iOS `GigaAMPostprocessing.swift`. The merge strategy is the
//! shared three-tier [`crate::token_seam_merge`]; the duration lane is absent
//! throughout. Parakeet is a TDT model whose joint predicts how many frames
//! each token covers; GigaAM is a plain RNN-T and predicts no such thing, so
//! carrying a parallel array of ones would be a lie in the shape of data.

use ferment_engine_core::TranscribedWord;
use ferment_utils::id;

use super::config::SECONDS_PER_ENCODER_FRAME;
use crate::token_seam_merge::{self, TokenSeamRow};

/// The one piece that is a name rather than text. Upstream's `sp.decode`
/// renders it ` ⁇ `; splicing the literal string into a caption is the one
/// thing it must not do.
const UNKNOWN_PIECE: &str = "<unk>";

/// SentencePiece marks a word start with U+2581 LOWER ONE EIGHTH BLOCK.
const WORD_START: char = '\u{2581}';

/// Build words from token ids using SentencePiece word boundaries.
///
/// Timing comes straight off the RNN-T emissions, which is all this model
/// offers: a word starts at its first token's encoder frame and ends one frame
/// past its last token's. That gives every word a 40 ms floor and keeps starts
/// monotonic. It is an emission-time span, not an acoustic one — RNN-T emits at
/// or just after the evidence, so a word's true onset can sit slightly earlier
/// than reported.
#[must_use]
pub fn build_words(
    tokens: &[i32],
    timestamps: &[i32],
    confidences: &[f32],
    vocabulary: &[String],
) -> Vec<TranscribedWord> {
    if tokens.is_empty() || tokens.len() != timestamps.len() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current = String::new();
    let mut word_start_frame = 0i32;
    let mut word_end_frame = 0i32;
    let mut word_confidences: Vec<f32> = Vec::new();
    // `token_seam_merge` can hand back a frame that steps backwards at a seam:
    // it matches anchors with a time tolerance and fills the gap between them
    // from whichever chunk has more tokens there, so a gap token can predate
    // the anchor before it. The token order carries the text, so time is
    // clamped to run with the sequence rather than reordering it.
    let mut frame_floor = 0i32;

    // The reset runs even on the empty-word path: piece id 3 is a bare `▁`,
    // which opens a word, cleans to "", and contributes a confidence — bailing
    // out before the reset would carry that confidence into the next word's
    // average.
    let flush = |words: &mut Vec<TranscribedWord>,
                 current: &mut String,
                 confidences: &mut Vec<f32>,
                 start_frame: i32,
                 end_frame: i32| {
        if !current.is_empty() {
            let average = if confidences.is_empty() {
                0.0
            } else {
                confidences.iter().sum::<f32>() / confidences.len() as f32
            };
            let start = f64::from(start_frame) * SECONDS_PER_ENCODER_FRAME;
            let end = f64::from(end_frame + 1) * SECONDS_PER_ENCODER_FRAME;
            words.push(TranscribedWord {
                id: id::short(),
                text: std::mem::take(current),
                start_time: start,
                duration: end - start,
                confidence: average,
            });
        }
        current.clear();
        confidences.clear();
    };

    for (index, &token) in tokens.iter().enumerate() {
        let piece = vocabulary
            .get(usize::try_from(token).unwrap_or(usize::MAX))
            .map_or("", String::as_str);
        let is_word_start = piece.starts_with(WORD_START);
        let stripped = piece.replace(WORD_START, "");
        let cleaned = if piece == UNKNOWN_PIECE {
            ""
        } else {
            stripped.trim()
        };

        let frame = timestamps[index].max(frame_floor);
        frame_floor = frame;

        if is_word_start {
            flush(
                &mut words,
                &mut current,
                &mut word_confidences,
                word_start_frame,
                word_end_frame,
            );
        }
        if current.is_empty() {
            word_start_frame = frame;
        }
        current.push_str(cleaned);
        word_end_frame = frame;
        word_confidences.push(confidences.get(index).copied().unwrap_or(0.0));
    }
    flush(
        &mut words,
        &mut current,
        &mut word_confidences,
        word_start_frame,
        word_end_frame,
    );

    words
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergeResult {
    pub tokens: Vec<i32>,
    pub timestamps: Vec<i32>,
    pub confidences: Vec<f32>,
    pub removed_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TokenRow {
    token: i32,
    timestamp: i32,
    confidence: f32,
}

impl TokenSeamRow for TokenRow {
    fn token(&self) -> i32 {
        self.token
    }

    fn timestamp(&self) -> i32 {
        self.timestamp
    }
}

/// Merge `current` into `previous`, resolving the overlap region via the
/// shared three-tier [`token_seam_merge`] (contiguous -> LCS -> midpoint).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn merge_chunks(
    previous_tokens: &[i32],
    previous_timestamps: &[i32],
    previous_confidences: &[f32],
    current_tokens: &[i32],
    current_timestamps: &[i32],
    current_confidences: &[f32],
    overlap_seconds: f64,
    seconds_per_frame: f64,
) -> MergeResult {
    let left = zip_rows(previous_tokens, previous_timestamps, previous_confidences);
    let right = zip_rows(current_tokens, current_timestamps, current_confidences);
    let (rows, removed_count) =
        token_seam_merge::merge(&left, &right, overlap_seconds, seconds_per_frame);

    MergeResult {
        tokens: rows.iter().map(|row| row.token).collect(),
        timestamps: rows.iter().map(|row| row.timestamp).collect(),
        confidences: rows.iter().map(|row| row.confidence).collect(),
        removed_count,
    }
}

fn zip_rows(tokens: &[i32], timestamps: &[i32], confidences: &[f32]) -> Vec<TokenRow> {
    assert_eq!(tokens.len(), timestamps.len(), "token rows must be aligned");
    assert_eq!(
        tokens.len(),
        confidences.len(),
        "token rows must be aligned"
    );
    (0..tokens.len())
        .map(|index| TokenRow {
            token: tokens[index],
            timestamp: timestamps[index],
            confidence: confidences[index],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny stand-in vocabulary: index == token id, `▁` opens a word.
    fn vocabulary() -> Vec<String> {
        [
            "<unk>",
            "\u{2581}",
            "\u{2581}при",
            "вет",
            "\u{2581}мир",
            "!",
        ]
        .iter()
        .map(|piece| (*piece).to_owned())
        .collect()
    }

    fn texts(words: &[TranscribedWord]) -> Vec<&str> {
        words.iter().map(|word| word.text.as_str()).collect()
    }

    #[test]
    fn subword_pieces_join_into_one_word() {
        let words = build_words(&[2, 3, 4, 5], &[0, 1, 3, 4], &[1.0; 4], &vocabulary());
        assert_eq!(texts(&words), ["привет", "мир!"]);
    }

    #[test]
    fn a_word_spans_from_its_first_frame_to_one_past_its_last() {
        let words = build_words(&[2, 3], &[5, 7], &[1.0, 1.0], &vocabulary());
        assert_eq!(words.len(), 1);
        assert!((words[0].start_time - 5.0 * 0.04).abs() < 1e-12);
        assert!((words[0].end_time() - 8.0 * 0.04).abs() < 1e-12);
    }

    #[test]
    fn confidence_is_the_mean_over_the_words_own_tokens() {
        let words = build_words(&[2, 3, 4], &[0, 1, 2], &[0.4, 0.6, 0.9], &vocabulary());
        assert!((words[0].confidence - 0.5).abs() < 1e-6);
        assert!((words[1].confidence - 0.9).abs() < 1e-6);
    }

    /// Piece id 3 in the real vocabulary is a bare `▁`: it opens a word,
    /// cleans to "", and still contributes a confidence. That confidence must
    /// not survive into the next word.
    #[test]
    fn a_bare_word_start_marker_does_not_leak_its_confidence() {
        let words = build_words(&[1, 4], &[0, 1], &[0.0, 1.0], &vocabulary());
        assert_eq!(texts(&words), ["мир"]);
        assert!(
            (words[0].confidence - 1.0).abs() < 1e-6,
            "the empty marker's 0.0 leaked in: {}",
            words[0].confidence
        );
    }

    #[test]
    fn the_unknown_piece_contributes_no_text() {
        let words = build_words(&[2, 0, 3], &[0, 1, 2], &[1.0; 3], &vocabulary());
        assert_eq!(texts(&words), ["привет"]);
    }

    /// A seam can hand back a timestamp that steps backwards; the clamp keeps
    /// word starts monotonic without reordering the tokens.
    #[test]
    fn timestamps_are_clamped_forward_at_a_seam() {
        let words = build_words(&[2, 3, 4], &[10, 11, 4], &[1.0; 3], &vocabulary());
        assert_eq!(texts(&words), ["привет", "мир"]);
        assert!(words[1].start_time >= words[0].start_time);
    }

    /// Kill: a seam running backwards inside a word creates a non-positive duration.
    /// Current Swift exact input, including the backward timestamp within alphabeta.
    #[test]
    fn current_backward_seam_keeps_positive_durations_and_word_order() {
        let vocabulary = vec![
            String::new(),
            "▁alpha".into(),
            "beta".into(),
            "▁gamma".into(),
        ];
        let words = build_words(&[1, 2, 3], &[100, 98, 99], &[1.0; 3], &vocabulary);
        assert_eq!(texts(&words), ["alphabeta", "gamma"]);
        assert!(words.iter().all(|word| word.duration > 0.0));
        assert!(
            words
                .windows(2)
                .all(|pair| pair[1].start_time >= pair[0].start_time)
        );
    }

    /// Kill: an empty bare marker's confidence survives the flush into the next word.
    #[test]
    fn current_bare_marker_case_preserves_both_word_confidences() {
        let vocabulary = vec![
            String::new(),
            "▁с".into(),
            "а".into(),
            "▁".into(),
            "▁мне".into(),
        ];
        let words = build_words(
            &[1, 2, 3, 4],
            &[0, 1, 2, 3],
            &[0.9, 0.9, 0.01, 0.99],
            &vocabulary,
        );
        assert_eq!(texts(&words), ["са", "мне"]);
        assert!((words[0].confidence - 0.9).abs() <= 1e-6);
        assert!((words[1].confidence - 0.99).abs() <= 1e-6);
    }

    #[test]
    fn degenerate_input_yields_no_words() {
        assert!(build_words(&[], &[], &[], &vocabulary()).is_empty());
        assert!(build_words(&[2], &[0, 1], &[1.0], &vocabulary()).is_empty());
    }

    /// The merge itself is covered in `token_seam_merge`; this pins that the
    /// confidence lane rides through in step with the tokens.
    #[test]
    fn merge_carries_the_confidence_lane_with_its_tokens() {
        let merged = merge_chunks(
            &[1, 2, 3, 4],
            &[0, 5, 8, 10],
            &[0.1, 0.2, 0.3, 0.4],
            &[3, 4, 5, 6],
            &[8, 10, 12, 14],
            &[0.7, 0.8, 0.5, 0.6],
            2.0,
            0.04,
        );
        assert_eq!(merged.tokens, [1, 2, 3, 4, 5, 6]);
        assert_eq!(merged.confidences, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(merged.removed_count, 2);
    }
}
