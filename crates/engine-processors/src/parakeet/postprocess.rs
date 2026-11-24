//! Pure post-processing: build words from token IDs + chunk-boundary
//! token deduplication. No I/O, no Inference.

use serde::{Deserialize, Serialize};

use super::config::SECONDS_PER_ENCODER_FRAME;
use super::vocab::Vocab;
use crate::token_seam_merge::{self, TokenSeamRow};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribedWord {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f32,
}

/// Reconstruct words from raw token IDs using SentencePiece word boundaries.
/// Tokens whose vocab string starts with a space or U+2581 begin a new word.
pub fn build_words(
    tokens: &[i32],
    timestamps: &[i32],
    confidences: &[f32],
    durations: &[i32],
    vocab: &Vocab,
) -> Vec<TranscribedWord> {
    if tokens.is_empty() || tokens.len() != timestamps.len() {
        return Vec::new();
    }

    let frame_duration = SECONDS_PER_ENCODER_FRAME;
    let mut words = Vec::new();
    let mut current = String::new();
    let mut word_start = 0.0_f64;
    let mut word_end = 0.0_f64;
    let mut word_confidences: Vec<f32> = Vec::new();

    let flush = |words: &mut Vec<TranscribedWord>,
                 current: &mut String,
                 word_start: f64,
                 word_end: f64,
                 word_confidences: &mut Vec<f32>| {
        if current.is_empty() {
            return;
        }
        let avg = if word_confidences.is_empty() {
            0.0
        } else {
            word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
        };
        words.push(TranscribedWord {
            text: std::mem::take(current),
            start_time: word_start,
            end_time: word_end,
            confidence: avg,
        });
        word_confidences.clear();
    };

    for i in 0..tokens.len() {
        let raw = vocab.lookup(tokens[i]).unwrap_or("");
        let is_word_start = raw.starts_with(' ') || raw.starts_with('\u{2581}');
        let cleaned: String = raw.replace('\u{2581}', "");
        let cleaned = cleaned.trim();

        let start_time = f64::from(timestamps[i]) * frame_duration;
        let dur = durations.get(i).copied().unwrap_or(1).max(1);
        let end_time = start_time + f64::from(dur) * frame_duration;

        if is_word_start {
            flush(
                &mut words,
                &mut current,
                word_start,
                word_end,
                &mut word_confidences,
            );
        }

        if current.is_empty() {
            word_start = start_time;
        }
        current.push_str(cleaned);
        word_end = end_time;
        if let Some(&c) = confidences.get(i) {
            word_confidences.push(c);
        }
    }
    flush(
        &mut words,
        &mut current,
        word_start,
        word_end,
        &mut word_confidences,
    );

    words
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergeResult {
    pub tokens: Vec<i32>,
    pub timestamps: Vec<i32>,
    pub durations: Vec<i32>,
    pub confidences: Vec<f32>,
    pub removed_count: usize,
}

/// Parakeet is a TDT model: its joint predicts how many frames each token
/// covers, and that `duration` lane rides through the shared seam merge
/// untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TokenRow {
    token: i32,
    timestamp: i32,
    duration: i32,
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
pub fn merge_chunks(
    previous_tokens: &[i32],
    previous_timestamps: &[i32],
    previous_durations: &[i32],
    previous_confidences: &[f32],
    current_tokens: &[i32],
    current_timestamps: &[i32],
    current_durations: &[i32],
    current_confidences: &[f32],
    overlap_seconds: f64,
    seconds_per_frame: f64,
) -> MergeResult {
    let left = zip_rows(
        previous_tokens,
        previous_timestamps,
        previous_durations,
        previous_confidences,
    );
    let right = zip_rows(
        current_tokens,
        current_timestamps,
        current_durations,
        current_confidences,
    );
    let (rows, removed_count) =
        token_seam_merge::merge(&left, &right, overlap_seconds, seconds_per_frame);

    MergeResult {
        tokens: rows.iter().map(|row| row.token).collect(),
        timestamps: rows.iter().map(|row| row.timestamp).collect(),
        durations: rows.iter().map(|row| row.duration).collect(),
        confidences: rows.iter().map(|row| row.confidence).collect(),
        removed_count,
    }
}

fn zip_rows(
    tokens: &[i32],
    timestamps: &[i32],
    durations: &[i32],
    confidences: &[f32],
) -> Vec<TokenRow> {
    assert_eq!(tokens.len(), timestamps.len(), "token rows must be aligned");
    assert_eq!(tokens.len(), durations.len(), "token rows must be aligned");
    assert_eq!(
        tokens.len(),
        confidences.len(),
        "token rows must be aligned"
    );
    (0..tokens.len())
        .map(|index| TokenRow {
            token: tokens[index],
            timestamp: timestamps[index],
            duration: durations[index],
            confidence: confidences[index],
        })
        .collect()
}
