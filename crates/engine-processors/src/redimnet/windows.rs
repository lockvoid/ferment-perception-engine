//! The tiling policy of the speaker embedder: which 2.005 s windows of the
//! decoded stream get scored, and which instants each one covers.
//!
//! The window is FIXED and the graph is never zero-padded: it only ever sees a
//! full window of real audio, so the four length-sensitive stages inside this
//! network — the waveform mean/std normalisation, the per-mel-bin mean
//! subtraction, the softmax over time in each transformer block, and the
//! attentive-statistics pooling — always see an honest frame count. Padding a
//! 2 s span out to a longer window costs nearly all of the answer (measured
//! upstream: cosine 0.04 against the same span scored honestly).
//!
//! The price is that a span shorter than one window scores nothing, and the
//! tail of a span below a full window is not scored. Neither costs accuracy on
//! the spans that ARE scored, because per-window vectors average back to the
//! full-span answer — which is the caller's job, not this module's.
//!
//! Ported from `Processors/ReDimNet/ReDimNetSpeakerEncoder.swift`.

use ferment_engine_core::TimeRange;

pub const SAMPLE_RATE: u32 = 16_000;
pub const HOP_LENGTH: usize = 160;

/// Embedding width. Asserted against the graph's own output at every
/// prediction, so it cannot drift away from the weights.
pub const EMBEDDING_DIMENSION: usize = 192;

/// The graph's window, in frames first — the backbone truncates its input to a
/// multiple of 4 frames, so a window whose frame count is not a multiple of 4
/// would silently discard its own tail. 200 frames is 2.005 s, and
/// `scripts/exports/redimnet/export.py` derives the sample count the same way.
pub const WINDOW_FRAMES: usize = 200;
pub const WINDOW_SAMPLES: usize = (WINDOW_FRAMES - 1) * HOP_LENGTH + 240; // 32_080

/// The shortest span that produces anything, in seconds. Measured upstream on 6
/// LibriSpeech dev-clean speakers × 6 utterances: a single 2 s window scores
/// 1.11% EER, while at 1.5 s it is 3.24% AND min(same-speaker) falls below
/// max(cross-speaker) — i.e. one short span starts being able to mislead rather
/// than merely be vague. 2 s is the floor worth having, not the floor the graph
/// could technically run.
#[must_use]
pub fn minimum_span_seconds() -> f64 {
    WINDOW_SAMPLES as f64 / f64::from(SAMPLE_RATE)
}

/// One prediction's worth of audio: where to read it, and the instants it
/// covers in the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowPlan {
    pub sample_offset: usize,
    pub start_time: f64,
    pub end_time: f64,
}

/// Tile each span into whole windows from its start.
///
/// Spans are clamped, not trusted: a span that runs past the end of the decoded
/// stream would otherwise read whatever was left in the buffer and embed it as
/// speech.
///
/// A window never crosses a span boundary, so every vector describes audio from
/// exactly one caller span, and windows within a span never overlap. Windows
/// from two spans that overlap EACH OTHER can, which is the caller's overlap
/// and not this function's to resolve.
///
/// Bounds are ROUNDED to the nearest sample, not truncated. A span's end is
/// `start + duration` in binary floating point, so a span meant to be exactly
/// one window wide lands a hair under it about half the time — `10.0 + 2.005`
/// is `12.004999999999999`, which truncates to one sample short and silently
/// scores nothing. The seconds are an approximation of a sample index; rounding
/// is what treats them as one.
///
/// The result is sorted by offset. Sorting the SPANS would not deliver that: a
/// long span with a short one starting inside it would emit every window of the
/// first before any of the second, and a consumer walking the results would see
/// time jump backwards.
#[must_use]
pub fn plan(spans: &[TimeRange], sample_count: usize) -> Vec<WindowPlan> {
    let rate = f64::from(SAMPLE_RATE);
    let mut plans = Vec::new();
    for span in spans {
        let from = sample_index(span.start * rate, sample_count);
        let to = sample_index((span.start + span.duration) * rate, sample_count);
        if to <= from {
            continue;
        }
        for index in 0..(to - from) / WINDOW_SAMPLES {
            let offset = from + index * WINDOW_SAMPLES;
            plans.push(WindowPlan {
                sample_offset: offset,
                start_time: offset as f64 / rate,
                end_time: (offset + WINDOW_SAMPLES) as f64 / rate,
            });
        }
    }
    plans.sort_unstable_by_key(|plan| plan.sample_offset);
    plans
}

/// A span bound in seconds as a sample index inside the decoded stream.
fn sample_index(samples: f64, sample_count: usize) -> usize {
    let rounded = samples.round();
    if rounded <= 0.0 {
        return 0;
    }
    if rounded >= sample_count as f64 {
        return sample_count;
    }
    rounded as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: f64, duration: f64) -> TimeRange {
        TimeRange { start, duration }
    }

    fn window_seconds() -> f64 {
        minimum_span_seconds()
    }

    #[test]
    fn the_window_is_the_frame_ladders_own_sample_count() {
        assert_eq!(WINDOW_SAMPLES, 32_080);
        assert_eq!(WINDOW_FRAMES % 4, 0, "the backbone strides time by 4");
        assert!((minimum_span_seconds() - 2.005).abs() < 1e-12);
    }

    #[test]
    fn spans_tile_into_whole_windows_from_their_own_offset() {
        let plans = plan(&[span(10.0, 3.0 * window_seconds())], 16_000 * 600);
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].sample_offset, 160_000, "10 s in at 16 kHz");
        assert_eq!(plans[1].sample_offset, 160_000 + WINDOW_SAMPLES);
        assert_eq!(plans[2].sample_offset, 160_000 + 2 * WINDOW_SAMPLES);
        assert!((plans[0].start_time - 10.0).abs() < 1e-9);
    }

    /// The tail below a full window is dropped rather than padded — the design
    /// decision the whole port rests on, pinned here so nobody "fixes" it into
    /// a zero-padded window later.
    #[test]
    fn the_tail_below_a_full_window_is_dropped_not_padded() {
        let plans = plan(&[span(0.0, 2.0 * window_seconds() - 0.001)], 16_000 * 600);
        assert_eq!(
            plans.len(),
            1,
            "1.999 windows of audio is one window, not two"
        );
        assert!((plans[0].end_time - window_seconds()).abs() < 1e-9);
    }

    #[test]
    fn spans_shorter_than_the_window_score_nothing() {
        for duration in [0.1, window_seconds() - 0.001] {
            assert!(
                plan(&[span(0.0, duration)], 16_000 * 600).is_empty(),
                "{duration}s must not reach the model"
            );
        }
    }

    /// A span of EXACTLY the minimum duration must produce exactly one window,
    /// wherever it starts.
    ///
    /// A span's end is `start + duration` in binary floating point, so it lands
    /// a hair under the true instant about half the time: `10.0 + 2.005` is
    /// `12.004999999999999`, and truncating that to samples gives 32079 where
    /// the window needs 32080. Before `plan` rounded, a minimum-length span
    /// starting at 10 s scored nothing while the same span starting at 20 s
    /// scored fine — a silent, offset-dependent hole in the middle of the
    /// stated contract.
    #[test]
    fn an_exactly_minimum_span_yields_one_window_at_every_offset() {
        for start in 0..=30 {
            let plans = plan(&[span(f64::from(start), window_seconds())], 16_000 * 600);
            assert_eq!(
                plans.len(),
                1,
                "a minimum-length span starting at {start}s scored {} windows",
                plans.len()
            );
        }
    }

    #[test]
    fn spans_are_tiled_in_time_order() {
        let plans = plan(
            &[span(20.0, window_seconds()), span(10.0, window_seconds())],
            16_000 * 600,
        );
        assert_eq!(
            plans.iter().map(|p| p.sample_offset).collect::<Vec<_>>(),
            vec![160_000, 320_000]
        );
    }

    /// A long span with a short one starting inside it must still come out
    /// time-ordered.
    #[test]
    fn overlapping_spans_still_yield_globally_time_ordered_windows() {
        let plans = plan(
            &[
                span(0.0, 5.0 * window_seconds()),
                span(window_seconds() * 1.5, 2.0 * window_seconds()),
            ],
            16_000 * 600,
        );
        assert_eq!(plans.len(), 7);
        let offsets: Vec<usize> = plans.iter().map(|p| p.sample_offset).collect();
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        assert_eq!(
            offsets, sorted,
            "windows must come out in time order regardless of span overlap"
        );
    }

    /// Clamped, not trusted. Without the clamp a span running past the end of
    /// the stream reads a window of whatever was left in the buffer.
    #[test]
    fn out_of_range_span_bounds_are_clamped() {
        let sample_count = 10 * 16_000;

        let negative = plan(&[span(-5.0, 5.0 + window_seconds())], sample_count);
        assert_eq!(negative.len(), 1, "clamped to the start of the stream");
        assert_eq!(negative[0].sample_offset, 0);

        let past_end = plan(&[span(9.0, 600.0)], sample_count);
        assert!(past_end.is_empty(), "only 1 s of stream exists past 9 s");

        let straddling = plan(&[span(5.0, 600.0)], sample_count);
        assert_eq!(straddling.len(), 2, "5 s of stream fits two windows");
        assert!(
            straddling.last().expect("two windows").sample_offset + WINDOW_SAMPLES <= sample_count,
            "a window must never run past the decoded stream"
        );
    }

    #[test]
    fn a_reversed_span_scores_nothing() {
        assert!(plan(&[span(10.0, -5.0)], 16_000 * 600).is_empty());
    }
}
