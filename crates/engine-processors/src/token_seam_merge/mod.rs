//! Three-tier chunk-seam reconciliation shared by the chunked speech ports,
//! ported from FluidAudio's `ChunkProcessor.mergeChunks` and mirroring the
//! Swift `Helpers/TokenSeamMerge.swift`:
//!
//!   1. longest contiguous run of time-tolerant `(token, time)` matches across
//!      the overlap (ids equal exactly, times within half the overlap window to
//!      allow frame drift);
//!   2. LCS when that run is too short — punctuation/case rewrites at the seam
//!      once the decoder's state restarts;
//!   3. midpoint cut when neither finds anchors, which cannot produce
//!      duplicates even if token matching fails outright.
//!
//! One body for every port: mirrors drift, this cannot. Matching and cutting
//! only ever read the token id and the frame timestamp, so the rest of the row
//! (Parakeet's TDT `duration` lane, GigaAM's confidence) rides through
//! untouched.

pub mod sequence_matcher;

use sequence_matcher::{find_contiguous_matches, find_longest_common_subsequence};

/// What the seam merge needs to know about one token row.
pub trait TokenSeamRow: Copy {
    fn token(&self) -> i32;
    fn timestamp(&self) -> i32;
}

/// Merge `right` (the next chunk) into `left`, resolving the overlap.
/// Returns the merged rows and how many rows the merge removed (logging only).
pub fn merge<R: TokenSeamRow>(
    left: &[R],
    right: &[R],
    overlap_seconds: f64,
    seconds_per_frame: f64,
) -> (Vec<R>, usize) {
    if left.is_empty() {
        return (right.to_vec(), 0);
    }
    if right.is_empty() {
        return (left.to_vec(), 0);
    }

    let start_time = |row: &R| f64::from(row.timestamp()) * seconds_per_frame;
    let end_time = |row: &R| start_time(row) + seconds_per_frame;
    let left_end_time = end_time(left.last().expect("left is non-empty"));
    let right_start_time = start_time(right.first().expect("right is non-empty"));

    // No overlap in time — concatenate verbatim.
    if left_end_time <= right_start_time {
        return (left.iter().chain(right).copied().collect(), 0);
    }

    // Rows that fall inside the overlap window on each side.
    let overlap_left = left
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| end_time(row) > right_start_time - overlap_seconds)
        .collect::<Vec<_>>();
    let overlap_right = right
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, row)| start_time(row) < left_end_time + overlap_seconds)
        .collect::<Vec<_>>();

    // Time-tolerant token equality.
    let half_overlap_window = overlap_seconds / 2.0;
    let matcher = |left: &(usize, R), right: &(usize, R)| {
        left.1.token() == right.1.token()
            && (start_time(&left.1) - start_time(&right.1)).abs() < half_overlap_window
    };
    let pairs_of = |matches: &[sequence_matcher::SequenceMatch]| {
        matches
            .iter()
            .map(|matched| {
                (
                    overlap_left[matched.left_start_index].0,
                    overlap_right[matched.right_start_index].0,
                )
            })
            .collect::<Vec<_>>()
    };

    let merged = if overlap_left.len() >= 2 && overlap_right.len() >= 2 {
        let minimum_pairs = (overlap_left.len() / 2).max(1);
        let contiguous = find_contiguous_matches(&overlap_left, &overlap_right, matcher);
        if contiguous.len() >= minimum_pairs {
            merge_using_matched_pairs(left, right, &pairs_of(&contiguous))
        } else {
            let lcs = find_longest_common_subsequence(&overlap_left, &overlap_right, matcher);
            if lcs.is_empty() {
                merge_by_midpoint(
                    left,
                    right,
                    left_end_time,
                    right_start_time,
                    seconds_per_frame,
                )
            } else {
                merge_using_matched_pairs(left, right, &pairs_of(&lcs))
            }
        }
    } else {
        merge_by_midpoint(
            left,
            right,
            left_end_time,
            right_start_time,
            seconds_per_frame,
        )
    };

    let removed = (left.len() + right.len()) - merged.len();
    (merged, removed)
}

/// Keep `left` up to the first anchor, then alternate anchors with gap fills
/// (taking whichever side has more tokens in the gap so nothing is dropped),
/// then append `right` past the last anchor. FluidAudio's `mergeUsingMatches`.
fn merge_using_matched_pairs<R: Copy>(left: &[R], right: &[R], pairs: &[(usize, usize)]) -> Vec<R> {
    let Some(&(first_left, _)) = pairs.first() else {
        return left.iter().chain(right).copied().collect();
    };

    let mut output = Vec::new();
    output.extend_from_slice(&left[..first_left]);
    for (pair_index, &(left_index, right_index)) in pairs.iter().enumerate() {
        output.push(left[left_index]);
        if let Some(&(next_left, next_right)) = pairs.get(pair_index + 1) {
            let left_gap = &left[(left_index + 1)..next_left];
            let right_gap = &right[(right_index + 1)..next_right];
            if right_gap.len() > left_gap.len() {
                output.extend_from_slice(right_gap);
            } else {
                output.extend_from_slice(left_gap);
            }
        }
    }
    let last_right = pairs.last().expect("pairs is non-empty").1;
    output.extend_from_slice(&right[(last_right + 1)..]);
    output
}

/// Final fallback: cut both sides at the time midpoint of the overlap.
fn merge_by_midpoint<R: TokenSeamRow>(
    left: &[R],
    right: &[R],
    left_end_time: f64,
    right_start_time: f64,
    seconds_per_frame: f64,
) -> Vec<R> {
    let cutoff = (left_end_time + right_start_time) / 2.0;
    left.iter()
        .filter(|row| f64::from(row.timestamp()) * seconds_per_frame < cutoff)
        .chain(
            right
                .iter()
                .filter(|row| f64::from(row.timestamp()) * seconds_per_frame >= cutoff),
        )
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row with no payload beyond what the merge itself reads — the
    /// generic contract, tested without either port's extra lanes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BareRow {
        token: i32,
        timestamp: i32,
    }

    impl TokenSeamRow for BareRow {
        fn token(&self) -> i32 {
            self.token
        }
        fn timestamp(&self) -> i32 {
            self.timestamp
        }
    }

    fn rows(tokens: &[i32], timestamps: &[i32]) -> Vec<BareRow> {
        tokens
            .iter()
            .zip(timestamps)
            .map(|(&token, &timestamp)| BareRow { token, timestamp })
            .collect()
    }

    #[test]
    fn empty_left_returns_right_untouched() {
        let right = rows(&[1, 2], &[0, 5]);
        let (merged, removed) = merge(&[], &right, 0.4, 0.08);
        assert_eq!(merged, right);
        assert_eq!(removed, 0);
    }

    #[test]
    fn empty_right_returns_left_untouched() {
        let left = rows(&[1, 2], &[0, 5]);
        let (merged, removed) = merge(&left, &[], 0.4, 0.08);
        assert_eq!(merged, left);
        assert_eq!(removed, 0);
    }

    #[test]
    fn disjoint_in_time_concatenates() {
        let (merged, removed) = merge(
            &rows(&[1, 2], &[0, 5]),
            &rows(&[3, 4], &[100, 105]),
            0.4,
            0.08,
        );
        assert_eq!(
            merged.iter().map(BareRow::token).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert_eq!(removed, 0);
    }

    #[test]
    fn contiguous_run_deduplicates_the_overlap() {
        let (merged, removed) = merge(
            &rows(&[1, 2, 3, 4], &[0, 5, 8, 10]),
            &rows(&[3, 4, 5, 6], &[8, 10, 12, 14]),
            0.4,
            0.08,
        );
        assert_eq!(
            merged.iter().map(BareRow::token).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 6]
        );
        assert_eq!(removed, 2);
    }

    #[test]
    fn lcs_recovers_a_rewritten_seam() {
        let (merged, _) = merge(
            &rows(&[1, 100, 200, 300], &[0, 5, 8, 10]),
            &rows(&[100, 999, 300, 400], &[5, 7, 10, 12]),
            0.4,
            0.08,
        );
        let tokens = merged.iter().map(BareRow::token).collect::<Vec<_>>();
        assert_eq!(tokens.iter().filter(|token| **token == 100).count(), 1);
        assert_eq!(tokens.iter().filter(|token| **token == 300).count(), 1);
        assert!(tokens.contains(&400));
    }

    /// The contiguous tier is not a fast path for the LCS tier — it is a
    /// different answer, and it wins when it is long enough.
    ///
    /// Here the sides share an adjacent run of 2 (`10, 20`, meeting the
    /// `overlap_left.len() / 2` minimum) and an LCS of 3 (`10, 20, 40`). The
    /// contiguous anchors end at index 1, so everything `right` carries past
    /// it — including the rewritten `99` — is taken verbatim. The LCS anchors
    /// reach index 3, so the gap between them is filled from `left` and `30`
    /// survives instead. Deleting the contiguous tier flips this case; without
    /// it, every other test here passes either way.
    #[test]
    fn a_long_enough_contiguous_run_outranks_a_longer_lcs() {
        let (merged, _) = merge(
            &rows(&[10, 20, 30, 40], &[0, 1, 2, 3]),
            &rows(&[10, 20, 99, 40], &[0, 1, 2, 3]),
            0.4,
            0.08,
        );
        assert_eq!(
            merged.iter().map(BareRow::token).collect::<Vec<_>>(),
            [10, 20, 99, 40],
            "the LCS answer would be [10, 20, 30, 40]"
        );
    }

    #[test]
    fn midpoint_cut_keeps_time_monotonic_when_nothing_matches() {
        let (merged, _) = merge(
            &rows(&[1, 2, 3], &[0, 4, 8]),
            &rows(&[9, 8, 7], &[6, 9, 12]),
            0.4,
            0.08,
        );
        for pair in merged.windows(2) {
            assert!(pair[1].timestamp >= pair[0].timestamp);
        }
    }

    /// A one-row overlap on either side cannot support the matcher's
    /// two-anchor minimum, so the midpoint cut takes over.
    #[test]
    fn single_row_overlap_falls_through_to_midpoint() {
        let (merged, _) = merge(&rows(&[1], &[10]), &rows(&[1], &[10]), 0.4, 0.08);
        assert_eq!(merged.len(), 1, "the duplicate row is cut, not doubled");
    }
}
