//! Generic sequence matching used at Parakeet chunk boundaries.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceMatch {
    pub left_start_index: usize,
    pub right_start_index: usize,
    pub length: usize,
}

/// Longest common subsequence via dynamic programming. Matches are returned
/// as ordered single-element alignments, mirroring the Swift implementation.
pub fn find_longest_common_subsequence<T>(
    left: &[T],
    right: &[T],
    matcher: impl Fn(&T, &T) -> bool,
) -> Vec<SequenceMatch> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }

    let mut dp = vec![vec![0usize; right.len() + 1]; left.len() + 1];
    for i in 1..=left.len() {
        for j in 1..=right.len() {
            dp[i][j] = if matcher(&left[i - 1], &right[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    let mut matches = Vec::new();
    let mut i = left.len();
    let mut j = right.len();
    while i > 0 && j > 0 {
        if matcher(&left[i - 1], &right[j - 1]) {
            matches.push(SequenceMatch {
                left_start_index: i - 1,
                right_start_index: j - 1,
                length: 1,
            });
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    matches.reverse();
    matches
}

/// Longest run of adjacent matches in both sequences.
pub fn find_contiguous_matches<T>(
    left: &[T],
    right: &[T],
    matcher: impl Fn(&T, &T) -> bool,
) -> Vec<SequenceMatch> {
    let mut best = Vec::new();
    for i in 0..left.len() {
        for j in 0..right.len() {
            if !matcher(&left[i], &right[j]) {
                continue;
            }
            let mut current = Vec::new();
            let mut k = i;
            let mut l = j;
            while k < left.len() && l < right.len() && matcher(&left[k], &right[l]) {
                current.push(SequenceMatch {
                    left_start_index: k,
                    right_start_index: l,
                    length: 1,
                });
                k += 1;
                l += 1;
            }
            if current.len() > best.len() {
                best = current;
            }
        }
    }
    best
}
