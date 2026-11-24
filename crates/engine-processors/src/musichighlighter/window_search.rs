//! Best-window search via prefix-sum sliding maximum.
//! `O(n)` over the per-second saliency array.

/// Returns the best window of `length` seconds in `saliency`. If
/// `length >= saliency.len()` the whole range is returned. Empty input or a
/// non-positive length returns `(0, 0)`.
pub fn best_window(saliency: &[f32], length: isize) -> (f32, f32) {
    if length <= 0 || saliency.is_empty() {
        return (0.0, 0.0);
    }
    let length = length as usize;
    if saliency.len() <= length {
        return (0.0, saliency.len() as f32);
    }

    let mut cs = vec![0.0f32; saliency.len()];
    cs[0] = saliency[0];
    for i in 1..saliency.len() {
        cs[i] = cs[i - 1] + saliency[i];
    }

    let n_windows = saliency.len() - length + 1;
    let mut best_idx: usize = 0;
    let mut best_sum: f32 = cs[length - 1];

    for i in 1..n_windows {
        let window_sum = cs[i + length - 1] - cs[i - 1];
        if window_sum > best_sum {
            best_sum = window_sum;
            best_idx = i;
        }
    }

    (best_idx as f32, (best_idx + length) as f32)
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    #[test]
    fn empty_returns_zero_zero() {
        assert_eq!(best_window(&[], 5), (0.0, 0.0));
    }

    #[test]
    fn zero_length_returns_zero_zero() {
        assert_eq!(best_window(&[1.0, 2.0, 3.0], 0), (0.0, 0.0));
    }

    #[test]
    fn length_exceeding_input_returns_full_range() {
        let s = vec![1.0; 10];
        assert_eq!(best_window(&s, 20), (0.0, 10.0));
    }

    #[test]
    fn finds_actual_max_window() {
        // Peak at indices 5..10
        let mut s = vec![0.1f32; 30];
        for v in s.iter_mut().take(10).skip(5) {
            *v = 1.0;
        }
        let (start, end) = best_window(&s, 5);
        assert_eq!((start, end), (5.0, 10.0));
    }

    #[test]
    fn ties_pick_earliest_window() {
        // All-equal saliency → first window wins.
        let s = vec![1.0; 30];
        let (start, _) = best_window(&s, 10);
        assert_eq!(start, 0.0);
    }

    #[test]
    fn matches_swift_python_fixture_when_available() {
        let Some(saliency) = fixtures::f32_le("MusicHighlighter/test_saliency.bin") else {
            return;
        };
        assert_eq!(best_window(&saliency, 15), (15.0, 30.0));
    }
}
