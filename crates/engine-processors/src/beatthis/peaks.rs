//! Peak picking on logits frames + neighbor deduplication.

/// Pool-max peak picking: index `i` is a peak if `logits[i] > 0` and is the
/// max across `[i - half_pool, i + half_pool]`.
pub fn find_peaks(logits: &[f32], pool_size: usize) -> Vec<usize> {
    let n = logits.len();
    let half_pool = pool_size / 2;
    let mut peaks = Vec::new();
    for i in 0..n {
        if logits[i] <= 0.0 {
            continue;
        }
        let lo = i.saturating_sub(half_pool);
        let hi = (i + half_pool + 1).min(n);
        let mut is_max = true;
        for j in lo..hi {
            if logits[j] > logits[i] {
                is_max = false;
                break;
            }
        }
        if is_max {
            peaks.push(i);
        }
    }
    peaks
}

/// Collapse runs of consecutive peak indices into their mean (Float
/// frame index). Mirrors iOS `deduplicatePeaks` exactly.
pub fn deduplicate_peaks(frames: &[usize]) -> Vec<f64> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut group_sum = frames[0] as f64;
    let mut group_count = 1;
    for i in 1..frames.len() {
        if frames[i] - frames[i - 1] <= 1 {
            group_sum += frames[i] as f64;
            group_count += 1;
        } else {
            result.push(group_sum / group_count as f64);
            group_sum = frames[i] as f64;
            group_count = 1;
        }
    }
    result.push(group_sum / group_count as f64);
    result
}

pub fn pick_beats(logits: &[f32], fps: f64, pool_size: usize) -> Vec<f64> {
    deduplicate_peaks(&find_peaks(logits, pool_size))
        .into_iter()
        .map(|frame| frame / fps)
        .collect()
}

/// Of the per-beat downbeat scores, pick those that are local maxima
/// (within `pool_size`) and return the corresponding beat times.
pub fn pick_raw_downbeats(
    beat_times: &[f64],
    downbeat_scores: &[f32],
    pool_size: usize,
) -> Vec<f64> {
    if beat_times.is_empty() || beat_times.len() != downbeat_scores.len() {
        return Vec::new();
    }
    let n = downbeat_scores.len();
    let half_pool = pool_size / 2;
    let mut result = Vec::new();
    for i in 0..n {
        if downbeat_scores[i] <= 0.0 {
            continue;
        }
        let lo = i.saturating_sub(half_pool);
        let hi = (i + half_pool + 1).min(n);
        let mut is_max = true;
        for j in lo..hi {
            if j == i {
                continue;
            }
            if downbeat_scores[j] > downbeat_scores[i] {
                is_max = false;
                break;
            }
        }
        if is_max {
            result.push(beat_times[i]);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finds_isolated_peak() {
        let mut logits = vec![-5.0; 20];
        logits[10] = 3.0;
        assert_eq!(find_peaks(&logits, 7), vec![10]);
    }

    #[test]
    fn test_ignores_negative_values() {
        assert!(find_peaks(&[-1.0, -2.0, -0.5, -3.0, -1.0], 7).is_empty());
    }

    #[test]
    fn test_ignores_zero() {
        assert!(find_peaks(&[-1.0, 0.0, -1.0], 7).is_empty());
    }

    #[test]
    fn test_finds_multiple_peaks() {
        let mut logits = vec![-5.0; 50];
        logits[10] = 5.0;
        logits[30] = 3.0;
        assert_eq!(find_peaks(&logits, 7), vec![10, 30]);
    }

    #[test]
    fn test_suppresses_non_max_in_window() {
        let mut logits = vec![-5.0; 20];
        logits[10] = 5.0;
        logits[12] = 3.0;
        assert_eq!(find_peaks(&logits, 7), vec![10]);
    }

    #[test]
    fn test_peaks_at_edges() {
        let mut logits = vec![-5.0; 10];
        logits[0] = 2.0;
        logits[9] = 3.0;
        let peaks = find_peaks(&logits, 7);
        assert!(peaks.contains(&0));
        assert!(peaks.contains(&9));
    }

    #[test]
    fn test_empty_input() {
        assert!(find_peaks(&[], 7).is_empty());
    }

    #[test]
    fn test_single_element() {
        assert_eq!(find_peaks(&[5.0], 7), vec![0]);
        assert!(find_peaks(&[-1.0], 7).is_empty());
    }

    #[test]
    fn test_custom_pool_size() {
        let mut logits = vec![-5.0; 20];
        logits[10] = 5.0;
        logits[12] = 4.0;
        let peaks = find_peaks(&logits, 3);
        assert!(peaks.contains(&10));
        assert!(peaks.contains(&12));
    }

    #[test]
    fn test_deduplicates_adjacent_frames() {
        let result = deduplicate_peaks(&[10, 11, 12, 20, 21]);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 11.0).abs() < 0.001);
        assert!((result[1] - 20.5).abs() < 0.001);
    }

    #[test]
    fn test_no_deduplication_needed() {
        assert_eq!(
            deduplicate_peaks(&[5, 10, 20, 30]),
            vec![5.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn test_single_frame() {
        assert_eq!(deduplicate_peaks(&[42]), vec![42.0]);
    }

    #[test]
    fn test_empty_frames() {
        assert!(deduplicate_peaks(&[]).is_empty());
    }

    #[test]
    fn test_all_adjacent() {
        let result = deduplicate_peaks(&[0, 1, 2, 3, 4]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_pick_beats_converts_to_seconds() {
        let mut logits = vec![-5.0; 100];
        logits[25] = 5.0;
        logits[50] = 5.0;
        logits[75] = 5.0;
        let beats = pick_beats(&logits, 50.0, 7);
        assert_eq!(beats.len(), 3);
        assert!((beats[0] - 0.5).abs() < 0.001);
        assert!((beats[1] - 1.0).abs() < 0.001);
        assert!((beats[2] - 1.5).abs() < 0.001);
    }
}
