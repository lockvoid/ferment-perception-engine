//! Viterbi HMM meter detection over candidate meters {3, 4, 5, 6, 7}.
//!
//! Emission likelihood is a Gaussian on the per-beat downbeat logit
//! (score). Transitions follow a strict modular cycle (state s → s+1 mod
//! meter) with a `transition_lambda` penalty for any other transition.
//!
//! Mirrors `BeatThisProcessing.detectMeter` with the same tuned constants.

const MU_DOWNBEAT: f32 = 10.097;
const SIGMA_DOWNBEAT: f32 = 2.276;
const MU_NON_DOWNBEAT: f32 = -9.657;
const SIGMA_NON_DOWNBEAT: f32 = 2.053;
const TRANSITION_LAMBDA: f32 = 5.0;
const CANDIDATE_METERS: [usize; 5] = [3, 4, 5, 6, 7];

#[derive(Debug, Clone, Copy)]
pub struct MeterResult {
    pub meter: usize,
    pub phase: usize,
    pub raw_meter: usize,
}

pub fn detect_meter(db_scores: &[f32]) -> Option<MeterResult> {
    if db_scores.len() < 2 {
        return None;
    }
    let (raw_meter, downbeat_indices) = viterbi_path(db_scores);
    let raw_phase = majority_phase(&downbeat_indices, raw_meter);

    // Collapse 2/4 and 6/3 to canonical forms.
    let (meter, phase) = match raw_meter {
        2 => (4, raw_phase * 2),
        6 => (3, raw_phase % 3),
        _ => (raw_meter, raw_phase),
    };

    Some(MeterResult {
        meter,
        phase,
        raw_meter,
    })
}

pub fn majority_phase(downbeat_indices: &[usize], meter: usize) -> usize {
    if downbeat_indices.is_empty() || meter == 0 {
        return 0;
    }
    let mut counts = vec![0usize; meter];
    for idx in downbeat_indices {
        counts[idx % meter] += 1;
    }
    let mut best_phase = 0;
    let mut best_count = counts[0];
    for (phase, &count) in counts.iter().enumerate().skip(1) {
        if count > best_count {
            best_count = count;
            best_phase = phase;
        }
    }
    best_phase
}

fn log_gaussian(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.918_938_5
}

/// Returns `(best_meter, downbeat_indices)`. `downbeat_indices` are the
/// frames whose Viterbi state was 0 (downbeat).
pub fn viterbi_path(db_scores: &[f32]) -> (usize, Vec<usize>) {
    let n = db_scores.len();
    if n == 0 {
        return (4, Vec::new());
    }

    let mut best_meter = 4usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_path: Vec<usize> = Vec::new();

    for &meter in &CANDIDATE_METERS {
        let mut v = vec![vec![f32::NEG_INFINITY; meter]; n];
        let mut bp = vec![vec![0usize; meter]; n];

        let log_prior = -(meter as f32).ln();
        for (s, value) in v[0].iter_mut().enumerate() {
            let emit = if s == 0 {
                log_gaussian(db_scores[0], MU_DOWNBEAT, SIGMA_DOWNBEAT)
            } else {
                log_gaussian(db_scores[0], MU_NON_DOWNBEAT, SIGMA_NON_DOWNBEAT)
            };
            *value = log_prior + emit;
        }

        for t in 1..n {
            let (previous_rows, current_rows) = v.split_at_mut(t);
            let previous = &previous_rows[t - 1];
            let current = &mut current_rows[0];
            for (s, (value, previous_state)) in current.iter_mut().zip(bp[t].iter_mut()).enumerate()
            {
                let emit = if s == 0 {
                    log_gaussian(db_scores[t], MU_DOWNBEAT, SIGMA_DOWNBEAT)
                } else {
                    log_gaussian(db_scores[t], MU_NON_DOWNBEAT, SIGMA_NON_DOWNBEAT)
                };
                let normal_prev = (s + meter - 1) % meter;
                let mut best_prev = normal_prev;
                let mut best_prev_score = previous[normal_prev];
                for (p, &previous_score) in previous.iter().enumerate() {
                    if p == normal_prev {
                        continue;
                    }
                    let sc = previous_score - TRANSITION_LAMBDA;
                    if sc > best_prev_score {
                        best_prev_score = sc;
                        best_prev = p;
                    }
                }
                *value = best_prev_score + emit;
                *previous_state = best_prev;
            }
        }

        let mut end_state = 0usize;
        let mut end_score = f32::NEG_INFINITY;
        for (s, &score) in v[n - 1].iter().enumerate() {
            if score > end_score {
                end_score = score;
                end_state = s;
            }
        }

        let normalized = end_score / n as f32;
        if normalized > best_score {
            best_score = normalized;
            best_meter = meter;
            let mut path = vec![0usize; n];
            path[n - 1] = end_state;
            for t in (0..n - 1).rev() {
                path[t] = bp[t + 1][path[t + 1]];
            }
            best_path = path;
        }
    }

    let downbeat_indices: Vec<usize> = best_path
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s == 0).then_some(i))
        .collect();
    (best_meter, downbeat_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_phase_simple() {
        assert_eq!(majority_phase(&[0, 4, 8, 12, 16, 1], 4), 0);
    }

    #[test]
    fn test_majority_phase_tie_goes_to_first() {
        assert_eq!(majority_phase(&[0, 1, 4, 5], 4), 0);
    }

    #[test]
    fn test_majority_phase_empty() {
        assert_eq!(majority_phase(&[], 4), 0);
    }

    #[test]
    fn test_majority_phase_all_same() {
        assert_eq!(majority_phase(&[2, 6, 10, 14], 4), 2);
    }

    #[test]
    fn test_detects_4_4() {
        let scores = (0..32)
            .map(|index| if index % 4 == 0 { 10.0 } else { -10.0 })
            .collect::<Vec<_>>();
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 4);
        assert_eq!(result.phase, 0);
    }

    #[test]
    fn test_detects_3_4() {
        let scores = (0..24)
            .map(|index| if index % 3 == 0 { 10.0 } else { -10.0 })
            .collect::<Vec<_>>();
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 3);
        assert_eq!(result.phase, 0);
    }

    #[test]
    fn test_detects_phase_offset() {
        let scores = (0..32)
            .map(|index| if index % 4 == 2 { 10.0 } else { -10.0 })
            .collect::<Vec<_>>();
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 4);
        assert_eq!(result.phase, 2);
    }

    #[test]
    fn test_normalizes_from_meter_6_to_meter_3() {
        let scores = (0..42)
            .map(|index| if index % 6 == 0 { 10.0 } else { -10.0 })
            .collect::<Vec<_>>();
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 3);
        assert_eq!(result.raw_meter, 6);
    }

    #[test]
    fn test_too_few_beats() {
        assert!(detect_meter(&[]).is_none());
        assert!(detect_meter(&[5.0]).is_none());
    }

    #[test]
    fn test_two_beats_minimum() {
        assert!(detect_meter(&[10.0, -10.0]).is_some());
    }

    #[test]
    fn test_noisy_scores_still_finds_pattern() {
        let noise = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let scores = (0..64)
            .map(|index| {
                let base = if index % 4 == 0 { 8.0 } else { -8.0 };
                base + noise[index % noise.len()]
            })
            .collect::<Vec<_>>();
        assert_eq!(detect_meter(&scores).expect("meter").meter, 4);
    }

    #[test]
    fn test_all_same_scores() {
        assert!(detect_meter(&[5.0; 20]).is_some());
    }
}
