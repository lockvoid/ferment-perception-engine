//! qm-dsp `TempoTrackV2`: RCF tempo posterior, Viterbi, and DP beats.

#[derive(Debug, Clone)]
pub struct TempoConfig {
    pub input_bpm: f64,
    pub tightness: f64,
    pub alpha: f64,
    pub rcf_len: usize,
    pub window_len: usize,
    pub hop_size: usize,
    pub comb_elements: usize,
    pub transition_sigma: f64,
    pub transition_border: usize,
}

impl Default for TempoConfig {
    fn default() -> Self {
        Self {
            input_bpm: 120.0,
            tightness: 4.0,
            alpha: 0.9,
            rcf_len: 128,
            window_len: 512,
            hop_size: 128,
            comb_elements: 4,
            transition_sigma: 8.0,
            transition_border: 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TempoResult {
    pub beat_period_sequence: Vec<usize>,
    pub bpm: f64,
    pub beat_frames: Vec<f64>,
}

pub fn track(onset: &[f32], envelope_rate: f64, config: &TempoConfig) -> TempoResult {
    if onset.len() <= config.window_len + config.hop_size {
        return TempoResult {
            beat_period_sequence: Vec::new(),
            bpm: 0.0,
            beat_frames: Vec::new(),
        };
    }
    let weighting = rayleigh_weighting(envelope_rate, config.input_bpm, config.rcf_len);
    let matrix = rcf_matrix(onset, &weighting, config);
    let periods = viterbi_beat_period(&matrix, &weighting, config);
    let beats = calculate_beats(
        onset,
        &periods,
        config.hop_size,
        config.alpha,
        config.tightness,
    );
    let median_period = median(
        &periods
            .iter()
            .map(|&value| value as f64)
            .collect::<Vec<_>>(),
    );
    let bpm = if median_period > 0.0 {
        60.0 * envelope_rate / median_period
    } else {
        0.0
    };
    TempoResult {
        beat_period_sequence: periods,
        bpm,
        beat_frames: beats,
    }
}

pub fn rayleigh_weighting(envelope_rate: f64, input_bpm: f64, len: usize) -> Vec<f64> {
    let r = envelope_rate * 60.0 / input_bpm.max(1.0);
    let r_squared = r * r;
    (0..len)
        .map(|i| {
            let x = i as f64;
            (x / r_squared) * (-(x * x) / (2.0 * r_squared)).exp()
        })
        .collect()
}

pub fn rcf_matrix(envelope: &[f32], weighting: &[f64], config: &TempoConfig) -> Vec<Vec<f64>> {
    if envelope.is_empty() || weighting.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let half_window = config.window_len / 2;
    let mut start = -(half_window as isize);
    let stop = envelope.len() as isize - half_window as isize;
    let mut frame = vec![0.0; config.window_len];
    while start < stop {
        frame.fill(0.0);
        let copy_start = start.max(0) as usize;
        let copy_end = (start + config.window_len as isize)
            .min(envelope.len() as isize)
            .max(0) as usize;
        let write_offset = copy_start as isize - start;
        for source in copy_start..copy_end {
            frame[write_offset as usize + source - copy_start] = envelope[source] as f64;
        }
        result.push(get_rcf(&frame, weighting, config));
        start += config.hop_size as isize;
    }
    result
}

pub fn get_rcf(frame: &[f64], weighting: &[f64], config: &TempoConfig) -> Vec<f64> {
    let data = adaptive_threshold_f64(frame);
    let mut rcf = vec![0.0; config.rcf_len];
    if data.len() < 2 {
        return rcf;
    }
    let max_lag =
        (config.comb_elements * config.rcf_len + config.comb_elements).min(data.len() - 1);
    let mut acf = vec![0.0; max_lag + 1];
    for (lag, value) in acf.iter_mut().enumerate() {
        let limit = data.len() - lag;
        *value = (0..limit)
            .map(|index| data[index] * data[index + lag])
            .sum::<f64>()
            / limit as f64;
    }
    for period in 2..config.rcf_len {
        let mut sum = 0.0;
        for harmonic in 1..=config.comb_elements {
            let denominator = (2 * harmonic - 1) as f64;
            for offset in (1 - harmonic as isize)..=(harmonic as isize - 1) {
                let index = harmonic as isize * period as isize + offset - 1;
                if index >= 0 && index as usize <= max_lag {
                    sum += acf[index as usize] / denominator;
                }
            }
        }
        rcf[period - 1] = sum * weighting.get(period - 1).copied().unwrap_or(0.0);
    }
    rcf = adaptive_threshold_f64(&rcf);
    let epsilon = 8.0e-7;
    for value in &mut rcf {
        *value += epsilon;
    }
    let inverse = 1.0 / (rcf.iter().sum::<f64>() + epsilon);
    for value in &mut rcf {
        *value *= inverse;
    }
    rcf
}

pub fn viterbi_beat_period(
    matrix: &[Vec<f64>],
    weighting: &[f64],
    config: &TempoConfig,
) -> Vec<usize> {
    if matrix.len() < 2 {
        return vec![0; matrix.len()];
    }
    let time_count = matrix.len();
    let state_count = matrix[0].len();
    if state_count == 0 || matrix.iter().any(|row| row.len() != state_count) {
        return Vec::new();
    }
    let mut transitions = vec![vec![0.0; state_count]; state_count];
    let lower = config.transition_border.min(state_count);
    let upper = state_count.saturating_sub(config.transition_border);
    let denominator = 2.0 * config.transition_sigma * config.transition_sigma;
    for (destination, row) in transitions.iter_mut().enumerate().take(upper).skip(lower) {
        for (source, value) in row.iter_mut().enumerate().take(upper).skip(lower) {
            let distance = source as f64 - destination as f64;
            *value = (-(distance * distance) / denominator).exp();
        }
    }

    let epsilon = 8.0e-7;
    let mut delta = vec![vec![0.0; state_count]; time_count];
    let mut psi = vec![vec![0usize; state_count]; time_count];
    for state in 0..state_count {
        delta[0][state] = weighting.get(state).copied().unwrap_or(0.0) * matrix[0][state];
    }
    normalize(&mut delta[0], epsilon);

    for time in 1..time_count {
        for destination in 0..state_count {
            let mut best_value = 0.0;
            let mut best_source = 0;
            for source in 0..state_count {
                let candidate = delta[time - 1][source] * transitions[destination][source];
                if candidate > best_value {
                    best_value = candidate;
                    best_source = source;
                }
            }
            delta[time][destination] = best_value * matrix[time][destination];
            psi[time][destination] = best_source;
        }
        normalize(&mut delta[time], epsilon);
    }

    let mut path = vec![0usize; time_count];
    path[time_count - 1] = delta[time_count - 1]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    if time_count > 2 {
        for time in (1..time_count - 1).rev() {
            path[time] = psi[time + 1][path[time + 1]];
        }
    }
    path[0] = psi[1][path[1]];
    path
}

fn normalize(values: &mut [f64], epsilon: f64) {
    let inverse = 1.0 / (values.iter().sum::<f64>() + epsilon);
    for value in values {
        *value *= inverse;
    }
}

pub fn calculate_beats(
    envelope: &[f32],
    beat_period: &[usize],
    hop_size: usize,
    alpha: f64,
    tightness: f64,
) -> Vec<f64> {
    if envelope.is_empty() || beat_period.is_empty() {
        return Vec::new();
    }
    let mut cumulative = vec![0.0f64; envelope.len()];
    let mut backlink = vec![0usize; envelope.len()];
    let mut old_period = 0;
    let mut weights = Vec::new();
    let mut range_min = 0isize;

    for index in 0..envelope.len() {
        let sequence_index = (index / hop_size.max(1)).min(beat_period.len() - 1);
        let period = beat_period[sequence_index].max(1);
        if period != old_period {
            old_period = period;
            range_min = -2 * period as isize;
            let range_max = -(period as isize) / 2;
            let len = (range_max - range_min + 1) as usize;
            weights = vec![0.0; len];
            for (j, weight) in weights.iter_mut().enumerate() {
                let argument = ((2.0 * period as f64).round() - j as f64) / period as f64;
                *weight = (-0.5 * (tightness * argument.ln()).powi(2)).exp();
            }
        }
        let mut best_value = 0.0;
        let mut best_index = 0;
        for (j, &weight) in weights.iter().enumerate() {
            let predecessor = index as isize + range_min + j as isize;
            if predecessor >= 0 {
                let candidate = weight * cumulative[predecessor as usize];
                if candidate > best_value {
                    best_value = candidate;
                    best_index = predecessor as usize;
                }
            }
        }
        cumulative[index] = alpha * best_value + (1.0 - alpha) * envelope[index] as f64;
        backlink[index] = best_index;
    }

    let last_period = beat_period.last().copied().unwrap_or(1).max(1);
    let tail_start = envelope.len().saturating_sub(last_period);
    let startpoint = cumulative[tail_start..]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| tail_start + index)
        .unwrap_or(tail_start);
    let mut beats = vec![startpoint];
    while let Some(&last) = beats.last() {
        let predecessor = backlink[last];
        if predecessor == 0 || predecessor == last {
            break;
        }
        beats.push(predecessor);
    }
    beats.reverse();
    beats.into_iter().map(|value| value as f64).collect()
}

pub fn adaptive_threshold_f64(values: &[f64]) -> Vec<f64> {
    adaptive_threshold_asymmetric(values, 8, 7)
}

fn adaptive_threshold_asymmetric(values: &[f64], before: usize, after: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut prefix = vec![0.0; values.len() + 1];
    for (index, value) in values.iter().enumerate() {
        prefix[index + 1] = prefix[index] + value;
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let first = index.saturating_sub(before);
            let last = (index + after).min(values.len() - 1);
            let mean = (prefix[last + 1] - prefix[first]) / (last - first + 1) as f64;
            (value - mean).max(0.0)
        })
        .collect()
}

pub fn adaptive_threshold(values: &[f32], width: usize) -> Vec<f32> {
    let values = values.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let output = if width == 0 {
        adaptive_threshold_f64(&values)
    } else {
        let half = (width / 2).max(1);
        adaptive_threshold_asymmetric(&values, half, half)
    };
    output.into_iter().map(|value| value as f32).collect()
}

pub fn estimate_period(
    envelope: &[f32],
    min_lag: usize,
    max_lag: usize,
    envelope_rate: Option<f64>,
    center_bpm: f64,
) -> f64 {
    if envelope.len() <= max_lag || min_lag >= max_lag {
        return 0.0;
    }
    let config = TempoConfig {
        rcf_len: max_lag + 1,
        ..TempoConfig::default()
    };
    let weighting = rayleigh_weighting(
        envelope_rate.unwrap_or(config.rcf_len as f64),
        center_bpm,
        config.rcf_len,
    );
    let frame = envelope
        .iter()
        .map(|&value| value as f64)
        .collect::<Vec<_>>();
    let rcf = get_rcf(&frame, &weighting, &config);
    let mut best_lag = min_lag;
    let mut best = f64::NEG_INFINITY;
    for (lag, &value) in rcf.iter().enumerate().take(max_lag + 1).skip(min_lag) {
        if value > best {
            best = value;
            best_lag = lag;
        }
    }
    if best_lag > min_lag && best_lag < max_lag {
        let left = rcf[best_lag - 1];
        let center = rcf[best_lag];
        let right = rcf[best_lag + 1];
        let denominator = left - 2.0 * center + right;
        if denominator.abs() > 1.0e-12 {
            let delta = 0.5 * (left - right) / denominator;
            if delta.abs() < 1.0 {
                return best_lag as f64 + delta;
            }
        }
    }
    best_lag as f64
}

pub fn dp_beats(envelope: &[f32], period_frames: f64, tightness: f64, alpha: f64) -> Vec<f64> {
    calculate_beats(
        envelope,
        &[period_frames.round().max(1.0) as usize],
        envelope.len() + 1,
        alpha,
        tightness,
    )
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimates_period_from_perfect_periodic_envelope() {
        let mut envelope = vec![0.0; 1_024];
        for index in (0..envelope.len()).step_by(43) {
            envelope[index] = 1.0;
            if index > 0 {
                envelope[index - 1] = 0.5;
            }
            if index + 1 < envelope.len() {
                envelope[index + 1] = 0.5;
            }
        }
        let period = estimate_period(&envelope, 10, 100, None, 120.0);
        assert!((period - 43.0).abs() < 0.6, "period={period}");
    }

    #[test]
    fn test_returns_zero_on_degenerate_range() {
        assert_eq!(estimate_period(&[1.0; 50], 60, 70, None, 120.0), 0.0);
    }

    #[test]
    fn test_dp_beats_land_on_envelope_peaks() {
        let mut envelope = vec![0.01; 600];
        let expected = (25..600).step_by(50).collect::<Vec<_>>();
        for &index in &expected {
            envelope[index] = 5.0;
        }
        let beats = dp_beats(&envelope, 50.0, 4.0, 0.8);
        assert!(beats.len() >= expected.len() - 2);
        let mean_error = beats
            .iter()
            .map(|beat| {
                expected
                    .iter()
                    .map(|&target| (target as f64 - beat).abs())
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / beats.len() as f64;
        assert!(mean_error < 3.0, "mean error={mean_error}");
    }

    #[test]
    fn test_track_recovers_known_bpm() {
        let mut envelope = vec![0.0; 1_500];
        for index in (43..envelope.len()).step_by(43) {
            envelope[index] = 1.0;
            envelope[index - 1] = 0.5;
            if index + 1 < envelope.len() {
                envelope[index + 1] = 0.5;
            }
        }
        let result = track(&envelope, 86.0, &TempoConfig::default());
        assert!((result.bpm - 120.0).abs() < 4.0, "bpm={}", result.bpm);
        assert!(result.beat_frames.len() > 20);
    }

    #[test]
    fn test_adaptive_threshold_clamps_at_zero() {
        let output = adaptive_threshold(&[0.1, 0.2, 0.1, 0.5, 0.1, 0.05, 0.1], 3);
        assert_eq!(output.len(), 7);
        assert!(output[3] > 0.0);
        assert!(output.iter().all(|&value| value >= 0.0));
    }

    #[test]
    fn test_viterbi_throughput() {
        let config = TempoConfig::default();
        let state_count = config.rcf_len;
        let time_count = 300;
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        let mut matrix = vec![vec![0.0; state_count]; time_count];
        for row in &mut matrix {
            let mut sum = 0.0;
            for value in row.iter_mut() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *value = ((state >> 32) % 1_000_000) as f64 / 1_000_000.0;
                sum += *value;
            }
            for value in row {
                *value /= sum.max(1.0e-12);
            }
        }
        let weighting = rayleigh_weighting(86.0, 120.0, state_count);

        let started = std::time::Instant::now();
        let production = viterbi_beat_period(&matrix, &weighting, &config);
        let elapsed = started.elapsed();
        let reference = scalar_viterbi(&matrix, &weighting, &config);

        assert_eq!(production, reference);
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "Viterbi pass took {elapsed:?}"
        );
    }

    fn scalar_viterbi(matrix: &[Vec<f64>], weighting: &[f64], config: &TempoConfig) -> Vec<usize> {
        if matrix.len() < 2 {
            return vec![0; matrix.len()];
        }
        let time_count = matrix.len();
        let state_count = matrix[0].len();
        let mut transitions = vec![vec![0.0; state_count]; state_count];
        let denominator = 2.0 * config.transition_sigma * config.transition_sigma;
        let lower = config.transition_border;
        let upper = state_count - config.transition_border;
        for (destination, row) in transitions.iter_mut().enumerate().take(upper).skip(lower) {
            for (source, value) in row.iter_mut().enumerate().take(upper).skip(lower) {
                let distance = source as f64 - destination as f64;
                *value = (-(distance * distance) / denominator).exp();
            }
        }

        let epsilon = 8.0e-7;
        let mut delta = vec![vec![0.0; state_count]; time_count];
        let mut backpointer = vec![vec![0usize; state_count]; time_count];
        for state in 0..state_count {
            delta[0][state] = weighting[state] * matrix[0][state];
        }
        let inverse = 1.0 / (delta[0].iter().sum::<f64>() + epsilon);
        for value in &mut delta[0] {
            *value *= inverse;
        }
        for time in 1..time_count {
            for destination in 0..state_count {
                let mut best_value = 0.0;
                let mut best_source = 0;
                for source in 0..state_count {
                    let candidate = delta[time - 1][source] * transitions[destination][source];
                    if candidate > best_value {
                        best_value = candidate;
                        best_source = source;
                    }
                }
                delta[time][destination] = best_value * matrix[time][destination];
                backpointer[time][destination] = best_source;
            }
            let inverse = 1.0 / (delta[time].iter().sum::<f64>() + epsilon);
            for value in &mut delta[time] {
                *value *= inverse;
            }
        }

        let mut path = vec![0usize; time_count];
        path[time_count - 1] = delta[time_count - 1]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        for time in (1..time_count - 1).rev() {
            path[time] = backpointer[time + 1][path[time + 1]];
        }
        path[0] = backpointer[1][path[1]];
        path
    }
}
