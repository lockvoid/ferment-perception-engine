//! Complex spectral-difference onset detector used by Queen Mary qm-dsp.

use rustfft::{FftPlanner, num_complex::Complex32};

/// The qm-dsp `DetectionFunction(DF_COMPLEXSD)` analysis front end.
#[derive(Debug, Clone)]
pub struct BeatQueenMaryOnset {
    pub step_size: usize,
    pub window_size: usize,
    window: Vec<f32>,
}

impl BeatQueenMaryOnset {
    pub const DEFAULT_STEP_SECS: f64 = 0.011_61;

    pub fn for_sample_rate(sample_rate: f64) -> Self {
        Self::for_sample_rate_and_step(sample_rate, Self::DEFAULT_STEP_SECS)
    }

    pub fn for_sample_rate_and_step(sample_rate: f64, step_secs: f64) -> Self {
        let step_size = (sample_rate * step_secs).round().max(1.0) as usize;
        let window_size = next_power_of_two(step_size * 2);
        Self::new(step_size, window_size)
    }

    pub fn new(step_size: usize, window_size: usize) -> Self {
        assert!(step_size > 0, "step_size must be positive");
        assert!(window_size >= 2 && window_size.is_power_of_two());
        assert!(step_size <= window_size);
        let window = (0..window_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / window_size as f32).cos())
            })
            .collect();
        Self {
            step_size,
            window_size,
            window,
        }
    }

    /// Compute one onset-strength value per hop, including qm-dsp's
    /// half-window zero padding at both ends.
    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let half = self.window_size / 2;
        let mut padded = vec![0.0; audio.len() + 2 * half];
        padded[half..half + audio.len()].copy_from_slice(audio);
        let frame_count = (padded.len() - self.window_size) / self.step_size + 1;
        let bin_count = half + 1;

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.window_size);
        let mut spectrum = vec![Complex32::new(0.0, 0.0); self.window_size];
        let mut mag_history = vec![0.0f32; bin_count];
        // Unit phasors encode phase without atan2/sin/cos. Histories start
        // at phase zero, matching atan2(0, 0) in the scalar oracle.
        let mut unit_real_history = vec![1.0f32; bin_count];
        let mut unit_imag_history = vec![0.0f32; bin_count];
        let mut unit_real_history_old = vec![1.0f32; bin_count];
        let mut unit_imag_history_old = vec![0.0f32; bin_count];
        let mut result = Vec::with_capacity(frame_count);

        for frame_index in 0..frame_count {
            let start = frame_index * self.step_size;
            for (i, value) in spectrum.iter_mut().enumerate() {
                *value = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut spectrum);

            let mut sum = 0.0f32;
            for k in 0..bin_count {
                let actual = spectrum[k];
                let magnitude = actual.norm();
                let (unit_real, unit_imag) = if magnitude > 0.0 {
                    (actual.re / magnitude, actual.im / magnitude)
                } else {
                    (1.0, 0.0)
                };

                // cos(φ_t − 2φ_{t−1} + φ_{t−2}) from unit phasors. Complex
                // SD only needs this cosine, so phase extraction, wrapping,
                // and sin/cos are avoidable without changing the algorithm.
                let previous_real = unit_real_history[k];
                let previous_imag = unit_imag_history[k];
                let previous_conjugate_squared_real =
                    previous_real * previous_real - previous_imag * previous_imag;
                let previous_conjugate_squared_imag = -2.0 * previous_real * previous_imag;
                let current_times_previous_real = unit_real * previous_conjugate_squared_real
                    - unit_imag * previous_conjugate_squared_imag;
                let current_times_previous_imag = unit_real * previous_conjugate_squared_imag
                    + unit_imag * previous_conjugate_squared_real;
                let cosine_deviation = (current_times_previous_real * unit_real_history_old[k]
                    - current_times_previous_imag * unit_imag_history_old[k])
                    .clamp(-1.0, 1.0);
                let previous_magnitude = mag_history[k];
                let difference_squared = previous_magnitude * previous_magnitude
                    + magnitude * magnitude
                    - 2.0 * previous_magnitude * magnitude * cosine_deviation;
                sum += difference_squared.max(0.0).sqrt();

                mag_history[k] = magnitude;
                unit_real_history_old[k] = previous_real;
                unit_imag_history_old[k] = previous_imag;
                unit_real_history[k] = unit_real;
                unit_imag_history[k] = unit_imag;
            }
            result.push(sum);
        }

        if let Some(value) = result.first_mut() {
            *value = 0.0;
        }
        if result.len() >= 2 {
            result[1] = 0.0;
        }
        result
    }
}

pub fn principal_argument(value: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    value - two_pi * (value / two_pi + 0.5).floor()
}

pub fn next_power_of_two(value: usize) -> usize {
    assert!(value > 0);
    value.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_step_for_22050_is_256() {
        let onset = BeatQueenMaryOnset::for_sample_rate(22_050.0);
        assert_eq!(onset.step_size, 256);
    }

    #[test]
    fn test_window_size_is_power_of_two_and_at_least_two_steps() {
        let onset = BeatQueenMaryOnset::for_sample_rate(22_050.0);
        assert_eq!(onset.window_size, 512);
        assert!(onset.window_size.is_power_of_two());
        assert!(onset.window_size >= onset.step_size * 2);
    }

    #[test]
    fn test_handles_non_standard_rates() {
        let onset_48k = BeatQueenMaryOnset::for_sample_rate(48_000.0);
        assert_eq!(onset_48k.step_size, 557);
        assert_eq!(onset_48k.window_size, 2_048);
    }

    #[test]
    fn test_empty_audio_returns_empty() {
        let onset = BeatQueenMaryOnset::for_sample_rate(22_050.0);
        assert!(onset.compute(&[]).is_empty());
    }

    #[test]
    fn test_silence_produces_near_zero_envelope() {
        let onset = BeatQueenMaryOnset::for_sample_rate(22_050.0);
        let envelope = onset.compute(&vec![0.0; 22_050]);
        assert!(envelope.len() > 50);
        let max_value = envelope.iter().copied().fold(0.0f32, f32::max);
        assert!(max_value < 1.0e-3, "max={max_value}");
    }

    #[test]
    fn test_impulses_produce_peaks_at_known_frames() {
        let mut audio = vec![0.0; 88_200];
        let impulses = [11_025, 22_050, 33_075, 44_100];
        for impulse in impulses {
            audio[impulse..impulse + 64].fill(1.0);
        }
        let onset = BeatQueenMaryOnset::for_sample_rate(22_050.0);
        let envelope = onset.compute(&audio);
        assert!(envelope.len() > 100);
        let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
        for impulse in impulses {
            let expected = impulse / onset.step_size;
            let local = envelope[expected.saturating_sub(10)..(expected + 10).min(envelope.len())]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            assert!(local > mean * 2.0, "expected peak near frame {expected}");
        }
    }

    #[test]
    fn test_princ_arg_wraps_to_minus_pi_to_pi() {
        for value in [
            0.0,
            std::f32::consts::PI,
            -std::f32::consts::PI,
            2.0 * std::f32::consts::PI,
            3.0 * std::f32::consts::PI,
            -3.0 * std::f32::consts::PI,
            5.5 * std::f32::consts::PI,
        ] {
            let wrapped = principal_argument(value);
            assert!(wrapped >= -std::f32::consts::PI - 1.0e-5);
            assert!(wrapped <= std::f32::consts::PI + 1.0e-5);
        }
        assert!(principal_argument(0.0).abs() < 1.0e-5);
        assert!(principal_argument(2.0 * std::f32::consts::PI).abs() < 1.0e-5);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(441), 512);
        assert_eq!(next_power_of_two(512), 512);
        assert_eq!(next_power_of_two(1_024), 1_024);
    }
}
