//! Davies–Plumbley spectral-difference downbeat detector from qm-dsp.

use rustfft::{FftPlanner, num_complex::Complex32};

use super::onset::next_power_of_two;

#[derive(Debug, Clone, PartialEq)]
pub struct DownbeatResult {
    pub downbeat_indices: Vec<usize>,
    pub beats_per_bar: usize,
    pub beat_spectral_difference: Vec<f64>,
}

pub fn decimation_factor(sample_rate: f64) -> usize {
    next_power_of_two((sample_rate / 3_000.0).round().max(1.0) as usize)
}

pub fn detect(
    audio: &[f32],
    sample_rate: f64,
    beat_frames: &[usize],
    beats_per_bar: Option<usize>,
) -> DownbeatResult {
    let fallback_meter = beats_per_bar.unwrap_or(4);
    if beat_frames.len() < 2 {
        return DownbeatResult {
            downbeat_indices: Vec::new(),
            beats_per_bar: fallback_meter,
            beat_spectral_difference: Vec::new(),
        };
    }
    let factor = decimation_factor(sample_rate);
    let decimated = decimate(audio, factor);
    if decimated.is_empty() {
        return DownbeatResult {
            downbeat_indices: Vec::new(),
            beats_per_bar: fallback_meter,
            beat_spectral_difference: Vec::new(),
        };
    }

    let decimated_rate = sample_rate / factor as f64;
    let frame_size = next_power_of_two((decimated_rate * 1.3).round().max(2.0) as usize);
    let spectrum_size = frame_size / 2;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame_size);
    let mut frame = vec![Complex32::new(0.0, 0.0); frame_size];
    let mut previous_spectrum = vec![0.0f32; spectrum_size];
    let mut spectral_difference = vec![0.0; beat_frames.len().saturating_sub(2)];

    for transition in 0..beat_frames.len() - 1 {
        let start = (beat_frames[transition] / factor).min(decimated.len() - 1);
        let end = (beat_frames[transition + 1] / factor)
            .max(start)
            .min(decimated.len() - 1);
        let beat_length = (end - start).max(1);
        let capacity = beat_length.min(frame_size);
        for (index, value) in frame.iter_mut().enumerate() {
            let sample = if index < capacity {
                let window = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * index as f32 / beat_length as f32).cos());
                decimated[start + index] * window
            } else {
                0.0
            };
            *value = Complex32::new(sample, 0.0);
        }
        fft.process(&mut frame);
        let spectrum = adaptive_threshold(
            &frame[..spectrum_size]
                .iter()
                .map(|value| value.norm())
                .collect::<Vec<_>>(),
        );
        if transition > 0 {
            spectral_difference[transition - 1] = jensen_shannon(&previous_spectrum, &spectrum);
        }
        previous_spectrum = spectrum;
    }

    let candidates = beats_per_bar.map_or_else(|| vec![3, 4], |meter| vec![meter]);
    let mut best_meter = candidates.first().copied().unwrap_or(4);
    let mut best_phase = 0;
    let mut best_score = f64::NEG_INFINITY;
    for meter in candidates {
        if meter == 0 {
            continue;
        }
        for phase in 0..meter {
            let mut sum = 0.0;
            let mut count = 0;
            let mut example = phase as isize - 1;
            while example < spectral_difference.len() as isize {
                if example >= 0 {
                    sum += spectral_difference[example as usize];
                    count += 1;
                }
                example += meter as isize;
            }
            if count > 0 {
                let mean = sum / count as f64;
                if mean > best_score {
                    best_score = mean;
                    best_phase = phase;
                    best_meter = meter;
                }
            }
        }
    }
    DownbeatResult {
        downbeat_indices: (best_phase..beat_frames.len())
            .step_by(best_meter)
            .collect(),
        beats_per_bar: best_meter,
        beat_spectral_difference: spectral_difference,
    }
}

pub fn decimate(audio: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return audio.to_vec();
    }
    if audio.len() < factor {
        return Vec::new();
    }
    let count = (audio.len() - factor) / factor + 1;
    (0..count)
        .map(|output| {
            audio[output * factor..output * factor + factor]
                .iter()
                .sum::<f32>()
                / factor as f32
        })
        .collect()
}

pub fn adaptive_threshold(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut prefix = vec![0.0f64; values.len() + 1];
    for (index, &value) in values.iter().enumerate() {
        prefix[index + 1] = prefix[index] + value as f64;
    }
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let first = index.saturating_sub(8);
            let last = (index + 7).min(values.len() - 1);
            let mean = (prefix[last + 1] - prefix[first]) / (last - first + 1) as f64;
            (value as f64 - mean).max(0.0) as f32
        })
        .collect()
}

pub fn jensen_shannon(left: &[f32], right: &[f32]) -> f64 {
    let participating = 512.min(left.len().min(right.len()) / 4);
    if participating == 0 {
        return 0.0;
    }
    let epsilon = 8.0e-7;
    let left_sum = left[..participating]
        .iter()
        .map(|&value| value as f64 + epsilon)
        .sum::<f64>();
    let right_sum = right[..participating]
        .iter()
        .map(|&value| value as f64 + epsilon)
        .sum::<f64>();
    if left_sum <= 0.0 || right_sum <= 0.0 {
        return 0.0;
    }
    (0..participating)
        .map(|index| {
            let p = (left[index] as f64 + epsilon) / left_sum;
            let q = (right[index] as f64 + epsilon) / right_sum;
            let middle = 0.5 * (p + q);
            -middle * middle.ln() + 0.5 * p * p.ln() + 0.5 * q * q.ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_decimation_factors() {
        assert_eq!(decimation_factor(22_050.0), 8);
        assert_eq!(decimation_factor(44_100.0), 16);
        assert_eq!(decimation_factor(48_000.0), 16);
    }

    #[test]
    fn test_decimate_length_is_approx_input_over_factor() {
        let audio = (0..2_048)
            .map(|value| value as f32 / 2_048.0)
            .collect::<Vec<_>>();
        assert_eq!(decimate(&audio, 8).len(), 256);
    }

    #[test]
    fn test_decimate_by_one_is_identity() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(decimate(&audio, 1), audio);
    }

    #[test]
    fn test_adaptive_threshold_suppresses_flat_regions() {
        let mut values = vec![0.5; 18];
        values[4] = 2.0;
        let output = adaptive_threshold(&values);
        assert_eq!(output.len(), values.len());
        assert!(output[4] > 0.0);
        assert!(output.iter().all(|&value| value >= 0.0));
    }

    #[test]
    fn test_js_identical_vectors_is_zero() {
        let vector = (0..128)
            .map(|index| (index % 5) as f32 + 0.1)
            .collect::<Vec<_>>();
        assert!(jensen_shannon(&vector, &vector).abs() < 1.0e-6);
    }

    #[test]
    fn test_js_different_spectra_is_positive() {
        let left = vec![0.1; 128];
        let mut right = left.clone();
        right[0] = 10.0;
        assert!(jensen_shannon(&left, &right) > 0.0);
    }

    #[test]
    fn test_detects_known_downbeat_phase_in_44_bar() {
        let sample_rate = 22_050.0;
        let beat_samples = 11_025;
        let beat_count = 32;
        let mut audio = vec![0.0; beat_samples * (beat_count + 1)];
        let beats = (0..beat_count)
            .map(|beat| {
                let start = beat * beat_samples;
                let downbeat = beat % 4 == 0;
                for sample in 0..beat_samples / 4 {
                    let time = sample as f32 / sample_rate as f32;
                    let envelope = (-12.0 * time).exp();
                    let bass = (2.0 * std::f32::consts::PI * 60.0 * time).sin();
                    let high = if downbeat {
                        0.7 * (2.0 * std::f32::consts::PI * 350.0 * time).sin()
                    } else {
                        0.0
                    };
                    audio[start + sample] =
                        if downbeat { 1.0 } else { 0.3 } * envelope * (bass + high);
                }
                start
            })
            .collect::<Vec<_>>();
        let result = detect(&audio, sample_rate, &beats, Some(4));
        assert_eq!(result.beats_per_bar, 4);
        assert!(result.downbeat_indices.len() >= 6);
        assert!(
            result
                .downbeat_indices
                .windows(2)
                .all(|pair| pair[1] - pair[0] == 4)
        );
        assert!((result.downbeat_indices[0] % 4) <= 3);
    }

    #[test]
    fn test_handles_empty_beats() {
        let result = detect(&vec![0.0; 22_050], 22_050.0, &[], Some(4));
        assert!(result.downbeat_indices.is_empty());
        assert!(result.beat_spectral_difference.is_empty());
    }
}
