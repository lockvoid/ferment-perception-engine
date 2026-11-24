//! Percival & Tzanetakis (2014) tempo estimation.

use std::sync::Arc;

use ferment_engine_audio::{AudioFrames, resample_fft};
use ferment_engine_core::Result;
use rustfft::{Fft, FftPlanner, num_complex::Complex32};

/// The rate the estimator runs at; frame, hop and lag constants assume it.
pub const SAMPLE_RATE: u32 = 44_100;
const FRAME: usize = 1024;
const HOP: usize = 128;
const OSS_FRAME: usize = 2048;
const OSS_HOP: usize = 128;
/// Onset-strength rate: one flux value per hop, 344.53 Hz.
const OSS_RATE: f32 = SAMPLE_RATE as f32 / HOP as f32;
/// `scipy.signal.firwin(15, 7 / (OSS_RATE / 2))`.
const LOW_PASS: [f32; 15] = [
    0.00933978, 0.01521148, 0.03163891, 0.05607187, 0.08390299, 0.10948195, 0.12742038, 0.13386527,
    0.12742038, 0.10948195, 0.08390299, 0.05607187, 0.03163891, 0.01521148, 0.00933978,
];
/// Correlation lags searched for tempo peaks: 210 down to 50 BPM, truncated
/// to whole lags.
const MIN_LAG: usize = (OSS_RATE as f64 * 60.0 / 210.0) as usize;
const MAX_LAG: f32 = (OSS_RATE as f64 * 60.0 / 50.0) as usize as f32;
/// Every window's lag adds a Gaussian of this width (in lags) to a histogram
/// long enough for the slowest tempo.
const HISTOGRAM_LAGS: usize = 414;
const LAG_SIGMA: f32 = 10.0;

/// Tempo of `frames` in BPM, or `None` when no analysis window holds a tempo
/// peak — for instance when the input is shorter than one 1024-sample frame
/// at 44.1 kHz. Other rates are resampled first.
pub fn percival_bpm(frames: &AudioFrames) -> Result<Option<f64>> {
    let resampled;
    let samples = if frames.sample_rate == SAMPLE_RATE {
        frames.samples.as_slice()
    } else {
        resampled = resample_fft(&frames.samples, frames.sample_rate, SAMPLE_RATE)?;
        resampled.as_slice()
    };
    let lags = window_lags(&onset_strength(samples));
    if lags.is_empty() {
        return Ok(None);
    }
    let histogram = lag_histogram(&lags);
    let lag = argmax(&histogram);
    let bpm = (f64::from(OSS_RATE) * 60.0 / lag as f64) as f32;
    Ok(Some(f64::from(octave(&histogram, lag, bpm) * bpm)))
}

/// Half-wave-rectified spectral flux of `ln(1 + 1000·|X| / N)` over Hamming
/// frames that fit the input entirely, low-passed.
fn onset_strength(samples: &[f32]) -> Vec<f32> {
    if samples.len() < FRAME {
        return Vec::new();
    }
    let window = hamming(FRAME);
    let fft = FftPlanner::new().plan_fft_forward(FRAME);
    let mut spectrum = vec![Complex32::default(); FRAME];
    let mut previous = vec![0.0f32; FRAME / 2 + 1];
    let flux: Vec<f32> = samples
        .windows(FRAME)
        .step_by(HOP)
        .map(|frame| {
            for ((bin, sample), weight) in spectrum.iter_mut().zip(frame).zip(&window) {
                *bin = Complex32::new(sample * weight, 0.0);
            }
            fft.process(&mut spectrum);
            let mut flux = 0.0;
            for (bin, last) in spectrum.iter().zip(previous.iter_mut()) {
                let level = (bin.norm() / FRAME as f32 * 1000.0 + 1.0).ln();
                if level > *last {
                    flux += level - *last;
                }
                *last = level;
            }
            flux
        })
        .collect();
    low_pass(&flux)
}

/// Hamming window `0.53836 − 0.46164·cos(2πn/(N−1))`.
fn hamming(size: usize) -> Vec<f32> {
    let denominator = (size - 1) as f64;
    (0..size)
        .map(|n| {
            (0.53836 - 0.46164 * (2.0 * std::f64::consts::PI * n as f64 / denominator).cos()) as f32
        })
        .collect()
}

fn low_pass(signal: &[f32]) -> Vec<f32> {
    (0..signal.len())
        .map(|n| {
            LOW_PASS
                .iter()
                .take(n + 1)
                .enumerate()
                .map(|(delay, coefficient)| coefficient * signal[n - delay])
                .sum()
        })
        .collect()
}

/// One tempo lag per 2048-value onset window, hop 128. The last window is
/// zero-padded, so any non-empty onset signal yields at least one window.
fn window_lags(onsets: &[f32]) -> Vec<usize> {
    if onsets.is_empty() {
        return Vec::new();
    }
    let windows = onsets.len().saturating_sub(OSS_FRAME).div_ceil(OSS_HOP) + 1;
    let mut correlation = Autocorrelation::new(OSS_FRAME);
    let mut window = vec![0.0f32; OSS_FRAME];
    let mut lags = Vec::with_capacity(windows);
    for index in 0..windows {
        let start = index * OSS_HOP;
        let end = (start + OSS_FRAME).min(onsets.len());
        window.fill(0.0);
        window[..end - start].copy_from_slice(&onsets[start..end]);
        lags.extend(strongest_lag(&enhance_harmonics(
            correlation.compute(&window),
        )));
    }
    lags
}

/// Generalized autocorrelation: the unnormalized inverse FFT of
/// `(|FFT(x)| / M)^0.5` over the input zero-padded to `M`, the next power of
/// two at or above twice its length; the first `len` lags.
struct Autocorrelation {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex32>,
}

impl Autocorrelation {
    fn new(len: usize) -> Self {
        let padded = (2 * len).next_power_of_two();
        let mut planner = FftPlanner::new();
        Self {
            forward: planner.plan_fft_forward(padded),
            inverse: planner.plan_fft_inverse(padded),
            buffer: vec![Complex32::default(); padded],
        }
    }

    fn compute(&mut self, signal: &[f32]) -> Vec<f32> {
        let padded = self.buffer.len() as f32;
        self.buffer.fill(Complex32::default());
        for (bin, sample) in self.buffer.iter_mut().zip(signal) {
            bin.re = *sample;
        }
        self.forward.process(&mut self.buffer);
        for bin in &mut self.buffer {
            *bin = Complex32::new((bin.norm() / padded).sqrt(), 0.0);
        }
        self.inverse.process(&mut self.buffer);
        self.buffer[..signal.len()]
            .iter()
            .map(|bin| bin.re)
            .collect()
    }
}

/// Adds each lag's doubled and quadrupled lag, in place.
fn enhance_harmonics(mut correlation: Vec<f32>) -> Vec<f32> {
    for lag in 0..correlation.len() / 4 {
        let harmonics = correlation[2 * lag] + correlation[4 * lag];
        correlation[lag] += harmonics;
    }
    correlation
}

/// The rounded position of the strongest correlation peak in the lag range.
///
/// The paper goes on to re-score candidate peaks against pulse trains; this
/// estimator keeps the strongest peak, the variant measured on the loop
/// library it was chosen with.
fn strongest_lag(correlation: &[f32]) -> Option<usize> {
    peaks(correlation)
        .into_iter()
        .reduce(|best, peak| if peak.1 > best.1 { peak } else { best })
        .map(|(position, _)| position.round() as usize)
}

/// Correlation peaks inside the lag range, in ascending position: interpolated
/// parabolically, a plateau peaking at its middle, the scan stopping at the
/// first peak past the range.
fn peaks(values: &[f32]) -> Vec<(f32, f32)> {
    let size = values.len();
    let mut peaks = Vec::new();
    let mut i = MIN_LAG;
    if size < MIN_LAG + 3 {
        return peaks;
    }
    if values[i] > values[i + 1] {
        peaks.push((i as f32, values[i]));
    }
    loop {
        while i + 1 < size - 1 && values[i] >= values[i + 1] {
            i += 1;
        }
        while i + 1 < size - 1 && values[i] < values[i + 1] {
            i += 1;
        }
        let mut j = i;
        while j + 1 < size - 1 && values[j] == values[j + 1] {
            j += 1;
        }
        if j + 1 < size - 1 && values[j + 1] < values[j] {
            let peak = if j == i {
                interpolate(values[j - 1], values[j], values[j + 1], j)
            } else {
                ((i + j) as f32 * 0.5, values[i])
            };
            if peak.0 > MAX_LAG {
                break;
            }
            peaks.push(peak);
        }
        i = j;
        if i + 1 >= size - 1 {
            if i == size - 2 && values[i - 1] < values[i] && values[i + 1] < values[i] {
                peaks.push(interpolate(values[i - 1], values[i], values[i + 1], i));
            }
            break;
        }
    }
    peaks
}

/// Vertex of the parabola through three neighbours: (position, value).
fn interpolate(left: f32, middle: f32, right: f32, bin: usize) -> (f32, f32) {
    let delta = 0.5 * ((left - right) / (left - 2.0 * middle + right));
    (bin as f32 + delta, middle - 0.25 * (left - right) * delta)
}

fn lag_histogram(lags: &[usize]) -> Vec<f32> {
    let scale = (1.0 / (f64::from(LAG_SIGMA) * (2.0 * std::f64::consts::PI).sqrt())) as f32;
    let spread = -2.0 * f64::from(LAG_SIGMA).powi(2);
    let mut histogram = vec![0.0f32; HISTOGRAM_LAGS];
    for &lag in lags {
        for (bin, value) in histogram.iter_mut().enumerate() {
            let offset = bin as f64 - lag as f64;
            *value += (f64::from(scale) * (offset * offset / spread).exp()) as f32;
        }
    }
    histogram
}

/// Index of the first maximum.
fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().fold(
        0,
        |best, (index, value)| if *value > values[best] { index } else { best },
    )
}

/// The paper's octave decider: three linear SVMs trained by its authors, over
/// the histogram energy below the lag, the energy around half the lag and the
/// tempo itself, each min–max normalised. Returns the tempo multiplier: ×2,
/// ×0.5 or ×1.
fn octave(histogram: &[f32], lag: usize, bpm: f32) -> f32 {
    const TOLERANCE: f32 = 10.0;
    // The trained second minimum is 1.68126e-83, zero in f32.
    const MINS: [f32; 3] = [0.0321812, 0.0, 50.1745];
    const MAXS: [f32; 3] = [0.863237, 0.449184, 208.807];
    // Pairwise classifiers; a positive decision favours the second option.
    const HALVE_OR_KEEP: [f32; 4] = [-1.9551, 0.4348, -4.6442, 3.2896];
    const HALVE_OR_DOUBLE: [f32; 4] = [-3.0408, 2.7591, -6.5367, 3.081];
    const KEEP_OR_DOUBLE: [f32; 4] = [-3.4624, 3.4397, -9.4897, 1.6297];
    let lag = lag as f32;
    let total = energy(histogram, 0.0, (histogram.len() - 1) as f32, 1.0);
    let features = [
        energy(histogram, 0.0, lag - TOLERANCE, 1.0 / total),
        energy(
            histogram,
            0.5 * lag - TOLERANCE,
            0.5 * lag + TOLERANCE,
            1.0 / total,
        ),
        bpm,
    ];
    let normalised: Vec<f32> = (0..3)
        .map(|i| (features[i] - MINS[i]) / (MAXS[i] - MINS[i]))
        .collect();
    let decide = |weights: [f32; 4]| {
        weights[3]
            + normalised[0] * weights[0]
            + normalised[1] * weights[1]
            + normalised[2] * weights[2]
    };
    if decide(HALVE_OR_DOUBLE) > 0.0 && decide(KEEP_OR_DOUBLE) > 0.0 {
        2.0
    } else if decide(HALVE_OR_KEEP) <= 0.0 && decide(HALVE_OR_DOUBLE) <= 0.0 {
        0.5
    } else {
        1.0
    }
}

/// `scale · Σ values[round(low) ..= round(high)]`, clamped to the histogram.
fn energy(values: &[f32], low: f32, high: f32, scale: f32) -> f32 {
    let low = (low.round() as isize).max(0) as usize;
    let high = (high.round() as isize).min(values.len() as isize - 1);
    if high < low as isize {
        return 0.0;
    }
    scale * values[low..=high as usize].iter().sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_bpm::test_signal::click_loop;

    /// Reference estimates for the same 32-beat click loops, computed outside
    /// this crate.
    const REFERENCE: [(f64, f64); 4] = [
        (90.0, 89.8777),
        (120.0, 120.1853),
        (128.0, 128.3967),
        (140.0, 139.6748),
    ];

    #[test]
    fn click_loops_match_the_reference_and_their_constructed_tempo() {
        for (tempo, reference) in REFERENCE {
            let frames = AudioFrames::new(click_loop(tempo, 32, SAMPLE_RATE), SAMPLE_RATE);
            let estimate = percival_bpm(&frames).unwrap().unwrap();
            assert!(
                (estimate - reference).abs() < 0.001,
                "{tempo} BPM loop: {estimate}, reference {reference}"
            );
            assert_eq!(estimate.round(), tempo);
        }
    }

    #[test]
    fn other_sample_rates_are_estimated_at_44_1_khz() {
        let frames = AudioFrames::new(click_loop(128.0, 32, 48_000), 48_000);
        let estimate = percival_bpm(&frames).unwrap().unwrap();
        assert_eq!(estimate.round(), 128.0, "{estimate}");
    }

    #[test]
    fn input_shorter_than_one_frame_has_no_estimate() {
        let frames = AudioFrames::new(vec![0.1; FRAME - 1], SAMPLE_RATE);
        assert_eq!(percival_bpm(&frames).unwrap(), None);
    }
}
