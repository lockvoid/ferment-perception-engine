//! Swappable frame-batch STFT/mel frontend.
//!
//! Keeping FFT, magnitude/power conversion, and mel projection behind one
//! boundary lets platform fast paths own their native spectrum layout. The
//! portable implementation remains the correctness reference on every target.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

use ferment_engine_core::{EngineError, Result};

use crate::{MelFilterbank, Padding, StftConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumMode {
    Magnitude,
    Power,
}

pub trait StftMelFrontend {
    fn analyze(&mut self, samples: &[f32], padding: Padding) -> Result<Vec<Vec<f32>>>;
}

pub struct PortableStftMelFrontend {
    config: StftConfig,
    filterbank: MelFilterbank,
    mode: SpectrumMode,
    spectrum_scale: f32,
    r2c: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    spectrum: Vec<f32>,
}

impl PortableStftMelFrontend {
    pub fn new(
        config: StftConfig,
        filterbank: MelFilterbank,
        mode: SpectrumMode,
        spectrum_scale: f32,
    ) -> Result<Self> {
        if filterbank.n_freqs != config.n_freq() {
            return Err(EngineError::invalid_config(format!(
                "filterbank n_freqs {} != STFT n_freq {}",
                filterbank.n_freqs,
                config.n_freq()
            )));
        }
        if !spectrum_scale.is_finite() {
            return Err(EngineError::invalid_config("spectrum scale must be finite"));
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(config.fft_size);
        let fft_input = r2c.make_input_vec();
        let fft_output = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();
        let spectrum = vec![0.0; config.n_freq()];
        Ok(Self {
            config,
            filterbank,
            mode,
            spectrum_scale,
            r2c,
            fft_input,
            fft_output,
            scratch,
            spectrum,
        })
    }

    fn padded_sample(samples: &[f32], position: usize, pad: usize, padding: Padding) -> f32 {
        if (pad..pad + samples.len()).contains(&position) {
            return samples[position - pad];
        }
        if padding == Padding::Zero || samples.is_empty() {
            return 0.0;
        }
        if position < pad {
            samples[(pad - position).min(samples.len() - 1)]
        } else {
            let offset = position - (pad + samples.len());
            let source = (samples.len() as isize - 2 - offset as isize).max(0) as usize;
            samples[source]
        }
    }
}

impl StftMelFrontend for PortableStftMelFrontend {
    fn analyze(&mut self, samples: &[f32], padding: Padding) -> Result<Vec<Vec<f32>>> {
        let pad = self.config.fft_size / 2;
        let padded_len = samples.len() + 2 * pad;
        if padded_len < self.config.fft_size {
            return Err(EngineError::AudioTooShort {
                samples: samples.len(),
                min_required: self.config.fft_size - 2 * pad,
            });
        }
        let n_frames = (padded_len - self.config.fft_size) / self.config.hop_size + 1;
        let mut output = Vec::with_capacity(n_frames);
        for frame_index in 0..n_frames {
            let start = frame_index * self.config.hop_size;
            let source_start = start.checked_sub(pad);
            let interior = source_start
                .and_then(|source_start| source_start.checked_add(self.config.fft_size))
                .is_some_and(|source_end| source_end <= samples.len());
            if interior {
                let source_start = source_start.unwrap_or(0);
                let source = &samples[source_start..source_start + self.config.fft_size];
                for ((input, sample), window) in self
                    .fft_input
                    .iter_mut()
                    .zip(source)
                    .zip(&self.config.window)
                {
                    *input = sample * window;
                }
            } else {
                for (offset, input) in self.fft_input.iter_mut().enumerate() {
                    *input = Self::padded_sample(samples, start + offset, pad, padding)
                        * self.config.window[offset];
                }
            }
            self.r2c
                .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.scratch)
                .map_err(|error| EngineError::StftFailed(error.to_string()))?;
            match self.mode {
                SpectrumMode::Magnitude => {
                    for (value, complex) in self.spectrum.iter_mut().zip(&self.fft_output) {
                        *value = complex.norm() * self.spectrum_scale;
                    }
                }
                SpectrumMode::Power => {
                    for (value, complex) in self.spectrum.iter_mut().zip(&self.fft_output) {
                        *value = complex.norm_sqr() * self.spectrum_scale;
                    }
                }
            }
            let mut mel = vec![0.0; self.filterbank.n_mels];
            self.filterbank.apply_into(&self.spectrum, &mut mel)?;
            output.push(mel);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::{MelNorm, MelOptions, MelScale, stft_centered_with, window};

    #[test]
    fn fused_frontend_matches_composed_portable_path() {
        let samples = (0..22_050)
            .map(|index| (index as f32 * 0.03125).sin())
            .collect::<Vec<_>>();
        let config = StftConfig::new(2_048, 512, window::hann_periodic(2_048)).unwrap();
        let filterbank = MelFilterbank::new(MelOptions {
            n_mels: 128,
            n_fft: 2_048,
            sample_rate: 22_050,
            f_min: 20.0,
            f_max: 11_025.0,
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
        })
        .unwrap();
        let frames = stft_centered_with(&samples, &config, Padding::Reflect).unwrap();
        let expected = frames
            .iter()
            .map(|frame| {
                let power = frame.iter().map(Complex::norm_sqr).collect::<Vec<_>>();
                filterbank.apply(&power).unwrap()
            })
            .collect::<Vec<_>>();
        let mut frontend =
            PortableStftMelFrontend::new(config, filterbank, SpectrumMode::Power, 1.0).unwrap();
        let actual = frontend.analyze(&samples, Padding::Reflect).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
        }
    }
}
