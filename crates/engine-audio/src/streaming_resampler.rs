//! Current Swift FIR/polyphase front-end, with bounded streaming history.
//! Both the flat and streaming faces evaluate the same windows in the same
//! order; only unsupported rational ratios use the older Rubato converter.

use ferment_engine_core::{EngineError, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// The file decoder's bounded converter. Small rationals use the current
/// front-end; unusual rates retain the platform-independent Rubato fallback.
pub enum AudioConverter {
    Current(StreamingResampler),
    Fallback(FallbackConverter),
}

impl AudioConverter {
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self> {
        if let Some(current) = StreamingResampler::new(input_rate, output_rate) {
            return Ok(Self::Current(current));
        }
        Ok(Self::Fallback(FallbackConverter::new(
            input_rate,
            output_rate,
        )?))
    }
    pub fn process(&mut self, block: &[f32]) -> Result<Vec<f32>> {
        match self {
            Self::Current(x) => x.process(block),
            Self::Fallback(x) => x.process(block),
        }
    }
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        match self {
            Self::Current(x) => x.finish(),
            Self::Fallback(x) => x.finish(),
        }
    }
}

pub struct FallbackConverter {
    resampler: SincFixedIn<f32>,
    pending: Vec<f32>,
    seen: usize,
    produced: usize,
    ratio: f64,
    finished: bool,
}

impl FallbackConverter {
    fn new(input_rate: u32, output_rate: u32) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 {
            return Err(EngineError::invalid_config("sample rates must be positive"));
        }
        let ratio = f64::from(output_rate) / f64::from(input_rate);
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let resampler = SincFixedIn::new(ratio, 2.0, params, 1024, 1)
            .map_err(|e| EngineError::internal(e.to_string()))?;
        Ok(Self {
            resampler,
            pending: Vec::new(),
            seen: 0,
            produced: 0,
            ratio,
            finished: false,
        })
    }
    fn process(&mut self, block: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(EngineError::invalid_config(
                "converter process after finish",
            ));
        }
        self.seen += block.len();
        self.pending.extend_from_slice(block);
        let mut output = Vec::new();
        while self.pending.len() >= 1024 {
            let result = self
                .resampler
                .process(&[&self.pending[..1024]], None)
                .map_err(|e| EngineError::internal(e.to_string()))?;
            output.extend_from_slice(&result[0]);
            self.pending.drain(..1024);
        }
        self.produced += output.len();
        Ok(output)
    }
    fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Err(EngineError::invalid_config("converter finish twice"));
        }
        self.finished = true;
        let target = (self.seen as f64 * self.ratio).round() as usize;
        let needed = target.saturating_sub(self.produced);
        let mut output = Vec::new();
        if !self.pending.is_empty() {
            self.pending.resize(1024, 0.0);
            let result = self
                .resampler
                .process_partial(Some(&[&self.pending[..]]), None)
                .map_err(|e| EngineError::internal(e.to_string()))?;
            output.extend_from_slice(&result[0]);
            self.pending.clear();
        }
        while output.len() < needed {
            let result = self
                .resampler
                .process_partial::<Vec<f32>>(None, None)
                .map_err(|e| EngineError::internal(e.to_string()))?;
            if result[0].is_empty() {
                return Err(EngineError::internal(
                    "converter did not drain its final content",
                ));
            }
            output.extend_from_slice(&result[0]);
        }
        output.truncate(needed);
        self.produced += output.len();
        Ok(output)
    }
}

#[derive(Debug, Clone)]
enum Kernel {
    Identity,
    Decimate {
        factor: usize,
        taps: Vec<f32>,
    },
    Polyphase {
        up: usize,
        down: usize,
        delay: usize,
        phases: Vec<Vec<f32>>,
        longest: usize,
    },
}

#[derive(Debug)]
pub struct StreamingResampler {
    kernel: Kernel,
    pending: Vec<f32>,
    base: isize,
    pub seen: usize,
    pub produced: usize,
    finished: bool,
}

impl StreamingResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Option<Self> {
        if input_rate == 0 || output_rate == 0 {
            return None;
        }
        let (kernel, pad) = if input_rate == output_rate {
            (Kernel::Identity, 0)
        } else if input_rate > output_rate && input_rate.is_multiple_of(output_rate) {
            let factor = (input_rate / output_rate) as usize;
            let nyquist = f64::from(output_rate) / 2.0;
            let length = length(0.1 * nyquist, f64::from(input_rate)) | 1;
            let taps = taps(length, 0.95 * nyquist, f64::from(input_rate));
            (Kernel::Decimate { factor, taps }, (length - 1) / 2)
        } else {
            let divisor = gcd(input_rate, output_rate);
            let up = (output_rate / divisor) as usize;
            let down = (input_rate / divisor) as usize;
            if up > 1000 || down > 1000 {
                return None;
            }
            let high = f64::from(input_rate) * up as f64;
            let nyquist = f64::from(input_rate.min(output_rate)) / 2.0;
            let estimate = length(0.1 * nyquist, high);
            let delay = (estimate - 1).div_ceil(2 * down);
            let size = 2 * delay * down + 1;
            let prototype = taps(size, 0.95 * nyquist, high);
            let phases: Vec<Vec<f32>> = (0..up)
                .map(|phase| {
                    let count = (size - phase).div_ceil(up);
                    (0..count)
                        .map(|i| prototype[phase + (count - 1 - i) * up] * up as f32)
                        .collect()
                })
                .collect();
            let longest = phases.iter().map(Vec::len).max().unwrap_or(1);
            (
                Kernel::Polyphase {
                    up,
                    down,
                    delay,
                    phases,
                    longest,
                },
                longest - 1,
            )
        };
        Some(Self {
            kernel,
            pending: vec![0.0; pad],
            base: -(pad as isize),
            seen: 0,
            produced: 0,
            finished: false,
        })
    }

    pub fn process(&mut self, block: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(EngineError::invalid_config(
                "resampler process after finish",
            ));
        }
        if block.is_empty() {
            return Ok(Vec::new());
        }
        self.seen += block.len();
        if matches!(self.kernel, Kernel::Identity) {
            self.produced += block.len();
            return Ok(block.to_vec());
        }
        self.pending.extend_from_slice(block);
        let available = match &self.kernel {
            Kernel::Identity => self.seen,
            Kernel::Decimate { factor, taps } => self
                .seen
                .checked_sub(1 + (taps.len() - 1) / 2)
                .map_or(0, |last| last / factor + 1),
            Kernel::Polyphase {
                up, down, delay, ..
            } => ((self.seen * up - 1) / down + 1).saturating_sub(*delay),
        };
        Ok(self.emit(available))
    }

    pub fn finish(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Err(EngineError::invalid_config("resampler finish twice"));
        }
        self.finished = true;
        let (target, pad) = match &self.kernel {
            Kernel::Identity => return Ok(Vec::new()),
            Kernel::Decimate { factor, taps } => (self.seen.div_ceil(*factor), taps.len()),
            Kernel::Polyphase {
                up, down, longest, ..
            } => ((self.seen * up).div_ceil(*down), *longest),
        };
        self.pending.resize(self.pending.len() + pad, 0.0);
        Ok(self.emit(target))
    }

    pub fn retained_samples(&self) -> usize {
        self.pending.len()
    }

    fn emit(&mut self, target: usize) -> Vec<f32> {
        if target <= self.produced {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(target - self.produced);
        for index in self.produced..target {
            let (start, filter) = match &self.kernel {
                Kernel::Identity => unreachable!(),
                Kernel::Decimate { factor, taps } => (
                    (index * factor) as isize - ((taps.len() - 1) / 2) as isize,
                    taps,
                ),
                Kernel::Polyphase {
                    up,
                    down,
                    delay,
                    phases,
                    ..
                } => {
                    let time = (index + delay) * down;
                    let phase = &phases[time % up];
                    ((time / up) as isize - (phase.len() - 1) as isize, phase)
                }
            };
            let offset = usize::try_from(start - self.base).expect("retained resampler window");
            let value = self.pending[offset..offset + filter.len()]
                .iter()
                .zip(filter)
                .map(|(x, h)| x * h)
                .sum();
            out.push(value);
        }
        self.produced = target;
        let next_base = match &self.kernel {
            Kernel::Identity => unreachable!(),
            Kernel::Decimate { factor, taps } => {
                (target * factor) as isize - ((taps.len() - 1) / 2) as isize
            }
            Kernel::Polyphase {
                up,
                down,
                delay,
                longest,
                ..
            } => (((target + delay) * down) / up) as isize - (longest - 1) as isize,
        };
        let drop = (next_base - self.base).max(0) as usize;
        self.pending.drain(..drop.min(self.pending.len()));
        self.base = next_base;
        out
    }
}

pub fn resample_current(samples: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
    let Some(mut resampler) = StreamingResampler::new(input_rate, output_rate) else {
        return crate::resample_sinc(samples, input_rate, output_rate);
    };
    let mut output = resampler.process(samples)?;
    output.extend(resampler.finish()?);
    Ok(output)
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}
fn length(transition: f64, rate: f64) -> usize {
    ((100.0 - 8.0) / (2.285 * 2.0 * std::f64::consts::PI * transition / rate)).ceil() as usize
}
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    loop {
        term *= x * x / 4.0 / (k * k);
        sum += term;
        if term < sum * 1e-16 {
            return sum;
        }
        k += 1.0;
    }
}
fn taps(length: usize, cutoff: f64, rate: f64) -> Vec<f32> {
    let beta = 0.1102 * (100.0 - 8.7);
    let middle = (length - 1) as f64 / 2.0;
    let normalized = 2.0 * cutoff / rate;
    let denominator = bessel_i0(beta);
    let mut kernel: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 - middle;
            let angle = std::f64::consts::PI * normalized * t;
            let sinc = if t == 0.0 { 1.0 } else { angle.sin() / angle };
            let r = 2.0 * t / (length - 1) as f64;
            normalized * sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denominator
        })
        .collect();
    let gain: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|v| *v /= gain);
    kernel.into_iter().map(|v| v as f32).collect()
}

#[cfg(test)]
mod design_tests;
