use std::f64::consts::PI;
use std::path::Path;

use async_trait::async_trait;
use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{
    AudioBuffer, LoudnessNormalizing, LoudnessReport, LoudnessTarget, NormalizedAudio,
    ProgressSink, Result,
};

use crate::audio_source::AudioSource;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

#[derive(Debug, Clone)]
pub struct LoudnessKWeighting {
    pub stage1: BiquadCoefficients,
    pub stage2: BiquadCoefficients,
    state1: [f64; 4],
    state2: [f64; 4],
}

impl LoudnessKWeighting {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            stage1: Self::high_shelf(
                sample_rate,
                1_681.974_450_955_533,
                0.707_175_236_955_419_6,
                3.999_843_853_973_347,
            ),
            stage2: Self::high_pass(sample_rate, 38.135_470_876_024_44, 0.500_327_037_323_877_3),
            state1: [0.0; 4],
            state2: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.state1 = [0.0; 4];
        self.state2 = [0.0; 4];
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let stage1 = filter_sample(f64::from(*sample), self.stage1, &mut self.state1);
            let stage2 = filter_sample(stage1, self.stage2, &mut self.state2);
            *sample = stage2 as f32;
        }
    }

    pub fn high_shelf(
        sample_rate: f64,
        frequency: f64,
        q: f64,
        gain_db: f64,
    ) -> BiquadCoefficients {
        let k = (PI * frequency / sample_rate).tan();
        let vh = 10_f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        BiquadCoefficients {
            b0: (vh + vb * k / q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        }
    }

    pub fn high_pass(sample_rate: f64, frequency: f64, q: f64) -> BiquadCoefficients {
        let k = (PI * frequency / sample_rate).tan();
        let a0 = 1.0 + k / q + k * k;
        BiquadCoefficients {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        }
    }
}

fn filter_sample(input: f64, coefficients: BiquadCoefficients, state: &mut [f64; 4]) -> f64 {
    let output = coefficients.b0 * input + coefficients.b1 * state[0] + coefficients.b2 * state[1]
        - coefficients.a1 * state[2]
        - coefficients.a2 * state[3];
    state[1] = state[0];
    state[0] = input;
    state[3] = state[2];
    state[2] = output;
    output
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoudnessMeter;

impl LoudnessMeter {
    pub const BLOCK_SECONDS: f64 = 0.4;
    pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
    pub const RELATIVE_GATE_LU: f64 = -10.0;
    pub const K_WEIGHTING_OFFSET_DB: f64 = -0.691;

    pub fn measure_lufs(samples: &[f32], sample_rate: f64) -> f64 {
        let block_samples = (Self::BLOCK_SECONDS * sample_rate) as usize;
        let stride_samples = (Self::BLOCK_SECONDS * 0.25 * sample_rate) as usize;
        if sample_rate <= 0.0 || samples.len() < block_samples || stride_samples == 0 {
            return f64::NEG_INFINITY;
        }
        let mut weighted = samples.to_vec();
        LoudnessKWeighting::new(sample_rate).process(&mut weighted);
        let mut cumulative_energy = Vec::with_capacity(weighted.len() + 1);
        cumulative_energy.push(0.0_f64);
        for sample in weighted {
            let sample = f64::from(sample);
            cumulative_energy
                .push(cumulative_energy.last().copied().unwrap_or(0.0) + sample * sample);
        }
        let mean_squares = (0..=cumulative_energy.len() - 1 - block_samples)
            .step_by(stride_samples)
            .map(|start| {
                (cumulative_energy[start + block_samples] - cumulative_energy[start])
                    / block_samples as f64
            })
            .collect::<Vec<_>>();
        Self::integrate(&mean_squares)
    }

    pub fn integrate(mean_squares: &[f64]) -> f64 {
        let absolute = Self::ms_for_loudness(Self::ABSOLUTE_GATE_LUFS);
        let stage1 = mean_squares
            .iter()
            .copied()
            .filter(|value| *value > absolute)
            .collect::<Vec<_>>();
        if stage1.is_empty() {
            return f64::NEG_INFINITY;
        }
        let provisional = stage1.iter().sum::<f64>() / stage1.len() as f64;
        let relative =
            Self::ms_for_loudness(Self::loudness_for_ms(provisional) + Self::RELATIVE_GATE_LU);
        let stage2 = stage1
            .into_iter()
            .filter(|value| *value > relative)
            .collect::<Vec<_>>();
        if stage2.is_empty() {
            return f64::NEG_INFINITY;
        }
        Self::loudness_for_ms(stage2.iter().sum::<f64>() / stage2.len() as f64)
    }

    pub fn loudness_for_ms(mean_square: f64) -> f64 {
        if mean_square <= 0.0 {
            f64::NEG_INFINITY
        } else {
            Self::K_WEIGHTING_OFFSET_DB + 10.0 * mean_square.log10()
        }
    }

    pub fn ms_for_loudness(lufs: f64) -> f64 {
        if lufs == f64::NEG_INFINITY {
            0.0
        } else {
            10_f64.powf((lufs - Self::K_WEIGHTING_OFFSET_DB) / 10.0)
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TruePeakMeter;

impl TruePeakMeter {
    const FACTOR: usize = 4;
    const TAPS_PER_PHASE: usize = 12;

    pub fn measure_dbtp(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return f64::NEG_INFINITY;
        }
        let peak = pulp::Arch::new().dispatch(TruePeakKernel {
            samples,
            taps: true_peak_taps(),
        });
        if peak == 0.0 {
            f64::NEG_INFINITY
        } else {
            20.0 * peak.log10()
        }
    }
}

struct TruePeakKernel<'a> {
    samples: &'a [f32],
    taps: &'a [[f64; TruePeakMeter::FACTOR]; TruePeakMeter::TAPS_PER_PHASE],
}

impl pulp::WithSimd for TruePeakKernel<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let mut peak = 0.0_f64;
        for index in 0..self.samples.len() {
            let tap_count = (index + 1).min(TruePeakMeter::TAPS_PER_PHASE);
            let mut values = [0.0_f64; TruePeakMeter::FACTOR];
            for tap_index in 0..tap_count {
                let scalar_sample = f64::from(self.samples[index - tap_index]);
                let sample = simd.splat_f64s(scalar_sample);
                let (values_head, values_tail) = S::as_mut_simd_f64s(&mut values);
                let (taps_head, taps_tail) = S::as_simd_f64s(&self.taps[tap_index]);
                for (value, tap) in values_head.iter_mut().zip(taps_head) {
                    *value = simd.add_f64s(*value, simd.mul_f64s(sample, *tap));
                }
                for (value, tap) in values_tail.iter_mut().zip(taps_tail) {
                    *value += scalar_sample * tap;
                }
            }
            for value in values {
                peak = peak.max(value.abs());
            }
        }
        peak
    }
}

fn true_peak_taps() -> &'static [[f64; TruePeakMeter::FACTOR]; TruePeakMeter::TAPS_PER_PHASE] {
    use std::sync::OnceLock;

    static PHASES: OnceLock<[[f64; TruePeakMeter::FACTOR]; TruePeakMeter::TAPS_PER_PHASE]> =
        OnceLock::new();
    PHASES.get_or_init(|| {
        const TOTAL: usize = TruePeakMeter::FACTOR * TruePeakMeter::TAPS_PER_PHASE;
        let middle = (TOTAL - 1) as f64;
        let cutoff = 1.0 / (2 * TruePeakMeter::FACTOR) as f64;
        let mut taps = [0.0; TOTAL];
        for (index, tap) in taps.iter_mut().enumerate() {
            let offset = index as f64 - middle / 2.0;
            let argument = 2.0 * cutoff * offset;
            let sinc = if argument.abs() < 1e-12 {
                1.0
            } else {
                (PI * argument).sin() / (PI * argument)
            };
            let window = 0.5 - 0.5 * (2.0 * PI * index as f64 / middle).cos();
            *tap = 2.0 * cutoff * sinc * window * TruePeakMeter::FACTOR as f64;
        }
        let mut phases = [[0.0; TruePeakMeter::FACTOR]; TruePeakMeter::TAPS_PER_PHASE];
        for (tap_index, phase) in phases.iter_mut().enumerate() {
            for (phase_index, tap) in phase.iter_mut().enumerate() {
                *tap = taps[phase_index + tap_index * TruePeakMeter::FACTOR];
            }
        }
        phases
    })
}

#[derive(Debug, Clone)]
pub struct LoudnessLimiter {
    pub ceiling_linear: f32,
    pub attack_coefficient: f32,
    pub release_coefficient: f32,
    current_gain: f32,
    pub engaged: bool,
}

impl LoudnessLimiter {
    pub fn new(sample_rate: f64, ceiling_dbfs: f64, attack_ms: f64, release_ms: f64) -> Self {
        Self {
            ceiling_linear: 10_f32.powf(ceiling_dbfs as f32 / 20.0),
            attack_coefficient: (1.0 - (-1.0 / (attack_ms / 1000.0 * sample_rate)).exp()) as f32,
            release_coefficient: (1.0 - (-1.0 / (release_ms / 1000.0 * sample_rate)).exp()) as f32,
            current_gain: 1.0,
            engaged: false,
        }
    }

    pub fn standard(sample_rate: f64, ceiling_dbfs: f64) -> Self {
        Self::new(sample_rate, ceiling_dbfs, 0.2, 50.0)
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let magnitude = sample.abs();
            let required = if magnitude > self.ceiling_linear {
                self.engaged = true;
                self.ceiling_linear / magnitude
            } else {
                1.0
            };
            let coefficient = if required < self.current_gain {
                self.attack_coefficient
            } else {
                self.release_coefficient
            };
            self.current_gain += (required - self.current_gain) * coefficient;
            *sample *= self.current_gain;
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoudnessLeveler {
    target_ms: f32,
    envelope_coefficient: f32,
    gain_attack_coefficient: f32,
    gain_release_coefficient: f32,
    min_gain: f32,
    max_gain: f32,
    silence_ms_gate: f32,
    ms_envelope: f32,
    current_gain: f32,
    pub engaged: bool,
}

impl LoudnessLeveler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f64,
        target_rms_dbfs: f64,
        envelope_ms: f64,
        gain_attack_ms: f64,
        gain_release_ms: f64,
        max_boost_db: f64,
        max_cut_db: f64,
        silence_gate_dbfs: f64,
    ) -> Self {
        let target = 10_f32.powf(target_rms_dbfs as f32 / 20.0);
        let silence = 10_f32.powf(silence_gate_dbfs as f32 / 20.0);
        Self {
            target_ms: target * target,
            envelope_coefficient: time_coefficient(envelope_ms, sample_rate),
            gain_attack_coefficient: time_coefficient(gain_attack_ms, sample_rate),
            gain_release_coefficient: time_coefficient(gain_release_ms, sample_rate),
            min_gain: 10_f32.powf(-max_cut_db as f32 / 20.0),
            max_gain: 10_f32.powf(max_boost_db as f32 / 20.0),
            silence_ms_gate: silence * silence,
            ms_envelope: 0.0,
            current_gain: 1.0,
            engaged: false,
        }
    }

    pub fn standard(sample_rate: f64) -> Self {
        Self::new(sample_rate, -20.0, 500.0, 100.0, 800.0, 0.0, 6.0, -45.0)
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            self.ms_envelope += (*sample * *sample - self.ms_envelope) * self.envelope_coefficient;
            if self.ms_envelope > self.silence_ms_gate {
                let required = (self.target_ms / self.ms_envelope.max(1e-12))
                    .sqrt()
                    .clamp(self.min_gain, self.max_gain);
                self.engaged |= required != 1.0;
                let coefficient = if required < self.current_gain {
                    self.gain_attack_coefficient
                } else {
                    self.gain_release_coefficient
                };
                self.current_gain += (required - self.current_gain) * coefficient;
            }
            *sample *= self.current_gain;
        }
    }
}

fn time_coefficient(milliseconds: f64, sample_rate: f64) -> f32 {
    (1.0 - (-1.0 / (milliseconds / 1000.0 * sample_rate)).exp()) as f32
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoudnessNormalizer;

impl LoudnessNormalizer {
    pub const TARGET_SAMPLE_RATE: u32 = 48_000;
    pub const SAFETY_MARGIN_DB: f64 = 1.5;

    pub async fn normalize_frames(
        &self,
        frames: AudioFrames,
        target: LoudnessTarget,
        progress: ProgressSink,
    ) -> Result<NormalizedAudio> {
        if frames.samples.is_empty() {
            return Err(ferment_engine_core::EngineError::NoAudioData);
        }
        progress.report(0.05, "preparing audio").await;
        let mut working = if frames.sample_rate == Self::TARGET_SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(
                &frames.samples,
                frames.sample_rate,
                Self::TARGET_SAMPLE_RATE,
            )?
        };
        let mut leveler = target
            .enable_leveler
            .then(|| LoudnessLeveler::standard(f64::from(Self::TARGET_SAMPLE_RATE)));
        if let Some(leveler) = &mut leveler {
            progress.report(0.15, "leveling speech dynamics").await;
            leveler.process(&mut working);
        }

        progress.report(0.25, "measuring loudness").await;
        let sample_rate = f64::from(Self::TARGET_SAMPLE_RATE);
        let measured_lufs = LoudnessMeter::measure_lufs(&working, sample_rate);
        let measured_true_peak = TruePeakMeter::measure_dbtp(&working);
        let (planned_gain_db, gain_applied_db) = if !measured_lufs.is_finite()
            || measured_lufs <= LoudnessMeter::ABSOLUTE_GATE_LUFS
        {
            (0.0, 0.0)
        } else {
            let planned = target.target_lufs - measured_lufs;
            let applied = if measured_true_peak.is_finite() {
                planned.min(target.peak_ceiling_dbtp - Self::SAFETY_MARGIN_DB - measured_true_peak)
            } else {
                planned
            };
            (planned, applied)
        };

        progress.report(0.60, "applying gain").await;
        let gain = 10_f32.powf(gain_applied_db as f32 / 20.0);
        for sample in &mut working {
            *sample *= gain;
        }
        let mut limiter = LoudnessLimiter::standard(
            sample_rate,
            target.peak_ceiling_dbtp - Self::SAFETY_MARGIN_DB,
        );
        limiter.process(&mut working);

        progress.report(0.90, "verifying").await;
        let post_lufs = LoudnessMeter::measure_lufs(&working, sample_rate);
        let post_peak = TruePeakMeter::measure_dbtp(&working);
        let in_tolerance = (post_lufs - target.target_lufs).abs() <= 1.0;
        let peak_capped = gain_applied_db < planned_gain_db - 1e-6;
        let accepted_undershoot =
            (limiter.engaged || peak_capped) && post_lufs <= target.target_lufs;
        let silent =
            !measured_lufs.is_finite() || measured_lufs <= LoudnessMeter::ABSOLUTE_GATE_LUFS;
        let report = LoudnessReport {
            spec_version: 1,
            measured_lufs,
            measured_true_peak,
            gain_applied_db,
            post_limit_lufs: post_lufs,
            post_limit_true_peak: post_peak,
            limiter_engaged: limiter.engaged,
            within_clip_leveler_engaged: leveler.is_some_and(|leveler| leveler.engaged),
            qa_pass: in_tolerance || accepted_undershoot || silent,
            target_lufs: target.target_lufs,
            peak_ceiling_dbtp: target.peak_ceiling_dbtp,
        };
        progress.report(1.0, "done").await;
        Ok(NormalizedAudio {
            audio: AudioBuffer::mono(working, Self::TARGET_SAMPLE_RATE),
            report,
        })
    }
}

#[async_trait(?Send)]
impl LoudnessNormalizing for LoudnessNormalizer {
    async fn normalize(
        &self,
        audio_path: &Path,
        target: LoudnessTarget,
        progress: ProgressSink,
    ) -> Result<NormalizedAudio> {
        self.normalize_frames(
            AudioSource::default().decode(audio_path).await?,
            target,
            progress,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48_000.0;

    fn sine(amplitude: f32, frequency: f64, seconds: f64) -> Vec<f32> {
        sine_with_phase(amplitude, frequency, seconds, 0.0)
    }

    fn sine_with_phase(amplitude: f32, frequency: f64, seconds: f64, phase: f64) -> Vec<f32> {
        (0..(SAMPLE_RATE * seconds) as usize)
            .map(|index| {
                amplitude * (2.0 * PI * frequency * index as f64 / SAMPLE_RATE + phase).sin() as f32
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f64 {
        (samples
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
    }

    fn normalize_sine(amplitude: f32) -> NormalizedAudio {
        futures_executor::block_on(LoudnessNormalizer.normalize_frames(
            AudioFrames::new(sine(amplitude, 1_000.0, 5.0), SAMPLE_RATE as u32),
            LoudnessTarget::STANDARD,
            ProgressSink::noop(),
        ))
        .unwrap()
    }

    // LoudnessNormalizerPipelineTests.swift. The Swift integration bundle uses
    // MP3/WAV assets; deterministic PCM signals preserve the report assertions
    // while remaining portable under the license-clean decoder policy.

    #[test]
    fn test_normalizes_music_within_tolerance() {
        let result = futures_executor::block_on(LoudnessNormalizer.normalize_frames(
            AudioFrames::new(sine(0.18, 997.0, 5.0), SAMPLE_RATE as u32),
            LoudnessTarget::STANDARD,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(result.report.qa_pass);
        assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
    }

    #[test]
    fn test_normalizes_speech_with_leveler() {
        let samples = (0..(SAMPLE_RATE * 5.0) as usize)
            .map(|index| {
                let second = index / SAMPLE_RATE as usize;
                let amplitude = if second.is_multiple_of(2) { 0.015 } else { 0.3 };
                amplitude * (2.0 * PI * 180.0 * index as f64 / SAMPLE_RATE).sin() as f32
            })
            .collect();
        let result = futures_executor::block_on(LoudnessNormalizer.normalize_frames(
            AudioFrames::new(samples, SAMPLE_RATE as u32),
            LoudnessTarget::SPEECH,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(
            result.report.within_clip_leveler_engaged
                || result.report.measured_lufs == f64::NEG_INFINITY
        );
        assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
    }

    #[test]
    fn test_normalizes_transient_fx_respects_ceiling() {
        let mut samples = sine(0.01, 440.0, 5.0);
        for start in (0..samples.len()).step_by(SAMPLE_RATE as usize / 2) {
            for offset in 0..128 {
                if let Some(sample) = samples.get_mut(start + offset) {
                    let window = 0.5 - 0.5 * (2.0 * PI * offset as f64 / 127.0).cos();
                    *sample += (0.95 * window) as f32;
                }
            }
        }
        let result = futures_executor::block_on(LoudnessNormalizer.normalize_frames(
            AudioFrames::new(samples, SAMPLE_RATE as u32),
            LoudnessTarget::STANDARD,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
        assert!(result.report.post_limit_lufs > -40.0);
    }

    // LoudnessKWeightingTests.swift

    mod loudness_k_weighting_tests {
        use super::*;

        #[test]
        fn stage1_matches_published_48k_coefficients() {
            let weighting = LoudnessKWeighting::new(SAMPLE_RATE);
            assert!((weighting.stage1.b0 - 1.535_124_859_586_97).abs() <= 1e-5);
            assert!((weighting.stage1.b1 - -2.691_696_189_406_38).abs() <= 1e-5);
            assert!((weighting.stage1.b2 - 1.198_392_810_852_85).abs() <= 1e-5);
            assert!((weighting.stage1.a1 - -1.690_659_293_182_41).abs() <= 1e-5);
            assert!((weighting.stage1.a2 - 0.732_480_774_215_85).abs() <= 1e-5);
        }

        #[test]
        fn stage2_matches_published_48k_coefficients() {
            let weighting = LoudnessKWeighting::new(SAMPLE_RATE);
            assert!((weighting.stage2.b0 - 1.0).abs() <= 1e-5);
            assert!((weighting.stage2.b1 - -2.0).abs() <= 1e-5);
            assert!((weighting.stage2.b2 - 1.0).abs() <= 1e-5);
            assert!((weighting.stage2.a1 - -1.990_047_454_833_98).abs() <= 1e-5);
            assert!((weighting.stage2.a2 - 0.990_072_250_366_21).abs() <= 1e-5);
        }

        #[test]
        fn gain_near_unity_at_one_k_hz() {
            let input = sine(0.1, 1_000.0, 1.0);
            let input_rms = rms(&input);
            let mut output = input;
            LoudnessKWeighting::new(SAMPLE_RATE).process(&mut output);
            let output_rms = rms(&output[(0.1 * SAMPLE_RATE) as usize..]);
            let gain_db = 20.0 * (output_rms / input_rms).log10();
            assert!((gain_db - 0.0).abs() <= 1.0, "gain: {gain_db} dB");
        }

        #[test]
        fn strong_attenuation_at_twenty_hz() {
            let input = sine(0.1, 20.0, 1.0);
            let input_rms = rms(&input);
            let mut output = input;
            LoudnessKWeighting::new(SAMPLE_RATE).process(&mut output);
            let output_rms = rms(&output[(0.2 * SAMPLE_RATE) as usize..]);
            let gain_db = 20.0 * (output_rms / input_rms).log10();
            assert!(gain_db < -10.0, "gain: {gain_db} dB");
        }
    }

    // LoudnessLevelerTests.swift

    mod loudness_leveler_tests {
        use super::*;

        #[test]
        fn boosts_quiet_signal_toward_target() {
            let mut leveler =
                LoudnessLeveler::new(SAMPLE_RATE, -20.0, 50.0, 50.0, 100.0, 6.0, 6.0, -45.0);
            let mut buffer = sine(10_f32.powf(-30.0 / 20.0) * 2_f32.sqrt(), 1_000.0, 2.0);
            let input_rms = rms(&buffer);
            leveler.process(&mut buffer);
            let output_rms = rms(&buffer[buffer.len() - (SAMPLE_RATE * 0.5) as usize..]);
            let gain_db = 20.0 * (output_rms / input_rms).log10();
            assert!((gain_db - 6.0).abs() <= 1.0, "gain: {gain_db} dB");
            assert!(leveler.engaged);
        }

        #[test]
        fn attenuates_loud_signal_toward_target() {
            let mut leveler =
                LoudnessLeveler::new(SAMPLE_RATE, -20.0, 50.0, 50.0, 100.0, 0.0, 6.0, -45.0);
            let mut buffer = sine(10_f32.powf(-10.0 / 20.0) * 2_f32.sqrt(), 1_000.0, 2.0);
            let input_rms = rms(&buffer);
            leveler.process(&mut buffer);
            let output_rms = rms(&buffer[buffer.len() - (SAMPLE_RATE * 0.5) as usize..]);
            let gain_db = 20.0 * (output_rms / input_rms).log10();
            assert!((gain_db - -6.0).abs() <= 1.0, "gain: {gain_db} dB");
        }

        #[test]
        fn default_leveler_does_not_boost_quiet_signal() {
            let mut leveler =
                LoudnessLeveler::new(SAMPLE_RATE, -20.0, 50.0, 50.0, 100.0, 0.0, 6.0, -45.0);
            let mut buffer = sine(10_f32.powf(-30.0 / 20.0) * 2_f32.sqrt(), 1_000.0, 1.0);
            let input_rms = rms(&buffer);
            leveler.process(&mut buffer);
            let output_rms = rms(&buffer[buffer.len() - (SAMPLE_RATE * 0.3) as usize..]);
            let gain_db = 20.0 * (output_rms / input_rms).log10();
            assert!((gain_db - 0.0).abs() <= 0.1, "gain: {gain_db} dB");
        }

        #[test]
        fn holds_gain_on_silence() {
            let mut leveler = LoudnessLeveler::standard(SAMPLE_RATE);
            let mut buffer = vec![0.0; SAMPLE_RATE as usize];
            leveler.process(&mut buffer);
            assert!(buffer.iter().all(|sample| sample.abs() <= 1e-6));
            assert!(!leveler.engaged);
        }
    }

    // LoudnessLimiterTests.swift

    mod loudness_limiter_tests {
        use super::*;

        #[test]
        fn clamps_steady_state_above_ceiling() {
            let mut limiter = LoudnessLimiter::new(SAMPLE_RATE, -6.0, 0.2, 50.0);
            let mut buffer = vec![1.0; (SAMPLE_RATE * 0.5) as usize];
            limiter.process(&mut buffer);
            let tail = &buffer[buffer.len() - (SAMPLE_RATE * 0.1) as usize..];
            let average = tail.iter().sum::<f32>() / tail.len() as f32;
            let ceiling = 10_f32.powf(-6.0 / 20.0);
            assert!((average - ceiling).abs() <= 0.005);
            assert!(limiter.engaged);
        }

        #[test]
        fn leaves_below_ceiling_signal_untouched() {
            let mut limiter = LoudnessLimiter::new(SAMPLE_RATE, -1.0, 0.2, 50.0);
            let mut buffer = sine(10_f32.powf(-20.0 / 20.0), 1_000.0, 0.2);
            let input = buffer.clone();
            limiter.process(&mut buffer);
            for (index, (actual, expected)) in buffer.iter().zip(input).enumerate() {
                assert!(
                    (*actual - expected).abs() <= 1e-5,
                    "sample {index}: {actual} != {expected}"
                );
            }
            assert!(!limiter.engaged);
        }

        #[test]
        fn streaming_state_persists_across_chunks() {
            let input = sine(1.0, 1_000.0, 0.5);
            let mut monolithic = input.clone();
            let mut chunked = input;
            LoudnessLimiter::standard(SAMPLE_RATE, -6.0).process(&mut monolithic);
            let mut chunked_limiter = LoudnessLimiter::standard(SAMPLE_RATE, -6.0);
            let chunk_size = chunked.len() / 4;
            for chunk in chunked.chunks_mut(chunk_size) {
                chunked_limiter.process(chunk);
            }
            for (index, (whole, split)) in monolithic.iter().zip(chunked).enumerate() {
                assert!(
                    (*whole - split).abs() <= 1e-5,
                    "sample {index}: {whole} != {split}"
                );
            }
        }
    }

    // LoudnessMeterTests.swift

    mod loudness_meter_tests {
        use super::*;

        #[test]
        fn reference_signal_1_k_hz_at_minus_23_dbfs() {
            let lufs = LoudnessMeter::measure_lufs(&sine(0.1, 1_000.0, 20.0), SAMPLE_RATE);
            assert!((lufs - -23.0).abs() <= 0.1, "measured: {lufs} LUFS");
        }

        #[test]
        fn linearity_with_amplitude_scaling() {
            let first = LoudnessMeter::measure_lufs(&sine(0.1, 1_000.0, 5.0), SAMPLE_RATE);
            let second = LoudnessMeter::measure_lufs(&sine(0.2, 1_000.0, 5.0), SAMPLE_RATE);
            assert!(((second - first) - 6.02).abs() <= 0.1);
        }

        #[test]
        fn silent_input_returns_negative_infinity() {
            assert_eq!(
                LoudnessMeter::measure_lufs(&vec![0.0; (SAMPLE_RATE * 5.0) as usize], SAMPLE_RATE),
                f64::NEG_INFINITY
            );
        }

        #[test]
        fn input_shorter_than_one_block_returns_negative_infinity() {
            assert_eq!(
                LoudnessMeter::measure_lufs(&vec![0.1; (SAMPLE_RATE * 0.1) as usize], SAMPLE_RATE),
                f64::NEG_INFINITY
            );
        }

        #[test]
        fn gating_passes_uniform_blocks() {
            let value = LoudnessMeter::ms_for_loudness(-20.0);
            assert!((LoudnessMeter::integrate(&[value; 10]) - -20.0).abs() <= 0.01);
        }

        #[test]
        fn gating_drops_absolute_floor() {
            let strong = LoudnessMeter::ms_for_loudness(-20.0);
            let floor = LoudnessMeter::ms_for_loudness(-80.0);
            let mut blocks = vec![strong; 5];
            blocks.extend(vec![floor; 100]);
            let result = LoudnessMeter::integrate(&blocks);
            assert!((result - -20.0).abs() <= 0.5, "measured: {result} LUFS");
        }

        #[test]
        fn relative_gate_excludes_quiet_passages() {
            let loud = LoudnessMeter::ms_for_loudness(-10.0);
            let quiet = LoudnessMeter::ms_for_loudness(-30.0);
            let mut blocks = vec![loud];
            blocks.extend(vec![quiet; 10]);
            let result = LoudnessMeter::integrate(&blocks);
            assert!(result > -15.0, "measured: {result} LUFS");
        }
    }

    /// EBU Tech 3341 §2.6 "Compliance tests" — the reference vectors for an
    /// EBU-mode integrated-loudness meter, which is the one part of BS.1770-4
    /// this file had no reference coverage for. The unit tests above check the
    /// pieces (biquad coefficients, gate arithmetic); these check the whole
    /// meter against numbers someone else published.
    ///
    /// Two adaptations, both forced and both harmless:
    ///
    /// * Tech 3341 states its signals as **stereo**, and BS.1770 sums the two
    ///   channel mean squares. `measure_lufs` takes one channel, so a mono
    ///   signal must sit 3.01 dB higher to read the same LUFS —
    ///   `amplitude_for_lufs` does that conversion and nothing else.
    /// * Tests 6-8 (5.1 channel weighting) and 9 (short-term window) are not
    ///   applicable: this meter has neither a channel matrix nor a
    ///   short-term/momentary lane. They are left out on purpose rather than
    ///   faked, and the gap is worth closing if either lands.
    ///
    /// Tests 1-5 are the interesting ones anyway: 3 exercises the relative
    /// gate, 4 the absolute gate, and 5 the plain energy mean.
    mod ebu_r128_conformance_tests {
        use super::*;

        /// Tech 3341 tolerance for integrated loudness.
        const TOLERANCE_LU: f64 = 0.1;

        /// Amplitude of a mono 1 kHz sine that reads `lufs` on this meter.
        fn amplitude_for_lufs(lufs: f64) -> f64 {
            // +3.01 dB for the missing second channel; the sine's own
            // peak-to-RMS factor is already inside the LUFS definition.
            10_f64.powf((lufs + 3.010_299_956_639_812) / 20.0)
        }

        /// One continuous 1 kHz tone whose amplitude steps between segments.
        ///
        /// 1 kHz at 48 kHz is exactly 48 samples per cycle and every segment
        /// boundary here lands on a whole number of seconds, so each step falls
        /// on a zero crossing: no click, no phase discontinuity, and nothing for
        /// the K-weighting filter to ring on.
        fn stepped_tone(segments: &[(f64, f64)]) -> Vec<f32> {
            let mut samples = Vec::new();
            let mut index = 0_usize;
            for &(seconds, lufs) in segments {
                let amplitude = amplitude_for_lufs(lufs);
                let count = (seconds * SAMPLE_RATE).round() as usize;
                samples.extend((0..count).map(|offset| {
                    let phase = 2.0 * PI * 1_000.0 * (index + offset) as f64 / SAMPLE_RATE;
                    (amplitude * phase.sin()) as f32
                }));
                index += count;
            }
            samples
        }

        fn assert_integrated(segments: &[(f64, f64)], expected: f64, label: &str) {
            let measured = LoudnessMeter::measure_lufs(&stepped_tone(segments), SAMPLE_RATE);
            assert!(
                (measured - expected).abs() <= TOLERANCE_LU,
                "{label}: measured {measured} LUFS, expected {expected} +/- {TOLERANCE_LU}"
            );
        }

        #[test]
        fn test_1_sine_at_minus_23_lufs() {
            assert_integrated(&[(20.0, -23.0)], -23.0, "EBU 3341 test 1");
        }

        #[test]
        fn test_2_sine_at_minus_33_lufs() {
            assert_integrated(&[(20.0, -33.0)], -33.0, "EBU 3341 test 2");
        }

        /// The -36 LUFS shoulders sit more than 10 LU below the provisional
        /// mean, so the relative gate has to drop them. A meter that skipped
        /// relative gating would read about -24.2 here.
        #[test]
        fn test_3_relative_gate_drops_quiet_shoulders() {
            assert_integrated(
                &[(10.0, -36.0), (60.0, -23.0), (10.0, -36.0)],
                -23.0,
                "EBU 3341 test 3",
            );
        }

        /// Adds -72 LUFS shoulders below the -70 LUFS absolute gate. Both gates
        /// must fire, in order, for this to come back at -23.
        #[test]
        fn test_4_absolute_and_relative_gates_together() {
            assert_integrated(
                &[
                    (10.0, -72.0),
                    (10.0, -36.0),
                    (60.0, -23.0),
                    (10.0, -36.0),
                    (10.0, -72.0),
                ],
                -23.0,
                "EBU 3341 test 4",
            );
        }

        /// Isolates the -70 LUFS absolute threshold, which test 4 does NOT pin
        /// despite its -72 LUFS shoulders: those are also more than 10 LU below
        /// the mean, so the relative gate discards them either way. Deleting the
        /// absolute gate leaves every other test in this module green.
        ///
        /// Content that is entirely sub-threshold is not quiet, it is nothing —
        /// BS.1770-4 drops those blocks outright and the integration is left
        /// with no input at all.
        #[test]
        fn content_below_the_absolute_threshold_integrates_to_silence() {
            let measured =
                LoudnessMeter::measure_lufs(&stepped_tone(&[(10.0, -75.0)]), SAMPLE_RATE);
            assert!(
                measured == f64::NEG_INFINITY,
                "sub-threshold content measured {measured} LUFS, expected silence"
            );
        }

        /// Nothing is gated here -- every block is within 10 LU of the mean --
        /// so this pins the energy average itself. Averaging in the log domain
        /// instead of the power domain would read about -23.8.
        #[test]
        fn test_5_energy_mean_with_nothing_gated() {
            assert_integrated(
                &[(20.0, -26.0), (20.1, -20.0), (20.0, -26.0)],
                -23.0,
                "EBU 3341 test 5",
            );
        }

        /// A full-scale 997 Hz sine has a true peak of exactly 0 dBTP by
        /// construction: 997 Hz does not divide 48 kHz, so the sample peak sits
        /// below the real one and only oversampling recovers it. Tech 3341
        /// allows +/- 0.4 dB for a true-peak meter reading at or below 0 dBTP.
        #[test]
        fn true_peak_of_full_scale_997_hz_sine_is_zero_dbtp() {
            let measured = TruePeakMeter::measure_dbtp(&sine(1.0, 997.0, 1.0));
            assert!(
                measured.abs() <= 0.4,
                "997 Hz full-scale sine measured {measured} dBTP, expected 0.0 +/- 0.4"
            );
        }
    }

    // LoudnessNormalizerTests.swift

    mod loudness_normalizer_tests {
        use super::*;

        #[test]
        fn quiet_input_reaches_target_lufs() {
            let result = normalize_sine(0.045);
            assert_eq!(result.report.target_lufs, -18.0);
            assert!((result.report.post_limit_lufs - -18.0).abs() <= 1.0);
            assert!(result.report.qa_pass);
            assert_eq!(result.report.spec_version, 1);
        }

        #[test]
        fn loud_input_reaches_target_lufs() {
            let result = normalize_sine(1.0);
            assert!((result.report.post_limit_lufs - -18.0).abs() <= 1.0);
            assert!(result.report.qa_pass);
            assert!(result.report.gain_applied_db < 0.0);
        }

        #[test]
        fn output_true_peak_never_exceeds_ceiling() {
            let result = normalize_sine(0.5);
            assert!(
                result.report.post_limit_true_peak
                    <= LoudnessTarget::STANDARD.peak_ceiling_dbtp + 0.1,
                "peak: {} dBTP",
                result.report.post_limit_true_peak
            );
        }

        #[test]
        fn report_fields_consistent() {
            let result = normalize_sine(0.1);
            let report = result.report;
            assert_eq!(report.spec_version, 1);
            assert_eq!(report.target_lufs, LoudnessTarget::STANDARD.target_lufs);
            assert_eq!(
                report.peak_ceiling_dbtp,
                LoudnessTarget::STANDARD.peak_ceiling_dbtp
            );
            assert!(report.measured_lufs.is_finite());
            assert!(report.post_limit_lufs.is_finite());
            assert!(!report.within_clip_leveler_engaged);
        }

        #[test]
        fn report_json_round_trip() {
            let original = LoudnessReport {
                spec_version: 1,
                measured_lufs: -22.5,
                measured_true_peak: -3.0,
                gain_applied_db: 4.5,
                post_limit_lufs: -18.0,
                post_limit_true_peak: -1.0,
                limiter_engaged: false,
                within_clip_leveler_engaged: true,
                qa_pass: true,
                target_lufs: -18.0,
                peak_ceiling_dbtp: -1.0,
            };
            let json = serde_json::to_string(&original).unwrap();
            assert!(json.contains("\"measured_lufs\""));
            assert!(json.contains("\"within_clip_leveler_engaged\""));
            assert!(json.contains("\"spec_version\""));
            let decoded: LoudnessReport = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, original);
        }
    }

    // TruePeakMeterTests.swift

    mod true_peak_meter_tests {
        use super::*;

        #[test]
        fn true_peak_matches_sample_peak_on_well_sampled_sine() {
            let peak = TruePeakMeter::measure_dbtp(&sine(0.5, 1_000.0, 1.0));
            assert!(
                (peak - 20.0 * 0.5_f64.log10()).abs() <= 0.3,
                "peak: {peak} dBTP"
            );
        }

        #[test]
        fn true_peak_detects_inter_sample_overshoot() {
            let samples = sine_with_phase(0.5, SAMPLE_RATE / 4.0, 0.5, PI / 4.0);
            let sample_peak = samples.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let true_peak = TruePeakMeter::measure_dbtp(&samples);
            let sample_peak_db = 20.0 * f64::from(sample_peak).log10();
            assert!(true_peak > sample_peak_db);
            assert!((true_peak - 20.0 * 0.5_f64.log10()).abs() <= 0.5);
        }

        #[test]
        fn silent_input_returns_negative_infinity() {
            assert_eq!(
                TruePeakMeter::measure_dbtp(&vec![0.0; SAMPLE_RATE as usize]),
                f64::NEG_INFINITY
            );
        }

        #[test]
        fn linearity_with_amplitude_scaling() {
            let first = TruePeakMeter::measure_dbtp(&sine(0.1, 1_000.0, 0.5));
            let second = TruePeakMeter::measure_dbtp(&sine(0.2, 1_000.0, 0.5));
            assert!(((second - first) - 6.02).abs() <= 0.1);
        }
    }
}
