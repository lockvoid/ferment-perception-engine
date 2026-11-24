//! Short-Time Fourier Transform — streaming and offline.
//!
//! Layout matches iOS `StreamingSTFT.forward`: each frame is `window ·
//! sliding_buffer`, the analysis memory holds the last `fft_size - hop_size`
//! raw (un-windowed) samples, and frames are emitted on every push of
//! `hop_size` samples.
//!
//! Forward STFT does **not** apply any normalization — caller multiplies by
//! their own `wnorm` if needed (DeepFilterNet uses `2*hop/fft²`, librosa
//! defaults to none, etc.).

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

use ferment_engine_core::{EngineError, Result};

#[derive(Debug, Clone)]
pub struct StftConfig {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: Vec<f32>,
}

impl StftConfig {
    pub fn new(fft_size: usize, hop_size: usize, window: Vec<f32>) -> Result<Self> {
        if fft_size < 2 || !fft_size.is_multiple_of(2) {
            return Err(EngineError::invalid_config(format!(
                "fft_size must be even and ≥ 2, got {fft_size}"
            )));
        }
        if hop_size == 0 || hop_size > fft_size {
            return Err(EngineError::invalid_config(format!(
                "hop_size must be in 1..=fft_size ({fft_size}), got {hop_size}"
            )));
        }
        if window.len() != fft_size {
            return Err(EngineError::invalid_config(format!(
                "window length ({}) must equal fft_size ({})",
                window.len(),
                fft_size
            )));
        }
        Ok(Self {
            fft_size,
            hop_size,
            window,
        })
    }

    pub fn n_freq(&self) -> usize {
        self.fft_size / 2 + 1
    }
}

/// Streaming STFT: feed `hop_size` samples per call, get one complex spectrum
/// frame back. Carries `fft_size - hop_size` samples of analysis memory.
pub struct StreamingStft {
    cfg: StftConfig,
    r2c: Arc<dyn RealToComplex<f32>>,
    analysis_mem: Vec<f32>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl StreamingStft {
    pub fn new(cfg: StftConfig) -> Result<Self> {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(cfg.fft_size);
        let fft_input = r2c.make_input_vec();
        let fft_output = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();
        let mem_len = cfg.fft_size - cfg.hop_size;
        Ok(Self {
            cfg,
            r2c,
            analysis_mem: vec![0.0; mem_len],
            fft_input,
            fft_output,
            scratch,
        })
    }

    pub fn n_freq(&self) -> usize {
        self.cfg.n_freq()
    }

    pub fn fft_size(&self) -> usize {
        self.cfg.fft_size
    }

    pub fn hop_size(&self) -> usize {
        self.cfg.hop_size
    }

    /// Push exactly `hop_size` samples (zero-padded if shorter), return one
    /// spectrum frame of length `n_freq`.
    pub fn frame(&mut self, samples: &[f32]) -> Result<&[Complex<f32>]> {
        let mem_len = self.cfg.fft_size - self.cfg.hop_size;
        let hop = self.cfg.hop_size;

        // First `mem_len` of fft input = analysis_mem * window
        for i in 0..mem_len {
            self.fft_input[i] = self.analysis_mem[i] * self.cfg.window[i];
        }
        // Next `hop_size` = (samples or 0) * window
        for i in 0..hop {
            let s = samples.get(i).copied().unwrap_or(0.0);
            self.fft_input[mem_len + i] = s * self.cfg.window[mem_len + i];
        }

        // Save raw (un-windowed) samples for next frame
        for i in 0..mem_len {
            // Slide left: keep last (mem_len - hop) of analysis_mem, append hop new samples
            if i + hop < mem_len {
                self.analysis_mem[i] = self.analysis_mem[i + hop];
            } else {
                let off = (i + hop) - mem_len;
                self.analysis_mem[i] = samples.get(off).copied().unwrap_or(0.0);
            }
        }

        self.r2c
            .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.scratch)
            .map_err(|e| EngineError::StftFailed(e.to_string()))?;

        Ok(&self.fft_output)
    }

    /// Reset analysis memory (use between independent inputs).
    pub fn reset(&mut self) {
        self.analysis_mem.fill(0.0);
    }
}

/// Offline STFT — emits `samples.len() / hop_size` frames of length `n_freq`.
/// Frame layout matches iOS `StreamingSTFT.forward`. Final partial hop is
/// zero-padded.
pub fn stft(samples: &[f32], cfg: &StftConfig) -> Result<Vec<Vec<Complex<f32>>>> {
    let n_frames = samples.len() / cfg.hop_size;
    if n_frames == 0 {
        return Err(EngineError::AudioTooShort {
            samples: samples.len(),
            min_required: cfg.hop_size,
        });
    }
    let mut s = StreamingStft::new(cfg.clone())?;
    let mut out = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let start = t * cfg.hop_size;
        let end = start + cfg.hop_size;
        let frame = s.frame(&samples[start..end])?.to_vec();
        out.push(frame);
    }
    Ok(out)
}

/// Edge-padding mode for `stft_centered`. Matches librosa / torchaudio
/// `center=True, pad_mode={"constant", "reflect"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// Zero-pad each side. Default for librosa, MusicHighlighter.
    #[default]
    Zero,
    /// Reflect signal across edges. torchaudio default; used by BeatThis.
    Reflect,
}

/// Librosa-style centered STFT.
///
/// Pads the input by `fft_size / 2` samples on each side (zero or reflect),
/// then frames at `audio[t*hop .. t*hop + fft_size]`. This matches
/// `librosa.stft(center=True)` and `torchaudio.transforms.MelSpectrogram`.
///
/// Returns `n_frames = floor((n + fft_size) / hop) + 1` complex spectra.
/// No `wnorm` scaling applied — caller multiplies if needed.
pub fn stft_centered(samples: &[f32], cfg: &StftConfig) -> Result<Vec<Vec<Complex<f32>>>> {
    stft_centered_with(samples, cfg, Padding::Zero)
}

/// `stft_centered` with explicit padding mode.
pub fn stft_centered_with(
    samples: &[f32],
    cfg: &StftConfig,
    padding: Padding,
) -> Result<Vec<Vec<Complex<f32>>>> {
    let pad = cfg.fft_size / 2;
    let padded_len = samples.len() + 2 * pad;
    if padded_len < cfg.fft_size {
        return Err(EngineError::AudioTooShort {
            samples: samples.len(),
            min_required: cfg.fft_size - 2 * pad,
        });
    }
    let n_frames = (padded_len - cfg.fft_size) / cfg.hop_size + 1;

    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    if padding == Padding::Reflect && !samples.is_empty() {
        // torch reflect: padded[pad-1-i] = samples[min(i+1, n-1)] for i in 0..pad
        // and padded[pad+n+i] = samples[max(n-2-i, 0)] for i in 0..pad
        let n = samples.len();
        for i in 0..pad {
            padded[pad - 1 - i] = samples[(i + 1).min(n - 1)];
        }
        for i in 0..pad {
            let src = (n as isize - 2 - i as isize).max(0) as usize;
            padded[pad + n + i] = samples[src];
        }
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(cfg.fft_size);
    let mut fft_input = r2c.make_input_vec();
    let mut fft_output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    let mut out = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let start = t * cfg.hop_size;
        for i in 0..cfg.fft_size {
            fft_input[i] = padded[start + i] * cfg.window[i];
        }
        r2c.process_with_scratch(&mut fft_input, &mut fft_output, &mut scratch)
            .map_err(|e| EngineError::StftFailed(e.to_string()))?;
        out.push(fft_output.clone());
    }
    Ok(out)
}

/// Streaming inverse STFT with overlap-add. Mirrors iOS `StreamingSTFT.inverse`.
/// Caller is responsible for any spectrum scaling — this performs only the
/// inverse FFT, the synthesis window multiplication, and OLA.
pub struct StreamingIstft {
    cfg: StftConfig,
    c2r: Arc<dyn ComplexToReal<f32>>,
    synthesis_mem: Vec<f32>,
    fft_input: Vec<Complex<f32>>,
    fft_output: Vec<f32>,
    scratch: Vec<Complex<f32>>,
}

impl StreamingIstft {
    pub fn new(cfg: StftConfig) -> Result<Self> {
        let mut planner = RealFftPlanner::<f32>::new();
        let c2r = planner.plan_fft_inverse(cfg.fft_size);
        let fft_input = c2r.make_input_vec();
        let fft_output = c2r.make_output_vec();
        let scratch = c2r.make_scratch_vec();
        let mem_len = cfg.fft_size - cfg.hop_size;
        Ok(Self {
            cfg,
            c2r,
            synthesis_mem: vec![0.0; mem_len],
            fft_input,
            fft_output,
            scratch,
        })
    }

    /// Push one spectrum frame, get `hop_size` time-domain samples back.
    pub fn frame(&mut self, spectrum: &[Complex<f32>]) -> Result<Vec<f32>> {
        if spectrum.len() != self.cfg.n_freq() {
            return Err(EngineError::invalid_config(format!(
                "spectrum length {} != n_freq {}",
                spectrum.len(),
                self.cfg.n_freq()
            )));
        }

        self.fft_input.copy_from_slice(spectrum);
        // A real-valued signal cannot carry an imaginary DC component or,
        // for an even FFT, an imaginary Nyquist component. Accelerate's
        // inverse DFT ignores those lanes, while `realfft` rejects them.
        // Core ML FP16 enhancement can introduce a few rounding bits there,
        // so canonicalize the half spectrum before the portable inverse.
        self.fft_input[0].im = 0.0;
        if let Some(nyquist) = self.fft_input.last_mut() {
            nyquist.im = 0.0;
        }
        self.c2r
            .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.scratch)
            .map_err(|e| EngineError::StftFailed(e.to_string()))?;

        // realfft inverse outputs un-normalized (by N) samples; caller normalizes.
        let inv_scale = 1.0 / self.cfg.fft_size as f32;
        let hop = self.cfg.hop_size;
        let mem_len = self.cfg.fft_size - hop;

        let mut out = vec![0.0f32; hop];
        for (i, output) in out.iter_mut().enumerate() {
            *output = self.fft_output[i] * inv_scale * self.cfg.window[i] + self.synthesis_mem[i];
        }
        // Refresh synthesis memory with the trailing windowed tail.
        for i in 0..mem_len {
            self.synthesis_mem[i] = self.fft_output[hop + i] * inv_scale * self.cfg.window[hop + i];
        }
        Ok(out)
    }

    pub fn reset(&mut self) {
        self.synthesis_mem.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_abs_diff_eq;
    use ferment_engine_test_fixtures as fixtures;

    use super::*;
    use crate::window;

    fn make_sine(freq: f32, sample_rate: u32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn config_rejects_odd_fft_size() {
        let w = vec![1.0; 1023];
        assert!(StftConfig::new(1023, 256, w).is_err());
    }

    #[test]
    fn config_accepts_non_power_of_two_even() {
        // DeepFilterNet uses 960 — not a power of two but supported.
        let w = vec![1.0; 960];
        assert!(StftConfig::new(960, 480, w).is_ok());
    }

    #[test]
    fn config_rejects_window_length_mismatch() {
        let w = vec![1.0; 1024];
        assert!(StftConfig::new(2048, 256, w).is_err());
    }

    #[test]
    fn config_rejects_zero_hop() {
        let w = vec![1.0; 1024];
        assert!(StftConfig::new(1024, 0, w).is_err());
    }

    #[test]
    fn sine_peaks_at_correct_bin() {
        // 1000 Hz sine at 16 kHz, 1024 fft → expected peak at bin 64
        let sr = 16_000;
        let fft_size = 1024;
        let hop = 256;
        let freq = 1000.0;
        let signal = make_sine(freq, sr, fft_size * 4);

        let cfg = StftConfig::new(fft_size, hop, window::hann(fft_size)).unwrap();
        let frames = stft(&signal, &cfg).unwrap();

        // Use a middle frame (steady-state)
        let mid = &frames[frames.len() / 2];
        let mags: Vec<f32> = mid.iter().map(|c| c.norm()).collect();
        let (peak_bin, _) = mags
            .iter()
            .enumerate()
            .skip(1) // ignore DC
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let expected_bin = (freq * fft_size as f32 / sr as f32) as usize;
        assert_eq!(
            peak_bin, expected_bin,
            "peak bin {peak_bin} != {expected_bin}"
        );
    }

    #[test]
    fn dc_signal_concentrates_at_bin_zero() {
        let fft_size = 512;
        let hop = 128;
        let signal = vec![1.0; fft_size * 2];

        // Use a rectangular window so DC stays DC
        let cfg = StftConfig::new(fft_size, hop, vec![1.0; fft_size]).unwrap();
        let frames = stft(&signal, &cfg).unwrap();
        let mid = &frames[frames.len() / 2];
        let mags: Vec<f32> = mid.iter().map(|c| c.norm()).collect();

        // Bin 0 should equal fft_size, all others ~0
        assert_abs_diff_eq!(mags[0], fft_size as f32, epsilon = 1e-3);
        for &m in &mags[1..] {
            assert!(m < 1e-3, "non-zero magnitude {m} at non-DC bin");
        }
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        // Streaming STFT/ISTFT introduces a `mem_len`-sample latency
        // (the analysis lookahead must accumulate before output emerges).
        // After that delay, Vorbis at 50% overlap is COLA-square so we
        // get bit-perfect recovery (modulo FFT round-off).
        let fft_size = 960;
        let hop = fft_size / 2;
        let mem_len = fft_size - hop; // = hop here (50% overlap)
        let signal: Vec<f32> = (0..fft_size * 8)
            .map(|i| 0.3 * (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();

        let win = window::vorbis(fft_size);
        let cfg = StftConfig::new(fft_size, hop, win).unwrap();

        let mut fwd = StreamingStft::new(cfg.clone()).unwrap();
        let mut inv = StreamingIstft::new(cfg.clone()).unwrap();

        let mut reconstructed = Vec::new();
        let n_frames = signal.len() / hop;
        for t in 0..n_frames {
            let start = t * hop;
            let end = start + hop;
            let spectrum = fwd.frame(&signal[start..end]).unwrap().to_vec();
            let samples = inv.frame(&spectrum).unwrap();
            reconstructed.extend_from_slice(&samples);
        }

        // Compare reconstructed[mem_len + skip ..] against signal[skip..],
        // skipping a couple of frames either side for window ramp.
        let skip = hop;
        let end = signal.len() - mem_len - hop;
        let mut max_err = 0.0f32;
        for i in skip..end {
            let err = (reconstructed[i + mem_len] - signal[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 1e-4, "STFT round-trip error {max_err} too large");
    }

    #[test]
    fn inverse_ignores_unrepresentable_endpoint_imaginary_values() {
        let cfg = StftConfig::new(16, 8, vec![1.0; 16]).unwrap();
        let mut inverse = StreamingIstft::new(cfg).unwrap();
        let mut spectrum = vec![Complex::new(0.0, 0.0); 9];
        spectrum[0].im = 0.25;
        spectrum[8].im = -0.5;
        assert_eq!(inverse.frame(&spectrum).unwrap().len(), 8);
    }

    #[test]
    fn deepfilter_stft_matches_libdf_swift_trace_when_available() {
        let Some(audio) = fixtures::f32_le("SpeechDenoising/trace_audio_padded.bin") else {
            return;
        };
        let expected = fixtures::require_f32_le("SpeechDenoising/trace_stft.bin");
        let fft_size = 960;
        let hop_size = 480;
        let config = StftConfig::new(fft_size, hop_size, window::vorbis(fft_size)).unwrap();
        let frames = stft(&audio, &config).unwrap();
        assert_eq!(frames.len(), 102);
        let normalization = (2 * hop_size) as f32 / (fft_size * fft_size) as f32;
        let actual = frames
            .iter()
            .flatten()
            .flat_map(|value| [value.re * normalization, value.im * normalization])
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        let max_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1e-5,
            "DeepFilterNet STFT max difference {max_difference} exceeded 1e-5"
        );
    }

    #[test]
    fn streaming_matches_offline() {
        let fft_size = 256;
        let hop = 64;
        let signal = make_sine(440.0, 16_000, fft_size * 4);
        let cfg = StftConfig::new(fft_size, hop, window::hann(fft_size)).unwrap();

        let offline = stft(&signal, &cfg).unwrap();

        let mut streaming = StreamingStft::new(cfg.clone()).unwrap();
        let mut streamed = Vec::new();
        for t in 0..(signal.len() / hop) {
            let start = t * hop;
            let frame = streaming.frame(&signal[start..start + hop]).unwrap();
            streamed.push(frame.to_vec());
        }

        assert_eq!(offline.len(), streamed.len());
        for (a, b) in offline.iter().zip(streamed.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_abs_diff_eq!(x.re, y.re, epsilon = 1e-5);
                assert_abs_diff_eq!(x.im, y.im, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn audio_too_short_errors() {
        let cfg = StftConfig::new(512, 128, window::hann(512)).unwrap();
        assert!(stft(&[0.0; 64], &cfg).is_err());
    }

    #[test]
    fn stft_centered_frame_count_matches_librosa() {
        // librosa center=True: pads fft//2 each side, then n_frames = n/hop + 1.
        // Real-world check: 30s @ 22.05 kHz, fft=2048, hop=512 → 1292 frames
        // (matches the iOS MusicHighlighter test_metadata.json).
        let fft_size = 2048;
        let hop = 512;
        let n_samples = 30 * 22_050;
        let cfg = StftConfig::new(fft_size, hop, window::hann_periodic(fft_size)).unwrap();
        let frames = stft_centered(&vec![0.0f32; n_samples], &cfg).unwrap();
        assert_eq!(frames.len(), 1292);
    }

    #[test]
    fn stft_centered_sine_peaks_at_correct_bin() {
        let sr = 22_050;
        let fft_size = 2048;
        let hop = 512;
        let freq = 1000.0;
        let signal = make_sine(freq, sr, sr as usize);
        let cfg = StftConfig::new(fft_size, hop, window::hann_periodic(fft_size)).unwrap();
        let frames = stft_centered(&signal, &cfg).unwrap();
        let mid = &frames[frames.len() / 2];
        let mags: Vec<f32> = mid.iter().map(|c| c.norm()).collect();
        let (peak_bin, _) = mags
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let expected = (freq * fft_size as f32 / sr as f32) as usize;
        assert!((peak_bin as i32 - expected as i32).abs() <= 1);
    }
}
