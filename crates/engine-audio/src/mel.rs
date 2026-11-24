//! Mel filterbank — librosa-compatible (`librosa.filters.mel`).
//!
//! Supports HTK and Slaney mel scales (selectable independently from
//! normalization). Each iOS processor builds its own filterbank with its
//! own params; this module is just the math.
//!
//! Conventions match librosa 0.10:
//! - Slaney: `2.0 / (hz_high - hz_low)` per-band normalization
//! - HTK scale: `2595 * log10(1 + hz / 700)`
//! - Slaney scale: linear below 1 kHz, log above (matches Auditory Toolbox)

use ferment_engine_core::{EngineError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    Htk,
    Slaney,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// No per-band normalization — raw triangles peak at 1.0.
    None,
    /// Slaney (librosa default `norm="slaney"`): scale band by `2 / (hz_high - hz_low)`.
    Slaney,
}

#[derive(Debug, Clone)]
pub struct MelOptions {
    pub n_mels: usize,
    pub n_fft: usize,
    pub sample_rate: u32,
    pub f_min: f32,
    pub f_max: f32,
    pub scale: MelScale,
    pub norm: MelNorm,
}

/// Mel filterbank weights. Stored as a flat `n_mels × n_freqs` row-major
/// matrix for cache-friendly mat-vec.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    pub n_mels: usize,
    pub n_freqs: usize,
    weights: Vec<f32>,
    row_ranges: Vec<(usize, usize)>,
}

impl MelFilterbank {
    pub fn new(opts: MelOptions) -> Result<Self> {
        if opts.n_mels == 0 {
            return Err(EngineError::invalid_config("n_mels must be > 0"));
        }
        if opts.n_fft < 2 || !opts.n_fft.is_multiple_of(2) {
            return Err(EngineError::invalid_config("n_fft must be even and ≥ 2"));
        }
        if opts.sample_rate == 0 {
            return Err(EngineError::invalid_config("sample_rate must be > 0"));
        }
        if !(0.0..opts.f_max).contains(&opts.f_min) {
            return Err(EngineError::invalid_config("f_min must be ≥ 0 and < f_max"));
        }

        let n_freqs = opts.n_fft / 2 + 1;
        let mel_min = hz_to_mel(opts.f_min, opts.scale);
        let mel_max = hz_to_mel(opts.f_max, opts.scale);

        // n_mels + 2 boundary points
        let mut mel_points = Vec::with_capacity(opts.n_mels + 2);
        for i in 0..=(opts.n_mels + 1) {
            let frac = i as f32 / (opts.n_mels + 1) as f32;
            mel_points.push(mel_min + frac * (mel_max - mel_min));
        }
        let hz_points: Vec<f32> = mel_points
            .iter()
            .map(|&m| mel_to_hz(m, opts.scale))
            .collect();
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * opts.n_fft as f32 / opts.sample_rate as f32)
            .collect();

        let mut weights = vec![0.0f32; opts.n_mels * n_freqs];
        for m in 0..opts.n_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            for (f, w) in weights[m * n_freqs..(m + 1) * n_freqs]
                .iter_mut()
                .enumerate()
            {
                let freq = f as f32;
                let v = if freq >= left && freq <= center && center > left {
                    (freq - left) / (center - left)
                } else if freq > center && freq <= right && right > center {
                    (right - freq) / (right - center)
                } else {
                    0.0
                };
                *w = v;
            }

            if opts.norm == MelNorm::Slaney {
                let bandwidth = hz_points[m + 2] - hz_points[m];
                if bandwidth > 0.0 {
                    let scale = 2.0 / bandwidth;
                    for w in &mut weights[m * n_freqs..(m + 1) * n_freqs] {
                        *w *= scale;
                    }
                }
            }
        }

        let row_ranges = weights
            .chunks_exact(n_freqs)
            .map(|row| {
                let start = row.iter().position(|weight| *weight != 0.0).unwrap_or(0);
                let end = row
                    .iter()
                    .rposition(|weight| *weight != 0.0)
                    .map_or(start, |index| index + 1);
                (start, end)
            })
            .collect();

        Ok(Self {
            n_mels: opts.n_mels,
            n_freqs,
            weights,
            row_ranges,
        })
    }

    /// Apply filterbank to a single power spectrum (length `n_freqs`):
    /// returns `n_mels` mel energies. `mel = W · power`.
    pub fn apply(&self, spectrum: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.n_mels];
        self.apply_into(spectrum, &mut out)?;
        Ok(out)
    }

    /// Apply the filterbank into caller-owned storage.
    ///
    /// Mel triangles have compact support. Traversing only each row's
    /// non-zero range preserves the dense dot product while avoiding the
    /// roughly 98% zero multiplies present at 128 bands / 2048 FFT.
    pub fn apply_into(&self, spectrum: &[f32], out: &mut [f32]) -> Result<()> {
        if spectrum.len() != self.n_freqs {
            return Err(EngineError::invalid_config(format!(
                "spectrum length {} != n_freqs {}",
                spectrum.len(),
                self.n_freqs
            )));
        }
        if out.len() != self.n_mels {
            return Err(EngineError::invalid_config(format!(
                "output length {} != n_mels {}",
                out.len(),
                self.n_mels
            )));
        }
        for (m, output) in out.iter_mut().enumerate() {
            let (start, end) = self.row_ranges[m];
            let row_start = m * self.n_freqs + start;
            *output = self.weights[row_start..m * self.n_freqs + end]
                .iter()
                .zip(&spectrum[start..end])
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(())
    }

    /// Raw row-major `n_mels × n_freqs` weight matrix — for callers that
    /// want to call BLAS / GEMV directly across many frames.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

// --- Hz ↔ Mel conversions ---

fn hz_to_mel(hz: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => slaney_hz_to_mel(hz),
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => slaney_mel_to_hz(mel),
    }
}

fn slaney_hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        let min_log_mel = MIN_LOG_HZ / F_SP;
        let logstep = (6.4f32).ln() / 27.0;
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if mel < min_log_mel {
        mel * F_SP
    } else {
        let logstep = (6.4f32).ln() / 27.0;
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    fn music_highlighter_opts() -> MelOptions {
        MelOptions {
            n_mels: 128,
            n_fft: 2048,
            sample_rate: 22_050,
            f_min: 0.0,
            f_max: 11_025.0,
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
        }
    }

    #[test]
    fn slaney_round_trip() {
        for hz in [100.0, 500.0, 999.0, 1000.0, 2000.0, 8000.0, 16000.0] {
            let mel = slaney_hz_to_mel(hz);
            let back = slaney_mel_to_hz(mel);
            assert_abs_diff_eq!(back, hz, epsilon = 1e-2);
        }
    }

    #[test]
    fn htk_matches_known_values() {
        // 0 Hz → 0 mel, 1000 Hz → 1000 mel (HTK is calibrated so 1k Hz ≈ 1k mel)
        assert_abs_diff_eq!(hz_to_mel(0.0, MelScale::Htk), 0.0, epsilon = 1e-3);
        let m1k = hz_to_mel(1000.0, MelScale::Htk);
        assert_abs_diff_eq!(m1k, 1000.0, epsilon = 1.0);
        // Round-trip
        assert_abs_diff_eq!(mel_to_hz(m1k, MelScale::Htk), 1000.0, epsilon = 1e-2);
    }

    #[test]
    fn shape_matches_options() {
        let opts = music_highlighter_opts();
        let fb = MelFilterbank::new(opts).unwrap();
        assert_eq!(fb.n_mels, 128);
        assert_eq!(fb.n_freqs, 1025);
        assert_eq!(fb.weights().len(), 128 * 1025);
    }

    #[test]
    fn each_band_is_triangular_and_local() {
        // Every band should have a single contiguous run of non-zero weights
        // (one rising edge, one falling edge).
        let fb = MelFilterbank::new(music_highlighter_opts()).unwrap();
        for m in 0..fb.n_mels {
            let row = &fb.weights[m * fb.n_freqs..(m + 1) * fb.n_freqs];
            let nonzero: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0.0)
                .map(|(i, _)| i)
                .collect();
            assert!(!nonzero.is_empty(), "band {m} has no support");
            let span = nonzero.last().unwrap() - nonzero.first().unwrap() + 1;
            assert_eq!(
                span,
                nonzero.len(),
                "band {m} has gaps in support — not triangular"
            );
        }
    }

    #[test]
    fn slaney_normalization_makes_total_energy_finite() {
        let fb = MelFilterbank::new(music_highlighter_opts()).unwrap();
        // For Slaney-normalized filterbank applied to flat power spectrum,
        // each band's output ≈ 1 (band integrates to unity energy).
        let flat = vec![1.0f32; fb.n_freqs];
        let out = fb.apply(&flat).unwrap();
        // Higher bands span more bins so this is approximate, but every
        // value should be finite and non-tiny.
        for &v in &out {
            assert!(v.is_finite());
            assert!(v > 0.0);
        }
    }

    #[test]
    fn applying_to_zero_yields_zero() {
        let fb = MelFilterbank::new(music_highlighter_opts()).unwrap();
        let zero = vec![0.0f32; fb.n_freqs];
        let out = fb.apply(&zero).unwrap();
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn rejects_wrong_spectrum_length() {
        let fb = MelFilterbank::new(music_highlighter_opts()).unwrap();
        assert!(fb.apply(&[0.0; 100]).is_err());
    }

    #[test]
    fn rejects_invalid_options() {
        let mut o = music_highlighter_opts();
        o.n_mels = 0;
        assert!(MelFilterbank::new(o).is_err());
        let mut o = music_highlighter_opts();
        o.f_min = 12_000.0;
        assert!(MelFilterbank::new(o).is_err());
    }

    #[test]
    fn whisper_config_builds() {
        // Whisper / Parakeet preprocess at 16 kHz, n_fft=400, n_mels=80
        let opts = MelOptions {
            n_mels: 80,
            n_fft: 400,
            sample_rate: 16_000,
            f_min: 0.0,
            f_max: 8_000.0,
            scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
        };
        let fb = MelFilterbank::new(opts).unwrap();
        assert_eq!(fb.n_mels, 80);
        assert_eq!(fb.n_freqs, 201);
    }
}
