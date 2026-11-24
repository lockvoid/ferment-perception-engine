//! Analysis windows used by STFT.
//!
//! All windows are length-`size` and symmetric (matches `librosa.filters.get_window`
//! with `fftbins=False`) unless noted. Tests cover the three iOS uses: Hann
//! (BeatThis, Parakeet, MusicHighlighter, Dasheng), Hamming (kept for parity),
//! and Vorbis (DeepFilterNet).
use std::f32::consts::PI;

/// Hann window: `0.5 * (1 - cos(2π·n/(N-1)))`.
pub fn hann(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let denom = (size - 1) as f32;
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / denom).cos()))
        .collect()
}

/// Hann window with the periodic convention used inside FFTs
/// (`librosa.filters.get_window("hann", N, fftbins=True)` / `np.hanning(N)` is
/// the symmetric variant; use this when matching `torch.hann_window(periodic=True)`).
pub fn hann_periodic(size: usize) -> Vec<f32> {
    if size == 0 {
        return Vec::new();
    }
    let denom = size as f32;
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / denom).cos()))
        .collect()
}

/// Hamming window: `0.54 - 0.46·cos(2π·n/(N-1))`.
pub fn hamming(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    let denom = (size - 1) as f32;
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / denom).cos())
        .collect()
}

/// Vorbis window — `sin(π/2·sin²(π·(n+0.5)/N_half))`. Mirrors iOS
/// `StreamingSTFT.makeVorbisWindow` byte-for-byte (used by DeepFilterNet,
/// Vorbis codec, libdf framing convention).
pub fn vorbis(size: usize) -> Vec<f32> {
    let half = (size / 2) as f64;
    (0..size)
        .map(|i| {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / half).sin();
            (0.5 * std::f64::consts::PI * s * s).sin() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hann_zero_at_endpoints() {
        let w = hann(8);
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(w[7], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn hann_peak_in_middle() {
        let w = hann(9);
        // size=9 → middle is index 4, should be 1.0
        assert_abs_diff_eq!(w[4], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn hann_symmetric() {
        let w = hann(64);
        for i in 0..32 {
            assert_abs_diff_eq!(w[i], w[63 - i], epsilon = 1e-6);
        }
    }

    #[test]
    fn hann_periodic_sum_matches_torch() {
        // For periodic Hann of size N, sum = N/2 exactly (because cos terms
        // sum to zero over a full period).
        let w = hann_periodic(64);
        let sum: f32 = w.iter().sum();
        assert_abs_diff_eq!(sum, 32.0, epsilon = 1e-4);
    }

    #[test]
    fn hamming_in_range() {
        let w = hamming(32);
        for &x in &w {
            assert!((0.0..=1.0 + 1e-6).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn vorbis_endpoints_near_zero() {
        // Vorbis isn't strictly zero at endpoints, but is very small.
        let w = vorbis(960);
        assert!(w[0] < 0.01, "w[0] = {}", w[0]);
        assert!(w[959] < 0.01, "w[959] = {}", w[959]);
    }

    #[test]
    fn vorbis_symmetric() {
        let w = vorbis(960);
        for i in 0..480 {
            assert_abs_diff_eq!(w[i], w[959 - i], epsilon = 1e-6);
        }
    }

    #[test]
    fn vorbis_satisfies_overlap_add_constraint() {
        // For 50% overlap, w[n]² + w[n + N/2]² should be ≈ constant
        // (Princen-Bradley / COLA-square condition that DeepFilterNet relies on).
        let n = 960;
        let w = vorbis(n);
        let half = n / 2;
        for i in 0..half {
            let sum_sq = w[i] * w[i] + w[i + half] * w[i + half];
            assert_abs_diff_eq!(sum_sq, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn size_zero_and_one_safe() {
        assert!(hann(0).is_empty());
        assert_eq!(hann(1), vec![1.0]);
        assert!(vorbis(0).is_empty());
    }
}
