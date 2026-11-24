//! Visualization mel spectrogram for the audio minimap. Configuration
//! matches iOS `AudioMinimapMelSpectrogram` byte-for-byte:
//!   22050 Hz, n_fft 2048, hop 512, 128 mel bands, Slaney scale,
//!   fmin 20 Hz, fmax sr/2, power=2, reference=max, top_db=80.
//!
//! Output is `[T, n_mels]` row-major in dB ∈ `[-top_db, 0]`.

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, Scale, Spectrum, TablePrecision,
    Window,
};
use ferment_engine_core::Result;

pub const SAMPLE_RATE: u32 = 22_050;
pub const FFT_SIZE: usize = 2048;
pub const HOP_LENGTH: usize = 512;
pub const N_MELS: usize = 128;
pub const F_MIN: f32 = 20.0;
pub const F_MAX: f32 = 11_025.0;
pub const TOP_DB: f32 = 80.0;

/// The same value used by the current iOS cache and minimap front-end.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: FFT_SIZE,
    hop_length: HOP_LENGTH,
    mel_bands: N_MELS,
    f_min: F_MIN,
    f_max: F_MAX,
    input_scale: 1.0,
    padding: Padding::Reflect,
    window: Window::HannPeriodic,
    window_length: None,
    dc_removal: false,
    pre_emphasis: None,
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Slaney,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::SlaneyArea,
    table_precision: TablePrecision::Float32,
    compression: Compression::Decibels { top_db: TOP_DB },
};

pub fn compute_mel_db(audio: &[f32]) -> Result<Vec<Vec<f32>>> {
    let frames = MEL_SPEC.compute(audio)?;
    Ok(frames
        .values
        .chunks_exact(frames.bands)
        .map(<[f32]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_yields_empty() {
        assert!(compute_mel_db(&[]).unwrap().is_empty());
    }

    #[test]
    fn silent_input_yields_zero_db_after_max_normalization() {
        // For all-zero input, reference=max=amin, so 10·log10(p/ref) = 0
        // for every bin. (No reference to normalize against → flat at 0 dB,
        // not -TOP_DB. Real audio has non-zero peaks that push the rest down.)
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let mel = compute_mel_db(&audio).unwrap();
        for row in &mel {
            for &v in row {
                assert!(v.abs() < 1e-3, "silent mel value {v} should be ~0 dB");
            }
        }
    }

    #[test]
    fn output_shape_matches_expected() {
        let audio = vec![0.0f32; 30 * SAMPLE_RATE as usize];
        let mel = compute_mel_db(&audio).unwrap();
        assert_eq!(mel.len(), 1292);
        assert!(mel.iter().all(|row| row.len() == N_MELS));
    }
}
