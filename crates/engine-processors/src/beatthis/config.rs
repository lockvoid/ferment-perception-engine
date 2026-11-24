//! BeatThis training-locked constants — the `beat_this` contract
//! (`ferment-models-export`, `models/beat_this/model.json`).

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, Scale, Spectrum, TablePrecision,
    Window,
};

pub const SAMPLE_RATE: u32 = 22_050;
pub const FFT_SIZE: usize = 1024;
pub const HOP_LENGTH: usize = 441;
pub const N_MELS: usize = 128;
pub const F_MIN: f32 = 30.0;
pub const F_MAX: f32 = 11_000.0;
pub const FPS: f64 = 50.0;

/// The authors' `LogMelSpect`, held as a value: torchaudio
/// `MelSpectrogram(22.05 kHz, n_fft 1024, hop 441, 30–11000 Hz, 128 Slaney
/// mels, norm None, power 1, normalized "frame_length")` then
/// `log1p(1000 · x)`. The same value as iOS `BeatThis.melSpec`.
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
    spectrum: Spectrum::Magnitude,
    normalization: Normalization::FrameLength,
    mel_scale: Scale::Slaney,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::None,
    table_precision: TablePrecision::Float32,
    compression: Compression::Log1p { multiplier: 1000.0 },
};

pub const CHUNK_SIZE: usize = 1500;
pub const BORDER_SIZE: usize = 6;

/// Offset applied to all beat/downbeat times after detection (seconds).
///
/// Zero: the stored grid is the model's truth. Attack-audibility leads (snap,
/// playhead stepping, loops) belong to the ACTION that needs them — baking a
/// lead into the data poisoned every cross-lane comparison and served only the
/// audio-snap consumers.
pub const DEFAULT_BEAT_OFFSET: f64 = 0.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stored_grid_carries_no_attack_lead() {
        assert_eq!(DEFAULT_BEAT_OFFSET, 0.0);
    }

    #[test]
    fn fps_matches_hop_at_22050() {
        // 22050 / 441 = 50 frames/s exactly.
        assert!((SAMPLE_RATE as f64 / HOP_LENGTH as f64 - FPS).abs() < 1e-9);
    }
}
