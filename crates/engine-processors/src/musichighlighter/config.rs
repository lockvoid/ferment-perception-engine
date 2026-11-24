//! The highlighter's training-locked constants — the `music_highlighter`
//! contract (`ferment-models-export`, `models/music_highlighter/model.json`).

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, Scale, Spectrum, TablePrecision,
    Window,
};

pub const SAMPLE_RATE: u32 = 22_050;
pub const FFT_SIZE: usize = 2048;
pub const HOP_LENGTH: usize = 512;
pub const N_MELS: usize = 128;
pub const F_MIN: f32 = 0.0;
pub const F_MAX: f32 = 11_025.0;
pub const LOG_MULTIPLIER: f32 = 10_000.0;

/// The front-end the highlighter was trained on, held as a value: librosa
/// `melspectrogram(sr 22050, n_fft 2048, hop 512, 128 mels)` — centred with
/// zero padding, power 2, Slaney mels, area-normalised triangles — then
/// `log(1 + 10000 · x)`. The same value as iOS `MusicHighlighter.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: FFT_SIZE,
    hop_length: HOP_LENGTH,
    mel_bands: N_MELS,
    f_min: F_MIN,
    f_max: F_MAX,
    input_scale: 1.0,
    padding: Padding::Constant,
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
    compression: Compression::Log1p {
        multiplier: LOG_MULTIPLIER,
    },
};

pub const CHUNK_SECONDS: usize = 3;
pub const CHUNK_FRAMES: usize = 129;
pub const MAX_CHUNKS: usize = 200;
pub const FEATURE_DIM: usize = 256;
pub const POS_ENC_DIM: usize = FEATURE_DIM; // pos_enc is featureDim long, dim*4 in iOS code

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_enc_dim_is_256() {
        assert_eq!(POS_ENC_DIM, 256);
    }

    #[test]
    fn one_chunk_seconds_match_frame_count() {
        // CHUNK_SECONDS at SAMPLE_RATE / HOP_LENGTH frames-per-second
        // ≈ CHUNK_FRAMES (within rounding).
        let frames_per_sec = SAMPLE_RATE as f32 / HOP_LENGTH as f32;
        let approx = (frames_per_sec * CHUNK_SECONDS as f32).round() as usize;
        // 22050/512 ≈ 43.07 frames/s; 3 seconds ≈ 129.2 → 129
        assert!((approx as i32 - CHUNK_FRAMES as i32).abs() <= 1);
    }
}
