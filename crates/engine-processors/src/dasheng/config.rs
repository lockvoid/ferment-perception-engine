//! Constants — the `dasheng_base_mel` contract
//! (`ferment-models-export`, `models/dasheng/model.json`).

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, Scale, Spectrum, TablePrecision,
    Window,
};

pub const SAMPLE_RATE: u32 = 16_000;
pub const WINDOW_SECONDS: usize = 10;
pub const WINDOW_SAMPLES: usize = SAMPLE_RATE as usize * WINDOW_SECONDS; // 160_000
pub const EMBEDDING_DIM: usize = 768;

/// Dasheng-Base's front-end, held as a value (read off the `dasheng` package's
/// `front_end`): torchaudio `MelSpectrogram(16 kHz, n_fft = win 512, hop 160,
/// centre reflect, power 2, 64 HTK mels, norm None, 0–8000 Hz)` then
/// `AmplitudeToDB`, whose `top_db` clamp is per window ([`TOP_DB`]). The same
/// value as iOS `DashengSemanticEncoder.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: 512,
    hop_length: 160,
    mel_bands: MEL_BANDS,
    f_min: 0.0,
    f_max: 8_000.0,
    input_scale: 1.0,
    padding: Padding::Reflect,
    window: Window::HannPeriodic,
    window_length: None,
    dc_removal: false,
    pre_emphasis: None,
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Htk,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::None,
    table_precision: TablePrecision::Float32,
    compression: Compression::DecibelsAbsolute { amin: 1e-10 },
};
/// torchaudio `AmplitudeToDB(top_db:)`'s floor, per window.
pub const TOP_DB: f32 = 120.0;

pub const MEL_BANDS: usize = 64;
/// `MEL_SPEC.frame_count(WINDOW_SAMPLES)` — 1 + 160 000 / 160.
pub const MEL_FRAMES: usize = 1001;
/// The model's patch: all 64 bands × 4 frames, stride 4 → 250 patches, the
/// 1001st frame dropped as the patch conv drops it.
pub const PATCH_FRAMES: usize = 4;
pub const PATCH_COUNT: usize = MEL_FRAMES / PATCH_FRAMES;
pub const PATCH_CHANNELS: usize = MEL_BANDS * PATCH_FRAMES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_one_thousand_and_one_frames_and_250_patches() {
        assert_eq!(WINDOW_SAMPLES, 160_000);
        assert_eq!(MEL_SPEC.frame_count(WINDOW_SAMPLES), MEL_FRAMES);
        assert_eq!(PATCH_COUNT, 250);
        assert_eq!(PATCH_CHANNELS, 256);
    }
}
