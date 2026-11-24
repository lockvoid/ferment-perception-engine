//! Parakeet-TDT-0.6B-v3 constants — the `parakeet` contract
//! (`ferment-models-export`, `models/parakeet/model.json`): the front-end, the
//! encoder's door and the greedy TDT decode.

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, PreEmphasis, Scale, Spectrum,
    TablePrecision, Window,
};

pub const SAMPLE_RATE: u32 = 16_000;

/// `ParakeetFeatureExtractor` (transformers 5.9), held as a value:
/// pre-emphasis 0.97 over the real samples, `torch.stft` centred with zero
/// padding, a 400-sample symmetric Hann in a 512-point transform, power, 128
/// Slaney mels 0–8000 Hz, area-normalised, `log(x + 2⁻²⁴)`. The same value as
/// iOS `Parakeet.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: 512,
    hop_length: HOP_LENGTH,
    mel_bands: MEL_BANDS,
    f_min: 0.0,
    f_max: 8_000.0,
    input_scale: 1.0,
    padding: Padding::Constant,
    window: Window::HannSymmetric,
    window_length: Some(400),
    dc_removal: false,
    pre_emphasis: Some(PreEmphasis::Signal { alpha: 0.97 }),
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Slaney,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::SlaneyArea,
    table_precision: TablePrecision::Float32,
    compression: Compression::Log {
        guard: 5.960_464_5e-8,
    },
};
/// Each band normalised over the window's valid frames: `(x − mean) / (std + ε)`,
/// std with n − 1.
pub const NORMALIZATION_EPSILON: f32 = 1e-5;

pub const HOP_LENGTH: usize = 160;
pub const MEL_BANDS: usize = 128;

/// One window: 15 s of samples → `mel [1, 1, 128, 1501]`, `mask [1, 1, 1, 1501]`
/// → `encoded [1, 640, 1, 188]`.
pub const WINDOW_SAMPLES: usize = 240_000;
pub const MEL_FRAMES: usize = 1_501;
pub const ENCODER_FRAMES: usize = 188;
/// The joint's width: the encoder's projection, the decoder's output.
pub const HIDDEN: usize = 640;

pub const VOCAB: usize = 8_193;
pub const BLANK_TOKEN_ID: usize = 8_192;
pub const DURATION_BINS: [usize; 5] = [0, 1, 2, 3, 4];
/// The joint's logits: the vocabulary, then the five durations.
pub const LOGITS: usize = VOCAB + DURATION_BINS.len();
/// `max_symbols_per_step × the window's encoder frames`, as `generate` sizes it.
pub const MAX_STEPS: usize = 10 * ENCODER_FRAMES;

/// Each encoder frame is 80 ms (1280 samples at 16 kHz).
pub const SAMPLES_PER_ENCODER_FRAME: usize = 1_280;
pub const SECONDS_PER_ENCODER_FRAME: f64 = SAMPLES_PER_ENCODER_FRAME as f64 / SAMPLE_RATE as f64;
pub const OVERLAP_FRAMES: usize = 25;

/// Mel frames backed by real samples: `floor(samples / 160)`.
#[must_use]
pub const fn valid_mel_frames(samples: usize) -> usize {
    samples / HOP_LENGTH
}

/// The subsampling's three kernel-3, stride-2, pad-1 convolutions:
/// `(L − 1) / 2 + 1`, three times.
#[must_use]
pub const fn valid_encoder_frames(mel_frames: usize) -> usize {
    let mut length = mel_frames;
    let mut step = 0;
    while step < 3 {
        length = if length == 0 { 0 } else { (length - 1) / 2 + 1 };
        step += 1;
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_seconds_match_80_ms() {
        assert!((SECONDS_PER_ENCODER_FRAME - 0.08).abs() < 1e-9);
    }

    #[test]
    fn a_full_window_is_the_doors_frames() {
        assert_eq!(MEL_SPEC.frame_count(WINDOW_SAMPLES), MEL_FRAMES);
        assert_eq!(valid_encoder_frames(MEL_FRAMES), ENCODER_FRAMES);
        assert_eq!(valid_mel_frames(WINDOW_SAMPLES), 1_500);
        assert_eq!(valid_encoder_frames(1_500), ENCODER_FRAMES);
        // Current ParakeetConfigTests: the full window's decode budget.
        // Kill: restore the old fixed loop cap or use mel rather than encoder frames.
        assert_eq!(MAX_STEPS, 1_880);
    }

    #[test]
    fn the_log_guard_is_two_to_the_minus_24() {
        let Compression::Log { guard } = MEL_SPEC.compression else {
            panic!("Parakeet's compression is NeMo's log");
        };
        assert_eq!(guard, 2.0_f32.powi(-24));
    }
}
