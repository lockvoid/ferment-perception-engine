//! FireRedVAD-AED constants — the `fireredvad_aed` contract
//! (`ferment-models-export`, `models/fireredvad_aed/model.json`) and
//! `FireRedAedConfig`'s post-processing defaults.

use ferment_engine_audio::MelSpec;
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, PreEmphasis, Scale, Spectrum,
    TablePrecision, Window,
};

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH_SAMPLES: usize = 400;
pub const FRAME_SHIFT_SAMPLES: usize = 160;
pub const FRAME_SHIFT_SECONDS: f64 = 0.010;

pub const INPUT_DIM: usize = 80;
pub const NUM_EVENT_CLASSES: usize = 3;

/// kaldi-native-fbank 1.22.3 at FireRedVAD's options, held as a value:
/// int16-scale samples; 25 ms povey frames every 10 ms, cut at the edges;
/// each frame's DC removed, then pre-emphasis 0.97 inside the frame; a
/// 512-point power spectrum; 80 triangles linear in mel, 20 Hz to Nyquist;
/// `log(max(x, FLT_EPSILON))`. The same value as iOS
/// `AudioEventDetector.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: 512,
    hop_length: FRAME_SHIFT_SAMPLES,
    mel_bands: INPUT_DIM,
    f_min: 20.0,
    f_max: 8_000.0,
    input_scale: 32_768.0,
    padding: Padding::None,
    window: Window::Povey,
    window_length: Some(FRAME_LENGTH_SAMPLES),
    dc_removal: true,
    pre_emphasis: Some(PreEmphasis::Frame { alpha: 0.97 }),
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Htk,
    filter_shape: FilterShape::MelLinear,
    filter_norm: FilterNorm::None,
    table_precision: TablePrecision::Float32,
    compression: Compression::LogFloored {
        floor: f32::EPSILON,
    },
};

/// The door's static frame count: `feat [1, 80, 1, 6000]`,
/// `mask [1, 1, 1, 6000]` → `probs [1, 3, 1, 6000]`.
pub const WINDOW_FRAMES: usize = 6_000;
/// The receptive field — 8 memories, each 19 frames back and 20 ahead — so a
/// window keeps only frames that saw everything the whole sequence sees.
pub const WINDOW_MARGIN: usize = 160;
pub const WINDOW_STRIDE: usize = WINDOW_FRAMES - 2 * WINDOW_MARGIN;

pub const SMOOTH_WINDOW_SIZE: usize = 5;
pub const SPEECH_THRESHOLD: f32 = 0.4;
pub const SINGING_THRESHOLD: f32 = 0.5;
pub const MUSIC_THRESHOLD: f32 = 0.5;
pub const MIN_EVENT_FRAME: usize = 20;
pub const MAX_EVENT_FRAME: usize = 2_000;
pub const MIN_SILENCE_FRAME: usize = 20;
pub const MERGE_SILENCE_FRAME: usize = 0;
pub const EXTEND_SPEECH_FRAME: usize = 0;

#[inline]
pub const fn samples_for_frames(frame_count: usize) -> usize {
    if frame_count == 0 {
        0
    } else {
        (frame_count - 1) * FRAME_SHIFT_SAMPLES + FRAME_LENGTH_SAMPLES
    }
}

#[inline]
pub const fn frames_for_samples(sample_count: usize) -> usize {
    if sample_count < FRAME_LENGTH_SAMPLES {
        0
    } else {
        (sample_count - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES + 1
    }
}

/// One window of the door: it reads `WINDOW_FRAMES` frames from `read_from`
/// (the tail past the input masked) and keeps `keep_from..keep_to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoorWindow {
    pub read_from: usize,
    pub keep_from: usize,
    pub keep_to: usize,
}

/// The contract's windows over `frame_count` frames: one when they fit the
/// door; otherwise window k reads from `max(0, k·stride − margin)` and keeps
/// `k·stride..min((k + 1)·stride, frames)`.
pub fn door_windows(frame_count: usize) -> Vec<DoorWindow> {
    if frame_count <= WINDOW_FRAMES {
        return vec![DoorWindow {
            read_from: 0,
            keep_from: 0,
            keep_to: frame_count,
        }];
    }
    (0..frame_count.div_ceil(WINDOW_STRIDE))
        .map(|k| DoorWindow {
            read_from: (k * WINDOW_STRIDE).saturating_sub(WINDOW_MARGIN),
            keep_from: k * WINDOW_STRIDE,
            keep_to: ((k + 1) * WINDOW_STRIDE).min(frame_count),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sample_math_matches_kaldi_snip_edges() {
        assert_eq!(frames_for_samples(399), 0);
        assert_eq!(frames_for_samples(400), 1);
        assert_eq!(frames_for_samples(560), 2);
        assert_eq!(samples_for_frames(1), 400);
        assert_eq!(samples_for_frames(WINDOW_FRAMES), 960_240);
    }

    #[test]
    fn windows_tile_the_input_and_keep_a_margin_of_context() {
        assert_eq!(
            door_windows(2_011),
            vec![DoorWindow {
                read_from: 0,
                keep_from: 0,
                keep_to: 2_011
            }]
        );
        let windows = door_windows(6_998);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[1].read_from, WINDOW_STRIDE - WINDOW_MARGIN);
        for frames in [6_001, 11_360, 11_361, 60_000] {
            let windows = door_windows(frames);
            assert_eq!(windows[0].keep_from, 0);
            assert_eq!(windows.last().unwrap().keep_to, frames);
            for pair in windows.windows(2) {
                assert_eq!(pair[0].keep_to, pair[1].keep_from);
            }
            for window in &windows {
                assert!(
                    window.keep_from == 0 || window.keep_from - window.read_from >= WINDOW_MARGIN
                );
                assert!(
                    window.keep_to == frames
                        || window.read_from + WINDOW_FRAMES - window.keep_to >= WINDOW_MARGIN
                );
            }
        }
    }
}
