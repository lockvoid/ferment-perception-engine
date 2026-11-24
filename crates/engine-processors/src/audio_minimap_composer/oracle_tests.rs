//! Scenario-for-scenario translations of `AudioMinimapCorrectnessTests` and
//! `AudioMinimapShapeTests` from the Swift oracle suite.

use std::f32::consts::PI;

use super::ImageFormat;
use super::colormap::{MAGMA_LUT, magma};
use super::layout::Layout;
use super::mel::{FFT_SIZE, HOP_LENGTH, N_MELS, SAMPLE_RATE, TOP_DB, compute_mel_db};
use super::renderer::{RenderInput, render};
use super::rms;

fn sine(frequency: f32, sample_count: usize) -> Vec<f32> {
    (0..sample_count)
        .map(|index| (2.0 * PI * frequency * index as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

fn deterministic_noise(sample_count: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    (0..sample_count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 0x00ff_ffff_u32 as f32 * 2.0 - 1.0
        })
        .collect()
}

fn peak_band(audio: &[f32]) -> usize {
    let mel = compute_mel_db(audio).unwrap();
    let mut sums = [0.0_f32; N_MELS];
    for frame in mel {
        for (index, value) in frame.into_iter().enumerate() {
            sums[index] += value;
        }
    }
    sums.iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0
}

// AudioMinimapShapeTests

#[test]
fn test_mel_output_shape_and_db_range() {
    let audio = sine(440.0, SAMPLE_RATE as usize);
    let mel = compute_mel_db(&audio).unwrap();
    let pad = FFT_SIZE / 2;
    let padded = audio.len() + 2 * pad;
    let expected = (padded - FFT_SIZE) / HOP_LENGTH + 1;
    assert_eq!(mel.len(), expected);
    assert_eq!(mel.first().map(Vec::len), Some(N_MELS));
    assert_eq!(mel.last().map(Vec::len), Some(N_MELS));

    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for value in mel.iter().flatten().copied() {
        assert!(!value.is_nan());
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    assert!(minimum >= -TOP_DB);
    assert!((maximum - 0.0).abs() <= 1.0e-3);
}

#[test]
fn test_mel_empty_audio_returns_empty() {
    assert!(compute_mel_db(&[]).unwrap().is_empty());
}

#[test]
fn test_mel_silent_audio_stays_within_db_range() {
    let mel = compute_mel_db(&vec![0.0; SAMPLE_RATE as usize]).unwrap();
    assert!(!mel.is_empty());
    for value in mel.iter().flatten().copied() {
        assert!(!value.is_nan());
        assert!(value >= -TOP_DB);
        assert!(value <= 0.0);
    }
}

#[test]
fn test_rms_frame_count_matches_mel() {
    let audio = deterministic_noise(SAMPLE_RATE as usize * 2);
    assert_eq!(
        compute_mel_db(&audio).unwrap().len(),
        rms::compute(&audio).len()
    );
}

#[test]
fn test_rms_values_in_db_range() {
    let audio = sine(440.0, SAMPLE_RATE as usize);
    for value in rms::compute(&audio) {
        assert!(!value.is_nan());
        assert!(value >= -TOP_DB);
        assert!(value <= 0.0);
    }
}

#[test]
fn test_rms_empty_audio() {
    assert!(rms::compute(&[]).is_empty());
}

#[test]
fn test_magma_lut_size() {
    assert_eq!(MAGMA_LUT.len(), 256 * 3);
}

#[test]
fn test_magma_endpoints() {
    let low = magma(0.0);
    let high = magma(1.0);
    assert!(u16::from(low.0) + u16::from(low.1) + u16::from(low.2) < 50);
    assert!(u16::from(high.0) + u16::from(high.1) + u16::from(high.2) > 600);
}

#[test]
fn test_magma_clamps_out_of_range_inputs() {
    assert_eq!(magma(-5.0).0, magma(0.0).0);
    assert_eq!(magma(2.0).0, magma(1.0).0);
}

#[test]
fn test_layout_width_clamps_to_min() {
    assert_eq!(Layout::resolve(1.0, 2_048).total_width, 1_024);
}

#[test]
fn test_layout_width_clamps_to_max() {
    assert_eq!(Layout::resolve(3_600.0, 2_048).total_width, 2_048);
}

#[test]
fn test_layout_height_stable_across_durations() {
    let short = Layout::resolve(1.0, 2_048);
    let long = Layout::resolve(600.0, 2_048);
    assert_eq!(short.total_height, long.total_height);
}

#[test]
fn test_image_format_extension_and_mime() {
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Png.mime_type(), "image/png");
    assert_eq!(ImageFormat::Jpeg { quality: 90 }.extension(), "jpg");
    assert_eq!(ImageFormat::Jpeg { quality: 90 }.mime_type(), "image/jpeg");
}

// AudioMinimapCorrectnessTests

#[test]
fn test_pure_sine_concentrates_energy_in_one_band() {
    let mel = compute_mel_db(&sine(440.0, SAMPLE_RATE as usize * 2)).unwrap();
    assert!(!mel.is_empty());
    let mut sums = [0.0_f32; N_MELS];
    for frame in mel {
        for (index, value) in frame.into_iter().enumerate() {
            sums[index] += value;
        }
    }
    let peak = sums
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert!(peak > 5);
    assert!(peak < 80);
}

#[test]
fn test_higher_frequency_peaks_higher_up_in_mel_bands() {
    let sample_count = SAMPLE_RATE as usize * 2;
    assert!(peak_band(&sine(4_000.0, sample_count)) > peak_band(&sine(440.0, sample_count)));
}

#[test]
fn test_mel_max_is_always_zero_db() {
    let sample_count = SAMPLE_RATE as usize;
    let cases = [
        sine(440.0, sample_count),
        deterministic_noise(sample_count),
        {
            let mut impulse = vec![0.0; sample_count];
            impulse[0] = 1.0;
            impulse
        },
    ];
    for audio in cases {
        let maximum = compute_mel_db(&audio)
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((maximum - 0.0).abs() <= 1.0e-3);
    }
}

#[test]
fn test_rms_matches_analytic_value_for_sine_wave() {
    let values = rms::compute(&sine(440.0, SAMPLE_RATE as usize));
    assert!(values.len() > 10);
    let steady = &values[5..values.len() - 5];
    let mean = steady.iter().sum::<f32>() / steady.len() as f32;
    for value in steady {
        assert!((value - mean).abs() <= 0.5);
    }
}

#[test]
fn test_rms_response_to_amplitude_step() {
    let half = SAMPLE_RATE as usize / 2;
    let mut audio = vec![0.0; half * 2];
    for (index, sample) in audio[..half].iter_mut().enumerate() {
        *sample = 0.05 * (2.0 * PI * 440.0 * index as f32 / SAMPLE_RATE as f32).sin();
    }
    for (index, sample) in audio[half..].iter_mut().enumerate() {
        let absolute_index = index + half;
        *sample = 0.5 * (2.0 * PI * 440.0 * absolute_index as f32 / SAMPLE_RATE as f32).sin();
    }
    let values = rms::compute(&audio);
    assert!(values.len() > 10);
    let middle = values.len() / 2;
    let quiet = &values[2..middle - 2];
    let loud = &values[middle + 2..values.len() - 2];
    let quiet_average = quiet.iter().sum::<f32>() / quiet.len() as f32;
    let loud_average = loud.iter().sum::<f32>() / loud.len() as f32;
    assert!(loud_average > quiet_average + 10.0);
}

#[test]
fn test_rms_aligns_with_mel_frame_by_frame() {
    let audio = deterministic_noise(SAMPLE_RATE as usize * 3);
    assert_eq!(
        compute_mel_db(&audio).unwrap().len(),
        rms::compute(&audio).len()
    );
}

#[test]
fn test_silence_produces_flat_bottom_mel() {
    let mel = compute_mel_db(&vec![0.0; SAMPLE_RATE as usize]).unwrap();
    let first = mel[0][0];
    for value in mel.iter().flatten() {
        assert!((value - first).abs() <= 0.001);
    }
}

#[test]
fn test_renderer_accepts_tiny_mel() {
    let mel = vec![vec![-40.0; N_MELS]; 3];
    let rms = [-30.0, -25.0, -28.0];
    let layout = Layout::resolve(1.0, 2_048);
    let output = render(&RenderInput {
        mel_db: &mel,
        rms_db: &rms,
        bar_ruler: &[],
        duration_secs: 1.0,
        layout,
    });
    assert!(output.pixmap.width() > 0);
    assert!(output.pixmap.height() > 0);
}

#[test]
fn test_renderer_accepts_empty_rms() {
    let mel = vec![vec![-40.0; N_MELS]; 3];
    let layout = Layout::resolve(1.0, 2_048);
    let output = render(&RenderInput {
        mel_db: &mel,
        rms_db: &[],
        bar_ruler: &[],
        duration_secs: 1.0,
        layout,
    });
    assert!(output.pixmap.width() > 0);
}
