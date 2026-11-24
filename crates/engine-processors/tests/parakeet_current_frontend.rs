//! Current ParakeetMelTests values from the frozen iOS oracle. The real host
//! frontend consumes the same three-tone audio; expectations are fixed values.
use ferment_engine_processors::parakeet::{config::*, features};

fn three_tone(count: usize) -> Vec<f32> {
    (0..count)
        .map(|sample| {
            let t = sample as f64 / 16_000.0;
            (0.50 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 1500.0 * t).sin()
                + 0.10 * (2.0 * std::f64::consts::PI * 5000.0 * t).sin()) as f32
        })
        .collect()
}

fn near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected} +/- {tolerance}"
    );
}

/// Kill: use ceil for valid audio or omit one of the three encoder subsamplings.
#[test]
fn frame_arithmetic_matches_the_traced_graph() {
    assert_eq!(MEL_SPEC.frame_count(1_723), 11);
    assert_eq!(valid_mel_frames(1_723), 10);
    assert_eq!(MEL_SPEC.frame_count(WINDOW_SAMPLES), 1_501);
    assert_eq!(valid_mel_frames(WINDOW_SAMPLES), 1_500);
    assert_eq!(valid_encoder_frames(1_501), 188);
    assert_eq!(valid_encoder_frames(501), 63);
}

/// Kill: change Hann, pre-emphasis, Slaney filters or per-band normalization.
#[test]
fn golden_cells_match_the_current_published_values() {
    let (mel, valid) = features(&three_tone(1_723)).expect("real Parakeet frontend");
    assert_eq!(valid, 10);
    assert_eq!(mel.len(), MEL_BANDS * MEL_FRAMES);
    for (frame, band, expected) in [
        (0, 0, 2.6243355),
        (0, 5, 2.8425014),
        (0, 64, 2.6714416),
        (0, 127, 2.4562137),
        (3, 10, 0.3125439),
        (5, 40, -0.4846673),
        (9, 0, -0.813_853),
        (9, 100, -0.4264047),
        (9, 127, -0.4515118),
    ] {
        // ONNX consumes band-major rather than Swift's frame-major storage.
        near(mel[band * MEL_FRAMES + frame], expected, 5e-4);
    }
}

/// Kill: swap frame/band indexing or normalize across the whole spectrogram.
#[test]
fn frame_five_aggregate_statistics_match_the_current_published_values() {
    let (mel, _) = features(&three_tone(1_723)).unwrap();
    let row: Vec<f32> = (0..MEL_BANDS)
        .map(|band| mel[band * MEL_FRAMES + 5])
        .collect();
    near(row.iter().sum::<f32>() / row.len() as f32, -0.3842894, 1e-3);
    near(
        row.iter().copied().fold(f32::INFINITY, f32::min),
        -0.7801333,
        5e-4,
    );
    near(
        row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        0.3192385,
        5e-4,
    );
}

/// Kill: retain normalized values in a padding frame the attention mask excludes.
#[test]
fn frames_beyond_the_valid_count_are_exactly_zero() {
    let (mel, valid) = features(&three_tone(1_723)).unwrap();
    for band in 0..MEL_BANDS {
        for frame in valid..MEL_FRAMES {
            assert_eq!(
                mel[band * MEL_FRAMES + frame],
                0.0,
                "band {band}, frame {frame}"
            );
        }
    }
}

/// Kill: use instance normalization, population variance, or include padding.
#[test]
fn each_mel_bin_is_independently_normalized_over_valid_frames() {
    let (mel, valid) = features(&three_tone(16_000)).unwrap();
    assert_eq!(valid, 100);
    for band in (0..MEL_BANDS).step_by(16) {
        let row = &mel[band * MEL_FRAMES..band * MEL_FRAMES + valid];
        let mean = row.iter().sum::<f32>() / valid as f32;
        let variance =
            row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / (valid - 1) as f32;
        near(mean, 0.0, 1e-3);
        near(variance.sqrt(), 1.0, 1e-3);
    }
}

/// Kill: let storage padding alter any real feature, including its pre-emphasis tail.
#[test]
fn a_static_window_does_not_change_the_valid_frames() {
    let audio = three_tone(1_723);
    let normalized = MEL_SPEC
        .compute(&audio)
        .unwrap()
        .normalized_per_band(valid_mel_frames(audio.len()), NORMALIZATION_EPSILON);
    let tight = normalized.bands_major(Some(11));
    let padded = normalized.bands_major(Some(64));
    let (actual_model_input, _) = features(&audio).unwrap();
    for band in 0..MEL_BANDS {
        for frame in 0..10 {
            near(tight[band * 11 + frame], padded[band * 64 + frame], 1e-5);
            near(
                tight[band * 11 + frame],
                actual_model_input[band * MEL_FRAMES + frame],
                1e-5,
            );
        }
    }
}
