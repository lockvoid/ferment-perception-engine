//! MusicHighlighter log-mel spectrogram: `config::MEL_SPEC` through the
//! engine's one front-end.

use ferment_engine_core::Result;

use super::config::MEL_SPEC;

/// Compute the log-mel spectrogram for a 22.05 kHz mono signal. Output:
/// `[T, N_MELS]` row-major, one row per frame.
pub fn compute_log_mel(audio: &[f32]) -> Result<Vec<Vec<f32>>> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    let frames = MEL_SPEC.compute(audio)?;
    Ok(frames
        .values
        .chunks_exact(frames.bands)
        .map(<[f32]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;
    use crate::musichighlighter::config::{N_MELS, SAMPLE_RATE};

    fn read_f32(name: &str) -> Option<Vec<f32>> {
        fixtures::f32_le(&format!("MusicHighlighter/{name}.bin"))
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let mel = compute_log_mel(&[]).unwrap();
        assert!(mel.is_empty());
    }

    #[test]
    fn output_shape_matches_expected() {
        // 30s of silence at 22050 Hz should produce 1292 frames × 128 mels.
        let audio = vec![0.0f32; 30 * SAMPLE_RATE as usize];
        let mel = compute_log_mel(&audio).unwrap();
        assert_eq!(mel.len(), 1292);
        assert!(mel.iter().all(|row| row.len() == N_MELS));
    }

    #[test]
    fn silence_yields_zero_log_mel() {
        // log(1 + 1e4 * 0) = log(1) = 0 — silent input → all zero mel.
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let mel = compute_log_mel(&audio).unwrap();
        for row in &mel {
            for &v in row {
                assert!(v.abs() < 1e-6, "non-zero value {v} in silent log-mel");
            }
        }
    }

    #[test]
    fn matches_swift_python_fixture_when_available() {
        let Some(audio) = read_f32("test_audio") else {
            return;
        };
        let expected = read_f32("test_mel").unwrap();
        let actual = compute_log_mel(&audio).unwrap();
        assert_eq!(actual.len(), 1_292);
        let max_difference = actual
            .iter()
            .flatten()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 0.001,
            "MusicHighlighter mel max difference {max_difference} exceeded 0.001"
        );
    }
}
