//! BeatThis log-mel spectrogram: `config::MEL_SPEC` through the engine's one
//! front-end. (Until 2026-09-21 this file carried its own STFT + filterbank —
//! one of the per-processor copies iOS retired in `beda1279c`.)

use ferment_engine_core::Result;

use super::config::MEL_SPEC;

/// Compute log-mel spectrogram for a 22.05 kHz mono signal. Output shape:
/// `[T, N_MELS]` row-major.
pub fn compute_log_mel(audio: &[f32]) -> Result<Vec<Vec<f32>>> {
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
    use crate::beatthis::config::{N_MELS, SAMPLE_RATE};

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(compute_log_mel(&[]).unwrap().is_empty());
    }

    #[test]
    fn silent_input_yields_zero_log_mel() {
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let mel = compute_log_mel(&audio).unwrap();
        for row in &mel {
            for &v in row {
                assert!(v.abs() < 1e-6, "non-zero {v} in silent log-mel");
            }
        }
    }

    #[test]
    fn test_beat_this_mel_spectrogram_matches_python_reference() {
        let Some(audio) = fixtures::f32_le("test_audio.bin") else {
            return;
        };
        let expected = fixtures::require_f32_le("test_mel_reference.bin");
        let actual = compute_log_mel(&audio).unwrap();
        assert_eq!(audio.len(), 44_100);
        assert_eq!(actual.len(), 101);
        assert_eq!(expected.len(), 101 * N_MELS);

        let max_difference = actual
            .iter()
            .flatten()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 0.001,
            "BeatThis mel max difference {max_difference} exceeded 0.001"
        );
    }

    #[test]
    fn frame_count_matches_50_fps() {
        // 1 second @ 22050 / 441 hop = 50 hops, plus 1 for the centered
        // first frame = 51. (Note: 22050 / 441 = 50 exactly, no remainder.)
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let mel = compute_log_mel(&audio).unwrap();
        assert_eq!(mel.len(), 51);
    }
}
