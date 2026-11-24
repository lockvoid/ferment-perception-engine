//! Sinusoidal positional encoding. Matches iOS
//! `MusicHighlighter.positionalEncoding`: `inv_freq[j] = 1 / 10000^(2·(j/2)/d)`,
//! interleaved sin/cos pairs, with position 0 left as zeros.

/// Build a `[n_pos, d_pos]` row-major positional encoding matrix.
/// Position 0 is all zeros (matches iOS).
pub fn positional_encoding(n_pos: usize, d_pos: usize) -> Vec<f32> {
    let mut inv_freq = vec![0.0f32; d_pos];
    for (j, frequency) in inv_freq.iter_mut().enumerate() {
        // (j/2) is integer division — same pair shares the frequency.
        let exp = 2.0 * ((j / 2) as f32) / d_pos as f32;
        *frequency = 1.0 / 10_000_f32.powf(exp);
    }

    let mut pe = vec![0.0f32; n_pos * d_pos];
    for pos in 1..n_pos {
        let pos_f = pos as f32;
        let mut j = 0;
        while j + 1 < d_pos {
            let angle = pos_f * inv_freq[j];
            pe[pos * d_pos + j] = angle.sin();
            pe[pos * d_pos + j + 1] = angle.cos();
            j += 2;
        }
        if !d_pos.is_multiple_of(2) {
            pe[pos * d_pos + d_pos - 1] = (pos_f * inv_freq[d_pos - 1]).sin();
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    #[test]
    fn position_zero_is_all_zeros() {
        let pe = positional_encoding(8, 64);
        for &value in pe.iter().take(64) {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn shape_matches_n_pos_d_pos() {
        let pe = positional_encoding(10, 256);
        assert_eq!(pe.len(), 10 * 256);
    }

    #[test]
    fn sin_cos_pairs_at_pos_one() {
        // For pos=1, j=0 → sin(1.0), j=1 → cos(1.0)
        let pe = positional_encoding(2, 8);
        let row1 = &pe[8..16];
        // inv_freq[0] = 1 / 10000^0 = 1.0
        let expected_sin = 1.0_f32.sin();
        let expected_cos = 1.0_f32.cos();
        assert!((row1[0] - expected_sin).abs() < 1e-6);
        assert!((row1[1] - expected_cos).abs() < 1e-6);
    }

    #[test]
    fn matches_swift_python_fixture_when_available() {
        let Some(expected) = fixtures::f32_le("MusicHighlighter/test_pos_enc.bin") else {
            return;
        };
        let actual = positional_encoding(10, 256);
        assert_eq!(actual.len(), expected.len());
        let max_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_difference < 1e-5);
    }
}
