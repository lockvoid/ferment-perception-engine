use super::config::INPUT_DIM;

/// FireRed corpus CMVN affine transform.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cmvn;

impl Cmvn {
    /// Apply `(x[t, d] - mean[d]) * inverse_stddev[d]` in place.
    ///
    /// Returns `false` and leaves the slice unchanged when its shape is not
    /// exactly `frame_count × 80`, matching the Swift helper's guard.
    pub fn apply(features: &mut [f32], frame_count: usize) -> bool {
        if features.len() != frame_count * INPUT_DIM {
            return false;
        }
        for row in features.chunks_exact_mut(INPUT_DIM) {
            for dimension in 0..INPUT_DIM {
                row[dimension] = (row[dimension] - MEANS[dimension]) * INVERSE_STDDEV[dimension];
            }
        }
        true
    }
}

// Keep the generated decimal literals byte-for-byte recognizable against the
// Swift conversion output. Rust rounds them to f32 at compile time.
#[allow(clippy::excessive_precision)]
pub const MEANS: [f32; INPUT_DIM] = [
    1.042295170e+01,
    1.086209774e+01,
    1.176454449e+01,
    1.249016476e+01,
    1.325983047e+01,
    1.389594364e+01,
    1.436494064e+01,
    1.459394836e+01,
    1.474972343e+01,
    1.466831493e+01,
    1.473079681e+01,
    1.477505207e+01,
    1.498905182e+01,
    1.517800522e+01,
    1.525352001e+01,
    1.532863712e+01,
    1.533401871e+01,
    1.528864193e+01,
    1.542766190e+01,
    1.524626637e+01,
    1.509257412e+01,
    1.529042149e+01,
    1.507575035e+01,
    1.518677330e+01,
    1.508867359e+01,
    1.517079735e+01,
    1.507017803e+01,
    1.515079498e+01,
    1.510853291e+01,
    1.511534500e+01,
    1.514128017e+01,
    1.513183212e+01,
    1.514519596e+01,
    1.519151878e+01,
    1.523547840e+01,
    1.530636978e+01,
    1.537302113e+01,
    1.541639423e+01,
    1.545985699e+01,
    1.539143276e+01,
    1.546357632e+01,
    1.539966106e+01,
    1.546290779e+01,
    1.544162941e+01,
    1.548496914e+01,
    1.555240154e+01,
    1.563809204e+01,
    1.570548916e+01,
    1.576700878e+01,
    1.585512352e+01,
    1.586726952e+01,
    1.589153767e+01,
    1.592314529e+01,
    1.597838306e+01,
    1.601480103e+01,
    1.604867554e+01,
    1.608202934e+01,
    1.609679985e+01,
    1.609373665e+01,
    1.607247925e+01,
    1.607551003e+01,
    1.602227020e+01,
    1.597676182e+01,
    1.589786434e+01,
    1.581274128e+01,
    1.571120548e+01,
    1.560419846e+01,
    1.555351925e+01,
    1.551025295e+01,
    1.546002388e+01,
    1.541568470e+01,
    1.537602806e+01,
    1.532834911e+01,
    1.529537106e+01,
    1.518547058e+01,
    1.501704502e+01,
    1.490507984e+01,
    1.462380695e+01,
    1.413809395e+01,
    1.331387043e+01,
];

#[allow(clippy::excessive_precision)]
pub const INVERSE_STDDEV: [f32; INPUT_DIM] = [
    2.494980842e-01,
    2.356323451e-01,
    2.314515263e-01,
    2.332233936e-01,
    2.318266034e-01,
    2.285335660e-01,
    2.243486941e-01,
    2.189892083e-01,
    2.183243781e-01,
    2.208259255e-01,
    2.229673564e-01,
    2.228841633e-01,
    2.223481089e-01,
    2.210064232e-01,
    2.199420184e-01,
    2.200544327e-01,
    2.207009196e-01,
    2.215081006e-01,
    2.223666757e-01,
    2.230529189e-01,
    2.233534157e-01,
    2.243890613e-01,
    2.254770100e-01,
    2.269007564e-01,
    2.282302380e-01,
    2.293147147e-01,
    2.304672748e-01,
    2.308355272e-01,
    2.314338237e-01,
    2.322065979e-01,
    2.325798869e-01,
    2.336197048e-01,
    2.343724072e-01,
    2.350825220e-01,
    2.357807904e-01,
    2.358919978e-01,
    2.360209823e-01,
    2.366379946e-01,
    2.374987602e-01,
    2.379845232e-01,
    2.389937788e-01,
    2.397481501e-01,
    2.403083593e-01,
    2.409769297e-01,
    2.414324880e-01,
    2.413546592e-01,
    2.407993823e-01,
    2.404740602e-01,
    2.399552464e-01,
    2.395228744e-01,
    2.394808829e-01,
    2.393650860e-01,
    2.392933965e-01,
    2.390219867e-01,
    2.385787368e-01,
    2.381470203e-01,
    2.380462140e-01,
    2.382419407e-01,
    2.386009544e-01,
    2.391540706e-01,
    2.392254025e-01,
    2.393830866e-01,
    2.397336066e-01,
    2.396056205e-01,
    2.402850240e-01,
    2.406181395e-01,
    2.406792939e-01,
    2.409620136e-01,
    2.404360622e-01,
    2.402152717e-01,
    2.397251427e-01,
    2.387199849e-01,
    2.374413162e-01,
    2.361950874e-01,
    2.333728075e-01,
    2.268023342e-01,
    2.257750332e-01,
    2.250384688e-01,
    2.263113707e-01,
    2.289949358e-01,
];

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    fn fixture(stem: &str, file: &str) -> String {
        format!("AudioEventDetector/{stem}/{file}")
    }

    fn assert_cmvn_matches(stem: &str) {
        let Some(mut actual) = fixtures::f32_le(&fixture(stem, "fbank.bin")) else {
            return;
        };
        let expected = fixtures::require_f32_le(&fixture(stem, "feat_cmvn.bin"));
        let frame_count = actual.len() / INPUT_DIM;
        assert!(Cmvn::apply(&mut actual, frame_count));
        assert_eq!(actual.len(), expected.len());
        let max_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1.0e-4,
            "{stem} CMVN max difference was {max_difference}"
        );
    }

    #[test]
    fn rejects_shape_mismatch_without_mutating() {
        let mut features = vec![1.0; INPUT_DIM - 1];
        let original = features.clone();
        assert!(!Cmvn::apply(&mut features, 1));
        assert_eq!(features, original);
    }

    #[test]
    fn affine_matches_first_dimension() {
        let mut features = vec![0.0; INPUT_DIM];
        features[0] = MEANS[0] + 2.0;
        assert!(Cmvn::apply(&mut features, 1));
        assert!((features[0] - 2.0 * INVERSE_STDDEV[0]).abs() < 1.0e-6);
    }

    #[test]
    fn test_cmvn_hairdryer() {
        assert_cmvn_matches("hairdryer");
    }

    #[test]
    fn test_cmvn_sec5() {
        assert_cmvn_matches("sec5");
    }
}
