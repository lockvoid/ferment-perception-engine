//! Feature extraction for DeepFilterNet3.
//!
//! Two features fed into the encoder:
//! - **ERB features**: 32-band log-energy with running mean normalization
//!   (`compute_band_corr` → `log10·10` → `band_mean_norm_erb` in libdf)
//! - **Spec features**: 96-bin complex spectrum, unit-normalized by running
//!   mean of magnitude (`band_unit_norm`). Returned in channel-first layout.

use super::config::{ERB_WIDTHS, N_FREQ, NB_DF, NB_ERB, NORM_ALPHA};

/// ERB log-band features with running-mean normalization.
///
/// `spec` layout: `[T, n_freq, 2]` flat (interleaved real/imag), length
/// `n_frames * n_freq * 2`. Output: `[T, nb_erb]` flat row-major.
pub fn erb_features(spec: &[f32], n_frames: usize) -> Vec<f32> {
    debug_assert_eq!(spec.len(), n_frames * N_FREQ * 2);
    let mut result = vec![0.0f32; n_frames * NB_ERB];

    // MEAN_NORM_INIT in libdf: linear ramp -60 → -90 across NB_ERB bands.
    let mut mean_state = [0.0f32; NB_ERB];
    for (i, m) in mean_state.iter_mut().enumerate() {
        *m = -60.0 + (i as f32) * (-90.0 - -60.0) / (NB_ERB - 1) as f32;
    }

    for t in 0..n_frames {
        let spec_off = t * N_FREQ * 2;
        let mut freq_idx = 0usize;

        for band in 0..NB_ERB {
            let width = ERB_WIDTHS[band];
            let mut energy: f32 = 0.0;
            for f in freq_idx..(freq_idx + width) {
                let re = spec[spec_off + f * 2];
                let im = spec[spec_off + f * 2 + 1];
                energy += re * re + im * im;
            }
            energy /= width as f32;
            // libdf's `feat_erb` adds the floor, it does not clamp to it: in a
            // band a codec emptied the two differ by up to 3 dB.
            let mut val = (energy + 1e-10).log10() * 10.0;

            mean_state[band] = val * (1.0 - NORM_ALPHA) + mean_state[band] * NORM_ALPHA;
            val -= mean_state[band];
            val /= 40.0;

            result[t * NB_ERB + band] = val;
            freq_idx += width;
        }
    }

    result
}

/// Unit-normalized complex spec features for the first `NB_DF` bins.
///
/// Output layout `[2, T, nb_df]` flat (channel-first for ONNX) — matches
/// libdf's `band_unit_norm` and the iOS reference.
pub fn spec_features(spec: &[f32], n_frames: usize) -> Vec<f32> {
    debug_assert_eq!(spec.len(), n_frames * N_FREQ * 2);
    let mut result = vec![0.0f32; 2 * n_frames * NB_DF];

    // UNIT_NORM_INIT in libdf: linear ramp 0.001 → 0.0001 across NB_DF bins.
    let mut unit_state = vec![0.0f32; NB_DF];
    for (i, u) in unit_state.iter_mut().enumerate() {
        *u = 0.001 + (i as f32) * (0.0001 - 0.001) / (NB_DF - 1) as f32;
    }

    for t in 0..n_frames {
        let spec_off = t * N_FREQ * 2;
        for f in 0..NB_DF {
            let re = spec[spec_off + f * 2];
            let im = spec[spec_off + f * 2 + 1];
            let mag = (re * re + im * im).sqrt();

            unit_state[f] = mag * (1.0 - NORM_ALPHA) + unit_state[f] * NORM_ALPHA;
            let scale = unit_state[f].max(1e-20).sqrt();

            result[t * NB_DF + f] = re / scale;
            result[n_frames * NB_DF + t * NB_DF + f] = im / scale;
        }
    }

    result
}

/// Apply DF lookahead: shift features back by `df_lookahead` frames so
/// that the encoder sees a small future context. Mirrors iOS
/// `applyLookaheadPad`, which leaves a span of no more frames than the shift
/// as it is. Returns a new buffer of the same shape.
///
/// `feature_size` is the per-frame element count (e.g. `NB_ERB` or
/// `2 * NB_DF` for the channel-interleaved spec features).
pub fn lookahead_pad(data: &[f32], n_frames: usize, feature_size: usize) -> Vec<f32> {
    use super::config::DF_LOOKAHEAD;
    if n_frames <= DF_LOOKAHEAD {
        return data.to_vec();
    }
    let total = n_frames * feature_size;

    if data.len() == total {
        // Flat case (ERB features): shift by `skip * feature_size`,
        // pad zeros at the end.
        let skip = DF_LOOKAHEAD * feature_size;
        let mut result = Vec::with_capacity(total);
        result.extend_from_slice(&data[skip..]);
        result.resize(total, 0.0);
        result
    } else {
        // Multi-channel case (spec features in `[2, T, nb_df]` layout):
        // shift each channel independently in-frame.
        let n_channels = data.len() / (n_frames * NB_DF);
        let mut result = vec![0.0f32; data.len()];
        for ch in 0..n_channels {
            let ch_off = ch * n_frames * NB_DF;
            for t in DF_LOOKAHEAD..n_frames {
                let src = ch_off + t * NB_DF;
                let dst = ch_off + (t - DF_LOOKAHEAD) * NB_DF;
                result[dst..dst + NB_DF].copy_from_slice(&data[src..src + NB_DF]);
            }
        }
        result
    }
}

/// What the door sees for one chunk of `frames` spectrum frames (`[T, 481,
/// 2]`): libdf's two features, their running normalisations started fresh,
/// each shifted `DF_LOOKAHEAD` frames earlier with the chunk's last frames
/// zero — iOS's `processSpec`, and `ferment-models-export`'s golden chunk.
/// Returns `feat_erb` `[T, 32]` and `feat_spec` `[2, T, 96]`.
#[must_use]
pub fn chunk_features(spectrum: &[f32], frames: usize) -> (Vec<f32>, Vec<f32>) {
    (
        lookahead_pad(&erb_features(spectrum, frames), frames, NB_ERB),
        lookahead_pad(&spec_features(spectrum, frames), frames, NB_DF),
    )
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures::export;

    use super::*;
    use approx::assert_abs_diff_eq;

    fn worst(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn erb_features_zero_input_yields_init_normalization() {
        // Zero input → log10(1e-10)*10 = -100 dB → after subtracting init
        // mean ramp (-60..-90) and dividing by 40, result is bounded.
        let spec = vec![0.0f32; 4 * N_FREQ * 2];
        let out = erb_features(&spec, 4);
        assert_eq!(out.len(), 4 * NB_ERB);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn spec_features_zero_input_yields_zero() {
        let spec = vec![0.0f32; 8 * N_FREQ * 2];
        let out = spec_features(&spec, 8);
        assert_eq!(out.len(), 2 * 8 * NB_DF);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// The export's golden chunk: libdf's own `df_features` over 15 s of the
    /// hairdryer clip, shifted as upstream's `pad_feat` shifts it. The fresh
    /// normalisations and the zeroed tail are what make these the features of
    /// ONE chunk rather than of a longer signal.
    #[test]
    fn a_chunks_features_are_libdfs() {
        let Some(spectrum) = export::f32_le("dfnet/goldens/dfnet_spectrum.f32") else {
            return;
        };
        let frames = spectrum.len() / (N_FREQ * 2);
        assert_eq!(frames, 1_500);
        let (feat_erb, feat_spec) = chunk_features(&spectrum, frames);
        let erb = worst(
            &feat_erb,
            &export::require_f32_le("dfnet/goldens/dfnet_feat_erb_in.f32"),
        );
        let spec = worst(
            &feat_spec,
            &export::require_f32_le("dfnet/goldens/dfnet_feat_spec_in.f32"),
        );
        eprintln!("[dfnet-features] max |Δ| vs libdf: feat_erb {erb:.3e}, feat_spec {spec:.3e}");
        assert!(erb < 1e-4, "feat_erb: {erb}");
        assert!(spec < 1e-3, "feat_spec: {spec}");
    }

    #[test]
    fn a_span_no_longer_than_the_lookahead_is_left_as_it_is() {
        use super::super::config::DF_LOOKAHEAD;
        for frames in 1..=DF_LOOKAHEAD {
            let erb = (0..frames * NB_ERB)
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            assert_eq!(lookahead_pad(&erb, frames, NB_ERB), erb);
            let spec = (0..2 * frames * NB_DF)
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            assert_eq!(lookahead_pad(&spec, frames, NB_DF), spec);
        }
    }

    #[test]
    fn spec_features_layout_is_channel_first() {
        // For nonzero input, real channel = result[t*NB_DF+f], imag channel
        // = result[n_frames*NB_DF + t*NB_DF + f]. Construct an input where
        // we can distinguish the two.
        let n = 4;
        let mut spec = vec![0.0f32; n * N_FREQ * 2];
        for t in 0..n {
            for f in 0..NB_DF {
                let off = t * N_FREQ * 2 + f * 2;
                spec[off] = 1.0; // re
                spec[off + 1] = 2.0; // im
            }
        }
        let out = spec_features(&spec, n);
        // Real channel: positive nonzero; imaginary channel: roughly 2x
        for t in 0..n {
            for f in 0..NB_DF {
                let re = out[t * NB_DF + f];
                let im = out[n * NB_DF + t * NB_DF + f];
                assert!(re > 0.0);
                assert!(im > 0.0);
                assert!(im > re);
            }
        }
    }

    #[test]
    fn lookahead_pad_flat_shifts_correctly() {
        use super::super::config::DF_LOOKAHEAD;
        let n_frames = 6;
        let feat_size = NB_ERB;
        let mut data = vec![0.0f32; n_frames * feat_size];
        // Tag each frame with its index for easy verification
        for t in 0..n_frames {
            for f in 0..feat_size {
                data[t * feat_size + f] = t as f32;
            }
        }
        let padded = lookahead_pad(&data, n_frames, feat_size);

        // First (n_frames - DF_LOOKAHEAD) frames should now be frames
        // [DF_LOOKAHEAD..n_frames]. Last DF_LOOKAHEAD frames should be zeros.
        for t in 0..(n_frames - DF_LOOKAHEAD) {
            assert_abs_diff_eq!(padded[t * feat_size], (t + DF_LOOKAHEAD) as f32);
        }
        for t in (n_frames - DF_LOOKAHEAD)..n_frames {
            assert_abs_diff_eq!(padded[t * feat_size], 0.0);
        }
    }
}
