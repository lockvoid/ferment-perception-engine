//! Spectral enhancement: apply ERB mask, deep filtering, merge low/high.

use super::config::{DF_LOOKAHEAD, DF_ORDER, ERB_WIDTHS, N_FREQ, NB_DF, NB_ERB};

/// Multiply the 32-band ERB mask back onto the 481-bin complex spectrum.
/// Each band's gain applies to all bins within that band's width.
pub fn apply_erb_mask(spec: &[f32], mask: &[f32], n_frames: usize) -> Vec<f32> {
    debug_assert_eq!(spec.len(), n_frames * N_FREQ * 2);
    debug_assert_eq!(mask.len(), n_frames * NB_ERB);
    let mut result = spec.to_vec();
    for t in 0..n_frames {
        let mut freq_idx = 0usize;
        for band in 0..NB_ERB {
            let width = ERB_WIDTHS[band];
            let gain = mask[t * NB_ERB + band];
            for f in freq_idx..(freq_idx + width) {
                let idx = t * N_FREQ * 2 + f * 2;
                result[idx] *= gain;
                result[idx + 1] *= gain;
            }
            freq_idx += width;
        }
    }
    result
}

/// Deep filtering: centered-window complex convolution over `[t-2, t+2]`
/// per frequency bin (`f < NB_DF`). Mirrors libdf `df_op`:
/// `output[t,f] = Σ_k coef[k,t,f] * spec[t-2+k, f]`.
///
/// `coefs` layout matches the iOS reorder: `[K, T, NB_DF, 2]`.
pub fn apply_deep_filtering(spec: &[f32], coefs: &[f32], n_frames: usize) -> Vec<f32> {
    debug_assert_eq!(spec.len(), n_frames * N_FREQ * 2);
    debug_assert_eq!(coefs.len(), DF_ORDER * n_frames * NB_DF * 2);
    let mut result = vec![0.0f32; n_frames * N_FREQ * 2];

    for t in 0..n_frames {
        for f in 0..NB_DF {
            let mut re_acc = 0.0f32;
            let mut im_acc = 0.0f32;

            for k in 0..DF_ORDER {
                let src_t = t as isize - DF_LOOKAHEAD as isize + k as isize;
                if src_t < 0 || src_t >= n_frames as isize {
                    continue;
                }
                let src_t = src_t as usize;

                let spec_idx = src_t * N_FREQ * 2 + f * 2;
                let coef_idx = k * n_frames * NB_DF * 2 + t * NB_DF * 2 + f * 2;

                let s_re = spec[spec_idx];
                let s_im = spec[spec_idx + 1];
                let c_re = coefs[coef_idx];
                let c_im = coefs[coef_idx + 1];

                re_acc += c_re * s_re - c_im * s_im;
                im_acc += c_re * s_im + c_im * s_re;
            }

            let out_idx = t * N_FREQ * 2 + f * 2;
            result[out_idx] = re_acc;
            result[out_idx + 1] = im_acc;
        }
    }

    result
}

/// Merge: `deep_filtered` for low bins (< NB_DF), `masked` for high bins.
/// Mutates a copy of `deep_filtered` and returns it.
pub fn merge(deep_filtered: &[f32], masked: &[f32], n_frames: usize) -> Vec<f32> {
    debug_assert_eq!(deep_filtered.len(), n_frames * N_FREQ * 2);
    debug_assert_eq!(masked.len(), n_frames * N_FREQ * 2);
    let mut result = deep_filtered.to_vec();
    for t in 0..n_frames {
        for f in NB_DF..N_FREQ {
            let idx = t * N_FREQ * 2 + f * 2;
            result[idx] = masked[idx];
            result[idx + 1] = masked[idx + 1];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_erb_mask_unit_gain_is_identity() {
        let n_frames = 3;
        let mut spec = vec![0.0f32; n_frames * N_FREQ * 2];
        for (i, x) in spec.iter_mut().enumerate() {
            *x = i as f32;
        }
        let mask = vec![1.0f32; n_frames * NB_ERB];
        let out = apply_erb_mask(&spec, &mask, n_frames);
        assert_eq!(out, spec);
    }

    #[test]
    fn apply_erb_mask_zero_gain_zeros_band() {
        let n_frames = 1;
        let mut spec = vec![1.0f32; n_frames * N_FREQ * 2];
        let mut mask = vec![1.0f32; NB_ERB];
        mask[0] = 0.0; // zero out first ERB band

        let out = apply_erb_mask(&spec, &mask, n_frames);
        // First band is `ERB_WIDTHS[0]` = 2 bins → 4 floats zeroed
        for f in 0..ERB_WIDTHS[0] {
            assert_eq!(out[f * 2], 0.0);
            assert_eq!(out[f * 2 + 1], 0.0);
        }
        // Beyond first band: untouched
        let next_idx = ERB_WIDTHS[0] * 2;
        assert_eq!(out[next_idx], 1.0);

        // Suppress unused warning for the spec
        spec.truncate(0);
    }

    #[test]
    fn merge_keeps_low_from_filtered_high_from_masked() {
        let n_frames = 1;
        let filtered = vec![1.0f32; N_FREQ * 2];
        let masked = vec![2.0f32; N_FREQ * 2];
        let out = merge(&filtered, &masked, n_frames);
        // Low bins (< NB_DF): from filtered (=1.0)
        for f in 0..NB_DF {
            assert_eq!(out[f * 2], 1.0);
        }
        // High bins (>= NB_DF): from masked (=2.0)
        for f in NB_DF..N_FREQ {
            assert_eq!(out[f * 2], 2.0);
        }
    }

    #[test]
    fn deep_filtering_zero_coefs_yields_zero() {
        let n_frames = 4;
        let spec = vec![1.0f32; n_frames * N_FREQ * 2];
        let coefs = vec![0.0f32; DF_ORDER * n_frames * NB_DF * 2];
        let out = apply_deep_filtering(&spec, &coefs, n_frames);
        // Output is zero in the NB_DF region; high bins also zero (since result is
        // initialised to zero and we only fill `f < NB_DF`).
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn deep_filtering_identity_at_lookahead_passes_signal() {
        // If only k=DF_LOOKAHEAD has coef=(1+0i), output should equal input
        // for f<NB_DF (current frame contribution only).
        let n_frames = 3;
        let mut spec = vec![0.0f32; n_frames * N_FREQ * 2];
        for t in 0..n_frames {
            for f in 0..NB_DF {
                spec[t * N_FREQ * 2 + f * 2] = (t * NB_DF + f) as f32;
            }
        }
        let mut coefs = vec![0.0f32; DF_ORDER * n_frames * NB_DF * 2];
        for t in 0..n_frames {
            for f in 0..NB_DF {
                let idx = DF_LOOKAHEAD * n_frames * NB_DF * 2 + t * NB_DF * 2 + f * 2;
                coefs[idx] = 1.0; // (1 + 0i)
            }
        }
        let out = apply_deep_filtering(&spec, &coefs, n_frames);
        for t in 0..n_frames {
            for f in 0..NB_DF {
                let idx = t * N_FREQ * 2 + f * 2;
                assert_eq!(out[idx], spec[idx]);
            }
        }
    }
}
