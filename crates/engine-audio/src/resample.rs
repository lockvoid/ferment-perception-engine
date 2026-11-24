//! Audio resampling backed by rubato. Two modes:
//! - `resample_sinc` — band-limited sinc interpolation. Default for production.
//! - `resample_fft` — FFT-based, ~3x faster, indistinguishable for most
//!   speech/music. Use when latency matters.

use rubato::{
    FftFixedIn, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use ferment_engine_core::{EngineError, Result};

/// Swift-oracle-compatible linear interpolation.
///
/// Production processors normally use [`resample_sinc`], but this helper is
/// retained for contracts whose timing was defined by the original vDSP
/// `vDSP_vlint` path.
#[must_use]
pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || source_rate == 0 || target_rate == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let output_count =
        (samples.len() as f64 * f64::from(target_rate) / f64::from(source_rate)) as usize;
    let maximum_index = samples.len() - 1;
    (0..output_count)
        .map(|output_index| {
            let position = (output_index as f64 * f64::from(source_rate) / f64::from(target_rate))
                .min(maximum_index as f64);
            let left = position as usize;
            let right = (left + 1).min(maximum_index);
            let fraction = (position - left as f64) as f32;
            samples[left] + (samples[right] - samples[left]) * fraction
        })
        .collect()
}

/// Downsample/upsample mono `f32` samples from `source_rate` to `target_rate`
/// using a high-quality band-limited sinc filter.
pub fn resample_sinc(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 0 || target_rate == 0 {
        return Err(EngineError::invalid_config("sample rates must be > 0"));
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size = 1024;
    let ratio = f64::from(target_rate) / f64::from(source_rate);

    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| EngineError::internal(format!("sinc resampler init: {e}")))?;

    let count = (samples.len() as f64 * ratio).round() as usize;
    // The pinned Rubato 0.16 SincFixedIn initializes last_index at half a
    // kernel BEFORE the input origin. Its emitted samples are time-aligned;
    // output_delay describes buffered lookahead, not leading zero samples.
    // Keep this distinct from the FFT path and grade both with real impulses.
    drive_resampler(&mut resampler, samples, chunk_size, count, 0)
}

/// Faster FFT-based resampler. Good for batch / non-realtime workloads when
/// you don't need sinc quality.
pub fn resample_fft(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 0 || target_rate == 0 {
        return Err(EngineError::invalid_config("sample rates must be > 0"));
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        chunk_size,
        2, // sub_chunks
        1,
    )
    .map_err(|e| EngineError::internal(format!("fft resampler init: {e}")))?;

    let count =
        (samples.len() as f64 * f64::from(target_rate) / f64::from(source_rate)).round() as usize;
    let delay = resampler.output_delay();
    drive_resampler(&mut resampler, samples, chunk_size, count, delay)
}

/// Common driver: feed `samples` to a resampler in `chunk_size` chunks,
/// drain the final real content, then remove only converter padding. The
/// returned clip has the source duration, rounded to the nearest output frame.
fn drive_resampler<R>(
    resampler: &mut R,
    samples: &[f32],
    chunk_size: usize,
    output_count: usize,
    leading_padding: usize,
) -> Result<Vec<f32>>
where
    R: Resampler<f32>,
{
    let mut output = Vec::<f32>::new();
    let mut input_buf = vec![vec![0.0f32; chunk_size]];
    let mut idx = 0;

    while idx + chunk_size <= samples.len() {
        input_buf[0].copy_from_slice(&samples[idx..idx + chunk_size]);
        let out = resampler
            .process(&input_buf, None)
            .map_err(|e| EngineError::internal(format!("resample chunk: {e}")))?;
        output.extend_from_slice(&out[0]);
        idx += chunk_size;
    }

    // Final partial chunk (if any)
    if idx < samples.len() {
        let remaining = samples.len() - idx;
        input_buf[0][..remaining].copy_from_slice(&samples[idx..]);
        input_buf[0][remaining..].fill(0.0);
        let out = resampler
            .process_partial(Some(&input_buf), None)
            .map_err(|e| EngineError::internal(format!("resample partial: {e}")))?;
        output.extend_from_slice(&out[0]);
    } else {
        // Drain any tail still inside the resampler
        let out = resampler
            .process_partial::<Vec<f32>>(None, None)
            .map_err(|e| EngineError::internal(format!("resample drain: {e}")))?;
        output.extend_from_slice(&out[0]);
    }

    let needed = output_count
        .checked_add(leading_padding)
        .ok_or_else(|| EngineError::invalid_config("resampled clip length overflow"))?;
    while output.len() < needed {
        let out = resampler
            .process_partial::<Vec<f32>>(None, None)
            .map_err(|e| EngineError::internal(format!("resample final content: {e}")))?;
        if out[0].is_empty() {
            return Err(EngineError::internal(
                "resampler stopped before the final content",
            ));
        }
        output.extend_from_slice(&out[0]);
    }
    if leading_padding > 0 {
        output.drain(..leading_padding);
    }
    output.truncate(output_count);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_abs_diff_eq;

    use super::*;

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_resample_identity() {
        let audio = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(resample_linear(&audio, 44_100, 44_100), audio);
    }

    #[test]
    fn test_resample_downsample() {
        let audio = (0..10).map(|value| value as f32).collect::<Vec<_>>();
        let result = resample_linear(&audio, 44_100, 22_050);
        assert_eq!(result.len(), 5);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 0.001);
    }

    #[test]
    fn test_resample_upsample() {
        let result = resample_linear(&[0.0, 1.0, 0.0], 22_050, 44_100);
        assert_eq!(result.len(), 6);
        assert_abs_diff_eq!(result[1], 0.5, epsilon = 0.01);
    }

    #[test]
    fn same_rate_is_identity_sinc() {
        let s = sine(440.0, 44_100, 4_410);
        let r = resample_sinc(&s, 44_100, 44_100).unwrap();
        assert_eq!(r.len(), s.len());
        for (a, b) in s.iter().zip(r.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let r = resample_sinc(&[], 16_000, 22_050).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn rejects_zero_rates() {
        assert!(resample_sinc(&[0.0; 10], 0, 16_000).is_err());
        assert!(resample_sinc(&[0.0; 10], 16_000, 0).is_err());
    }

    #[test]
    fn downsample_44k_to_16k_preserves_exact_duration() {
        // One second remains one second. The real current iOS AudioReader
        // independently yields 16,000 samples for this source duration.
        let s = sine(440.0, 44_100, 44_100);
        let r = resample_sinc(&s, 44_100, 16_000).unwrap();
        assert_eq!(r.len(), 16_000);
    }

    /// An impulse at 50 ms must still appear at 50 ms. A sinc/FFT filter's
    /// internal delay is not part of the audio. Current iOS AudioReader's
    /// recorded oracle also puts this peak at output sample 800.
    #[test]
    fn rate_conversion_preserves_content_time_zero() {
        let mut source = vec![0.0; 44_100];
        source[2_205] = 1.0;
        for resample in [resample_sinc, resample_fft] {
            let output = resample(&source, 44_100, 16_000).unwrap();
            let peak = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            assert!(
                peak.abs_diff(800) <= 1,
                "50 ms impulse moved to sample {peak}"
            );
        }
    }

    /// Non-multiple input blocks must retain the final real content while
    /// excluding the padding used only to drain the converter.
    #[test]
    fn partial_blocks_keep_the_last_content_without_a_padded_tail() {
        let mut source = vec![0.0; 44_117];
        source[44_000] = 1.0;
        for resample in [resample_sinc, resample_fft] {
            let output = resample(&source, 44_100, 16_000).unwrap();
            assert_eq!(output.len(), 16_006);
            let peak = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            assert!(peak.abs_diff(15_964) <= 1, "last impulse moved to {peak}");
            assert!(
                output[peak].abs() > 0.2,
                "the last pulse was lost while draining"
            );
        }
    }

    #[test]
    fn downsample_preserves_frequency_content() {
        // Resample a 1 kHz tone from 44.1 kHz to 16 kHz. The output should
        // still be a recognizable 1 kHz tone — measure peak FFT bin.
        let sr_src = 44_100u32;
        let sr_dst = 16_000u32;
        let dur_secs = 0.5;
        let n_src = (f64::from(sr_src) * dur_secs) as usize;
        let s = sine(1000.0, sr_src, n_src);
        let r = resample_sinc(&s, sr_src, sr_dst).unwrap();

        // Skip transients at edges, take a middle section
        let middle = &r[r.len() / 4..r.len() * 3 / 4];

        // Trivial frequency check: estimate frequency by zero-crossings.
        let mut zc = 0usize;
        for i in 1..middle.len() {
            if (middle[i - 1] >= 0.0) != (middle[i] >= 0.0) {
                zc += 1;
            }
        }
        let secs = middle.len() as f32 / sr_dst as f32;
        let est_freq = zc as f32 / (2.0 * secs);
        assert!(
            (est_freq - 1000.0).abs() < 20.0,
            "estimated freq {est_freq} != 1000 Hz"
        );
    }

    #[test]
    fn fft_resampler_works_for_simple_case() {
        let s = sine(440.0, 48_000, 24_000); // 0.5s
        let r = resample_fft(&s, 48_000, 16_000).unwrap();
        assert_eq!(r.len(), 8_000);
    }
}
