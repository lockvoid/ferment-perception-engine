//! 9-bit per-frame color histograms (3 bits per channel), L2-normalized.
//!
//! Bin assignment matches `AutoShot/ColorHistograms.compute_color_histograms`:
//!   `bin = ((R >> 5) << 6) + ((G >> 5) << 3) + (B >> 5)`  (range 0..511)
//!
//! Per-frame L2 norm matches `torch.norm(p=2, dim=2)`.

use super::config::{FRAME_HEIGHT, FRAME_WIDTH, N_COLOR_BINS};

/// Compute color histograms for a (T, H, W, RGB) uint8 buffer.
/// Returns `[T * 512]` flat — `result[t*512 + b]` is the L2-normalized count.
pub fn color_histograms(frames: &[u8], n_frames: usize) -> Vec<f32> {
    color_histograms_with_dimensions(frames, n_frames, FRAME_HEIGHT, FRAME_WIDTH)
}

fn color_histograms_with_dimensions(
    frames: &[u8],
    n_frames: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let pixels_per_frame = height * width;
    debug_assert_eq!(frames.len(), n_frames * pixels_per_frame * 3);

    let mut histograms = vec![0.0f32; n_frames * N_COLOR_BINS];
    for t in 0..n_frames {
        let frame_off = t * pixels_per_frame * 3;
        let hist_off = t * N_COLOR_BINS;
        for p in 0..pixels_per_frame {
            let r = frames[frame_off + p * 3] as usize;
            let g = frames[frame_off + p * 3 + 1] as usize;
            let b = frames[frame_off + p * 3 + 2] as usize;
            let bin = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
            histograms[hist_off + bin] += 1.0;
        }
        // Per-frame L2 normalize
        let mut norm: f32 = 0.0;
        for v in &histograms[hist_off..hist_off + N_COLOR_BINS] {
            norm += v * v;
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in &mut histograms[hist_off..hist_off + N_COLOR_BINS] {
                *v *= inv;
            }
        }
    }
    histograms
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    #[test]
    fn pure_black_yields_one_at_bin_zero() {
        let frames = vec![0u8; FRAME_HEIGHT * FRAME_WIDTH * 3];
        let h = color_histograms(&frames, 1);
        // All pixels at bin 0; after L2 normalize, bin 0 = 1.0, rest 0.
        assert!((h[0] - 1.0).abs() < 1e-6);
        for &v in &h[1..] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn even_split_two_bins_yields_sqrt_half_each() {
        let n = FRAME_HEIGHT * FRAME_WIDTH;
        let mut frames = vec![0u8; n * 3];
        // Half pixels black (bin 0), half pixels (255,0,0) → bin = (7<<6) = 448
        for p in 0..n / 2 {
            frames[(n / 2 + p) * 3] = 255;
        }
        let h = color_histograms(&frames, 1);
        let bin_high = 7usize << 6;
        let expected = (1.0_f32 / 2.0).sqrt();
        assert!((h[0] - expected).abs() < 1e-6);
        assert!((h[bin_high] - expected).abs() < 1e-6);
    }

    // VideoSceneDetectorProcessingTests.swift

    #[test]
    fn color_histogram_matches_python_reference() {
        let Some(frames) = fixtures::bytes("VideoSceneDetector/test_frames.bin") else {
            return;
        };
        let expected = fixtures::require_f32_le("VideoSceneDetector/test_color_hist.bin");
        let actual = color_histograms(&frames, 100);
        assert_eq!(actual.len(), expected.len());
        let max_difference = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_difference < 1e-5,
            "histogram max difference {max_difference} exceeded 1e-5"
        );
        for (frame, histogram) in actual.chunks_exact(N_COLOR_BINS).enumerate() {
            let norm = histogram
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "frame {frame} histogram norm {norm} drifted from one"
            );
        }
    }

    #[test]
    fn color_histogram_single_pixel_goes_into_one_bin() {
        let histogram = color_histograms_with_dimensions(&[224, 32, 0], 1, 1, 1);
        assert_eq!(histogram.len(), 512);
        assert!((histogram[456] - 1.0).abs() < 1e-6);
        for (index, value) in histogram.iter().enumerate() {
            if index != 456 {
                assert!(
                    value.abs() < 1e-6,
                    "unexpected value {value} in bin {index}"
                );
            }
        }
    }
}
