//! Local contrast normalization of an 8-bit luma plane — the frame as
//! STRUCTURE, with brightness divided out: every pixel becomes its z-score
//! against the mean/variance of a 15×15 window around it, re-quantized to
//! 8 bits (128 + 32·z). A strobe hit, a sweeping beam or a dimming stage
//! changes brightness, not structure; on the normalized plane the same
//! texture matches the same texture whatever the light did.
//!
//! Port of iOS `PhotometricNormalizer.swift`. iOS runs the two box passes
//! through `vImageConvolve_PlanarF` with `kvImageEdgeExtend`; this is the
//! same separable 15-tap uniform convolution with edge replication, written
//! out. A running-sum box blur would be cheaper but accumulates f32 drift
//! across a row, and the very next step subtracts `E[x]²` from `E[x²]` —
//! catastrophic cancellation on a flat wall is exactly what `VARIANCE_FLOOR`
//! exists to bound, so the taps stay honest.

use crate::slow_motion_renderer::LumaPlane;

/// Side of the local window, in pixels.
const WINDOW_SIDE: usize = 15;
/// σ ⇒ 32 levels, so a z of ±4 fills the byte.
const SCALE: f32 = 32.0;
/// Added to the local variance before dividing: a flat wall carrying only
/// sensor noise (σ ≈ 2–3 luma) must not blow up into full-contrast texture.
/// 25 = (5 luma)².
const VARIANCE_FLOOR: f32 = 25.0;

/// The z-score plane. Returns a packed plane of the same dimensions; an
/// invalid or empty input comes back unchanged in shape with no data.
#[must_use]
pub fn normalize(plane: &LumaPlane) -> LumaPlane {
    let (width, height) = (plane.width, plane.height);
    if !plane.is_valid() {
        return LumaPlane {
            width,
            height,
            bytes_per_row: width,
            data: vec![128; width * height],
        };
    }

    let mut source = vec![0.0_f32; width * height];
    for y in 0..height {
        let row = y * plane.bytes_per_row;
        for x in 0..width {
            source[y * width + x] = f32::from(plane.data[row + x]);
        }
    }
    let squared: Vec<f32> = source.iter().map(|value| value * value).collect();

    let mean = box_blur(&source, width, height);
    let mean_of_squares = box_blur(&squared, width, height);

    let mut data = vec![0_u8; width * height];
    for index in 0..width * height {
        let variance = mean_of_squares[index] - mean[index] * mean[index] + VARIANCE_FLOOR;
        let z = (source[index] - mean[index]) / variance.sqrt();
        // `vDSP_vfixru8` rounds; ties are measure-zero on real footage.
        data[index] = (z * SCALE + 128.0).clamp(0.0, 255.0).round() as u8;
    }
    LumaPlane {
        width,
        height,
        bytes_per_row: width,
        data,
    }
}

/// Mean absolute difference between two same-size planes — the global change
/// gate. Planes must be packed (`bytes_per_row == width`), which every plane
/// on this path is: raw planes come from the analysis downscale, normalized
/// ones from [`normalize`].
#[must_use]
pub fn mean_absolute_difference(a: &LumaPlane, b: &LumaPlane) -> f64 {
    let count = a.width * a.height;
    if count == 0 || b.width != a.width || b.height != a.height {
        return 0.0;
    }
    if a.data.len() < count || b.data.len() < count {
        return 0.0;
    }
    let total: u64 = (0..count)
        .map(|index| u64::from(a.data[index].abs_diff(b.data[index])))
        .sum();
    total as f64 / count as f64
}

/// Separable box mean: a 1×15 pass then a 15×1 pass — 30 taps per pixel
/// instead of 225.
fn box_blur(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let horizontal = convolve_horizontal(input, width, height);
    convolve_vertical(&horizontal, width, height)
}

fn convolve_horizontal(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let radius = (WINDOW_SIDE / 2) as isize;
    let weight = 1.0 / WINDOW_SIDE as f32;
    let last = width as isize - 1;
    let mut output = vec![0.0_f32; width * height];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0_f32;
            for tap in -radius..=radius {
                // kvImageEdgeExtend: out-of-bounds replicates the edge pixel.
                let sx = (x as isize + tap).clamp(0, last) as usize;
                sum += input[row + sx];
            }
            output[row + x] = sum * weight;
        }
    }
    output
}

fn convolve_vertical(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let radius = (WINDOW_SIDE / 2) as isize;
    let weight = 1.0 / WINDOW_SIDE as f32;
    let last = height as isize - 1;
    let mut output = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            for tap in -radius..=radius {
                let sy = (y as isize + tap).clamp(0, last) as usize;
                sum += input[sy * width + x];
            }
            output[y * width + x] = sum * weight;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_scene_detector::synthetic::{flashed, texture};

    // PhotometricNormalizer — the property the jump-cut and transient laws
    // rest on, pinned directly rather than only through their verdicts.

    #[test]
    fn a_global_brightness_change_barely_moves_the_normalized_plane() {
        let plane = texture(256, 144, 7);
        let lit = flashed(&plane, 60);
        let raw = mean_absolute_difference(&plane, &lit);
        let structural = mean_absolute_difference(&normalize(&plane), &normalize(&lit));
        assert!(raw > 55.0, "the raw planes must differ a lot: {raw}");
        assert!(
            structural < raw / 10.0,
            "normalization must divide the light out: raw {raw}, normalized {structural}"
        );
    }

    #[test]
    fn unrelated_texture_still_differs_after_normalization() {
        let a = normalize(&texture(256, 144, 7));
        let b = normalize(&texture(256, 144, 99));
        assert!(
            mean_absolute_difference(&a, &b) > 20.0,
            "structure change must survive normalization"
        );
    }

    /// A flat wall carries only sensor noise; without the variance floor it
    /// would come back as full-contrast texture and every block would look
    /// matchable.
    #[test]
    fn a_flat_plane_normalizes_to_the_mid_grey_pedestal() {
        let flat = LumaPlane {
            width: 64,
            height: 64,
            bytes_per_row: 64,
            data: vec![97; 64 * 64],
        };
        let normalized = normalize(&flat);
        assert!(normalized.data.iter().all(|&value| value == 128));
    }

    #[test]
    fn the_variance_floor_keeps_sensor_noise_from_becoming_texture() {
        // ±2 luma of noise on a flat wall: σ² ≈ 4, far under the 25 floor,
        // so z stays under ±1 and the byte stays near the pedestal.
        let mut data = vec![128_u8; 64 * 64];
        for (index, value) in data.iter_mut().enumerate() {
            *value = 128 + u8::try_from(index % 3).expect("0..3 fits a byte") * 2 - 2;
        }
        let normalized = normalize(&LumaPlane {
            width: 64,
            height: 64,
            bytes_per_row: 64,
            data,
        });
        let extreme = normalized
            .data
            .iter()
            .map(|&value| i32::from(value).abs_diff(128))
            .max()
            .expect("plane is not empty");
        assert!(
            extreme <= 32,
            "noise amplified to ±{extreme} of the pedestal"
        );
    }

    #[test]
    fn normalization_honours_row_padding() {
        let packed = texture(48, 32, 3);
        let mut padded = LumaPlane {
            width: 48,
            height: 32,
            bytes_per_row: 64,
            data: vec![0; 64 * 32],
        };
        for y in 0..32 {
            padded.data[y * 64..y * 64 + 48].copy_from_slice(&packed.data[y * 48..y * 48 + 48]);
        }
        assert_eq!(normalize(&packed).data, normalize(&padded).data);
    }

    #[test]
    fn mean_absolute_difference_is_zero_for_a_plane_against_itself() {
        let plane = texture(64, 64, 11);
        assert!(mean_absolute_difference(&plane, &plane).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_absolute_difference_rejects_mismatched_geometry() {
        let a = texture(64, 64, 1);
        let b = texture(32, 64, 1);
        assert!(mean_absolute_difference(&a, &b).abs() < f64::EPSILON);
    }
}
