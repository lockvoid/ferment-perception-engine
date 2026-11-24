//! Deterministic synthetic luma for the scene-detection suites — the Rust
//! twin of iOS `Tests/FermentEngineTests/TestSupport/SyntheticLuma.swift` plus
//! the per-suite helpers the Swift tests declare privately (`flashed`,
//! `jittered`, `teleported`). Shared here because the jump-cut, transient and
//! verdict suites all build the same sequences.

use crate::slow_motion_renderer::LumaPlane;

/// Seeded value-noise texture smoothed by two 3×3 box passes — every block
/// carries matchable structure, no single-pixel noise gradients.
pub fn texture(width: usize, height: usize, seed: u64) -> LumaPlane {
    let mut state = seed;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8
    };
    let mut data = vec![0_u8; width * height];
    for value in &mut data {
        *value = next();
    }
    for _ in 0..2 {
        let mut out = data.clone();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0_u32;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        sum += u32::from(
                            data[y.wrapping_add_signed(dy) * width + x.wrapping_add_signed(dx)],
                        );
                    }
                }
                out[y * width + x] = (sum / 9) as u8;
            }
        }
        data = out;
    }
    LumaPlane {
        width,
        height,
        bytes_per_row: width,
        data,
    }
}

/// The plane translated by integer (dx, dy); vacated edges replicate the
/// border.
pub fn shifted(plane: &LumaPlane, dx: isize, dy: isize) -> LumaPlane {
    let mut data = vec![0_u8; plane.width * plane.height];
    for y in 0..plane.height {
        let source_y = (y as isize - dy).clamp(0, plane.height as isize - 1) as usize;
        for x in 0..plane.width {
            let source_x = (x as isize - dx).clamp(0, plane.width as isize - 1) as usize;
            data[y * plane.width + x] = plane.data[source_y * plane.bytes_per_row + source_x];
        }
    }
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data,
    }
}

/// The plane under a global illumination change (a strobe hit).
pub fn flashed(plane: &LumaPlane, delta: i32) -> LumaPlane {
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data: plane
            .data
            .iter()
            .map(|&value| (i32::from(value) + delta).clamp(0, 255) as u8)
            .collect(),
    }
}

/// Sensor noise (−1…2 luma, seeded per frame) so the baseline is not zero —
/// real footage never is.
pub fn jittered(plane: &LumaPlane, frame: usize) -> LumaPlane {
    let mut state = (frame as u64)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(12_345);
    let mut data = plane.data.clone();
    for value in &mut data {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let noise = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 62) as i32 - 1;
        *value = (i32::from(*value) + noise).clamp(0, 255) as u8;
    }
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data,
    }
}

/// A 96×96 centre region swapped for unrelated texture — the subject
/// teleported, the set stayed. Geometry is relative so 256- and 640-wide
/// frames both get a subject that spans several blocks.
pub fn teleported(plane: &LumaPlane, seed: u64) -> LumaPlane {
    let replacement = texture(plane.width, plane.height, seed);
    let mut data = plane.data.clone();
    let (x0, x1) = (plane.width * 5 / 16, plane.width * 11 / 16);
    let (y0, y1) = (plane.height / 6, plane.height * 5 / 6);
    for y in y0..y1 {
        for x in x0..x1 {
            data[y * plane.width + x] = replacement.data[y * plane.width + x];
        }
    }
    LumaPlane {
        width: plane.width,
        height: plane.height,
        bytes_per_row: plane.width,
        data,
    }
}

/// `length` jittered copies of one texture — a still shot with sensor noise.
pub fn static_sequence(width: usize, height: usize, seed: u64, length: usize) -> Vec<LumaPlane> {
    let base = texture(width, height, seed);
    (0..length).map(|frame| jittered(&base, frame)).collect()
}
