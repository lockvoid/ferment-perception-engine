use std::cmp::Ordering;

use super::{LumaPlane, MotionField};

/// Estimates source shutter angle as blur-trail / displacement × 360°.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShutterAngleEstimator;

impl ShutterAngleEstimator {
    pub const MIN_DISPLACEMENT: f32 = 4.0;
    pub const CANDIDATES: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0];
    pub const MIN_BLOCKS: usize = 3;
    pub const MAX_BLOCKS: usize = 12;
    pub const MIN_PAIRS: usize = 3;
    pub const TRAIL_GAIN: f64 = 1.4;

    #[must_use]
    pub fn estimate(plane: &LumaPlane, field: &MotionField) -> Option<f64> {
        Self::estimate_with_min_displacement(plane, field, Self::MIN_DISPLACEMENT)
    }

    #[must_use]
    pub fn estimate_with_min_displacement(
        plane: &LumaPlane,
        field: &MotionField,
        min_displacement: f32,
    ) -> Option<f64> {
        if !plane.is_valid() {
            return None;
        }
        let radius = 6_usize;
        let reach = (Self::CANDIDATES.last().copied().unwrap_or(0.0).ceil() as usize) / 2 + 2;
        let margin = radius + reach + 2;

        let mut qualifying = field
            .vectors
            .iter()
            .copied()
            .filter_map(|vector| {
                let magnitude = f64::from(vector.dx.hypot(vector.dy));
                (magnitude >= f64::from(min_displacement)
                    && vector.cx >= margin
                    && vector.cx < plane.width.saturating_sub(margin)
                    && vector.cy >= margin
                    && vector.cy < plane.height.saturating_sub(margin))
                .then_some((vector, magnitude))
            })
            .collect::<Vec<_>>();
        qualifying.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        qualifying.truncate(Self::MAX_BLOCKS);

        let mut block_angles = Vec::with_capacity(qualifying.len());
        for (vector, magnitude) in qualifying {
            let ux = f64::from(vector.dx) / magnitude;
            let uy = f64::from(vector.dy) / magnitude;
            let plain = energies(plane, vector.cx, vector.cy, radius, ux, uy, 0.0);
            if plain.1 <= 1e-6 {
                continue;
            }
            let observed = plain.0 / plain.1;
            let mut curve = vec![(0.0, 1.0)];
            for length in Self::CANDIDATES {
                if length <= magnitude * 1.3 {
                    let probe = energies(plane, vector.cx, vector.cy, radius, ux, uy, length);
                    curve.push((length, probe.1 / plain.1));
                }
            }
            if curve.len() <= 1 {
                continue;
            }
            let trail = look_up_trail(observed, &curve) * Self::TRAIL_GAIN;
            block_angles.push((360.0 * trail / magnitude).min(360.0));
        }

        (block_angles.len() >= Self::MIN_BLOCKS).then(|| median(&mut block_angles))
    }

    #[must_use]
    pub fn aggregate(per_pair: &[Option<f64>]) -> Option<f64> {
        let mut valid = per_pair.iter().flatten().copied().collect::<Vec<_>>();
        (valid.len() >= Self::MIN_PAIRS).then(|| median(&mut valid))
    }
}

#[allow(clippy::too_many_arguments)]
fn energies(
    plane: &LumaPlane,
    cx: usize,
    cy: usize,
    radius: usize,
    ux: f64,
    uy: f64,
    test_blur: f64,
) -> (f64, f64) {
    let perpendicular_x = -uy;
    let perpendicular_y = ux;
    let taps = if test_blur > 0.0 {
        ((test_blur.ceil() as usize).max(3) | 1).min(9)
    } else {
        0
    };
    let sample = |x: f64, y: f64| {
        if taps == 0 {
            return bilinear(plane, x, y);
        }
        let mut sum = 0.0;
        for tap in 0..taps {
            let offset = (tap as f64 / (taps - 1) as f64 - 0.5) * test_blur;
            sum += bilinear(
                plane,
                x + perpendicular_x * offset,
                y + perpendicular_y * offset,
            );
        }
        sum / taps as f64
    };

    let mut parallel = 0.0;
    let mut perpendicular = 0.0;
    for wy in (-(radius as isize)..=radius as isize).step_by(2) {
        for wx in (-(radius as isize)..=radius as isize).step_by(2) {
            let x = cx as f64 + wx as f64;
            let y = cy as f64 + wy as f64;
            let gradient_parallel = sample(x + ux, y + uy) - sample(x - ux, y - uy);
            let gradient_perpendicular = sample(x + perpendicular_x, y + perpendicular_y)
                - sample(x - perpendicular_x, y - perpendicular_y);
            parallel += gradient_parallel * gradient_parallel;
            perpendicular += gradient_perpendicular * gradient_perpendicular;
        }
    }
    (parallel, perpendicular)
}

fn bilinear(plane: &LumaPlane, x: f64, y: f64) -> f64 {
    let clamped_x = x.clamp(0.0, (plane.width - 1) as f64);
    let clamped_y = y.clamp(0.0, (plane.height - 1) as f64);
    let x0 = clamped_x as usize;
    let y0 = clamped_y as usize;
    let x1 = (x0 + 1).min(plane.width - 1);
    let y1 = (y0 + 1).min(plane.height - 1);
    let fraction_x = clamped_x - x0 as f64;
    let fraction_y = clamped_y - y0 as f64;
    let p00 = f64::from(plane.data[y0 * plane.bytes_per_row + x0]);
    let p10 = f64::from(plane.data[y0 * plane.bytes_per_row + x1]);
    let p01 = f64::from(plane.data[y1 * plane.bytes_per_row + x0]);
    let p11 = f64::from(plane.data[y1 * plane.bytes_per_row + x1]);
    p00 * (1.0 - fraction_x) * (1.0 - fraction_y)
        + p10 * fraction_x * (1.0 - fraction_y)
        + p01 * (1.0 - fraction_x) * fraction_y
        + p11 * fraction_x * fraction_y
}

fn look_up_trail(observed: f64, curve: &[(f64, f64)]) -> f64 {
    if observed >= curve[0].1 {
        return curve[0].0;
    }
    for pair in curve.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if observed >= current.1 {
            let span = previous.1 - current.1;
            if span <= 1e-9 {
                return current.0;
            }
            let interpolation = (previous.1 - observed) / span;
            return previous.0 + interpolation * (current.0 - previous.0);
        }
    }
    curve.last().map_or(0.0, |entry| entry.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    values[values.len() / 2]
}
