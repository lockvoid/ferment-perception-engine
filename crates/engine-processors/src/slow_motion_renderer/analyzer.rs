use super::{BlockMotionEstimator, LumaPlane, ShutterAngleEstimator, SlowMotionAnalysis};

/// Pure CPU portion of the slow-motion pre-pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlowMotionAnalyzer;

impl SlowMotionAnalyzer {
    pub const SAMPLE_PAIRS: usize = 12;
    pub const ANALYSIS_WIDTH: usize = 256;

    /// Sampling timestamps for the platform decoder adapter. The range is in
    /// seconds and follows the Swift midpoint scheduling exactly.
    #[must_use]
    pub fn sample_times(range_start: f64, range_duration: f64, nominal_fps: f64) -> Vec<f64> {
        let frame_interval = if nominal_fps > 0.0 {
            1.0 / nominal_fps
        } else {
            1.0 / 30.0
        };
        let usable = (range_duration - 3.0 * frame_interval).max(0.0);
        let pair_count = Self::SAMPLE_PAIRS.min(((usable / frame_interval) as usize).max(1));
        (0..pair_count)
            .map(|index| range_start + usable * (index as f64 + 0.5) / pair_count as f64)
            .collect()
    }

    /// Analyze already-decoded consecutive luma pairs. At most twelve pairs
    /// are consumed, matching the production sampling budget.
    #[must_use]
    pub fn analyze_pairs(pairs: &[(LumaPlane, LumaPlane)]) -> SlowMotionAnalysis {
        let mut displacements = Vec::new();
        let mut shutters = Vec::new();
        let mut pairs_sampled = 0;

        for (first, second) in pairs.iter().take(Self::SAMPLE_PAIRS) {
            pairs_sampled += 1;
            let field = BlockMotionEstimator::estimate(first, second);
            if field.vectors.is_empty() {
                continue;
            }

            let mut magnitudes = field
                .vectors
                .iter()
                .map(|vector| f64::from(vector.dx.hypot(vector.dy)) / first.height as f64)
                .collect::<Vec<_>>();
            magnitudes.sort_by(f64::total_cmp);
            displacements.push(magnitudes[magnitudes.len() / 2]);
            shutters.push(ShutterAngleEstimator::estimate(first, &field));
        }

        displacements.sort_by(f64::total_cmp);
        let (median_displacement, p95_displacement) = if displacements.is_empty() {
            (0.0, 0.0)
        } else {
            let p95_index =
                ((displacements.len() as f64 * 0.95) as usize).min(displacements.len() - 1);
            (
                displacements[displacements.len() / 2],
                displacements[p95_index],
            )
        };

        SlowMotionAnalysis {
            source_shutter_angle: ShutterAngleEstimator::aggregate(&shutters),
            median_displacement,
            p95_displacement,
            pairs_sampled,
        }
    }

    /// Bilinearly downscale a luma plane to the analysis width, never
    /// upscaling. Platform adapters can use this after extracting Y planes.
    #[must_use]
    pub fn downscaled_luma(source: &LumaPlane) -> Option<LumaPlane> {
        if !source.is_valid() {
            return None;
        }
        let scale = (Self::ANALYSIS_WIDTH as f64 / source.width as f64).min(1.0);
        let target_width = ((source.width as f64 * scale).round() as usize).max(16);
        let target_height = ((source.height as f64 * scale).round() as usize).max(16);
        if target_width == source.width && target_height == source.height {
            return Some(source.clone());
        }

        let mut data = vec![0_u8; target_width * target_height];
        for target_y in 0..target_height {
            let source_y = if target_height == 1 {
                0.0
            } else {
                target_y as f64 * (source.height - 1) as f64 / (target_height - 1) as f64
            };
            for target_x in 0..target_width {
                let source_x = if target_width == 1 {
                    0.0
                } else {
                    target_x as f64 * (source.width - 1) as f64 / (target_width - 1) as f64
                };
                data[target_y * target_width + target_x] = bilinear(source, source_x, source_y)
                    .round()
                    .clamp(0.0, 255.0)
                    as u8;
            }
        }
        Some(LumaPlane {
            width: target_width,
            height: target_height,
            bytes_per_row: target_width,
            data,
        })
    }
}

fn bilinear(plane: &LumaPlane, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(plane.width - 1);
    let y1 = (y0 + 1).min(plane.height - 1);
    let fraction_x = x - x0 as f64;
    let fraction_y = y - y0 as f64;
    let p00 = f64::from(plane.data[y0 * plane.bytes_per_row + x0]);
    let p10 = f64::from(plane.data[y0 * plane.bytes_per_row + x1]);
    let p01 = f64::from(plane.data[y1 * plane.bytes_per_row + x0]);
    let p11 = f64::from(plane.data[y1 * plane.bytes_per_row + x1]);
    p00 * (1.0 - fraction_x) * (1.0 - fraction_y)
        + p10 * fraction_x * (1.0 - fraction_y)
        + p01 * (1.0 - fraction_x) * fraction_y
        + p11 * fraction_x * fraction_y
}
