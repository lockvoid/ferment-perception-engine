/// An owned 8-bit luma plane, decoupled from decoder buffer lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LumaPlane {
    pub width: usize,
    pub height: usize,
    pub bytes_per_row: usize,
    pub data: Vec<u8>,
}

impl LumaPlane {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.bytes_per_row >= self.width
            && self
                .bytes_per_row
                .checked_mul(self.height)
                .is_some_and(|required| self.data.len() >= required)
    }
}

/// One sparse block-motion vector, centered on `(cx, cy)` in analysis pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    pub cx: usize,
    pub cy: usize,
    pub dx: f32,
    pub dy: f32,
}

/// Sparse motion field between two same-size luma planes.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionField {
    pub vectors: Vec<MotionVector>,
    /// Blocks above the texture floor. `vectors` is the subset that also
    /// passed the best-match residual gate.
    pub textured_blocks: usize,
    /// Grid coordinates (block column, block row) of the textured blocks that
    /// found NO acceptable match — the tear, with its shape. The jump-cut
    /// verdict reads the shape, not just the count: a real edit tears a
    /// connected body, beams and blur scatter into singletons.
    pub torn_blocks: Vec<(usize, usize)>,
    pub width: usize,
    pub height: usize,
}

/// Coarse block matching using SAD on downscaled luma.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockMotionEstimator;

impl BlockMotionEstimator {
    pub const BLOCK_SIZE: usize = 16;
    pub const SEARCH_RADIUS: usize = 20;
    pub const VARIANCE_FLOOR: f64 = 20.0;
    pub const MAX_MEAN_RESIDUAL: f64 = 10.0;

    /// The slow-motion pre-pass form: texture judged on `from`, matches gated
    /// at the default residual.
    #[must_use]
    pub fn estimate(from: &LumaPlane, to: &LumaPlane) -> MotionField {
        Self::estimate_matching(from, to, from, Self::MAX_MEAN_RESIDUAL)
    }

    /// `texture_reference` decides which blocks count as textured; the
    /// jump-cut pass matches on photometrically normalized planes but must
    /// judge texture on the RAW luma — a normalized flat wall is unit-variance
    /// noise and would pass any texture floor. `max_mean_residual` is the
    /// match gate in the matched planes' own units.
    #[must_use]
    pub fn estimate_matching(
        from: &LumaPlane,
        to: &LumaPlane,
        texture_reference: &LumaPlane,
        max_mean_residual: f64,
    ) -> MotionField {
        let mut field = MotionField {
            vectors: Vec::new(),
            textured_blocks: 0,
            torn_blocks: Vec::new(),
            width: from.width,
            height: from.height,
        };
        if !from.is_valid()
            || !to.is_valid()
            || !texture_reference.is_valid()
            || from.width != to.width
            || from.height != to.height
            || texture_reference.width != from.width
            || texture_reference.height != from.height
        {
            return field;
        }

        let block_size = Self::BLOCK_SIZE;
        let margin = Self::SEARCH_RADIUS;
        let Some(right_edge) = from.width.checked_sub(margin) else {
            return field;
        };
        let Some(bottom_edge) = from.height.checked_sub(margin) else {
            return field;
        };
        if margin + block_size > right_edge || margin + block_size > bottom_edge {
            return field;
        }

        let mut by = margin;
        let mut row = 0;
        while by + block_size <= bottom_edge {
            let mut bx = margin;
            let mut column = 0;
            while bx + block_size <= right_edge {
                if block_variance(texture_reference, bx, by, block_size) >= Self::VARIANCE_FLOOR {
                    field.textured_blocks += 1;
                    let best = exhaustive_search(from, to, bx, by, block_size, Self::SEARCH_RADIUS);
                    if f64::from(best.sad) / (block_size * block_size) as f64 <= max_mean_residual {
                        field.vectors.push(MotionVector {
                            cx: bx + block_size / 2,
                            cy: by + block_size / 2,
                            dx: best.dx as f32,
                            dy: best.dy as f32,
                        });
                    } else {
                        field.torn_blocks.push((column, row));
                    }
                }
                bx += block_size;
                column += 1;
            }
            by += block_size;
            row += 1;
        }
        field
    }
}

fn block_variance(plane: &LumaPlane, bx: usize, by: usize, block_size: usize) -> f64 {
    let mut sum = 0_u64;
    let mut sum_squared = 0_u64;
    for y in by..by + block_size {
        let row = y * plane.bytes_per_row;
        for x in bx..bx + block_size {
            let value = u64::from(plane.data[row + x]);
            sum += value;
            sum_squared += value * value;
        }
    }
    let count = (block_size * block_size) as f64;
    let mean = sum as f64 / count;
    sum_squared as f64 / count - mean * mean
}

#[derive(Debug, Clone, Copy)]
struct Match {
    dx: isize,
    dy: isize,
    sad: u32,
}

fn exhaustive_search(
    source: &LumaPlane,
    target: &LumaPlane,
    bx: usize,
    by: usize,
    block_size: usize,
    radius: usize,
) -> Match {
    let radius = radius as isize;
    let mut coarse = (0_isize, 0_isize);
    let mut coarse_sad = u32::MAX;
    for dy in center_out(radius, 2) {
        for dx in center_out(radius, 2) {
            let total = sad(source, target, bx, by, block_size, dx, dy, 2, coarse_sad);
            if total < coarse_sad {
                coarse_sad = total;
                coarse = (dx, dy);
            }
        }
    }

    let mut best = coarse;
    let mut best_sad = u32::MAX;
    for dy in (coarse.1 - 1).max(-radius)..=(coarse.1 + 1).min(radius) {
        for dx in (coarse.0 - 1).max(-radius)..=(coarse.0 + 1).min(radius) {
            let total = sad(source, target, bx, by, block_size, dx, dy, 1, best_sad);
            if total < best_sad {
                best_sad = total;
                best = (dx, dy);
            }
        }
    }
    Match {
        dx: best.0,
        dy: best.1,
        sad: best_sad,
    }
}

#[allow(clippy::too_many_arguments)]
fn sad(
    source: &LumaPlane,
    target: &LumaPlane,
    bx: usize,
    by: usize,
    block_size: usize,
    dx: isize,
    dy: isize,
    sample_stride: usize,
    bound: u32,
) -> u32 {
    let mut total = 0_u32;
    let mut y = 0;
    while y < block_size {
        let source_row = (by + y) * source.bytes_per_row + bx;
        let target_y = (by + y).wrapping_add_signed(dy);
        let target_x = bx.wrapping_add_signed(dx);
        let target_row = target_y * target.bytes_per_row + target_x;
        let mut x = 0;
        while x < block_size {
            total += u32::from(source.data[source_row + x].abs_diff(target.data[target_row + x]));
            x += sample_stride;
        }
        if total >= bound {
            break;
        }
        y += sample_stride;
    }
    total
}

fn center_out(radius: isize, step: isize) -> Vec<isize> {
    let mut order = vec![0];
    let mut magnitude = step;
    while magnitude <= radius {
        order.push(magnitude);
        order.push(-magnitude);
        magnitude += step;
    }
    order
}
