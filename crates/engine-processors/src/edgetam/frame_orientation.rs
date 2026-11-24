//! EdgeTAM works in displayed pixel coordinates, including mirrored tracks.
//! Native decoders supply the coded raster. Apply the container's affine matrix
//! once at that RGB boundary, with exact sample permutation for quarter turns.
use ferment_engine_core::{EngineError, Result};
use ferment_engine_media::{CpuFrame, CpuPixelFormat};

#[derive(Debug, Clone)]
pub(super) struct DisplayOrientation {
    source_width: usize,
    source_height: usize,
    pub width: usize,
    pub height: usize,
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    min_x: f64,
    min_y: f64,
    quarter_turns: Option<u8>,
}

const IDENTITY: [i32; 9] = [65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30];

pub(super) fn quarter_turns_counterclockwise(matrix: Option<[i32; 9]>) -> Option<u8> {
    let m = matrix.unwrap_or(IDENTITY);
    if m[2] != 0 || m[5] != 0 || m[8] != 1 << 30 {
        return None;
    }
    // Fixed 16.16 values have a coarser quantum than Swift's 1e-6 tolerance;
    // exact comparison is its equivalent at this container boundary.
    match (m[0], m[1], m[3], m[4]) {
        (65536, 0, 0, 65536) => Some(0),
        (0, -65536, 65536, 0) => Some(1),
        (-65536, 0, 0, -65536) => Some(2),
        (0, 65536, -65536, 0) => Some(3),
        _ => None,
    }
}

impl DisplayOrientation {
    pub fn new(width: usize, height: usize, matrix: Option<[i32; 9]>) -> Result<Self> {
        let m = matrix.unwrap_or(IDENTITY);
        if width == 0 || height == 0 || m[2] != 0 || m[5] != 0 || m[8] != 1 << 30 {
            return Err(EngineError::Decode(
                "invalid EdgeTAM affine display geometry".into(),
            ));
        }
        let [a, b, c, d] = [m[0], m[1], m[3], m[4]].map(|v| f64::from(v) / 65536.0);
        if (a * d - b * c).abs() < f64::EPSILON {
            return Err(EngineError::Decode(
                "singular EdgeTAM display transform".into(),
            ));
        }
        // Translation cancels when the transformed bounding rectangle is
        // standardized to a raster origin. Keep the full linear transform.
        let corners = [
            (0., 0.),
            (width as f64, 0.),
            (0., height as f64),
            (width as f64, height as f64),
        ];
        let xs = corners.map(|(x, y)| a * x + c * y);
        let ys = corners.map(|(x, y)| b * x + d * y);
        let min_x = xs.into_iter().fold(f64::INFINITY, f64::min);
        let min_y = ys.into_iter().fold(f64::INFINITY, f64::min);
        let w = (xs.into_iter().fold(f64::NEG_INFINITY, f64::max) - min_x).round();
        let h = (ys.into_iter().fold(f64::NEG_INFINITY, f64::max) - min_y).round();
        if w < 1. || h < 1. || w > u32::MAX as f64 || h > u32::MAX as f64 {
            return Err(EngineError::Decode(
                "invalid EdgeTAM displayed dimensions".into(),
            ));
        }
        let (output_width, output_height) = (w as usize, h as usize);
        output_width
            .checked_mul(output_height)
            .and_then(|n| n.checked_mul(3))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| EngineError::Decode("EdgeTAM display byte count overflow".into()))?;
        Ok(Self {
            source_width: width,
            source_height: height,
            width: output_width,
            height: output_height,
            a,
            b,
            c,
            d,
            min_x,
            min_y,
            quarter_turns: quarter_turns_counterclockwise(matrix),
        })
    }

    pub fn apply(&self, source: &CpuFrame) -> Result<CpuFrame> {
        if source.width as usize != self.source_width
            || source.height as usize != self.source_height
        {
            return Err(EngineError::Decode(
                "EdgeTAM decoded dimensions changed within a track".into(),
            ));
        }
        let source = source.to_rgb24();
        if self.quarter_turns == Some(0) {
            return Ok(source);
        }
        let mut pixels = vec![0; self.width * self.height * 3];
        if let Some(turns) = self.quarter_turns {
            for y in 0..self.source_height {
                for x in 0..self.source_width {
                    let (dx, dy) = match turns {
                        1 => (y, self.source_width - 1 - x),
                        2 => (self.source_width - 1 - x, self.source_height - 1 - y),
                        3 => (self.source_height - 1 - y, x),
                        _ => unreachable!(),
                    };
                    let s = y * source.stride + x * 3;
                    let d = (dy * self.width + dx) * 3;
                    pixels[d..d + 3].copy_from_slice(&source.data[s..s + 3]);
                }
            }
        } else {
            // Mirrors and scale/shear matrices use an orientation-aware affine
            // sampler. Only this one decoded picture and its output are live;
            // there is no duration-sized image-generator queue on native.
            let det = self.a * self.d - self.b * self.c;
            for y in 0..self.height {
                for x in 0..self.width {
                    let dx = x as f64 + 0.5 + self.min_x;
                    let dy = y as f64 + 0.5 + self.min_y;
                    let sx = (self.d * dx - self.c * dy) / det - 0.5;
                    let sy = (-self.b * dx + self.a * dy) / det - 0.5;
                    if sx < -0.5
                        || sy < -0.5
                        || sx > self.source_width as f64 - 0.5
                        || sy > self.source_height as f64 - 0.5
                    {
                        continue;
                    }
                    let sx = sx.clamp(0., (self.source_width - 1) as f64);
                    let sy = sy.clamp(0., (self.source_height - 1) as f64);
                    let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
                    let (x1, y1) = (
                        (x0 + 1).min(self.source_width - 1),
                        (y0 + 1).min(self.source_height - 1),
                    );
                    let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
                    for channel in 0..3 {
                        let p =
                            |xx, yy| f64::from(source.data[yy * source.stride + xx * 3 + channel]);
                        let upper = p(x0, y0) * (1. - fx) + p(x1, y0) * fx;
                        let lower = p(x0, y1) * (1. - fx) + p(x1, y1) * fx;
                        pixels[(y * self.width + x) * 3 + channel] =
                            (upper * (1. - fy) + lower * fy).round().clamp(0., 255.) as u8;
                    }
                }
            }
        }
        CpuFrame::new(
            self.width as u32,
            self.height as u32,
            self.width * 3,
            CpuPixelFormat::Rgb24,
            pixels,
        )
        .map_err(|e| EngineError::VideoIo(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn matrix(a: i32, b: i32, c: i32, d: i32, tx: i32, ty: i32) -> [i32; 9] {
        [a, b, 0, c, d, 0, tx * 65536, ty * 65536, 1 << 30]
    }
    /// Kill: treat a mirrored/scaled track as a unit rotation, or omit a source classifier/budget branch.
    #[test]
    fn current_transform_classification_and_native_reader_budgets() {
        for (m, want) in [
            (None, Some(0)),
            (Some(matrix(0, -65536, 65536, 0, 48, 0)), Some(1)),
            (Some(matrix(-65536, 0, 0, -65536, 48, 32)), Some(2)),
            (Some(matrix(0, 65536, -65536, 0, 0, 32)), Some(3)),
            (Some(matrix(-65536, 0, 0, 65536, 0, 0)), None),
            (Some(matrix(32768, 0, 0, 32768, 0, 0)), None),
        ] {
            assert_eq!(quarter_turns_counterclockwise(m), want);
        }
        use crate::edgetam::frame_source::EdgeTamFrameSource;
        for ((w, h), expected) in [((16, 16), 1024), ((1920, 1080), 10), ((3840, 2160), 2)] {
            assert_eq!(
                EdgeTamFrameSource::reverse_reader_batch_size(w, h),
                expected
            );
        }
        // AVAssetImageGenerator's 32/2/1 batch knobs have no native analogue:
        // the production decoder is pulled once per frame. The real-file test
        // below grades that path's retained RGB cache instead of a fake knob.
    }

    /// Kill: transpose without reversing a quarter-turn axis, drop the preferred mirror, or ignore affine scale.
    #[test]
    fn displayed_pixels_preserve_exact_quarter_turns_mirrors_and_scale() {
        let pixels: Vec<u8> = (0..8)
            .flat_map(|v| [v * 20, v * 20 + 1, v * 20 + 2])
            .collect();
        let source = CpuFrame::new(4, 2, 12, CpuPixelFormat::Rgb24, pixels.clone()).unwrap();
        for (m, expected) in [
            (
                matrix(0, -65536, 65536, 0, 48, 0),
                vec![3, 7, 2, 6, 1, 5, 0, 4],
            ),
            (
                matrix(-65536, 0, 0, -65536, 48, 32),
                vec![7, 6, 5, 4, 3, 2, 1, 0],
            ),
            (
                matrix(0, 65536, -65536, 0, 0, 32),
                vec![4, 0, 5, 1, 6, 2, 7, 3],
            ),
            (
                matrix(-65536, 0, 0, 65536, 0, 0),
                vec![3, 2, 1, 0, 7, 6, 5, 4],
            ),
        ] {
            let transform = DisplayOrientation::new(4, 2, Some(m)).unwrap();
            let result = transform.apply(&source).unwrap();
            let wanted: Vec<u8> = expected
                .into_iter()
                .flat_map(|v| [v * 20, v * 20 + 1, v * 20 + 2])
                .collect();
            assert_eq!(&*result.data, &wanted);
        }
        let scaled = DisplayOrientation::new(4, 2, Some(matrix(32768, 0, 0, 32768, 0, 0)))
            .unwrap()
            .apply(&source)
            .unwrap();
        assert_eq!((scaled.width, scaled.height), (2, 1));
        assert_eq!(&*scaled.data, &[50, 51, 52, 90, 91, 92]);
    }
}
