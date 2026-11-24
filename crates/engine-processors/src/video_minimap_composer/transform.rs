//! Preferred affine display transforms for selected minimap pictures only.
use super::processor::KeyframeBuffer;
use crate::video_source::RgbFrame;
use ferment_engine_core::{EngineError, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct DisplayTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}
impl DisplayTransform {
    pub(super) fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }
    pub(super) fn from_matrix(matrix: Option<[i32; 9]>) -> Result<Self> {
        let Some(m) = matrix else {
            return Ok(Self::identity());
        };
        if m[2] != 0 || m[5] != 0 || m[8] == 0 {
            return Err(EngineError::invalid_config(
                "minimap requires an affine display matrix",
            ));
        }
        let w = f64::from(m[8]) / 1_073_741_824.0;
        let value = Self {
            a: f64::from(m[0]) / 65536.0 / w,
            b: f64::from(m[1]) / 65536.0 / w,
            c: f64::from(m[3]) / 65536.0 / w,
            d: f64::from(m[4]) / 65536.0 / w,
            tx: f64::from(m[6]) / 65536.0 / w,
            ty: f64::from(m[7]) / 65536.0 / w,
        };
        let determinant = value.a * value.d - value.b * value.c;
        if !determinant.is_finite() || determinant == 0.0 {
            return Err(EngineError::invalid_config(
                "minimap display matrix is not invertible",
            ));
        }
        Ok(value)
    }
    /// CGSize.applying ignores translation and applies the affine linear part.
    pub(super) fn display_size(self, width: u32, height: u32) -> (f64, f64) {
        (
            (self.a * f64::from(width) + self.c * f64::from(height)).abs(),
            (self.b * f64::from(width) + self.d * f64::from(height)).abs(),
        )
    }
    fn point(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )
    }
    fn orthogonal(self) -> bool {
        let values = [self.a, self.b, self.c, self.d];
        values.iter().all(|v| [-1.0, 0.0, 1.0].contains(v))
            && self.a.abs() + self.c.abs() == 1.0
            && self.b.abs() + self.d.abs() == 1.0
            && (self.a * self.d - self.b * self.c).abs() == 1.0
    }
    pub(super) fn thumbnail(self, frame: &RgbFrame, maximum: (u32, u32)) -> Result<KeyframeBuffer> {
        if frame.width == 0
            || frame.height == 0
            || maximum.0 == 0
            || maximum.1 == 0
            || frame.data.len() != frame.width as usize * frame.height as usize * 3
        {
            return Err(EngineError::Decode("invalid minimap RGB frame".into()));
        }
        let points = [
            self.point(0.0, 0.0),
            self.point(f64::from(frame.width), 0.0),
            self.point(0.0, f64::from(frame.height)),
            self.point(f64::from(frame.width), f64::from(frame.height)),
        ];
        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let width = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max) - min_x;
        let height = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) - min_y;
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(EngineError::invalid_config(
                "unbounded minimap display geometry",
            ));
        }
        let scale = (f64::from(maximum.0) / width)
            .min(f64::from(maximum.1) / height)
            .min(1.0);
        let out_w = (width * scale).round().clamp(1.0, f64::from(maximum.0)) as u32;
        let out_h = (height * scale).round().clamp(1.0, f64::from(maximum.1)) as u32;
        let size = (out_w as usize)
            .checked_mul(out_h as usize)
            .and_then(|n| n.checked_mul(3))
            .filter(|n| *n <= 64 * 1024 * 1024)
            .ok_or_else(|| EngineError::invalid_config("unbounded minimap thumbnail extent"))?;
        // Pure quarter turns/reflections are exact byte permutations. Resize
        // only after that permutation when a selected cell needs fewer pixels.
        if self.orthogonal() {
            let w = width.round() as u32;
            let h = height.round() as u32;
            let identity = self.a == 1.0 && self.d == 1.0;
            let mut rotated = Vec::new();
            let bytes = if identity {
                frame.data.as_slice()
            } else {
                rotated.resize(frame.data.len(), 0);
                for y in 0..frame.height {
                    for x in 0..frame.width {
                        let (dx, dy) = self.point(f64::from(x) + 0.5, f64::from(y) + 0.5);
                        let ix = (dx - min_x).floor() as usize;
                        let iy = (dy - min_y).floor() as usize;
                        let from = (y as usize * frame.width as usize + x as usize) * 3;
                        let to = (iy * w as usize + ix) * 3;
                        rotated[to..to + 3].copy_from_slice(&frame.data[from..from + 3]);
                    }
                }
                rotated.as_slice()
            };
            let data = if (w, h) == (out_w, out_h) {
                bytes.to_vec()
            } else {
                let image =
                    image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(w, h, bytes).unwrap();
                image::imageops::resize(&image, out_w, out_h, image::imageops::FilterType::Triangle)
                    .into_raw()
            };
            return Ok(KeyframeBuffer {
                width: out_w as usize,
                height: out_h as usize,
                data,
            });
        }
        // Arbitrary finite affine scale/shear/rotation: inverse-sample directly
        // into the bounded target. Never allocate the full transformed canvas.
        let mut data = vec![0; size];
        let det = self.a * self.d - self.b * self.c;
        for y in 0..out_h {
            for x in 0..out_w {
                let dx = min_x + (f64::from(x) + 0.5) / f64::from(out_w) * width - self.tx;
                let dy = min_y + (f64::from(y) + 0.5) / f64::from(out_h) * height - self.ty;
                let sx = (self.d * dx - self.c * dy) / det - 0.5;
                let sy = (-self.b * dx + self.a * dy) / det - 0.5;
                if sx < -0.5
                    || sy < -0.5
                    || sx > f64::from(frame.width) - 0.5
                    || sy > f64::from(frame.height) - 0.5
                {
                    continue;
                }
                let sx = sx.clamp(0.0, f64::from(frame.width - 1));
                let sy = sy.clamp(0.0, f64::from(frame.height - 1));
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(frame.width as usize - 1);
                let y1 = (y0 + 1).min(frame.height as usize - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;
                for channel in 0..3 {
                    let get =
                        |x, y| f64::from(frame.data[(y * frame.width as usize + x) * 3 + channel]);
                    let top = get(x0, y0) * (1.0 - fx) + get(x1, y0) * fx;
                    let bottom = get(x0, y1) * (1.0 - fx) + get(x1, y1) * fx;
                    data[(y as usize * out_w as usize + x as usize) * 3 + channel] =
                        (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(KeyframeBuffer {
            width: out_w as usize,
            height: out_h as usize,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Kill: ignore affine scale/shear or allocate its full canvas; this
    /// independently specified map is x'=2x+y+7,y'=y-3 on a4×4 source.
    #[test]
    fn scaled_sheared_picture_uses_inverse_mapping_and_bounded_canvas() {
        let transform = DisplayTransform::from_matrix(Some([
            131072, 0, 0, 65536, 65536, 0, 458752, -196608, 1073741824,
        ]))
        .unwrap();
        let source = RgbFrame {
            width: 4,
            height: 4,
            presentation_time: 3.0,
            data: [80, 160, 240].repeat(16),
        };
        assert_eq!(transform.display_size(4, 4), (12.0, 4.0));
        let picture = transform.thumbnail(&source, (120, 40)).unwrap();
        assert_eq!((picture.width, picture.height), (12, 4));
        let pixel = |x: usize, y: usize| &picture.data[(y * 12 + x) * 3..(y * 12 + x + 1) * 3];
        assert_eq!(pixel(3, 1), [80, 160, 240]);
        assert_eq!(pixel(0, 3), [0, 0, 0]);
        assert_eq!(pixel(11, 0), [0, 0, 0]);
        let small = transform.thumbnail(&source, (6, 2)).unwrap();
        assert_eq!((small.width, small.height, small.data.len()), (6, 2, 36));
    }
    /// Kill: silently accept a collapsed affine transform or misinterpret a
    /// perspective matrix as Swift's six-coefficient affine contract.
    #[test]
    fn malformed_or_noninvertible_affine_is_rejected() {
        assert!(DisplayTransform::from_matrix(Some([0, 0, 0, 0, 0, 0, 0, 0, 1073741824])).is_err());
        assert!(
            DisplayTransform::from_matrix(Some([65536, 0, 1, 0, 65536, 0, 0, 0, 1073741824]))
                .is_err()
        );
    }
}
