//! The RGB sampling performed by iOS's high-quality vImage preparation.
//!
//! Apple documents a five-lobe Lanczos kernel. Independent float impulse
//! probes establish its phase convention: vertical centres use the lower
//! phase; horizontal centres use the nearest phase. The phase table starts at
//! 64 phases and halves for each power-of-two reduction in scale. Independent
//! public custom-filter captures establish the Float32-to-Q14 DC-preserving
//! conversion below. Each integer dot is rounded and saturated to UInt8 before
//! the next pass, with vertical processing first.
//! The kernel is computed for the actual dimensions; no fixture pixels or
//! dimension-specific coefficient tables participate in production.

#[derive(Clone, Copy)]
enum Axis {
    Vertical,
    Horizontal,
}

#[derive(Debug)]
struct Tap {
    index: usize,
    weight: i32,
}

/// Convert a normalized filter into the integer coefficients used by the UInt8
/// operation. A shared truncation bias is bisected in Float32 until the DC
/// residual is at most one. Equal-coefficient plateaus can straddle that goal;
/// stop at the Float32 fixed point and restore DC at the first largest tap.
/// This law is independent of input pixels and image dimensions.
fn quantize(weights: &[f32]) -> Vec<i32> {
    let values: Vec<_> = weights.iter().map(|value| value * 16384.0).collect();
    let mut largest = 0;
    for index in 1..values.len() {
        if values[index] > values[largest] {
            largest = index;
        }
    }
    let (mut low, mut high, mut bias) = (0.0_f32, 1.0_f32, 0.5_f32);
    let mut result = vec![0; values.len()];
    let mut residual = 0;
    for _ in 0..64 {
        for (value, output) in values.iter().zip(&mut result) {
            *output = (value + bias).trunc() as i32;
        }
        residual = 16384 - result.iter().sum::<i32>();
        if residual.abs() <= 1 {
            break;
        }
        if residual < 0 {
            high = bias;
        } else {
            low = bias;
        }
        let following = (low + high) * 0.5;
        if following == bias {
            break;
        }
        bias = following;
    }
    result[largest] += residual;
    result
}

fn sinc(value: f64) -> f64 {
    if value.abs() < f64::EPSILON {
        1.0
    } else {
        let angle = std::f64::consts::PI * value;
        angle.sin() / angle
    }
}

fn taps(size: usize, target: usize, axis: Axis) -> Vec<Vec<Tap>> {
    let ratio = size as f64 / target as f64;
    let scale = ratio.max(1.0);
    let support = 5.0 * scale;
    let reduction = (scale.log2().floor() as u32).min(6);
    let phases = f64::from(64_u32 >> reduction);
    (0..target)
        .map(|position| {
            let source_center = (position as f64 + 0.5) * ratio - 0.5;
            let phase_bias = match axis {
                Axis::Vertical => 0.0,
                Axis::Horizontal => 0.5,
            };
            let center = (source_center * phases + phase_bias).floor() / phases;
            let first = (center - support).ceil() as isize;
            let last = (center + support).floor() as isize;
            let samples: Vec<_> = (first..=last)
                .map(|index| {
                    let distance = (index as f64 - center) / scale;
                    let weight = if distance.abs() >= 5.0 {
                        0.0
                    } else {
                        sinc(distance) * sinc(distance / 5.0)
                    };
                    (index.clamp(0, size as isize - 1) as usize, weight)
                })
                .collect();
            let total: f64 = samples.iter().map(|sample| sample.1).sum();
            let normalized: Vec<_> = samples
                .iter()
                .map(|sample| (sample.1 / total) as f32)
                .collect();
            samples
                .into_iter()
                .zip(quantize(&normalized))
                .map(|((index, _), weight)| Tap { index, weight })
                .collect()
        })
        .collect()
}

fn sample8(value: i64) -> u8 {
    ((value + 8192) >> 14).clamp(0, 255) as u8
}

/// Precomputed separable HQ taps and bounded UInt8 intermediate storage.
/// Reused for every video picture with the same immutable display dimensions.
#[derive(Debug)]
pub(crate) struct Workspace {
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
    rows: Vec<Vec<Tap>>,
    columns: Vec<Vec<Tap>>,
    vertical: Vec<u8>,
}
impl Workspace {
    pub fn new(width: usize, height: usize, target_width: usize, target_height: usize) -> Self {
        assert!(width > 0 && height > 0 && target_width > 0 && target_height > 0);
        Self {
            width,
            height,
            target_width,
            target_height,
            rows: if height == target_height {
                Vec::new()
            } else {
                taps(height, target_height, Axis::Vertical)
            },
            columns: if width == target_width {
                Vec::new()
            } else {
                taps(width, target_width, Axis::Horizontal)
            },
            vertical: if height == target_height {
                Vec::new()
            } else {
                vec![0; width * target_height * 3]
            },
        }
    }
    pub fn resize_into(&mut self, rgb: &[u8], output: &mut [u8]) {
        assert_eq!(rgb.len(), self.width * self.height * 3);
        assert_eq!(output.len(), self.target_width * self.target_height * 3);
        let vertical = if self.height == self.target_height {
            rgb
        } else {
            for (y, row) in self.rows.iter().enumerate() {
                for x in 0..self.width {
                    let mut channels = [0_i64; 3];
                    for tap in row {
                        let source = (tap.index * self.width + x) * 3;
                        for channel in 0..3 {
                            channels[channel] +=
                                i64::from(rgb[source + channel]) * i64::from(tap.weight);
                        }
                    }
                    let destination = (y * self.width + x) * 3;
                    for (channel, value) in channels.into_iter().enumerate() {
                        self.vertical[destination + channel] = sample8(value);
                    }
                }
            }
            &self.vertical
        };
        if self.width == self.target_width {
            output.copy_from_slice(vertical);
            return;
        }
        for y in 0..self.target_height {
            for (x, column) in self.columns.iter().enumerate() {
                let mut channels = [0_i64; 3];
                for tap in column {
                    let source = (y * self.width + tap.index) * 3;
                    for channel in 0..3 {
                        channels[channel] +=
                            i64::from(vertical[source + channel]) * i64::from(tap.weight);
                    }
                }
                let destination = (y * self.target_width + x) * 3;
                for channel in 0..3 {
                    output[destination + channel] = sample8(channels[channel]);
                }
            }
        }
    }
    pub fn retained_byte_count(&self) -> usize {
        self.vertical.capacity()
            + [&self.rows, &self.columns]
                .into_iter()
                .map(|axis| {
                    axis.capacity() * std::mem::size_of::<Vec<Tap>>()
                        + axis
                            .iter()
                            .map(|row| row.capacity() * std::mem::size_of::<Tap>())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
}

pub(crate) fn resize_rgb(
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    if (width, height) == (target_width, target_height) {
        return rgb;
    }
    let mut workspace = Workspace::new(width, height, target_width, target_height);
    let mut output = vec![0; target_width * target_height * 3];
    workspace.resize_into(&rgb, &mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: use nearest rounding, lose signed truncation, select the last equal
    /// maximum, or loop forever on an equal-tap Float32 plateau. Expected values
    /// come from three independent public custom-filter captures, not an image.
    #[test]
    fn independent_signed_and_flat_filters_preserve_the_original_integer_law() {
        assert_eq!(quantize(&[0.1, 0.8, 0.1]), [1638, 13108, 1638]);
        assert_eq!(
            quantize(&[-0.1, 0.3, 0.6, 0.3, -0.1]),
            [-1638, 4915, 9830, 4915, -1638]
        );
        let mut flat = vec![863; 19];
        flat[0] = 850;
        assert_eq!(quantize(&[1.0 / 19.0; 19]), flat);
    }
}
