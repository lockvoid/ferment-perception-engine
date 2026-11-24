//! Pillow-compatible RGB8 bilinear resize used by the pinned SigLIP processor.
//!
//! Compatibility specification: Pillow 11.3.0 `libImaging/Resample.c`:
//! https://github.com/python-pillow/Pillow/blob/11.3.0/src/libImaging/Resample.c
//! Upstream notice: `../THIRD_PARTY_NOTICES.md` (MIT-CMU).
//! Each separable pass rounds to RGB8 using Q22 normalized coefficients;
//! doing both passes in float, or vertical first, differs by a byte at edges.
//! This is a bounded, safe-Rust implementation of that numerical convention,
//! not a general-purpose image resizer. Input validation belongs to the door.

const BITS: u32 = 22;
const ONE: i64 = 1 << BITS;

struct Tap {
    start: usize,
    weights: Vec<i64>,
}

fn taps(input: usize, output: usize) -> Vec<Tap> {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = (center - support + 0.5).max(0.0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (start..end)
                .map(|p| (1.0 - ((p as f64 + 0.5 - center) / support).abs()).max(0.0))
                .collect();
            let sum: f64 = weights.iter().sum();
            Tap {
                start,
                weights: weights
                    .into_iter()
                    .map(|w| (w / sum * ONE as f64 + 0.5) as i64)
                    .collect(),
            }
        })
        .collect()
}

fn round_channel(sum: i64) -> u8 {
    ((sum + ONE / 2) >> BITS).clamp(0, 255) as u8
}

pub(super) fn rgb_bilinear(rgb: &[u8], width: usize, height: usize, side: usize) -> Vec<u8> {
    let x_taps = taps(width, side);
    let y_taps = taps(height, side);
    // The intermediate is width=224, never a full-size source copy.
    let mut horizontal = vec![0_u8; side * height * 3];
    for y in 0..height {
        for (x, tap) in x_taps.iter().enumerate() {
            for channel in 0..3 {
                let sum = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| i64::from(rgb[(y * width + tap.start + j) * 3 + channel]) * w)
                    .sum();
                horizontal[(y * side + x) * 3 + channel] = round_channel(sum);
            }
        }
    }
    let mut output = vec![0_u8; side * side * 3];
    for (y, tap) in y_taps.iter().enumerate() {
        for x in 0..side {
            for channel in 0..3 {
                let sum = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| {
                        i64::from(horizontal[((tap.start + j) * side + x) * 3 + channel]) * w
                    })
                    .sum();
                output[(y * side + x) * 3 + channel] = round_channel(sum);
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_and_single_pixel_are_exact() {
        let input: Vec<_> = (0..12 * 12 * 3).map(|i| (i % 256) as u8).collect();
        assert_eq!(rgb_bilinear(&input, 12, 12, 12), input);
        assert_eq!(
            rgb_bilinear(&[31, 79, 192], 1, 1, 3),
            [31, 79, 192].repeat(9)
        );
    }
}
