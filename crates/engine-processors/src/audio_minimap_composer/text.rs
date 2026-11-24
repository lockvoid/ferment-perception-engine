//! Minimap labels rendered with bundled Noto Sans.
//! The same outlines and metrics are used on every platform.

use std::sync::OnceLock;

use ab_glyph::{Font, FontRef, ScaleFont, point};
use tiny_skia::{Pixmap, PremultipliedColorU8};

#[derive(Clone, Copy)]
pub(crate) enum Weight {
    Regular,
    Semibold,
}

fn font(weight: Weight) -> &'static FontRef<'static> {
    static REGULAR: OnceLock<FontRef<'static>> = OnceLock::new();
    static SEMIBOLD: OnceLock<FontRef<'static>> = OnceLock::new();
    match weight {
        Weight::Regular => REGULAR.get_or_init(|| {
            FontRef::try_from_slice(include_bytes!("../resources/fonts/NotoSans-Regular.ttf"))
                .expect("the bundled Noto Sans regular face must parse")
        }),
        Weight::Semibold => SEMIBOLD.get_or_init(|| {
            FontRef::try_from_slice(include_bytes!("../resources/fonts/NotoSans-SemiBold.ttf"))
                .expect("the bundled Noto Sans semibold face must parse")
        }),
    }
}

fn scale(font: &FontRef<'_>, size: f32) -> f32 {
    // CoreText fontSize measures the em, whereas ab_glyph measures ascent
    // minus descent. Convert once so drawing and measured widths agree.
    size * font.height_unscaled() / font.units_per_em().expect("OpenType em units")
}

pub(crate) fn width(text: &str, size: f32, weight: Weight) -> f32 {
    let font = font(weight);
    let scaled = font.as_scaled(scale(font, size));
    let digit_width = ('0'..='9')
        .map(|c| scaled.h_advance(scaled.glyph_id(c)))
        .fold(0.0, f32::max);
    text.chars()
        .map(|c| {
            if c.is_ascii_digit() {
                digit_width
            } else {
                scaled.h_advance(scaled.glyph_id(c))
            }
        })
        .sum()
}

pub(crate) fn draw(
    pixmap: &mut Pixmap,
    text: &str,
    origin: (f32, f32),
    size: f32,
    weight: Weight,
    color: [u8; 4],
) {
    draw_oriented(pixmap, text, origin, size, weight, color, false);
}

pub(crate) fn draw_rotated(
    pixmap: &mut Pixmap,
    text: &str,
    origin: (f32, f32),
    size: f32,
    color: [u8; 4],
) {
    draw_oriented(pixmap, text, origin, size, Weight::Regular, color, true);
}

fn draw_oriented(
    pixmap: &mut Pixmap,
    text: &str,
    origin: (f32, f32),
    size: f32,
    weight: Weight,
    color: [u8; 4],
    rotated: bool,
) {
    let font = font(weight);
    let pixel_scale = scale(font, size);
    let scaled = font.as_scaled(pixel_scale);
    let digit_width = ('0'..='9')
        .map(|c| scaled.h_advance(scaled.glyph_id(c)))
        .fold(0.0, f32::max);
    let (canvas_width, canvas_height) = (pixmap.width() as i32, pixmap.height() as i32);
    let pixels = pixmap.pixels_mut();
    let mut cursor = 0.0;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        let natural_advance = scaled.h_advance(id);
        let advance = if c.is_ascii_digit() {
            digit_width
        } else {
            natural_advance
        };
        let glyph = id.with_scale_and_position(
            pixel_scale,
            point(cursor + (advance - natural_advance) / 2.0, size * 0.85),
        );
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            outline.draw(|x, y, coverage| {
                let local_x = bounds.min.x + x as f32;
                let local_y = bounds.min.y + y as f32;
                let (x, y) = if rotated {
                    (origin.0 + local_y, origin.1 - local_x)
                } else {
                    (origin.0 + local_x, origin.1 + local_y)
                };
                let (x, y) = (x.round() as i32, y.round() as i32);
                if x < 0 || y < 0 || x >= canvas_width || y >= canvas_height {
                    return;
                }
                let pixel = &mut pixels[(y * canvas_width + x) as usize];
                let alpha = coverage * f32::from(color[3]) / 255.0;
                let blend = |source: u8, destination: u8| {
                    (f32::from(source) * alpha + f32::from(destination) * (1.0 - alpha)).round()
                        as u8
                };
                // Minimap backgrounds are opaque, including under labels.
                *pixel = PremultipliedColorU8::from_rgba(
                    blend(color[0], pixel.red()),
                    blend(color[1], pixel.green()),
                    blend(color[2], pixel.blue()),
                    255,
                )
                .expect("opaque glyph color");
            });
        }
        cursor += advance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The same digit cell is used by layout and drawing. Kill: use natural
    /// advances for digits, losing the CoreText tabular-numbers contract.
    #[test]
    fn numeric_labels_use_tabular_digits_and_real_glyphs() {
        for weight in [Weight::Regular, Weight::Semibold] {
            assert_eq!(width("11:11", 12.0, weight), width("88:88", 12.0, weight));
            let mut pixmap = Pixmap::new(96, 32).unwrap();
            pixmap.fill(tiny_skia::Color::WHITE);
            draw(
                &mut pixmap,
                "0:30 · Hz",
                (4.0, 5.0),
                12.0,
                weight,
                [38, 38, 38, 255],
            );
            let dark = pixmap
                .pixels()
                .iter()
                .filter(|pixel| pixel.red() < 150)
                .count();
            assert!(dark > 40, "actual label pixels: {dark}");
            assert_eq!(pixmap.pixels()[0].red(), 255);
        }
    }
}
