//! Current video minimap panels, readable labels, fitted pictures and timeline.
//! Raster/text engines are portable tiny-skia + the pinned app fonts; metadata
//! and the source display/clock geometry do not depend on host font discovery.

use super::allocate::SceneAllocation;
use super::layout::Layout;
use super::processor::{KeyframeBuffer, PoolEntry};
use crate::audio_minimap_composer::text::{self, Weight};
use tiny_skia::{
    Color, FillRule, Paint, PathBuilder, Pixmap, PremultipliedColorU8, Stroke, Transform,
};

pub struct RenderInput<'a> {
    pub allocations: &'a [SceneAllocation],
    pub keyframes: &'a [(usize, Vec<KeyframeBuffer>)],
    pub barcode_pool: &'a [PoolEntry],
    pub layout: Layout,
    pub duration: f64,
    pub boundary_times: &'a [f64],
    pub total_scene_count: usize,
    pub video_size: (f64, f64),
    pub frame_rate: f64,
}

pub struct RenderOutput {
    pub pixmap: Pixmap,
}

pub fn render(input: &RenderInput<'_>) -> RenderOutput {
    let l = input.layout;
    let mut pixmap =
        Pixmap::new(l.total_width as u32, l.total_height as u32).expect("minimap extent");
    pixmap.fill(Color::WHITE);
    stroke_line(
        &mut pixmap,
        0.0,
        l.header_height as f32 - 0.5,
        l.total_width as f32,
        l.header_height as f32 - 0.5,
        1.0,
        gray(166),
    );
    text::draw(
        &mut pixmap,
        &header_label(input),
        (14.0, l.header_height as f32 / 2.0 - 7.0),
        14.0,
        Weight::Semibold,
        [38, 38, 38, 255],
    );

    for (row, alloc) in input.allocations.iter().enumerate() {
        let row_y = l.header_height + row as i32 * l.row_height();
        draw_scene_label(&mut pixmap, &l, row_y, alloc);
        let frames = input
            .keyframes
            .iter()
            .find(|(index, _)| *index == alloc.scene_index)
            .map_or(&[][..], |(_, frames)| frames.as_slice());
        for (cell, &time) in alloc.cell_times.iter().enumerate() {
            let x =
                l.outer_padding + l.label_gutter_width + cell as i32 * (l.cell_width + l.cell_gap);
            let y = row_y + l.row_vertical_padding;
            if let Some(picture) = frames.get(cell) {
                blit_rgb(&mut pixmap, picture, x, y, l.cell_width, l.cell_height);
            } else {
                fill_rect(
                    &mut pixmap,
                    x as f32,
                    y as f32,
                    l.cell_width as f32,
                    l.cell_height as f32,
                    gray(26),
                );
                text::draw(
                    &mut pixmap,
                    "no frame",
                    (x as f32 + 6.0, y as f32 + l.cell_height as f32 / 2.0 - 6.0),
                    11.0,
                    Weight::Regular,
                    [179, 179, 179, 255],
                );
            }
            stroke_rect(
                &mut pixmap,
                x as f32,
                y as f32,
                l.cell_width as f32,
                l.cell_height as f32,
                0.5,
                gray(153),
            );
            text::draw(
                &mut pixmap,
                &format_mmss(time),
                (x as f32 + 4.0, (y + l.cell_height + 2) as f32),
                11.0,
                Weight::Semibold,
                [51, 51, 51, 255],
            );
        }
    }
    let rows_bottom = l.header_height + input.allocations.len() as i32 * l.row_height();
    let divider = (l.outer_padding + l.label_gutter_width) as f32 - 0.5;
    stroke_line(
        &mut pixmap,
        divider,
        l.header_height as f32,
        divider,
        rows_bottom as f32,
        1.0,
        gray(204),
    );
    stroke_line(
        &mut pixmap,
        0.0,
        rows_bottom as f32 + 0.5,
        l.total_width as f32,
        rows_bottom as f32 + 0.5,
        1.0,
        gray(166),
    );
    draw_barcode(
        &mut pixmap,
        &l,
        input.barcode_pool,
        input.duration,
        input.boundary_times,
    );
    draw_time_axis(&mut pixmap, &l, input.duration);
    RenderOutput { pixmap }
}

fn header_label(input: &RenderInput<'_>) -> String {
    let shown = input.allocations.len();
    let scenes = if shown == input.total_scene_count {
        format!("{} scenes", input.total_scene_count)
    } else {
        format!("{} scenes ({shown} shown)", input.total_scene_count)
    };
    format!(
        "{}  ·  {scenes}  ·  {:.0} fps  ·  {}",
        format_mmss(input.duration),
        input.frame_rate,
        format_aspect(input.video_size)
    )
}

fn format_aspect((width, height): (f64, f64)) -> String {
    if width <= 0.0 || height <= 0.0 {
        return "?".into();
    }
    let ratio = width / height;
    for (r, label) in [
        (16.0 / 9.0, "16:9"),
        (9.0 / 16.0, "9:16"),
        (4.0 / 3.0, "4:3"),
        (3.0 / 4.0, "3:4"),
        (1.0, "1:1"),
        (21.0 / 9.0, "21:9"),
        (2.39, "2.39:1"),
    ] {
        if (ratio - r).abs() < 0.02 {
            return label.into();
        }
    }
    let (width, height) = (width.round() as u64, height.round() as u64);
    if width == 0 || height == 0 {
        return format!("{ratio:.2}:1");
    }
    let (mut a, mut b) = (width, height);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    format!("{}:{}", width / a, height / a)
}

fn draw_scene_label(pixmap: &mut Pixmap, l: &Layout, y: i32, a: &SceneAllocation) {
    let x = (l.outer_padding + 8) as f32;
    text::draw(
        pixmap,
        &format!("Scene #{}", a.scene_index + 1),
        (x, y as f32 + 8.0),
        14.0,
        Weight::Semibold,
        [26, 26, 26, 255],
    );
    text::draw(
        pixmap,
        &format!(
            "{} – {}",
            format_mmss(a.scene.start_time),
            format_mmss(a.scene.start_time + a.scene.duration)
        ),
        (x, y as f32 + 28.0),
        12.0,
        Weight::Regular,
        [77, 77, 77, 255],
    );
    text::draw(
        pixmap,
        &format!("{:.2}s", a.scene.duration),
        (x, y as f32 + 46.0),
        12.0,
        Weight::Regular,
        [77, 77, 77, 255],
    );
}

fn draw_barcode(
    pixmap: &mut Pixmap,
    l: &Layout,
    pool: &[PoolEntry],
    duration: f64,
    boundaries: &[f64],
) {
    let rect = l.barcode_rect();
    fill_rect(
        pixmap,
        rect.x as f32,
        rect.y as f32,
        rect.width as f32,
        rect.height as f32,
        Color::BLACK,
    );
    if pool.is_empty() || !duration.is_finite() || duration <= 0.0 {
        return;
    }
    for (index, entry) in pool.iter().enumerate() {
        let next = pool
            .get(index + 1)
            .map_or(duration, |entry| entry.start_time);
        let start = (entry.start_time / duration).clamp(0.0, 1.0);
        let end = (next / duration).clamp(start, 1.0);
        fill_rect(
            pixmap,
            (rect.x + start * rect.width) as f32,
            rect.y as f32,
            ((end - start) * rect.width).max(1.0) as f32,
            rect.height as f32,
            Color::from_rgba8(entry.mean_r, entry.mean_g, entry.mean_b, 255),
        );
    }
    for &time in boundaries {
        if !time.is_finite() || time < 0.0 || time > duration {
            continue;
        }
        let x = (rect.x + time / duration * rect.width) as f32;
        stroke_line(
            pixmap,
            x,
            rect.y as f32,
            x,
            (rect.y + rect.height) as f32,
            1.0,
            Color::from_rgba8(255, 255, 255, 217),
        );
    }
    stroke_rect(
        pixmap,
        rect.x as f32,
        rect.y as f32,
        rect.width as f32,
        rect.height as f32,
        0.5,
        gray(153),
    );
}

#[derive(Debug)]
struct Tick {
    time: f64,
    x: f32,
    label_x: f32,
    label_width: f32,
}
fn axis_ticks(l: &Layout, duration: f64) -> Vec<Tick> {
    if !duration.is_finite() || duration <= 0.0 {
        return vec![];
    }
    let rect = l.time_axis_rect();
    let max_labels = (rect.width / (36.0 + 16.0)).floor().max(2.0);
    let ideal = duration / max_labels;
    let step = [
        1., 2., 5., 10., 15., 30., 60., 120., 300., 600., 900., 1800., 3600.,
    ]
    .into_iter()
    .find(|step| *step >= ideal)
    .unwrap_or(3600.0);
    let mut ticks = vec![];
    let mut time = 0.0;
    while time <= duration + 0.001 {
        let x = (rect.x + time / duration * rect.width) as f32;
        let width = text::width(&format_mmss(time), 10.0, Weight::Regular);
        ticks.push(Tick {
            time,
            x,
            label_x: x,
            label_width: width,
        });
        time += step;
    }
    let count = ticks.len();
    for (index, tick) in ticks.iter_mut().enumerate() {
        tick.label_x = if index == 0 {
            tick.x
        } else if index + 1 == count {
            tick.x - tick.label_width
        } else {
            tick.x - tick.label_width / 2.0
        };
    }
    ticks
}
fn draw_time_axis(pixmap: &mut Pixmap, l: &Layout, duration: f64) {
    let rect = l.time_axis_rect();
    for tick in axis_ticks(l, duration) {
        stroke_line(
            pixmap,
            tick.x,
            rect.y as f32,
            tick.x,
            rect.y as f32 + 4.0,
            0.5,
            gray(102),
        );
        text::draw(
            pixmap,
            &format_mmss(tick.time),
            (tick.label_x, rect.y as f32 + 6.0),
            10.0,
            Weight::Regular,
            [77, 77, 77, 255],
        );
    }
}

/// Contain the source, preserve top-to-bottom display orientation and paint
/// neutral letter/pillar boxes. The final draw is filtered, as the source's
/// high-quality CGContext draw is; it never stretches a mismatched picture.
pub fn blit_rgb(pixmap: &mut Pixmap, buf: &KeyframeBuffer, x: i32, y: i32, w: i32, h: i32) {
    fill_rect(pixmap, x as f32, y as f32, w as f32, h as f32, gray(26));
    if buf.width == 0 || buf.height == 0 || w <= 0 || h <= 0 {
        return;
    }
    let scale = (w as f64 / buf.width as f64).min(h as f64 / buf.height as f64);
    let fit_w = (buf.width as f64 * scale).round().clamp(1.0, w as f64) as u32;
    let fit_h = (buf.height as f64 * scale).round().clamp(1.0, h as f64) as u32;
    let source = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
        buf.width as u32,
        buf.height as u32,
        &buf.data,
    )
    .expect("validated RGB thumbnail");
    let image = image::imageops::resize(
        &source,
        fit_w,
        fit_h,
        image::imageops::FilterType::CatmullRom,
    );
    let left = x + (w - fit_w as i32) / 2;
    let top = y + (h - fit_h as i32) / 2;
    let stride = pixmap.width() as usize;
    let height = pixmap.height() as i32;
    let pixels = pixmap.pixels_mut();
    for (ix, iy, pixel) in image.enumerate_pixels() {
        let dx = left + ix as i32;
        let dy = top + iy as i32;
        if dx >= 0 && dx < stride as i32 && dy >= 0 && dy < height {
            pixels[dy as usize * stride + dx as usize] =
                PremultipliedColorU8::from_rgba(pixel[0], pixel[1], pixel[2], 255).unwrap();
        }
    }
}
fn format_mmss(seconds: f64) -> String {
    let value = seconds.round().max(0.0) as u64;
    format!("{}:{:02}", value / 60, value % 60)
}
fn gray(value: u8) -> Color {
    Color::from_rgba8(value, value, value, 255)
}
fn fill_rect(pixmap: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, color: Color) {
    if w <= 0.0 || h <= 0.0 {
        return;
    }
    let path = PathBuilder::from_rect(tiny_skia::Rect::from_xywh(x, y, w, h).unwrap());
    let mut paint = Paint::default();
    paint.set_color(color);
    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}
fn stroke_rect(pixmap: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, width: f32, color: Color) {
    let path = PathBuilder::from_rect(tiny_skia::Rect::from_xywh(x, y, w, h).unwrap());
    let mut paint = Paint::default();
    paint.set_color(color);
    paint.anti_alias = true;
    pixmap.stroke_path(
        &path,
        &paint,
        &Stroke {
            width,
            ..Stroke::default()
        },
        Transform::identity(),
        None,
    );
}
fn stroke_line(
    pixmap: &mut Pixmap,
    x: f32,
    y: f32,
    to_x: f32,
    to_y: f32,
    width: f32,
    color: Color,
) {
    let mut path = PathBuilder::new();
    path.move_to(x, y);
    path.line_to(to_x, to_y);
    let mut paint = Paint::default();
    paint.set_color(color);
    paint.anti_alias = true;
    pixmap.stroke_path(
        &path.finish().unwrap(),
        &paint,
        &Stroke {
            width,
            ..Stroke::default()
        },
        Transform::identity(),
        None,
    );
}

#[cfg(test)]
mod barcode_source_tests {
    use super::*;
    use crate::video_minimap_composer::{Scene, allocate::allocate, layout::resolve};
    /// Kill: ordinal stripes or painting the leading centered-request gap.
    #[test]
    fn barcode_stripes_follow_requested_time_and_preserve_the_leading_gap() {
        let a = allocate(&[Scene {
            start_frame: 0,
            frame_count: 30,
            start_time: 0.0,
            duration: 1.0,
        }]);
        let l = resolve(&a, 640, 16.0 / 9.0);
        let r = l.barcode_rect();
        let mut image = Pixmap::new(l.total_width as u32, l.total_height as u32).unwrap();
        image.fill(Color::WHITE);
        draw_barcode(
            &mut image,
            &l,
            &[
                PoolEntry {
                    start_time: 0.25,
                    end_time: 0.75,
                    mean_r: 240,
                    mean_g: 0,
                    mean_b: 0,
                },
                PoolEntry {
                    start_time: 0.75,
                    end_time: 1.0,
                    mean_r: 0,
                    mean_g: 230,
                    mean_b: 0,
                },
            ],
            1.0,
            &[],
        );
        let pixel = |f: f64| {
            let p = image
                .pixel((r.x + f * r.width) as u32, (r.y + 10.0) as u32)
                .unwrap()
                .demultiply();
            [p.red(), p.green(), p.blue()]
        };
        assert_eq!(pixel(0.10), [0, 0, 0]);
        assert_eq!(pixel(0.40), [240, 0, 0]);
        assert_eq!(pixel(0.90), [0, 230, 0]);
    }
}

#[cfg(test)]
mod renderer_source_tests {
    use super::*;
    use crate::video_minimap_composer::{Scene, allocate::allocate_indexed, layout::resolve};
    fn rows() -> Vec<SceneAllocation> {
        allocate_indexed(
            &[
                Scene {
                    start_frame: 0,
                    frame_count: 60,
                    start_time: 0.0,
                    duration: 2.0,
                },
                Scene {
                    start_frame: 60,
                    frame_count: 60,
                    start_time: 2.0,
                    duration: 2.0,
                },
            ],
            Some(&[4, 12]),
        )
    }
    fn rgb(image: &Pixmap, x: u32, y: u32) -> [u8; 3] {
        let p = image.pixel(x, y).unwrap().demultiply();
        [p.red(), p.green(), p.blue()]
    }
    fn dark_pixels(image: &Pixmap, x: u32, y: u32, w: u32, h: u32) -> usize {
        (y..y + h)
            .flat_map(|y| (x..x + w).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let p = rgb(image, x, y);
                p.iter().all(|v| *v < 200)
            })
            .count()
    }

    /// Kill: blank header/gutter/timestamps, renumbering sampled original scene
    /// identities, or replacing bundled glyph drawing with metadata only.
    #[test]
    fn real_header_scene_duration_and_cell_time_glyphs_reach_the_raster() {
        let a = rows();
        let l = resolve(&a, 640, 16.0 / 9.0);
        let input = RenderInput {
            allocations: &a,
            keyframes: &[],
            barcode_pool: &[],
            layout: l,
            duration: 4.0,
            boundary_times: &[2.0],
            total_scene_count: 20,
            video_size: (1080.0, 1920.0),
            frame_rate: 30.0,
        };
        assert_eq!(
            header_label(&input),
            "0:04  ·  20 scenes (2 shown)  ·  30 fps  ·  9:16"
        );
        let image = render(&input).pixmap;
        assert_eq!(rgb(&image, 0, 0), [255, 255, 255]);
        assert!(
            dark_pixels(&image, 14, 10, (l.total_width - 28) as u32, 22) > 100,
            "header glyphs are absent"
        );
        for row in 0..2 {
            let y = (l.header_height + row * l.row_height()) as u32;
            assert!(
                dark_pixels(&image, 20, y + 8, 110, 17) > 35,
                "scene-number glyphs are absent"
            );
            assert!(
                dark_pixels(&image, 20, y + 28, 110, 15) > 20,
                "scene-range glyphs are absent"
            );
            assert!(
                dark_pixels(&image, 20, y + 46, 110, 15) > 20,
                "duration glyphs are absent"
            );
            assert!(
                dark_pixels(
                    &image,
                    (l.outer_padding + l.label_gutter_width + 4) as u32,
                    y + (l.row_vertical_padding + l.cell_height + 2) as u32,
                    48,
                    14
                ) > 10,
                "cell time glyphs are absent"
            );
        }
        let mut alternate = rows();
        alternate[0].scene_index = 0;
        let changed = render(&RenderInput {
            allocations: &alternate,
            ..input
        })
        .pixmap;
        let y = l.header_height as u32 + 8;
        assert!(
            (y..y + 17).any(|y| (20..130).any(|x| rgb(&image, x, y) != rgb(&changed, x, y))),
            "sampled original Scene #5 identity was lost"
        );
    }

    /// Kill: stretch a portrait into its cell, flip top/bottom, or leave the
    /// side bands as the white page instead of the source's neutral gray.
    #[test]
    fn aspect_fit_keeps_colored_display_corners_and_neutral_pillarboxes() {
        let mut data = vec![];
        for y in 0..8 {
            for _ in 0..4 {
                data.extend_from_slice(if y < 4 {
                    &[230, 20, 10]
                } else {
                    &[10, 200, 30]
                });
            }
        }
        let mut image = Pixmap::new(80, 40).unwrap();
        image.fill(Color::WHITE);
        blit_rgb(
            &mut image,
            &KeyframeBuffer {
                width: 4,
                height: 8,
                data,
            },
            0,
            0,
            80,
            40,
        );
        assert_eq!(rgb(&image, 5, 20), [26, 26, 26]);
        assert_eq!(rgb(&image, 75, 20), [26, 26, 26]);
        assert_eq!(rgb(&image, 40, 5), [230, 20, 10]);
        assert_eq!(rgb(&image, 40, 35), [10, 200, 30]);
        assert_eq!(rgb(&image, 29, 20), [26, 26, 26]);
        assert_ne!(rgb(&image, 30, 20), [26, 26, 26]);
        assert_ne!(rgb(&image, 49, 20), [26, 26, 26]);
        assert_eq!(rgb(&image, 50, 20), [26, 26, 26]);
    }

    /// Kill: skip the black fallback on an empty sparse pool, exclude the scene
    /// label gutter from the barcode, or draw evenly spaced invented cuts.
    #[test]
    fn full_width_barcode_empty_fallback_and_real_cut_positions_are_visible() {
        let a = rows();
        let l = resolve(&a, 640, 16.0 / 9.0);
        let r = l.barcode_rect();
        assert_eq!(r.x, 12.0);
        assert_eq!(r.width, (l.total_width - 24) as f64);
        let mut empty = Pixmap::new(l.total_width as u32, l.total_height as u32).unwrap();
        empty.fill(Color::WHITE);
        draw_barcode(&mut empty, &l, &[], 4.0, &[1.0]);
        assert_eq!(rgb(&empty, 20, (r.y + 20.0) as u32), [0, 0, 0]);
        assert_eq!(
            rgb(&empty, (r.x + r.width - 2.0) as u32, (r.y + 20.0) as u32),
            [0, 0, 0]
        );
        let mut image = empty.clone();
        draw_barcode(
            &mut image,
            &l,
            &[PoolEntry {
                start_time: 0.0,
                end_time: 4.0,
                mean_r: 200,
                mean_g: 0,
                mean_b: 0,
            }],
            4.0,
            &[1.0],
        );
        let y = (r.y + 30.0) as u32;
        let cut = (r.x + r.width / 4.0) as u32;
        assert_eq!(rgb(&image, 20, y), [200, 0, 0]);
        assert!(
            (cut - 1..=cut + 1)
                .map(|x| rgb(&image, x, y)[1])
                .max()
                .unwrap()
                > 70,
            "actual quarter-time cut absent"
        );
        assert_eq!(
            rgb(&image, (r.x + r.width / 2.0) as u32, y),
            [200, 0, 0],
            "an invented midpoint cut was drawn"
        );
    }

    /// Kill: eleven fixed unlabeled marks or edge labels clipped outside the
    /// original20-minute time domain/plot. Real glyph pixels must accompany ticks.
    #[test]
    fn twenty_minute_axis_uses_clean_adaptive_ticks_with_inset_edge_labels() {
        let a = rows();
        let l = resolve(&a, 640, 16.0 / 9.0);
        let r = l.time_axis_rect();
        let ticks = axis_ticks(&l, 1200.0);
        assert_eq!(
            ticks.iter().map(|t| t.time).collect::<Vec<_>>(),
            (0..=10).map(|n| n as f64 * 120.0).collect::<Vec<_>>()
        );
        assert!((f64::from(ticks[0].x) - r.x).abs() < 1e-5);
        assert!((f64::from(ticks.last().unwrap().x) - r.x - r.width).abs() < 1e-5);
        assert!(ticks.iter().all(|t| f64::from(t.label_x) >= r.x - 0.001
            && f64::from(t.label_x + t.label_width) <= r.x + r.width + 0.001));
        assert!(
            ticks
                .windows(2)
                .all(|p| p[0].label_x + p[0].label_width < p[1].label_x)
        );
        let mut image = Pixmap::new(l.total_width as u32, l.total_height as u32).unwrap();
        image.fill(Color::WHITE);
        draw_time_axis(&mut image, &l, 1200.0);
        for tick in ticks {
            assert!(
                dark_pixels(
                    &image,
                    tick.label_x.floor() as u32,
                    (r.y + 6.0) as u32,
                    tick.label_width.ceil() as u32 + 1,
                    14
                ) > 8,
                "tick label was only metadata"
            );
        }
    }
}
