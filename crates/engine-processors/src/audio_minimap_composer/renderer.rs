//! Tiny-skia rasterizer for the audio minimap. Produces an RGBA8 pixmap;
//! callers encode to PNG/JPEG via the `image` crate.
//!
//! Coverage:
//! - mel spectrogram (magma-mapped pixels)
//! - RMS envelope (gray polyline + fill)
//! - shared time axis with tick marks
//! - colorbar (right gutter)
//!
//! - bar ruler (downbeat verticals through mel + RMS)
//!
//! Header, frequency/level axes, adaptive time ticks, and numbered bars are
//! painted into the image, so a vision consumer can read the chart itself.

use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use super::colormap::magma;
use super::layout::Layout;
use super::mel::{N_MELS, SAMPLE_RATE, TOP_DB};
use super::text::{self, Weight};

#[derive(Debug)]
pub struct RenderInput<'a> {
    pub mel_db: &'a [Vec<f32>],
    pub rms_db: &'a [f32],
    /// Downbeat times in track seconds. Empty = no bar ruler.
    pub bar_ruler: &'a [f64],
    pub duration_secs: f64,
    pub layout: Layout,
}

#[derive(Debug)]
pub struct RenderOutput {
    pub pixmap: Pixmap,
}

pub fn render(input: &RenderInput<'_>) -> RenderOutput {
    let layout = input.layout;
    let mut pixmap =
        Pixmap::new(layout.total_width as u32, layout.total_height as u32).expect("pixmap");
    pixmap.fill(Color::WHITE);

    draw_header(&mut pixmap, input);
    draw_mel(&mut pixmap, input);
    draw_mel_separator(&mut pixmap, &layout);
    draw_mel_y_axis(&mut pixmap, &layout);
    draw_rms(&mut pixmap, input);
    draw_rms_separator(&mut pixmap, &layout);
    draw_rms_y_axis(&mut pixmap, &layout);
    draw_bar_ruler(&mut pixmap, input);
    draw_time_axis(&mut pixmap, &layout, input.duration_secs);
    draw_colorbar(&mut pixmap, &layout);

    RenderOutput { pixmap }
}

fn draw_header(pixmap: &mut Pixmap, input: &RenderInput<'_>) {
    let height = input.layout.header_height as f32;
    line(
        pixmap,
        (0.0, height - 0.5),
        (input.layout.total_width as f32, height - 0.5),
        1.0,
        Color::from_rgba8(166, 166, 166, 255),
    );
    let label = format!(
        "{}  ·  {}",
        format_mmss(input.duration_secs),
        format_sample_rate(SAMPLE_RATE as f64)
    );
    text::draw(
        pixmap,
        &label,
        (14.0, height / 2.0 - 7.0),
        14.0,
        Weight::Semibold,
        [38, 38, 38, 255],
    );
}

fn draw_mel(pixmap: &mut Pixmap, input: &RenderInput<'_>) {
    let rect = input.layout.mel_rect();
    let frames = input.mel_db.len() as f32;
    if frames <= 0.0 {
        return;
    }

    let pixels = pixmap.pixels_mut();
    let stride = input.layout.total_width as usize;
    let x_start = rect.x as usize;
    let y_start = rect.y as usize;
    let w = rect.width.max(0.0) as usize;
    let h = rect.height.max(0.0) as usize;

    for px in 0..w {
        let frame_f = (px as f32 / w as f32) * frames;
        let frame_idx = (frame_f as usize).min(input.mel_db.len() - 1);
        let row = &input.mel_db[frame_idx];

        for py in 0..h {
            // Flip vertically: top of image = high mel band.
            let mel_idx = N_MELS - 1 - ((py as f32 / h as f32) * N_MELS as f32) as usize;
            let mel_idx = mel_idx.min(N_MELS - 1);
            let db = row[mel_idx];
            let normalized = ((db + TOP_DB) / TOP_DB).clamp(0.0, 1.0);
            let (r, g, b) = magma(normalized);

            let dst_x = x_start + px;
            let dst_y = y_start + py;
            if dst_x >= stride || dst_y >= input.layout.total_height as usize {
                continue;
            }
            pixels[dst_y * stride + dst_x] =
                tiny_skia::PremultipliedColorU8::from_rgba(r, g, b, 255).expect("color");
        }
    }
}

fn draw_mel_separator(pixmap: &mut Pixmap, layout: &Layout) {
    let rect = layout.mel_rect();
    stroke_rect(
        pixmap,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        Color::from_rgba8(153, 153, 153, 255),
    );
}

fn draw_mel_y_axis(pixmap: &mut Pixmap, layout: &Layout) {
    let rect = layout.mel_rect();
    for (label, fraction) in [
        ("16k", 0.93),
        ("8k", 0.78),
        ("4k", 0.62),
        ("2k", 0.47),
        ("1k", 0.34),
        ("500", 0.22),
        ("125", 0.06),
    ] {
        let y = rect.y + rect.height * (1.0 - fraction);
        line(
            pixmap,
            (rect.x - 4.0, y),
            (rect.x, y),
            0.5,
            Color::from_rgba8(153, 153, 153, 128),
        );
        text::draw(
            pixmap,
            label,
            (rect.x - 36.0, y - 5.0),
            9.0,
            Weight::Regular,
            [77, 77, 77, 255],
        );
    }
    text::draw_rotated(
        pixmap,
        "Hz (mel)",
        (rect.x - 50.0, rect.y + rect.height / 2.0 + 22.0),
        9.0,
        [77, 77, 77, 255],
    );
}

fn draw_rms(pixmap: &mut Pixmap, input: &RenderInput<'_>) {
    let rect = input.layout.rms_rect();
    let mut background = Paint::default();
    background.set_color(Color::from_rgba8(242, 242, 242, 255));
    pixmap.fill_rect(
        tiny_skia::Rect::from_xywh(rect.x, rect.y, rect.width, rect.height).expect("RMS rect"),
        &background,
        Transform::identity(),
        None,
    );
    let frames = input.rms_db.len();
    if frames < 2 {
        return;
    }

    // Build a filled polyline from baseline → samples → baseline.
    let mut path = PathBuilder::new();
    let baseline_y = rect.y + rect.height;
    path.move_to(rect.x, baseline_y);
    for i in 0..frames {
        let x = rect.x + (i as f32 / (frames - 1) as f32) * rect.width;
        let normalized = ((input.rms_db[i] + TOP_DB) / TOP_DB).clamp(0.0, 1.0);
        let y = rect.y + (1.0 - normalized) * rect.height;
        path.line_to(x, y);
    }
    path.line_to(rect.x + rect.width, baseline_y);
    path.close();
    let path = path.finish().expect("rms path");

    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba8(77, 77, 77, 255));
    paint.anti_alias = true;
    pixmap.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
}

fn draw_rms_separator(pixmap: &mut Pixmap, layout: &Layout) {
    let rect = layout.rms_rect();
    stroke_rect(
        pixmap,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        Color::from_rgba8(153, 153, 153, 255),
    );
}

fn draw_rms_y_axis(pixmap: &mut Pixmap, layout: &Layout) {
    let rect = layout.rms_rect();
    for (label, fraction) in [("0 dB", 0.0), ("-80 dB", 1.0)] {
        let y = rect.y + fraction * rect.height;
        line(
            pixmap,
            (rect.x - 4.0, y),
            (rect.x, y),
            0.5,
            Color::from_rgba8(153, 153, 153, 128),
        );
        text::draw(
            pixmap,
            label,
            (rect.x - 38.0, y - 5.0),
            9.0,
            Weight::Regular,
            [77, 77, 77, 255],
        );
    }
    text::draw_rotated(
        pixmap,
        "RMS",
        (rect.x - 50.0, rect.y + rect.height / 2.0 + 12.0),
        9.0,
        [77, 77, 77, 255],
    );
}

/// Downbeat verticals through mel + RMS — the axis a vision LLM reads
/// structure against ("the drop is bar 35"). Labelled bars get the readable
/// weight (alpha 0.9, width 2), the rest are context (0.35, width 1).
/// White digits over a black offset stay readable on the mel surface.
fn draw_bar_ruler(pixmap: &mut Pixmap, input: &RenderInput<'_>) {
    const BAR_LABEL_SIZE: f32 = 14.0;

    let ticks = input
        .layout
        .bar_ruler_ticks(input.bar_ruler, input.duration_secs);
    if ticks.is_empty() {
        return;
    }
    let top = input.layout.mel_rect().y;
    let rms = input.layout.rms_rect();
    let bottom = rms.y + rms.height;

    for tick in ticks {
        let mut path = PathBuilder::new();
        path.move_to(tick.x, top);
        path.line_to(tick.x, bottom);
        let Some(path) = path.finish() else { continue };
        let mut paint = Paint::default();
        paint.set_color(
            Color::from_rgba(1.0, 0.82, 0.12, if tick.labeled { 0.9 } else { 0.35 })
                .expect("bar ruler color"),
        );
        paint.anti_alias = true;
        let stroke = Stroke {
            width: if tick.labeled { 2.0 } else { 1.0 },
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        if tick.labeled {
            let label = tick.bar.to_string();
            text::draw(
                pixmap,
                &label,
                (tick.x + 4.0, top + 3.0),
                BAR_LABEL_SIZE,
                Weight::Semibold,
                [0, 0, 0, 230],
            );
            text::draw(
                pixmap,
                &label,
                (tick.x + 3.0, top + 2.0),
                BAR_LABEL_SIZE,
                Weight::Semibold,
                [255, 255, 255, 255],
            );
        }
    }
}

fn draw_time_axis(pixmap: &mut Pixmap, layout: &Layout, duration: f64) {
    let rect = layout.time_axis_rect();
    let ticks = time_ticks(duration, rect.width);
    for (i, time) in ticks.iter().enumerate() {
        let x = rect.x + (*time / duration) as f32 * rect.width;
        line(
            pixmap,
            (x, rect.y),
            (x, rect.y + 4.0),
            0.5,
            Color::from_rgba8(102, 102, 102, 255),
        );
        let label = format_mmss(*time);
        let width = text::width(&label, 10.0, Weight::Regular);
        let label_x = if i == 0 {
            x
        } else if i == ticks.len() - 1 {
            x - width
        } else {
            x - width / 2.0
        };
        text::draw(
            pixmap,
            &label,
            (label_x, rect.y + 6.0),
            10.0,
            Weight::Regular,
            [77, 77, 77, 255],
        );
    }
}

fn time_ticks(duration: f64, width: f32) -> Vec<f64> {
    if !duration.is_finite() || duration <= 0.0 || !width.is_finite() {
        return Vec::new();
    }
    let maximum = ((width / 52.0) as usize).max(2);
    let ideal = duration / maximum as f64;
    let step = [
        1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0,
    ]
    .into_iter()
    .find(|step| *step >= ideal)
    .unwrap_or(3600.0);
    let count = ((duration + 0.001) / step).floor() as usize;
    (0..=count).map(|index| index as f64 * step).collect()
}

fn format_mmss(seconds: f64) -> String {
    let total = seconds.round().max(0.0) as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

fn format_sample_rate(sample_rate: f64) -> String {
    if sample_rate >= 1000.0 {
        let khz = sample_rate / 1000.0;
        if khz == khz.round() {
            format!("{khz:.0} kHz")
        } else {
            format!("{khz:.2} kHz")
        }
    } else {
        format!("{sample_rate:.0} Hz")
    }
}

fn draw_colorbar(pixmap: &mut Pixmap, layout: &Layout) {
    // Vertical magma gradient strip in the right gutter, aligned with mel rect.
    let mel_rect = layout.mel_rect();
    let gutter_x = mel_rect.x + mel_rect.width + 8.0;
    let bar_w = 12.0_f32;
    if bar_w <= 0.0 {
        return;
    }

    let pixels = pixmap.pixels_mut();
    let stride = layout.total_width as usize;
    let x_start = gutter_x as usize;
    let y_start = mel_rect.y as usize;
    let w = bar_w.max(1.0) as usize;
    let h = mel_rect.height.max(1.0) as usize;

    for py in 0..h {
        let normalized = 1.0 - (py as f32 / h.saturating_sub(1).max(1) as f32);
        let (r, g, b) = magma(normalized);
        for px in 0..w {
            let dst_x = x_start + px;
            let dst_y = y_start + py;
            if dst_x >= stride || dst_y >= layout.total_height as usize {
                continue;
            }
            pixels[dst_y * stride + dst_x] =
                tiny_skia::PremultipliedColorU8::from_rgba(r, g, b, 255).expect("color");
        }
    }
    stroke_rect(
        pixmap,
        gutter_x,
        mel_rect.y,
        bar_w,
        mel_rect.height,
        Color::from_rgba8(153, 153, 153, 255),
    );
    text::draw(
        pixmap,
        "0 dB",
        (gutter_x + bar_w + 4.0, mel_rect.y - 4.0),
        9.0,
        Weight::Regular,
        [77, 77, 77, 255],
    );
    text::draw(
        pixmap,
        "-80",
        (gutter_x + bar_w + 4.0, mel_rect.y + mel_rect.height - 6.0),
        9.0,
        Weight::Regular,
        [77, 77, 77, 255],
    );
}

fn line(pixmap: &mut Pixmap, from: (f32, f32), to: (f32, f32), width: f32, color: Color) {
    let mut path = PathBuilder::new();
    path.move_to(from.0, from.1);
    path.line_to(to.0, to.1);
    let mut paint = Paint::default();
    paint.set_color(color);
    paint.anti_alias = true;
    pixmap.stroke_path(
        &path.finish().expect("line"),
        &paint,
        &Stroke {
            width,
            ..Stroke::default()
        },
        Transform::identity(),
        None,
    );
}

fn stroke_rect(pixmap: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, color: Color) {
    let mut path = PathBuilder::new();
    path.move_to(x, y);
    path.line_to(x + w, y);
    path.line_to(x + w, y + h);
    path.line_to(x, y + h);
    path.close();
    let path = path.finish().expect("rect");
    let mut paint = Paint::default();
    paint.set_color(color);
    let stroke = Stroke {
        width: 0.5,
        ..Stroke::default()
    };
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Current Swift's adaptive clean steps and half-away rounded labels.
    /// Kill: reinstate ten equally spaced, unlabeled divisions.
    #[test]
    fn time_ticks_and_header_values_follow_the_current_ios_chart() {
        assert_eq!(time_ticks(4.0, 900.0), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            time_ticks(60.0, 900.0),
            (0..=12).map(|i| i as f64 * 5.0).collect::<Vec<_>>()
        );
        assert_eq!(
            time_ticks(3600.0, 1924.0),
            (0..=30).map(|i| i as f64 * 120.0).collect::<Vec<_>>()
        );
        assert!(time_ticks(0.0, 900.0).is_empty());
        assert_eq!(format_mmss(59.5), "1:00");
        assert_eq!(format_mmss(-0.5), "0:00");
        assert_eq!(format_sample_rate(22050.0), "22.05 kHz");
        assert_eq!(format_sample_rate(48000.0), "48 kHz");
        assert_eq!(format_sample_rate(800.0), "800 Hz");
    }

    /// Read actual bitmap regions that the Swift chart reserves for labels.
    /// Kill: omit text drawing, keep only line geometry, or restore dark fill.
    #[test]
    fn chart_labels_and_bar_numbers_are_painted_into_the_image() {
        let layout = Layout::resolve(4.0, 2048);
        let mel = vec![vec![-40.0; N_MELS]; 64];
        let rms = vec![-80.0; 64];
        let plain = render(&RenderInput {
            mel_db: &mel,
            rms_db: &rms,
            bar_ruler: &[],
            duration_secs: 4.0,
            layout,
        });
        let ruled = render(&RenderInput {
            mel_db: &mel,
            rms_db: &rms,
            bar_ruler: &[2.0],
            duration_secs: 4.0,
            layout,
        });
        assert_eq!((plain.pixmap.width(), plain.pixmap.height()), (1024, 586));
        let pixel = |image: &Pixmap, x: usize, y: usize| image.pixels()[y * 1024 + x];
        assert_eq!(
            pixel(&plain.pixmap, 1018, 580),
            tiny_skia::PremultipliedColorU8::from_rgba(255, 255, 255, 255).unwrap()
        );
        assert_eq!(
            pixel(&plain.pixmap, 300, 500).red(),
            242,
            "empty RMS surface is light gray"
        );
        let dark_count = |x0: usize, y0: usize, x1: usize, y1: usize| {
            (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| pixel(&plain.pixmap, x, y).red() < 150)
                .count()
        };
        assert!(
            dark_count(14, 11, 180, 29) > 40,
            "duration/sample-rate header"
        );
        assert!(
            dark_count(12, 65, 62, 402) > 40,
            "frequency labels and rotated Hz title"
        );
        assert!(dark_count(12, 431, 64, 552) > 20, "RMS dB labels");
        assert!(
            dark_count(68, 556, 968, 572) > 40,
            "time labels below their ticks"
        );
        // The sole downbeat is bar 1 at 2 / 4 of [68, 968], x = 518.
        let yellow = pixel(&ruled.pixmap, 518, 234);
        assert!(yellow.red() > 180 && yellow.green() > 140 && yellow.blue() < yellow.red() - 60);
        let white_digits = (56..70)
            .flat_map(|y| (521..534).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let p = pixel(&ruled.pixmap, x, y);
                p.red() > 220 && p.green() > 220 && p.blue() > 220
            })
            .count();
        assert!(
            white_digits > 4,
            "bar number must be visible over the mel: {white_digits}"
        );
        assert_ne!(plain.pixmap.data(), ruled.pixmap.data());
    }

    #[test]
    fn renders_without_panic_for_short_clip() {
        // 2 seconds of silence: ~86 mel frames, ~86 RMS samples.
        let layout = Layout::resolve(2.0, 2048);
        let mel = vec![vec![-TOP_DB; N_MELS]; 86];
        let rms = vec![-TOP_DB; 86];
        let out = render(&RenderInput {
            mel_db: &mel,
            rms_db: &rms,
            bar_ruler: &[],
            duration_secs: 2.0,
            layout,
        });
        assert_eq!(out.pixmap.width(), layout.total_width as u32);
        assert_eq!(out.pixmap.height(), layout.total_height as u32);
    }

    #[test]
    fn renders_empty_inputs_gracefully() {
        let layout = Layout::resolve(2.0, 2048);
        let out = render(&RenderInput {
            mel_db: &[],
            rms_db: &[],
            bar_ruler: &[],
            duration_secs: 2.0,
            layout,
        });
        assert!(out.pixmap.width() > 0);
    }

    /// The ruler must PAINT, not merely be planned: a labelled vertical puts
    /// yellow on the mel panel where the plain render has magma.
    #[test]
    fn a_bar_ruler_paints_yellow_verticals_the_plain_render_does_not_have() {
        let layout = Layout::resolve(30.0, 2048);
        let mel = vec![vec![-TOP_DB; N_MELS]; 1_290];
        let rms = vec![-TOP_DB; 1_290];
        let downbeats = (0..15).map(|bar| f64::from(bar) * 2.0).collect::<Vec<_>>();
        let input = |bar_ruler: &'static [f64]| RenderInput {
            mel_db: &mel,
            rms_db: &rms,
            bar_ruler,
            duration_secs: 30.0,
            layout,
        };
        let plain = render(&input(&[]));
        let ruled = render(&RenderInput {
            bar_ruler: &downbeats,
            ..input(&[])
        });

        let yellow_pixels = |pixmap: &Pixmap| {
            pixmap
                .pixels()
                .iter()
                .filter(|pixel| {
                    let color = pixel.demultiply();
                    color.red() > 180 && color.green() > 130 && color.blue() < 90
                })
                .count()
        };
        assert_eq!(yellow_pixels(&plain.pixmap), 0, "no ruler, no yellow");
        let painted = yellow_pixels(&ruled.pixmap);
        let ruler_height = layout.rms_rect().y + layout.rms_rect().height - layout.mel_rect().y;
        assert!(
            painted > ruler_height as usize,
            "15 verticals over {ruler_height} px painted only {painted} yellow pixels"
        );
    }
}
