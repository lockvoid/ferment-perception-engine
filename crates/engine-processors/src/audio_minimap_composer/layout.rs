//! Resolved pixel-layout for the audio minimap.
//!
//! Layout matches iOS `AudioMinimapComposer.resolveDimensions`:
//! - 40 px/sec width, clamped 1024..2048
//! - Fixed-height panels (mel 360, RMS 110, axis 24) + gutters + header
//! - Outer padding, left gutter (Hz/dB labels), right gutter (colorbar)

use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub total_width: i32,
    pub total_height: i32,
    pub outer_padding: i32,
    pub gutter_left: i32,
    pub gutter_right: i32,
    pub header_height: i32,
    pub header_to_mel_gap: i32,
    pub mel_height: i32,
    pub mel_to_rms_gap: i32,
    pub rms_height: i32,
    pub rms_to_axis_gap: i32,
    pub time_axis_height: i32,
}

/// Geometry exposed in API responses so vision-LLM consumers can index
/// into pixel rects without reverse-engineering the renderer.
#[derive(Debug, Clone, Serialize)]
pub struct LayoutGeometry {
    pub mel_rect: Rect,
    pub rms_rect: Rect,
    pub time_axis_rect: Rect,
    pub header_rect: Rect,
    pub plot_left: f32,
    pub plot_right: f32,
    pub time_min: f64,
    pub time_max: f64,
    /// The bar ruler actually drawn, including the numbered bar labels.
    pub bar_ruler: Vec<BarRulerTick>,
}

/// One drawn bar vertical. `bar` is 1-based, matching the beats cook's bar
/// indices and the minimap's own numbering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BarRulerTick {
    pub bar: usize,
    pub x: f32,
    /// Labelled verticals are the readable axis (alpha 0.9, width 2); the rest
    /// are context (alpha 0.35, width 1).
    pub labeled: bool,
}

/// Labels start every 4 bars — the musical phrase, and chart-reading research
/// says models localize by printed labels rather than geometry. Long material
/// thins deterministically: the label stride doubles until neighbouring labels
/// keep at least `MIN_LABEL_SPACING_PX`, the line stride until lines keep at
/// least `MIN_LINE_SPACING_PX`. A 67-minute mix once rendered 1374 lines into
/// an unreadable picket fence.
const BASE_LABEL_STRIDE: usize = 4;
const MIN_LABEL_SPACING_PX: f32 = 40.0;
const MIN_LINE_SPACING_PX: f32 = 3.0;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Layout {
    pub fn resolve(duration_secs: f64, max_width: usize) -> Self {
        let outer_padding = 12i32;
        let gutter_left = 56i32;
        let gutter_right = 44i32;
        let header_height = 36i32;
        let header_to_mel_gap = 18i32;
        let mel_to_rms_gap = 22i32;
        let rms_to_axis_gap = 4i32;
        let mel_height = 360i32;
        let rms_height = 110i32;
        let time_axis_height = 24i32;

        let px_per_sec = 40.0_f64;
        let width_min = 1024i32;
        let width_max = max_width.max(width_min as usize) as i32;
        let nominal_width =
            (duration_secs * px_per_sec) as i32 + 2 * outer_padding + gutter_left + gutter_right;
        let total_width = nominal_width.clamp(width_min, width_max);

        let total_height = header_height
            + header_to_mel_gap
            + mel_height
            + mel_to_rms_gap
            + rms_height
            + rms_to_axis_gap
            + time_axis_height
            + outer_padding;

        Self {
            total_width,
            total_height,
            outer_padding,
            gutter_left,
            gutter_right,
            header_height,
            header_to_mel_gap,
            mel_height,
            mel_to_rms_gap,
            rms_height,
            rms_to_axis_gap,
            time_axis_height,
        }
    }

    pub fn plot_left(&self) -> i32 {
        self.outer_padding + self.gutter_left
    }

    pub fn plot_right(&self) -> i32 {
        self.total_width - self.outer_padding - self.gutter_right
    }

    pub fn header_rect(&self) -> Rect {
        Rect {
            x: 0.0,
            y: 0.0,
            width: self.total_width as f32,
            height: self.header_height as f32,
        }
    }

    pub fn mel_rect(&self) -> Rect {
        let y = self.header_height + self.header_to_mel_gap;
        Rect {
            x: self.plot_left() as f32,
            y: y as f32,
            width: (self.plot_right() - self.plot_left()) as f32,
            height: self.mel_height as f32,
        }
    }

    pub fn rms_rect(&self) -> Rect {
        let y = self.header_height + self.header_to_mel_gap + self.mel_height + self.mel_to_rms_gap;
        Rect {
            x: self.plot_left() as f32,
            y: y as f32,
            width: (self.plot_right() - self.plot_left()) as f32,
            height: self.rms_height as f32,
        }
    }

    pub fn time_axis_rect(&self) -> Rect {
        let y = self.header_height
            + self.header_to_mel_gap
            + self.mel_height
            + self.mel_to_rms_gap
            + self.rms_height
            + self.rms_to_axis_gap;
        Rect {
            x: self.plot_left() as f32,
            y: y as f32,
            width: (self.plot_right() - self.plot_left()) as f32,
            height: self.time_axis_height as f32,
        }
    }

    pub fn geometry(&self, duration_secs: f64, bar_ruler: &[f64]) -> LayoutGeometry {
        LayoutGeometry {
            mel_rect: self.mel_rect(),
            rms_rect: self.rms_rect(),
            time_axis_rect: self.time_axis_rect(),
            header_rect: self.header_rect(),
            plot_left: self.plot_left() as f32,
            plot_right: self.plot_right() as f32,
            time_min: 0.0,
            time_max: duration_secs,
            bar_ruler: self.bar_ruler_ticks(bar_ruler, duration_secs),
        }
    }

    /// The verticals the ruler draws for `downbeats` (track seconds), after
    /// the label/line thinning above. Empty when there is no ruler to draw.
    pub fn bar_ruler_ticks(&self, downbeats: &[f64], duration_secs: f64) -> Vec<BarRulerTick> {
        if downbeats.is_empty() || duration_secs <= 0.0 {
            return Vec::new();
        }
        let plot_left = self.plot_left() as f32;
        let plot_width = (self.plot_right() - self.plot_left()) as f32;
        let bar_width = plot_width / downbeats.len() as f32;
        let mut label_stride = BASE_LABEL_STRIDE;
        while bar_width * (label_stride as f32) < MIN_LABEL_SPACING_PX {
            label_stride *= 2;
        }
        let mut line_stride = 1usize;
        while bar_width * (line_stride as f32) < MIN_LINE_SPACING_PX {
            line_stride *= 2;
        }

        downbeats
            .iter()
            .enumerate()
            .filter_map(|(index, &time)| {
                let bar = index + 1;
                let labeled = bar == 1 || bar.is_multiple_of(label_stride);
                (labeled || bar.is_multiple_of(line_stride)).then(|| BarRulerTick {
                    bar,
                    x: plot_left + (time / duration_secs) as f32 * plot_width,
                    labeled,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_clip_clamps_to_min_width() {
        let l = Layout::resolve(2.0, 2048);
        assert_eq!(l.total_width, 1024);
    }

    #[test]
    fn long_clip_clamps_to_max_width() {
        let l = Layout::resolve(120.0, 2048);
        assert_eq!(l.total_width, 2048);
    }

    #[test]
    fn medium_clip_scales_with_duration() {
        // 30s * 40px/sec + padding+gutters ~= 1312, within [1024, 2048].
        let l = Layout::resolve(30.0, 2048);
        assert!(l.total_width > 1024);
        assert!(l.total_width < 2048);
    }

    #[test]
    fn a_short_track_labels_every_fourth_bar_and_draws_the_rest() {
        // 32 bars over a 1024 px canvas: plot width 912, bar width 28.5 px, so
        // 4 × 28.5 = 114 px clears the 40 px label spacing at the base stride
        // and every bar clears the 3 px line floor.
        let layout = Layout::resolve(2.0, 2048);
        let downbeats = (0..32).map(|bar| f64::from(bar) * 2.0).collect::<Vec<_>>();
        let ticks = layout.bar_ruler_ticks(&downbeats, 64.0);
        assert_eq!(ticks.len(), 32, "every bar keeps its line");
        assert_eq!(
            ticks
                .iter()
                .filter(|tick| tick.labeled)
                .map(|tick| tick.bar)
                .collect::<Vec<_>>(),
            vec![1, 4, 8, 12, 16, 20, 24, 28, 32]
        );
        assert!((ticks[0].x - layout.plot_left() as f32).abs() < 1e-3);
    }

    #[test]
    fn a_long_mix_thins_both_strides_instead_of_drawing_a_picket_fence() {
        // 1374 bars — the 67-minute mix. Bar width is 2048-wide plot / 1374 =
        // 1.42 px, so lines thin to a stride of 4 (5.7 px) and labels to 32
        // (45.5 px).
        let layout = Layout::resolve(4_000.0, 2048);
        let downbeats = (0..1_374)
            .map(|bar| f64::from(bar) * 2.5)
            .collect::<Vec<_>>();
        let ticks = layout.bar_ruler_ticks(&downbeats, 1_374.0 * 2.5);
        let labeled = ticks.iter().filter(|tick| tick.labeled).collect::<Vec<_>>();
        assert_eq!(labeled[0].bar, 1);
        assert_eq!(labeled[1].bar, 32, "label stride doubled to 32");
        let lines = ticks
            .iter()
            .filter(|tick| !tick.labeled)
            .map(|tick| tick.bar)
            .collect::<Vec<_>>();
        assert_eq!(lines[0], 4, "line stride doubled to 4");
        let plot_width = (layout.plot_right() - layout.plot_left()) as f32;
        let spacing = plot_width / 1_374.0;
        assert!(spacing * 4.0 >= 3.0, "lines keep 3 px");
        assert!(spacing * 32.0 >= 40.0, "labels keep 40 px");
        assert!(ticks.len() < 1_374 / 2, "thinned: {}", ticks.len());
    }

    #[test]
    fn no_downbeats_and_no_duration_draw_no_ruler() {
        let layout = Layout::resolve(30.0, 2048);
        assert!(layout.bar_ruler_ticks(&[], 30.0).is_empty());
        assert!(layout.bar_ruler_ticks(&[0.0, 2.0], 0.0).is_empty());
    }

    #[test]
    fn rects_sum_to_total_height() {
        let l = Layout::resolve(30.0, 2048);
        let last_y = l.time_axis_rect().y + l.time_axis_rect().height;
        // total_height = last_y + outer_padding (bottom)
        assert_eq!(last_y as i32 + l.outer_padding, l.total_height);
    }
}
