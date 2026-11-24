//! Pixel layout resolver for the video minimap.
//!
//! Cell sizing is aspect-preserving and adapts to scene count:
//! more scenes → smaller cells. Width is clamped to `max_width`.

use serde::Serialize;

#[allow(unused_imports)]
use super::allocate::SceneAllocation;

const MIN_CELL_WIDTH: i32 = 90;
const TARGET_CELL_WIDTH: i32 = 240;
pub const FALLBACK_CELL_ASPECT: f32 = 16.0 / 9.0;
pub const BARCODE_POOL_FPS: f64 = 2.0;
// The encoded5MiB ceiling alone cannot bound an extremely tall raw canvas.
//64 megapixels caps the RGBA allocation at256MiB before any rendering.
const MAX_RASTER_PIXELS: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub total_width: i32,
    pub total_height: i32,
    pub label_gutter_width: i32,
    pub cell_width: i32,
    pub cell_height: i32,
    pub cell_gap: i32,
    pub cell_time_label_height: i32,
    pub row_vertical_padding: i32,
    pub header_height: i32,
    pub barcode_height: i32,
    pub time_axis_height: i32,
    pub outer_padding: i32,
    pub max_cells_per_row: i32,
}

impl Layout {
    pub fn cells_area_width(&self) -> i32 {
        self.total_width - self.label_gutter_width - 2 * self.outer_padding
    }

    pub fn row_height(&self) -> i32 {
        self.cell_height
            .saturating_add(self.cell_time_label_height)
            .saturating_add(self.row_vertical_padding.saturating_mul(2))
    }

    pub fn validate_raster_extent(&self) -> ferment_engine_core::Result<()> {
        let pixels = u64::try_from(self.total_width)
            .ok()
            .zip(u64::try_from(self.total_height).ok())
            .and_then(|(width, height)| width.checked_mul(height));
        if self.total_width <= 0
            || self.total_height <= 0
            || pixels.is_none_or(|count| count > MAX_RASTER_PIXELS)
        {
            return Err(ferment_engine_core::EngineError::invalid_config(format!(
                "minimap raster extent {}x{} exceeds the64-megapixel allocation limit",
                self.total_width, self.total_height
            )));
        }
        Ok(())
    }

    pub fn header_rect(&self) -> ferment_engine_core::Rect {
        ferment_engine_core::Rect {
            x: 0.0,
            y: 0.0,
            width: f64::from(self.total_width),
            height: f64::from(self.header_height),
        }
    }

    pub fn barcode_rect(&self) -> ferment_engine_core::Rect {
        ferment_engine_core::Rect {
            x: f64::from(self.outer_padding),
            y: f64::from(
                self.total_height
                    - self.barcode_height
                    - self.time_axis_height
                    - 2 * self.outer_padding,
            ),
            width: f64::from(self.total_width - 2 * self.outer_padding),
            height: f64::from(self.barcode_height),
        }
    }

    pub fn time_axis_rect(&self) -> ferment_engine_core::Rect {
        let barcode = self.barcode_rect();
        ferment_engine_core::Rect {
            x: barcode.x,
            y: barcode.y + barcode.height,
            width: barcode.width,
            height: f64::from(self.time_axis_height),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LayoutGeometry {
    pub header_rect: ferment_engine_core::Rect,
    pub barcode_rect: ferment_engine_core::Rect,
    pub time_axis_rect: ferment_engine_core::Rect,
    pub width: i32,
    pub height: i32,
    pub cell_width: i32,
    pub cell_height: i32,
    pub max_cells_per_row: i32,
    pub barcode_height: i32,
    pub header_height: i32,
}

pub fn resolve(allocations: &[SceneAllocation], max_width: i32, cell_aspect: f32) -> Layout {
    let outer_padding: i32 = 12;
    let label_gutter_width = 132;
    let cell_gap = 6;
    let cell_time_label_height = 18;
    let row_vertical_padding = 6;
    let header_height = 44;
    let barcode_height = 72;
    let time_axis_height = 22;

    let max_cells_per_row = i32::try_from(
        allocations
            .iter()
            .map(|a| a.cell_frames.len())
            .max()
            .unwrap_or(1)
            .max(1),
    )
    .unwrap_or(i32::MAX);

    let target_cell_height = match allocations.len() {
        0..=7 => 200,
        8..=14 => 170,
        15..=24 => 130,
        _ => 100,
    };

    let mut cell_height = target_cell_height;
    let mut cell_width = ((cell_height as f32 * cell_aspect).round() as i32).max(1);

    if cell_width < MIN_CELL_WIDTH {
        cell_width = MIN_CELL_WIDTH;
        cell_height = ((cell_width as f32 / cell_aspect).round() as i32).max(60);
    }
    if cell_width > TARGET_CELL_WIDTH {
        cell_width = TARGET_CELL_WIDTH;
        cell_height = ((cell_width as f32 / cell_aspect).round() as i32).max(60);
    }

    let available_for_cells = max_width
        .saturating_sub(2 * outer_padding)
        .saturating_sub(label_gutter_width);
    let cells_total_gap = (max_cells_per_row - 1).saturating_mul(cell_gap);
    if max_cells_per_row > 0 {
        let width_budget_per_cell =
            available_for_cells.saturating_sub(cells_total_gap) / max_cells_per_row;
        if cell_width > width_budget_per_cell {
            cell_width = width_budget_per_cell.max(MIN_CELL_WIDTH);
            cell_height = ((cell_width as f32 / cell_aspect).round() as i32).max(60);
        }
    }

    let total_width = (label_gutter_width + 2 * outer_padding)
        .saturating_add(cells_total_gap)
        .saturating_add(max_cells_per_row.saturating_mul(cell_width));
    let row_height = cell_height
        .saturating_add(cell_time_label_height)
        .saturating_add(2 * row_vertical_padding);
    let total_height = i32::try_from(allocations.len())
        .unwrap_or(i32::MAX)
        .saturating_mul(row_height)
        .saturating_add(header_height)
        .saturating_add(barcode_height)
        .saturating_add(time_axis_height)
        .saturating_add(3 * outer_padding);

    Layout {
        total_width,
        total_height,
        label_gutter_width,
        cell_width,
        cell_height,
        cell_gap,
        cell_time_label_height,
        row_vertical_padding,
        header_height,
        barcode_height,
        time_axis_height,
        outer_padding,
        max_cells_per_row,
    }
}

#[cfg(test)]
mod tests {
    use super::super::allocate::{Scene, allocate};
    use super::super::processor::ImageFormat;
    use super::*;

    fn scene(start: usize, count: usize) -> Scene {
        Scene {
            start_frame: start,
            frame_count: count,
            start_time: start as f64 / 30.0,
            duration: count as f64 / 30.0,
        }
    }

    #[test]
    fn total_width_respects_max_cap() {
        let scenes = (0..10)
            .map(|index| scene(index * 600, 600))
            .collect::<Vec<_>>();
        let layout = resolve(&allocate(&scenes), 1_568, 16.0 / 9.0);
        assert!(layout.total_width <= 1_568);
    }

    #[test]
    fn respects_custom_max_width() {
        let scenes = (0..6)
            .map(|index| scene(index * 300, 300))
            .collect::<Vec<_>>();
        let layout = resolve(&allocate(&scenes), 800, 16.0 / 9.0);
        assert!(layout.total_width <= 800);
    }

    #[test]
    fn few_scene_few_cells_gets_larger_cells() {
        let few = (0..3)
            .map(|index| scene(index * 300, 300))
            .collect::<Vec<_>>();
        let many = (0..30)
            .map(|index| scene(index * 30, 30))
            .collect::<Vec<_>>();
        let few_layout = resolve(&allocate(&few), 1_568, 16.0 / 9.0);
        let many_layout = resolve(&allocate(&many), 1_568, 16.0 / 9.0);
        assert!(few_layout.cell_height > many_layout.cell_height);
    }

    #[test]
    fn portrait_aspect_produces_narrower_cells_than_landscape() {
        let scenes = (0..8)
            .map(|index| scene(index * 60, 60))
            .collect::<Vec<_>>();
        let allocations = allocate(&scenes);
        let landscape = resolve(&allocations, 1_568, 16.0 / 9.0);
        let portrait = resolve(&allocations, 1_568, 9.0 / 16.0);
        assert!(portrait.cell_width < landscape.cell_width);
    }

    #[test]
    fn cell_height_at_least_sixty_pixels() {
        let scenes = (0..60)
            .map(|index| scene(index * 30, 30))
            .collect::<Vec<_>>();
        let layout = resolve(&allocate(&scenes), 1_568, 0.5);
        assert!(layout.cell_height >= 60);
    }

    #[test]
    fn total_height_grows_linearly_with_scene_count() {
        let small = (0..5)
            .map(|index| scene(index * 60, 60))
            .collect::<Vec<_>>();
        let large = (0..20)
            .map(|index| scene(index * 60, 60))
            .collect::<Vec<_>>();
        let small_layout = resolve(&allocate(&small), 1_568, 16.0 / 9.0);
        let large_layout = resolve(&allocate(&large), 1_568, 16.0 / 9.0);
        assert!(large_layout.total_height > small_layout.total_height);
    }

    #[test]
    fn image_format_extensions() {
        assert_eq!(ImageFormat::Png.file_extension(), "png");
        assert_eq!(ImageFormat::Jpeg { quality: 90 }.file_extension(), "jpg");
    }

    #[test]
    fn image_format_mime_types() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg { quality: 90 }.mime_type(), "image/jpeg");
    }
}
