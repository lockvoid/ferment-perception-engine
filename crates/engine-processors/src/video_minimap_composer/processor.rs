//! Top-level orchestrator. Takes pre-extracted RGB keyframes (per scene)
//! and a barcode pool (mean RGB samples over time), renders a minimap.
//!
//! Path-based callers extract through one sequential platform reader; only
//! selected, cell-sized thumbnails and scalar barcode colors are retained.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_media::VideoBackend;
use serde::{Deserialize, Serialize};

use super::allocate::{Scene, SceneAllocation, allocate_indexed};
use super::layout::{FALLBACK_CELL_ASPECT, LayoutGeometry, resolve};
use super::renderer::{RenderInput, render};
use super::sampler::{SamplingStrategy, sample_indices};
use super::transform::DisplayTransform;
use crate::video_source::{RgbFrame, VideoSource};

const MAX_OUTPUT_BYTES: usize = 5 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg { quality: u8 },
}

impl ImageFormat {
    #[must_use]
    pub const fn file_extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpg",
        }
    }

    #[must_use]
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg { .. } => "image/jpeg",
        }
    }
}

/// One thumbnail buffer: row-major RGB8, `width × height × 3` bytes.
#[derive(Debug, Clone)]
pub struct KeyframeBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// One sparse barcode sample: requested start time and the next requested
/// time (or movie end), with mean RGB from an aspect-fitted ≤64×64 image.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PoolEntry {
    pub start_time: f64,
    pub end_time: f64,
    pub mean_r: u8,
    pub mean_g: u8,
    pub mean_b: u8,
}

/// Output cell geometry — useful for vision LLMs to map regions back to
/// scene/frame indices.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub scene_index: usize,
    pub frame_index: usize,
    pub time: f64,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct VideoMinimapComposerConfig {
    pub max_width: i32,
    pub format: ImageFormat,
    pub cell_aspect: f32,
    pub max_scenes_shown: usize,
    pub sampling_strategy: SamplingStrategy,
}

impl Default for VideoMinimapComposerConfig {
    fn default() -> Self {
        Self {
            max_width: 1568,
            format: ImageFormat::Jpeg { quality: 90 },
            cell_aspect: FALLBACK_CELL_ASPECT,
            max_scenes_shown: 25,
            sampling_strategy: SamplingStrategy::StratifiedLongestPerBin,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoMinimapResult {
    pub image_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub layout: LayoutGeometry,
    pub cells: Vec<Cell>,
}

#[derive(Clone, Copy)]
struct Viewport {
    width: f64,
    height: f64,
    frame_rate: f64,
    duration: f64,
}
#[derive(Clone, Copy)]
struct SourceClock {
    duration: f64,
    frame_rate: f64,
}
impl Viewport {
    fn aspect(self) -> f32 {
        if self.width > 0.0 && self.height > 0.0 {
            (self.width / self.height) as f32
        } else {
            FALLBACK_CELL_ASPECT
        }
    }
}
struct DecodedInputs {
    keyframes: Vec<(usize, Vec<KeyframeBuffer>)>,
    barcode: Vec<PoolEntry>,
    viewport: Viewport,
}

pub struct VideoMinimapComposer {
    config: VideoMinimapComposerConfig,
    video: VideoSource,
}

impl VideoMinimapComposer {
    pub fn new(config: VideoMinimapComposerConfig) -> Self {
        Self {
            config,
            video: VideoSource::default(),
        }
    }

    #[must_use]
    pub fn with_video_backend(mut self, backend: VideoBackend) -> Self {
        self.video = VideoSource::new(backend);
        self
    }

    pub fn with_defaults() -> Self {
        Self::new(VideoMinimapComposerConfig::default())
    }

    fn sampled_allocations(&self, scenes: &[Scene]) -> Result<Vec<SceneAllocation>> {
        let indices = sample_indices(
            scenes,
            self.config.max_scenes_shown,
            self.config.sampling_strategy,
        );
        if self.config.sampling_strategy == SamplingStrategy::AllScenes
            && indices.len() > self.config.max_scenes_shown
        {
            return Err(EngineError::invalid_config(format!(
                "video has {} scenes; minimap supports up to {} with all_scenes",
                indices.len(),
                self.config.max_scenes_shown
            )));
        }
        if indices.is_empty() {
            return Err(EngineError::invalid_config(
                "scene sampler returned an empty selection",
            ));
        }
        let selected = indices
            .iter()
            .map(|index| scenes[*index])
            .collect::<Vec<_>>();
        Ok(allocate_indexed(&selected, Some(&indices)))
    }

    pub async fn decode_inputs(
        &self,
        video_path: &Path,
        scenes: &[Scene],
    ) -> Result<(Vec<(usize, Vec<KeyframeBuffer>)>, Vec<PoolEntry>)> {
        self.decode_inputs_with_progress(video_path, scenes, &ProgressSink::noop())
            .await
    }

    /// Retain one thumbnail per selected cell, never the decoded movie. The
    /// Swift extractor caps its keyframe images at twice the resolved cell
    /// size; the sequential reader preserves exact frame-index selection.
    pub async fn decode_inputs_with_progress(
        &self,
        video_path: &Path,
        scenes: &[Scene],
        progress: &ProgressSink,
    ) -> Result<(Vec<(usize, Vec<KeyframeBuffer>)>, Vec<PoolEntry>)> {
        let decoded = self
            .decode_resolved(video_path, scenes, progress, None)
            .await?;
        Ok((decoded.keyframes, decoded.barcode))
    }

    async fn decode_resolved(
        &self,
        video_path: &Path,
        scenes: &[Scene],
        progress: &ProgressSink,
        source_clock: Option<SourceClock>,
    ) -> Result<DecodedInputs> {
        let allocations = self.sampled_allocations(scenes)?;
        let mut reader = self.video.open_minimap_rgb(video_path, progress).await?;
        let first = reader
            .next(progress)
            .await?
            .ok_or_else(|| EngineError::Decode("video decoder produced no frames".into()))?;
        let transform = DisplayTransform::from_matrix(reader.display_matrix())?;
        let (display_width, display_height) = transform.display_size(first.width, first.height);
        // Match CGSize.applying(preferredTransform) for source cell aspect;
        // selected picture extraction separately contains the affine image.
        let mut viewport = Viewport {
            width: display_width,
            height: display_height,
            frame_rate: source_clock.map_or(reader.frame_rate(), |clock| clock.frame_rate),
            duration: source_clock.map_or_else(
                || {
                    scenes
                        .iter()
                        .map(|scene| scene.start_time + scene.duration)
                        .fold(0.0_f64, f64::max)
                },
                |clock| clock.duration,
            ),
        };
        let layout = resolve(&allocations, self.config.max_width, viewport.aspect());
        layout.validate_raster_extent()?;
        let maximum_size = (
            layout.cell_width.max(1) as u32 * 2,
            layout.cell_height.max(1) as u32 * 2,
        );
        let mut destinations: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        let mut selected: Vec<Vec<Option<KeyframeBuffer>>> = allocations
            .iter()
            .enumerate()
            .map(|(row, allocation)| {
                for (cell, frame) in allocation.cell_frames.iter().enumerate() {
                    destinations.entry(*frame).or_default().push((row, cell));
                }
                vec![None; allocation.cell_frames.len()]
            })
            .collect();
        let mut barcode = BarcodeSampler::with_transform(viewport.duration, transform);
        let mut index = 0usize;
        let mut next = Some(first);
        while let Some(frame) = next {
            if let Some(cells) = destinations.remove(&index) {
                let thumbnail = transform.thumbnail(&frame, maximum_size)?;
                for (row, cell) in cells {
                    selected[row][cell] = Some(thumbnail.clone());
                }
            }
            // The sampler owns at most the previous frame while it chooses
            // the closest presentation around each sparse requested time.
            barcode.push(frame)?;
            index += 1;
            if index.is_multiple_of(64) {
                let fraction = reader
                    .expected_frame_count()
                    .map_or(0.0, |count| (index as f64 / count as f64).min(1.0));
                progress
                    .report(
                        0.05 + fraction * 0.50,
                        format!("decoding minimap frame {index}"),
                    )
                    .await;
            }
            next = reader.next(progress).await?;
        }
        viewport.frame_rate = source_clock.map_or(reader.frame_rate(), |clock| clock.frame_rate);
        if index == 0 {
            return Err(EngineError::Decode(
                "video decoder produced no frames".into(),
            ));
        }
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        let barcode = barcode.finish()?;
        let keyframes = allocations
            .into_iter()
            .zip(selected)
            .map(|(allocation, frames)| {
                (
                    allocation.scene_index,
                    frames.into_iter().flatten().collect(),
                )
            })
            .collect();
        Ok(DecodedInputs {
            keyframes,
            barcode,
            viewport,
        })
    }

    /// The path-based public capability uses the decoded display viewport for
    /// both thumbnail bounds and the final image/sidecar geometry.
    pub async fn compose_file(
        &self,
        video_path: &Path,
        scenes: &[Scene],
        total_frames: usize,
        frame_rate: f64,
        progress: ProgressSink,
    ) -> Result<VideoMinimapResult> {
        if !frame_rate.is_finite() || frame_rate <= 0.0 {
            return Err(EngineError::invalid_config("frame rate must be positive"));
        }
        let clock = SourceClock {
            duration: total_frames as f64 / frame_rate,
            frame_rate,
        };
        let decoded = self
            .decode_resolved(video_path, scenes, &progress, Some(clock))
            .await?;
        self.compose_resolved(
            scenes,
            &decoded.keyframes,
            &decoded.barcode,
            progress,
            Some(decoded.viewport),
        )
        .await
    }

    /// Render a video minimap.
    ///
    /// Caller pre-decoded:
    /// - `scenes`: scene boundaries (from VideoSceneDetector)
    /// - `keyframes_per_scene[i]`: RGB keyframe buffers for scene i,
    ///   one per allocated cell (allocate count comes from `allocate::cell_count`)
    /// - `barcode_pool`: mean RGB samples evenly spaced through the clip
    pub async fn compose(
        &self,
        scenes: &[Scene],
        keyframes_per_scene: &[(usize, Vec<KeyframeBuffer>)],
        barcode_pool: &[PoolEntry],
        progress: ProgressSink,
    ) -> Result<VideoMinimapResult> {
        self.compose_resolved(scenes, keyframes_per_scene, barcode_pool, progress, None)
            .await
    }

    async fn compose_resolved(
        &self,
        scenes: &[Scene],
        keyframes_per_scene: &[(usize, Vec<KeyframeBuffer>)],
        barcode_pool: &[PoolEntry],
        progress: ProgressSink,
        viewport: Option<Viewport>,
    ) -> Result<VideoMinimapResult> {
        if scenes.is_empty() {
            return Err(EngineError::invalid_config("scenes must not be empty"));
        }

        progress.report(0.55, "allocating cells").await;
        let mut allocations = self.sampled_allocations(scenes)?;
        if let Some(view) = viewport {
            for row in &mut allocations {
                row.cell_times = row
                    .cell_frames
                    .iter()
                    .map(|frame| *frame as f64 / view.frame_rate)
                    .collect();
            }
        }

        progress.report(0.55, "resolving layout").await;
        let layout = resolve(
            &allocations,
            self.config.max_width,
            viewport.map_or(self.config.cell_aspect, Viewport::aspect),
        );
        layout.validate_raster_extent()?;
        let duration = scenes
            .iter()
            .map(|scene| scene.start_time + scene.duration)
            .fold(0.0_f64, f64::max);
        let boundary_times = scenes
            .iter()
            .skip(1)
            .map(|scene| scene.start_time)
            .collect::<Vec<_>>();
        let fallback_rate = scenes
            .iter()
            .find(|scene| scene.duration > 0.0)
            .map_or(0.0, |scene| scene.frame_count as f64 / scene.duration);
        let view = viewport.unwrap_or(Viewport {
            width: f64::from(self.config.cell_aspect) * 1000.0,
            height: 1000.0,
            frame_rate: fallback_rate,
            duration,
        });
        let duration = view.duration;

        progress.report(0.85, "rendering").await;
        let rendered = render(&RenderInput {
            allocations: &allocations,
            keyframes: keyframes_per_scene,
            barcode_pool,
            layout,
            duration,
            boundary_times: &boundary_times,
            total_scene_count: scenes.len(),
            video_size: (view.width, view.height),
            frame_rate: view.frame_rate,
        });

        progress.report(0.95, "encoding").await;
        let image_data = encode(&rendered.pixmap, self.config.format)?;
        if image_data.len() > MAX_OUTPUT_BYTES {
            return Err(EngineError::invalid_config(format!(
                "video minimap is {} KB, exceeds {} KB limit",
                image_data.len() / 1024,
                MAX_OUTPUT_BYTES / 1024
            )));
        }

        // Build cell geometry for the response sidecar.
        let mut cells = Vec::new();
        let row_y_start = layout.header_height;
        for (row_idx, alloc) in allocations.iter().enumerate() {
            let row_y = row_y_start + row_idx as i32 * layout.row_height();
            let cells_x_start = layout.outer_padding + layout.label_gutter_width;
            let cell_y = row_y + layout.row_vertical_padding;
            for (cell_idx, &frame_index) in alloc.cell_frames.iter().enumerate() {
                let cell_x =
                    cells_x_start + cell_idx as i32 * (layout.cell_width + layout.cell_gap);
                cells.push(Cell {
                    scene_index: alloc.scene_index,
                    frame_index,
                    time: alloc.cell_times[cell_idx],
                    x: cell_x,
                    y: cell_y,
                    width: layout.cell_width,
                    height: layout.cell_height,
                });
            }
        }

        progress.report(1.0, "done").await;
        Ok(VideoMinimapResult {
            image_data,
            width: layout.total_width as u32,
            height: layout.total_height as u32,
            format: self.config.format,
            layout: LayoutGeometry {
                header_rect: layout.header_rect(),
                barcode_rect: layout.barcode_rect(),
                time_axis_rect: layout.time_axis_rect(),
                width: layout.total_width,
                height: layout.total_height,
                cell_width: layout.cell_width,
                cell_height: layout.cell_height,
                max_cells_per_row: layout.max_cells_per_row,
                barcode_height: layout.barcode_height,
                header_height: layout.header_height,
            },
            cells,
        })
    }
}

/// Current Swift uses centered 2 fps requests at CMTime timescale 600,
/// 64×64 maximum pictures, and ±0.5 s decode tolerance. A sequential pass is
/// already needed for exact cell indices; sharing that pass avoids a second
/// reader and retains at most one preceding full frame, never the movie.
struct BarcodeSampler {
    transform: DisplayTransform,
    duration: f64,
    times: Vec<f64>,
    next: usize,
    previous: Option<RgbFrame>,
    entries: Vec<PoolEntry>,
}

impl BarcodeSampler {
    #[cfg(test)]
    fn new(duration: f64) -> Self {
        Self::with_transform(duration, DisplayTransform::identity())
    }
    fn with_transform(duration: f64, transform: DisplayTransform) -> Self {
        let count = if duration.is_finite() && duration > 0.0 {
            (duration * 2.0).round().max(1.0) as usize
        } else {
            0
        };
        let times = (0..count)
            .map(|index| {
                let seconds = (index as f64 + 0.5) / count as f64 * duration;
                (seconds * 600.0).round() / 600.0
            })
            .collect();
        Self {
            transform,
            duration,
            times,
            next: 0,
            previous: None,
            entries: Vec::with_capacity(count),
        }
    }

    fn push(&mut self, frame: RgbFrame) -> Result<()> {
        while let Some(&time) = self
            .times
            .get(self.next)
            .filter(|time| **time <= frame.presentation_time)
        {
            let selected = self
                .previous
                .as_ref()
                .filter(|previous| {
                    (previous.presentation_time - time).abs()
                        <= (frame.presentation_time - time).abs()
                })
                .unwrap_or(&frame);
            if let Some(entry) = self.sample_transformed(selected, time)? {
                self.entries.push(entry);
            }
            self.next += 1;
        }
        self.previous = Some(frame);
        Ok(())
    }

    #[cfg(test)]
    fn sample(frame: &RgbFrame, time: f64) -> Result<Option<PoolEntry>> {
        Self::new(1.0).sample_transformed(frame, time)
    }
    fn sample_transformed(&self, frame: &RgbFrame, time: f64) -> Result<Option<PoolEntry>> {
        // A missing image beyond the source's ±0.5 s tolerance is omitted,
        // as AVAssetImageGenerator failures are in the source extractor.
        if (frame.presentation_time - time).abs() > 0.5 + f64::EPSILON {
            return Ok(None);
        }
        let picture = self.transform.thumbnail(frame, (64, 64))?;
        let (red, green, blue) =
            picture
                .data
                .chunks_exact(3)
                .fold((0u64, 0u64, 0u64), |(r, g, b), pixel| {
                    (
                        r + u64::from(pixel[0]),
                        g + u64::from(pixel[1]),
                        b + u64::from(pixel[2]),
                    )
                });
        let divisor = (picture.width * picture.height).max(1) as u64;
        Ok(Some(PoolEntry {
            start_time: time,
            end_time: time,
            mean_r: (red / divisor) as u8,
            mean_g: (green / divisor) as u8,
            mean_b: (blue / divisor) as u8,
        }))
    }

    fn finish(mut self) -> Result<Vec<PoolEntry>> {
        if let Some(previous) = self.previous.as_ref() {
            for &time in &self.times[self.next..] {
                if let Some(entry) = self.sample_transformed(previous, time)? {
                    self.entries.push(entry);
                }
            }
        }
        for index in 0..self.entries.len() {
            self.entries[index].end_time = self
                .entries
                .get(index + 1)
                .map_or(self.duration, |entry| entry.start_time);
        }
        Ok(self.entries)
    }
}

#[cfg(test)]
fn thumbnail(frame: &RgbFrame, maximum: (u32, u32)) -> Result<KeyframeBuffer> {
    DisplayTransform::identity().thumbnail(frame, maximum)
}

fn encode(pixmap: &tiny_skia::Pixmap, format: ImageFormat) -> Result<Vec<u8>> {
    let w = pixmap.width();
    let h = pixmap.height();
    let rgba: Vec<u8> = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let c = p.demultiply();
            [c.red(), c.green(), c.blue(), c.alpha()]
        })
        .collect();
    let img = image::RgbaImage::from_raw(w, h, rgba)
        .ok_or_else(|| EngineError::internal("rgba buffer size mismatch"))?;

    let mut out = Cursor::new(Vec::new());
    match format {
        ImageFormat::Png => img
            .write_to(&mut out, image::ImageFormat::Png)
            .map_err(|e| EngineError::internal(format!("png encode: {e}")))?,
        ImageFormat::Jpeg { quality } => {
            let rgb = image::DynamicImage::ImageRgba8(img).into_rgb8();
            let mut encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.min(100));
            encoder
                .encode(&rgb, w, h, image::ExtendedColorType::Rgb8)
                .map_err(|e| EngineError::internal(format!("jpeg encode: {e}")))?;
        }
    }
    Ok(out.into_inner())
}

#[cfg(test)]
mod barcode_source_tests {
    use super::*;

    /// Kill: per-frame barcode cadence or unquantized/left-edge requests.
    #[test]
    fn centered_two_fps_requests_use_the_source_cmtime_clock() {
        let sampler = BarcodeSampler::new(1.025);
        assert_eq!(sampler.times, vec![154.0 / 600.0, 461.0 / 600.0]);
        assert_eq!(BarcodeSampler::new(0.1).times, vec![30.0 / 600.0]);
        assert!(BarcodeSampler::new(0.0).times.is_empty());
    }

    /// Kill: keep original-sized barcode pictures or average only one channel.
    #[test]
    fn barcode_pictures_fit_sixty_four_pixels_before_exact_integer_mean() {
        let frame = RgbFrame {
            width: 512,
            height: 256,
            presentation_time: 0.25,
            data: [7, 11, 13].repeat(512 * 256),
        };
        let small = thumbnail(&frame, (64, 64)).unwrap();
        assert_eq!(
            (small.width, small.height, small.data.len()),
            (64, 32, 64 * 32 * 3)
        );
        let entry = BarcodeSampler::sample(&frame, 0.25166666666666665)
            .unwrap()
            .unwrap();
        assert_eq!([entry.mean_r, entry.mean_g, entry.mean_b], [7, 11, 13]);
        assert_eq!(entry.start_time, 151.0 / 600.0);
        assert!(
            BarcodeSampler::sample(&frame, 0.7516666666666667)
                .unwrap()
                .is_none()
        );
    }
}
