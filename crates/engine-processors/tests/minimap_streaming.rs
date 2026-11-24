// The allocator is a test-only observer of real Rust allocations. Its pointer
// operations delegate unchanged to System; no production unsafe code is added.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, ProgressSink, VideoMinimapComposing, VideoScene, VideoSceneDetectionResult,
};
use ferment_engine_media::{
    Codec, ColorInfo, ContainerFormat, CpuFrame, CpuPixelFormat, DecodeSession,
    DecoderConfiguration, EncodeSession, EncoderSettings, FrameBuffer, MediaInfo, MediaKind,
    MediaTime, StreamInfo, TimeBase, VideoFrame, VideoIo,
};
use ferment_engine_processors::{
    VideoMinimapComposer,
    video_minimap_composer::{
        SamplingStrategy, Scene, VideoMinimapComposerConfig, allocate::allocate, layout::resolve,
    },
};

struct ObservedAllocator;
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());
fn added(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::SeqCst);
}
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let value = unsafe { System.alloc(layout) };
        if !value.is_null() {
            added(layout.size());
        }
        value
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let value = unsafe { System.alloc_zeroed(layout) };
        if !value.is_null() {
            added(layout.size());
        }
        value
    }
    unsafe fn dealloc(&self, value: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(value, layout) };
    }
    unsafe fn realloc(&self, value: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(value, layout, size) };
        if !result.is_null() {
            if size >= layout.size() {
                added(size - layout.size());
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - size, Ordering::SeqCst);
            }
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

#[derive(Debug)]
struct State {
    frames: usize,
    size: (u32, u32),
    declared_rate: Option<(u32, u32)>,
    origin_ticks: i64,
    matrix: Option<[i32; 9]>,
    corner_pattern: bool,
    coded_opens: AtomicUsize,
    reads: AtomicUsize,
    drops: AtomicUsize,
    progress: ProgressSink,
    cancel_read: usize,
    fail_read: usize,
}
impl State {
    fn new(frames: usize) -> Self {
        Self {
            frames,
            size: (512, 256),
            declared_rate: Some((30, 1)),
            origin_ticks: 0,
            matrix: None,
            corner_pattern: false,
            coded_opens: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            progress: ProgressSink::noop(),
            cancel_read: usize::MAX,
            fail_read: usize::MAX,
        }
    }
}
fn color(index: usize) -> [u8; 3] {
    [
        (index % 251) as u8,
        ((index * 3) % 251) as u8,
        ((index * 7) % 251) as u8,
    ]
}
#[derive(Debug)]
struct Decoder {
    state: Arc<State>,
    index: usize,
}
impl Drop for Decoder {
    fn drop(&mut self) {
        self.state.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[async_trait(?Send)]
impl DecodeSession for Decoder {
    async fn seek(&mut self, _: MediaTime) -> ferment_engine_media::Result<()> {
        panic!("the exact indexed pass must remain sequential")
    }
    async fn decode_next(&mut self) -> ferment_engine_media::Result<Option<VideoFrame>> {
        let read = self.state.reads.fetch_add(1, Ordering::SeqCst) + 1;
        if read == self.state.cancel_read {
            self.state.progress.cancel();
        }
        if read == self.state.fail_read {
            return Err(ferment_engine_media::MediaError::InvalidData(
                "controlled decode failure".into(),
            ));
        }
        if self.index == self.state.frames {
            return Ok(None);
        }
        let index = self.index;
        self.index += 1;
        let (width, height) = self.state.size;
        let mut bytes = vec![0; width as usize * height as usize * 3];
        for (offset, pixel) in bytes.chunks_exact_mut(3).enumerate() {
            let value = if self.state.corner_pattern {
                let x = offset % width as usize;
                let y = offset / width as usize;
                [[230, 20, 10], [10, 210, 30], [20, 30, 220], [220, 200, 10]]
                    [(y >= height as usize / 2) as usize * 2 + (x >= width as usize / 2) as usize]
            } else {
                color(index)
            };
            pixel.copy_from_slice(&value);
        }
        Ok(Some(VideoFrame {
            width,
            height,
            pts: MediaTime::new(
                self.state.origin_ticks + index as i64,
                TimeBase::new_unchecked(1, 30),
            ),
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(CpuFrame::new(
                width,
                height,
                width as usize * 3,
                CpuPixelFormat::Rgb24,
                bytes,
            )?),
        }))
    }
}
#[derive(Debug)]
struct Video(Arc<State>);
#[async_trait(?Send)]
impl VideoIo for Video {
    async fn probe(&self, _: &Path) -> ferment_engine_media::Result<MediaInfo> {
        let duration = Some(MediaTime::new(
            self.0.frames as i64,
            TimeBase::new_unchecked(1, 30),
        ));
        Ok(MediaInfo {
            container: ContainerFormat::Mp4,
            duration,
            streams: vec![StreamInfo {
                index: 0,
                kind: MediaKind::Video,
                codec: Codec::H264,
                time_base: TimeBase::new_unchecked(1, 30),
                start_time: Some(MediaTime::new(
                    self.0.origin_ticks,
                    TimeBase::new_unchecked(1, 30),
                )),
                duration,
                width: Some(self.0.size.0),
                height: Some(self.0.size.1),
                sample_rate: None,
                channels: None,
                frame_rate: self.0.declared_rate,
                display_matrix: self.0.matrix,
                bit_rate: None,
                language: None,
                color: ColorInfo::default(),
                audio_timing: None,
                decoder_configuration: DecoderConfiguration::from_codec_private(
                    Codec::H264,
                    vec![],
                )?,
            }],
        })
    }
    async fn open_decoder(
        &self,
        _: &Path,
        _: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        Ok(Box::new(Decoder {
            state: self.0.clone(),
            index: 0,
        }))
    }
    async fn open_coded_decoder(
        &self,
        path: &Path,
        index: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.0.coded_opens.fetch_add(1, Ordering::SeqCst);
        self.open_decoder(path, index).await
    }
    async fn open_encoder(
        &self,
        _: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        panic!("minimap extraction never encodes video")
    }
}
fn config() -> VideoMinimapComposerConfig {
    VideoMinimapComposerConfig {
        max_width: 640,
        cell_aspect: 2.0,
        sampling_strategy: SamplingStrategy::AllScenes,
        ..VideoMinimapComposerConfig::default()
    }
}
fn scenes(frames: usize) -> Vec<Scene> {
    let half = frames / 2;
    [0, half]
        .into_iter()
        .map(|start| Scene {
            start_frame: start,
            frame_count: half,
            start_time: start as f64 / 30.0,
            duration: half as f64 / 30.0,
        })
        .collect()
}
fn core_scenes(source: &[Scene]) -> VideoSceneDetectionResult {
    VideoSceneDetectionResult {
        total_frames: source.iter().map(|s| s.frame_count).sum(),
        frame_rate: 30.0,
        scenes: source
            .iter()
            .enumerate()
            .map(|(index, s)| VideoScene {
                index,
                start_frame: s.start_frame,
                frame_count: s.frame_count,
                start_time: s.start_time,
                duration: s.duration,
                start_confidence: Some(1.0),
                embedding: None,
            })
            .collect(),
    }
}

/// Kill: decode and retain the movie before selecting cells, or retain source-
/// sized cell clones. The allocator measures the public extraction's ACTUAL
/// live Rust heap, independent of OS compression and allocator RSS retention.
#[test]
fn long_extraction_heap_is_bounded_by_cells_and_current_frames() {
    let _serial = SERIAL.lock().unwrap();
    let state = Arc::new(State::new(300));
    let source = scenes(state.frames);
    let allocations = allocate(&source);
    let cfg = config();
    let layout = resolve(&allocations, cfg.max_width, cfg.cell_aspect);
    let cells: usize = allocations.iter().map(|row| row.cell_frames.len()).sum();
    let retained_ceiling =
        cells * (layout.cell_width * 2) as usize * (layout.cell_height * 2) as usize * 3;
    let frame_bytes = state.size.0 as usize * state.size.1 as usize * 3;
    let budget = retained_ceiling + 8 * frame_bytes + 2 * 1_048_576;
    let composer =
        VideoMinimapComposer::new(cfg).with_video_backend(Arc::new(Video(state.clone())));
    let baseline = LIVE_BYTES.load(Ordering::SeqCst);
    PEAK_BYTES.store(baseline, Ordering::SeqCst);
    let (keyframes, barcode) =
        futures_executor::block_on(composer.decode_inputs(Path::new("observed.mp4"), &source))
            .unwrap();
    let peak_growth = PEAK_BYTES.load(Ordering::SeqCst).saturating_sub(baseline);
    assert!(
        peak_growth < budget,
        "minimap retained the movie: measured{peak_growth} bytes, bounded-cell/current-frame ceiling{budget}"
    );
    assert_eq!(state.reads.load(Ordering::SeqCst), state.frames + 1);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert_eq!(keyframes.len(), allocations.len());
    for ((scene_index, frames), allocation) in keyframes.iter().zip(&allocations) {
        assert_eq!(*scene_index, allocation.scene_index);
        assert_eq!(frames.len(), allocation.cell_frames.len());
        for (frame, original_index) in frames.iter().zip(&allocation.cell_frames) {
            assert!(frame.width <= (layout.cell_width * 2) as usize);
            assert!(frame.height <= (layout.cell_height * 2) as usize);
            assert_eq!(frame.data.len(), frame.width * frame.height * 3);
            assert!(
                frame
                    .data
                    .chunks_exact(3)
                    .all(|pixel| pixel == color(*original_index)),
                "wrong original frame was retained"
            );
        }
    }
    assert_eq!(barcode.len(), 20); // 10 seconds at the current source barcode cadence, 2 fps.
    eprintln!(
        "MINIMAP_RUST_HEAP {{\"frames\":{},\"cells\":{},\"baseline_bytes\":{},\"peak_growth_bytes\":{},\"ceiling_bytes\":{}}}",
        state.frames, cells, baseline, peak_growth, budget
    );
}

/// Kill: deduplicate repeated cell requests, use the next scene's frame, alter
/// exact mean color, or stamp decoder PTS in place of centered requested time.
#[test]
fn duplicate_cells_and_sparse_requested_barcode_timing_survive_streaming() {
    let _serial = SERIAL.lock().unwrap();
    let mut value = State::new(32);
    value.size = (32, 18);
    value.declared_rate = None;
    value.origin_ticks = 45;
    let state = Arc::new(value);
    let source = vec![Scene {
        start_frame: 10,
        frame_count: 2,
        start_time: 1.0 / 3.0,
        duration: 25.0,
    }];
    let composer =
        VideoMinimapComposer::new(config()).with_video_backend(Arc::new(Video(state.clone())));
    let (keyframes, barcode) =
        futures_executor::block_on(composer.decode_inputs(Path::new("timed.mp4"), &source))
            .unwrap();
    assert_eq!(keyframes.len(), 1);
    assert_eq!(keyframes[0].1.len(), 5);
    for (frame, index) in keyframes[0].1.iter().zip([10, 10, 11, 11, 11]) {
        assert_eq!((frame.width, frame.height), (32, 18));
        assert!(
            frame
                .data
                .chunks_exact(3)
                .all(|pixel| pixel == color(index))
        );
    }
    // The deliberately long allocation still repeats both original cells;
    // sparse barcode requests outside ±0.5 s of this shifted source are absent.
    assert_eq!(barcode.len(), 4);
    for (entry, (time, index)) in barcode.iter().zip([
        (1.2416666666666667, 0),
        (1.7383333333333333, 7),
        (2.235, 22),
        (2.7316666666666665, 31),
    ]) {
        assert!((entry.start_time - time).abs() < 1e-9);
        assert_eq!([entry.mean_r, entry.mean_g, entry.mean_b], color(index));
    }
    for entries in barcode.windows(2) {
        assert_eq!(entries[0].end_time, entries[1].start_time);
    }
    assert!((barcode.last().unwrap().end_time - (25.0 + 1.0 / 3.0)).abs() < 1e-9);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: the public composing door discards its caller's cancellation sink
/// while decoding a long movie, retaining partial cells/reader until EOF.
#[test]
fn cancelling_public_composition_during_decode_releases_the_reader() {
    let _serial = SERIAL.lock().unwrap();
    let mut value = State::new(300);
    value.cancel_read = 7;
    let state = Arc::new(value);
    let composer =
        VideoMinimapComposer::new(config()).with_video_backend(Arc::new(Video(state.clone())));
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("cancel.mp4"),
        &core_scenes(&scenes(state.frames)),
        state.progress.clone(),
    ));
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(state.reads.load(Ordering::SeqCst), 7);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: failed decode leaves the reader or previously extracted frames alive
/// after the error crosses the actual path-based capability boundary.
#[test]
fn failed_public_decode_releases_the_reader_and_returns_the_original_error() {
    let _serial = SERIAL.lock().unwrap();
    let mut value = State::new(300);
    value.fail_read = 9;
    let state = Arc::new(value);
    let composer =
        VideoMinimapComposer::new(config()).with_video_backend(Arc::new(Video(state.clone())));
    let baseline = LIVE_BYTES.load(Ordering::SeqCst);
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("error.mp4"),
        &core_scenes(&scenes(state.frames)),
        state.progress.clone(),
    ));
    assert!(
        matches!(&result, Err(EngineError::Decode(message)) if message.contains("controlled decode failure"))
    );
    assert_eq!(state.reads.load(Ordering::SeqCst), 9);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    assert!(
        LIVE_BYTES.load(Ordering::SeqCst) < baseline + 1_048_576,
        "failed extraction retained partial frame storage"
    );
}

/// Kill: use relative ProgressTracker increments as absolute sink values,
/// regress at encoding, or return a public JNI-ready minimap without completion.
#[test]
fn successful_public_composition_reports_monotone_terminal_progress() {
    let _serial = SERIAL.lock().unwrap();
    let state = Arc::new(State::new(72));
    let events = Arc::new(Mutex::new(Vec::new()));
    let observed = events.clone();
    let progress = ProgressSink::from_callback(move |event| observed.lock().unwrap().push(event));
    let composer =
        VideoMinimapComposer::new(config()).with_video_backend(Arc::new(Video(state.clone())));
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("complete.mp4"),
        &core_scenes(&scenes(state.frames)),
        progress,
    ))
    .unwrap();
    assert!(!result.image_data.is_empty());
    let events = events.lock().unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.stage.starts_with("decoding minimap frame "))
    );
    assert!(
        events
            .iter()
            .all(|event| (0.0..=1.0).contains(&event.progress))
    );
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].progress <= pair[1].progress),
        "progress regressed: {events:?}"
    );
    assert_eq!(
        events
            .iter()
            .find(|event| event.stage == "rendering")
            .unwrap()
            .progress,
        0.85
    );
    assert_eq!(
        events
            .iter()
            .find(|event| event.stage == "encoding")
            .unwrap()
            .progress,
        0.95
    );
    assert_eq!(events.last().unwrap().progress, 1.0);
    assert_eq!(events.last().unwrap().stage, "done");
    assert_eq!(
        events.iter().filter(|event| event.progress == 1.0).count(),
        1
    );
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: fixed landscape geometry on the actual file capability, or a sidecar
/// that reports absent/misaligned panels despite a correctly rendered image.
#[test]
fn public_portrait_minimap_preserves_display_aspect_and_real_panel_bounds() {
    let _serial = SERIAL.lock().unwrap();
    let mut value = State::new(72);
    value.size = (128, 256);
    value.corner_pattern = true;
    let state = Arc::new(value);
    let mut cfg = config();
    cfg.format = ferment_engine_processors::video_minimap_composer::ImageFormat::Png;
    let composer =
        VideoMinimapComposer::new(cfg).with_video_backend(Arc::new(Video(state.clone())));
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("portrait.mp4"),
        &core_scenes(&scenes(state.frames)),
        ProgressSink::noop(),
    ))
    .unwrap();
    let image = image::load_from_memory(&result.image_data)
        .unwrap()
        .to_rgb8();
    assert_eq!(image.dimensions(), (result.width, result.height));
    assert_eq!(result.layout.cells.len(), 4);
    let first = result.layout.cells[0].rect;
    assert!(first.height > first.width);
    assert!((first.width / first.height - 0.5).abs() < 0.01);
    assert_eq!(
        first.y, 50.0,
        "row must start immediately below44px header plus6px padding"
    );
    let header = result.layout.header_rect.unwrap();
    let barcode = result.layout.barcode_rect.unwrap();
    let axis = result.layout.time_axis_rect.unwrap();
    assert_eq!(
        (header.x, header.y, header.width, header.height),
        (0.0, 0.0, result.width as f64, 44.0)
    );
    assert_eq!(
        (barcode.x, barcode.width, barcode.height),
        (12.0, result.width as f64 - 24.0, 72.0)
    );
    assert_eq!(
        (axis.x, axis.y, axis.width, axis.height),
        (barcode.x, barcode.y + barcode.height, barcode.width, 22.0)
    );
    assert!(axis.y + axis.height <= result.height as f64);
    for cell in &result.layout.cells {
        let r = cell.rect;
        assert!(
            r.x >= 0.0
                && r.y >= header.height
                && r.x + r.width <= result.width as f64
                && r.y + r.height < barcode.y
        );
    }
    let leading = image.get_pixel(20, (barcode.y + 20.0) as u32).0;
    assert_eq!(
        leading,
        [0, 0, 0],
        "midpoint-sampled barcode leading gap must be black across the old gutter area"
    );
    assert_corner_pixels(
        &image,
        first,
        [[230, 20, 10], [10, 210, 30], [20, 30, 220], [220, 200, 10]],
    );
    assert_eq!(state.coded_opens.load(Ordering::SeqCst), 1);
    assert_eq!(state.reads.load(Ordering::SeqCst), state.frames + 1);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

fn assert_corner_pixels(
    image: &image::RgbImage,
    rect: ferment_engine_core::Rect,
    expected: [[u8; 3]; 4],
) {
    for ((xf, yf), color) in [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        .into_iter()
        .zip(expected)
    {
        let pixel = image
            .get_pixel(
                (rect.x + rect.width * xf) as u32,
                (rect.y + rect.height * yf) as u32,
            )
            .0;
        assert_eq!(pixel, color, "display corner({xf},{yf})");
    }
}
fn transformed_public(matrix: [i32; 9], size: (u32, u32), expected: [[u8; 3]; 4], aspect: f64) {
    let mut value = State::new(12);
    value.size = size;
    value.matrix = Some(matrix);
    value.corner_pattern = true;
    let state = Arc::new(value);
    let mut cfg = config();
    cfg.format = ferment_engine_processors::video_minimap_composer::ImageFormat::Png;
    let composer =
        VideoMinimapComposer::new(cfg).with_video_backend(Arc::new(Video(state.clone())));
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("display-matrix.mp4"),
        &core_scenes(&scenes(state.frames)),
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(
        state.coded_opens.load(Ordering::SeqCst),
        1,
        "must use explicit coded policy"
    );
    assert_eq!(state.reads.load(Ordering::SeqCst), 13);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    let image = image::load_from_memory(&result.image_data)
        .unwrap()
        .to_rgb8();
    let rect = result.layout.cells[0].rect;
    assert!((rect.width / rect.height - aspect).abs() < 0.01);
    assert_corner_pixels(&image, rect, expected);
    assert_eq!(
        result
            .layout
            .cells
            .iter()
            .map(|cell| cell.frame_index)
            .collect::<Vec<_>>(),
        vec![0, 6]
    );
    assert_eq!(
        result.layout.cells[1].time, 0.2,
        "display does not alter frame clock"
    );
}
/// Kill: omit the matrix, rotate the already coded-policy output twice, or
/// resolve portrait layout from coded dimensions instead of preferred display.
#[test]
fn coded_landscape_quarter_turn_reaches_the_public_portrait_raster_once() {
    let _serial = SERIAL.lock().unwrap();
    transformed_public(
        [0, 65536, 0, -65536, 0, 0, 32 * 65536, 0, 1073741824],
        (64, 32),
        [[20, 30, 220], [230, 20, 10], [220, 200, 10], [10, 210, 30]],
        0.5,
    );
}
/// Kill: infer transform application from dimensions;180° keeps square size.
#[test]
fn square_half_turn_preserves_the_actual_opposite_pixel_corners() {
    let _serial = SERIAL.lock().unwrap();
    transformed_public(
        [
            -65536,
            0,
            0,
            0,
            -65536,
            0,
            48 * 65536,
            48 * 65536,
            1073741824,
        ],
        (48, 48),
        [[220, 200, 10], [20, 30, 220], [10, 210, 30], [230, 20, 10]],
        1.0,
    );
}
/// Kill: drop reflection because the matrix is not an ordinary quarter-turn.
#[test]
fn preferred_horizontal_reflection_is_applied_to_pixels_once() {
    let _serial = SERIAL.lock().unwrap();
    transformed_public(
        [-65536, 0, 0, 0, 65536, 0, 64 * 65536, 0, 1073741824],
        (64, 32),
        [[10, 210, 30], [230, 20, 10], [220, 200, 10], [20, 30, 220]],
        2.0,
    );
}
/// Kill: use rounded/partial scene endpoints or the decoder's30fps in the
/// barcode/header/cell clocks while publishing the source-result24fps sidecar.
#[test]
fn public_scene_result_clock_controls_cells_and_sparse_barcode_domain() {
    let _serial = SERIAL.lock().unwrap();
    let state = Arc::new(State::new(72));
    let mut cfg = config();
    cfg.format = ferment_engine_processors::video_minimap_composer::ImageFormat::Png;
    let composer =
        VideoMinimapComposer::new(cfg).with_video_backend(Arc::new(Video(state.clone())));
    let mut source = core_scenes(&scenes(72));
    source.frame_rate = 24.0;
    source.scenes[0].start_time = 0.125;
    source.scenes[0].duration = 1.7;
    source.scenes[1].start_time = 1.825;
    source.scenes[1].duration = 1.6;
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("clock.mp4"),
        &source,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(
        result.layout.time_range,
        ferment_engine_core::TimeRange::new(0.0, 3.0)
    );
    assert_eq!(
        result
            .layout
            .cells
            .iter()
            .map(|cell| cell.frame_index)
            .collect::<Vec<_>>(),
        vec![0, 35, 36, 71]
    );
    for cell in &result.layout.cells {
        assert_eq!(cell.time, cell.frame_index as f64 / 24.0);
    }
    let image = image::load_from_memory(&result.image_data)
        .unwrap()
        .to_rgb8();
    let r = result.layout.barcode_rect.unwrap();
    // Six samples across3.0s place the first requested-time stripe at1/12 of
    // the plot. Using max(endpoints)=3.425 instead incorrectly paints at1/14.
    let x = (r.x + r.width * 0.078) as u32;
    let y = (r.y + 20.0) as u32;
    assert_eq!(image.get_pixel(x, y).0, [0, 0, 0]);
    let visible = image.get_pixel((r.x + r.width * 0.12) as u32, y).0;
    assert_ne!(
        visible,
        [0, 0, 0],
        "correct first source-clock stripe must be present"
    );
}

/// Kill: bound only thumbnail storage or encoded output while an invertible
/// display scale makes the raw raster enormous. Both entrypoints must refuse
/// the extent before retaining frames/rendering; normal portrait cases remain.
#[test]
fn pathological_affine_extent_is_rejected_before_frame_or_raster_retention() {
    let _serial = SERIAL.lock().unwrap();
    let mut value = State::new(12);
    value.size = (128, 64);
    value.matrix = Some([1, 0, 0, 0, 65536, 0, 0, 0, 1073741824]);
    let state = Arc::new(value);
    let composer =
        VideoMinimapComposer::new(config()).with_video_backend(Arc::new(Video(state.clone())));
    let source = scenes(state.frames);
    let result = futures_executor::block_on(VideoMinimapComposing::compose(
        &composer,
        Path::new("extreme-affine.mp4"),
        &core_scenes(&source),
        ProgressSink::noop(),
    ));
    assert!(
        matches!(result, Err(EngineError::InvalidConfiguration(ref message))
        if message.contains("raster extent")),
        "{result:?}"
    );
    assert_eq!(state.coded_opens.load(Ordering::SeqCst), 1);
    assert_eq!(
        state.reads.load(Ordering::SeqCst),
        1,
        "must reject after the first source picture"
    );
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    let mut cfg = config();
    cfg.cell_aspect = 1.0 / 32768.0;
    let composer = VideoMinimapComposer::new(cfg);
    let result =
        futures_executor::block_on(composer.compose(&source, &[], &[], ProgressSink::noop()));
    assert!(
        matches!(result, Err(EngineError::InvalidConfiguration(ref message))
        if message.contains("raster extent")),
        "{result:?}"
    );
}
