use super::*;
use ferment_engine_infer::InferenceBackend;
use ferment_engine_media::{
    Codec, ColorInfo, ContainerFormat, CpuFrame, CpuPixelFormat, DecodeOptions, DecodeSession,
    DecoderConfiguration, EncodeSession, EncoderSettings, FrameBuffer, MediaInfo, MediaKind,
    MediaTime, StreamInfo, TimeBase, VideoFrame, VideoIo,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

fn frame_bytes(index: usize) -> Vec<u8> {
    vec![(index % 251) as u8; PIXELS_PER_FRAME]
}

#[derive(Debug)]
struct State {
    progress: ProgressSink,
    frames: usize,
    source_size: (u32, u32),
    metadata_size: bool,
    declared_rate: Option<(u32, u32)>,
    pts_rate: i64,
    honor_scaling: bool,
    decoded_size: Option<(u32, u32)>,
    requested_sizes: Mutex<Vec<Option<(u32, u32)>>>,
    probes: AtomicUsize,
    opens: AtomicUsize,
    reads: AtomicUsize,
    drops: AtomicUsize,
    models: AtomicUsize,
    cancel_read: usize,
    cancel_model: usize,
    fail_read: usize,
    allow_identity: bool,
}
impl State {
    fn new(frames: usize) -> Self {
        Self {
            progress: ProgressSink::noop(),
            frames,
            source_size: (FRAME_WIDTH as u32, FRAME_HEIGHT as u32),
            metadata_size: true,
            declared_rate: Some((30, 1)),
            pts_rate: 30,
            honor_scaling: false,
            decoded_size: None,
            requested_sizes: Mutex::new(Vec::new()),
            probes: AtomicUsize::new(0),
            opens: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            models: AtomicUsize::new(0),
            cancel_read: usize::MAX,
            cancel_model: usize::MAX,
            fail_read: usize::MAX,
            allow_identity: false,
        }
    }
}
#[derive(Debug)]
struct Decoder {
    state: Arc<State>,
    next: usize,
    size: (u32, u32),
}
impl Drop for Decoder {
    fn drop(&mut self) {
        self.state.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[async_trait(?Send)]
impl DecodeSession for Decoder {
    async fn seek(&mut self, _: MediaTime) -> ferment_engine_media::Result<()> {
        panic!("a streaming pass must never seek")
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
        if self.next == self.state.frames {
            return Ok(None);
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(VideoFrame {
            width: self.size.0,
            height: self.size.1,
            pts: MediaTime::new(
                index as i64,
                TimeBase::new_unchecked(1, self.state.pts_rate),
            ),
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(CpuFrame::new(
                self.size.0,
                self.size.1,
                self.size.0 as usize * 3,
                CpuPixelFormat::Rgb24,
                vec![(index % 251) as u8; self.size.0 as usize * self.size.1 as usize * 3],
            )?),
        }))
    }
}
#[derive(Debug)]
struct Video(Arc<State>);
#[async_trait(?Send)]
impl VideoIo for Video {
    async fn probe(&self, _: &Path) -> ferment_engine_media::Result<MediaInfo> {
        self.0.probes.fetch_add(1, Ordering::SeqCst);
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
                start_time: Some(MediaTime::ZERO),
                duration,
                width: self.0.metadata_size.then_some(self.0.source_size.0),
                height: self.0.metadata_size.then_some(self.0.source_size.1),
                sample_rate: None,
                channels: None,
                frame_rate: self.0.declared_rate,
                display_matrix: None,
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
        self.0.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Decoder {
            state: self.0.clone(),
            next: 0,
            size: self.0.decoded_size.unwrap_or(self.0.source_size),
        }))
    }
    async fn open_decoder_with(
        &self,
        _: &Path,
        _: u32,
        options: DecodeOptions,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.0
            .requested_sizes
            .lock()
            .unwrap()
            .push(options.output_size);
        self.0.opens.fetch_add(1, Ordering::SeqCst);
        let size = self.0.decoded_size.unwrap_or_else(|| {
            if self.0.honor_scaling {
                options.output_size.unwrap_or(self.0.source_size)
            } else {
                self.0.source_size
            }
        });
        Ok(Box::new(Decoder {
            state: self.0.clone(),
            next: 0,
            size,
        }))
    }
    async fn open_encoder(
        &self,
        _: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        panic!("scene detection cannot encode")
    }
}
#[derive(Debug)]
struct Models(Arc<State>);
#[async_trait]
impl InferenceBackend for Models {
    async fn infer(
        &self,
        model: ModelId,
        _: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        assert!(
            model == ModelId::Autoshot || self.0.allow_identity,
            "model-only diagnostics must not enter DINO"
        );
        let call = self.0.models.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.0.cancel_model {
            self.0.progress.cancel();
        }
        if model == ModelId::Dinov2Vits14 {
            let mut embedding = vec![0.0; 384];
            embedding[0] = 1.0;
            let mut output = TensorMap::new();
            output.insert(
                "embedding".into(),
                Tensor::from_f32(vec![1, 384], embedding)?,
            );
            return Ok(output);
        }
        let frames = inputs["frames"].to_f32()?;
        let logits = (0..BATCH_SIZE)
            .map(|index| {
                let value = frames[index * FRAME_HEIGHT * FRAME_WIDTH] as usize;
                if value % 17 == 7 { 2.0 } else { -20.0 }
            })
            .collect();
        let mut result = TensorMap::new();
        result.insert("one_hot".into(), Tensor::from_f32(vec![1, 100], logits)?);
        Ok(result)
    }
}
fn detector(state: Arc<State>) -> VideoSceneDetector {
    VideoSceneDetector::new(Arc::new(Models(state.clone())))
        .with_video_backend(Arc::new(Video(state)))
}
fn path() -> &'static Path {
    Path::new("controlled-sequential-source")
}

/// Kill: alter the leading/trailing replication, skip an overlap, or retain a
/// clip's pixels instead of consuming the same 100-frame slices as the oracle.
#[test]
fn every_streaming_window_is_byte_identical_to_buffered_padding() {
    for count in [1, 24, 25, 49, 50, 51, 74, 75, 99, 100, 101, 149, 150, 151] {
        let frames: Vec<u8> = (0..count).flat_map(frame_bytes).collect();
        let padded = pad_for_batching(&frames, count);
        let expected: Vec<_> = (0..count.div_ceil(CENTER_SIZE))
            .map(|batch| {
                let start = batch * CENTER_SIZE * PIXELS_PER_FRAME;
                padded[start..start + BATCH_SIZE * PIXELS_PER_FRAME].to_vec()
            })
            .collect();
        let mut window = ModelWindow::new();
        let mut actual = Vec::new();
        for frame in frames.chunks_exact(PIXELS_PER_FRAME) {
            window.push(frame);
            if window.ready() {
                actual.push(window.pixels.clone());
                window.consume();
            }
        }
        while !window.finished() {
            window.pad_tail();
            actual.push(window.pixels.clone());
            window.consume();
        }
        assert_eq!(actual, expected, "frame count {count}");
        assert_eq!(window.frame_count, count);
        assert_eq!(window.pixels.capacity(), BATCH_SIZE * PIXELS_PER_FRAME);
    }
}

/// Kill: let window allocation grow with clip duration despite passing the short
/// padding comparisons. Result metadata is deliberately outside this pixel bound.
#[test]
fn a_ten_minute_input_keeps_only_one_model_window() {
    let mut window = ModelWindow::new();
    for index in 0..18_000 {
        window.push(&frame_bytes(index));
        assert!(window.pixels.len() <= BATCH_SIZE * PIXELS_PER_FRAME);
        assert_eq!(window.pixels.capacity(), BATCH_SIZE * PIXELS_PER_FRAME);
        assert_eq!(window.last_frame.len(), PIXELS_PER_FRAME);
        if window.ready() {
            window.consume();
        }
    }
    while !window.finished() {
        window.pad_tail();
        window.consume();
    }
    assert_eq!(window.batches, 360);
}

/// Kill: shift the central predictions, trim the tail incorrectly, drop a decoded
/// frame, or compute different scene timestamps in the production orchestrator.
#[test]
fn streaming_model_results_equal_the_independent_buffered_oracle() {
    futures_executor::block_on(async {
        for count in [1, 50, 51, 75, 100, 101, 151, 301] {
            let state = Arc::new(State::new(count));
            let detector = detector(state.clone());
            let streamed = detector
                .detect_streaming_with_options(path(), false, false, ProgressSink::noop())
                .await
                .unwrap();
            let frames: Vec<u8> = (0..count).flat_map(frame_bytes).collect();
            let buffered = detector
                .detect(&frames, count, 30.0, ProgressSink::noop())
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(&streamed).unwrap(),
                serde_json::to_value(&buffered).unwrap()
            );
            assert_eq!(
                streamed
                    .scenes
                    .iter()
                    .map(|scene| scene.frame_count)
                    .sum::<usize>(),
                count
            );
            assert_eq!(state.opens.load(Ordering::SeqCst), 1);
            assert_eq!(state.reads.load(Ordering::SeqCst), count + 1);
            assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        }
    });
}

/// Kill: admit any probe, reader or model after the caller was already cancelled.
#[test]
fn an_already_cancelled_stream_opens_nothing() {
    let state = Arc::new(State::new(120));
    state.progress.cancel();
    let result = futures_executor::block_on(
        detector(state.clone()).detect_streaming(path(), state.progress.clone()),
    );
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(state.probes.load(Ordering::SeqCst), 0);
    assert_eq!(state.opens.load(Ordering::SeqCst), 0);
    assert_eq!(state.models.load(Ordering::SeqCst), 0);
}

/// Kill: deliver a frame decoded after cancellation, issue another read, or
/// retain the native reader on the error return.
#[test]
fn cancellation_inside_native_decode_drops_the_reader_before_return() {
    let mut state = State::new(120);
    state.cancel_read = 2;
    let state = Arc::new(state);
    let result = futures_executor::block_on(detector(state.clone()).detect_streaming_with_options(
        path(),
        false,
        false,
        state.progress.clone(),
    ));
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(state.reads.load(Ordering::SeqCst), 2);
    assert_eq!(state.models.load(Ordering::SeqCst), 0);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: infer or decode again after the first graph completed under cancellation.
#[test]
fn cancellation_inside_inference_stops_before_the_next_decode() {
    let mut state = State::new(301);
    state.cancel_model = 1;
    let state = Arc::new(state);
    let result = futures_executor::block_on(detector(state.clone()).detect_streaming_with_options(
        path(),
        false,
        false,
        state.progress.clone(),
    ));
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(state.reads.load(Ordering::SeqCst), 75);
    assert_eq!(state.models.load(Ordering::SeqCst), 1);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: let the identity tap continue into another graph or a second decode
/// when the first DINO call completed after cancellation.
#[test]
fn cancellation_inside_identity_stops_before_another_graph_or_decode() {
    let mut state = State::new(301);
    state.allow_identity = true;
    state.cancel_model = 1;
    let state = Arc::new(state);
    let result = futures_executor::block_on(
        detector(state.clone()).detect_streaming(path(), state.progress.clone()),
    );
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(state.reads.load(Ordering::SeqCst), 1);
    assert_eq!(state.models.load(Ordering::SeqCst), 1);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Kill: turn an empty/error source into a successful zero-scene result or leave
/// the owned native reader alive after either failure.
#[test]
fn empty_and_failed_streams_release_the_reader_and_do_not_infer() {
    for fail in [false, true] {
        let mut state = State::new(if fail { 120 } else { 0 });
        if fail {
            state.fail_read = 2;
        }
        let state = Arc::new(state);
        let result =
            futures_executor::block_on(detector(state.clone()).detect_streaming_with_options(
                path(),
                false,
                false,
                state.progress.clone(),
            ));
        assert!(matches!(result, Err(EngineError::Decode(_))));
        assert_eq!(state.models.load(Ordering::SeqCst), 0);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    }
}

/// Kill: a second owner outlives a consumer break and reads or disposes the same
/// decoder later. Forty independent early exits must each drop exactly one owner.
#[test]
fn early_consumer_termination_releases_its_only_decoder_forty_times() {
    let state = Arc::new(State::new(120));
    let source = VideoSource::new(Arc::new(Video(state.clone())));
    futures_executor::block_on(async {
        for round in 1..=40 {
            let mut stream = source.open_rgb(path(), &state.progress).await.unwrap();
            for index in 0..2 {
                let frame = stream.next(&state.progress).await.unwrap().unwrap();
                assert_eq!(frame.data, frame_bytes(index));
            }
            drop(stream);
            assert_eq!(state.drops.load(Ordering::SeqCst), round);
        }
    });
    assert_eq!(state.opens.load(Ordering::SeqCst), 40);
    assert_eq!(state.reads.load(Ordering::SeqCst), 80);
}

/// Kill: lose the tail, reorder a picture, or reopen a decoder after EOF.
#[test]
fn the_full_stream_reads_all_120_frames_in_order_and_finishes_once() {
    let state = Arc::new(State::new(120));
    let source = VideoSource::new(Arc::new(Video(state.clone())));
    futures_executor::block_on(async {
        let mut stream = source.open_rgb(path(), &state.progress).await.unwrap();
        for index in 0..120 {
            let frame = stream.next(&state.progress).await.unwrap().unwrap();
            assert_eq!(frame.data, frame_bytes(index));
            assert!((frame.presentation_time - index as f64 / 30.0).abs() < 1e-9);
        }
        assert!(stream.next(&state.progress).await.unwrap().is_none());
        assert!(stream.next(&state.progress).await.unwrap().is_none());
        assert_eq!(stream.frame_rate(), 30.0);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    });
    assert_eq!(state.reads.load(Ordering::SeqCst), 121);
}

#[derive(Clone)]
struct AnalysisOnlyIdentity;
#[async_trait(?Send)]
impl ShotIdentifying for AnalysisOnlyIdentity {
    async fn embedding(&self, _: &FrameSnapshot) -> Vec<f32> {
        Vec::new()
    }
    fn same_shot_distance(&self) -> f32 {
        0.72
    }
    fn similar_shot_distance(&self) -> f32 {
        0.80
    }
}

#[path = "analysis_source_tests.rs"]
mod analysis_source_tests;

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod real_reader {
    use super::*;
    use std::{path::PathBuf, process::Command};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture {
        directory: PathBuf,
        path: PathBuf,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
    fn fixture() -> Fixture {
        fixture_at("320x180")
    }
    fn fixture_at(size: &str) -> Fixture {
        let directory = std::env::temp_dir().join(format!(
            "ferment-scene-stream-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let fixture = Fixture {
            path: directory.join("120-frames.mp4"),
            directory,
        };
        let output = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={size}:rate=30"),
                "-frames:v",
                "120",
                "-an",
                "-c:v",
                "libx264",
                "-g",
                "12",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&fixture.path)
            .output()
            .expect("real reader gate requires ffmpeg");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        fixture
    }
    /// Kill: termination races an in-flight native reader owner. This repeats the
    /// current iOS 40× two-frame exit through a real codec, not the test decoder.
    #[test]
    #[ignore = "requires ffmpeg and the real platform decoder"]
    fn real_stream_early_termination_40_rounds() {
        let fixture = fixture();
        futures_executor::block_on(async {
            for _ in 0..40 {
                let source = VideoSource::default();
                let progress = ProgressSink::noop();
                let mut stream = source.open_rgb(&fixture.path, &progress).await.unwrap();
                for _ in 0..2 {
                    let frame = stream.next(&progress).await.unwrap().unwrap();
                    let image =
                        image::RgbImage::from_raw(frame.width, frame.height, frame.data).unwrap();
                    let small = image::imageops::resize(
                        &image,
                        FRAME_WIDTH as u32,
                        FRAME_HEIGHT as u32,
                        image::imageops::FilterType::Triangle,
                    );
                    assert_eq!(small.as_raw().len(), PIXELS_PER_FRAME);
                }
                drop(stream);
            }
        });
    }
    /// Kill: omit frames at end-of-stream or terminate without draining all 120
    /// encoded pictures from the actual file reader.
    #[test]
    #[ignore = "requires ffmpeg and the real platform decoder"]
    fn real_stream_yields_all_120_frames() {
        let fixture = fixture();
        futures_executor::block_on(async {
            let source = VideoSource::default();
            let progress = ProgressSink::noop();
            let mut stream = source.open_rgb(&fixture.path, &progress).await.unwrap();
            let mut count = 0;
            let mut previous_time = -1.0;
            while let Some(frame) = stream.next(&progress).await.unwrap() {
                assert!(frame.presentation_time > previous_time);
                previous_time = frame.presentation_time;
                count += 1;
            }
            assert_eq!(count, 120);
            assert_eq!(stream.frame_rate(), 30.0);
        });
    }
    /// The actual codec may honor the requested output size or refuse it; both
    /// paths must keep only the iOS-sized pictures in the transient look-behind.
    /// Kill: retain full-resolution4K snapshots,32 snapshots, or drop actual decoded frames to meet the bound.
    #[test]
    #[ignore = "requires ffmpeg and the real platform decoder"]
    fn a_real_4k_reader_drains_every_frame_with_analysis_sized_retention() {
        let fixture = fixture_at("3840x2160");
        futures_executor::block_on(async {
            let source = VideoSource::default();
            let progress = ProgressSink::noop();
            let mut stream = source
                .open_scene_rgb(&fixture.path, &progress)
                .await
                .unwrap();
            let mut collector = TransientCollector::new(AnalysisOnlyIdentity);
            let mut count = 0;
            let mut previous_time = -1.0;
            let mut max_storage = (0, 0, 0);
            while let Some(frame) = stream.next(&progress).await.unwrap() {
                assert_eq!((frame.width, frame.height), (256, 144));
                assert_eq!(frame.data.len(), 256 * 144 * 3);
                assert!(frame.presentation_time > previous_time);
                previous_time = frame.presentation_time;
                let luma = jump_cut::analysis_luma(&frame).unwrap();
                assert_eq!((luma.width, luma.height), (256, 144));
                let snapshot = FrameSnapshot::packed(256, 144, frame.data);
                collector.push(&luma, snapshot).await;
                let storage = collector.retained_pixel_storage();
                assert!(storage.0 <= 31);
                assert_eq!(storage.1, storage.0 * 256 * 144 * 3);
                assert_eq!(storage.2, storage.0 * 256 * 144);
                if storage.1 + storage.2 > max_storage.1 + max_storage.2 {
                    max_storage = storage;
                }
                count += 1;
            }
            assert_eq!(count, 120);
            assert_eq!(stream.frame_rate(), 30.0);
            assert_eq!(max_storage.0, 31);
            assert!(max_storage.1 + max_storage.2 < 5 * 1_048_576);
            println!(
                "SCENE_ANALYSIS_RETAINED source=3840x2160 frames={count} snapshots={} rgbBytes={} lumaBytes={} method=owned-buffer-lengths-not-RSS",
                max_storage.0, max_storage.1, max_storage.2
            );
        });
    }
}
