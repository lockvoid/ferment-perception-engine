//! Each consumer that owns preferred-transform application must request coded
//! pictures. The ordinary test door deliberately behaves like an Android
//! output Surface and has already applied the track rotation.
use super::*;
use async_trait::async_trait;
use ferment_engine_core::{
    FaceRect, FaceRectSource, RgbFrameRef, SaliencyHeatMap, SaliencyHeatSource, SaliencyMode,
    VideoObjectPropagationDirection::{Forward, Reverse},
    VideoScene,
};
use ferment_engine_media::{
    ContainerFormat, DecodeOptions, DecodeSession, DecoderConfiguration, EncodeSession,
    EncodedVideo, MediaInfo, MediaKind, StreamInfo, UnavailableAudioIo, VideoIo,
};
use std::sync::{Arc, Mutex};

const TIMES: [i64; 3] = [0, 33_333, 83_333];
const DURATIONS: [i64; 3] = [33_333, 50_000, 16_667];
fn clock() -> TimeBase {
    TimeBase::new_unchecked(1, 1_000_000)
}
fn stamp(value: i64) -> MediaTime {
    MediaTime::new(value, clock())
}
fn matrix(turn: u8) -> [i32; 9] {
    let (a, b, c, d) = match turn {
        0 => (65536, 0, 0, 65536),
        1 => (0, 65536, -65536, 0),
        2 => (-65536, 0, 0, -65536),
        3 => (0, -65536, 65536, 0),
        _ => unreachable!(),
    };
    [a, b, 0, c, d, 0, 0, 0, 1073741824]
}
fn coded_frame(index: usize, width: u32, height: u32, origin: i64) -> VideoFrame {
    let data: Vec<u8> = (0..width * height)
        .flat_map(|pixel| {
            let v = (pixel % 8) as u8 * 20 + index as u8;
            [v, v + 1, v + 2]
        })
        .collect();
    VideoFrame {
        width,
        height,
        pts: stamp(origin + TIMES[index]),
        duration: Some(stamp(DURATIONS[index])),
        color: ColorInfo::default(),
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(
                width,
                height,
                width as usize * 3,
                CpuPixelFormat::Rgb24,
                data,
            )
            .unwrap(),
        ),
    }
}
fn wanted(turn: u8, index: usize) -> (u32, u32, Vec<u8>) {
    let order: &[u8] = match turn {
        0 => &[0, 1, 2, 3, 4, 5, 6, 7],
        1 => &[4, 0, 5, 1, 6, 2, 7, 3],
        2 => &[7, 6, 5, 4, 3, 2, 1, 0],
        3 => &[3, 7, 2, 6, 1, 5, 0, 4],
        _ => unreachable!(),
    };
    let bytes = order
        .iter()
        .flat_map(|v| {
            let v = *v * 20 + index as u8;
            [v, v + 1, v + 2]
        })
        .collect();
    let (w, h) = if turn % 2 == 1 { (2, 4) } else { (4, 2) };
    (w, h, bytes)
}
fn assert_frame(frame: &VideoFrame, turn: u8, index: usize, relative: i64) {
    let (w, h, bytes) = wanted(turn, index);
    let cpu = frame.to_cpu_rgb().unwrap().to_rgb24();
    assert_eq!(
        (frame.width, frame.height, cpu.width, cpu.height),
        (w, h, w, h)
    );
    assert_eq!(&*cpu.data, &bytes);
    assert_eq!(frame.pts, stamp(relative));
    assert_eq!(frame.duration, Some(stamp(DURATIONS[index])));
    assert_eq!(frame.color, ColorInfo::default());
}

#[derive(Debug, Default)]
struct Calls {
    regular: usize,
    coded: usize,
    options: Vec<DecodeOptions>,
    seeks: Vec<MediaTime>,
    settings: Vec<EncoderSettings>,
    drops: usize,
}
#[derive(Debug)]
struct Video {
    turn: u8,
    size: (u32, u32),
    origin: i64,
    calls: Arc<Mutex<Calls>>,
}
impl Video {
    fn new(turn: u8, size: (u32, u32), origin: i64) -> Arc<Self> {
        Arc::new(Self {
            turn,
            size,
            origin,
            calls: Arc::new(Mutex::new(Calls::default())),
        })
    }
    fn decoder(&self, automatic: bool) -> Box<dyn DecodeSession> {
        let frames = (0..3)
            .map(|index| {
                let frame = coded_frame(index, self.size.0, self.size.1, self.origin);
                if automatic {
                    frame.rotated_clockwise(self.turn).unwrap()
                } else {
                    frame
                }
            })
            .collect();
        Box::new(Reader {
            frames,
            cursor: 0,
            calls: self.calls.clone(),
        })
    }
    fn backends(self: &Arc<Self>) -> crate::EncodeBackends {
        crate::EncodeBackends {
            video: self.clone(),
            audio: Arc::new(UnavailableAudioIo),
            muxer: Arc::new(ferment_engine_media::platform::libav::LibavMuxer),
        }
    }
}
#[derive(Debug)]
struct Reader {
    frames: Vec<VideoFrame>,
    cursor: usize,
    calls: Arc<Mutex<Calls>>,
}
impl Drop for Reader {
    fn drop(&mut self) {
        self.calls.lock().unwrap().drops += 1;
    }
}
#[async_trait(?Send)]
impl DecodeSession for Reader {
    async fn seek(&mut self, target: MediaTime) -> ferment_engine_media::Result<()> {
        self.calls.lock().unwrap().seeks.push(target);
        self.cursor = self
            .frames
            .iter()
            .position(|frame| frame.pts.seconds() >= target.seconds())
            .unwrap_or(self.frames.len());
        Ok(())
    }
    async fn decode_next(&mut self) -> ferment_engine_media::Result<Option<VideoFrame>> {
        let frame = self.frames.get(self.cursor).cloned();
        self.cursor += usize::from(frame.is_some());
        Ok(frame)
    }
}
#[derive(Debug)]
struct NoWrite;
#[async_trait(?Send)]
impl EncodeSession for NoWrite {
    async fn push_frame(&mut self, _frame: VideoFrame) -> ferment_engine_media::Result<()> {
        panic!("capture must stop at the inference seam before encoding");
    }
    async fn finalize(self: Box<Self>) -> ferment_engine_media::Result<EncodedVideo> {
        panic!("capture must not fabricate an encoded file");
    }
}
#[async_trait(?Send)]
impl VideoIo for Video {
    async fn probe(&self, _: &Path) -> ferment_engine_media::Result<MediaInfo> {
        Ok(MediaInfo {
            container: ContainerFormat::Mp4,
            duration: Some(stamp(100_000)),
            streams: vec![StreamInfo {
                index: 0,
                kind: MediaKind::Video,
                codec: Codec::H264,
                time_base: clock(),
                start_time: Some(stamp(self.origin)),
                duration: Some(stamp(100_000)),
                width: Some(self.size.0),
                height: Some(self.size.1),
                sample_rate: None,
                channels: None,
                frame_rate: Some((30, 1)),
                display_matrix: Some(matrix(self.turn)),
                bit_rate: Some(20_000_000),
                language: None,
                color: ColorInfo::default(),
                audio_timing: None,
                decoder_configuration: DecoderConfiguration::from_codec_private(
                    Codec::H264,
                    vec![],
                )
                .unwrap(),
            }],
        })
    }
    async fn open_decoder(
        &self,
        _: &Path,
        _: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.calls.lock().unwrap().regular += 1;
        Ok(self.decoder(true))
    }
    async fn open_coded_decoder(
        &self,
        _: &Path,
        _: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.calls.lock().unwrap().coded += 1;
        Ok(self.decoder(false))
    }
    async fn open_coded_decoder_with(
        &self,
        path: &Path,
        index: u32,
        options: DecodeOptions,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.calls.lock().unwrap().options.push(options);
        self.open_coded_decoder(path, index).await
    }
    async fn open_encoder(
        &self,
        settings: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        self.calls.lock().unwrap().settings.push(settings);
        Ok(Box::new(NoWrite))
    }
}
fn window() -> core::MediaTimeRange {
    core::MediaTimeRange {
        start: core::MediaTime::new(33_333, 1_000_000),
        duration: core::MediaTime::new(66_667, 1_000_000),
    }
}

// Kill: SourceWindow auto-rotates and then rotates again, or the repair retimes
// VFR pictures/crops, loses duration/color, or reintroduces transform metadata.
#[test]
fn shared_render_source_applies_one_transform_after_seek_without_retiming() {
    futures_executor::block_on(async {
        for turn in 0..4 {
            let video = Video::new(turn, (4, 2), 0);
            let mut source = SourceWindow::open_with_backends(
                video.backends(),
                Path::new("preferred.mp4"),
                Some(window()),
            )
            .await
            .unwrap();
            let (w, h, _) = wanted(turn, 0);
            assert_eq!(
                (source.stream.width, source.stream.height),
                (Some(w), Some(h))
            );
            assert_eq!(source.stream.display_matrix, None);
            let _output = VideoOutput::open(&source).await.unwrap();
            let first = source.next(&ProgressSink::noop()).await.unwrap().unwrap();
            let second = source.next(&ProgressSink::noop()).await.unwrap().unwrap();
            assert_frame(&first, turn, 1, 0);
            assert_frame(&second, turn, 2, 50_000);
            assert!(source.next(&ProgressSink::noop()).await.unwrap().is_none());
            drop(source);
            let calls = video.calls.lock().unwrap();
            assert_eq!((calls.regular, calls.coded, calls.drops), (0, 1, 1));
            assert_eq!(calls.seeks, [stamp(33_333)]);
            assert_eq!(calls.settings.len(), 1);
            let settings = &calls.settings[0];
            assert_eq!((settings.width, settings.height), (w, h));
            assert_eq!(settings.codec, Codec::Hevc);
            assert_eq!(settings.frame_rate, Some((30, 1)));
            assert_eq!(settings.time_base, clock());
            assert_eq!(settings.bit_rate, Some(20_000_000));
        }
    });
}

struct CaptureUpscale {
    frames: Vec<VideoFrame>,
    ended: bool,
}
#[async_trait(?Send)]
impl FrameUpscaling for CaptureUpscale {
    fn native_scale_factor(&self) -> u32 {
        1
    }
    fn start_session(&mut self, _: Dimensions, _: bool) -> Result<()> {
        Ok(())
    }
    async fn upscale(&mut self, frame: &VideoFrame) -> Result<FrameBuffer> {
        self.frames.push(frame.clone());
        Err(EngineError::internal("orientation captured"))
    }
    fn end_session(&mut self) {
        self.ended = true;
    }
}
// Kill: only repair the shared denoise/slow-motion door while the standalone
// upscale processor still sees an already transformed Surface picture.
#[test]
fn upscale_inference_receives_once_oriented_pixels_with_original_interval() {
    futures_executor::block_on(async {
        for turn in 1..4 {
            let video = Video::new(turn, (4, 2), 0);
            let mut processor = CaptureUpscale {
                frames: vec![],
                ended: false,
            };
            let error = upscale_video_with_processor(
                video.backends(),
                Path::new("preferred.mp4"),
                Path::new("unused-output.mp4"),
                UpscaleMode::Enhance,
                Some(window()),
                ProgressSink::noop(),
                &mut processor,
                None,
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("orientation captured"));
            assert!(processor.ended);
            assert_eq!(processor.frames.len(), 1);
            assert_frame(&processor.frames[0], turn, 1, 0);
            let calls = video.calls.lock().unwrap();
            assert_eq!((calls.regular, calls.coded, calls.drops), (0, 1, 1));
            assert_eq!(calls.seeks, [stamp(33_333)]);
        }
    });
}
#[derive(Debug, Default)]
struct CaptureHeat {
    frames: Mutex<Vec<(u32, u32, Vec<u8>)>>,
}
#[async_trait(?Send)]
impl SaliencyHeatSource for CaptureHeat {
    fn mode(&self) -> SaliencyMode {
        SaliencyMode::Objectness
    }
    async fn heat(&self, frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap> {
        self.frames
            .lock()
            .unwrap()
            .push((frame.width, frame.height, frame.data.to_vec()));
        Ok(SaliencyHeatMap {
            values: vec![1.; frame.width as usize * frame.height as usize],
            width: frame.width as usize,
            height: frame.height as usize,
        })
    }
}
struct NoFaces;
#[async_trait(?Send)]
impl FaceRectSource for NoFaces {
    async fn faces(&self, _: RgbFrameRef<'_>) -> Result<Vec<FaceRect>> {
        Ok(vec![])
    }
}
// Kill: model measurements land in a double-rotated coordinate space, including
// 180 degrees where width/height alone cannot expose the error.
#[test]
fn saliency_model_sees_display_pixels_once_across_real_grabber_seeks() {
    futures_executor::block_on(async {
        for turn in 1..4 {
            let video = Video::new(turn, (4, 2), 0);
            let heat = Arc::new(CaptureHeat::default());
            let analyzer =
                crate::video_saliency::VideoSaliencyAnalyzer::new(heat.clone(), Arc::new(NoFaces))
                    .with_video_backend(video.clone());
            let scenes = (0..2)
                .map(|index| VideoScene {
                    index,
                    start_frame: index,
                    frame_count: 1,
                    start_time: index as f64 * 0.05,
                    duration: 0.05,
                    start_confidence: None,
                    embedding: None,
                })
                .collect::<Vec<_>>();
            let result = analyzer
                .analyze(Path::new("preferred.mp4"), &scenes, ProgressSink::noop())
                .await
                .unwrap();
            let (w, h, _) = wanted(turn, 0);
            assert_eq!((result.frame_width, result.frame_height), (w, h));
            assert_eq!(result.samples.len(), 2);
            let frames = heat.frames.lock().unwrap();
            assert_eq!(frames.as_slice(), &[wanted(turn, 1), wanted(turn, 2)]);
            let calls = video.calls.lock().unwrap();
            assert_eq!((calls.regular, calls.coded, calls.drops), (0, 1, 1));
            assert_eq!(calls.seeks, [stamp(20_000), stamp(70_000)]);
        }
    });
}
// Kill: timeline indexing uses Surface dimensions while forward/reverse reads
// use coded dimensions, or either decoder admission applies the transform twice.
// Compare exact microsecond PTS after a nonzero origin; no tolerance relaxation.
#[test]
fn edge_timeline_forward_and_reverse_use_coded_pixels_and_exact_timestamps() {
    futures_executor::block_on(async {
        for turn in 1..4 {
            let video = Video::new(turn, (4, 2), 500_000);
            let mut source = crate::edgetam::frame_source::EdgeTamFrameSource::new(
                video.clone(),
                Path::new("preferred.mp4"),
            )
            .await
            .unwrap();
            let (w, h, _) = wanted(turn, 0);
            let info = source.source_info();
            assert_eq!(
                (info.width, info.height, info.frame_count),
                (w as usize, h as usize, 3)
            );
            for direction in [Forward, Reverse] {
                let indices: Vec<usize> = if direction == Forward {
                    (0..3).collect()
                } else {
                    (0..3).rev().collect()
                };
                for index in indices {
                    let frame = source.sequential_frame(index, direction).await.unwrap();
                    let (_, _, bytes) = wanted(turn, index);
                    assert_eq!((frame.width, frame.height), (w as usize, h as usize));
                    assert_eq!(frame.pixels, bytes);
                    assert_eq!(frame.presentation_timestamp, stamp(TIMES[index]));
                    assert_eq!(
                        source.presentation_time(index).unwrap(),
                        TIMES[index] as f64 / 1_000_000.
                    );
                }
            }
            drop(source);
            let calls = video.calls.lock().unwrap();
            assert_eq!(calls.regular, 0);
            assert!(
                calls.coded >= 2,
                "timeline and reopened reader must both choose coded pixels"
            );
            assert_eq!(calls.drops, calls.coded);
        }
    });
}
// Kill: keep the coded-only door for minimap yet use rotating Surface pixels
// before scene analysis's existing naturalSize resize. Unknown output_size on
// small sources must still take the coded-with-options door.
#[test]
fn scene_reader_keeps_coded_geometry_resize_policy_and_original_frame_clock() {
    futures_executor::block_on(async {
        for (size, output) in [((4, 2), None), ((512, 256), Some((256, 128)))] {
            let video = Video::new(1, size, 0);
            let source = crate::video_source::VideoSource::new(video.clone());
            let mut reader = source
                .open_scene_rgb(Path::new("preferred.mp4"), &ProgressSink::noop())
                .await
                .unwrap();
            for time in TIMES {
                let frame = reader.next(&ProgressSink::noop()).await.unwrap().unwrap();
                assert_eq!((frame.width, frame.height), output.unwrap_or(size));
                assert_eq!(frame.presentation_time, time as f64 / 1_000_000.);
            }
            assert!(reader.next(&ProgressSink::noop()).await.unwrap().is_none());
            assert_eq!(reader.frame_rate(), 30.);
            drop(reader);
            let calls = video.calls.lock().unwrap();
            assert_eq!((calls.regular, calls.coded, calls.drops), (0, 1, 1));
            assert_eq!(
                calls.options,
                [DecodeOptions {
                    output_size: output,
                    acceleration: ferment_engine_media::DecodeAcceleration::Software,
                    ..Default::default()
                }]
            );
        }
    });
}

// Kill: automatic slow-motion analysis opens the surface-oriented door even
// though the current Swift AVAssetReader analyzer evaluates coded luma.
#[test]
fn automatic_motion_analysis_keeps_coded_admission_on_rotated_sources() {
    for turn in 0..4 {
        let video = Video::new(turn, (4, 2), 0);
        let source = futures_executor::block_on(SourceWindow::open_with_backends(
            video.backends(),
            Path::new("preferred.mp4"),
            None,
        ))
        .unwrap();
        let _ = analyze_motion(&source, Path::new("preferred.mp4"), &ProgressSink::noop()).unwrap();
        drop(source);
        let calls = video.calls.lock().unwrap();
        assert_eq!((calls.regular, calls.coded, calls.drops), (0, 2, 2));
        assert!(
            calls.settings.is_empty(),
            "analysis must not open an encoder"
        );
    }
}
