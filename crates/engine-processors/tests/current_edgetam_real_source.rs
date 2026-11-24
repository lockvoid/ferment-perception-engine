//! Current EdgeTAM fixture, cancellation and process-memory assertions. These
//! gates use real platform decode; no picture, timestamp, model or RSS is faked.
#![cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]

use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, ProgressSink,
    VideoObjectPropagationDirection::{Forward, Reverse},
};
use ferment_engine_media::{
    DecodeSession, EncodeSession, EncoderSettings, MediaInfo, MediaTime, TimeBase, VideoBackend,
    VideoFrame, VideoIo,
};
use ferment_engine_processors::{
    edgetam::{
        frame_source::EdgeTamFrameSource,
        image_preprocessing,
        preprocessing::EdgeTamRgbFrame,
        processing::PromptTensor,
        session::ObjectState,
        temporal_state::{
            EdgeTamStoredFrame, LOW_RESOLUTION_MASK_COUNT, MEMORY_FEATURE_COUNT,
            OBJECT_POINTER_COUNT,
        },
    },
    video_backend,
};
use half::f16;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
#[path = "support/current_process_memory.rs"]
mod process_memory;
use process_memory::footprint_bytes;

fn fixture(name: &str) -> PathBuf {
    let root = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(PathBuf::from)
        .expect("current EdgeTAM gate requires FERMENT_ENGINE_FIXTURES_DIR");
    let path = root.join(name);
    assert!(
        path.is_file(),
        "required original fixture missing: {}",
        path.display()
    );
    path
}
fn open(path: &Path) -> EdgeTamFrameSource {
    futures_executor::block_on(EdgeTamFrameSource::new(video_backend(), path)).unwrap()
}
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}
fn exact_time(value: i64, denominator: i64) -> MediaTime {
    MediaTime::new(value, TimeBase::new_unchecked(1, denominator))
}

/// Kill: reconstruct the timeline from duration/FPS or round a sample before
/// indexing. The exact counts and Double-bit FNVs come from the current iOS files.
#[test]
#[ignore = "requires the three original current iOS videos and real platform decoder"]
fn real_timelines_keep_actual_sample_counts_and_exact_double_hashes() {
    for (name, count, expected_hash) in [
        (
            "Video/formula1-trackside.mov",
            1_013,
            0xbe16_799e_8461_388e_u64,
        ),
        (
            "Video/japanese-rain-garden.mov",
            2_954,
            0xb1ea_1113_a9b9_76a0,
        ),
        ("Video/dhh-interview.mp4", 7_104, 0x45cf_1621_3f1e_cc08),
    ] {
        let mut source = open(&fixture(name));
        assert_eq!(source.source_info().frame_count, count, "{name}");
        assert!(source.presentation_time(0).unwrap().abs() < 1e-6);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for index in 0..count {
            hash ^= source.presentation_time(index).unwrap().to_bits();
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        println!("EDGETAM_TIMELINE {name} frames={count} fnv64={hash:016x}");
        assert_eq!(hash, expected_hash, "{name} exact current sample timeline");
        for index in [0, count - 1] {
            let frame = futures_executor::block_on(source.frame(index)).unwrap();
            assert_eq!(frame.frame_index, index);
            assert!(
                (frame.presentation_time() - source.presentation_time(index).unwrap()).abs() < 1e-9
            );
        }
        source.cancel_decoder_resources();
        assert!(!source.has_decoder_resources());
    }
}

/// Kill: seek to a neighboring HEVC sync sample or quantize the indexed 248/249
/// timestamps. The nanosecond comparison is unchanged from current iOS.
#[test]
#[ignore = "requires the original hdr-hlg-12s.mov and real platform decoder"]
fn hlg_nonkeyframes_248_and_249_are_exact_in_both_directions() {
    let mut source = open(&fixture("Video/hdr-hlg-12s.mov"));
    let random = futures_executor::block_on(source.frame(248)).unwrap();
    let forward248 = futures_executor::block_on(source.sequential_frame(248, Forward)).unwrap();
    let forward249 = futures_executor::block_on(source.sequential_frame(249, Forward)).unwrap();
    assert!((forward248.presentation_time() - 248.0 / 30.0).abs() < 1e-9);
    assert!((forward249.presentation_time() - 249.0 / 30.0).abs() < 1e-9);
    assert_eq!(random.pixels, forward248.pixels);
    let reverse249 = futures_executor::block_on(source.sequential_frame(249, Reverse)).unwrap();
    let reverse248 = futures_executor::block_on(source.sequential_frame(248, Reverse)).unwrap();
    assert_eq!(reverse249.pixels, forward249.pixels);
    assert_eq!(reverse248.pixels, forward248.pixels);
    assert_eq!(
        reverse249.presentation_timestamp,
        forward249.presentation_timestamp
    );
    assert_eq!(
        reverse248.presentation_timestamp,
        forward248.presentation_timestamp
    );
    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());
}

/// Kill: discard rational metadata between the decoder, cached image features,
/// stored FP16 state, and public mask timestamp conversion.
#[test]
fn decoded_and_stored_frames_preserve_thirtieths_without_a_microsecond_round_trip() {
    for index in [1, 248, 249, 17_999] {
        let time = exact_time(index as i64, 30);
        let decoded = EdgeTamRgbFrame::new(index, time, 1, 1, vec![0; 3]).unwrap();
        let stored = state_frame(index);
        assert_eq!(decoded.presentation_timestamp, time);
        assert_eq!(stored.presentation_timestamp, time);
        assert_eq!(decoded.presentation_time(), index as f64 / 30.0);
        assert_eq!(stored.presentation_time(), index as f64 / 30.0);
    }
}

struct PeakSampler {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<u64>>,
}
impl PeakSampler {
    fn start(baseline: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        Self {
            stop,
            worker: Some(thread::spawn(move || {
                let mut peak = baseline;
                while !worker_stop.load(Ordering::Acquire) {
                    peak = peak.max(footprint_bytes());
                    thread::sleep(Duration::from_millis(1));
                }
                peak.max(footprint_bytes())
            })),
        }
    }
    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Release);
        self.worker.take().unwrap().join().unwrap()
    }
}
impl Drop for PeakSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Debug)]
struct CancellationWindow {
    entered: mpsc::SyncSender<()>,
    progress: ProgressSink,
}
#[derive(Debug, Default)]
struct Calls {
    active: AtomicUsize,
    peak: AtomicUsize,
    active_decoders: AtomicUsize,
    window: Mutex<Option<CancellationWindow>>,
}
struct ActiveRead(Arc<Calls>);
impl ActiveRead {
    fn new(calls: &Arc<Calls>) -> Self {
        let active = calls.active.fetch_add(1, Ordering::SeqCst) + 1;
        calls.peak.fetch_max(active, Ordering::SeqCst);
        Self(calls.clone())
    }
}
impl Drop for ActiveRead {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct ObservedBackend {
    real: VideoBackend,
    calls: Arc<Calls>,
}
#[derive(Debug)]
struct ObservedDecoder {
    real: Box<dyn DecodeSession>,
    calls: Arc<Calls>,
}
impl Drop for ObservedDecoder {
    fn drop(&mut self) {
        self.calls.active_decoders.fetch_sub(1, Ordering::SeqCst);
    }
}
#[async_trait(?Send)]
impl DecodeSession for ObservedDecoder {
    async fn seek(&mut self, target: MediaTime) -> ferment_engine_media::Result<()> {
        let _active = ActiveRead::new(&self.calls);
        let window = self.calls.window.lock().unwrap().take();
        if let Some(window) = &window {
            window.entered.send(()).unwrap();
        }
        let result = self.real.seek(target).await;
        if let Some(window) = &window {
            // Hold this real native result at the ownership boundary until the
            // scheduled cancellation arrives. A codec faster than 1/10ms must
            // not turn an in-flight cancellation test into a completed-before-
            // cancellation case. No frame or measured native work is replaced.
            let deadline = Instant::now() + Duration::from_secs(2);
            while !window.progress.is_closed() {
                assert!(Instant::now() < deadline, "cancellation worker did not run");
                thread::yield_now();
            }
        }
        result
    }
    async fn decode_next(&mut self) -> ferment_engine_media::Result<Option<VideoFrame>> {
        let _active = ActiveRead::new(&self.calls);
        self.real.decode_next().await
    }
}
#[async_trait(?Send)]
impl VideoIo for ObservedBackend {
    async fn probe(&self, source: &Path) -> ferment_engine_media::Result<MediaInfo> {
        self.real.probe(source).await
    }
    async fn open_decoder(
        &self,
        source: &Path,
        index: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        let real = self.real.open_decoder(source, index).await?;
        self.calls.active_decoders.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(ObservedDecoder {
            real,
            calls: self.calls.clone(),
        }))
    }
    async fn open_encoder(
        &self,
        settings: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        self.real.open_encoder(settings).await
    }
}
fn observed_source(path: &Path) -> (EdgeTamFrameSource, Arc<Calls>) {
    let calls = Arc::new(Calls::default());
    let backend = Arc::new(ObservedBackend {
        real: video_backend(),
        calls: calls.clone(),
    });
    let source = futures_executor::block_on(EdgeTamFrameSource::new(backend, path)).unwrap();
    (source, calls)
}

fn cancellation_cycles(forward: bool) {
    let (mut source, calls) = observed_source(&fixture("Video/hdr-hlg-12s.mov"));
    let info = source.source_info();
    assert_eq!((info.width, info.height), (1_920, 1_080));
    let warm = futures_executor::block_on(source.frame(248)).unwrap();
    drop(warm);
    source.cancel_decoder_resources();
    let baseline = footprint_bytes();
    let start = Instant::now();
    let mut maximum_latency = Duration::ZERO;
    let cycles = if forward { 64 } else { 16 };
    for iteration in 0..cycles {
        let progress = ProgressSink::noop();
        let worker_progress = progress.clone();
        let (send, receive) = mpsc::sync_channel(0);
        *calls.window.lock().unwrap() = Some(CancellationWindow {
            entered: send,
            progress: progress.clone(),
        });
        let cancelled_at = Arc::new(Mutex::new(None));
        let worker_time = cancelled_at.clone();
        let worker = thread::spawn(move || {
            receive
                .recv_timeout(Duration::from_secs(2))
                .expect("real native seek must start");
            thread::sleep(Duration::from_millis(if forward { 1 } else { 10 }));
            *worker_time.lock().unwrap() = Some(Instant::now());
            worker_progress.cancel();
        });
        let result = if forward {
            futures_executor::block_on(
                source.sequential_frame_with_cancellation(248, Forward, progress),
            )
        } else {
            futures_executor::block_on(source.frame_with_cancellation(248, progress))
        };
        worker.join().unwrap();
        let latency = cancelled_at.lock().unwrap().unwrap().elapsed();
        maximum_latency = maximum_latency.max(latency);
        assert!(
            latency < Duration::from_secs(1),
            "reset cycle {iteration}: {latency:?}"
        );
        assert!(
            matches!(result, Err(EngineError::Cancelled)),
            "cycle {iteration}: stale frame or wrong error: {:?}",
            result
                .as_ref()
                .map(|frame| (frame.frame_index, frame.presentation_time()))
        );
        source.cancel_decoder_resources();
        assert!(!source.has_decoder_resources());
        assert_eq!(calls.active.load(Ordering::SeqCst), 0);
        assert_eq!(calls.active_decoders.load(Ordering::SeqCst), 0);
    }
    let elapsed = start.elapsed();
    let final_bytes = footprint_bytes();
    println!(
        "EDGETAM_CANCEL forward={forward} cycles={cycles} elapsed={elapsed:?} max_reset={maximum_latency:?} peak_active={} baseline={baseline} final={final_bytes}",
        calls.peak.load(Ordering::SeqCst)
    );
    assert_eq!(calls.peak.load(Ordering::SeqCst), 1);
    assert!(
        elapsed < Duration::from_secs(10),
        "{cycles} resets took {elapsed:?}"
    );
    assert!(final_bytes.saturating_sub(baseline) < 128 * 1_048_576);
}

/// Kill: continue the native forward decode or retain one decoder per reset.
/// Current iOS fixture, 64 cycles, 1ms handoff, <1s each/<10s total/<128MiB.
#[test]
#[ignore = "real HLG codec/time/memory gate; run alone"]
fn forward_hlg_64_cancellations_are_prompt_and_bounded() {
    cancellation_cycles(true);
}

/// Kill: return a stale random-access picture after cancellation or keep its
/// native surfaces. Current iOS fixture, 16 cycles and 10ms handoff, same limits.
#[test]
#[ignore = "real HLG codec/time/memory gate; run alone"]
fn random_hlg_16_cancellations_are_prompt_and_bounded() {
    cancellation_cycles(false);
}

/// Kill: retain duration-sized decoded frames while indexing the real18,000-frame
/// clip. Current iOS 16MiB indexing budget and exact 30Hz sample timestamps.
#[test]
#[ignore = "real ten-minute file and process-memory gate; run alone"]
fn the_real_ten_minute_index_is_exact_and_below_16_mib_growth() {
    let path = fixture("EdgeTAM/ten-minute-30fps-16x16.mp4");
    let baseline = footprint_bytes();
    let source = open(&path);
    assert_eq!(source.source_info().frame_count, 18_000);
    for index in [0, 4_500, 9_000, 13_500, 17_999] {
        assert!((source.presentation_time(index).unwrap() - index as f64 / 30.0).abs() < 1e-9);
    }
    let final_bytes = footprint_bytes();
    println!("EDGETAM_REAL_INDEX baseline={baseline} final={final_bytes}");
    assert!(final_bytes.saturating_sub(baseline) < 16 * 1_048_576);
}

/// Kill: a reverse batch returns neighboring frames, leaks decoder ownership, or
/// grows beyond the current 256MiB peak /64MiB final bounds on the exact HLG clip.
/// Pixel equality uses an independent current decoder, as the current six-file
/// iOS provider test prescribes; old Apple-toolchain-specific FNVs are not reused.
#[test]
#[ignore = "real1080p HLG reverse and process-memory gate; run alone"]
fn reverse_1080p_all_360_frames_are_exact_and_bounded() {
    let path = fixture("Video/hdr-hlg-12s.mov");
    let checkpoints = [0, 180, 248, 249, 359];
    let mut reference = open(&path);
    let expected: BTreeMap<_, _> = checkpoints
        .into_iter()
        .map(|index| {
            let frame = futures_executor::block_on(reference.frame(index)).unwrap();
            (index, fnv(&frame.pixels))
        })
        .collect();
    let warm = futures_executor::block_on(reference.sequential_frame(359, Reverse)).unwrap();
    drop(warm);
    reference.cancel_decoder_resources();
    drop(reference);
    let (mut source, calls) = observed_source(&path);
    let info = source.source_info();
    assert_eq!(
        (info.width, info.height, info.frame_count),
        (1_920, 1_080, 360)
    );
    let baseline = footprint_bytes();
    let sampler = PeakSampler::start(baseline);
    let start = Instant::now();
    for index in (0..360).rev() {
        let frame = futures_executor::block_on(source.sequential_frame(index, Reverse)).unwrap();
        assert_eq!(frame.frame_index, index);
        assert!(
            (frame.presentation_time() - source.presentation_time(index).unwrap()).abs() < 1e-9
        );
        assert!((frame.presentation_time() - index as f64 / 30.0).abs() < 1e-9);
        if let Some(expected) = expected.get(&index) {
            assert_eq!(fnv(&frame.pixels), *expected, "HLG checkpoint {index}");
            println!("EDGETAM_REVERSE checkpoint={index} fnv64={expected:016x}");
        }
    }
    let peak = sampler.finish();
    let final_bytes = footprint_bytes();
    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());
    assert_eq!(calls.active.load(Ordering::SeqCst), 0);
    assert_eq!(calls.active_decoders.load(Ordering::SeqCst), 0);
    assert_eq!(calls.peak.load(Ordering::SeqCst), 1);
    println!(
        "EDGETAM_REVERSE frames=360 elapsed={:?} baseline={baseline} peak={peak} final={final_bytes}",
        start.elapsed()
    );
    assert!(peak.saturating_sub(baseline) < 256 * 1_048_576);
    assert!(final_bytes.saturating_sub(baseline) < 64 * 1_048_576);
}

fn state_frame(index: usize) -> EdgeTamStoredFrame {
    EdgeTamStoredFrame::new(
        index,
        exact_time(index as i64, 30),
        vec![f16::from_f32((index % 11) as f32); LOW_RESOLUTION_MASK_COUNT],
        vec![f16::from_f32((index % 13) as f32); MEMORY_FEATURE_COUNT],
        vec![f16::from_f32((index % 17) as f32); OBJECT_POINTER_COUNT],
        0.5,
        1.0,
    )
    .unwrap()
}
fn prompt() -> PromptTensor {
    PromptTensor {
        coordinates: vec![512.0, 512.0],
        labels: vec![1],
    }
}

/// Kill: retain ordinary tracking state or duplicate anchors over18,000 frames.
/// All32 anchors and every subsequent frame own distinct actual FP16 allocations.
#[test]
#[ignore = "ten-minute temporal-state physical-memory gate; run alone"]
fn ten_minute_tracking_with_all_32_prompt_anchors_stays_below_64_mib_growth() {
    let mut state = ObjectState {
        conditioning: state_frame(0),
        non_conditioning: BTreeMap::new(),
        prompts: BTreeMap::from([(0, prompt())]),
        conditioning_was_tracked: true,
        directions: BTreeMap::from([(0, Forward)]),
    };
    for index in 1..32 {
        state.non_conditioning.insert(index, state_frame(index));
        state.store_prompt(prompt(), index).unwrap();
        state.directions.insert(index, Forward);
    }
    for index in 32..512 {
        state.non_conditioning.insert(index, state_frame(index));
        state.directions.insert(index, Forward);
        state.prune_tracking_history(index, Forward);
    }
    let baseline = footprint_bytes();
    let mut peak = baseline;
    for index in 512..18_000 {
        state.non_conditioning.insert(index, state_frame(index));
        state.directions.insert(index, Forward);
        state.prune_tracking_history(index, Forward);
        if index % 4_500 == 0 || index == 17_999 {
            peak = peak.max(footprint_bytes());
            assert!(state.non_conditioning.len() < 47);
            assert_eq!(state.prompts.len(), 32);
            assert!(state.directions.len() <= 47);
        }
    }
    println!(
        "EDGETAM_TEMPORAL frames=18000 anchors=32 retained={} baseline={baseline} peak={peak}",
        state.non_conditioning.len() + 1
    );
    assert!(peak.saturating_sub(baseline) < 64 * 1_048_576);
}

/// Kill: allocate an unbounded high-resolution intermediate for still-image
/// normalization. Same3840×2160 input, FP16 shape and512MiB peak budget as iOS.
#[test]
#[ignore = "4K preprocessing physical-memory gate; run alone"]
fn a_4k_still_preprocess_stays_below_512_mib_peak_growth() {
    let pixels = (0..3_840 * 2_160 * 3)
        .map(|index| (index % 251) as u8)
        .collect();
    let source = EdgeTamRgbFrame::new(0, MediaTime::ZERO, 3_840, 2_160, pixels).unwrap();
    let baseline = footprint_bytes();
    let sampler = PeakSampler::start(baseline);
    let start = Instant::now();
    let normalized = image_preprocessing::normalized_image(&source).unwrap();
    assert_eq!(normalized.len(), 3 * 1_024 * 1_024);
    let peak = sampler.finish();
    let final_bytes = footprint_bytes();
    println!(
        "EDGETAM_4K baseline={baseline} peak={peak} final={final_bytes} elapsed={:?}",
        start.elapsed()
    );
    assert!(peak >= baseline);
    assert!(peak - baseline < 512 * 1_048_576);
    std::hint::black_box(normalized);
}

/// Kill: decode a different frame after a cursor reuse, return nondeterministic
/// HDR pixels, or lose the last sample. Same six fixtures/checkpoints as iOS.
/// The off-Apple independent oracle is direct platform VideoIO, not an Apple
/// image-generator byte signature that changes with its tone-mapping toolchain.
#[test]
#[ignore = "requires all six original current video fixtures and real decoder"]
fn all_six_forward_providers_match_independent_current_decoder_checkpoints() {
    for name in [
        "Video/formula1-trackside.mov",
        "Video/japanese-rain-garden.mov",
        "Video/dhh-interview.mp4",
        "EdgeTAM/rotated-48x32.mp4",
        "Video/hdr-pq-15s.mp4",
        "Video/hdr-hlg-12s.mov",
    ] {
        let path = fixture(name);
        let mut source = open(&path);
        let count = source.source_info().frame_count;
        let checkpoints = [0, count / 2, count - 1];
        let mut expected = BTreeMap::new();
        for index in 0..count {
            let frame =
                futures_executor::block_on(source.sequential_frame(index, Forward)).unwrap();
            assert_eq!(frame.frame_index, index);
            assert!(
                (frame.presentation_time() - source.presentation_time(index).unwrap()).abs() < 1e-6
            );
            if checkpoints.contains(&index) {
                expected.insert(index, frame.pixels);
            }
        }
        source.cancel_decoder_resources();
        assert!(!source.has_decoder_resources());
        let mut second = open(&path);
        for index in checkpoints {
            let frame =
                futures_executor::block_on(second.sequential_frame(index, Forward)).unwrap();
            assert_eq!(
                frame.pixels, expected[&index],
                "independent source {name} frame {index}"
            );
        }
        second.cancel_decoder_resources();
        assert!(!second.has_decoder_resources());
        futures_executor::block_on(async {
            let backend = video_backend();
            let info = backend.probe(&path).await.unwrap();
            let stream = info.video_streams().next().unwrap();
            let mut decoder = backend.open_decoder(&path, stream.index).await.unwrap();
            let mut index = 0;
            while let Some(frame) = decoder.decode_next().await.unwrap() {
                if let Some(expected) = expected.get(&index) {
                    let cpu = frame.to_cpu_rgb().unwrap();
                    // Swift's independent AVAssetImageGenerator observer sets
                    // appliesPreferredTrackTransform=true. VideoIO supplies
                    // coded pixels, so apply this original fixture's pinned
                    // 90-degree matrix independently of EdgeTAM's orientation
                    // implementation before comparing exact displayed bytes.
                    let direct: std::borrow::Cow<'_, [u8]> = if name == "EdgeTAM/rotated-48x32.mp4"
                    {
                        assert_eq!(
                            stream.display_matrix,
                            Some([0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30])
                        );
                        assert_eq!((cpu.width, cpu.height), (48, 32));
                        assert_eq!(
                            (source.source_info().width, source.source_info().height),
                            (32, 48)
                        );
                        let mut displayed = Vec::with_capacity(32 * 48 * 3);
                        for row in 0..48 {
                            for column in 0..32 {
                                let offset = column * cpu.stride + (47 - row) * 3;
                                displayed.extend_from_slice(&cpu.data[offset..offset + 3]);
                            }
                        }
                        std::borrow::Cow::Owned(displayed)
                    } else {
                        std::borrow::Cow::Borrowed(cpu.data.as_ref())
                    };
                    assert_eq!(direct.len(), expected.len(), "displayed byte count {name}");
                    assert!(
                        direct.as_ref() == expected.as_slice(),
                        "exact displayed decoder pixels {name} frame {index}: direct={:016x} expected={:016x}",
                        fnv(direct.as_ref()),
                        fnv(expected)
                    );
                    println!(
                        "EDGETAM_FORWARD {name} frame={index} fnv64={:016x}",
                        fnv(expected)
                    );
                }
                index += 1;
            }
            assert_eq!(index, count, "{name}: complete decode count");
        });
    }
}
