//! Current original-media encoder assertions. No generated replacement video,
//! decoder, encoder or RSS result substitutes for the pinned Swift fixtures.
#![cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
use ferment_engine_core::{
    MultiVideoEncoding, ProgressSink, ThumbStripOptions, ThumbnailFrame, ThumbnailSink,
    VariantThumbnailSink, VideoEncoding, VideoEncodingPreset, VideoEncodingVariant,
    VideoRateControl,
};
use ferment_engine_processors::{VideoEncoder, platform_encode_backends};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
#[path = "support/current_process_memory.rs"]
mod process_memory;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self::under(&std::env::temp_dir())
    }
    fn under(parent: &Path) -> Self {
        std::fs::create_dir_all(parent).unwrap();
        loop {
            let path = parent.join(format!(
                "ferment-current-encoder-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot reserve owned test directory: {error}"),
            }
        }
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture(name: &str) -> PathBuf {
    let root = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(PathBuf::from)
        .expect("original pinned Swift fixture root required");
    let path = root.join("Video").join(name);
    assert!(path.is_file(), "missing original {}", path.display());
    path
}
fn encoder() -> VideoEncoder {
    VideoEncoder::new().with_backends(platform_encode_backends())
}
fn thumb() -> ThumbStripOptions {
    ThumbStripOptions {
        interval: 0.5,
        max_width: 192,
        max_height: 192,
    }
}
fn variant(
    id: &str,
    path: PathBuf,
    preset: VideoEncodingPreset,
    thumb_strip: Option<ThumbStripOptions>,
) -> VideoEncodingVariant {
    VideoEncodingVariant {
        id: id.into(),
        output_path: path,
        preset,
        thumb_strip,
    }
}
fn requested(d: &Directory, strip: bool) -> Vec<VideoEncodingVariant> {
    vec![
        variant(
            "proxy",
            d.file("proxy.mov"),
            VideoEncoder::proxy_preset(),
            strip.then_some(thumb()),
        ),
        variant("ai", d.file("ai.mp4"), VideoEncoder::ai_preset(), None),
    ]
}

#[derive(Debug, Default)]
struct ThumbnailStats {
    cells: Vec<(usize, f64, u32, u32)>,
    failures: usize,
    jpeg_bytes: usize,
}
#[derive(Debug, Default)]
struct JpegSink(Mutex<ThumbnailStats>);
impl JpegSink {
    fn consume(&self, frame: ThumbnailFrame) {
        let mut stats = self.0.lock().unwrap();
        let (width, height) = (frame.frame.width, frame.frame.height);
        let plane = &frame.frame.planes[0];
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = plane.offset + y * plane.stride + x * 4;
                let p = &frame.frame.data[offset..offset + 4];
                rgb.extend_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        let mut bytes = Vec::new();
        if image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 80)
            .encode(&rgb, width, height, image::ExtendedColorType::Rgb8)
            .is_err()
        {
            stats.failures += 1;
            return;
        }
        assert!(bytes.starts_with(&[0xff, 0xd8]) && bytes.ends_with(&[0xff, 0xd9]));
        stats.jpeg_bytes += bytes.len();
        stats
            .cells
            .push((frame.index, frame.presentation_time, width, height));
    }
    fn count(&self) -> usize {
        self.0.lock().unwrap().cells.len()
    }
    fn failures(&self) -> usize {
        self.0.lock().unwrap().failures
    }
}
impl ThumbnailSink for JpegSink {
    fn emit(&self, frame: ThumbnailFrame) {
        self.consume(frame);
    }
}
impl VariantThumbnailSink for JpegSink {
    fn emit(&self, id: &str, frame: ThumbnailFrame) {
        assert_eq!(id, "proxy");
        self.consume(frame);
    }
}
#[derive(Debug, Default)]
struct DiscardSink;
impl VariantThumbnailSink for DiscardSink {
    fn emit(&self, _id: &str, _frame: ThumbnailFrame) {}
}

#[derive(Debug)]
struct Facts {
    path: std::path::PathBuf,
    stream: Value,
    packets: Vec<Value>,
    duration: f64,
}
impl Facts {
    fn count(&self) -> usize {
        self.packets.len()
    }
    fn keys(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| p["flags"].as_str().unwrap().contains('K'))
            .count()
    }
    fn all_intra(&self) -> bool {
        self.count() > 0 && self.count() == self.keys()
    }
    fn width(&self) -> u64 {
        self.stream["width"].as_u64().unwrap()
    }
    fn height(&self) -> u64 {
        self.stream["height"].as_u64().unwrap()
    }
}
fn facts(path: &Path) -> Facts {
    let result = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_packets",
            "-show_format",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("independent ffprobe required");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let json: Value = serde_json::from_slice(&result.stdout).unwrap();
    let stream = json["streams"][0].clone();
    assert!(!stream.is_null());
    let duration = json["format"]["duration"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    Facts {
        path: path.to_owned(),
        stream,
        packets: json["packets"].as_array().unwrap().clone(),
        duration,
    }
}
fn decoded_count(path: &Path) -> usize {
    futures_executor::block_on(async {
        let backends = platform_encode_backends();
        let info = backends.video.probe(path).await.unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = backends
            .video
            .open_decoder(path, stream.index)
            .await
            .unwrap();
        let mut count = 0;
        while decoder.decode_next().await.unwrap().is_some() {
            count += 1;
        }
        count
    })
}
fn parity(shared: &Facts, separate: &Facts) {
    for field in [
        "width",
        "height",
        "codec_tag_string",
        "color_transfer",
        "color_primaries",
        "side_data_list",
    ] {
        assert_eq!(
            shared.stream[field], separate.stream[field],
            "media contract {field}"
        );
    }
    assert_eq!(shared.count(), separate.count());
    assert_eq!(shared.all_intra(), separate.all_intra());
    assert!((shared.duration - separate.duration).abs() <= 0.05);
}
fn hdr(facts: &Facts, transfer: &str) {
    assert_eq!(facts.stream["codec_tag_string"], "hvc1");
    assert_eq!(facts.stream["color_primaries"], "bt2020");
    assert_eq!(facts.stream["color_transfer"], transfer);
    assert_eq!(
        decoded_count(&facts.path),
        facts.count(),
        "every HDR sample must decode through EOS"
    );
    let info = ferment_engine_media::demux::probe(&facts.path).unwrap();
    let configuration = &info
        .video_streams()
        .next()
        .unwrap()
        .decoder_configuration
        .description;
    assert!(
        configuration.len() >= 23 && configuration[0] == 1,
        "hvc1 must carry hvcC"
    );
    let mut cursor = 23;
    let mut complete = [false; 3];
    for _ in 0..configuration[22] {
        let header = configuration[cursor];
        let kind = header & 63;
        let count = u16::from_be_bytes(configuration[cursor + 1..cursor + 3].try_into().unwrap());
        cursor += 3;
        if (32..=34).contains(&kind) {
            assert!(
                header & 128 != 0 && count > 0,
                "hvc1 parameter arrays must be complete"
            );
            complete[(kind - 32) as usize] = true;
        }
        for _ in 0..count {
            let bytes =
                u16::from_be_bytes(configuration[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2;
            assert!(bytes >= 2 && cursor + bytes <= configuration.len());
            assert_eq!((configuration[cursor] >> 1) & 63, kind);
            cursor += bytes;
        }
    }
    assert_eq!(complete, [true; 3]);
    assert_eq!(cursor, configuration.len());
}

/// Kill: diverge shared and separate output contracts or silently lose decoded frames/thumbnails.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs/ffprobe"]
fn shared_decode_matches_both_separate_original_file_outputs() {
    let source = fixture("reel-1/sec1.mp4");
    let d = Directory::new();
    let single = JpegSink::default();
    let shared = JpegSink::default();
    let e = encoder();
    let decoded_source_frames = decoded_count(&source);
    futures_executor::block_on(async {
        e.encode(
            &source,
            &d.file("separate-ai.mp4"),
            &VideoEncoder::ai_preset(),
            None,
            ProgressSink::noop(),
            None,
        )
        .await
        .unwrap();
        e.encode(
            &source,
            &d.file("separate-proxy.mov"),
            &VideoEncoder::proxy_preset(),
            Some(thumb()),
            ProgressSink::noop(),
            Some(&single),
        )
        .await
        .unwrap();
        let result = e
            .encode_variants(
                &source,
                &requested(&d, true),
                ProgressSink::noop(),
                Some(&shared),
            )
            .await
            .unwrap();
        assert_eq!(result.variants.len(), 2);
        assert!(d.file("ai.mp4").exists() && d.file("proxy.mov").exists());
        parity(
            &facts(&d.file("ai.mp4")),
            &facts(&d.file("separate-ai.mp4")),
        );
        parity(
            &facts(&d.file("proxy.mov")),
            &facts(&d.file("separate-proxy.mov")),
        );
        assert_eq!(result.decoded_frames, decoded_source_frames);
        assert_eq!(shared.count(), single.count());
        assert_eq!(shared.failures(), 0);
    });
}

/// Kill: encode a proxy inter-frame GOP or collapse the AI feed to all-intra.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs/ffprobe"]
fn the_actual_shared_proxy_is_all_intra_and_ai_keeps_its_bounded_gop() {
    let d = Directory::new();
    futures_executor::block_on(encoder().encode_variants(
        &fixture("reel-1/sec1.mp4"),
        &requested(&d, false),
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    let proxy = facts(&d.file("proxy.mov"));
    let ai = facts(&d.file("ai.mp4"));
    assert!(proxy.all_intra(), "{}/{} keys", proxy.keys(), proxy.count());
    assert_eq!(proxy.stream["codec_tag_string"], "avc1");
    assert_eq!(ai.stream["codec_tag_string"], "avc1");
    assert!(ai.keys() < ai.count());
    assert!(ai.width().max(ai.height()) <= 640);
    assert!(proxy.width().max(proxy.height()) <= 1080);
    let max_interval = VideoEncoder::ai_preset().key_frame_interval as usize;
    let mut last_key = 0;
    for (index, p) in ai.packets.iter().enumerate() {
        if p["flags"].as_str().unwrap().contains('K') {
            assert!(index - last_key <= max_interval);
            last_key = index;
        }
    }
    assert!(ai.count() - last_key <= max_interval);
}

/// Kill: emit a different inline strip cadence/geometry on shared decode or lose any of the original60 cells.
#[test]
#[ignore = "original thumbstrip-test-30s.mp4 and real platform codecs"]
fn actual_thirty_second_thumbnail_cadence_matches_the_single_pass() {
    let source = fixture("thumbstrip-test-30s.mp4");
    let d = Directory::new();
    let single = JpegSink::default();
    let shared = JpegSink::default();
    futures_executor::block_on(async {
        let e = encoder();
        e.encode(
            &source,
            &d.file("single.mov"),
            &VideoEncoder::proxy_preset(),
            Some(thumb()),
            ProgressSink::noop(),
            Some(&single),
        )
        .await
        .unwrap();
        e.encode_variants(
            &source,
            &requested(&d, true),
            ProgressSink::noop(),
            Some(&shared),
        )
        .await
        .unwrap();
    });
    let a = single.0.lock().unwrap();
    let b = shared.0.lock().unwrap();
    assert_eq!(a.cells.len(), b.cells.len());
    assert_eq!(b.cells.len(), 60);
    assert_eq!(b.failures, 0);
    assert_eq!(
        a.cells.first().map(|c| (c.2, c.3)),
        b.cells.first().map(|c| (c.2, c.3))
    );
    for (a, b) in a.cells.iter().zip(&b.cells) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-9);
    }
}

/// Kill: derive an SDR AI stream while retaining HDR tags, or normalize the shared HDR lead.
#[test]
#[ignore = "original hdr-hlg-12s.mov and real platform codecs/ffprobe"]
fn original_hlg_mixed_eligibility_preserves_proxy_and_normalizes_ai() {
    let d = Directory::new();
    let result = futures_executor::block_on(encoder().encode_variants(
        &fixture("hdr-hlg-12s.mov"),
        &requested(&d, false),
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    assert_eq!(
        result
            .variants
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["ai", "proxy"]
    );
    hdr(&facts(&d.file("proxy.mov")), "arib-std-b67");
    let ai = facts(&d.file("ai.mp4"));
    assert_eq!(ai.stream["codec_tag_string"], "avc1");
    assert_ne!(ai.stream["color_primaries"], "bt2020");
    assert_ne!(ai.stream["color_transfer"], "arib-std-b67");
}

/// Kill: lose HLG/BT2020 or all-intra when the original source fans out to two HDR-preserving sizes.
#[test]
#[ignore = "original hdr-hlg-12s.mov and real platform codecs/ffprobe"]
fn original_hlg_uniform_eligibility_preserves_both_hdr_outputs() {
    let d = Directory::new();
    let mut small = VideoEncoder::proxy_preset();
    small.label = "proxy-small".into();
    small.max_width = 540;
    small.rate_control = VideoRateControl::Average(4_000_000);
    let requested = [
        variant(
            "large",
            d.file("large.mov"),
            VideoEncoder::proxy_preset(),
            None,
        ),
        variant("small", d.file("small.mov"), small, None),
    ];
    futures_executor::block_on(encoder().encode_variants(
        &fixture("hdr-hlg-12s.mov"),
        &requested,
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    for name in ["large.mov", "small.mov"] {
        let p = facts(&d.file(name));
        hdr(&p, "arib-std-b67");
        assert!(p.all_intra());
    }
}

struct Sampler {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<Vec<u64>>>,
}
impl Sampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let worker = thread::spawn(move || {
            let mut series = Vec::new();
            while !flag.load(Ordering::Acquire) {
                series.push(process_memory::footprint_bytes());
                thread::sleep(Duration::from_millis(100));
            }
            series
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }
    fn finish(mut self) -> Vec<u64> {
        self.stop.store(true, Ordering::Release);
        self.worker.take().unwrap().join().unwrap()
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Kill: retain duration-sized decoded pictures; the last-third physical footprint must not rise256MiB.
#[test]
#[ignore = "original djs-walking-city.mp4; real platform codecs; fresh process"]
fn actual_shared_decode_preserves_the_original_256mib_growth_bound() {
    let source = fixture("djs-walking-city.mp4");
    let d = Directory::new();
    let sampler = Sampler::start();
    futures_executor::block_on(encoder().encode_variants(
        &source,
        &requested(&d, true),
        ProgressSink::noop(),
        Some(&DiscardSink),
    ))
    .unwrap();
    let series = sampler.finish();
    assert!(
        series.len() >= 8,
        "unqualified: fewer than the original8 physical-footprint samples"
    );
    let third = series.len() / 3;
    let mean = |v: &[u64]| v.iter().map(|x| u128::from(*x)).sum::<u128>() / v.len() as u128;
    let first = mean(&series[..third]) as i128;
    let last = mean(&series[series.len() - third..]) as i128;
    println!(
        "CURRENT_ENCODER_MEMORY samples={} first={} last={} growth={}",
        series.len(),
        first,
        last,
        last - first
    );
    assert!(
        last - first < 256 * 1024 * 1024,
        "footprint growth {}",
        last - first
    );
}

/// Kill: report a real SDR source as HDR and select an unnecessary10-bit codec path.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform demux"]
fn the_original_sdr_reel_has_no_hdr_transfer() {
    let info = futures_executor::block_on(
        platform_encode_backends()
            .video
            .probe(&fixture("reel-1/sec1.mp4")),
    )
    .unwrap();
    assert_eq!(
        VideoEncoder::detect_hdr_transfer(info.video_streams().next().unwrap().color),
        None
    );
}

/// Kill: return a tiny/nonexistent AI file or exceed the original640-pixel longer-axis cap.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn the_original_reel_encodes_a_valid_ai_file() {
    let d = Directory::new();
    let result = futures_executor::block_on(encoder().encode(
        &fixture("reel-1/sec1.mp4"),
        &d.file("ai.mp4"),
        &VideoEncoder::ai_preset(),
        None,
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    assert!(result.bytes > 1024);
    assert!(result.width.max(result.height) <= 640);
}

/// Kill: drop the source HLG transfer or inline strip while encoding the real HDR proxy.
#[test]
#[ignore = "original hdr-hlg-12s.mov and real platform codecs/ffprobe"]
fn the_original_hlg_proxy_preserves_transfer_primaries_and_thumbnails() {
    let d = Directory::new();
    let sink = JpegSink::default();
    let options = ThumbStripOptions {
        interval: 1.0,
        max_width: 96,
        max_height: 96,
    };
    let result = futures_executor::block_on(encoder().encode(
        &fixture("hdr-hlg-12s.mov"),
        &d.file("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        Some(options),
        ProgressSink::noop(),
        Some(&sink),
    ))
    .unwrap();
    assert!(result.bytes > 1024);
    assert!(sink.count() > 0);
    assert_eq!(sink.failures(), 0);
    hdr(&facts(&d.file("proxy.mov")), "arib-std-b67");
}

/// Kill: coerce the original PQ source to HLG or lose its BT2020/HEVC metadata.
#[test]
#[ignore = "original hdr-pq-15s.mp4 and real platform codecs/ffprobe"]
fn the_original_pq_proxy_preserves_its_own_transfer_and_primaries() {
    let d = Directory::new();
    let sink = JpegSink::default();
    let options = ThumbStripOptions {
        interval: 1.0,
        max_width: 96,
        max_height: 96,
    };
    futures_executor::block_on(encoder().encode(
        &fixture("hdr-pq-15s.mp4"),
        &d.file("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        Some(options),
        ProgressSink::noop(),
        Some(&sink),
    ))
    .unwrap();
    hdr(&facts(&d.file("proxy.mov")), "smpte2084");
}

/// Kill: return a missing/tiny proxy, wrong output URL, excessive dimensions or unsolicited thumbnails.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn the_original_reel_encodes_a_valid_proxy_file() {
    let d = Directory::new();
    let output = d.file("proxy.mov");
    let result = futures_executor::block_on(encoder().encode(
        &fixture("reel-1/sec1.mp4"),
        &output,
        &VideoEncoder::proxy_preset(),
        None,
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    assert!(output.exists());
    assert_eq!(result.output_path, output);
    assert!(result.bytes > 1024);
    assert!(result.width.max(result.height) <= 1080);
    assert_eq!(result.thumbnails_emitted, 0);
}

/// Kill: omit or overproduce inline thumbnails from the original29-second source.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs/ffprobe"]
fn the_original_reel_emits_the_expected_one_second_strip_count() {
    let source = fixture("reel-1/sec1.mp4");
    let duration = facts(&source).duration;
    assert!(duration > 5.0);
    let expected = duration as isize;
    let d = Directory::new();
    let sink = JpegSink::default();
    let options = ThumbStripOptions {
        interval: 1.0,
        max_width: 96,
        max_height: 96,
    };
    let result = futures_executor::block_on(encoder().encode(
        &source,
        &d.file("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        Some(options),
        ProgressSink::noop(),
        Some(&sink),
    ))
    .unwrap();
    assert_eq!(result.thumbnails_emitted, sink.count());
    assert!((sink.count() as isize - expected).abs() <= 2);
    assert_eq!(sink.failures(), 0);
}

// Warm the same real codec and container doors before the measured region,
// matching the source benchmark's explicit VideoEncoder.prewarm boundary.
fn prewarm(directory: &Directory) {
    use ferment_engine_media::{
        Codec, ColorInfo, CpuFrame, CpuPixelFormat, EncoderSettings, FrameBuffer, MediaTime,
        MuxSettings, TimeBase, VideoFrame, VideoMuxTrack,
    };
    futures_executor::block_on(async {
        let backends = platform_encode_backends();
        let base = TimeBase::new_unchecked(1, 30);
        let mut encoder = backends
            .video
            .open_encoder(EncoderSettings {
                frame_rate: Some((30, 1)),
                ..EncoderSettings::new(Codec::H264, 64, 64, base)
            })
            .await
            .unwrap();
        let frame = VideoFrame {
            width: 64,
            height: 64,
            pts: MediaTime::ZERO,
            duration: Some(MediaTime::new(1, base)),
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(
                CpuFrame::new(
                    64,
                    64,
                    64 * 3,
                    CpuPixelFormat::Rgb24,
                    vec![127; 64 * 64 * 3],
                )
                .unwrap(),
            ),
        };
        encoder
            .push_frame(frame.encodable_scaled(64, 64, frame.color).unwrap())
            .await
            .unwrap();
        let encoded = encoder.finalize().await.unwrap();
        let path = directory.file("warmup.mp4");
        let mut mux = backends.muxer.open(&path, MuxSettings::default()).unwrap();
        let track = mux
            .add_video_track(VideoMuxTrack {
                display_matrix: None,
                configuration: encoded.configuration,
                width: 64,
                height: 64,
                time_base: base,
                color: ColorInfo::default(),
                duration: Some(MediaTime::new(1, base)),
            })
            .unwrap();
        for packet in encoded.packets {
            mux.write_packet(track, packet).unwrap();
        }
        mux.finalize().unwrap();
        std::fs::remove_file(path).unwrap();
    });
}

/// Kill: record a benchmark without consuming real JPEG cells or preserving every source sample in each requested variant.
#[test]
#[ignore = "explicit original benchmark environment and real codecs/ffprobe; one mode per process"]
fn the_current_encoder_benchmark_records_before_asserting_media_contracts() {
    assert_eq!(
        std::env::var("FERMENT_VIDEO_ENCODER_BENCHMARK").unwrap(),
        "1"
    );
    let required = |name| {
        std::env::var(name).expect("complete current encoder benchmark environment required")
    };
    let mode = required("FERMENT_VIDEO_ENCODER_BENCHMARK_MODE");
    assert!(["baseline", "multi", "proxy-old", "proxy-multi"].contains(&mode.as_str()));
    let source = PathBuf::from(required("FERMENT_VIDEO_ENCODER_BENCHMARK_INPUT"));
    assert!(source.is_file());
    let root = PathBuf::from(required("FERMENT_VIDEO_ENCODER_BENCHMARK_OUTPUT_ROOT"));
    let result_path = PathBuf::from(required("FERMENT_VIDEO_ENCODER_BENCHMARK_RESULT"));
    let trial: u32 = required("FERMENT_VIDEO_ENCODER_BENCHMARK_TRIAL")
        .parse()
        .unwrap();
    let d = Directory::under(&root);
    let sink = JpegSink::default();
    let e = encoder();
    let produces_ai = mode == "baseline" || mode == "multi";
    prewarm(&d);
    let sampler = Sampler::start();
    let started = Instant::now();
    let mut decoded = None;
    let mut stages = Vec::new();
    futures_executor::block_on(async {
        if mode == "baseline" {
            let start = Instant::now();
            e.encode(
                &source,
                &d.file("ai.mp4"),
                &VideoEncoder::ai_preset(),
                None,
                ProgressSink::noop(),
                None,
            )
            .await
            .unwrap();
            stages.push(json!({"id":"ai","wallSeconds":start.elapsed().as_secs_f64()}));
        }
        let start = Instant::now();
        if mode == "baseline" || mode == "proxy-old" {
            e.encode(
                &source,
                &d.file("proxy.mov"),
                &VideoEncoder::proxy_preset(),
                Some(thumb()),
                ProgressSink::noop(),
                Some(&sink),
            )
            .await
            .unwrap();
        } else {
            let mut wanted = requested(&d, true);
            if !produces_ai {
                wanted.truncate(1);
            }
            let result = e
                .encode_variants(&source, &wanted, ProgressSink::noop(), Some(&sink))
                .await
                .unwrap();
            decoded = Some(result.decoded_frames);
        }
        stages.push(json!({"id":if mode=="multi"{"shared"}else{"proxy"},"wallSeconds":start.elapsed().as_secs_f64()}));
    });
    let wall = started.elapsed().as_secs_f64();
    let memory = sampler.finish();
    let source_facts = facts(&source);
    let proxy = facts(&d.file("proxy.mov"));
    let ai = produces_ai.then(|| facts(&d.file("ai.mp4")));
    let stats = sink.0.lock().unwrap();
    let record = json!({"schemaVersion":2,"mode":mode,"trial":trial,"source":{"path":source,"frameCount":source_facts.count(),"durationSeconds":source_facts.duration,"width":source_facts.width(),"height":source_facts.height()},"totalWallSeconds":wall,"stages":stages,"decodedFrames":decoded,"realTimeFactor":source_facts.duration/wall,"variants":{"proxy":{"frames":proxy.count(),"width":proxy.width(),"height":proxy.height()},"ai":ai.as_ref().map(|v|json!({"frames":v.count(),"width":v.width(),"height":v.height()}))},"thumbnails":{"consumed":stats.cells.len(),"jpegBytes":stats.jpeg_bytes,"jpegFailures":stats.failures,"quality":80,"compressor":"portable image JPEG"},"memory":{"counter":if cfg!(target_os="macos"){"TASK_VM_INFO.phys_footprint"}else{"VmRSS"},"sampleIntervalMilliseconds":100,"sampleCount":memory.len(),"baselineFootprintBytes":memory.first(),"peakFootprintBytes":memory.iter().max()},"platformDiagnostics":{"cpuAndThermal":"not reported by this portable harness; no Apple counter equivalence claimed"}});
    std::fs::write(&result_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    println!("CURRENT_ENCODER_BENCHMARK {}", result_path.display());
    assert_eq!(
        proxy.count(),
        source_facts.count(),
        "proxy dropped source samples"
    );
    if let Some(ai) = ai {
        assert_eq!(
            ai.count(),
            source_facts.count(),
            "AI dropped source samples"
        );
    }
    assert_eq!(stats.failures, 0);
    assert!(!stats.cells.is_empty());
}
