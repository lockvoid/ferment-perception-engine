//! End-to-end proof of the real encode pipelines against platform backends
//! and the engine-media fixture corpus: decode → scale → encode → mux →
//! probe-back. macOS-gated; the Windows box runs the same suite through
//! `scripts/sync run` with Media Foundation backends.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ferment_engine_core::{
    AudioEncoding, AudioEncodingPreset, MultiVideoEncoding, ProgressSink, ThumbStripOptions,
    ThumbnailFrame, ThumbnailSink, VariantThumbnailSink, VideoEncoding, VideoEncodingVariant,
    VideoRateControl,
};
use ferment_engine_media::platform::libav::LibavAudioIo;
use ferment_engine_media::platform::macos::VideoToolboxVideoIo;
use ferment_engine_media::{
    Codec, ColorPrimaries, EncoderSettings, MediaKind, TransferCharacteristic,
};
use ferment_engine_processors::{AudioEncoder, EncodeBackends, SourceVideo, VideoEncoder};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-media/tests/fixtures/generated")
        .join(name)
}

fn backends() -> EncodeBackends {
    EncodeBackends {
        video: Arc::new(VideoToolboxVideoIo),
        audio: Arc::new(LibavAudioIo),
        muxer: Arc::new(ferment_engine_media::platform::libav::LibavMuxer),
    }
}

/// An encoded output under the system temp directory, removed when the test
/// drops it — including when the test fails part-way, which is when a plain
/// tail cleanup would leak.
struct TempOutput(PathBuf);

impl TempOutput {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("ferment-encode-e2e-{}-{name}", std::process::id()));
        Self(path)
    }
}

impl std::ops::Deref for TempOutput {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempOutput {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn temp_output(name: &str) -> TempOutput {
    TempOutput::new(name)
}

fn probe(path: &Path) -> ferment_engine_media::MediaInfo {
    futures_executor::block_on(ferment_engine_media::VideoIo::probe(
        &VideoToolboxVideoIo,
        path,
    ))
    .expect("output probes")
}

fn video_stream(path: &Path) -> ferment_engine_media::StreamInfo {
    probe(path)
        .streams
        .into_iter()
        .find(|stream| stream.kind == MediaKind::Video)
        .expect("output has video")
}

/// The major brand from the container's leading `ftyp` box.
fn ftyp_brand(path: &Path) -> [u8; 4] {
    let bytes = std::fs::read(path).expect("output readable");
    assert_eq!(
        &bytes[4..8],
        b"ftyp",
        "output does not start with an ftyp box"
    );
    bytes[8..12].try_into().expect("4 brand bytes")
}

#[derive(Debug, Default)]
struct CountingSink(Mutex<Vec<ThumbnailFrame>>);

impl ThumbnailSink for CountingSink {
    fn emit(&self, thumbnail: ThumbnailFrame) {
        self.0.lock().unwrap().push(thumbnail);
    }
}

#[derive(Debug, Default)]
struct VariantSink(Mutex<Vec<(String, ThumbnailFrame)>>);

impl VariantThumbnailSink for VariantSink {
    fn emit(&self, variant_id: &str, thumbnail: ThumbnailFrame) {
        self.0
            .lock()
            .unwrap()
            .push((variant_id.to_owned(), thumbnail));
    }
}

#[test]
fn ai_preset_transcodes_a_real_h264_fixture_and_probes_back() {
    let output = temp_output("ai.mp4");
    let encoder = VideoEncoder::new().with_backends(backends());
    let result = futures_executor::block_on(encoder.encode(
        &fixture("h264-gop4.mp4"),
        &output,
        &VideoEncoder::ai_preset(),
        None,
        ProgressSink::noop(),
        None,
    ))
    .expect("ai transcode succeeds");
    assert!(result.bytes > 0);
    assert!(result.width <= 640 && result.height <= 640);
    assert_eq!(result.thumbnails_emitted, 0);

    let video = video_stream(&output);
    assert_eq!(video.codec, Codec::H264);
    assert_eq!(video.width, Some(result.width));
    assert_eq!(video.height, Some(result.height));
}

#[test]
fn proxy_preset_emits_thumbnails_during_the_single_decode_pass() {
    let output = temp_output("proxy.mov");
    let sink = CountingSink::default();
    let encoder = VideoEncoder::new().with_backends(backends());
    let result = futures_executor::block_on(encoder.encode(
        &fixture("h264-gop4.mp4"),
        &output,
        &VideoEncoder::proxy_preset(),
        Some(ThumbStripOptions {
            interval: 0.5,
            max_width: 192,
            max_height: 192,
        }),
        ProgressSink::noop(),
        Some(&sink),
    ))
    .expect("proxy transcode succeeds");
    let thumbnails = sink.0.lock().unwrap();
    assert_eq!(result.thumbnails_emitted, thumbnails.len());
    assert!(!thumbnails.is_empty(), "proxy pass emits thumbnails");
    assert!(thumbnails.iter().all(|thumb| {
        thumb.frame.width <= 192 && thumb.frame.height <= 192 && !thumb.frame.data.is_empty()
    }));
    // Indices and presentation times are monotonic.
    assert!(
        thumbnails
            .windows(2)
            .all(|pair| pair[0].presentation_time < pair[1].presentation_time
                && pair[0].index + 1 == pair[1].index)
    );
}

/// The production shape: one decode, both derivatives, the strip tapped from
/// the same frames. Replaces the per-derivative decode the operators used to
/// pay for.
#[test]
fn one_decode_feeds_every_variant_and_the_strip() {
    let proxy_output = temp_output("variants-proxy.mov");
    let ai_output = temp_output("variants-ai.mp4");
    let sink = VariantSink::default();
    let encoder = VideoEncoder::new().with_backends(backends());
    let variants = vec![
        VideoEncodingVariant {
            id: "proxy".to_owned(),
            output_path: proxy_output.to_path_buf(),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: Some(ThumbStripOptions {
                interval: 0.25,
                max_width: 192,
                max_height: 192,
            }),
        },
        VideoEncodingVariant {
            id: "ai".to_owned(),
            output_path: ai_output.to_path_buf(),
            preset: VideoEncoder::ai_preset(),
            thumb_strip: None,
        },
    ];
    let result = futures_executor::block_on(encoder.encode_variants(
        &fixture("h264-gop4.mp4"),
        &variants,
        ProgressSink::noop(),
        Some(&sink),
    ))
    .expect("multi-variant transcode succeeds");

    assert_eq!(result.variants.len(), 2);
    assert!(result.decoded_frames > 0);
    assert!(result.wall_seconds > 0.0);

    let proxy = &result.variants["proxy"];
    let ai = &result.variants["ai"];
    assert_eq!(proxy.output_path, *proxy_output);
    assert_eq!(ai.output_path, *ai_output);
    assert!(proxy.bytes > 0 && ai.bytes > 0);
    assert_eq!(ai.thumbnails_emitted, 0, "only the proxy asked for a strip");
    assert!(proxy.thumbnails_emitted > 0);

    // Both containers exist and hold the codec their preset states.
    assert_eq!(video_stream(&proxy_output).codec, Codec::H264);
    assert_eq!(video_stream(&ai_output).codec, Codec::H264);
    // The container brand comes from the preset's file type: the proxy is a
    // QuickTime movie, the AI feed an MP4.
    assert_eq!(&ftyp_brand(&proxy_output), b"qt  ");
    assert_eq!(&ftyp_brand(&ai_output), b"isom");

    // Every emitted cell is attributed to the variant that asked for it, and
    // tapped from the shared decode rather than from the proxy's own output.
    let emitted = sink.0.lock().unwrap();
    assert_eq!(emitted.len(), proxy.thumbnails_emitted);
    assert!(emitted.iter().all(|(id, _)| id == "proxy"));
}

/// An HDR source splits: the HDR-preserving variants share the decode, and the
/// SDR variant is derived from the finished proxy. The derived output must be
/// tagged BT.709 — inheriting the source's BT.2020/PQ would describe
/// normalized pixels with the source's colorimetry.
#[test]
fn an_hdr_source_splits_the_decode_and_tags_the_derived_variant_bt709() {
    let proxy_output = temp_output("hdr-proxy.mov");
    let ai_output = temp_output("hdr-ai.mp4");
    let source = fixture("synthetic-hdr-nclx.mp4");
    let source_stream = video_stream(&source);
    assert_eq!(source_stream.color.transfer, TransferCharacteristic::Pq);
    assert_eq!(source_stream.color.primaries, ColorPrimaries::Bt2020);

    let encoder = VideoEncoder::new().with_backends(backends());
    let variants = vec![
        VideoEncodingVariant {
            id: "proxy".to_owned(),
            output_path: proxy_output.to_path_buf(),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: None,
        },
        VideoEncodingVariant {
            id: "ai".to_owned(),
            output_path: ai_output.to_path_buf(),
            preset: VideoEncoder::ai_preset(),
            thumb_strip: None,
        },
    ];
    let result = futures_executor::block_on(encoder.encode_variants(
        &source,
        &variants,
        ProgressSink::noop(),
        None,
    ))
    .expect("split transcode succeeds");
    assert_eq!(result.variants.len(), 2);

    let proxy = video_stream(&proxy_output);
    assert_eq!(proxy.codec, Codec::Hevc, "the HDR proxy stays HEVC");
    assert_eq!(proxy.color.transfer, TransferCharacteristic::Pq);
    assert_eq!(proxy.color.primaries, ColorPrimaries::Bt2020);

    let ai = video_stream(&ai_output);
    assert_eq!(ai.codec, Codec::H264, "the AI feed is always SDR H.264");
    assert_eq!(
        ai.color.transfer,
        TransferCharacteristic::Bt709,
        "the derived SDR variant must not inherit the source transfer"
    );
    assert_eq!(ai.color.primaries, ColorPrimaries::Bt709);

    // The shared pass decoded the source once; the SDR variant re-read the
    // finished proxy, so it did not add to that count.
    let source_frames = result.decoded_frames;
    assert!(source_frames > 0);
}

/// "SDR variants encode FROM the finished proxy" is only observable when the
/// two sources differ in size, so the HDR variant here is capped small: an AI
/// feed derived from a 32x24 proxy cannot be bigger than 32x24, while one that
/// re-read the 64x48 source would come out at 64x48. Second-generation quality
/// is the accepted price for decoding the source once.
#[test]
fn the_sdr_variant_of_an_hdr_source_is_derived_from_the_finished_proxy() {
    let proxy_output = temp_output("derive-proxy.mov");
    let ai_output = temp_output("derive-ai.mp4");
    let mut small_proxy = VideoEncoder::proxy_preset();
    small_proxy.max_width = 32;
    small_proxy.caps_short_edge = false;

    let encoder = VideoEncoder::new().with_backends(backends());
    let result = futures_executor::block_on(encoder.encode_variants(
        &fixture("synthetic-hdr-nclx.mp4"),
        &[
            VideoEncodingVariant {
                id: "proxy".to_owned(),
                output_path: proxy_output.to_path_buf(),
                preset: small_proxy,
                thumb_strip: None,
            },
            VideoEncodingVariant {
                id: "ai".to_owned(),
                output_path: ai_output.to_path_buf(),
                preset: VideoEncoder::ai_preset(),
                thumb_strip: None,
            },
        ],
        ProgressSink::noop(),
        None,
    ))
    .expect("split transcode succeeds");

    assert_eq!(
        (
            result.variants["proxy"].width,
            result.variants["proxy"].height
        ),
        (32, 24)
    );
    assert_eq!(
        (result.variants["ai"].width, result.variants["ai"].height),
        (32, 24),
        "the SDR variant re-read the source instead of the finished proxy"
    );
}

/// The 10-bit read-back path, measured against the real buffer VideoToolbox
/// hands us for an HDR proxy (`x420`, 10-bit codes right-aligned in 16-bit
/// words). A flat mid-grey encoded as HEVC Main10 with BT.2020/PQ tags comes
/// back at 77-78 rather than 128, and that is NOT a narrowing bug — the same
/// grey through the 8-bit lane with the SAME tags round-trips to exactly
/// 128.0, so the shift is VideoToolbox inverse-tone-mapping display-referred
/// RGB into PQ on the encode side. Taking the high byte of each word instead
/// of dividing by four reads the frame as near-black, so this band is a live
/// guard on the layout.
#[test]
fn a_ten_bit_hdr_output_reads_back_at_the_measured_pq_level() {
    use ferment_engine_media::{
        ColorInfo, ColorMatrix, ColorRange, CpuFrame, CpuPixelFormat, MediaTime, MuxFormat,
        MuxSettings, Muxer, TimeBase, VideoMuxTrack,
    };

    const FED_GREY: u8 = 128;
    let time_base = TimeBase::new_unchecked(1, 600);
    let grey = CpuFrame::new(
        64,
        48,
        64 * 3,
        CpuPixelFormat::Rgb24,
        vec![FED_GREY; 64 * 48 * 3],
    )
    .unwrap();

    let mut means = Vec::new();
    for (label, codec, profile, bits) in [
        ("h264-8bit", Codec::H264, "high", 8u8),
        ("hevc-10bit", Codec::Hevc, "main10", 10),
    ] {
        let color = ColorInfo {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Pq,
            matrix: ColorMatrix::Bt2020NonConstant,
            bits_per_channel: Some(bits),
        };
        let mut encoder = futures_executor::block_on(ferment_engine_media::VideoIo::open_encoder(
            &VideoToolboxVideoIo,
            EncoderSettings {
                frame_rate: Some((12, 1)),
                bit_rate: Some(20_000_000),
                profile: Some(profile.to_owned()),
                max_keyframe_interval: Some(1),
                allow_frame_reordering: false,
                color: Some(color),
                // HEVC only: VideoToolbox answers `kVTParameterErr` (-12902)
                // when an H.264 session is asked to insert HDR metadata, and
                // that refusal is surfaced rather than swallowed.
                insert_hdr_metadata: bits == 10,
                ..EncoderSettings::new(codec, 64, 48, time_base)
            },
        ))
        .expect("the HDR encoder session configures");
        for index in 0..6i64 {
            let frame = ferment_engine_media::VideoFrame::encodable_from_cpu(
                &grey,
                64,
                48,
                MediaTime::new(index * 50, time_base),
                Some(MediaTime::new(50, time_base)),
                color,
            )
            .unwrap();
            futures_executor::block_on(encoder.push_frame(frame)).unwrap();
        }
        let encoded = futures_executor::block_on(encoder.finalize()).unwrap();

        let output = temp_output(&format!("grey-{label}.mov"));
        let mut session = Muxer::open(
            &ferment_engine_media::platform::libav::LibavMuxer,
            &output,
            MuxSettings {
                format: MuxFormat::Mov,
                ..MuxSettings::default()
            },
        )
        .unwrap();
        let track = session
            .add_video_track(VideoMuxTrack {
                display_matrix: None,
                configuration: encoded.configuration.clone(),
                width: 64,
                height: 48,
                time_base,
                color,
                duration: None,
            })
            .unwrap();
        for packet in encoded.packets {
            session.write_packet(track, packet).unwrap();
        }
        session.finalize().unwrap();

        let stream = video_stream(&output);
        let mut decoder = futures_executor::block_on(ferment_engine_media::VideoIo::open_decoder(
            &VideoToolboxVideoIo,
            &output,
            stream.index,
        ))
        .unwrap();
        let frame = futures_executor::block_on(decoder.decode_next())
            .unwrap()
            .expect("the HDR output decodes back");
        let cpu = frame
            .to_cpu_rgb()
            .expect("a ten-bit HDR buffer converts to CPU RGB");
        let sum: u64 = cpu.data.iter().map(|byte| u64::from(*byte)).sum();
        means.push((label, sum as f64 / cpu.data.len() as f64));
    }

    let eight_bit = means[0].1;
    let ten_bit = means[1].1;
    assert!(
        (eight_bit - f64::from(FED_GREY)).abs() < 1.0,
        "the 8-bit lane must round-trip the fed grey, got {eight_bit}"
    );
    assert!(
        (60.0..100.0).contains(&ten_bit),
        "ten-bit read-back left its measured band (77.7): {ten_bit}"
    );
}

/// Determinism, measured rather than asserted in the abstract: the rate is a
/// pure function of the probed geometry and frame rate, so two independent
/// probes of the same file resolve the same number.
#[test]
fn the_resolved_proxy_bit_rate_is_identical_across_independent_probes() {
    let source = fixture("h264-gop4.mp4");
    let preset = VideoEncoder::proxy_preset();
    let first = SourceVideo::from_stream(&video_stream(&source)).unwrap();
    let second = SourceVideo::from_stream(&video_stream(&source)).unwrap();
    assert_eq!(first, second);

    let plan_one = VideoEncoder::plan(first, &preset);
    let plan_two = VideoEncoder::plan(second, &preset);
    assert_eq!(plan_one, plan_two);

    // And it is the formula, not a stored constant.
    let VideoRateControl::BitsPerPixel(bits_per_pixel) = preset.rate_control else {
        panic!("the proxy is bits-per-pixel");
    };
    let expected = (bits_per_pixel
        * f64::from(plan_one.width)
        * f64::from(plan_one.height)
        * first.frames_per_second) as u64;
    assert_eq!(plan_one.bit_rate, expected);
    assert!(first.frames_per_second > 0.0, "the fixture states a rate");
}

/// The reordering knob, measured on the real encoder. With it off, delivery
/// order equals presentation order — the property the scene detectors index
/// against. Both values must be accepted by the session, so a platform that
/// refuses the property fails here instead of silently reordering.
#[test]
fn videotoolbox_delivers_presentation_order_when_reordering_is_off() {
    let source = fixture("h264-gop4.mp4");
    let stream = video_stream(&source);

    for allow_frame_reordering in [false, true] {
        let mut decoder = futures_executor::block_on(ferment_engine_media::VideoIo::open_decoder(
            &VideoToolboxVideoIo,
            &source,
            stream.index,
        ))
        .expect("decoder opens");
        let mut encoder = futures_executor::block_on(ferment_engine_media::VideoIo::open_encoder(
            &VideoToolboxVideoIo,
            EncoderSettings {
                frame_rate: stream.frame_rate,
                bit_rate: Some(2_500_000),
                profile: Some("high".to_owned()),
                max_keyframe_interval: Some(60),
                allow_frame_reordering,
                ..EncoderSettings::new(
                    Codec::H264,
                    stream.width.unwrap(),
                    stream.height.unwrap(),
                    stream.time_base,
                )
            },
        ))
        .expect("the encoder accepts the reordering property in both directions");

        while let Some(frame) =
            futures_executor::block_on(decoder.decode_next()).expect("decode succeeds")
        {
            let cpu = frame.to_cpu_rgb().expect("frame converts");
            let encodable = ferment_engine_media::VideoFrame::encodable_from_cpu(
                &cpu,
                stream.width.unwrap(),
                stream.height.unwrap(),
                frame.pts,
                frame.duration,
                frame.color,
            )
            .expect("frame uploads");
            futures_executor::block_on(encoder.push_frame(encodable)).expect("frame encodes");
        }
        let encoded = futures_executor::block_on(encoder.finalize()).expect("encode finalizes");
        assert!(!encoded.packets.is_empty());

        if !allow_frame_reordering {
            let mut previous = f64::NEG_INFINITY;
            for packet in &encoded.packets {
                let pts = packet.pts.expect("encoded packet carries a pts").seconds();
                assert!(
                    pts > previous,
                    "reordering-off must emit strictly increasing presentation times"
                );
                previous = pts;
                if let Some(dts) = packet.dts {
                    assert!(
                        (dts.seconds() - pts).abs() < 1e-9,
                        "reordering-off must not offset composition times"
                    );
                }
            }
        }
    }
}

#[test]
fn audio_only_sources_reencode_to_aac_m4a() {
    let output = temp_output("audio.m4a");
    // A 48 kHz source with a 44.1 kHz preset target forces the real re-encode
    // path (the passthrough branch requires an exact target match).
    let encoder = AudioEncoder::new().with_backends(backends());
    let result = futures_executor::block_on(encoder.encode(
        &fixture("aac-lc-48000-stereo.mp4"),
        &output,
        AudioEncodingPreset::default(),
        ProgressSink::noop(),
    ))
    .expect("audio transcode succeeds");
    assert!(result.bytes > 0);
    assert!(!result.did_passthrough);
    assert!(result.duration > 0.0);

    let info = ferment_engine_media::demux::probe(&output).expect("m4a probes");
    let audio = info
        .streams
        .iter()
        .find(|stream| stream.kind == MediaKind::Audio)
        .expect("output has audio");
    assert_eq!(audio.codec, Codec::Aac);
}

/// Wall-clock for the production request (proxy + ai + thumb strip) on a
/// real clip: `FERMENT_BENCH_CLIP=/path/to.mp4 cargo test -p
/// ferment-engine-processors --test encode_pipeline_e2e bench_ -- --ignored
/// --nocapture`. `FERMENT_BENCH_VARIANTS=proxy|ai|both` (default both) and
/// `FERMENT_BENCH_THUMBS=0` narrow the request to attribute the time.
/// Ignored: it needs a clip and prints, it asserts nothing.
#[test]
#[ignore]
fn bench_encode_variants_on_env_clip() {
    let Ok(clip) = std::env::var("FERMENT_BENCH_CLIP") else {
        eprintln!("FERMENT_BENCH_CLIP not set");
        return;
    };
    let which = std::env::var("FERMENT_BENCH_VARIANTS").unwrap_or_else(|_| "both".into());
    let thumbs = std::env::var("FERMENT_BENCH_THUMBS")
        .ok()
        .is_none_or(|value| value != "0");
    let sink = VariantSink::default();
    let proxy = TempOutput::new("bench-proxy.mp4");
    let ai = TempOutput::new("bench-ai.mp4");
    let mut variants = Vec::new();
    if which != "ai" {
        variants.push(VideoEncodingVariant {
            id: "proxy".into(),
            output_path: proxy.to_path_buf(),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: thumbs.then_some(ThumbStripOptions {
                max_width: 160,
                max_height: 160,
                interval: 1.0,
            }),
        });
    }
    if which != "proxy" {
        variants.push(VideoEncodingVariant {
            id: "ai".into(),
            output_path: ai.to_path_buf(),
            preset: VideoEncoder::ai_preset(),
            thumb_strip: None,
        });
    }
    let encoder = VideoEncoder::new().with_backends(backends());
    let started = std::time::Instant::now();
    let result = futures_executor::block_on(encoder.encode_variants(
        Path::new(&clip),
        &variants,
        ProgressSink::noop(),
        Some(&sink),
    ))
    .expect("encode_variants");
    eprintln!(
        "BENCH encode_variants [{which}, thumbs={thumbs}]: {:.2}s wall, {} frames decoded, {} thumbnails, {} variants",
        started.elapsed().as_secs_f64(),
        result.decoded_frames,
        sink.0.lock().unwrap().len(),
        result.variants.len()
    );
}
