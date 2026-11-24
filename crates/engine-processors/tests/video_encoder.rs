//! Preset, geometry, rate-control and encoder-settings contracts.
//!
//! Ported from `VideoEncoderTests.swift` plus the regressions the 2026-08
//! proxy recut left behind. Everything here is media-free; the real
//! decode → encode → mux pass is `encode_pipeline_e2e`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_core::{
    EngineError, MultiVideoEncoding, ProgressSink, ThumbStripOptions, ThumbnailFrame,
    ThumbnailSink, VideoEncoding, VideoEncodingVariant, VideoFileType, VideoRateControl,
};
use ferment_engine_media::{Codec, ColorInfo, ColorMatrix, ColorPrimaries, TransferCharacteristic};
use ferment_engine_processors::{
    AtomicFileDelivery, HdrTransfer, SourceVideo, VideoEncoder, VideoEncodingError, VideoProfile,
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ferment-video-encoder-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sdr() -> ColorInfo {
    ColorInfo {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferCharacteristic::Bt709,
        matrix: ColorMatrix::Bt709,
        ..ColorInfo::default()
    }
}

fn hdr(transfer: TransferCharacteristic) -> ColorInfo {
    ColorInfo {
        primaries: ColorPrimaries::Bt2020,
        transfer,
        matrix: ColorMatrix::Bt2020NonConstant,
        bits_per_channel: Some(10),
        ..ColorInfo::default()
    }
}

fn source(width: u32, height: u32, frames_per_second: f64, color: ColorInfo) -> SourceVideo {
    SourceVideo {
        width,
        height,
        frames_per_second,
        color,
    }
}

fn thumb_strip() -> ThumbStripOptions {
    ThumbStripOptions {
        interval: 1.0,
        max_width: 96,
        max_height: 96,
    }
}

// MARK: - Preset geometry

/// The proxy caps the SHORT edge (a portrait 4K proxies to 1080x1920, the NLE
/// half-resolution convention and never smaller than a phone display); the AI
/// feed caps the LONG edge at 640 for its model consumers.
#[test]
fn proxy_scales_by_short_edge_and_ai_by_long_edge() {
    let proxy = VideoEncoder::proxy_preset();
    let ai = VideoEncoder::ai_preset();
    assert!(proxy.caps_short_edge);
    assert!(!ai.caps_short_edge);

    assert_eq!(
        VideoEncoder::scaled_size(2_160, 3_840, proxy.max_width, proxy.caps_short_edge),
        (1_080, 1_920)
    );
    assert_eq!(
        VideoEncoder::scaled_size(2_160, 3_840, ai.max_width, ai.caps_short_edge),
        (360, 640)
    );
}

/// Regression for the long-axis proxy (fixed upstream 2026-08-25): capping the
/// LONG edge of a portrait 4K put the proxy at 606x1080 — softer than the
/// display it upscales onto. The proxy must be 1080 WIDE.
#[test]
fn portrait_four_k_proxies_at_1080_wide_not_606() {
    let preset = VideoEncoder::proxy_preset();
    let (width, height) =
        VideoEncoder::scaled_size(2_160, 3_840, preset.max_width, preset.caps_short_edge);
    assert_eq!(width, 1_080, "the proxy caps the short edge, not the long");
    assert_eq!(height, 1_920);

    // What the bug produced, kept as the counter-example.
    assert_eq!(
        VideoEncoder::scaled_size(2_160, 3_840, preset.max_width, false),
        (606, 1_080)
    );
}

#[test]
fn scaled_size_never_upscales_and_stays_even() {
    // Landscape 4K: short edge 2160 -> 1080, long edge follows.
    assert_eq!(
        VideoEncoder::scaled_size(3_840, 2_160, 1_080, true),
        (1_920, 1_080)
    );
    // Already under the cap: untouched.
    assert_eq!(VideoEncoder::scaled_size(720, 900, 1_080, true), (720, 900));
    // Odd geometry floors to even in both axes.
    assert_eq!(
        VideoEncoder::scaled_size(1_001, 667, 500, false),
        (500, 332)
    );
    // A degenerate source falls back to the square cap.
    assert_eq!(VideoEncoder::scaled_size(0, 0, 640, false), (640, 640));
    // A sub-two-pixel result is clamped rather than handed to the encoder.
    assert_eq!(VideoEncoder::scaled_size(1, 1, 1, false), (2, 2));
}

// MARK: - Rate control

/// The proxy is bitrate-driven everywhere, and the bitrate is COMPUTED:
/// 0.75 bpp x output pixels x source fps. No quality probe exists to make the
/// file platform-dependent.
#[test]
fn proxy_bit_rate_is_deterministic_bits_per_pixel() {
    let preset = VideoEncoder::proxy_preset();
    assert_eq!(
        preset.rate_control,
        VideoRateControl::BitsPerPixel(0.75),
        "the proxy resolves its rate, it does not request a quality"
    );

    let plan = VideoEncoder::plan(source(2_160, 3_840, 60.0, sdr()), &preset);
    assert_eq!((plan.width, plan.height), (1_080, 1_920));
    assert_eq!(plan.bit_rate, (0.75 * 1_080.0 * 1_920.0 * 60.0) as u64);

    // The HDR branch carries the same computed average.
    let hdr_plan = VideoEncoder::plan(
        source(2_160, 3_840, 60.0, hdr(TransferCharacteristic::Hlg)),
        &preset,
    );
    assert_eq!(hdr_plan.bit_rate, plan.bit_rate);
    assert_eq!(hdr_plan.codec, Codec::Hevc);
}

/// A container that states no usable frame rate resolves at the phone-standard
/// 30 rather than collapsing the proxy to a zero bitrate.
#[test]
fn a_source_with_no_stated_frame_rate_resolves_the_proxy_at_thirty() {
    let preset = VideoEncoder::proxy_preset();
    let at_thirty = VideoEncoder::plan(source(2_160, 3_840, 30.0, sdr()), &preset);
    for unusable in [0.0, -24.0, f64::NAN, f64::INFINITY] {
        let plan = VideoEncoder::plan(source(2_160, 3_840, unusable, sdr()), &preset);
        assert_eq!(
            plan.bit_rate, at_thirty.bit_rate,
            "frame rate {unusable} must fall back to 30"
        );
        assert!(plan.bit_rate > 0);
    }
}

#[test]
fn the_ai_feed_holds_a_fixed_average_regardless_of_geometry() {
    let preset = VideoEncoder::ai_preset();
    assert_eq!(preset.rate_control, VideoRateControl::Average(2_500_000));
    for (width, height, fps) in [(2_160, 3_840, 60.0), (640, 360, 24.0), (1_920, 1_080, 0.0)] {
        let plan = VideoEncoder::plan(source(width, height, fps, sdr()), &preset);
        assert_eq!(plan.bit_rate, 2_500_000);
    }
}

#[test]
fn planning_the_same_source_twice_resolves_the_same_rate() {
    let preset = VideoEncoder::proxy_preset();
    let first = VideoEncoder::plan(source(2_160, 3_840, 59.94, sdr()), &preset);
    let second = VideoEncoder::plan(source(2_160, 3_840, 59.94, sdr()), &preset);
    assert_eq!(first, second);
    assert_eq!(first.bit_rate, 93_218_688);
}

// MARK: - Encoder settings

#[test]
fn the_proxy_preset_is_all_intra_with_reordering_off() {
    let preset = VideoEncoder::proxy_preset();
    assert_eq!(preset.file_type, VideoFileType::Mov);
    assert_eq!(preset.max_width, 1_080);
    assert_eq!(preset.b_frames, 0);
    assert_eq!(preset.key_frame_interval, 1);
    assert!(preset.hdr_eligible);

    let plan = VideoEncoder::plan(source(720, 900, 30.0, sdr()), &preset);
    assert_eq!((plan.width, plan.height), (720, 900));
    assert_eq!(plan.profile, VideoProfile::H264HighAutoLevel);
    assert_eq!(plan.max_key_frame_interval, Some(1));
    assert!(!plan.allow_frame_reordering);
}

/// The point of the AI preset. Its GOP is bounded, not all-intra, and a
/// bounded GOP is exactly the case where an encoder's default (VideoToolbox
/// documents `true`) would silently reintroduce B-frames. The scene detectors
/// index frames by delivery position and cannot survive reordering.
#[test]
fn the_ai_preset_states_frame_reordering_off_on_a_bounded_gop() {
    let preset = VideoEncoder::ai_preset();
    assert_eq!(preset.file_type, VideoFileType::Mp4);
    assert_eq!(preset.max_width, 640);
    assert_eq!(preset.b_frames, 0);
    assert_eq!(preset.key_frame_interval, 60);
    assert!(!preset.hdr_eligible);

    let plan = VideoEncoder::plan(source(720, 900, 30.0, sdr()), &preset);
    assert_eq!((plan.width, plan.height), (512, 640));
    assert_eq!(plan.max_key_frame_interval, Some(60));
    assert!(
        !plan.allow_frame_reordering,
        "the bounded-GOP AI feed must state reordering OFF"
    );
    assert_eq!(plan.profile, VideoProfile::H264HighAutoLevel);
}

#[test]
fn both_stock_presets_state_reordering_off_through_the_settings_builders() {
    for settings in [
        VideoEncoder::sdr_video_settings(640, 360, 2_500_000, 0, 60),
        VideoEncoder::sdr_video_settings(1_080, 1_920, 8_000_000, 0, 1),
        VideoEncoder::hdr_video_settings(1_080, 1_920, 8_000_000, 0, 1, HdrTransfer::Hlg),
    ] {
        assert!(!settings.allow_frame_reordering);
    }
    // A preset that DOES want B-frames says so, and drops the keyframe bound
    // to the encoder — the upstream `else` branch.
    let reordered = VideoEncoder::sdr_video_settings(640, 360, 2_500_000, 2, 60);
    assert!(reordered.allow_frame_reordering);
    assert_eq!(reordered.max_key_frame_interval, None);
    // `key_frame_interval == 0` means "encoder's choice" even without B-frames.
    assert_eq!(
        VideoEncoder::sdr_video_settings(640, 360, 2_500_000, 0, 0).max_key_frame_interval,
        None
    );
}

#[test]
fn hdr_settings_tag_hlg_and_pq_as_bt2020_main10() {
    for (transfer, expected) in [
        (HdrTransfer::Hlg, TransferCharacteristic::Hlg),
        (HdrTransfer::Pq, TransferCharacteristic::Pq),
    ] {
        let settings = VideoEncoder::hdr_video_settings(1_080, 1_920, 8_000_000, 0, 1, transfer);
        assert_eq!(settings.codec, Codec::Hevc);
        assert_eq!(settings.profile, VideoProfile::HevcMain10AutoLevel);
        assert_eq!(settings.color.primaries, ColorPrimaries::Bt2020);
        assert_eq!(settings.color.transfer, expected);
        assert_eq!(settings.color.matrix, ColorMatrix::Bt2020NonConstant);
        assert_eq!(settings.color.bits_per_channel, Some(10));
        assert!(settings.insert_hdr_metadata);
    }
}

/// The SDR path STATES BT.709 instead of leaving the tags to be inherited.
/// An SDR variant of an HDR source that inherits 2020/HLG describes washed
/// pixels with the source's colorimetry (fixed upstream in `203deb5d9`).
#[test]
fn sdr_settings_state_bt709_rather_than_inheriting_source_tags() {
    let settings = VideoEncoder::sdr_video_settings(640, 360, 2_500_000, 0, 60);
    assert_eq!(settings.codec, Codec::H264);
    assert_eq!(settings.color.primaries, ColorPrimaries::Bt709);
    assert_eq!(settings.color.transfer, TransferCharacteristic::Bt709);
    assert_eq!(settings.color.matrix, ColorMatrix::Bt709);
    assert!(!settings.insert_hdr_metadata);

    // Planning the AI feed off an HDR source lands on exactly those tags.
    let plan = VideoEncoder::plan(
        source(1_080, 1_920, 30.0, hdr(TransferCharacteristic::Hlg)),
        &VideoEncoder::ai_preset(),
    );
    assert_eq!(plan.codec, Codec::H264);
    assert_eq!(plan.color.primaries, ColorPrimaries::Bt709);
    assert_eq!(plan.color.transfer, TransferCharacteristic::Bt709);
}

#[test]
fn detect_hdr_transfer_reads_hlg_and_pq_only() {
    assert_eq!(VideoEncoder::detect_hdr_transfer(sdr()), None);
    assert_eq!(
        VideoEncoder::detect_hdr_transfer(hdr(TransferCharacteristic::Hlg)),
        Some(HdrTransfer::Hlg)
    );
    assert_eq!(
        VideoEncoder::detect_hdr_transfer(hdr(TransferCharacteristic::Pq)),
        Some(HdrTransfer::Pq)
    );
}

#[test]
fn the_hdr_proxy_preserves_its_transfer() {
    for transfer in [TransferCharacteristic::Hlg, TransferCharacteristic::Pq] {
        let plan = VideoEncoder::plan(
            source(1_080, 1_920, 30.0, hdr(transfer)),
            &VideoEncoder::proxy_preset(),
        );
        assert_eq!(plan.codec, Codec::Hevc);
        assert_eq!(plan.color.transfer, transfer);
        assert_eq!(plan.color.primaries, ColorPrimaries::Bt2020);
    }
}

// MARK: - Request validation

#[test]
fn a_thumb_strip_without_a_handler_is_refused() {
    let error = VideoEncoder::validate_thumb_strip(Some(thumb_strip()), false).unwrap_err();
    assert_eq!(error, VideoEncodingError::ThumbStripWithoutHandler);
    assert!(VideoEncoder::validate_thumb_strip(Some(thumb_strip()), true).is_ok());
    assert!(VideoEncoder::validate_thumb_strip(None, false).is_ok());
}

fn variant(id: &str, thumb_strip: Option<ThumbStripOptions>) -> VideoEncodingVariant {
    VideoEncodingVariant {
        id: id.to_owned(),
        output_path: PathBuf::from(format!("/tmp/{id}.mov")),
        preset: VideoEncoder::proxy_preset(),
        thumb_strip,
    }
}

#[test]
fn a_multi_variant_request_needs_variants_unique_ids_and_a_thumb_handler() {
    assert_eq!(
        VideoEncoder::validate_variants(&[], true).unwrap_err(),
        VideoEncodingError::NoVariants
    );
    assert_eq!(
        VideoEncoder::validate_variants(&[variant("proxy", None), variant("proxy", None)], true)
            .unwrap_err(),
        VideoEncodingError::DuplicateVariantId("proxy".to_owned())
    );
    assert_eq!(
        VideoEncoder::validate_variants(&[variant("proxy", Some(thumb_strip()))], false)
            .unwrap_err(),
        VideoEncodingError::ThumbStripWithoutHandler
    );
    assert!(
        VideoEncoder::validate_variants(
            &[variant("proxy", Some(thumb_strip())), variant("ai", None)],
            true
        )
        .is_ok()
    );
}

// MARK: - Atomic delivery

/// Kill: remove the completed destination when admitting a replacement or abandoning its stage.
#[test]
fn an_abandoned_replacement_preserves_the_completed_file() {
    let directory = TestDirectory::new("old-completed");
    let output = directory.join("sound.m4a");
    std::fs::write(&output, b"completed original").unwrap();
    let delivery = AtomicFileDelivery::begin(&output).unwrap();
    std::fs::write(delivery.partial_path(), b"unfinished replacement").unwrap();
    assert_eq!(delivery.partial_path().extension().unwrap(), "m4a");
    assert_eq!(std::fs::read(&output).unwrap(), b"completed original");
    drop(delivery);
    assert_eq!(std::fs::read(&output).unwrap(), b"completed original");
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
}

/// Kill: use one shared partial name; an abandoned attempt then erases another attempt's bytes.
#[test]
fn two_attempts_own_separate_stages_and_late_cleanup_preserves_the_winner() {
    let directory = TestDirectory::new("separate-attempts");
    let output = directory.join("proxy.mov");
    let first = AtomicFileDelivery::begin(&output).unwrap();
    std::fs::write(first.partial_path(), b"abandoned attempt").unwrap();
    let second = AtomicFileDelivery::begin(&output).unwrap();
    std::fs::write(second.partial_path(), b"winning attempt").unwrap();
    assert_ne!(first.partial_path(), second.partial_path());
    assert_eq!(
        std::fs::read(first.partial_path()).unwrap(),
        b"abandoned attempt"
    );
    second.commit().unwrap();
    drop(first);
    assert_eq!(std::fs::read(&output).unwrap(), b"winning attempt");
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
}

/// Kill: promote an already completed stage after its actual work scope was cancelled.
#[test]
fn cancellation_before_commit_keeps_old_bytes_and_removes_only_its_stage() {
    use std::sync::{Arc, atomic::AtomicBool};
    let directory = TestDirectory::new("cancelled-commit");
    let output = directory.join("ai.mp4");
    std::fs::write(&output, b"previous finished output").unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let work = ferment_engine_core::WorkActivity::from_callback(|_| {})
        .with_cancellation(Arc::clone(&cancelled));
    let _scope = work.enter();
    let delivery = AtomicFileDelivery::begin(&output).unwrap();
    std::fs::write(delivery.partial_path(), b"late finished output").unwrap();
    cancelled.store(true, Ordering::SeqCst);
    assert!(matches!(delivery.commit(), Err(EngineError::Cancelled)));
    assert_eq!(std::fs::read(&output).unwrap(), b"previous finished output");
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
}

#[test]
fn the_output_path_is_invisible_until_the_encode_completes() {
    let directory = TestDirectory::new("atomic");
    let output = directory.join("proxy.mov");
    let worker_output = output.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel();

    let worker = std::thread::spawn(move || {
        AtomicFileDelivery::write(&worker_output, |partial| {
            std::fs::write(partial, vec![7_u8; 2_048])?;
            ready_tx.send(()).unwrap();
            finish_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            Ok(())
        })
    });

    let ready = ready_rx.recv_timeout(std::time::Duration::from_secs(5));
    let invisible = !output.exists();
    let released = finish_tx.send(());
    let finished = worker.join();
    ready.unwrap();
    released.unwrap();
    finished.unwrap().unwrap();
    assert!(
        invisible,
        "VideoEncoder must deliver output_path atomically"
    );

    assert!(output.exists());
    assert!(std::fs::metadata(output).unwrap().len() > 1_024);
}

#[test]
fn cancelling_mid_encode_delivers_neither_output_nor_partial() {
    let directory = TestDirectory::new("cancel");
    let output = directory.join("proxy.mov");

    let result = AtomicFileDelivery::write(&output, |staging_path| {
        std::fs::write(staging_path, vec![9_u8; 2_048])?;
        Err::<(), _>(EngineError::Cancelled)
    });

    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(!output.exists(), "cancelled encode delivered output");
    assert_eq!(
        std::fs::read_dir(&directory.0).unwrap().count(),
        0,
        "cancelled encode left a file"
    );
}

// MARK: - Capability boundary without backends

#[derive(Debug)]
struct NoopThumbnailSink;

impl ThumbnailSink for NoopThumbnailSink {
    fn emit(&self, _thumbnail: ThumbnailFrame) {}
}

#[test]
fn the_encode_trait_validates_the_thumb_strip_before_the_backend_boundary() {
    let directory = TestDirectory::new("trait-validation");
    let encoder = VideoEncoder::new();
    let service: &dyn VideoEncoding = &encoder;
    let error = futures_executor::block_on(service.encode(
        &directory.join("source.mp4"),
        &directory.join("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        Some(thumb_strip()),
        ProgressSink::noop(),
        None,
    ))
    .unwrap_err();

    match error {
        EngineError::InvalidConfiguration(message) => assert_eq!(
            message,
            "VideoEncoder: thumbStrip options provided but onThumbnail handler is nil"
        ),
        other => panic!("expected thumb-strip validation error, got {other}"),
    }
}

#[test]
fn a_backendless_encoder_reports_the_boundary_and_writes_nothing() {
    let directory = TestDirectory::new("trait-mux");
    let output = directory.join("proxy.mov");
    let encoder = VideoEncoder::new();
    let sink = NoopThumbnailSink;

    for thumb_strip_options in [None, Some(thumb_strip())] {
        let handler: Option<&dyn ThumbnailSink> =
            thumb_strip_options.map(|_| &sink as &dyn ThumbnailSink);
        let error = futures_executor::block_on(VideoEncoding::encode(
            &encoder,
            &directory.join("source.mp4"),
            &output,
            &VideoEncoder::proxy_preset(),
            thumb_strip_options,
            ProgressSink::noop(),
            handler,
        ))
        .unwrap_err();
        assert!(matches!(
            error,
            EngineError::Unsupported(message) if message == VideoEncoder::MUX_UNAVAILABLE
        ));
    }
    assert!(!output.exists(), "unsupported encode delivered output");
    assert_eq!(
        std::fs::read_dir(&directory.0).unwrap().count(),
        0,
        "unsupported encode left a file"
    );
}

#[test]
fn a_backendless_multi_variant_encode_validates_first_then_reports_the_boundary() {
    let encoder = VideoEncoder::new();
    let service: &dyn MultiVideoEncoding = &encoder;

    let invalid = futures_executor::block_on(service.encode_variants(
        &PathBuf::from("source.mp4"),
        &[],
        ProgressSink::noop(),
        None,
    ))
    .unwrap_err();
    assert!(matches!(invalid, EngineError::InvalidConfiguration(_)));

    let boundary = futures_executor::block_on(service.encode_variants(
        &PathBuf::from("source.mp4"),
        &[variant("proxy", None)],
        ProgressSink::noop(),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        boundary,
        EngineError::Unsupported(message) if message == VideoEncoder::MUX_UNAVAILABLE
    ));
}
