//! File delivery through installed ONNX graphs and platform codecs, decoded by
//! the independently installed FFmpeg CLI. No codec/runtime seam is mocked.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]
use ferment_engine_core::{
    self as core, ProgressSink, SlowMotionRendering, VideoNoiseFiltering, VideoUpscaling,
};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::video_upscaler::UpscaleMode;
use ferment_engine_processors::{SlowMotionRenderer, VideoNoiseFilter, VideoUpscaler};
use ferment_engine_test_fixtures::export;
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn backend() -> Backend {
    let master = |name| {
        export::artifact(name).unwrap_or_else(|| {
            panic!("{name} is required; set FERMENT_ENGINE_MODELS_DIR to the published fleet")
        })
    };
    let paths = ModelPaths::new()
        .with_path(
            ModelId::RealesrGeneralX4,
            ModelVersion::Latest,
            master("realesr_general_x4v3.onnx"),
        )
        .with_path(
            ModelId::RifeV425,
            ModelVersion::Latest,
            master("rife_v4_25.onnx"),
        )
        .with_path(
            ModelId::Fastdvdnet,
            ModelVersion::Latest,
            master("fastdvdnet.onnx"),
        );
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}
fn run(tool: &str, args: &[&str]) -> Vec<u8> {
    let result = Command::new(tool)
        .args(args)
        .output()
        .expect("independent media oracle must be installed");
    assert!(
        result.status.success(),
        "{tool}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}
fn paths(name: &str) -> (PathBuf, PathBuf) {
    let dir = root()
        .join("target/render-delivery")
        .join(format!("{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    (dir.join("source.mp4"), dir.join("output.mp4"))
}
fn video(name: &str) -> (PathBuf, PathBuf) {
    video_at_rate(name, 10)
}
fn video_at_rate(name: &str, frame_rate: u32) -> (PathBuf, PathBuf) {
    let paths = paths(name);
    let picture = format!("testsrc2=size=64x48:rate={frame_rate}:duration=1");
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &picture,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=997:sample_rate=48000:duration=1",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            paths.0.to_str().unwrap(),
        ],
    );
    paths
}
fn probe(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path.to_str().unwrap(),
        ],
    ))
    .unwrap()
}
fn assert_video(path: &Path, width: u32, height: u32, frames: &str, audio: bool) {
    let json = probe(path);
    let streams = json["streams"].as_array().unwrap();
    let video = streams.iter().find(|s| s["codec_type"] == "video").unwrap();
    assert_eq!(video["width"], width);
    assert_eq!(video["height"], height);
    assert_eq!(video["nb_read_frames"], frames);
    assert_eq!(streams.iter().any(|s| s["codec_type"] == "audio"), audio);
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ],
    );
}

/// Kill: skip Compact inference and upscale ordinary pixels; independent frozen Compact pixels diverge.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn image_delivery_matches_the_independent_compact_golden() {
    let (source, output) = paths("image");
    let source = source.with_extension("png");
    // The export's independent real-frame checkpoint pair, not the retired
    // local FP16 converter's synthetic output. The delivery gate stays 30 dB.
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(export::require_path("realesr/model.json")).unwrap())
            .unwrap();
    let crop = u32::try_from(contract["goldens"]["crop"].as_u64().unwrap()).unwrap();
    let shown = |planes: Vec<f32>| {
        assert_eq!(planes.len() % 3, 0);
        let pixels = planes.len() / 3;
        (0..planes.len())
            .map(|index| {
                (planes[(index % 3) * pixels + index / 3].clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect::<Vec<_>>()
    };
    let pixels = shown(export::require_f32_le(
        "realesr/goldens/realesr_street_lr_in.f32",
    ));
    image::RgbImage::from_raw(crop, crop, pixels)
        .unwrap()
        .save(&source)
        .unwrap();
    let size =
        futures_executor::block_on(ferment_engine_processors::render_pipeline::upscale_image(
            backend(),
            &source,
            &output,
            UpscaleMode::Upscale {
                longest_side: crop * 4,
            },
            ProgressSink::noop(),
        ))
        .unwrap();
    assert_eq!((size.width, size.height), (crop * 4, crop * 4));
    let actual = run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
    );
    let expected = shown(export::require_f32_le(
        "realesr/goldens/realesr_general_x4v3_street_sr_out.f32",
    ));
    assert_eq!(actual.len(), expected.len());
    let mse = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    let psnr = 10.0 * (255.0f64.powi(2) / mse).log10();
    eprintln!("image JPEG95 vs independently frozen Compact pixels: {psnr:.2} dB (floor 30 dB)");
    assert!(psnr >= 30.0);
}

/// Kill: return the native 4x dimensions without the requested final resample.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn upscale_honors_final_size_and_preserves_audio() {
    let (source, output) = video("upscale");
    let result = futures_executor::block_on(VideoUpscaler::new().with_backend(backend()).render(
        &source,
        &output,
        core::VideoUpscaleMode::Upscale { longest_side: 96 },
        None,
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(result.frames_written, 10);
    assert_video(&output, 96, 72, "10", true);
}

/// Kill: route Enhance to the previous unconditional unsupported guard or native 4x delivery.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn enhance_delivers_source_dimensions() {
    let (source, output) = video("enhance");
    futures_executor::block_on(VideoUpscaler::new().with_backend(backend()).render(
        &source,
        &output,
        core::VideoUpscaleMode::Enhance,
        None,
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_video(&output, 64, 48, "10", true);
}

/// Kill: leave the file noise-filter façade permanently unsupported, or lose its scene/window frames.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn denoise_window_is_decodable_and_keeps_selected_audio() {
    let (source, output) = video("denoise");
    let range = core::MediaTimeRange {
        start: core::MediaTime::new(2, 10),
        duration: core::MediaTime::new(5, 10),
    };
    futures_executor::block_on(VideoNoiseFilter::new().with_backend(backend()).render(
        &source,
        &output,
        0.5,
        Some(range),
        &[core::MediaTime::new(4, 10)],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_video(&output, 64, 48, "5", true);
}

/// Kill: omit RIFE interpolation from the delivered stream; only ten source frames are written.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn slow_motion_writes_real_intermediates_to_the_stretched_timeline() {
    let (source, output) = video("slow");
    let result =
        futures_executor::block_on(SlowMotionRenderer::new().with_backend(backend()).render(
            &source,
            &output,
            2,
            None,
            &[],
            core::MotionBlurMode::Disabled,
            180.0,
            ProgressSink::noop(),
        ))
        .unwrap();
    assert_eq!(result.frames_written, 19);
    assert!((result.output_seconds - 1.8).abs() < 0.000_01);
    assert_video(&output, 64, 48, "19", false);
    let stream = run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
    );
    let frame = 64 * 48 * 3;
    assert_ne!(
        &stream[..frame],
        &stream[frame..2 * frame],
        "an interpolated slot must not duplicate its anchor"
    );
}

/// Kill: discard explicit blur strength while adapting FFI/core mode; the result reports a different request.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn blur_strength_reaches_the_file_renderer() {
    let (source, output) = video("blur");
    let result =
        futures_executor::block_on(SlowMotionRenderer::new().with_backend(backend()).render(
            &source,
            &output,
            2,
            None,
            &[],
            core::MotionBlurMode::Enabled { strength: 27 },
            180.0,
            ProgressSink::noop(),
        ))
        .unwrap();
    let decision = result.motion_blur.unwrap();
    assert!(decision.enabled);
    assert_eq!(decision.strength, 27);
    assert_eq!(decision.reason, "explicitOn");
    assert_video(&output, 64, 48, "19", false);
}

/// Kill: fail to check cancellation before commit; the cancelled operation publishes a playable file.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn cancelled_upscale_leaves_no_output_or_partial() {
    let (source, output) = video("cancel");
    let sink = ProgressSink::noop();
    sink.cancel();
    let error = futures_executor::block_on(VideoUpscaler::new().with_backend(backend()).render(
        &source,
        &output,
        core::VideoUpscaleMode::Enhance,
        None,
        &[],
        sink,
    ))
    .unwrap_err();
    assert!(matches!(error, core::EngineError::Cancelled));
    assert!(!output.exists());
    assert!(!output.with_file_name("output.partial.mp4").exists());
}

/// Kill: strip the selected window's audio, the old upscale-only audio omission.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn upscale_trims_audio_with_the_video_window() {
    let (source, output) = video("upscale-trim");
    let range = core::MediaTimeRange {
        start: core::MediaTime::new(2, 10),
        duration: core::MediaTime::new(5, 10),
    };
    futures_executor::block_on(VideoUpscaler::new().with_backend(backend()).render(
        &source,
        &output,
        core::VideoUpscaleMode::Enhance,
        Some(range),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_video(&output, 64, 48, "5", true);
    let json = probe(&output);
    let duration = json["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["codec_type"] == "audio")
        .unwrap()["duration"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!(
        (duration - 0.5).abs() < 0.025,
        "trimmed audio duration {duration}"
    );
}

/// Kill: discard the phone video's display matrix while re-encoding coded pixels.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn rotated_phone_video_is_delivered_upright() {
    let (source, output) = video("rotated");
    let rotated = source.with_file_name("phone.mp4");
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            source.to_str().unwrap(),
            "-c",
            "copy",
            rotated.to_str().unwrap(),
        ],
    );
    futures_executor::block_on(VideoUpscaler::new().with_backend(backend()).render(
        &rotated,
        &output,
        core::VideoUpscaleMode::Enhance,
        None,
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_video(&output, 48, 64, "10", true);
    let pixels = |path: &Path| {
        run(
            "ffmpeg",
            &[
                "-nostdin",
                "-v",
                "error",
                "-i",
                path.to_str().unwrap(),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
        )
    };
    let (actual, expected) = (pixels(&output), pixels(&rotated));
    assert_eq!(actual.len(), expected.len());
    let mse = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    let psnr = 10.0 * (255.0f64.powi(2) / mse).log10();
    eprintln!(
        "upright enhanced pixels against independently display-oriented source: {psnr:.2} dB (floor 20 dB)"
    );
    assert!(psnr >= 20.0, "wrong display orientation: {psnr:.2} dB");
}

/// Kill: check cancellation only before the first frame and still commit after Stop.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn cancelling_an_active_model_render_never_commits() {
    let (source, output) = video("active-cancel");
    let holder = Arc::new(std::sync::Mutex::new(None::<ProgressSink>));
    let callback = holder.clone();
    let progress = ProgressSink::from_callback(move |_| {
        callback.lock().unwrap().as_ref().unwrap().cancel();
    });
    *holder.lock().unwrap() = Some(progress.clone());
    let result =
        futures_executor::block_on(SlowMotionRenderer::new().with_backend(backend()).render(
            &source,
            &output,
            2,
            None,
            &[],
            core::MotionBlurMode::Disabled,
            180.0,
            progress,
        ));
    holder.lock().unwrap().take();
    assert!(
        matches!(result, Err(core::EngineError::Cancelled)),
        "{result:?}"
    );
    assert!(!output.exists());
    assert!(
        !output.parent().unwrap().read_dir().unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial"))
    );
}

/// Kill: ignore the selected half-second window, fail to stretch PTS, retain
/// source audio, report an unfinished/regressing progress value, or leave a
/// partial file. The checks inspect real RIFE output through independent tools.
#[test]
#[ignore = "requires installed ONNX Runtime, model artifacts and ffmpeg/ffprobe"]
fn a_windowed_slow_render_preserves_frame_rate_duration_and_atomic_delivery() {
    let (source, output) = video_at_rate("slow-window-current", 30);
    let ticks = Arc::new(std::sync::Mutex::new(Vec::new()));
    let collected = ticks.clone();
    let progress = ProgressSink::from_callback(move |event| {
        collected.lock().unwrap().push(event.progress);
    });
    let result =
        futures_executor::block_on(SlowMotionRenderer::new().with_backend(backend()).render(
            &source,
            &output,
            2,
            Some(core::MediaTimeRange {
                start: core::MediaTime::new(2, 10),
                duration: core::MediaTime::new(5, 10),
            }),
            &[],
            core::MotionBlurMode::Disabled,
            180.0,
            progress,
        ))
        .unwrap();
    assert!(output.is_file());
    assert!(!output.parent().unwrap().read_dir().unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")
    }));
    let before = probe(&source);
    let after = probe(&output);
    let original = before["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|stream| stream["codec_type"] == "video")
        .unwrap();
    let streams = after["streams"].as_array().unwrap();
    let delivered = streams
        .iter()
        .find(|stream| stream["codec_type"] == "video")
        .unwrap();
    let ratio = |value: &serde_json::Value| {
        let (numerator, denominator) = value.as_str().unwrap().split_once('/').unwrap();
        numerator.parse::<f64>().unwrap() / denominator.parse::<f64>().unwrap()
    };
    let source_fps = ratio(&original["avg_frame_rate"]);
    let output_fps = ratio(&delivered["avg_frame_rate"]);
    assert!((output_fps - source_fps).abs() <= 2.0_f64.max(source_fps * 0.1));
    let duration = after["format"]["duration"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!(
        (duration - 1.0).abs() <= 0.2,
        "actual output duration {duration}"
    );
    assert_eq!(result.frames_written % 2, 1);
    assert!(result.frames_written > 10);
    assert_eq!(
        delivered["nb_read_frames"]
            .as_str()
            .unwrap()
            .parse::<usize>()
            .unwrap(),
        result.frames_written
    );
    assert!(!streams.iter().any(|stream| stream["codec_type"] == "audio"));
    let ticks = ticks.lock().unwrap();
    assert_eq!(ticks.last(), Some(&1.0));
    assert!(ticks.windows(2).all(|pair| pair[0] <= pair[1]), "{ticks:?}");
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ],
    );
}
