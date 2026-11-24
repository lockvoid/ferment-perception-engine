#![cfg(any(target_os = "macos", target_os = "windows"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ferment_engine_core::{
    ProgressSink, VideoMinimapComposing, VideoSceneDetecting, VideoSceneDetectionResult,
};
use ferment_engine_infer::{Backend, InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_media::VideoBackend;
use ferment_engine_processors::{VideoMinimapComposer, VideoSceneDetector, video_minimap_composer};
use video_minimap_composer::VideoMinimapComposerConfig;

#[cfg(all(target_os = "macos", feature = "coreml", not(feature = "ort")))]
use ferment_engine_infer::{CoreMlBackend, ModelPaths};

#[derive(Debug)]
struct NoCuts;

#[async_trait]
impl InferenceBackend for NoCuts {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        _inputs: TensorMap,
    ) -> ferment_engine_core::Result<TensorMap> {
        let mut outputs = TensorMap::new();
        match model {
            ModelId::Autoshot => {
                outputs.insert(
                    "one_hot".into(),
                    Tensor::from_f32(vec![1, 100], vec![-20.0; 100])?,
                );
            }
            // Scene detection asks the identity encoder for every frame now.
            // An all-zero vector is a legitimate abstention, which keeps this
            // test about video IO sequencing rather than about embeddings.
            ModelId::Dinov2Vits14 => {
                outputs.insert(
                    "embedding".into(),
                    Tensor::from_f32(vec![1, 384], vec![0.0; 384])?,
                );
            }
            other => panic!("scene detection asked for an unexpected model: {other}"),
        }
        Ok(outputs)
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-media/tests/fixtures/generated/h264-gop4.mp4")
}

fn video_backend() -> VideoBackend {
    #[cfg(target_os = "macos")]
    return Arc::new(ferment_engine_media::platform::macos::VideoToolboxVideoIo);
    #[cfg(target_os = "windows")]
    return Arc::new(ferment_engine_media::platform::libav::LibavVideoIo);
}

#[cfg(all(target_os = "macos", feature = "coreml", not(feature = "ort")))]
fn coreml_backend() -> Backend {
    let directory = std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ferment-cuts-ios/Packages/FermentEngine/LocalModels")
        });
    let model = directory.join("autoshot.mlmodelc");
    assert!(
        model.is_dir(),
        "AutoShot model not found at {}; set FERMENT_ENGINE_MODELS_DIR",
        model.display()
    );
    let mut paths = ModelPaths::new();
    paths.insert(ModelId::Autoshot, ModelVersion::Latest, model);
    Arc::new(CoreMlBackend::new(Arc::new(paths)))
}

#[cfg(any(feature = "ort", all(target_os = "macos", feature = "coreml")))]
fn current_model_backend() -> Backend {
    #[cfg(feature = "ort")]
    return streaming_profile_backend();
    #[cfg(all(not(feature = "ort"), target_os = "macos", feature = "coreml"))]
    return coreml_backend();
}

fn require_opt_in(name: &str) {
    assert_eq!(
        std::env::var(name).as_deref(),
        Ok("1"),
        "manual scenario is disabled; set {name}=1 deliberately"
    );
}

fn required_path(name: &str) -> PathBuf {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} to the real media path"));
    assert!(path.is_file(), "{name} does not exist: {}", path.display());
    path
}

#[test]
fn scene_detection_and_minimap_consume_sequential_video_io() {
    futures_executor::block_on(async {
        let detected = VideoSceneDetecting::detect(
            &VideoSceneDetector::new(Arc::new(NoCuts)).with_video_backend(video_backend()),
            &fixture(),
            ProgressSink::noop(),
        )
        .await
        .unwrap();
        assert_eq!(detected.total_frames, 12);
        assert_eq!(detected.scenes.len(), 1);

        let composer = VideoMinimapComposer::new(VideoMinimapComposerConfig {
            max_width: 640,
            format: video_minimap_composer::ImageFormat::Png,
            ..VideoMinimapComposerConfig::default()
        })
        .with_video_backend(video_backend());
        let minimap = VideoMinimapComposing::compose(
            &composer,
            &fixture(),
            &VideoSceneDetectionResult {
                scenes: detected.scenes,
                total_frames: detected.total_frames,
                frame_rate: detected.frame_rate,
            },
            ProgressSink::noop(),
        )
        .await
        .unwrap();
        assert_eq!(&minimap.image_data[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(minimap.layout.cells.len(), 2);
    });
}

// _CloudyDumpFramesTest.swift. This is intentionally ignored: the source
// scenario writes roughly 30k JPEGs (~650 MB).
#[cfg(any(feature = "ort", all(target_os = "macos", feature = "coreml")))]
#[test]
#[ignore = "manual 650 MB visualizer; requires FERMENT_RUN_CLOUDY_FRAME_DUMP=1 and real paths"]
fn dump_frames_with_cut_suffix() {
    require_opt_in("FERMENT_RUN_CLOUDY_FRAME_DUMP");
    let source = required_path("FERMENT_CLOUDY_VIDEO");
    let output_root = std::env::var_os("FERMENT_VIDEO_SCENE_DUMP_DIR")
        .map(PathBuf::from)
        .expect("set FERMENT_VIDEO_SCENE_DUMP_DIR explicitly");
    assert!(
        output_root.is_dir(),
        "dump root must already exist: {}",
        output_root.display()
    );
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let output = output_root.join(format!("cloudy_frames_{stamp}"));
    std::fs::create_dir(&output).unwrap();

    futures_executor::block_on(async {
        let video = video_backend();
        let result = VideoSceneDetecting::detect(
            &VideoSceneDetector::new(current_model_backend()).with_video_backend(video.clone()),
            &source,
            ProgressSink::noop(),
        )
        .await
        .unwrap();
        let info = video.probe(&source).await.unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = video.open_decoder(&source, stream.index).await.unwrap();
        let mut frame_index = 0_usize;
        while let Some(frame) = decoder.decode_next().await.unwrap() {
            let Some(scene) = result.scenes.iter().find(|scene| {
                (scene.start_frame..scene.start_frame + scene.frame_count).contains(&frame_index)
            }) else {
                frame_index += 1;
                continue;
            };
            let scene_directory = output.join(format!("scene_{:03}", scene.index));
            std::fs::create_dir_all(&scene_directory).unwrap();
            let is_cut = scene.index + 1 < result.scenes.len()
                && frame_index == scene.start_frame + scene.frame_count - 1;
            let suffix = if is_cut { "_cut" } else { "" };
            let cpu = frame.to_cpu_rgb().unwrap();
            let image = image::RgbImage::from_raw(cpu.width, cpu.height, cpu.data.to_vec())
                .expect("valid decoded RGB frame");
            let path = scene_directory.join(format!(
                "scene_{:03}_frame_{:05}{suffix}.jpg",
                scene.index, frame_index
            ));
            let file = std::fs::File::create(path).unwrap();
            image::codecs::jpeg::JpegEncoder::new_with_quality(file, 80)
                .encode_image(&image)
                .unwrap();
            frame_index += 1;
        }
        assert_eq!(frame_index, result.total_frames);
        std::fs::write(
            output.join("result.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
        eprintln!("wrote {frame_index} frames to {}", output.display());
    });
}

// _CloudyProfileTest.swift — a real wall-clock run, never a synthetic or
// no-op backend benchmark.
#[cfg(any(feature = "ort", all(target_os = "macos", feature = "coreml")))]
#[test]
#[ignore = "manual 20-minute actual-model profile; requires FERMENT_RUN_CLOUDY_PROFILE=1"]
fn profile_cloudy_20_min() {
    require_opt_in("FERMENT_RUN_CLOUDY_PROFILE");
    let source = required_path("FERMENT_CLOUDY_VIDEO");
    let started = std::time::Instant::now();
    let result = futures_executor::block_on(VideoSceneDetecting::detect(
        &VideoSceneDetector::new(current_model_backend()).with_video_backend(video_backend()),
        &source,
        ProgressSink::noop(),
    ))
    .unwrap();
    let elapsed = started.elapsed();
    assert!(result.total_frames > 0);
    assert!(!result.scenes.is_empty());
    eprintln!(
        "cloudy profile: {:.2?}, {} frames, {} scenes, {:.3}s source",
        elapsed,
        result.total_frames,
        result.scenes.len(),
        result.duration()
    );
}

#[cfg(feature = "ort")]
fn streaming_profile_backend() -> Backend {
    use ferment_engine_infer::{ModelPaths, OrtBackend, OrtExecutionProvider};
    use ferment_engine_test_fixtures::export;
    let autoshot =
        export::artifact("autoshot.onnx").expect("streaming gate requires published autoshot.onnx");
    let dinov2 = export::artifact("dinov2_vits14.onnx")
        .expect("streaming gate requires published dinov2_vits14.onnx");
    let paths = ModelPaths::new()
        .with_path(ModelId::Autoshot, ModelVersion::Latest, autoshot)
        .with_path(ModelId::Dinov2Vits14, ModelVersion::Latest, dinov2);
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}

#[cfg(feature = "ort")]
fn assert_scene_results_equal(
    actual: &ferment_engine_processors::video_scene_detector::VideoSceneDetectionResult,
    expected: &ferment_engine_processors::video_scene_detector::VideoSceneDetectionResult,
) {
    assert_eq!(actual.total_frames, expected.total_frames);
    assert_eq!(actual.frame_rate, expected.frame_rate);
    assert_eq!(actual.cuts.len(), expected.cuts.len());
    for (actual, expected) in actual.cuts.iter().zip(&expected.cuts) {
        assert_eq!(actual.frame, expected.frame);
        assert_eq!(actual.time, expected.time);
        assert!((actual.confidence - expected.confidence).abs() <= 1e-6);
    }
    assert_eq!(actual.scenes.len(), expected.scenes.len());
    for (actual, expected) in actual.scenes.iter().zip(&expected.scenes) {
        assert_eq!(actual.index, expected.index);
        assert_eq!(actual.start_frame, expected.start_frame);
        assert_eq!(actual.frame_count, expected.frame_count);
        assert_eq!(actual.start_time, expected.start_time);
        assert_eq!(actual.duration, expected.duration);
        match (&actual.embedding, &expected.embedding) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.len(), 384);
                assert_eq!(expected.len(), 384);
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 1e-5)
                );
            }
            (None, None) => (),
            _ => panic!("streaming changed the all-or-nothing scene embedding column"),
        }
    }
    assert_eq!(
        actual
            .scenes
            .iter()
            .map(|scene| scene.frame_count)
            .sum::<usize>(),
        actual.total_frames
    );
}

#[cfg(feature = "ort")]
fn run_streaming_comparison(media_variable: &str) {
    require_opt_in("FERMENT_RUN_STREAMING_PROFILE");
    let source = required_path(media_variable);
    futures_executor::block_on(async {
        let detector = VideoSceneDetector::new(streaming_profile_backend())
            .with_video_backend(video_backend());
        let updates = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = updates.clone();
        let progress =
            ProgressSink::from_callback(move |event| captured.lock().unwrap().push(event.progress));
        let start = std::time::Instant::now();
        let streaming = detector.detect_streaming(&source, progress).await.unwrap();
        let streaming_elapsed = start.elapsed();
        let start = std::time::Instant::now();
        // This diagnostic buffers only the small model frames; it must not first
        // retain every full-resolution decoded picture of a twenty-minute clip.
        let decoded = detector.decode_for_detection(&source).await.unwrap();
        let model = detector
            .detect(
                &decoded.frames,
                decoded.frame_count,
                decoded.frame_rate,
                ProgressSink::noop(),
            )
            .await
            .unwrap();
        let merged = VideoSceneDetector::merge_boundaries(
            model.clone(),
            &decoded.jump_cuts,
            &decoded.verdicts,
        );
        let buffered = VideoSceneDetector::attach_embeddings(merged, &decoded.samples);
        let buffered_elapsed = start.elapsed();
        assert!(streaming.total_frames > 0);
        assert!(
            !decoded.samples.is_empty(),
            "the actual DINO graph must answer"
        );
        assert_scene_results_equal(&streaming, &buffered);

        // The current flash veto intentionally removes model boundaries. Pin the
        // original profiler's no-lost-predictions intent on model-only outputs,
        // then independently require exact current post-pass equality above.
        let model_stream = detector
            .detect_streaming_with_options(&source, false, false, ProgressSink::noop())
            .await
            .unwrap();
        assert_scene_results_equal(&model_stream, &model);
        let updates = updates.lock().unwrap();
        assert_eq!(updates.last(), Some(&1.0));
        assert!(updates.windows(2).all(|pair| pair[0] <= pair[1]));
        eprintln!(
            "{}: streamed={streaming_elapsed:?}, buffered={buffered_elapsed:?}, frames={}, cuts={}",
            source.display(),
            streaming.total_frames,
            streaming.cuts.len()
        );
    });
}

// Kill: route the production API back through whole-clip buffering, lose a
// central model prediction, or change the current veto/identity post-passes.
#[cfg(feature = "ort")]
#[test]
#[ignore = "requires ORT masters, real cooking fixture and FERMENT_RUN_STREAMING_PROFILE=1"]
fn cooking_clip_streaming_vs_buffered() {
    run_streaming_comparison("FERMENT_COOKING_VIDEO");
}

// Kill: a duration-dependent tail/window error hidden by the short cooking clip.
#[cfg(feature = "ort")]
#[test]
#[ignore = "requires ORT masters, real twenty-minute Cloudy fixture and FERMENT_RUN_STREAMING_PROFILE=1"]
fn cloudy_streaming_vs_buffered() {
    run_streaming_comparison("FERMENT_CLOUDY_VIDEO");
}

#[cfg(any(feature = "ort", all(target_os = "macos", feature = "coreml")))]
mod manual_video_minimap_fixtures {
    use std::collections::{HashMap, HashSet};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use video_minimap_composer::allocate::{Scene, allocate};
    use video_minimap_composer::{KeyframeBuffer, SamplingStrategy};

    use super::*;

    const HAIRDRYER_VIDEO_ENV: &str = "FERMENT_HAIRDRYER_VIDEO";
    const CLOUDY_VIDEO_ENV: &str = "FERMENT_CLOUDY_VIDEO";

    fn required_path(variable: &str) -> PathBuf {
        let path = std::env::var_os(variable)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "manual VideoMinimap fixture gate: set {variable} before running ignored tests"
                )
            });
        assert!(
            path.is_file(),
            "manual VideoMinimap fixture gate: {variable}={} is not a file",
            path.display()
        );
        path
    }

    async fn detect_scenes(video_path: &Path, threshold: f32) -> VideoSceneDetectionResult {
        let detector = VideoSceneDetector::new(current_model_backend())
            .with_threshold(threshold)
            .with_video_backend(video_backend());
        VideoSceneDetecting::detect(&detector, video_path, ProgressSink::noop())
            .await
            .expect("AutoShot scene detection should succeed")
    }

    fn source_scenes(detected: &VideoSceneDetectionResult) -> Vec<Scene> {
        detected
            .scenes
            .iter()
            .map(|scene| Scene {
                start_frame: scene.start_frame,
                frame_count: scene.frame_count,
                start_time: scene.start_time,
                duration: scene.duration,
            })
            .collect()
    }

    async fn ground_truth_frames(
        video_path: &Path,
        needed_indices: &HashSet<usize>,
    ) -> HashMap<usize, KeyframeBuffer> {
        let backend = video_backend();
        let info = backend
            .probe(video_path)
            .await
            .expect("video fixture should probe");
        let stream_index = info
            .video_streams()
            .next()
            .expect("video fixture should have a video stream")
            .index;
        let mut decoder = backend
            .open_decoder(video_path, stream_index)
            .await
            .expect("video fixture should open for sequential decode");
        let last_needed = needed_indices.iter().copied().max().unwrap_or_default();
        let mut collected = HashMap::with_capacity(needed_indices.len());
        let mut frame_index = 0usize;
        while let Some(frame) = decoder
            .decode_next()
            .await
            .expect("sequential fixture decode should succeed")
        {
            if needed_indices.contains(&frame_index) {
                let cpu = frame
                    .to_cpu_rgb()
                    .expect("fixture frame should convert to RGB");
                collected.insert(
                    frame_index,
                    KeyframeBuffer {
                        width: cpu.width as usize,
                        height: cpu.height as usize,
                        data: cpu.data.as_ref().to_vec(),
                    },
                );
            }
            if frame_index >= last_needed {
                break;
            }
            frame_index += 1;
        }
        collected
    }

    fn perceptual_signature(frame: &KeyframeBuffer) -> Vec<u8> {
        // The current Swift observer draws to16×16 with CGContext .medium
        // interpolation BEFORE integer luma. Nearest sparse source pixels
        // alias fine texture and are not that independent filtered observer.
        let source = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
            frame.width as u32,
            frame.height as u32,
            &frame.data,
        )
        .unwrap();
        let reduced =
            image::imageops::resize(&source, 16, 16, image::imageops::FilterType::Triangle);
        reduced
            .pixels()
            .map(|pixel| {
                let [r, g, b] = pixel.0.map(usize::from);
                ((r * 299 + g * 587 + b * 114) / 1000) as u8
            })
            .collect()
    }

    /// Kill: restore the old nearest-grid observer. Original truth frames
    /// remain independently decoded; the strict scene inequality is unchanged.
    #[test]
    fn precision_signature_filters_texture_before_grayscale() {
        let mut data = Vec::with_capacity(512 * 256 * 3);
        for y in 0..256 {
            for x in 0..512 {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                data.extend_from_slice(&[value, value, value]);
            }
        }
        let signature = perceptual_signature(&KeyframeBuffer {
            width: 512,
            height: 256,
            data,
        });
        assert_eq!(signature.len(), 256);
        assert!(
            signature.iter().all(|value| (120..=135).contains(value)),
            "filtered texture aliased into a sparse source grid"
        );
        let red = perceptual_signature(&KeyframeBuffer {
            width: 32,
            height: 32,
            data: [255, 0, 0].repeat(32 * 32),
        });
        assert!(red.iter().all(|value| *value == 76));
    }

    fn signature_distance(left: &[u8], right: &[u8]) -> usize {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(left, right)| usize::from(left.abs_diff(*right)))
            .sum()
    }

    async fn run_precision_at(video_path: &Path, threshold: f32, sample_stride_hint: usize) {
        let detected = detect_scenes(video_path, threshold).await;
        assert!(
            detected.scenes.len() > 2,
            "precision fixture needs at least three detected scenes"
        );

        let scenes = source_scenes(&detected);
        let allocations = allocate(&scenes);
        let all_candidates = allocations
            .iter()
            .enumerate()
            .filter(|(scene_index, allocation)| {
                allocation.cell_frames.len() >= 2 && scene_index + 1 < scenes.len()
            })
            .map(|(scene_index, allocation)| {
                (
                    *allocation.cell_frames.last().unwrap(),
                    scenes[scene_index + 1].start_frame,
                    scene_index,
                )
            })
            .collect::<Vec<_>>();
        let candidates = all_candidates
            .into_iter()
            .step_by(sample_stride_hint.max(1))
            .collect::<Vec<_>>();
        assert!(
            candidates.len() >= 3,
            "need a few multi-cell scenes to get a meaningful signal"
        );

        let needed_indices = candidates
            .iter()
            .flat_map(|(last_in_scene, first_of_next, _)| [*last_in_scene, *first_of_next])
            .collect::<HashSet<_>>();
        let ground_truth = ground_truth_frames(video_path, &needed_indices).await;

        let composer = VideoMinimapComposer::new(VideoMinimapComposerConfig {
            max_scenes_shown: scenes.len(),
            sampling_strategy: SamplingStrategy::AllScenes,
            ..VideoMinimapComposerConfig::default()
        })
        .with_video_backend(video_backend());
        let (keyframes, _) = composer
            .decode_inputs(video_path, &scenes)
            .await
            .expect("composer keyframe extraction should succeed");
        assert_eq!(keyframes.len(), allocations.len());

        let mut checked = 0usize;
        for (last_in_scene, first_of_next, scene_index) in candidates {
            let extracted = keyframes
                .iter()
                .find(|(index, _)| *index == scene_index)
                .and_then(|(_, frames)| frames.last())
                .expect("candidate scene should have an extracted last cell");
            let truth_in = ground_truth
                .get(&last_in_scene)
                .expect("ground truth should contain the in-scene frame");
            let truth_out = ground_truth
                .get(&first_of_next)
                .expect("ground truth should contain the next-scene frame");
            let extracted_signature = perceptual_signature(extracted);
            let distance_in =
                signature_distance(&extracted_signature, &perceptual_signature(truth_in));
            let distance_out =
                signature_distance(&extracted_signature, &perceptual_signature(truth_out));
            assert!(
                distance_in < distance_out,
                "scene at frames …{last_in_scene}: extractor result is closer to next scene's first frame {first_of_next} (d={distance_out}) than to in-scene last frame {last_in_scene} (d={distance_in})"
            );
            checked += 1;
        }
        assert!(checked > 0, "no candidates produced a usable comparison");
    }

    #[test]
    #[ignore = "manual: requires FERMENT_HAIRDRYER_VIDEO and FERMENT_ENGINE_MODELS_DIR"]
    fn last_cell_of_scene_stays_inside_the_scene_hairdryer() {
        futures_executor::block_on(run_precision_at(
            &required_path(HAIRDRYER_VIDEO_ENV),
            0.5,
            1,
        ));
    }

    #[test]
    #[ignore = "manual: requires FERMENT_CLOUDY_VIDEO and FERMENT_ENGINE_MODELS_DIR"]
    fn last_cell_of_scene_stays_inside_the_scene_cloudy() {
        futures_executor::block_on(run_precision_at(
            &required_path(CLOUDY_VIDEO_ENV),
            0.296,
            14,
        ));
    }

    #[test]
    #[ignore = "manual: requires FERMENT_CLOUDY_VIDEO and FERMENT_ENGINE_MODELS_DIR"]
    fn compose_minimap_for_cloudy_20_min() {
        futures_executor::block_on(async {
            let video_path = required_path(CLOUDY_VIDEO_ENV);

            let detect_started = Instant::now();
            let scenes = detect_scenes(&video_path, 0.296).await;
            let detect_elapsed = detect_started.elapsed();
            eprintln!(
                "VideoSceneDetector: {:6.2}s -> {} scenes, {} boundaries",
                detect_elapsed.as_secs_f64(),
                scenes.scenes.len(),
                scenes.scenes.len().saturating_sub(1)
            );

            let compose_started = Instant::now();
            let result = VideoMinimapComposing::compose(
                &VideoMinimapComposer::with_defaults().with_video_backend(video_backend()),
                &video_path,
                &scenes,
                ProgressSink::noop(),
            )
            .await
            .expect("long-form cloudy minimap composition should succeed");
            let compose_elapsed = compose_started.elapsed();
            eprintln!(
                "Composer: {:6.2}s -> {}x{} px ({} KB)",
                compose_elapsed.as_secs_f64(),
                result.width,
                result.height,
                result.image_data.len() / 1_024
            );

            if let Some(output_directory) =
                std::env::var_os("FERMENT_MINIMAP_OUTPUT_DIR").map(PathBuf::from)
            {
                std::fs::create_dir_all(&output_directory)
                    .expect("FERMENT_MINIMAP_OUTPUT_DIR should be creatable");
                let stamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock should be after Unix epoch")
                    .as_secs();
                let output = output_directory.join(format!("cloudy_minimap_{stamp}.jpg"));
                std::fs::write(&output, &result.image_data)
                    .expect("cloudy minimap output should be writable");
                eprintln!("output: {}", output.display());
            } else {
                eprintln!("output not saved; set FERMENT_MINIMAP_OUTPUT_DIR to retain it");
            }

            let total = detect_elapsed + compose_elapsed;
            eprintln!(
                "TOTAL {:.2}s (detect {:.0}%, compose {:.0}%)",
                total.as_secs_f64(),
                detect_elapsed.as_secs_f64() / total.as_secs_f64() * 100.0,
                compose_elapsed.as_secs_f64() / total.as_secs_f64() * 100.0
            );
            assert!(result.image_data.len() > 1_024);
            assert!(scenes.scenes.len() > 100);
        });
    }
}
