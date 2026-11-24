//! The strobe regression, ported from iOS
//! `VideoSceneDetectorStrobeRegressionTests.swift`.
//!
//! The bug: a 20-minute festival clip came out with ~1 000 scenes — AutoShot
//! fires on strobe flashes (58 of the 67 boundaries in its first minute were
//! single white frames inside one shot). `strobe-cuts.mp4` is 14.4 s of that
//! footage: three real cuts, ten flashes, one real cut hidden in a flash.
//! Ground truth from the frame sheets, by eye:
//!
//!   * a cut at 39  (a white DJ frame at 38, the crowd shot begins at 39),
//!   * a cut at 189 (DJ booth → crowd wide with beams),
//!   * a cut at 337 (crowd wide → DJ back-shot lit by the flash).
//!
//! Everything else the model fires on (110, 120, 138, 166, 335, 346, 356, …)
//! is the light, not an edit — which the second test pins, so the regression
//! this file holds is real rather than one the model would pass anyway.
//!
//! Needs a video decoder (macOS/Windows platform backend), ONNX Runtime, the
//! published AutoShot + DINOv2 masters (`FERMENT_ENGINE_MODELS_DIR`, or
//! `ferment-models-export/artifacts`), and the iOS fixture tree. Missing any of
//! them, the test prints `SKIP` and says which.

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::sync::Arc;

use ferment_engine_core::ProgressSink;
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::video_scene_detector::{Decoded, VideoSceneDetector};
use ferment_engine_test_fixtures::export;

const FIXTURE: &str = "RegressionTests/VideoSceneDetector/strobe-cuts.mp4";
/// Ground truth: the scene starts a human sees in the clip.
const REAL_CUTS: [usize; 3] = [39, 189, 337];
/// Model boundaries that are flashes, not edits — the veto must remove them
/// and the veto-off run must still show them.
const FLASHES: [usize; 3] = [110, 120, 138];

/// The detector with both published masters behind ORT, or `None` with a
/// printed reason.
fn detector() -> Option<VideoSceneDetector> {
    let autoshot = export::artifact("autoshot.onnx")?;
    let dinov2 = export::artifact("dinov2_vits14.onnx")?;
    let paths = ModelPaths::new()
        .with_path(ModelId::Autoshot, ModelVersion::Latest, autoshot)
        .with_path(ModelId::Dinov2Vits14, ModelVersion::Latest, dinov2);
    let backend = match std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() {
        Ok("directml") => {
            OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::DirectMl)
        }
        Ok("cuda") => {
            OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cuda)
        }
        Ok("windows") => OrtBackend::windows_default(Arc::new(paths)),
        _ => OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu),
    };
    Some(VideoSceneDetector::new(Arc::new(backend)))
}

/// One decode of the fixture, shared shape for both tests.
fn decode() -> Option<(VideoSceneDetector, Decoded)> {
    let fixture = ferment_engine_test_fixtures::path(FIXTURE)?;
    let detector = detector()?;
    let decoded = futures_executor::block_on(detector.decode_for_detection(&fixture))
        .expect("the fixture decodes");
    Some((detector, decoded))
}

fn scene_starts(detector: &VideoSceneDetector, decoded: &Decoded, veto: bool) -> Vec<usize> {
    let model = futures_executor::block_on(detector.detect(
        &decoded.frames,
        decoded.frame_count,
        decoded.frame_rate,
        ProgressSink::noop(),
    ))
    .expect("AutoShot runs");
    let verdicts: &[_] = if veto { &decoded.verdicts } else { &[] };
    let merged = VideoSceneDetector::merge_boundaries(model, &decoded.jump_cuts, verdicts);
    merged
        .scenes
        .iter()
        .skip(1)
        .map(|scene| scene.start_frame)
        .collect()
}

#[test]
#[ignore = "needs the iOS fixture tree, the ONNX fleet and ONNX Runtime"]
fn strobe_footage_yields_only_the_real_cuts() {
    let (detector, decoded) =
        decode().expect("original strobe fixture and approved models are required");
    assert!(
        !decoded.verdicts.is_empty(),
        "the transient scanner found no runs at all in a clip that is ten flashes long — \
         the DINOv2 identity is not answering"
    );
    let starts = scene_starts(&detector, &decoded, true);
    assert_eq!(starts.len(), REAL_CUTS.len(), "scene starts: {starts:?}");
    for (start, expected) in starts.iter().zip(REAL_CUTS) {
        assert!(
            start.abs_diff(expected) <= 1,
            "{starts:?} vs {REAL_CUTS:?} (±1: the white frame may fall either side)"
        );
    }
}

/// With the veto off the model alone still fires on the flashes. Without this
/// the first test could pass on a fixture the model never got wrong.
#[test]
#[ignore = "needs the iOS fixture tree, the ONNX fleet and ONNX Runtime"]
fn the_model_alone_still_fires_on_the_flashes() {
    let (detector, decoded) =
        decode().expect("original strobe fixture and approved models are required");
    let starts = scene_starts(&detector, &decoded, false);
    assert!(starts.len() > 6, "{starts:?}");
    for flash in FLASHES {
        assert!(
            starts.contains(&flash),
            "the model should still fire on the flash at {flash}: {starts:?}"
        );
    }
}

/// The per-scene embedding column: all-or-nothing, one unit vector of 384 per
/// final scene, and distinguishable between the clip's three real shots.
#[test]
#[ignore = "needs the iOS fixture tree, the ONNX fleet and ONNX Runtime"]
fn every_final_scene_carries_a_unit_embedding() {
    let (detector, decoded) =
        decode().expect("original strobe fixture and approved models are required");
    let model = futures_executor::block_on(detector.detect(
        &decoded.frames,
        decoded.frame_count,
        decoded.frame_rate,
        ProgressSink::noop(),
    ))
    .expect("AutoShot runs");
    let merged = VideoSceneDetector::merge_boundaries(model, &decoded.jump_cuts, &decoded.verdicts);
    let result = VideoSceneDetector::attach_embeddings(merged, &decoded.samples);

    assert!(!decoded.samples.is_empty(), "the tap collected no samples");
    for scene in &result.scenes {
        let embedding = scene
            .embedding
            .as_ref()
            .unwrap_or_else(|| panic!("scene {} has no embedding", scene.index));
        assert_eq!(embedding.len(), 384);
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "scene {}: norm {norm}",
            scene.index
        );
    }
}

/// Kill: pass the retained buffered strobe regression while the production
/// streaming path omits a tail, disables the veto, or loses scene identities.
#[test]
#[ignore = "requires the real strobe fixture, both ORT masters and platform decoder"]
fn production_streaming_matches_every_buffered_strobe_scene_and_embedding() {
    let fixture =
        ferment_engine_test_fixtures::path(FIXTURE).expect("strobe fixture must be provisioned");
    let detector = detector().expect("both ORT masters must be provisioned");
    futures_executor::block_on(async {
        let streamed = detector
            .detect_streaming(&fixture, ProgressSink::noop())
            .await
            .unwrap();
        let decoded = detector.decode_for_detection(&fixture).await.unwrap();
        assert!(!decoded.verdicts.is_empty(), "real flash verdicts required");
        assert!(
            !decoded.samples.is_empty(),
            "real scene identity samples required"
        );
        let model = detector
            .detect(
                &decoded.frames,
                decoded.frame_count,
                decoded.frame_rate,
                ProgressSink::noop(),
            )
            .await
            .unwrap();
        let merged =
            VideoSceneDetector::merge_boundaries(model, &decoded.jump_cuts, &decoded.verdicts);
        let buffered = VideoSceneDetector::attach_embeddings(merged, &decoded.samples);
        assert_eq!(streamed.total_frames, decoded.frame_count);
        assert_eq!(streamed.frame_rate, decoded.frame_rate);
        assert_eq!(streamed.scenes.len(), REAL_CUTS.len() + 1);
        for ((actual, expected), real_cut) in streamed
            .scenes
            .iter()
            .skip(1)
            .zip(buffered.scenes.iter().skip(1))
            .zip(REAL_CUTS)
        {
            assert_eq!(actual.start_frame, expected.start_frame);
            assert!(actual.start_frame.abs_diff(real_cut) <= 1);
        }
        assert_eq!(
            streamed
                .scenes
                .iter()
                .map(|scene| scene.frame_count)
                .sum::<usize>(),
            decoded.frame_count
        );
        for (actual, expected) in streamed.scenes.iter().zip(&buffered.scenes) {
            assert_eq!(actual.index, expected.index);
            assert_eq!(actual.start_frame, expected.start_frame);
            assert_eq!(actual.frame_count, expected.frame_count);
            assert_eq!(actual.start_time, expected.start_time);
            assert_eq!(actual.duration, expected.duration);
            let embedding = actual.embedding.as_ref().expect("streamed scene identity");
            let baseline = expected
                .embedding
                .as_ref()
                .expect("buffered scene identity");
            assert_eq!(embedding.len(), 384);
            assert!((embedding.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs() < 1e-3);
            assert!(
                embedding
                    .iter()
                    .zip(baseline)
                    .all(|(a, b)| (a - b).abs() < 1e-5)
            );
        }
    });
}

/// Kill: collapse real different shots into the same identity under a lighting change.
/// This preserves the source's stimuli/order law using Android's DINO provider;
/// it does not claim to reproduce Apple's Vision feature-print vector space.
#[test]
#[ignore = "actual original strobe media and published DINO model; run explicitly"]
fn the_original_real_shot_stays_nearer_under_plus_sixty_light() {
    use ferment_engine_processors::video_scene_detector::transient_verdict::distance;
    use ferment_engine_processors::{
        dinov2::{Dinov2SemanticEncoder, FrameSnapshot},
        video_scene_detector::shot_identity::EncoderShotIdentity,
    };
    futures_executor::block_on(async {
        let path = ferment_engine_test_fixtures::path(FIXTURE)
            .expect("original strobe fixture is required");
        let video = ferment_engine_processors::video_backend();
        let info = video.probe(&path).await.unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = video.open_decoder(&path, stream.index).await.unwrap();
        let mut frames = std::collections::HashMap::new();
        let mut index = 0usize;
        while let Some(frame) = decoder.decode_next().await.unwrap() {
            if [20, 100, 250, 300].contains(&index) {
                let rgb = frame.to_cpu_rgb().unwrap().to_rgb24();
                frames.insert(
                    index,
                    FrameSnapshot::packed(
                        rgb.width as usize,
                        rgb.height as usize,
                        rgb.data.to_vec(),
                    ),
                );
            }
            index += 1;
            if index > 300 {
                break;
            }
        }
        assert_eq!(frames.len(), 4, "every original stimulus must decode");
        let model = export::artifact("dinov2_vits14.onnx").expect("published DINO model required");
        let paths = ModelPaths::new().with_path(ModelId::Dinov2Vits14, ModelVersion::Latest, model);
        let encoder = Dinov2SemanticEncoder::new(Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        )));
        let shot = encoder.embedding(&frames[&100]).await;
        assert_eq!(shot.len(), 384);
        let mut lit = frames[&100].clone();
        for byte in &mut lit.rgb {
            *byte = byte.saturating_add(60);
        }
        let lit = encoder.embedding(&lit).await;
        assert_eq!(lit.len(), 384);
        let light_distance = distance(&shot, &lit);
        assert!(
            light_distance < EncoderShotIdentity::SAME_SHOT_DISTANCE,
            "+60 light changed the shot: {light_distance}"
        );
        for index in [20, 250, 300] {
            let other = encoder.embedding(&frames[&index]).await;
            assert_eq!(other.len(), 384);
            let other_distance = distance(&shot, &other);
            assert!(
                other_distance > EncoderShotIdentity::SAME_SHOT_DISTANCE,
                "original other shot {index}: {other_distance}"
            );
            assert!(
                light_distance < other_distance,
                "original ordering failed for shot {index}"
            );
        }
    });
}

/// Kill: bypass the production AI encode variant, omit source detection, or
/// report agreement without comparing actual original/feed scene boundaries.
/// The original is a measurement gate: agreement is reported, not thresholded.
#[test]
#[ignore = "actual original configured video, platform encoder and two scene models; run explicitly"]
fn current_ai_feed_records_actual_original_scene_agreement() {
    use ferment_engine_core::{MultiVideoEncoding, VideoEncodingVariant, VideoSceneDetecting};
    use ferment_engine_processors::VideoEncoder;
    let source = std::path::PathBuf::from(
        std::env::var_os("FERMENT_SCENE_FEED_BENCHMARK_INPUT")
            .expect("explicit original feed benchmark input required"),
    );
    assert!(source.is_file());
    let directory = std::env::temp_dir().join(format!(
        "ferment-current-scene-feed-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _owned = Owned(directory.clone());
    futures_executor::block_on(async {
        let feed = directory.join("ai.mp4");
        let start = std::time::Instant::now();
        let encoded = VideoEncoder::new()
            .with_backends(ferment_engine_processors::platform_encode_backends())
            .encode_variants(
                &source,
                &[VideoEncodingVariant {
                    id: "ai".into(),
                    output_path: feed.clone(),
                    preset: VideoEncoder::ai_preset(),
                    thumb_strip: None,
                }],
                ProgressSink::noop(),
                None,
            )
            .await
            .unwrap();
        assert!(encoded.variants.contains_key("ai"));
        assert!(feed.is_file());
        let encode_wall = start.elapsed().as_secs_f64();
        let detector = detector().expect("both approved scene models required");
        let start = std::time::Instant::now();
        let original = VideoSceneDetecting::detect(&detector, &source, ProgressSink::noop())
            .await
            .unwrap();
        let original_wall = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let derived = VideoSceneDetecting::detect(&detector, &feed, ProgressSink::noop())
            .await
            .unwrap();
        let feed_wall = start.elapsed().as_secs_f64();
        let original_cuts: Vec<_> = original
            .scenes
            .iter()
            .skip(1)
            .map(|s| s.start_frame)
            .collect();
        let feed_cuts: Vec<_> = derived
            .scenes
            .iter()
            .skip(1)
            .map(|s| s.start_frame)
            .collect();
        let agreed: Vec<_> = original_cuts
            .iter()
            .filter(|o| feed_cuts.iter().any(|f| f.abs_diff(**o) <= 1))
            .copied()
            .collect();
        let missed: Vec<_> = original_cuts
            .iter()
            .filter(|o| !feed_cuts.iter().any(|f| f.abs_diff(**o) <= 1))
            .copied()
            .collect();
        let extra: Vec<_> = feed_cuts
            .iter()
            .filter(|f| !original_cuts.iter().any(|o| o.abs_diff(**f) <= 1))
            .copied()
            .collect();
        let original_frames = original
            .scenes
            .iter()
            .map(|scene| scene.frame_count)
            .sum::<usize>();
        let feed_frames = derived
            .scenes
            .iter()
            .map(|scene| scene.frame_count)
            .sum::<usize>();
        assert_eq!(
            feed_frames, original_frames,
            "the feed preserves every source frame"
        );
        assert_eq!(encoded.decoded_frames, original_frames);
        let variant = &encoded.variants["ai"];
        let record = serde_json::json!({"feedEncodeWallSeconds":encode_wall,"scenesOnOriginalWallSeconds":original_wall,"scenesOnFeedWallSeconds":feed_wall,"scenesSpeedupPercent":(original_wall-feed_wall)/original_wall*100.0,"scenesFromOriginal":original.scenes.len(),"cutsFromOriginal":original_cuts.len(),"scenesFromFeed":derived.scenes.len(),"cutsFromFeed":feed_cuts.len(),"agreedWithinOneFrame":agreed.len(),"missedByFeed":missed,"extraFromFeed":extra,"sourceFrames":original_frames,"feedFrames":feed_frames,"feedWidth":variant.width,"feedHeight":variant.height});
        if let Some(path) = std::env::var_os("FERMENT_SCENE_FEED_BENCHMARK_RESULT") {
            std::fs::write(path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        }
        println!("CURRENT_SCENE_FEED {record}");
        assert!(!original.scenes.is_empty(), "original scene assertion");
    });
}
