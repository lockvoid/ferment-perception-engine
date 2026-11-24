//! Origin-parity oracles for the video-saliency backings, plus the end-to-end
//! measurement over a committed synthetic clip.
//!
//! Both masters ship as released ONNX rather than as an export of ours, so the
//! "origin" is the same graph run through onnxruntime's CPU EP in Python
//! (`scripts/exports/video_saliency/gen_goldens.py`) — the pyannote pattern.
//! What these oracles prove is that the Rust pre-processing, tensor plumbing
//! and YuNet decode agree with an independent Python implementation on
//! committed pixels, and that the ported math turns that into the right
//! measurement.
//!
//! Reference frames are committed at each graph's own input size, so no resize
//! runs on either side and the comparison is exact rather than kernel-dependent.
//!
//! Needs a loadable ONNX Runtime (`ORT_DYLIB_PATH`); select the execution
//! provider with `FERMENT_ENGINE_ORT_EP` (cpu | directml | cuda | windows).

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_core::{
    FaceRectSource, ProgressSink, RgbFrameRef, SaliencyHeatSource, SaliencyMode, VideoScene,
};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::video_saliency::{
    SampleSource, U2NetpHeatSource, VideoSaliencyAnalyzer, VideoSaliencyConfig, YuNetFaceSource,
    measure,
};
use ferment_engine_test_fixtures::export;

/// CPU EP against CPU-EP goldens is exact in practice; the bound leaves room
/// for a GPU provider's own arithmetic when `FERMENT_ENGINE_ORT_EP` names one.
const HEAT_TOLERANCE: f32 = 1e-4;
/// Face boxes are normalized to the frame, so this is a fraction of the frame:
/// a tenth of a pixel on a 640-wide input.
const BOX_TOLERANCE: f64 = 2e-4;

/// Both backings and their ladder are `ferment-models-export`'s
/// (`models/saliency`), the same bytes that grade every other runtime: the two
/// released graphs, the two frames they are measured on, U²-Netp's heat and
/// YuNet's twelve tensors with the DECODED boxes beside them.
fn golden(name: &str) -> PathBuf {
    export::require_path(&format!("saliency/goldens/{name}"))
}

/// The same explicitly configured published models as every other oracle.
fn master(file_name: &str) -> PathBuf {
    export::artifact(file_name).unwrap_or_else(|| {
        panic!("{file_name} is required; set FERMENT_ENGINE_MODELS_DIR to the published fleet")
    })
}

fn execution_provider() -> OrtExecutionProvider {
    match std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() {
        Ok("directml") => OrtExecutionProvider::DirectMl,
        Ok("cuda") => OrtExecutionProvider::Cuda,
        Ok("cpu") | Ok("windows") | Err(_) => OrtExecutionProvider::Cpu,
        Ok(other) => panic!("unsupported FERMENT_ENGINE_ORT_EP: {other}"),
    }
}

fn backend() -> Backend {
    let paths = ModelPaths::new()
        .with_path(ModelId::U2Netp, ModelVersion::Latest, master("u2netp.onnx"))
        .with_path(
            ModelId::YuNetFace,
            ModelVersion::Latest,
            master("yunet_face.onnx"),
        );
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        execution_provider(),
    ))
}

/// The frame each graph is measured on, at its own grid, as raw RGB24 — the
/// export writes the pixels rather than a PNG, so nothing about a decoder sits
/// between the golden and the door.
fn reference_frame(name: &str, side: u32) -> image::RgbImage {
    let path = golden(name);
    let pixels = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("missing reference frame {}: {error}", path.display()));
    image::RgbImage::from_raw(side, side, pixels).expect("packed RGB24 at the graph's grid")
}

fn golden_f32(name: &str) -> Vec<f32> {
    let path = golden(name);
    std::fs::read(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()))
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn frame_ref(image: &image::RgbImage) -> RgbFrameRef<'_> {
    RgbFrameRef::new(image.as_raw(), image.width(), image.height()).expect("packed RGB24")
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the in-repo masters"]
fn u2netp_heat_matches_the_python_origin() {
    let source = U2NetpHeatSource::new(backend());
    assert_eq!(source.mode(), SaliencyMode::Objectness);
    let frame = reference_frame("u2netp_selfie_320_in.u8", 320);
    let heat = futures_executor::block_on(source.heat(frame_ref(&frame))).expect("heat map");
    assert_eq!((heat.width, heat.height), (320, 320));

    let expected = golden_f32("u2netp_heat_out.f32");
    assert_eq!(heat.values.len(), expected.len());
    let difference = heat
        .values
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!("u2netp: observed maximum difference {difference}");
    assert!(
        difference <= HEAT_TOLERANCE,
        "u2netp heat drifted {difference} from the Python origin"
    );
}

#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the in-repo masters"]
fn yunet_faces_match_the_python_origin() {
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(golden("yunet_face_faces.json")).unwrap())
            .unwrap();
    let expected = expected.as_array().expect("golden is an array").clone();

    let frame = reference_frame("yunet_face_selfie_640_in.u8", 640);
    let detector = YuNetFaceSource::new(backend());
    let faces = futures_executor::block_on(detector.faces(frame_ref(&frame))).expect("faces");
    assert_eq!(
        faces.len(),
        expected.len(),
        "face count changed: {faces:#?} vs golden {expected:#?}"
    );
    for (actual, golden) in faces.iter().zip(&expected) {
        let field = |key: &str| golden[key].as_f64().expect("golden field");
        for (label, actual, expected) in [
            ("x", actual.x, field("x")),
            ("y", actual.y, field("y")),
            ("width", actual.width, field("width")),
            ("height", actual.height, field("height")),
            ("confidence", actual.confidence, field("confidence")),
        ] {
            assert!(
                (actual - expected).abs() <= BOX_TOLERANCE,
                "{label}: {actual} vs golden {expected}"
            );
        }
    }
}

/// The Swift `VideoSaliencyDominantFaceTests` case, on a committed crowd photo:
/// the verdict must be ONE face, not the band spanning the whole group.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the in-repo masters"]
fn a_crowd_measures_as_one_dominant_face_not_the_union() {
    let backend = backend();
    let heat = U2NetpHeatSource::new(Arc::clone(&backend));
    let faces = YuNetFaceSource::new(backend);
    let frame = reference_frame("yunet_face_selfie_640_in.u8", 640);
    let detected = futures_executor::block_on(faces.faces(frame_ref(&frame))).expect("faces");
    assert!(detected.len() > 1, "need a crowd, got {detected:#?}");

    let sample = futures_executor::block_on(measure::measure(
        &heat,
        &faces,
        frame_ref(&frame),
        0,
        0.0,
        &VideoSaliencyConfig::default(),
    ))
    .expect("sample");
    eprintln!(
        "crowd: src={:?} cx={} cy={} w={} h={}",
        sample.source, sample.cx, sample.cy, sample.w, sample.h
    );
    assert_eq!(sample.source, SampleSource::Face);

    let union_left = detected.iter().map(|f| f.x).fold(f64::INFINITY, f64::min);
    let union_right = detected
        .iter()
        .map(|f| f.x + f.width)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        sample.w < (union_right - union_left) / 2.0,
        "the verdict is still the union band: w={}",
        sample.w
    );
    let dominant = detected
        .iter()
        .max_by(|a, b| (a.width * a.height).total_cmp(&(b.width * b.height)))
        .unwrap();
    assert!((sample.cx - (dominant.x + dominant.width / 2.0)).abs() < 1e-9);
}

/// End to end over the committed synthetic clip: a still scene must produce a
/// still verdict on the subject, and a travelling scene must keep its curve
/// instead of averaging into one wrong point.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the in-repo masters and a platform decoder"]
fn a_synthetic_clip_measures_its_subject_end_to_end() {
    let clip = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/saliency-two-scenes.mp4");
    let backend = backend();
    let analyzer = VideoSaliencyAnalyzer::new(
        Arc::new(U2NetpHeatSource::new(Arc::clone(&backend))),
        Arc::new(YuNetFaceSource::new(backend)),
    );
    let scenes = [
        VideoScene {
            index: 0,
            start_frame: 0,
            frame_count: 120,
            start_time: 0.0,
            duration: 4.0,
            start_confidence: None,
            embedding: None,
        },
        VideoScene {
            index: 1,
            start_frame: 120,
            frame_count: 180,
            start_time: 4.0,
            duration: 6.0,
            start_confidence: Some(0.9),
            embedding: None,
        },
    ];
    let result = futures_executor::block_on(analyzer.analyze(&clip, &scenes, ProgressSink::noop()))
        .expect("analysis");
    eprintln!("e2e: {:#?}", result.scenes);
    eprintln!(
        "e2e samples: {:?}",
        result
            .samples
            .iter()
            .map(|s| (s.time, s.cx, s.cy, s.w, s.confidence))
            .collect::<Vec<_>>()
    );

    assert_eq!((result.frame_width, result.frame_height), (512, 288));
    assert_eq!(result.mode, SaliencyMode::Objectness);
    assert_eq!(result.scenes.len(), 2);
    assert!(
        result
            .samples
            .iter()
            .all(|sample| sample.source == SampleSource::Saliency),
        "a disc is not a face"
    );

    // Scene 0: the disc is parked at (0.30, 0.50) with a diameter of
    // 2 * 0.12 * 288 = 69 px, i.e. 0.135 of the width and 0.240 of the height.
    let still = result.scenes[0];
    assert_eq!(still.sample_count, 3);
    assert!((still.cx - 0.30).abs() < 0.02, "cx {}", still.cx);
    assert!((still.cy - 0.50).abs() < 0.02, "cy {}", still.cy);
    assert!((still.w - 0.135).abs() < 0.03, "w {}", still.w);
    assert!((still.h - 0.240).abs() < 0.04, "h {}", still.h);
    assert!(still.spread < 0.05, "spread {}", still.spread);
    assert_eq!(still.face_share, 0.0);

    // Scene 1: x 0.15 -> 0.85, so the samples must climb and the spread must be
    // large enough for a consumer to keep the path.
    let travelling = result.scenes[1];
    assert_eq!(travelling.sample_count, 3);
    assert!(travelling.spread > 0.15, "spread {}", travelling.spread);
    let path: Vec<f64> = result
        .samples
        .iter()
        .filter(|sample| sample.scene_index == 1)
        .map(|sample| sample.cx)
        .collect();
    assert!(
        path.windows(2).all(|pair| pair[0] < pair[1]),
        "cx must rise with time: {path:?}"
    );
    assert!(path[0] < 0.35, "first {}", path[0]);
    assert!(
        *path.last().unwrap() > 0.65,
        "last {}",
        path.last().unwrap()
    );

    // The plan asks for 4.6 / 7.0 / 9.4 s with 0.25 s of seek slack, and the
    // grabber spends it: it takes the first frame at or after `time -
    // tolerance`, so the subject is measured 0.25 s early. The disc crosses
    // 0.15 -> 0.85 over frames 120..299, i.e. 4.000 -> 9.967 s, which puts the
    // three grabs at x = 0.191 / 0.473 / 0.754. A grabber that ignored the
    // tolerance would read 0.220 / 0.501 / 0.782 and fail here while still
    // satisfying every assertion above.
    for (measured, expected) in path.iter().zip([0.191, 0.473, 0.754]) {
        assert!(
            (measured - expected).abs() < 0.01,
            "grab landed at cx={measured}, expected {expected} from time - tolerance"
        );
    }
}

/// Ported from `VideoSaliencyFaceTests.swift`: 20 s of a static interview shot
/// as one scene -> 10 grabs, and a talking head must come back as a FACE.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the in-repo masters and a platform decoder"]
fn a_talking_head_is_measured_from_faces() {
    let Some(clip) = ferment_engine_test_fixtures::path("Video/dhh-interview.mp4") else {
        return;
    };
    let backend = backend();
    let analyzer = VideoSaliencyAnalyzer::new(
        Arc::new(U2NetpHeatSource::new(Arc::clone(&backend))),
        Arc::new(YuNetFaceSource::new(backend)),
    );
    let scenes = [VideoScene {
        index: 0,
        start_frame: 0,
        frame_count: 600,
        start_time: 0.0,
        duration: 20.0,
        start_confidence: None,
        embedding: None,
    }];
    let result = futures_executor::block_on(analyzer.analyze(&clip, &scenes, ProgressSink::noop()))
        .expect("analysis");
    let verdict = result.scenes[0];
    eprintln!(
        "dhh-interview: faceShare={} cx={} cy={} w={} h={} confidence={} spread={}",
        verdict.face_share,
        verdict.cx,
        verdict.cy,
        verdict.w,
        verdict.h,
        verdict.confidence,
        verdict.spread
    );
    assert_eq!(result.samples.len(), 10);
    assert_eq!((result.frame_width, result.frame_height), (640, 360));
    assert!(
        verdict.face_share >= 0.6,
        "faceShare={} over {} samples",
        verdict.face_share,
        result.samples.len()
    );
    for sample in result
        .samples
        .iter()
        .filter(|s| s.source == SampleSource::Face)
    {
        assert!(
            sample.w < 0.6,
            "face box too wide at t={}: w={}",
            sample.time,
            sample.w
        );
        assert!(sample.w > 0.0);
        assert!(sample.confidence > 0.0);
    }
}

/// Ported from `VideoSaliencyOrientationTests.swift`: a portrait source has to
/// come back portrait, with every coordinate inside the frame.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the in-repo masters and a platform decoder"]
fn a_portrait_source_reports_a_portrait_frame_and_in_frame_coordinates() {
    let Some(clip) = ferment_engine_test_fixtures::path("Video/cloudy_9x16.mp4") else {
        return;
    };
    let backend = backend();
    let analyzer = VideoSaliencyAnalyzer::new(
        Arc::new(U2NetpHeatSource::new(Arc::clone(&backend))),
        Arc::new(YuNetFaceSource::new(backend)),
    );
    let scenes = [VideoScene {
        index: 0,
        start_frame: 0,
        frame_count: 300,
        start_time: 0.0,
        duration: 12.0,
        start_confidence: None,
        embedding: None,
    }];
    let result = futures_executor::block_on(analyzer.analyze(&clip, &scenes, ProgressSink::noop()))
        .expect("analysis");
    eprintln!(
        "cloudy_9x16: frame={}x{} centroids={:?}",
        result.frame_width,
        result.frame_height,
        result
            .samples
            .iter()
            .map(|s| (s.cx, s.cy))
            .collect::<Vec<_>>()
    );
    assert_eq!(result.samples.len(), 6); // 12 s x 0.5/s
    assert_eq!((result.frame_width, result.frame_height), (608, 1080));
    assert!(result.frame_height > result.frame_width);
    for sample in &result.samples {
        for (label, value) in [
            ("cx", sample.cx),
            ("cy", sample.cy),
            ("w", sample.w),
            ("h", sample.h),
        ] {
            assert!(
                (0.0..=1.0).contains(&value),
                "{label}={value} out of frame at t={}",
                sample.time
            );
        }
    }
    assert_eq!(result.scenes[0].sample_count, 6);
}

/// Ported from `VideoSaliencyDominantFaceTests.swift`: two men walking side by
/// side at 1.86 s. The union of their faces is a wide band whose centre is the
/// wall between them; the verdict must be ONE face, narrow and off that
/// midpoint.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the in-repo masters and a platform decoder"]
fn two_faces_resolve_to_one_face_not_the_band_between_them() {
    let Some(clip) = ferment_engine_test_fixtures::path("Video/djs-walking-city.mp4") else {
        return;
    };
    let backend = backend();
    let analyzer = VideoSaliencyAnalyzer::new(
        Arc::new(U2NetpHeatSource::new(Arc::clone(&backend))),
        Arc::new(YuNetFaceSource::new(backend)),
    );
    // One 0.1 s scene around 1.86 s, so the plan's single midpoint grab lands
    // exactly where the Swift test grabbed.
    let scenes = [VideoScene {
        index: 0,
        start_frame: 55,
        frame_count: 3,
        start_time: 1.81,
        duration: 0.1,
        start_confidence: None,
        embedding: None,
    }];
    let result = futures_executor::block_on(analyzer.analyze(&clip, &scenes, ProgressSink::noop()))
        .expect("analysis");
    let sample = result.samples[0];
    eprintln!(
        "djs@1.86: src={:?} cx={} cy={} w={} h={}",
        sample.source, sample.cx, sample.cy, sample.w, sample.h
    );
    assert_eq!(sample.source, SampleSource::Face);
    assert!(
        sample.w < 0.2,
        "the verdict is still the union band: w={}",
        sample.w
    );
    assert!(
        sample.cx < 0.25 || sample.cx > 0.45,
        "centroid {} sits on the wall between the two faces (union midpoint ~0.34)",
        sample.cx
    );
}

#[path = "support/current_saliency_source_tests.rs"]
mod current_source;
