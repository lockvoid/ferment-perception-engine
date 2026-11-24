//! Real four-graph EdgeTAM execution. Reference tensors/masks are emitted by
//! the pinned upstream variable-length attention and pre-existing iOS PyTorch
//! wrappers (`scripts/exports/edgetam/verify_onnx.py`), not by this Rust code.
//! Set FERMENT_EDGETAM_ORACLE_DIR to that frozen oracle export and
//! FERMENT_ENGINE_ONNX_DIR to the current fleet's downloaded ONNX files.
//! The retained numerical gates have not yet been qualified against this fleet.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use ferment_engine_core::{
    ImageObjectSegmenting, NormalizedVideoPoint, RgbFrameRef, VideoObjectId, VideoObjectMask,
    VideoObjectMaskFrame, VideoObjectMaskSink, VideoObjectPointLabel, VideoObjectPointPrompt,
    VideoObjectPrompt, VideoObjectPropagationDirection, VideoObjectSegmentationSession,
};
use ferment_engine_infer::{Backend, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::edgetam::frame_source::EdgeTamFrameSource;
use ferment_engine_processors::edgetam::{EdgeTam, EdgeTamModel, EdgeTamSession};
use ferment_engine_processors::video_backend;
use half::f16;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn directory() -> PathBuf {
    std::env::var_os("FERMENT_EDGETAM_ORACLE_DIR")
        .map(PathBuf::from)
        .expect("FERMENT_EDGETAM_ORACLE_DIR must name the independent oracle export")
}
fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(directory().join("goldens").join(name))
        .expect("required independent EdgeTAM golden")
}
fn halves(name: &str) -> Vec<f16> {
    let data = bytes(name);
    assert_eq!(data.len() % 2, 0);
    data.chunks_exact(2)
        .map(|v| f16::from_bits(u16::from_le_bytes([v[0], v[1]])))
        .collect()
}
fn manifest() -> serde_json::Value {
    serde_json::from_slice(&bytes("manifest.json")).unwrap()
}
fn backend() -> Backend {
    let models = std::env::var_os("FERMENT_ENGINE_ONNX_DIR")
        .map(PathBuf::from)
        .expect("FERMENT_ENGINE_ONNX_DIR must name the current fleet's verified ONNX files");
    let mut paths = ModelPaths::new();
    for id in EdgeTamModel::MODEL_FILES {
        paths = paths.with_path(
            id,
            ModelVersion::Latest,
            models.join(format!("{}.onnx", id.name())),
        );
    }
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}
fn point(x: f32, y: f32, label: VideoObjectPointLabel) -> VideoObjectPrompt {
    VideoObjectPrompt::Points(vec![VideoObjectPointPrompt::new(
        NormalizedVideoPoint::new(x, y),
        label,
    )])
}
fn mask_agreement(name: &str, actual: &VideoObjectMask) {
    let expected = bytes(name);
    assert_eq!(actual.bytes().len(), expected.len());
    let mut union = 0usize;
    let mut intersection = 0usize;
    let mut same = 0usize;
    for (&a, &e) in actual.bytes().iter().zip(&expected) {
        let a = a > 0;
        let e = e > 0;
        union += usize::from(a || e);
        intersection += usize::from(a && e);
        same += usize::from(a == e);
    }
    assert!(union > 0, "oracle must have foreground");
    let iou = intersection as f64 / union as f64;
    let agreement = same as f64 / expected.len() as f64;
    eprintln!("{name}: IoU={iou:.9}, agreement={agreement:.9}, foreground union={union}");
    // The pre-existing iOS local mask gates, unchanged after measurement.
    assert!(iou >= 0.98, "{name} IoU {iou}");
    assert!(agreement >= 0.995, "{name} agreement {agreement}");
}

/// KILL: leave invalid slots in softmax, rotate global/pointer tokens, or use
/// slot indices rather than the actual temporal position on sparse histories.
#[test]
#[ignore = "requires four actual EdgeTAM ONNX graphs, independent goldens and ORT_DYLIB_PATH"]
fn fixed_temporal_banks_match_upstream_and_ignore_poisoned_padding() {
    let model = EdgeTamModel::new(backend());
    let meta = manifest();
    let vision = halves("vision.fp16");
    let spatial = halves("spatial.fp16");
    let pointers = halves("pointers.fp16");
    for bank in meta["banks"].as_array().unwrap() {
        let name = bank["name"].as_str().unwrap();
        let mut sv = vec![f16::ZERO; 7];
        let mut pv = vec![f16::ZERO; 16];
        for i in bank["spatial"].as_array().unwrap() {
            sv[i.as_u64().unwrap() as usize] = f16::ONE;
        }
        for i in bank["pointers"].as_array().unwrap() {
            pv[i.as_u64().unwrap() as usize] = f16::ONE;
        }
        let actual =
            futures_executor::block_on(model.condition(&vision, &spatial, &pointers, &sv, &pv))
                .expect("actual memory attention runs");
        let expected = halves(&format!("{name}.fp16"));
        assert_eq!(actual.len(), expected.len());
        let mut maximum = 0f64;
        let mut delta2 = 0f64;
        let mut a2 = 0f64;
        let mut e2 = 0f64;
        let mut dot = 0f64;
        for (a, e) in actual.iter().zip(&expected) {
            let a = a.to_f64();
            let e = e.to_f64();
            assert!(a.is_finite());
            maximum = maximum.max((a - e).abs());
            delta2 += (a - e).powi(2);
            a2 += a * a;
            e2 += e * e;
            dot += a * e;
        }
        let relative = (delta2 / e2).sqrt();
        let cosine = dot / (a2 * e2).sqrt();
        eprintln!("{name}: max={maximum}, relative_l2={relative}, cosine={cosine}");
        assert!(maximum <= 0.006);
        assert!(relative <= 0.001);
        assert!(cosine >= 0.99999);
        let mut dirty_s = spatial.clone();
        let mut dirty_p = pointers.clone();
        for (slot, valid) in sv.iter().enumerate() {
            if *valid == f16::ZERO {
                dirty_s[slot * 512 * 64..(slot + 1) * 512 * 64].fill(f16::from_f32(511.));
            }
        }
        for (slot, valid) in pv.iter().enumerate() {
            if *valid == f16::ZERO {
                dirty_p[slot * 256..(slot + 1) * 256].fill(f16::from_f32(-511.));
            }
        }
        let poisoned =
            futures_executor::block_on(model.condition(&vision, &dirty_s, &dirty_p, &sv, &pv))
                .unwrap();
        assert_eq!(
            actual, poisoned,
            "{name}: invalid memory leaked into attention"
        );
    }
}

/// KILL: replace image inference with a blank/rectangle mask, choose the wrong
/// multimask candidate, or omit the previous-mask correction input.
#[test]
#[ignore = "requires four actual EdgeTAM ONNX graphs, independent goldens and ORT_DYLIB_PATH"]
fn image_prompts_and_accumulated_corrections_match_independent_masks() {
    futures_executor::block_on(async {
        let meta = manifest();
        let width = meta["image"]["width"].as_u64().unwrap() as u32;
        let height = meta["image"]["height"].as_u64().unwrap() as u32;
        let rgb = bytes("image.rgb");
        let engine = EdgeTam::new(backend(), video_backend());
        let mut session = ImageObjectSegmenting::make_session(
            &engine,
            RgbFrameRef {
                data: &rgb,
                width,
                height,
            },
        )
        .await
        .unwrap();
        let id = VideoObjectId::new("person");
        let first = session
            .apply(
                &point(0.7, 0.55, VideoObjectPointLabel::Foreground),
                &id,
                true,
            )
            .await
            .unwrap();
        mask_agreement("image-first.mask", &first.mask);
        let corrected = session
            .apply(
                &point(0.05, 0.05, VideoObjectPointLabel::Background),
                &id,
                false,
            )
            .await
            .unwrap();
        mask_agreement("image-corrected.mask", &corrected.mask);
        session.remove(&id).await;
        let repeat = session
            .apply(
                &point(0.7, 0.55, VideoObjectPointLabel::Foreground),
                &id,
                true,
            )
            .await
            .unwrap();
        assert_eq!(
            first.mask, repeat.mask,
            "remove must release correction state but retain the encoded image"
        );
        session.reset().await;
    });
}

struct Masks {
    limit: usize,
    frames: Mutex<Vec<VideoObjectMaskFrame>>,
}
impl VideoObjectMaskSink for Masks {
    fn emit(&self, frame: VideoObjectMaskFrame) -> bool {
        let mut frames = self.frames.lock().unwrap();
        frames.push(frame);
        frames.len() < self.limit
    }
}

/// KILL: omit temporal attention, ignore callback stop, retain decoder after
/// stop, reuse forward state on reverse, or keep prompts after reset.
#[test]
#[ignore = "requires actual EdgeTAM graphs, iOS qualification video and ORT_DYLIB_PATH"]
fn video_forward_reverse_cancel_and_reset_release_actual_session_resources() {
    futures_executor::block_on(async {
        let path = directory().join("checkpoints/qualifier-sec1.mp4");
        let source = EdgeTamFrameSource::new(video_backend(), &path)
            .await
            .unwrap();
        let mut session = EdgeTamSession::new(EdgeTamModel::new(backend()), source);
        assert_eq!(session.source_info().width, 720);
        assert_eq!(session.source_info().height, 900);
        let id = VideoObjectId::new("person");
        session
            .apply(
                &point(0.7, 0.55, VideoObjectPointLabel::Foreground),
                &id,
                0,
                true,
            )
            .await
            .unwrap();
        let forward = Masks {
            limit: 4,
            frames: Mutex::new(vec![]),
        };
        session
            .propagate(0, VideoObjectPropagationDirection::Forward, &forward)
            .await
            .unwrap();
        {
            let frames = forward.frames.lock().unwrap();
            assert_eq!(frames.len(), 4);
            for (i, frame) in frames.iter().enumerate() {
                assert_eq!(frame.frame_index, i);
                assert!((frame.presentation_time - i as f64 / 30.).abs() < 1e-5);
                assert!(frame.mask.bytes().iter().any(|v| *v > 0));
            }
        }
        assert!(!session.memory_snapshot().has_decoder_resources);
        assert!(session.memory_snapshot().has_temporal_tensor_bank);
        assert_eq!(session.memory_snapshot().prompted_frame_count, 1);
        session
            .apply(
                &point(0.05, 0.05, VideoObjectPointLabel::Background),
                &id,
                3,
                false,
            )
            .await
            .unwrap();
        let reverse = Masks {
            limit: 3,
            frames: Mutex::new(vec![]),
        };
        session
            .propagate(3, VideoObjectPropagationDirection::Reverse, &reverse)
            .await
            .unwrap();
        assert_eq!(
            reverse
                .frames
                .lock()
                .unwrap()
                .iter()
                .map(|f| f.frame_index)
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        assert!(!session.memory_snapshot().has_decoder_resources);
        session.reset().await;
        let memory = session.memory_snapshot();
        assert_eq!(memory.prompted_frame_count, 0);
        assert_eq!(memory.stored_frame_count, 0);
        assert!(!memory.has_cached_encoded_features);
        assert!(!memory.has_temporal_tensor_bank);
        assert!(!memory.has_decoder_resources);
        assert!(
            session
                .propagate(0, VideoObjectPropagationDirection::Forward, &forward)
                .await
                .is_err()
        );
        session
            .apply(
                &point(0.7, 0.55, VideoObjectPointLabel::Foreground),
                &VideoObjectId::new("new"),
                0,
                true,
            )
            .await
            .unwrap();
        session.reset().await;
    });
}
