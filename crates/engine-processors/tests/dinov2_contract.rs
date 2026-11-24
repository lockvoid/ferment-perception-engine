//! `dinov2_vits14` against its contract and its golden ladder — both owned by
//! `ferment-models-export` (`models/dinov2/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — the frame this processor prepares, the door, and the two
//!   distances the flash veto holds;
//! * `…_image_in.u8` → `…_embedding_out.f32` — 26 frames of `strobe-cuts.mp4`
//!   (either side of each real cut, of each flash, and three pairs inside the
//!   decision band) and the resampler's synthetic frame, with the checkpoint's
//!   unit vectors: every frame within `COSINE` through the real backend, every
//!   pair on the reference's side of 0.72 and 0.80 and within `DISTANCE` of
//!   it;
//! * the encoder as the veto calls it — snapshot, rotation, resample — on the
//!   published master: the same picture under another light nearer than
//!   another picture.
//!
//! The contract runs everywhere; the lowered half is `#[ignore]`d like every
//! oracle and needs the artifacts, which live in the export repository only
//! between an export and its upload.

use ferment_engine_processors::Dinov2SemanticEncoder;
use ferment_engine_processors::video_scene_detector::EncoderShotIdentity;
use ferment_engine_test_fixtures::export;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("dinov2/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

#[test]
fn the_contract_is_the_processors_door_and_the_vetos_distances() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "dinov2_vits14");
    let frame = &contract["frontend"]["frame"];
    assert_eq!(frame["width"], Dinov2SemanticEncoder::INPUT_WIDTH);
    assert_eq!(frame["height"], Dinov2SemanticEncoder::INPUT_HEIGHT);

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "image");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([
            1,
            3,
            Dinov2SemanticEncoder::INPUT_HEIGHT,
            Dinov2SemanticEncoder::INPUT_WIDTH
        ])
    );
    assert_eq!(main["outputs"][0]["name"], "embedding");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, Dinov2SemanticEncoder::EMBEDDING_SIZE])
    );
    assert_eq!(contract["artifacts"]["onnx"]["file"], "dinov2_vits14.onnx");

    let identity = &contract["identity"];
    assert_eq!(
        identity["same_shot"].as_f64().unwrap() as f32,
        EncoderShotIdentity::SAME_SHOT_DISTANCE
    );
    assert_eq!(
        identity["similar_shot"].as_f64().unwrap() as f32,
        EncoderShotIdentity::SIMILAR_SHOT_DISTANCE
    );
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
        TensorMap,
    };

    use super::*;

    /// fp32 behind the same door: the export measured cosine 1.000000 and no
    /// pair drift.
    const COSINE: f32 = 0.99999;
    const DISTANCE: f32 = 0.015;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    }

    /// `EncoderShotIdentity`'s measure: Euclidean distance of the unit vectors.
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let (na, nb) = (norm(a), norm(b));
        a.iter()
            .zip(b)
            .map(|(x, y)| (x / na - y / nb).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn verdict(value: f32) -> &'static str {
        if value < EncoderShotIdentity::SAME_SHOT_DISTANCE {
            "same"
        } else if value < EncoderShotIdentity::SIMILAR_SHOT_DISTANCE {
            "similar"
        } else {
            "different"
        }
    }

    /// The door fed a golden frame's own planes — the graph, not the host's
    /// resample, is what this measures.
    fn embed(backend: &Backend, frame: &str) -> Vec<f32> {
        let planes = std::fs::read(export::require_path(&format!(
            "dinov2/goldens/dinov2_{frame}_image_in.u8"
        )))
        .unwrap();
        let image = Tensor::from_f32(
            vec![
                1,
                3,
                Dinov2SemanticEncoder::INPUT_HEIGHT as i64,
                Dinov2SemanticEncoder::INPUT_WIDTH as i64,
            ],
            planes.into_iter().map(f32::from).collect(),
        )
        .unwrap();
        let mut inputs = TensorMap::new();
        inputs.insert("image".into(), image);
        let outputs = futures_executor::block_on(backend.infer(
            ModelId::Dinov2Vits14,
            &ModelVersion::default(),
            inputs,
        ))
        .unwrap();
        outputs["embedding"].to_f32().unwrap()
    }

    fn assert_answers(backend: Backend, label: &str) {
        let Some(contract) = contract() else { return };
        let frames = contract["goldens"]["frames"].as_object().unwrap();
        let mut vectors = std::collections::HashMap::new();
        let mut worst = 1.0_f32;
        for frame in frames.keys() {
            let vector = embed(&backend, frame);
            let expected =
                export::require_f32_le(&format!("dinov2/goldens/dinov2_{frame}_embedding_out.f32"));
            let similarity = cosine(&vector, &expected);
            worst = worst.min(similarity);
            assert!(similarity >= COSINE, "{label} {frame}: {similarity}");
            vectors.insert(frame.clone(), vector);
        }
        let mut widest = 0.0_f32;
        for pair in contract["goldens"]["pairs"].as_array().unwrap() {
            let (before, after) = (
                pair["before"].as_str().unwrap(),
                pair["after"].as_str().unwrap(),
            );
            let expected = pair["distance"].as_f64().unwrap() as f32;
            let found = distance(&vectors[before], &vectors[after]);
            widest = widest.max((found - expected).abs());
            assert_eq!(
                verdict(found),
                verdict(expected),
                "{label} {}: {found} against the reference's {expected}",
                pair["label"]
            );
            assert!(
                (found - expected).abs() <= DISTANCE,
                "{label} {}: {found} against the reference's {expected}",
                pair["label"]
            );
        }
        eprintln!("[dinov2] {label}: worst cosine {worst:.6}, widest pair drift {widest:.4}");
    }

    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_encoder_answers_the_checkpoint() {
        let Some(artifact) = export::artifact("dinov2_vits14.onnx") else {
            return;
        };
        let paths =
            ModelPaths::new().with_path(ModelId::Dinov2Vits14, ModelVersion::Latest, artifact);
        let backend =
            OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu);
        assert_answers(Arc::new(backend), "ort");
    }

    /// The whole encoder — snapshot preparation, rotation, resample, tensor
    /// plumbing — asserting the one property every consumer rides: the same
    /// picture under a different light is nearer than a different picture.
    /// The door test above starts from prepared planes and cannot see any of
    /// that. The source is the resampler's committed 640-wide frame.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_encoder_puts_light_nearer_than_content() {
        use ferment_engine_processors::FrameSnapshot;

        let Some(artifact) = export::artifact("dinov2_vits14.onnx") else {
            return;
        };
        let paths =
            ModelPaths::new().with_path(ModelId::Dinov2Vits14, ModelVersion::Latest, artifact);
        let encoder = Dinov2SemanticEncoder::new(Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        )));
        let source = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/exports/dinov2/goldens/resample_source_640x360.rgb"),
        )
        .expect("the committed 640-wide source frame");
        let frame = FrameSnapshot::packed(640, 360, source.clone());
        let lit = FrameSnapshot::packed(
            640,
            360,
            source.iter().map(|byte| byte.saturating_add(60)).collect(),
        );
        // The same frame upside down: same colours, different structure.
        let flipped = FrameSnapshot::packed(640, 360, {
            let mut bytes = source.clone();
            bytes.reverse();
            bytes
        });

        let embed = |snapshot| futures_executor::block_on(encoder.embedding(snapshot));
        let (a, b, c) = (embed(&frame), embed(&lit), embed(&flipped));
        assert_eq!(a.len(), Dinov2SemanticEncoder::EMBEDDING_SIZE);
        let under_light = Dinov2SemanticEncoder::cosine_distance(&a, &b);
        let other_content = Dinov2SemanticEncoder::cosine_distance(&a, &c);
        eprintln!(
            "[dinov2] same picture under light {under_light}, different content {other_content}"
        );
        assert!(
            under_light < other_content,
            "light {under_light} must be nearer than content {other_content}"
        );

        // A portrait crop of the same frame embeds through the rotation rather
        // than a squash, so it must not come back empty or degenerate.
        let portrait = FrameSnapshot {
            width: 360,
            height: 640,
            bytes_per_row: 360 * 3,
            rgb: {
                let mut bytes = vec![0_u8; 360 * 640 * 3];
                for y in 0..640 {
                    let row = &source[(y % 360) * 640 * 3..(y % 360) * 640 * 3 + 360 * 3];
                    bytes[y * 360 * 3..(y + 1) * 360 * 3].copy_from_slice(row);
                }
                bytes
            },
        };
        assert_eq!(
            embed(&portrait).len(),
            Dinov2SemanticEncoder::EMBEDDING_SIZE
        );
    }
}
