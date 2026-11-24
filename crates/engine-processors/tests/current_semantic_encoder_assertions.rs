//! Current iOS semantic-encoder assertions through the production ORT path.
//! Required resources fail closed; the explicit oracle gate runs these ignored tests.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use ferment_engine_audio::AudioFrames;
use ferment_engine_core::ProgressSink;
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::dasheng::config::{EMBEDDING_DIM, SAMPLE_RATE, WINDOW_SAMPLES};
use ferment_engine_processors::video_scene_detector::{EncoderShotIdentity, ShotIdentifying};
use ferment_engine_processors::{
    AudioSemanticEncoder, AudioSource, Dinov2SemanticEncoder, FrameSnapshot, MusicTagger,
};
use ferment_engine_test_fixtures::export;
use futures_executor::block_on;
use std::{path::PathBuf, sync::Arc};

fn backend(models: &[(ModelId, &str)]) -> Backend {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "name the pinned ORT runtime"
    );
    let mut paths = ModelPaths::new();
    for &(id, name) in models {
        paths = paths.with_path(
            id,
            ModelVersion::Latest,
            export::artifact(name).expect("required published ONNX graph"),
        );
    }
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}

fn encoder() -> Dinov2SemanticEncoder {
    Dinov2SemanticEncoder::new(backend(&[(ModelId::Dinov2Vits14, "dinov2_vits14.onnx")]))
}

// Same independent xorshift texture, two 3x3 box passes and channel formula as
// current Swift SyntheticLuma.texture + DINOv2SemanticEncoderTests.picture.
fn picture(mut seed: u64, brightness: u8, portrait: bool) -> FrameSnapshot {
    let (width, height) = if portrait { (144, 256) } else { (256, 144) };
    let mut data: Vec<u8> = (0..width * height)
        .map(|_| {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect();
    for _ in 0..2 {
        let mut output = data.clone();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0_u32;
                for sy in y - 1..=y + 1 {
                    for sx in x - 1..=x + 1 {
                        sum += u32::from(data[sy * width + sx]);
                    }
                }
                output[y * width + x] = (sum / 9) as u8;
            }
        }
        data = output;
    }
    let rgb = data
        .into_iter()
        .flat_map(|value| {
            let value = value.saturating_add(brightness);
            [255 - value, value / 2, value]
        })
        .collect();
    FrameSnapshot::packed(width, height, rgb)
}

fn assert_unit(vector: &[f32]) {
    assert_eq!(vector.len(), 384);
    assert!(vector.iter().all(|value| value.is_finite()));
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() <= 0.02, "unit-vector norm: {norm}");
}

#[test]
#[ignore = "requires pinned ORT and the published DINOv2 graph"]
fn current_landscape_embeddings_are_unit_vectors_and_preserve_light_identity() {
    let encoder = encoder();
    let a = block_on(encoder.embedding(&picture(1, 0, false)));
    let lit = block_on(encoder.embedding(&picture(1, 60, false)));
    let other = block_on(encoder.embedding(&picture(2, 0, false)));
    for vector in [&a, &lit, &other] {
        assert_unit(vector);
    }
    assert!(
        Dinov2SemanticEncoder::cosine_distance(&a, &lit)
            < Dinov2SemanticEncoder::cosine_distance(&a, &other)
    );
}

#[test]
#[ignore = "requires pinned ORT and the published DINOv2 graph"]
fn current_portrait_embeddings_preserve_light_identity_through_rotation() {
    let encoder = encoder();
    let a = block_on(encoder.embedding(&picture(3, 0, true)));
    let lit = block_on(encoder.embedding(&picture(3, 60, true)));
    let other = block_on(encoder.embedding(&picture(4, 0, true)));
    for vector in [&a, &lit, &other] {
        assert_unit(vector);
    }
    assert!(
        Dinov2SemanticEncoder::cosine_distance(&a, &lit)
            < Dinov2SemanticEncoder::cosine_distance(&a, &other)
    );
}

#[test]
#[ignore = "requires pinned ORT and the published DINOv2 graph"]
fn packed_and_padded_rgb_inputs_agree_through_the_real_encoder() {
    let encoder = encoder();
    let packed = picture(5, 0, false);
    let stride = packed.bytes_per_row + 64;
    let mut padded = FrameSnapshot {
        width: packed.width,
        height: packed.height,
        bytes_per_row: stride,
        rgb: vec![197; stride * packed.height],
    };
    for y in 0..packed.height {
        padded.rgb[y * stride..y * stride + packed.bytes_per_row]
            .copy_from_slice(&packed.rgb[y * packed.bytes_per_row..(y + 1) * packed.bytes_per_row]);
    }
    let a = block_on(encoder.embedding(&packed));
    let b = block_on(encoder.embedding(&padded));
    assert_unit(&a);
    assert_unit(&b);
    assert!(Dinov2SemanticEncoder::cosine_distance(&a, &b) < 0.01);
}

#[test]
#[ignore = "requires pinned ORT and the published DINOv2 graph"]
fn the_production_encoder_and_shot_identity_supply_real_scene_vectors() {
    let encoder = encoder();
    assert_unit(&block_on(encoder.embedding(&picture(11, 0, false))));
    let identity = EncoderShotIdentity::new(encoder);
    let vector = block_on(identity.embedding(&picture(12, 0, false)));
    assert_unit(&vector);
    assert!(vector.iter().any(|value| value.abs() > 0.000_001));
}

#[test]
#[ignore = "requires pinned ORT and the published Dasheng graph"]
fn a_silent_window_has_finite_repeatable_embeddings() {
    let encoder =
        AudioSemanticEncoder::new(backend(&[(ModelId::DashengBase, "dasheng_base_mel.onnx")]));
    let silence = || AudioFrames {
        identity: None,
        samples: vec![0.0; WINDOW_SAMPLES],
        sample_rate: SAMPLE_RATE,
    };
    let a = block_on(encoder.encode(silence(), false, ProgressSink::noop())).unwrap();
    let b = block_on(encoder.encode(silence(), false, ProgressSink::noop())).unwrap();
    assert_eq!(a.pooled.len(), EMBEDDING_DIM);
    assert_eq!(b.pooled.len(), EMBEDDING_DIM);
    assert!(
        a.pooled
            .iter()
            .chain(&b.pooled)
            .all(|value| value.is_finite())
    );
    let dot: f32 = a.pooled.iter().zip(&b.pooled).map(|(a, b)| a * b).sum();
    let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(dot / (norm(&a.pooled) * norm(&b.pooled)) > 0.999);
}

#[test]
#[ignore = "requires pinned ORT, Dasheng and MusicTagger graphs, current iOS techno fixture and native MP3 decoder"]
fn the_dominant_genre_survives_the_real_audio_encoder_pipeline() {
    let root = PathBuf::from(
        std::env::var_os("FERMENT_CURRENT_IOS_ORACLE")
            .expect("name the frozen current iOS checkout"),
    );
    let source = root.join("Packages/FermentEngine/Tests/Fixtures/techno-1-30s.mp3");
    assert!(
        source.is_file(),
        "required current Swift fixture: {}",
        source.display()
    );
    let audio = block_on(AudioSource::platform_default().decode(&source)).unwrap();
    assert!(!audio.samples.is_empty());
    let backend = backend(&[
        (ModelId::DashengBase, "dasheng_base_mel.onnx"),
        (ModelId::MusicTaggerHeads, "music_tagger_heads.onnx"),
    ]);
    let embedding = block_on(AudioSemanticEncoder::new(backend.clone()).encode(
        audio,
        false,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(embedding.pooled.len(), EMBEDDING_DIM);
    let tags = block_on(MusicTagger::new(backend).tag(&embedding.pooled, 10)).unwrap();
    let genres: Vec<_> = tags.genre.iter().map(|tag| tag.name.as_str()).collect();
    assert_eq!(genres.first(), Some(&"electronic"));
    assert!(genres.contains(&"techno"), "genres: {genres:?}");
}
