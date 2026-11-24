//! Current iOS speech-identity corpus assertions through native file ingest,
//! the production frontend, and actual ONNX rows. No fabricated graph answers.
//! The required fixture/model roots fail closed when these ignored cases are selected.

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::sync::Arc;

use ferment_engine_audio::{decode_path, resample_sinc};
use ferment_engine_core::{ProgressSink, TimeRange};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::ambernet::{self, AmberNetLanguageId, LanguageHypothesis};
use ferment_engine_processors::audio_source::{ArrayAudioSampleSource, AudioSource};
use ferment_engine_processors::pyannote::{PyannoteSpeakerSegmenter, stitcher};
use ferment_engine_processors::redimnet::{ReDimNetSpeakerEncoder, windows};
use ferment_engine_test_fixtures::{self as fixtures, export};

fn backend(model: ModelId, file: &str) -> Backend {
    let artifact = export::artifact(file).expect("required speech model artifact");
    let paths = ModelPaths::new().with_path(model, ModelVersion::Latest, artifact);
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}

fn language(relative: &str) -> Vec<LanguageHypothesis> {
    let path = fixtures::path(relative).expect("required current iOS audio fixture");
    let decoded = futures_executor::block_on(AudioSource::platform_default().decode(&path))
        .expect("the actual compressed fixture decodes through production AudioSource");
    let samples = if decoded.sample_rate == ambernet::SAMPLE_RATE {
        decoded.samples
    } else {
        resample_sinc(&decoded.samples, decoded.sample_rate, ambernet::SAMPLE_RATE)
            .expect("resample to the model's own rate")
    };
    let source = ArrayAudioSampleSource::new(samples);
    let identifier =
        AmberNetLanguageId::new(backend(ModelId::AmberNetLangId, "ambernet_langid.onnx"));
    let result =
        futures_executor::block_on(identifier.identify(&source, None, ProgressSink::noop()))
            .expect("actual AmberNet answers");
    assert_eq!(
        result.hypotheses.len(),
        107,
        "every VoxLingua hypothesis is returned"
    );
    assert!(
        result
            .hypotheses
            .iter()
            .all(|answer| answer.probability.is_finite())
    );
    let sum: f32 = result
        .hypotheses
        .iter()
        .map(|answer| answer.probability)
        .sum();
    assert!((sum - 1.0).abs() <= 1e-3, "probability sum {sum}");
    assert!(
        result
            .hypotheses
            .windows(2)
            .all(|pair| pair[0].probability >= pair[1].probability)
    );
    result.hypotheses
}

/// Kill: changed audio ingest/front-end/vocabulary makes the Russian vlog another language.
#[test]
#[ignore = "needs current fixtures, export artifacts and ONNX Runtime"]
fn russian_vlog_is_identified_as_russian() {
    assert_eq!(language("volkova3.mp3")[0].code, "ru");
}

/// Kill: classify the actual Russian advert incorrectly while synthetic features still pass.
#[test]
#[ignore = "needs current fixtures, export artifacts and ONNX Runtime"]
fn russian_product_placement_is_identified_as_russian() {
    assert_eq!(language("hairdryer-product-placement.mp3")[0].code, "ru");
}

/// Kill: the English narration is routed to the wrong language branch.
#[test]
#[ignore = "needs current fixtures, export artifacts and ONNX Runtime"]
fn english_edu_reel_is_identified_as_english() {
    assert_eq!(
        language("Audio/animal-evolution-edu-reel.mp3")[0].code,
        "en"
    );
}

/// Kill: add an implicit speech gate; current Swift documents that confidence is not VAD.
#[test]
#[ignore = "needs current fixtures, export artifacts and ONNX Runtime"]
fn music_without_speech_still_returns_a_confident_answer() {
    assert!(language("Audio/classical-piano.mp3")[0].probability > 0.5);
}

fn encoder() -> ReDimNetSpeakerEncoder {
    ReDimNetSpeakerEncoder::new(backend(ModelId::RedimnetSpeaker, "redimnet_speaker.onnx"))
}

fn embed(
    encoder: &ReDimNetSpeakerEncoder,
    stem: &str,
    start: f64,
    duration: f64,
) -> Vec<ferment_engine_processors::redimnet::SpeakerSegmentEmbedding> {
    let path = fixtures::path(&format!("SpeakerEncoder/{stem}.wav"))
        .expect("required current speaker corpus clip");
    let audio = ReDimNetSpeakerEncoder::to_model_rate(decode_path(&path).expect("WAV decode"))
        .expect("model-rate audio");
    futures_executor::block_on(encoder.embed(&audio, &[TimeRange { start, duration }]))
        .expect("actual ReDimNet answers")
}

fn golden(stem: &str) -> Vec<Vec<f32>> {
    let raw = export::require_f32_le(&format!(
        "redimnet/goldens/redimnet_{stem}_embedding_out.f32"
    ));
    assert_eq!(raw.len() % windows::EMBEDDING_DIMENSION, 0);
    assert!(!raw.is_empty(), "speaker golden must contain windows");
    raw.chunks_exact(windows::EMBEDDING_DIMENSION)
        .map(<[f32]>::to_vec)
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Kill: pad the minimum span or offset the first window; independent torch cosine falls.
#[test]
#[ignore = "needs current speaker fixtures, export goldens and ONNX Runtime"]
fn a_single_window_span_matches_the_torch_golden() {
    let encoder = encoder();
    for stem in ["speaker-a-1", "speaker-b-1"] {
        let embeddings = embed(&encoder, stem, 0.0, windows::minimum_span_seconds());
        assert_eq!(embeddings.len(), 1, "{stem}: minimum span");
        let similarity = cosine(&embeddings[0].vector, &golden(stem)[0]);
        assert!(similarity >= 0.999, "{stem}: cosine {similarity}");
    }
}

/// Kill: shift, drop or duplicate a decoded-file window while raw mel row tests still pass.
#[test]
#[ignore = "needs current speaker fixtures, export goldens and ONNX Runtime"]
fn every_window_of_a_long_span_matches_the_torch_golden_and_times() {
    let encoder = encoder();
    let seconds = windows::minimum_span_seconds();
    for stem in ["speaker-a-1", "speaker-a-2", "speaker-b-1", "speaker-b-2"] {
        let expected = golden(stem);
        assert!(expected.len() > 1, "{stem}: a multi-window fixture");
        let actual = embed(&encoder, stem, 0.0, 3600.0);
        assert_eq!(actual.len(), expected.len(), "{stem}: window count");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!((actual.start_time - index as f64 * seconds).abs() <= 1e-9);
            assert!((actual.end_time - (index + 1) as f64 * seconds).abs() <= 1e-9);
            let similarity = cosine(&actual.vector, expected);
            assert!(
                similarity >= 0.999,
                "{stem} window {index}: cosine {similarity}"
            );
        }
    }
}

/// Kill: dilute the shortest legal speech span until another speaker is a better match.
#[test]
#[ignore = "needs current speaker fixtures, export artifacts and ONNX Runtime"]
fn a_short_piece_identifies_the_speaker_better_than_any_impostor() {
    let encoder = encoder();
    let short = embed(
        &encoder,
        "speaker-a-1",
        0.0,
        windows::minimum_span_seconds(),
    );
    assert_eq!(short.len(), 1);
    let pooled = |stem| {
        let vectors = embed(&encoder, stem, 0.0, 3600.0)
            .into_iter()
            .map(|embedding| embedding.vector)
            .collect::<Vec<_>>();
        assert!(!vectors.is_empty());
        stitcher::pooled(&vectors)
    };
    let same = cosine(&short[0].vector, &pooled("speaker-a-2"));
    for stem in ["speaker-b-1", "speaker-b-2"] {
        let other = cosine(&short[0].vector, &pooled(stem));
        assert!(same > other, "same speaker {same}, impostor {stem} {other}");
    }
}

/// Kill: emit a wrong-width, non-finite or non-unit vector from the actual row.
#[test]
#[ignore = "needs current speaker fixtures, export artifacts and ONNX Runtime"]
fn every_embedding_is_finite_unit_length_and_the_declared_width() {
    let embeddings = embed(&encoder(), "speaker-a-1", 0.0, 3600.0);
    assert!(!embeddings.is_empty());
    for (index, embedding) in embeddings.iter().enumerate() {
        assert_eq!(
            embedding.vector.len(),
            windows::EMBEDDING_DIMENSION,
            "window {index}"
        );
        assert!(
            embedding.vector.iter().all(|value| value.is_finite()),
            "window {index}"
        );
        let norm = embedding
            .vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1e-3, "window {index}: norm {norm}");
    }
}

/// Kill: return segment-wide, overlapping or relative times from the scored span.
#[test]
#[ignore = "needs current speaker fixtures, export artifacts and ONNX Runtime"]
fn a_long_span_yields_ordered_non_overlapping_sub_boundaries() {
    let window = windows::minimum_span_seconds();
    let embeddings = embed(&encoder(), "speaker-a-1", 1.0, 3.0 * window);
    assert_eq!(embeddings.len(), 3);
    for (index, embedding) in embeddings.iter().enumerate() {
        assert!((embedding.end_time - embedding.start_time - window).abs() <= 1e-9);
        assert!(embedding.start_time >= 1.0);
        assert!(embedding.end_time <= 1.0 + 3.0 * window + 1e-9);
        if index > 0 {
            assert!(embedding.start_time >= embeddings[index - 1].end_time - 1e-9);
        }
    }
}

/// Kill: publish an anonymous, malformed or embedder-derived segmentation identity.
#[test]
fn the_segmenter_names_the_weights_it_ran_on() {
    let hash = PyannoteSpeakerSegmenter::SEGMENTER_HASH;
    assert_eq!(hash.len(), 64);
    assert!(
        hash.bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    assert_eq!(
        ModelId::PyannoteSegmentation.name(),
        "pyannote_segmentation"
    );
    assert_ne!(hash, ReDimNetSpeakerEncoder::ENCODER_HASH);
}

/// Kill: regress the frontend or weights on Russian until different presenters
/// are as close as non-adjacent halves of the same original recording.
/// The English cross-session identity case remains the stronger identity proof.
#[test]
#[ignore = "needs current Russian MP3 fixtures, export artifacts and ONNX Runtime"]
fn russian_speakers_separate_the_same_way_english_ones_do() {
    let encoder = encoder();
    let mut pooled = Vec::new();
    for (relative, spans) in [
        ("volkova3.mp3", [(0.0, 60.0), (80.0, 140.0)]),
        (
            "hairdryer-product-placement.mp3",
            [(0.0, 9.0), (10.0, 19.0)],
        ),
    ] {
        let path = fixtures::path(relative).expect("required current Russian recording");
        let decoded = futures_executor::block_on(AudioSource::platform_default().decode(&path))
            .expect("the actual MP3 decodes through production AudioSource");
        let audio = ReDimNetSpeakerEncoder::to_model_rate(decoded).expect("model-rate audio");
        for (start, end) in spans {
            let embeddings = futures_executor::block_on(encoder.embed(
                &audio,
                &[TimeRange {
                    start,
                    duration: end - start,
                }],
            ))
            .expect("actual ReDimNet answers for the original span");
            assert!(embeddings.len() > 1, "{relative} [{start}, {end}]");
            let vectors = embeddings
                .into_iter()
                .map(|value| value.vector)
                .collect::<Vec<_>>();
            pooled.push(stitcher::pooled(&vectors));
        }
    }
    let same = [
        cosine(&pooled[0], &pooled[1]),
        cosine(&pooled[2], &pooled[3]),
    ];
    let cross = [
        cosine(&pooled[0], &pooled[2]),
        cosine(&pooled[0], &pooled[3]),
        cosine(&pooled[1], &pooled[2]),
        cosine(&pooled[1], &pooled[3]),
    ];
    let margin = same.iter().copied().fold(f32::INFINITY, f32::min)
        - cross.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        margin >= 0.25,
        "ru same-speaker {same:?}, ru cross-speaker {cross:?}: margin {margin}"
    );
}
