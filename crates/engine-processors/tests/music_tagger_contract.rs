//! `music_tagger_heads` against its contract and its golden rows — both owned
//! by `ferment-models-export` (`models/music_tagger/`), the same bytes that
//! grade the iOS engine:
//!
//! * `model.json` — the door every lowering shares, and the two tables this
//!   crate bundles (tag names and tuned thresholds, in output order);
//! * `music_tagger_embedding_in` → `music_tagger_<axis>_probs_out` — sixteen
//!   embeddings the heads can meet (row 0 is Dasheng's own music golden, a
//!   techno window; rows 1–15 MTG-Jamendo validation tracks) through the
//!   trained heads in fp32 torch.
//!
//! The contract half runs everywhere; the lowered halves are `#[ignore]`d like
//! every oracle and need the artifacts, which live in the export repository
//! only between an export and its upload.

use ferment_engine_processors::dasheng::config::EMBEDDING_DIM;
use ferment_engine_processors::musictagger::vocab::{
    MusicTaggerVocab, THRESHOLDS_JSON, VOCAB_JSON,
};
use ferment_engine_test_fixtures::export;

const EMBEDDINGS: &str = "music_tagger/goldens/music_tagger_embedding_in.f32";
const AXES: [&str; 3] = ["genre", "mood", "instrument"];

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

fn widths() -> [usize; 3] {
    let vocab = MusicTaggerVocab::load().unwrap();
    [
        vocab.genre_tags.len(),
        vocab.mood_tags.len(),
        vocab.instrument_tags.len(),
    ]
}

/// One golden row: the embedding, and the genre, mood and instrument
/// probabilities the trained heads give it.
type Row = (Vec<f32>, [Vec<f32>; 3]);

/// Every golden row, or `None` when the export repository is not beside this
/// checkout.
fn golden_rows() -> Option<Vec<Row>> {
    let embeddings = export::f32_le(EMBEDDINGS)?;
    let rows = embeddings.len() / EMBEDDING_DIM;
    assert_eq!(rows, 16, "the original independent corpus has sixteen rows");
    assert_eq!(embeddings.len(), rows * EMBEDDING_DIM);
    let probabilities = AXES.map(|axis| {
        export::require_f32_le(&format!(
            "music_tagger/goldens/music_tagger_{axis}_probs_out.f32"
        ))
    });
    let widths = widths();
    for (index, probabilities) in probabilities.iter().enumerate() {
        assert_eq!(probabilities.len(), rows * widths[index]);
        assert!(
            probabilities
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }
    Some(
        (0..rows)
            .map(|row| {
                let axis = |index: usize| {
                    probabilities[index][row * widths[index]..(row + 1) * widths[index]].to_vec()
                };
                (
                    embeddings[row * EMBEDDING_DIM..(row + 1) * EMBEDDING_DIM].to_vec(),
                    [axis(0), axis(1), axis(2)],
                )
            })
            .collect(),
    )
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(path) = export::path("music_tagger/model.json") else {
        return;
    };
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(contract["name"], "music_tagger_heads");
    assert_eq!(contract["input"]["from"], "dasheng_base_mel");

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "embedding");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, EMBEDDING_DIM])
    );
    for (index, (axis, width)) in AXES.iter().zip(widths()).enumerate() {
        assert_eq!(main["outputs"][index]["name"], format!("{axis}_probs"));
        assert_eq!(
            main["outputs"][index]["shape"],
            serde_json::json!([1, width])
        );
    }
    assert_eq!(contract["labels"]["axes"], serde_json::json!(AXES));

    // The tables are the export's bytes, not a re-serialization of them.
    for (file, bundled) in [
        ("music_tagger_vocab.json", VOCAB_JSON),
        ("music_tagger_thresholds.json", THRESHOLDS_JSON),
    ] {
        let exported =
            std::fs::read(export::require_path(&format!("music_tagger/{file}"))).unwrap();
        assert_eq!(bundled.as_bytes(), exported, "{file}");
        assert_eq!(contract["labels"]["tables"][file]["bytes"], bundled.len());
    }

    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "music_tagger_heads.onnx"
    );
    assert_eq!(contract["artifacts"]["onnx"]["dtype"], "float32");
    assert_eq!(contract["artifacts"]["coreml"]["dtype"], "float16");
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend,
        OrtExecutionProvider, Tensor, TensorMap,
    };
    use ferment_engine_processors::MusicTagger;

    use super::*;

    fn backend() -> Option<OrtBackend> {
        let master = export::artifact("music_tagger_heads.onnx")?;
        let paths =
            ModelPaths::new().with_path(ModelId::MusicTaggerHeads, ModelVersion::Latest, master);
        Some(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ))
    }

    /// The graph alone, one row at a time through its batch-1 door: what the
    /// export measured through Python onnxruntime (2.4e-7), through this
    /// engine's backend.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_master_matches_the_trained_heads() {
        let (Some(rows), Some(backend)) = (golden_rows(), backend()) else {
            return;
        };
        let mut delta = 0.0_f32;
        for (embedding, expected) in &rows {
            let mut inputs = TensorMap::new();
            inputs.insert(
                "embedding".into(),
                Tensor::from_f32(vec![1, EMBEDDING_DIM as i64], embedding.clone()).unwrap(),
            );
            let outputs = futures_executor::block_on(backend.infer(
                ModelId::MusicTaggerHeads,
                &ModelVersion::Latest,
                inputs,
            ))
            .unwrap();
            for (axis, expected) in AXES.iter().zip(expected) {
                let actual = outputs[&format!("{axis}_probs")].to_f32().unwrap();
                assert!(actual.iter().all(|value| (0.0..=1.0).contains(value)));
                delta = delta.max(worst(&actual, expected));
            }
        }
        eprintln!(
            "[tagger-graph] {} rows: max |Δ| vs the trained heads {delta:.3e}",
            rows.len()
        );
        assert!(delta < 1e-5, "{delta}");
    }

    /// The processor as the app calls it: raw probabilities on every row, and
    /// the tags that fire on Dasheng's techno window at the tuned thresholds.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_processor_tags_as_the_trained_heads() {
        let (Some(rows), Some(backend)) = (golden_rows(), backend()) else {
            return;
        };
        let backend: Backend = Arc::new(backend);
        let tagger = MusicTagger::new(backend);
        let vocab = MusicTaggerVocab::load().unwrap();
        let mut decisions = 0;
        let mut delta = 0.0_f32;
        for (embedding, expected) in &rows {
            let raw = futures_executor::block_on(tagger.raw_probabilities(embedding)).unwrap();
            for ((actual, expected), thresholds) in [&raw.genre, &raw.mood, &raw.instrument]
                .into_iter()
                .zip(expected)
                .zip([
                    &vocab.genre_thresholds,
                    &vocab.mood_thresholds,
                    &vocab.instrument_thresholds,
                ])
            {
                assert_eq!(actual.scores.len(), thresholds.len());
                assert!(
                    actual
                        .scores
                        .iter()
                        .all(|value| (0.0..=1.0).contains(value))
                );
                delta = delta.max(worst(&actual.scores, expected));
                for ((actual, expected), threshold) in
                    actual.scores.iter().zip(expected).zip(thresholds)
                {
                    if (expected - threshold).abs() > 0.005 {
                        decisions += 1;
                        assert_eq!(
                            actual >= threshold,
                            expected >= threshold,
                            "{actual} vs trained {expected} at threshold {threshold}"
                        );
                    }
                }
            }
        }
        assert!(decisions > 0, "the independent corpus must decide tags");
        eprintln!("[tagger-e2e] max |Δ| vs the trained heads {delta:.3e}");
        assert!(delta < 1e-5, "{delta}");

        let techno = futures_executor::block_on(tagger.tag(&rows[0].0, 10)).unwrap();
        let genres: Vec<_> = techno.genre.iter().map(|tag| tag.name.as_str()).collect();
        eprintln!("[tagger-e2e] Dasheng's techno window fires {genres:?}");
        assert_eq!(genres[..2], ["electronic", "techno"]);
    }
    #[test]
    #[ignore = "requires pinned ORT and the published MusicTagger graph"]
    fn real_zero_embedding_and_invalid_dimension_keep_the_current_contract() {
        let backend: Backend = Arc::new(backend().expect("required MusicTagger graph"));
        let tagger = MusicTagger::new(backend);
        let zero = futures_executor::block_on(tagger.tag(&vec![0.0; EMBEDDING_DIM], 5)).unwrap();
        for axis in [&zero.genre, &zero.mood, &zero.instrument] {
            assert!(axis.len() <= 5);
        }
        assert!(futures_executor::block_on(tagger.tag(&[0.0; 100], 5)).is_err());
    }

    #[test]
    #[ignore = "requires pinned ORT, Dasheng/MusicTagger graphs and the frozen current techno fixture"]
    fn real_techno_audio_tags_are_capped_sorted_probabilities() {
        use ferment_engine_core::ProgressSink;
        use ferment_engine_processors::{AudioSemanticEncoder, AudioSource};
        let root = std::path::PathBuf::from(
            std::env::var_os("FERMENT_CURRENT_IOS_ORACLE")
                .expect("name the frozen current iOS checkout"),
        );
        let file = root.join("Packages/FermentEngine/Tests/Fixtures/techno-1-30s.mp3");
        assert!(file.is_file(), "required original techno fixture");
        let paths = ModelPaths::new()
            .with_path(
                ModelId::MusicTaggerHeads,
                ModelVersion::Latest,
                export::artifact("music_tagger_heads.onnx").expect("required MusicTagger graph"),
            )
            .with_path(
                ModelId::DashengBase,
                ModelVersion::Latest,
                export::artifact("dasheng_base_mel.onnx").expect("required Dasheng graph"),
            );
        let backend: Backend = Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ));
        let audio =
            futures_executor::block_on(AudioSource::platform_default().decode(&file)).unwrap();
        let embedding = futures_executor::block_on(
            AudioSemanticEncoder::new(backend.clone()).encode(audio, false, ProgressSink::noop()),
        )
        .unwrap();
        let tags = futures_executor::block_on(MusicTagger::new(backend).tag(&embedding.pooled, 5))
            .unwrap();
        for axis in [&tags.genre, &tags.mood, &tags.instrument] {
            assert!(axis.len() <= 5);
            assert!(axis.windows(2).all(|pair| pair[0].score >= pair[1].score));
            assert!(axis.iter().all(|tag| (0.0..=1.0).contains(&tag.score)));
        }
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion,
    };
    use ferment_engine_processors::MusicTagger;

    use super::*;

    /// The Core ML lane of the same contract, on the units both runtimes pin
    /// (`CpuOnly`). The bundle is float16 at both doors and the processor
    /// speaks float32: Core ML converts. The export measured 1.8e-3 on
    /// `CPU_ONLY` (`parity.coreml.goldens.CPU_ONLY`); the bound keeps twice
    /// that.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/music_tagger_heads.mlmodelc"]
    fn the_processor_matches_the_trained_heads_on_core_ml() {
        assert_eq!(
            ferment_engine_models::default_fleet_manifest()
                .model(ModelId::MusicTaggerHeads)
                .unwrap()
                .placement_for(ferment_engine_models::BackendKind::CoreMl),
            Some(ferment_engine_models::Placement::CoreMl {
                units: ferment_engine_models::CoreMlComputeUnits::CpuOnly
            })
        );
        let Some(rows) = golden_rows() else { return };
        let Some(bundle) = export::artifact("music_tagger_heads.mlmodelc") else {
            return;
        };
        let paths =
            ModelPaths::new().with_path(ModelId::MusicTaggerHeads, ModelVersion::Latest, bundle);
        let backend: Backend = Arc::new(CoreMlBackend::with_compute_units(
            Arc::new(paths),
            ComputeUnits::CpuOnly,
        ));
        let tagger = MusicTagger::new(backend);
        let mut delta = 0.0_f32;
        for (embedding, expected) in &rows {
            let raw = futures_executor::block_on(tagger.raw_probabilities(embedding)).unwrap();
            for (actual, expected) in [&raw.genre, &raw.mood, &raw.instrument]
                .into_iter()
                .zip(expected)
            {
                delta = delta.max(worst(&actual.scores, expected));
            }
        }
        eprintln!("[tagger-coreml] max |Δ| vs the trained heads {delta:.3e}");
        assert!(delta < 3.7e-3, "{delta}");
    }
}
