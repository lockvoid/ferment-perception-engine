//! `dasheng_base_mel` against its contract and its golden ladder — both owned
//! by `ferment-models-export` (`models/dasheng/`), the same bytes that grade
//! the iOS engine:
//!
//! * `model.json` — the front-end this processor holds as a constant, and the
//!   graph's names and shapes;
//! * `dasheng_audio_in` → `dasheng_mel_out` — the host half, against the
//!   `dasheng` package's own `front_end` (torchaudio), quiet sweep and loud
//!   music;
//! * `dasheng_mel_out` → `dasheng_embedding_out` — the graph through the real
//!   ONNX Runtime backend, and the processor end to end.
//!
//! The first two run everywhere; the ORT half is `#[ignore]`d like every
//! oracle (`ORT_DYLIB_PATH`) and needs the lowered `dasheng_base_mel.onnx`,
//! which lives in the export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::dasheng::config::{
    EMBEDDING_DIM, MEL_BANDS, MEL_FRAMES, MEL_SPEC, PATCH_CHANNELS, PATCH_COUNT, PATCH_FRAMES,
    TOP_DB, WINDOW_SAMPLES,
};
use ferment_engine_processors::dasheng::frontend::{mel, patches};
use ferment_engine_test_fixtures::export;

const CASES: [(&str, &str, &str, &str); 2] = [
    (
        "sweep",
        "dasheng/goldens/dasheng_audio_in.bin",
        "dasheng/goldens/dasheng_mel_out.f32",
        "dasheng/goldens/dasheng_embedding_out.bin",
    ),
    (
        "music",
        "dasheng/goldens/dasheng_music_audio_in.bin",
        "dasheng/goldens/dasheng_music_mel_out.f32",
        "dasheng/goldens/dasheng_music_embedding_out.bin",
    ),
];

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(path) = export::path("dasheng/model.json") else {
        return;
    };
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(contract["name"], "dasheng_base_mel");

    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(
        frontend["perWindow"][0]["clampBelowMaxBy"],
        f64::from(TOP_DB)
    );
    assert_eq!(frontend["window"]["samples"], WINDOW_SAMPLES);
    assert_eq!(frontend["window"]["frames"], MEL_FRAMES);
    assert_eq!(frontend["layout"]["patchFrames"], PATCH_FRAMES);
    assert_eq!(frontend["layout"]["patches"], PATCH_COUNT);

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "mel");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, PATCH_CHANNELS, 1, PATCH_COUNT])
    );
    assert_eq!(main["outputs"][0]["name"], "embedding");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, EMBEDDING_DIM, 1, 1])
    );
    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "dasheng_base_mel.onnx"
    );
    assert_eq!(contract["artifacts"]["onnx"]["dtype"], "float32");
}

/// The gate iOS holds the same front-end to (`MelSpecTests`): under 1e-3 dB
/// of torchaudio — on every cell within 60 dB of the window's peak. A float32
/// transform's own rounding sits ~132 dB under a full-scale bin
/// (`(ε · n)²` against `(n / 2)²`), so a cell 60 dB down still has 72 dB over
/// it (2.5e-4 relative, 1.1e-3 dB) and one 100 dB down has 32; below the line
/// two correct transforms differ by their summation order, which is held to
/// 0.05 dB and, by the end-to-end oracle, never reaches the embedding.
#[test]
fn the_mel_is_the_stock_front_ends() {
    for (label, audio, golden, _) in CASES {
        let Some(audio) = export::f32_le(audio) else {
            return;
        };
        let golden = export::require_f32_le(golden);
        assert_eq!(golden.len(), MEL_BANDS * MEL_FRAMES);
        let ours = mel(&audio).unwrap().bands_major(None);
        let peak = golden.iter().copied().fold(f32::MIN, f32::max);
        let (mut loud, mut quiet) = (0.0_f32, 0.0_f32);
        for (a, e) in ours.iter().zip(&golden) {
            let slot = if *e >= peak - 60.0 {
                &mut loud
            } else {
                &mut quiet
            };
            *slot = slot.max((a - e).abs());
        }
        eprintln!(
            "[dasheng-mel] {label}: max |Δ| vs torchaudio {loud:.3e} dB within 60 dB of the peak, {quiet:.3e} dB below"
        );
        assert!(loud < 1e-3, "{label}: {loud}");
        assert!(quiet < 5e-2, "{label}: {quiet}");
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_audio::{AudioFrames, MelFrames};
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::{
        Backend, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend,
        OrtExecutionProvider, Tensor, TensorMap,
    };
    use ferment_engine_processors::AudioSemanticEncoder;
    use ferment_engine_processors::dasheng::config::SAMPLE_RATE;

    use super::*;

    fn backend() -> Option<OrtBackend> {
        let master = export::artifact("dasheng_base_mel.onnx")?;
        let paths = ModelPaths::new().with_path(ModelId::DashengBase, ModelVersion::Latest, master);
        Some(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ))
    }

    /// The graph alone, fed the golden mel: what the export measured through
    /// Python onnxruntime (2.9e-6), through this engine's backend.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_graph_matches_python_origin() {
        let Some(backend) = backend() else { return };
        for (label, _, golden_mel, embedding) in CASES {
            let bands_major = export::require_f32_le(golden_mel);
            let mut values = vec![0.0; bands_major.len()];
            for band in 0..MEL_BANDS {
                for frame in 0..MEL_FRAMES {
                    values[frame * MEL_BANDS + band] = bands_major[band * MEL_FRAMES + frame];
                }
            }
            let mut inputs = TensorMap::new();
            inputs.insert(
                "mel".into(),
                Tensor::from_f32(
                    vec![1, PATCH_CHANNELS as i64, 1, PATCH_COUNT as i64],
                    patches(&MelFrames {
                        bands: MEL_BANDS,
                        values,
                    }),
                )
                .unwrap(),
            );
            let outputs = futures_executor::block_on(backend.infer(
                ModelId::DashengBase,
                &ModelVersion::Latest,
                inputs,
            ))
            .unwrap();
            let actual = outputs["embedding"].to_f32().unwrap();
            let delta = worst(&actual, &export::require_f32_le(embedding));
            eprintln!("[dasheng-graph] {label}: max |Δ| vs stock torch {delta:.3e}");
            assert!(delta < 1e-4, "{label}: {delta}");
        }
    }

    /// Audio in, embedding out — the processor as the app calls it.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_processor_matches_python_origin_end_to_end() {
        let Some(backend) = backend() else { return };
        let backend: Backend = Arc::new(backend);
        let encoder = AudioSemanticEncoder::new(backend);
        for (label, audio, _, embedding) in CASES {
            let frames = AudioFrames {
                identity: None,
                samples: export::require_f32_le(audio),
                sample_rate: SAMPLE_RATE,
            };
            let result =
                futures_executor::block_on(encoder.encode(frames, false, ProgressSink::noop()))
                    .unwrap();
            let expected = export::require_f32_le(embedding);
            let delta = worst(&result.pooled, &expected);
            assert_eq!(result.pooled.len(), EMBEDDING_DIM);
            assert!(result.pooled.iter().all(|value| value.is_finite()));
            let dot: f32 = result
                .pooled
                .iter()
                .zip(&expected)
                .map(|(a, b)| a * b)
                .sum();
            let norm =
                |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
            let cosine = dot / (norm(&result.pooled) * norm(&expected));
            assert!(cosine > 0.9999, "{label}: cosine {cosine}");
            eprintln!("[dasheng-e2e] {label}: max |Δ| vs stock torch {delta:.3e}");
            assert!(delta < 1e-3, "{label}: {delta}");
        }
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_audio::AudioFrames;
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::{
        Backend, ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion,
    };
    use ferment_engine_processors::AudioSemanticEncoder;
    use ferment_engine_processors::dasheng::config::SAMPLE_RATE;

    use super::*;

    /// The Core ML lane of the same contract, on the units the fleet pins.
    /// The bundle declares `mel` and `embedding` as float16 and the processor
    /// speaks float32 to every backend: Core ML converts at both doors. The
    /// export measured 2.4e-3 (sweep) and 3.3e-3 (music) there
    /// (`parity.coreml.<case>.CPU_AND_NE`); the bound keeps twice that.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/dasheng_base_mel.mlmodelc"]
    fn the_processor_matches_python_origin_on_the_neural_engine() {
        assert_eq!(
            ferment_engine_models::default_fleet_manifest()
                .model(ModelId::DashengBase)
                .unwrap()
                .placement_for(ferment_engine_models::BackendKind::CoreMl),
            Some(ferment_engine_models::Placement::CoreMl {
                units: ferment_engine_models::CoreMlComputeUnits::CpuAndNeuralEngine
            })
        );
        let Some(bundle) = export::artifact("dasheng_base_mel.mlmodelc") else {
            return;
        };
        let paths = ModelPaths::new().with_path(ModelId::DashengBase, ModelVersion::Latest, bundle);
        let backend: Backend = Arc::new(CoreMlBackend::with_compute_units(
            Arc::new(paths),
            ComputeUnits::CpuAndNeuralEngine,
        ));
        let encoder = AudioSemanticEncoder::new(backend);
        for (label, audio, _, embedding) in CASES {
            let frames = AudioFrames {
                identity: None,
                samples: export::require_f32_le(audio),
                sample_rate: SAMPLE_RATE,
            };
            let result =
                futures_executor::block_on(encoder.encode(frames, false, ProgressSink::noop()))
                    .unwrap();
            let delta = worst(&result.pooled, &export::require_f32_le(embedding));
            eprintln!("[dasheng-coreml] {label}: max |Δ| vs stock torch {delta:.3e}");
            assert!(delta < 6.7e-3, "{label}: {delta}");
        }
    }
}
