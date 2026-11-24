//! `music_highlighter` against its contract and its golden ladder — both owned
//! by `ferment-models-export` (`models/music_highlighter/`), the same bytes
//! that grade the iOS engine:
//!
//! * `model.json` — the front-end, the chunking and the door this processor
//!   holds as constants;
//! * `beat_this_audio_in` → `music_highlighter_music_mel_out` — the host's
//!   `MelSpec` against librosa (the audio is BeatThis's golden, reused);
//! * `…_music_logits_out`, `…_track_logits_out` — the door over 10 chunks (a
//!   count that divides 200) and over 37 (one that does not, so the tiling
//!   repeats the first 15 three times), through the real backends.
//!
//! The contract and the mel run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts, which live in the
//! export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::musichighlighter::config::{
    CHUNK_FRAMES, CHUNK_SECONDS, FEATURE_DIM, MAX_CHUNKS, MEL_SPEC, N_MELS, SAMPLE_RATE,
};
use ferment_engine_processors::musichighlighter::mel::compute_log_mel;
use ferment_engine_processors::musichighlighter::pos_enc::positional_encoding;
use ferment_engine_test_fixtures::export;

const AUDIO: &str = "beat_this/goldens/beat_this_audio_in.bin";
const MEL: &str = "music_highlighter/goldens/music_highlighter_music_mel_out.f32";

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

fn contract() -> Option<serde_json::Value> {
    let path = export::path("music_highlighter/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

/// The processor's chunking over a `[frames × 128]` mel of `samples` samples.
fn chunk(mel: &[f32], samples: usize) -> (Vec<f32>, usize) {
    let frames = mel.len() / N_MELS;
    let count = (samples / SAMPLE_RATE as usize) / CHUNK_SECONDS;
    let length = frames / count;
    let copied = length.min(CHUNK_FRAMES);
    let mut chunks = vec![0.0_f32; count * CHUNK_FRAMES * N_MELS];
    for j in 0..count {
        for t in 0..copied {
            let source = (j * length + t) * N_MELS;
            let target = (j * CHUNK_FRAMES + t) * N_MELS;
            chunks[target..target + N_MELS].copy_from_slice(&mel[source..source + N_MELS]);
        }
    }
    (chunks, count)
}

/// One golden case: its label, its real chunks, their count and the door's
/// logits over them.
type Case = (&'static str, Vec<f32>, usize, Vec<f32>);

fn cases() -> Option<Vec<Case>> {
    let audio = export::f32_le(AUDIO)?;
    let (music, music_count) = chunk(&export::require_f32_le(MEL), audio.len());
    let track =
        export::require_f32_le("music_highlighter/goldens/music_highlighter_track_chunks_in.f32");
    let track_count = track.len() / (CHUNK_FRAMES * N_MELS);
    Some(vec![
        (
            "music",
            music,
            music_count,
            export::require_f32_le(
                "music_highlighter/goldens/music_highlighter_music_logits_out.f32",
            ),
        ),
        (
            "track",
            track,
            track_count,
            export::require_f32_le(
                "music_highlighter/goldens/music_highlighter_track_logits_out.f32",
            ),
        ),
    ])
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "music_highlighter");
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(frontend["chunks"]["seconds"], CHUNK_SECONDS);
    assert_eq!(frontend["chunks"]["frames"], CHUNK_FRAMES);
    assert_eq!(frontend["chunks"]["max"], MAX_CHUNKS);
    assert_eq!(frontend["positionalEncoding"]["dimension"], FEATURE_DIM);

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "chunks");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([MAX_CHUNKS, CHUNK_FRAMES, N_MELS])
    );
    assert_eq!(main["inputs"][1]["name"], "pos_enc");
    assert_eq!(
        main["inputs"][1]["shape"],
        serde_json::json!([1, MAX_CHUNKS, FEATURE_DIM])
    );
    assert_eq!(main["outputs"][0]["name"], "logits");
    assert_eq!(main["outputs"][0]["shape"], serde_json::json!([MAX_CHUNKS]));
    assert_eq!(
        contract["artifacts"]["onnx"]["file"],
        "music_highlighter.onnx"
    );
}

/// The host half against librosa, on 30 s of techno.
#[test]
fn the_mel_is_librosas() {
    let Some(audio) = export::f32_le(AUDIO) else {
        return;
    };
    let golden = export::require_f32_le(MEL);
    let ours: Vec<f32> = compute_log_mel(&audio).unwrap().concat();
    let delta = worst(&ours, &golden);
    eprintln!("[highlighter-mel] max |Δ| vs librosa {delta:.3e}");
    assert!(delta < 1e-3, "{delta}");
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use std::sync::Arc;

    use ferment_engine_audio::AudioFrames;
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::{
        Backend, InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap,
    };
    use ferment_engine_processors::musichighlighter::MusicHighlighter;

    use super::*;

    /// The door's two tensors: `count` real chunks tiled over 200 slots, slot
    /// s holding chunk `s mod count`, the encoding alike.
    fn door(chunks: &[f32], count: usize) -> TensorMap {
        let stride = CHUNK_FRAMES * N_MELS;
        let encoding = positional_encoding(count, FEATURE_DIM);
        let mut tiled = vec![0.0_f32; MAX_CHUNKS * stride];
        let mut positions = vec![0.0_f32; MAX_CHUNKS * FEATURE_DIM];
        for slot in 0..MAX_CHUNKS {
            let source = slot % count;
            tiled[slot * stride..(slot + 1) * stride]
                .copy_from_slice(&chunks[source * stride..(source + 1) * stride]);
            positions[slot * FEATURE_DIM..(slot + 1) * FEATURE_DIM]
                .copy_from_slice(&encoding[source * FEATURE_DIM..(source + 1) * FEATURE_DIM]);
        }
        let mut inputs = TensorMap::new();
        inputs.insert(
            "chunks".into(),
            Tensor::from_f32(
                vec![MAX_CHUNKS as i64, CHUNK_FRAMES as i64, N_MELS as i64],
                tiled,
            )
            .unwrap(),
        );
        inputs.insert(
            "pos_enc".into(),
            Tensor::from_f32(vec![1, MAX_CHUNKS as i64, FEATURE_DIM as i64], positions).unwrap(),
        );
        inputs
    }

    /// The graph alone on both golden cases.
    pub(super) fn assert_graph(backend: &dyn InferenceBackend, label: &str, bound: f32) {
        let Some(cases) = cases() else { return };
        for (case, chunks, count, golden) in cases {
            let outputs = futures_executor::block_on(backend.infer(
                ModelId::MusicHighlighter,
                &ModelVersion::Latest,
                door(&chunks, count),
            ))
            .unwrap();
            let logits = outputs["logits"].to_f32().unwrap();
            let delta = worst(&logits, &golden);
            eprintln!(
                "[highlighter-graph] {label} {case} ({count} chunks): max |Δ| vs the door {delta:.3e}"
            );
            assert!(delta < bound, "{label} {case}: {delta}");
        }
    }

    /// Audio in, saliency out — the processor as the app calls it, against
    /// the saliency the golden logits make.
    pub(super) fn assert_processor(backend: Backend, label: &str, bound: f32) {
        let Some(audio) = export::f32_le(AUDIO) else {
            return;
        };
        let golden = export::require_f32_le(
            "music_highlighter/goldens/music_highlighter_music_logits_out.f32",
        );
        let count = (audio.len() / SAMPLE_RATE as usize) / CHUNK_SECONDS;
        let head = &golden[..count];
        let peak = head.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = head.iter().map(|value| (value - peak).exp()).collect();
        let top = exps.iter().copied().fold(0.0_f32, f32::max);
        let expected: Vec<f32> = exps
            .iter()
            .flat_map(|&value| std::iter::repeat_n(value / top, CHUNK_SECONDS))
            .collect();

        let result = futures_executor::block_on(MusicHighlighter::new(backend).detect(
            AudioFrames::new(audio, SAMPLE_RATE),
            vec![15],
            ProgressSink::noop(),
        ))
        .unwrap();
        let delta = worst(&result.saliency, &expected);
        eprintln!(
            "[highlighter-e2e] {label}: saliency max |Δ| vs the door {delta:.3e}, best 15 s window at {}",
            result.highlights[0].start
        );
        assert!(delta < bound, "{label}: {delta}");
    }

    #[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
    mod ort {
        use ferment_engine_infer::{ModelPaths, OrtBackend, OrtExecutionProvider};

        use super::*;

        fn backend() -> Option<OrtBackend> {
            let master = export::artifact("music_highlighter.onnx")?;
            let paths = ModelPaths::new().with_path(
                ModelId::MusicHighlighter,
                ModelVersion::Latest,
                master,
            );
            Some(OrtBackend::with_execution_provider(
                Arc::new(paths),
                OrtExecutionProvider::Cpu,
            ))
        }

        /// fp32 behind the same door: the export measured 5.2e-6.
        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_master_matches_the_door() {
            let Some(backend) = backend() else { return };
            assert_graph(&backend, "ort", 1e-4);
        }

        #[test]
        #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
        fn the_processor_highlights_as_the_door() {
            let Some(backend) = backend() else { return };
            assert_processor(Arc::new(backend), "ort", 1e-4);
        }
    }

    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    mod coreml {
        use ferment_engine_infer::{ComputeUnits, CoreMlBackend, ModelPaths};
        use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

        use super::*;

        /// The fp16 bundle on the units the fleet pins, the CPU.
        fn backend() -> Option<CoreMlBackend> {
            assert_eq!(
                ferment_engine_models::default_fleet_manifest()
                    .model(ModelId::MusicHighlighter)
                    .unwrap()
                    .placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuOnly
                })
            );
            let bundle = export::artifact("music_highlighter.mlmodelc")?;
            let paths = ModelPaths::new().with_path(
                ModelId::MusicHighlighter,
                ModelVersion::Latest,
                bundle,
            );
            Some(CoreMlBackend::with_compute_units(
                Arc::new(paths),
                ComputeUnits::CpuOnly,
            ))
        }

        /// The export measured the logits 0.051 (music) and 0.028 (track) from
        /// the door's there (`parity.coreml.CPU_ONLY`); the bound keeps twice
        /// the worse.
        #[test]
        #[ignore = "needs ferment-models-export/artifacts/music_highlighter.mlmodelc"]
        fn the_bundle_matches_the_door() {
            let Some(backend) = backend() else { return };
            assert_graph(&backend, "coreml", 0.103);
        }

        #[test]
        #[ignore = "needs ferment-models-export/artifacts/music_highlighter.mlmodelc"]
        fn the_processor_highlights_as_the_door_on_core_ml() {
            let Some(backend) = backend() else { return };
            assert_processor(Arc::new(backend), "coreml", 0.103);
        }
    }
}
