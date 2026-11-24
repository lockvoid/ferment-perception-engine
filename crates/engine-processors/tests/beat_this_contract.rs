//! `beat_this_{1,2,3}` against their contract and their golden ladder — both
//! owned by `ferment-models-export` (`models/beat_this/`), the same bytes that
//! grade the iOS engine:
//!
//! * `model.json` — the front-end this processor holds as a constant, the
//!   chunk it cuts, and the door every lowering shares;
//! * `beat_this_audio_in` → `beat_this_mel_out` — the host half, against the
//!   authors' own `LogMelSpect` (torchaudio);
//! * `beat_this_mel_out` → `beat_this_<n>_logits_out` — each shipping member's
//!   master through the real ONNX Runtime backend, against the authors' own
//!   network.
//!
//! The first two run everywhere; the ORT half is `#[ignore]`d like every
//! oracle (`ORT_DYLIB_PATH`) and needs the lowered masters, which live in the
//! export repository only between an export and its upload.

use ferment_engine_audio::MelSpec;
use ferment_engine_processors::beatthis::config::{
    BORDER_SIZE, CHUNK_SIZE, HOP_LENGTH, MEL_SPEC, N_MELS, SAMPLE_RATE,
};
use ferment_engine_processors::beatthis::mel::compute_log_mel;
use ferment_engine_test_fixtures::export;

const AUDIO: &str = "beat_this/goldens/beat_this_audio_in.bin";
const MEL: &str = "beat_this/goldens/beat_this_mel_out.f32";

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
    let Some(path) = export::path("beat_this/model.json") else {
        return;
    };
    let contract: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(frontend["chunk"]["frames"], CHUNK_SIZE);
    assert_eq!(frontend["chunk"]["border"], BORDER_SIZE);

    let main = &contract["functions"][0];
    assert_eq!(main["inputs"][0]["name"], "mel");
    assert_eq!(
        main["inputs"][0]["shape"],
        serde_json::json!([1, 1, N_MELS, CHUNK_SIZE])
    );
    assert_eq!(main["outputs"][0]["name"], "logits");
    assert_eq!(
        main["outputs"][0]["shape"],
        serde_json::json!([1, 2, 1, CHUNK_SIZE])
    );
    assert_eq!(
        main["outputs"][0]["rows"],
        serde_json::json!(["beat", "downbeat"])
    );
    for member in ["beat_this_1", "beat_this_2", "beat_this_3"] {
        let onnx = &contract["members"][member]["artifacts"]["onnx"];
        assert_eq!(onnx["file"], format!("{member}.onnx"));
        assert_eq!(onnx["dtype"], "float32");
    }
}

/// `log1p(1000 · x)` flattens the transform's own float32 rounding instead of
/// amplifying it the way a decibel scale does, so one bound holds on every
/// cell: 1e-3 of the authors' torchaudio front-end.
#[test]
fn the_mel_is_the_authors_front_end() {
    let Some(audio) = export::f32_le(AUDIO) else {
        return;
    };
    let golden = export::require_f32_le(MEL);
    let frames = audio.len() / HOP_LENGTH + 1;
    assert_eq!(audio.len(), 30 * SAMPLE_RATE as usize);
    assert_eq!(golden.len(), N_MELS * frames);

    let ours = compute_log_mel(&audio).unwrap();
    assert_eq!(ours.len(), frames);
    let mut bands_major = vec![0.0_f32; golden.len()];
    for (frame, row) in ours.iter().enumerate() {
        for (band, value) in row.iter().enumerate() {
            bands_major[band * frames + frame] = *value;
        }
    }
    let delta = worst(&bands_major, &golden);
    eprintln!("[beat-this-mel] max |Δ| vs torchaudio {delta:.3e}");
    assert!(delta < 1e-3, "{delta}");
}

/// The golden mel cut to the door: `mel [1, 1, 128, 1500]`, bands major.
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
fn door() -> Option<Vec<f32>> {
    let bands_major = export::f32_le(MEL)?;
    let frames = bands_major.len() / N_MELS;
    let mut door = vec![0.0_f32; N_MELS * CHUNK_SIZE];
    for band in 0..N_MELS {
        door[band * CHUNK_SIZE..(band + 1) * CHUNK_SIZE]
            .copy_from_slice(&bands_major[band * frames..band * frames + CHUNK_SIZE]);
    }
    Some(door)
}

/// Each shipping member, fed the golden mel at the door, against the authors'
/// network: its logits' max |Δ| under `bound`.
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
fn assert_members(
    backend: impl Fn(ferment_engine_infer::ModelId, std::path::PathBuf) -> ferment_engine_infer::Backend,
    extension: &str,
    label: &str,
    bounds: [f32; 2],
) {
    use ferment_engine_infer::{ModelId, ModelVersion, Tensor, TensorMap};

    let Some(door) = door() else { return };
    for ((id, name), bound) in [
        (ModelId::BeatThis1, "beat_this_1"),
        (ModelId::BeatThis2, "beat_this_2"),
    ]
    .into_iter()
    .zip(bounds)
    {
        let Some(artifact) = export::artifact(&format!("{name}.{extension}")) else {
            return;
        };
        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(vec![1, 1, N_MELS as i64, CHUNK_SIZE as i64], door.clone()).unwrap(),
        );
        let outputs = futures_executor::block_on(backend(id, artifact).infer(
            id,
            &ModelVersion::Latest,
            inputs,
        ))
        .unwrap();
        let actual = outputs["logits"].to_f32().unwrap();
        let golden = export::require_f32_le(&format!("beat_this/goldens/{name}_logits_out.f32"));
        let delta = worst(&actual, &golden);
        eprintln!("[beat-this-graph] {label} {name}: max |Δ| vs the authors' network {delta:.3e}");
        assert!(delta < bound, "{label} {name}: {delta}");
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_infer::{ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};

    use super::*;

    /// fp32 arithmetic over fp16-stored weights: the export measured the
    /// masters 5.9e-3 (member 1) and 7.3e-3 (member 2) from the authors'
    /// network (`parity.<member>.onnx.vs_stock`).
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the beat_this artifacts"]
    fn the_shipping_masters_match_the_authors_network() {
        assert_members(
            |id, master| {
                let paths = ModelPaths::new().with_path(id, ModelVersion::Latest, master);
                Arc::new(OrtBackend::with_execution_provider(
                    Arc::new(paths),
                    OrtExecutionProvider::Cpu,
                ))
            },
            "onnx",
            "ort",
            [2e-2, 2e-2],
        );
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion};
    use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

    use super::*;

    /// The palettized fp16 bundles on the units the fleet pins, the CPU and
    /// the Neural Engine: the export measured them 0.194 (member 1) and 0.179
    /// (member 2) from the authors' network there
    /// (`parity.<member>.coreml.<member>.CPU_AND_NE`); the bounds keep twice
    /// that.
    #[test]
    #[ignore = "needs the beat_this_{1,2}.mlmodelc artifacts"]
    fn the_shipping_bundles_match_the_authors_network() {
        let fleet = ferment_engine_models::default_fleet_manifest();
        for id in [ModelId::BeatThis1, ModelId::BeatThis2] {
            assert_eq!(
                fleet.model(id).unwrap().placement_for(BackendKind::CoreMl),
                Some(Placement::CoreMl {
                    units: CoreMlComputeUnits::CpuAndNeuralEngine
                })
            );
        }
        assert_members(
            |id, bundle| {
                let paths = ModelPaths::new().with_path(id, ModelVersion::Latest, bundle);
                Arc::new(CoreMlBackend::with_compute_units(
                    Arc::new(paths),
                    ComputeUnits::CpuAndNeuralEngine,
                ))
            },
            "mlmodelc",
            "coreml",
            [0.39, 0.36],
        );
    }
}
