//! `dfnet` against its contract and its golden chunk — both owned by
//! `ferment-models-export` (`models/dfnet/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — ONE door on every runtime, the STREAMING one: 100 frames a
//!   call, the three GRU states and the three causal convolutions' input tails
//!   handed from each call to the next, a 1500-frame chunk starting from zeros;
//!   and the two steps the HOST owns either side of it — libdf's analysis with
//!   the complex axis folded into channels and shifted two frames, then the
//!   mask, the deep filter and the inverse STFT;
//! * `feat_erb` + `feat_spec` → `erb_mask`, `df_coefs`, `lsnr` on fifteen
//!   seconds of the hairdryer fixture, graded against the checkpoint's
//!   whole-chunk answer through `DeepFilterNet::run_chunk` — the processor's
//!   own fifteen calls, so a seam the host drops is a seam these catch;
//! * `audio_in` → `enhanced_out`: the same fifteen seconds through
//!   `DeepFilterNet::denoise` as the app calls it, every sample the chunk's own
//!   frames make held to the checkpoint's.
//!
//! The contract runs everywhere; the lowered halves are `#[ignore]`d like every
//! oracle and need the artifacts.

use ferment_engine_processors::deepfilternet::config::{
    CHUNK_FRAMES, DF_LOOKAHEAD, DF_ORDER, FFT_SIZE, HOP_SIZE, NB_DF, NB_ERB, SAMPLE_RATE,
    STREAM_FRAMES, STREAM_STATE,
};
use ferment_engine_test_fixtures::export;
use serde_json::json;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("dfnet/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn names_and_shapes(tensors: &serde_json::Value) -> Vec<(String, serde_json::Value)> {
    tensors
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| {
            (
                tensor["name"].as_str().unwrap().to_owned(),
                tensor["shape"].clone(),
            )
        })
        .collect()
}

/// The processor's door, name for name and shape for shape — a state the host
/// forgets to carry is a seam, not an error, so the state list is graded too.
#[test]
fn the_contract_is_one_streaming_door_on_every_runtime() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "dfnet");

    let functions = contract["functions"].as_array().unwrap();
    assert_eq!(functions.len(), 1, "one door");
    let door = &functions[0];
    assert_eq!(door["name"], "dfnet");
    assert_eq!(door["artifacts"], json!(["coreai", "coreml", "onnx"]));

    let mut inputs = vec![
        ("feat_erb".to_owned(), json!([1, 1, STREAM_FRAMES, NB_ERB])),
        ("feat_spec".to_owned(), json!([1, 2, STREAM_FRAMES, NB_DF])),
    ];
    inputs.extend(
        STREAM_STATE
            .iter()
            .map(|state| (state.input.to_owned(), json!(state.shape))),
    );
    assert_eq!(names_and_shapes(&door["inputs"]), inputs);

    let mut outputs = vec![
        ("erb_mask".to_owned(), json!([1, 1, STREAM_FRAMES, NB_ERB])),
        (
            "df_coefs".to_owned(),
            json!([1, DF_ORDER, STREAM_FRAMES, NB_DF, 2]),
        ),
        ("lsnr".to_owned(), json!([1, STREAM_FRAMES, 1])),
    ];
    outputs.extend(
        STREAM_STATE
            .iter()
            .map(|state| (state.output.to_owned(), json!(state.shape))),
    );
    assert_eq!(names_and_shapes(&door["outputs"]), outputs);

    let geometry = &contract["geometry"];
    assert_eq!(geometry["sample_rate"], SAMPLE_RATE);
    assert_eq!(geometry["hop"], HOP_SIZE);
    assert_eq!(geometry["frames"], CHUNK_FRAMES);
    assert_eq!(geometry["stream_frames"], STREAM_FRAMES);
    assert_eq!(geometry["erb_bands"], NB_ERB);
    assert_eq!(geometry["df_bins"], NB_DF);
    assert_eq!(geometry["df_order"], DF_ORDER);
    // The host's causal shift: features move forward two frames, the spectrum
    // does not. Getting this wrong is silence in the wrong places.
    assert_eq!(geometry["lookahead"], DF_LOOKAHEAD);

    // Every lane ships the streaming door, and nothing ships beside it.
    let artifacts = contract["members"]["dfnet"]["artifacts"]
        .as_object()
        .unwrap();
    let mut lanes = artifacts.keys().collect::<Vec<_>>();
    lanes.sort();
    assert_eq!(lanes, ["coreai", "coreml", "onnx"]);
    for (lane, file, dtype) in [
        ("coreml", "dfnet.mlmodelc", "float16"),
        ("onnx", "dfnet.onnx", "float32"),
    ] {
        assert_eq!(artifacts[lane]["file"], file);
        assert_eq!(artifacts[lane]["dtype"], dtype);
        assert_eq!(artifacts[lane]["port"], "streaming");
        assert_eq!(artifacts[lane]["ships"], true);
    }
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use ferment_engine_audio::AudioFrames;
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::Backend;
    use ferment_engine_processors::DeepFilterNet;

    use super::*;

    /// Largest `|Δ|` a lane may put on each output of the golden chunk.
    pub struct Bounds {
        pub erb_mask: f32,
        pub df_coefs: f32,
        pub lsnr: f32,
    }

    fn golden(name: &str) -> Vec<f32> {
        export::require_f32_le(&format!("dfnet/goldens/dfnet_{name}.f32"))
    }

    fn worst(label: &str, produced: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(produced.len(), expected.len(), "{label}");
        produced
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// The golden chunk through the processor's own streaming loop, every
    /// frame after every seam held to the checkpoint's whole-chunk answer.
    pub fn assert_the_golden_chunk(backend: Backend, lane: &str, bounds: &Bounds) {
        let answer = futures_executor::block_on(DeepFilterNet::new(backend).run_chunk(
            &golden("feat_erb_in"),
            &golden("feat_spec_in"),
            CHUNK_FRAMES,
        ))
        .unwrap();
        let erb_mask = worst("erb_mask", &answer.erb_mask, &golden("erb_mask_out"));
        let df_coefs = worst("df_coefs", &answer.df_coefs, &golden("df_coefs_out"));
        let lsnr = worst("lsnr", &answer.lsnr, &golden("lsnr_out"));
        eprintln!(
            "[dfnet] {lane} vs the checkpoint, {CHUNK_FRAMES} frames — max |Δ| erb_mask \
             {erb_mask:.3e}, df_coefs {df_coefs:.3e}, lsnr {lsnr:.3e}"
        );
        assert!(erb_mask < bounds.erb_mask, "{lane} erb_mask: {erb_mask}");
        assert!(df_coefs < bounds.df_coefs, "{lane} df_coefs: {df_coefs}");
        assert!(lsnr < bounds.lsnr, "{lane} lsnr: {lsnr}");
    }

    /// The golden chunk's audio through the processor as the app calls it.
    /// The golden is the synthesis as it came out, warm-up included, and the
    /// processor trims the warm-up; the samples after it come from the
    /// processor's second chunk, which the golden does not have.
    pub fn assert_the_processor(backend: Backend, lane: &str, bound: f32) {
        let denoised = futures_executor::block_on(DeepFilterNet::new(backend).denoise(
            AudioFrames::new(golden("audio_in"), SAMPLE_RATE),
            ProgressSink::noop(),
        ))
        .unwrap()
        .samples;
        let expected = &golden("enhanced_out")[FFT_SIZE - HOP_SIZE..];
        let audio = worst("audio", &denoised[..expected.len()], expected);
        eprintln!("[dfnet] {lane} processor vs the checkpoint, audio max |Δ| {audio:.3e}");
        assert!(audio < bound, "{lane} audio: {audio}");
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_infer::{
        ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };

    use super::lowered::{Bounds, assert_the_golden_chunk, assert_the_processor};
    use super::*;

    /// fp32 behind the streaming door: the export measured every tensor at its
    /// PSNR ceiling (`parity.onnx`) — max |Δ| 3.5e-6 on the mask, 1.7e-6 on
    /// the coefficients, 2.1e-5 dB on lsnr. The bounds keep ten times that for
    /// another ONNX Runtime build's kernels, and sit four orders under what a
    /// dropped seam costs: carrying only the GRU states moves the mask by 0.36
    /// at every seam (`models/dfnet/port.py`).
    const BOUNDS: Bounds = Bounds {
        erb_mask: 3.5e-5,
        df_coefs: 1.7e-5,
        lsnr: 2.1e-4,
    };

    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_streaming_door_answers_the_checkpoint() {
        if contract().is_none() {
            return;
        }
        let Some(artifact) = export::artifact("dfnet.onnx") else {
            return;
        };
        let paths = ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, artifact);
        assert_the_golden_chunk(
            Arc::new(OrtBackend::with_execution_provider(
                Arc::new(paths),
                OrtExecutionProvider::Cpu,
            )),
            "ort",
            &BOUNDS,
        );
    }

    /// The export measured the fp32 door's audio 2.98e-7 from the
    /// checkpoint's (`parity.onnx.audio`); the host's own libdf half adds its
    /// ~6e-7 on the features.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_processor_answers_the_checkpoint() {
        if contract().is_none() {
            return;
        }
        let Some(artifact) = export::artifact("dfnet.onnx") else {
            return;
        };
        let paths = ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, artifact);
        assert_the_processor(
            Arc::new(OrtBackend::with_execution_provider(
                Arc::new(paths),
                OrtExecutionProvider::Cpu,
            )),
            "ort",
            1e-5,
        );
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion};
    use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

    use super::lowered::{Bounds, assert_the_golden_chunk, assert_the_processor};
    use super::*;

    /// fp16 at both ends. The export measured (`parity.coreml`), max |Δ| on
    /// the mask / the coefficients / lsnr: 0.212 / 0.0168 / 0.155 dB on the
    /// CPU and 0.211 / 0.0158 / 0.257 on the Neural Engine. Carrying only the
    /// GRU states moves the coefficients by 0.446 and lsnr by 6.83 dB after
    /// the seams, so those two bounds tell a seam from fp16; the mask's cannot
    /// — its fp16 drift and the seam's (0.36) are neighbours. The bounds are
    /// iOS's (`DeepFilterNetCoreAIParityTests`).
    const BOUNDS: Bounds = Bounds {
        erb_mask: 0.5,
        df_coefs: 0.1,
        lsnr: 2.0,
    };

    /// The bundle on the units the fleet declares for it.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/dfnet.mlmodelc"]
    fn the_core_ml_streaming_door_answers_the_checkpoint() {
        if contract().is_none() {
            return;
        }
        let Some(bundle) = export::artifact("dfnet.mlmodelc") else {
            return;
        };
        let declared = ferment_engine_models::default_fleet_manifest()
            .model(ModelId::Dfnet)
            .unwrap()
            .placement_for(BackendKind::CoreMl);
        assert_eq!(
            declared,
            Some(Placement::CoreMl {
                units: CoreMlComputeUnits::CpuAndNeuralEngine
            })
        );
        let paths = ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, bundle);
        assert_the_golden_chunk(
            Arc::new(CoreMlBackend::with_compute_units(
                Arc::new(paths),
                ComputeUnits::CpuAndNeuralEngine,
            )),
            "coreml",
            &BOUNDS,
        );
    }

    /// The export measured the bundle's audio 3.9e-3 from the checkpoint's on
    /// the Neural Engine (`parity.coreml.CPU_AND_NE.audio`); the bound keeps
    /// twice that.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/dfnet.mlmodelc"]
    fn the_processor_answers_the_checkpoint_on_core_ml() {
        if contract().is_none() {
            return;
        }
        let Some(bundle) = export::artifact("dfnet.mlmodelc") else {
            return;
        };
        let paths = ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, bundle);
        assert_the_processor(
            Arc::new(CoreMlBackend::with_compute_units(
                Arc::new(paths),
                ComputeUnits::CpuAndNeuralEngine,
            )),
            "coreml",
            7.8e-3,
        );
    }
}
