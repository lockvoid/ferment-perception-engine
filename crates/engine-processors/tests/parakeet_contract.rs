//! `parakeet` against its contract and its golden ladder — both owned by
//! `ferment-models-export` (`models/parakeet/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — the front-end, the encoder's door, the decode and the
//!   host decode's tensors (`members.parakeet_decode`) this processor holds as
//!   constants;
//! * `…_audio_in.f32` → `…_mel_out.f32` — the host's `MelSpec` and its
//!   per-band normalisation against `ParakeetFeatureExtractor`;
//! * `parakeet_decoder_out.f32`, `parakeet_joint_out.f32` — the checkpoint's
//!   prediction network and joint along `reel`'s decode, which the host decode
//!   answers from `parakeet_decode.bin`;
//! * `parakeet_tokens.json` — `ParakeetForTDT.generate`'s tokens, the frame
//!   each came at and the text: what the processor must emit, window by
//!   window, through the real backends.
//!
//! The contract and the mel run everywhere; the lowered halves are
//! `#[ignore]`d like every oracle and need the artifacts.

use ferment_engine_audio::MelSpec;
use ferment_engine_infer::ModelId;
use ferment_engine_processors::Parakeet;
use ferment_engine_processors::parakeet::config::{
    BLANK_TOKEN_ID, DURATION_BINS, ENCODER_FRAMES, HIDDEN, MEL_BANDS, MEL_FRAMES, MEL_SPEC,
    WINDOW_SAMPLES,
};
use ferment_engine_processors::parakeet::features;
use ferment_engine_processors::parakeet::host_decode::{FILE, TENSORS};
use ferment_engine_processors::parakeet::postprocess::build_words;
use ferment_engine_processors::parakeet::vocab::{VOCAB_JSON, Vocab};
use ferment_engine_test_fixtures::export;
use serde_json::json;

const CLIPS: [&str; 3] = ["reel", "ru", "short"];

fn contract() -> Option<serde_json::Value> {
    let path = export::path("parakeet/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

/// `parakeet_tokens.json`; `None` when the export tree is not beside this one.
fn goldens() -> Option<serde_json::Value> {
    let path = export::path("parakeet/goldens/parakeet_tokens.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn audio(clip: &str) -> Vec<f32> {
    export::require_f32_le(&format!("parakeet/goldens/parakeet_{clip}_audio_in.f32"))
}

/// `generate`'s `(token, frame)` pairs for a clip.
fn expected_tokens(goldens: &serde_json::Value, clip: &str) -> Vec<(i32, usize)> {
    goldens[clip]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            (
                pair[0].as_i64().unwrap() as i32,
                pair[1].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

#[test]
fn the_contract_is_the_processors_constants() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "parakeet");
    // Kill: the export delivers an unregistered graph/resource lane.
    for (name, member) in contract["members"].as_object().unwrap() {
        let mut actual = member["artifacts"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let expected = match name.as_str() {
            "parakeet_encoder" => vec!["coreai", "coreml", "onnx"],
            "parakeet_decode" => vec!["resource"],
            _ => vec!["onnx"],
        };
        assert_eq!(actual, expected, "{name}");
    }
    let frontend = &contract["frontend"];
    let spec: MelSpec = serde_json::from_value(frontend["spec"].clone()).unwrap();
    assert_eq!(spec, MEL_SPEC);
    assert_eq!(frontend["window"]["samples"], WINDOW_SAMPLES);
    assert_eq!(frontend["window"]["melFrames"], MEL_FRAMES);
    assert_eq!(frontend["window"]["encoderFrames"], ENCODER_FRAMES);
    assert_eq!(contract["decode"]["blank"], BLANK_TOKEN_ID);
    assert_eq!(contract["decode"]["durations"], json!(DURATION_BINS));

    // The one graph: the encoder, on both lanes.
    assert_eq!(Parakeet::MODEL_FILES, [ModelId::ParakeetEncoder]);
    let encoder = &contract["members"]["parakeet_encoder"];
    assert_eq!(
        encoder["function"]["inputs"],
        json!([
            {"name": "mel", "shape": [1, 1, MEL_BANDS, MEL_FRAMES], "dtype": "per artifact"},
            {"name": "mask", "shape": [1, 1, 1, MEL_FRAMES], "dtype": "per artifact"},
        ])
    );
    assert_eq!(
        encoder["function"]["outputs"],
        json!([{"name": "encoded", "shape": [1, HIDDEN, 1, ENCODER_FRAMES], "dtype": "per artifact"}])
    );
    assert_eq!(
        encoder["artifacts"]["coreml"]["file"],
        "parakeet_encoder.mlmodelc"
    );
    assert_eq!(
        encoder["artifacts"]["onnx"]["file"],
        "parakeet_encoder.onnx"
    );

    // The host decode's weights, tensor for tensor in file order.
    assert_eq!(Parakeet::RESOURCE_FILES, [FILE]);
    let resource = &contract["members"]["parakeet_decode"]["artifacts"]["resource"];
    assert_eq!(resource["file"], FILE);
    assert_eq!(resource["dtype"], "float16");
    assert_eq!(resource["byteOrder"], "little");
    assert_eq!(resource["layout"], "row major, back to back");
    assert_eq!(
        resource["gates"],
        "input, forget, cell, output; each layer's bias b_ih + b_hh"
    );
    assert_eq!(
        resource["joint"],
        "head · relu(encoded[t] + decoder) + head_bias"
    );
    let tensors = resource["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| {
            (
                tensor["name"].as_str().unwrap().to_owned(),
                serde_json::from_value::<Vec<usize>>(tensor["shape"].clone()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let ours = TENSORS
        .iter()
        .map(|(name, shape)| ((*name).to_owned(), shape.to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(tensors, ours);
    let bytes: usize = TENSORS
        .iter()
        .map(|(_, shape)| 2 * shape.iter().product::<usize>())
        .sum();
    assert_eq!(resource["pins"][0]["bytes"], bytes);

    // The vocabulary is the export's bytes, not a re-serialization of them.
    let vocabulary = &contract["vocabulary"];
    assert_eq!(vocabulary["file"], "parakeet_vocab.json");
    assert_eq!(vocabulary["pieces"], BLANK_TOKEN_ID);
    assert_eq!(vocabulary["wordStart"], "\u{2581}");
    assert_eq!(vocabulary["bytes"], VOCAB_JSON.len());
    // Kill: two matching vocab copies disagree with the export's authoritative digest.
    assert_eq!(
        vocabulary["sha256"].as_str().unwrap(),
        ferment_engine_models::sha256_bytes(VOCAB_JSON.as_bytes())
    );
    assert_eq!(Vocab::load().unwrap().len(), BLANK_TOKEN_ID);
    let exported = std::fs::read(export::require_path("parakeet/parakeet_vocab.json")).unwrap();
    assert_eq!(VOCAB_JSON.as_bytes(), exported);
}

/// The host half against `ParakeetFeatureExtractor`: the log-mel of the real
/// samples, each band normalised over the valid frames, the rest zero.
#[test]
fn the_mel_is_the_extractors() {
    let Some(goldens) = goldens() else { return };
    for clip in CLIPS {
        let golden =
            export::require_f32_le(&format!("parakeet/goldens/parakeet_{clip}_mel_out.f32"));
        let (ours, valid) = features(&audio(clip)).unwrap();
        // Kill: accept a truncated or overlong golden by grading only its prefix.
        assert_eq!(golden.len(), MEL_BANDS * MEL_FRAMES, "{clip}");
        assert_eq!(ours.len(), golden.len(), "{clip}");
        assert_eq!(
            valid as u64,
            goldens[clip]["validMelFrames"].as_u64().unwrap(),
            "{clip}"
        );
        let mut worst = 0.0_f32;
        for band in 0..MEL_BANDS {
            for frame in 0..MEL_FRAMES {
                let delta = ours[band * MEL_FRAMES + frame] - golden[frame * MEL_BANDS + band];
                worst = worst.max(delta.abs());
            }
        }
        eprintln!("[parakeet-mel] {clip}: max |Δ| vs ParakeetFeatureExtractor {worst:.3e}");
        assert!(worst < 1e-3, "{clip}: {worst}");
    }
}

/// The vocabulary the processor compiles in and its word building, over
/// `generate`'s tokens: the text the transformers tokenizer decodes.
#[test]
fn the_golden_tokens_spell_the_golden_text() {
    let Some(goldens) = goldens() else { return };
    let vocab = Vocab::load().unwrap();
    for clip in CLIPS {
        let (tokens, frames): (Vec<i32>, Vec<i32>) = expected_tokens(&goldens, clip)
            .into_iter()
            .map(|(token, frame)| (token, frame as i32))
            .unzip();
        let ones = vec![1; tokens.len()];
        let words = build_words(&tokens, &frames, &vec![1.0; tokens.len()], &ones, vocab);
        let text = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(text, goldens[clip]["text"].as_str().unwrap(), "{clip}");
    }
}

mod host {
    use ferment_engine_processors::parakeet::config::LOGITS;
    use ferment_engine_processors::parakeet::decoder::{Encoded, decode_window};
    use ferment_engine_processors::parakeet::host_decode::{HostDecode, Prediction};

    use super::*;

    /// Kill: deliver a same-sized but changed host decoder weight file.
    /// Current Swift contract grades the installed resource's actual bytes.
    #[test]
    #[ignore = "needs the export contract and actual parakeet_decode.bin"]
    fn published_decode_weights_match_the_contract() {
        let contract = contract().expect("required current Parakeet export contract");
        let resource = &contract["members"]["parakeet_decode"]["artifacts"]["resource"];
        let pin = &resource["pins"][0];
        assert_eq!(pin["object"], FILE);
        let path = export::artifact(FILE).expect("required actual decoder weights");
        let bytes = std::fs::read(path).expect("read the installed decoder resource");
        assert_eq!(bytes.len() as u64, pin["bytes"].as_u64().unwrap());
        assert_eq!(
            ferment_engine_models::sha256_bytes(&bytes),
            pin["tree_sha256"].as_str().unwrap()
        );
    }

    /// The host decode is fp32 arithmetic over the fp16 tensors — the same
    /// weights the ONNX graphs store fp16 and run fp32. The export measured
    /// that lane against the checkpoint (`parity.onnx`) at max |Δ| 3.16e-3 on
    /// the prediction network and 2.65e-2 on the joint; the bounds keep twice
    /// that.
    const DECODER_BOUND: f32 = 6.4e-3;
    const JOINT_BOUND: f32 = 5.3e-2;

    fn weights() -> Option<HostDecode> {
        Some(HostDecode::load(&export::artifact(FILE)?).unwrap())
    }

    fn encoded(clip: &str, valid: usize) -> Encoded {
        let values =
            export::require_f32_le(&format!("parakeet/goldens/parakeet_{clip}_encoded_out.f32"));
        Encoded::from_door(&values, valid).unwrap()
    }

    fn worst(ours: &[f32], golden: &[f32]) -> f32 {
        assert_eq!(ours.len(), golden.len());
        ours.iter()
            .zip(golden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// `parakeet_decoder_out.f32` is `[3, 3200]`: output ‖ hidden 0 ‖ hidden 1
    /// ‖ cell 0 ‖ cell 1 after blank from zero, then after `reel`'s first two
    /// tokens; `parakeet_joint_out.f32` the joint at frames 0, 40 and 120
    /// against those three outputs.
    #[test]
    #[ignore = "needs parakeet_decode.bin (FERMENT_ENGINE_MODELS_DIR or ferment-models-export/artifacts)"]
    fn the_host_decode_is_the_checkpoints_prediction_network_and_joint() {
        let Some(goldens) = goldens() else { return };
        let Some(weights) = weights() else { return };
        let tokens = expected_tokens(&goldens, "reel");
        let mut steps = Vec::new();
        let mut state = Prediction::zero();
        for token in [BLANK_TOKEN_ID, tokens[0].0 as usize, tokens[1].0 as usize] {
            state = weights.predict(token, &state);
            steps.push(state.clone());
        }
        let decoder = steps
            .iter()
            .flat_map(|step| {
                [
                    &step.output,
                    &step.hidden[0],
                    &step.hidden[1],
                    &step.cell[0],
                    &step.cell[1],
                ]
                .into_iter()
                .flatten()
                .copied()
            })
            .collect::<Vec<f32>>();
        let decoder_drift = worst(
            &decoder,
            &export::require_f32_le("parakeet/goldens/parakeet_decoder_out.f32"),
        );

        let reel = encoded("reel", ENCODER_FRAMES);
        let mut joint = Vec::new();
        for (frame, step) in [(0, &steps[0]), (40, &steps[1]), (120, &steps[2])] {
            let mut logits = vec![0.0; LOGITS];
            weights.joint(reel.frame(frame), &step.output, &mut logits);
            joint.extend(logits);
        }
        let joint_drift = worst(
            &joint,
            &export::require_f32_le("parakeet/goldens/parakeet_joint_out.f32"),
        );
        eprintln!(
            "[parakeet-host] vs the checkpoint: max |Δ| decoder {decoder_drift:.3e}, joint {joint_drift:.3e}"
        );
        assert!(decoder_drift < DECODER_BOUND, "decoder: {decoder_drift}");
        assert!(joint_drift < JOINT_BOUND, "joint: {joint_drift}");
    }

    /// On the checkpoint's own encoder output the host decode is `generate`,
    /// token for token and frame for frame — the export's `parity.host`.
    #[test]
    #[ignore = "needs parakeet_decode.bin (FERMENT_ENGINE_MODELS_DIR or ferment-models-export/artifacts)"]
    fn the_host_decode_says_generates_tokens() {
        let Some(goldens) = goldens() else { return };
        let Some(weights) = weights() else { return };
        for clip in CLIPS {
            let valid = goldens[clip]["validEncoderFrames"].as_u64().unwrap() as usize;
            let emitted = decode_window(&weights, &encoded(clip, valid));
            let ours = emitted
                .tokens
                .iter()
                .copied()
                .zip(emitted.frames.iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(ours, expected_tokens(&goldens, clip), "{clip}");
            if clip == "reel" {
                // The only hard assertion in current ParakeetHostDecodeCostTests.
                // Kill: change the nonblank advance count while preserving timing logs.
                assert_eq!(ours.len(), 85, "reel's nonblank prediction advances");
            }
        }
    }
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use ferment_engine_infer::Backend;

    use super::*;

    /// The processor as the app runs it — audio in, a window's mel, the lane's
    /// encoder, the host decode — for each golden clip.
    pub fn decode(backend: Backend) -> Option<Vec<Vec<(i32, usize)>>> {
        let decode = export::artifact(FILE)?;
        let parakeet = Parakeet::new(backend, decode);
        Some(
            CLIPS
                .iter()
                .map(|clip| {
                    let emitted =
                        futures_executor::block_on(parakeet.decode(&audio(clip))).unwrap();
                    emitted
                        .tokens
                        .iter()
                        .copied()
                        .zip(emitted.frames.iter().copied())
                        .collect()
                })
                .collect(),
        )
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_infer::{ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};

    use super::*;

    /// fp32 arithmetic over fp16-stored encoder weights: the export measured
    /// the ONNX lane's tokens identical to `generate` on all three clips
    /// (`parity.onnx.tokens`), and the host decode is fp32 too.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the parakeet artifacts"]
    fn the_processor_answers_generate() {
        let Some(goldens) = goldens() else { return };
        let Some(encoder) = export::artifact("parakeet_encoder.onnx") else {
            return;
        };
        let paths =
            ModelPaths::new().with_path(ModelId::ParakeetEncoder, ModelVersion::Latest, encoder);
        let backend =
            OrtBackend::with_execution_provider(Arc::new(paths), OrtExecutionProvider::Cpu);
        let Some(emitted) = super::lowered::decode(Arc::new(backend)) else {
            return;
        };
        for (clip, ours) in CLIPS.iter().zip(emitted) {
            let expected = expected_tokens(&goldens, clip);
            eprintln!(
                "[parakeet-e2e] ort {clip}: {} tokens against generate's {}",
                ours.len(),
                expected.len()
            );
            assert_eq!(ours, expected, "ort {clip}");
        }
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{ComputeUnits, CoreMlBackend, ModelPaths, ModelVersion};
    use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

    use super::lowered::decode;
    use super::*;

    fn words(tokens: &[(i32, usize)]) -> Vec<String> {
        let vocab = Vocab::load().unwrap();
        tokens
            .iter()
            .map(|(token, _)| vocab.lookup(*token).unwrap())
            .collect::<String>()
            .replace('\u{2581}', " ")
            .split_whitespace()
            .map(str::to_owned)
            .collect()
    }

    /// Substitutions, insertions and deletions between two word sequences.
    fn word_errors(expected: &[String], said: &[String]) -> usize {
        let mut previous = (0..=said.len()).collect::<Vec<usize>>();
        for (row, word) in expected.iter().enumerate() {
            let mut current = vec![row + 1; said.len() + 1];
            for (column, other) in said.iter().enumerate() {
                current[column + 1] = (previous[column] + usize::from(word != other))
                    .min(previous[column + 1] + 1)
                    .min(current[column] + 1);
            }
            previous = current;
        }
        previous[said.len()]
    }

    /// The palettized fp16 encoder on the units the fleet declares for it. The
    /// export grades this lane by the words, not the tokens — fp16 flips a
    /// near-tie — and measured it on the CPU and the Neural Engine
    /// (`parity.coreml.CPU_AND_NE.tokens`) at 0 words wrong on `reel`, 1 on
    /// `ru` and 0 on `short`; each clip is held to that.
    const WORD_ERRORS: [usize; 3] = [0, 1, 0];

    #[test]
    #[ignore = "needs the parakeet_encoder.mlmodelc and parakeet_decode.bin artifacts"]
    fn the_processor_says_generates_words() {
        let Some(goldens) = goldens() else { return };
        let Some(encoder) = export::artifact("parakeet_encoder.mlmodelc") else {
            return;
        };
        let declared = ferment_engine_models::default_fleet_manifest()
            .model(ModelId::ParakeetEncoder)
            .unwrap()
            .placement_for(BackendKind::CoreMl);
        assert_eq!(
            declared,
            Some(Placement::CoreMl {
                units: CoreMlComputeUnits::CpuAndNeuralEngine
            })
        );
        let paths =
            ModelPaths::new().with_path(ModelId::ParakeetEncoder, ModelVersion::Latest, encoder);
        let backend =
            CoreMlBackend::with_compute_units(Arc::new(paths), ComputeUnits::CpuAndNeuralEngine);
        let Some(emitted) = decode(Arc::new(backend)) else {
            return;
        };
        for ((clip, ours), allowed) in CLIPS.iter().zip(emitted).zip(WORD_ERRORS) {
            let expected = words(&expected_tokens(&goldens, clip));
            let said = words(&ours);
            let errors = word_errors(&expected, &said);
            eprintln!(
                "[parakeet-e2e] coreml {clip}: {errors} of {} words wrong: «{}»",
                expected.len(),
                said.join(" ")
            );
            assert!(errors <= allowed, "coreml {clip}: {errors} words wrong");
        }
    }
}
