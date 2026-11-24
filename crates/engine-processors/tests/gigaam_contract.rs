//! `gigaam_*` against its contract and its golden chunk — both owned by
//! `ferment-models-export` (`models/gigaam/`), the same bytes that grade the
//! iOS engine:
//!
//! * `model.json` — the four doors, the encoder ONE door on every runtime (the
//!   mel and a pad mask the host counts, the frames out), the chunk geometry
//!   (240 000 samples → 1499 mel → 375 encoder frames, 640 samples each) and
//!   the RNN-T facts a decoder needs: the blank id that primes the prediction
//!   network and the symbol budget per frame;
//! * the chain — fifteen seconds of real Russian speech, and the SENTENCE the
//!   greedy loop builds out of it. A lowering that drifts one token is a
//!   lowering that says another word, so the gate is the text;
//! * the processor over the external `Speech/speech-ru.m4a`, a clip shorter than
//!   one chunk — where the mask is load-bearing.
//!
//! The contract runs everywhere; the lowered halves are `#[ignore]`d like every
//! oracle and need the artifacts.

use ferment_engine_processors::gigaam::config::{
    BLANK_TOKEN_ID, DECODER_HIDDEN, ENCODER_HIDDEN, MAX_CHUNK_SAMPLES, MAX_ENCODER_FRAMES,
    MAX_MEL_FRAMES, MAX_SYMBOLS_PER_STEP, N_MELS, SAMPLES_PER_ENCODER_FRAME,
};
use ferment_engine_test_fixtures::export;
use serde_json::json;

fn contract() -> Option<serde_json::Value> {
    let path = export::path("gigaam/model.json")?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn doors<'a>(contract: &'a serde_json::Value, name: &str) -> Vec<&'a serde_json::Value> {
    contract["functions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|function| function["name"] == name)
        .collect()
}

fn names_and_shapes(tensors: &serde_json::Value) -> Vec<(&str, serde_json::Value)> {
    tensors
        .as_array()
        .unwrap()
        .iter()
        .map(|tensor| (tensor["name"].as_str().unwrap(), tensor["shape"].clone()))
        .collect()
}

#[test]
fn the_contract_is_the_four_doors_and_the_decoding_rules() {
    let Some(contract) = contract() else { return };
    assert_eq!(contract["name"], "gigaam");

    let preprocessor = doors(&contract, "gigaam_preprocessor");
    assert_eq!(
        names_and_shapes(&preprocessor[0]["inputs"]),
        [("audio", json!([1, MAX_CHUNK_SAMPLES]))]
    );
    assert_eq!(
        names_and_shapes(&preprocessor[0]["outputs"]),
        [("mel", json!([1, N_MELS, MAX_MEL_FRAMES]))]
    );

    // ONE encoder door on every runtime: the mel and the pad mask in, the
    // frames out. No `length` in, no `encoded_len` out — the host counts the
    // frames and hands them over as the mask.
    let encoder = doors(&contract, "gigaam_encoder");
    assert_eq!(encoder.len(), 1, "one encoder door");
    assert_eq!(encoder[0]["artifacts"], json!(["coreai", "coreml", "onnx"]));
    assert_eq!(
        names_and_shapes(&encoder[0]["inputs"]),
        [
            ("mel", json!([1, N_MELS, MAX_MEL_FRAMES])),
            ("mask", json!([1, 1, MAX_ENCODER_FRAMES])),
        ]
    );
    assert_eq!(
        names_and_shapes(&encoder[0]["outputs"]),
        [("encoded", json!([1, ENCODER_HIDDEN, MAX_ENCODER_FRAMES]))]
    );

    let decoder = doors(&contract, "gigaam_decoder");
    assert_eq!(
        decoder[0]["outputs"][0]["shape"],
        json!([1, 1, DECODER_HIDDEN])
    );
    let joint = doors(&contract, "gigaam_joint");
    assert_eq!(joint[0]["outputs"][0]["name"], "token_id");

    let decoding = &contract["decoding"];
    assert_eq!(decoding["blank_id"], BLANK_TOKEN_ID);
    assert_eq!(
        decoding["samples_per_encoder_frame"],
        SAMPLES_PER_ENCODER_FRAME
    );
    assert_eq!(decoding["max_symbols_per_frame"], MAX_SYMBOLS_PER_STEP);

    // The encoder's lanes are the one port, and nothing ships beside them: the
    // Core ML row is that port PALETTIZED under the canonical name, the ONNX
    // master of the same graph is what this engine runs.
    let encoder = contract["members"]["gigaam_encoder"]["artifacts"]
        .as_object()
        .unwrap();
    let mut lanes = encoder.keys().collect::<Vec<_>>();
    lanes.sort();
    assert_eq!(lanes, ["coreai", "coreml", "onnx"]);
    for lane in ["coreml", "coreai", "onnx"] {
        assert_eq!(encoder[lane]["port"], "ane_encoder", "{lane}");
        assert_eq!(encoder[lane]["ships"], true, "{lane}");
    }
    assert!(
        encoder["coreml"]["compression"]
            .as_str()
            .unwrap()
            .starts_with("8-bit k-means palettized")
    );
    assert_eq!(encoder["onnx"]["file"], "gigaam_encoder.fp16.onnx");
}

#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreml", target_vendor = "apple")
))]
mod lowered {
    use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

    use super::*;

    pub fn golden_f32(name: &str) -> Vec<f32> {
        export::require_f32_le(&format!("gigaam/goldens/gigaam_{name}.f32"))
    }

    pub fn tensor(shape: &[usize], values: Vec<f32>) -> Tensor {
        Tensor::from_f32(shape.iter().map(|value| *value as i64).collect(), values).unwrap()
    }

    pub fn run(backend: &Backend, model: ModelId, inputs: TensorMap) -> TensorMap {
        futures_executor::block_on(backend.infer(model, &ModelVersion::default(), inputs)).unwrap()
    }

    pub fn worst(produced: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(produced.len(), expected.len());
        produced
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// The encoder door on `mel`, its mask 1 on the first `valid` frames.
    pub fn encode(backend: &Backend, mel: Vec<f32>, valid: usize) -> Vec<f32> {
        let mut mask = vec![1.0; valid];
        mask.resize(MAX_ENCODER_FRAMES, 0.0);
        let mut inputs = TensorMap::new();
        inputs.insert("mel".into(), tensor(&[1, N_MELS, MAX_MEL_FRAMES], mel));
        inputs.insert("mask".into(), tensor(&[1, 1, MAX_ENCODER_FRAMES], mask));
        run(backend, ModelId::GigaAmEncoder, inputs)["encoded"]
            .to_f32()
            .unwrap()
    }

    /// The pieces, id-indexed, as the runtime joins them: `▁` is the word
    /// boundary SentencePiece marks and `sp.decode` throws away.
    fn text(ids: &[i64]) -> String {
        let pieces = ferment_engine_processors::gigaam::vocab::pieces().unwrap();
        ids.iter()
            .map(|id| pieces[*id as usize].as_str())
            .collect::<String>()
            .replace('\u{2581}', " ")
            .trim()
            .to_string()
    }

    /// The golden chunk through the lane's four doors and the host's own
    /// greedy loop. Returns the encoder's frames and the sentence. The mel's
    /// bound keeps six times the ONNX preprocessor's 1.67e-4 (`parity.onnx`,
    /// at the PSNR ceiling); the Core ML one measured 9.5e-7.
    pub fn the_chain(backend: &Backend) -> (Vec<f32>, String) {
        let mut inputs = TensorMap::new();
        inputs.insert(
            "audio".into(),
            tensor(&[1, MAX_CHUNK_SAMPLES], golden_f32("audio_in")),
        );
        let mel = run(backend, ModelId::GigaAmPreprocessor, inputs)["mel"]
            .to_f32()
            .unwrap();
        let mel_drift = worst(&mel, &golden_f32("mel_out"));
        eprintln!("[gigaam] mel: max |Δ| vs the checkpoint {mel_drift:.3e}");
        assert!(mel_drift < 1e-3, "mel: {mel_drift}");

        // The golden chunk is fifteen seconds of speech: every frame is real.
        let encoded = encode(backend, golden_f32("mel_out"), MAX_ENCODER_FRAMES);

        let step_decoder = |token: i32, hidden: &[f32], cell: &[f32]| {
            let mut inputs = TensorMap::new();
            inputs.insert(
                "x".into(),
                Tensor::from_i32(vec![1, 1], vec![token]).unwrap(),
            );
            inputs.insert(
                "h_in".into(),
                tensor(&[1, 1, DECODER_HIDDEN], hidden.to_vec()),
            );
            inputs.insert(
                "c_in".into(),
                tensor(&[1, 1, DECODER_HIDDEN], cell.to_vec()),
            );
            let outputs = run(backend, ModelId::GigaAmDecoder, inputs);
            (
                outputs["dec"].to_f32().unwrap(),
                outputs["h_out"].to_f32().unwrap(),
                outputs["c_out"].to_f32().unwrap(),
            )
        };

        let (mut dec, mut hidden, mut cell) = step_decoder(
            BLANK_TOKEN_ID,
            &[0.0; DECODER_HIDDEN],
            &[0.0; DECODER_HIDDEN],
        );
        let mut ids: Vec<i64> = Vec::new();
        for frame in 0..MAX_ENCODER_FRAMES {
            let enc: Vec<f32> = (0..ENCODER_HIDDEN)
                .map(|channel| encoded[channel * MAX_ENCODER_FRAMES + frame])
                .collect();
            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let mut inputs = TensorMap::new();
                inputs.insert("enc".into(), tensor(&[1, ENCODER_HIDDEN, 1], enc.clone()));
                inputs.insert("dec".into(), tensor(&[1, DECODER_HIDDEN, 1], dec.clone()));
                let emitted = run(backend, ModelId::GigaAmJoint, inputs)["token_id"]
                    .to_i64()
                    .unwrap()[0];
                if emitted == i64::from(BLANK_TOKEN_ID) {
                    break;
                }
                ids.push(emitted);
                (dec, hidden, cell) = step_decoder(emitted as i32, &hidden, &cell);
            }
        }
        // Current GigaAMCoreAIParityTests compares all 134 checkpoint IDs.
        // Reconstructing the same string alone could hide changed tokenization.
        let raw =
            std::fs::read(export::require_path("gigaam/goldens/gigaam_tokens_out.i32")).unwrap();
        assert_eq!(raw.len() % 4, 0, "complete little-endian token IDs");
        let expected = raw
            .chunks_exact(4)
            .map(|bytes| i64::from(i32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), 134, "the current checkpoint token count");
        assert_eq!(
            ids, expected,
            "the graph must emit the exact checkpoint tokens"
        );
        (encoded, text(&ids))
    }

    pub fn the_sentence(contract: &serde_json::Value) -> &str {
        contract["goldens"]["text"].as_str().unwrap()
    }
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort {
    use std::sync::Arc;

    use ferment_engine_audio::AudioFrames;
    use ferment_engine_core::ProgressSink;
    use ferment_engine_infer::{
        Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };
    use ferment_engine_media::AudioIo;
    use ferment_engine_media::platform::libav::LibavAudioIo;
    use ferment_engine_processors::GigaAm;

    use super::lowered::{encode, golden_f32, the_chain, the_sentence, worst};
    use super::*;

    /// fp32 arithmetic over fp16-stored weights: the export measured the
    /// encoder's frames 1.41e-3 from the checkpoint's (`parity.onnx`, 79.3 dB
    /// against a 79.8 dB ceiling) and every one of 1057 tokens over eleven
    /// chunks the checkpoint's. The bound keeps about three times that.
    const ENCODED_BOUND: f32 = 5e-3;

    fn backend() -> Option<Backend> {
        let mut paths = ModelPaths::new();
        for (model, file) in [
            (ModelId::GigaAmPreprocessor, "gigaam_preprocessor.onnx"),
            (ModelId::GigaAmEncoder, "gigaam_encoder.fp16.onnx"),
            (ModelId::GigaAmDecoder, "gigaam_decoder.onnx"),
            (ModelId::GigaAmJoint, "gigaam_joint.onnx"),
        ] {
            paths = paths.with_path(model, ModelVersion::Latest, export::artifact(file)?);
        }
        Some(Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        )))
    }

    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_four_doors_say_the_sentence() {
        let Some(contract) = contract() else { return };
        let Some(backend) = backend() else { return };
        let (encoded, said) = the_chain(&backend);
        let drift = worst(&encoded, &golden_f32("encoded_out"));
        eprintln!("[gigaam] ort encoded: max |Δ| vs the checkpoint {drift:.3e}");
        eprintln!("[gigaam] ort: «{said}»");
        assert!(drift < ENCODED_BOUND, "encoded: {drift}");
        let golden = golden_f32("encoded_out");
        let noise = encoded
            .iter()
            .zip(&golden)
            .map(|(ours, expected)| f64::from((ours - expected) * (ours - expected)))
            .sum::<f64>()
            / golden.len() as f64;
        let peak = golden
            .iter()
            .map(|sample| f64::from(sample * sample))
            .fold(0.0_f64, f64::max);
        let psnr = 10.0 * (peak / noise.max(1e-20)).log10();
        // Keep the current Swift Core AI floor alongside the stronger ORT
        // max-absolute bound, rather than replacing either assertion.
        assert!(psnr > 55.0, "encoder PSNR {psnr} dB");
        assert_eq!(
            said,
            the_sentence(&contract),
            "the lane said another sentence"
        );
    }

    /// Current GigaAMCoreAIParityTests concurrency body over the actual ORT
    /// graph: six distinct masks, three callers with rotated input order.
    /// KILL: reuse mutable input/output storage across concurrent model calls.
    #[test]
    #[ignore = "needs admitted ONNX Runtime and the exact GigaAM encoder/golden mel"]
    fn concurrent_encodes_match_the_six_serial_answers() {
        assert!(std::env::var_os("ORT_DYLIB_PATH").is_some());
        let backend = backend().expect("required GigaAM graph artifacts");
        let mel = golden_f32("mel_out");
        let lengths = (0..6)
            .map(|index| MAX_MEL_FRAMES - index * 150)
            .collect::<Vec<_>>();
        let serial = lengths
            .iter()
            .map(|frames| {
                encode(
                    &backend,
                    mel.clone(),
                    ferment_engine_processors::gigaam::config::encoder_frames(*frames),
                )
            })
            .collect::<Vec<_>>();
        for answer in &serial {
            assert_eq!(answer.len(), ENCODER_HIDDEN * MAX_ENCODER_FRAMES);
        }
        for first in 0..serial.len() {
            for second in first + 1..serial.len() {
                assert!(
                    serial[first] != serial[second],
                    "six valid lengths must differ"
                );
            }
        }
        let answers = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for task in 0..3 {
                let backend = backend.clone();
                let mel = &mel;
                let lengths = &lengths;
                handles.push(scope.spawn(move || {
                    (0..lengths.len())
                        .map(|offset| {
                            let index = (task * 2 + offset) % lengths.len();
                            let frames = lengths[index];
                            let answer = encode(
                                &backend,
                                mel.clone(),
                                ferment_engine_processors::gigaam::config::encoder_frames(frames),
                            );
                            (index, answer)
                        })
                        .collect::<Vec<_>>()
                }));
            }
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("concurrent encoder caller"))
                .collect::<Vec<_>>()
        });
        assert_eq!(answers.len(), 3 * lengths.len());
        for (index, answer) in answers {
            assert_eq!(answer.len(), serial[index].len());
            assert!(
                answer == serial[index],
                "{} mel frames: concurrent output differs, max delta {}",
                lengths[index],
                worst(&answer, &serial[index]),
            );
        }
    }

    /// The same mel, masked to the 125 encoder frames of 499 real mel frames
    /// (80 000 samples of a 240 000-sample window), must move the frames the
    /// two answers share: a door that ignored its mask would still say the
    /// full chunk's sentence and read every short chunk's zero tail as
    /// speech. The retired `length` door moved them by 2.08 on the same test.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_mask_reaches_the_graph() {
        if contract().is_none() {
            return;
        }
        let Some(backend) = backend() else { return };
        let mel = golden_f32("mel_out");
        let full = encode(&backend, mel.clone(), MAX_ENCODER_FRAMES);
        let short = encode(&backend, mel, 125);
        let spread = (0..ENCODER_HIDDEN)
            .flat_map(|channel| (0..125).map(move |frame| channel * MAX_ENCODER_FRAMES + frame))
            .map(|index| (full[index] - short[index]).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[gigaam] ort mask: the shared 125 frames moved by {spread:.3}");
        assert!(spread > 0.5, "the mask is not reaching the graph: {spread}");
    }

    /// The processor end to end over a clip shorter than one chunk — its
    /// resampling, its chunk plan, the mask it counts, the decode, the words.
    #[test]
    #[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
    fn the_processor_transcribes_russian_speech_end_to_end() {
        let Some(backend) = backend() else { return };
        let fixture = ferment_engine_test_fixtures::require_path("Speech/speech-ru.m4a");
        let io = LibavAudioIo;
        let info = futures_executor::block_on(io.probe(&fixture)).unwrap();
        let stream = info
            .streams
            .iter()
            .find(|stream| stream.kind == ferment_engine_media::MediaKind::Audio)
            .unwrap();
        let mut decoder =
            futures_executor::block_on(io.open_audio_decoder(&fixture, stream.index)).unwrap();
        let mut samples = Vec::new();
        let mut sample_rate = 0;
        while let Some(frame) = futures_executor::block_on(decoder.decode_next()).unwrap() {
            sample_rate = frame.sample_rate;
            assert_eq!(frame.channels, 1, "fixture is mono");
            samples.extend_from_slice(&frame.samples);
        }
        assert!(
            samples.len() < MAX_CHUNK_SAMPLES * sample_rate as usize / 16_000,
            "the clip must be shorter than one chunk for the mask to matter"
        );

        let words = futures_executor::block_on(GigaAm::new(backend).transcribe(
            AudioFrames::new(samples, sample_rate),
            None,
            ProgressSink::noop(),
        ))
        .unwrap();
        let transcript = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        eprintln!("[gigaam] ort transcript: {transcript:?}");
        assert!(
            transcript.contains("лиса") && transcript.contains("собаку"),
            "unexpected transcript: {transcript:?}"
        );
        assert!(
            words
                .windows(2)
                .all(|pair| pair[1].start_time >= pair[0].start_time),
            "word starts must be monotonic"
        );
    }
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml {
    use std::sync::Arc;

    use ferment_engine_infer::{
        Backend, ComputeUnits, CoreMlBackend, ModelId, ModelPaths, ModelVersion,
    };
    use ferment_engine_models::{BackendKind, CoreMlComputeUnits, Placement};

    use super::lowered::{golden_f32, the_chain, the_sentence, worst};
    use super::*;

    /// The palettized port. The export measured its frames 0.0904 from the
    /// checkpoint's on the CPU and 0.0562 on the Neural Engine
    /// (`parity.coreml`), and on both units the sentence the checkpoint's; the
    /// bound keeps about twice the CPU's drift.
    const ENCODED_BOUND: f32 = 0.18;

    /// The four bundles on the units the fleet declares for each.
    #[test]
    #[ignore = "needs ferment-models-export/artifacts/gigaam_*.mlmodelc"]
    fn the_four_core_ml_doors_say_the_sentence() {
        let Some(contract) = contract() else { return };
        let fleet = ferment_engine_models::default_fleet_manifest();
        let mut paths = ModelPaths::new();
        let mut units = Vec::new();
        for model in [
            ModelId::GigaAmPreprocessor,
            ModelId::GigaAmEncoder,
            ModelId::GigaAmDecoder,
            ModelId::GigaAmJoint,
        ] {
            let Some(bundle) = export::artifact(&format!("{}.mlmodelc", model.name())) else {
                return;
            };
            paths.insert(model, ModelVersion::Latest, bundle);
            let Some(Placement::CoreMl { units: declared }) = fleet
                .model(model)
                .unwrap()
                .placement_for(BackendKind::CoreMl)
            else {
                panic!("{model} has no Core ML placement");
            };
            units.push((
                model,
                match declared {
                    CoreMlComputeUnits::All => ComputeUnits::All,
                    CoreMlComputeUnits::CpuOnly => ComputeUnits::CpuOnly,
                    CoreMlComputeUnits::CpuAndGpu => ComputeUnits::CpuAndGpu,
                    CoreMlComputeUnits::CpuAndNeuralEngine => ComputeUnits::CpuAndNeuralEngine,
                },
            ));
        }
        let backend: Backend = Arc::new(CoreMlBackend::new(Arc::new(paths)).with_placements(units));
        let (encoded, said) = the_chain(&backend);
        let drift = worst(&encoded, &golden_f32("encoded_out"));
        eprintln!("[gigaam] coreml encoded: max |Δ| vs the checkpoint {drift:.3e}");
        eprintln!("[gigaam] coreml: «{said}»");
        assert!(drift < ENCODED_BOUND, "encoded: {drift}");
        assert_eq!(
            said,
            the_sentence(&contract),
            "the lane said another sentence"
        );
    }
}
