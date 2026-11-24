//! Current Swift integration fixtures through the real native pipeline.
//! This is CPU/host evidence when run on macOS. It does not qualify Android
//! MediaCodec or XNNPACK; those remain separately measured device boundaries.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use ferment_engine_core::{LoudnessTarget, ProgressSink};
use ferment_engine_infer::{
    InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
    TensorMap,
};
use ferment_engine_processors::{AudioSource, DeepFilterNet, LoudnessNormalizer};
use futures_executor::block_on;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn fixture(relative: &str) -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("FERMENT_CURRENT_IOS_ORACLE")
            .expect("set FERMENT_CURRENT_IOS_ORACLE to the captured current iOS checkout"),
    );
    let path = root.join(relative);
    assert!(
        path.is_file(),
        "required current Swift fixture: {}",
        path.display()
    );
    path
}

#[test]
#[ignore = "requires the immutable current iOS fixture tree and native MP3 decoder"]
fn real_music_normalization_preserves_the_source_report_contract() {
    let audio = block_on(
        AudioSource::platform_default()
            .decode(&fixture("Tests/Fixtures/Audio/industrial-techno.mp3")),
    )
    .unwrap();
    assert!(!audio.samples.is_empty());
    let result = block_on(LoudnessNormalizer.normalize_frames(
        audio,
        LoudnessTarget::STANDARD,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.report.qa_pass);
    assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
}

#[test]
#[ignore = "requires the immutable current iOS fixture tree"]
fn real_speech_normalization_engages_the_leveler_and_respects_the_ceiling() {
    let audio = block_on(
        AudioSource::platform_default()
            .decode(&fixture("Tests/Fixtures/Audio/synthetic-speech-1.wav")),
    )
    .unwrap();
    assert!(!audio.samples.is_empty());
    let result = block_on(LoudnessNormalizer.normalize_frames(
        audio,
        LoudnessTarget::SPEECH,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(
        result.report.within_clip_leveler_engaged
            || result.report.measured_lufs == f64::NEG_INFINITY
    );
    assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
}

#[test]
#[ignore = "requires the immutable current iOS fixture tree and native MP3 decoder"]
fn real_transient_normalization_respects_the_ceiling_and_audible_floor() {
    let audio = block_on(
        AudioSource::platform_default().decode(&fixture("Tests/Fixtures/Audio/q3-quad-damage.mp3")),
    )
    .unwrap();
    assert!(!audio.samples.is_empty());
    let result = block_on(LoudnessNormalizer.normalize_frames(
        audio,
        LoudnessTarget::STANDARD,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.report.post_limit_true_peak <= result.report.peak_ceiling_dbtp + 0.1);
    assert!(result.report.post_limit_lufs > -40.0);
}

#[test]
#[ignore = "requires the immutable current iOS fixture tree and native MP3 decoder"]
fn phonk_loads_through_the_real_audio_source() {
    let audio = block_on(
        AudioSource::platform_default().decode(&fixture("Tests/Fixtures/Audio/phonk-1.mp3")),
    )
    .unwrap();
    assert!(!audio.samples.is_empty());
    assert!(audio.sample_rate > 0);
}

#[test]
#[ignore = "requires explicit pinned ORT, published dfnet, current hairdryer MP3 and native decoder"]
fn real_hairdryer_denoising_preserves_duration_level_and_progress() {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "name the pinned ORT runtime"
    );
    let root = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
            .expect("name the frozen published model directory"),
    );
    let graph = root.join("dfnet.onnx");
    assert!(graph.is_file(), "required graph: {}", graph.display());
    let backend = Arc::new(OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, graph)),
        OrtExecutionProvider::Cpu,
    ));
    let audio = block_on(AudioSource::platform_default().decode(&fixture(
        "Packages/FermentEngine/Tests/Fixtures/hairdryer-product-placement.mp3",
    )))
    .unwrap();
    let stages = Arc::new(Mutex::new(Vec::new()));
    let observed = stages.clone();
    let progress = ProgressSink::from_callback(move |event| {
        observed.lock().unwrap().push(event.stage);
    });
    let result = block_on(DeepFilterNet::new(backend.clone()).denoise(audio, progress)).unwrap();
    assert!(!result.samples.is_empty());
    assert!(result.sample_rate > 0);
    let seconds = result.samples.len() as f64 / f64::from(result.sample_rate);
    assert!(
        seconds > 15.0 && seconds < 25.0,
        "denoised duration: {seconds}"
    );
    assert!(result.samples.iter().all(|value| value.is_finite()));
    let maximum = result
        .samples
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum < 2.0 && maximum > 0.01, "max amplitude: {maximum}");
    let rms = (result
        .samples
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / result.samples.len() as f32)
        .sqrt();
    assert!(rms > 0.001, "denoised RMS: {rms}");
    assert!(
        !stages.lock().unwrap().is_empty(),
        "the real processor must report progress"
    );
    assert_eq!(
        backend.selected_provider_for(ModelId::Dfnet),
        Some(OrtExecutionProvider::Cpu)
    );
    assert_eq!(backend.cache_snapshot().resident, 1);
}

fn assert_warm_repeated_outputs(id: ModelId, file: &str, inputs: TensorMap) {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "name the pinned ORT runtime"
    );
    let root = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
            .expect("name the frozen published model directory"),
    );
    let graph = root.join(file);
    assert!(graph.is_file(), "required graph: {}", graph.display());
    let backend = OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(id, ModelVersion::Latest, graph)),
        OrtExecutionProvider::Cpu,
    );
    block_on(backend.warmup(&[id])).unwrap();
    assert_eq!(backend.cache_snapshot().resident, 1);
    let reference = block_on(backend.infer(id, &ModelVersion::Latest, inputs.clone())).unwrap();
    assert!(!reference.is_empty());
    for _ in 0..2 {
        let actual = block_on(backend.infer(id, &ModelVersion::Latest, inputs.clone())).unwrap();
        assert_eq!(actual.len(), reference.len());
        for (name, expected) in &reference {
            assert_eq!(actual[name].shape, expected.shape, "{name}");
            let expected = expected.to_f32().unwrap();
            assert!(expected.iter().all(|value| value.is_finite()), "{name}");
            assert_eq!(
                actual[name].to_f32().unwrap(),
                expected,
                "CPU repeated-call drift: {name}"
            );
        }
        assert_eq!(backend.cache_snapshot().resident, 1);
    }
    assert_eq!(
        backend.selected_provider_for(id),
        Some(OrtExecutionProvider::Cpu)
    );
    assert_eq!(backend.cache_snapshot().loading, 0);
    assert_eq!(backend.cache_snapshot().waiters, 0);
}

#[test]
#[ignore = "requires explicit pinned ORT, published BeatThis graph and export goldens"]
fn warmed_beat_this_is_stable_over_three_real_calls() {
    use ferment_engine_processors::beatthis::config::{CHUNK_SIZE, N_MELS};
    use ferment_engine_test_fixtures::export;
    assert!(
        std::env::var_os("FERMENT_MODELS_EXPORT_DIR").is_some(),
        "name the frozen export tree"
    );
    let mel = export::require_f32_le("beat_this/goldens/beat_this_mel_out.f32");
    let frames = mel.len() / N_MELS;
    assert!(frames >= CHUNK_SIZE);
    let mut door = Vec::with_capacity(N_MELS * CHUNK_SIZE);
    for band in 0..N_MELS {
        door.extend_from_slice(&mel[band * frames..band * frames + CHUNK_SIZE]);
    }
    let mut inputs = TensorMap::new();
    inputs.insert(
        "mel".into(),
        Tensor::from_f32(vec![1, 1, N_MELS as i64, CHUNK_SIZE as i64], door).unwrap(),
    );
    assert_warm_repeated_outputs(ModelId::BeatThis1, "beat_this_1.onnx", inputs);
}

#[test]
#[ignore = "requires explicit pinned ORT, published dfnet graph and export goldens"]
fn warmed_dfnet_is_stable_over_three_real_calls() {
    use ferment_engine_processors::deepfilternet::config::{
        CHUNK_FRAMES, NB_DF, NB_ERB, STREAM_FRAMES, STREAM_STATE,
    };
    use ferment_engine_test_fixtures::export;
    assert!(
        std::env::var_os("FERMENT_MODELS_EXPORT_DIR").is_some(),
        "name the frozen export tree"
    );
    let erb = export::require_f32_le("dfnet/goldens/dfnet_feat_erb_in.f32");
    let spec = export::require_f32_le("dfnet/goldens/dfnet_feat_spec_in.f32");
    assert_eq!(erb.len(), CHUNK_FRAMES * NB_ERB);
    assert_eq!(spec.len(), 2 * CHUNK_FRAMES * NB_DF);
    let mut first_spec = spec[..STREAM_FRAMES * NB_DF].to_vec();
    first_spec.extend_from_slice(
        &spec[CHUNK_FRAMES * NB_DF..CHUNK_FRAMES * NB_DF + STREAM_FRAMES * NB_DF],
    );
    let mut inputs = TensorMap::new();
    inputs.insert(
        "feat_erb".into(),
        Tensor::from_f32(
            vec![1, 1, STREAM_FRAMES as i64, NB_ERB as i64],
            erb[..STREAM_FRAMES * NB_ERB].to_vec(),
        )
        .unwrap(),
    );
    inputs.insert(
        "feat_spec".into(),
        Tensor::from_f32(vec![1, 2, STREAM_FRAMES as i64, NB_DF as i64], first_spec).unwrap(),
    );
    for state in STREAM_STATE {
        inputs.insert(
            state.input.into(),
            Tensor::from_f32(
                state.shape.to_vec(),
                vec![0.0; state.shape.iter().map(|n| *n as usize).product()],
            )
            .unwrap(),
        );
    }
    assert_warm_repeated_outputs(ModelId::Dfnet, "dfnet.onnx", inputs);
}

fn parakeet_backend() -> OrtBackend {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "name the pinned ORT runtime"
    );
    let root = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
            .expect("name the frozen published model directory"),
    );
    let graph = root.join("parakeet_encoder.onnx");
    assert!(graph.is_file(), "required graph: {}", graph.display());
    OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(
            ModelId::ParakeetEncoder,
            ModelVersion::Latest,
            graph,
        )),
        OrtExecutionProvider::Cpu,
    )
}

#[test]
#[ignore = "requires explicit pinned ORT and the current Parakeet encoder plus host-decode weights"]
fn the_current_parakeet_bundle_loads_its_graph_decode_and_vocabulary() {
    use ferment_engine_processors::parakeet::host_decode::{FILE, HostDecode};
    use ferment_engine_processors::parakeet::vocab::Vocab;
    let backend = parakeet_backend();
    block_on(backend.warmup(&[ModelId::ParakeetEncoder])).unwrap();
    let root = PathBuf::from(std::env::var_os("FERMENT_ENGINE_MODELS_DIR").unwrap());
    let _weights = HostDecode::load(&root.join(FILE)).unwrap();
    let vocab = Vocab::load().unwrap();
    assert!(vocab.len() > 8_000);
    assert!(vocab.values().any(|piece| piece.starts_with('▁')));
    assert_eq!(backend.cache_snapshot().resident, 1);
    assert_eq!(
        backend.selected_provider_for(ModelId::ParakeetEncoder),
        Some(OrtExecutionProvider::Cpu)
    );
}

#[test]
#[ignore = "requires explicit pinned ORT and the current Parakeet encoder"]
fn the_current_parakeet_encoder_answers_its_declared_tensor_door() {
    use ferment_engine_processors::parakeet::config::{
        ENCODER_FRAMES, HIDDEN, MEL_BANDS, MEL_FRAMES,
    };
    let backend = parakeet_backend();
    let mut inputs = TensorMap::new();
    inputs.insert(
        "mel".into(),
        Tensor::from_f32(
            vec![1, 1, MEL_BANDS as i64, MEL_FRAMES as i64],
            vec![0.0; MEL_BANDS * MEL_FRAMES],
        )
        .unwrap(),
    );
    inputs.insert(
        "mask".into(),
        Tensor::from_f32(vec![1, 1, 1, MEL_FRAMES as i64], vec![1.0; MEL_FRAMES]).unwrap(),
    );
    let result =
        block_on(backend.infer(ModelId::ParakeetEncoder, &ModelVersion::Latest, inputs)).unwrap();
    assert_eq!(
        result["encoded"].shape,
        vec![1, HIDDEN as i64, 1, ENCODER_FRAMES as i64]
    );
    assert!(
        result["encoded"]
            .to_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
