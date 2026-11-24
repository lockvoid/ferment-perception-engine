use std::f32::consts::TAU;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{AudioEvent, AudioEventSegment, EngineError, ProgressSink};
use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};
use ferment_engine_test_fixtures::export;

use super::DeepFilterNet;
use super::config::{
    CHUNK_FRAMES, DF_LOOKAHEAD, DF_ORDER, FFT_SIZE, HOP_SIZE, NB_DF, NB_ERB, STREAM_FRAMES,
    STREAM_STATE,
};
use super::enhance::{apply_deep_filtering, apply_erb_mask, merge};
use super::features::chunk_features;
use super::processor::{Analysis, Synthesis};

/// Every call is the streaming door — its eight inputs, each at the shape
/// `models/dfnet/model.json` declares, and nothing else.
fn assert_the_streaming_door(inputs: &TensorMap) {
    let mut expected = vec![
        (
            "feat_erb".to_owned(),
            vec![1, 1, STREAM_FRAMES as i64, NB_ERB as i64],
        ),
        (
            "feat_spec".to_owned(),
            vec![1, 2, STREAM_FRAMES as i64, NB_DF as i64],
        ),
    ];
    expected.extend(
        STREAM_STATE
            .iter()
            .map(|state| (state.input.to_owned(), state.shape.to_vec())),
    );
    expected.sort();
    let actual = inputs
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.shape.clone()))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

/// The identity enhancement — a unit mask, the deep filter's identity tap at
/// `DF_LOOKAHEAD`, lsnr 0 — with the state handed back as it came in.
fn identity_answer(inputs: &TensorMap) -> TensorMap {
    let mut coefficients = vec![0.0; DF_ORDER * STREAM_FRAMES * NB_DF * 2];
    let identity_plane = DF_LOOKAHEAD * STREAM_FRAMES * NB_DF * 2;
    for frame in 0..STREAM_FRAMES {
        for frequency in 0..NB_DF {
            coefficients[identity_plane + frame * NB_DF * 2 + frequency * 2] = 1.0;
        }
    }
    let mut output = TensorMap::new();
    output.insert(
        "erb_mask".into(),
        Tensor::from_f32(
            vec![1, 1, STREAM_FRAMES as i64, NB_ERB as i64],
            vec![1.0; STREAM_FRAMES * NB_ERB],
        )
        .unwrap(),
    );
    output.insert(
        "df_coefs".into(),
        Tensor::from_f32(
            vec![1, DF_ORDER as i64, STREAM_FRAMES as i64, NB_DF as i64, 2],
            coefficients,
        )
        .unwrap(),
    );
    output.insert(
        "lsnr".into(),
        Tensor::from_f32(vec![1, STREAM_FRAMES as i64, 1], vec![0.0; STREAM_FRAMES]).unwrap(),
    );
    for state in &STREAM_STATE {
        output.insert(state.output.into(), inputs[state.input].clone());
    }
    output
}

#[derive(Debug, Default)]
struct IdentityStreamingBackend;

#[async_trait]
impl InferenceBackend for IdentityStreamingBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> ferment_engine_core::Result<TensorMap> {
        assert_eq!(model, ModelId::Dfnet);
        assert_the_streaming_door(&inputs);
        Ok(identity_answer(&inputs))
    }
}

/// Answers the identity enhancement and hands every state tensor back one
/// larger than it came in, recording what it was handed — so each record is
/// the call's index within its chunk.
#[derive(Debug, Default)]
struct CountingStateBackend {
    handed: Mutex<Vec<f32>>,
}

#[async_trait]
impl InferenceBackend for CountingStateBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> ferment_engine_core::Result<TensorMap> {
        assert_eq!(model, ModelId::Dfnet);
        assert_the_streaming_door(&inputs);
        let handed = inputs[STREAM_STATE[0].input].to_f32()?[0];
        let mut output = identity_answer(&inputs);
        for state in &STREAM_STATE {
            let values = inputs[state.input].to_f32()?;
            assert!(
                values.iter().all(|value| *value == handed),
                "{} was not handed the same call's state as the rest",
                state.input
            );
            output.insert(
                state.output.into(),
                Tensor::from_f32(
                    state.shape.to_vec(),
                    values.iter().map(|value| value + 1.0).collect(),
                )?,
            );
        }
        self.handed.lock().unwrap().push(handed);
        Ok(output)
    }
}

/// Answers each call with its own features — `erb_mask` is `feat_erb`,
/// `df_coefs[k, t, f, c]` is `feat_spec[c, t, f] + k`, `lsnr[t]` is
/// `feat_erb[t, 0]` — and records the features it was fed.
#[derive(Debug, Default)]
struct EchoBackend {
    fed: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
}

#[async_trait]
impl InferenceBackend for EchoBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> ferment_engine_core::Result<TensorMap> {
        assert_eq!(model, ModelId::Dfnet);
        assert_the_streaming_door(&inputs);
        let erb = inputs["feat_erb"].to_f32()?;
        let spec = inputs["feat_spec"].to_f32()?;
        let mut coefficients = Vec::with_capacity(DF_ORDER * STREAM_FRAMES * NB_DF * 2);
        for plane in 0..DF_ORDER {
            for frame in 0..STREAM_FRAMES {
                for frequency in 0..NB_DF {
                    for channel in 0..2 {
                        coefficients.push(
                            spec[channel * STREAM_FRAMES * NB_DF + frame * NB_DF + frequency]
                                + plane as f32,
                        );
                    }
                }
            }
        }
        let mut output = identity_answer(&inputs);
        output.insert("erb_mask".into(), inputs["feat_erb"].clone());
        output.insert(
            "df_coefs".into(),
            Tensor::from_f32(
                vec![1, DF_ORDER as i64, STREAM_FRAMES as i64, NB_DF as i64, 2],
                coefficients,
            )?,
        );
        output.insert(
            "lsnr".into(),
            Tensor::from_f32(
                vec![1, STREAM_FRAMES as i64, 1],
                (0..STREAM_FRAMES)
                    .map(|frame| erb[frame * NB_ERB])
                    .collect(),
            )?,
        );
        self.fed.lock().unwrap().push((erb, spec));
        Ok(output)
    }
}

/// Hands state `i` of `STREAM_STATE` back `i + 1` larger than it came in and
/// records the first value of every state it was handed, in door order.
#[derive(Debug, Default)]
struct StepPerStateBackend {
    handed: Mutex<Vec<Vec<f32>>>,
}

#[async_trait]
impl InferenceBackend for StepPerStateBackend {
    async fn infer(
        &self,
        model: ModelId,
        _version: &ModelVersion,
        inputs: TensorMap,
    ) -> ferment_engine_core::Result<TensorMap> {
        assert_eq!(model, ModelId::Dfnet);
        assert_the_streaming_door(&inputs);
        let mut output = identity_answer(&inputs);
        let mut handed = Vec::with_capacity(STREAM_STATE.len());
        for (index, state) in STREAM_STATE.iter().enumerate() {
            let values = inputs[state.input].to_f32()?;
            handed.push(values[0]);
            output.insert(
                state.output.into(),
                Tensor::from_f32(
                    state.shape.to_vec(),
                    values
                        .iter()
                        .map(|value| value + (index + 1) as f32)
                        .collect(),
                )?,
            );
        }
        self.handed.lock().unwrap().push(handed);
        Ok(output)
    }
}

fn detector() -> DeepFilterNet {
    DeepFilterNet::new(Arc::new(IdentityStreamingBackend))
}

fn tone(sample_rate: u32, seconds: f64, frequency: f32, amplitude: f32) -> AudioFrames {
    let samples = (0..(seconds * f64::from(sample_rate)) as usize)
        .map(|index| (TAU * frequency * index as f32 / sample_rate as f32).sin() * amplitude)
        .collect();
    AudioFrames::new(samples, sample_rate)
}

fn speech_like(sample_rate: u32, seconds: f64) -> AudioFrames {
    let samples = (0..(seconds * f64::from(sample_rate)) as usize)
        .map(|index| {
            let time = index as f32 / sample_rate as f32;
            let envelope = 0.35 + 0.65 * (TAU * 2.3 * time).sin().abs();
            envelope * (0.04 * (TAU * 180.0 * time).sin() + 0.02 * (TAU * 720.0 * time).sin())
        })
        .collect();
    AudioFrames::new(samples, sample_rate)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn speech_segment(start: f64, duration: f64) -> AudioEventSegment {
    AudioEventSegment {
        event: AudioEvent::Speech,
        start_time: start,
        duration,
        confidence: 1.0,
    }
}

// DeepFilterNetTests.swift — deterministic orchestration coverage.

#[test]
fn output_length_matches_input_duration_at_48k() {
    let result = futures_executor::block_on(
        detector().denoise(tone(48_000, 5.0, 440.0, 0.05), ProgressSink::noop()),
    )
    .unwrap();
    assert_eq!(result.sample_rate, 48_000);
    let expected = 5 * 48_000;
    assert!(result.samples.len().abs_diff(expected) < 2 * HOP_SIZE);
}

#[test]
fn resamples_non_48k_inputs() {
    for input_rate in [16_000, 22_050, 44_100] {
        let result = futures_executor::block_on(
            detector().denoise(tone(input_rate, 1.5, 440.0, 0.05), ProgressSink::noop()),
        )
        .unwrap();
        assert_eq!(result.sample_rate, 48_000);
        let expected = 72_000;
        let tolerance = (expected as f64 * 0.05) as usize;
        assert!(
            result.samples.len().abs_diff(expected) < tolerance,
            "resample drift at {input_rate} Hz: got {}",
            result.samples.len()
        );
    }
}

#[test]
fn output_is_finite_and_bounded() {
    let result = futures_executor::block_on(
        detector().denoise(tone(48_000, 2.0, 440.0, 0.05), ProgressSink::noop()),
    )
    .unwrap();
    assert!(result.samples.iter().all(|sample| sample.is_finite()));
    let maximum = result
        .samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);
    assert!(maximum < 2.0, "output amplitude {maximum} is unreasonable");
}

#[test]
fn audio_too_short_throws() {
    let error = futures_executor::block_on(
        detector().denoise(tone(48_000, 0.005, 440.0, 0.05), ProgressSink::noop()),
    )
    .unwrap_err();
    assert!(matches!(error, EngineError::AudioTooShort { .. }));
}

#[test]
fn chunked_path_produces_valid_output_on_speech() {
    let audio = speech_like(48_000, 8.0);
    let chunked = futures_executor::block_on(
        detector()
            .with_max_frames_per_chunk(300)
            .denoise(audio.clone(), ProgressSink::noop()),
    )
    .unwrap();
    let single =
        futures_executor::block_on(detector().denoise(audio, ProgressSink::noop())).unwrap();
    assert!(chunked.samples.len().abs_diff(single.samples.len()) < 2 * HOP_SIZE);
    let chunked_rms = rms(&chunked.samples);
    let single_rms = rms(&single.samples);
    assert!(chunked_rms > single_rms * 0.5);
    assert!(chunked_rms < single_rms * 1.5);
}

/// 7.5 s is 752 frames: at 300 frames a chunk that is chunks of 300, 300 and
/// 152 — three calls, three, and two (100 + 52). Every call hands its state to
/// the next; every chunk starts from zeros (`DeepFilterNetModel.stream`).
#[test]
fn the_state_is_handed_from_call_to_call_and_reset_at_every_chunk() {
    let backend = Arc::new(CountingStateBackend::default());
    futures_executor::block_on(
        DeepFilterNet::new(backend.clone())
            .with_max_frames_per_chunk(300)
            .denoise(speech_like(48_000, 7.5), ProgressSink::noop()),
    )
    .unwrap();
    assert_eq!(
        *backend.handed.lock().unwrap(),
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0]
    );
}

/// 250 frames are three calls — 100, 100 and 50. Every answer lands at its
/// frames in `[T, 32]`, `[5, T, 96, 2]` and `[T]`; the last call is fed zeros
/// past its 50 real rows and nothing it answers for them survives.
#[test]
fn a_chunk_lands_in_the_layouts_the_dsp_indexes() {
    const FRAMES: usize = 250;
    let feat_erb = (0..FRAMES * NB_ERB)
        .map(|index| 1.0 + index as f32)
        .collect::<Vec<_>>();
    let feat_spec = (0..2 * FRAMES * NB_DF)
        .map(|index| -1.0 - index as f32)
        .collect::<Vec<_>>();
    let backend = Arc::new(EchoBackend::default());
    let answer = futures_executor::block_on(
        DeepFilterNet::new(backend.clone()).run_chunk(&feat_erb, &feat_spec, FRAMES),
    )
    .unwrap();

    assert_eq!(answer.erb_mask, feat_erb);
    assert_eq!(
        answer.lsnr,
        (0..FRAMES)
            .map(|frame| feat_erb[frame * NB_ERB])
            .collect::<Vec<_>>()
    );
    assert_eq!(answer.df_coefs.len(), DF_ORDER * FRAMES * NB_DF * 2);
    for plane in 0..DF_ORDER {
        for frame in 0..FRAMES {
            for frequency in 0..NB_DF {
                for channel in 0..2 {
                    assert_eq!(
                        answer.df_coefs
                            [((plane * FRAMES + frame) * NB_DF + frequency) * 2 + channel],
                        feat_spec[channel * FRAMES * NB_DF + frame * NB_DF + frequency]
                            + plane as f32,
                        "plane {plane}, frame {frame}, bin {frequency}, channel {channel}"
                    );
                }
            }
        }
    }

    let fed = backend.fed.lock().unwrap();
    assert_eq!(fed.len(), 3);
    let (erb, spec) = &fed[2];
    assert_eq!(erb[..50 * NB_ERB], feat_erb[200 * NB_ERB..]);
    assert!(erb[50 * NB_ERB..].iter().all(|value| *value == 0.0));
    for channel in 0..2 {
        let call = channel * STREAM_FRAMES * NB_DF;
        let chunk = channel * FRAMES * NB_DF;
        assert_eq!(
            spec[call..call + 50 * NB_DF],
            feat_spec[chunk + 200 * NB_DF..chunk + FRAMES * NB_DF]
        );
        assert!(
            spec[call + 50 * NB_DF..call + STREAM_FRAMES * NB_DF]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}

/// The calls that fed one chunk of `frames` frames carried exactly `erb` and
/// `spec`, that chunk's `[T, 32]` and `[2, T, 96]` features, in door-sized
/// slices.
fn assert_fed(
    calls: &[(Vec<f32>, Vec<f32>)],
    erb: &[f32],
    spec: &[f32],
    frames: usize,
    label: &str,
) {
    assert_eq!(
        calls.len(),
        frames.div_ceil(STREAM_FRAMES),
        "{label}: calls"
    );
    for ((fed_erb, fed_spec), call) in calls.iter().zip((0..frames).step_by(STREAM_FRAMES)) {
        let valid = (frames - call).min(STREAM_FRAMES);
        assert_eq!(
            fed_erb[..valid * NB_ERB],
            erb[call * NB_ERB..(call + valid) * NB_ERB],
            "feat_erb, {label}, call at {call}"
        );
        for channel in 0..2 {
            let fed_rows = channel * STREAM_FRAMES * NB_DF;
            let rows = channel * frames * NB_DF + call * NB_DF;
            assert_eq!(
                fed_spec[fed_rows..fed_rows + valid * NB_DF],
                spec[rows..rows + valid * NB_DF],
                "feat_spec channel {channel}, {label}, call at {call}"
            );
        }
    }
}

/// The door is fed each chunk's own features: computed from that chunk's
/// frames alone, the running normalisations fresh, and shifted with the
/// chunk's last two frames zero — never the next chunk's first two
/// (`DeepFilterNet.processSpec`). 7.5 s is chunks of 300, 300 and 152.
#[test]
fn every_chunk_is_fed_its_own_features() {
    const CHUNK: usize = 300;
    let audio = speech_like(48_000, 7.5);
    let backend = Arc::new(EchoBackend::default());
    futures_executor::block_on(
        DeepFilterNet::new(backend.clone())
            .with_max_frames_per_chunk(CHUNK)
            .denoise(audio.clone(), ProgressSink::noop()),
    )
    .unwrap();

    let frame_count = (audio.samples.len() + FFT_SIZE) / HOP_SIZE;
    let mut analysis = Analysis::new().unwrap();
    let fed = backend.fed.lock().unwrap();
    let mut calls = &fed[..];
    for start in (0..frame_count).step_by(CHUNK) {
        let frames = (frame_count - start).min(CHUNK);
        let (erb, spec) = chunk_features(&analysis.next(&audio.samples, frames).unwrap(), frames);
        assert!(
            erb[(frames - DF_LOOKAHEAD) * NB_ERB..]
                .iter()
                .all(|value| *value == 0.0),
            "the chunk at frame {start} ends on its own zeros"
        );
        let (chunk, rest) = calls.split_at(frames.div_ceil(STREAM_FRAMES));
        assert_fed(
            chunk,
            &erb,
            &spec,
            frames,
            &format!("chunk at frame {start}"),
        );
        calls = rest;
    }
    assert!(calls.is_empty());
}

/// iOS's `canSkipChunk`: a chunk whose output range overlaps no keep segment
/// never reaches the door. 7.5 s at 300 frames a chunk is chunks of 300, 300
/// and 152 — three calls, three and two — whose output ranges are
/// [0, 143 520), [143 520, 287 520) and [287 520, 360 000); a segment inside
/// the first leaves only its three.
#[test]
fn a_chunk_outside_every_keep_segment_costs_no_door_call() {
    let backend = Arc::new(CountingStateBackend::default());
    futures_executor::block_on(
        DeepFilterNet::new(backend.clone())
            .with_max_frames_per_chunk(300)
            .denoise_with_keep_segments(
                speech_like(48_000, 7.5),
                &[speech_segment(0.5, 1.0)],
                ProgressSink::noop(),
            ),
    )
    .unwrap();
    assert_eq!(*backend.handed.lock().unwrap(), [0.0, 1.0, 2.0]);
}

/// A skipped chunk runs silence through the analysis — iOS's
/// `stftFwd.process(zeroAudio)` — so the chunk after it starts from a silent
/// memory: its features are a fresh analysis of that chunk's own samples.
/// 5 s at 300 frames a chunk is chunks of 300 and 202; the segment lies in
/// the second chunk's output range [143 520, 240 000) alone.
#[test]
fn a_skipped_chunk_leaves_the_analysis_silent() {
    const CHUNK: usize = 300;
    let audio = speech_like(48_000, 5.0);
    let backend = Arc::new(EchoBackend::default());
    futures_executor::block_on(
        DeepFilterNet::new(backend.clone())
            .with_max_frames_per_chunk(CHUNK)
            .denoise_with_keep_segments(
                audio.clone(),
                &[speech_segment(3.5, 1.0)],
                ProgressSink::noop(),
            ),
    )
    .unwrap();

    let frames = (audio.samples.len() + FFT_SIZE) / HOP_SIZE - CHUNK;
    let after_silence = Analysis::new()
        .unwrap()
        .next(&audio.samples[CHUNK * HOP_SIZE..], frames)
        .unwrap();
    let (erb, spec) = chunk_features(&after_silence, frames);
    let fed = backend.fed.lock().unwrap();
    let second = &fed[fed.len() - frames.div_ceil(STREAM_FRAMES)..];
    assert_fed(
        second,
        &erb,
        &spec,
        frames,
        "the chunk after the skipped one",
    );
}

/// Each `<state>_out` is handed to its own `<state>` — not a neighbour's —
/// and the next chunk starts from zeros again.
#[test]
fn each_state_is_handed_to_itself_and_a_chunk_starts_from_zeros() {
    const FRAMES: usize = 300;
    let backend = Arc::new(StepPerStateBackend::default());
    let detector = DeepFilterNet::new(backend.clone());
    for _ in 0..2 {
        futures_executor::block_on(detector.run_chunk(
            &vec![0.5; FRAMES * NB_ERB],
            &vec![0.5; 2 * FRAMES * NB_DF],
            FRAMES,
        ))
        .unwrap();
    }
    let chunk = [
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    ];
    assert_eq!(
        *backend.handed.lock().unwrap(),
        [chunk.clone(), chunk].concat()
    );
}

#[test]
fn a_chunk_is_one_to_fifteen_hundred_frames_of_matching_features() {
    let detector = detector();
    for (frames, erb, spec) in [
        (0, 0, 0),
        (1_501, 1_501 * NB_ERB, 2 * 1_501 * NB_DF),
        (100, 99 * NB_ERB, 2 * 100 * NB_DF),
        (100, 100 * NB_ERB, 100 * NB_DF),
    ] {
        let error = futures_executor::block_on(detector.run_chunk(
            &vec![0.0; erb],
            &vec![0.0; spec],
            frames,
        ))
        .unwrap_err();
        assert!(
            matches!(error, EngineError::InvalidConfiguration(_)),
            "{frames}: {error}"
        );
    }
}

#[test]
fn events_mask_silences_non_speech_regions() {
    let result = futures_executor::block_on(detector().denoise_with_keep_segments(
        tone(48_000, 3.0, 800.0, 0.2),
        &[speech_segment(1.0, 1.0)],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.samples[(0.2 * 48_000.0) as usize].abs() < 1e-6);
    assert!(result.samples[(2.8 * 48_000.0) as usize].abs() < 1e-6);
    let middle = (1.5 * 48_000.0) as usize;
    assert!(rms(&result.samples[middle..middle + 4_800]) > 1e-5);
}

// DeepFilterNetMemoryTests.swift. The chunk loop holds one chunk's spectrum at
// a time; what grows with the source is the input it was handed and the
// output it returns.

#[test]
#[ignore = "manual RSS measurement: set FERMENT_RUN_DFNET_MEMORY=1"]
fn five_minute_input_stays_bounded_in_memory() {
    memory_regression(300.0, 300.0);
}

#[test]
#[ignore = "manual RSS measurement: set FERMENT_RUN_DFNET_MEMORY=1"]
fn ten_minute_input_stays_bounded_in_memory() {
    memory_regression(600.0, 200.0);
}

fn memory_regression(seconds: f64, maximum_delta_mb: f64) {
    assert_eq!(
        std::env::var("FERMENT_RUN_DFNET_MEMORY").as_deref(),
        Ok("1"),
        "set FERMENT_RUN_DFNET_MEMORY=1 deliberately"
    );
    let input = tone(48_000, seconds, 440.0, 0.05);
    let _warmup = futures_executor::block_on(
        detector().denoise(tone(48_000, 2.0, 440.0, 0.05), ProgressSink::noop()),
    )
    .unwrap();
    let before = resident_set_bytes().expect("RSS probe is only implemented through POSIX ps");
    let result = futures_executor::block_on(detector().denoise(input, ProgressSink::noop()))
        .expect("long offline input must complete");
    let after = resident_set_bytes().expect("RSS probe must remain available");
    let delta_mb = after.saturating_sub(before) as f64 / 1_048_576.0;
    assert!(
        delta_mb < maximum_delta_mb,
        "RSS delta {delta_mb:.0} MB exceeded {maximum_delta_mb:.0} MB"
    );
    let expected = (seconds * 48_000.0) as usize;
    assert!(result.samples.len().abs_diff(expected) < 2 * HOP_SIZE);
}

fn resident_set_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let kib = std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kib * 1_024)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

// libdf's own host half on `ferment-models-export`'s golden chunk
// (`models/dfnet/goldens`): 720 000 samples of the hairdryer clip, analysed
// from an empty memory into 1500 frames, and the checkpoint's own mask and
// coefficients synthesised back.

fn golden(name: &str) -> Vec<f32> {
    export::require_f32_le(&format!("dfnet/goldens/dfnet_{name}.f32"))
}

fn worst(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn the_analysis_is_libdfs() {
    let Some(audio) = export::f32_le("dfnet/goldens/dfnet_audio_in.f32") else {
        return;
    };
    assert_eq!(audio.len(), CHUNK_FRAMES * HOP_SIZE);
    let spectrum = Analysis::new().unwrap().next(&audio, CHUNK_FRAMES).unwrap();
    let delta = worst(&spectrum, &golden("spectrum"));
    eprintln!("[dfnet-analysis] max |Δ| vs libdf {delta:.3e}");
    assert!(delta < 1e-5, "{delta}");
}

/// The mask, the deep filter over the first 96 bins, the merge and the
/// inverse STFT — upstream's `DfNet.forward` from the spectrum on, whose
/// output the export wrote unshifted: the synthesis's warm-up included.
#[test]
fn the_synthesis_is_libdfs() {
    let Some(spectrum) = export::f32_le("dfnet/goldens/dfnet_spectrum.f32") else {
        return;
    };
    let masked = apply_erb_mask(&spectrum, &golden("erb_mask_out"), CHUNK_FRAMES);
    let filtered = apply_deep_filtering(&spectrum, &golden("df_coefs_out"), CHUNK_FRAMES);
    let mut audio = Vec::new();
    Synthesis::new()
        .unwrap()
        .push(&merge(&filtered, &masked, CHUNK_FRAMES), &mut audio)
        .unwrap();
    let delta = worst(&audio, &golden("enhanced_out"));
    eprintln!("[dfnet-synthesis] max |Δ| vs libdf {delta:.3e}");
    assert!(delta < 1e-4, "{delta}");
}

// Owned file output uses the same chunk DSP with bounded reads and writes.
mod disk_output_tests {
    use super::*;
    use crate::audio_source::{ArrayAudioSampleSource, AudioSampleSource};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "dfnet-file-test-{}-{time}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn input(&self) -> (PathBuf, AudioFrames) {
            let audio = tone(48_000, 1.031, 440.0, 0.05);
            let path = self.0.join("original.wav");
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for &sample in &audio.samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            (path, audio)
        }
        fn names(&self) -> Vec<String> {
            let mut names = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|p| p.unwrap().file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            names.sort();
            names
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn file_output_matches_array_across_mask_and_chunk_boundaries_and_owns_only_its_output() {
        let scratch = Scratch::new();
        let (path, audio) = scratch.input();
        let original = std::fs::read(&path).unwrap();
        let keep = [
            speech_segment(0.073, 0.101),
            speech_segment(0.174, 0.163),
            speech_segment(0.901, 0.13),
        ];
        let model = detector().with_max_frames_per_chunk(7);
        let expected = futures_executor::block_on(model.denoise_with_keep_segments(
            audio,
            &keep,
            ProgressSink::noop(),
        ))
        .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let source = futures_executor::block_on(model.denoise_to_file(
            &path,
            &scratch.0,
            Some(&keep),
            ProgressSink::from_callback(move |e| captured.lock().unwrap().push(e.progress)),
        ))
        .unwrap();
        let output = source.path().to_owned();
        assert!(output.starts_with(&scratch.0));
        assert_ne!(output, path);
        assert_eq!(source.sample_count(), expected.samples.len());
        let mut actual = vec![0.0; source.sample_count()];
        assert_eq!(
            source
                .copy_samples(&mut actual, 0, expected.samples.len())
                .unwrap(),
            expected.samples.len()
        );
        assert_eq!(
            actual, expected.samples,
            "mask, delay trim, tail and every chunk seam"
        );
        let mut window = [0.0; 257];
        source.copy_samples(&mut window, 3_301, 257).unwrap();
        assert_eq!(window.as_slice(), &actual[3_301..3_558]);
        assert_eq!(
            scratch.names().len(),
            2,
            "decoded input scratch is already gone"
        );
        let progress = events.lock().unwrap();
        assert_eq!(progress.last(), Some(&1.0));
        assert!(progress.windows(2).all(|p| p[1] >= p[0]));
        source.cleanup().unwrap();
        source.cleanup().unwrap();
        drop(source);
        assert!(!output.exists());
        assert_eq!(scratch.names(), ["original.wav"]);
        assert_eq!(std::fs::read(path).unwrap(), original);
    }

    #[test]
    fn windowed_input_reads_and_output_writes_are_bounded_independently_of_source_length() {
        struct Watched {
            inner: ArrayAudioSampleSource,
            largest: AtomicUsize,
        }
        impl AudioSampleSource for Watched {
            fn sample_count(&self) -> usize {
                self.inner.sample_count()
            }
            fn copy_samples(
                &self,
                out: &mut [f32],
                offset: usize,
                count: usize,
            ) -> ferment_engine_core::Result<usize> {
                self.largest.fetch_max(count, Ordering::Relaxed);
                self.inner.copy_samples(out, offset, count)
            }
        }
        let source = Watched {
            inner: ArrayAudioSampleSource::new(tone(48_000, 1.031, 440.0, 0.05).samples),
            largest: AtomicUsize::new(0),
        };
        let mut count = 0;
        let mut largest = 0;
        futures_executor::block_on(detector().with_max_frames_per_chunk(7).run_windowed(
            &source,
            None,
            ProgressSink::noop(),
            |block| {
                count += block.len();
                largest = largest.max(block.len());
                Ok(())
            },
        ))
        .unwrap();
        assert_eq!(count, source.sample_count());
        assert_eq!(source.largest.load(Ordering::Relaxed), 7 * HOP_SIZE);
        assert!(largest <= 7 * HOP_SIZE);
    }

    #[derive(Debug)]
    struct FailSecond(AtomicUsize);
    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl InferenceBackend for FailSecond {
        async fn infer(
            &self,
            _: ModelId,
            _: &ModelVersion,
            inputs: TensorMap,
        ) -> ferment_engine_core::Result<TensorMap> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 1 {
                return Err(EngineError::decode("forced second chunk failure"));
            }
            Ok(identity_answer(&inputs))
        }
    }

    #[test]
    fn model_failure_after_partial_output_removes_both_owned_files_and_preserves_input() {
        let scratch = Scratch::new();
        let (path, _) = scratch.input();
        let original = std::fs::read(&path).unwrap();
        let model = DeepFilterNet::new(Arc::new(FailSecond(AtomicUsize::new(0))))
            .with_max_frames_per_chunk(7);
        let error = futures_executor::block_on(model.denoise_to_file(
            &path,
            &scratch.0,
            None,
            ProgressSink::noop(),
        ))
        .unwrap_err();
        assert!(error.to_string().contains("forced second chunk failure"));
        assert_eq!(scratch.names(), ["original.wav"]);
        assert_eq!(std::fs::read(path).unwrap(), original);
    }

    #[test]
    fn cancellation_after_partial_output_removes_both_owned_files_and_preserves_input() {
        let scratch = Scratch::new();
        let (path, _) = scratch.input();
        let original = std::fs::read(&path).unwrap();
        let slot = Arc::new(Mutex::new(None::<ProgressSink>));
        let captured = slot.clone();
        let progress = ProgressSink::from_callback(move |e| {
            if e.progress > 0.05 {
                captured.lock().unwrap().as_ref().unwrap().cancel();
            }
        });
        *slot.lock().unwrap() = Some(progress.clone());
        let result = futures_executor::block_on(
            detector()
                .with_max_frames_per_chunk(7)
                .denoise_to_file(&path, &scratch.0, None, progress),
        );
        *slot.lock().unwrap() = None;
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert_eq!(scratch.names(), ["original.wav"]);
        assert_eq!(std::fs::read(path).unwrap(), original);
    }

    #[test]
    fn invalid_input_and_unwritable_scratch_transfer_no_file() {
        let scratch = Scratch::new();
        let (path, _) = scratch.input();
        let original = std::fs::read(&path).unwrap();
        assert!(
            futures_executor::block_on(detector().denoise_to_file(
                &path,
                Path::new(&path),
                None,
                ProgressSink::noop()
            ))
            .is_err()
        );
        let bad = scratch.0.join("bad.audio");
        std::fs::write(&bad, b"not an audio container").unwrap();
        assert!(
            futures_executor::block_on(detector().denoise_to_file(
                &bad,
                &scratch.0,
                None,
                ProgressSink::noop()
            ))
            .is_err()
        );
        assert_eq!(scratch.names(), ["bad.audio", "original.wav"]);
        assert_eq!(std::fs::read(path).unwrap(), original);
        assert_eq!(std::fs::read(bad).unwrap(), b"not an audio container");
    }
}
