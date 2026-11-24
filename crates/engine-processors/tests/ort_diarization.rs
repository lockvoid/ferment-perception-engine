//! Diarization end to end: the two ONNX masters running through the real ONNX
//! Runtime backend, on corpus-labelled speakers. Each graph's own parity moved
//! with its export (`tests/pyannote_contract.rs`, `tests/redimnet_contract.rs`);
//! here it is the diarizer — segmentation, stitching, the embedder — and what it
//! answers about who spoke.
//!
//! Both masters come from `ferment-models-export/artifacts`. Needs a loadable
//! ONNX Runtime (`ORT_DYLIB_PATH`).

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::sync::Arc;

use ferment_engine_audio::decode_path;
use ferment_engine_core::{ProgressSink, TimeRange};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::pyannote::{PyannoteSpeakerSegmenter, windows};
use ferment_engine_processors::redimnet::{ReDimNetSpeakerEncoder, windows as encoder_windows};
use ferment_engine_test_fixtures as fixtures;

const SPEAKER_CLIPS: [&str; 4] = ["speaker-a-1", "speaker-a-2", "speaker-b-1", "speaker-b-2"];

/// One backend serving both graphs — the diarizer needs them together.
fn backend() -> Backend {
    let mut paths = ModelPaths::new();
    if let Some(segmenter) = fixtures::export::artifact("pyannote_segmentation.onnx") {
        paths = paths.with_path(
            ModelId::PyannoteSegmentation,
            ModelVersion::Latest,
            segmenter,
        );
    }
    if let Some(encoder) = fixtures::export::artifact("redimnet_speaker.onnx") {
        paths = paths.with_path(ModelId::RedimnetSpeaker, ModelVersion::Latest, encoder);
    }
    let locator = Arc::new(paths);
    Arc::new(match std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() {
        Ok("directml") => {
            OrtBackend::with_execution_provider(locator, OrtExecutionProvider::DirectMl)
        }
        Ok("cuda") => OrtBackend::with_execution_provider(locator, OrtExecutionProvider::Cuda),
        Ok("windows") => OrtBackend::windows_default(locator),
        Ok("cpu") | Err(_) => {
            OrtBackend::with_execution_provider(locator, OrtExecutionProvider::Cpu)
        }
        Ok(other) => panic!("unsupported FERMENT_ENGINE_ORT_EP: {other}"),
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// A speaker clip decoded through the engine's own WAV path — the path the
/// processor uses, so this measures the port rather than a test helper.
fn clip(stem: &str) -> Option<Vec<f32>> {
    let path = fixtures::path(&format!("SpeakerEncoder/{stem}.wav"))?;
    let frames = decode_path(&path).expect("speaker clip decodes");
    assert_eq!(frames.sample_rate, encoder_windows::SAMPLE_RATE);
    Some(frames.samples)
}

/// The property the embedder exists for, on corpus-labelled speakers recorded
/// in different sessions: two utterances by one person must sit closer together
/// than either sits to another person. Asserted as a MARGIN because absolute
/// cosines move with recording conditions.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and ferment-models-export/artifacts"]
fn different_speakers_separate_by_a_clear_margin() {
    let encoder = ReDimNetSpeakerEncoder::new(backend());

    let mut pooled = Vec::new();
    for stem in SPEAKER_CLIPS {
        let Some(audio) = clip(stem) else { return };
        let embeddings = futures_executor::block_on(encoder.embed(
            &audio,
            &[TimeRange {
                start: 0.0,
                duration: 3600.0,
            }],
        ))
        .expect("the encoder runs");
        assert!(
            embeddings.len() > 1,
            "{stem} should tile into several windows"
        );
        pooled.push(ferment_engine_processors::pyannote::stitcher::pooled(
            &embeddings
                .into_iter()
                .map(|embedding| embedding.vector)
                .collect::<Vec<_>>(),
        ));
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
    eprintln!("redimnet identity: same {same:?} cross {cross:?} margin {margin:.4}");
    assert!(
        margin >= 0.15,
        "same-speaker {same:?}, cross-speaker {cross:?} — margin {margin}"
    );
}

fn diarize(audio: Vec<f32>) -> Vec<ferment_engine_core::LocalVoice> {
    let segmenter = PyannoteSpeakerSegmenter::new(backend());
    let frames = ferment_engine_audio::AudioFrames::new(audio, windows::SAMPLE_RATE);
    futures_executor::block_on(segmenter.diarize(frames, ProgressSink::noop()))
        .expect("diarization runs")
}

/// The case this round was built for. One person reading continuously is ONE
/// voice — the failure that started it was a cut solo pitch coming back as two,
/// because a 2 s window is below the working floor of any embedder and the old
/// wiring had nothing longer to pool.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and both masters in ferment-models-export/artifacts"]
fn one_person_reading_is_one_voice() {
    let Some(mut audio) = clip("speaker-a-1") else {
        return;
    };
    audio.extend(clip("speaker-a-2").expect("companion clip"));

    let voices = diarize(audio);

    assert_eq!(voices.len(), 1, "two reads by one speaker are one voice");
    assert_eq!(voices[0].vector.len(), encoder_windows::EMBEDDING_DIMENSION);
    let norm: f32 = voices[0].vector.iter().map(|v| v * v).sum();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "the pooled fingerprint is renormalised"
    );
    assert!(voices[0].confidence > 0.5);
    let spoken: f64 = voices[0].turns.iter().map(|t| t.duration()).sum();
    assert!(
        spoken > 15.0,
        "most of 24 s of reading is speech ({spoken}s)"
    );
}

/// The current iOS regression: eight short replies by one person still pool
/// enough clean speech for a fingerprint, although no reply fills one window.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and both masters in ferment-models-export/artifacts"]
fn short_replies_still_fingerprint_the_voice() {
    let speech = clip("speaker-a-1").expect("required current speaker fixture");
    let reply = 19_200;
    let pause = 16_000;
    let mut talk = Vec::new();
    for index in 0..8 {
        if (index + 1) * reply > speech.len() {
            break;
        }
        talk.extend_from_slice(&speech[index * reply..(index + 1) * reply]);
        talk.resize(talk.len() + pause, 0.0);
    }
    assert!(
        !talk.is_empty(),
        "the speaker fixture must contain complete replies"
    );

    let voices = diarize(talk);

    assert_eq!(voices.len(), 1, "one person talking is one voice");
    assert!(
        voices
            .iter()
            .flat_map(|voice| &voice.turns)
            .all(|turn| { turn.duration() < encoder_windows::minimum_span_seconds() }),
        "every reply is shorter than an encoder window"
    );
    assert_eq!(
        voices[0].vector.len(),
        encoder_windows::EMBEDDING_DIMENSION,
        "the voice has a fingerprint"
    );
}

/// Two people are two voices, and each voice's speech sits in its own half.
///
/// The bar is on CONFUSION — the part that is actually about telling two people
/// apart — rather than on a full DER, whose floor here is set by the pauses
/// inside each read rather than by the diarization.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and both masters in ferment-models-export/artifacts"]
fn two_people_are_two_voices_and_the_seam_is_found() {
    let Some(first) = clip("speaker-a-1") else {
        return;
    };
    let second = clip("speaker-b-1").expect("companion clip");
    let seam = first.len() as f64 / f64::from(windows::SAMPLE_RATE);
    let reference_end = (first.len() + second.len()) as f64 / f64::from(windows::SAMPLE_RATE);
    let mut audio = first;
    audio.extend(second);

    let voices = diarize(audio);
    assert_eq!(voices.len(), 2);
    // Kill: miss most speech while preserving perfect per-voice confusion.
    // Current Swift also grades total error: a diarizer that misses most speech
    // can have perfect per-voice confusion and still be unusable.
    let score = score_two_halves(&voices, seam, reference_end);
    eprintln!("two-speaker current iOS score: {score:?}");
    assert_eq!(
        score.mapping_count, 2,
        "each voice maps to a different person"
    );
    assert!(score.reference_speech > 0.0);
    assert!(score.confusion / score.reference_speech < 0.05);
    assert!(score.der < 0.25, "current iOS DER bound: {score:?}");

    let mut halves = Vec::new();
    for voice in &voices {
        let before: f64 = voice
            .turns
            .iter()
            .map(|turn| (seam.min(turn.end_time) - turn.start_time).max(0.0))
            .sum();
        let total: f64 = voice.turns.iter().map(|t| t.duration()).sum();
        halves.push((before, total - before));
    }
    eprintln!("diarization halves (before, after) per voice: {halves:?}");

    // Each voice belongs to a different half, and hardly any of its speech
    // lands in the other one.
    let owner = |(before, after): (f64, f64)| usize::from(after > before);
    assert_ne!(
        owner(halves[0]),
        owner(halves[1]),
        "each voice maps to a different person"
    );
    for (before, after) in halves {
        let confusion = before.min(after) / (before + after);
        assert!(
            confusion < 0.05,
            "under 5% of a voice's speech may land in the other person's half (got {confusion:.3})"
        );
    }
}

/// Audio with no speech in it is no voices — not one voice with no replies, and
/// not a failure.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and both masters in ferment-models-export/artifacts"]
fn silence_is_no_voices() {
    assert!(diarize(vec![0.0; 16_000 * 12]).is_empty());
}

/// Audio too short for a single model frame is answered, not thrown at.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and both masters in ferment-models-export/artifacts"]
fn audio_shorter_than_one_frame_is_no_voices() {
    assert!(diarize(vec![0.1; 500]).is_empty());
}

#[derive(Debug)]
struct TwoSpeakerScore {
    der: f64,
    confusion: f64,
    reference_speech: f64,
    mapping_count: usize,
}

/// Current Swift test-only DiarizationScorer specialized to this two-person
/// fixture: 10 ms midpoint rasterization, best global one-to-one assignment,
/// and the same 0.5 s full-width reference collar. With two voices the two
/// permutations exactly replace Hungarian assignment; no production seam.
fn score_two_halves(
    voices: &[ferment_engine_core::LocalVoice],
    seam: f64,
    reference_end: f64,
) -> TwoSpeakerScore {
    assert_eq!(voices.len(), 2);
    const STEP: f64 = 0.01;
    const HALF_COLLAR: f64 = 0.25;
    let end = voices
        .iter()
        .flat_map(|voice| &voice.turns)
        .map(|turn| turn.end_time)
        .fold(reference_end, f64::max);
    let frames = (end / STEP).ceil() as usize + 1;
    let mut truth = vec![[false; 2]; frames];
    let mut answer = vec![[false; 2]; frames];
    let rasterize = |rows: &mut [[bool; 2]], speaker: usize, start: f64, end: f64| {
        if end <= start {
            return;
        }
        let from = (start / STEP - 0.5).ceil().max(0.0) as usize;
        let to = ((end / STEP - 0.5).ceil().max(0.0) as usize).min(frames);
        for row in rows.iter_mut().take(to).skip(from) {
            row[speaker] = true;
        }
    };
    rasterize(&mut truth, 0, 0.0, seam);
    rasterize(&mut truth, 1, seam, reference_end);
    for (speaker, voice) in voices.iter().enumerate() {
        for turn in &voice.turns {
            rasterize(&mut answer, speaker, turn.start_time, turn.end_time);
        }
    }
    let mut agreement = [[0_usize; 2]; 2];
    for (truth, answer) in truth.iter().zip(&answer) {
        for system in 0..2 {
            for reference in 0..2 {
                agreement[system][reference] += usize::from(answer[system] && truth[reference]);
            }
        }
    }
    let mapping = if agreement[0][0] + agreement[1][1] >= agreement[0][1] + agreement[1][0] {
        [0, 1]
    } else {
        [1, 0]
    };
    let mapping_count = (0..2)
        .filter(|&system| agreement[system][mapping[system]] > 0)
        .count();
    let mut scorable = vec![true; frames];
    for boundary in [0.0, seam, reference_end] {
        let from = ((boundary - HALF_COLLAR) / STEP).floor().max(0.0) as usize;
        let to = (((boundary + HALF_COLLAR) / STEP).ceil().max(0.0) as usize).min(frames);
        for scored in scorable.iter_mut().take(to).skip(from) {
            *scored = false;
        }
    }
    let (mut missed, mut spurious, mut confused, mut spoken) = (0_usize, 0_usize, 0_usize, 0_usize);
    for ((truth, answer), scorable) in truth.iter().zip(&answer).zip(scorable) {
        if !scorable {
            continue;
        }
        let in_truth = truth.iter().filter(|&&active| active).count();
        let in_answer = answer.iter().filter(|&&active| active).count();
        let agreed = (0..2)
            .filter(|&system| {
                answer[system] && agreement[system][mapping[system]] > 0 && truth[mapping[system]]
            })
            .count();
        missed += in_truth.saturating_sub(in_answer);
        spurious += in_answer.saturating_sub(in_truth);
        confused += in_truth.min(in_answer) - agreed;
        spoken += in_truth;
    }
    assert!(
        spoken > 0,
        "reference fixture must contain speech outside the collar"
    );
    TwoSpeakerScore {
        der: (missed + spurious + confused) as f64 / spoken as f64,
        confusion: confused as f64 * STEP,
        reference_speech: spoken as f64 * STEP,
        mapping_count,
    }
}
