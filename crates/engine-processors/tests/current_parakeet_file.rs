//! Real-model qualification of the bounded file door against the existing
//! array door. The input repeats the pinned original export speech samples;
//! it is a synthetic long fixture, not a claimed original iOS recording.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]
use ferment_engine_audio::{AudioFrames, write_wav_f32};
use ferment_engine_core::{AudioEvent, AudioEventSegment, ProgressSink};
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::{
    Parakeet,
    parakeet::{config::SAMPLE_RATE, speech_segment_coalescer::coalesce as coalesce_segments},
};
use ferment_engine_test_fixtures::export;
use futures_executor::block_on;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Input {
    directory: PathBuf,
    path: PathBuf,
    audio: Vec<f32>,
}
impl Input {
    fn new() -> Self {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "admitted ORT is required"
        );
        assert!(
            std::env::var_os("FERMENT_MODELS_EXPORT_DIR").is_some(),
            "admitted original export goldens are required"
        );
        let original = export::require_f32_le("parakeet/goldens/parakeet_reel_audio_in.f32");
        assert!(!original.is_empty());
        let audio: Vec<_> = original
            .iter()
            .copied()
            .cycle()
            .take(SAMPLE_RATE as usize * 43 + 137)
            .collect();
        let directory = std::env::temp_dir().join(format!(
            "parakeet-real-file-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("input.wav");
        write_wav_f32(&path, &audio, SAMPLE_RATE).unwrap();
        Self {
            directory,
            path,
            audio,
        }
    }
    fn model(&self) -> Parakeet {
        let encoder =
            export::artifact("parakeet_encoder.onnx").expect("approved encoder is mandatory");
        let decode = export::artifact("parakeet_decode.bin")
            .expect("approved host decode weights are mandatory");
        let paths =
            ModelPaths::new().with_path(ModelId::ParakeetEncoder, ModelVersion::Latest, encoder);
        let backend = Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        ));
        Parakeet::new(backend, decode)
    }
    fn verify_cleanup(&self, before: &[u8]) {
        assert_eq!(
            std::fs::read_dir(&self.directory).unwrap().count(),
            1,
            "no PCM survives return"
        );
        assert_eq!(
            std::fs::read(&self.path).unwrap(),
            before,
            "caller input preserved byte for byte"
        );
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
type Word = (String, f64, f64, f32);
fn array_words(
    result: ferment_engine_processors::parakeet::TranscriptionResult,
    offset: f64,
) -> Vec<Word> {
    result
        .captions
        .into_iter()
        .flat_map(|c| c.words)
        .map(|w| {
            (
                w.text,
                w.start_time + offset,
                (w.end_time - w.start_time).max(0.0),
                w.confidence,
            )
        })
        .collect()
}
fn file_words(result: ferment_engine_core::TranscriptionResult) -> Vec<Word> {
    result
        .captions
        .into_iter()
        .flat_map(|c| c.words)
        .map(|w| (w.text, w.start_time, w.duration, w.confidence))
        .collect()
}

/// Kill: any window read, context, final-tail or token-merging difference
/// between caller-owned disk input and the pre-existing in-memory caller.
#[test]
#[ignore = "requires the approved ORT runtime, parakeet encoder/decode and original export reel PCM"]
fn real_file_and_array_paths_keep_every_word_and_chunk_seam() {
    let input = Input::new();
    let original = std::fs::read(&input.path).unwrap();
    let model = input.model();
    let expected = array_words(
        block_on(model.transcribe(
            AudioFrames::new(input.audio.clone(), SAMPLE_RATE),
            ProgressSink::noop(),
        ))
        .unwrap(),
        0.0,
    );
    assert!(
        !expected.is_empty(),
        "empty inference cannot establish speech parity"
    );
    let actual = block_on(model.transcribe_file_in(
        &input.path,
        &input.directory,
        None,
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!(
        file_words(actual),
        expected,
        "text, sample-aligned word times and confidence at every seam"
    );
    input.verify_cleanup(&original);
}

/// Kill: materialize/shift a >30 s selected run, reset its timestamps, fail to
/// coalesce the current padded runs, regress progress or leak successful scratch.
#[test]
#[ignore = "requires the approved ORT runtime, parakeet encoder/decode and original export reel PCM"]
fn real_selected_runs_keep_absolute_words_and_monotonic_terminal_progress() {
    let input = Input::new();
    let original = std::fs::read(&input.path).unwrap();
    let model = input.model();
    let segments =
        [(3.0, 16.0), (19.5, 15.5), (40.0, 2.0)].map(|(start_time, duration)| AudioEventSegment {
            event: AudioEvent::Speech,
            start_time,
            duration,
            confidence: 1.0,
        });
    let runs = coalesce_segments(
        &segments,
        1.0,
        input.audio.len() as f64 / f64::from(SAMPLE_RATE),
    );
    assert_eq!(runs.len(), 2);
    assert!(
        runs[0].duration > 30.0,
        "long selected speech must also stay windowed"
    );
    let mut expected = Vec::new();
    for run in runs {
        let start = (run.start_time * f64::from(SAMPLE_RATE)) as usize;
        let end = (run.end_time() * f64::from(SAMPLE_RATE)) as usize;
        expected.extend(array_words(
            block_on(model.transcribe(
                AudioFrames::new(input.audio[start..end].to_vec(), SAMPLE_RATE),
                ProgressSink::noop(),
            ))
            .unwrap(),
            run.start_time,
        ));
    }
    assert!(!expected.is_empty());
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let actual = block_on(model.transcribe_file_in(
        &input.path,
        &input.directory,
        Some(&segments),
        ProgressSink::from_callback(move |e| captured.lock().unwrap().push((e.progress, e.stage))),
    ))
    .unwrap();
    assert_eq!(file_words(actual), expected);
    let events = events.lock().unwrap();
    assert!(
        events.windows(2).all(|p| p[0].0 <= p[1].0),
        "progress restarts between speech runs"
    );
    assert_eq!(events.last(), Some(&(1.0, "done".to_owned())));
    assert_eq!(events.iter().filter(|e| e.0 == 1.0).count(), 1);
    input.verify_cleanup(&original);
}
