//! Current iOS empty-speech assertions at the real public Android façade.
//! Missing audio and model paths make any accidental work fail before success.

use std::sync::{Arc, Mutex};

use crate::{
    EngineService, LanguageIdentifier, ModelCacheState, ProgressListener, SpeakerEncoder,
    SpeechTranscriber, TranscriberKind,
};

#[derive(Default)]
struct Stages(Mutex<Vec<(f64, String)>>);

impl ProgressListener for Stages {
    fn on_progress(&self, progress: f64, stage: String) {
        self.0.lock().expect("stage lock").push((progress, stage));
    }
}

fn empty_service() -> Arc<EngineService> {
    Arc::new(EngineService::new("/dev/null/no-models".to_owned()).expect("lazy service"))
}

fn assert_terminal_without_models(service: &EngineService, stages: &Stages, expected: &str) {
    let observed = stages.0.lock().expect("stage lock");
    assert_eq!(observed.last(), Some(&(1.0, expected.to_owned())));
    assert_eq!(
        service.model_cache_state(),
        ModelCacheState {
            resident: 0,
            loading: 0,
            waiters: 0
        }
    );
}

/// Kill: decode the missing file, load AmberNet, or omit terminal progress on [].
/// iOS: AmberNetLanguageIDTests.testEmptySpeechSegmentsShortCircuitsBeforeAnyLoad.
#[test]
fn language_id_empty_segments_finish_before_decode_or_model_load() {
    let service = empty_service();
    let stages = Arc::new(Stages::default());
    let listener: Arc<dyn ProgressListener> = stages.clone();
    let result = LanguageIdentifier::new(service.clone())
        .identify(
            "/dev/null/does-not-exist.wav".to_owned(),
            Some(Vec::new()),
            Some(listener),
            None,
        )
        .expect("empty speech is an answer without an input file");
    assert!(result.is_empty());
    assert_terminal_without_models(&service, &stages, "No speech detected");
}

/// Kill: decode the missing file, load ReDimNet, or omit terminal progress on [].
/// iOS: ReDimNetSpeakerEncoderTests.testEmptySegmentsShortCircuitBeforeAnyLoad.
#[test]
fn speaker_encoding_empty_segments_finish_before_decode_or_model_load() {
    let service = empty_service();
    let stages = Arc::new(Stages::default());
    let listener: Arc<dyn ProgressListener> = stages.clone();
    let result = SpeakerEncoder::new(service.clone())
        .embed(
            "/dev/null/nope.wav".to_owned(),
            Vec::new(),
            Some(listener),
            None,
        )
        .expect("empty speech is an answer without an input file");
    assert!(result.is_empty());
    assert_terminal_without_models(&service, &stages, "No speech to embed");
}

/// Kill: a real decoded no-word result advertises the model language as detected speech.
/// Current Swift GigaAMEdgeProbeTests requires nil language even after decoding.
#[test]
fn gigaam_sub_window_audio_has_no_language_without_loading_models() {
    let service = empty_service();
    let input = test_audio_file(&[0.05]);
    let result = SpeechTranscriber::new(service.clone(), TranscriberKind::GigaAm)
        .transcribe(
            input.path(),
            std::env::temp_dir().to_string_lossy().into_owned(),
            None,
            None,
            None,
        )
        .expect("one sample is a valid no-speech input");
    assert!(result.transcription.is_empty());
    assert!(result.captions.is_empty());
    assert!(result.language.is_none());
    assert_eq!(
        service.model_cache_state(),
        ModelCacheState {
            resident: 0,
            loading: 0,
            waiters: 0
        }
    );
}

/// Kill: the empty-segment gate starts the loading ladder or names a language.
#[test]
fn gigaam_empty_segments_have_only_the_no_speech_terminal_stage() {
    let service = empty_service();
    let stages = Arc::new(Stages::default());
    let listener: Arc<dyn ProgressListener> = stages.clone();
    let result = SpeechTranscriber::new(service.clone(), TranscriberKind::GigaAm)
        .transcribe(
            "/nonexistent/gigaam/not-a-real-file.mp3".to_owned(),
            std::env::temp_dir().to_string_lossy().into_owned(),
            Some(Vec::new()),
            Some(listener),
            None,
        )
        .expect("no input or model load");
    assert!(result.transcription.is_empty() && result.captions.is_empty());
    assert!(result.language.is_none());
    assert_eq!(
        *stages.0.lock().expect("stages"),
        [(1.0, "No speech detected".to_owned())]
    );
    assert_terminal_without_models(&service, &stages, "No speech detected");
}

struct TestAudio(std::path::PathBuf);
impl TestAudio {
    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}
impl Drop for TestAudio {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.0) {
            eprintln!(
                "speech fixture cleanup failed at {}: {error}",
                self.0.display()
            );
        }
    }
}

/// Standard little-endian float WAV. No codec helper substitutes for native ingest.
fn test_audio_file(samples: &[f32]) -> TestAudio {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferment-speech-{}-{}.wav",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut bytes = Vec::new();
    let data_size = (samples.len() * 4) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&3_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&16_000_u32.to_le_bytes());
    bytes.extend_from_slice(&64_000_u32.to_le_bytes());
    bytes.extend_from_slice(&4_u16.to_le_bytes());
    bytes.extend_from_slice(&32_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for value in samples {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).expect("write owned WAV");
    TestAudio(path)
}
