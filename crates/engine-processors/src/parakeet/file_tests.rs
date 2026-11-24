//! File-lifetime and bounded-read regressions. Stubbed inference tests here
//! prove admission/cleanup only; real ORT file parity has a separate gate.
use super::*;
use async_trait::async_trait;
use ferment_engine_audio::write_wav_f32;
use ferment_engine_infer::InferenceBackend;
use futures_executor::block_on;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct VirtualSource {
    count: usize,
    calls: Mutex<Vec<(usize, usize)>>,
    short: bool,
}
impl AudioSampleSource for VirtualSource {
    fn sample_count(&self) -> usize {
        self.count
    }
    fn copy_samples(&self, output: &mut [f32], offset: usize, count: usize) -> Result<usize> {
        assert!(count <= WINDOW_SAMPLES, "duration-sized read");
        assert_eq!(count, output.len());
        assert!(offset + count <= self.count);
        self.calls.lock().unwrap().push((offset, count));
        let actual = count - usize::from(self.short && count > 0);
        for (i, sample) in output[..actual].iter_mut().enumerate() {
            *sample = ((offset + i) % 1009) as f32;
        }
        Ok(actual)
    }
}
fn virtual_source(count: usize) -> VirtualSource {
    VirtualSource {
        count,
        calls: Mutex::default(),
        short: false,
    }
}

/// Kill: materialize a whole twenty-minute run, lose overlap context, shift the
/// selected-run origin, or omit its final non-window-aligned samples.
#[test]
fn twenty_minutes_and_a_selected_run_read_only_exact_encoder_windows() {
    let from = 5_073;
    let count = SAMPLE_RATE as usize * 1_200 + 137;
    let source = virtual_source(from + count + 89);
    let run = from..from + count;
    let planned = windows(count);
    assert!(planned.len() >= 80);
    for window in &planned {
        let audio = read_window(&source, &run, *window).unwrap();
        assert!(audio.len() <= WINDOW_SAMPLES);
        assert_eq!(audio.len(), window.to - window.from);
        assert_eq!(audio[0], ((from + window.from) % 1009) as f32);
        assert_eq!(
            *audio.last().unwrap(),
            ((from + window.to - 1) % 1009) as f32
        );
        assert_eq!(window.first_frame * SAMPLES_PER_ENCODER_FRAME, window.from);
    }
    assert_eq!(planned.last().unwrap().to, count);
    let calls = source.calls.lock().unwrap();
    assert_eq!(calls.len(), planned.len());
    assert_eq!(calls.first().unwrap().0, from);
    assert_eq!(calls.last().unwrap().0 + calls.last().unwrap().1, run.end);
    for pair in planned.windows(2) {
        assert_eq!(
            pair[0].to - (pair[1].from + CONTEXT_SAMPLES),
            OVERLAP_SAMPLES
        );
    }
}

/// Kill: silently zero-pad an unexpectedly truncated source and move seam words.
#[test]
fn declared_short_read_is_an_error_instead_of_silence() {
    let mut source = virtual_source(16_001);
    source.short = true;
    let error = read_window(&source, &(0..16_001), windows(16_001)[0]).unwrap_err();
    assert!(error.to_string().contains("16000 of 16001"));
}

/// Kill: permit out-of-range slices or a duration-sized allocation before read.
#[test]
fn invalid_ranges_and_oversized_windows_fail_before_touching_the_source() {
    let source = virtual_source(WINDOW_SAMPLES * 2);
    let tiny = windows(100)[0];
    assert!(read_window(&source, &std::ops::Range { start: 9, end: 8 }, tiny).is_err());
    assert!(read_window(&source, &(0..source.count + 1), tiny).is_err());
    assert!(read_window(&source, &(0..99), tiny).is_err());
    assert!(
        read_window(
            &source,
            &(0..source.count),
            Window {
                from: 0,
                to: WINDOW_SAMPLES + 1,
                first_frame: 0,
            }
        )
        .is_err()
    );
    assert!(source.calls.lock().unwrap().is_empty());
}

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferment-parakeet-file-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("sentinel"), b"another caller's file").unwrap();
        Self(path)
    }
    fn input(&self) -> PathBuf {
        let input = self.0.join("input.wav");
        let audio = (0..31 * SAMPLE_RATE as usize)
            .map(|n| 0.02 * (n as f32 * 0.0864).sin())
            .collect::<Vec<_>>();
        write_wav_f32(&input, &audio, SAMPLE_RATE).unwrap();
        input
    }
    fn names(&self) -> Vec<String> {
        names(&self.0)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn names(path: &Path) -> Vec<String> {
    let mut result = std::fs::read_dir(path)
        .unwrap()
        .map(|p| p.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    result.sort();
    result
}

#[derive(Debug)]
struct InspectThenFail {
    directory: PathBuf,
    calls: AtomicUsize,
}
#[async_trait]
impl InferenceBackend for InspectThenFail {
    async fn infer(&self, model: ModelId, _: &ModelVersion, _: TensorMap) -> Result<TensorMap> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(model, ModelId::ParakeetEncoder);
        let files = names(&self.directory);
        assert_eq!(
            files.len(),
            3,
            "input + another caller's sentinel + one owned PCM"
        );
        let pcm = files.iter().find(|n| n.ends_with(".f32")).unwrap();
        assert_eq!(
            std::fs::metadata(self.directory.join(pcm)).unwrap().len(),
            31 * 16_000 * 4
        );
        Err(EngineError::inference(
            "intentional encoder failure after PCM admission",
        ))
    }
}
fn model(scratch: &Scratch) -> (Parakeet, Arc<InspectThenFail>) {
    let backend = Arc::new(InspectThenFail {
        directory: scratch.0.clone(),
        calls: AtomicUsize::new(0),
    });
    (
        Parakeet::new(backend.clone(), scratch.0.join("absent-decode.bin")),
        backend,
    )
}
fn assert_owned_files_released(scratch: &Scratch, input: &Path, original: &[u8]) {
    assert_eq!(scratch.names(), ["input.wav", "sentinel"]);
    assert_eq!(std::fs::read(input).unwrap(), original);
    assert_eq!(
        std::fs::read(scratch.0.join("sentinel")).unwrap(),
        b"another caller's file"
    );
}

/// Kill: use an in-memory file path, leak PCM when inference fails, or delete
/// the source/other caller's file during cleanup.
#[test]
fn model_failure_sees_one_disk_source_then_releases_only_its_owned_pcm() {
    let scratch = Scratch::new();
    let input = scratch.input();
    let original = std::fs::read(&input).unwrap();
    let (model, backend) = model(&scratch);
    let error = block_on(model.transcribe_file_in(&input, &scratch.0, None, ProgressSink::noop()))
        .unwrap_err();
    assert!(error.to_string().contains("intentional encoder failure"));
    assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
    assert_owned_files_released(&scratch, &input, &original);
}

/// Kill: a progress-triggered cancellation still calls the encoder, or leaves
/// the already decoded source after returning cancellation to the caller.
#[test]
fn cancellation_after_decode_cleans_pcm_before_any_model_load() {
    let scratch = Scratch::new();
    let input = scratch.input();
    let original = std::fs::read(&input).unwrap();
    let (model, backend) = model(&scratch);
    let holder: Arc<Mutex<Option<ProgressSink>>> = Arc::default();
    let cancellation = holder.clone();
    let directory = scratch.0.clone();
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let progress = ProgressSink::from_callback(move |event| {
        captured.lock().unwrap().push(event.stage.clone());
        if event.stage == "loading vocab" {
            assert_eq!(
                names(&directory).len(),
                3,
                "PCM is admitted before cancellation"
            );
            cancellation.lock().unwrap().as_ref().unwrap().cancel();
        }
    });
    *holder.lock().unwrap() = Some(progress.clone());
    let error = block_on(model.transcribe_file_in(&input, &scratch.0, None, progress)).unwrap_err();
    holder.lock().unwrap().take();
    assert!(matches!(error, EngineError::Cancelled));
    assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    assert_eq!(*events.lock().unwrap(), ["loading audio", "loading vocab"]);
    assert_owned_files_released(&scratch, &input, &original);
}

/// Kill: rejected scratch storage mutates the input or begins expensive inference.
#[test]
fn unwritable_scratch_fails_without_input_mutation_or_inference() {
    let scratch = Scratch::new();
    let input = scratch.input();
    let original = std::fs::read(&input).unwrap();
    let (model, backend) = model(&scratch);
    let error = block_on(model.transcribe_file_in(
        &input,
        &scratch.0.join("missing/child"),
        None,
        ProgressSink::noop(),
    ))
    .unwrap_err();
    assert!(matches!(error, EngineError::Io(_)));
    assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    assert_owned_files_released(&scratch, &input, &original);
}

/// Kill: a canceled request decodes the nonexistent input or opens scratch/model files.
#[test]
fn pre_cancelled_call_never_opens_audio_or_scratch() {
    let scratch = Scratch::new();
    let (model, backend) = model(&scratch);
    let progress = ProgressSink::noop();
    progress.cancel();
    let error = block_on(model.transcribe_file_in(
        Path::new("/no/audio"),
        Path::new("/no/scratch"),
        None,
        progress,
    ))
    .unwrap_err();
    assert!(matches!(error, EngineError::Cancelled));
    assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    assert_eq!(scratch.names(), ["sentinel"]);
}

/// Kill: [] attempts audio/model admission or loses its sole terminal callback.
#[test]
fn empty_segments_short_circuit_the_file_path_before_scratch_admission() {
    let scratch = Scratch::new();
    let (model, backend) = model(&scratch);
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let result = block_on(model.transcribe_file_in(
        Path::new("/no/audio"),
        Path::new("/no/scratch"),
        Some(&[]),
        ProgressSink::from_callback(move |e| captured.lock().unwrap().push((e.progress, e.stage))),
    ))
    .unwrap();
    assert!(result.transcription.is_empty() && result.captions.is_empty());
    assert!(result.language.is_none());
    assert_eq!(
        *events.lock().unwrap(),
        [(1.0, "No speech detected".to_owned())]
    );
    assert_eq!(backend.calls.load(Ordering::Relaxed), 0);
    assert_eq!(scratch.names(), ["sentinel"]);
}
