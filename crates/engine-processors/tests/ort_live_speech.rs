//! The Nemotron streaming transcriber against real speech, through the real
//! three-graph export.
//!
//! The ported iOS suite is `NemotronTranscriberTests`. Every one of its cases
//! is REAL by FLEET_LAW — real audio, real weights, and an assertion about the
//! WORLD (which language, which words) rather than about the code — and every
//! one is model-dependent, so they live here behind `--ignored` and the same
//! artifact guards the rest of the ORT oracle uses:
//!
//! ```text
//! FERMENT_ENGINE_ONNX_DIR   the flat directory holding the fleet's own ONNX
//!                           file names and the two nemotron_*.json sidecars
//!                           (`ferment-models-export/artifacts` is one)
//! FERMENT_ENGINE_FIXTURES_DIR  FermentEngine/Tests/Fixtures
//! ORT_DYLIB_PATH            the ONNX Runtime
//! ```
//!
//! A missing artifact prints `SKIP` and returns rather than failing, because a
//! 2.4 GB export is not a checkout.

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_core::{
    LiveSpeechTranscribing, ProgressSink, SpeechTranscribing, TranscriptionResult,
};
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_models::BackendKind;
use ferment_engine_processors::NemotronTranscriber;
use ferment_engine_processors::audio_source::ArrayAudioSampleSource;
use ferment_engine_processors::nemotron::config;

/// The file the FLEET publishes for this graph, not `id.name() + ".onnx"`:
/// an artifact whose name says something about itself — the encoder's weights
/// are stored fp16, so it is `nemotron_encoder.fp16.onnx` — would otherwise be
/// looked for under a name nobody writes, and every one of these tests would
/// SKIP forever while reading as green.
fn ort_file_name(id: &ModelId) -> String {
    ferment_engine_models::default_fleet_manifest()
        .model(*id)
        .and_then(|model| {
            model
                .artifacts
                .iter()
                .find(|artifact| artifact.backend == BackendKind::Ort)
                .map(|artifact| artifact.file_name.clone())
        })
        .unwrap_or_else(|| format!("{}.onnx", id.name()))
}

fn onnx_directory() -> Option<PathBuf> {
    let path = std::env::var_os("FERMENT_ENGINE_ONNX_DIR")
        .or_else(|| std::env::var_os("FERMENT_ENGINE_MODELS_DIR"))
        .map(PathBuf::from)?;
    path.is_dir().then_some(path)
}

/// The models directory, or `None` with a printed SKIP naming what is absent.
fn models(test: &str) -> Option<PathBuf> {
    let Some(directory) = onnx_directory() else {
        println!("SKIP {test}: set FERMENT_ENGINE_ONNX_DIR to the Nemotron export");
        return None;
    };
    let missing: Vec<String> = NemotronTranscriber::MODEL_FILES
        .iter()
        .map(ort_file_name)
        .chain([
            format!("{}.json", config::VOCAB_FILE),
            format!("{}.json", config::METADATA_FILE),
        ])
        .filter(|name| !directory.join(name).exists())
        .collect();
    if missing.is_empty() {
        Some(directory)
    } else {
        println!(
            "SKIP {test}: {} lacks {}",
            directory.display(),
            missing.join(", ")
        );
        None
    }
}

fn fixture(test: &str, relative: &str) -> Option<PathBuf> {
    let root = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").map(PathBuf::from)?;
    let path = root.join(relative);
    if path.exists() {
        Some(path)
    } else {
        println!("SKIP {test}: no fixture at {}", path.display());
        None
    }
}

fn transcriber(directory: &Path) -> NemotronTranscriber {
    let mut paths = ModelPaths::new();
    for id in NemotronTranscriber::MODEL_FILES {
        paths.insert(id, ModelVersion::Latest, directory.join(ort_file_name(&id)));
    }
    NemotronTranscriber::new(
        Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        )),
        directory,
    )
}

fn samples(path: &Path, seconds: Option<f64>) -> Vec<f32> {
    let frames = futures_executor::block_on(
        ferment_engine_processors::audio_source::AudioSource::default().decode(path),
    )
    .expect("decode fixture");
    let audio = if frames.sample_rate == config::SAMPLE_RATE {
        frames.samples
    } else {
        ferment_engine_audio::resample_sinc(
            &frames.samples,
            frames.sample_rate,
            config::SAMPLE_RATE,
        )
        .expect("resample")
    };
    match seconds {
        Some(seconds) => {
            let count = ((seconds * f64::from(config::SAMPLE_RATE)) as usize).min(audio.len());
            audio[..count].to_vec()
        }
        None => audio,
    }
}

fn finish(directory: &Path, audio: Vec<f32>) -> TranscriptionResult {
    let transcriber = transcriber(directory);
    futures_executor::block_on(async {
        let mut session = transcriber.make_live_session().await.expect("live session");
        session
            .finish(&ArrayAudioSampleSource::new(audio))
            .await
            .expect("finish")
    })
}

/// The English reel is transcribed as English and says what it says.
///
/// iOS asserts the same phrase from the same file.
///
/// kill: stop carrying the RNN-T decoder's LSTM state across symbols
/// (`self.hidden`/`self.cell`) and the greedy walk loses its history — the
/// phrase goes.
///
/// NOT kills, both MEASURED on this file:
/// * zeroing the nine-frame mel cache instead of prepending the previous
///   chunk's tail;
/// * throwing away `cache_channel_out` so every chunk encodes with an empty
///   attention cache.
///
/// The second is the whole point of a cache-AWARE model, and this assertion
/// cannot see it: a 4½-minute reel still says "which animal survived" with no
/// cross-chunk context at all. A test whose subject IS the cache carry is owed
/// and is not written here (report §8) — naming the hole is worth more than a
/// second phrase assert that would not close it either.
#[test]
#[ignore = "needs the Nemotron export and the iOS fixtures"]
fn the_english_reel_is_transcribed_as_english() {
    let Some(directory) = models("the_english_reel_is_transcribed_as_english") else {
        return;
    };
    let Some(path) = fixture(
        "the_english_reel_is_transcribed_as_english",
        "Audio/animal-evolution-edu-reel.mp3",
    ) else {
        return;
    };
    let transcriber = transcriber(&directory);
    let result =
        futures_executor::block_on(transcriber.transcribe(&path, None, ProgressSink::noop()))
            .expect("transcribe");
    println!("[NEMOTRON] en text: {}", result.transcription);
    assert_eq!(result.language.as_deref(), Some("en"));
    assert!(
        result.transcription.contains("which animal survived"),
        "{}",
        result.transcription
    );
}

/// The Russian take is transcribed as Russian.
///
/// kill: pin `prompt_id` to `en-US` (0) instead of the metadata's
/// `default_prompt_id` (`auto`, 101) and the model stops picking the language
/// itself — MEASURED: under 0 this exact file transcribes to the EMPTY string
/// while the English reel stays word-perfect, which is how the hard-coded 0
/// was found.
///
/// The current iOS assertion is the same stable phrase heard by this export;
/// the owned 2026-09-25 real-model receipt contains it verbatim.
#[test]
#[ignore = "needs the Nemotron export and the iOS fixtures"]
fn the_russian_take_is_transcribed_as_russian() {
    let Some(directory) = models("the_russian_take_is_transcribed_as_russian") else {
        return;
    };
    let Some(path) = fixture("the_russian_take_is_transcribed_as_russian", "volkova3.mp3") else {
        return;
    };
    let transcriber = transcriber(&directory);
    let result =
        futures_executor::block_on(transcriber.transcribe(&path, None, ProgressSink::noop()))
            .expect("transcribe");
    println!("[NEMOTRON] ru language: {:?}", result.language);
    println!("[NEMOTRON] ru text: {}", result.transcription);
    assert_eq!(result.language.as_deref(), Some("ru"));
    assert!(
        result.transcription.contains("мотивация ходить в зал"),
        "{}",
        result.transcription
    );
    assert!(
        result.transcription.contains("собираться в"),
        "{}",
        result.transcription
    );
    assert!(
        result.transcription.contains("спортивная форм"),
        "{}",
        result.transcription
    );
}

/// Every word lands inside the audio it came from, in order, and the transcript
/// SPANS it — both ends, not just the top.
///
/// kill: scale `SECONDS_PER_ENCODER_FRAME` by 2 (the ceiling catches it) or by
/// 0.5 (the SPAN catches it). Measured: with only the ceiling, a uniform 0.5×
/// compression of the whole timeline passed — every word stayed non-negative,
/// positive-duration, in order and under the bound, and the entire transcript
/// sat in the first half of the audio.
///
/// Also kill: drop the `frame_base` advance in `NemotronStream::feed`, so every
/// chunk's words restart at zero. Measured red — but by a PANIC inside
/// `words_of` (`last_frame - first_frame` underflowing on a non-monotonic
/// token list), not by the asserts below. That underflow is now a
/// `saturating_sub`, so this kill lands on the ordering assert instead.
#[test]
#[ignore = "needs the Nemotron export and the iOS fixtures"]
fn word_timings_stay_inside_the_audio() {
    let Some(directory) = models("word_timings_stay_inside_the_audio") else {
        return;
    };
    let Some(path) = fixture(
        "word_timings_stay_inside_the_audio",
        "Audio/animal-evolution-edu-reel.mp3",
    ) else {
        return;
    };
    let audio = samples(&path, None);
    let seconds = audio.len() as f64 / f64::from(config::SAMPLE_RATE);
    let result = finish(&directory, audio);
    let words: Vec<_> = result
        .captions
        .iter()
        .flat_map(|caption| &caption.words)
        .collect();
    assert!(!words.is_empty());
    let ceiling = seconds + (config::FLUSH_CHUNKS + 1) as f64 * config::CHUNK_SECONDS;
    for word in &words {
        assert!(word.start_time >= 0.0, "{}", word.text);
        assert!(word.duration > 0.0, "{}", word.text);
        assert!(
            word.end_time() <= ceiling,
            "{} ends at {}",
            word.text,
            word.end_time()
        );
    }
    let starts: Vec<f64> = words.iter().map(|word| word.start_time).collect();
    let mut sorted = starts.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(starts, sorted, "word starts must be in time order");

    // The LOWER bound. The reel is continuous speech, so its last word has to
    // land near the end of the audio; without this a timeline compressed by any
    // factor < 1 satisfies every assertion above.
    let last_end = words.last().expect("words").end_time();
    let slack = (config::FLUSH_CHUNKS + 1) as f64 * config::CHUNK_SECONDS;
    assert!(
        last_end > seconds - slack,
        "the transcript ends at {last_end:.3} s but the audio runs to {seconds:.3} s — the \
         timeline is compressed"
    );
}

/// Live and batch say the same thing, and no committed word is ever revised.
///
/// kill: make `transcript()` answer `committed = words()` with no volatile tail
/// and the newest word is published before its chunk closed — the prefix check
/// below sees it change.
#[test]
#[ignore = "needs the Nemotron export and the iOS fixtures"]
fn live_finish_says_what_the_batch_says_and_never_revises() {
    let Some(directory) = models("live_finish_says_what_the_batch_says_and_never_revises") else {
        return;
    };
    let Some(path) = fixture(
        "live_finish_says_what_the_batch_says_and_never_revises",
        "volkova3.mp3",
    ) else {
        return;
    };
    let audio = samples(&path, Some(20.0));
    let step = (0.25 * f64::from(config::SAMPLE_RATE)) as usize;

    let transcriber = transcriber(&directory);
    let (live, updates) = futures_executor::block_on(async {
        let mut session = transcriber.make_live_session().await.expect("live session");
        let mut updates = Vec::new();
        let mut cursor = 0;
        while cursor < audio.len() {
            cursor = (cursor + step).min(audio.len());
            let prefix = ArrayAudioSampleSource::new(audio[..cursor].to_vec());
            updates.push(session.update(&prefix).await.expect("update"));
        }
        let whole = ArrayAudioSampleSource::new(audio.clone());
        (session.finish(&whole).await.expect("finish"), updates)
    });
    let batch = finish(&directory, audio);

    assert_eq!(live.transcription, batch.transcription);
    assert_eq!(live.language.as_deref(), Some("ru"));

    let mut previous: Vec<String> = Vec::new();
    for (index, transcript) in updates.iter().enumerate() {
        assert!(
            transcript.volatile.len() <= 1,
            "update {index} had more than the newest word open"
        );
        let committed: Vec<String> = transcript
            .committed
            .iter()
            .map(|word| word.text.clone())
            .collect();
        // Kill: shortening the committed list must fail, not compare only the shorter prefix.
        assert!(
            committed.len() >= previous.len(),
            "update {index} lost committed words"
        );
        assert_eq!(
            committed[..previous.len()],
            previous[..],
            "update {index} revised committed words"
        );
        previous = committed;
    }
    assert!(!previous.is_empty());
}

/// Kill: finish without flushing lookahead drops a word already spoken before release.
/// Current Swift NemotronTranscriberTests.testAReleaseAfterTheLastWordKeepsIt.
#[test]
#[ignore = "needs the Nemotron export and the current iOS fixtures"]
fn a_release_after_the_last_word_keeps_it() {
    let directory =
        models("a_release_after_the_last_word_keeps_it").expect("required Nemotron export");
    let path = fixture("a_release_after_the_last_word_keeps_it", "volkova3.mp3")
        .expect("required current Russian take");
    let audio = samples(&path, Some(25.0));
    let rate = config::SAMPLE_RATE as usize;
    let window = (0.3 * rate as f64) as usize;
    let pause = (5 * rate..audio.len().saturating_sub(window) + 1)
        .step_by(window / 6)
        .find(|&offset| {
            let rms = (audio[offset..offset + window]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / window as f32)
                .sqrt();
            20.0 * rms.max(1e-9).log10() < -35.0
        })
        .expect("a pause in the first 25 seconds");
    let pause_seconds = pause as f64 / rate as f64;
    let whole = finish(&directory, audio.clone());
    let whole_words = whole
        .captions
        .iter()
        .flat_map(|caption| &caption.words)
        .collect::<Vec<_>>();
    let before = whole_words
        .iter()
        .filter(|word| word.end_time() <= pause_seconds)
        .count();
    let released = finish(
        &directory,
        audio[..pause + (0.15 * rate as f64) as usize].to_vec(),
    );
    let heard = released
        .captions
        .iter()
        .flat_map(|caption| &caption.words)
        .collect::<Vec<_>>();
    assert!(!heard.is_empty());
    assert!(
        heard.len() >= before,
        "release dropped a word spoken before the pause"
    );
    assert!(
        heard.len() <= whole_words.len(),
        "the released take is a prefix"
    );
    // Punctuation used by the pinned Russian corpus. Keep case and every letter intact.
    let bare = |word: &str| {
        word.trim_matches(|c: char| {
            c.is_ascii_punctuation()
                || ['…', '«', '»', '„', '“', '”', '’', '‘', '—', '–', '‐', '‑'].contains(&c)
        })
        .to_owned()
    };
    assert_eq!(
        heard
            .iter()
            .map(|word| bare(&word.text))
            .collect::<Vec<_>>(),
        whole_words
            .iter()
            .take(heard.len())
            .map(|word| bare(&word.text))
            .collect::<Vec<_>>(),
        "the released take is a prefix of the continuous one"
    );
}
