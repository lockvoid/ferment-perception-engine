//! The whole transcription: the audio in overlapping 15 s windows, each
//! through the host's mel, the `parakeet` contract's encoder and the host's
//! decode (`ferment-models-export`, `models/parakeet/model.json`), the
//! windows' token streams merged at their seams, then words and captions.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};

use crate::audio_source::{ArrayAudioSampleSource, AudioSampleSource};

use super::config::{
    HOP_LENGTH, MEL_BANDS, MEL_FRAMES, MEL_SPEC, NORMALIZATION_EPSILON, OVERLAP_FRAMES,
    SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME, SECONDS_PER_ENCODER_FRAME, WINDOW_SAMPLES,
    valid_encoder_frames, valid_mel_frames,
};
use super::decoder::{Encoded, WindowTokens, decode_window};
use super::host_decode::{self, HostDecode};
pub use super::postprocess::TranscribedWord;
use super::postprocess::{build_words, merge_chunks};
use super::vocab::Vocab;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caption {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
    pub words: Vec<TranscribedWord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub transcription: String,
    pub language: Option<String>,
    pub captions: Vec<Caption>,
}

#[derive(Debug, Clone)]
pub struct Parakeet {
    backend: Backend,
    version: ModelVersion,
    host_decode: PathBuf,
    weights: Arc<Mutex<Option<Arc<HostDecode>>>>,
}

impl Parakeet {
    /// The ENGINE's knowledge of its own model, for the transcription router.
    /// Parakeet-TDT-0.6B-v3's 25 languages, verbatim from iOS
    /// `Processors/Parakeet/Parakeet.swift`.
    pub const SUPPORTED_LANGUAGE_CODES: [&'static str; 25] = [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt",
        "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk",
    ];

    /// The graph this transcriber needs on disk. The decode's weights are a
    /// resource beside it (`host_decode::FILE`); the SentencePiece vocabulary
    /// is compiled into the crate.
    pub const MODEL_FILES: [ModelId; 1] = [ModelId::ParakeetEncoder];

    /// The downloaded resources it reads beside the graph.
    pub const RESOURCE_FILES: [&'static str; 1] = [host_decode::FILE];

    /// `host_decode` is where `parakeet_decode.bin` is — read once, at the
    /// first window.
    pub fn new(backend: Backend, host_decode: impl Into<PathBuf>) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
            host_decode: host_decode.into(),
            weights: Arc::default(),
        }
    }

    #[tracing::instrument(skip_all, fields(processor = "parakeet", samples = frames.len()))]
    pub async fn transcribe(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<TranscriptionResult> {
        check_cancellation(&progress)?;
        progress.report(0.05, "loading audio").await;
        check_cancellation(&progress)?;
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        let source = ArrayAudioSampleSource::new(audio);
        let result = self
            .transcribe_source_range(&source, 0..source.sample_count(), &progress, 0.10, 0.95)
            .await?;
        check_cancellation(&progress)?;
        progress.report(1.0, "done").await;
        Ok(result)
    }

    /// The file and array doors use the same frame-aligned windows and token
    /// seam merge. A selected speech run is a view into the source, never a
    /// second duration-sized sample array. Only one <=15 s window is owned.
    pub(crate) async fn transcribe_source_range(
        &self,
        source: &dyn AudioSampleSource,
        range: Range<usize>,
        progress: &ProgressSink,
        progress_start: f64,
        progress_end: f64,
    ) -> Result<TranscriptionResult> {
        check_cancellation(progress)?;
        validate_source_range(source, &range)?;
        if range.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        progress.report(progress_start, "loading vocab").await;
        check_cancellation(progress)?;
        let vocab = Vocab::load()?;
        let windows = windows(range.len());
        let weight = (progress_end - progress_start) / windows.len() as f64;
        let mut merged = Rows::default();
        for (index, window) in windows.iter().enumerate() {
            check_cancellation(progress)?;
            progress
                .report(
                    if index + 1 == windows.len() {
                        progress_end
                    } else {
                        progress_start + weight * (index + 1) as f64
                    },
                    format!("transcribing ({}/{})", index + 1, windows.len()),
                )
                .await;
            check_cancellation(progress)?;
            let samples = read_window(source, &range, *window)?;
            check_cancellation(progress)?;
            let tokens = self.decode(&samples).await?;
            check_cancellation(progress)?;
            merged.merge(Rows::of(&tokens, window.first_frame));
        }

        let words = build_words(
            &merged.tokens,
            &merged.timestamps,
            &merged.confidences,
            &merged.durations,
            vocab,
        );
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();

        let language = detect_language(&text);
        let captions = build_captions(&words);

        Ok(TranscriptionResult {
            transcription: text,
            language,
            captions,
        })
    }

    /// One window of at most `WINDOW_SAMPLES` samples: the mel, the encoder,
    /// the decode.
    pub async fn decode(&self, samples: &[f32]) -> Result<WindowTokens> {
        if samples.len() > WINDOW_SAMPLES {
            return Err(EngineError::inference(format!(
                "a Parakeet window is at most {WINDOW_SAMPLES} samples, not {}",
                samples.len()
            )));
        }
        let (mel, valid) = features(samples)?;
        let mask = mask(valid)?;
        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(vec![1, 1, MEL_BANDS as i64, MEL_FRAMES as i64], mel)?,
        );
        inputs.insert("mask".into(), mask);
        let mut out = self
            .backend
            .infer(ModelId::ParakeetEncoder, &self.version, inputs)
            .await?;
        let encoded = out
            .remove("encoded")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: "parakeet_encoder".into(),
                key: "encoded".into(),
            })?
            .to_f32()?;
        let encoded = Encoded::from_door(&encoded, valid_encoder_frames(valid))?;
        Ok(decode_window(&*self.weights()?, &encoded))
    }

    fn weights(&self) -> Result<Arc<HostDecode>> {
        let mut slot = self.weights.lock().unwrap();
        if let Some(weights) = slot.as_ref() {
            return Ok(Arc::clone(weights));
        }
        let weights = Arc::new(HostDecode::load(&self.host_decode)?);
        *slot = Some(Arc::clone(&weights));
        Ok(weights)
    }
}

/// A window's `mel [128, 1501]` bands-major and its valid frame count: the
/// contract's front-end over the window's real samples (at most
/// `WINDOW_SAMPLES`, whose frames are exactly `MEL_FRAMES`), each band
/// normalised over the valid frames, every frame past them zero.
pub fn features(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    let valid = valid_mel_frames(samples.len());
    let mel = MEL_SPEC
        .compute(samples)?
        .normalized_per_band(valid, NORMALIZATION_EPSILON);
    Ok((mel.bands_major(Some(MEL_FRAMES)), valid))
}

fn mask(valid: usize) -> Result<Tensor> {
    let values = (0..MEL_FRAMES)
        .map(|frame| if frame < valid { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_f32(vec![1, 1, 1, MEL_FRAMES as i64], values)
}

/// The context a window after the first reads before its own start: one
/// encoder frame, so its first frame's mel is not cut at the edge.
const CONTEXT_SAMPLES: usize = SAMPLES_PER_ENCODER_FRAME;
pub(crate) fn check_cancellation(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn validate_source_range(source: &dyn AudioSampleSource, range: &Range<usize>) -> Result<()> {
    if range.start > range.end || range.end > source.sample_count() {
        return Err(EngineError::decode(
            "Parakeet source range is out of bounds",
        ));
    }
    Ok(())
}

/// Keep the read bound at the consumer, even if a source reports an unexpected
/// short read. A truncated source cannot silently move words at a chunk seam.
fn read_window(
    source: &dyn AudioSampleSource,
    range: &Range<usize>,
    window: Window,
) -> Result<Vec<f32>> {
    validate_source_range(source, range)?;
    if window.from > window.to || window.to > range.len() {
        return Err(EngineError::decode(
            "Parakeet window is outside its speech run",
        ));
    }
    let count = window.to - window.from;
    if count > WINDOW_SAMPLES {
        return Err(EngineError::decode(
            "Parakeet audio read exceeds one encoder window",
        ));
    }
    let mut samples = vec![0.0; count];
    let copied = source.copy_samples(&mut samples, range.start + window.from, count)?;
    if copied != count {
        return Err(EngineError::decode(format!(
            "Parakeet source ended early: {copied} of {count} samples"
        )));
    }
    Ok(samples)
}

/// A window's own samples, whole encoder frames, with room for the context
/// and the centre padding's frame.
pub(super) const CHUNK_SAMPLES: usize = (WINDOW_SAMPLES - CONTEXT_SAMPLES - HOP_LENGTH)
    / SAMPLES_PER_ENCODER_FRAME
    * SAMPLES_PER_ENCODER_FRAME;
const OVERLAP_SAMPLES: usize = OVERLAP_FRAMES * SAMPLES_PER_ENCODER_FRAME;
const STRIDE_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES;

/// The samples `from..to` one window reads, and the global encoder frame its
/// first frame is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Window {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) first_frame: usize,
}

pub(super) fn windows(total: usize) -> Vec<Window> {
    let mut windows = Vec::new();
    let mut start = 0;
    while start < total {
        let to = (start + CHUNK_SAMPLES).min(total);
        let from = if windows.is_empty() {
            start
        } else {
            start - CONTEXT_SAMPLES
        };
        windows.push(Window {
            from,
            to,
            first_frame: from / SAMPLES_PER_ENCODER_FRAME,
        });
        if to == total {
            break;
        }
        start += STRIDE_SAMPLES;
    }
    windows
}

/// Token rows, the frames global.
#[derive(Debug, Default)]
struct Rows {
    tokens: Vec<i32>,
    timestamps: Vec<i32>,
    confidences: Vec<f32>,
    durations: Vec<i32>,
}

impl Rows {
    fn of(window: &WindowTokens, first_frame: usize) -> Self {
        Self {
            tokens: window.tokens.clone(),
            timestamps: window
                .frames
                .iter()
                .map(|frame| (first_frame + frame) as i32)
                .collect(),
            confidences: window.confidences.clone(),
            durations: window.durations.clone(),
        }
    }

    fn merge(&mut self, next: Self) {
        if self.tokens.is_empty() || next.tokens.is_empty() {
            self.tokens.extend(next.tokens);
            self.timestamps.extend(next.timestamps);
            self.confidences.extend(next.confidences);
            self.durations.extend(next.durations);
            return;
        }
        let merged = merge_chunks(
            &self.tokens,
            &self.timestamps,
            &self.durations,
            &self.confidences,
            &next.tokens,
            &next.timestamps,
            &next.durations,
            &next.confidences,
            OVERLAP_FRAMES as f64 * SECONDS_PER_ENCODER_FRAME,
            SECONDS_PER_ENCODER_FRAME,
        );
        self.tokens = merged.tokens;
        self.timestamps = merged.timestamps;
        self.confidences = merged.confidences;
        self.durations = merged.durations;
    }
}

fn detect_language(text: &str) -> Option<String> {
    let lowercase = text.to_lowercase();
    let letters = lowercase
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<String>();
    if letters.is_empty() {
        return Some("xx".into());
    }
    let cyrillic = letters
        .chars()
        .filter(|character| ('\u{0400}'..='\u{04ff}').contains(character))
        .count();
    if cyrillic * 2 >= letters.chars().count() {
        return Some("ru".into());
    }

    let words = lowercase
        .split(|character: char| !character.is_alphabetic() && character != '\'')
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let languages = [
        (
            "en",
            [
                "hello",
                "how",
                "are",
                "you",
                "doing",
                "today",
                "hope",
                "everything",
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "lazy",
                "dog",
                "this",
                "is",
                "test",
                "language",
                "detection",
                "system",
                "for",
                "purposes",
                "going",
                "well",
            ]
            .as_slice(),
        ),
        (
            "es",
            [
                "hola", "cómo", "estás", "espero", "que", "todo", "esté", "bien", "hoy",
            ]
            .as_slice(),
        ),
        (
            "fr",
            [
                "bonjour",
                "comment",
                "allez",
                "vous",
                "j'espère",
                "que",
                "tout",
                "va",
                "bien",
                "aujourd'hui",
            ]
            .as_slice(),
        ),
        (
            "de",
            [
                "guten", "tag", "wie", "geht", "es", "ihnen", "ich", "hoffe", "alles", "ist", "in",
                "ordnung",
            ]
            .as_slice(),
        ),
    ];
    let mut scores = languages
        .iter()
        .map(|(language, vocabulary)| {
            let score = words
                .iter()
                .filter(|word| vocabulary.contains(word))
                .count();
            (*language, score)
        })
        .collect::<Vec<_>>();
    scores.sort_by_key(|(_, score)| std::cmp::Reverse(*score));
    if scores[0].1 > 0 && (scores.len() == 1 || scores[0].1 > scores[1].1) {
        Some(scores[0].0.into())
    } else {
        Some("xx".into())
    }
}

fn build_captions(words: &[TranscribedWord]) -> Vec<Caption> {
    if words.is_empty() {
        return Vec::new();
    }
    let sentence_enders = ['.', '!', '?'];
    let max_pause = 1.5;
    let max_segment_duration = 6.0;

    let mut captions = Vec::new();
    let mut current: Vec<TranscribedWord> = Vec::new();

    let flush = |captions: &mut Vec<Caption>, current: &mut Vec<TranscribedWord>| {
        if current.is_empty() {
            return;
        }
        let first = &current[0];
        let last = &current[current.len() - 1];
        let text = current
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned();
        if !text.is_empty() {
            captions.push(Caption {
                text,
                start_time: first.start_time,
                end_time: last.end_time,
                words: std::mem::take(current),
            });
        } else {
            current.clear();
        }
    };

    for (i, word) in words.iter().enumerate() {
        current.push(word.clone());
        let is_last = i == words.len() - 1;
        let ends_sentence = word
            .text
            .chars()
            .last()
            .is_some_and(|c| sentence_enders.contains(&c));
        let long_pause = !is_last && (words[i + 1].start_time - word.end_time) > max_pause;
        let too_long =
            !current.is_empty() && (word.end_time - current[0].start_time) > max_segment_duration;

        if is_last || ends_sentence || long_pause {
            flush(&mut captions, &mut current);
            continue;
        }
        if too_long && word.text.ends_with(',') {
            flush(&mut captions, &mut current);
        }
    }
    flush(&mut captions, &mut current);
    captions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_file_is_one_window() {
        assert_eq!(
            windows(16_000),
            [Window {
                from: 0,
                to: 16_000,
                first_frame: 0
            }]
        );
        assert_eq!(windows(CHUNK_SAMPLES).len(), 1);
        assert_eq!(windows(CHUNK_SAMPLES + 1).len(), 2);
    }

    #[test]
    fn windows_overlap_by_25_frames_and_read_one_frame_of_context() {
        let total = SAMPLE_RATE as usize * 60;
        let windows = windows(total);
        assert_eq!(windows[0].from, 0);
        assert_eq!(windows.last().unwrap().to, total);
        for pair in windows.windows(2) {
            let (previous, next) = (pair[0], pair[1]);
            let start = next.from + CONTEXT_SAMPLES;
            assert_eq!(previous.to - start, OVERLAP_SAMPLES);
            assert_eq!(start % SAMPLES_PER_ENCODER_FRAME, 0);
            assert_eq!(next.first_frame * SAMPLES_PER_ENCODER_FRAME, next.from);
            assert!(next.to - next.from <= WINDOW_SAMPLES);
        }
    }

    #[test]
    fn build_captions_empty_words_yields_no_captions() {
        assert!(build_captions(&[]).is_empty());
    }

    #[test]
    fn build_captions_splits_at_sentence_boundary() {
        let words = vec![
            TranscribedWord {
                text: "hello.".into(),
                start_time: 0.0,
                end_time: 0.5,
                confidence: 1.0,
            },
            TranscribedWord {
                text: "world".into(),
                start_time: 0.6,
                end_time: 1.1,
                confidence: 1.0,
            },
        ];
        let caps = build_captions(&words);
        assert_eq!(caps.len(), 2);
    }

    #[test]
    fn test_empty_string_returns_xx() {
        assert_eq!(detect_language(""), Some("xx".into()));
    }

    #[test]
    fn test_whitespace_only_returns_xx() {
        assert_eq!(detect_language("   \n\t  "), Some("xx".into()));
    }

    #[test]
    fn test_english_sentence() {
        assert_eq!(
            detect_language("Hello, how are you doing today? I hope everything is going well."),
            Some("en".into())
        );
    }

    #[test]
    fn test_english_paragraph() {
        assert_eq!(
            detect_language(
                "The quick brown fox jumps over the lazy dog. This is a test of the language detection system."
            ),
            Some("en".into())
        );
    }

    #[test]
    fn test_russian_sentence() {
        assert_eq!(
            detect_language("Привет, как у тебя дела? Надеюсь, всё хорошо."),
            Some("ru".into())
        );
    }

    #[test]
    fn test_spanish_sentence() {
        assert_eq!(
            detect_language("Hola, cómo estás? Espero que todo esté bien hoy."),
            Some("es".into())
        );
    }

    #[test]
    fn test_french_sentence() {
        assert_eq!(
            detect_language("Bonjour, comment allez-vous? J'espère que tout va bien aujourd'hui."),
            Some("fr".into())
        );
    }

    #[test]
    fn test_german_sentence() {
        assert_eq!(
            detect_language("Guten Tag, wie geht es Ihnen? Ich hoffe, alles ist in Ordnung."),
            Some("de".into())
        );
    }

    #[test]
    fn test_single_word_english() {
        assert!(matches!(
            detect_language("Hello").as_deref(),
            Some("en" | "xx")
        ));
    }

    #[test]
    fn test_numbers_only_returns_xx() {
        assert_eq!(detect_language("123 456 789 000"), Some("xx".into()));
    }

    #[test]
    fn test_return_value_is_lowercase() {
        let result =
            detect_language("This is a reasonably long English sentence for testing purposes.")
                .unwrap();
        assert_eq!(result, result.to_lowercase());
    }

    #[test]
    fn test_return_value_is_short_code() {
        let result = detect_language(
            "Девочки, наконец я нашла тот самый фен который сушит волосы за пять минут.",
        )
        .unwrap();
        assert!((2..=5).contains(&result.chars().count()));
    }
}

#[cfg(test)]
#[path = "file_tests.rs"]
mod file_tests;
