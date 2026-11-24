//! On-device streaming transcription on NVIDIA's Nemotron 3.5 ASR
//! (`nemotron-3.5-asr-streaming-0.6b`, multilingual, 560 ms tier): a
//! cache-aware FastConformer encoder with an RNN-T decoder, punctuation and
//! casing in the output, forty locales picked by the model itself under the
//! `auto` prompt.
//!
//! Ported from `Processors/Nemotron/NemotronTranscriber.swift`.
//!
//! This is the dictation lane's model. The caption lane keeps the batch
//! Conformers: on the same Russian speech this one measures about twice
//! GigaAM's word error rate, and it earns its place by emitting text while the
//! microphone is still open, which no full-attention encoder can.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ferment_engine_core::{
    AudioEventSegment, EngineError, LiveSpeechTranscribing, LiveTranscript,
    LiveTranscriptionSession, ProgressSink, ProgressTracker, Result, SpeechTranscribing,
    TranscribedWord, TranscriptionResult,
};
use ferment_engine_infer::{Backend, ModelId};

use super::config;
use super::live_session::NemotronLiveSession;
use super::models::NemotronModels;
use super::stream::NemotronStream;
use crate::CaptionSegmenter;
use crate::audio_source::{ArrayAudioSampleSource, AudioSource};
use crate::parakeet::speech_segment_coalescer::coalesce;

#[derive(Debug, Clone)]
pub struct NemotronTranscriber {
    backend: Backend,
    models_directory: PathBuf,
}

impl NemotronTranscriber {
    /// The graphs this transcriber needs on disk. The vocabulary and the
    /// metadata sidecar are fleet RESOURCES, read from the same directory.
    pub const MODEL_FILES: [ModelId; 3] = [
        ModelId::NemotronPreprocessor,
        ModelId::NemotronEncoder,
        ModelId::NemotronDecoderJoint,
    ];

    #[must_use]
    pub fn new(backend: Backend, models_directory: impl Into<PathBuf>) -> Self {
        Self {
            backend,
            models_directory: models_directory.into(),
        }
    }

    /// A stream over an ALREADY-LOADED model. The sidecars are a 210 KB
    /// vocabulary and a metadata file; iOS loads them once and hands the same
    /// `models` to every session, and re-parsing them per speech region was
    /// this port's own addition.
    fn stream_over(&self, models: NemotronModels) -> NemotronStream {
        let prompt_id = models.auto_prompt_id;
        NemotronStream::new(self.backend.clone(), models, prompt_id)
    }
}

#[async_trait(?Send)]
impl SpeechTranscribing for NemotronTranscriber {
    async fn transcribe(
        &self,
        audio_path: &Path,
        speech_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<TranscriptionResult> {
        let sink = progress.clone();
        let progress = ProgressTracker::new(progress);
        if speech_segments.is_some_and(<[AudioEventSegment]>::is_empty) {
            progress.advance(1.0, "No speech detected").await;
            return Ok(TranscriptionResult {
                transcription: String::new(),
                language: None,
                captions: Vec::new(),
            });
        }

        progress.advance(0.02, "Loading audio...").await;
        let sample_rate = f64::from(config::SAMPLE_RATE);
        let frames = AudioSource::default().decode(audio_path).await?;
        let samples = if frames.sample_rate == config::SAMPLE_RATE {
            frames.samples
        } else {
            ferment_engine_audio::resample_sinc(
                &frames.samples,
                frames.sample_rate,
                config::SAMPLE_RATE,
            )?
        };
        if samples.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        let sample_count = samples.len();

        progress.advance(0.08, "Loading models...").await;
        let models = NemotronModels::load(&self.models_directory)?;

        let regions: Vec<(usize, usize)> = match speech_segments {
            Some(segments) => {
                let duration_seconds = sample_count as f64 / sample_rate;
                coalesce(segments, config::SEGMENT_PAD_SECONDS, duration_seconds)
                    .into_iter()
                    .map(|segment| {
                        (
                            (segment.start_time * sample_rate).max(0.0) as usize,
                            ((segment.end_time() * sample_rate) as usize).min(sample_count),
                        )
                    })
                    .filter(|(start, end)| end > start)
                    .collect()
            }
            None => vec![(0, sample_count)],
        };

        let budget = 0.85 / regions.len().max(1) as f64;
        let mut words: Vec<TranscribedWord> = Vec::new();
        let mut language: Option<String> = None;
        for (start, end) in regions {
            // The house cancellation seam, which parakeet honours the same way:
            // iOS checks `Task.checkCancellation()` per chunk and per encoder
            // frame, and without this a cancelled 4½-minute take decodes to
            // completion because the token only silences progress.
            if sink.is_closed() {
                return Err(EngineError::Cancelled);
            }
            let mut session = NemotronLiveSession::new(self.stream_over(models.clone()));
            let slice = ArrayAudioSampleSource::new(samples[start..end].to_vec());
            let result = session.finish(&slice).await?;
            let offset = start as f64 / sample_rate;
            words.extend(
                result
                    .captions
                    .iter()
                    .flat_map(|caption| &caption.words)
                    .map(|word| TranscribedWord {
                        id: word.id.clone(),
                        text: word.text.clone(),
                        start_time: word.start_time + offset,
                        duration: word.duration,
                        confidence: word.confidence,
                    }),
            );
            if language.is_none() {
                language = result.language;
            }
            progress.advance(budget, "Transcribing...").await;
        }

        progress.advance(0.05, "Done").await;
        Ok(TranscriptionResult {
            language: (!words.is_empty()).then_some(language).flatten(),
            transcription: LiveTranscript::text_of(&words),
            // iOS's `TranscriptionResult.init(words:transcription:language:)`
            // runs `CaptionSegmenter.captions(from:)`, so Nemotron's result
            // carries SEGMENTED captions there — as Parakeet's and GigaAM's do
            // here. One caption spanning the whole take was this port's own.
            captions: CaptionSegmenter::captions(&words),
        })
    }
}

#[async_trait(?Send)]
impl LiveSpeechTranscribing for NemotronTranscriber {
    async fn make_live_session(&self) -> Result<Box<dyn LiveTranscriptionSession>> {
        let models = NemotronModels::load(&self.models_directory)?;
        Ok(Box::new(NemotronLiveSession::new(self.stream_over(models))))
    }
}
