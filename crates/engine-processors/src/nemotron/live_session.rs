//! The Nemotron stream behind the engine's live seam.
//!
//! Ported from `Processors/Nemotron/NemotronLiveSession.swift`. `update` feeds
//! every whole chunk the buffer has grown by since the last call; `finish` feeds
//! the tail zero-padded to a chunk and then enough silence for the encoder's
//! lookahead to close over the last word. The only volatile word is the newest
//! one, while a chunk can still add pieces to it.

use async_trait::async_trait;
use ferment_engine_core::{
    AudioSampleSource, LiveTranscript, LiveTranscriptionSession, Result, TranscriptionResult,
};

use crate::CaptionSegmenter;

use super::config;
use super::stream::NemotronStream;

pub struct NemotronLiveSession {
    stream: NemotronStream,
    cursor: usize,
    scratch: Vec<f32>,
}

impl NemotronLiveSession {
    #[must_use]
    pub fn new(stream: NemotronStream) -> Self {
        Self {
            stream,
            cursor: 0,
            scratch: vec![0.0; config::CHUNK_SAMPLES],
        }
    }

    async fn feed_whole_chunks(
        &mut self,
        source: &dyn AudioSampleSource,
        total: usize,
    ) -> Result<()> {
        let chunk = config::CHUNK_SAMPLES;
        while total - self.cursor >= chunk {
            source.copy_samples(&mut self.scratch, self.cursor, chunk)?;
            self.stream.feed(&self.scratch).await?;
            self.cursor += chunk;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl LiveTranscriptionSession for NemotronLiveSession {
    async fn update(&mut self, source: &dyn AudioSampleSource) -> Result<LiveTranscript> {
        let total = source.sample_count();
        self.feed_whole_chunks(source, total).await?;
        Ok(self.stream.transcript())
    }

    async fn finish(&mut self, source: &dyn AudioSampleSource) -> Result<TranscriptionResult> {
        let total = source.sample_count();
        self.feed_whole_chunks(source, total).await?;
        let remaining = total - self.cursor;
        if remaining > 0 {
            source.copy_samples(&mut self.scratch, self.cursor, remaining)?;
            self.scratch[remaining..].fill(0.0);
            self.stream.feed(&self.scratch).await?;
            self.cursor = total;
        }
        let silence = vec![0.0_f32; config::CHUNK_SAMPLES];
        for _ in 0..config::FLUSH_CHUNKS {
            self.stream.feed(&silence).await?;
        }
        let words = self.stream.words();
        let language = self.stream.language();
        Ok(TranscriptionResult {
            transcription: LiveTranscript::text_of(&words),
            language: (!words.is_empty()).then_some(language).flatten(),
            // Segmented, as iOS's `TranscriptionResult` init does — see
            // `NemotronTranscriber::transcribe`.
            captions: CaptionSegmenter::captions(&words),
        })
    }

    async fn cancel(&mut self) {}
}
