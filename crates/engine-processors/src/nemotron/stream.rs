//! The cache-aware streaming decode: one encoder pass per 560 ms chunk with the
//! attention and convolution caches carried between passes, then a greedy RNN-T
//! walk over the chunk's seven encoder frames. Nothing is ever re-encoded and no
//! emitted token is revised, which is what makes every token final the moment
//! it appears.
//!
//! Ported from `Processors/Nemotron/NemotronStream.swift`. Adapted there from
//! FluidAudio's `StreamingNemotronMultilingualAsrManager` (Apache 2.0); the
//! speculative-blank batching, hotword biasing and blank rescue of the
//! reference are not carried over.

use ferment_engine_core::{EngineError, LiveTranscript, Result, TranscribedWord};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

use super::config;
use super::models::{self, NemotronModels};

/// One token the decoder emitted, on the absolute encoder frame it came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NemotronToken {
    pub id: i32,
    pub frame: usize,
}

pub struct NemotronStream {
    backend: Backend,
    version: ModelVersion,
    models: NemotronModels,

    cache_channel: Vec<f32>,
    cache_time: Vec<f32>,
    cache_length: i32,
    hidden: Vec<f32>,
    cell: Vec<f32>,
    /// The previous chunk's last mel frames, bin-major
    /// `[MEL_FEATURES × PRE_ENCODE_CACHE_FRAMES]`; `None` before the first chunk.
    mel_cache: Option<Vec<f32>>,
    last_token: i32,
    frame_base: usize,
    prompt_id: i32,

    pub tokens: Vec<NemotronToken>,
    /// The first `<xx-XX>` tag the decoder emitted, without its brackets.
    pub language_tag: Option<String>,
}

impl NemotronStream {
    pub fn new(backend: Backend, models: NemotronModels, prompt_id: i32) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
            models,
            cache_channel: vec![0.0; element_count(&config::CACHE_CHANNEL_SHAPE)],
            cache_time: vec![0.0; element_count(&config::CACHE_TIME_SHAPE)],
            // One frame of silence in an otherwise empty cache: a zero-length
            // cache slice fails the graph's shape inference on the first chunk.
            cache_length: 1,
            hidden: vec![0.0; config::DECODER_LAYERS * config::DECODER_HIDDEN],
            cell: vec![0.0; config::DECODER_LAYERS * config::DECODER_HIDDEN],
            mel_cache: None,
            last_token: config::BLANK_TOKEN_ID,
            frame_base: 0,
            prompt_id,
            tokens: Vec::new(),
            language_tag: None,
        }
    }

    /// The language the decoder announced, as a bare code (`ru`), or `None`
    /// before it has.
    #[must_use]
    pub fn language(&self) -> Option<String> {
        self.language_tag.as_ref().map(|tag| {
            tag.split(['-', '_'])
                .next()
                .unwrap_or(tag)
                .to_ascii_lowercase()
        })
    }

    /// Exactly one chunk of audio, `CHUNK_SAMPLES` long.
    pub async fn feed(&mut self, chunk: &[f32]) -> Result<()> {
        assert_eq!(chunk.len(), config::CHUNK_SAMPLES);

        let mut preprocessor_inputs = TensorMap::new();
        preprocessor_inputs.insert(
            models::PREPROCESSOR_AUDIO_KEY.into(),
            Tensor::from_f32(vec![1, config::CHUNK_SAMPLES as i64], chunk.to_vec())?,
        );
        let preprocessor_outputs = self
            .backend
            .infer(
                ModelId::NemotronPreprocessor,
                &self.version,
                preprocessor_inputs,
            )
            .await?;
        let mel = preprocessor_outputs
            .get(models::PREPROCESSOR_MEL_KEY)
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::NemotronPreprocessor.name().to_owned(),
                key: models::PREPROCESSOR_MEL_KEY.to_owned(),
            })?;
        // The graph emits `chunk_mel_frames + 1` frames and zeroes the last
        // one, so the valid count is a constant rather than an output: the
        // shapes here are all static.
        let encoder_mel = self.assemble(mel, config::CHUNK_MEL_FRAMES)?;

        let mut encoder_inputs = TensorMap::new();
        encoder_inputs.insert(
            models::ENCODER_MEL_KEY.into(),
            Tensor::from_f32(
                vec![
                    1,
                    config::MEL_FEATURES as i64,
                    config::ENCODER_MEL_FRAMES as i64,
                ],
                encoder_mel,
            )?,
        );
        encoder_inputs.insert(
            models::ENCODER_CACHE_CHANNEL_KEY.into(),
            Tensor::from_f32(
                config::CACHE_CHANNEL_SHAPE.to_vec(),
                self.cache_channel.clone(),
            )?,
        );
        encoder_inputs.insert(
            models::ENCODER_CACHE_TIME_KEY.into(),
            Tensor::from_f32(config::CACHE_TIME_SHAPE.to_vec(), self.cache_time.clone())?,
        );
        encoder_inputs.insert(
            models::ENCODER_CACHE_LENGTH_KEY.into(),
            Tensor::from_i32(vec![1], vec![self.cache_length])?,
        );
        encoder_inputs.insert(
            models::ENCODER_PROMPT_KEY.into(),
            Tensor::from_i32(vec![1], vec![self.prompt_id])?,
        );
        let encoder_outputs = self
            .backend
            .infer(ModelId::NemotronEncoder, &self.version, encoder_inputs)
            .await?;
        let encoded = required(&encoder_outputs, models::ENCODER_OUTPUT_KEY)?;
        let frames = usize::try_from(encoded.shape[2]).map_err(|_| EngineError::InvalidOutput)?;
        let encoded_values = encoded.to_f32()?;
        self.cache_channel =
            required(&encoder_outputs, models::ENCODER_CACHE_CHANNEL_OUT_KEY)?.to_f32()?;
        self.cache_time =
            required(&encoder_outputs, models::ENCODER_CACHE_TIME_OUT_KEY)?.to_f32()?;
        self.cache_length = required(&encoder_outputs, models::ENCODER_CACHE_LENGTH_OUT_KEY)?
            .to_i64()?
            .first()
            .copied()
            .unwrap_or(self.cache_length.into()) as i32;

        self.decode(&encoded_values, frames).await?;
        self.frame_base += frames;
        Ok(())
    }

    /// The tokens as timed words — [`words_of`] over this stream's own state.
    #[must_use]
    pub fn words(&self) -> Vec<TranscribedWord> {
        words_of(
            &self.models.pieces,
            &self.models.language_tag_token_ids,
            &self.tokens,
        )
    }

    /// The committed/volatile split — [`split_transcript`] over this stream's
    /// own state.
    #[must_use]
    pub fn transcript(&self) -> LiveTranscript {
        split_transcript(self.words(), &self.tokens, self.frame_base)
    }

    // MARK: - Steps

    /// `encoder_mel` = the previous chunk's tail, then this chunk's first
    /// `CHUNK_MEL_FRAMES`; the tail of this chunk becomes the next cache.
    fn assemble(&mut self, mel: &Tensor, valid_frames: usize) -> Result<Vec<f32>> {
        if mel.shape.len() != 3
            || usize::try_from(mel.shape[1]).unwrap_or(0) != config::MEL_FEATURES
        {
            return Err(EngineError::InvalidOutput);
        }
        // `shape[2]` is the row STRIDE, not the valid-frame count: NeMo's
        // featurizer masks past `mel_length` with `pad_value = 0`, and a chunk
        // of 8 960 samples comes back as 57 frames of which 56 are audio. Real
        // log-mel here runs about −11…−1, so that zero frame is a +4.5 outlier
        // across all 128 bins — and it is the LAST one, so a tail taken by
        // stride puts it into the next chunk's convolution context.
        let frames = usize::try_from(mel.shape[2]).unwrap_or(0);
        if frames < valid_frames || valid_frames < config::CHUNK_MEL_FRAMES {
            return Err(EngineError::InvalidOutput);
        }
        let source = mel.to_f32()?;
        let cache_frames = config::PRE_ENCODE_CACHE_FRAMES;
        let previous = self.mel_cache.take();
        let mut next = vec![0.0_f32; config::MEL_FEATURES * cache_frames];
        let mut destination = vec![0.0_f32; config::MEL_FEATURES * config::ENCODER_MEL_FRAMES];

        for bin in 0..config::MEL_FEATURES {
            for t in 0..cache_frames {
                destination[bin * config::ENCODER_MEL_FRAMES + t] = previous
                    .as_ref()
                    .map_or(0.0, |cache| cache[bin * cache_frames + t]);
                next[bin * cache_frames + t] =
                    source[bin * frames + (valid_frames - cache_frames + t)];
            }
            for t in 0..config::CHUNK_MEL_FRAMES {
                destination[bin * config::ENCODER_MEL_FRAMES + cache_frames + t] =
                    source[bin * frames + t];
            }
        }
        self.mel_cache = Some(next);
        Ok(destination)
    }

    async fn decode(&mut self, encoded: &[f32], frames: usize) -> Result<()> {
        for t in 0..frames {
            let mut encoder_step = vec![0.0_f32; config::ENCODER_DIM];
            for (channel, value) in encoder_step.iter_mut().enumerate() {
                *value = encoded[channel * frames + t];
            }

            for _ in 0..config::MAX_SYMBOLS_PER_STEP {
                let mut inputs = TensorMap::new();
                inputs.insert(
                    models::DECODER_TOKEN_KEY.into(),
                    Tensor::from_i32(vec![1, 1], vec![self.last_token])?,
                );
                inputs.insert(
                    models::DECODER_HIDDEN_IN_KEY.into(),
                    Tensor::from_f32(
                        vec![
                            config::DECODER_LAYERS as i64,
                            1,
                            config::DECODER_HIDDEN as i64,
                        ],
                        self.hidden.clone(),
                    )?,
                );
                inputs.insert(
                    models::DECODER_CELL_IN_KEY.into(),
                    Tensor::from_f32(
                        vec![
                            config::DECODER_LAYERS as i64,
                            1,
                            config::DECODER_HIDDEN as i64,
                        ],
                        self.cell.clone(),
                    )?,
                );
                inputs.insert(
                    models::DECODER_ENCODER_KEY.into(),
                    Tensor::from_f32(vec![1, config::ENCODER_DIM as i64, 1], encoder_step.clone())?,
                );
                let outputs = self
                    .backend
                    .infer(ModelId::NemotronDecoderJoint, &self.version, inputs)
                    .await?;
                let logits = required(&outputs, models::DECODER_LOGITS_KEY)?.to_f32()?;
                let hidden_out = required(&outputs, models::DECODER_HIDDEN_OUT_KEY)?.to_f32()?;
                let cell_out = required(&outputs, models::DECODER_CELL_OUT_KEY)?.to_f32()?;

                let id = argmax(&logits)?;
                if id == config::BLANK_TOKEN_ID {
                    break;
                }
                self.tokens.push(NemotronToken {
                    id,
                    frame: self.frame_base + t,
                });
                self.last_token = id;
                self.hidden = hidden_out;
                self.cell = cell_out;
                if self.language_tag.is_none() && self.models.language_tag_token_ids.contains(&id) {
                    let piece = &self.models.pieces[id as usize];
                    // `<ru-RU>` without its brackets.
                    self.language_tag = Some(
                        piece
                            .strip_prefix('<')
                            .and_then(|rest| rest.strip_suffix('>'))
                            .unwrap_or(piece)
                            .to_owned(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// The tokens as timed words: a piece carrying the SentencePiece boundary
/// marker opens a word, every other piece extends the current one, and the
/// language tags carry no text.
///
/// A free function, not a method, so the tests grade THIS code rather than a
/// second copy of it written beside them.
#[must_use]
pub fn words_of(
    pieces: &[String],
    language_tag_token_ids: &std::collections::BTreeSet<i32>,
    tokens: &[NemotronToken],
) -> Vec<TranscribedWord> {
    let seconds = config::SECONDS_PER_ENCODER_FRAME;
    let mut words = Vec::new();
    let mut text = String::new();
    let mut first_frame = 0_usize;
    let mut last_frame = 0_usize;

    for token in tokens {
        if language_tag_token_ids.contains(&token.id) {
            continue;
        }
        let mut piece = pieces[token.id as usize].as_str();
        if let Some(stripped) = piece.strip_prefix('\u{2581}') {
            close(&mut words, &mut text, first_frame, last_frame, seconds);
            piece = stripped;
        }
        if text.is_empty() {
            first_frame = token.frame;
        }
        text.push_str(piece);
        last_frame = token.frame;
    }
    close(&mut words, &mut text, first_frame, last_frame, seconds);
    words
}

/// The words split into what will not change and what still might. A word's
/// pieces can straddle a chunk, so the newest word is open while its last piece
/// came out of the most recent chunk; one silent chunk after it, nothing can
/// extend it.
/// The LAST word is always open.
///
/// iOS opens it only while its last piece came out of the most recent chunk
/// (`NemotronStream.swift:152`, `lastToken.frame >= frameBase - encoderFramesPerChunk`),
/// and this port carried that rule until it was caught violating the contract
/// it exists to keep. A SentencePiece piece without the `▁` boundary marker
/// extends whatever word is last, and nothing bounds how many chunks later that
/// piece may arrive — punctuation in particular trails. Measured on
/// `volkova3.mp3` at the 560 ms tier: `"спорт"` was published as committed and
/// a later chunk turned it into `"спорт,"`, which is exactly the revision
/// `LiveTranscript`'s doc forbids and the iOS suite's own live test asserts
/// against.
///
/// Committing one word later costs at most one word of latency. Publishing a
/// word that then changes costs the caller's text rewriting itself under the
/// cursor, so the trade is not close. `finish` commits everything, as before.
///
/// Deviation from iOS (report D21).
#[must_use]
pub fn split_transcript(
    mut words: Vec<TranscribedWord>,
    tokens: &[NemotronToken],
    _frame_base: usize,
) -> LiveTranscript {
    if tokens.is_empty() || words.is_empty() {
        return LiveTranscript {
            committed: words,
            volatile: Vec::new(),
        };
    }
    let open = words.pop().expect("checked above");
    LiveTranscript {
        committed: words,
        volatile: vec![open],
    }
}

fn close(
    words: &mut Vec<TranscribedWord>,
    text: &mut String,
    first_frame: usize,
    last_frame: usize,
    seconds: f64,
) {
    if !text.is_empty() {
        words.push(TranscribedWord {
            id: format!("nemotron-{first_frame}-{}", words.len()),
            text: std::mem::take(text),
            start_time: first_frame as f64 * seconds,
            // `saturating_sub`, not `-`: the tokens a stream emits are
            // monotonic by construction, but `words_of` is public and takes any
            // token list. A subtraction here panics in debug and WRAPS in
            // release, turning a bad token order into a word whose duration is
            // astronomical rather than an obviously wrong one.
            duration: (last_frame.saturating_sub(first_frame) + 1) as f64 * seconds,
            confidence: 1.0,
        });
    }
    text.clear();
}

/// First index of the maximum, matching `vDSP_maxvi`'s first-on-tie answer.
///
/// An empty slice would otherwise answer 0 — a real token id, emitted as a
/// word — so the length is checked rather than assumed.
fn argmax(values: &[f32]) -> Result<i32> {
    if values.is_empty() {
        return Err(EngineError::InvalidOutput);
    }
    let mut best = f32::NEG_INFINITY;
    let mut index = 0_usize;
    for (candidate, value) in values.iter().copied().enumerate() {
        if value > best {
            best = value;
            index = candidate;
        }
    }
    i32::try_from(index).map_err(|_| EngineError::InvalidOutput)
}

fn required<'map>(outputs: &'map TensorMap, key: &str) -> Result<&'map Tensor> {
    outputs
        .get(key)
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: "nemotron".into(),
            key: key.to_owned(),
        })
}

fn element_count(shape: &[i64]) -> usize {
    shape.iter().product::<i64>() as usize
}
