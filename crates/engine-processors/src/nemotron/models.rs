//! The three graphs and two sidecars of the Nemotron streaming export.
//!
//! Ported from `Processors/Nemotron/NemotronModels.swift`. Feature names are
//! pinned by the export:
//!
//! ```text
//! preprocessor   audio [1, 8960] f32 → mel [1, 128, 57] f32
//! encoder        mel [1, 128, 65] f32, cache_channel [24, 1, 56, 1024] f32,
//!                cache_time [24, 1, 1024, 8] f32, cache_len [1] i32,
//!                prompt_id [1] i32
//!                → encoded [1, 1024, 7] f32,
//!                  cache_channel_out, cache_time_out, cache_len_out
//! decoder_joint  token [1, 1] i32, h_in / c_in [2, 1, 640] f32,
//!                encoder [1, 1024, 1] f32
//!                → logits [1, 1, 1, 13088] f32, h_out / c_out
//!
//! NO `*_length` INPUTS. Every shape here is static, so a length is a number
//! the caller already knows: the chunk is always 8960 samples, the mel always
//! 57 frames with the 57th zeroed, the token always one. The three that were
//! declared here — `audio_length`, `mel_length`, `token_length` — were in
//! neither the shipped iOS row nor the export, and the session refused them
//! by name (`Invalid input name: audio_length`, 2026-09-23).
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use ferment_engine_core::{EngineError, Result};
use serde::Deserialize;

use super::config;

pub const PREPROCESSOR_AUDIO_KEY: &str = "audio";
pub const PREPROCESSOR_MEL_KEY: &str = "mel";
pub const ENCODER_MEL_KEY: &str = "mel";
pub const ENCODER_CACHE_CHANNEL_KEY: &str = "cache_channel";
pub const ENCODER_CACHE_TIME_KEY: &str = "cache_time";
pub const ENCODER_CACHE_LENGTH_KEY: &str = "cache_len";
pub const ENCODER_PROMPT_KEY: &str = "prompt_id";
pub const ENCODER_OUTPUT_KEY: &str = "encoded";
pub const ENCODER_CACHE_CHANNEL_OUT_KEY: &str = "cache_channel_out";
pub const ENCODER_CACHE_TIME_OUT_KEY: &str = "cache_time_out";
pub const ENCODER_CACHE_LENGTH_OUT_KEY: &str = "cache_len_out";
pub const DECODER_TOKEN_KEY: &str = "token";
pub const DECODER_HIDDEN_IN_KEY: &str = "h_in";
pub const DECODER_CELL_IN_KEY: &str = "c_in";
pub const DECODER_ENCODER_KEY: &str = "encoder";
pub const DECODER_LOGITS_KEY: &str = "logits";
pub const DECODER_HIDDEN_OUT_KEY: &str = "h_out";
pub const DECODER_CELL_OUT_KEY: &str = "c_out";

/// The export's own sidecars, checked against [`super::config`] so a
/// re-converted bundle cannot run under stale shapes.
#[derive(Debug, Clone)]
pub struct NemotronModels {
    /// SentencePiece pieces, index == token id.
    pub pieces: Vec<String>,
    /// Ids of the `<xx-XX>` language-tag tokens the decoder emits before text.
    pub language_tag_token_ids: BTreeSet<i32>,
    /// The prompt embedding under which the model picks the language itself.
    pub auto_prompt_id: i32,
}

impl NemotronModels {
    /// Read and validate the two sidecars from the models directory — the flat
    /// tree the fleet downloads into, iOS's `modelsDirectory`.
    pub fn load(models_directory: &Path) -> Result<Self> {
        let pieces =
            load_vocabulary(&models_directory.join(format!("{}.json", config::VOCAB_FILE)))?;
        let metadata =
            load_metadata(&models_directory.join(format!("{}.json", config::METADATA_FILE)))?;
        Ok(Self {
            pieces,
            language_tag_token_ids: metadata.lang_tag_token_ids.into_iter().collect(),
            auto_prompt_id: metadata.default_prompt_id,
        })
    }
}

/// The conversion writes `{"<id>": "<piece>"}`; the decoder indexes by id.
fn load_vocabulary(path: &Path) -> Result<Vec<String>> {
    let bytes = read(path)?;
    let table: BTreeMap<String, String> = serde_json::from_slice(&bytes)
        .map_err(|error| EngineError::invalid_config(format!("{}: {error}", path.display())))?;
    if table.len() != config::VOCAB_SIZE {
        return Err(EngineError::invalid_config(format!(
            "{}.json has {} pieces, expected {}",
            config::VOCAB_FILE,
            table.len(),
            config::VOCAB_SIZE
        )));
    }
    let mut pieces = vec![String::new(); config::VOCAB_SIZE];
    for (key, piece) in table {
        let id: usize = key
            .parse()
            .ok()
            .filter(|id| *id < config::VOCAB_SIZE)
            .ok_or_else(|| {
                EngineError::invalid_config(format!(
                    "{}.json carries token id '{key}'",
                    config::VOCAB_FILE
                ))
            })?;
        pieces[id] = piece;
    }
    Ok(pieces)
}

#[derive(Debug, Deserialize)]
struct Metadata {
    chunk_mel_frames: usize,
    pre_encode_cache: usize,
    vocab_size: usize,
    blank_idx: i32,
    cache_channel_shape: Vec<i64>,
    cache_time_shape: Vec<i64>,
    default_prompt_id: i32,
    lang_tag_token_ids: Vec<i32>,
    /// The encoder frames one chunk yields — NeMo's `valid_out_len` for the
    /// tier the export selected. It is 7 for the 560 ms row and 4 for the row
    /// the first export traced by accident, so checking it is what tells a
    /// correctly-tiered bundle from a wrongly-tiered one.
    #[serde(default)]
    encoder_frames_per_chunk: Option<usize>,
}

fn load_metadata(path: &Path) -> Result<Metadata> {
    let bytes = read(path)?;
    let metadata: Metadata = serde_json::from_slice(&bytes)
        .map_err(|error| EngineError::invalid_config(format!("{}: {error}", path.display())))?;
    if metadata.chunk_mel_frames != config::CHUNK_MEL_FRAMES
        || metadata.pre_encode_cache != config::PRE_ENCODE_CACHE_FRAMES
        || metadata.vocab_size != config::VOCAB_SIZE
        || metadata.blank_idx != config::BLANK_TOKEN_ID
        || metadata.cache_channel_shape != config::CACHE_CHANNEL_SHAPE
        || metadata.cache_time_shape != config::CACHE_TIME_SHAPE
        || metadata
            .encoder_frames_per_chunk
            .is_some_and(|frames| frames != config::ENCODER_FRAMES_PER_CHUNK)
    {
        return Err(EngineError::invalid_config(format!(
            "{}.json describes a different Nemotron export than this build",
            config::METADATA_FILE
        )));
    }
    Ok(metadata)
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            EngineError::ModelNotFound(path.display().to_string())
        } else {
            EngineError::Io(error)
        }
    })
}
