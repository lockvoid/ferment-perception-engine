use ferment_engine_core::{EngineError, Result};
use sentencepiece_rs::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const MAX_TEXT_BYTES: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GlapLanguage {
    En,
    Ru,
}

/// The verified SONAR SentencePiece model, not a similarly named tokenizer.
pub struct GlapTokenizer(SentencePieceProcessor);

impl GlapTokenizer {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = read_verified(
            path,
            4_852_054,
            "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a",
        )?;
        SentencePieceProcessor::from_serialized_model(&bytes)
            .map(Self)
            .map_err(token_error)
    }

    pub fn encode(&self, text: &str, language: GlapLanguage) -> Result<Vec<i32>> {
        check_text(text)?;
        let prefix = match language {
            GlapLanguage::En => 256047,
            GlapLanguage::Ru => 256147,
        };
        let pieces = self.0.encode_to_ids(text).map_err(token_error)?;
        // Upstream appends EOS before slicing: a truncated sequence can have
        // no EOS. Do not silently substitute a different truncation policy.
        Ok(std::iter::once(prefix)
            .chain(pieces.into_iter().map(|id| id as i32 + 1))
            .chain(std::iter::once(3))
            .take(514)
            .collect())
    }
}

pub struct Siglip2Tokenizer(Tokenizer);

impl Siglip2Tokenizer {
    pub fn from_path(path: &Path) -> Result<Self> {
        let bytes = read_verified(
            path,
            34_363_039,
            "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
        )?;
        let mut tokenizer = Tokenizer::from_bytes(&bytes).map_err(token_error)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(64),
            direction: PaddingDirection::Right,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "<pad>".into(),
            pad_to_multiple_of: None,
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 64,
                ..Default::default()
            }))
            .map_err(token_error)?;
        Ok(Self(tokenizer))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>> {
        check_text(text)?;
        let encoding = self.0.encode(text, true).map_err(token_error)?;
        let ids = encoding.get_ids();
        if ids.len() != 64 {
            return Err(EngineError::invalid_config(
                "SigLIP 2 must pad to 64 tokens",
            ));
        }
        Ok(ids.iter().map(|&id| id as i32).collect())
    }
}

fn read_verified(path: &Path, length: u64, expected_hash: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    if file.metadata()?.len() != length {
        return Err(EngineError::Models("tokenizer length mismatch".into()));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.by_ref().take(length + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != length || format!("{:x}", Sha256::digest(&bytes)) != expected_hash {
        return Err(EngineError::Models("tokenizer hash mismatch".into()));
    }
    Ok(bytes)
}

fn check_text(text: &str) -> Result<()> {
    if text.len() > MAX_TEXT_BYTES {
        Err(EngineError::invalid_config("retrieval text exceeds 1 MiB"))
    } else {
        Ok(())
    }
}

fn token_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::invalid_config(format!("retrieval tokenizer: {error}"))
}
