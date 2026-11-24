//! GigaAM's SentencePiece vocabulary, bundled with the crate.
//!
//! `ferment-models-export` writes it (`models/gigaam`, `gigaam_vocab.json`): a
//! flat JSON array of pieces, index == token id, exactly [`BLANK_TOKEN_ID`]
//! entries long — the blank has no piece. The runtime needs per-token pieces
//! (`▁слово` / `сло` / `во`) to rebuild word boundaries and per-word timings;
//! `sp.decode(ids)` collapses that away.

use std::sync::OnceLock;

use ferment_engine_core::{EngineError, Result};

use super::config::BLANK_TOKEN_ID;

const VOCAB_JSON: &str = include_str!("vocab.json");

/// The 1024 pieces, index == token id.
pub fn pieces() -> Result<&'static [String]> {
    static CACHE: OnceLock<Vec<String>> = OnceLock::new();
    if let Some(cached) = CACHE.get() {
        return Ok(cached.as_slice());
    }
    let parsed: Vec<String> = serde_json::from_str(VOCAB_JSON)
        .map_err(|error| EngineError::invalid_config(format!("gigaam vocab.json: {error}")))?;
    if parsed.len() != BLANK_TOKEN_ID as usize {
        return Err(EngineError::invalid_config(format!(
            "gigaam vocab.json has {} pieces, expected {BLANK_TOKEN_ID}",
            parsed.len()
        )));
    }
    Ok(CACHE.get_or_init(|| parsed).as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vocabulary_is_exactly_the_blank_id_long() {
        let pieces = pieces().expect("gigaam vocab");
        assert_eq!(pieces.len(), 1024);
        assert_eq!(pieces[0], "<unk>", "id 0 is the unknown piece");
        assert_eq!(pieces[3], "\u{2581}", "id 3 is a bare word-start marker");
        assert!(
            pieces.iter().filter(|p| p.starts_with('\u{2581}')).count() > 100,
            "a SentencePiece vocabulary marks word starts with U+2581"
        );
    }

    #[test]
    fn pieces_are_cached() {
        assert!(std::ptr::eq(
            pieces().unwrap().as_ptr(),
            pieces().unwrap().as_ptr()
        ));
    }
}
