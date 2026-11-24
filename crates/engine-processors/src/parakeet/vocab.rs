//! The `parakeet` contract's vocabulary (`model.json` `vocabulary`), compiled
//! in: the tokenizer's pieces in id order, a word starting at a piece that
//! begins with `▁` — the export's bytes, which `tests/parakeet_contract.rs`
//! holds to the file it wrote.

use std::collections::HashMap;
use std::sync::OnceLock;

use ferment_engine_core::{EngineError, Result};

pub const VOCAB_JSON: &str = include_str!("vocab.json");

#[derive(Debug)]
pub struct Vocab {
    table: HashMap<i32, String>,
}

impl Vocab {
    pub fn load() -> Result<&'static Self> {
        static CACHE: OnceLock<Vocab> = OnceLock::new();
        if let Some(v) = CACHE.get() {
            return Ok(v);
        }
        let parsed = Self::parse()?;
        Ok(CACHE.get_or_init(|| parsed))
    }

    fn parse() -> Result<Self> {
        let pieces: Vec<String> = serde_json::from_str(VOCAB_JSON)
            .map_err(|e| EngineError::invalid_config(format!("parakeet vocab.json: {e}")))?;
        Ok(Self {
            table: pieces
                .into_iter()
                .enumerate()
                .map(|(id, piece)| (id as i32, piece))
                .collect(),
        })
    }

    pub fn lookup(&self, id: i32) -> Option<&str> {
        self.table.get(&id).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn values(&self) -> impl Iterator<Item = &str> {
        self.table.values().map(String::as_str)
    }

    #[cfg(test)]
    pub(crate) fn from_entries(entries: impl IntoIterator<Item = (i32, &'static str)>) -> Self {
        Self {
            table: entries
                .into_iter()
                .map(|(id, token)| (id, token.to_owned()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_is_every_piece_but_the_blank() {
        let v = Vocab::load().expect("load parakeet vocab");
        assert_eq!(v.len(), crate::parakeet::config::BLANK_TOKEN_ID);
        assert_eq!(v.lookup(0), Some("<unk>"));
    }

    #[test]
    fn vocab_caches_load() {
        let a = Vocab::load().unwrap();
        let b = Vocab::load().unwrap();
        assert!(std::ptr::eq(a, b));
    }
}
