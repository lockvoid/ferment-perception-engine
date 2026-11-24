//! Vocab + per-tag thresholds, parsed once at startup. The two tables are
//! half of the heads' contract: `ferment-models-export` writes them from the
//! three training runs beside `models/music_tagger/model.json`, pins them
//! there, and this crate bundles the same bytes (`tests/music_tagger_contract.rs`
//! holds it to that), so names, order and thresholds can never drift from
//! the graph in a production build.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ferment_engine_core::{EngineError, Result};

pub const VOCAB_JSON: &str = include_str!("music_tagger_vocab.json");
pub const THRESHOLDS_JSON: &str = include_str!("music_tagger_thresholds.json");

#[derive(Debug, Clone)]
pub struct MusicTaggerVocab {
    pub genre_tags: Vec<String>,
    pub mood_tags: Vec<String>,
    pub instrument_tags: Vec<String>,
    pub genre_thresholds: Vec<f32>,
    pub mood_thresholds: Vec<f32>,
    pub instrument_thresholds: Vec<f32>,
}

impl MusicTaggerVocab {
    pub fn load() -> Result<&'static Self> {
        static CACHE: OnceLock<MusicTaggerVocab> = OnceLock::new();
        if let Some(v) = CACHE.get() {
            return Ok(v);
        }
        let parsed = Self::parse()?;
        Ok(CACHE.get_or_init(|| parsed))
    }

    fn parse() -> Result<Self> {
        let vocab: BTreeMap<String, Vec<String>> = serde_json::from_str(VOCAB_JSON)
            .map_err(|e| EngineError::invalid_config(format!("vocab.json parse: {e}")))?;
        let thresh: BTreeMap<String, Vec<f32>> = serde_json::from_str(THRESHOLDS_JSON)
            .map_err(|e| EngineError::invalid_config(format!("thresholds.json parse: {e}")))?;

        let genre_tags = take(&vocab, "genre")?;
        let mood_tags = take(&vocab, "mood")?;
        let instrument_tags = take(&vocab, "instrument")?;
        let genre_thresholds = take(&thresh, "genre")?;
        let mood_thresholds = take(&thresh, "mood")?;
        let instrument_thresholds = take(&thresh, "instrument")?;

        if genre_tags.len() != genre_thresholds.len()
            || mood_tags.len() != mood_thresholds.len()
            || instrument_tags.len() != instrument_thresholds.len()
        {
            return Err(EngineError::invalid_config(
                "vocab/thresholds per-axis count mismatch",
            ));
        }

        Ok(Self {
            genre_tags,
            mood_tags,
            instrument_tags,
            genre_thresholds,
            mood_thresholds,
            instrument_thresholds,
        })
    }
}

fn take<T: Clone>(m: &BTreeMap<String, T>, key: &str) -> Result<T> {
    m.get(key)
        .cloned()
        .ok_or_else(|| EngineError::invalid_config(format!("missing axis '{key}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_bundled_vocab() {
        let v = MusicTaggerVocab::load().expect("load vocab");
        assert_eq!(v.genre_tags.len(), 87);
        assert_eq!(v.mood_tags.len(), 8);
        assert_eq!(v.instrument_tags.len(), 13);
        assert_eq!(v.genre_tags.len(), v.genre_thresholds.len());
        assert_eq!(v.mood_tags.len(), v.mood_thresholds.len());
        assert_eq!(v.instrument_tags.len(), v.instrument_thresholds.len());
    }

    #[test]
    fn caches_load_result() {
        let a = MusicTaggerVocab::load().unwrap();
        let b = MusicTaggerVocab::load().unwrap();
        // Same `&'static` reference — proves caching.
        assert!(std::ptr::eq(a, b));
    }
}
