//! The VoxLingua107 label set, in the exported checkpoint's own order — the
//! contract's vocabulary (`ferment-models-export`, `models/ambernet/`,
//! written from `model.cfg.train_ds.labels`), committed here byte for byte and
//! pinned to the export by `tests/ambernet_contract.rs`; iOS bundles the same
//! file. Re-checked against the graph's 107 logits at every prediction.

use std::sync::OnceLock;

use ferment_engine_core::{EngineError, Result};

use super::LANGUAGE_COUNT;

pub const VOCAB_JSON: &str = include_str!("ambernet_langid_vocab.json");

/// The 107 language tags, index == logit index.
pub fn languages() -> Result<&'static [String]> {
    static CACHE: OnceLock<Vec<String>> = OnceLock::new();
    if let Some(cached) = CACHE.get() {
        return Ok(cached.as_slice());
    }
    let parsed: Vec<String> = serde_json::from_str(VOCAB_JSON).map_err(|error| {
        EngineError::invalid_config(format!("ambernet_langid_vocab.json: {error}"))
    })?;
    if parsed.len() != LANGUAGE_COUNT {
        return Err(EngineError::invalid_config(format!(
            "ambernet_langid_vocab.json carries {} languages, expected {LANGUAGE_COUNT}",
            parsed.len()
        )));
    }
    Ok(CACHE.get_or_init(|| parsed).as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_label_set_is_the_checkpoint_order() {
        let langs = languages().expect("ambernet langs");
        assert_eq!(langs.len(), LANGUAGE_COUNT);
        // Head, tail and the two legacy tags the router modernises — the
        // order is alphabetical in the checkpoint, so a re-export that
        // reshuffled would move these.
        assert_eq!(&langs[..4], ["ab", "af", "am", "ar"]);
        assert_eq!(langs[LANGUAGE_COUNT - 1], "zh");
        assert_eq!(langs[44], "iw", "the legacy Hebrew tag rides at 44");
        assert_eq!(langs[46], "jw", "the legacy Javanese tag rides at 46");
        assert_eq!(langs[77], "ru");
        assert_eq!(langs[7], "be", "the be↔ru neighbour confusion's tag");
    }

    #[test]
    fn languages_are_cached() {
        assert!(std::ptr::eq(
            languages().unwrap().as_ptr(),
            languages().unwrap().as_ptr()
        ));
    }
}
