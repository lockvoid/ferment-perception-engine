//! Nemotron streaming-decode tests.
//!
//! The five iOS cases (`NemotronTranscriberTests`) are all REAL and all
//! model-dependent: they run real speech through the real weights and assert a
//! language and a phrase from outside the code. They are ported in
//! `crates/engine-processors/tests/ort_live_speech.rs`, which SKIPS visibly
//! until the export lands. What is gradable without weights lives here: the
//! sidecar contract (raw JSON bytes) and the committed/volatile split, whose
//! source is the iOS test's own invariant — "committed words are never revised,
//! at most one word is open".

use std::collections::BTreeSet;

use super::config;
use super::models::NemotronModels;
use super::stream::{NemotronToken, split_transcript, words_of};

fn write_sidecars(directory: &std::path::Path, metadata: serde_json::Value) {
    let vocabulary: serde_json::Map<String, serde_json::Value> = (0..config::VOCAB_SIZE)
        .map(|id| (id.to_string(), serde_json::Value::String(format!("p{id}"))))
        .collect();
    std::fs::write(
        directory.join(format!("{}.json", config::VOCAB_FILE)),
        serde_json::to_vec(&vocabulary).unwrap(),
    )
    .unwrap();
    std::fs::write(
        directory.join(format!("{}.json", config::METADATA_FILE)),
        serde_json::to_vec(&metadata).unwrap(),
    )
    .unwrap();
}

fn good_metadata() -> serde_json::Value {
    serde_json::json!({
        "chunk_mel_frames": config::CHUNK_MEL_FRAMES,
        "pre_encode_cache": config::PRE_ENCODE_CACHE_FRAMES,
        "vocab_size": config::VOCAB_SIZE,
        "blank_idx": config::BLANK_TOKEN_ID,
        "cache_channel_shape": config::CACHE_CHANNEL_SHAPE,
        "cache_time_shape": config::CACHE_TIME_SHAPE,
        "default_prompt_id": 7,
        "lang_tag_token_ids": [11, 12, 13],
    })
}

/// A re-converted bundle whose shapes moved must be REFUSED, not run under the
/// build's stale constants: the caches are carried between chunks by shape, so
/// a mismatch is silent corruption rather than a crash.
///
/// kill: drop any one clause of the shape comparison in `load_metadata` and the
/// matching case below loads a bundle this build cannot decode.
#[test]
fn a_re_converted_export_that_moved_a_shape_is_refused() {
    let directory = tempdir();
    write_sidecars(directory.path(), good_metadata());
    let models = NemotronModels::load(directory.path()).expect("a matching export loads");
    assert_eq!(models.auto_prompt_id, 7);
    assert_eq!(models.language_tag_token_ids, BTreeSet::from([11, 12, 13]));
    assert_eq!(models.pieces.len(), config::VOCAB_SIZE);
    assert_eq!(models.pieces[5], "p5");

    for (field, value) in [
        ("chunk_mel_frames", serde_json::json!(48)),
        ("pre_encode_cache", serde_json::json!(8)),
        ("vocab_size", serde_json::json!(2828)),
        ("blank_idx", serde_json::json!(2828)),
        ("cache_channel_shape", serde_json::json!([1, 24, 70, 1024])),
        ("cache_time_shape", serde_json::json!([1, 24, 1024, 16])),
    ] {
        let mut metadata = good_metadata();
        metadata[field] = value;
        write_sidecars(directory.path(), metadata);
        assert!(
            NemotronModels::load(directory.path()).is_err(),
            "an export whose {field} moved must be refused"
        );
    }
}

/// The vocabulary is indexed BY ID, not by position: the conversion writes a
/// JSON object, whose key order is not the token order.
///
/// kill: build `pieces` by iterating the map's values in order and every piece
/// after id 9 lands under the wrong token (`"10"` sorts before `"2"`).
#[test]
fn the_vocabulary_is_indexed_by_token_id_not_by_key_order() {
    let directory = tempdir();
    write_sidecars(directory.path(), good_metadata());
    let models = NemotronModels::load(directory.path()).expect("load");
    for id in [0_usize, 2, 10, 999, config::VOCAB_SIZE - 1] {
        assert_eq!(models.pieces[id], format!("p{id}"), "token {id}");
    }
}

/// A vocabulary that is not the export's size is refused: a 2828-piece Latin
/// bundle under the multilingual build would index past the decoder's logits.
///
/// kill: drop the count check and `pieces[13086]` becomes an empty string
/// instead of an error.
#[test]
fn a_vocabulary_of_the_wrong_size_is_refused() {
    let directory = tempdir();
    write_sidecars(directory.path(), good_metadata());
    let short: serde_json::Map<String, serde_json::Value> = (0..2828)
        .map(|id| (id.to_string(), serde_json::Value::String(format!("p{id}"))))
        .collect();
    std::fs::write(
        directory
            .path()
            .join(format!("{}.json", config::VOCAB_FILE)),
        serde_json::to_vec(&short).unwrap(),
    )
    .unwrap();
    assert!(NemotronModels::load(directory.path()).is_err());
}

/// The iOS suite's own live invariant: what is published as COMMITTED is never
/// revised, and at most one word is open.
///
/// A piece without the `\u{2581}` boundary marker extends whatever word is
/// last, and nothing bounds how many chunks later it may arrive — so the last
/// word is open until `finish`, however old its newest token is. iOS bounds it
/// by chunk age; measured on real audio, that rule published `"спорт"` and then
/// turned it into `"спорт,"` (report D21).
///
/// kill: restore the age test (`last_token.frame + ENCODER_FRAMES_PER_CHUNK >=
/// frame_base`) — the trailing-piece case below then commits a word a later
/// chunk still extends.
#[test]
fn the_last_word_is_open_until_a_later_piece_can_no_longer_extend_it() {
    let boundary = '\u{2581}';
    let pieces: Vec<String> = (0..config::VOCAB_SIZE)
        .map(|id| match id {
            1 => format!("{boundary}alpha"),
            2 => format!("{boundary}beta"),
            3 => ",".to_owned(),
            _ => format!("p{id}"),
        })
        .collect();
    let no_language_tags = BTreeSet::new();
    let per_chunk = config::ENCODER_FRAMES_PER_CHUNK;
    let split = |tokens: &[NemotronToken], frame_base: usize| {
        split_transcript(
            words_of(&pieces, &no_language_tags, tokens),
            tokens,
            frame_base,
        )
    };
    let texts = |words: &[ferment_engine_core::TranscribedWord]| {
        words.iter().map(|w| w.text.clone()).collect::<Vec<_>>()
    };

    // "beta" closed several chunks ago — old enough that the age rule would
    // commit it.
    let stale = vec![
        NemotronToken { id: 1, frame: 0 },
        NemotronToken { id: 2, frame: 1 },
    ];
    let before = split(&stale, per_chunk * 6);
    assert_eq!(texts(&before.committed), ["alpha"]);
    assert_eq!(
        texts(&before.volatile),
        ["beta"],
        "the last word stays open"
    );

    // …and a later chunk's trailing comma extends it, which is precisely the
    // revision that must never reach a caller as committed.
    let extended = [
        stale.as_slice(),
        &[NemotronToken {
            id: 3,
            frame: per_chunk * 6,
        }],
    ]
    .concat();
    let after = split(&extended, per_chunk * 7);
    assert_eq!(texts(&after.committed), ["alpha"]);
    assert_eq!(texts(&after.volatile), ["beta,"]);
    assert_eq!(
        texts(&before.committed),
        texts(&after.committed),
        "nothing published as committed may change"
    );

    // At most one word is ever open, and an empty stream opens nothing.
    assert!(before.volatile.len() <= 1 && after.volatile.len() <= 1);
    let empty = split(&[], 0);
    assert!(empty.committed.is_empty() && empty.volatile.is_empty());
}

/// A per-process scratch directory, removed on drop. The crate's own idiom
/// (`engine-media`'s suites) rather than a new dev-dependency.
struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> ScratchDir {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferment-nemotron-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).expect("scratch dir");
    ScratchDir(path)
}
