//! Rank parity with the server, against vectors the server emits from its OWN
//! implementation (`bin/rails crdt:ranks`) — the same fixture iOS pins
//! (`RankVectorsTests.swift`). Ranks are minted on every platform and compared
//! bytewise under `COLLATE "C"`, so they have to come from one rule.

use ferment_utils::rank;

#[derive(serde::Deserialize)]
struct Vectors {
    between: Vec<Pair>,
    #[serde(rename = "appendChain")]
    append_chain: Vec<String>,
    alphabet: String,
}

#[derive(serde::Deserialize)]
struct Pair {
    low: Option<String>,
    high: Option<String>,
    rank: String,
}

fn vectors() -> Vectors {
    let raw = include_str!("fixtures/rank_vectors.json");
    serde_json::from_str(raw).expect("rank_vectors.json decodes")
}

#[test]
fn the_vectors_are_present() {
    let v = vectors();
    assert!(!v.between.is_empty());
    assert!(!v.append_chain.is_empty());
}

#[test]
fn every_between_vector_matches_the_server() {
    for pair in vectors().between {
        let minted = rank::between(pair.low.as_deref(), pair.high.as_deref()).unwrap();
        assert_eq!(
            minted, pair.rank,
            "between({:?}, {:?}) diverged from the server",
            pair.low, pair.high
        );
    }
}

#[test]
fn the_append_chain_matches_the_server() {
    let chain = vectors().append_chain;
    let mut cursor: Option<String> = None;
    for expected in &chain {
        let minted = rank::append(cursor.as_deref()).unwrap();
        assert_eq!(&minted, expected, "append after {cursor:?} diverged");
        cursor = Some(minted);
    }
}

#[test]
fn the_alphabet_is_the_servers() {
    assert_eq!(vectors().alphabet.as_bytes(), rank::ALPHABET);
}
