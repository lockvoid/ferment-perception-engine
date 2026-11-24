//! Fractional ranks — the ONE generator, ported from iOS `Rank.swift`, itself
//! a port of the server's `Rankable.rank_between`.
//!
//! Ranks are minted on BOTH sides (this device when it inserts offline, the
//! server on create) and compared bytewise under Postgres `COLLATE "C"`. They
//! must interleave, so they must come from the same rule. Parity is pinned by
//! the vectors the server emits from its own implementation
//! (`bin/rails crdt:ranks` → `tests/fixtures/rank_vectors.json`).
//!
//! The algorithm is plain midpoint bisection over base-62, so ANY string over
//! the alphabet is valid input — there is no format to violate.

/// Ascending base-62: digits → uppercase → lowercase. Matches the server's
/// `Rankable::ALPHABET` and sorts bytewise under `COLLATE "C"`.
pub const ALPHABET: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const MIN_DIGIT: u8 = ALPHABET[0];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RankError {
    #[error("{0} >= {1}")]
    NotAscending(String, String),
    #[error("rank contains '{0}', which is not in the alphabet")]
    InvalidCharacter(char),
}

fn digit_index(byte: u8) -> Option<i64> {
    match byte {
        b'0'..=b'9' => Some(i64::from(byte - b'0')),
        b'A'..=b'Z' => Some(i64::from(byte - b'A') + 10),
        b'a'..=b'z' => Some(i64::from(byte - b'a') + 36),
        _ => None,
    }
}

fn validate(rank: &str) -> Result<(), RankError> {
    match rank.bytes().find(|byte| digit_index(*byte).is_none()) {
        Some(byte) => Err(RankError::InvalidCharacter(char::from(byte))),
        None => Ok(()),
    }
}

/// A rank strictly between `low` and `high`. `None` low means "before
/// everything", `None` high means "after everything".
///
/// The result never ends on the minimum digit — a trailing min digit would
/// leave no room to insert before it later.
pub fn between(low: Option<&str>, high: Option<&str>) -> Result<String, RankError> {
    let low = low.unwrap_or("");
    if let Some(high) = high
        && low >= high
    {
        return Err(RankError::NotAscending(low.to_owned(), high.to_owned()));
    }
    validate(low)?;
    if let Some(high) = high {
        validate(high)?;
    }

    let low_bytes = low.as_bytes();
    let mut high: Option<&[u8]> = high.map(str::as_bytes);
    let mut prefix = Vec::new();
    let mut i = 0usize;
    loop {
        let l = low_bytes
            .get(i)
            .map_or(-1, |byte| digit_index(*byte).expect("validated"));
        // Past the end of `high` (or with no `high` at all) the bound is the
        // top of the alphabet.
        let h = high
            .and_then(|bound| bound.get(i))
            .map_or(62, |byte| digit_index(*byte).expect("validated"));

        if h - l > 1 {
            let mid = (l + h) / 2;

            // Never end on the min digit: nothing can be inserted before such
            // a rank (there is no digit below the minimum), so it strands its
            // neighbour and the dead pair surfaces as an error one insert
            // later. Descend instead — `mid < h` here, so the high bound is
            // satisfied for every continuation.
            if mid == 0 {
                prefix.push(MIN_DIGIT);
                high = None;
                i += 1;
                continue;
            }

            prefix.push(ALPHABET[usize::try_from(mid).expect("0..62")]);
            return Ok(String::from_utf8(prefix).expect("alphabet is ASCII"));
        }

        // No room at this digit: pin low's digit (the min digit once low is
        // exhausted) and descend. The high bound stops constraining only if
        // the pinned digit is STRICTLY below high's — dropping it on an equal
        // digit is what produced ranks above the bound.
        let pinned = if l >= 0 { low_bytes[i] } else { MIN_DIGIT };
        prefix.push(pinned);
        if let Some(bound) = high
            && let Some(byte) = bound.get(i)
            && pinned < *byte
        {
            high = None;
        }
        i += 1;
    }
}

pub fn append(after_last: Option<&str>) -> Result<String, RankError> {
    between(after_last, None)
}

pub fn prepend(before_first: Option<&str>) -> Result<String, RankError> {
    between(None, before_first)
}

/// `count` ascending ranks after `low` (and before `high`, if given).
pub fn n_between(
    low: Option<&str>,
    high: Option<&str>,
    count: usize,
) -> Result<Vec<String>, RankError> {
    let mut out = Vec::with_capacity(count);
    let mut cursor = low.map(ToOwned::to_owned);
    for _ in 0..count {
        let next = between(cursor.as_deref(), high)?;
        out.push(next.clone());
        cursor = Some(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_never_end_on_the_min_digit() {
        for (low, high) in [
            (None, Some("01")),
            (Some("0"), Some("01")),
            (None, Some("001")),
        ] {
            let rank = between(low, high).unwrap();
            assert!(!rank.ends_with('0'), "{rank} ends on the min digit");
            assert!(low.is_none_or(|low| low < rank.as_str()));
            assert!(high.is_none_or(|high| rank.as_str() < high));
        }
    }

    #[test]
    fn not_ascending_and_bad_characters_are_refused() {
        assert!(matches!(
            between(Some("b"), Some("a")),
            Err(RankError::NotAscending(_, _))
        ));
        assert!(matches!(
            between(Some("!"), None),
            Err(RankError::InvalidCharacter('!'))
        ));
    }
}
