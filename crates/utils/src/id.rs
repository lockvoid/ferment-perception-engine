//! The ONE identity module, three forms, mirroring the server's `Ferment::Id`
//! and iOS `FermentUtils/FermentID.swift` bit for bit.
//!
//! - [`long`] — 26 chars, anything the server stores under an `id` column. A
//!   device mints `projects.id` and `elements.id` itself and the server adopts
//!   them — that is what lets an offline device create a project, reference it
//!   from clips and cooks, and push the whole thing later under the id it
//!   already used locally. UUIDv7 bits in Crockford base32: decodes back to a
//!   valid RFC 9562 UUIDv7 (so a `uuid` column stays an option server-side) and
//!   its fixed-length big-endian form keeps lexicographic order ==
//!   chronological order.
//! - [`short`] — 10 chars base62, random. Ids that live inside ONE document and
//!   never join across projects: clip keys, caption/word ids.
//! - [`compact`] — the last 6 of a long, display-only for model-facing
//!   surfaces. The HEAD is the millisecond timestamp, shared by everything
//!   minted in one import; the TAIL is the random half. Never stored:
//!   [`resolve`] is the door-side inverse.
//!
//! Three implementations of this encoding now exist — Ruby
//! (`lib/ferment/id.rb`), Swift (`FermentID.swift`) and this one. The test
//! module pins the same example vectors all three suites pin, so a drift shows
//! up on one shared example rather than on a project that will not save.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::Rng;
use sha2::{Digest, Sha256};

/// Crockford base32 — no I, L, O or U, so an id can be read aloud or retyped
/// without the classic confusions.
const ALPHABET: &[u8; 32] = b"0123456789abcdefghjkmnpqrstvwxyz";

/// 26 × 5 bits = 130, which covers the 128 a UUID carries; the top two bits are
/// always zero, so a minted id never exceeds `7z…`.
pub const LENGTH: usize = 26;

/// Base62 — the full case-sensitive space; a short id is copied, never read
/// aloud, so Crockford's confusion-avoidance buys nothing here.
const SHORT_ALPHABET: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// 62¹⁰ ≈ 8.4 × 10¹⁷ — birthday-paradox collision at 10M ids is ~6 × 10⁻⁹.
pub const SHORT_LENGTH: usize = 10;

/// 62¹¹ exceeds `u64`, which is what caps [`short_seeded_with_length`].
const MAX_SEEDED_SHORT_LENGTH: usize = 10;

pub const COMPACT_LENGTH: usize = 6;

/// The outro slot's clip key. Domain meaning lives with the outro feature; the
/// name is HERE because admission is here: `outro` is a legal id precisely
/// because it is reserved, and a rename that touched only the domain side would
/// silently make the slot unsaveable.
pub const OUTRO_KEY: &str = "outro";

/// The branded outro element that ships inside the app bundle — an id with no
/// `Element` row behind it, which the server admits for the same reason.
pub const FERMENT_OUTRO_ELEMENT_ID: &str = "ferment-outro";

/// Well-known keys that predate (and outlive) the convention. The server's
/// `Ferment::Id::RESERVED` names the same two.
pub const RESERVED: [&str; 2] = [OUTRO_KEY, FERMENT_OUTRO_ELEMENT_ID];

/// A compact tail matched more than one id. Carries the contenders so the door
/// can refuse in its own words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ambiguous {
    pub compact: String,
    pub matches: Vec<String>,
}

impl fmt::Display for Ambiguous {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "compact id '{}' matches {} ids: {}",
            self.compact,
            self.matches.len(),
            self.matches.join(", ")
        )
    }
}

impl std::error::Error for Ambiguous {}

fn decode_digit(character: char) -> Option<u64> {
    let byte = u8::try_from(character).ok()?;
    ALPHABET
        .iter()
        .position(|candidate| *candidate == byte)
        .map(|index| index as u64)
}

// MARK: - Mint

/// A fresh long id stamped with the current millisecond.
#[must_use]
pub fn long() -> String {
    long_at(SystemTime::now())
}

/// [`long`] with the clock supplied, so ordering claims are testable.
#[must_use]
pub fn long_at(now: SystemTime) -> String {
    let milliseconds = now
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as u64);

    // RFC 9562 §5.7: 48-bit big-endian timestamp, version nibble 7, 12 random
    // bits, variant `10`, 62 random bits.
    let mut rng = rand::rng();
    let high = (milliseconds << 16) | (0x7 << 12) | rng.random_range(0..1u64 << 12);
    let low = (0b10 << 62) | rng.random_range(0..1u64 << 62);

    encode(high, low)
}

/// A fresh random short id.
#[must_use]
pub fn short() -> String {
    short_with_length(SHORT_LENGTH)
}

/// A fresh random UUID (RFC 9562 v4), hyphenated lowercase — the DEVICE
/// identity's form, mirroring iOS `UUID().uuidString.lowercased()`. Not a
/// fourth id form for rows: the server treats `X-Device-Id` as an opaque
/// string, and the two platforms should present the same shape for it.
#[must_use]
pub fn device() -> String {
    let mut rng = rand::rng();
    let high = (rng.random::<u64>() & !(0xF << 12)) | (0x4 << 12);
    let low = (rng.random::<u64>() & !(0b11 << 62)) | (0b10 << 62);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        high >> 32,
        (high >> 16) & 0xFFFF,
        high & 0xFFFF,
        low >> 48,
        low & 0xFFFF_FFFF_FFFF
    )
}

/// [`short`] at an explicit length. Pass one only when there is a reason
/// (sub-entity scoped ids, tests).
#[must_use]
pub fn short_with_length(length: usize) -> String {
    let mut rng = rand::rng();
    (0..length)
        .map(|_| char::from(SHORT_ALPHABET[rng.random_range(0..SHORT_ALPHABET.len())]))
        .collect()
}

/// A short id that is a pure function of `seed` — SHA-256, first 8 bytes big
/// endian, mod 62^length, zero-padded.
///
/// This is the server-side mint discipline, mirrored bit for bit: a retried
/// request must land on the id it already minted. Device mints stay random
/// (see [`short`]) because in-process calls cannot lose their result.
#[must_use]
pub fn short_seeded(seed: &str) -> String {
    short_seeded_with_length(SHORT_LENGTH, seed)
}

/// [`short_seeded`] at an explicit length.
///
/// # Panics
/// If `length` exceeds 10 — 62¹¹ does not fit the 64-bit derivation, and
/// silently wrapping would hand back ids that collide across platforms.
#[must_use]
pub fn short_seeded_with_length(length: usize, seed: &str) -> String {
    assert!(
        length <= MAX_SEEDED_SHORT_LENGTH,
        "seeded short caps at {MAX_SEEDED_SHORT_LENGTH} chars (62^{length} exceeds the 64-bit derivation)"
    );

    let digest = Sha256::digest(seed.as_bytes());
    let head = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 always yields 32 bytes"),
    );
    // `length <= 10` was just asserted, so 62^length fits `u64`.
    let mut value = head % 62u64.pow(length as u32);

    let mut characters = vec![SHORT_ALPHABET[0]; length];
    for slot in characters.iter_mut().rev() {
        *slot = SHORT_ALPHABET[(value % 62) as usize];
        value /= 62;
    }
    String::from_utf8(characters).expect("base62 is ASCII")
}

// MARK: - Compact

/// Display-only. A reserved or foreign value passes through — it is already the
/// spelling the reader knows.
#[must_use]
pub fn compact(id: &str) -> &str {
    if is_convention(id) {
        // `is_convention` proved every character is in the ASCII alphabet, so
        // this byte index is a character boundary.
        &id[id.len() - COMPACT_LENGTH..]
    } else {
        id
    }
}

/// The door-side inverse of [`compact`]: a full or reserved id passes through
/// (membership stays the caller's lookup), a tail expands against `among`,
/// ambiguity refuses loudly, anything else is `None` — unknown, refused by the
/// caller in its own words.
///
/// # Errors
/// [`Ambiguous`] when more than one candidate ends with `value`.
pub fn resolve(value: &str, among: &[String]) -> Result<Option<String>, Ambiguous> {
    if is_valid(value) {
        return Ok(Some(value.to_owned()));
    }

    let length = value.chars().count();
    if !(COMPACT_LENGTH..LENGTH).contains(&length)
        || !value
            .chars()
            .all(|character| decode_digit(character).is_some())
    {
        return Ok(None);
    }

    let matches = among
        .iter()
        .filter(|candidate| candidate.ends_with(value))
        .cloned()
        .collect::<Vec<_>>();

    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.into_iter().next()),
        _ => Err(Ambiguous {
            compact: value.to_owned(),
            matches,
        }),
    }
}

// MARK: - Convert

/// The canonical base32 form of a UUID string, in any casing, with or without
/// dashes. `None` when the input is not 32 hex digits.
///
/// Slightly stricter than the Swift twin, which inherits `UInt64(_:radix:)`'s
/// tolerance of a leading `+`. No real UUID reaches that branch, and rejecting
/// it cannot turn a valid id away.
#[must_use]
pub fn of_uuid(uuid: &str) -> Option<String> {
    let hex = uuid
        .chars()
        .filter(|character| *character != '-')
        .collect::<String>()
        .to_lowercase();
    if hex.len() != 32 || !hex.chars().all(|character| character.is_ascii_hexdigit()) {
        return None;
    }

    let high = u64::from_str_radix(&hex[..16], 16).ok()?;
    let low = u64::from_str_radix(&hex[16..], 16).ok()?;
    Some(encode(high, low))
}

/// The dashed lowercase UUID an id encodes. `None` for anything that is not 26
/// in-alphabet characters — notably the reserved names, which are legal ids but
/// carry no UUID.
#[must_use]
pub fn to_uuid(id: &str) -> Option<String> {
    let (high, low) = decode(id)?;
    let hex = format!("{high:016x}{low:016x}");
    Some(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}

/// The 48-bit big-endian millisecond timestamp the id carries up front — mint
/// time, without a round trip through a UUID. `None` for a reserved name.
#[must_use]
pub fn timestamp_ms(id: &str) -> Option<u64> {
    // The timestamp occupies bits 127..80, and 80 == 64 + 16.
    decode(id).map(|(high, _)| high >> 16)
}

/// Whether `id` has the shape [`short`] mints: [`SHORT_LENGTH`] base62
/// characters.
///
/// Shape only — a short id carries no checkable structure, so this cannot say
/// where one came from, only that it could have come from here. Every base62
/// byte is ASCII, so a byte scan of the right count is also a character scan.
#[must_use]
pub fn is_short_shaped(id: &str) -> bool {
    id.len() == SHORT_LENGTH && id.bytes().all(|byte| SHORT_ALPHABET.contains(&byte))
}

// MARK: - Admit

/// Admission: the reserved allowlist OR the convention. Never the pattern
/// alone — callers that skip the allowlist reject the outro slot.
#[must_use]
pub fn is_valid(id: &str) -> bool {
    RESERVED.contains(&id) || is_convention(id)
}

fn is_convention(id: &str) -> bool {
    id.chars().count() == LENGTH
        && id
            .chars()
            .all(|character| decode_digit(character).is_some())
}

// MARK: - Base32

fn encode(mut high: u64, mut low: u64) -> String {
    let mut characters = [ALPHABET[0]; LENGTH];

    // Fill from the least significant end: take five bits, then shift the
    // 128-bit pair right by five, carrying the low five bits of the high word
    // into the top of the low word.
    for slot in characters.iter_mut().rev() {
        *slot = ALPHABET[(low & 0b11111) as usize];
        low = (low >> 5) | (high << 59);
        high >>= 5;
    }

    String::from_utf8(characters.to_vec()).expect("Crockford base32 is ASCII")
}

fn decode(id: &str) -> Option<(u64, u64)> {
    if id.chars().count() != LENGTH {
        return None;
    }

    let mut high: u64 = 0;
    let mut low: u64 = 0;
    for character in id.chars() {
        let value = decode_digit(character)?;

        // The mirror of `encode`: the top five bits of the low word spill into
        // the bottom of the high word.
        high = (high << 5) | (low >> 59);
        low = (low << 5) | value;
    }

    Some((high, low))
}

/// Case for case with `FermentIDTests.swift`, which is itself case for case
/// with `test/unit/lib/ferment/id_test.rb`.
#[cfg(test)]
mod tests {
    #[test]
    fn a_device_id_is_a_lowercase_v4_uuid() {
        let id = super::device();
        assert_eq!(id.len(), 36);
        let groups: Vec<&str> = id.split('-').collect();
        assert_eq!(
            groups.iter().map(|group| group.len()).collect::<Vec<_>>(),
            [8, 4, 4, 4, 12]
        );
        assert!(id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert!(!id.chars().any(|c| c.is_ascii_uppercase()));
        assert!(groups[2].starts_with('4'), "version nibble: {id}");
        assert!(
            matches!(groups[3].as_bytes()[0], b'8' | b'9' | b'a' | b'b'),
            "variant bits: {id}"
        );
        assert_ne!(super::device(), id);
    }

    use std::collections::HashSet;
    use std::time::Duration;

    use super::*;

    /// Six real element ids minted inside one import: the first six characters
    /// are a shared millisecond-bucket timestamp, the tails are all distinct —
    /// the reason compact is a TAIL. The Swift and Ruby suites pin the same six.
    fn dancing() -> Vec<String> {
        [
            "01m0qwrt88f5v8r0abm12yrtys",
            "01m0qwry1kf2h9gmyr6ej97gq5",
            "01m0qws2wkf4xvt8a8y7hrq4pd",
            "01m0qws7wzfs5b8s30dwb3akdy",
            "01m0qwsbkze1c8gfthtjde41nn",
            "01m0qwsfh0f5ks95c2x9wxbgdw",
        ]
        .iter()
        .map(|id| (*id).to_owned())
        .collect()
    }

    mod long_tests {
        use super::*;

        #[test]
        fn mints_26_lowercase_crockford_characters() {
            for _ in 0..100 {
                let id = long();
                assert_eq!(id.chars().count(), LENGTH);
                assert!(
                    id.chars()
                        .all(|character| "0123456789abcdefghjkmnpqrstvwxyz".contains(character)),
                    "{id} left the alphabet"
                );
            }
        }

        #[test]
        fn round_trips_to_a_valid_rfc_9562_uuidv7() {
            let uuid = to_uuid(&long()).expect("a minted long decodes");
            let characters = uuid.chars().collect::<Vec<_>>();
            // Version nibble 7, variant bits 0b10 — the two fields that make it a v7.
            assert_eq!(characters[14], '7', "{uuid} is not version 7");
            assert!("89ab".contains(characters[19]), "{uuid} has a bad variant");
        }

        #[test]
        fn lexicographic_order_of_sequential_ids_is_chronological() {
            // Same-millisecond mints share a timestamp prefix and sort by their
            // random tail, so compare the timestamp segment only — that is the
            // ordering claim.
            let stamps = (0..1_000)
                .map(|_| long().chars().take(10).collect::<String>())
                .collect::<Vec<_>>();
            let mut sorted = stamps.clone();
            sorted.sort();
            assert_eq!(stamps, sorted);
        }

        #[test]
        fn the_timestamp_segment_decodes_to_the_mint_time() {
            let before = SystemTime::now();
            let id = long_at(before + Duration::from_millis(5));
            let expected = before
                .duration_since(UNIX_EPOCH)
                .expect("after the epoch")
                .as_millis() as u64
                + 5;
            assert_eq!(timestamp_ms(&id), Some(expected));
        }

        #[test]
        fn a_pre_epoch_clock_degrades_to_zero_rather_than_panicking() {
            let id = long_at(UNIX_EPOCH - Duration::from_secs(1));
            assert_eq!(timestamp_ms(&id), Some(0));
        }
    }

    mod short_tests {
        use super::*;

        #[test]
        fn mints_10_base62_characters_by_default() {
            let allowed = SHORT_ALPHABET
                .iter()
                .map(|b| char::from(*b))
                .collect::<HashSet<_>>();
            for _ in 0..100 {
                let id = short();
                assert_eq!(id.chars().count(), SHORT_LENGTH);
                assert!(id.chars().all(|character| allowed.contains(&character)));
            }
        }

        #[test]
        fn honours_an_explicit_length() {
            for length in [1, 4, 8, 16] {
                assert_eq!(short_with_length(length).chars().count(), length);
            }
        }

        #[test]
        fn does_not_repeat() {
            let ids = (0..1_000).map(|_| short()).collect::<HashSet<_>>();
            assert_eq!(ids.len(), 1_000);
        }

        #[test]
        fn a_seeded_short_is_deterministic_and_seed_sensitive() {
            assert_eq!(short_seeded("ferment:1"), short_seeded("ferment:1"));
            assert_ne!(short_seeded("ferment:1"), short_seeded("ferment:2"));
        }

        /// SHA-256, first 8 bytes big-endian, mod 62¹⁰, base62 zero-padded.
        /// The Ruby and Swift suites pin the same three.
        #[test]
        fn seeds_the_documented_example_vectors() {
            assert_eq!(short_seeded("ferment"), "h1nT4Qi2DJ");
            assert_eq!(short_seeded("ferment:1"), "q1kVBirBnG");
            assert_eq!(short_seeded("seed:2"), "1qCxxiFVKB");
        }

        #[test]
        fn a_seeded_short_is_zero_padded_to_its_full_length() {
            for length in 1..=MAX_SEEDED_SHORT_LENGTH {
                assert_eq!(
                    short_seeded_with_length(length, "ferment").chars().count(),
                    length
                );
            }
        }

        #[test]
        #[should_panic(expected = "seeded short caps at 10 chars")]
        fn a_seeded_short_refuses_a_length_past_the_64_bit_derivation() {
            let _ = short_seeded_with_length(11, "ferment");
        }
    }

    mod compact_tests {
        use super::*;

        #[test]
        fn is_the_last_six_characters_of_a_long_id() {
            let id = long();
            assert_eq!(compact(&id), &id[id.len() - COMPACT_LENGTH..]);
        }

        #[test]
        fn leaves_reserved_and_foreign_values_untouched() {
            assert_eq!(compact("outro"), "outro");
            assert_eq!(compact("ferment-outro"), "ferment-outro");
            assert_eq!(compact("v1"), "v1");
        }
    }

    mod resolve_tests {
        use super::*;

        #[test]
        fn expands_a_compact_tail_to_its_one_full_id() {
            assert_eq!(
                resolve("2yrtys", &dancing()).unwrap().as_deref(),
                Some("01m0qwrt88f5v8r0abm12yrtys")
            );
        }

        #[test]
        fn accepts_a_longer_tail_too() {
            assert_eq!(
                resolve("6ej97gq5", &dancing()).unwrap().as_deref(),
                Some("01m0qwry1kf2h9gmyr6ej97gq5")
            );
        }

        #[test]
        fn passes_a_full_id_through_untouched_in_or_out_of_the_list() {
            let ids = dancing();
            assert_eq!(resolve(&ids[0], &ids).unwrap().as_ref(), Some(&ids[0]));

            let other = long();
            assert_eq!(resolve(&other, &ids).unwrap().as_ref(), Some(&other));
        }

        #[test]
        fn passes_reserved_names_through() {
            assert_eq!(
                resolve("outro", &dancing()).unwrap().as_deref(),
                Some("outro")
            );
        }

        #[test]
        fn returns_none_for_an_unknown_tail() {
            assert_eq!(resolve("zzzzzz", &dancing()).unwrap(), None);
        }

        #[test]
        fn returns_none_for_values_outside_the_grammar() {
            let ids = dancing();
            assert_eq!(
                resolve("yrtys", &ids).unwrap(),
                None,
                "shorter than compact"
            );
            assert_eq!(resolve("2YRTYS", &ids).unwrap(), None, "uppercase");
            assert_eq!(resolve("", &ids).unwrap(), None);
        }

        #[test]
        fn refuses_an_ambiguous_tail_loudly_naming_the_contenders() {
            let twins = vec![
                format!("{}2yrtys", "a".repeat(20)),
                format!("{}2yrtys", "b".repeat(20)),
            ];
            let error = resolve("2yrtys", &twins).expect_err("twins are ambiguous");
            assert_eq!(error.compact, "2yrtys");
            assert_eq!(
                error.matches.iter().collect::<HashSet<_>>(),
                twins.iter().collect::<HashSet<_>>()
            );
        }
    }

    mod is_valid_tests {
        use super::*;

        #[test]
        fn accepts_minted_ids() {
            for _ in 0..100 {
                assert!(is_valid(&long()));
            }
        }

        #[test]
        fn accepts_the_reserved_allowlist() {
            assert!(is_valid("outro"));
            assert!(is_valid("ferment-outro"));
        }

        #[test]
        fn rejects_uppercase_wrong_length_and_crockford_excluded_letters() {
            let id = long();
            let stem = id.chars().take(LENGTH - 1).collect::<String>();

            assert!(!is_valid(&id.to_uppercase()));
            assert!(!is_valid(&stem));
            assert!(!is_valid(&format!("{id}0")));
            for excluded in ['i', 'l', 'o', 'u'] {
                assert!(
                    !is_valid(&format!("{stem}{excluded}")),
                    "{excluded} is Crockford-excluded"
                );
            }
            assert!(!is_valid(""));
            assert!(!is_valid("proj_01ky9ph4wvf9ab2q7x3m5kjtne"));
        }
    }

    mod cross_platform_vector_tests {
        use super::*;

        /// THE shared example. The Ruby and Swift suites assert the identical
        /// pair — if any two encoders disagree by a bit, all three go red on
        /// this line.
        #[test]
        fn encodes_the_documented_example_vector() {
            assert_eq!(
                of_uuid("01983265-89db-7d2a-8b0b-3ce68b2a5cee").as_deref(),
                Some("01k0s6b2evfmn8p2swwt5jmq7e")
            );
            assert_eq!(
                to_uuid("01k0s6b2evfmn8p2swwt5jmq7e").as_deref(),
                Some("01983265-89db-7d2a-8b0b-3ce68b2a5cee")
            );
        }

        #[test]
        fn compacts_the_documented_example_vector() {
            assert_eq!(compact("01k0s6b2evfmn8p2swwt5jmq7e"), "5jmq7e");
        }

        #[test]
        fn of_uuid_accepts_the_casing_and_punctuation_callers_hand_back() {
            let canonical = "01983265-89db-7d2a-8b0b-3ce68b2a5cee";
            assert_eq!(
                of_uuid(&canonical.to_uppercase()).as_deref(),
                Some("01k0s6b2evfmn8p2swwt5jmq7e")
            );
            assert_eq!(
                of_uuid("0198326589db7d2a8b0b3ce68b2a5cee").as_deref(),
                Some("01k0s6b2evfmn8p2swwt5jmq7e"),
                "dashes are optional"
            );
        }

        #[test]
        fn to_uuid_inverts_of_uuid() {
            for _ in 0..100 {
                let id = long();
                let uuid = to_uuid(&id).expect("a minted long decodes");
                assert_eq!(of_uuid(&uuid).as_deref(), Some(id.as_str()));
            }
        }

        #[test]
        fn rejects_malformed_input_instead_of_guessing() {
            assert_eq!(of_uuid(""), None);
            assert_eq!(of_uuid("not-a-uuid"), None);
            assert_eq!(
                of_uuid("01983265-89db-7d2a-8b0b-3ce68b2a5ce"),
                None,
                "31 hex digits"
            );
            assert_eq!(
                of_uuid("01983265-89db-7d2a-8b0b-3ce68b2a5ceg"),
                None,
                "'g' is not hex"
            );

            assert_eq!(to_uuid(""), None);
            assert_eq!(to_uuid("outro"), None);
            assert_eq!(to_uuid("01k0s6b2evfmn8p2swwt5jmq7"), None, "25 chars");
            assert_eq!(
                to_uuid("01k0s6b2evfmn8p2swwt5jmq7i"),
                None,
                "Crockford-excluded"
            );
            assert_eq!(timestamp_ms("outro"), None);
        }

        /// All-zero and all-one exercise the 64-bit carry seam in both
        /// directions — a shift that dropped the top five bits of the low word
        /// would still pass every random case above.
        #[test]
        fn the_full_128_bit_range_survives_the_round_trip() {
            assert_eq!(
                of_uuid("00000000-0000-0000-0000-000000000000").as_deref(),
                Some("00000000000000000000000000")
            );
            assert_eq!(
                to_uuid("00000000000000000000000000").as_deref(),
                Some("00000000-0000-0000-0000-000000000000")
            );

            let ones = "ffffffff-ffff-ffff-ffff-ffffffffffff";
            let encoded_ones = of_uuid(ones).expect("all-ones encodes");
            assert_eq!(encoded_ones, "7zzzzzzzzzzzzzzzzzzzzzzzzz");
            assert_eq!(to_uuid(&encoded_ones).as_deref(), Some(ones));

            // A value whose set bits sit exactly on the word boundary.
            let seam = "00000000-0000-0001-0000-000000000000";
            let encoded_seam = of_uuid(seam).expect("the seam encodes");
            assert_eq!(to_uuid(&encoded_seam).as_deref(), Some(seam));
        }
    }
}
