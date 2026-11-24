//! The transcription router's decision table — a pure function, so the choice
//! is testable without a model on disk.
//!
//! Ported from iOS `Features/Project/Processors/TranscriberRouting.swift`
//! (`88ac816ec`, chip axis removed in `3dd704d80`) minus the Apple lane:
//! desktop has no `SpeechTranscriber`, so the order is GigaAM owns Russian
//! (it beats Parakeet on it by ear) → Parakeet covers its 25 → anything else
//! is honestly unsupported. There is no chip axis anywhere.

use ferment_engine_infer::ModelId;

use crate::gigaam::GigaAm;
use crate::parakeet::Parakeet;

/// VoxLingua107 emits legacy ISO tags verbatim; the modern spellings are what
/// every transcriber speaks.
const LEGACY_TAGS: [(&str, &str); 2] = [("iw", "he"), ("jw", "jv")];

/// How many language hypotheses the verdict walk considers.
const HYPOTHESIS_WALK: usize = 3;

/// The tag reported when there is nothing to route.
const UNDETERMINED: &str = "und";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    GigaAm,
    Parakeet,
    Unsupported { code: String },
}

/// Route one language code. `code` is lowercased and legacy-modernised first.
#[must_use]
pub fn route_code(code: &str) -> Route {
    let lowered = code.to_lowercase();
    let modern = LEGACY_TAGS
        .iter()
        .find_map(|(legacy, modern)| (*legacy == lowered).then_some(*modern))
        .unwrap_or(lowered.as_str());

    if GigaAm::SUPPORTED_LANGUAGE_CODES.contains(&modern) {
        Route::GigaAm
    } else if Parakeet::SUPPORTED_LANGUAGE_CODES.contains(&modern) {
        Route::Parakeet
    } else {
        Route::Unsupported {
            code: modern.to_owned(),
        }
    }
}

/// The verdict-level route: walk the TOP-3 language hypotheses in order and
/// take the first one a transcriber actually supports.
///
/// AmberNet's neighbour confusions — be↔ru, ca↔es, ro/uk noise — otherwise
/// skip real speech (live 2026-08-26: Russian speech IDed «be 0.885 /
/// ru 0.113» lost its captions to a language nobody transcribes). Nothing
/// supported in the top three = an honest `Unsupported` carrying the top
/// hypothesis.
pub fn route<'a>(hypothesis_codes: impl IntoIterator<Item = &'a str>) -> Route {
    let mut top = None;
    for code in hypothesis_codes.into_iter().take(HYPOTHESIS_WALK) {
        let candidate = route_code(code);
        if !matches!(candidate, Route::Unsupported { .. }) {
            return candidate;
        }
        if top.is_none() {
            top = Some(candidate);
        }
    }
    top.unwrap_or(Route::Unsupported {
        code: UNDETERMINED.to_owned(),
    })
}

/// The graphs the routed transcriber needs on disk — with
/// [`resource_files`], what the lead's transcript-model download node ensures
/// before inference starts. The SentencePiece vocabularies are compiled into
/// this crate, so they are not listed anywhere: nothing to fetch.
#[must_use]
pub fn model_files(route: &Route) -> &'static [ModelId] {
    match route {
        Route::GigaAm => &GigaAm::MODEL_FILES,
        Route::Parakeet => &Parakeet::MODEL_FILES,
        Route::Unsupported { .. } => &[],
    }
}

/// The downloaded files the routed transcriber reads beside its graphs:
/// Parakeet's host decode weights.
#[must_use]
pub fn resource_files(route: &Route) -> &'static [&'static str] {
    match route {
        Route::Parakeet => &Parakeet::RESOURCE_FILES,
        Route::GigaAm | Route::Unsupported { .. } => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unsupported(code: &str) -> Route {
        Route::Unsupported {
            code: code.to_owned(),
        }
    }

    /// The hypothesis walk (ruled 2026-08-26): AmberNet's neighbour
    /// confusions must not skip real speech.
    #[test]
    fn the_walk_takes_the_first_supported_of_the_top_three_hypotheses() {
        assert_eq!(
            route(["be", "ru", "uk"]),
            Route::GigaAm,
            "the be→ru live case routes to the Russian transcriber"
        );
        assert_eq!(
            route(["ca", "es"]),
            Route::Parakeet,
            "the ca→es confusion routes to Spanish, which Parakeet speaks"
        );
        assert_eq!(
            route(["ru", "be"]),
            Route::GigaAm,
            "a supported top-1 routes exactly as before"
        );
    }

    #[test]
    fn the_walk_stops_at_three_and_skips_honestly() {
        assert_eq!(
            route(["xx", "yy", "zz", "en"]),
            unsupported("xx"),
            "a supported 4th hypothesis is noise, not a verdict"
        );
        assert_eq!(
            route(std::iter::empty()),
            unsupported("und"),
            "no hypotheses — nothing to route"
        );
    }

    #[test]
    fn russian_always_rides_gigaam() {
        assert_eq!(route_code("ru"), Route::GigaAm);
        assert_eq!(route_code("RU"), Route::GigaAm, "the tag is case-folded");
    }

    #[test]
    fn parakeet_covers_its_twenty_five() {
        assert_eq!(Parakeet::SUPPORTED_LANGUAGE_CODES.len(), 25);
        for code in Parakeet::SUPPORTED_LANGUAGE_CODES {
            let expected = if code == "ru" {
                Route::GigaAm
            } else {
                Route::Parakeet
            };
            assert_eq!(route_code(code), expected, "code {code}");
        }
    }

    #[test]
    fn legacy_tags_are_modernized_before_routing() {
        // Neither `he` nor `jv` is transcribable on desktop, but the tag the
        // refusal carries must be the modern one — that string reaches the UI.
        assert_eq!(route_code("iw"), unsupported("he"));
        assert_eq!(route_code("jw"), unsupported("jv"));
    }

    #[test]
    fn a_language_nobody_speaks_is_honestly_unsupported() {
        assert_eq!(route_code("haw"), unsupported("haw"));
        assert_eq!(route_code(""), unsupported(""));
        assert_eq!(route_code("ja"), unsupported("ja"));
    }

    #[test]
    fn every_routed_model_is_named_and_unsupported_names_nothing() {
        assert_eq!(model_files(&Route::GigaAm).len(), 4);
        assert!(resource_files(&Route::GigaAm).is_empty());
        assert_eq!(model_files(&Route::Parakeet), [ModelId::ParakeetEncoder]);
        assert_eq!(resource_files(&Route::Parakeet), ["parakeet_decode.bin"]);
        assert!(model_files(&unsupported("haw")).is_empty());
        assert!(resource_files(&unsupported("haw")).is_empty());
        for route in [Route::GigaAm, Route::Parakeet] {
            let files = model_files(&route);
            let names = files.iter().map(|id| id.name()).collect::<Vec<_>>();
            assert_eq!(
                names.len(),
                names.iter().collect::<std::collections::HashSet<_>>().len(),
                "{route:?} names a model twice"
            );
        }
    }
}
