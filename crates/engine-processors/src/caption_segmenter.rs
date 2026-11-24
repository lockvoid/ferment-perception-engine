use ferment_engine_core::{Caption, TranscribedWord};
use ferment_utils::id;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, Default)]
pub struct CaptionSegmenter;

impl CaptionSegmenter {
    const SENTENCE_ENDERS: [char; 3] = ['.', '!', '?'];
    const MAX_PAUSE: f64 = 1.5;
    const MAX_DURATION: f64 = 6.0;
    const MAX_CHARS: usize = 60;
    const MIN_DURATION: f64 = 5.0 / 6.0;
    const MIN_SIDE_SHARE: f64 = 0.15;
    const MIN_SPLIT_GAP: f64 = 0.3;

    pub fn captions(words: &[TranscribedWord]) -> Vec<Caption> {
        let runs = Self::hard_runs(words);
        let fitted = runs
            .into_iter()
            .flat_map(|run| Self::fitted(&run))
            .collect::<Vec<_>>();
        Self::merge_flash_runs(fitted)
            .into_iter()
            .map(Self::caption)
            .collect()
    }

    /// Sentence-final punctuation ends a caption UNLESS the period is an
    /// abbreviation's (the stable-ts `ignore_special_periods` guard, mirrored
    /// in the Ruby twin `Captions::Segmenter` in ferment-api): a word starting
    /// with a capital or digit, not ending in `".."`, with fewer than 3
    /// characters outside capitals/digits/periods — `Dr.`, `U.S.`, `No.9`,
    /// `США.` read on; `friend.` ends.
    ///
    /// The character classes are Unicode, as both twins are: Swift tests
    /// `Character.isUppercase`/`isNumber`, and `char::is_uppercase`/
    /// `is_numeric` are the same properties. (Ruby's `[[:digit:]]` is Nd only,
    /// so a word opening with a Nl/No numeral — `Ⅷ.` — reads on here and cuts
    /// there. No transcriber emits those.)
    #[must_use]
    pub fn ends_sentence(text: &str) -> bool {
        if !text.ends_with(Self::SENTENCE_ENDERS) {
            return false;
        }
        let abbreviation = text
            .chars()
            .next()
            .is_some_and(|first| first.is_uppercase() || first.is_numeric())
            && !text.ends_with("..")
            && text
                .chars()
                .filter(|c| !c.is_uppercase() && !c.is_numeric() && *c != '.')
                .count()
                < 3;
        !abbreviation
    }

    fn hard_runs(words: &[TranscribedWord]) -> Vec<Vec<TranscribedWord>> {
        let mut runs = Vec::new();
        let mut current = Vec::new();
        for (index, word) in words.iter().enumerate() {
            current.push(word.clone());
            let long_pause = words
                .get(index + 1)
                .is_some_and(|next| next.start_time - word.end_time() > Self::MAX_PAUSE);
            if Self::ends_sentence(&word.text) || long_pause {
                runs.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            runs.push(current);
        }
        runs
    }

    fn fitted(run: &[TranscribedWord]) -> Vec<Vec<TranscribedWord>> {
        if run.len() < 2 || !Self::exceeds_limits(run) {
            return vec![run.to_vec()];
        }
        let split = Self::best_split_index(run);
        let mut output = Self::fitted(&run[..=split]);
        output.extend(Self::fitted(&run[split + 1..]));
        output
    }

    fn exceeds_limits(run: &[TranscribedWord]) -> bool {
        Self::duration(run) > Self::MAX_DURATION || Self::char_count(run) > Self::MAX_CHARS
    }

    fn best_split_index(run: &[TranscribedWord]) -> usize {
        let total = Self::char_count(run);
        let min_side = (total as f64 * Self::MIN_SIDE_SHARE).ceil() as usize;
        let mut midpoint = (0, usize::MAX);
        let mut clause: Option<(usize, usize)> = None;
        let mut breath: Option<(usize, f64)> = None;
        let mut prefix = 0;

        for index in 0..run.len() - 1 {
            prefix += usize::from(index > 0) + grapheme_count(&run[index].text);
            let suffix = total - prefix - 1;
            let imbalance = prefix.abs_diff(suffix);
            if imbalance < midpoint.1 {
                midpoint = (index, imbalance);
            }
            if prefix < min_side || suffix < min_side {
                continue;
            }
            if run[index].text.ends_with([',', ';', ':', '—'])
                && clause.is_none_or(|(_, best)| imbalance < best)
            {
                clause = Some((index, imbalance));
            }
            let gap = run[index + 1].start_time - run[index].end_time();
            if gap >= Self::MIN_SPLIT_GAP && breath.is_none_or(|(_, best)| gap > best) {
                breath = Some((index, gap));
            }
        }
        clause
            .map(|(index, _)| index)
            .or_else(|| breath.map(|(index, _)| index))
            .unwrap_or(midpoint.0)
    }

    fn merge_flash_runs(runs: Vec<Vec<TranscribedWord>>) -> Vec<Vec<TranscribedWord>> {
        let mut output: Vec<Vec<TranscribedWord>> = Vec::new();
        for run in runs {
            if output
                .last()
                .is_some_and(|previous| Self::should_merge(previous, &run))
            {
                output.last_mut().unwrap().extend(run);
            } else {
                output.push(run);
            }
        }
        output
    }

    fn should_merge(left: &[TranscribedWord], right: &[TranscribedWord]) -> bool {
        if Self::duration(left) >= Self::MIN_DURATION && Self::duration(right) >= Self::MIN_DURATION
        {
            return false;
        }
        if right[0].start_time - left.last().unwrap().end_time() > Self::MAX_PAUSE {
            return false;
        }
        let mut merged = left.to_vec();
        merged.extend_from_slice(right);
        Self::duration(&merged) <= Self::MAX_DURATION
            && Self::char_count(&merged) <= Self::MAX_CHARS
    }

    fn duration(run: &[TranscribedWord]) -> f64 {
        run.last().unwrap().end_time() - run[0].start_time
    }

    fn char_count(run: &[TranscribedWord]) -> usize {
        run.iter()
            .map(|word| grapheme_count(&word.text))
            .sum::<usize>()
            + run.len().saturating_sub(1)
    }

    fn caption(words: Vec<TranscribedWord>) -> Caption {
        Caption {
            id: id::short(),
            text: words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            start_time: words[0].start_time,
            duration: Self::duration(&words),
            words,
        }
    }
}

fn grapheme_count(value: &str) -> usize {
    value.graphemes(true).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard. These ids used to come from a process-global counter
    /// starting at 1, so every app launch reminted `0000000001`, `0000000002`,
    /// ... — two elements captioned in two sessions collided inside one
    /// project document. They are `FermentID.short()` now: 10 base62
    /// characters, random, same as iOS.
    #[test]
    fn caption_ids_are_short_ferment_ids_and_do_not_restart_per_process() {
        let captions = CaptionSegmenter::captions(&[
            word("one", 0.0, 0.4),
            word("two.", 0.5, 0.9),
            word("three", 3.0, 3.4),
            word("four.", 3.5, 3.9),
        ]);
        assert!(
            captions.len() >= 2,
            "need at least two captions to compare ids"
        );

        for caption in &captions {
            assert!(
                ferment_utils::id::is_short_shaped(&caption.id),
                "{} is not a short FermentID",
                caption.id
            );
        }
        let unique = captions
            .iter()
            .map(|caption| caption.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), captions.len(), "caption ids collided");
        assert_ne!(
            captions[0].id, "0000000001",
            "the old sequential counter is back"
        );
    }

    fn word(text: &str, start: f64, end: f64) -> TranscribedWord {
        TranscribedWord {
            id: format!("w{start:.3}"),
            text: text.into(),
            start_time: start,
            duration: end - start,
            confidence: 1.0,
        }
    }

    fn stream(n: usize, text: &str, start: f64, duration: f64, gap: f64) -> Vec<TranscribedWord> {
        (0..n)
            .map(|index| {
                let start = start + index as f64 * (duration + gap);
                word(text, start, start + duration)
            })
            .collect()
    }

    #[test]
    fn test_run_on_speech_without_punctuation_is_capped_at_max_duration() {
        let captions = CaptionSegmenter::captions(&stream(30, "a", 0.0, 0.4, 0.2));
        assert!(captions.len() >= 3);
        assert!(captions.iter().all(|caption| caption.duration <= 6.0));
    }

    #[test]
    fn test_dense_speech_is_capped_by_character_budget() {
        let captions = CaptionSegmenter::captions(&stream(20, "abcdefgh", 0.0, 0.2, 0.02));
        assert!(captions.len() >= 3);
        assert!(
            captions
                .iter()
                .all(|caption| grapheme_count(&caption.text) <= 60)
        );
    }

    #[test]
    fn test_overlong_run_prefers_clause_boundary_split() {
        let mut words = stream(18, "look", 0.0, 0.45, 0.05);
        words[8].text = "look,".into();
        let captions = CaptionSegmenter::captions(&words);
        assert_eq!(captions.len(), 2);
        assert!(captions[0].text.ends_with(','));
        assert!(captions.iter().all(|caption| caption.duration <= 6.0));
    }

    #[test]
    fn test_overlong_run_splits_at_largest_gap() {
        let mut words = stream(8, "go", 0.0, 0.45, 0.05);
        words.extend(stream(8, "go", 4.95, 0.45, 0.05));
        let captions = CaptionSegmenter::captions(&words);
        assert_eq!(captions.len(), 2);
        assert_eq!(captions[0].words.len(), 8);
        assert!((captions[1].start_time - 4.95).abs() < 1e-9);
    }

    #[test]
    fn test_split_never_strands_single_word_fragments() {
        let mut words = vec![word("Yes,", 0.0, 0.3)];
        words.extend(stream(15, "word", 0.35, 0.3, 0.05));
        let captions = CaptionSegmenter::captions(&words);
        assert!(
            captions
                .iter()
                .all(|caption| { grapheme_count(&caption.text) <= 60 && caption.words.len() >= 2 })
        );
    }

    #[test]
    fn test_single_overlong_word_is_emitted_as_is() {
        let captions = CaptionSegmenter::captions(&[word("Aaaaaaaaah", 0.0, 8.0)]);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].duration, 8.0);
    }

    #[test]
    fn test_flash_caption_merges_into_neighbor() {
        let mut words = vec![word("Yeah.", 0.0, 0.4)];
        for (index, text) in ["I", "know", "what", "you", "mean", "okay."]
            .iter()
            .enumerate()
        {
            let start = 0.7 + index as f64 * 0.5;
            words.push(word(text, start, start + 0.45));
        }
        let captions = CaptionSegmenter::captions(&words);
        assert_eq!(captions.len(), 1);
        assert_eq!(captions[0].text, "Yeah. I know what you mean okay.");
    }

    // -- Abbreviation guard (mirrored in the Ruby twin `Captions::Segmenter`,
    //    ferment-api packs/project/app/lib/captions/segmenter.rb; cases from
    //    both twins' suites)

    fn phrase(texts: &[&str]) -> Vec<TranscribedWord> {
        texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let start = index as f64 * 0.5;
                word(text, start, start + 0.45)
            })
            .collect()
    }

    #[test]
    fn test_abbreviation_period_is_not_a_sentence_end() {
        assert_eq!(
            CaptionSegmenter::captions(&phrase(&["the", "U.S.", "economy", "keeps", "growing"]))
                .len(),
            1,
            "U.S. must not end the caption"
        );
        assert_eq!(
            CaptionSegmenter::captions(&phrase(&["ask", "Dr.", "Smith", "today"])).len(),
            1,
            "Dr. must not end the caption"
        );
    }

    /// `Character.isUppercase` on the device is Unicode: «США.» reads on
    /// there, so both twins must read on too — or RU born captions cut
    /// differently from RU recognized captions.
    #[test]
    fn test_the_abbreviation_guard_is_unicode() {
        assert_eq!(
            CaptionSegmenter::captions(&phrase(&["в", "США.", "едут", "туристы"])).len(),
            1,
            "США. must not end the caption"
        );
    }

    #[test]
    fn test_sentence_ending_punctuation_still_ends_a_caption() {
        let mut words = phrase(&["Hello", "there", "friend."]);
        words.extend(stream(3, "more", 1.6, 0.4, 0.1));
        let captions = CaptionSegmenter::captions(&words);
        assert_eq!(captions.len(), 2);
        assert!(captions[0].text.ends_with("friend."));
    }

    /// The guard's four clauses, each isolated: the sentence-ender check, the
    /// capital/digit opener, the `".."` escape hatch, and the three-character
    /// budget. The last rows are the quirk both twins share and neither
    /// documents: a SHORT capitalised word reads on even when it is a real
    /// sentence end, because "Hey." has only two characters outside the
    /// capital/digit/period class.
    #[test]
    fn test_ends_sentence_clause_by_clause() {
        for (text, expected, why) in [
            (
                "friend.",
                true,
                "a lowercase opener is never an abbreviation",
            ),
            ("Hello.", true, "4 characters outside the class"),
            (
                "Word.",
                true,
                "exactly 3 outside the class is not an abbreviation",
            ),
            (
                "Dr.",
                false,
                "capital opener, 1 character outside the class",
            ),
            (
                "U.S.",
                false,
                "capital opener, 0 characters outside the class",
            ),
            ("9.", false, "a digit opener counts like a capital"),
            (
                "No.9",
                false,
                "no sentence-final punctuation - it ends on a digit",
            ),
            ("Yes...", true, "an ellipsis ends the sentence"),
            ("Ok..", true, "the '..' escape hatch fires at exactly two"),
            (
                "Hey.",
                false,
                "the shared quirk: 2 characters outside the class read on",
            ),
            ("word", false, "no sentence-final punctuation at all"),
            ("", false, "the empty string ends nothing"),
            (
                "Wow!",
                true,
                "'o', 'w' and '!' are 3 characters outside the class",
            ),
            (
                "Ok!",
                false,
                "the guard excuses ! and ? too, not only periods",
            ),
        ] {
            assert_eq!(
                CaptionSegmenter::ends_sentence(text),
                expected,
                "{text:?}: {why}"
            );
        }
    }

    #[test]
    fn test_words_preserved_in_order_across_splits() {
        let mut words = vec![
            word("Hello", 0.0, 0.45),
            word("there", 0.5, 0.95),
            word("friend.", 1.0, 1.45),
        ];
        words.extend(stream(20, "beta", 3.5, 0.4, 0.1));
        let captions = CaptionSegmenter::captions(&words);
        let ids = captions
            .iter()
            .flat_map(|caption| caption.words.iter().map(|word| &word.id))
            .collect::<Vec<_>>();
        assert_eq!(ids, words.iter().map(|word| &word.id).collect::<Vec<_>>());
        for caption in captions {
            assert_eq!(
                caption.text,
                caption
                    .words
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            assert_eq!(
                caption.start_time,
                caption.words.first().unwrap().start_time
            );
            let last = caption.words.last().unwrap();
            assert!((caption.end_time() - (last.start_time + last.duration)).abs() < 1e-9);
        }
    }
}
