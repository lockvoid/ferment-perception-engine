//! The track's loudness structure over the bar lattice, computed once at
//! beat detection and carried by BOTH detectors.
//!
//! No hand-tuned thresholds: boundaries come from changepoint detection
//! (optimal partitioning, L2 cost) under the BIC penalty `2·σ̂²·ln n`, with
//! σ̂ estimated from the track's own bar-to-bar jitter (MAD of first
//! differences). A flat techno master therefore segments sensitively while a
//! noisy live recording segments conservatively — same code, no knobs. The
//! constants that remain are physics, not tuning: 1 dB (loudness JND),
//! 2 bars (the shortest musical phrase — anything shorter is an event, not a
//! section), −45 dBFS (digital silence).
//!
//! Port of iOS `Protocols/EnergyStructure.swift` + `BeatAnalysisResult
//! .energyPerBar`.

use std::collections::BTreeSet;

use ferment_engine_core::{
    EnergyEvent, EnergyEventKind, EnergyLevel, EnergySegment, EnergyStructure,
};

/// Loudness just-noticeable difference — plateaus closer than this are one
/// plateau to an ear.
const JND_DB: f64 = 1.0;
/// The shortest musical phrase, in bars: a segment at or under it, sandwiched
/// inside one plateau, is an event.
const PHRASE_BARS: usize = 2;
/// Digital near-silence, dBFS.
const SILENCE_DB: f64 = -45.0;
/// The most segments a structure may answer with — past this the line stops
/// being structure (a 67-minute DJ mix once answered ~90 segments). Merging
/// is weakest-first, so the biggest contrasts survive the coarsening.
const MAX_SEGMENTS: usize = 24;
/// Under this many bars there is no structure to speak of — quartiles over a
/// handful of values are noise.
const MIN_BARS: usize = 8;
/// The energy column's floor and grain: dBFS clamped at −80, one decimal.
const FLOOR_DB: f64 = -80.0;

/// `energy_per_bar` is the beats cook's column: RMS dBFS per bar of the
/// quantized downbeat lattice.
#[must_use]
pub fn detect(energy_per_bar: &[f64]) -> EnergyStructure {
    if energy_per_bar.len() < MIN_BARS {
        return EnergyStructure::default();
    }
    let merged = merge(partition(energy_per_bar), energy_per_bar);
    let (surviving, events) = collapse_events(merged, energy_per_bar);
    EnergyStructure {
        segments: leveled(&surviving),
        events,
    }
}

/// The energy column: RMS per bar over the downbeat lattice, dBFS (0 dB =
/// full scale), floored at −80, one decimal — the wire grain. One home for
/// the arithmetic; every detector that holds samples calls this with its own
/// lattice.
#[must_use]
pub fn energy_per_bar(
    samples: &[f32],
    sample_rate: f64,
    downbeats: &[f64],
    duration: f64,
) -> Vec<f64> {
    if downbeats.is_empty() || samples.is_empty() || sample_rate <= 0.0 {
        return Vec::new();
    }
    let sample_index = |seconds: f64| -> usize {
        let raw = seconds * sample_rate;
        if raw <= 0.0 {
            0
        } else {
            (raw as usize).min(samples.len())
        }
    };
    downbeats
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let end = downbeats.get(k + 1).copied().unwrap_or(duration);
            let lo = sample_index(start);
            let hi = sample_index(end);
            if hi <= lo {
                return FLOOR_DB;
            }
            let sum: f64 = samples[lo..hi]
                .iter()
                .map(|&sample| f64::from(sample) * f64::from(sample))
                .sum();
            let rms = (sum / (hi - lo) as f64).sqrt();
            if rms <= 0.0 {
                return FLOOR_DB;
            }
            rounded((20.0 * rms.log10()).max(FLOOR_DB))
        })
        .collect()
}

/// The same sequential sum, rounding and bar boundaries over bounded source
/// reads. A short read fails; it cannot silently lower a bar's measured energy.
pub(crate) fn energy_per_bar_source(
    source: &dyn ferment_engine_core::AudioSampleSource,
    sample_rate: f64,
    downbeats: &[f64],
    duration: f64,
    check: &dyn Fn() -> ferment_engine_core::Result<()>,
) -> ferment_engine_core::Result<Vec<f64>> {
    if source.sample_count() == 0 || sample_rate <= 0. {
        return Ok(Vec::new());
    }
    let index =
        |seconds: f64| ((seconds * sample_rate).max(0.) as usize).min(source.sample_count());
    let mut buffer = vec![0.; 4096];
    downbeats
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let lo = index(start);
            let hi = index(downbeats.get(k + 1).copied().unwrap_or(duration));
            if hi <= lo {
                return Ok(FLOOR_DB);
            }
            let mut sum = 0.;
            for at in (lo..hi).step_by(buffer.len()) {
                check()?;
                let count = (hi - at).min(buffer.len());
                if source.copy_samples(&mut buffer, at, count)? != count {
                    return Err(ferment_engine_core::EngineError::internal(
                        "Incomplete bar PCM",
                    ));
                }
                for &sample in &buffer[..count] {
                    sum += f64::from(sample) * f64::from(sample);
                }
            }
            let rms = (sum / (hi - lo) as f64).sqrt();
            Ok(if rms <= 0. {
                FLOOR_DB
            } else {
                rounded((20. * rms.log10()).max(FLOOR_DB))
            })
        })
        .collect()
}

/// One piecewise-constant stretch, 0-based inclusive.
#[derive(Debug, Clone, Copy)]
struct Run {
    from: usize,
    to: usize,
    mean: f64,
}

// MARK: - Changepoint partition (optimal partitioning, L2 cost, BIC penalty)

fn partition(bars: &[f64]) -> Vec<Run> {
    let n = bars.len();
    let mut sum = vec![0.0; n + 1];
    let mut sum_sq = vec![0.0; n + 1];
    for (i, &x) in bars.iter().enumerate() {
        sum[i + 1] = sum[i] + x;
        sum_sq[i + 1] = sum_sq[i] + x * x;
    }
    // SSE of bars[a..=b] around its own mean.
    let cost = |a: usize, b: usize| -> f64 {
        let s = sum[b + 1] - sum[a];
        let q = sum_sq[b + 1] - sum_sq[a];
        let m = (b - a + 1) as f64;
        (q - s * s / m).max(0.0)
    };

    let sigma = sigma(bars);
    let beta = 2.0 * sigma * sigma * (n as f64).ln();
    let mut best = vec![0.0; n + 1];
    let mut back = vec![0usize; n + 1];
    best[0] = -beta;
    for t in 1..=n {
        let mut best_value = f64::INFINITY;
        let mut best_split = 0;
        for (s, &head) in best[..t].iter().enumerate() {
            let value = head + cost(s, t - 1) + beta;
            if value < best_value {
                best_value = value;
                best_split = s;
            }
        }
        best[t] = best_value;
        back[t] = best_split;
    }

    let mut runs = Vec::new();
    let mut end = n;
    while end > 0 {
        let start = back[end];
        runs.push(Run {
            from: start,
            to: end - 1,
            mean: (sum[end] - sum[start]) / (end - start) as f64,
        });
        end = start;
    }
    runs.reverse();
    runs
}

/// Robust noise scale of the bar-to-bar jitter: MAD of first differences,
/// scaled to σ (1.4826·MAD/√2). Floored at 0.1 dB — a purely numerical guard
/// for bit-identical bars, an order under anything audible.
fn sigma(bars: &[f64]) -> f64 {
    let diffs = bars
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let center = median(&diffs);
    let mad = median(&diffs.iter().map(|d| (d - center).abs()).collect::<Vec<_>>());
    (1.4826 * mad / std::f64::consts::SQRT_2).max(0.1)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}

// MARK: - Perceptual passes

/// Closest-pair-first merging under two laws at once: a boundary quieter than
/// the JND is inaudible, and past `MAX_SEGMENTS` the WEAKEST real boundaries
/// go too — a line of 40 sections is not a structure. Merging weakest-first is
/// what "the biggest contrasts survive" means; the answer this replaced
/// upstream (re-partition with a doubled penalty until it fits) had no such
/// property and fell off a cliff — a 60-phrase DJ set with 6–13 dB contrast
/// answered ONE segment.
fn merge(mut runs: Vec<Run>, bars: &[f64]) -> Vec<Run> {
    while runs.len() > 1 {
        let mut closest = 0;
        let mut gap = f64::INFINITY;
        for i in 0..runs.len() - 1 {
            let candidate = (runs[i].mean - runs[i + 1].mean).abs();
            if candidate < gap {
                gap = candidate;
                closest = i;
            }
        }
        if !(gap < JND_DB || runs.len() > MAX_SEGMENTS) {
            break;
        }
        runs[closest] = merged(runs[closest], runs[closest + 1], bars, &BTreeSet::new());
        runs.remove(closest + 1);
    }
    runs
}

/// A run no longer than a phrase, sandwiched between two halves of the SAME
/// plateau (their means within the JND), is an event: the halves fuse and the
/// run reports as a peak/dip against the fused mean.
fn collapse_events(mut runs: Vec<Run>, bars: &[f64]) -> (Vec<Run>, Vec<EnergyEvent>) {
    let mut events: Vec<EnergyEvent> = Vec::new();
    // Every bar already reported as an event stays out of every later host
    // mean: folding a collapsed spike back into the plateau it was measured
    // against made identical hits report different depths (+20.0, +19.5,
    // +19.2, +19.0 for four copies of one spike).
    let mut spent: BTreeSet<usize> = BTreeSet::new();
    let mut index = 1;
    while runs.len() >= 2 && index < runs.len() - 1 {
        let run = runs[index];
        let is_event = run.to - run.from < PHRASE_BARS
            && (runs[index - 1].mean - runs[index + 1].mean).abs() < JND_DB;
        if !is_event {
            index += 1;
            continue;
        }
        let mut skipping = spent.clone();
        skipping.extend(run.from..=run.to);
        let host = merged(runs[index - 1], runs[index + 1], bars, &skipping);
        // The bar furthest from the host mean; the first one wins a tie, as
        // the Swift `max(by:)` this ports does. Seeded with the window's own
        // head — folding from NaN would never replace, since `x > NaN` is
        // false and the event would report a NaN depth.
        let window = &bars[run.from..=run.to];
        let extreme = window.iter().skip(1).fold(window[0], |best, &value| {
            if (value - host.mean).abs() > (best - host.mean).abs() {
                value
            } else {
                best
            }
        });
        events.push(EnergyEvent {
            bar: run.from + 1,
            kind: if run.mean > host.mean {
                EnergyEventKind::Peak
            } else {
                EnergyEventKind::Dip
            },
            delta_db: rounded(extreme - host.mean),
        });
        spent.extend(run.from..=run.to);
        runs.splice(index - 1..=index + 1, [host]);
        index = (index - 1).max(1);
    }
    events.sort_by_key(|event| event.bar);
    (runs, events)
}

fn merged(a: Run, b: Run, bars: &[f64], spent: &BTreeSet<usize>) -> Run {
    let mut total = 0.0;
    let mut count = 0usize;
    for (i, &value) in bars.iter().enumerate().take(b.to + 1).skip(a.from) {
        if !spent.contains(&i) {
            total += value;
            count += 1;
        }
    }
    Run {
        from: a.from,
        to: b.to,
        mean: if count > 0 {
            total / count as f64
        } else {
            bars[a.from]
        },
    }
}

/// Levels are headroom below the track's OWN full power — the loudest
/// sustained plateau (a phrase or longer; any segment when none is).
/// Distribution scales (quartiles) collapse when one plateau covers most of
/// the track — a 76-bar hip-hop body once flattened a −43 dB fade into "mid".
/// Headroom bands don't, and their widths are psychoacoustics, not tuning:
/// within 2·JND of full power reads as loud; more than 10 dB under (perceived
/// half loudness) reads as quiet.
fn leveled(runs: &[Run]) -> Vec<EnergySegment> {
    let sustained = runs
        .iter()
        .filter(|run| run.to - run.from + 1 >= PHRASE_BARS && run.mean > SILENCE_DB)
        .collect::<Vec<_>>();
    let reference = if sustained.is_empty() {
        runs.iter().map(|run| run.mean).fold(f64::NAN, f64::max)
    } else {
        sustained
            .iter()
            .map(|run| run.mean)
            .fold(f64::NAN, f64::max)
    };
    let reference = if reference.is_nan() { 0.0 } else { reference };

    runs.iter()
        .map(|run| EnergySegment {
            from_bar: run.from + 1,
            to_bar: run.to + 1,
            level: if run.mean <= SILENCE_DB {
                EnergyLevel::Silence
            } else if run.mean >= reference - 2.0 * JND_DB {
                EnergyLevel::Loud
            } else if run.mean < reference - 10.0 {
                EnergyLevel::Quiet
            } else {
                EnergyLevel::Mid
            },
            mean_db: rounded(run.mean),
        })
        .collect()
}

fn rounded(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(groups: &[(usize, f64)]) -> Vec<f64> {
        groups
            .iter()
            .flat_map(|&(count, db)| std::iter::repeat_n(db, count))
            .collect()
    }

    fn from_bars(structure: &EnergyStructure) -> Vec<usize> {
        structure
            .segments
            .iter()
            .map(|segment| segment.from_bar)
            .collect()
    }

    fn levels(structure: &EnergyStructure) -> Vec<EnergyLevel> {
        structure
            .segments
            .iter()
            .map(|segment| segment.level)
            .collect()
    }

    // MARK: - Degenerate inputs

    #[test]
    fn under_eight_bars_has_no_structure() {
        assert_eq!(detect(&[]), EnergyStructure::default());
        assert_eq!(
            detect(&bars(&[(7, -10.0)])),
            EnergyStructure::default(),
            "seven bars is not a structure"
        );
    }

    #[test]
    fn flat_track_is_one_loud_segment() {
        let structure = detect(&bars(&[(40, -9.5)]));
        assert_eq!(
            structure.segments,
            vec![EnergySegment {
                from_bar: 1,
                to_bar: 40,
                level: EnergyLevel::Loud,
                mean_db: -9.5,
            }],
            "a uniform track sits at its own full power — loud on its own scale"
        );
        assert!(structure.events.is_empty());
    }

    // MARK: - Boundaries

    #[test]
    fn clean_step_splits_at_the_exact_bar() {
        let structure = detect(&bars(&[(20, -20.0), (20, -10.0)]));
        assert_eq!(from_bars(&structure), vec![1, 21]);
        assert_eq!(
            structure
                .segments
                .iter()
                .map(|segment| segment.to_bar)
                .collect::<Vec<_>>(),
            vec![20, 40]
        );
        assert_eq!(
            levels(&structure),
            vec![EnergyLevel::Mid, EnergyLevel::Loud],
            "10 dB under full power is the mid/quiet edge — still mid, not yet quiet"
        );
        assert_eq!(
            structure
                .segments
                .iter()
                .map(|segment| segment.mean_db)
                .collect::<Vec<_>>(),
            vec![-20.0, -10.0]
        );
    }

    #[test]
    fn staircase_keeps_every_audible_step() {
        let structure = detect(&bars(&[(10, -30.0), (10, -20.0), (10, -10.0)]));
        assert_eq!(from_bars(&structure), vec![1, 11, 21]);
        assert_eq!(
            levels(&structure),
            vec![EnergyLevel::Quiet, EnergyLevel::Mid, EnergyLevel::Loud]
        );
        assert!(
            structure.events.is_empty(),
            "a staircase step is a section, never an event"
        );
    }

    #[test]
    fn silence_head_is_its_own_segment() {
        let structure = detect(&bars(&[(4, -80.0), (30, -10.0)]));
        assert_eq!(
            structure.segments.first(),
            Some(&EnergySegment {
                from_bar: 1,
                to_bar: 4,
                level: EnergyLevel::Silence,
                mean_db: -80.0,
            })
        );
        assert_eq!(
            structure.segments.last().map(|segment| segment.level),
            Some(EnergyLevel::Loud),
            "the audible half is the track's own full power — silence doesn't drag the scale"
        );
    }

    // MARK: - The JND law

    #[test]
    fn plateaus_within_one_db_are_one_plateau() {
        let structure = detect(&bars(&[(20, -10.0), (20, -10.5)]));
        assert_eq!(
            structure.segments.len(),
            1,
            "0.5 dB is under the loudness JND — no boundary"
        );
    }

    // MARK: - The phrase floor (events)

    #[test]
    fn single_bar_spike_is_a_peak_event_not_a_section() {
        let mut energy = bars(&[(30, -10.0)]);
        energy[14] = -2.0;
        let structure = detect(&energy);
        assert_eq!(structure.segments.len(), 1);
        assert_eq!(structure.segments[0].to_bar, 30);
        assert_eq!(
            structure.events,
            vec![EnergyEvent {
                bar: 15,
                kind: EnergyEventKind::Peak,
                delta_db: 8.0,
            }]
        );
    }

    #[test]
    fn two_bar_hole_is_a_dip_event() {
        let mut energy = bars(&[(40, -9.0)]);
        energy[19] = -25.0;
        energy[20] = -25.0;
        let structure = detect(&energy);
        assert_eq!(structure.segments.len(), 1);
        assert_eq!(structure.events.len(), 1);
        assert_eq!(structure.events[0].bar, 20);
        assert_eq!(structure.events[0].kind, EnergyEventKind::Dip);
        assert!(
            (structure.events[0].delta_db + 16.0).abs() <= 0.11,
            "delta={}",
            structure.events[0].delta_db
        );
    }

    #[test]
    fn three_bar_break_is_a_section_not_an_event() {
        let mut energy = bars(&[(40, -9.0)]);
        energy[19] = -25.0;
        energy[20] = -25.0;
        energy[21] = -25.0;
        let structure = detect(&energy);
        assert_eq!(
            from_bars(&structure),
            vec![1, 20, 23],
            "three bars exceed the phrase floor — a real breakdown"
        );
        assert!(structure.events.is_empty());
    }

    // MARK: - Track-adaptive sensitivity (no fixed threshold)

    /// A fixed white-noise sample (gauss σ=1, seeded once, no RNG at test
    /// time) — strict ± alternation would be bar-to-bar STRUCTURE, wrecking
    /// the MAD-of-differences noise estimate the law under test relies on.
    const NOISE: [f64; 40] = [
        2.338, -0.663, 0.395, 0.147, 0.835, -1.402, -0.415, -0.751, -1.075, -0.844, -0.512, -0.287,
        -0.907, 0.422, -0.548, -3.198, 1.191, -0.392, -0.743, 0.268, 0.23, 0.053, -0.854, 0.192,
        -1.537, 1.443, -1.266, -0.207, 0.019, 0.219, -0.247, 0.483, -3.635, -0.234, -0.289, -0.564,
        1.401, -1.107, -0.213, -2.167,
    ];

    fn stepped_with_noise(scale: f64) -> Vec<f64> {
        bars(&[(20, -12.0), (20, -9.0)])
            .iter()
            .zip(NOISE)
            .map(|(level, noise)| level + noise * scale)
            .collect()
    }

    #[test]
    fn the_same_step_splits_over_a_calm_floor_and_holds_over_a_loud_one() {
        assert_eq!(
            from_bars(&detect(&stepped_with_noise(0.2))),
            vec![1, 21],
            "a 3 dB step over a ±0.2 dB floor is a section change at the exact bar"
        );
        assert_eq!(
            detect(&stepped_with_noise(6.0)).segments.len(),
            1,
            "the same 3 dB step inside a ±6 dB floor is statistically nothing"
        );
    }

    // MARK: - The ceiling

    /// A 25-minute DJ set: 60 eight-bar phrases cycling four levels 6–13 dB
    /// apart. Every boundary carries near-equal gain, so "re-partition with a
    /// doubled penalty until it fits" fell off a cliff here — 25 segments, one
    /// doubling, then ONE segment covering the whole set. Weakest-first
    /// merging coarsens instead.
    #[test]
    fn a_long_set_of_equally_contrasted_phrases_coarsens_instead_of_collapsing() {
        let levels = [-22.0, -16.0, -9.0, -13.0];
        let set = (0..60)
            .flat_map(|phrase: usize| std::iter::repeat_n(levels[phrase % 4], 8))
            .collect::<Vec<f64>>();
        let structure = detect(&set);

        assert!(
            structure.segments.len() <= 24,
            "{}",
            structure.segments.len()
        );
        assert!(
            structure.segments.len() >= 20,
            "the ceiling coarsens; it must not collapse the set into one plateau ({})",
            structure.segments.len()
        );
        assert_eq!(structure.segments.first().map(|s| s.from_bar), Some(1));
        assert_eq!(
            structure.segments.last().map(|s| s.to_bar),
            Some(480),
            "the segments still cover every bar"
        );
    }

    // MARK: - The event law, both halves

    #[test]
    fn identical_spikes_report_identical_depths() {
        let mut energy = bars(&[(60, -20.0)]);
        for spike in [9, 24, 39, 54] {
            energy[spike] = 0.0;
        }
        let structure = detect(&energy);

        assert_eq!(
            structure
                .events
                .iter()
                .map(|event| event.bar)
                .collect::<Vec<_>>(),
            vec![10, 25, 40, 55]
        );
        assert_eq!(
            structure
                .events
                .iter()
                .map(|event| event.delta_db)
                .collect::<Vec<_>>(),
            vec![20.0, 20.0, 20.0, 20.0],
            "a collapsed spike folded back into its own host reported +20/+19.5/+19.2/+19"
        );
        assert_eq!(
            structure.segments.first().map(|segment| segment.mean_db),
            Some(-20.0),
            "the surviving plateau is the plateau, not the plateau plus its spikes"
        );
    }

    #[test]
    fn a_short_run_between_two_different_plateaus_is_a_section_not_an_event() {
        // The other half of the event law: "sandwiched between two halves of
        // ONE plateau". Here the neighbours are 8 dB apart, so the 2-bar run
        // is the step between them — a section, whatever its length.
        let structure = detect(&bars(&[(12, -24.0), (2, -16.0), (12, -8.0)]));
        assert!(
            structure.events.is_empty(),
            "a step between two plateaus is not a peak"
        );
        assert_eq!(from_bars(&structure), vec![1, 13, 15]);
    }

    #[test]
    fn a_one_bar_segment_is_not_the_tracks_loudness_reference() {
        // The reference is the loudest SUSTAINED plateau. This 1-bar hit sits
        // between two DIFFERENT plateaus, so it survives as a segment (not an
        // event) — and it still must not become the scale everything else is
        // measured against, or the body of the track reads quiet.
        let structure = detect(&bars(&[(12, -24.0), (1, 0.0), (12, -8.0)]));
        assert_eq!(from_bars(&structure), vec![1, 13, 14]);
        assert_eq!(
            levels(&structure),
            vec![EnergyLevel::Quiet, EnergyLevel::Loud, EnergyLevel::Loud],
            "the -8 body is the reference; the 1-bar hit rides along, it does not set the scale"
        );
    }

    #[test]
    fn the_loud_band_is_two_jnd_wide_on_both_sides() {
        // 1.5 dB under the reference is still loud (within 2·JND); 3 dB under
        // is not. Pins the band in both directions.
        assert_eq!(
            levels(&detect(&bars(&[(20, -9.0), (20, -10.5)]))),
            vec![EnergyLevel::Loud, EnergyLevel::Loud]
        );
        assert_eq!(
            levels(&detect(&bars(&[(20, -9.0), (20, -12.0)]))),
            vec![EnergyLevel::Loud, EnergyLevel::Mid]
        );
    }

    // MARK: - The energy column

    /// Loud bar / silent bar / partial tail — the profile the montage agent
    /// reads. 4 s of audio, downbeats at 0/1/2/3: bar 1 full-scale sine, bar 2
    /// silence, bar 3 half-scale, tail (3→4 s) full-scale.
    #[test]
    fn energy_follows_the_bars() {
        let rate = 1000.0;
        let mut samples = vec![0.0f32; 4000];
        for (i, sample) in samples.iter_mut().enumerate().take(1000) {
            *sample = (i as f32 * 0.3).sin();
        }
        for (i, sample) in samples.iter_mut().enumerate().take(3000).skip(2000) {
            *sample = 0.5 * (i as f32 * 0.3).sin();
        }
        for (i, sample) in samples.iter_mut().enumerate().take(4000).skip(3000) {
            *sample = (i as f32 * 0.3).sin();
        }

        let energy = energy_per_bar(&samples, rate, &[0.0, 1.0, 2.0, 3.0], 4.0);

        assert_eq!(energy.len(), 4, "3 full bars + the partial tail");
        assert!(
            (energy[0] + 3.0).abs() <= 0.5,
            "full-scale sine ≈ −3 dBFS, got {}",
            energy[0]
        );
        assert_eq!(energy[1], -80.0, "silence floors at −80");
        assert!(
            (energy[2] + 9.0).abs() <= 0.5,
            "half scale = −6 dB under the sine, got {}",
            energy[2]
        );
        assert!(
            (energy[3] - energy[0]).abs() <= 0.2,
            "tail bar matches bar 1"
        );
    }

    #[test]
    fn energy_degenerate_inputs() {
        assert!(energy_per_bar(&[], 22_050.0, &[0.0, 1.0], 2.0).is_empty());
        assert!(energy_per_bar(&[0.5], 22_050.0, &[], 1.0).is_empty());
        assert!(energy_per_bar(&[0.5], 0.0, &[0.0], 1.0).is_empty());
    }

    /// A downbeat lattice running past the samples must floor, not panic:
    /// `quantized_downbeats` can extend half a period past the last sample.
    #[test]
    fn energy_past_the_end_of_the_samples_floors() {
        let energy = energy_per_bar(&[0.5; 100], 100.0, &[0.0, 1.0, 2.0], 3.0);
        assert_eq!(energy.len(), 3);
        assert_eq!(energy[1], -80.0);
        assert_eq!(energy[2], -80.0);
    }
}
