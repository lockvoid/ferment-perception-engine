//! The warp-marker lattice: piecewise-constant tempo segmentation of a beat
//! sequence, in place of a single global fit.
//!
//! Shape (all constants canonical, none tuned on any track):
//!   1. Ornament gate — beats whose IBI sits further than 3·σ̂ from the
//!      running median are ornaments (drum fills, doubled detections); σ̂ is
//!      the track's own noise scale, MAD × 1.4826. Ornaments stay in the RAW
//!      arrays; the lattice never sees them.
//!   2. PELT changepoint detection (Killick 2012) over the cleaned IBI
//!      series: exact segmentation minimizing Σ residual²/σ̂² + BIC·k — the
//!      number of segments is model selection, not a threshold. A steady
//!      track earns exactly one segment because a second never repays its
//!      BIC penalty. Bridge intervals — the gaps left where ornament beats
//!      were removed — carry no tempo evidence and are masked out of the
//!      cost entirely: a segment spans them, so the lattice covers the hole
//!      instead of splitting on it.
//!   3. Per segment the existing machinery: linear-fit period refinement,
//!      circular-mean anchor, downbeat phase by raw-downbeat votes (in
//!      `quantize`).
//!
//! Port of iOS `Processors/BeatThis/BeatLatticeSegmentation.swift`.

use std::collections::BTreeSet;
use std::ops::Range;

/// One constant-tempo stretch, in beat-index space of the CLEANED series.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Range of cleaned-beat indices covered.
    pub beat_range: Range<usize>,
    /// Seconds per beat (the segment's period estimate).
    pub period: f64,
}

/// Centered running median, window clamped at the edges. The window is a
/// STRUCTURAL constant (two bars of IBIs — long enough that a one-bar fill
/// cannot capture the median), not a tuned threshold.
const RUNNING_MEDIAN_WINDOW: usize = 17;

/// σ̂ = MAD × 1.4826 — the canonical robust normal-scale estimate.
#[must_use]
pub fn mad_sigma(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let center = median(values);
    median(
        &values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>(),
    ) * 1.4826
}

/// The series' own quantization step: both trackers emit IBIs on a discrete
/// grid (BeatThis's 20 ms logits frame, Queen Mary's tempo lattice), so most
/// first differences are EXACTLY zero and a raw MAD collapses. The median
/// nonzero |difference| measures the grid step Δ; uniform rounding on Δ
/// carries σ = Δ/√12 — the canonical quantization noise, and the floor under
/// every scale estimate here.
#[must_use]
pub fn quantization_sigma(values: &[f64]) -> f64 {
    let nonzero = values
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .filter(|delta| *delta > 0.0)
        .collect::<Vec<_>>();
    if nonzero.is_empty() {
        return 0.0;
    }
    median(&nonzero) / 12.0_f64.sqrt()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

fn running_median(values: &[f64]) -> Vec<f64> {
    let half = RUNNING_MEDIAN_WINDOW / 2;
    (0..values.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(values.len());
            median(&values[lo..hi])
        })
        .collect()
}

/// Indices of ornament IBIs. An ornament must clear BOTH tests:
///
///   1. Statistical — |ibi − running median| > 3·σ̂ (MAD-scale of the
///      residuals, canonical robust statistics).
///   2. Metrical — the interval is closer to ANOTHER metrical level than to
///      the beat level: ratio to the local median < 0.75 (nearer ½×) or
///      ≥ 1.5 (nearer 2×). The bounds are the midpoints between metrical
///      levels — musical structure, not tuning.
///
/// The conjunction is what survives discretized trackers: both engines emit
/// IBIs on a few discrete values, which collapses MAD to ~zero — an
/// additive-only gate then flags every legitimate step (108kick lost 21 % of
/// its beats to exactly this). A frame step is never metrically displaced, so
/// test 2 keeps it; a genuine fill subdivision fails both.
#[must_use]
pub fn ornament_ibis(ibis: &[f64]) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    if ibis.len() < 3 {
        return out;
    }
    let baseline = running_median(ibis);
    let residuals = ibis
        .iter()
        .zip(&baseline)
        .map(|(ibi, base)| ibi - base)
        .collect::<Vec<_>>();
    let sigma = mad_sigma(&residuals)
        .max(quantization_sigma(ibis))
        .max(1e-9);
    for i in 0..ibis.len() {
        if residuals[i].abs() <= 3.0 * sigma || baseline[i] <= 0.0 {
            continue;
        }
        let ratio = ibis[i] / baseline[i];
        if !(0.75..1.5).contains(&ratio) {
            out.insert(i);
        }
    }
    out
}

/// PELT over the cleaned IBI series (Killick 2012, L2 cost, exact).
///
/// Cost of a segment = RSS/σ̂² over its NON-bridge samples (−2·loglik under
/// Normal with the segment's own mean); σ̂ is the difference-based robust
/// noise scale — MAD of first differences × 1.4826 / √2 — immune to the very
/// mean shifts we are hunting. Penalty per changepoint = 2·ln n (BIC: a mean
/// and a location). Prefix sums make each cost O(1); pruning keeps the
/// candidate set tiny. `bridges` marks intervals that exist only because
/// beats between them were gated away — zero tempo evidence, excluded from
/// cost and period alike, spanned in index space.
#[must_use]
pub fn segment(clean_ibis: &[f64], bridges: &BTreeSet<usize>) -> Vec<Segment> {
    let y = clean_ibis;
    let n = y.len();
    let evidence = (0..n).filter(|i| !bridges.contains(i)).collect::<Vec<_>>();
    if n < 2 || evidence.is_empty() {
        let period = evidence
            .first()
            .map(|&i| y[i])
            .or_else(|| y.first().copied())
            .unwrap_or(0.0);
        return vec![Segment {
            beat_range: 0..(n + 1),
            period,
        }];
    }

    // Difference-based noise scale over consecutive EVIDENCE samples.
    let evidence_values = evidence.iter().map(|&i| y[i]).collect::<Vec<_>>();
    let diffs = evidence_values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let sigma = (mad_sigma(&diffs) / std::f64::consts::SQRT_2)
        .max(quantization_sigma(&evidence_values))
        .max(1e-9);
    let sigma2 = sigma * sigma;
    let penalty = 2.0 * (evidence.len() as f64).ln();

    // Winsorize the series the SEGMENTATION sees: an isolated spike (the
    // tracker skipping a beat — between metrical levels, so not an ornament)
    // otherwise carries RSS/σ̂² in the hundreds and buys itself a
    // micro-segment (hip-hop-1 shattered into 52). Clipping to running median
    // ± 3σ̂ is canonical robust preprocessing; a SUSTAINED tempo change moves
    // the running median itself within a window and stays fully visible to
    // PELT.
    let baseline = running_median(y);
    let winsorized = (0..n)
        .map(|i| y[i].clamp(baseline[i] - 3.0 * sigma, baseline[i] + 3.0 * sigma))
        .collect::<Vec<_>>();

    // Prefix sums over evidence only → O(1) segment RSS.
    let mut s1 = vec![0.0; n + 1];
    let mut s2 = vec![0.0; n + 1];
    let mut cnt = vec![0.0; n + 1];
    for i in 0..n {
        let is_evidence = !bridges.contains(&i);
        s1[i + 1] = s1[i] + if is_evidence { winsorized[i] } else { 0.0 };
        s2[i + 1] = s2[i]
            + if is_evidence {
                winsorized[i] * winsorized[i]
            } else {
                0.0
            };
        cnt[i + 1] = cnt[i] + if is_evidence { 1.0 } else { 0.0 };
    }
    let cost = |a: usize, b: usize| -> f64 {
        let len = cnt[b] - cnt[a];
        if len <= 0.0 {
            return 0.0;
        }
        let sum = s1[b] - s1[a];
        let rss = (s2[b] - s2[a]) - sum * sum / len;
        rss.max(0.0) / sigma2
    };

    // PELT with pruning.
    let mut f = vec![0.0; n + 1];
    let mut previous = vec![0usize; n + 1];
    f[0] = -penalty;
    let mut candidates = vec![0usize];
    for t in 1..=n {
        let mut best = f64::INFINITY;
        let mut best_s = 0;
        for &s in &candidates {
            let value = f[s] + cost(s, t) + penalty;
            if value < best {
                best = value;
                best_s = s;
            }
        }
        f[t] = best;
        previous[t] = best_s;
        candidates.retain(|&s| f[s] + cost(s, t) <= f[t]);
        candidates.push(t);
    }

    // Backtrack.
    let mut bounds = Vec::new();
    let mut t = n;
    while t > 0 {
        bounds.push(t);
        t = previous[t];
    }
    bounds.push(0);
    bounds.reverse();

    // MEDIAN over the segment's evidence: isolated flagged-but-kept IBIs (a
    // fill edge whose partner interval stayed) contaminate a mean —
    // j8VwTEV3HD pulled 0.7053 → 0.7134 and the phase vote walked off the
    // grid. On a symmetric within-segment drift the median and mean agree, so
    // the ramp property holds either way.
    let period_of = |a: usize, b: usize| -> f64 {
        let values = (a..b)
            .filter(|i| !bridges.contains(i))
            .map(|i| y[i])
            .collect::<Vec<_>>();
        if values.is_empty() {
            0.0
        } else {
            median(&values)
        }
    };

    // A boundary directly at a bridge means PELT had no evidence to separate
    // the halves — snap such empty-evidence splits away.
    let mut merged = vec![bounds[0]];
    for &bound in &bounds[1..] {
        let a = *merged.last().expect("merged is never empty");
        if bound < n && cnt[bound] - cnt[a] == 0.0 {
            continue;
        }
        merged.push(bound);
    }
    if merged.len() < 2 {
        merged = vec![0, n];
    }

    // Coalesce neighbours whose winsorized means agree under the two-sample
    // z-test at 3σ: |Δ| ≤ 3·σ̂·√(1/n₁+1/n₂). PELT may still pay for cuts
    // around clipped spike clusters; two fragments of the SAME tempo must
    // chain into one grid instead of re-anchoring — that re-anchoring was the
    // boundary-IBI zoo. Sample size does the discrimination: a real ramp's
    // long segments have tiny standard errors and stay split.
    let mean_of = |a: usize, b: usize| -> Option<(f64, f64)> {
        let len = cnt[b] - cnt[a];
        (len > 0.0).then(|| ((s1[b] - s1[a]) / len, len))
    };
    let mut i = 0;
    while merged.len() > 2 && i < merged.len() - 2 {
        let (Some(left), Some(right)) = (
            mean_of(merged[i], merged[i + 1]),
            mean_of(merged[i + 1], merged[i + 2]),
        ) else {
            merged.remove(i + 1);
            i = i.saturating_sub(1);
            continue;
        };
        let standard_error = sigma * (1.0 / left.1 + 1.0 / right.1).sqrt();
        if (left.0 - right.0).abs() <= 3.0 * standard_error {
            merged.remove(i + 1);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }

    merged
        .windows(2)
        .map(|pair| {
            let (a, b) = (pair[0], pair[1]);
            Segment {
                beat_range: a..(if b == n { n + 1 } else { b }),
                period: period_of(a, b),
            }
        })
        .collect()
}

/// Deterministic jitter in [-amp, +amp] — the same LCG the Swift suites use,
/// so the synthetic series are identical on both sides. Shared with the
/// quantizer's suite, which builds beat sequences from the same recipe.
#[cfg(test)]
pub(super) fn jitter(n: usize, amp: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (u64::MAX >> 11) as f64 * 2.0 - 1.0) * amp
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn beats(periods: &[(usize, f64)], jitter_amp: f64) -> Vec<f64> {
        let mut time = 1.0;
        let mut out = Vec::new();
        let noise = jitter(periods.iter().map(|block| block.0).sum(), jitter_amp, 42);
        let mut i = 0;
        for &(count, period) in periods {
            for _ in 0..count {
                out.push(time + noise[i]);
                time += period;
                i += 1;
            }
        }
        out
    }

    fn ibis(beats: &[f64]) -> Vec<f64> {
        beats.windows(2).map(|pair| pair[1] - pair[0]).collect()
    }

    fn no_bridges() -> BTreeSet<usize> {
        BTreeSet::new()
    }

    // MARK: - σ̂

    #[test]
    fn mad_sigma_matches_the_canonical_scale() {
        // Symmetric residuals around 0.5 with MAD 0.01 → σ̂ = 0.01 × 1.4826.
        let values = [0.49, 0.50, 0.51, 0.50, 0.49, 0.51, 0.50];
        assert!((mad_sigma(&values) - 0.01 * 1.4826).abs() < 1e-9);
    }

    // MARK: - Ornament gate

    #[test]
    fn drum_fill_ibis_are_ornaments_and_steady_ones_are_not() {
        // A fill: one beat splits into a 0.16/0.34 pair.
        let mut series = vec![0.5; 60];
        series[30] = 0.16;
        series[31] = 0.34;
        let flagged = ornament_ibis(&series);
        assert!(
            flagged.contains(&30) && flagged.contains(&31),
            "fill IBIs must be flagged: {flagged:?}"
        );
        assert_eq!(
            flagged,
            BTreeSet::from([30, 31]),
            "steady IBIs must survive"
        );
    }

    #[test]
    fn discretized_tracker_steps_are_not_ornaments() {
        // BOTH engines emit discretized IBIs — the majority value collapses
        // MAD to zero and an additive-only gate then flags every legitimate
        // step. The metrical test keeps them: ±2 % of the beat level is
        // nowhere near another level.
        let mut series = vec![0.6153; 240];
        for i in (7..240).step_by(9) {
            series[i] = 0.6037;
        }
        for i in (11..240).step_by(23) {
            series[i] = 0.6269;
        }
        assert_eq!(
            ornament_ibis(&series),
            no_bridges(),
            "quantization wobble is not an ornament"
        );
    }

    #[test]
    fn frame_alternation_on_a_steady_kick_is_not_an_ornament() {
        // 108kick's shape: one IBI value alternating with its frame
        // neighbour. Nothing here is closer to ½× or 2× than to 1×.
        let series = (0..300)
            .map(|i| if i % 3 == 0 { 0.48 } else { 0.46 })
            .collect::<Vec<f64>>();
        assert_eq!(ornament_ibis(&series), no_bridges());
    }

    // MARK: - PELT

    #[test]
    fn steady_track_earns_exactly_one_segment() {
        let series = ibis(&beats(&[(300, 0.5)], 0.004));
        let segments = segment(&series, &no_bridges());
        assert_eq!(segments.len(), 1);
        assert!((segments[0].period - 0.5).abs() < 0.002);
        assert_eq!(segments[0].beat_range, 0..300);
    }

    /// A discretized tracker emits most IBIs at exactly one value, so the MAD
    /// of first differences is zero and the noise scale would collapse to the
    /// 1e-9 guard — every 1-frame step then costs RSS/1e-18 and PELT shatters
    /// the track. The quantization-noise floor (Δ/√12) is what holds it
    /// together, and this is the site where it is load-bearing.
    #[test]
    fn a_discretized_ibi_series_is_one_segment() {
        let mut series = vec![0.6153; 240];
        for i in (7..240).step_by(9) {
            series[i] = 0.6037;
        }
        for i in (11..240).step_by(23) {
            series[i] = 0.6269;
        }
        let segments = segment(&series, &no_bridges());
        assert_eq!(
            segments.len(),
            1,
            "quantization wobble is not a tempo change: {:?}",
            segments
                .iter()
                .map(|s| s.beat_range.clone())
                .collect::<Vec<_>>()
        );
        assert!((segments[0].period - 0.6153).abs() < 1e-9);
    }

    #[test]
    fn tempo_change_is_cut_within_two_beats() {
        let series = ibis(&beats(&[(150, 0.5), (150, 0.6)], 0.004));
        let segments = segment(&series, &no_bridges());
        assert_eq!(
            segments.len(),
            2,
            "one sustained tempo change → exactly two segments"
        );
        assert!((segments[0].period - 0.5).abs() < 0.003);
        assert!((segments[1].period - 0.6).abs() < 0.003);
        assert!(
            (segments[0].beat_range.end as f64 - 150.0).abs() <= 2.0,
            "cut at {}",
            segments[0].beat_range.end
        );
    }

    #[test]
    fn slow_drift_segments_bound_the_lattice_error() {
        // Period ramps 0.500 → 0.525 over 400 beats (≈5 % drift, a live band).
        let ramp = (0..400)
            .map(|i| (1usize, 0.5 + 0.000_062_5 * f64::from(i)))
            .collect::<Vec<_>>();
        let series = ibis(&beats(&ramp, 0.003));
        let segments = segment(&series, &no_bridges());
        assert!(
            segments.len() >= 2,
            "a global fit cannot hold a 5 % ramp; got {}",
            segments.len()
        );

        // Reconstruct the piecewise lattice and bound its error to the series.
        // The honest claim: piecewise beats the single global fit by an order
        // of magnitude.
        let mut time = 0.0_f64;
        let mut true_time = 0.0_f64;
        let mut max_error = 0.0_f64;
        let mut index = 0;
        for piece in &segments {
            for _ in piece.beat_range.clone() {
                max_error = max_error.max((time - true_time).abs());
                if index < series.len() {
                    true_time += series[index];
                    time += piece.period;
                    index += 1;
                }
            }
        }
        let global_period = series.iter().sum::<f64>() / series.len() as f64;
        let mut global_time = 0.0_f64;
        let mut true_time = 0.0_f64;
        let mut global_error = 0.0_f64;
        for value in &series {
            true_time += value;
            global_time += global_period;
            global_error = global_error.max((global_time - true_time).abs());
        }
        assert!(
            max_error < global_error / 10.0,
            "piecewise must beat the global fit by ≥10×: {max_error} vs {global_error}"
        );
        assert!(
            max_error < 0.100,
            "sanity ceiling on an extreme 5 % ramp: {max_error}"
        );
    }

    #[test]
    fn bridge_gaps_do_not_split_agreeing_tempo() {
        // A steady 0.5 s track whose middle beats were gated away: the cleaned
        // series carries one huge bridge IBI. Same tempo on both sides → ONE
        // segment spanning the bridge, so the lattice covers the hole.
        let noise = jitter(200, 0.003, 42);
        let mut series = (0..200).map(|i| 0.5 + noise[i]).collect::<Vec<f64>>();
        series[100] = 6.5; // ~12 gated beats bridged by one interval
        let segments = segment(&series, &BTreeSet::from([100]));
        assert_eq!(
            segments.len(),
            1,
            "agreeing tempo across a bridge must not split"
        );
        assert!(
            (segments[0].period - 0.5).abs() < 0.003,
            "the bridge must not poison the period: {}",
            segments[0].period
        );
        assert_eq!(segments[0].beat_range, 0..201);
    }

    /// The case bridge masking exists for, and the one winsorization cannot
    /// cover: a BURST of gated beats. Ten consecutive bridge intervals own the
    /// running median inside their own window, so the winsorizer keeps them
    /// whole; only masking them out of the evidence stops PELT reading the
    /// hole as a tempo change and cutting a 6 s "segment" out of a steady
    /// track.
    #[test]
    fn a_burst_of_bridges_is_spanned_not_segmented() {
        let noise = jitter(200, 0.003, 42);
        let mut series = (0..200).map(|i| 0.5 + noise[i]).collect::<Vec<f64>>();
        let burst = 95..=104;
        for i in burst.clone() {
            series[i] = 3.0;
        }
        let bridges = burst.collect::<BTreeSet<usize>>();

        let masked = segment(&series, &bridges);
        assert_eq!(
            masked.len(),
            1,
            "a burst of bridges must be spanned: {:?}",
            masked
                .iter()
                .map(|s| s.beat_range.clone())
                .collect::<Vec<_>>()
        );
        assert!(
            (masked[0].period - 0.5).abs() < 0.003,
            "the bridges must not poison the period: {}",
            masked[0].period
        );

        // Same series with nothing declared a bridge: the hole reads as real
        // tempo evidence and PELT pays for the cuts. This is what the masking
        // buys — stated as a contrast so the property cannot quietly become
        // "winsorization happened to cover it".
        assert!(
            segment(&series, &no_bridges()).len() > 1,
            "without masking the burst must look like a tempo change"
        );
    }

    #[test]
    fn sparse_skipped_beat_spikes_do_not_shatter_the_segmentation() {
        // hip-hop-1's shape: a steady 0.70 grid with ~30 isolated ~1.31×
        // intervals (the tracker skipping a swung beat). They are neither
        // ornaments (between metrical levels) nor tempo changes — an L2 cost
        // without winsorization isolates every one into its own micro-segment
        // (52 pieces, 23 % of beats lost).
        let noise = jitter(400, 0.004, 42);
        let mut series = (0..400).map(|i| 0.70 + noise[i]).collect::<Vec<f64>>();
        for i in (13..400).step_by(13) {
            series[i] = 0.92;
        }
        let segments = segment(&series, &no_bridges());
        assert_eq!(
            segments.len(),
            1,
            "sparse spikes must not shatter a steady track"
        );
        assert!((segments[0].period - 0.70).abs() < 0.005);
    }

    #[test]
    fn short_and_empty_series_do_not_crash() {
        assert_eq!(segment(&[], &no_bridges()).len(), 1);
        assert_eq!(segment(&[0.5], &no_bridges()).len(), 1);
    }
}
