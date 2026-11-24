//! Beat-grid quantization over the warp lattice: ornament gate → PELT
//! changepoints → per-segment linear fit, circular-mean anchor and downbeat
//! phase by raw-downbeat votes. The emission PARTITIONS the track: every span
//! belongs to exactly one grid, so no beat is orphaned between segments.
//!
//! Port of iOS `BeatThisProcessing.quantizeBeats`.

use std::collections::BTreeSet;

use ferment_engine_core::BeatLatticeSegment;

use super::lattice;
use super::meter::MeterResult;
#[cfg(test)]
mod support_tests;

#[cfg(test)]
#[test]
fn bounded_lattice_rejects_expansion_before_allocation_and_preserves_normal_results() {
    let beats = (0..32).map(|n| n as f64 * 0.5).collect::<Vec<_>>();
    let support = vec![true; beats.len()];
    assert!(quantize_bounded(&beats, &[], None, None, Some(&support), 16).is_err());
    let actual = quantize_bounded(&beats, &[], None, None, Some(&support), 64).unwrap();
    let expected = quantize_supported_beats(&beats, &[], None, None, &support);
    assert_eq!(actual.beats, expected.beats);
    assert_eq!(actual.segments, expected.segments);
}

#[derive(Debug, Clone)]
pub struct QuantizationResult {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    /// Duration-weighted dominant segment tempo — the single number a
    /// consumer shows; the per-segment truth lives in `segments`.
    pub bpm: Option<f64>,
    /// The warp-marker lattice, one entry per constant-tempo stretch. The
    /// flattened `beats`/`downbeats` above are its concatenated expansion.
    pub segments: Vec<BeatLatticeSegment>,
}

impl QuantizationResult {
    fn empty() -> Self {
        Self {
            beats: Vec::new(),
            downbeats: Vec::new(),
            bpm: None,
            segments: Vec::new(),
        }
    }
}

pub fn local_tempo_bpm(beats: &[f64], window_radius: usize) -> Vec<f64> {
    if beats.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(beats.len());
    let last = beats.len() - 1;
    for i in 0..beats.len() {
        let lo = i.saturating_sub(window_radius);
        let hi = (i + window_radius).min(last);
        let interval_count = hi - lo;
        if interval_count < 2 {
            continue;
        }
        let span = beats[hi] - beats[lo];
        if span <= 0.0 {
            continue;
        }
        let local_ibi = span / interval_count as f64;
        if local_ibi > 0.0 {
            out.push(60.0 / local_ibi);
        }
    }
    out
}

/// Least squares over RECOVERED grid indices: consecutive index steps are
/// `round(Δt / period)`, so a bridge (beats gated away between two kept ones)
/// advances the axis by its true integer span instead of compressing it — the
/// classic deleted-sample slope bias. Returns `None` on a degenerate axis.
pub fn indexed_linear_fit(beats: &[f64], period: f64) -> Option<(f64, f64)> {
    if beats.len() < 2 || period <= 0.0 {
        return None;
    }
    let mut indices = vec![0.0; beats.len()];
    let mut current = 0.0;
    for i in 1..beats.len() {
        current += (((beats[i] - beats[i - 1]) / period).round()).max(1.0);
        indices[i] = current;
    }
    let n = beats.len() as f64;
    let sum_x: f64 = indices.iter().sum();
    let sum_y: f64 = beats.iter().sum();
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (&x, &y) in indices.iter().zip(beats) {
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator <= 0.0 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope > 0.0).then_some((intercept, slope))
}

/// Ornament gate: a beat is an ornament when BOTH of its adjacent IBIs are
/// flagged — exactly the signature of an inserted beat splitting one interval.
/// Returns the surviving beats plus the cleaned-index set of BRIDGE intervals,
/// the IBIs that only exist because beats between them were removed.
fn clean_ornaments(beats: &[f64], support: Option<&[bool]>) -> (Vec<f64>, BTreeSet<usize>) {
    let indices = (0..beats.len())
        .filter(|&i| support.is_none_or(|s| s[i]))
        .collect::<Vec<_>>();
    let evidence = indices.iter().map(|&i| beats[i]).collect::<Vec<_>>();
    let beats = evidence.as_slice();
    let raw_ibis = inter_beat_intervals(beats);
    let flagged = lattice::ornament_ibis(&raw_ibis);
    let mut cleaned = Vec::with_capacity(beats.len());
    let mut bridges = BTreeSet::new();
    let mut dropped = false;
    let mut previous = None;
    for (i, &beat) in beats.iter().enumerate() {
        let left_flagged = i > 0 && flagged.contains(&(i - 1));
        let right_flagged = i < raw_ibis.len() && flagged.contains(&i);
        if left_flagged && right_flagged {
            dropped = true;
            continue;
        }
        if (dropped || previous.is_some_and(|p| indices[i] > p + 1)) && !cleaned.is_empty() {
            // The IBI about to be created spans removed beats — a bridge,
            // structural rather than tempo evidence.
            bridges.insert(cleaned.len() - 1);
        }
        dropped = false;
        cleaned.push(beat);
        previous = Some(indices[i]);
    }
    if cleaned.len() < 4 && support.is_none() {
        return (beats.to_vec(), BTreeSet::new());
    }
    (cleaned, bridges)
}

/// Materialize the changepoint pieces into beat runs, folding degenerate ones
/// (fewer than 2 beats — no tempo of their own) into a neighbour so the
/// emission always PARTITIONS the track: every span belongs to exactly one
/// grid, no clamp gaps. The Ballroom hold-out found beats orphaned between
/// clamps.
fn runs_from_pieces(cleaned: &[f64], pieces: &[lattice::Segment]) -> Vec<Vec<f64>> {
    let mut runs: Vec<Vec<f64>> = Vec::new();
    for piece in pieces {
        let lo = piece.beat_range.start.min(cleaned.len());
        let hi = piece.beat_range.end.clamp(lo, cleaned.len());
        let segment_beats = &cleaned[lo..hi];
        if segment_beats.is_empty() {
            continue;
        }
        match runs.last_mut() {
            Some(previous) if segment_beats.len() < 2 => previous.extend_from_slice(segment_beats),
            _ => runs.push(segment_beats.to_vec()),
        }
    }
    while runs.len() >= 2 && runs[0].len() < 2 {
        let head = runs.remove(0);
        runs[0].splice(0..0, head);
    }
    runs
}

pub fn quantize_beats(
    beats: &[f64],
    raw_downbeats: &[f64],
    meter: Option<MeterResult>,
    track_duration: Option<f64>,
) -> QuantizationResult {
    quantize_bounded(
        beats,
        raw_downbeats,
        meter,
        track_duration,
        None,
        usize::MAX,
    )
    .expect("unbounded result")
}

/// DBN-inferred beats remain in the raw result, but an unsupported emission
/// must not become evidence of a new tempo. Missing observations are bridges,
/// not slower beats. This is separate from (and does not alter) Viterbi decoding.
pub(super) fn quantize_supported_beats(
    beats: &[f64],
    raw_downbeats: &[f64],
    meter: Option<MeterResult>,
    track_duration: Option<f64>,
    support: &[bool],
) -> QuantizationResult {
    if support.len() != beats.len() {
        return QuantizationResult::empty();
    }
    quantize_bounded(
        beats,
        raw_downbeats,
        meter,
        track_duration,
        Some(support),
        usize::MAX,
    )
    .expect("unbounded result")
}

pub(super) fn quantize_bounded(
    beats: &[f64],
    raw_downbeats: &[f64],
    meter: Option<MeterResult>,
    track_duration: Option<f64>,
    support: Option<&[bool]>,
    max_events: usize,
) -> ferment_engine_core::Result<QuantizationResult> {
    if beats.len() > max_events
        || raw_downbeats.len() > max_events
        || support.is_some_and(|support| support.len() != beats.len())
    {
        return Err(ferment_engine_core::EngineError::invalid_config(
            "Beat result quota or support mismatch",
        ));
    }
    if beats.len() < 4 {
        return Ok(QuantizationResult::empty());
    }

    // 1. Ornaments leave the lattice fit (they stay in the raw arrays).
    let (cleaned, bridges) = clean_ornaments(beats, support);
    if cleaned.len() < 4 {
        return Ok(QuantizationResult::empty());
    }

    // 2. Changepoints: PELT + BIC over the cleaned IBIs — the number of
    //    constant-tempo stretches is model selection, not a threshold;
    //    bridges are masked out of the evidence and spanned in place.
    let pieces = lattice::segment(&inter_beat_intervals(&cleaned), &bridges);
    let runs = runs_from_pieces(&cleaned, &pieces);

    // 3. Per run: index-recovered least squares for the period, circular
    //    anchor fallback, downbeat phase voted by the raw downbeats inside.
    //    Emission windows are half-open [boundary, boundary): midpoints
    //    between neighbouring runs, the track edges padded half a period.
    let mut all_beats = Vec::new();
    let mut all_downbeats = Vec::new();
    let mut segments: Vec<BeatLatticeSegment> = Vec::new();

    for (run_index, run_beats) in runs.iter().enumerate() {
        let run_ibis = inter_beat_intervals(run_beats);
        if run_beats.len() < 2 || run_ibis.is_empty() {
            continue;
        }
        // The upper-middle IBI, not the averaged even-count median: this is
        // the segment's own pulse, and averaging two neighbours across a
        // quantization step lands the grid between frames.
        let base_period = {
            let mut sorted = run_ibis.clone();
            sorted.sort_by(f64::total_cmp);
            sorted[sorted.len() / 2]
        };

        let (anchor, reference_ibi) = match indexed_linear_fit(run_beats, base_period) {
            Some((fit_anchor, fit_ibi))
                if base_period > 0.0 && (fit_ibi - base_period).abs() / base_period <= 0.15 =>
            {
                (fit_anchor, fit_ibi)
            }
            _ => (grid_anchor(run_beats, base_period), base_period),
        };
        if reference_ibi <= 0.0 {
            continue;
        }

        let first_beat = run_beats[0];
        let last_beat = run_beats[run_beats.len() - 1];
        let lower_bound = match run_index.checked_sub(1).and_then(|i| runs[i].last()) {
            Some(previous_last) => (previous_last + first_beat) / 2.0,
            None => (first_beat - reference_ibi / 2.0).max(0.0),
        };
        let upper_bound = match runs.get(run_index + 1).and_then(|next| next.first()) {
            Some(next_first) => (last_beat + next_first) / 2.0,
            None => track_duration
                .unwrap_or(f64::INFINITY)
                .min(last_beat + reference_ibi / 2.0),
        };
        let min_index = ((lower_bound - anchor) / reference_ibi - 1e-9).ceil() as i64;
        let max_index = ((upper_bound - anchor) / reference_ibi - 1e-9).floor() as i64;
        if min_index > max_index {
            continue;
        }
        let count = (max_index as i128 - min_index as i128 + 1) as u128;
        if count > max_events.saturating_sub(all_beats.len()) as u128 {
            return Err(ferment_engine_core::EngineError::invalid_config(
                "Quantized beat result quota exceeded",
            ));
        }

        // max(0, ·): the head node sits AT zero mathematically but can land a
        // few ULPs below it in float — the lattice must not claim a beat
        // before the track starts.
        let node = |index: i64| (anchor + index as f64 * reference_ibi).max(0.0);
        let segment_quantized = (min_index..=max_index).map(node).collect::<Vec<_>>();
        let mut segment_downbeats = Vec::new();
        let mut downbeat_phase = None;

        if let Some(meter) = meter.filter(|meter| meter.meter > 0) {
            // Phase from the raw downbeats INSIDE this stretch — the same vote
            // as the global one, scoped. Regression: BeatQuantizer/j8VwTEV3HD.
            let mut phase_counts = vec![0usize; meter.meter];
            for &downbeat in raw_downbeats {
                if downbeat < lower_bound - reference_ibi / 2.0
                    || downbeat > upper_bound + reference_ibi / 2.0
                {
                    continue;
                }
                let grid_index = ((downbeat - anchor) / reference_ibi).round() as i64;
                phase_counts[positive_modulo(grid_index, meter.meter as i64) as usize] += 1;
            }
            let fallback_phase = positive_modulo(meter.phase as i64, meter.meter as i64) as usize;
            let mut best_phase = fallback_phase;
            let mut best_count = -1i64;
            for (phase, &count) in phase_counts.iter().enumerate() {
                let count = count as i64;
                if count > best_count || (count == best_count && phase == fallback_phase) {
                    best_count = count;
                    best_phase = phase;
                }
            }
            segment_downbeats = (min_index..=max_index)
                .filter(|&index| {
                    positive_modulo(index - best_phase as i64, meter.meter as i64) == 0
                })
                .map(node)
                .collect();
            downbeat_phase = segment_downbeats.first().map(|first| {
                ((first - segment_quantized[0]) / reference_ibi)
                    .round()
                    .max(0.0) as usize
            });
        }

        all_beats.extend_from_slice(&segment_quantized);
        all_downbeats.extend_from_slice(&segment_downbeats);
        segments.push(BeatLatticeSegment {
            bpm: 60.0 / reference_ibi,
            first: segment_quantized[0],
            count: segment_quantized.len(),
            downbeat_phase,
            meter: meter.map(|meter| meter.meter),
        });
    }

    if all_beats.is_empty() {
        return Ok(QuantizationResult::empty());
    }

    // Dominant tempo = the segment covering the most time. First one wins a
    // tie, matching the Swift `max(by:)` this ports.
    let dominant = segments
        .iter()
        .fold(None::<&BeatLatticeSegment>, |best, segment| {
            let span = |segment: &BeatLatticeSegment| segment.count as f64 * 60.0 / segment.bpm;
            match best {
                Some(best) if span(best) >= span(segment) => Some(best),
                _ => Some(segment),
            }
        });
    Ok(QuantizationResult {
        bpm: dominant.map(|segment| segment.bpm),
        beats: all_beats,
        downbeats: all_downbeats,
        segments,
    })
}

/// Circular mean of beat residuals modulo a fixed IBI. This chooses the
/// intercept when the index-recovered fit's tempo is rejected in favour of the
/// segment's median IBI.
fn grid_anchor(beats: &[f64], ibi: f64) -> f64 {
    if ibi <= 0.0 || beats.is_empty() {
        return 0.0;
    }
    let mut sine_sum = 0.0;
    let mut cosine_sum = 0.0;
    for &beat in beats {
        let angle = 2.0 * std::f64::consts::PI * (beat % ibi) / ibi;
        sine_sum += angle.sin();
        cosine_sum += angle.cos();
    }
    sine_sum.atan2(cosine_sum) / (2.0 * std::f64::consts::PI) * ibi
}

pub fn grid_stability(beats: &[f64]) -> f64 {
    if beats.len() < 3 {
        return 0.0;
    }
    let intervals = inter_beat_intervals(beats);
    if intervals.len() < 2 {
        return 0.0;
    }
    let reference_ibi = median(&intervals);
    if reference_ibi <= 0.0 {
        return 0.0;
    }
    let threshold = 0.05;
    let stable = intervals
        .iter()
        .filter(|&&v| (v - reference_ibi).abs() / reference_ibi <= threshold)
        .count();
    stable as f64 / intervals.len() as f64
}

pub fn tempo_stability(beats: &[f64]) -> f64 {
    let local_bpm = local_tempo_bpm(beats, 4);
    if local_bpm.len() < 2 {
        return 0.0;
    }
    let reference = median(&local_bpm);
    if reference <= 0.0 {
        return 0.0;
    }
    let threshold = 0.03;
    let stable = local_bpm
        .iter()
        .filter(|&&b| (b - reference).abs() / reference <= threshold)
        .count();
    stable as f64 / local_bpm.len() as f64
}

pub fn bpm_from_beats(beats: &[f64]) -> Option<f64> {
    let intervals = inter_beat_intervals(beats);
    if intervals.is_empty() {
        return None;
    }
    let m = median(&intervals);
    if m <= 0.0 {
        return None;
    }
    Some(60.0 / m)
}

fn inter_beat_intervals(beats: &[f64]) -> Vec<f64> {
    beats
        .windows(2)
        .filter_map(|w| {
            let d = w[1] - w[0];
            if d > 0.0 { Some(d) } else { None }
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn positive_modulo(value: i64, modulus: i64) -> i64 {
    ((value % modulus) + modulus) % modulus
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct BeatFixture {
        time_signature: String,
        beats: Vec<f64>,
        downbeats: Vec<f64>,
    }

    fn regression_fixture(name: &str) -> BeatFixture {
        let json = match name {
            "j8VwTEV3HD" => include_str!("testdata/j8VwTEV3HD.json"),
            "v8H5hd4SnF" => include_str!("testdata/v8H5hd4SnF.json"),
            "sdOAq0IPSq" => include_str!("testdata/sdOAq0IPSq.json"),
            _ => panic!("unknown BeatQuantizer fixture {name}"),
        };
        serde_json::from_str(json).expect("valid BeatQuantizer fixture")
    }

    fn assert_quantized_downbeats_align_with_raw(name: &str, tolerance: f64) {
        let fixture = regression_fixture(name);
        let meter_n = fixture
            .time_signature
            .split('/')
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4);
        // Deliberately wrong phase: raw downbeats, rather than this fallback,
        // must establish the actual bar phase.
        let meter = MeterResult {
            meter: meter_n,
            phase: 0,
            raw_meter: meter_n,
        };
        let result = quantize_beats(&fixture.beats, &fixture.downbeats, Some(meter), None);
        assert!(result.beats.len() > 1, "{name}: degenerate grid");
        assert!(
            !result.downbeats.is_empty(),
            "{name}: no quantized downbeats produced"
        );

        let lower = result.beats[0] - tolerance;
        let upper = result.beats[result.beats.len() - 1] + tolerance;
        let in_range = fixture
            .downbeats
            .iter()
            .copied()
            .filter(|value| *value >= lower && *value <= upper)
            .collect::<Vec<_>>();
        assert!(
            in_range.len() >= fixture.downbeats.len().saturating_sub(2),
            "{name}: {} raw downbeats fell outside the quantized grid",
            fixture.downbeats.len() - in_range.len()
        );
        for downbeat in in_range {
            let nearest = result
                .downbeats
                .iter()
                .min_by(|left, right| {
                    (*left - downbeat)
                        .abs()
                        .total_cmp(&(*right - downbeat).abs())
                })
                .copied()
                .expect("quantized downbeat");
            assert!(
                (nearest - downbeat).abs() < tolerance,
                "{name}: raw downbeat {downbeat}s has no quantized downbeat within {}ms (nearest {nearest}s)",
                (tolerance * 1_000.0) as usize
            );
        }
    }

    #[test]
    fn local_tempo_with_constant_120bpm() {
        let beats: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        let bpm = local_tempo_bpm(&beats, 4);
        for v in bpm {
            assert_abs_diff_eq!(v, 120.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn grid_stability_metronomic_is_one() {
        let beats: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
        assert_abs_diff_eq!(grid_stability(&beats), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn bpm_from_metronomic_120() {
        let beats: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        let bpm = bpm_from_beats(&beats).unwrap();
        assert_abs_diff_eq!(bpm, 120.0, epsilon = 1e-9);
    }

    #[test]
    fn quantize_metronomic_produces_clean_grid() {
        // 32 perfect beats every 0.5 s, range 0..15.5 s. One segment; with the
        // half-IBI margin (0.25 s) the emission window is [0, 15.75) and the
        // lattice holds 32 nodes.
        let beats: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
        let q = quantize_beats(&beats, &[], None, Some(16.0));
        assert_eq!(q.beats.len(), 32);
        assert_abs_diff_eq!(q.bpm.unwrap(), 120.0, epsilon = 1.0);
        assert_eq!(q.segments.len(), 1, "a metronome is one constant tempo");
        assert_eq!(q.segments[0].count, 32);
        assert_abs_diff_eq!(q.segments[0].first, 0.0, epsilon = 1e-6);
        assert_eq!(q.segments[0].meter, None);
        assert_eq!(q.segments[0].downbeat_phase, None);
    }

    /// The emission must PARTITION the track: at a tempo change the two grids
    /// meet at the midpoint between the neighbouring beats — no beat may be
    /// left without a lattice node nearby, and the lattice stays strictly
    /// increasing. This is the Ballroom hold-out defect: beats orphaned at
    /// segment boundaries (p95 jumped to 580 ms).
    #[test]
    fn segment_joins_lose_no_beats() {
        let noise = lattice::jitter(300, 0.003, 42);
        let mut beats = Vec::with_capacity(300);
        let mut time = 1.0;
        for (index, period) in (0..300).map(|i| (i, if i < 150 { 0.5 } else { 0.64 })) {
            beats.push(time + noise[index]);
            time += period;
        }
        let meter = MeterResult {
            meter: 4,
            phase: 0,
            raw_meter: 4,
        };
        let result = quantize_beats(&beats, &[], Some(meter), None);
        let grid = result.beats;
        assert!(
            grid.len() as f64 >= beats.len() as f64 * 0.97,
            "the lattice must cover ≥97% of the beats: {} of {}",
            grid.len(),
            beats.len()
        );
        assert!(
            grid.windows(2).all(|pair| pair[1] > pair[0]),
            "lattice must be strictly increasing"
        );
        let mut j = 0;
        let mut worst = 0.0f64;
        for beat in &beats {
            while j + 1 < grid.len() && (grid[j + 1] - beat).abs() <= (grid[j] - beat).abs() {
                j += 1;
            }
            worst = worst.max((grid[j] - beat).abs());
        }
        assert!(worst < 0.25, "beat orphaned at a segment join: {worst}s");

        assert_eq!(result.segments.len(), 2, "one sustained tempo change");
        assert_abs_diff_eq!(result.segments[0].bpm, 120.0, epsilon = 1.0);
        assert_abs_diff_eq!(result.segments[1].bpm, 93.75, epsilon = 1.0);
    }

    /// The flattened arrays ARE the segments' concatenated expansion — the
    /// property every consumer of the wire shape relies on.
    #[test]
    fn segments_expand_to_the_flattened_grid() {
        let noise = lattice::jitter(300, 0.003, 7);
        let mut beats = Vec::with_capacity(300);
        let mut time = 1.0;
        for (index, period) in (0..300).map(|i| (i, if i < 150 { 0.5 } else { 0.64 })) {
            beats.push(time + noise[index]);
            time += period;
        }
        let downbeats = beats.iter().step_by(4).copied().collect::<Vec<_>>();
        let meter = MeterResult {
            meter: 4,
            phase: 0,
            raw_meter: 4,
        };
        let result = quantize_beats(&beats, &downbeats, Some(meter), None);

        let mut expanded = Vec::new();
        let mut expanded_downbeats = Vec::new();
        for segment in &result.segments {
            let period = 60.0 / segment.bpm;
            for index in 0..segment.count {
                let time = segment.first + index as f64 * period;
                expanded.push(time);
                if let (Some(phase), Some(meter)) = (segment.downbeat_phase, segment.meter)
                    && index >= phase
                    && (index - phase).is_multiple_of(meter)
                {
                    expanded_downbeats.push(time);
                }
            }
        }
        assert_eq!(expanded.len(), result.beats.len());
        for (got, want) in expanded.iter().zip(&result.beats) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-9);
        }
        assert_eq!(expanded_downbeats.len(), result.downbeats.len());
        for (got, want) in expanded_downbeats.iter().zip(&result.downbeats) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-9);
        }
    }

    /// The ornament gate must reach the quantizer: a drum fill that doubles
    /// two beats leaves the lattice fit but stays in the raw arrays.
    #[test]
    fn a_drum_fill_does_not_bend_the_grid() {
        let steady = (0..80).map(|i| f64::from(i) * 0.5).collect::<Vec<_>>();
        let clean = quantize_beats(&steady, &[], None, None);

        // Split beat 40 into a 0.16/0.34 pair — the fill signature.
        let mut with_fill = steady.clone();
        with_fill.insert(41, steady[40] + 0.16);
        let filled = quantize_beats(&with_fill, &[], None, None);

        assert_eq!(filled.segments.len(), 1, "a fill is not a tempo change");
        assert_abs_diff_eq!(filled.segments[0].bpm, clean.segments[0].bpm, epsilon = 0.5);
    }

    #[test]
    fn test_j8_vw_tev3_hd_quantized_downbeats_on_raw_phase() {
        assert_quantized_downbeats_align_with_raw("j8VwTEV3HD", 0.06);

        // Pin the disputed bar: 33.779s is the real downbeat. 33.085s is
        // merely a beat and was the old one-beat-early result.
        let fixture = regression_fixture("j8VwTEV3HD");
        let meter = MeterResult {
            meter: 4,
            phase: 0,
            raw_meter: 4,
        };
        let result = quantize_beats(&fixture.beats, &fixture.downbeats, Some(meter), None);
        let near_33 = result
            .downbeats
            .iter()
            .copied()
            .filter(|value| (33.0..=34.5).contains(value))
            .collect::<Vec<_>>();
        assert!(
            near_33.iter().any(|value| (value - 33.779).abs() < 0.06),
            "expected a downbeat at ~33.779s; got {near_33:?}"
        );
        assert!(
            !near_33.iter().any(|value| (value - 33.085).abs() < 0.06),
            "33.085s is a beat, not a downbeat"
        );
    }

    #[test]
    fn test_v8_h5hd4_sn_f_correct_stays_correct() {
        assert_quantized_downbeats_align_with_raw("v8H5hd4SnF", 0.06);
    }

    #[test]
    fn test_sd_o_aq0_ip_sq_quantized_tempo_matches_median() {
        let fixture = regression_fixture("sdOAq0IPSq");
        let median_bpm = 60.0 / median(&inter_beat_intervals(&fixture.beats));
        let meter = MeterResult {
            meter: 4,
            phase: 0,
            raw_meter: 4,
        };
        let result = quantize_beats(&fixture.beats, &fixture.downbeats, Some(meter), None);
        let bpm = result.bpm.expect("quantized BPM");
        assert_abs_diff_eq!(bpm, median_bpm, epsilon = median_bpm * 0.1);
        assert!(
            bpm < 85.0,
            "must not be the ~98 BPM density latch; bpm={bpm}"
        );
    }

    fn deterministic_unit(index: usize) -> f64 {
        let mut value = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn variable_beats(count: usize, center: f64, variation: f64) -> Vec<f64> {
        let mut beats = Vec::with_capacity(count);
        let mut time = 0.0;
        for index in 0..count {
            beats.push(time);
            let jitter = (deterministic_unit(index) * 2.0 - 1.0) * variation;
            time += center * (1.0 + jitter);
        }
        beats
    }

    mod grid_stability_tests {
        use super::*;

        #[test]
        fn test_empty_beats() {
            assert_eq!(grid_stability(&[]), 0.0);
        }

        #[test]
        fn test_single_beat() {
            assert_eq!(grid_stability(&[1.0]), 0.0);
        }

        #[test]
        fn test_two_beats() {
            assert_eq!(grid_stability(&[0.0, 0.5]), 0.0);
        }

        #[test]
        fn test_three_beats_minimum() {
            assert_eq!(grid_stability(&[0.0, 0.5, 1.0]), 1.0);
        }

        #[test]
        fn test_perfect_metronome_120_bpm() {
            let beats = (0..100).map(|index| index as f64 * 0.5).collect::<Vec<_>>();
            assert_eq!(grid_stability(&beats), 1.0);
        }

        #[test]
        fn test_perfect_metronome_60_bpm() {
            let beats = (0..50).map(|index| index as f64).collect::<Vec<_>>();
            assert_eq!(grid_stability(&beats), 1.0);
        }

        #[test]
        fn test_perfect_metronome_180_bpm() {
            let beats = (0..200)
                .map(|index| index as f64 * (60.0 / 180.0))
                .collect::<Vec<_>>();
            assert_eq!(grid_stability(&beats), 1.0);
        }

        #[test]
        fn test_slight_jitter() {
            assert!(grid_stability(&variable_beats(100, 0.5, 0.02)) > 0.9);
        }

        #[test]
        fn test_beat_tracker_typical_error() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for index in 0..100 {
                beats.push(time);
                time += 0.5 + if index % 2 == 0 { 0.01 } else { -0.01 };
            }
            assert!(grid_stability(&beats) > 0.95);
        }

        #[test]
        fn test_moderate_rubato() {
            let stability = grid_stability(&variable_beats(100, 0.5, 0.08));
            assert!(stability < 0.8, "stability={stability}");
            assert!(stability > 0.2, "stability={stability}");
        }

        #[test]
        fn test_high_rubato() {
            let stability = grid_stability(&variable_beats(100, 0.5, 0.20));
            assert!(stability < 0.5, "stability={stability}");
        }

        #[test]
        fn test_random_intervals() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for index in 0..50 {
                beats.push(time);
                time += 0.2 + deterministic_unit(index) * 1.3;
            }
            let stability = grid_stability(&beats);
            assert!(stability < 0.3, "stability={stability}");
        }

        #[test]
        fn test_few_outliers_dont_tank_score() {
            let mut beats = (0..100).map(|index| index as f64 * 0.5).collect::<Vec<_>>();
            beats[50] += 0.25;
            beats[75] += 0.3;
            assert!(grid_stability(&beats) > 0.9);
        }

        #[test]
        fn test_tempo_change_middle() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for _ in 0..50 {
                beats.push(time);
                time += 0.5;
            }
            for _ in 0..50 {
                beats.push(time);
                time += 60.0 / 140.0;
            }
            assert!(grid_stability(&beats) < 0.7);
        }

        #[test]
        fn test_exactly_at_threshold() {
            let beats = (0..100)
                .map(|index| index as f64 * 0.5 * 1.05)
                .collect::<Vec<_>>();
            assert_eq!(grid_stability(&beats), 1.0);
        }

        #[test]
        fn test_just_over_threshold() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for index in 0..100 {
                beats.push(time);
                time += if index < 50 { 0.5 } else { 0.5 * 1.06 };
            }
            assert_abs_diff_eq!(grid_stability(&beats), 0.5, epsilon = 0.1);
        }
    }

    mod tempo_stability_tests {
        use super::*;

        #[test]
        fn test_empty_beats() {
            assert_eq!(tempo_stability(&[]), 0.0);
        }

        #[test]
        fn test_perfect_metronome_120_bpm() {
            let beats = (0..100).map(|index| index as f64 * 0.5).collect::<Vec<_>>();
            assert_eq!(tempo_stability(&beats), 1.0);
        }

        #[test]
        fn test_small_beat_jitter_preserves_tempo_stability() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for index in 0..100 {
                beats.push(time);
                time += 0.5 + if index % 2 == 0 { 0.01 } else { -0.01 };
            }
            assert!(tempo_stability(&beats) > 0.95);
        }

        #[test]
        fn test_gradual_tempo_drift_reduces_tempo_stability() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for index in 0..120 {
                beats.push(time);
                let progress = index as f64 / 119.0;
                time += 0.5 * (1.0 - 0.08 * progress);
            }
            assert!(tempo_stability(&beats) < 0.8);
        }

        #[test]
        fn test_section_tempo_change_reduces_tempo_stability() {
            let mut beats = Vec::new();
            let mut time = 0.0;
            for _ in 0..50 {
                beats.push(time);
                time += 0.5;
            }
            for _ in 0..50 {
                beats.push(time);
                time += 60.0 / 140.0;
            }
            assert!(tempo_stability(&beats) < 0.7);
        }
    }
}
