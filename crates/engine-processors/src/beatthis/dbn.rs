//! Port of madmom's `DBNDownBeatTrackingProcessor` (Böck, Krebs, Widmer —
//! "Joint Beat and Downbeat Tracking with Recurrent Neural Networks", ISMIR
//! 2016) over the BeatThis logits, with `beat_this`'s `postp_dbn` input
//! massaging. One bar-pointer HMM per bar length (3 and 4 beats), exponential
//! tempo transitions at beat boundaries, exact Viterbi, best bar length by
//! final log-probability, then beat correction to the activation peaks.
//!
//! Everything runs in `f64`: madmom decodes in float64, and the parity gate
//! against its output is exact frame indices, not "close enough".
//!
//! Port of iOS `Processors/BeatThis/BeatThisDBN.swift`.

use super::meter::{MeterResult, majority_phase};
use super::postprocess::{self, ProcessResult};
use super::quantize::quantize_supported_beats;

mod paged;
#[cfg(test)]
use paged::decode as decode_paged;
pub(super) use paged::process as process_paged;

// madmom defaults (DBNDownBeatTrackingProcessor) + beat_this's call:
// beats_per_bar=[3,4], min/max bpm 55/215, num_tempi=60, λ=100,
// observation_lambda=16, threshold=0.05, correct=True.
const BAR_LENGTHS: [usize; 2] = [3, 4];
const MIN_BPM: f64 = 55.0;
const MAX_BPM: f64 = 215.0;
const NUM_TEMPI: usize = 60;
const TRANSITION_LAMBDA: f64 = 100.0;
const OBSERVATION_LAMBDA: usize = 16;
const ACTIVATION_THRESHOLD: f64 = 0.05;
/// `np.spacing(1)` — transition probabilities at or below this are dropped.
const TRANSITION_FLOOR: f64 = f64::EPSILON;
/// `beat_this`'s `postp_dbn` squeeze: activations land in (ε/2, 1−ε/2).
const SQUEEZE_EPSILON: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub beat_times: Vec<f64>,
    /// 1-based position of each beat inside its bar (downbeat = 1).
    pub beat_numbers: Vec<usize>,
    pub bar_length: usize,
    pub log_probability: f64,
}

impl Decoded {
    /// Times of the beats the decoder placed on beat one of a bar.
    #[must_use]
    pub fn downbeat_times(&self) -> Vec<f64> {
        self.beat_numbers
            .iter()
            .zip(&self.beat_times)
            .filter(|(number, _)| **number == 1)
            .map(|(_, time)| *time)
            .collect()
    }
}

/// DBN decode shaped like [`postprocess::process`]. The shared quantizer uses
/// observed beats as tempo evidence and bridges inferred ones; raw decoder
/// output is unchanged. The meter is not re-detected from downbeat scores:
/// the winning HMM's bar length IS the meter, and the phase falls out of the
/// decoded beat numbers. Falls back to the minimal peak-pick when the decode
/// yields fewer than two beats.
#[must_use]
pub fn process(
    beat_logits: &[f32],
    downbeat_logits: &[f32],
    fps: f64,
    track_duration: Option<f64>,
) -> ProcessResult {
    let Some(decoded) =
        decode(beat_logits, downbeat_logits, fps).filter(|decoded| decoded.beat_times.len() >= 2)
    else {
        tracing::warn!("DBN decode produced no usable grid — falling back to minimal postprocess");
        return postprocess::process(beat_logits, downbeat_logits, fps, track_duration);
    };

    let support = decoded
        .beat_times
        .iter()
        .map(|time| {
            let frame = (time * fps).round() as usize;
            beat_logits.get(frame).is_some_and(|v| *v > 0.)
                || downbeat_logits.get(frame).is_some_and(|v| *v > 0.)
        })
        .collect::<Vec<_>>();
    finish(decoded, &support, track_duration, usize::MAX).expect("unbounded DBN result")
}

fn finish(
    decoded: Decoded,
    support: &[bool],
    track_duration: Option<f64>,
    max_events: usize,
) -> ferment_engine_core::Result<ProcessResult> {
    let downbeat_indices = decoded
        .beat_numbers
        .iter()
        .enumerate()
        .filter(|(_, number)| **number == 1)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let downbeats = downbeat_indices
        .iter()
        .map(|&index| decoded.beat_times[index])
        .collect::<Vec<_>>();
    let meter = MeterResult {
        meter: decoded.bar_length,
        phase: majority_phase(&downbeat_indices, decoded.bar_length),
        raw_meter: decoded.bar_length,
    };
    // A Viterbi path includes inferred beats through weak observations. They
    // are valid raw timing hypotheses, not evidence of a local tempo change.
    // Reuse the minimal peak-picker's positive-logit boundary (p > 0.5), on
    // either beat OR downbeat channel, at the decoder's corrected frame.
    let supported_downbeats = downbeat_indices
        .iter()
        .filter(|&&i| support[i])
        .map(|&i| decoded.beat_times[i])
        .collect::<Vec<_>>();
    let quantized = if max_events == usize::MAX {
        quantize_supported_beats(
            &decoded.beat_times,
            &supported_downbeats,
            Some(meter),
            track_duration,
            support,
        )
    } else {
        super::quantize::quantize_bounded(
            &decoded.beat_times,
            &supported_downbeats,
            Some(meter),
            track_duration,
            Some(support),
            max_events,
        )?
    };

    tracing::info!(
        beats = decoded.beat_times.len(),
        downbeats = downbeats.len(),
        bar = decoded.bar_length,
        log_probability = decoded.log_probability,
        "DBN decode"
    );

    Ok(ProcessResult {
        beats: decoded.beat_times,
        downbeats,
        quantized_beats: quantized.beats,
        quantized_downbeats: quantized.downbeats,
        quantized_bpm: quantized.bpm,
        segments: quantized.segments,
    })
}

/// The decoder. `None` when the activations support no path at all — never a
/// fabricated grid.
#[must_use]
pub fn decode(beat_logits: &[f32], downbeat_logits: &[f32], fps: f64) -> Option<Decoded> {
    let frames = beat_logits.len().min(downbeat_logits.len());
    if frames == 0 {
        return None;
    }

    // beat_this postp_dbn: sigmoid in double, squeeze into (ε/2, 1−ε/2), then
    // the Böck two-column form [max(beat−downbeat, ε/2), downbeat].
    let squeeze =
        |logit: f32| sigmoid(f64::from(logit)) * (1.0 - SQUEEZE_EPSILON) + SQUEEZE_EPSILON / 2.0;
    let mut beat_activation = vec![0.0; frames];
    let mut downbeat_activation = vec![0.0; frames];
    for i in 0..frames {
        let beat = squeeze(beat_logits[i]);
        let down = squeeze(downbeat_logits[i]);
        beat_activation[i] = (beat - down).max(SQUEEZE_EPSILON / 2.0);
        downbeat_activation[i] = down;
    }

    // threshold_activations: keep the first..last frame where any column
    // clears the threshold; frame indices shift back by `first` at the end.
    let active = |i: usize| {
        beat_activation[i] >= ACTIVATION_THRESHOLD || downbeat_activation[i] >= ACTIVATION_THRESHOLD
    };
    let first = (0..frames).find(|&i| active(i))?;
    let last = (0..frames).rev().find(|&i| active(i))? + 1;
    let beat_window = &beat_activation[first..last];
    let downbeat_window = &downbeat_activation[first..last];

    // Observation log densities, one row per frame: none / beat / downbeat.
    let count = beat_window.len();
    let lambda = OBSERVATION_LAMBDA as f64;
    let mut log_none = vec![0.0; count];
    let mut log_beat = vec![0.0; count];
    let mut log_down = vec![0.0; count];
    for i in 0..count {
        log_none[i] = ((1.0 - beat_window[i] - downbeat_window[i]) / (lambda - 1.0)).ln();
        log_beat[i] = beat_window[i].ln();
        log_down[i] = downbeat_window[i].ln();
    }

    let min_interval = 60.0 * fps / MAX_BPM;
    let max_interval = 60.0 * fps / MIN_BPM;
    let mut best: Option<(BarHmm, Vec<u16>, f64)> = None;
    for bar_length in BAR_LENGTHS {
        let hmm = BarHmm::new(bar_length, min_interval, max_interval, NUM_TEMPI);
        let Some((path, log_probability)) = hmm.viterbi(&log_none, &log_beat, &log_down) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(_, _, best)| log_probability > *best)
        {
            best = Some((hmm, path, log_probability));
        }
    }
    let (hmm, path, log_probability) = best?;

    // correct=True: within each stretch of beat-range states, the beat sits on
    // the frame with the highest activation of EITHER column (madmom argmaxes
    // the flattened (T, 2) array).
    let mut beat_frames = Vec::new();
    let mut region_start = None;
    let in_beat_range = |t: usize| t < count && hmm.observation[usize::from(path[t])] >= 1;
    for t in 0..=count {
        match (in_beat_range(t), region_start) {
            (true, None) => region_start = Some(t),
            (false, Some(start)) => {
                let mut peak = start;
                let mut peak_value = f64::NEG_INFINITY;
                for i in start..t {
                    if beat_window[i] > peak_value {
                        peak_value = beat_window[i];
                        peak = i;
                    }
                    if downbeat_window[i] > peak_value {
                        peak_value = downbeat_window[i];
                        peak = i;
                    }
                }
                beat_frames.push(peak);
                region_start = None;
            }
            _ => {}
        }
    }
    if beat_frames.is_empty() {
        return None;
    }

    Some(Decoded {
        beat_times: beat_frames
            .iter()
            .map(|&frame| (frame + first) as f64 / fps)
            .collect(),
        beat_numbers: beat_frames
            .iter()
            .map(|&frame| hmm.positions[usize::from(path[frame])] as usize + 1)
            .collect(),
        bar_length: hmm.bar_length,
        log_probability,
    })
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// State space + transitions for one bar length. States chain linearly inside
/// a beat (`prev = state − 1`, probability 1), so back-pointers are stored
/// only for the first states of each beat — the one place with a real choice.
/// madmom's dense (frames × states) pointer matrix is ~100 MB per bar on a
/// 3-minute track; this is ~3 MB.
#[derive(Debug)]
struct BarHmm {
    bar_length: usize,
    state_count: usize,
    /// Fractional bar position of each state, in `0..bar_length`.
    positions: Vec<f64>,
    /// 0 = no-beat density, 1 = beat, 2 = downbeat.
    observation: Vec<u8>,
    /// Slot into the incoming tables for first states, `None` for chain states.
    first_slot: Vec<i32>,
    /// Flattened incoming tables — slot `i` owns `slot_start[i]..slot_start[i+1]`
    /// (predecessor ids ascending, matching madmom's CSR order).
    slot_start: Vec<usize>,
    incoming_previous: Vec<u16>,
    incoming_log_probability: Vec<f64>,
}

impl BarHmm {
    fn new(bar_length: usize, min_interval: f64, max_interval: f64, num_tempi: usize) -> Self {
        let intervals = beat_intervals(min_interval, max_interval, num_tempi);
        let states_per_beat: usize = intervals.iter().sum();
        let state_count = states_per_beat * bar_length;
        assert!(
            state_count <= usize::from(u16::MAX),
            "back-pointers are u16 — widen them before widening the state space"
        );

        let mut positions = vec![0.0; state_count];
        let mut first_states = Vec::with_capacity(bar_length);
        let mut last_states = Vec::with_capacity(bar_length);
        let mut offset = 0;
        for beat in 0..bar_length {
            let mut firsts = Vec::with_capacity(intervals.len());
            let mut lasts = Vec::with_capacity(intervals.len());
            for &interval in &intervals {
                firsts.push(offset);
                lasts.push(offset + interval - 1);
                // np.linspace(0, 1, i, endpoint=False) is k · (1/i) in float64
                // — reproduce the multiplication, not k/i, so the
                // observation-border comparison below matches bit for bit.
                let step = 1.0 / interval as f64;
                for k in 0..interval {
                    positions[offset + k] = beat as f64 + k as f64 * step;
                }
                offset += interval;
            }
            first_states.push(firsts);
            last_states.push(lasts);
        }

        // RNNDownBeatTrackingObservationModel pointers.
        let border = 1.0 / OBSERVATION_LAMBDA as f64;
        let observation = positions
            .iter()
            .map(|&position| {
                if position < border {
                    2
                } else if position % 1.0 < border {
                    1
                } else {
                    0
                }
            })
            .collect();

        // BarTransitionModel: at each beat boundary, last states of the
        // previous beat connect to first states of this beat with
        // exp(−λ·|ratio − 1|), floored at np.spacing(1) and row-normalized
        // over the destination intervals.
        let mut first_slot = vec![-1i32; state_count];
        let mut slot_start = vec![0usize];
        let mut incoming_previous = Vec::new();
        let mut incoming_log_probability = Vec::new();
        for (beat, firsts) in first_states.iter().enumerate() {
            let from_beat = (beat + bar_length - 1) % bar_length;
            let from_states = &last_states[from_beat];
            // Row normalization runs over the FROM interval's outgoing
            // probabilities, exactly as madmom builds the matrix.
            let mut probability = vec![vec![0.0; intervals.len()]; intervals.len()];
            for (i, &from_interval) in intervals.iter().enumerate() {
                let mut row_sum = 0.0;
                for (j, &to_interval) in intervals.iter().enumerate() {
                    let ratio = to_interval as f64 / from_interval as f64;
                    let mut p = (-TRANSITION_LAMBDA * (ratio - 1.0).abs()).exp();
                    if p <= TRANSITION_FLOOR {
                        p = 0.0;
                    }
                    probability[i][j] = p;
                    row_sum += p;
                }
                for value in &mut probability[i] {
                    if *value != 0.0 {
                        *value /= row_sum;
                    }
                }
            }
            for (j, &to_state) in firsts.iter().enumerate() {
                first_slot[to_state] = (slot_start.len() - 1) as i32;
                for (i, &from_state) in from_states.iter().enumerate() {
                    if probability[i][j] != 0.0 {
                        incoming_previous.push(from_state as u16);
                        incoming_log_probability.push(probability[i][j].ln());
                    }
                }
                slot_start.push(incoming_previous.len());
            }
        }

        Self {
            bar_length,
            state_count,
            positions,
            observation,
            first_slot,
            slot_start,
            incoming_previous,
            incoming_log_probability,
        }
    }

    /// Exact Viterbi, log domain, uniform initial distribution. Returns the
    /// state path and the final best log-probability; `None` when the
    /// activations support no path, mirroring madmom's empty answer.
    fn viterbi(
        &self,
        log_none: &[f64],
        log_beat: &[f64],
        log_down: &[f64],
    ) -> Option<(Vec<u16>, f64)> {
        let frames = log_none.len();
        if frames == 0 {
            return None;
        }
        let slots = self.slot_start.len() - 1;
        let mut previous = vec![-(self.state_count as f64).ln(); self.state_count];
        let mut current = vec![0.0; self.state_count];
        // Back-pointers only where a choice exists (first states), one u16 per
        // slot per frame.
        let mut back_pointers = vec![0u16; frames * slots];

        for t in 0..frames {
            let row = &mut back_pointers[t * slots..(t + 1) * slots];
            self.step(
                &previous,
                &mut current,
                row,
                [log_none[t], log_beat[t], log_down[t]],
            );
            std::mem::swap(&mut previous, &mut current);
        }

        let (best_state, best_log_probability) = previous.iter().enumerate().fold(
            (0usize, f64::NEG_INFINITY),
            |(best_state, best), (state, &value)| {
                if value > best {
                    (state, value)
                } else {
                    (best_state, best)
                }
            },
        );
        if best_log_probability == f64::NEG_INFINITY {
            return None;
        }

        let mut path = vec![0u16; frames];
        let mut state = best_state;
        for t in (0..frames).rev() {
            path[t] = state as u16;
            let slot = self.first_slot[state];
            state = if slot < 0 {
                state - 1
            } else {
                usize::from(back_pointers[t * slots + slot as usize])
            };
        }
        Some((path, best_log_probability))
    }

    /// Shared exact transition step; storage never changes tie ordering or
    /// the global forward state carried between observations.
    fn step(&self, previous: &[f64], current: &mut [f64], row: &mut [u16], density: [f64; 3]) {
        for state in 0..self.state_count {
            let slot = self.first_slot[state];
            let value = if slot < 0 {
                previous[state - 1]
            } else {
                let slot = slot as usize;
                let mut value = f64::NEG_INFINITY;
                let mut best_previous = 0u16;
                for k in self.slot_start[slot]..self.slot_start[slot + 1] {
                    let candidate = previous[usize::from(self.incoming_previous[k])]
                        + self.incoming_log_probability[k];
                    if candidate > value {
                        value = candidate;
                        best_previous = self.incoming_previous[k];
                    }
                }
                row[slot] = best_previous;
                value
            };
            current[state] = value + density[usize::from(self.observation[state])];
        }
    }
}

/// madmom's `BeatStateSpace` intervals: linear integer intervals unless
/// `num_tempi` is tighter, in which case they are log-spaced and de-duplicated
/// until exactly `num_tempi` distinct values survive. At the shipping 50 fps
/// the range is 14…55 (42 values < 60), so the log branch never runs there —
/// it exists because `fps` is a caller parameter.
fn beat_intervals(min_interval: f64, max_interval: f64, num_tempi: usize) -> Vec<usize> {
    let lo = min_interval.round().max(1.0) as usize;
    let hi = max_interval.round().max(lo as f64) as usize;
    let linear = (lo..=hi).collect::<Vec<_>>();
    if num_tempi >= linear.len() {
        return linear;
    }
    let log_min = min_interval.log2();
    let log_max = max_interval.log2();
    let mut target = num_tempi;
    loop {
        let step = (log_max - log_min) / (target - 1) as f64;
        let mut spaced = (0..target)
            .map(|i| 2.0_f64.powf(log_min + i as f64 * step).round() as usize)
            .collect::<Vec<_>>();
        spaced.sort_unstable();
        spaced.dedup();
        if spaced.len() >= num_tempi {
            return spaced;
        }
        target += 1;
    }
}

#[cfg(test)]
mod tests {
    use ferment_engine_test_fixtures as fixtures;
    use serde::Deserialize;

    use super::*;

    #[test]
    fn paged_viterbi_matches_the_global_path_across_disk_pages() {
        for (frames, period, meter) in [(129, 25, 4), (8203, 27, 3), (8197, 23, 4)] {
            let (mut beat, mut down) = synthetic_logits(frames, period, meter);
            // Silence at edges and a weak middle passage exercise global
            // thresholding and the inferred path through page boundaries.
            for i in 2100..3100.min(frames) {
                beat[i] = -6.;
                down[i] = -7.;
            }
            let expected = decode(&beat, &down, 50.);
            let mut scratch = || {
                tempfile::tempfile()
                    .map_err(|e| ferment_engine_core::EngineError::internal(e.to_string()))
            };
            let mut logits = super::super::storage::Logits::new(scratch().unwrap()).unwrap();
            for (b, d) in beat.chunks(157).zip(down.chunks(157)) {
                logits.append(b, d).unwrap();
            }
            let actual = decode_paged(&mut logits, &mut scratch, 50., 4096, &|| Ok(())).unwrap();
            assert_eq!(actual, expected, "frames={frames}, meter={meter}");
            assert!(logits.resident_bytes() <= 64 * 1024);
            assert!(decode_paged(&mut logits, &mut scratch, 50., 1, &|| Ok(())).is_err());
            assert!(
                decode_paged(&mut logits, &mut scratch, 50., 4096, &|| Err(
                    ferment_engine_core::EngineError::Cancelled
                ))
                .is_err()
            );
        }
    }

    /// A perfect click lattice: beat pulse every `period` frames starting at
    /// frame 10, downbeat pulse every `beats_per_bar`-th beat. ±8 logits are
    /// far past sigmoid saturation, so the observation is unambiguous.
    fn synthetic_logits(
        frames: usize,
        period: usize,
        beats_per_bar: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut beat = vec![-8.0; frames];
        let mut down = vec![-8.0; frames];
        let mut index = 0;
        let mut frame = 10;
        while frame < frames {
            beat[frame] = 8.0;
            if index % beats_per_bar == 0 {
                down[frame] = 8.0;
            }
            index += 1;
            frame += period;
        }
        (beat, down)
    }

    #[test]
    fn decodes_steady_four_four() {
        // 120 bpm at 50 fps → 25 frames per beat, 60 s of it.
        let (beat, down) = synthetic_logits(3_000, 25, 4);
        let decoded = decode(&beat, &down, 50.0).expect("decode");

        assert_eq!(decoded.bar_length, 4);
        assert!(
            decoded.beat_times.len() > 110,
            "{}",
            decoded.beat_times.len()
        );
        for (time, number) in decoded.beat_times.iter().zip(&decoded.beat_numbers) {
            let offset = (time * 50.0 - 10.0) % 25.0;
            let distance = offset.abs().min(25.0 - offset.abs());
            assert!(distance <= 1.01, "beat off the pulse lattice at {time}s");
            assert!((1..=4).contains(number));
        }

        let downbeats = decoded.downbeat_times();
        assert!(downbeats.len() > 25, "{}", downbeats.len());
        for time in downbeats {
            let frame = (time * 50.0).round() as usize;
            assert_eq!(
                down[frame], 8.0,
                "downbeat at {time}s not on a downbeat pulse"
            );
        }
    }

    #[test]
    fn picks_three_beat_bar() {
        let (beat, down) = synthetic_logits(3_000, 25, 3);
        assert_eq!(decode(&beat, &down, 50.0).expect("decode").bar_length, 3);
    }

    /// The parity fixture is a mid-tempo track, so it decodes identically
    /// under a narrower tempo range — it cannot pin 55/215. These two click
    /// tracks can: 54 frames per beat is 55.6 bpm and 14 frames is 214.3 bpm,
    /// the slowest and fastest pulses the state space can represent at 50 fps.
    /// Narrow the range at all and neither lattice is reachable.
    #[test]
    fn the_tempo_range_reaches_its_own_edges() {
        for period in [54_usize, 14] {
            let (beat, down) = synthetic_logits(6_000, period, 4);
            let decoded = decode(&beat, &down, 50.0).expect("decode");
            assert!(decoded.beat_times.len() > 2, "period={period}");
            let spacing = (decoded.beat_times[decoded.beat_times.len() - 1]
                - decoded.beat_times[0])
                / (decoded.beat_times.len() - 1) as f64;
            let expected = period as f64 / 50.0;
            assert!(
                (spacing - expected).abs() < 0.01,
                "period {period} frames ({:.1} bpm) decoded at {:.1} bpm",
                60.0 / expected,
                60.0 / spacing
            );
        }
    }

    #[test]
    fn silence_decodes_to_nothing() {
        // Every activation sits under the 0.05 threshold — the trim leaves no
        // frames and the decoder answers None, never a fabricated grid.
        let silent = vec![-8.0f32; 3_000];
        assert_eq!(decode(&silent, &silent, 50.0), None);
    }

    #[test]
    fn empty_logits_decode_to_nothing() {
        assert_eq!(decode(&[], &[], 50.0), None);
    }

    /// The shipping frame rate never reaches the log-spaced branch, and the
    /// branch that never runs is the branch nobody checks.
    #[test]
    fn shipping_intervals_are_linear_and_a_wide_range_is_log_spaced() {
        let shipping = beat_intervals(60.0 * 50.0 / MAX_BPM, 60.0 * 50.0 / MIN_BPM, NUM_TEMPI);
        assert_eq!(shipping.first().copied(), Some(14));
        assert_eq!(shipping.last().copied(), Some(55));
        assert_eq!(shipping.len(), 42, "42 < 60 tempi — no log spacing");

        // 200 fps spans 56…218, far past 60 tempi.
        let wide = beat_intervals(60.0 * 200.0 / MAX_BPM, 60.0 * 200.0 / MIN_BPM, NUM_TEMPI);
        assert_eq!(
            wide.len(),
            NUM_TEMPI,
            "log spacing lands exactly on num_tempi"
        );
        assert!(
            wide.windows(2).all(|pair| pair[0] < pair[1]),
            "sorted, distinct"
        );
        assert_eq!(wide.first().copied(), Some(56));
        assert_eq!(wide.last().copied(), Some(218));
    }

    /// The fallback door, both arms: a decode that cannot reach two beats hands
    /// the whole answer back to the minimal peak-pick rather than publishing a
    /// one-beat grid or an empty one.
    #[test]
    fn process_falls_back_to_the_minimal_postprocess_when_the_decode_is_too_thin() {
        // One isolated spike: the activation trim leaves a single frame, so the
        // decode yields exactly one beat.
        let mut beat = vec![-8.0f32; 400];
        beat[100] = 8.0;
        let down = vec![-8.0f32; 400];
        assert_eq!(
            decode(&beat, &down, 50.0).expect("decode").beat_times.len(),
            1
        );
        assert_eq!(
            process(&beat, &down, 50.0, None),
            postprocess::process(&beat, &down, 50.0, None)
        );
        assert_eq!(process(&beat, &down, 50.0, None).beats, vec![2.0]);

        // Nothing at all: `decode` answers None and the fallback answers empty.
        let silent = vec![-8.0f32; 400];
        assert_eq!(decode(&silent, &silent, 50.0), None);
        assert!(process(&silent, &silent, 50.0, None).beats.is_empty());
    }

    /// What the DBN buys over the peak-pick: two hits 4 s apart are not a
    /// 15 bpm pulse, they are a gap. The bar-pointer HMM has to walk a
    /// continuous bar across it at a tempo inside 55–215 bpm, so it fills in
    /// the beats the peak-picker simply does not see.
    #[test]
    fn the_bar_pointer_hmm_walks_a_continuous_bar_across_a_gap() {
        let mut beat = vec![-8.0f32; 400];
        beat[100] = 8.0;
        beat[300] = 8.0;
        let down = vec![-8.0f32; 400];

        let minimal = postprocess::process(&beat, &down, 50.0, None);
        assert_eq!(minimal.beats, vec![2.0, 6.0], "the peak-pick sees two hits");

        let decoded = decode(&beat, &down, 50.0).expect("decode");
        assert!(decoded.beat_times.len() > 2, "{:?}", decoded.beat_times);
        assert_eq!(decoded.beat_times.first().copied(), Some(2.0));
        assert_eq!(decoded.beat_times.last().copied(), Some(6.0));
        let bpm = 60.0 * (decoded.beat_times.len() - 1) as f64 / 4.0;
        assert!(
            (MIN_BPM..=MAX_BPM).contains(&bpm),
            "interpolated tempo {bpm} left the DBN's range"
        );
    }

    // MARK: - madmom parity

    #[derive(Deserialize)]
    struct LogitsFixture {
        #[serde(rename = "beatLogits")]
        beat_logits: Vec<f32>,
        #[serde(rename = "downbeatLogits")]
        downbeat_logits: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct ExpectedFixture {
        beats: Vec<f64>,
        downbeats: Vec<f64>,
    }

    /// The port's ground truth: real ensemble logits (fp32 lane) decoded by
    /// madmom's `DBNDownBeatTrackingProcessor`. This decoder must reproduce
    /// every beat and downbeat exactly.
    #[test]
    fn matches_madmom_on_real_logits() {
        let Some(logits_path) = fixtures::path("BeatThis/dbn-parity-logits.json") else {
            return;
        };
        let logits: LogitsFixture =
            serde_json::from_slice(&std::fs::read(logits_path).expect("read logits fixture"))
                .expect("parse logits fixture");
        let expected: ExpectedFixture = serde_json::from_slice(&fixtures::require_bytes(
            "BeatThis/dbn-parity-expected.json",
        ))
        .expect("parse expected fixture");

        let decoded = decode(&logits.beat_logits, &logits.downbeat_logits, 50.0).expect("decode");

        assert_eq!(decoded.beat_times.len(), expected.beats.len());
        for (got, want) in decoded.beat_times.iter().zip(&expected.beats) {
            assert!((got - want).abs() < 1e-6, "beat {got} != {want}");
        }

        let downbeats = decoded.downbeat_times();
        assert_eq!(downbeats.len(), expected.downbeats.len());
        for (got, want) in downbeats.iter().zip(&expected.downbeats) {
            assert!((got - want).abs() < 1e-6, "downbeat {got} != {want}");
        }
    }
}
