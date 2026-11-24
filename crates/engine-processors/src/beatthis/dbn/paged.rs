//! Exact global Viterbi with paged logits, back-pointers and reverse path.
//! Only state vectors and a few file pages reside in memory. Output has an
//! explicit event quota; exceeding it fails instead of truncating a grid.
use super::super::storage::{Array, Logits, Scratch};
use super::*;
use ferment_engine_core::{EngineError, Result};

fn activation(logits: [f32; 2]) -> [f64; 2] {
    let squeeze =
        |logit: f32| sigmoid(f64::from(logit)) * (1. - SQUEEZE_EPSILON) + SQUEEZE_EPSILON / 2.;
    let beat = squeeze(logits[0]);
    let down = squeeze(logits[1]);
    [(beat - down).max(SQUEEZE_EPSILON / 2.), down]
}

pub(in crate::beatthis) fn process(
    logits: &mut Logits,
    scratch: &mut Scratch<'_>,
    fps: f64,
    duration: Option<f64>,
    limit: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<ProcessResult> {
    let Some(decoded) =
        decode(logits, scratch, fps, limit, check)?.filter(|d| d.beat_times.len() >= 2)
    else {
        return postprocess::process_paged(logits, fps, duration, limit, check);
    };
    let support = decoded
        .beat_times
        .iter()
        .map(|time| {
            let [beat, down] = logits.get((time * fps).round() as usize)?;
            Ok(beat > 0. || down > 0.)
        })
        .collect::<Result<Vec<_>>>()?;
    finish(decoded, &support, duration, limit)
}

pub(in crate::beatthis) fn decode(
    logits: &mut Logits,
    scratch: &mut Scratch<'_>,
    fps: f64,
    max_events: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Option<Decoded>> {
    check()?;
    let mut first = None;
    let mut last = 0;
    for t in 0..logits.len() {
        if t.is_multiple_of(1024) {
            check()?;
        }
        if activation(logits.get(t)?)
            .iter()
            .any(|v| *v >= ACTIVATION_THRESHOLD)
        {
            first.get_or_insert(t);
            last = t + 1;
        }
    }
    let Some(first) = first else {
        return Ok(None);
    };
    let count = last - first;
    let mut best: Option<(BarHmm, Array<2>, f64)> = None;
    for bar in BAR_LENGTHS {
        let hmm = BarHmm::new(bar, 60. * fps / MAX_BPM, 60. * fps / MIN_BPM, NUM_TEMPI);
        let Some((path, probability)) = hmm.paged_path(logits, first..last, scratch, check)? else {
            continue;
        };
        if best.as_ref().is_none_or(|(_, _, p)| probability > *p) {
            best = Some((hmm, path, probability));
        }
    }
    let Some((hmm, mut path, log_probability)) = best else {
        return Ok(None);
    };
    let mut beat_times = Vec::new();
    let mut beat_numbers = Vec::new();
    // Correct each beat-range region to the first peak of either activation,
    // exactly as the flattened (T, 2) argmax in the resident implementation.
    let mut peak: Option<(usize, f64, u16)> = None;
    for t in 0..=count {
        if t.is_multiple_of(1024) {
            check()?;
        }
        let state = if t < count {
            u16::from_le_bytes(path.get(count - 1 - t)?)
        } else {
            0
        };
        if t < count && hmm.observation[usize::from(state)] >= 1 {
            for value in activation(logits.get(first + t)?) {
                if peak.is_none_or(|(_, best, _)| value > best) {
                    peak = Some((t, value, state));
                }
            }
        } else if let Some((frame, _, state)) = peak.take() {
            if beat_times.len() >= max_events {
                return Err(EngineError::invalid_config(
                    "Beat event result quota exceeded",
                ));
            }
            beat_times.push((frame + first) as f64 / fps);
            beat_numbers.push(hmm.positions[usize::from(state)] as usize + 1);
        }
    }
    check()?;
    Ok((!beat_times.is_empty()).then_some(Decoded {
        beat_times,
        beat_numbers,
        bar_length: hmm.bar_length,
        log_probability,
    }))
}

impl BarHmm {
    fn paged_path(
        &self,
        logits: &mut Logits,
        range: std::ops::Range<usize>,
        scratch: &mut Scratch<'_>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<(Array<2>, f64)>> {
        let count = range.len();
        let slots = self.slot_start.len() - 1;
        let mut back = Array::<2>::new(scratch()?)?;
        let mut previous = vec![-(self.state_count as f64).ln(); self.state_count];
        let mut current = vec![0.; self.state_count];
        let mut row = vec![0; slots];
        let mut encoded = vec![0; slots * 2];
        for t in range {
            if t.is_multiple_of(128) {
                check()?;
            }
            let [beat, down] = activation(logits.get(t)?);
            self.step(
                &previous,
                &mut current,
                &mut row,
                [
                    ((1. - beat - down) / (OBSERVATION_LAMBDA as f64 - 1.)).ln(),
                    beat.ln(),
                    down.ln(),
                ],
            );
            for (value, bytes) in row.iter().zip(encoded.as_chunks_mut::<2>().0) {
                *bytes = value.to_le_bytes();
            }
            back.append(&encoded)?;
            std::mem::swap(&mut previous, &mut current);
        }
        let (mut state, probability) = previous.iter().enumerate().fold(
            (0, f64::NEG_INFINITY),
            |(best_state, best), (state, &p)| {
                if p > best {
                    (state, p)
                } else {
                    (best_state, best)
                }
            },
        );
        if probability == f64::NEG_INFINITY {
            return Ok(None);
        }
        let mut path = Array::<2>::new(scratch()?)?;
        for t in (0..count).rev() {
            if t.is_multiple_of(1024) {
                check()?;
            }
            path.append(&(state as u16).to_le_bytes())?;
            let slot = self.first_slot[state];
            state = if slot < 0 {
                state - 1
            } else {
                usize::from(u16::from_le_bytes(back.get(t * slots + slot as usize)?))
            };
        }
        Ok(Some((path, probability)))
    }
}
