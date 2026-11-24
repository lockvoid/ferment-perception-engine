//! Tempo of music loops: the Percival & Tzanetakis (2014) estimate and the
//! Font & Serra (ISMIR 2016) loop confidence.
//!
//! Processors only. Which files count as loops, which confidence accepts an
//! estimate and how it is combined with other detectors are decided by the
//! consumer.

mod confidence;
mod percival;

pub use confidence::{
    LoopBpmMatch, MAX_LOOP_BEATS, loop_bpm_confidence, loop_bpm_confidence_with_max_beats,
    match_loop_length,
};
pub use percival::{SAMPLE_RATE, percival_bpm};

#[cfg(test)]
mod test_signal {
    /// A loop of `beats` clicks at `bpm`, cut on the beat grid. Bars accent
    /// their downbeat and every beat carries an off-beat tick. A click is a
    /// 10 ms alternating-sign ramp built from plain f32 arithmetic and grid
    /// positions rounded half away from zero, so the same samples can be
    /// rebuilt outside Rust to compute reference values.
    pub(super) fn click_loop(bpm: f64, beats: usize, rate: u32) -> Vec<f32> {
        let period = 60.0 * f64::from(rate) / bpm;
        let mut samples = vec![0.0f32; (beats as f64 * period).round() as usize];
        let mut click = |at: f64, gain: f32| {
            let start = at.round() as usize;
            for k in 0..441usize {
                let Some(sample) = samples.get_mut(start + k) else {
                    break;
                };
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                *sample += gain * sign * (1.0 - k as f32 / 441.0);
            }
        };
        for beat in 0..beats {
            let at = beat as f64 * period;
            click(at, if beat % 4 == 0 { 1.0 } else { 0.6 });
            click(at + period / 2.0, 0.3);
        }
        samples
    }
}
