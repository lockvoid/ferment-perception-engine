//! Pure BeatThis logits post-processing, separated from model inference.

use ferment_engine_core::BeatLatticeSegment;

use super::{
    meter::detect_meter,
    peaks::{deduplicate_peaks, find_peaks, pick_raw_downbeats},
    quantize::quantize_bounded,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessResult {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub quantized_beats: Vec<f64>,
    pub quantized_downbeats: Vec<f64>,
    pub quantized_bpm: Option<f64>,
    /// The warp lattice `quantized_beats` expands from.
    pub segments: Vec<BeatLatticeSegment>,
}

pub fn process(
    beat_logits: &[f32],
    downbeat_logits: &[f32],
    fps: f64,
    track_duration: Option<f64>,
) -> ProcessResult {
    if beat_logits.is_empty() {
        return empty_result();
    }
    let beat_frames = deduplicate_peaks(&find_peaks(beat_logits, 7));
    let downbeat_scores = beat_frames
        .iter()
        .map(|frame| {
            downbeat_logits
                .get(*frame as usize)
                .copied()
                .unwrap_or(-100.0)
        })
        .collect::<Vec<_>>();
    finish(
        beat_frames,
        downbeat_scores,
        fps,
        track_duration,
        usize::MAX,
    )
    .expect("unbounded peak result")
}

fn finish(
    beat_frames: Vec<f64>,
    downbeat_scores: Vec<f32>,
    fps: f64,
    track_duration: Option<f64>,
    limit: usize,
) -> ferment_engine_core::Result<ProcessResult> {
    let beat_times = beat_frames
        .iter()
        .map(|frame| frame / fps)
        .collect::<Vec<_>>();
    if beat_times.len() < 2 {
        return Ok(ProcessResult {
            beats: beat_times,
            ..empty_result()
        });
    }
    let raw_downbeats = pick_raw_downbeats(&beat_times, &downbeat_scores, 3);
    let meter = detect_meter(&downbeat_scores);
    let quantized = quantize_bounded(
        &beat_times,
        &raw_downbeats,
        meter,
        track_duration,
        None,
        limit,
    )?;
    Ok(ProcessResult {
        beats: beat_times,
        downbeats: raw_downbeats,
        quantized_beats: quantized.beats,
        quantized_downbeats: quantized.downbeats,
        quantized_bpm: quantized.bpm,
        segments: quantized.segments,
    })
}

/// Same pool-max and consecutive-index mean, reading a bounded neighborhood.
pub(super) fn process_paged(
    logits: &mut super::storage::Logits,
    fps: f64,
    duration: Option<f64>,
    limit: usize,
    check: &dyn Fn() -> ferment_engine_core::Result<()>,
) -> ferment_engine_core::Result<ProcessResult> {
    let mut peaks = Vec::new();
    let mut group: Option<(usize, f64, usize)> = None;
    let mut emit = |sum: f64, count: usize| {
        if peaks.len() >= limit {
            return Err(ferment_engine_core::EngineError::invalid_config(
                "Beat event result quota exceeded",
            ));
        }
        peaks.push(sum / count as f64);
        Ok(())
    };
    for i in 0..logits.len() {
        if i.is_multiple_of(1024) {
            check()?;
        }
        let value = logits.get(i)?[0];
        if value <= 0. {
            continue;
        }
        let mut is_max = true;
        for j in i.saturating_sub(3)..(i + 4).min(logits.len()) {
            if logits.get(j)?[0] > value {
                is_max = false;
                break;
            }
        }
        if !is_max {
            continue;
        }
        group = Some(match group.take() {
            Some((last, sum, count)) if i - last <= 1 => (i, sum + i as f64, count + 1),
            Some((_, sum, count)) => {
                emit(sum, count)?;
                (i, i as f64, 1)
            }
            None => (i, i as f64, 1),
        });
    }
    if let Some((_, sum, count)) = group {
        emit(sum, count)?;
    }
    let scores = peaks
        .iter()
        .map(|frame| logits.get(*frame as usize).map(|row| row[1]))
        .collect::<ferment_engine_core::Result<Vec<_>>>()?;
    check()?;
    finish(peaks, scores, fps, duration, limit)
}

fn empty_result() -> ProcessResult {
    ProcessResult {
        beats: Vec::new(),
        downbeats: Vec::new(),
        quantized_beats: Vec::new(),
        quantized_downbeats: Vec::new(),
        quantized_bpm: None,
        segments: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn synthetic_logits(
        beat_count: usize,
        beat_interval: usize,
        meter: usize,
        downbeat_phase: usize,
        downbeat_strength: f32,
        non_downbeat_strength: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let frame_count = beat_count * beat_interval + 10;
        let mut beats = vec![-5.0; frame_count];
        let mut downbeats = vec![-5.0; frame_count];
        for beat in 0..beat_count {
            let frame = beat * beat_interval;
            beats[frame] = 5.0;
            downbeats[frame] = if (beat + meter - downbeat_phase).is_multiple_of(meter) {
                downbeat_strength
            } else {
                non_downbeat_strength
            };
        }
        (beats, downbeats)
    }

    fn standard_logits(beat_count: usize, meter: usize) -> (Vec<f32>, Vec<f32>) {
        synthetic_logits(beat_count, 21, meter, 0, 10.0, -10.0)
    }

    #[test]
    fn test_basic_4_4() {
        let (beats, downbeats) = standard_logits(32, 4);
        let result = process(&beats, &downbeats, 50.0, None);
        assert!(result.beats.len().abs_diff(32) <= 1);
        assert!(result.downbeats.len().abs_diff(8) <= 1);
        assert!(result.quantized_beats.len().abs_diff(32) <= 1);
        assert!(result.quantized_downbeats.len().abs_diff(8) <= 1);
    }

    #[test]
    fn test_basic_3_4() {
        let (beats, downbeats) = standard_logits(24, 3);
        let result = process(&beats, &downbeats, 50.0, None);
        assert!(result.beats.len().abs_diff(24) <= 1);
        assert!(result.downbeats.len().abs_diff(8) <= 1);
        assert!(result.quantized_beats.len().abs_diff(24) <= 1);
        assert!(result.quantized_downbeats.len().abs_diff(8) <= 1);
    }

    #[test]
    fn test_downbeats_are_subset_of_beats() {
        let (beats, downbeats) = standard_logits(32, 4);
        let result = process(&beats, &downbeats, 50.0, None);
        let beat_set = result
            .beats
            .iter()
            .map(|value| (value * 1_000.0) as i64)
            .collect::<HashSet<_>>();
        assert!(
            result
                .downbeats
                .iter()
                .all(|value| beat_set.contains(&((*value * 1_000.0) as i64)))
        );
        let quantized_set = result
            .quantized_beats
            .iter()
            .map(|value| (value * 1_000.0) as i64)
            .collect::<HashSet<_>>();
        assert!(
            result
                .quantized_downbeats
                .iter()
                .all(|value| quantized_set.contains(&((*value * 1_000.0) as i64)))
        );
    }

    #[test]
    fn test_downbeats_are_regularly_spaced() {
        let (beats, downbeats) = standard_logits(32, 4);
        let result = process(&beats, &downbeats, 50.0, None);
        assert!(result.quantized_downbeats.len() >= 3);
        let intervals = result
            .quantized_downbeats
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect::<Vec<_>>();
        let mut sorted = intervals.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        assert!(intervals.iter().all(|value| (value - median).abs() <= 0.05));
    }

    #[test]
    fn test_downbeat_phase_is_preserved() {
        let (beats, downbeats) = synthetic_logits(32, 21, 4, 2, 10.0, -10.0);
        let result = process(&beats, &downbeats, 50.0, None);
        let first_downbeat = result.quantized_downbeats[0];
        let index = result
            .quantized_beats
            .iter()
            .position(|beat| (beat - first_downbeat).abs() < 0.01)
            .expect("downbeat in beats");
        assert_eq!(index % 4, 2);
    }

    #[test]
    fn test_empty_input() {
        let result = process(&[], &[], 50.0, None);
        assert!(result.beats.is_empty());
        assert!(result.downbeats.is_empty());
    }

    #[test]
    fn test_single_beat() {
        let mut logits = vec![-5.0; 20];
        logits[10] = 5.0;
        let result = process(&logits, &logits, 50.0, None);
        assert_eq!(result.beats.len(), 1);
        assert!(result.downbeats.is_empty());
        assert!(result.quantized_beats.is_empty());
        assert!(result.quantized_downbeats.is_empty());
    }

    #[test]
    fn test_two_beats() {
        let mut beats = vec![-5.0; 50];
        beats[10] = 5.0;
        beats[30] = 5.0;
        let mut downbeats = vec![-5.0; 50];
        downbeats[10] = 10.0;
        downbeats[30] = -10.0;
        let result = process(&beats, &downbeats, 50.0, None);
        assert_eq!(result.beats.len(), 2);
        assert!(!result.downbeats.is_empty());
        assert!(result.quantized_beats.is_empty());
    }

    #[test]
    fn test_all_negative_logits() {
        let logits = vec![-1.0; 100];
        let result = process(&logits, &logits, 50.0, None);
        assert!(result.beats.is_empty());
        assert!(result.downbeats.is_empty());
        assert!(result.quantized_beats.is_empty());
        assert!(result.quantized_downbeats.is_empty());
    }

    #[test]
    fn test_very_long_track() {
        let (beats, downbeats) = synthetic_logits(2_400, 21, 4, 0, 10.0, -10.0);
        let result = process(&beats, &downbeats, 50.0, None);
        assert!(result.beats.len().abs_diff(2_400) <= 2);
        assert!(result.quantized_downbeats.len().abs_diff(600) <= 2);
    }

    #[test]
    fn test_weak_signal() {
        let (beats, downbeats) = synthetic_logits(32, 21, 4, 0, 0.5, -0.5);
        assert!(!process(&beats, &downbeats, 50.0, None).beats.is_empty());
    }

    #[test]
    fn test_beat_logits_mismatch_downbeat_logits_length() {
        let beats = [5.0, -5.0, -5.0, -5.0, 5.0, -5.0, -5.0, -5.0, 5.0, -5.0];
        let downbeats = [10.0, -10.0, -10.0];
        assert!(!process(&beats, &downbeats, 50.0, None).beats.is_empty());
    }

    #[test]
    fn test_bpm_consistency() {
        let (beats, downbeats) = synthetic_logits(100, 21, 4, 0, 10.0, -10.0);
        let result = process(&beats, &downbeats, 50.0, None);
        let interval = (result.beats[result.beats.len() - 1] - result.beats[0])
            / (result.beats.len() - 1) as f64;
        assert!((60.0 / interval - 142.9).abs() <= 1.0);
    }

    #[test]
    fn test_quantized_beats_do_not_extrapolate_into_silence() {
        let (beats, downbeats) = standard_logits(32, 4);
        let duration = 20.0;
        let result = process(&beats, &downbeats, 50.0, Some(duration));
        let last_detected = result.beats[result.beats.len() - 1];
        let ibi = last_detected / (result.beats.len() - 1) as f64;
        assert!(result.quantized_beats[0] >= 0.0);
        assert!(result.quantized_beats[result.quantized_beats.len() - 1] <= last_detected + ibi);
        assert!(duration - result.quantized_beats[result.quantized_beats.len() - 1] > 5.0);
    }

    #[test]
    fn test_quantized_beats_fill_gaps_between_detected_beats() {
        let (mut beats, downbeats) = standard_logits(32, 4);
        beats[15 * 21] = -5.0;
        beats[16 * 21] = -5.0;
        let result = process(&beats, &downbeats, 50.0, None);
        assert!(result.beats.len().abs_diff(30) <= 1);
        assert!(result.quantized_beats.len() >= 31);
    }
}
