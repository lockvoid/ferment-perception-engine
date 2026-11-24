//! AED-driven post-denoise keep mask.

use std::f32::consts::PI;

use ferment_engine_core::AudioEventSegment;

/// Keep only the requested event segments, with five-millisecond raised-cosine
/// edges. Touching or overlapping segments remain at full gain internally.
pub fn apply_events_mask(
    samples: &mut [f32],
    keep_segments: &[AudioEventSegment],
    sample_rate: u32,
) {
    apply_events_mask_window(samples, 0, samples.len(), keep_segments, sample_rate);
}

/// Apply exactly the full-buffer mask to a bounded output window. Edge
/// coordinates and overlapping-segment tests use the complete file's sample
/// positions, so a synthesis chunk boundary cannot restart a fade.
pub(super) fn apply_events_mask_window(
    samples: &mut [f32],
    offset: usize,
    total_samples: usize,
    keep_segments: &[AudioEventSegment],
    sample_rate: u32,
) {
    if samples.is_empty() {
        return;
    }
    if keep_segments.is_empty() {
        samples.fill(0.0);
        return;
    }
    let fade_samples = ((0.005 * f64::from(sample_rate)) as usize).max(1);
    let end_window = offset.saturating_add(samples.len());
    let spans = keep_segments
        .iter()
        .map(|segment| {
            (
                seconds_to_sample(segment.start_time, sample_rate),
                seconds_to_sample(segment.end_time(), sample_rate),
            )
        })
        .collect::<Vec<_>>();
    let mut gain = vec![0.0_f32; samples.len()];
    for &(start, end) in &spans {
        let lo = start.min(total_samples).max(offset);
        let hi = end.min(total_samples).min(end_window);
        if lo < hi {
            gain[lo - offset..hi - offset].fill(1.0);
        }
    }
    let in_kept_span = |index| {
        spans
            .iter()
            .any(|&(start, end)| start <= index && index < end)
    };
    for &(start, end) in &spans {
        for distance in 1..=fade_samples {
            let Some(index) = start.checked_sub(distance) else {
                continue;
            };
            if index >= total_samples {
                continue;
            }
            if in_kept_span(index) {
                break;
            }
            if index < offset || index >= end_window {
                continue;
            }
            let phase = (fade_samples - distance + 1) as f32 / (fade_samples + 1) as f32;
            gain[index - offset] = gain[index - offset].max(0.5 - 0.5 * (PI * phase).cos());
        }
        for distance in 0..fade_samples {
            let index = end.saturating_add(distance);
            if index >= total_samples {
                break;
            }
            if in_kept_span(index) {
                break;
            }
            if index < offset || index >= end_window {
                continue;
            }
            let phase = (fade_samples - distance) as f32 / (fade_samples + 1) as f32;
            gain[index - offset] = gain[index - offset].max(0.5 - 0.5 * (PI * phase).cos());
        }
    }
    for (sample, gain) in samples.iter_mut().zip(gain) {
        *sample *= gain;
    }
}

fn seconds_to_sample(seconds: f64, sample_rate: u32) -> usize {
    (seconds.max(0.0) * f64::from(sample_rate)) as usize
}

#[cfg(test)]
mod tests {
    use ferment_engine_core::{AudioEvent, AudioEventSegment};

    use super::*;

    const SAMPLE_RATE: u32 = 48_000;
    const FADE_SAMPLES: usize = 240;

    fn unit_input(seconds: f64) -> Vec<f32> {
        vec![1.0; (seconds * f64::from(SAMPLE_RATE)) as usize]
    }

    fn keep(segments: &[(f64, f64)]) -> Vec<AudioEventSegment> {
        segments
            .iter()
            .map(|&(start, end)| AudioEventSegment {
                event: AudioEvent::Speech,
                start_time: start,
                duration: end - start,
                confidence: 1.0,
            })
            .collect()
    }

    // DeepFilterNetEventsMaskTests.swift.

    #[test]
    fn test_no_speech_segments_mutes_entire_buffer() {
        let mut samples = unit_input(1.0);
        apply_events_mask(&mut samples, &keep(&[]), SAMPLE_RATE);
        assert!(samples.iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn test_speech_segment_interior_is_unchanged() {
        let mut samples = unit_input(2.0);
        apply_events_mask(&mut samples, &keep(&[(0.5, 1.5)]), SAMPLE_RATE);
        assert!((samples[SAMPLE_RATE as usize] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gap_between_segments_is_silenced() {
        let mut samples = unit_input(3.0);
        apply_events_mask(&mut samples, &keep(&[(0.0, 0.5), (2.5, 3.0)]), SAMPLE_RATE);
        assert!((samples[(1.5 * f64::from(SAMPLE_RATE)) as usize]).abs() < 1e-6);
    }

    #[test]
    fn test_rising_edge_is_monotonic_raised_cosine() {
        let mut samples = unit_input(1.0);
        apply_events_mask(&mut samples, &keep(&[(0.5, 0.9)]), SAMPLE_RATE);
        let start = (0.5 * f64::from(SAMPLE_RATE)) as usize;
        let mut previous = -1.0_f32;
        for &gain in &samples[start - FADE_SAMPLES..start] {
            assert!(gain >= previous);
            assert!((0.0..=1.0).contains(&gain));
            previous = gain;
        }
        assert!((samples[start] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_falling_edge_is_monotonic_raised_cosine() {
        let mut samples = unit_input(1.0);
        apply_events_mask(&mut samples, &keep(&[(0.1, 0.5)]), SAMPLE_RATE);
        let end = (0.5 * f64::from(SAMPLE_RATE)) as usize;
        assert!((samples[end - 1] - 1.0).abs() < 1e-6);
        let mut previous = 2.0_f32;
        for &gain in &samples[end..end + FADE_SAMPLES] {
            assert!(gain <= previous);
            previous = gain;
        }
        assert!(samples[end + FADE_SAMPLES + 100].abs() < 1e-6);
    }

    #[test]
    fn test_adjacent_segments_keep_inner_boundary_at_full_gain() {
        let mut samples = unit_input(2.0);
        apply_events_mask(&mut samples, &keep(&[(0.5, 1.0), (1.0, 1.5)]), SAMPLE_RATE);
        let boundary = SAMPLE_RATE as usize;
        for offset in -100_isize..=100 {
            let gain = samples[boundary.wrapping_add_signed(offset)];
            assert!(
                (gain - 1.0).abs() < 1e-6,
                "inner offset {offset} faded to {gain}"
            );
        }
    }

    #[test]
    fn test_rising_edge_curve_matches_raised_cosine_formula() {
        let mut samples = unit_input(1.0);
        apply_events_mask(&mut samples, &keep(&[(0.5, 0.9)]), SAMPLE_RATE);
        let start = (0.5 * f64::from(SAMPLE_RATE)) as usize;
        for distance in 1..=FADE_SAMPLES {
            let phase = (FADE_SAMPLES - distance + 1) as f32 / (FADE_SAMPLES + 1) as f32;
            let expected = 0.5 - 0.5 * (PI * phase).cos();
            let actual = samples[start - distance];
            assert!(
                (actual - expected).abs() < 1e-6,
                "rising fade at distance {distance}: got {actual}, expected {expected}"
            );
        }
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use ferment_engine_core::AudioEvent;
    #[test]
    fn chunk_boundaries_preserve_the_absolute_raised_cosine_and_adjacent_unity_region() {
        let spans = [
            (24_000_usize, 48_000_usize),
            (48_000, 72_000),
            (96_000, 108_000),
        ];
        let segments = spans
            .iter()
            .map(|&(start, end)| AudioEventSegment {
                event: AudioEvent::Speech,
                start_time: start as f64 / 48_000.0,
                duration: (end - start) as f64 / 48_000.0,
                confidence: 1.0,
            })
            .collect::<Vec<_>>();
        let mut actual = vec![1.0; 120_000];
        for (chunk, block) in actual.chunks_mut(137).enumerate() {
            apply_events_mask_window(block, chunk * 137, 120_000, &segments, 48_000);
        }
        // Independent analytical ramps about the merged audible intervals;
        // 48000 is inside the joined interval, so no fade may appear there.
        for (index, value) in actual.iter().enumerate() {
            let mut expected = 0.0_f32;
            for (start, end) in [(24_000_usize, 72_000_usize), (96_000, 108_000)] {
                let gain = if (start..end).contains(&index) {
                    1.0
                } else if index < start && start - index <= 240 {
                    0.5 - 0.5
                        * (std::f32::consts::PI * (241 - (start - index)) as f32 / 241.0).cos()
                } else if index >= end && index - end < 240 {
                    0.5 - 0.5 * (std::f32::consts::PI * (240 - (index - end)) as f32 / 241.0).cos()
                } else {
                    0.0
                };
                expected = expected.max(gain);
            }
            assert!(
                (value - expected).abs() < 1e-6,
                "sample {index}: {value} vs {expected}"
            );
        }
    }
}
