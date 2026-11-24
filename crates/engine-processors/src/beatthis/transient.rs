//! Transient-attack refinement for beat markers.

/// Broadband overlapping RMS envelope.
pub fn energy_envelope(audio: &[f32], hop: usize, window: usize) -> Vec<f32> {
    if hop == 0 || window == 0 || audio.len() < window {
        return Vec::new();
    }
    let count = (audio.len() - window) / hop + 1;
    (0..count)
        .map(|index| {
            (audio[index * hop..index * hop + window]
                .iter()
                .map(|sample| sample * sample)
                .sum::<f32>()
                / window as f32)
                .sqrt()
        })
        .collect()
}

/// Pull each beat to the foot of the strongest nearby positive log-energy
/// flux, with a maximum-movement guard and a final non-crossing clamp.
pub fn snap_to_attack_foot(
    beats: &[f64],
    energy: &[f32],
    hop_seconds: f64,
    search_seconds: f64,
    pre_roll: f64,
    flux_threshold: f32,
    max_move_seconds: f64,
) -> Vec<f64> {
    if energy.len() < 3 || hop_seconds <= 0.0 || beats.is_empty() {
        return beats.to_vec();
    }
    let mut flux = vec![0.0f32; energy.len()];
    for index in 1..energy.len() {
        flux[index] = ((energy[index] + 1.0e-6).ln() - (energy[index - 1] + 1.0e-6).ln()).max(0.0);
    }
    let window = (search_seconds / hop_seconds).floor().max(1.0) as usize;
    let last = energy.len() - 1;
    let mut output = beats.to_vec();
    for (beat_index, &beat) in beats.iter().enumerate() {
        let center = (beat / hop_seconds).round() as isize;
        if center < 0 || center as usize > last {
            continue;
        }
        let low = (center as usize).saturating_sub(window).max(1);
        let high = (center as usize + window).min(last);
        if low >= high {
            continue;
        }
        let peak = (low..=high)
            .max_by(|&left, &right| flux[left].total_cmp(&flux[right]))
            .unwrap_or(low);
        let peak_value = flux[peak];
        if peak_value <= 0.0 {
            continue;
        }
        let threshold = peak_value * flux_threshold;
        let mut foot = peak;
        while foot > low && flux[foot - 1] > threshold {
            foot -= 1;
        }
        let foot_time = (foot as f64 * hop_seconds - pre_roll).max(0.0);
        if (foot_time - beat).abs() <= max_move_seconds {
            output[beat_index] = foot_time;
        }
    }
    for index in 1..output.len() {
        output[index] = output[index].max(output[index - 1]);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENERGY: [f32; 10] = [0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 0.6, 0.3, 0.1, 0.1];
    const HOP: f64 = 0.01;

    fn snap(beats: &[f64], energy: &[f32], pre_roll: f64) -> Vec<f64> {
        snap_to_attack_foot(beats, energy, HOP, 0.06, pre_roll, 0.1, 0.05)
    }

    #[test]
    fn test_snap_pulls_beat_to_attack_start() {
        let output = snap_to_attack_foot(&[5.0 * HOP], &ENERGY, HOP, 0.06, 0.0, 0.1, 0.1);
        assert!((output[0] - 3.0 * HOP).abs() <= HOP / 2.0);
    }

    #[test]
    fn test_lead_shifts_earlier() {
        let output = snap_to_attack_foot(&[5.0 * HOP], &ENERGY, HOP, 0.06, 0.005, 0.1, 0.1);
        assert!((output[0] - (3.0 * HOP - 0.005)).abs() <= HOP / 4.0);
    }

    #[test]
    fn test_max_move_guard() {
        let beat = 8.0 * HOP;
        let output = snap_to_attack_foot(&[beat], &ENERGY, HOP, 0.1, 0.0, 0.1, 0.04);
        assert!((output[0] - beat).abs() < 1.0e-9);
    }

    #[test]
    fn test_no_transient_leaves_beat() {
        let beat = 10.0 * HOP;
        let output = snap(&[beat], &[0.2; 20], 0.0);
        assert!((output[0] - beat).abs() < 1.0e-9);
    }

    #[test]
    fn test_degenerate_inputs() {
        assert_eq!(snap(&[1.0], &[], 0.006), vec![1.0]);
        assert!(snap(&[], &ENERGY, 0.006).is_empty());
    }

    #[test]
    fn test_energy_envelope_value_and_length() {
        let audio = (0..4_000)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let envelope = energy_envelope(&audio, 64, 128);
        assert_eq!(envelope.len(), (audio.len() - 128) / 64 + 1);
        assert!((envelope[10] - 1.0).abs() < 1.0e-3);
        assert!(energy_envelope(&[0.0, 0.0, 0.0], 64, 128).is_empty());
    }

    #[test]
    fn test_energy_envelope_tracks_burst() {
        let mut audio = vec![0.0; 4_000];
        for (index, sample) in audio.iter_mut().enumerate().take(2_200).skip(2_000) {
            *sample = if index % 2 == 0 { 0.5 } else { -0.5 };
        }
        let envelope = energy_envelope(&audio, 64, 128);
        assert!((envelope[2_000 / 64] - 0.5).abs() < 0.05);
        assert!(envelope[0] < 0.01);
    }

    #[test]
    fn test_tiny_envelope_no_crash() {
        let output = snap_to_attack_foot(
            &[0.0, 2.0 * HOP],
            &[0.1, 0.9, 0.1],
            HOP,
            0.01,
            0.0,
            0.1,
            0.1,
        );
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_two_beats_stay_sorted() {
        let output =
            snap_to_attack_foot(&[5.0 * HOP, 6.0 * HOP], &ENERGY, HOP, 0.06, 0.0, 0.1, 0.1);
        assert!(output[0] <= output[1]);
    }
}
