//! RMS dB envelope, hop-aligned with the visualization mel spectrogram so
//! the two panels share a time axis. Mirrors
//! `librosa.feature.rms(y, frame_length=N_FFT, hop_length=HOP)` then
//! `librosa.amplitude_to_db(rms, ref=max)` clipped to `[-top_db, 0]`.

use super::mel::{FFT_SIZE, HOP_LENGTH, TOP_DB};

pub fn compute(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let frame_length = FFT_SIZE;
    let hop = HOP_LENGTH;
    let pad = frame_length / 2;

    // librosa center=True, pad_mode='reflect' (same as mel)
    let mut padded = vec![0.0f32; audio.len() + 2 * pad];
    let n = audio.len();
    for i in 0..pad {
        padded[pad - 1 - i] = audio[(i + 1).min(n - 1)];
    }
    padded[pad..pad + n].copy_from_slice(audio);
    for i in 0..pad {
        let src = (n as isize - 2 - i as isize).max(0) as usize;
        padded[pad + n + i] = audio[src];
    }

    let num_frames = if padded.len() >= frame_length {
        (padded.len() - frame_length) / hop + 1
    } else {
        0
    };
    if num_frames == 0 {
        return Vec::new();
    }

    let mut cumulative_energy = Vec::with_capacity(padded.len() + 1);
    cumulative_energy.push(0.0_f64);
    for sample in padded {
        cumulative_energy.push(
            cumulative_energy.last().copied().unwrap_or(0.0)
                + f64::from(sample) * f64::from(sample),
        );
    }
    let rms = (0..num_frames)
        .map(|index| {
            let start = index * hop;
            ((cumulative_energy[start + frame_length] - cumulative_energy[start])
                / frame_length as f64)
                .sqrt() as f32
        })
        .collect::<Vec<_>>();

    // amplitude_to_db: 20 * log10(rms / ref), ref = max(rms), clip to [-top_db, 0]
    let max_rms = rms.iter().copied().fold(0.0f32, f32::max);
    let amin: f32 = 1e-5;
    let reference = max_rms.max(amin);
    let log_ref = 20.0 * reference.log10();
    let lower_bound = -TOP_DB;

    rms.into_iter()
        .map(|value| (20.0 * value.max(amin).log10() - log_ref).max(lower_bound))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_yields_empty() {
        assert!(compute(&[]).is_empty());
    }

    #[test]
    fn silent_input_yields_zero_db_after_max_normalization() {
        // ref=max=amin, so 20·log10(rms/ref) = 0 across every frame.
        let audio = vec![0.0f32; 22_050];
        let db = compute(&audio);
        for &v in &db {
            assert!(v.abs() < 1e-3, "silent rms value {v} should be ~0 dB");
        }
    }
}
