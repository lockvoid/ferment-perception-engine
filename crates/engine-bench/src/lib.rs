//! Non-shipping DSP benchmark support.

#[cfg(all(target_os = "macos", feature = "accel-fastpath"))]
#[allow(unsafe_code)]
pub mod accelerate;

use std::f32::consts::TAU;

/// Duration used by every published benchmark result.
pub const PINNED_SECONDS: usize = 60;

/// Deterministic music-like mono workload with tones, transients, and noise.
#[must_use]
pub fn signal(sample_rate: u32, seconds: usize, channel: usize) -> Vec<f32> {
    let sample_count = sample_rate as usize * seconds;
    let mut noise_state = 0x9e37_79b9_u32 ^ channel as u32;
    (0..sample_count)
        .map(|index| {
            noise_state = noise_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise = (noise_state as f32 / u32::MAX as f32 - 0.5) * 0.025;
            let time = index as f32 / sample_rate as f32;
            let base = if channel == 0 { 110.0 } else { 164.81 };
            let tone = 0.36 * (TAU * base * time).sin()
                + 0.18 * (TAU * base * 2.01 * time).sin()
                + 0.08 * (TAU * 2_300.0 * time).sin();
            let transient_phase = index % (sample_rate as usize / 2);
            let transient = if transient_phase < 256 {
                0.5 * (1.0 - transient_phase as f32 / 256.0)
            } else {
                0.0
            };
            (tone + transient + noise).clamp(-1.0, 1.0)
        })
        .collect()
}

/// The pinned 60-second stereo 48 kHz workload.
#[must_use]
pub fn stereo_48k() -> (Vec<f32>, Vec<f32>) {
    (
        signal(48_000, PINNED_SECONDS, 0),
        signal(48_000, PINNED_SECONDS, 1),
    )
}

/// The pinned mono workload used by the 22.05 kHz minimap pipeline.
#[must_use]
pub fn mono_22050() -> Vec<f32> {
    signal(22_050, PINNED_SECONDS, 0)
}

/// The pinned mono input for 44.1 kHz to 48 kHz resampling.
#[must_use]
pub fn mono_44100() -> Vec<f32> {
    signal(44_100, PINNED_SECONDS, 0)
}
