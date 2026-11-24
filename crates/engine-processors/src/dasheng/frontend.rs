//! The host half of the contract: one window's log-mel, in the graph's patch
//! layout.

use ferment_engine_audio::MelFrames;
use ferment_engine_core::{EngineError, Result};

use super::config::{
    MEL_BANDS, MEL_FRAMES, MEL_SPEC, PATCH_CHANNELS, PATCH_COUNT, PATCH_FRAMES, TOP_DB,
    WINDOW_SAMPLES,
};

/// The stock front-end over one 10 s window: `MEL_SPEC`, then the window's
/// `top_db` floor.
pub fn mel(window: &[f32]) -> Result<MelFrames> {
    if window.len() != WINDOW_SAMPLES {
        return Err(EngineError::invalid_config(format!(
            "dasheng mel: expected {WINDOW_SAMPLES} samples, got {}",
            window.len()
        )));
    }
    let frames = MEL_SPEC.compute(window)?.clamped_below_max_by(TOP_DB);
    if frames.frame_count() != MEL_FRAMES {
        return Err(EngineError::InvalidOutput);
    }
    Ok(frames)
}

/// The frames in the graph's layout, `[256 channels × 250 patches]` row-major
/// (`mel [1, 256, 1, 250]`): channel `band · 4 + dt` is band `band` of frame
/// `4 · patch + dt`.
#[must_use]
pub fn patches(mel: &MelFrames) -> Vec<f32> {
    let mut out = vec![0.0; PATCH_CHANNELS * PATCH_COUNT];
    for patch in 0..PATCH_COUNT {
        for dt in 0..PATCH_FRAMES {
            let frame = mel.frame(patch * PATCH_FRAMES + dt);
            for band in 0..MEL_BANDS {
                out[(band * PATCH_FRAMES + dt) * PATCH_COUNT + patch] = frame[band];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_patch_channel_is_a_band_of_a_frame() {
        let values = (0..MEL_FRAMES * MEL_BANDS).map(|i| i as f32).collect();
        let laid = patches(&MelFrames {
            bands: MEL_BANDS,
            values,
        });
        let cell = |band: usize, frame: usize| (frame * MEL_BANDS + band) as f32;
        assert_eq!(laid.len(), 256 * 250);
        assert_eq!(laid[0], cell(0, 0));
        assert_eq!(laid[(5 * 4 + 3) * 250 + 7], cell(5, 4 * 7 + 3));
        assert_eq!(laid[(63 * 4 + 3) * 250 + 249], cell(63, 999));
    }

    #[test]
    fn a_short_window_is_refused() {
        assert!(mel(&[0.0; 100]).is_err());
    }
}
