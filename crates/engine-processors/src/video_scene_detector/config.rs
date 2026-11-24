//! AutoShot model + batching constants.

pub const N_FRAMES_PER_BATCH: usize = 100;
pub const FRAME_HEIGHT: usize = 27;
pub const FRAME_WIDTH: usize = 48;
pub const N_COLOR_BINS: usize = 512;

/// Default boundary-probability threshold (matches AutoShot best-F1 on SHOT test set).
pub const DEFAULT_THRESHOLD: f32 = 0.296;

/// Frames per padded batch the Inference model sees.
pub const BATCH_SIZE: usize = N_FRAMES_PER_BATCH;
/// Center frames (predictions kept) per batch.
pub const CENTER_SIZE: usize = BATCH_SIZE / 2;
/// Border frames padded on each side of the center.
pub const BORDER_SIZE: usize = (BATCH_SIZE - CENTER_SIZE) / 2;

/// Total RGB bytes per frame.
pub const PIXELS_PER_FRAME: usize = FRAME_HEIGHT * FRAME_WIDTH * 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_split_consistent() {
        assert_eq!(BORDER_SIZE * 2 + CENTER_SIZE, BATCH_SIZE);
    }

    #[test]
    fn pixels_per_frame_is_27x48x3() {
        assert_eq!(PIXELS_PER_FRAME, 27 * 48 * 3);
    }
}
