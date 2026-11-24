//! Where the segmenter's 10 s windows sit in a file, and where its frames sit
//! in a window.
//!
//! Every constant here is the CHECKPOINT's, derived from SincNet's stride
//! ladder rather than chosen: `conv(k=251, s=10) → pool(3) → conv(5) →
//! pool(3) → conv(5) → pool(3)` turns 160 000 samples into 589 frames, one
//! every 270 samples, each seeing 991 samples of audio. The staging script
//! refuses a model whose output is not `[1, 589, 7]`, so these cannot drift
//! away from the weights without the staging step saying so.
//!
//! The one number that IS a choice is [`STEP_FRAMES`], and it is a choice made
//! in FRAMES rather than in seconds on purpose: a step that is a whole number
//! of frames makes chunk k's frame i land exactly on global frame
//! `k * STEP_FRAMES + i`, so overlapping chunks can be averaged by index with
//! no interpolation anywhere. 4.978 s is not a rounder number than 5 s — it is
//! the same number said in the unit that keeps the grids aligned.
//!
//! Ported from `Processors/Pyannote/SegmentationWindows.swift`, itself adapted
//! from FluidAudio (Apache-2.0), `Sources/FluidAudio/Diarizer/Segmentation/
//! SlidingWindow.swift` — the frame geometry. The whole-frame chunk step and
//! the global grid are ours.

pub const SAMPLE_RATE: u32 = 16_000;

/// The graph's fixed input, in samples, and what it produces.
pub const WINDOW_SAMPLES: usize = 160_000;
pub const WINDOW_FRAMES: usize = 589;

/// Samples between two frame STARTS, and how much audio one frame sees.
/// 270 = 10 · 3 · 3 · 3 (the stride ladder); 991 = 251 + 2·10 + 4·30 + 2·30 +
/// 4·90 + 2·90 (the receptive field).
pub const FRAME_STRIDE: usize = 270;
pub const FRAME_SAMPLES: usize = 991;

/// Chunk hop. The LARGEST whole-frame step for which two steps still fit
/// inside one window (2 × 294 = 588 ≤ 589), which is what makes "every frame
/// away from the two ends is measured twice" true rather than nearly true.
/// Half of 589 is 295, and at 295 the two windows fall one frame short of
/// touching: every 590th frame would be measured once, by a chunk that
/// happened to be at its own seam.
///
/// Being measured twice is the point of overlapping at all — a speaker turn
/// that lands on a chunk seam, where SincNet's edge and the LSTM's cold start
/// make the model least sure, is also seen mid-window by the neighbour, and
/// the two are averaged.
pub const STEP_FRAMES: usize = 294;

/// Two steps must fit inside one window, or "the interior is measured twice" —
/// the property the whole averaging rests on — becomes a coincidence. Checked
/// at compile time because it is a statement about two constants.
const _: () = assert!(2 * STEP_FRAMES <= WINDOW_FRAMES);

/// 79 380 samples.
pub const STEP_SAMPLES: usize = STEP_FRAMES * FRAME_STRIDE;

/// Seconds between two frame starts — 16.9 ms.
#[must_use]
pub fn frame_seconds() -> f64 {
    FRAME_STRIDE as f64 / f64::from(SAMPLE_RATE)
}

/// One prediction: where to read its audio, and which global frames its own
/// frames land on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub sample_offset: usize,
    pub frame_offset: usize,
    /// How many of this chunk's [`WINDOW_FRAMES`] frames are real audio rather
    /// than the zero pad the last chunk reads past the end.
    pub frames: usize,
}

/// Frames whose receptive field fits inside `sample_count`. Audio past the
/// last whole frame is not a frame the model can be asked about.
#[must_use]
pub fn frame_count(sample_count: usize) -> usize {
    if sample_count < FRAME_SAMPLES {
        return 0;
    }
    (sample_count - FRAME_SAMPLES) / FRAME_STRIDE + 1
}

/// Tile the file. The last chunk is zero-padded on the right by the reader;
/// the frames that fall in the pad are dropped here rather than decoded, so
/// silence the file never had can never become a turn.
#[must_use]
pub fn chunks(sample_count: usize) -> Vec<Chunk> {
    let total = frame_count(sample_count);
    let mut chunks = Vec::new();
    let mut frame_offset = 0;
    while frame_offset < total {
        chunks.push(Chunk {
            sample_offset: frame_offset * FRAME_STRIDE,
            frame_offset,
            frames: WINDOW_FRAMES.min(total - frame_offset),
        });
        // A chunk that already reaches the end is the last one: stepping again
        // would only re-measure audio two chunks have covered.
        if frame_offset + WINDOW_FRAMES >= total {
            break;
        }
        frame_offset += STEP_FRAMES;
    }
    chunks
}

/// The instant a global frame starts. A frame is 61.9 ms of audio taken every
/// 16.9 ms, so consecutive frames OVERLAP — [`end_time`] is deliberately not
/// the next frame's `start_time`.
#[must_use]
pub fn start_time(frame: usize) -> f64 {
    (frame * FRAME_STRIDE) as f64 / f64::from(SAMPLE_RATE)
}

/// The instant a global frame's audio ends.
#[must_use]
pub fn end_time(frame: usize) -> f64 {
    (frame * FRAME_STRIDE + FRAME_SAMPLES) as f64 / f64::from(SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 160 000 samples through `conv(251, s10) → pool3 → conv5 → pool3 →
    /// conv5 → pool3` is exactly 589 frames. If this stops holding, the staged
    /// model is not the model this code was written for.
    #[test]
    fn the_window_holds_exactly_the_models_frame_count() {
        assert_eq!(frame_count(WINDOW_SAMPLES), WINDOW_FRAMES);
        assert_eq!(FRAME_STRIDE, 270);
        assert_eq!(FRAME_SAMPLES, 991);
    }

    /// A frame exists only when its whole receptive field is real audio. Audio
    /// one sample short of the next frame's field is not a frame the model can
    /// be asked about, and rounding it up would ask it about the zero pad.
    #[test]
    fn a_frame_needs_its_whole_receptive_field() {
        assert_eq!(frame_count(0), 0);
        assert_eq!(frame_count(FRAME_SAMPLES - 1), 0);
        assert_eq!(frame_count(FRAME_SAMPLES), 1);
        assert_eq!(frame_count(FRAME_SAMPLES + FRAME_STRIDE - 1), 1);
        assert_eq!(frame_count(FRAME_SAMPLES + FRAME_STRIDE), 2);
    }

    /// A frame is 61.9 ms of audio taken every 16.9 ms, so consecutive frames
    /// OVERLAP — `end_time` is deliberately not the next frame's `start_time`,
    /// and a reply built from a run of frames runs to the last one's audio.
    #[test]
    fn frames_overlap_each_other() {
        assert!((start_time(0)).abs() < 1e-12);
        assert!((end_time(0) - FRAME_SAMPLES as f64 / f64::from(SAMPLE_RATE)).abs() < 1e-12);
        assert!(end_time(0) > start_time(1));
    }

    /// The last frame's audio never runs past the file — which is why nothing
    /// downstream has to clamp a reply's end to the duration.
    #[test]
    fn the_last_frames_audio_fits_in_the_file() {
        for samples in [FRAME_SAMPLES, 12_345, 160_000, 1_000_000] {
            let last = frame_count(samples) - 1;
            assert!(
                end_time(last) <= samples as f64 / f64::from(SAMPLE_RATE),
                "{samples} samples"
            );
        }
    }

    /// The chunk step is a whole number of FRAMES, which is the whole reason
    /// overlapping chunks can be averaged by index.
    #[test]
    fn chunks_land_on_the_global_frame_grid() {
        let chunks = chunks(16_000 * 60);
        assert!(chunks.len() > 2);
        for chunk in &chunks {
            assert_eq!(chunk.sample_offset, chunk.frame_offset * FRAME_STRIDE);
            assert_eq!(chunk.sample_offset % FRAME_STRIDE, 0);
        }
        assert_eq!(STEP_SAMPLES, STEP_FRAMES * FRAME_STRIDE);
    }

    /// Every frame is measured, and every frame away from the two ends is
    /// measured TWICE — the overlap that lets a seam twitch be outvoted.
    #[test]
    fn every_frame_is_covered_and_the_middle_is_covered_twice() {
        let samples = 16_000 * 47;
        let total = frame_count(samples);
        let mut coverage = vec![0_usize; total];
        for chunk in chunks(samples) {
            for seen in &mut coverage[chunk.frame_offset..chunk.frame_offset + chunk.frames] {
                *seen += 1;
            }
        }

        assert!(!coverage.contains(&0), "a frame no chunk measured");
        assert!(
            coverage[WINDOW_FRAMES..total - WINDOW_FRAMES]
                .iter()
                .all(|&seen| seen >= 2),
            "the interior must be seen by two chunks"
        );
    }

    /// A chunk never claims frames the file does not have: the last one reads
    /// into the zero pad, and the frames that land there are dropped by count
    /// rather than decoded into silence the file never had.
    #[test]
    fn no_chunk_claims_a_frame_that_is_not_there() {
        for seconds in [0.5, 3.0, 10.0, 10.5, 25.0, 60.0] {
            let samples = (16_000.0 * seconds) as usize;
            let total = frame_count(samples);
            let chunks = chunks(samples);
            for chunk in &chunks {
                assert!(chunk.frames > 0, "{seconds}s");
                assert!(chunk.frames <= WINDOW_FRAMES);
                assert!(chunk.frame_offset + chunk.frames <= total, "{seconds}s");
            }
            assert_eq!(
                chunks.last().map_or(0, |c| c.frame_offset + c.frames),
                total,
                "{seconds}s"
            );
        }
    }

    /// Audio shorter than one window is ONE chunk, padded — not zero chunks,
    /// and not a second chunk that would only re-read the pad.
    #[test]
    fn short_audio_is_a_single_chunk() {
        let chunks = chunks(16_000 * 3);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].frame_offset, 0);
        assert_eq!(chunks[0].frames, frame_count(16_000 * 3));
    }

    #[test]
    fn audio_with_no_whole_frame_is_no_chunks() {
        assert!(chunks(FRAME_SAMPLES - 1).is_empty());
        assert!(chunks(0).is_empty());
    }
}
