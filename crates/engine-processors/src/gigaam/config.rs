//! GigaAM-v3-e2e-rnnt model constants.
//!
//! Every number here is pinned by `ferment-models-export`
//! (`models/gigaam/model.json`) — the exported shapes are static, so a
//! mismatch is a load-time failure rather than a silent quality loss. Keep the
//! two in step. Verified against iOS `Processors/GigaAM/GigaAMConfig.swift`.

pub const SAMPLE_RATE: u32 = 16_000;
pub const HOP_LENGTH: usize = 160;
pub const WINDOW_SIZE: usize = 320;
pub const N_MELS: usize = 64;

pub const ENCODER_HIDDEN: usize = 768;
pub const DECODER_HIDDEN: usize = 320;
pub const DECODER_LAYERS: usize = 1;
/// `num_classes - 1` — the RNN-T convention upstream's `RNNTDecoder` uses, and
/// the id its embedding is built with as `padding_idx`.
pub const BLANK_TOKEN_ID: i32 = 1024;

/// Samples per encoder output frame: 160-sample mel hop x 4 subsampling.
pub const SAMPLES_PER_ENCODER_FRAME: usize = 640;
/// 640 / 16000 = 0.04 s. Twice Parakeet's resolution, which is where the finer
/// word timings come from.
pub const SECONDS_PER_ENCODER_FRAME: f64 = SAMPLES_PER_ENCODER_FRAME as f64 / SAMPLE_RATE as f64;

/// Maximum audio samples per inference chunk (15 seconds). Static shape: the
/// exported preprocessor takes exactly this many samples.
pub const MAX_CHUNK_SAMPLES: usize = 240_000;
/// Overlap between chunks in encoder frames — 50 x 0.04 s = 2.0 s, the same
/// acoustic overlap the Parakeet port uses at its coarser frame rate.
pub const OVERLAP_FRAMES: usize = 50;

/// Context samples prepended to non-first chunks for mel stability
/// (1 encoder frame).
pub const MEL_CONTEXT_SAMPLES: usize = SAMPLES_PER_ENCODER_FRAME;

/// Tokens emitted at one encoder frame before the loop force-advances.
/// Upstream `RNNTGreedyDecoding` uses 10.
pub const MAX_SYMBOLS_PER_STEP: usize = 10;
/// Runaway guard per chunk. 15 s of dense Russian speech runs ~180
/// SentencePiece tokens against this 1024-piece vocabulary, so 300 is slack
/// rather than a quality ceiling.
pub const MAX_TOKENS_PER_CHUNK: usize = 300;

/// Half-width (seconds) the per-segment decode path pads each caller-supplied
/// speech segment by before slicing, and — because merging happens after
/// padding — the gap threshold at which two segments coalesce into one decode
/// run. Same 1.0 s the Parakeet path uses.
pub const SEGMENT_PAD_SECONDS: f64 = 1.0;

/// Content samples per chunk, leaving room for the prepended context so a
/// chunk never exceeds the exported model's fixed input length, and rounded
/// down to a whole number of encoder frames.
pub const CHUNK_SAMPLES: usize = (MAX_CHUNK_SAMPLES - MEL_CONTEXT_SAMPLES - HOP_LENGTH)
    / SAMPLES_PER_ENCODER_FRAME
    * SAMPLES_PER_ENCODER_FRAME;
/// Derived from [`OVERLAP_FRAMES`], not from a second seconds constant: the
/// merge window is computed from the same number, and two independently
/// rounded copies of "2 seconds" agreeing is luck, not design.
pub const OVERLAP_SAMPLES: usize = OVERLAP_FRAMES * SAMPLES_PER_ENCODER_FRAME;
pub const STRIDE_SAMPLES: usize =
    (CHUNK_SAMPLES - OVERLAP_SAMPLES) / SAMPLES_PER_ENCODER_FRAME * SAMPLES_PER_ENCODER_FRAME;

// The chunk geometry's invariants hold at compile time, not at test time: a
// chunk that outgrew the exported graph's fixed input, or a stride that
// rounded to zero, should not build at all.
const _: () = assert!(CHUNK_SAMPLES + MEL_CONTEXT_SAMPLES <= MAX_CHUNK_SAMPLES);
const _: () = assert!(STRIDE_SAMPLES > 0);
const _: () = assert!(CHUNK_SAMPLES.is_multiple_of(SAMPLES_PER_ENCODER_FRAME));
const _: () = assert!(STRIDE_SAMPLES.is_multiple_of(SAMPLES_PER_ENCODER_FRAME));

/// Mel frames a buffer of `sample_count` samples yields — upstream's
/// `FeatureExtractor.out_len` with `center: false`.
///
/// This is what the encoder's pad mask is counted from, and it is
/// load-bearing: the preprocessor always sees a full 240k buffer, so the mask
/// is the only thing telling the Conformer which frames are real. Mask the
/// full 375 on a short chunk and the zero padding reads as speech — which
/// changes the transcript, measurably, not just the compute.
///
/// Returns a signed count because a buffer shorter than one window yields a
/// non-positive frame count, which the caller treats as "nothing to decode".
/// Integer division truncates toward zero on both sides, so an input of
/// 161..=319 samples answers 1 here where upstream's `out_len` answers 0; the
/// encoder then reads one frame of a near-empty buffer and emits nothing,
/// which is the same outcome.
#[must_use]
pub const fn mel_frames(sample_count: usize) -> i64 {
    (sample_count as i64 - WINDOW_SIZE as i64) / HOP_LENGTH as i64 + 1
}

/// Encoder frames for `mel_frame_count` — upstream's
/// `StridingSubsampling.calc_output_length`: two stride-2 conv steps, each
/// `floor((L - 1) / 2) + 1` for the kernel-5 conv1d this model uses.
#[must_use]
pub const fn encoder_frames(mel_frame_count: usize) -> usize {
    let mut length = mel_frame_count;
    let mut step = 0;
    while step < 2 {
        length = (length - 1) / 2 + 1;
        step += 1;
    }
    length
}

/// The static mel axis of the encoder's door: a full 240 000-sample chunk.
pub const MAX_MEL_FRAMES: usize = mel_frames(MAX_CHUNK_SAMPLES) as usize;
/// The static time axis of `mask` and `encoded`.
pub const MAX_ENCODER_FRAMES: usize = encoder_frames(MAX_MEL_FRAMES);

/// The encoder's pad mask for a chunk with `mel_frame_count` real mel frames:
/// 1 on its first `encoder_frames(mel_frame_count)` frames, 0 after.
#[must_use]
pub fn encoder_mask(mel_frame_count: usize) -> Vec<f32> {
    let valid = encoder_frames(mel_frame_count);
    (0..MAX_ENCODER_FRAMES)
        .map(|frame| if frame < valid { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: count the padded input rather than upstream's valid content geometry.
    /// Current Swift GigaAMTranscriberTests.testFrameArithmeticMatchesUpstream.
    #[test]
    fn frame_arithmetic_matches_all_current_upstream_probes() {
        for (samples, mel, encoded) in [
            (240_000, 1499, 375),
            (239_360, 1495, 374),
            (238_720, 1491, 373),
            (32_000, 199, 50),
            (640, 3, 1),
            (320, 1, 1),
        ] {
            assert_eq!(mel_frames(samples), mel, "{samples} samples");
            assert_eq!(encoder_frames(mel as usize), encoded, "{mel} mel frames");
        }
        assert_eq!(
            MAX_CHUNK_SAMPLES / encoder_frames(mel_frames(MAX_CHUNK_SAMPLES) as usize),
            SAMPLES_PER_ENCODER_FRAME
        );
        assert!((SECONDS_PER_ENCODER_FRAME - 0.04).abs() <= 1e-9);
    }

    #[test]
    fn frame_seconds_match_40_ms() {
        assert!((SECONDS_PER_ENCODER_FRAME - 0.04).abs() < 1e-12);
    }

    /// The exported graph's geometry: 240 000 samples -> 1499 mel -> 375
    /// encoder frames, and 375 x 640 = 240 000 exactly.
    #[test]
    fn the_exported_chunk_geometry_closes() {
        assert_eq!(mel_frames(MAX_CHUNK_SAMPLES), 1499);
        assert_eq!(encoder_frames(1499), 375);
        assert_eq!((MAX_MEL_FRAMES, MAX_ENCODER_FRAMES), (1499, 375));
        assert_eq!(375 * SAMPLES_PER_ENCODER_FRAME, MAX_CHUNK_SAMPLES);
    }

    /// `floor((L - 1) / 2) + 1` twice (`models/gigaam/model.json`,
    /// `frontend.mask`), at the lengths where the two steps round.
    #[test]
    fn the_mask_covers_the_chunks_real_encoder_frames() {
        for (mel, valid) in [
            (1, 1),
            (4, 1),
            (5, 2),
            (8, 2),
            (9, 3),
            (499, 125),
            (1499, 375),
        ] {
            assert_eq!(encoder_frames(mel), valid, "{mel} mel frames");
            let mask = encoder_mask(mel);
            assert_eq!(mask.len(), MAX_ENCODER_FRAMES);
            assert!(mask[..valid].iter().all(|value| *value == 1.0), "{mel}");
            assert!(mask[valid..].iter().all(|value| *value == 0.0), "{mel}");
        }
    }

    /// The values themselves; their invariants are compile-time asserts above.
    #[test]
    fn chunk_geometry_is_frame_aligned_and_fits_the_graph() {
        assert_eq!(CHUNK_SAMPLES, 238_720);
        assert_eq!(OVERLAP_SAMPLES, 32_000);
        assert_eq!(STRIDE_SAMPLES, 206_720);
    }

    /// A buffer under one analysis window yields nothing to decode rather
    /// than a negative frame count reaching the graph. The 161..=319 row is
    /// the documented divergence from upstream's `out_len`: integer division
    /// truncates toward zero on both sides, so this answers 1 where upstream
    /// answers 0 — the encoder then reads one frame of a near-empty buffer and
    /// emits nothing, which is the same outcome.
    #[test]
    fn a_sub_window_buffer_yields_no_frames() {
        assert_eq!(mel_frames(0), -1);
        assert_eq!(mel_frames(159), 0);
        assert_eq!(mel_frames(160), 0);
        assert_eq!(mel_frames(161), 1);
        assert_eq!(mel_frames(319), 1);
        assert_eq!(mel_frames(320), 1);
        assert_eq!(mel_frames(480), 2);
    }
}
