//! Fixed shapes of the `nvidia/nemotron-3.5-asr-streaming-0.6b` multilingual
//! export, 560 ms tier. Every number is baked into the exported graphs;
//! [`super::models::NemotronModels::load`] checks the sidecar metadata against
//! them before a chunk runs.
//!
//! Ported from `Processors/Nemotron/NemotronConfig.swift`.

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_FEATURES: usize = 128;
pub const MEL_HOP_SAMPLES: usize = 160;
/// Mel frames of new audio per encoder call.
pub const CHUNK_MEL_FRAMES: usize = 56;
/// Mel frames of the previous chunk prepended for convolution context.
pub const PRE_ENCODE_CACHE_FRAMES: usize = 9;
pub const ENCODER_MEL_FRAMES: usize = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE_FRAMES;
pub const CHUNK_SAMPLES: usize = CHUNK_MEL_FRAMES * MEL_HOP_SAMPLES;
pub const CHUNK_SECONDS: f64 = CHUNK_SAMPLES as f64 / SAMPLE_RATE as f64;
/// 8× subsampling: one encoder frame per 80 ms.
pub const SAMPLES_PER_ENCODER_FRAME: usize = 1_280;
pub const SECONDS_PER_ENCODER_FRAME: f64 = SAMPLES_PER_ENCODER_FRAME as f64 / SAMPLE_RATE as f64;
pub const ENCODER_FRAMES_PER_CHUNK: usize = CHUNK_SAMPLES / SAMPLES_PER_ENCODER_FRAME;
pub const ENCODER_DIM: usize = 1_024;
/// `[layers, batch, left_context_frames, dim]`, the shape
/// `scripts/exports/nemotron/export.py` emits from the checkpoint's own
/// streaming state.
///
/// iOS pins `[1, 24, 42, 1024]`. BOTH differences — the axis order and the 42
/// — belong to the FluidInference Core ML conversion iOS runs, not to the
/// weights: this checkpoint's 560 ms row (`att_context_size` index 3) is
/// `[56, 13]`, and `get_initial_cache_state` hands back `(24, 1, 56, 1024)`.
/// A transcript is text, so the two platforms are free to differ here — but
/// which left context BOTH should take is a ruling, not this lane's call
/// (report §7, ruling N-1).
pub const CACHE_CHANNEL_SHAPE: [i64; 4] = [24, 1, 56, 1024];
/// `[layers, batch, dim, conv_context]` — the same axis order as
/// [`CACHE_CHANNEL_SHAPE`], and the same divergence from iOS's leading `1`.
pub const CACHE_TIME_SHAPE: [i64; 4] = [24, 1, 1024, 8];
pub const DECODER_LAYERS: usize = 2;
pub const DECODER_HIDDEN: usize = 640;
pub const VOCAB_SIZE: usize = 13_087;
pub const BLANK_TOKEN_ID: i32 = 13_087;
pub const MAX_SYMBOLS_PER_STEP: usize = 10;
/// Encoder frames of right context baked into the export's attention mask
/// (`att_context_size` [42, 13]): a token for audio at frame f is not emitted
/// until the encoder has seen frame f + 13.
pub const LOOKAHEAD_FRAMES: usize = 13;
/// Silent chunks fed after the last audio so the lookahead closes over the
/// final word.
pub const FLUSH_CHUNKS: usize = LOOKAHEAD_FRAMES.div_ceil(ENCODER_FRAMES_PER_CHUNK);

pub const VOCAB_FILE: &str = "nemotron_vocab";
pub const METADATA_FILE: &str = "nemotron_metadata";

/// Padding around AED speech segments before they are decoded in isolation.
pub const SEGMENT_PAD_SECONDS: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// kill: change `SAMPLES_PER_ENCODER_FRAME` to 1024 and the 80 ms frame
    /// this whole timing column is built on stops being 80 ms.
    #[test]
    fn frame_seconds_match_80_ms() {
        assert!((SECONDS_PER_ENCODER_FRAME - 0.08).abs() < 1e-9);
    }

    /// kill: drop `LOOKAHEAD_FRAMES` to 6 and one flush chunk covers it, so
    /// the last word of every take is never emitted.
    #[test]
    fn flush_chunks_cover_the_whole_lookahead() {
        assert_eq!(ENCODER_FRAMES_PER_CHUNK, 7);
        assert_eq!(FLUSH_CHUNKS, 2);
        const { assert!(FLUSH_CHUNKS * ENCODER_FRAMES_PER_CHUNK >= LOOKAHEAD_FRAMES) };
    }
}
