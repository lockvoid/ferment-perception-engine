//! DeepFilterNet3 model constants. Mirrors `DeepFilterNetConfig.swift`
//! and `dfnet_config.json` byte-for-byte — these values are what the
//! ONNX models were trained against; do not change them.

pub const SAMPLE_RATE: u32 = 48_000;
pub const FFT_SIZE: usize = 960;
pub const HOP_SIZE: usize = 480;
pub const N_FREQ: usize = FFT_SIZE / 2 + 1; // 481
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;
pub const DF_LOOKAHEAD: usize = 2;
pub const NORM_ALPHA: f32 = 0.99;

/// ERB band widths — sums to 481 (`N_FREQ`). Maps frequency bins → ERB bands.
pub const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37,
    42, 50, 56, 67,
];

/// Frames per chunk: where the streaming door's state is reset
/// (`ferment-models-export`, `models/dfnet/model.json`, `geometry.frames`).
pub const CHUNK_FRAMES: usize = 1_500;
/// The streaming door's static frame count — one call
/// (`geometry.stream_frames`). Fifteen calls are one chunk.
pub const STREAM_FRAMES: usize = 100;

/// One tensor a streaming call hands the next: fed as `input`, answered as
/// `output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamState {
    pub input: &'static str,
    pub output: &'static str,
    pub shape: &'static [i64],
}

/// The door's state, in its argument order: the GRU states `[layers, 1, 256]`
/// — `enc.emb_gru` 1 layer, `erb_dec.emb_gru` 2, `df_dec.df_gru` 2 — then the
/// last input frames of the three causal convolutions (`erb_conv0` and
/// `df_conv0` look 2 frames back, `df_convp` 4). A chunk starts from zeros.
pub const STREAM_STATE: [StreamState; 6] = [
    StreamState {
        input: "h_enc",
        output: "h_enc_out",
        shape: &[1, 1, 256],
    },
    StreamState {
        input: "h_erb",
        output: "h_erb_out",
        shape: &[2, 1, 256],
    },
    StreamState {
        input: "h_df",
        output: "h_df_out",
        shape: &[2, 1, 256],
    },
    StreamState {
        input: "erb_tail",
        output: "erb_tail_out",
        shape: &[1, 1, 2, NB_ERB as i64],
    },
    StreamState {
        input: "spec_tail",
        output: "spec_tail_out",
        shape: &[1, 2, 2, NB_DF as i64],
    },
    StreamState {
        input: "c0_tail",
        output: "c0_tail_out",
        shape: &[1, 64, 4, NB_DF as i64],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erb_widths_sum_to_n_freq() {
        let total: usize = ERB_WIDTHS.iter().sum();
        assert_eq!(total, N_FREQ);
    }

    #[test]
    fn n_freq_matches_fft_size() {
        assert_eq!(N_FREQ, FFT_SIZE / 2 + 1);
    }
}
