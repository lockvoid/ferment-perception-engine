//! The prediction network and the joint as host arithmetic over
//! `parakeet_decode.bin` (`ferment-models-export`, `models/parakeet`,
//! `members.parakeet_decode`): fp16 tensors, row major, little-endian, back to
//! back, widened to f32 once at load. A step is a row of the embedding, each
//! LSTM layer's input and recurrent gemv, the projection; the joint is
//! `head · relu(frame + decoder) + head_bias`, one 8198 × 640 gemv. iOS
//! `ParakeetHostDecode.swift`, arithmetic for arithmetic.
//!
//! As graphs the prediction network multiplied a one-hot by the whole
//! 8193 × 640 embedding on every token and cost a runtime call a step; here a
//! token is a row of the table.

use std::path::Path;

use ferment_engine_core::{EngineError, Result};
use half::f16;

use super::config::{HIDDEN, LOGITS, VOCAB};

/// The object's name on the CDN and in a models directory.
pub const FILE: &str = "parakeet_decode.bin";

const LAYERS: usize = 2;
const GATES: usize = 4 * HIDDEN;

/// `port.DECODE_TENSORS`, in file order. LSTM gates are torch's (input,
/// forget, cell, output); each layer's bias is `b_ih + b_hh`.
pub const TENSORS: [(&str, &[usize]); 11] = [
    ("embedding", &[VOCAB, HIDDEN]),
    ("lstm_0_input", &[GATES, HIDDEN]),
    ("lstm_0_recurrent", &[GATES, HIDDEN]),
    ("lstm_0_bias", &[GATES]),
    ("lstm_1_input", &[GATES, HIDDEN]),
    ("lstm_1_recurrent", &[GATES, HIDDEN]),
    ("lstm_1_bias", &[GATES]),
    ("projection", &[HIDDEN, HIDDEN]),
    ("projection_bias", &[HIDDEN]),
    ("head", &[LOGITS, HIDDEN]),
    ("head_bias", &[LOGITS]),
];

struct Layer {
    input: Vec<f32>,
    recurrent: Vec<f32>,
    bias: Vec<f32>,
}

/// The decode's weights, widened.
pub struct HostDecode {
    embedding: Vec<f32>,
    layers: [Layer; LAYERS],
    projection: Vec<f32>,
    projection_bias: Vec<f32>,
    head: Vec<f32>,
    head_bias: Vec<f32>,
}

impl std::fmt::Debug for HostDecode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HostDecode { .. }")
    }
}

/// The prediction network after a token: its output, and each layer's hidden
/// and cell state.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub output: Vec<f32>,
    pub hidden: [Vec<f32>; LAYERS],
    pub cell: [Vec<f32>; LAYERS],
}

impl Prediction {
    /// The state a window starts from: every layer zero, no output yet.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            output: vec![0.0; HIDDEN],
            hidden: [vec![0.0; HIDDEN], vec![0.0; HIDDEN]],
            cell: [vec![0.0; HIDDEN], vec![0.0; HIDDEN]],
        }
    }
}

impl HostDecode {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            EngineError::Models(format!("{} could not be read: {error}", path.display()))
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let values: usize = TENSORS
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum();
        if bytes.len() != values * 2 {
            return Err(EngineError::Models(format!(
                "{FILE} is {} bytes; the contract's tensors are {} fp16 values",
                bytes.len(),
                values
            )));
        }
        let mut halves = bytes
            .chunks_exact(2)
            .map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32());
        let mut tensors = TENSORS.map(|(_, shape)| {
            halves
                .by_ref()
                .take(shape.iter().product())
                .collect::<Vec<f32>>()
        });
        let mut take = |index: usize| std::mem::take(&mut tensors[index]);
        Ok(Self {
            embedding: take(0),
            layers: [
                Layer {
                    input: take(1),
                    recurrent: take(2),
                    bias: take(3),
                },
                Layer {
                    input: take(4),
                    recurrent: take(5),
                    bias: take(6),
                },
            ],
            projection: take(7),
            projection_bias: take(8),
            head: take(9),
            head_bias: take(10),
        })
    }

    /// The prediction network on `token` from `state`.
    #[must_use]
    pub fn predict(&self, token: usize, state: &Prediction) -> Prediction {
        let mut x = self.embedding[token * HIDDEN..(token + 1) * HIDDEN].to_vec();
        let mut next = Prediction::zero();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut gates = layer.bias.clone();
            gemv(&layer.input, &x, &mut gates);
            gemv(&layer.recurrent, &state.hidden[index], &mut gates);
            let (input, rest) = gates.split_at(HIDDEN);
            let (forget, rest) = rest.split_at(HIDDEN);
            let (candidate, output) = rest.split_at(HIDDEN);
            for unit in 0..HIDDEN {
                let cell = sigmoid(forget[unit]) * state.cell[index][unit]
                    + sigmoid(input[unit]) * candidate[unit].tanh();
                next.cell[index][unit] = cell;
                next.hidden[index][unit] = sigmoid(output[unit]) * cell.tanh();
            }
            x.copy_from_slice(&next.hidden[index]);
        }
        next.output.copy_from_slice(&self.projection_bias);
        gemv(&self.projection, &x, &mut next.output);
        next
    }

    /// `logits [8198]` for one encoder frame against the decoder's output:
    /// the vocabulary, then the five durations.
    pub fn joint(&self, frame: &[f32], decoder: &[f32], logits: &mut [f32]) {
        let sum = frame
            .iter()
            .zip(decoder)
            .map(|(frame, decoder)| (frame + decoder).max(0.0))
            .collect::<Vec<f32>>();
        logits.copy_from_slice(&self.head_bias);
        gemv(&self.head, &sum, logits);
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// `out += matrix · x`, the matrix row major `[out.len(), x.len()]`.
fn gemv(matrix: &[f32], x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(matrix.len(), out.len() * x.len());
    pulp::Arch::new().dispatch(Gemv { matrix, x, out });
}

struct Gemv<'a> {
    matrix: &'a [f32],
    x: &'a [f32],
    out: &'a mut [f32],
}

impl pulp::WithSimd for Gemv<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) {
        let (x_head, x_tail) = S::as_simd_f32s(self.x);
        for (row, out) in self
            .matrix
            .chunks_exact(self.x.len())
            .zip(self.out.iter_mut())
        {
            let (row_head, row_tail) = S::as_simd_f32s(row);
            let mut sums = [simd.splat_f32s(0.0); 4];
            let mut rows = row_head.chunks_exact(4);
            let mut xs = x_head.chunks_exact(4);
            for (row, x) in rows.by_ref().zip(xs.by_ref()) {
                for lane in 0..4 {
                    sums[lane] = simd.mul_add_e_f32s(row[lane], x[lane], sums[lane]);
                }
            }
            for (row, x) in rows.remainder().iter().zip(xs.remainder()) {
                sums[0] = simd.mul_add_e_f32s(*row, *x, sums[0]);
            }
            let sum = simd.add_f32s(
                simd.add_f32s(sums[0], sums[1]),
                simd.add_f32s(sums[2], sums[3]),
            );
            let mut total = simd.reduce_sum_f32s(sum);
            for (row, x) in row_tail.iter().zip(x_tail) {
                total += row * x;
            }
            *out += total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gemv_is_every_row_dotted_with_the_vector() {
        for (rows, columns) in [(3, 640), (5, 17), (2, 1), (4, 37)] {
            let matrix = (0..rows * columns)
                .map(|index| ((index * 7919) % 101) as f32 / 50.0 - 1.0)
                .collect::<Vec<f32>>();
            let x = (0..columns)
                .map(|index| ((index * 31) % 23) as f32 / 11.0 - 1.0)
                .collect::<Vec<f32>>();
            let mut out = vec![0.5; rows];
            gemv(&matrix, &x, &mut out);
            for (row, out) in out.iter().enumerate() {
                let expected = 0.5
                    + (0..columns)
                        .map(|column| {
                            f64::from(matrix[row * columns + column]) * f64::from(x[column])
                        })
                        .sum::<f64>();
                assert!(
                    (f64::from(*out) - expected).abs() < 1e-4,
                    "{rows}×{columns} row {row}"
                );
            }
        }
    }

    #[test]
    fn a_file_of_the_wrong_size_is_refused() {
        let error = HostDecode::from_bytes(&[0; 64]).unwrap_err();
        assert!(error.to_string().contains("fp16 values"), "{error}");
    }

    #[test]
    fn the_tensors_are_the_contracts_byte_count() {
        let values: usize = TENSORS
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum();
        assert_eq!(values * 2, 34_934_796);
    }
}
