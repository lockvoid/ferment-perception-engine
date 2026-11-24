//! Tensor abstraction for the inference client.
//!
//! Wraps inference's "raw bytes" tensor format — every infer call uses
//! `raw_input_contents` / `raw_output_contents` for performance (avoids
//! protobuf field-decode overhead, see grpc_service.proto comment).
//!
//! Tensors are stored as little-endian bytes plus shape + dtype.
//! Conversion helpers handle the most common Rust → bytes paths
//! (`Vec<f32>`, `Vec<i64>`, etc.).

use std::collections::BTreeMap;

use bytes::Bytes;
use half::{bf16, f16};

use ferment_engine_core::{EngineError, Result};

/// inference data types as defined in `model_config.proto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Fp16,
    Fp32,
    Fp64,
    Bf16,
    Bytes,
}

impl DType {
    /// Wire string used in `InferInputTensor::datatype` (`"FP32"`, `"INT64"`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::UInt8 => "UINT8",
            Self::UInt16 => "UINT16",
            Self::UInt32 => "UINT32",
            Self::UInt64 => "UINT64",
            Self::Int8 => "INT8",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Bf16 => "BF16",
            Self::Bytes => "BYTES",
        }
    }

    /// Bytes per scalar (None for variable-length BYTES).
    pub fn size_bytes(self) -> Option<usize> {
        match self {
            Self::Bool | Self::UInt8 | Self::Int8 => Some(1),
            Self::UInt16 | Self::Int16 | Self::Fp16 | Self::Bf16 => Some(2),
            Self::UInt32 | Self::Int32 | Self::Fp32 => Some(4),
            Self::UInt64 | Self::Int64 | Self::Fp64 => Some(8),
            Self::Bytes => None,
        }
    }

    pub fn from_wire(s: &str) -> Result<Self> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "UINT8" => Self::UInt8,
            "UINT16" => Self::UInt16,
            "UINT32" => Self::UInt32,
            "UINT64" => Self::UInt64,
            "INT8" => Self::Int8,
            "INT16" => Self::Int16,
            "INT32" => Self::Int32,
            "INT64" => Self::Int64,
            "FP16" => Self::Fp16,
            "FP32" => Self::Fp32,
            "FP64" => Self::Fp64,
            "BF16" => Self::Bf16,
            "BYTES" => Self::Bytes,
            other => {
                return Err(EngineError::inference(format!(
                    "unknown tensor datatype: {other}"
                )));
            }
        })
    }
}

/// One tensor — shape + dtype + flat little-endian bytes.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub dtype: DType,
    pub data: Bytes,
}

impl Tensor {
    pub fn new(shape: Vec<i64>, dtype: DType, data: Bytes) -> Result<Self> {
        let expected = checked_element_count(&shape)?;
        if let Some(size) = dtype.size_bytes() {
            let expected_bytes = expected.checked_mul(size).ok_or_else(|| {
                EngineError::inference("tensor byte count overflows addressable storage")
            })?;
            if data.len() != expected_bytes {
                return Err(EngineError::inference(format!(
                    "tensor size mismatch: {} bytes for shape {:?} dtype {} (expected {})",
                    data.len(),
                    shape,
                    dtype.as_str(),
                    expected_bytes
                )));
            }
        }
        Ok(Self { shape, dtype, data })
    }

    /// Build an FP32 tensor from a `Vec<f32>` without copying. Bytes are
    /// reinterpreted little-endian — fine on every platform we ship to.
    pub fn from_f32(shape: Vec<i64>, values: Vec<f32>) -> Result<Self> {
        let len = values.len();
        let expected = checked_element_count(&shape)?;
        if expected != len {
            return Err(EngineError::inference(format!(
                "shape {shape:?} has {expected} elements, got {len}"
            )));
        }
        let bytes = f32_vec_into_le_bytes(values);
        Self::new(shape, DType::Fp32, Bytes::from(bytes))
    }

    pub fn from_i64(shape: Vec<i64>, values: Vec<i64>) -> Result<Self> {
        let len = values.len();
        let expected = checked_element_count(&shape)?;
        if expected != len {
            return Err(EngineError::inference(format!(
                "shape {shape:?} has {expected} elements, got {len}"
            )));
        }
        let mut bytes = Vec::with_capacity(len * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::new(shape, DType::Int64, Bytes::from(bytes))
    }

    pub fn from_i32(shape: Vec<i64>, values: Vec<i32>) -> Result<Self> {
        let len = values.len();
        let expected = checked_element_count(&shape)?;
        if expected != len {
            return Err(EngineError::inference(format!(
                "shape {shape:?} has {expected} elements, got {len}"
            )));
        }
        let mut bytes = Vec::with_capacity(len * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::new(shape, DType::Int32, Bytes::from(bytes))
    }

    /// Build an FP16 tensor from a `Vec<f16>`.
    ///
    /// EdgeTAM's every neural boundary is FP16 and its parity fixtures pin
    /// BIT PATTERNS, so the half values cannot travel as `f32` and be rounded
    /// at the edge — the round trip is what a golden would catch.
    pub fn from_f16(shape: Vec<i64>, values: Vec<f16>) -> Result<Self> {
        let len = values.len();
        let expected = checked_element_count(&shape)?;
        if expected != len {
            return Err(EngineError::inference(format!(
                "shape {shape:?} has {expected} elements, got {len}"
            )));
        }
        let mut bytes = Vec::with_capacity(len * 2);
        for v in values {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        Self::new(shape, DType::Fp16, Bytes::from(bytes))
    }

    /// Decode a floating-point tensor into `Vec<f16>`, rounding once at the
    /// end for a wider source. An FP16 tensor keeps its exact bit patterns.
    pub fn to_f16(&self) -> Result<Vec<f16>> {
        if self.dtype == DType::Fp16 {
            return Ok(self
                .data
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect());
        }
        Ok(self.to_f32()?.into_iter().map(f16::from_f32).collect())
    }

    /// Decode a floating-point tensor into `Vec<f32>`.
    ///
    /// Core ML commonly exposes FP16 outputs even when the processor inputs
    /// are FP32, while ORT CPU exports usually return FP32. Keeping the
    /// conversion here makes processor code backend-neutral.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(self.element_count().max(0) as usize);
        match self.dtype {
            DType::Fp16 => {
                for chunk in self.data.chunks_exact(2) {
                    output.push(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32());
                }
            }
            DType::Bf16 => {
                for chunk in self.data.chunks_exact(2) {
                    output.push(bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32());
                }
            }
            DType::Fp32 => {
                for chunk in self.data.chunks_exact(4) {
                    output.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            DType::Fp64 => {
                for chunk in self.data.chunks_exact(8) {
                    let mut bytes = [0; 8];
                    bytes.copy_from_slice(chunk);
                    output.push(f64::from_le_bytes(bytes) as f32);
                }
            }
            dtype => {
                return Err(EngineError::inference(format!(
                    "expected floating-point tensor, got {}",
                    dtype.as_str()
                )));
            }
        }
        Ok(output)
    }

    pub fn to_i64(&self) -> Result<Vec<i64>> {
        let mut output = Vec::with_capacity(self.element_count().max(0) as usize);
        match self.dtype {
            DType::Int32 => {
                for chunk in self.data.chunks_exact(4) {
                    output.push(i64::from(i32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ])));
                }
            }
            DType::Int64 => {
                for chunk in self.data.chunks_exact(8) {
                    let mut bytes = [0; 8];
                    bytes.copy_from_slice(chunk);
                    output.push(i64::from_le_bytes(bytes));
                }
            }
            dtype => {
                return Err(EngineError::inference(format!(
                    "expected signed integer tensor, got {}",
                    dtype.as_str()
                )));
            }
        }
        Ok(output)
    }

    pub fn to_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::Int32 {
            return Err(EngineError::inference(format!(
                "expected INT32 tensor, got {}",
                self.dtype.as_str()
            )));
        }
        let mut out = Vec::with_capacity(self.data.len() / 4);
        for chunk in self.data.chunks_exact(4) {
            out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    pub fn element_count(&self) -> i64 {
        self.shape.iter().copied().product()
    }
}

/// Validate every dimension before multiplying; an even number of negative
/// dimensions and a wrapped product must never become a valid packed shape.
fn checked_element_count(shape: &[i64]) -> Result<usize> {
    if shape.iter().any(|dimension| *dimension < 0) {
        return Err(EngineError::inference("tensor shape contains negative dim"));
    }
    let count = shape
        .iter()
        .try_fold(1_i64, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| EngineError::inference("tensor element count overflows"))?;
    usize::try_from(count)
        .map_err(|_| EngineError::inference("tensor element count exceeds addressable storage"))
}

/// Named map of input/output tensors — `BTreeMap` so iteration order is
/// stable (matters for raw_input_contents alignment with inputs vec).
pub type TensorMap = BTreeMap<String, Tensor>;

fn f32_vec_into_le_bytes(values: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn malformed_shapes_cannot_wrap_or_hide_negative_dimensions() {
        for shape in [vec![-2, -2], vec![0, -1], vec![i64::MAX, 2]] {
            assert!(Tensor::new(shape.clone(), DType::Fp16, Bytes::new()).is_err());
            assert!(Tensor::from_f32(shape.clone(), vec![]).is_err());
            assert!(Tensor::from_f16(shape.clone(), vec![]).is_err());
            assert!(Tensor::from_i64(shape.clone(), vec![]).is_err());
            assert!(Tensor::from_i32(shape, vec![]).is_err());
        }
        assert!(Tensor::new(vec![i64::MAX], DType::Fp64, Bytes::new()).is_err());
    }

    #[test]
    fn packed_half_shape_count_and_bit_patterns_are_exact() {
        assert!(Tensor::from_f16(vec![1, 2, 3], vec![f16::ONE; 3]).is_err());
        let values = [-0.0_f32, 0.0, -2.5, 1.25, 7.5, f32::INFINITY].map(f16::from_f32);
        let tensor = Tensor::from_f16(vec![1, 2, 3], values.to_vec()).unwrap();
        assert_eq!(tensor.dtype, DType::Fp16);
        assert_eq!(tensor.shape, [1, 2, 3]);
        assert_eq!(tensor.data.len(), 12);
        assert_eq!(
            tensor
                .to_f16()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.map(f16::to_bits)
        );
    }

    // MLMultiArrayFloatsTests.swift. The fifth Swift case exercises an
    // Objective-C `MLMultiArray` with padded physical strides. The pinned
    // coreml-native 0.2.0 reads those correctly in `Prediction::get_f32`,
    // which `coreml.rs` calls for every floating Core ML output, but its
    // public tensor constructors always compute contiguous strides and its
    // `Prediction` cannot be constructed around a caller-supplied array.
    // Keep that exact scenario outstanding until the facade exposes one of
    // those seams; a Rust-only strided copy would not exercise Core ML.

    #[test]
    fn test_contiguous_float32() {
        let tensor = Tensor::from_f32(
            vec![1, 3, 2],
            (0..6).map(|index| index as f32 * 1.5).collect(),
        )
        .unwrap();
        assert_eq!(tensor.to_f32().unwrap(), vec![0.0, 1.5, 3.0, 4.5, 6.0, 7.5]);
    }

    #[test]
    fn test_contiguous_float16() {
        let values = [-2.5_f32, 0.0, 1.25, 7.5];
        let bytes = values
            .into_iter()
            .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = Tensor::new(vec![4], DType::Fp16, bytes.into()).unwrap();
        assert_eq!(tensor.to_f32().unwrap(), values);
    }

    #[test]
    fn test_contiguous_double() {
        let bytes = [-1.0_f64, 0.0, 1.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        let tensor = Tensor::new(vec![3], DType::Fp64, bytes.into()).unwrap();
        assert_eq!(tensor.to_f32().unwrap(), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_empty() {
        let tensor = Tensor::new(vec![0], DType::Fp32, Bytes::new()).unwrap();
        assert!(tensor.to_f32().unwrap().is_empty());
    }

    #[test]
    fn dtype_round_trips_through_wire_string() {
        for d in [DType::Fp32, DType::Int64, DType::Bool, DType::Fp16] {
            assert_eq!(DType::from_wire(d.as_str()).unwrap(), d);
        }
    }

    #[test]
    fn dtype_rejects_unknown_wire() {
        assert!(DType::from_wire("FLOAT").is_err());
    }

    #[test]
    fn dtype_byte_sizes_match_spec() {
        assert_eq!(DType::Fp32.size_bytes(), Some(4));
        assert_eq!(DType::Int64.size_bytes(), Some(8));
        assert_eq!(DType::Fp16.size_bytes(), Some(2));
        assert_eq!(DType::Bytes.size_bytes(), None);
    }

    #[test]
    fn from_f32_round_trips() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_f32(vec![2, 2], v.clone()).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.dtype, DType::Fp32);
        let back = t.to_f32().unwrap();
        for (a, b) in v.iter().zip(back.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 0.0);
        }
    }

    #[test]
    fn fp16_and_fp64_convert_to_f32() {
        let fp16 = [f16::from_f32(1.5), f16::from_f32(-2.25)];
        let fp16_bytes = fp16
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = Tensor::new(vec![2], DType::Fp16, fp16_bytes.into()).unwrap();
        assert_eq!(tensor.to_f32().unwrap(), vec![1.5, -2.25]);

        let fp64_bytes = [1.25_f64, -4.5]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();
        let tensor = Tensor::new(vec![2], DType::Fp64, fp64_bytes.into()).unwrap();
        assert_eq!(tensor.to_f32().unwrap(), vec![1.25, -4.5]);
    }

    #[test]
    fn from_i64_round_trips() {
        let v = vec![-1_i64, 0, 1, 1_000_000_000_000];
        let t = Tensor::from_i64(vec![4], v.clone()).unwrap();
        assert_eq!(t.dtype, DType::Int64);
        assert_eq!(t.to_i64().unwrap(), v);
    }

    #[test]
    fn int32_converts_to_i64() {
        let tensor = Tensor::from_i32(vec![2], vec![-7, 9]).unwrap();
        assert_eq!(tensor.to_i64().unwrap(), vec![-7, 9]);
    }

    #[test]
    fn from_i32_round_trips() {
        let tensor = Tensor::from_i32(vec![2], vec![-7, 9]).unwrap();
        assert_eq!(tensor.to_i32().unwrap(), vec![-7, 9]);
    }

    #[test]
    fn from_f32_rejects_shape_mismatch() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!(Tensor::from_f32(vec![2, 2], v).is_err());
    }

    #[test]
    fn to_f32_rejects_wrong_dtype() {
        let t = Tensor::from_i64(vec![3], vec![1, 2, 3]).unwrap();
        assert!(t.to_f32().is_err());
    }

    #[test]
    fn element_count_matches_product() {
        let t = Tensor::from_f32(vec![2, 3, 4], vec![0.0; 24]).unwrap();
        assert_eq!(t.element_count(), 24);
    }

    #[test]
    fn new_validates_byte_count() {
        let bytes = Bytes::from(vec![0u8; 8]);
        assert!(Tensor::new(vec![3], DType::Fp32, bytes.clone()).is_err()); // expects 12 bytes
        assert!(Tensor::new(vec![2], DType::Fp32, bytes).is_ok());
    }
}
