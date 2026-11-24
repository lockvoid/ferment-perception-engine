//! The inference boundary for the four EdgeTAM physical graphs.
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMModel.swift` (with
//! `EdgeTAMTensor.swift`'s shape table): the host holds every tensor in fp16,
//! as iOS does, and widens it at the door, which takes fp32 — the ONNX masters'
//! dtype (`ferment-models-export`, `models/edgetam/model.json`); a Core ML
//! fp16 bundle converts back at its own door. This type owns no video/session
//! state. It validates every tensor before and after inference.

use ferment_engine_core::{EngineError, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use half::f16;

use super::error::EdgeTamError;
use super::processing::PromptTensor;

/// Ported from `EdgeTAMTensorShape`.
pub mod shape {
    pub const NORMALIZED_IMAGE: [i64; 4] = [1, 3, 1024, 1024];
    pub const VISION_FEATURES: [i64; 4] = [1, 256, 64, 64];
    pub const HIGH_RESOLUTION_FEATURES_0: [i64; 4] = [1, 32, 256, 256];
    pub const HIGH_RESOLUTION_FEATURES_1: [i64; 4] = [1, 64, 128, 128];
    pub const LOW_RESOLUTION_MASK: [i64; 4] = [1, 1, 256, 256];
    pub const CANDIDATE_MASKS: [i64; 4] = [1, 4, 256, 256];
    pub const IOU_PREDICTIONS: [i64; 2] = [1, 4];
    pub const CANDIDATE_OBJECT_POINTERS: [i64; 3] = [1, 4, 256];
    pub const OBJECT_SCORE_LOGITS: [i64; 2] = [1, 1];
    pub const OBJECT_POINTER: [i64; 2] = [1, 256];
    pub const MEMORY_FEATURES: [i64; 3] = [1, 512, 64];
    pub const SPATIAL_MEMORY_FEATURES: [i64; 4] = [1, 7, 512, 64];
    pub const OBJECT_POINTERS: [i64; 3] = [1, 16, 256];
    pub const SPATIAL_MEMORY_VALIDITY: [i64; 2] = [1, 7];
    pub const OBJECT_POINTER_VALIDITY: [i64; 2] = [1, 16];

    #[must_use]
    pub fn point_coordinates(count: usize) -> Vec<i64> {
        vec![1, count as i64, 2]
    }

    #[must_use]
    pub fn point_labels(count: usize) -> Vec<i64> {
        vec![1, count as i64]
    }

    #[must_use]
    pub fn element_count(shape: &[i64]) -> usize {
        shape.iter().product::<i64>() as usize
    }
}

/// Immutable image features shared by prompt decoding and temporal memory.
#[derive(Debug, Clone)]
pub struct EdgeTamImageFeatures {
    pub vision_features: Vec<f16>,
    pub high_resolution_features_0: Vec<f16>,
    pub high_resolution_features_1: Vec<f16>,
}

impl EdgeTamImageFeatures {
    pub fn new(
        vision_features: Vec<f16>,
        high_resolution_features_0: Vec<f16>,
        high_resolution_features_1: Vec<f16>,
    ) -> Result<Self> {
        validate("vision_features", &vision_features, &shape::VISION_FEATURES)?;
        validate(
            "high_resolution_features_0",
            &high_resolution_features_0,
            &shape::HIGH_RESOLUTION_FEATURES_0,
        )?;
        validate(
            "high_resolution_features_1",
            &high_resolution_features_1,
            &shape::HIGH_RESOLUTION_FEATURES_1,
        )?;
        Ok(Self {
            vision_features,
            high_resolution_features_0,
            high_resolution_features_1,
        })
    }

    /// Reuse the encoder's high-resolution maps with a vision feature map
    /// conditioned by temporal memory attention.
    pub fn replacing_vision_features(&self, conditioned: Vec<f16>) -> Result<Self> {
        Self::new(
            conditioned,
            self.high_resolution_features_0.clone(),
            self.high_resolution_features_1.clone(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct EdgeTamSelectedCandidate {
    /// Selected low-resolution mask logits, `[1,1,256,256]` FP16.
    pub mask: Vec<f16>,
    /// Selected object pointer, `[1,256]` FP16.
    pub pointer: Vec<f16>,
    pub predicted_iou: f32,
    pub object_score: f32,
}

/// The four raw decoder candidates. Selection is deliberately outside the
/// graph so its prompt-dependent policy is deterministic and testable.
#[derive(Debug, Clone)]
pub struct EdgeTamDecodeOutput {
    pub candidate_masks: Vec<f16>,
    pub iou_predictions: Vec<f16>,
    pub candidate_object_pointers: Vec<f16>,
    pub object_score_logits: Vec<f16>,
}

impl EdgeTamDecodeOutput {
    pub fn new(
        candidate_masks: Vec<f16>,
        iou_predictions: Vec<f16>,
        candidate_object_pointers: Vec<f16>,
        object_score_logits: Vec<f16>,
    ) -> Result<Self> {
        validate("candidate_masks", &candidate_masks, &shape::CANDIDATE_MASKS)?;
        validate("iou_predictions", &iou_predictions, &shape::IOU_PREDICTIONS)?;
        validate(
            "candidate_object_pointers",
            &candidate_object_pointers,
            &shape::CANDIDATE_OBJECT_POINTERS,
        )?;
        validate(
            "object_score_logits",
            &object_score_logits,
            &shape::OBJECT_SCORE_LOGITS,
        )?;
        Ok(Self {
            candidate_masks,
            iou_predictions,
            candidate_object_pointers,
            object_score_logits,
        })
    }

    pub fn iou_prediction(&self, index: usize) -> std::result::Result<f32, EdgeTamError> {
        self.iou_predictions
            .get(index)
            .map(|value| value.to_f32())
            .ok_or(EdgeTamError::InvalidModelBoundary(format!(
                "iou_predictions has no candidate {index}"
            )))
    }

    /// Bit-exact FP16 extraction of one candidate.
    pub fn select_candidate(
        &self,
        index: usize,
    ) -> std::result::Result<EdgeTamSelectedCandidate, EdgeTamError> {
        if index >= 4 {
            return Err(EdgeTamError::InvalidModelBoundary(format!(
                "EdgeTAM candidate index {index} is outside 0...3."
            )));
        }
        let mask_stride = 256 * 256;
        let pointer_stride = 256;
        Ok(EdgeTamSelectedCandidate {
            mask: self.candidate_masks[index * mask_stride..(index + 1) * mask_stride].to_vec(),
            pointer: self.candidate_object_pointers
                [index * pointer_stride..(index + 1) * pointer_stride]
                .to_vec(),
            predicted_iou: self.iou_prediction(index)?,
            object_score: self.object_score_logits[0].to_f32(),
        })
    }
}

/// The narrow prompt-decoder seam used by image and video session policy.
///
/// Keeping this independent from model loading lets staged tests prove the
/// exact flags, prompt accumulation and previous-mask bytes presented to the
/// graph without replacing the encoder and memory models.
#[async_trait::async_trait(?Send)]
pub trait EdgeTamPromptDecoding {
    async fn decode(
        &self,
        features: &EdgeTamImageFeatures,
        prompt: &PromptTensor,
        add_no_memory_embedding: bool,
        apply_object_score_mask_gate: bool,
        mask_input: Option<&[f16]>,
    ) -> Result<EdgeTamDecodeOutput>;
}

#[derive(Debug, Clone)]
pub struct EdgeTamModel {
    backend: Backend,
    version: ModelVersion,
}

impl EdgeTamModel {
    /// The graphs this segmenter needs on disk.
    pub const MODEL_FILES: [ModelId; 4] = [
        ModelId::EdgetamImageEncoder,
        ModelId::EdgetamPromptMaskDecoder,
        ModelId::EdgetamMemoryEncoder,
        ModelId::EdgetamMemoryAttention,
    ];

    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    pub async fn encode_image(&self, normalized_image: &[f16]) -> Result<EdgeTamImageFeatures> {
        validate(
            "normalized_image",
            normalized_image,
            &shape::NORMALIZED_IMAGE,
        )?;
        let mut inputs = TensorMap::new();
        inputs.insert(
            "normalized_image".into(),
            door(&shape::NORMALIZED_IMAGE, normalized_image)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::EdgetamImageEncoder, &self.version, inputs)
            .await?;
        EdgeTamImageFeatures::new(
            required(&outputs, "vision_features", ModelId::EdgetamImageEncoder)?,
            required(
                &outputs,
                "high_resolution_features_0",
                ModelId::EdgetamImageEncoder,
            )?,
            required(
                &outputs,
                "high_resolution_features_1",
                ModelId::EdgetamImageEncoder,
            )?,
        )
    }

    pub async fn encode_memory(
        &self,
        raw_vision_features: &[f16],
        selected_low_resolution_mask: &[f16],
    ) -> Result<Vec<f16>> {
        validate(
            "raw_vision_features",
            raw_vision_features,
            &shape::VISION_FEATURES,
        )?;
        validate(
            "selected_low_resolution_mask",
            selected_low_resolution_mask,
            &shape::LOW_RESOLUTION_MASK,
        )?;
        let mut inputs = TensorMap::new();
        inputs.insert(
            "raw_vision_features".into(),
            door(&shape::VISION_FEATURES, raw_vision_features)?,
        );
        inputs.insert(
            "selected_low_resolution_mask".into(),
            door(&shape::LOW_RESOLUTION_MASK, selected_low_resolution_mask)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::EdgetamMemoryEncoder, &self.version, inputs)
            .await?;
        let memory_features = required(&outputs, "memory_features", ModelId::EdgetamMemoryEncoder)?;
        validate("memory_features", &memory_features, &shape::MEMORY_FEATURES)?;
        Ok(memory_features)
    }

    pub async fn condition(
        &self,
        raw_vision_features: &[f16],
        spatial_memory_features: &[f16],
        object_pointers: &[f16],
        spatial_validity: &[f16],
        pointer_validity: &[f16],
    ) -> Result<Vec<f16>> {
        validate(
            "raw_vision_features",
            raw_vision_features,
            &shape::VISION_FEATURES,
        )?;
        validate(
            "spatial_memory_features",
            spatial_memory_features,
            &shape::SPATIAL_MEMORY_FEATURES,
        )?;
        validate("object_pointers", object_pointers, &shape::OBJECT_POINTERS)?;
        validate(
            "spatial_memory_validity",
            spatial_validity,
            &shape::SPATIAL_MEMORY_VALIDITY,
        )?;
        validate(
            "object_pointer_validity",
            pointer_validity,
            &shape::OBJECT_POINTER_VALIDITY,
        )?;

        let mut inputs = TensorMap::new();
        inputs.insert(
            "raw_vision_features".into(),
            door(&shape::VISION_FEATURES, raw_vision_features)?,
        );
        inputs.insert(
            "spatial_memory_features".into(),
            door(&shape::SPATIAL_MEMORY_FEATURES, spatial_memory_features)?,
        );
        inputs.insert(
            "object_pointers".into(),
            door(&shape::OBJECT_POINTERS, object_pointers)?,
        );
        inputs.insert(
            "spatial_memory_validity".into(),
            door(&shape::SPATIAL_MEMORY_VALIDITY, spatial_validity)?,
        );
        inputs.insert(
            "object_pointer_validity".into(),
            door(&shape::OBJECT_POINTER_VALIDITY, pointer_validity)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::EdgetamMemoryAttention, &self.version, inputs)
            .await?;
        let conditioned = required(
            &outputs,
            "conditioned_vision_features",
            ModelId::EdgetamMemoryAttention,
        )?;
        validate(
            "conditioned_vision_features",
            &conditioned,
            &shape::VISION_FEATURES,
        )?;
        Ok(conditioned)
    }
}

#[async_trait::async_trait(?Send)]
impl EdgeTamPromptDecoding for EdgeTamModel {
    async fn decode(
        &self,
        features: &EdgeTamImageFeatures,
        prompt: &PromptTensor,
        add_no_memory_embedding: bool,
        apply_object_score_mask_gate: bool,
        mask_input: Option<&[f16]>,
    ) -> Result<EdgeTamDecodeOutput> {
        let point_count = prompt.labels.len();
        if !(1..=8).contains(&point_count) {
            return Err(EngineError::Inference(format!(
                "EdgeTAM supports 1...8 prompt tokens, not {point_count}."
            )));
        }
        if prompt.coordinates.len() != point_count * 2 {
            return Err(EngineError::Inference(format!(
                "EdgeTAM prompt has {} coordinate scalars and {point_count} labels.",
                prompt.coordinates.len()
            )));
        }
        if let Some(mask_input) = mask_input {
            validate("mask_input", mask_input, &shape::LOW_RESOLUTION_MASK)?;
        }

        let mut inputs = TensorMap::new();
        inputs.insert(
            "vision_features".into(),
            door(&shape::VISION_FEATURES, &features.vision_features)?,
        );
        inputs.insert(
            "high_resolution_features_0".into(),
            door(
                &shape::HIGH_RESOLUTION_FEATURES_0,
                &features.high_resolution_features_0,
            )?,
        );
        inputs.insert(
            "high_resolution_features_1".into(),
            door(
                &shape::HIGH_RESOLUTION_FEATURES_1,
                &features.high_resolution_features_1,
            )?,
        );
        inputs.insert(
            "point_coordinates".into(),
            Tensor::from_f32(
                shape::point_coordinates(point_count),
                prompt.coordinates.clone(),
            )?,
        );
        inputs.insert(
            "point_labels".into(),
            Tensor::from_f32(
                shape::point_labels(point_count),
                prompt.labels.iter().map(|label| *label as f32).collect(),
            )?,
        );
        inputs.insert(
            "add_no_memory_embedding".into(),
            Tensor::from_f32(vec![1], vec![flag(add_no_memory_embedding)])?,
        );
        inputs.insert(
            "apply_object_score_mask_gate".into(),
            Tensor::from_f32(vec![1], vec![flag(apply_object_score_mask_gate)])?,
        );
        // iOS selects one of sixteen specialized Core ML FUNCTIONS by
        // (point count, has-mask); ONNX has no multifunction container, so the
        // mask is always bound and `has_mask_input` gates it inside the graph —
        // the shape the upstream SAM2 ONNX export uses.
        inputs.insert(
            "mask_input".into(),
            match mask_input {
                Some(mask) => door(&shape::LOW_RESOLUTION_MASK, mask)?,
                None => Tensor::from_f32(
                    shape::LOW_RESOLUTION_MASK.to_vec(),
                    vec![0.0; shape::element_count(&shape::LOW_RESOLUTION_MASK)],
                )?,
            },
        );
        inputs.insert(
            "has_mask_input".into(),
            Tensor::from_f32(vec![1], vec![flag(mask_input.is_some())])?,
        );

        let outputs = self
            .backend
            .infer(ModelId::EdgetamPromptMaskDecoder, &self.version, inputs)
            .await?;
        EdgeTamDecodeOutput::new(
            required(
                &outputs,
                "candidate_masks",
                ModelId::EdgetamPromptMaskDecoder,
            )?,
            required(
                &outputs,
                "iou_predictions",
                ModelId::EdgetamPromptMaskDecoder,
            )?,
            required(
                &outputs,
                "candidate_object_pointers",
                ModelId::EdgetamPromptMaskDecoder,
            )?,
            required(
                &outputs,
                "object_score_logits",
                ModelId::EdgetamPromptMaskDecoder,
            )?,
        )
    }
}

fn flag(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

/// A host tensor widened to the door's fp32.
fn door(shape: &[i64], values: &[f16]) -> Result<Tensor> {
    Tensor::from_f32(
        shape.to_vec(),
        values.iter().map(|value| value.to_f32()).collect(),
    )
}

fn required(outputs: &TensorMap, name: &str, model: ModelId) -> Result<Vec<f16>> {
    outputs
        .get(name)
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: model.name().to_owned(),
            key: name.to_owned(),
        })?
        .to_f16()
}

fn validate(name: &str, values: &[f16], expected_shape: &[i64]) -> Result<()> {
    let expected = shape::element_count(expected_shape);
    if values.len() == expected {
        Ok(())
    } else {
        Err(EngineError::Inference(format!(
            "{name} has {} values; expected {expected} for shape {expected_shape:?}.",
            values.len()
        )))
    }
}
