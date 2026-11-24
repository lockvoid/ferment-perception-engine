use ferment_engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// Identity includes the checkpoint AND host preprocessing/pooling policy.
/// A new policy requires a new space; same dimension never means compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingSpace {
    #[serde(rename = "glap:79b13014:unit-window-v1")]
    GlapWindowV1,
    #[serde(rename = "siglip2-base-patch16-224:75de2d55:unit-v1")]
    Siglip2Base224V1,
}

impl EmbeddingSpace {
    pub const fn id(self) -> &'static str {
        match self {
            Self::GlapWindowV1 => "glap:79b13014:unit-window-v1",
            Self::Siglip2Base224V1 => "siglip2-base-patch16-224:75de2d55:unit-v1",
        }
    }

    pub const fn dimensions(self) -> usize {
        match self {
            Self::GlapWindowV1 => 1024,
            Self::Siglip2Base224V1 => 768,
        }
    }
}

/// Validated finite unit vector. Deserialization goes through the same
/// constructor so persisted or agent-supplied values cannot bypass validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StoredEmbedding")]
pub struct Embedding {
    space: EmbeddingSpace,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct StoredEmbedding {
    space: EmbeddingSpace,
    values: Vec<f32>,
}

impl TryFrom<StoredEmbedding> for Embedding {
    type Error = EngineError;
    fn try_from(value: StoredEmbedding) -> Result<Self> {
        Self::from_unit_values(value.space, value.values)
    }
}

impl Embedding {
    pub fn from_unit_values(space: EmbeddingSpace, values: Vec<f32>) -> Result<Self> {
        let norm = values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt();
        if values.len() != space.dimensions() || !norm.is_finite() || (norm - 1.0).abs() > 0.002 {
            return Err(EngineError::invalid_config(format!(
                "invalid unit embedding for {}",
                space.id()
            )));
        }
        Ok(Self { space, values })
    }

    pub fn space(&self) -> EmbeddingSpace {
        self.space
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn cosine(&self, other: &Self) -> Result<f32> {
        if self.space != other.space {
            return Err(EngineError::invalid_config(
                "cannot compare different embedding spaces",
            ));
        }
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for (&a, &b) in self.values.iter().zip(&other.values) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        Ok((dot / (norm_a * norm_b).sqrt()).clamp(-1.0, 1.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn unit(space: EmbeddingSpace) -> Embedding {
        let mut values = vec![0.0; space.dimensions()];
        values[0] = 1.0;
        Embedding::from_unit_values(space, values).unwrap()
    }
    #[test]
    fn invalid_vectors_and_mixed_spaces_are_rejected() {
        let space = EmbeddingSpace::GlapWindowV1;
        for values in [
            vec![],
            vec![0.0; 1024],
            vec![f32::NAN; 1024],
            vec![1.0; 1024],
            vec![f32::INFINITY; 1024],
        ] {
            assert!(Embedding::from_unit_values(space, values).is_err());
        }
        let glap = unit(space);
        assert_eq!(glap.cosine(&glap).unwrap(), 1.0);
        assert!(
            glap.cosine(&unit(EmbeddingSpace::Siglip2Base224V1))
                .is_err()
        );
        let mut json = serde_json::to_value(&glap).unwrap();
        json["values"] = serde_json::json!([0.0]);
        assert!(serde_json::from_value::<Embedding>(json).is_err());
    }
}
