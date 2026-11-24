//! Orchestrator: 768-dim embedding in, three-axis tag predictions out.
//! Pure inference — no audio decode here, the embedding comes from
//! `AudioSemanticEncoder`.

use ferment_engine_core::{EngineError, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};

use super::vocab::MusicTaggerVocab;
use crate::dasheng::config::EMBEDDING_DIM;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredTag {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicTagsResult {
    pub genre: Vec<ScoredTag>,
    pub mood: Vec<ScoredTag>,
    pub instrument: Vec<ScoredTag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisProbabilities {
    pub names: Vec<String>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTagProbabilities {
    pub genre: AxisProbabilities,
    pub mood: AxisProbabilities,
    pub instrument: AxisProbabilities,
}

#[derive(Debug, Clone)]
pub struct MusicTagger {
    backend: Backend,
    version: ModelVersion,
}

impl MusicTagger {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    /// Top-K tags per axis, filtered by per-tag thresholds. May return
    /// fewer than `top_k` per axis if not enough tags clear their threshold.
    #[tracing::instrument(skip(self, embedding), fields(processor = "music_tagger", top_k))]
    pub async fn tag(&self, embedding: &[f32], top_k: usize) -> Result<MusicTagsResult> {
        let probs = self.predict(embedding).await?;
        let v = MusicTaggerVocab::load()?;
        Ok(MusicTagsResult {
            genre: top_tags(&probs.genre, &v.genre_tags, &v.genre_thresholds, top_k),
            mood: top_tags(&probs.mood, &v.mood_tags, &v.mood_thresholds, top_k),
            instrument: top_tags(
                &probs.instrument,
                &v.instrument_tags,
                &v.instrument_thresholds,
                top_k,
            ),
        })
    }

    /// Raw per-axis sigmoid probabilities (full vocab, no threshold filtering).
    pub async fn raw_probabilities(&self, embedding: &[f32]) -> Result<RawTagProbabilities> {
        let probs = self.predict(embedding).await?;
        let v = MusicTaggerVocab::load()?;
        Ok(RawTagProbabilities {
            genre: AxisProbabilities {
                names: v.genre_tags.clone(),
                scores: probs.genre,
            },
            mood: AxisProbabilities {
                names: v.mood_tags.clone(),
                scores: probs.mood,
            },
            instrument: AxisProbabilities {
                names: v.instrument_tags.clone(),
                scores: probs.instrument,
            },
        })
    }

    async fn predict(&self, embedding: &[f32]) -> Result<AxisProbs> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(EngineError::invalid_config(format!(
                "embedding must be {EMBEDDING_DIM}-dim, got {}",
                embedding.len()
            )));
        }
        let mut inputs = TensorMap::new();
        inputs.insert(
            "embedding".into(),
            Tensor::from_f32(vec![1, EMBEDDING_DIM as i64], embedding.to_vec())?,
        );
        let out = self
            .backend
            .infer(ModelId::MusicTaggerHeads, &self.version, inputs)
            .await?;
        Ok(AxisProbs {
            genre: take_axis(&out, "genre_probs")?,
            mood: take_axis(&out, "mood_probs")?,
            instrument: take_axis(&out, "instrument_probs")?,
        })
    }
}

struct AxisProbs {
    genre: Vec<f32>,
    mood: Vec<f32>,
    instrument: Vec<f32>,
}

fn take_axis(map: &TensorMap, key: &str) -> Result<Vec<f32>> {
    let t = map
        .get(key)
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: "music_tagger_heads".into(),
            key: key.into(),
        })?;
    t.to_f32()
}

/// Tags whose `prob ≥ threshold[i]`, sorted descending, capped at `top_k`.
pub(crate) fn top_tags(
    probs: &[f32],
    names: &[String],
    thresholds: &[f32],
    top_k: usize,
) -> Vec<ScoredTag> {
    debug_assert_eq!(probs.len(), names.len());
    debug_assert_eq!(probs.len(), thresholds.len());
    let mut scored: Vec<ScoredTag> = probs
        .iter()
        .zip(names.iter())
        .zip(thresholds.iter())
        .filter_map(|((&p, n), &t)| {
            if p >= t {
                Some(ScoredTag {
                    name: n.clone(),
                    score: p,
                })
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use ferment_engine_infer::InferenceBackend;

    use super::*;

    #[derive(Debug)]
    struct FixedHeadsBackend;

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl InferenceBackend for FixedHeadsBackend {
        async fn infer(
            &self,
            _model: ModelId,
            _version: &ModelVersion,
            _inputs: TensorMap,
        ) -> Result<TensorMap> {
            let vocab = MusicTaggerVocab::load()?;
            let mut outputs = TensorMap::new();
            for (name, count) in [
                ("genre_probs", vocab.genre_tags.len()),
                ("mood_probs", vocab.mood_tags.len()),
                ("instrument_probs", vocab.instrument_tags.len()),
            ] {
                outputs.insert(
                    name.into(),
                    Tensor::from_f32(vec![1, count as i64], vec![0.0; count])?,
                );
            }
            Ok(outputs)
        }
    }

    #[test]
    fn test_top_tags_respects_threshold() {
        let probabilities = [0.9, 0.4, 0.6, 0.2, 0.8];
        let names = ["a", "b", "c", "d", "e"].map(String::from);
        let thresholds = [0.5; 5];
        let result = top_tags(&probabilities, &names, &thresholds, 10);
        assert_eq!(
            result
                .iter()
                .map(|tag| tag.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "e", "c"]
        );
    }

    #[test]
    fn test_top_tags_respects_top_k() {
        let probabilities = [0.9, 0.85, 0.8, 0.75, 0.7];
        let names = ["a", "b", "c", "d", "e"].map(String::from);
        let result = top_tags(&probabilities, &names, &[0.0; 5], 2);
        assert_eq!(
            result
                .iter()
                .map(|tag| tag.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn test_top_tags_per_tag_thresholds() {
        let probabilities = [0.5; 3];
        let names = ["a", "b", "c"].map(String::from);
        let result = top_tags(&probabilities, &names, &[0.4, 0.6, 0.5], 5);
        let mut result = result.into_iter().map(|tag| tag.name).collect::<Vec<_>>();
        result.sort();
        assert_eq!(result, ["a", "c"]);
    }

    #[test]
    fn test_top_tags_handles_empty_above_threshold() {
        let probabilities = [0.1, 0.2, 0.3];
        let names = ["a", "b", "c"].map(String::from);
        assert!(top_tags(&probabilities, &names, &[0.5; 3], 5).is_empty());
    }

    #[test]
    fn test_tag_from_zero_embedding() {
        let tagger = MusicTagger::new(Arc::new(FixedHeadsBackend));
        let result = futures_executor::block_on(tagger.tag(&[0.0; EMBEDDING_DIM], 5)).unwrap();
        assert!(result.genre.len() <= 5);
        assert!(result.mood.len() <= 5);
        assert!(result.instrument.len() <= 5);
    }

    #[test]
    fn test_tag_from_wrong_dim_embedding_throws() {
        let tagger = MusicTagger::new(Arc::new(FixedHeadsBackend));
        assert!(futures_executor::block_on(tagger.tag(&[0.0; 100], 5)).is_err());
    }
}
