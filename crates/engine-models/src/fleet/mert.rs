//! The two official MERT-v1 checkpoints, qualified 2026-11-15.
//! Feature extraction is lazy; the caller chooses the layer and aggregation.
//! ONNX temporal means accumulate in FP32: half CUDA reductions overflowed.
//! The FP16 weighted trunk and both outputs pass the full A4000 GPU corpus.
//! Windows ML automatic NVIDIA TensorRT RTX also passed on 2026-11-16.
use super::{
    CoreMlComputeUnits, ModelDefinition, ModelId, OrtPlacement, Pins, Published, published,
};

#[cfg(test)]
pub(super) const IDS: [ModelId; 2] = [ModelId::Mert95M, ModelId::Mert330M];

pub(super) fn models() -> [ModelDefinition; 2] {
    [
        published(
            ModelId::Mert95M,
            Published {
                file_name: "mert_v1_95m_fp16.mlmodelc",
                tree_sha256: "56380ad05de604c6910b32040b8a48527af7749c235009913cc05f1824a5ad21",
                manifest_sha256: "053f20ba303246d6317bec6ef3da54088158094e49257379c441821c0980c90b",
            },
            Published {
                file_name: "mert_v1_95m_fp16.onnx",
                tree_sha256: "8da785bbf4902c3850326f5a26fe0850e66170a54b89bdd3e179a93507c28e55",
                manifest_sha256: "6ab465796c58988f141f2e907c3e3ccd77fb077002b5d3ca66f3a4aea47e72c0",
            },
            None,
            "CC-BY-NC-4.0",
            "https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
        published(
            ModelId::Mert330M,
            Published {
                file_name: "mert_v1_330m_fp16.mlmodelc",
                tree_sha256: "bdc1fc5b738f6d0fbc5995ed0a3fcf04c5d752a23df837acc9b1c1cb6f9327ac",
                manifest_sha256: "7e656d867d7d2fafa1142ac1cf9c565353989a3382b7bea45c27bd3b11c2c5d3",
            },
            Published {
                file_name: "mert_v1_330m_fp16.onnx",
                tree_sha256: "078d00c0019ff1f069f9a5c89c020b0a05696807f2199abd66bd1d9c3d2dc73f",
                manifest_sha256: "73748f11fbfc5b42018d151585f930a1631380df2df72b46ba497bf945b46b41",
            },
            None,
            "CC-BY-NC-4.0",
            "https://huggingface.co/m-a-p/MERT-v1-330M/tree/5240c2708a5acaee1007f43fb9735c7dcd0b78c9",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_test_fixtures::export;

    /// Kill: substitute the source-code license for the weights' NC license,
    /// change a checkpoint revision, or eagerly load these large feature models.
    #[test]
    fn mert_preserves_the_checkpoint_provenance_and_lazy_loading() {
        let Some(path) = export::path("mert/model.json") else {
            return;
        };
        let contract: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for model in models() {
            assert!(model.warmup_priority.is_none());
            let source = &contract["members"][&model.id]["source"];
            for artifact in model.artifacts {
                assert_eq!(artifact.license.as_deref(), Some("CC-BY-NC-4.0"));
                assert_eq!(artifact.license.as_deref(), source["license"].as_str());
                assert_eq!(
                    artifact.source_url.as_deref(),
                    source["source_url"].as_str()
                );
            }
        }
    }
}
