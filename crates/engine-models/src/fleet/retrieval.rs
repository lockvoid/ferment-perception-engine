//! Paired retrieval inventory copied from the verified 2026-11-15 FP16 exports.
//! No eager warmup: the text encoders are large and load only for a query.
//! Core ML passed all four modes; these exact ONNX bytes passed CUDA/DirectML
//! on A4000, including the public Rust frontend with the shipped Windows DLLs.
//! Architecture-specific Core AI objects are attached by the fleet selector.
//! Windows ML automatic NVIDIA TensorRT RTX also passed on 2026-11-16.
use super::{
    ArtifactSource, CoreMlComputeUnits, MODEL_BASE_URL, MODEL_VERSION, ModelDefinition, ModelId,
    OrtPlacement, Pins, Published, ResourceDefinition, published,
};

#[cfg(test)]
pub(super) const IDS: [ModelId; 4] = [
    ModelId::GlapAudio,
    ModelId::GlapText,
    ModelId::Siglip2Image,
    ModelId::Siglip2Text,
];

pub(super) fn models() -> [ModelDefinition; 4] {
    [
        published(
            ModelId::GlapAudio,
            Published {
                file_name: "glap_audio_mel_fp16.mlmodelc",
                tree_sha256: "f917697bdebdc0ee55125901e99c1df90f04f1db6090e1dde43a61f5f3af156f",
                manifest_sha256: "01d11edb7dc851037f95c7753dcc01b8888dfa4ae14656b4f5fc73781a4a5aad",
            },
            Published {
                file_name: "glap_audio_mel_fp16.onnx",
                tree_sha256: "b77661e27eee80262a02d3cb7cb7b9407649cbceaaa55768e0713b51f83679d7",
                manifest_sha256: "14cf022d455e2e9c0810886888aba76195239a713f05453fab9e9bb5cdd2ef59",
            },
            None,
            "Apache-2.0",
            "https://huggingface.co/mispeech/GLAP/tree/79b13014511e2d5628cb57c4712a90edd19cebbd",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
        published(
            ModelId::GlapText,
            Published {
                file_name: "glap_text_fp16.mlmodelc",
                tree_sha256: "094fa81063a71fdfa9826baebb58b529fe2c1586a97b9d631277c656a7f6b86e",
                manifest_sha256: "f7e657155a5c13397ede14987e1d059c0a2f30c5c4034ca6cc1b5cb551b1410b",
            },
            Published {
                file_name: "glap_text_fp16.onnx",
                tree_sha256: "61ea63a4fea79e8057c3bc551959b3f65cafe0267ca201bf1f2c4370480ed434",
                manifest_sha256: "a87d18d580fb7ef643e60e396e704bdc1a564e239bb5ccc39892708791a64d75",
            },
            None,
            "Apache-2.0",
            "https://huggingface.co/mispeech/GLAP/tree/79b13014511e2d5628cb57c4712a90edd19cebbd",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
        published(
            ModelId::Siglip2Image,
            Published {
                file_name: "siglip2_base_224_image_fp16.mlmodelc",
                tree_sha256: "01db5a200aa8af0669b6110679df19616cde6792863b63ba54121be84217e2e7",
                manifest_sha256: "8bd95d7ac0efb51081e2a80cdfc925c2d1b20c960773c2cba111d0698d3e432d",
            },
            Published {
                file_name: "siglip2_base_224_image_fp16.onnx",
                tree_sha256: "1791818ffd86dcb237d34cfac2444057f4f3286515d70c1e38adea074d0c742f",
                manifest_sha256: "9928b67548c94fc60a60a471f0c053cb6df4d7ea4fa313358ca76c5c9de260f9",
            },
            None,
            "Apache-2.0",
            "https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
        published(
            ModelId::Siglip2Text,
            Published {
                file_name: "siglip2_base_224_text_fp16.mlmodelc",
                tree_sha256: "633b24d990b3c7471eb11bd9bb3059c7860618c7f6f88d7631f9928cd9fcee53",
                manifest_sha256: "b634a1c53bae6fab087ed1746882bd0d5db39b46f74a7e7a412d997d0368be3e",
            },
            Published {
                file_name: "siglip2_base_224_text_fp16.onnx",
                tree_sha256: "7533560d4c542548e83edba556d7333d6360fdec24d5b95cb4c55c221db04182",
                manifest_sha256: "0b5f3b9a6bca88261619b01852bf45cd199dc5383a571c925c8a5601e6aee6d5",
            },
            None,
            "Apache-2.0",
            "https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            Pins {
                coreml: CoreMlComputeUnits::All,
                ort: OrtPlacement::WindowsMl,
            },
        ),
    ]
}

pub(super) fn resources() -> [ResourceDefinition; 2] {
    [
        ResourceDefinition {
            id: "glap_sentencepiece".into(), version: MODEL_VERSION.into(), file_name: "glap_sentencepiece.model".into(),
            source: ArtifactSource::ManifestPrefix { prefix_url: format!("{MODEL_BASE_URL}/glap_sentencepiece.model"), manifest_sha256: "182df253180f801583b4ae7996613a201469fa71aa426983cdd9eb3d7eba91fb".into() },
            sha256: Some("14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a".into()), bytes: Some(4852054),
            license: Some("Apache-2.0".into()), source_url: Some("https://huggingface.co/mispeech/GLAP/tree/79b13014511e2d5628cb57c4712a90edd19cebbd".into()),
        },
        ResourceDefinition {
            id: "siglip2_tokenizer".into(), version: MODEL_VERSION.into(), file_name: "siglip2_tokenizer.json".into(),
            source: ArtifactSource::ManifestPrefix { prefix_url: format!("{MODEL_BASE_URL}/siglip2_tokenizer.json"), manifest_sha256: "8a2ec6dae3f1dae5623643eb5ab931bc7f616943ad762a45289768e8dab1eabe".into() },
            sha256: Some("cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322".into()), bytes: Some(34363039),
            license: Some("Apache-2.0".into()), source_url: Some("https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2".into()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendKind, Placement};

    /// Kill: eagerly load a text encoder or substitute an unqualified placement.
    #[test]
    fn retrieval_is_lazy_and_qualified_on_both_base_lanes() {
        for model in models() {
            assert!(model.warmup_priority.is_none());
            assert_eq!(model.artifacts.len(), 2);
            for artifact in model.artifacts {
                let expected = match artifact.backend {
                    BackendKind::CoreAi => panic!("base fleet selects no Core AI architecture"),
                    BackendKind::CoreMl => Placement::CoreMl {
                        units: CoreMlComputeUnits::All,
                    },
                    BackendKind::Ort => Placement::Ort {
                        providers: OrtPlacement::WindowsMl,
                    },
                };
                assert_eq!(artifact.placement, Some(expected));
            }
        }
    }
}
