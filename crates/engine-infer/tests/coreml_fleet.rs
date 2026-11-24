//! Real-artifact smoke tests for the native Core ML backend.
//!
//! These are ignored in the normal suite because the model fleet is roughly
//! one gigabyte and is intentionally not checked into this repository. Run
//! them on macOS after placing the Swift fleet in `FERMENT_ENGINE_MODELS_DIR`.

#![cfg(all(feature = "coreml", target_vendor = "apple"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_infer::{CoreMlBackend, InferenceBackend, ModelId, ModelPaths, ModelVersion};

fn model_directory() -> PathBuf {
    std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ferment-cuts-ios/Packages/FermentEngine/LocalModels")
        })
}

fn fleet_paths(directory: &Path) -> ModelPaths {
    let mut paths = ModelPaths::new();
    for (id, file_name) in [
        (ModelId::BeatThis1, "beat_this_1.mlmodelc"),
        (ModelId::BeatThis2, "beat_this_2.mlmodelc"),
        (ModelId::BeatThis3, "beat_this_3.mlmodelc"),
        (ModelId::ParakeetEncoder, "parakeet_encoder.mlmodelc"),
        (ModelId::Dfnet, "dfnet.mlmodelc"),
        (ModelId::MusicHighlighter, "music_highlighter.mlmodelc"),
        (ModelId::DashengBase, "dasheng_base_mel.mlmodelc"),
        (ModelId::MusicTaggerHeads, "music_tagger_heads.mlmodelc"),
        (ModelId::Autoshot, "autoshot.mlmodelc"),
        (ModelId::FireRedVadAed, "fireredvad_aed.mlmodelc"),
    ] {
        paths.insert(id, ModelVersion::Latest, directory.join(file_name));
    }
    paths
}

#[test]
#[ignore = "requires the external 1 GB Core ML model fleet"]
fn loads_every_shipped_coreml_model() {
    let directory = model_directory();
    assert!(
        directory.is_dir(),
        "Core ML fleet not found at {}; set FERMENT_ENGINE_MODELS_DIR",
        directory.display()
    );
    let models = [
        ModelId::BeatThis1,
        ModelId::BeatThis2,
        ModelId::BeatThis3,
        ModelId::ParakeetEncoder,
        ModelId::Dfnet,
        ModelId::MusicHighlighter,
        ModelId::DashengBase,
        ModelId::MusicTaggerHeads,
        ModelId::Autoshot,
        ModelId::FireRedVadAed,
    ];
    let backend = CoreMlBackend::new(Arc::new(fleet_paths(&directory)));
    futures_executor::block_on(backend.warmup(&models)).unwrap();
}
