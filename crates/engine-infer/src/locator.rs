use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_core::{EngineError, EngineService, Result};

use crate::{ModelId, ModelVersion};

/// Resolves a logical model/version to the artifact selected for this backend.
pub trait ModelLocator: EngineService + std::fmt::Debug {
    fn locate(&self, model: ModelId, version: &ModelVersion) -> Result<PathBuf>;
}

pub type SharedModelLocator = Arc<dyn ModelLocator>;

/// In-memory locator used by desktop applications after `engine-models` has
/// selected and verified the appropriate platform artifacts.
#[derive(Debug, Clone, Default)]
pub struct ModelPaths {
    paths: HashMap<(ModelId, ModelVersion), PathBuf>,
}

impl ModelPaths {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_path(
        mut self,
        model: ModelId,
        version: ModelVersion,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.insert(model, version, path);
        self
    }

    pub fn insert(
        &mut self,
        model: ModelId,
        version: ModelVersion,
        path: impl Into<PathBuf>,
    ) -> Option<PathBuf> {
        self.paths.insert((model, version), path.into())
    }

    pub fn latest(&self, model: ModelId) -> Option<&Path> {
        self.paths
            .get(&(model, ModelVersion::Latest))
            .map(PathBuf::as_path)
    }
}

impl ModelLocator for ModelPaths {
    fn locate(&self, model: ModelId, version: &ModelVersion) -> Result<PathBuf> {
        self.paths
            .get(&(model, version.clone()))
            .cloned()
            .ok_or_else(|| {
                EngineError::Models(format!(
                    "no artifact path configured for {model} version '{}'",
                    version.as_str()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_and_latest_paths_are_distinct() {
        let paths = ModelPaths::new()
            .with_path(ModelId::DashengBase, ModelVersion::Latest, "/latest.onnx")
            .with_path(ModelId::DashengBase, ModelVersion::pinned("2"), "/v2.onnx");
        assert_eq!(
            paths
                .locate(ModelId::DashengBase, &ModelVersion::Latest)
                .unwrap(),
            PathBuf::from("/latest.onnx")
        );
        assert_eq!(
            paths
                .locate(ModelId::DashengBase, &ModelVersion::pinned("2"))
                .unwrap(),
            PathBuf::from("/v2.onnx")
        );
    }
}
