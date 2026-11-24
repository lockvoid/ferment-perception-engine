//! Runtime model fleet manifest, integrity validation, and versioned cache.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferment_engine_core::{EngineError, Result};
use ferment_engine_infer::ModelId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod fleet;
mod verified_cache;

pub use fleet::{
    MODEL_FLEET_VERSION, SWIFT_MODEL_BASE_URL, default_fleet_manifest, fleet_manifest_for_coreai,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    CoreAi,
    CoreMl,
    Ort,
}

impl BackendKind {
    fn directory(self) -> &'static str {
        match self {
            Self::CoreAi => "coreai",
            Self::CoreMl => "coreml",
            Self::Ort => "ort",
        }
    }
}

/// Where one compiled Core ML bundle runs. The four `MLComputeUnits` cases,
/// under the names iOS's `ModelPlacement.units` uses, so a pin can be read
/// against `FermentEngine`'s tables without translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlComputeUnits {
    /// Let Core ML's planner choose. The value every artifact had before this
    /// axis existed, and now only a fallback for an unpinned artifact.
    All,
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
}

/// Which ONNX Runtime execution providers an artifact is qualified for. CPU is
/// always the floor — every variant ends there — so this names the accelerator
/// that may be *tried* first, not one that is required.
///
/// The values are the 016 ship matrix (`migration/016-report.md` §4.2/§4.1):
/// a family is only lifted off [`Self::Cpu`] where a GPU provider was actually
/// measured against the oracle and passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrtPlacement {
    /// CPU everywhere. Either the only column 016 measured for this family, or
    /// the column a measured GPU provider failed.
    Cpu,
    /// Windows ML catalog and automatic device policy on Windows, CPU elsewhere.
    /// Qualified against the A4000 TensorRT RTX corpus in November 2026.
    WindowsMl,
    /// Explicit legacy DirectML placement; CPU alone elsewhere. The shipping
    /// fleet uses WindowsMl; retained for callers with older manifests.
    DirectMl,
    /// The ONNX Runtime Core ML provider ahead of CPU on Apple; CPU alone
    /// anywhere else. No fleet entry carries this yet — it is the shape a
    /// promotion takes once the measurement earns it, and DINOv2 is the
    /// candidate: iOS runs it on the ANE from a compiled bundle Studio does
    /// not have, so the ORT provider is the only route to the same hardware.
    CoreMl,
}

/// The placement axis: where a model's graph runs *within* the lane its
/// artifact belongs to. The lane itself is chosen upstream by which backend
/// has a non-`Missing` artifact — placement never moves a model between lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "lane", rename_all = "snake_case")]
pub enum Placement {
    /// The qualified compiled model with Core AI's default device planner.
    CoreAi,
    CoreMl {
        units: CoreMlComputeUnits,
    },
    Ort {
        providers: OrtPlacement,
    },
}

impl Placement {
    /// The lane this placement can be applied to. Cross-lane placements are
    /// rejected by [`ModelManifest::validate`].
    #[must_use]
    pub fn backend(self) -> BackendKind {
        match self {
            Self::CoreAi => BackendKind::CoreAi,
            Self::CoreMl { .. } => BackendKind::CoreMl,
            Self::Ort { .. } => BackendKind::Ort,
        }
    }

    #[must_use]
    pub fn compute_units(self) -> Option<CoreMlComputeUnits> {
        match self {
            Self::CoreMl { units } => Some(units),
            Self::CoreAi | Self::Ort { .. } => None,
        }
    }

    #[must_use]
    pub fn ort_providers(self) -> Option<OrtPlacement> {
        match self {
            Self::Ort { providers } => Some(providers),
            Self::CoreAi | Self::CoreMl { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactSource {
    /// Shipped with the application/package rather than downloaded.
    Bundled,
    /// A prefix of objects plus the manifest that names them: the CDN carries
    /// `<prefix_url>/<path>` for every entry of `<prefix_url>.manifest.json`,
    /// and those objects ARE the artifact — there is no archive to unpack.
    ///
    /// `manifest_sha256` is the digest of the manifest's exact bytes, compiled
    /// in here rather than read from the CDN's `.sha256` sidecar. That makes
    /// the pin a stronger trust anchor than iOS's (which fetches the sidecar):
    /// the set of files a client will accept is fixed at build time, so a
    /// swapped manifest is refused for the price of the 2 KB it took to read
    /// it — before a single payload byte moves. The sidecar still ships beside
    /// the manifest for tooling parity; nothing in this crate reads it.
    ManifestPrefix {
        prefix_url: String,
        manifest_sha256: String,
    },
    Missing {
        reason: String,
    },
}

/// What one artifact is made of on the CDN, and what every piece must hash to.
///
/// The byte format is the iOS pipeline's, exactly — compact JSON
/// `{"files":[{"path","bytes","sha256"},…]}` with one trailing newline, as
/// `Scripts/cdn/make-manifests.py` writes it and
/// `ferment-cuts-ios`'s `ModelFileManifest` reads it. It has to be exact: the
/// digest of those bytes is what [`ArtifactSource::ManifestPrefix`] pins, so a
/// re-serialization that moved a separator would fail the pin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFileManifest {
    pub files: Vec<ModelFileEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFileEntry {
    /// Relative to the artifact's root: the CDN object is `<prefix>/<path>`
    /// and the staged file is `<staging>/<path>`.
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

impl ModelFileManifest {
    /// Decode, then refuse. Every byte here arrived from the network, so
    /// nothing in it is trusted until it has been checked: the paths become
    /// filesystem writes and the digests are the only proof the payload is the
    /// artifact.
    pub fn parse(bytes: &[u8], label: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(bytes).map_err(|error| {
            EngineError::Models(format!("manifest for '{label}' could not be read: {error}"))
        })?;
        manifest.validate(label)?;
        Ok(manifest)
    }

    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|entry| entry.bytes).sum()
    }

    pub fn validate(&self, label: &str) -> Result<()> {
        if self.files.is_empty() {
            return Err(EngineError::Models(format!(
                "manifest for '{label}' lists no files"
            )));
        }
        // Case-folded, because both shipping filesystems are case-insensitive
        // by default (APFS, NTFS): `Weight.bin` and `weight.bin` are two
        // manifest entries and one file on disk, and whichever landed second
        // would silently win.
        let mut claimed = std::collections::HashSet::new();
        for entry in &self.files {
            if !is_safe_relative_path(&entry.path) {
                return Err(EngineError::Models(format!(
                    "manifest for '{label}' names an unsafe path: '{}'",
                    entry.path
                )));
            }
            validate_digest(&entry.sha256, &entry.path)?;
            if !claimed.insert(entry.path.to_lowercase()) {
                return Err(EngineError::Models(format!(
                    "manifest for '{label}' lists '{}' twice",
                    entry.path
                )));
            }
        }
        // `a/b` and `a` cannot both exist — one of them needs `a` to be a
        // directory and the other needs it to be a file.
        for entry in &self.files {
            let segments: Vec<&str> = entry.path.split('/').collect();
            let mut ancestor = String::new();
            for segment in &segments[..segments.len() - 1] {
                if !ancestor.is_empty() {
                    ancestor.push('/');
                }
                ancestor.push_str(segment);
                if claimed.contains(&ancestor.to_lowercase()) {
                    return Err(EngineError::Models(format!(
                        "manifest for '{label}' needs '{}' to be both a file and a directory",
                        entry.path
                    )));
                }
            }
        }
        Ok(())
    }
}

/// A path that can only ever land INSIDE the staging directory. `..`, absolute
/// paths, NUL and empty segments are the whole attack surface of a manifest —
/// it is a network document that names files to write. Backslashes go too,
/// which iOS does not need to say: on Windows they are a second separator, and
/// `a\..\..\b` would escape a check that only knows about `/`.
fn is_safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\0') || path.contains('\\') {
        return false;
    }
    path.split('/')
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub backend: BackendKind,
    pub file_name: String,
    pub source: ArtifactSource,
    /// Lowercase SHA-256 of the downloaded file or staged directory tree.
    pub sha256: Option<String>,
    /// Provenance license: an SPDX-style identifier for third-party artifacts
    /// (`BSD-3-Clause`, `MIT`, …) or `proprietary-internal` for first-party
    /// exports. Required — with [`Self::source_url`] — for every non-`Missing`
    /// artifact; [`ModelManifest::validate`] rejects a manifest that omits it,
    /// so no third-party weight can enter the fleet without recorded provenance.
    #[serde(default)]
    pub license: Option<String>,
    /// Where the artifact came from: an `https://` URL for third-party
    /// downloads, or an `internal:` reference to the first-party pipeline that
    /// produced it (e.g. `internal:scripts/engine_export_onnx.py`).
    #[serde(default)]
    pub source_url: Option<String>,
    /// Where this artifact's graph runs: Core ML compute units on the mlmodelc
    /// lane, execution-provider preference on the ORT lane. Required for every
    /// artifact that ships bytes and forbidden on a `Missing` one, which has no
    /// graph to place — the same rule the integrity fields already follow.
    #[serde(default)]
    pub placement: Option<Placement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    pub version: String,
    pub artifacts: Vec<ModelArtifact>,
    pub warmup_priority: Option<u16>,
}

impl ModelDefinition {
    pub fn model_id(&self) -> Option<ModelId> {
        all_model_ids()
            .into_iter()
            .find(|candidate| candidate.name() == self.id)
    }

    /// The model's one artifact on `backend`'s lane, a declared gap included.
    #[must_use]
    pub fn artifact_for(&self, backend: BackendKind) -> Option<&ModelArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.backend == backend)
    }

    /// Where this model runs on one lane. `None` when the lane carries no
    /// artifact or only a declared gap — an absent placement means "this lane
    /// does not serve this model", which is exactly what the router already
    /// reads `Missing` for.
    #[must_use]
    pub fn placement_for(&self, backend: BackendKind) -> Option<Placement> {
        self.artifact_for(backend)?.placement
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub version: String,
    pub models: Vec<ModelDefinition>,
    /// Small non-model inputs that processors load alongside the fleet.
    #[serde(default)]
    pub resources: Vec<ResourceDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDefinition {
    pub id: String,
    pub version: String,
    pub file_name: String,
    pub source: ArtifactSource,
    /// Lowercase SHA-256 of the final resource bytes.
    pub sha256: Option<String>,
    pub bytes: Option<u64>,
    /// See [`ModelArtifact::license`] — same provenance contract for the small
    /// non-model inputs (vocabularies, thresholds) bundled with the fleet.
    #[serde(default)]
    pub license: Option<String>,
    /// See [`ModelArtifact::source_url`].
    #[serde(default)]
    pub source_url: Option<String>,
}

impl ModelManifest {
    pub fn model(&self, id: ModelId) -> Option<&ModelDefinition> {
        self.models.iter().find(|model| model.id == id.name())
    }

    pub fn resource(&self, id: &str) -> Option<&ResourceDefinition> {
        self.resources.iter().find(|resource| resource.id == id)
    }

    /// The whole placement table for one lane, in manifest order — what a
    /// backend constructor reads to pin its models. Models the lane does not
    /// serve are absent, not defaulted.
    #[must_use]
    pub fn placements(&self, backend: BackendKind) -> Vec<(ModelId, Placement)> {
        self.models
            .iter()
            .filter_map(|model| Some((model.model_id()?, model.placement_for(backend)?)))
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.version.trim().is_empty() {
            return Err(EngineError::Models(
                "manifest version cannot be empty".into(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for model in &self.models {
            if model.model_id().is_none() {
                return Err(EngineError::Models(format!(
                    "unknown model id '{}'",
                    model.id
                )));
            }
            if !ids.insert(&model.id) {
                return Err(EngineError::Models(format!(
                    "duplicate model id '{}'",
                    model.id
                )));
            }
            if model.artifacts.is_empty() {
                return Err(EngineError::Models(format!(
                    "model '{}' has no artifacts",
                    model.id
                )));
            }
            validate_component("model version", &model.version)?;

            let mut lanes = std::collections::HashSet::new();
            for artifact in &model.artifacts {
                validate_artifact(artifact)?;
                if !lanes.insert(artifact.backend) {
                    return Err(EngineError::Models(format!(
                        "model '{}' has two {:?} artifacts",
                        model.id, artifact.backend
                    )));
                }
            }
        }

        let mut resource_ids = std::collections::HashSet::new();
        for resource in &self.resources {
            validate_component("resource id", &resource.id)?;
            validate_component("resource version", &resource.version)?;
            validate_file_name(&resource.file_name)?;
            validate_source(&resource.source, &resource.file_name)?;
            validate_optional_digest(resource.sha256.as_deref(), &resource.file_name)?;
            if matches!(resource.source, ArtifactSource::Missing { .. }) {
                if resource.sha256.is_some() || resource.bytes.is_some() {
                    return Err(EngineError::Models(format!(
                        "missing resource '{}' cannot declare integrity or size metadata",
                        resource.file_name
                    )));
                }
            } else {
                validate_provenance(
                    resource.license.as_deref(),
                    resource.source_url.as_deref(),
                    &resource.file_name,
                )?;
            }
            if !resource_ids.insert(&resource.id) {
                return Err(EngineError::Models(format!(
                    "duplicate resource id '{}'",
                    resource.id
                )));
            }
        }
        Ok(())
    }

    /// Validate that a deployment manifest accounts for every inference
    /// identifier, including explicit `Missing` entries for unfinished ports.
    pub fn validate_complete_fleet(&self) -> Result<()> {
        self.validate()?;
        for id in all_model_ids() {
            if self.model(id).is_none() {
                return Err(EngineError::Models(format!(
                    "manifest does not account for model '{}'",
                    id.name()
                )));
            }
        }
        Ok(())
    }
}

fn validate_artifact(artifact: &ModelArtifact) -> Result<()> {
    validate_file_name(&artifact.file_name)?;
    validate_source(&artifact.source, &artifact.file_name)?;
    validate_optional_digest(artifact.sha256.as_deref(), &artifact.file_name)?;
    if matches!(artifact.source, ArtifactSource::ManifestPrefix { .. }) {
        // Three checks guard a manifest-lane install: the compiled-in manifest
        // pin, then each file's size and digest, then this — the hash of the
        // whole staged tree, re-taken by the store before it adopts anything.
        // Without it a manifest that validated could still install a set the
        // fleet never qualified.
        if artifact.sha256.is_none() {
            return Err(EngineError::Models(format!(
                "manifest-prefix artifact '{}' has no installed-tree digest",
                artifact.file_name
            )));
        }
    }
    if matches!(artifact.source, ArtifactSource::Missing { .. }) {
        if artifact.sha256.is_some() {
            return Err(EngineError::Models(format!(
                "missing artifact '{}' cannot declare an integrity digest",
                artifact.file_name
            )));
        }
        if artifact.placement.is_some() {
            return Err(EngineError::Models(format!(
                "missing artifact '{}' cannot declare a placement: it has no graph to place",
                artifact.file_name
            )));
        }
    } else {
        validate_provenance(
            artifact.license.as_deref(),
            artifact.source_url.as_deref(),
            &artifact.file_name,
        )?;
        validate_placement(artifact)?;
    }
    Ok(())
}

/// Every artifact that ships bytes says where it runs, and says it for its own
/// lane. A missing pin is not a benign default here: it is the difference
/// between a graph that was measured onto a compute unit and one that fell back
/// to whatever the planner picked, and that difference is the whole axis.
fn validate_placement(artifact: &ModelArtifact) -> Result<()> {
    let placement = artifact.placement.ok_or_else(|| {
        EngineError::Models(format!(
            "artifact '{}' has no placement: every shipped artifact declares where it runs",
            artifact.file_name
        ))
    })?;
    if placement.backend() == artifact.backend {
        Ok(())
    } else {
        Err(EngineError::Models(format!(
            "artifact '{}' is a {:?} artifact but carries a {:?} placement",
            artifact.file_name,
            artifact.backend,
            placement.backend()
        )))
    }
}

/// Every artifact/resource that actually ships bytes must record both a license
/// and a source. This is the manifest-level enforcement of the 016 rule: no
/// third-party artifact enters the fleet without auditable provenance (the
/// downstream hook is a `cargo deny`-style provenance audit over the manifest).
/// `Missing` entries are exempt — they are declared gaps, not shipped bytes.
fn validate_provenance(license: Option<&str>, source_url: Option<&str>, label: &str) -> Result<()> {
    if license.map(str::trim).unwrap_or_default().is_empty() {
        return Err(EngineError::Models(format!(
            "artifact '{label}' has no license: every shipped artifact must record provenance"
        )));
    }
    if source_url.map(str::trim).unwrap_or_default().is_empty() {
        return Err(EngineError::Models(format!(
            "artifact '{label}' has no source_url: every shipped artifact must record provenance"
        )));
    }
    Ok(())
}

fn validate_source(source: &ArtifactSource, label: &str) -> Result<()> {
    match source {
        ArtifactSource::Bundled => Ok(()),
        ArtifactSource::ManifestPrefix {
            prefix_url,
            manifest_sha256,
        } => {
            if !(prefix_url.starts_with("https://") || prefix_url.starts_with("http://")) {
                return Err(EngineError::Models(format!(
                    "manifest prefix for '{label}' is not an HTTP(S) URL"
                )));
            }
            // The prefix is joined two different ways — `<prefix>/<path>` for
            // the objects and `<prefix>.manifest.json` for the document — and
            // a trailing slash silently breaks the second one.
            if prefix_url.ends_with('/') {
                return Err(EngineError::Models(format!(
                    "manifest prefix for '{label}' must not end in '/'"
                )));
            }
            validate_digest(manifest_sha256, label)
        }
        ArtifactSource::Missing { reason } => {
            if reason.trim().is_empty() {
                Err(EngineError::Models(format!(
                    "missing source for '{label}' has no reason"
                )))
            } else {
                Ok(())
            }
        }
    }
}

fn validate_optional_digest(digest: Option<&str>, label: &str) -> Result<()> {
    match digest {
        Some(value) => validate_digest(value, label),
        None => Ok(()),
    }
}

fn validate_digest(digest: &str, label: &str) -> Result<()> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(EngineError::Models(format!(
            "'{label}' has an invalid lowercase SHA-256 digest"
        )))
    }
}

fn validate_component(kind: &str, value: &str) -> Result<()> {
    if value.trim().is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
    {
        Err(EngineError::Models(format!(
            "{kind} '{value}' is not a safe path component"
        )))
    } else {
        Ok(())
    }
}

fn validate_file_name(file_name: &str) -> Result<()> {
    validate_component("artifact file name", file_name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    Missing,
    Unverified,
    Ready,
    Corrupt,
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
    manifest: ModelManifest,
    verified: std::sync::Arc<verified_cache::VerifiedCache>,
}

impl ModelStore {
    pub fn new(root: impl Into<PathBuf>, manifest: ModelManifest) -> Result<Self> {
        manifest.validate()?;
        Ok(Self {
            root: root.into(),
            manifest,
            verified: Default::default(),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn artifact_path(&self, model: &ModelDefinition, artifact: &ModelArtifact) -> PathBuf {
        self.root
            .join(&self.manifest.version)
            .join(artifact.backend.directory())
            .join(&model.id)
            .join(&model.version)
            .join(&artifact.file_name)
    }

    pub fn resource_path(&self, resource: &ResourceDefinition) -> PathBuf {
        self.root
            .join(&self.manifest.version)
            .join("resources")
            .join(&resource.id)
            .join(&resource.version)
            .join(&resource.file_name)
    }

    pub fn availability(
        &self,
        model: &ModelDefinition,
        artifact: &ModelArtifact,
    ) -> Result<Availability> {
        self.verified.availability(
            &self.artifact_path(model, artifact),
            artifact.sha256.as_deref(),
        )
    }

    pub fn resource_availability(&self, resource: &ResourceDefinition) -> Result<Availability> {
        self.verified
            .availability(&self.resource_path(resource), resource.sha256.as_deref())
    }

    /// Install a fetched single-file artifact. The caller owns networking;
    /// this method verifies before any final path is visible.
    pub fn install_bytes(
        &self,
        model: &ModelDefinition,
        artifact: &ModelArtifact,
        bytes: &[u8],
    ) -> Result<PathBuf> {
        let expected = artifact.sha256.as_deref().ok_or_else(|| {
            EngineError::Models(format!(
                "artifact '{}' has no integrity digest",
                artifact.file_name
            ))
        })?;
        let actual = sha256_bytes(bytes);
        if actual != expected {
            return Err(EngineError::Models(format!(
                "SHA-256 mismatch for '{}': expected {expected}, got {actual}",
                artifact.file_name
            )));
        }

        let final_path = self.artifact_path(model, artifact);
        if final_path.exists() {
            if sha256_path(&final_path)? == expected {
                return Ok(final_path);
            }
            return Err(EngineError::Models(format!(
                "refusing to overwrite corrupt artifact at '{}'",
                final_path.display()
            )));
        }
        let parent = final_path
            .parent()
            .ok_or_else(|| EngineError::Models("artifact path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let temp_path = parent.join(format!(".{}.partial", artifact.file_name));
        if temp_path.exists() {
            return Err(EngineError::Models(format!(
                "partial install already exists at '{}'",
                temp_path.display()
            )));
        }
        let mut file = File::create(&temp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp_path, &final_path)?;
        write_receipt(parent, model, artifact, expected)?;
        Ok(final_path)
    }

    /// Install a small manifest resource after checking its exact bytes.
    pub fn install_resource_bytes(
        &self,
        resource: &ResourceDefinition,
        bytes: &[u8],
    ) -> Result<PathBuf> {
        let expected = resource.sha256.as_deref().ok_or_else(|| {
            EngineError::Models(format!(
                "resource '{}' has no integrity digest",
                resource.file_name
            ))
        })?;
        let actual = sha256_bytes(bytes);
        if actual != expected {
            return Err(EngineError::Models(format!(
                "SHA-256 mismatch for resource '{}': expected {expected}, got {actual}",
                resource.file_name
            )));
        }
        if let Some(expected_bytes) = resource.bytes
            && bytes.len() as u64 != expected_bytes
        {
            return Err(EngineError::Models(format!(
                "size mismatch for resource '{}': expected {expected_bytes}, got {}",
                resource.file_name,
                bytes.len()
            )));
        }

        let final_path = self.resource_path(resource);
        if final_path.exists() {
            if sha256_path(&final_path)? == expected {
                return Ok(final_path);
            }
            return Err(EngineError::Models(format!(
                "refusing to overwrite corrupt resource at '{}'",
                final_path.display()
            )));
        }
        let parent = final_path
            .parent()
            .ok_or_else(|| EngineError::Models("resource path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let temp_path = parent.join(format!(".{}.partial", resource.file_name));
        if temp_path.exists() {
            return Err(EngineError::Models(format!(
                "partial install already exists at '{}'",
                temp_path.display()
            )));
        }
        let mut file = File::create(&temp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp_path, &final_path)?;
        write_resource_receipt(parent, resource, expected)?;
        Ok(final_path)
    }

    /// Adopt a resource fetched to disk — never into memory, which is how the
    /// 35 MB Parakeet decode travels — after re-verifying its size and digest.
    /// The rename is the install, so the resource is whole or absent.
    pub fn install_staged_resource(
        &self,
        resource: &ResourceDefinition,
        staged: &Path,
    ) -> Result<PathBuf> {
        let expected = resource.sha256.as_deref().ok_or_else(|| {
            EngineError::Models(format!(
                "resource '{}' has no integrity digest",
                resource.file_name
            ))
        })?;
        let refused = || {
            EngineError::Models(format!(
                "staged resource '{}' failed integrity validation",
                resource.file_name
            ))
        };
        if !staged.is_file()
            || resource.bytes.is_some_and(|bytes| {
                fs::metadata(staged).map(|meta| meta.len()).ok() != Some(bytes)
            })
            || sha256_file(staged)? != expected
        {
            return Err(refused());
        }
        let final_path = self.resource_path(resource);
        if final_path.exists() {
            return Err(EngineError::Models(format!(
                "versioned resource already exists at '{}'",
                final_path.display()
            )));
        }
        let parent = final_path
            .parent()
            .ok_or_else(|| EngineError::Models("resource path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        fs::rename(staged, &final_path)?;
        write_resource_receipt(parent, resource, expected)?;
        Ok(final_path)
    }

    /// Adopt a fully staged artifact — an extracted directory bundle (Core ML)
    /// or a single staged file (raw ONNX) — on the same filesystem, after
    /// re-verifying its integrity digest. Updates use a new versioned path, so
    /// a known-good prior version is never removed.
    pub fn install_staged_directory(
        &self,
        model: &ModelDefinition,
        artifact: &ModelArtifact,
        staged: &Path,
    ) -> Result<PathBuf> {
        let expected = artifact.sha256.as_deref().ok_or_else(|| {
            EngineError::Models(format!(
                "artifact '{}' has no integrity digest",
                artifact.file_name
            ))
        })?;
        if !staged.exists() || sha256_path(staged)? != expected {
            return Err(EngineError::Models(format!(
                "staged artifact for '{}' failed integrity validation",
                artifact.file_name
            )));
        }
        let final_path = self.artifact_path(model, artifact);
        if final_path.exists() {
            return Err(EngineError::Models(format!(
                "versioned artifact already exists at '{}'",
                final_path.display()
            )));
        }
        let parent = final_path
            .parent()
            .ok_or_else(|| EngineError::Models("artifact path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        fs::rename(staged, &final_path)?;
        write_receipt(parent, model, artifact, expected)?;
        Ok(final_path)
    }
}

fn write_receipt(
    directory: &Path,
    model: &ModelDefinition,
    artifact: &ModelArtifact,
    sha256: &str,
) -> Result<()> {
    #[derive(Serialize)]
    struct Receipt<'a> {
        model: &'a str,
        version: &'a str,
        file_name: &'a str,
        sha256: &'a str,
    }
    let bytes = serde_json::to_vec_pretty(&Receipt {
        model: &model.id,
        version: &model.version,
        file_name: &artifact.file_name,
        sha256,
    })
    .map_err(|error| EngineError::Models(error.to_string()))?;
    let receipt = directory.join(format!("{}.receipt.json", artifact.file_name));
    let mut file = File::create(receipt)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_resource_receipt(
    directory: &Path,
    resource: &ResourceDefinition,
    sha256: &str,
) -> Result<()> {
    #[derive(Serialize)]
    struct Receipt<'a> {
        resource: &'a str,
        version: &'a str,
        file_name: &'a str,
        sha256: &'a str,
    }
    let bytes = serde_json::to_vec_pretty(&Receipt {
        resource: &resource.id,
        version: &resource.version,
        file_name: &resource.file_name,
        sha256,
    })
    .map_err(|error| EngineError::Models(error.to_string()))?;
    let receipt = directory.join(format!("{}.receipt.json", resource.file_name));
    let mut file = File::create(receipt)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn sha256_path(path: &Path) -> Result<String> {
    if path.is_file() {
        return sha256_file(path);
    }
    if !path.is_dir() {
        return Err(EngineError::Models(format!(
            "integrity path '{}' is neither file nor directory",
            path.display()
        )));
    }
    let mut files = Vec::new();
    collect_files(path, path, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, absolute) in files {
        hasher.update(relative.as_os_str().as_encoded_bytes());
        hasher.update([0]);
        let mut file = File::open(absolute)?;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Public because the manifest lane's first check is over bytes that never
/// reach the disk: the fetched manifest is proved against the compiled-in pin
/// before anything it names is requested.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn collect_files(root: &Path, path: &Path, output: &mut Vec<(PathBuf, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let absolute = entry.path();
        if absolute.is_dir() {
            collect_files(root, &absolute, output)?;
        } else if absolute.is_file() {
            let relative = absolute
                .strip_prefix(root)
                .map_err(|error| EngineError::Models(error.to_string()))?
                .to_path_buf();
            output.push((relative, absolute));
        }
    }
    Ok(())
}

fn all_model_ids() -> [ModelId; ModelId::ALL.len()] {
    ModelId::ALL
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) struct TempRoot(pub(super) PathBuf);

    impl TempRoot {
        pub(super) fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let suffix = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "ferment-models-{label}-{}-{suffix}",
                std::process::id()
            )))
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            if self.0.exists() {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
    }

    fn artifact(hash: Option<String>) -> ModelArtifact {
        ModelArtifact {
            backend: BackendKind::Ort,
            file_name: "model.onnx".into(),
            source: ArtifactSource::Bundled,
            sha256: hash,
            license: Some("proprietary-internal".into()),
            source_url: Some("internal:test".into()),
            placement: Some(Placement::Ort {
                providers: OrtPlacement::Cpu,
            }),
        }
    }

    #[test]
    fn install_rejects_bad_hash_before_writing() {
        let root = std::env::temp_dir().join(format!("ferment-model-test-{}", std::process::id()));
        let model = ModelDefinition {
            id: ModelId::DashengBase.name().into(),
            version: "1".into(),
            artifacts: vec![artifact(Some("0".repeat(64)))],
            warmup_priority: None,
        };
        let manifest = ModelManifest {
            version: "test".into(),
            models: vec![model.clone()],
            resources: Vec::new(),
        };
        let store = ModelStore::new(&root, manifest).unwrap();
        assert!(
            store
                .install_bytes(&model, &model.artifacts[0], b"bad")
                .is_err()
        );
        assert!(!store.artifact_path(&model, &model.artifacts[0]).exists());
    }

    #[test]
    fn unverified_artifact_is_explicit() {
        let model = ModelDefinition {
            id: ModelId::DashengBase.name().into(),
            version: "1".into(),
            artifacts: vec![artifact(None)],
            warmup_priority: None,
        };
        let manifest = ModelManifest {
            version: "test".into(),
            models: vec![model],
            resources: Vec::new(),
        };
        assert!(manifest.validate().is_ok());
    }

    fn manifest_prefix_artifact(sha256: Option<String>) -> ModelArtifact {
        ModelArtifact {
            backend: BackendKind::CoreMl,
            file_name: "toy.mlmodelc".into(),
            source: ArtifactSource::ManifestPrefix {
                prefix_url: "https://example.invalid/models/toy.mlmodelc".into(),
                manifest_sha256: "a".repeat(64),
            },
            sha256,
            license: Some("proprietary-internal".into()),
            source_url: Some("https://example.invalid/models/toy.mlmodelc.zip".into()),
            placement: Some(Placement::CoreMl {
                units: CoreMlComputeUnits::CpuOnly,
            }),
        }
    }

    fn manifest_of(artifacts: Vec<ModelArtifact>) -> ModelManifest {
        ModelManifest {
            version: "test".into(),
            models: vec![ModelDefinition {
                id: ModelId::Autoshot.name().into(),
                version: "1".into(),
                artifacts,
                warmup_priority: None,
            }],
            resources: Vec::new(),
        }
    }

    /// The manifest pin proves delivery; the tree hash proves the install. A
    /// source that carries only the first is a source that would adopt a
    /// verified set of files the fleet never qualified.
    #[test]
    fn manifest_prefix_requires_the_installed_tree_digest() {
        assert!(
            manifest_of(vec![manifest_prefix_artifact(Some("0".repeat(64)))])
                .validate()
                .is_ok()
        );
        let error = manifest_of(vec![manifest_prefix_artifact(None)])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("installed-tree digest"), "{error}");
    }

    #[test]
    fn manifest_prefix_rejects_bad_prefixes() {
        let mut trailing = manifest_prefix_artifact(Some("0".repeat(64)));
        trailing.source = ArtifactSource::ManifestPrefix {
            prefix_url: "https://example.invalid/models/toy.mlmodelc/".into(),
            manifest_sha256: "a".repeat(64),
        };
        let error = manifest_of(vec![trailing])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("must not end in '/'"), "{error}");

        let mut short_pin = manifest_prefix_artifact(Some("0".repeat(64)));
        short_pin.source = ArtifactSource::ManifestPrefix {
            prefix_url: "https://example.invalid/models/toy.mlmodelc".into(),
            manifest_sha256: "abc".into(),
        };
        assert!(manifest_of(vec![short_pin]).validate().is_err());
    }

    fn file_manifest(paths: &[(&str, u64)]) -> ModelFileManifest {
        ModelFileManifest {
            files: paths
                .iter()
                .map(|(path, bytes)| ModelFileEntry {
                    path: (*path).into(),
                    bytes: *bytes,
                    sha256: "b".repeat(64),
                })
                .collect(),
        }
    }

    /// A manifest is a network document that names files to write, so every
    /// refusal here is a write that never happens.
    #[test]
    fn file_manifest_refuses_what_it_cannot_safely_write() {
        assert!(
            file_manifest(&[("weights/weight.bin", 4), ("coremldata.bin", 2)])
                .validate("toy")
                .is_ok()
        );

        for path in [
            "../escape.bin",
            "/absolute.bin",
            "weights/../../escape.bin",
            "weights\\weight.bin",
            "",
            "weights//weight.bin",
        ] {
            assert!(
                file_manifest(&[(path, 1)]).validate("toy").is_err(),
                "'{path}' must be refused"
            );
        }

        assert!(
            ModelFileManifest { files: Vec::new() }
                .validate("toy")
                .is_err()
        );

        // Case-folded duplicates: two entries, one file on APFS/NTFS.
        let error = file_manifest(&[("Weight.bin", 1), ("weight.bin", 1)])
            .validate("toy")
            .unwrap_err()
            .to_string();
        assert!(error.contains("twice"), "{error}");

        // `a` cannot be both the file one entry names and the directory the
        // other one needs.
        let error = file_manifest(&[("weights", 1), ("weights/weight.bin", 1)])
            .validate("toy")
            .unwrap_err()
            .to_string();
        assert!(error.contains("both a file and a directory"), "{error}");

        let mut bad_digest = file_manifest(&[("weight.bin", 1)]);
        bad_digest.files[0].sha256 = "B".repeat(64);
        assert!(bad_digest.validate("toy").is_err(), "digests are lowercase");
    }

    #[test]
    fn file_manifest_parses_the_ios_byte_format() {
        // Byte-for-byte what `scripts/models-publish` writes and the CDN
        // serves — compact separators, one trailing newline.
        let body = b"{\"files\":[{\"path\":\"coremldata.bin\",\"bytes\":3,\"sha256\":\"\
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"},\
            {\"path\":\"weights/weight.bin\",\"bytes\":7,\"sha256\":\"\
            bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}]}\n";
        let manifest = ModelFileManifest::parse(body, "toy").unwrap();
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(manifest.total_bytes(), 10);
        assert_eq!(manifest.files[1].path, "weights/weight.bin");

        // A negative size is unrepresentable rather than rejected: iOS needs an
        // `invalidSize` refusal because Swift decodes it into an `Int64`.
        let negative = b"{\"files\":[{\"path\":\"a.bin\",\"bytes\":-1,\"sha256\":\"\
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]}\n";
        assert!(ModelFileManifest::parse(negative, "toy").is_err());
        assert!(ModelFileManifest::parse(b"not json", "toy").is_err());
    }

    #[test]
    fn resource_install_is_versioned_and_verified() {
        let root = TempRoot::new("resource");
        let bytes = b"bundled metadata";
        let resource = ResourceDefinition {
            id: "vocab".into(),
            version: "1".into(),
            file_name: "vocab.json".into(),
            source: ArtifactSource::Bundled,
            sha256: Some(sha256_bytes(bytes)),
            bytes: Some(bytes.len() as u64),
            license: Some("proprietary-internal".into()),
            source_url: Some("internal:test".into()),
        };
        let manifest = ModelManifest {
            version: "fleet-v1".into(),
            models: Vec::new(),
            resources: vec![resource.clone()],
        };
        let store = ModelStore::new(&root.0, manifest).unwrap();
        let path = store.install_resource_bytes(&resource, bytes).unwrap();

        assert!(path.ends_with("fleet-v1/resources/vocab/1/vocab.json"));
        assert_eq!(
            store.resource_availability(&resource).unwrap(),
            Availability::Ready
        );
        assert!(path.with_file_name("vocab.json.receipt.json").exists());
    }

    #[test]
    fn a_staged_resource_is_adopted_only_as_the_bytes_it_pins() {
        let root = TempRoot::new("staged-resource");
        let bytes = b"host decode weights";
        let resource = ResourceDefinition {
            id: "decode".into(),
            version: "1".into(),
            file_name: "decode.bin".into(),
            source: ArtifactSource::ManifestPrefix {
                prefix_url: "https://cdn.example/decode.bin".into(),
                manifest_sha256: "c".repeat(64),
            },
            sha256: Some(sha256_bytes(bytes)),
            bytes: Some(bytes.len() as u64),
            license: Some("CC-BY-4.0".into()),
            source_url: Some("https://example.test/upstream".into()),
        };
        let store = ModelStore::new(
            &root.0,
            ModelManifest {
                version: "fleet-v1".into(),
                models: Vec::new(),
                resources: vec![resource.clone()],
            },
        )
        .unwrap();
        let staging = root.0.join("staging");
        fs::create_dir_all(&staging).unwrap();

        let tampered = staging.join("tampered.bin");
        fs::write(&tampered, b"host decode WEIGHTS").unwrap();
        let error = store
            .install_staged_resource(&resource, &tampered)
            .unwrap_err()
            .to_string();
        assert!(error.contains("failed integrity validation"), "{error}");
        assert_eq!(
            store.resource_availability(&resource).unwrap(),
            Availability::Missing
        );

        let staged = staging.join("decode.bin");
        fs::write(&staged, bytes).unwrap();
        let path = store.install_staged_resource(&resource, &staged).unwrap();
        assert!(path.ends_with("fleet-v1/resources/decode/1/decode.bin"));
        assert!(!staged.exists(), "adoption is a rename");
        assert_eq!(
            store.resource_availability(&resource).unwrap(),
            Availability::Ready
        );
        assert!(path.with_file_name("decode.bin.receipt.json").exists());
    }

    #[test]
    fn validation_rejects_artifact_without_provenance() {
        // A shipped (non-Missing) artifact with no license must be rejected —
        // this is the manifest-level guard that keeps unprovenanced third-party
        // weights out of the fleet.
        let mut bare = artifact(Some("0".repeat(64)));
        bare.license = None;
        let manifest = ModelManifest {
            version: "test".into(),
            models: vec![ModelDefinition {
                id: ModelId::DashengBase.name().into(),
                version: "1".into(),
                artifacts: vec![bare],
                warmup_priority: None,
            }],
            resources: Vec::new(),
        };
        let error = manifest.validate().unwrap_err().to_string();
        assert!(error.contains("license"), "unexpected error: {error}");
    }

    /// `artifact_for` answers with the first artifact on a lane, so a second
    /// one would be shipped by nobody and pinned by nothing — the iOS
    /// Simulator's fp32 twin was the last such pair.
    #[test]
    fn validation_rejects_two_artifacts_on_one_lane() {
        let mut twin = manifest_prefix_artifact(Some("b".repeat(64)));
        twin.file_name = "toy_fp32.mlmodelc".into();
        let error = manifest_of(vec![manifest_prefix_artifact(Some("a".repeat(64))), twin])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("two CoreMl artifacts"), "{error}");
    }

    #[test]
    fn validation_rejects_a_placement_from_the_wrong_lane() {
        let mut artifact = manifest_prefix_artifact(Some("b".repeat(64)));
        artifact.placement = Some(Placement::Ort {
            providers: OrtPlacement::Cpu,
        });
        let error = manifest_of(vec![artifact])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("placement"), "unexpected error: {error}");
    }

    #[test]
    fn validation_requires_a_placement_on_every_shipped_artifact() {
        let mut artifact = manifest_prefix_artifact(Some("b".repeat(64)));
        artifact.placement = None;
        let error = manifest_of(vec![artifact])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("no placement"), "unexpected error: {error}");
    }

    #[test]
    fn validation_rejects_a_placement_on_a_declared_gap() {
        let artifact = ModelArtifact {
            backend: BackendKind::CoreMl,
            file_name: "toy.mlmodelc".into(),
            source: ArtifactSource::Missing {
                reason: "no audited export".into(),
            },
            sha256: None,
            license: None,
            source_url: None,
            placement: Some(Placement::CoreMl {
                units: CoreMlComputeUnits::CpuOnly,
            }),
        };
        let error = manifest_of(vec![artifact])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no graph to place"),
            "unexpected error: {error}"
        );
    }
}
