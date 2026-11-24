//! `EngineService` — the lifetime the Kotlin façade holds.
//!
//! On iOS `FermentEngine` is a namespace of statics: a logger, a models
//! directory, and per-processor `placements` tables that `CoreMLCache` reads.
//! Here the equivalent state is one ORT backend over one model catalog, and it
//! is an object rather than a static for the reason E0 wrote into FLEET_LAW: a
//! process-wide cache behind an FFI door makes every caller after the first one
//! a liar. Two services with different model directories are two backends.

use std::sync::Arc;

use ferment_engine_infer::{
    Backend, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_models::{
    ArtifactSource, BackendKind, ModelArtifact, ModelDefinition, ModelManifest,
    default_fleet_manifest,
};

use crate::EngineFfiError;
use crate::FfiResult;
use crate::capabilities::TranscriberKind;
use crate::progress::{Cancellation, call_progress};

/// One file the model delivery has to put in the models directory.
///
/// The shape iOS's `ModelDefinition` has (`name` + `ext`, the file's name),
/// carrying the object the fleet publishes it as and the pin of that object's
/// manifest, so the download is proved against the fleet. This is what makes
/// the Kotlin `ModelManifest.all` a derivative of the fleet rather than a
/// hand-mirrored second list.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FleetModelFile {
    /// The engine's model id (`ModelId::name`) — `beat_this_1`, `dfnet`, … —
    /// or, for a resource a processor downloads beside its graphs, the
    /// resource's id (`parakeet_decode`).
    pub model_id: String,
    /// The download name, iOS's `ModelDefinition.name`: the file's stem.
    pub name: String,
    /// The download extension, iOS's `ModelDefinition.ext`: `onnx` here where
    /// iOS says `mlmodelc`, because the file on this platform IS the ONNX
    /// master.
    pub ext: String,
    /// The object's prefix on the CDN: its manifest is
    /// `<prefix_url>.manifest.json`, its files `<prefix_url>/<path>`.
    pub prefix_url: String,
    /// Lowercase SHA-256 of the object's manifest — the pin a fetched manifest
    /// is proved against and an install is recorded under, so the model's
    /// version: a re-export is a new pin.
    pub manifest_sha256: String,
    /// SPDX identifier or `proprietary-internal`.
    pub license: Option<String>,
}

/// Every file the fleet publishes for this platform — each ONNX artifact and
/// each downloaded resource — with the object it is fetched from and its pin.
///
/// A free function so the Kotlin model catalog can be built before any engine
/// exists — `ModelManifest.all` on iOS is a `static let` read at launch, long
/// before a processor runs.
#[uniffi::export]
pub fn fleet_model_files() -> Vec<FleetModelFile> {
    let manifest = default_fleet_manifest();
    android_model_files(&manifest)
}

/// The fleet files ONE transcriber needs, from the processor's own
/// `MODEL_FILES` constant.
///
/// The Kotlin `placements` tables name model ids as strings, and nothing on
/// that side can tell a right id from a wrong-but-published one — a table that
/// asked for `beat_this_1` where it needs `parakeet_encoder` would still
/// resolve. This is the crate's own answer for the three processors that
/// publish one, so the transcriber tables the download router reads are graded
/// against the engine rather than against themselves.
#[uniffi::export]
pub fn transcriber_model_files(kind: TranscriberKind) -> Vec<String> {
    let (ids, resources): (&[ModelId], &[&str]) = match kind {
        TranscriberKind::Parakeet => (
            &ferment_engine_processors::Parakeet::MODEL_FILES,
            &ferment_engine_processors::Parakeet::RESOURCE_FILES,
        ),
        TranscriberKind::GigaAm => (&ferment_engine_processors::GigaAm::MODEL_FILES, &[]),
    };
    let manifest = default_fleet_manifest();
    let graphs = ids.iter().filter_map(|id| {
        let artifact = android_artifact(manifest.model(*id)?)?;
        Some(artifact.file_name.clone())
    });
    let sidecars = resources.iter().filter_map(|file_name| {
        manifest
            .resources
            .iter()
            .find(|resource| resource.file_name == *file_name)
            .map(|resource| resource.file_name.clone())
    });
    graphs.chain(sidecars).collect()
}

/// The one ONNX artifact this platform takes for a fleet model, if it has one.
///
/// The ORT lane, unless it is a declared gap — a `Missing` entry exists to be
/// visible to tooling, not to be fetched. Every reader of the fleet goes
/// through here, so the download table, the path table the engine looks
/// models up in, and `model_path` cannot answer differently from one another.
fn android_artifact(model: &ModelDefinition) -> Option<&ModelArtifact> {
    model
        .artifact_for(BackendKind::Ort)
        .filter(|artifact| !matches!(artifact.source, ArtifactSource::Missing { .. }))
}

/// Every file this platform downloads: each model's ONNX artifact, then each
/// resource the fleet publishes rather than bundles — a resource's id stands
/// where a model's does.
fn android_model_files(manifest: &ModelManifest) -> Vec<FleetModelFile> {
    let models = manifest.models.iter().filter_map(|model| {
        let artifact = android_artifact(model)?;
        file_row(
            &model.id,
            &artifact.file_name,
            &artifact.source,
            artifact.license.clone(),
        )
    });
    let resources = manifest.resources.iter().filter_map(|resource| {
        file_row(
            &resource.id,
            &resource.file_name,
            &resource.source,
            resource.license.clone(),
        )
    });
    models.chain(resources).collect()
}

/// The row of a file the fleet publishes as an object on the CDN; `None` for
/// one it does not — a bundled resource is compiled into the engine and asks
/// for nothing.
fn file_row(
    id: &str,
    file_name: &str,
    source: &ArtifactSource,
    license: Option<String>,
) -> Option<FleetModelFile> {
    let ArtifactSource::ManifestPrefix {
        prefix_url,
        manifest_sha256,
    } = source
    else {
        return None;
    };
    let (name, ext) = split_extension(file_name);
    Some(FleetModelFile {
        model_id: id.to_owned(),
        name,
        ext,
        prefix_url: prefix_url.clone(),
        manifest_sha256: manifest_sha256.clone(),
        license,
    })
}

/// `<stem>` and `<extension>` of a fleet file name.
///
/// `beat_this_1.onnx` splits at the LAST dot, so `gigaam_encoder.fp16.onnx`
/// is `("gigaam_encoder.fp16", "onnx")` — the fp16 marker belongs to the name
/// because it names a different file, not a different format, and iOS's
/// `ModelDefinition.localDirName` is exactly `"\(name).\(ext)"`.
fn split_extension(file_name: &str) -> (String, String) {
    match file_name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => (stem.to_owned(), extension.to_owned()),
        _ => (file_name.to_owned(), String::new()),
    }
}

/// Only cache-owned state. A caller can safely finish using an evicted
/// session, so residency here is not a measurement of total native memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct ModelCacheState {
    pub resident: u64,
    pub loading: u64,
    pub waiters: u64,
}

/// The engine's lifetime: one ORT backend over one models directory.
#[derive(uniffi::Object)]
pub struct EngineService {
    backend: Arc<OrtBackend>,
    models_directory: String,
}

#[uniffi::export]
impl EngineService {
    /// Open the engine over a flat directory of ONNX masters.
    ///
    /// `models_directory` holds the fleet's files under their own names — the
    /// layout `ModelManager` downloads into, and iOS's `modelsDirectory`
    /// exactly. A file that is not there yet is not an error here: the model
    /// is located when a capability first needs it, which is where iOS raises
    /// `modelNotFound` too.
    ///
    /// `engine_init` must already have named the ONNX Runtime — a session
    /// created before it makes `ort` load its platform default lazily, and
    /// `engine_init` then refuses (E0's F1). Constructing a service does not
    /// create a session, so the order Kotlin has to keep is only
    /// `engineInit` → first capability call.
    #[uniffi::constructor]
    pub fn new(models_directory: String) -> FfiResult<Self> {
        let manifest = default_fleet_manifest();
        let mut paths = ModelPaths::new();
        let root = std::path::Path::new(&models_directory);
        for model in &manifest.models {
            let (Some(id), Some(artifact)) = (model.model_id(), android_artifact(model)) else {
                continue;
            };
            paths.insert(id, ModelVersion::Latest, root.join(&artifact.file_name));
        }
        // Android normally attempts XNNPACK then CPU, including the verified
        // YuNet graph. U2NetP's pinned Resize nodes provide empty scales and
        // dynamic sizes. ORT 1.29's XNNPACK capability probe indexes those
        // scales without checking their length and terminates the process,
        // before provider fallback can run. Select CPU before that probe.
        let mut backend =
            OrtBackend::with_provider_chain(Arc::new(paths), OrtExecutionProvider::android_chain());
        // Parakeet, retrieval and MERT exports have a CPU baseline. Do not
        // silently promote them to an unqualified mobile execution provider.
        // The remaining Android families retain their existing measured chain.
        for id in [
            ModelId::U2Netp,
            ModelId::ParakeetEncoder,
            ModelId::GlapAudio,
            ModelId::GlapText,
            ModelId::Siglip2Image,
            ModelId::Siglip2Text,
            ModelId::Mert95M,
            ModelId::Mert330M,
        ] {
            backend = backend.with_model_provider_chain(id, [OrtExecutionProvider::Cpu]);
        }
        Ok(Self {
            backend: Arc::new(backend),
            models_directory,
        })
    }

    pub fn models_directory(&self) -> String {
        self.models_directory.clone()
    }

    /// The backend-wide chain, in attempt order — `["XNNPACK", "CPU"]`.
    pub fn provider_chain(&self) -> Vec<String> {
        self.backend
            .configured_chain()
            .iter()
            .map(|provider| provider.name().to_owned())
            .collect()
    }

    /// Which provider actually loaded `model_id`, once something has. `None`
    /// before the first session — the honest answer, and the one that makes a
    /// "we got XNNPACK" assertion able to fail.
    pub fn selected_provider(&self, model_id: String) -> Option<String> {
        let id = parse_model_id(&model_id)?;
        self.backend
            .selected_provider_for(id)
            .map(|provider| provider.name().to_owned())
    }

    /// Load and optimize one ONNX graph in this service's real session cache.
    /// Resources have no graph and are rejected. This does not perform dummy
    /// inference or claim a processor result; it makes the provider/session
    /// visible before the first user operation.
    pub fn warm_model(
        &self,
        model_id: String,
        cancellation: Option<Arc<Cancellation>>,
    ) -> FfiResult<()> {
        let progress = call_progress(None, cancellation);
        ferment_engine_core::WorkActivity::check_current()?;
        let id = parse_model_id(&model_id)
            .ok_or_else(|| EngineFfiError::new(format!("'{model_id}' is not a graph model")))?;
        futures_executor::block_on(progress.sink().until_cancelled(self.backend.warmup(&[id])))?;
        Ok(())
    }

    /// Release cache references on a memory warning. Pending loads and the
    /// explicitly excepted graph ids survive; accepted predictions retain
    /// their sessions until they have copied their outputs.
    pub fn evict_idle_models(&self, except: Vec<String>) -> FfiResult<Vec<String>> {
        let except: Vec<_> = except
            .iter()
            .map(|name| {
                parse_model_id(name)
                    .ok_or_else(|| EngineFfiError::new(format!("'{name}' is not a graph model")))
            })
            .collect::<FfiResult<_>>()?;
        Ok(self
            .backend
            .evict_idle_models(&except)
            .into_iter()
            .map(|id| id.name().to_owned())
            .collect())
    }

    /// Call after replacing one installed ONNX graph. This retires only its
    /// cached/pending versions; resources such as JSON sidecars are rejected.
    pub fn invalidate_model(&self, model_id: String) -> FfiResult<bool> {
        let id = parse_model_id(&model_id)
            .ok_or_else(|| EngineFfiError::new(format!("'{model_id}' is not a graph model")))?;
        Ok(self.backend.invalidate_model(id))
    }

    pub fn model_cache_state(&self) -> ModelCacheState {
        let snapshot = self.backend.cache_snapshot();
        ModelCacheState {
            resident: snapshot.resident as u64,
            loading: snapshot.loading as u64,
            waiters: snapshot.waiters as u64,
        }
    }

    /// The absolute path this service will look for `model_id` — a model's or
    /// a downloaded resource's — at.
    pub fn model_path(&self, model_id: String) -> FfiResult<String> {
        let file = android_model_files(&default_fleet_manifest())
            .into_iter()
            .find(|file| file.model_id == model_id)
            .ok_or_else(|| {
                EngineFfiError::new(format!("'{model_id}' is no file this platform downloads"))
            })?;
        Ok(std::path::Path::new(&self.models_directory)
            .join(format!("{}.{}", file.name, file.ext))
            .to_string_lossy()
            .into_owned())
    }
}

impl EngineService {
    pub(crate) fn backend(&self) -> Backend {
        Arc::clone(&self.backend) as Backend
    }
}

pub(crate) fn parse_model_id(name: &str) -> Option<ModelId> {
    default_fleet_manifest()
        .models
        .iter()
        .find(|model| model.id == name)
        .and_then(|model| model.model_id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn warmup_refuses_resources_and_unknown_ids_before_opening_runtime() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        for id in ["parakeet_decode", "unknown-graph"] {
            let failure = service
                .warm_model(id.to_owned(), None)
                .expect_err("not a graph");
            assert!(failure.to_string().contains(id));
            assert_eq!(service.selected_provider(id.to_owned()), None);
        }
    }

    /// The Android model table is a DERIVATIVE of the fleet manifest, never a
    /// second list — CLAUDE.md's "server models are the manifest" applied to
    /// the model fleet. Every ONNX artifact and every downloaded resource the
    /// fleet publishes appears once, and its name+ext rejoin into the file the
    /// CDN serves.
    ///
    /// Kill: split on the FIRST dot instead of the last — `gigaam_encoder.fp16`
    /// becomes `("gigaam_encoder", "fp16.onnx")` and the download asks for a
    /// file that does not exist.
    #[test]
    fn every_fleet_file_rejoins_into_the_name_the_cdn_serves() {
        let manifest = default_fleet_manifest();
        let files = android_model_files(&manifest);
        assert!(
            files.len() >= 30,
            "only {} files: {:?}",
            files.len(),
            files
                .iter()
                .map(|file| format!("{}.{}", file.name, file.ext))
                .collect::<Vec<_>>()
        );
        let mut resources = Vec::new();
        for file in &files {
            let published = match manifest
                .models
                .iter()
                .find(|model| model.id == file.model_id)
            {
                Some(model) => android_artifact(model)
                    .expect("the row exists because the artifact does")
                    .file_name
                    .clone(),
                None => {
                    let resource = manifest
                        .resource(&file.model_id)
                        .expect("every row names a fleet model or a fleet resource");
                    resources.push(resource.file_name.clone());
                    resource.file_name.clone()
                }
            };
            assert_eq!(format!("{}.{}", file.name, file.ext), published);
        }
        assert_eq!(
            resources,
            [
                "parakeet_decode.bin",
                "nemotron_vocab.json",
                "nemotron_metadata.json",
                "glap_sentencepiece.model",
                "siglip2_tokenizer.json"
            ]
        );
    }

    /// Every file the live transcriber reads is a file this platform
    /// downloads: its three graphs through the fleet's ONNX artifacts, and the
    /// vocabulary and metadata `NemotronModels::load` reads from the models
    /// directory through the fleet's resources.
    ///
    /// Kill: drop either sidecar from `fleet.rs` — the transcriber downloads
    /// its graphs and fails its first session on a file nobody fetched.
    #[test]
    fn every_file_the_live_transcriber_reads_is_downloaded() {
        use ferment_engine_processors::NemotronTranscriber;
        use ferment_engine_processors::nemotron::config::{METADATA_FILE, VOCAB_FILE};

        let manifest = default_fleet_manifest();
        let downloaded = android_model_files(&manifest)
            .into_iter()
            .map(|file| format!("{}.{}", file.name, file.ext))
            .collect::<Vec<_>>();
        let sidecars = [
            format!("{VOCAB_FILE}.json"),
            format!("{METADATA_FILE}.json"),
        ];
        assert_eq!(sidecars, ["nemotron_vocab.json", "nemotron_metadata.json"]);
        let graphs = NemotronTranscriber::MODEL_FILES.iter().map(|id| {
            android_artifact(manifest.model(*id).unwrap())
                .expect("a published graph")
                .file_name
                .clone()
        });
        for file in graphs.chain(sidecars) {
            assert!(
                downloaded.contains(&file),
                "the live transcriber reads {file}, which this platform does not download"
            );
        }
    }

    /// Every offered file names the object it is fetched from and the pin its
    /// manifest must hash to, exactly as its model or resource declares them:
    /// the app proves what it downloads against the fleet, never against the
    /// CDN's word about itself. `yunet_face`'s pin is the SHA-256 of the
    /// manifest the CDN serves for it.
    ///
    /// Kill: hand the file's own digest over as `manifest_sha256` — the app
    /// refuses every manifest it fetches.
    #[test]
    fn every_offered_file_names_its_object_and_its_pin() {
        let manifest = default_fleet_manifest();
        for file in android_model_files(&manifest) {
            let source = match manifest
                .models
                .iter()
                .find(|model| model.id == file.model_id)
            {
                Some(model) => &android_artifact(model).expect("offered").source,
                None => {
                    &manifest
                        .resource(&file.model_id)
                        .expect("a model or a resource")
                        .source
                }
            };
            let ArtifactSource::ManifestPrefix {
                prefix_url,
                manifest_sha256,
            } = source
            else {
                panic!("{} is offered but not published", file.model_id);
            };
            assert_eq!(&file.prefix_url, prefix_url, "{}", file.model_id);
            assert_eq!(&file.manifest_sha256, manifest_sha256, "{}", file.model_id);
        }
        let yunet = fleet_model_files()
            .into_iter()
            .find(|file| file.model_id == "yunet_face")
            .expect("yunet is offered");
        assert_eq!(
            yunet.prefix_url,
            "https://public.lockvoid.com/ferment/models/yunet_face.onnx"
        );
        assert_eq!(
            yunet.manifest_sha256,
            "ce6c5e14ba14d46d4eecd9120889a1faac7abecd73d84af7701c3b5a0b342927"
        );
    }

    /// A fleet artifact with a declared GAP is not offered for download.
    /// `beat_this_3` has `Missing` on both lanes, and a table that listed it
    /// would send the app after a file the CDN does not have.
    ///
    /// Kill: give `beat_this_3`'s ORT lane a `ManifestPrefix` in `fleet.rs`
    /// with nothing uploaded behind it — the app is sent after a manifest the
    /// CDN does not serve.
    #[test]
    fn a_declared_gap_is_not_a_downloadable_file() {
        let files = fleet_model_files();
        assert!(
            !files.iter().any(|file| file.model_id == "beat_this_3"),
            "the declared ORT gap was offered as a download"
        );
        assert!(files.iter().any(|file| file.model_id == "beat_this_1"));
    }

    /// U2NetP's empty-scales Resize nodes must not enter the XNNPACK
    /// capability probe: its bounds failure aborts before CPU fallback.
    #[test]
    fn u2netp_bypasses_the_unsafe_xnnpack_resize_probe() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        assert_eq!(
            service.backend.chain_for(ModelId::U2Netp),
            [OrtExecutionProvider::Cpu]
        );
        assert_eq!(service.selected_provider("u2netp".to_owned()), None);
    }

    /// Parakeet's exact published encoder stays on the provider its fleet row
    /// qualifies. Opening the service is lazy: policy selection must not warm
    /// a graph, replace its bytes, or create another capability-owned session.
    ///
    /// Kill: remove the Parakeet override in the real service constructor —
    /// the fleet row still says CPU, but the actual backend would try XNNPACK.
    #[test]
    fn parakeet_runs_its_declared_cpu_provider_without_opening_or_replacing_the_graph() {
        let manifest = default_fleet_manifest();
        let artifact = android_artifact(
            manifest
                .model(ModelId::ParakeetEncoder)
                .expect("declared Parakeet model"),
        )
        .expect("published Parakeet ORT graph");
        assert_eq!(
            artifact.placement,
            Some(ferment_engine_models::Placement::Ort {
                providers: ferment_engine_models::OrtPlacement::Cpu,
            })
        );
        let service = EngineService::new("/speech-models".to_owned()).expect("service");
        assert_eq!(
            service.backend.chain_for(ModelId::ParakeetEncoder),
            [OrtExecutionProvider::Cpu]
        );
        assert_eq!(
            Path::new(
                &service
                    .model_path("parakeet_encoder".to_owned())
                    .expect("graph path")
            ),
            Path::new("/speech-models").join(&artifact.file_name)
        );
        assert_eq!(
            service.selected_provider("parakeet_encoder".to_owned()),
            None
        );
        let cache = service.model_cache_state();
        assert_eq!([cache.resident, cache.loading, cache.waiters], [0, 0, 0]);
    }

    /// Kill: let desktop GPU qualification remove an Android CPU override,
    /// or offer only encoder graphs without their paired tokenizer.
    #[test]
    fn fp16_feature_models_keep_lazy_android_cpu_and_downloadable_tokenizers() {
        let manifest = default_fleet_manifest();
        let service = EngineService::new("/retrieval-models".to_owned()).unwrap();
        let files = fleet_model_files();
        for id in [
            ModelId::GlapAudio,
            ModelId::GlapText,
            ModelId::Siglip2Image,
            ModelId::Siglip2Text,
            ModelId::Mert95M,
            ModelId::Mert330M,
        ] {
            let artifact = android_artifact(manifest.model(id).unwrap()).unwrap();
            // The fleet records the measured Windows lane. Android retains
            // its explicit CPU override until a device qualifies another EP.
            assert_eq!(
                artifact.placement,
                Some(ferment_engine_models::Placement::Ort {
                    providers: ferment_engine_models::OrtPlacement::WindowsMl,
                })
            );
            assert_eq!(service.backend.chain_for(id), [OrtExecutionProvider::Cpu]);
            assert_eq!(service.selected_provider(id.name().to_owned()), None);
            assert!(files.iter().any(|file| file.model_id == id.name()));
        }
        for name in ["glap_sentencepiece.model", "siglip2_tokenizer.json"] {
            assert!(
                files
                    .iter()
                    .any(|file| format!("{}.{}", file.name, file.ext) == name)
            );
        }
        let cache = service.model_cache_state();
        assert_eq!([cache.resident, cache.loading, cache.waiters], [0, 0, 0]);
    }

    /// Other graphs keep the measured Android chain. The scoped CPU
    /// correction does not promote/reassign any other family's provider.
    /// YuNet in particular remains eligible for XNNPACK.
    #[test]
    fn other_fleet_models_keep_the_android_chain() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        assert_eq!(service.provider_chain(), ["XNNPACK", "CPU"]);
        assert_eq!(
            service.backend.chain_for(ModelId::YuNetFace),
            [OrtExecutionProvider::Xnnpack, OrtExecutionProvider::Cpu]
        );
        for file in fleet_model_files() {
            let Some(id) = parse_model_id(&file.model_id) else {
                assert!(
                    default_fleet_manifest().resource(&file.model_id).is_some(),
                    "{} names neither a model nor a resource",
                    file.model_id
                );
                continue;
            };
            if matches!(
                id,
                ModelId::U2Netp
                    | ModelId::ParakeetEncoder
                    | ModelId::GlapAudio
                    | ModelId::GlapText
                    | ModelId::Siglip2Image
                    | ModelId::Siglip2Text
                    | ModelId::Mert95M
                    | ModelId::Mert330M
            ) {
                continue;
            }
            assert_eq!(
                service.backend.chain_for(id),
                [OrtExecutionProvider::Xnnpack, OrtExecutionProvider::Cpu],
                "{} does not run the android chain",
                file.model_id
            );
        }
    }

    /// The service resolves a model to the flat file name the fleet publishes,
    /// under the caller's directory — not to `ModelId::name()`, which is the
    /// engine's id and still differs from the file wherever an artifact says
    /// something about itself the id does not (`gigaam_encoder` →
    /// `gigaam_encoder.fp16.onnx`, the rung whose weights are stored halved).
    ///
    /// The families `ferment-models-export` now owns publish under their id
    /// (`yunet_face.onnx`, 2026-09-22, where the object used to carry the
    /// opencv_zoo release's own name) — which is why the id must not be the
    /// rule: it is right for most and wrong for the ones that matter.
    ///
    /// Kill: build the path from `id.name()` + ".onnx" — every fp16 export
    /// points at nothing.
    #[test]
    fn a_model_resolves_to_the_fleets_file_name_not_its_id() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        assert_eq!(
            Path::new(&service.model_path("yunet_face".to_owned()).unwrap()),
            Path::new("/models/yunet_face.onnx")
        );
        assert_eq!(
            Path::new(&service.model_path("gigaam_encoder".to_owned()).unwrap()),
            Path::new("/models/gigaam_encoder.fp16.onnx")
        );
    }

    /// The download table and the table the engine looks models up in are ONE
    /// selection, not two that happen to agree. A row offered for download
    /// that the service will never look for is a file the app fetches and
    /// never opens; the reverse is a model the engine expects and the app was
    /// never told to fetch.
    ///
    /// Kill: change the artifact filter in one of the two readers.
    #[test]
    fn the_download_table_and_the_path_table_name_the_same_files() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        for file in fleet_model_files() {
            let path = service
                .model_path(file.model_id.clone())
                .unwrap_or_else(|error| panic!("{}: {error}", file.model_id));
            assert_eq!(
                Path::new(&path),
                Path::new("/models").join(format!("{}.{}", file.name, file.ext))
            );
        }
    }

    /// The backend-wide chain is E0's measured one, and it is a chain: XNNPACK
    /// claims Conv/Gemm/MatMul and CPU takes the rest.
    ///
    /// Kill: construct the backend with `with_execution_provider(Xnnpack)` —
    /// a runtime built without XNNPACK then hard-errors instead of running.
    #[test]
    fn the_service_runs_the_android_chain() {
        let service = EngineService::new("/models".to_owned()).expect("service");
        assert_eq!(service.provider_chain(), vec!["XNNPACK", "CPU"]);
        assert_eq!(service.selected_provider("yunet_face".to_owned()), None);
    }
}
