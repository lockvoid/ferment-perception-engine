use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::session_cache::{ModelCacheSnapshot, SessionCache};

use async_trait::async_trait;
use ferment_engine_core::{EngineError, Result, WorkActivity};
use ort::session::Session;
use ort::value::{DynTensor, Tensor as OrtTensor, TensorElementType, ValueType};

use crate::{
    DType, InferenceBackend, ModelId, ModelVersion, SharedModelLocator, Tensor, TensorMap,
};

type ModelKey = (ModelId, ModelVersion);

struct LoadedSession {
    /// `None` only inside `Drop`, which releases it in its provider's turn
    /// (`directml_turn`).
    session: Mutex<Option<Session>>,
    predictions: crate::prediction_permit::PredictionPermitQueue,
    provider: OrtExecutionProvider,
    #[cfg(windows)]
    graph_assignment: Option<Vec<crate::OrtAssignedSubgraph>>,
}

const LIVE: &str = "a loaded session is taken only when it is dropped";

impl Drop for LoadedSession {
    fn drop(&mut self) {
        let session = self
            .session
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        crate::directml_turn::in_turn(self.provider, move || drop(session));
    }
}

/// Metadata identity catches atomic replacement and same-size in-place
/// rewriting, without reading a multi-gigabyte graph on every prediction.
/// Downloaded artifact contents are separately checked against fleet pins.
fn file_identity(path: &std::path::Path) -> Result<(String, u64)> {
    let canonical = crate::model_io::at_path(path, path.canonicalize())?;
    let meta = crate::model_io::at_path(&canonical, canonical.metadata())?;
    if !meta.is_file() {
        return Err(EngineError::invalid_config(format!(
            "ORT graph is not a file: {}",
            canonical.display(),
        )));
    }
    #[cfg(unix)]
    let identity = {
        use std::os::unix::fs::MetadataExt;
        format!(
            "{canonical:?}:{}:{}:{}:{}:{}:{}:{}",
            meta.dev(),
            meta.ino(),
            meta.len(),
            meta.mtime(),
            meta.mtime_nsec(),
            meta.ctime(),
            meta.ctime_nsec()
        )
    };
    #[cfg(not(unix))]
    let identity = format!(
        "{canonical:?}:{}:{:?}:{:?}",
        meta.len(),
        crate::model_io::at_path(&canonical, meta.modified())?,
        meta.created().ok()
    );
    Ok((identity, meta.len()))
}

/// ONNX Runtime execution provider for newly-created sessions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OrtExecutionProvider {
    /// Portable CPU execution provider. Always available.
    #[default]
    Cpu,
    /// Windows ML catalog installs/registers compatible certified providers;
    /// the runtime's MaxPerformance device policy selects the EP and device.
    /// Inspect `graph_assignment_for` on Windows for actual node placement.
    WindowsMl,
    /// Explicit DirectML for diagnostics and previously qualified callers. Registered
    /// with `error_on_failure`, so requesting it on a runtime that lacks the
    /// provider fails loudly rather than silently falling back to CPU.
    DirectMl,
    /// Explicit CUDA for diagnostics with a separately provisioned runtime.
    /// Windows ML obtains its own NVIDIA provider through the catalog.
    /// Also registered with `error_on_failure` — same fail-loud discipline.
    Cuda,
    /// Core ML as an ONNX Runtime execution provider, on Apple only.
    ///
    /// Not the same thing as [`crate::CoreMlBackend`]: that loads a compiled
    /// `.mlmodelc` through Core ML directly, this hands an ONNX graph to Core
    /// ML through ORT. It exists for the families Studio has no `.mlmodelc`
    /// for — DINOv2 above all, which iOS runs on the ANE and Studio currently
    /// runs on the ORT CPU provider. Nothing in the shipping fleet selects it;
    /// it is reached by an explicit per-model chain, which is what keeps the
    /// experiment off by default.
    ///
    /// Registered as `MLProgram` (Core ML 5) rather than the EP's
    /// `NeuralNetwork` default: the older format lowers far fewer operators,
    /// so measuring against it would measure the format, not the hardware.
    /// Compute units are left at the EP's `ALL` — the planner's choice, which
    /// is the honest first question. Pinning units the way the `.mlmodelc`
    /// lane does is a follow-up the numbers have to earn.
    CoreMl,
    /// XNNPACK, the mobile floor. Registered by name through
    /// `SessionOptionsAppendExecutionProvider("XNNPACK")`, which `load-dynamic`
    /// already enables — no extra `ort` cargo feature is involved.
    ///
    /// It is a CPU provider with its own threadpool, not an accelerator: it
    /// claims `Conv`/`Gemm`/`MatMul` and leaves the rest to the CPU provider
    /// behind it, so a chain that ends in [`Self::Cpu`] is the shape that makes
    /// sense. `error_on_failure` like the others, so a runtime built without
    /// XNNPACK fails loudly instead of pretending.
    ///
    /// `supported_by_platform()` in `ort` is aarch64 / arm-linux / x86-64, so a
    /// green run on an arm64 emulator says nothing about a Snapdragon's QNN —
    /// that is a separate tier, measured separately.
    Xnnpack,
}

impl OrtExecutionProvider {
    /// Human-readable name used in fallback reporting and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::WindowsMl => "WindowsML",
            Self::DirectMl => "DirectML",
            Self::Cuda => "CUDA",
            Self::CoreMl => "CoreML",
            Self::Xnnpack => "XNNPACK",
        }
    }

    /// Windows ML automatic device policy, then explicit CPU if preparation
    /// or session creation fails. No vendor EP is hard-coded in this chain.
    pub fn windows_chain() -> [Self; 2] {
        [Self::WindowsMl, Self::Cpu]
    }

    /// The Android floor: XNNPACK for the operators it claims, CPU for the
    /// rest and for a runtime that was built without it. There is no
    /// accelerator above this tier yet — NNAPI is deprecated and QNN is
    /// vendor-specific, so both are measured before either enters a chain.
    pub fn android_chain() -> [Self; 2] {
        [Self::Xnnpack, Self::Cpu]
    }
}

/// ONNX Runtime backend for Windows, Linux, and native parity tests.
///
/// Sessions are created lazily and cached per `(model, version)`. Provider
/// selection is a *chain with fail-loud fallback*: the model's chain is
/// attempted in order, each GPU provider registered with `error_on_failure` so
/// an unavailable provider errors instead of silently downgrading. The provider
/// that loads a model is recorded by [`Self::selected_provider_for`]. Windows
/// ML selects devices automatically; `graph_assignment_for` reports the actual
/// per-node providers, including any CPU partitions. A single-element chain (from
/// [`Self::with_execution_provider`], or a placement that qualifies no
/// accelerator) therefore hard-errors when its one provider is missing: an
/// explicit request is never downgraded.
///
/// Which chain a model gets is the placement axis: [`Self::with_placements`]
/// installs the fleet manifest's ORT column, and anything it does not name
/// falls back to the backend-wide chain.
pub struct OrtBackend {
    locator: SharedModelLocator,
    providers: Vec<OrtExecutionProvider>,
    /// The placement axis on this lane: one provider chain per model, read
    /// from the fleet manifest by whoever constructs the backend. A model
    /// without a chain of its own uses `providers`.
    model_providers: HashMap<ModelId, Vec<OrtExecutionProvider>>,
    sessions: SessionCache<ModelKey, LoadedSession>,
    /// The graphics card the host chose for models, by its PCI device ID:
    /// Windows ML runs on that card (and the CPU) instead of choosing one.
    /// `None` keeps Windows ML's automatic choice. Shared with session loads,
    /// which read it when they build, not when they are asked for.
    gpu: Arc<std::sync::RwLock<Option<u32>>>,
}

impl std::fmt::Debug for OrtBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OrtBackend")
            .field("locator", &self.locator)
            .field("providers", &self.providers)
            .field("model_providers", &self.model_providers)
            .field("cache", &self.sessions.snapshot())
            .finish_non_exhaustive()
    }
}

/// One entry of ONNX Runtime's device list, as far as choosing a card reads it.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug)]
struct DeviceEntry {
    gpu: bool,
    /// The hardware's PCI device ID.
    id: u32,
    cpu_provider: bool,
}

/// The devices a model runs on when the host chose a graphics card: every
/// provider registered for that card, then the CPU for what they cannot run.
/// With no provider for that card, the CPU alone: never Windows ML's own
/// choice, which put the models on a card nobody chose. The flag tells
/// whether the card is among them.
#[cfg(any(windows, test))]
fn card_device_indices(devices: &[DeviceEntry], card: u32) -> (Vec<usize>, bool) {
    let on_card: Vec<usize> = devices
        .iter()
        .enumerate()
        .filter(|(_, device)| device.gpu && device.id == card)
        .map(|(index, _)| index)
        .collect();
    let found = !on_card.is_empty();
    let cpu = devices.iter().position(|device| device.cpu_provider);
    (on_card.into_iter().chain(cpu).collect(), found)
}

impl OrtBackend {
    /// CPU-only backend (parity tests, platforms without a GPU provider).
    pub fn new(locator: SharedModelLocator) -> Self {
        Self::with_provider_chain(locator, [OrtExecutionProvider::Cpu])
    }

    /// Force a single execution provider with no fallback. The requested
    /// provider must load or session creation fails loudly — this is how a
    /// caller pins DirectML/CUDA and refuses a silent CPU downgrade.
    pub fn with_execution_provider(
        locator: SharedModelLocator,
        execution_provider: OrtExecutionProvider,
    ) -> Self {
        Self::with_provider_chain(locator, [execution_provider])
    }

    /// The default Windows backend: attempt Windows ML → CPU in order,
    /// reporting a skipped provider and recording the one that wins.
    pub fn windows_default(locator: SharedModelLocator) -> Self {
        Self::with_provider_chain(locator, OrtExecutionProvider::windows_chain())
    }

    /// Backend with an explicit ordered provider chain. The chain is attempted
    /// front-to-back; the first provider that loads a model is selected.
    ///
    /// # Panics
    /// Panics if `providers` is empty — a backend with no provider can never
    /// create a session.
    pub fn with_provider_chain(
        locator: SharedModelLocator,
        providers: impl IntoIterator<Item = OrtExecutionProvider>,
    ) -> Self {
        let providers: Vec<_> = providers.into_iter().collect();
        assert!(
            !providers.is_empty(),
            "OrtBackend requires at least one execution provider"
        );
        Self {
            locator,
            providers,
            model_providers: HashMap::new(),
            sessions: SessionCache::new(),
            gpu: Arc::default(),
        }
    }

    /// Run models on this graphics card, by its PCI device ID, or let Windows
    /// ML choose (`None`). Sessions already running keep their card; the next
    /// session of every model is created on the new one.
    pub fn set_gpu(&self, device: Option<u32>) {
        let mut gpu = self.gpu.write().unwrap_or_else(PoisonError::into_inner);
        if *gpu != device {
            *gpu = device;
            drop(gpu);
            self.sessions.invalidate(|_| true);
        }
    }

    /// The graphics card models run on, if the host chose one.
    pub fn gpu(&self) -> Option<u32> {
        *self.gpu.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Pin one model's provider chain, overriding the backend-wide one.
    ///
    /// # Panics
    /// Panics if `chain` is empty, for the same reason the constructor does.
    #[must_use]
    pub fn with_model_provider_chain(
        mut self,
        model: ModelId,
        chain: impl IntoIterator<Item = OrtExecutionProvider>,
    ) -> Self {
        let chain: Vec<_> = chain.into_iter().collect();
        assert!(
            !chain.is_empty(),
            "a pinned chain for {model} needs at least one execution provider"
        );
        if self.chain_for(model) != chain.as_slice() {
            self.invalidate_model(model);
        }
        self.model_providers.insert(model, chain);
        self
    }

    /// Pin a whole placement table at once — the fleet manifest's ORT column,
    /// as `engine-models` hands it over.
    ///
    /// # Panics
    /// Panics if any chain is empty.
    #[must_use]
    pub fn with_placements(
        mut self,
        placements: impl IntoIterator<Item = (ModelId, Vec<OrtExecutionProvider>)>,
    ) -> Self {
        for (model, chain) in placements {
            self = self.with_model_provider_chain(model, chain);
        }
        self
    }

    /// The backend-wide provider chain, in attempt order. The floor for any
    /// model the placement table does not name.
    pub fn configured_chain(&self) -> &[OrtExecutionProvider] {
        &self.providers
    }

    /// The chain this model will actually be attempted on: its pin, or the
    /// backend-wide chain.
    pub fn chain_for(&self, model: ModelId) -> &[OrtExecutionProvider] {
        self.model_providers
            .get(&model)
            .map_or(&self.providers, Vec::as_slice)
    }

    /// The policy/provider that loaded this model, or `None` before its session
    /// exists. `WindowsMl` means automatic selection, not a claim of GPU work;
    /// `graph_assignment_for` reports the actual execution providers on Windows.
    pub fn selected_provider_for(&self, model: ModelId) -> Option<OrtExecutionProvider> {
        self.sessions
            .peek(&(model, ModelVersion::Latest))
            .map(|loaded| loaded.provider)
    }

    /// Actual per-node EP assignment from the cached Windows session. This
    /// includes nodes inside compiled EP subgraphs, not just available devices.
    #[cfg(windows)]
    pub fn graph_assignment_for(&self, model: ModelId) -> Option<Vec<crate::OrtAssignedSubgraph>> {
        self.sessions
            .peek(&(model, ModelVersion::Latest))
            .and_then(|loaded| loaded.graph_assignment.clone())
    }

    fn prepare_provider(provider: OrtExecutionProvider) -> Result<()> {
        if provider != OrtExecutionProvider::WindowsMl {
            return Ok(());
        }
        #[cfg(windows)]
        {
            let preparation = winml_native::prepare().map_err(EngineError::inference)?;
            for warning in &preparation.warnings {
                tracing::warn!(%warning, "Windows ML EP preparation failed; automatic selection uses available EPs");
            }
            tracing::info!(providers = ?preparation.providers, devices = ?preparation.devices,
                "Windows ML catalog prepared");
            Ok(())
        }
        #[cfg(not(windows))]
        Err(EngineError::inference("Windows ML requires Windows"))
    }

    /// Configure an explicit provider or Windows ML's automatic device policy.
    /// Explicit providers use `error_on_failure`; actual graph placement is
    /// recorded separately because automatic policies can partition onto CPU.
    fn session_builder(
        model: ModelId,
        provider: OrtExecutionProvider,
        #[cfg_attr(not(windows), allow(unused_variables))] gpu: Option<u32>,
    ) -> Result<ort::session::builder::SessionBuilder> {
        let mut builder =
            Session::builder().map_err(|error| EngineError::inference(error.to_string()))?;
        #[cfg(windows)]
        {
            builder = builder
                .with_config_entry("session.record_ep_graph_assignment_info", "1")
                .map_err(|error| EngineError::inference(error.to_string()))?;
        }
        // Parakeet stores 608 weights as FP16 followed by Cast(to=FLOAT).
        // ConstantFolding and the separate Level4 initializer-fusion pass
        // eagerly retain 2.44 GB of widened weights alongside the embedded
        // halves during session initialization. Exclude both by their ORT
        // transformer names so these exact casts run in the CPU graph and
        // sequential execution can release each temporary after its consumers.
        // ORT's config parser splits this list on semicolons. Graph bytes,
        // arithmetic, other optimizers and every other model/provider stay
        // unchanged; real token, memory and elapsed-time gates grade this lane.
        if model == ModelId::ParakeetEncoder && provider == OrtExecutionProvider::Cpu {
            builder = builder
                .with_config_entry(
                    "optimization.disable_specified_optimizers",
                    "ConstantFolding;FuseFp16InitializerToFp32NodeTransformer",
                )
                .map_err(|error| EngineError::inference(error.to_string()))?;
        }
        // BeatThis reuses a fixed 1500-frame graph many times. ORT's memory
        // pattern begins allocating its combined activation block on the
        // second run, after the first run has already grown the CPU arena.
        // Retaining both layouts exceeded the original warmup+8 <32 MiB
        // growth gate. Reuse the ordinary arena layout on the mobile CPU
        // routes instead; graph weights, math and session caching are unchanged.
        if matches!(
            model,
            ModelId::BeatThis1 | ModelId::BeatThis2 | ModelId::BeatThis3
        ) && matches!(
            provider,
            OrtExecutionProvider::Cpu | OrtExecutionProvider::Xnnpack
        ) {
            builder = builder
                .with_memory_pattern(false)
                .map_err(|error| EngineError::inference(error.to_string()))?;
        }
        match provider {
            OrtExecutionProvider::Cpu => {
                #[cfg(windows)]
                {
                    // SessionBuilder otherwise defaults to MaxEfficiency and
                    // can pick a catalog EP after another model registers it.
                    // Explicit CPU placements must remain CPU-only.
                    let environment = ort::environment::Environment::current()
                        .map_err(|error| EngineError::inference(error.to_string()))?;
                    let devices: Vec<_> = environment
                        .devices()
                        .filter(|device| device.ep().ok() == Some("CPUExecutionProvider"))
                        .collect();
                    if devices.is_empty() {
                        return Err(EngineError::inference("ORT exposes no CPU device"));
                    }
                    builder = builder
                        .with_devices(devices, None)
                        .map_err(|error| EngineError::inference(error.to_string()))?;
                }
            }
            OrtExecutionProvider::WindowsMl => {
                // The catalog was prepared outside the GPU turn. On the card
                // the host chose, Windows ML runs every provider registered
                // for it and the CPU. Without a choice it picks among its
                // devices itself: on a laptop with an integrated and a
                // discrete card its MaxPerformance policy picked the
                // integrated one, whose 2020 Intel D3D12 driver then crashed
                // the process while the library was being indexed.
                builder = builder
                    .with_memory_pattern(false)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                #[cfg(windows)]
                let environment = ort::environment::Environment::current()
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                #[cfg(windows)]
                let chosen = gpu.map(|card| {
                    let devices: Vec<_> = environment.devices().collect();
                    let entries: Vec<_> = devices
                        .iter()
                        .map(|device| DeviceEntry {
                            gpu: device.ty() == ort::memory::DeviceType::GPU,
                            id: device.id(),
                            cpu_provider: device.ep().ok() == Some("CPUExecutionProvider"),
                        })
                        .collect();
                    let (indices, found) = card_device_indices(&entries, card);
                    if found {
                        tracing::info!(model = model.name(), card, devices = indices.len(),
                            "Windows ML runs on the chosen graphics card");
                    } else {
                        tracing::warn!(model = model.name(), card,
                            "No Windows ML provider serves the chosen graphics card; the model runs on the CPU");
                    }
                    devices
                        .into_iter()
                        .enumerate()
                        .filter(|(index, _)| indices.contains(index))
                        .map(|(_, device)| device)
                        .collect::<Vec<_>>()
                });
                #[cfg(not(windows))]
                let chosen: Option<Vec<ort::device::Device<'_>>> = None;
                builder = match chosen {
                    Some(devices) => builder.with_devices(devices, None),
                    None => builder
                        .with_auto_device(ort::session::builder::AutoDevicePolicy::MaxPerformance),
                }
                .map_err(|error| EngineError::inference(error.to_string()))?;
                // TensorRT RTX's implicit profile uses opt=1 for symbolic
                // dimensions. These graphs cannot compile at that shape.
                // Supply their valid ranges without pinning the selected EP.
                let profile = match model {
                    ModelId::GlapAudio => Some(["mel:1x64x4", "mel:1x64x1001", "mel:1x64x1001"]),
                    ModelId::Fastdvdnet => Some([
                        "frames:1x15x4x4,noise_map:1x1x4x4",
                        "frames:1x15x256x256,noise_map:1x1x256x256",
                        // Preserve the EP's implicit i16::MAX ceiling, rounded
                        // down to this model's required multiple of four.
                        // The EP allocates activation memory for actual shapes.
                        "frames:1x15x32764x32764,noise_map:1x1x32764x32764",
                    ]),
                    _ => None,
                };
                if let Some(profile) = profile {
                    for (key, shape) in [
                        "nv_profile_min_shapes",
                        "nv_profile_opt_shapes",
                        "nv_profile_max_shapes",
                    ]
                    .into_iter()
                    .zip(profile)
                    {
                        builder = builder
                            .with_config_entry(
                                format!("ep.nvtensorrtrtxexecutionprovider.{key}"),
                                shape,
                            )
                            .map_err(|error| EngineError::inference(error.to_string()))?;
                    }
                }
            }
            OrtExecutionProvider::DirectMl => {
                builder = builder
                    .with_memory_pattern(false)
                    .map_err(|error| EngineError::inference(error.to_string()))?
                    .with_execution_providers([ort::ep::DirectML::default()
                        .build()
                        .error_on_failure()])
                    .map_err(|error| EngineError::inference(error.to_string()))?;
            }
            OrtExecutionProvider::Cuda => {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(|error| EngineError::inference(error.to_string()))?;
            }
            OrtExecutionProvider::CoreMl => {
                builder = builder
                    .with_execution_providers([ort::ep::CoreML::default()
                        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                        .build()
                        .error_on_failure()])
                    .map_err(|error| EngineError::inference(error.to_string()))?;
            }
            OrtExecutionProvider::Xnnpack => {
                builder = builder
                    .with_execution_providers([ort::ep::XNNPACK::default()
                        .build()
                        .error_on_failure()])
                    .map_err(|error| EngineError::inference(error.to_string()))?;
            }
        }
        Ok(builder)
    }

    fn build_session(
        model: ModelId,
        provider: OrtExecutionProvider,
        path: &std::path::Path,
        gpu: Option<u32>,
    ) -> Result<Session> {
        Self::session_builder(model, provider, gpu)?
            .commit_from_file(path)
            .map_err(|error| EngineError::inference(error.to_string()))
    }

    async fn ensure_session(
        &self,
        model: ModelId,
        version: &ModelVersion,
    ) -> Result<Arc<LoadedSession>> {
        WorkActivity::check_current()?;
        let key = (model, version.clone());
        let resolve = || {
            let path = self.locator.locate(model, version)?;
            let (identity, weight) = file_identity(&path)?;
            Ok::<_, EngineError>((path, identity, weight))
        };
        let (path, identity, weight) = match resolve() {
            Ok(value) => value,
            Err(error) => {
                self.sessions.invalidate(|candidate| candidate == &key);
                return Err(error);
            }
        };
        let expected_identity = identity.clone();
        let chain = self.chain_for(model).to_vec();
        // Read when the session is built: a change before then is not lost to
        // a load that was asked for earlier, and one after it retires the load.
        let gpu = Arc::clone(&self.gpu);
        self.sessions
            .load(key, identity, weight, move || {
                let gpu = *gpu.read().unwrap_or_else(PoisonError::into_inner);
                let (session, provider) = Self::resolve_session(model, &path, &chain, gpu)?;
                // Owned at once, so a refused session is released in its turn.
                #[allow(unused_mut)]
                let mut loaded = LoadedSession {
                    session: Mutex::new(Some(session)),
                    predictions: crate::prediction_permit::PredictionPermitQueue::default(),
                    provider,
                    #[cfg(windows)]
                    graph_assignment: None,
                };
                #[cfg(windows)]
                {
                    let assignment = {
                        let session = loaded
                            .session
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner);
                        winml_native::graph_assignment(session.as_ref().expect(LIVE))
                    };
                    loaded.graph_assignment = match assignment {
                        Ok(graphs) => Some(graphs),
                        Err(error) if provider == OrtExecutionProvider::WindowsMl => {
                            return Err(EngineError::inference(format!(
                                "Windows ML graph assignment: {error}"
                            )));
                        }
                        Err(error) => {
                            // Explicit legacy ORT providers remain usable with
                            // older runtimes; no diagnostic claim is fabricated.
                            tracing::warn!(%error, "ORT graph assignment unavailable");
                            None
                        }
                    };
                    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
                    for graph in loaded.graph_assignment.iter().flatten() {
                        *counts.entry(&graph.provider).or_default() += graph.nodes.len();
                    }
                    tracing::info!(model = model.name(), assignment = ?counts,
                        "ORT actual graph execution providers");
                }
                if file_identity(&path)?.0 != expected_identity {
                    return Err(EngineError::invalid_config(format!(
                        "model {} changed while its native session was loading",
                        model.name(),
                    )));
                }
                Self::record_selection(model, provider, &chain);
                Ok(loaded)
            })
            .await
    }

    /// Drop cache ownership heaviest-first; in-flight loads and named model
    /// versions survive. Active inference holds its own session reference.
    pub fn evict_idle_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.sessions
            .evict_idle(|(model, _)| except.contains(model))
            .into_iter()
            .map(|(model, _)| model)
            .collect()
    }

    /// Drop only unleased models, heaviest-first, regardless of memory pressure.
    /// Active/queued predictions stay registered and joinable. Pending loads
    /// and `except` IDs also stay. Prefer this to `evict_idle_models` when
    /// moving between analysis stages without deliberately detaching live calls.
    pub fn evict_inactive_models(&self, except: &[ModelId]) -> Vec<ModelId> {
        self.sessions
            .evict_inactive(|(model, _)| except.contains(model))
            .into_iter()
            .map(|(model, _)| model)
            .collect()
    }

    /// Retire only this resource, including pending versions. An older load
    /// cannot overwrite a newer graph's cache entry when it finally finishes.
    pub fn invalidate_model(&self, model: ModelId) -> bool {
        self.sessions
            .invalidate(|(candidate, _)| *candidate == model)
            != 0
    }

    pub fn cache_snapshot(&self) -> ModelCacheSnapshot {
        self.sessions.snapshot()
    }

    /// Snapshot the actual cached native model lane; separate model instances
    /// keep separate permits and are not serialized against one another.
    pub async fn prediction_snapshot(
        &self,
        model: ModelId,
        version: &ModelVersion,
    ) -> Result<crate::prediction_permit::PredictionSnapshot> {
        Ok(self
            .ensure_session(model, version)
            .await?
            .predictions
            .snapshot())
    }

    /// Walk the provider chain, returning the first session that loads together
    /// with the provider that produced it. Every skipped provider is reported
    /// via `tracing::warn`; if the whole chain fails the error lists each
    /// provider's failure so nothing is swallowed silently.
    fn resolve_session(
        model: ModelId,
        path: &std::path::Path,
        chain: &[OrtExecutionProvider],
        gpu: Option<u32>,
    ) -> Result<(Session, OrtExecutionProvider)> {
        let mut failures = Vec::new();
        for &provider in chain {
            match Self::prepare_provider(provider).and_then(|()| {
                crate::directml_turn::in_turn(provider, || {
                    Self::build_session(model, provider, path, gpu)
                })
            }) {
                Ok(session) => return Ok((session, provider)),
                Err(error) => {
                    tracing::warn!(
                        model = model.name(),
                        provider = provider.name(),
                        %error,
                        "ORT execution provider unavailable; falling through to next in chain",
                    );
                    failures.push(format!("{}: {error}", provider.name()));
                }
            }
        }
        let attempted = chain
            .iter()
            .map(|provider| provider.name())
            .collect::<Vec<_>>()
            .join(" -> ");
        Err(EngineError::inference(format!(
            "no ORT execution provider in chain [{attempted}] could load '{}': {}",
            model.name(),
            failures.join("; ")
        )))
    }

    /// Record the winning provider and, when it is not the chain's first
    /// choice, emit a loud downgrade warning so a silent fallback to a slower
    /// provider is never invisible.
    fn record_selection(
        model: ModelId,
        provider: OrtExecutionProvider,
        chain: &[OrtExecutionProvider],
    ) {
        if chain.first() == Some(&provider) {
            tracing::info!(
                model = model.name(),
                provider = provider.name(),
                "ORT execution provider selected"
            );
        } else {
            tracing::warn!(
                model = model.name(),
                preferred = chain[0].name(),
                selected = provider.name(),
                "ORT downgraded from the preferred execution provider",
            );
        }
    }
}

#[async_trait]
impl InferenceBackend for OrtBackend {
    async fn infer(
        &self,
        model: ModelId,
        version: &ModelVersion,
        inputs: TensorMap,
    ) -> Result<TensorMap> {
        let work = WorkActivity::current();
        let check_cancellation = || {
            work.as_ref()
                .map_or(Ok(()), WorkActivity::check_cancellation)
        };
        let loaded = self.ensure_session(model, version).await?;
        check_cancellation()?;
        let permit = loaded.predictions.acquire().await;
        check_cancellation()?;
        // Sample after admission, including a queued cache hit. Keep this
        // graph hot and preserve other admitted/queued callers and loads.
        self.sessions
            .evict_inactive_if_pressure(|(candidate, candidate_version)| {
                *candidate == model && candidate_version == version
            });
        let result: Result<TensorMap> = (|| {
            let inputs = inputs
                .into_iter()
                .map(|(name, tensor)| Ok((name, to_ort_tensor(tensor)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            let mut session = loaded.session.lock().unwrap();
            let session = session.as_mut().expect(LIVE);
            crate::directml_turn::in_turn(loaded.provider, || {
                let outputs = session
                    .run(inputs)
                    .map_err(|error| EngineError::inference(error.to_string()))?;
                outputs
                    .iter()
                    .map(|(name, value)| Ok((name.to_owned(), from_ort_tensor(&value)?)))
                    .collect()
            })
        })();
        permit.finish(result.is_ok());
        check_cancellation()?;
        if result.is_ok()
            && let Some(work) = &work
        {
            work.checkpoint();
        }
        result
    }

    async fn warmup(&self, models: &[ModelId]) -> Result<()> {
        for &model in models {
            self.ensure_session(model, &ModelVersion::Latest).await?;
        }
        Ok(())
    }

    async fn metadata(
        &self,
        model: ModelId,
        version: &ModelVersion,
        key: &str,
    ) -> Result<Option<String>> {
        let loaded = self.ensure_session(model, version).await?;
        let session = loaded.session.lock().unwrap();
        let metadata = session
            .as_ref()
            .expect(LIVE)
            .metadata()
            .map_err(|error| EngineError::inference(error.to_string()))?;
        // ort 2.0.0-rc.12 turns ORT's null missing-value pointer into Some("").
        // Check the key table so a nameless encoder cannot masquerade as one
        // with a declared (empty) checkpoint identity.
        if !metadata
            .custom_keys()
            .map_err(|error| EngineError::inference(error.to_string()))?
            .iter()
            .any(|candidate| candidate == key)
        {
            return Ok(None);
        }
        Ok(metadata.custom(key))
    }
}

fn to_ort_tensor(tensor: Tensor) -> Result<DynTensor> {
    let shape = tensor.shape.clone();
    let result = match tensor.dtype {
        DType::Fp32 => OrtTensor::from_array((shape, tensor.to_f32()?)).map(OrtTensor::upcast),
        // EdgeTAM's every boundary is FP16 and its fixtures pin BIT PATTERNS,
        // so the halves cross as halves rather than being promoted here and
        // rounded again inside the graph.
        DType::Fp16 => OrtTensor::from_array((shape, tensor.to_f16()?)).map(OrtTensor::upcast),
        DType::Int32 => OrtTensor::from_array((shape, tensor.to_i32()?)).map(OrtTensor::upcast),
        DType::Int64 => OrtTensor::from_array((shape, tensor.to_i64()?)).map(OrtTensor::upcast),
        dtype => {
            return Err(EngineError::Unsupported(format!(
                "ORT input dtype {} is not implemented",
                dtype.as_str()
            )));
        }
    };
    result.map_err(|error| EngineError::inference(error.to_string()))
}

fn from_ort_tensor(value: &ort::value::DynValue) -> Result<Tensor> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        return Err(EngineError::Unsupported(format!(
            "ORT returned non-tensor value {}",
            value.dtype()
        )));
    };
    match ty {
        TensorElementType::Float32 => {
            let (shape, values) = value
                .try_extract_tensor::<f32>()
                .map_err(|error| EngineError::inference(error.to_string()))?;
            Tensor::from_f32(shape.to_vec(), values.to_vec())
        }
        TensorElementType::Float16 => {
            let (shape, values) = value
                .try_extract_tensor::<half::f16>()
                .map_err(|error| EngineError::inference(error.to_string()))?;
            Tensor::from_f16(shape.to_vec(), values.to_vec())
        }
        TensorElementType::Int32 => {
            let (shape, values) = value
                .try_extract_tensor::<i32>()
                .map_err(|error| EngineError::inference(error.to_string()))?;
            Tensor::from_i32(shape.to_vec(), values.to_vec())
        }
        TensorElementType::Int64 => {
            let (shape, values) = value
                .try_extract_tensor::<i64>()
                .map_err(|error| EngineError::inference(error.to_string()))?;
            Tensor::from_i64(shape.to_vec(), values.to_vec())
        }
        other => Err(EngineError::Unsupported(format!(
            "ORT output dtype {other} is not implemented"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[test]
    fn physical_identity_records_actual_file_weight_and_atomic_replacement() {
        let root = std::env::temp_dir().join(format!(
            "ort-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("model.onnx");
        std::fs::write(&path, b"first").unwrap();
        let first = file_identity(&path).unwrap();
        assert_eq!(first.1, 5);
        assert_eq!(file_identity(&path).unwrap(), first);
        let replacement = root.join("replacement.onnx");
        std::fs::write(&replacement, b"other").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let second = file_identity(&path).unwrap();
        assert_eq!(second.1, first.1);
        assert_ne!(second.0, first.0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn same_size_in_place_rewrite_is_detected_even_if_modified_time_is_preserved() {
        let path = std::env::temp_dir().join(format!(
            "ort-rewrite-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"first").unwrap();
        let modified = path.metadata().unwrap().modified().unwrap();
        let first = file_identity(&path).unwrap();
        std::fs::write(&path, b"other").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        let second = file_identity(&path).unwrap();
        assert_eq!(path.metadata().unwrap().modified().unwrap(), modified);
        assert_eq!(second.1, first.1);
        assert_ne!(
            second.0, first.0,
            "Unix ctime distinguishes rewritten same-path, same-length graphs"
        );
        std::fs::remove_file(path).unwrap();
    }

    /// A HISTORICAL DEFECT, made permanent: `to_ort_tensor` had no FP16 arm, so
    /// every EdgeTAM call — whose every boundary is FP16 — failed with
    /// "ORT input dtype FP16 is not implemented". No host suite saw it, because
    /// none built an FP16 tensor; `EngineFacadeSmokeTest` on `cuts-36` did, on
    /// the first call that crossed the boundary.
    ///
    /// kill: delete the `DType::Fp16` arm and this reds with that same message.
    #[test]
    fn an_fp16_tensor_crosses_into_ort_with_its_bit_patterns_intact() {
        use half::f16;

        let values = vec![
            f16::from_f32(-32.0),
            f16::from_f32(0.5),
            f16::ZERO,
            f16::from_f32(65504.0),
        ];
        let tensor = super::Tensor::from_f16(vec![1, 4], values.clone()).expect("f16 tensor");
        let ort = super::to_ort_tensor(tensor).expect("FP16 must reach ORT");
        let (shape, extracted) = ort
            .try_extract_tensor::<f16>()
            .expect("and come back as halves");
        assert_eq!(shape.to_vec(), vec![1_i64, 4]);
        assert_eq!(
            extracted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    use crate::ModelPaths;

    use super::OrtExecutionProvider::{CoreMl, Cpu, Cuda, DirectMl, WindowsMl, Xnnpack};
    use super::*;

    fn backend(chain: impl IntoIterator<Item = OrtExecutionProvider>) -> OrtBackend {
        OrtBackend::with_provider_chain(Arc::new(ModelPaths::new()), chain)
    }

    #[test]
    fn a_chosen_card_runs_on_its_own_providers_and_the_cpu() {
        // As the laptop's list reads: the CPU, the integrated Intel card three
        // times (one entry per provider), the NVIDIA card once.
        let device = |gpu, id, cpu_provider| DeviceEntry {
            gpu,
            id,
            cpu_provider,
        };
        let devices = [
            device(false, 0, true),
            device(true, 0x9bc4, false),
            device(true, 0x9bc4, false),
            device(true, 0x1f14, false),
            device(true, 0x9bc4, false),
        ];
        assert_eq!(card_device_indices(&devices, 0x1f14), (vec![3, 0], true));
        assert_eq!(
            card_device_indices(&devices, 0x9bc4),
            (vec![1, 2, 4, 0], true)
        );
        // A card no provider serves runs on the CPU, not on Windows ML's choice.
        assert_eq!(card_device_indices(&devices, 0x2204), (vec![0], false));
    }

    #[test]
    fn a_chosen_card_is_remembered_until_it_changes() {
        let backend = OrtBackend::new(Arc::new(ModelPaths::new()));
        assert_eq!(backend.gpu(), None);
        backend.set_gpu(Some(0x1f14));
        assert_eq!(backend.gpu(), Some(0x1f14));
        backend.set_gpu(Some(0x1f14));
        backend.set_gpu(None);
        assert_eq!(backend.gpu(), None);
    }

    #[test]
    fn windows_chain_is_windows_ml_then_cpu() {
        assert_eq!(OrtExecutionProvider::windows_chain(), [WindowsMl, Cpu]);
    }

    #[test]
    fn provider_names_are_stable() {
        assert_eq!(Cpu.name(), "CPU");
        assert_eq!(WindowsMl.name(), "WindowsML");
        assert_eq!(DirectMl.name(), "DirectML");
        assert_eq!(Cuda.name(), "CUDA");
        assert_eq!(CoreMl.name(), "CoreML");
        // The Kotlin binding sends provider names as strings, so this one is a
        // wire value: renaming it silently unpins XNNPACK on every device.
        assert_eq!(Xnnpack.name(), "XNNPACK");
    }

    #[test]
    fn new_backend_is_cpu_only_and_unresolved() {
        let backend = OrtBackend::new(Arc::new(ModelPaths::new()));
        assert_eq!(backend.configured_chain(), [Cpu]);
        assert_eq!(backend.selected_provider_for(ModelId::Autoshot), None);
    }

    #[test]
    fn placements_pin_per_model_and_leave_the_rest_on_the_chain() {
        let backend = OrtBackend::new(Arc::new(ModelPaths::new())).with_placements(vec![
            (ModelId::Autoshot, vec![DirectMl, Cpu]),
            (ModelId::MusicHighlighter, vec![Cpu]),
        ]);
        assert_eq!(backend.chain_for(ModelId::Autoshot), [DirectMl, Cpu]);
        assert_eq!(backend.chain_for(ModelId::MusicHighlighter), [Cpu]);
        // An unpinned model does not inherit a neighbour's accelerator.
        assert_eq!(backend.chain_for(ModelId::Dinov2Vits14), [Cpu]);
    }

    /// The DINOv2 experiment seam: a per-model chain can reach the ORT Core ML
    /// provider, and nothing but an explicit pin puts a model there.
    #[test]
    fn the_coreml_provider_is_reachable_only_by_an_explicit_pin() {
        let backend = OrtBackend::new(Arc::new(ModelPaths::new()))
            .with_model_provider_chain(ModelId::Dinov2Vits14, [CoreMl, Cpu]);
        assert_eq!(backend.chain_for(ModelId::Dinov2Vits14), [CoreMl, Cpu]);
        assert_eq!(backend.chain_for(ModelId::U2Netp), [Cpu]);
    }

    /// The Android floor is a chain, not a pin: XNNPACK claims the operators it
    /// implements and CPU takes the rest, so CPU has to be reachable behind it.
    /// A single-element `[Xnnpack]` chain would hard-error on a runtime built
    /// without the provider instead of running the model slowly.
    #[test]
    fn the_android_chain_is_xnnpack_then_cpu() {
        assert_eq!(OrtExecutionProvider::android_chain(), [Xnnpack, Cpu]);
        let backend = backend(OrtExecutionProvider::android_chain());
        assert_eq!(backend.configured_chain(), [Xnnpack, Cpu]);
        // Nothing is on XNNPACK until a session is created, so a caller cannot
        // read the chain as a claim about where a model actually ran.
        assert_eq!(backend.selected_provider_for(ModelId::YuNetFace), None);
    }

    /// XNNPACK never arrives by accident: it is an explicit chain or a pin, the
    /// same discipline the Core ML provider is held to.
    #[test]
    fn xnnpack_is_reachable_only_by_an_explicit_chain_or_pin() {
        let backend = OrtBackend::new(Arc::new(ModelPaths::new()))
            .with_model_provider_chain(ModelId::YuNetFace, [Xnnpack, Cpu]);
        assert_eq!(backend.chain_for(ModelId::YuNetFace), [Xnnpack, Cpu]);
        assert_eq!(backend.chain_for(ModelId::U2Netp), [Cpu]);
    }

    #[test]
    #[should_panic(expected = "needs at least one execution provider")]
    fn an_empty_pinned_chain_panics() {
        let _ = OrtBackend::new(Arc::new(ModelPaths::new()))
            .with_model_provider_chain(ModelId::Autoshot, []);
    }

    #[test]
    fn windows_default_uses_the_full_fallback_chain() {
        assert_eq!(
            backend(OrtExecutionProvider::windows_chain()).configured_chain(),
            [WindowsMl, Cpu]
        );
    }

    #[test]
    fn forced_provider_has_no_fallback() {
        // A pinned provider is a single-element chain: if it cannot load, the
        // resolver has nothing to fall through to, so it errors instead of
        // silently downgrading to CPU.
        let backend = OrtBackend::with_execution_provider(Arc::new(ModelPaths::new()), DirectMl);
        assert_eq!(backend.configured_chain(), [DirectMl]);
    }

    #[test]
    #[should_panic(expected = "at least one execution provider")]
    fn empty_chain_panics() {
        let _ = backend([]);
    }

    fn assert_real_warmup_reuses_sessions(models: &[(ModelId, &str)]) {
        assert!(
            std::env::var_os("ORT_DYLIB_PATH").is_some(),
            "name the pinned ORT runtime"
        );
        let root = std::path::PathBuf::from(
            std::env::var_os("FERMENT_ENGINE_MODELS_DIR")
                .expect("name the frozen published model directory"),
        );
        let mut paths = ModelPaths::new();
        for &(id, filename) in models {
            let path = root.join(filename);
            assert!(path.is_file(), "required graph: {}", path.display());
            paths = paths.with_path(id, ModelVersion::Latest, path);
        }
        let backend = OrtBackend::with_execution_provider(Arc::new(paths), Cpu);
        let ids = models.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        futures_executor::block_on(backend.warmup(&ids)).unwrap();
        assert_eq!(backend.cache_snapshot().resident, models.len());
        for &id in &ids {
            // Keep the actual warm instance, then use the same session door
            // infer() enters. Count equality alone would miss a reload.
            let warmed = backend.sessions.peek(&(id, ModelVersion::Latest)).unwrap();
            let requested =
                futures_executor::block_on(backend.ensure_session(id, &ModelVersion::Latest))
                    .unwrap();
            assert!(
                Arc::ptr_eq(&warmed, &requested),
                "{} reloaded after warmup",
                id.name()
            );
            assert_eq!(requested.provider, Cpu);
        }
        assert_eq!(backend.cache_snapshot().resident, models.len());
        assert_eq!(backend.cache_snapshot().loading, 0);
        assert_eq!(backend.cache_snapshot().waiters, 0);
    }

    #[test]
    #[ignore = "requires explicit pinned ORT and published AmberNet graph"]
    fn real_ambernet_warmup_hands_inference_the_same_session() {
        assert_real_warmup_reuses_sessions(&[(ModelId::AmberNetLangId, "ambernet_langid.onnx")]);
    }

    #[test]
    #[ignore = "requires explicit pinned ORT and published GigaAM graphs"]
    fn real_gigaam_warmup_hands_inference_the_same_sessions() {
        assert_real_warmup_reuses_sessions(&[
            (ModelId::GigaAmPreprocessor, "gigaam_preprocessor.onnx"),
            (ModelId::GigaAmEncoder, "gigaam_encoder.fp16.onnx"),
            (ModelId::GigaAmDecoder, "gigaam_decoder.onnx"),
            (ModelId::GigaAmJoint, "gigaam_joint.onnx"),
        ]);
    }
}

#[cfg(test)]
#[path = "ort_parakeet_policy_tests.rs"]
mod parakeet_policy_tests;

#[cfg(test)]
#[path = "ort_work_tests.rs"]
mod work_tests;
