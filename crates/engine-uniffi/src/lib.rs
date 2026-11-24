//! The Android/Kotlin door onto the perception engine.
//!
//! E0 exported the smallest surface that proves the engine cross-compiles,
//! links, loads an ONNX Runtime and infers on a device. E2 grows it into the
//! capability façade: one object per iOS `Protocols/*.swift` contract, with
//! the same member names, over an [`EngineService`] that owns one ORT backend
//! and one models directory. `infer_smoke` stays — it is E0's device gate and
//! the only test that pins the provider chain end to end.
//!
//! The crate lives here rather than in the Android repo because a `path`
//! dependency on `crates/*` cannot resolve from a lane worktree of that repo —
//! the same reason `ferment-kine` keeps its bindings beside its crate.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use ferment_engine_core::{FaceRectSource, RgbFrameRef};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use ferment_engine_processors::video_saliency::YuNetFaceSource;

uniffi::setup_scaffolding!("engine");

mod android_codecs;
mod audio_file;
mod audio_front_end;
mod capabilities;
mod live_transcription;
mod progress;
mod service;
#[cfg(test)]
mod speech_identity_tests;
mod types;

pub use android_codecs::*;
pub use audio_file::*;
pub use audio_front_end::*;
pub use capabilities::*;
pub use live_transcription::LiveTranscriptionSession;
pub use progress::{Cancellation, NativeWorkPhase, ProgressListener, WorkListener};
pub use service::{EngineService, FleetModelFile, ModelCacheState, fleet_model_files};
pub use types::*;

/// The reference frame the smoke runs on, compiled in.
///
/// The same bytes `crates/engine-processors/tests/ort_saliency.rs` reads for
/// `yunet_faces_match_the_python_origin`. Compiled in rather than delivered so
/// that the device physically cannot run a different input from the Mac: the
/// only thing the two runs are allowed to differ in is the execution provider.
pub(crate) const SMOKE_FRAME_PNG: &[u8] =
    include_bytes!("../../../scripts/exports/video_saliency/frames/selfie_640.png");

/// Everything that crosses this boundary as a failure. One case: the engine's
/// own errors are strings by the time they reach Kotlin, because E0 has no
/// caller that could branch on them.
///
/// The field is `reason`, not `message`: UniFFI lowers an error variant's
/// fields to Kotlin `val`s on a `kotlin.Exception` subclass, and a field named
/// `message` collides with `Throwable.message` — the generated file does not
/// compile. Named here rather than rewritten by a `refine.sh` pass, so the
/// binding stays exactly what `uniffi-bindgen` emitted.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum EngineFfiError {
    #[error("{reason}")]
    Engine { reason: String },
}

impl EngineFfiError {
    fn new(reason: impl Into<String>) -> Self {
        Self::Engine {
            reason: reason.into(),
        }
    }
}

/// Every engine failure crosses as its `Display`. One variant, because the app
/// branches on the OPERATION, never on the failure kind — iOS's
/// `FermentEngineError` cases are all rendered through `errorDescription` and
/// shown, and a second variant here would be a distinction no caller reads.
/// `EngineError::Cancelled` is the one exception the Kotlin side matches on,
/// by its message, which is why it is spelled out below.
impl From<ferment_engine_core::EngineError> for EngineFfiError {
    fn from(error: ferment_engine_core::EngineError) -> Self {
        Self::new(error.to_string())
    }
}

pub(crate) type FfiResult<T> = Result<T, EngineFfiError>;

/// The engine's version — the workspace version every `ferment-engine-*` crate
/// carries, `ferment-engine-processors` included.
///
/// Read from this crate's own `CARGO_PKG_VERSION`: both packages take
/// `version.workspace = true`, so the two literals cannot diverge without an
/// edit to the workspace root. The Kotlin gate compares it to the `version`
/// line of the workspace manifest, which is what makes a stale `.so` red.
#[uniffi::export]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// The ONNX Runtime this process loaded: the path the first successful
/// [`engine_init`] was given, and the answer it produced.
///
/// A `Mutex` rather than a `OnceLock` because the whole load has to be inside
/// it. With a `OnceLock` two threads both see `None`, both load, `ort` keeps
/// one library and the lock keeps the other's version string — so the answer
/// could name a runtime that is not the live one.
static ORT_RUNTIME: Mutex<Option<(String, String)>> = Mutex::new(None);

/// Point `ort` at the ONNX Runtime to `dlopen`, once, before any session.
///
/// Android has no `ORT_DYLIB_PATH` environment to inherit, so the name of the
/// runtime can only come from Kotlin. Returns
/// `<OrtGetApiBase()->GetVersionString()> — <GetBuildInfoString()>`.
///
/// The version string is read here, off the probe handle, rather than through
/// `ort`: `GetVersionString` sits on `OrtApiBase`, which `ort` does not
/// re-expose once the library is loaded, and the build-info string is not a
/// substitute — the ONNX Runtime AAR reports `git-branch=HEAD`, not
/// `rel-1.29.0`, so it names no version at all.
///
/// Idempotent for the SAME path, an error for a different one. Measured in
/// `ort` 2.0.0-rc.12: `init_from` is `OnceLock::get_or_try_init` over the dylib
/// handle and `commit()` is `try_insert_with`, so a second call there is a
/// silent no-op that keeps the FIRST library — it does not fail, and it does
/// not switch. Passing that silence through would be a lie: a caller naming a
/// second runtime would be told it loaded. It is refused instead.
///
/// That is not a theoretical tidiness. While this lane was under review the
/// device suite's `ortLoadsFromTheAarAndNamesItsVersion` was measured to pass
/// with `engineInit("libdefinitelynotort.so")` — a whole test grading a cached
/// string, green, because an earlier test in the process had already
/// initialised. Refusing the mismatch is what makes that test able to fail.
///
/// The other order is refused too: a session created BEFORE `engine_init`
/// makes `ort` load its platform default lazily (`libonnxruntime.so`, which
/// the APK's linker namespace resolves), and `commit()` then answers `false`
/// because the environment already existed. Recording the path at that point
/// would name a runtime this process never loaded through it.
///
/// # The probe
/// The load is attempted HERE first, and every failure is reported from here.
/// `ort` 2.0.0-rc.12 **hangs** rather than erroring on ANY load failure: its
/// loader closure runs inside `G_ORT_LIB`'s `std::sync::Once`, and building
/// the failure (`ort::error::Error::new_internal`) calls `ort::api()`, which
/// re-enters that same still-running `Once` through `setup_api` and blocks on
/// its futex forever. Not an Android quirk — the same stack reproduces on
/// macOS in a few microseconds of CPU and then never returns.
///
/// `ort::load_dylib_from_path` can fail three ways inside that `Once`, so
/// [`probe_runtime_version`] checks all three: the `dlopen`, the presence of
/// `OrtGetApiBase`, and the API-level floor (a runtime below `1.MINOR_VERSION`
/// is rejected — the one mode a version string alone can see, and the one that
/// a change of ONNX Runtime coordinate would otherwise turn into a hang).
#[uniffi::export]
pub fn engine_init(ort_library_path: String) -> FfiResult<String> {
    let mut runtime = ORT_RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((loaded, info)) = runtime.as_ref() {
        if *loaded != ort_library_path {
            return Err(EngineFfiError::new(format!(
                "ONNX Runtime is already loaded from '{loaded}'; '{ort_library_path}' cannot replace it"
            )));
        }
        return Ok(info.clone());
    }
    let version = probe_runtime_version(&ort_library_path)?;
    let committed = ort::init_from(&ort_library_path)
        .map_err(|error| {
            EngineFfiError::new(format!("could not load {ort_library_path}: {error}"))
        })?
        .commit();
    if !committed {
        return Err(EngineFfiError::new(format!(
            "ONNX Runtime was initialised before engine_init (a session ran first); '{ort_library_path}' is not what this process loaded"
        )));
    }
    let info = format!("{version} — {}", ort::info());
    *runtime = Some((ort_library_path, info.clone()));
    Ok(info)
}

/// What one smoke run answered.
#[derive(Debug, uniffi::Record)]
pub struct SmokeResult {
    /// The provider that actually loaded the model — `OrtBackend`'s own record,
    /// not the chain's first element. A chain that silently downgraded to CPU
    /// says `CPU` here.
    pub provider_selected: String,
    /// The detections, `x,y,w,h,confidence` per face separated by `;`, in the
    /// order the detector returned them, at full `f64` precision. A string
    /// rather than numbers because the FFI record is fixed at three fields;
    /// the caller splits and compares with the golden's own tolerance.
    pub output_summary: String,
    /// Wall-clock of pre-processing, session creation and inference together —
    /// the first call pays for the session, which is the number a cold start
    /// actually feels.
    pub elapsed_ms: f64,
}

/// Run YuNet over the compiled-in reference frame on the given provider chain.
///
/// The smallest ONNX master in the fleet — `yunet_face.onnx`, 224 KB, as
/// `ferment-models-export` publishes it (`models/saliency`) — through the
/// processor's real entry point, `YuNetFaceSource::faces`, exactly what
/// `yunet_faces_match_the_python_origin` calls. Nothing about the
/// pre-processing or the decode is re-implemented here.
///
/// `providers` are chain elements in attempt order, by
/// [`OrtExecutionProvider::name`] (`XNNPACK`, `CPU`, …). An unknown name is an
/// error rather than a silent CPU fallback.
#[uniffi::export]
pub fn infer_smoke(model_path: String, providers: Vec<String>) -> FfiResult<SmokeResult> {
    let chain = providers
        .iter()
        .map(|name| parse_provider(name))
        .collect::<FfiResult<Vec<_>>>()?;
    if chain.is_empty() {
        return Err(EngineFfiError::new("an empty provider chain cannot load"));
    }

    let frame = image::load_from_memory(SMOKE_FRAME_PNG)
        .map_err(|error| EngineFfiError::new(format!("reference frame: {error}")))?
        .to_rgb8();

    let paths = ModelPaths::new().with_path(
        ModelId::YuNetFace,
        ModelVersion::Latest,
        PathBuf::from(&model_path),
    );
    let backend = Arc::new(OrtBackend::with_provider_chain(Arc::new(paths), chain));
    let detector = YuNetFaceSource::new(Arc::clone(&backend) as Backend);

    let started = Instant::now();
    let reference = RgbFrameRef::new(frame.as_raw(), frame.width(), frame.height())
        .ok_or_else(|| EngineFfiError::new("reference frame is not packed RGB24"))?;
    let faces = futures_executor::block_on(detector.faces(reference))
        .map_err(|error| EngineFfiError::new(error.to_string()))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let provider_selected = backend
        .selected_provider_for(ModelId::YuNetFace)
        .ok_or_else(|| EngineFfiError::new("no provider was recorded for YuNetFace"))?
        .name()
        .to_owned();
    let output_summary = faces
        .iter()
        .map(|face| {
            format!(
                "{},{},{},{},{}",
                face.x, face.y, face.width, face.height, face.confidence
            )
        })
        .collect::<Vec<_>>()
        .join(";");

    Ok(SmokeResult {
        provider_selected,
        output_summary,
        elapsed_ms,
    })
}

/// `dlopen` the runtime and ask it its version, before `ort` ever touches it.
///
/// Two jobs in one load. It is the probe described on [`engine_init`], and it
/// is the only route to `OrtGetApiBase()->GetVersionString()` — the one place
/// ONNX Runtime states its release.
///
/// It reproduces every check `ort::load_dylib_from_path` makes inside its
/// `Once`, because each of them is a deadlock there and a message here: the
/// `dlopen` itself, the presence of `OrtGetApiBase`, and the API-level floor.
/// The version parse is `ort`'s own — minor component, unparsable reads as 0 —
/// so the two cannot disagree about which runtimes are acceptable.
///
/// The handle is dropped at the end of the call, before `ort` loads the same
/// file itself. `dlopen` is refcounted and the second load re-opens it, so the
/// only cost is that the library may be unmapped and remapped in between; the
/// probe hands `ort` a path, never a handle, so nothing outlives the drop.
fn probe_runtime_version(path: &str) -> FfiResult<String> {
    type ApiBaseGetter = unsafe extern "C" fn() -> *const ort::sys::OrtApiBase;

    let library = unsafe { libloading::Library::new(path) }
        .map_err(|error| EngineFfiError::new(format!("could not dlopen '{path}': {error}")))?;
    let getter: libloading::Symbol<'_, ApiBaseGetter> = unsafe { library.get(b"OrtGetApiBase") }
        .map_err(|error| EngineFfiError::new(format!("'{path}' has no OrtGetApiBase: {error}")))?;
    let base = unsafe { getter() };
    if base.is_null() {
        return Err(EngineFfiError::new(format!(
            "'{path}' returned a null OrtApiBase"
        )));
    }
    let version = unsafe { std::ffi::CStr::from_ptr(((*base).GetVersionString)()) }
        .to_string_lossy()
        .into_owned();
    let minor = version
        .split('.')
        .nth(1)
        .map_or(0, |minor| minor.parse::<u32>().unwrap_or(0));
    if minor < ort::MINOR_VERSION {
        return Err(EngineFfiError::new(format!(
            "'{path}' is ONNX Runtime {version}, below ort's floor of 1.{}.x",
            ort::MINOR_VERSION
        )));
    }
    Ok(version)
}

/// Every provider Kotlin is allowed to name.
///
/// The wire strings are [`OrtExecutionProvider::name`]'s and nothing else —
/// matched, not restated — so `provider_names_are_stable` in `engine-infer`
/// guards this boundary too and a rename cannot leave a stale spelling here.
const PROVIDERS: [OrtExecutionProvider; 5] = [
    OrtExecutionProvider::Cpu,
    OrtExecutionProvider::Xnnpack,
    OrtExecutionProvider::CoreMl,
    OrtExecutionProvider::Cuda,
    OrtExecutionProvider::DirectMl,
];

fn parse_provider(name: &str) -> FfiResult<OrtExecutionProvider> {
    PROVIDERS
        .into_iter()
        .find(|provider| provider.name().eq_ignore_ascii_case(name))
        .ok_or_else(|| EngineFfiError::new(format!("unknown execution provider '{name}'")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The other side of the seam. Prints the CPU answer the device's
    /// `EngineSmokeTest` is graded against, on the Mac, from the same compiled-in
    /// frame and the same master:
    ///
    /// ```text
    /// ORT_DYLIB_PATH=$PWD/vendor/onnxruntime/aarch64-apple-darwin/libonnxruntime.dylib \
    ///   cargo test -p ferment-engine-uniffi --lib -- --ignored --nocapture
    /// ```
    ///
    /// Kill: change the pre-processing, the decode or the master and the numbers
    /// move, which is exactly what the device assertion then catches.
    #[test]
    #[ignore = "needs an ONNX Runtime (ORT_DYLIB_PATH) and the published yunet_face.onnx"]
    fn prints_the_mac_cpu_answer_for_the_device_suite() {
        let Some(model) = ferment_engine_test_fixtures::export::artifact("yunet_face.onnx") else {
            return;
        };
        let result = infer_smoke(model.to_string_lossy().into_owned(), vec!["CPU".to_owned()])
            .expect("smoke");
        assert_eq!(result.provider_selected, "CPU");
        eprintln!("provider_selected = {}", result.provider_selected);
        eprintln!("elapsed_ms        = {}", result.elapsed_ms);
        eprintln!("output_summary    = {}", result.output_summary);
    }

    /// The provider names are a WIRE contract: Kotlin sends them as strings and
    /// the device suite pins XNNPACK by spelling it. Parsing goes through
    /// [`OrtExecutionProvider::name`], so this also proves the two directions
    /// cannot drift apart.
    ///
    /// Kill: restate a spelling in [`PROVIDERS`] instead of matching `name()`,
    /// or drop the case-insensitivity the Kotlin caller relies on.
    #[test]
    fn provider_names_round_trip_case_insensitively() {
        for provider in PROVIDERS {
            assert_eq!(parse_provider(provider.name()).unwrap(), provider);
            assert_eq!(
                parse_provider(&provider.name().to_ascii_lowercase()).unwrap(),
                provider
            );
        }
        // The chain the device smoke actually sends.
        assert_eq!(
            parse_provider("XNNPACK").unwrap(),
            OrtExecutionProvider::Xnnpack
        );
        assert_eq!(parse_provider("CPU").unwrap(), OrtExecutionProvider::Cpu);
    }

    /// An unknown name is an error, and the message quotes what the caller
    /// actually sent — not a normalised echo of it.
    #[test]
    fn an_unknown_provider_is_rejected_verbatim() {
        let error = parse_provider("tpu").unwrap_err();
        assert_eq!(error.to_string(), "unknown execution provider 'tpu'");
    }

    /// A session (or any `ort::init`) that created the environment BEFORE
    /// `engine_init` leaves `ort` on the runtime it loaded lazily; `commit()`
    /// answers `false` and `engine_init` must refuse rather than record a path
    /// this process never loaded through it.
    ///
    /// Kill: discard `commit()`'s `bool` — the call returns `Ok` and records
    /// the path.
    #[test]
    #[ignore = "needs an ONNX Runtime (ORT_DYLIB_PATH)"]
    fn engine_init_after_a_lazy_environment_is_refused() {
        let path = std::env::var("ORT_DYLIB_PATH").expect("ORT_DYLIB_PATH");
        // Whoever gets there first — this call or a sibling test's session.
        let _ = ort::init_from(&path).expect("the runtime loads").commit();
        let error = engine_init(path).unwrap_err();
        assert!(
            error.to_string().contains("initialised before engine_init"),
            "{error}"
        );
    }
}
