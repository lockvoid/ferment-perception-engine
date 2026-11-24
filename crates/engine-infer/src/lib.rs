//! Inference contracts and backend-neutral tensor representation.

mod backend;
#[cfg(all(feature = "coreai", target_os = "macos"))]
mod coreai;
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
mod coreml;
#[cfg(all(any(feature = "coreml", feature = "coreai"), target_os = "macos"))]
mod coreml_disk;
#[cfg(any(
    all(feature = "coreml", target_vendor = "apple"),
    all(feature = "coreai", target_os = "macos")
))]
mod coreml_identity;
#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod directml_turn;
mod locator;
mod model;
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreai", target_os = "macos"),
    all(feature = "coreml", target_vendor = "apple")
))]
mod model_io;
#[cfg(any(
    all(feature = "coreml", target_vendor = "apple"),
    all(feature = "coreai", target_os = "macos")
))]
mod native_worker;
#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
mod ort_backend;
pub mod prediction_permit;
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreai", target_os = "macos"),
    all(feature = "coreml", target_vendor = "apple")
))]
mod session_cache;
mod tensor;

pub use backend::{Backend, InferenceBackend, UnavailableBackend};
#[cfg(all(feature = "coreai", target_os = "macos"))]
pub use coreai::CoreAiBackend;
#[cfg(all(feature = "coreai", target_os = "macos"))]
pub use coreai_native::device_architecture as coreai_device_architecture;
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub use coreml::CoreMlBackend;
/// Re-exported so a caller can pin compute units without depending on
/// `coreml-native` directly — the placement axis crosses this crate boundary.
#[cfg(all(feature = "coreml", target_vendor = "apple"))]
pub use coreml_native::ComputeUnits;
pub use locator::{ModelLocator, ModelPaths, SharedModelLocator};
pub use model::{ModelId, ModelVersion};
#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
pub use ort_backend::{OrtBackend, OrtExecutionProvider};
#[cfg(any(
    all(feature = "ort", not(target_arch = "wasm32")),
    all(feature = "coreai", target_os = "macos"),
    all(feature = "coreml", target_vendor = "apple")
))]
pub use session_cache::{EVICTION_FLOOR_BYTES, ModelCacheSnapshot};
pub use tensor::{DType, Tensor, TensorMap};
#[cfg(all(feature = "ort", windows))]
pub use winml_native::{AssignedNode as OrtAssignedNode, AssignedSubgraph as OrtAssignedSubgraph};
