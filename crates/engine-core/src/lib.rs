//! Shared, orchestration-free engine contracts.

mod capabilities;
mod error;
mod image_object_segmenting;
mod progress;
mod types;
mod video_object_segmenting;
mod work_activity;

#[cfg(not(target_arch = "wasm32"))]
pub trait EngineService: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> EngineService for T {}

#[cfg(target_arch = "wasm32")]
pub trait EngineService {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> EngineService for T {}

pub use capabilities::*;
pub use error::{EngineError, Result};
pub use image_object_segmenting::*;
pub use progress::{ProgressEvent, ProgressSink, ProgressThrottler, ProgressTracker};
pub use types::*;
pub use video_object_segmenting::*;
pub use work_activity::{NativeWorkTicket, WorkActivity, WorkActivityScope, WorkEvent, WorkPhase};
