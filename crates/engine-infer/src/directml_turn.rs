//! DirectML work in this process, one piece at a time.
//!
//! ONNX Runtime's DirectML execution provider shares a GPU allocator, command
//! queue and descriptor pool between the sessions of one process without
//! locking them (onnxruntime #20713, PR #28007). Creating, running or releasing
//! two DirectML sessions at once corrupts that state and kills the process:
//! measured on the Windows server with ORT 1.24.4 and with Windows ML's ORT
//! 1.28.3 (`tests/directml_concurrency.rs`). The GPU runs one workload at a
//! time anyway and separate processes do not interfere, so one turn per
//! process costs little; Microsoft's own earlier Windows ML serialized its GPU
//! sessions the same way. Windows ML's automatic policy can select DirectML,
//! so automatic sessions share this turn too. Explicit other providers do not.

use std::sync::{Mutex, PoisonError};

use crate::OrtExecutionProvider;

static TURN: Mutex<()> = Mutex::new(());

/// Run `work` alone among this process's DirectML work when `provider` is
/// DirectML or Windows ML automatic selection; explicit other providers run freely.
pub(crate) fn in_turn<T>(provider: OrtExecutionProvider, work: impl FnOnce() -> T) -> T {
    if !matches!(
        provider,
        OrtExecutionProvider::DirectMl | OrtExecutionProvider::WindowsMl
    ) {
        return work();
    }
    // A panic inside earlier work leaves no DirectML state half-held by this
    // lock, which guards ordering only.
    let _turn = TURN.lock().unwrap_or_else(PoisonError::into_inner);
    work()
}
