//! Actual completed work and native lifetime observations, without scheduling
//! or timeout policy. The host owns the attempt and its clock.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkPhase {
    Preparing,
    Processing,
    LoadingModel,
    Finalizing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkEvent {
    /// None retains physical work; Some enters a logical phase only.
    Started {
        id: u64,
        phase: Option<WorkPhase>,
    },
    Finished {
        id: u64,
    },
    Checkpoint,
}

type Callback = dyn Fn(WorkEvent) + Send + Sync;
type CancellationCheck = dyn Fn() -> bool + Send + Sync;

/// A cloneable observer carried across owned native thread handoffs.
#[derive(Clone)]
pub struct WorkActivity {
    callback: Arc<Callback>,
    cancelled: Arc<AtomicBool>,
    cancellation_check: Option<Arc<CancellationCheck>>,
}

impl std::fmt::Debug for WorkActivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkActivity")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

thread_local! {
    static CURRENT: RefCell<Option<WorkActivity>> = const { RefCell::new(None) };
}

impl WorkActivity {
    pub fn from_callback(callback: impl Fn(WorkEvent) + Send + Sync + 'static) -> Self {
        Self {
            callback: Arc::new(callback),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellation_check: None,
        }
    }

    pub fn with_cancellation(mut self, cancelled: Arc<AtomicBool>) -> Self {
        self.cancelled = cancelled;
        self
    }

    pub fn with_cancellation_check(
        mut self,
        check: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        self.cancellation_check = Some(Arc::new(check));
        self
    }

    pub fn current() -> Option<Self> {
        CURRENT.with(|current| current.borrow().clone())
    }

    /// Synchronous, thread-bound scope. It must not span an executor hop.
    pub fn enter(&self) -> WorkActivityScope {
        WorkActivityScope {
            previous: CURRENT.with(|current| current.replace(Some(self.clone()))),
            thread_bound: PhantomData,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
            || self
                .cancellation_check
                .as_ref()
                .is_some_and(|check| check())
    }

    pub fn check_cancellation(&self) -> crate::Result<()> {
        if self.is_cancelled() {
            Err(crate::EngineError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn check_current() -> crate::Result<()> {
        Self::current()
            .as_ref()
            .map_or(Ok(()), Self::check_cancellation)
    }

    /// A completed frame/chunk/read, never a periodic keep-alive.
    pub fn checkpoint(&self) {
        if !self.is_cancelled() {
            (self.callback)(WorkEvent::Checkpoint);
        }
    }

    pub fn retain_work(&self) -> NativeWorkTicket {
        self.start(None)
    }

    pub fn enter_phase(&self, phase: WorkPhase) -> NativeWorkTicket {
        self.start(Some(phase))
    }

    fn start(&self, phase: Option<WorkPhase>) -> NativeWorkTicket {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        (self.callback)(WorkEvent::Started { id, phase });
        NativeWorkTicket {
            _ticket: Arc::new(Ticket {
                callback: Arc::clone(&self.callback),
                id,
            }),
        }
    }
}

/// Restores a nested synchronous context on the same thread.
pub struct WorkActivityScope {
    previous: Option<WorkActivity>,
    thread_bound: PhantomData<Rc<()>>,
}

impl Drop for WorkActivityScope {
    fn drop(&mut self) {
        CURRENT.with(|current| current.replace(self.previous.take()));
    }
}

/// Last-owner release; logical cancellation cannot release another clone.
#[derive(Clone)]
pub struct NativeWorkTicket {
    _ticket: Arc<Ticket>,
}

struct Ticket {
    callback: Arc<Callback>,
    id: u64,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        (self.callback)(WorkEvent::Finished { id: self.id });
    }
}
