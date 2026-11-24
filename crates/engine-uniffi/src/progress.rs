//! The two seams every long capability call needs on the Kotlin side:
//! a progress callback and a cancellation handle.
//!
//! Ported from the Swift engine's own shape — every protocol member takes
//! `onProgress: @escaping @Sendable @MainActor (Double, String) -> Void`, and
//! cancellation is the enclosing `Task`'s. UniFFI has no Task, so the handle
//! is an object the caller keeps and the call registers itself with.

use std::sync::{Arc, Mutex};

use ferment_engine_core::{ProgressSink, WorkActivity, WorkActivityScope, WorkEvent, WorkPhase};

/// The Kotlin side of `onProgress(fraction, stage)`.
///
/// A callback interface rather than a record: the engine calls it many times
/// per operation, from the thread the capability runs on. Kotlin hops to its
/// own dispatcher — the Swift `@MainActor` annotation is the same instruction
/// on the other platform and is not something the FFI can enforce.
#[uniffi::export(with_foreign)]
pub trait ProgressListener: Send + Sync {
    fn on_progress(&self, progress: f64, stage: String);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum NativeWorkPhase {
    Preparing,
    Processing,
    LoadingModel,
    Finalizing,
}

impl From<WorkPhase> for NativeWorkPhase {
    fn from(phase: WorkPhase) -> Self {
        match phase {
            WorkPhase::Preparing => Self::Preparing,
            WorkPhase::Processing => Self::Processing,
            WorkPhase::LoadingModel => Self::LoadingModel,
            WorkPhase::Finalizing => Self::Finalizing,
        }
    }
}

/// None retains physical work; Some enters a logical phase without retaining it.
#[uniffi::export(with_foreign)]
pub trait WorkListener: Send + Sync {
    fn is_cancelled(&self) -> bool;
    fn on_started(&self, id: u64, phase: Option<NativeWorkPhase>);
    fn on_finished(&self, id: u64);
    fn on_checkpoint(&self);
}

/// The caller's handle on a running capability call.
///
/// Swift cancels by cancelling the enclosing `Task`; the engine's processors
/// read that through `ProgressSink::is_closed`. Here the caller makes one of
/// these BEFORE the call, passes it in, and may cancel from any thread.
///
/// Every sink a call builds registers itself here, so one token can cover a
/// pipeline of several calls — and a token cancelled before the call starts
/// cancels it immediately rather than losing the request.
#[derive(uniffi::Object, Default)]
pub struct Cancellation {
    state: Mutex<CancellationState>,
}

#[derive(Default)]
struct CancellationState {
    cancelled: bool,
    next: u64,
    sinks: Vec<(u64, ProgressSink)>,
    work_listener: Option<Arc<dyn WorkListener>>,
}

#[uniffi::export]
impl Cancellation {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self::default()
    }

    /// Idempotent. Cancels every call already registered and every call this
    /// token is handed to afterwards.
    pub fn cancel(&self) {
        let sinks = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.cancelled {
                return;
            }
            state.cancelled = true;
            state
                .sinks
                .iter()
                .map(|(_, sink)| sink.clone())
                .collect::<Vec<_>>()
        };
        for sink in sinks {
            sink.cancel();
        }
    }

    pub fn set_work_listener(&self, listener: Option<Arc<dyn WorkListener>>) {
        let previous = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            std::mem::replace(&mut state.work_listener, listener)
        };
        drop(previous);
    }

    pub fn is_cancelled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cancelled
    }
}

impl Cancellation {
    /// Register a live call's sink, or cancel it on the spot if the token has
    /// already fired. The check and the push are one critical section, so a
    /// `cancel()` racing a starting call either cancels the sink through the
    /// list or through this branch — never neither.
    fn register(&self, sink: &ProgressSink) -> Option<u64> {
        let registration = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.cancelled {
                None
            } else {
                let id = state.next;
                state.next += 1;
                state.sinks.push((id, sink.clone()));
                Some(id)
            }
        };
        if registration.is_none() {
            sink.cancel();
        }
        registration
    }

    fn work_listener(&self) -> Option<Arc<dyn WorkListener>> {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .work_listener
            .clone()
    }

    fn deregister(&self, id: u64) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.sinks.retain(|(registered, _)| *registered != id);
    }

    #[cfg(test)]
    fn registered(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .sinks
            .len()
    }
}

/// One capability call's progress seam, live for exactly that call.
///
/// The registration is scoped rather than permanent: a token is meant to
/// outlive the calls made under it, so anything it keeps per call it keeps
/// forever. What a sink holds is not bytes — it is the `ProgressListener`, and
/// a foreign listener is a handle into the Kotlin runtime.
pub(crate) struct CallProgress {
    sink: ProgressSink,
    registration: Option<(Arc<Cancellation>, u64)>,
    _work: Option<WorkActivityScope>,
}

impl CallProgress {
    /// The sink to hand the capability. A clone, because `ProgressSink` shares
    /// its cancellation flag and this guard has to outlive the call it covers.
    pub(crate) fn sink(&self) -> ProgressSink {
        self.sink.clone()
    }

    /// Deliver one event from a synchronous façade method. `report` is `async`
    /// only because the engine's own callers are; there is nothing to await.
    pub(crate) fn report(&self, progress: f64, stage: &str) {
        futures_executor::block_on(self.sink.report(progress, stage));
    }
}

impl Drop for CallProgress {
    fn drop(&mut self) {
        if let Some((cancellation, id)) = &self.registration {
            cancellation.deregister(*id);
        }
    }
}

/// The progress seam one capability call runs under.
///
/// `None` for either half is the honest no-op: a caller that wants neither
/// progress nor cancellation passes nothing and pays nothing.
pub(crate) fn call_progress(
    listener: Option<Arc<dyn ProgressListener>>,
    cancellation: Option<Arc<Cancellation>>,
) -> CallProgress {
    let mut sink = match listener {
        Some(listener) => ProgressSink::from_callback(move |event| {
            listener.on_progress(event.progress, event.stage);
        }),
        None => ProgressSink::noop(),
    };
    if let Some(listener) = cancellation
        .as_ref()
        .and_then(|token| token.work_listener())
    {
        let permission = Arc::clone(&listener);
        sink = sink.with_work_activity(
            WorkActivity::from_callback(move |event| match event {
                WorkEvent::Started { id, phase } => listener.on_started(id, phase.map(Into::into)),
                WorkEvent::Finished { id } => listener.on_finished(id),
                WorkEvent::Checkpoint => listener.on_checkpoint(),
            })
            .with_cancellation_check(move || permission.is_cancelled()),
        );
    }
    let registration = cancellation
        .and_then(|cancellation| Some((Arc::clone(&cancellation), cancellation.register(&sink)?)));
    let work = sink.enter_work();
    CallProgress {
        sink,
        registration,
        _work: work,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct Counting(AtomicUsize);

    impl ProgressListener for Counting {
        fn on_progress(&self, _progress: f64, _stage: String) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A token outlives the calls made under it — that is its whole point,
    /// and the doc above says so ("one token can cover a pipeline of several
    /// calls"). What it must NOT do is accumulate them: a session-scoped token
    /// registering one sink per call grows a list that is never read again.
    ///
    /// Kill: register without deregistering — the list is the call count.
    #[test]
    fn a_token_does_not_retain_the_calls_that_finished_under_it() {
        let token = Arc::new(Cancellation::new());
        for _ in 0..64 {
            let _call = call_progress(None, Some(Arc::clone(&token)));
        }
        assert_eq!(token.registered(), 0);
    }

    /// And the thing being retained is a KOTLIN object. The sink owns the
    /// callback, the callback owns the `ProgressListener`, and a foreign
    /// listener is a handle into the other runtime — so a leaked sink pins a
    /// Kotlin object for the life of the token, not just some bytes.
    ///
    /// Kill: the same one. This is the consequence spelled out.
    #[test]
    fn a_finished_call_releases_its_kotlin_listener() {
        let token = Arc::new(Cancellation::new());
        let listener = Arc::new(Counting(AtomicUsize::new(0)));
        {
            let _call = call_progress(
                Some(Arc::clone(&listener) as Arc<dyn ProgressListener>),
                Some(Arc::clone(&token)),
            );
        }
        assert_eq!(Arc::strong_count(&listener), 1);
    }

    /// A token cancelled BEFORE the call cancels it anyway. The window is real:
    /// Kotlin makes the token, launches a coroutine, and the coroutine may be
    /// cancelled before the FFI call has built its sink.
    ///
    /// Kill: make `register` push unconditionally instead of checking the flag
    /// — the sink then starts live and the capability runs to completion.
    #[test]
    fn a_token_cancelled_before_the_call_cancels_it_at_registration() {
        let token = Cancellation::new();
        token.cancel();
        let sink = call_progress(None, Some(Arc::new(token)));
        assert!(sink.sink().is_closed());
    }

    /// One token covers several calls — the pipeline shape (`AED -> transcribe`)
    /// the app drives.
    ///
    /// Kill: keep only the newest sink in `Cancellation` (a `Mutex<Option<_>>`
    /// rather than a `Vec`) and the first call keeps running after `cancel()`.
    #[test]
    fn one_token_cancels_every_call_registered_under_it() {
        let token = Arc::new(Cancellation::new());
        let first = call_progress(None, Some(Arc::clone(&token)));
        let second = call_progress(None, Some(Arc::clone(&token)));
        assert!(!first.sink().is_closed() && !second.sink().is_closed());
        token.cancel();
        assert!(first.sink().is_closed(), "the first call was left running");
        assert!(second.sink().is_closed());
    }

    /// The listener is called with the engine's own numbers, not a re-scaled
    /// copy, and a call with no listener costs no allocation.
    ///
    /// Kill: swallow `report` when the sink has no callback but the token is
    /// present — the progress bar then never moves under cancellation-capable
    /// calls, which is every call the app makes.
    #[test]
    fn a_listener_receives_every_report() {
        let counter = Arc::new(Counting(AtomicUsize::new(0)));
        let sink = call_progress(
            Some(Arc::clone(&counter) as Arc<dyn ProgressListener>),
            None,
        );
        futures_executor::block_on(async {
            sink.sink().report(0.25, "one").await;
            sink.sink().report(0.5, "two").await;
        });
        assert_eq!(counter.0.load(Ordering::Relaxed), 2);
    }

    /// Kill: cancel sinks under the token registry lock; a synchronous executor wake cannot inspect or retire its call.
    #[test]
    fn cancellation_wakes_pending_work_outside_the_token_registry_lock() {
        use std::future::Future;
        use std::sync::atomic::AtomicBool;
        use std::task::{Context, Wake, Waker};

        struct ReentrantWake {
            token: std::sync::Weak<Cancellation>,
            unlocked: Arc<AtomicBool>,
        }
        impl Wake for ReentrantWake {
            fn wake(self: Arc<Self>) {
                let token = self.token.upgrade().unwrap();
                self.unlocked
                    .store(token.state.try_lock().is_ok(), Ordering::SeqCst);
            }
        }
        let token = Arc::new(Cancellation::new());
        let call = call_progress(None, Some(Arc::clone(&token)));
        let sink = call.sink();
        let unlocked = Arc::new(AtomicBool::new(false));
        let waker = Waker::from(Arc::new(ReentrantWake {
            token: Arc::downgrade(&token),
            unlocked: Arc::clone(&unlocked),
        }));
        let mut pending = Box::pin(
            sink.until_cancelled(std::future::pending::<ferment_engine_core::Result<()>>()),
        );
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        token.cancel();
        assert!(matches!(
            futures_executor::block_on(pending),
            Err(ferment_engine_core::EngineError::Cancelled)
        ));
        assert!(unlocked.load(Ordering::SeqCst));
    }
}
