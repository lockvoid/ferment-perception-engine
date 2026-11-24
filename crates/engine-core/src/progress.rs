use std::collections::BTreeMap;
use std::future::{Future, poll_fn};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::task::{Poll, Waker};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub progress: f64,
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl ProgressEvent {
    pub fn new(progress: f64, stage: impl Into<String>) -> Self {
        Self {
            progress: progress.clamp(0.0, 1.0),
            stage: stage.into(),
            data: None,
        }
    }

    #[must_use]
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}

#[cfg(not(target_arch = "wasm32"))]
type ProgressCallback = dyn Fn(ProgressEvent) + Send + Sync + 'static;
#[cfg(target_arch = "wasm32")]
type ProgressCallback = dyn Fn(ProgressEvent) + 'static;

/// Cloneable, wasm-safe replacement for the Swift progress closures.
#[derive(Clone)]
pub struct ProgressSink {
    callback: Option<Arc<ProgressCallback>>,
    cancelled: Arc<AtomicBool>,
    cancellation_callbacks: Arc<Mutex<CancellationCallbacks>>,
    work: Option<crate::WorkActivity>,
}

#[derive(Default)]
struct CancellationCallbacks {
    next: u64,
    callbacks: BTreeMap<u64, Arc<dyn Fn() + Send + Sync>>,
}

struct CancellationRegistration {
    owner: Weak<Mutex<CancellationCallbacks>>,
    id: Option<u64>,
}
impl Drop for CancellationRegistration {
    fn drop(&mut self) {
        if let (Some(owner), Some(id)) = (self.owner.upgrade(), self.id) {
            owner.lock().unwrap().callbacks.remove(&id);
        }
    }
}

impl std::fmt::Debug for ProgressSink {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProgressSink")
            .field("has_callback", &self.callback.is_some())
            .field("cancelled", &self.is_closed())
            .finish()
    }
}

impl Default for ProgressSink {
    fn default() -> Self {
        Self::noop()
    }
}

impl ProgressSink {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_callback(callback: impl Fn(ProgressEvent) + Send + Sync + 'static) -> Self {
        Self {
            callback: Some(Arc::new(callback)),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellation_callbacks: Arc::new(Mutex::new(CancellationCallbacks::default())),
            work: crate::WorkActivity::current(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_callback(callback: impl Fn(ProgressEvent) + 'static) -> Self {
        Self {
            callback: Some(Arc::new(callback)),
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellation_callbacks: Arc::new(Mutex::new(CancellationCallbacks::default())),
            work: crate::WorkActivity::current(),
        }
    }

    /// A live sink which intentionally discards updates.
    pub fn noop() -> Self {
        Self {
            callback: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            cancellation_callbacks: Arc::new(Mutex::new(CancellationCallbacks::default())),
            work: crate::WorkActivity::current(),
        }
    }

    pub async fn report(&self, progress: f64, stage: impl Into<String>) {
        self.report_event(ProgressEvent::new(progress, stage)).await;
    }

    pub fn with_work_activity(mut self, work: crate::WorkActivity) -> Self {
        self.work = Some(work.with_cancellation(Arc::clone(&self.cancelled)));
        self
    }

    /// UniFFI is synchronous: its executor polls on this same calling thread.
    pub fn enter_work(&self) -> Option<crate::WorkActivityScope> {
        self.work.as_ref().map(crate::WorkActivity::enter)
    }

    pub fn enter_phase(&self, phase: crate::WorkPhase) -> Option<crate::NativeWorkTicket> {
        self.work.as_ref().map(|work| work.enter_phase(phase))
    }

    pub fn checkpoint(&self) {
        if !self.is_closed()
            && let Some(work) = &self.work
        {
            work.checkpoint();
        }
    }

    pub async fn report_event(&self, event: ProgressEvent) {
        if !self.is_closed()
            && let Some(callback) = &self.callback
        {
            callback(event);
        }
    }

    pub fn cancel(&self) {
        if self.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }
        let callbacks = std::mem::take(&mut self.cancellation_callbacks.lock().unwrap().callbacks);
        // Callbacks may drop their registration or wake/poll a pending future.
        // Invoke after releasing the registry lock to avoid re-entrant deadlock.
        for callback in callbacks.into_values() {
            callback();
        }
    }

    fn on_cancellation(&self, callback: Arc<dyn Fn() + Send + Sync>) -> CancellationRegistration {
        let closed = self.is_closed();
        let mut callbacks = self.cancellation_callbacks.lock().unwrap();
        if closed || self.cancelled.load(Ordering::Acquire) {
            drop(callbacks);
            callback();
            return CancellationRegistration {
                owner: Arc::downgrade(&self.cancellation_callbacks),
                id: None,
            };
        }
        let id = callbacks.next;
        callbacks.next = callbacks
            .next
            .checked_add(1)
            .expect("cancellation registration counter overflow");
        callbacks.callbacks.insert(id, callback);
        CancellationRegistration {
            owner: Arc::downgrade(&self.cancellation_callbacks),
            id: Some(id),
        }
    }

    /// Cancel pending asynchronous ownership without preempting a native call.
    /// A synchronous native poll drains before returning; its result is then
    /// discarded. A queued poll is woken and dropped promptly, releasing its
    /// queue registration rather than retaining a cancelled model waiter.
    pub async fn until_cancelled<T>(
        &self,
        future: impl Future<Output = crate::Result<T>>,
    ) -> crate::Result<T> {
        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let wake = waker.clone();
        let registration = self.on_cancellation(Arc::new(move || {
            let waiting = wake.lock().unwrap().take();
            if let Some(waker) = waiting {
                waker.wake();
            }
        }));
        let mut future = std::pin::pin!(future);
        let result = poll_fn(|cx| {
            *waker.lock().unwrap() = Some(cx.waker().clone());
            if self.is_closed() {
                return Poll::Ready(Err(crate::EngineError::Cancelled));
            }
            let result = future.as_mut().poll(cx);
            if self.is_closed() {
                Poll::Ready(Err(crate::EngineError::Cancelled))
            } else {
                result
            }
        })
        .await;
        drop(registration);
        result
    }

    pub fn is_closed(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
            || self
                .work
                .as_ref()
                .is_some_and(crate::WorkActivity::is_cancelled)
    }
}

/// Accumulates fractional work before reporting an absolute 0...1 value.
///
/// This mirrors the Swift engine helper while keeping delivery in the
/// orchestration-free [`ProgressSink`] seam.
#[derive(Clone, Debug)]
pub struct ProgressTracker {
    sink: ProgressSink,
    value: Arc<Mutex<f64>>,
}

impl ProgressTracker {
    pub fn new(sink: ProgressSink) -> Self {
        Self {
            sink,
            value: Arc::new(Mutex::new(0.0)),
        }
    }

    pub fn accumulate(&self, delta: f64) {
        if delta.is_finite() && delta > 0.0 {
            self.sink.checkpoint();
        }
        let mut value = self.value.lock().unwrap();
        *value = (*value + delta).min(1.0);
    }

    pub async fn emit(&self, stage: impl Into<String>) {
        let value = *self.value.lock().unwrap();
        self.sink.report(value, stage).await;
    }

    pub async fn advance(&self, delta: f64, stage: impl Into<String>) {
        self.accumulate(delta);
        self.emit(stage).await;
    }

    pub fn value(&self) -> f64 {
        *self.value.lock().unwrap()
    }
}

/// Accumulates every update but rate-limits progress delivery.
#[derive(Clone, Debug)]
pub struct ProgressThrottler {
    tracker: ProgressTracker,
    minimum_interval: Duration,
    last_emit: Arc<Mutex<Option<Instant>>>,
}

impl ProgressThrottler {
    pub fn new(tracker: ProgressTracker, minimum_interval: Duration) -> Self {
        Self {
            tracker,
            minimum_interval,
            last_emit: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn advance(&self, delta: f64, stage: impl Into<String>, force: bool) {
        self.tracker.accumulate(delta);
        let should_emit = {
            let now = Instant::now();
            let mut last_emit = self.last_emit.lock().unwrap();
            if force
                || last_emit.is_none_or(|last| now.duration_since(last) >= self.minimum_interval)
            {
                *last_emit = Some(now);
                true
            } else {
                false
            }
        };
        if should_emit {
            self.tracker.emit(stage).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn progress_is_clamped_and_delivered() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let sink = ProgressSink::from_callback(move |event| captured.lock().unwrap().push(event));
        futures_executor::block_on(sink.report(2.0, "done"));
        assert_eq!(events.lock().unwrap()[0].progress, 1.0);
    }

    #[test]
    fn noop_is_not_cancellation() {
        let sink = ProgressSink::noop();
        assert!(!sink.is_closed());
        sink.cancel();
        assert!(sink.is_closed());
    }

    fn captured_tracker() -> (ProgressTracker, Arc<Mutex<Vec<ProgressEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let sink = ProgressSink::from_callback(move |event| captured.lock().unwrap().push(event));
        (ProgressTracker::new(sink), events)
    }

    #[test]
    fn test_advance_accumulates() {
        let (tracker, events) = captured_tracker();
        futures_executor::block_on(async {
            tracker.advance(0.3, "Step 1").await;
            tracker.advance(0.5, "Step 2").await;
            tracker.advance(0.2, "Step 3").await;
        });
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert!((events[0].progress - 0.3).abs() < 0.001);
        assert!((events[1].progress - 0.8).abs() < 0.001);
        assert!((events[2].progress - 1.0).abs() < 0.001);
        assert_eq!(events[0].stage, "Step 1");
        assert_eq!(events[1].stage, "Step 2");
        assert_eq!(events[2].stage, "Step 3");
    }

    #[test]
    fn test_clamps_to_one() {
        let (tracker, events) = captured_tracker();
        futures_executor::block_on(async {
            tracker.advance(0.7, "A").await;
            tracker.advance(0.7, "B").await;
        });
        assert!((events.lock().unwrap()[1].progress - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_delta_preserves_value() {
        let (tracker, events) = captured_tracker();
        futures_executor::block_on(async {
            tracker.advance(0.5, "A").await;
            tracker.advance(0.0, "B").await;
        });
        let values = events
            .lock()
            .unwrap()
            .iter()
            .map(|event| event.progress)
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0.5, 0.5]);
    }

    #[test]
    fn test_messages_pass_through() {
        let (tracker, events) = captured_tracker();
        futures_executor::block_on(async {
            tracker.advance(0.1, "Loading...").await;
            tracker.advance(0.2, "Processing...").await;
            tracker.advance(0.7, "Done").await;
        });
        let messages = events
            .lock()
            .unwrap()
            .iter()
            .map(|event| event.stage.clone())
            .collect::<Vec<_>>();
        assert_eq!(messages, ["Loading...", "Processing...", "Done"]);
    }

    #[test]
    fn test_accumulate_does_not_emit() {
        let (tracker, events) = captured_tracker();
        tracker.accumulate(0.3);
        tracker.accumulate(0.5);
        assert!(events.lock().unwrap().is_empty());
    }

    #[test]
    fn test_accumulate_then_emit() {
        let (tracker, events) = captured_tracker();
        tracker.accumulate(0.3);
        tracker.accumulate(0.2);
        futures_executor::block_on(tracker.emit("Combined"));
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert!((events[0].progress - 0.5).abs() < 0.001);
        assert_eq!(events[0].stage, "Combined");
    }

    #[test]
    fn test_first_advance_always_emits() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_secs(1));
        futures_executor::block_on(throttler.advance(0.1, "Start", false));
        assert_eq!(events.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_rapid_advances_throttled() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_secs(1));
        futures_executor::block_on(async {
            for index in 0..10 {
                throttler
                    .advance(0.05, format!("Chunk {index}"), false)
                    .await;
            }
        });
        assert_eq!(events.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_force_bypasses_throttle() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_secs(1));
        futures_executor::block_on(async {
            throttler.advance(0.1, "Start", false).await;
            throttler.advance(0.1, "Throttled", false).await;
            throttler.advance(0.1, "Forced", true).await;
        });
        assert_eq!(events.lock().unwrap().len(), 2);
    }

    #[test]
    fn test_accumulates_even_when_throttled() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_secs(1));
        futures_executor::block_on(async {
            for _ in 0..5 {
                throttler.advance(0.1, "Chunk", false).await;
            }
            throttler.advance(0.0, "Final", true).await;
        });
        assert!((events.lock().unwrap().last().unwrap().progress - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_emits_after_interval_elapses() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_millis(10));
        futures_executor::block_on(throttler.advance(0.1, "First", false));
        thread::sleep(Duration::from_millis(20));
        futures_executor::block_on(throttler.advance(0.1, "After interval", false));
        assert_eq!(events.lock().unwrap().len(), 2);
    }

    #[test]
    fn test_zero_delta_force_emits_current_value() {
        let (tracker, events) = captured_tracker();
        let throttler = ProgressThrottler::new(tracker, Duration::from_secs(1));
        futures_executor::block_on(async {
            throttler.advance(0.3, "A", false).await;
            throttler.advance(0.2, "B", false).await;
            throttler.advance(0.0, "Flush", true).await;
        });
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert!((events[0].progress - 0.3).abs() < 0.001);
        assert!((events[1].progress - 0.5).abs() < 0.001);
    }

    /// Swift's completed positive units reach activity before presentation throttling.
    /// Kill: remove accumulate's activity hook; seventeen chunks become one UI report and zero checkpoints.
    #[test]
    fn completed_units_report_activity_before_progress_throttling() {
        use std::sync::atomic::AtomicUsize;
        let checkpoints = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&checkpoints);
        let work = crate::WorkActivity::from_callback(move |event| {
            if event == crate::WorkEvent::Checkpoint {
                captured.fetch_add(1, Ordering::SeqCst);
            }
        });
        let deliveries = Arc::new(AtomicUsize::new(0));
        let captured = Arc::clone(&deliveries);
        let sink = ProgressSink::from_callback(move |_| {
            captured.fetch_add(1, Ordering::SeqCst);
        })
        .with_work_activity(work);
        let tracker = ProgressTracker::new(sink);
        let throttler = ProgressThrottler::new(tracker.clone(), Duration::MAX);
        futures_executor::block_on(async {
            for _ in 0..17 {
                throttler.advance(0.01, "completed chunk", false).await;
            }
        });
        tracker.accumulate(0.0);
        tracker.accumulate(-0.01);
        tracker.accumulate(f64::NAN);
        assert_eq!(deliveries.load(Ordering::SeqCst), 1);
        assert_eq!(checkpoints.load(Ordering::SeqCst), 17);
    }
}
