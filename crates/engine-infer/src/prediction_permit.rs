//! One FIFO native-prediction lane per cached model instance. A pending future
//! owns its registration; dropping it removes the waiter immediately, even
//! while a non-preemptible native prediction still holds the permit.
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PredictionSnapshot {
    pub active: usize,
    pub waiting: usize,
    pub peak_active: usize,
    pub peak_waiting: usize,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Waiting,
    Granted,
    Claimed,
    Abandoned,
}
#[derive(Debug)]
struct WaiterState {
    status: Status,
    waker: Option<Waker>,
}
#[derive(Debug)]
struct Waiter {
    id: u64,
    state: Mutex<WaiterState>,
}
#[derive(Debug, Default)]
struct State {
    next: u64,
    active: Option<u64>,
    waiting: VecDeque<Arc<Waiter>>,
    metrics: PredictionSnapshot,
}
#[derive(Debug, Default)]
struct Shared(Mutex<State>);

#[derive(Debug, Clone, Default)]
pub struct PredictionPermitQueue {
    shared: Arc<Shared>,
}
impl PredictionPermitQueue {
    pub fn acquire(&self) -> Acquire {
        let mut state = self.shared.0.lock().unwrap();
        let id = state.next;
        state.next = state
            .next
            .checked_add(1)
            .expect("prediction permit identity overflow");
        let granted = state.active.is_none();
        let waiter = Arc::new(Waiter {
            id,
            state: Mutex::new(WaiterState {
                status: if granted {
                    Status::Granted
                } else {
                    Status::Waiting
                },
                waker: None,
            }),
        });
        if granted {
            state.active = Some(id);
            state.metrics.active = 1;
            state.metrics.peak_active = state.metrics.peak_active.max(1);
        } else {
            state.waiting.push_back(waiter.clone());
            state.metrics.waiting = state.waiting.len();
            state.metrics.peak_waiting = state.metrics.peak_waiting.max(state.metrics.waiting);
        }
        Acquire {
            shared: self.shared.clone(),
            waiter,
            claimed: false,
        }
    }
    pub fn snapshot(&self) -> PredictionSnapshot {
        self.shared.0.lock().unwrap().metrics
    }
}

#[derive(Debug)]
pub struct Acquire {
    shared: Arc<Shared>,
    waiter: Arc<Waiter>,
    claimed: bool,
}
impl Future for Acquire {
    type Output = PredictionPermit;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = this.waiter.state.lock().unwrap();
        match state.status {
            Status::Granted => {
                state.status = Status::Claimed;
                state.waker = None;
                this.claimed = true;
                Poll::Ready(PredictionPermit {
                    shared: this.shared.clone(),
                    id: this.waiter.id,
                    finished: false,
                })
            }
            Status::Waiting => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            _ => panic!("prediction acquisition polled after completion"),
        }
    }
}
impl Drop for Acquire {
    fn drop(&mut self) {
        if self.claimed {
            return;
        }
        let wake = {
            let mut state = self.shared.0.lock().unwrap();
            let mut waiter = self.waiter.state.lock().unwrap();
            match waiter.status {
                Status::Waiting => {
                    state
                        .waiting
                        .retain(|candidate| candidate.id != self.waiter.id);
                    state.metrics.waiting = state.waiting.len();
                    waiter.status = Status::Abandoned;
                    waiter.waker = None;
                    None
                }
                Status::Granted => {
                    assert_eq!(state.active, Some(self.waiter.id));
                    waiter.status = Status::Abandoned;
                    drop(waiter);
                    grant_next(&mut state)
                }
                _ => None,
            }
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    }
}

// Caller owns the lane state. Never invoke a waker while retaining that lock:
// a custom executor is allowed to poll the awakened future synchronously.
fn grant_next(state: &mut State) -> Option<Waker> {
    state.active = None;
    state.metrics.active = 0;
    let wake = state.waiting.pop_front().and_then(|waiter| {
        let mut waiting = waiter.state.lock().unwrap();
        assert_eq!(waiting.status, Status::Waiting);
        waiting.status = Status::Granted;
        state.active = Some(waiter.id);
        state.metrics.active = 1;
        state.metrics.peak_active = state.metrics.peak_active.max(1);
        waiting.waker.take()
    });
    state.metrics.waiting = state.waiting.len();
    wake
}

#[derive(Debug)]
pub struct PredictionPermit {
    shared: Arc<Shared>,
    id: u64,
    finished: bool,
}
impl PredictionPermit {
    pub fn finish(mut self, success: bool) {
        self.release(success);
    }
    fn release(&mut self, success: bool) {
        if self.finished {
            return;
        }
        let wake = {
            let mut state = self.shared.0.lock().unwrap();
            assert_eq!(state.active, Some(self.id));
            if success {
                state.metrics.completed += 1;
            } else {
                state.metrics.failed += 1;
            }
            self.finished = true;
            grant_next(&mut state)
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    }
}
impl Drop for PredictionPermit {
    fn drop(&mut self) {
        self.release(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_core::{EngineError, ProgressSink};
    use futures_executor::block_on;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;
    #[derive(Default)]
    struct WakeCount(AtomicUsize);
    impl Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Kill: retain cancelled predictions behind the still-held native permit, or let one cancellation acquire it successfully.
    #[test]
    fn one_thousand_cancelled_prediction_waiters_release_behind_a_held_permit() {
        let queue = PredictionPermitQueue::default();
        let held = block_on(queue.acquire());
        let mut waiters = Vec::new();
        let mut cancellations = Vec::new();
        let wake_count = Arc::new(WakeCount::default());
        let waker = Waker::from(wake_count.clone());
        let mut cx = Context::from_waker(&waker);
        for _ in 0..1_000 {
            let cancellation = ProgressSink::noop();
            cancellations.push(cancellation.clone());
            let lane = queue.clone();
            let mut waiter = Box::pin(async move {
                cancellation
                    .until_cancelled(async { Ok(lane.acquire().await) })
                    .await
            });
            assert!(waiter.as_mut().poll(&mut cx).is_pending());
            waiters.push(waiter);
        }
        assert_eq!(queue.snapshot().waiting, 1_000);
        assert_eq!(queue.snapshot().peak_waiting, 1_000);
        for cancellation in cancellations {
            cancellation.cancel();
        }
        assert_eq!(
            wake_count.0.load(Ordering::SeqCst),
            1_000,
            "every cancelled queued task must be woken without a permit handoff"
        );
        let mut cancelled = 0;
        for waiter in waiters {
            assert!(matches!(block_on(waiter), Err(EngineError::Cancelled)));
            cancelled += 1;
        }
        assert_eq!(cancelled, 1_000);
        assert_eq!(queue.snapshot().waiting, 0);
        assert_eq!(
            queue.snapshot().active,
            1,
            "the original native call still owns its permit"
        );
        held.finish(true);
        assert_eq!(queue.snapshot().active, 0);
    }

    /// Kill: replace FIFO admission with LIFO or allow more than one live prediction to claim the model.
    #[test]
    fn sixteen_live_prediction_waiters_remain_fifo() {
        let queue = PredictionPermitQueue::default();
        let held = block_on(queue.acquire());
        let mut cx = Context::from_waker(Waker::noop());
        let mut waiting = Vec::new();
        for index in 0..16 {
            let mut waiter = Box::pin(queue.acquire());
            assert!(waiter.as_mut().poll(&mut cx).is_pending());
            waiting.push(waiter);
            assert_eq!(queue.snapshot().waiting, index + 1);
        }
        held.finish(true);
        let mut order = Vec::new();
        for index in 0..16 {
            let mut ready = Vec::new();
            for (candidate, waiter) in waiting.iter_mut().enumerate().skip(index) {
                if let Poll::Ready(permit) = waiter.as_mut().poll(&mut cx) {
                    ready.push((candidate, permit));
                }
            }
            assert_eq!(ready.len(), 1);
            let (candidate, permit) = ready.pop().unwrap();
            assert_eq!(candidate, index);
            order.push(candidate);
            permit.finish(true);
        }
        assert_eq!(order, (0..16).collect::<Vec<_>>());
        assert_eq!(queue.snapshot().waiting, 0);
        assert_eq!(queue.snapshot().active, 0);
        assert_eq!(queue.snapshot().peak_active, 1);
    }

    /// Kill: strand a permit when cancellation wins after handoff but before its future resumes.
    #[test]
    fn cancellation_after_handoff_releases_the_unclaimed_permit() {
        let queue = PredictionPermitQueue::default();
        let held = block_on(queue.acquire());
        let cancellation = ProgressSink::noop();
        let token = cancellation.clone();
        let lane = queue.clone();
        let mut pending = Box::pin(async move {
            token
                .until_cancelled(async { Ok(lane.acquire().await) })
                .await
        });
        let mut cx = Context::from_waker(Waker::noop());
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        let next = queue.acquire();
        held.finish(true);
        cancellation.cancel();
        assert!(matches!(block_on(pending), Err(EngineError::Cancelled)));
        block_on(next).finish(true);
        assert_eq!(queue.snapshot().active, 0);
        assert_eq!(queue.snapshot().waiting, 0);
    }
    /// Kill: wake an obsolete executor registration, omit the cancellation wake,
    /// or retain a callback/waker after a cancelled queued operation completes.
    #[test]
    fn cancellation_wakes_the_latest_registered_task_exactly_once() {
        let queue = PredictionPermitQueue::default();
        let held = block_on(queue.acquire());
        let cancellation = ProgressSink::noop();
        let lane = queue.clone();
        let mut pending =
            Box::pin(cancellation.until_cancelled(async { Ok(lane.acquire().await) }));
        let first = Arc::new(WakeCount::default());
        let second = Arc::new(WakeCount::default());
        let first_waker = Waker::from(first.clone());
        let second_waker = Waker::from(second.clone());
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(&first_waker))
                .is_pending()
        );
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(&second_waker))
                .is_pending()
        );
        cancellation.cancel();
        assert_eq!(first.0.load(Ordering::SeqCst), 0);
        assert_eq!(second.0.load(Ordering::SeqCst), 1);
        assert!(matches!(block_on(pending), Err(EngineError::Cancelled)));
        assert_eq!(queue.snapshot().waiting, 0);
        cancellation.cancel();
        held.finish(true);
        assert_eq!(second.0.load(Ordering::SeqCst), 1);
        assert_eq!(queue.snapshot().active, 0);
    }
}
