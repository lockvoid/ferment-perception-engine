//! A model instance owns one bounded native execution lane. Dropping a caller
//! detaches its reply, not the native operation or the buffers it is using.
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};

use ferment_engine_core::{EngineError, Result, WorkActivity};

type Job = Box<dyn FnOnce() + Send>;

pub(crate) struct NativeWorker(SyncSender<Job>);

impl NativeWorker {
    pub fn new(name: String) -> Result<Self> {
        let (sender, receiver) = sync_channel::<Job>(1);
        std::thread::Builder::new().name(name).spawn(move || {
            while let Ok(job) = receiver.recv() {
                job();
            }
        })?;
        Ok(Self(sender))
    }

    /// Admission is the cached instance's prediction permit. No unbounded
    /// work queue and no native call on the async executor/UI thread.
    pub async fn run<T: Send + 'static>(
        &self,
        operation: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<T> {
        WorkActivity::check_current()?;
        let work = WorkActivity::current();
        let ownership = work.as_ref().map(WorkActivity::retain_work);
        let (sender, receiver) = futures_channel::oneshot::channel();
        let job = Box::new(move || {
            let _ownership = ownership;
            let _scope = work.as_ref().map(WorkActivity::enter);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation))
                .unwrap_or_else(|_| Err(EngineError::internal("native prediction panicked")));
            if result.is_ok()
                && let Some(work) = &work
            {
                work.checkpoint();
            }
            let _ = sender.send(result);
        });
        self.0.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => EngineError::internal("native prediction lane is full"),
            TrySendError::Disconnected(_) => {
                EngineError::internal("native prediction worker stopped")
            }
        })?;
        receiver
            .await
            .map_err(|_| EngineError::internal("native prediction reply was lost"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction_permit::PredictionPermitQueue;
    use std::future::Future;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    struct Buffers(Arc<AtomicBool>);
    impl Drop for Buffers {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn cancellation_detaches_reply_but_keeps_native_buffers_and_permit_alive() {
        use ferment_engine_core::{WorkActivity, WorkEvent};
        let worker = NativeWorker::new("cancellation-test".into()).unwrap();
        let queue = PredictionPermitQueue::default();
        let permit = futures_executor::block_on(queue.acquire());
        let dropped = Arc::new(AtomicBool::new(false));
        let buffers = Buffers(Arc::clone(&dropped));
        let (started, running) = mpsc::channel();
        let (release, barrier) = mpsc::channel();
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = Arc::clone(&events);
        let work = WorkActivity::from_callback(move |event| recorded.lock().unwrap().push(event));
        let mut future = Box::pin(worker.run(move || {
            started.send(()).unwrap();
            barrier.recv().unwrap();
            drop(buffers);
            permit.finish(true);
            Ok(17)
        }));
        {
            let _scope = work.enter();
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
        }
        running.recv_timeout(Duration::from_secs(5)).unwrap();
        let owners_before_drop = events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, WorkEvent::Started { phase: None, .. }))
            .count();
        drop(future);
        let released_before_native_return = events
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, WorkEvent::Finished { .. }));
        assert!(!dropped.load(Ordering::SeqCst));
        assert_eq!(queue.snapshot().active, 1);
        release.send(()).unwrap();
        assert_eq!(
            futures_executor::block_on(worker.run(|| Ok(42))).unwrap(),
            42
        );
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(
            owners_before_drop, 1,
            "the native job must retain its calling scope"
        );
        assert!(
            !released_before_native_return,
            "dropping the reply cannot release native work"
        );
        let events = events.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, WorkEvent::Finished { .. }))
                .count(),
            1
        );
        let snapshot = queue.snapshot();
        assert_eq!(
            (snapshot.active, snapshot.completed, snapshot.failed),
            (0, 1, 0)
        );
    }

    #[test]
    fn a_panicking_job_reports_failure_without_poisoning_the_next_job() {
        let worker = NativeWorker::new("panic-test".into()).unwrap();
        let error =
            futures_executor::block_on(worker.run(|| -> Result<()> { panic!("native fixture") }))
                .unwrap_err();
        assert!(error.to_string().contains("native prediction panicked"));
        assert_eq!(futures_executor::block_on(worker.run(|| Ok(7))).unwrap(), 7);
    }
}
