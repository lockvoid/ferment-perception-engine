//! Session lifetime, independent of ORT's synchronous compiler. One worker
//! owns a native load; cancelling or timing out a waiter only detaches that
//! waiter. Bookkeeping never holds a lock across loading or inference.

use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use ferment_engine_core::{EngineError, NativeWorkTicket, Result, WorkActivity, WorkPhase};

pub const EVICTION_FLOOR_BYTES: u64 = 768 * 1_048_576;
const LOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ModelCacheSnapshot {
    pub resident: usize,
    pub loading: usize,
    pub waiters: usize,
}

type SharedResult<V> = std::result::Result<Arc<V>, Arc<EngineError>>;

enum EntryValue<V> {
    Resident(Arc<V>),
    Loading(Arc<Flight<V>>),
}

struct Entry<V> {
    identity: String,
    weight: u64,
    value: EntryValue<V>,
}

struct Inner<K, V> {
    entries: Mutex<HashMap<K, Entry<V>>>,
    available_memory: Box<dyn Fn() -> u64 + Send + Sync>,
    timeout: Duration,
}

pub(crate) struct SessionCache<K, V> {
    inner: Arc<Inner<K, V>>,
}

impl<K, V> SessionCache<K, V>
where
    K: Clone + Eq + Hash + Debug + Send + 'static,
    V: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self::with_policy(LOAD_WAIT_TIMEOUT, available_memory)
    }

    fn with_policy(
        timeout: Duration,
        available_memory: impl Fn() -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                entries: Mutex::new(HashMap::new()),
                available_memory: Box::new(available_memory),
                timeout,
            }),
        }
    }

    /// Register before starting the producer, including an immediately
    /// completing producer. The physical identity is separate from the
    /// logical key: replacing a graph at the same path invalidates its slot.
    pub fn load(
        &self,
        key: K,
        identity: String,
        weight: u64,
        producer: impl FnOnce() -> Result<V> + Send + 'static,
    ) -> LoadWaiter<V> {
        let work = WorkActivity::current();
        if work.as_ref().is_some_and(WorkActivity::is_cancelled) {
            return LoadWaiter::ready(Err(Arc::new(EngineError::Cancelled)));
        }
        let mut retired = Vec::new();
        let (flight, is_new) = {
            let mut entries = self.inner.entries.lock().unwrap();
            if let Some(entry) = entries.get(&key)
                && entry.identity == identity
            {
                match &entry.value {
                    EntryValue::Resident(value) => return LoadWaiter::ready(Ok(Arc::clone(value))),
                    EntryValue::Loading(flight) => (Arc::clone(flight), false),
                }
            } else {
                if let Some(old) = entries.remove(&key) {
                    retired.push(old);
                }
                // Unknown headroom is not evidence of memory pressure.
                let available = (self.inner.available_memory)();
                if available != 0 && available < EVICTION_FLOOR_BYTES {
                    let victims = inactive_victims(&entries, |_| false);
                    for victim in victims {
                        retired.extend(entries.remove(&victim));
                    }
                }
                let flight = Arc::new(Flight::new(format!("{key:?}")));
                entries.insert(
                    key.clone(),
                    Entry {
                        identity,
                        weight,
                        value: EntryValue::Loading(Arc::clone(&flight)),
                    },
                );
                (flight, true)
            }
        };
        // Native destructors can be expensive. Free memory before creating a
        // new graph, but outside the registry lock.
        drop(retired);

        let timeout = work.is_none().then_some(self.inner.timeout);
        let waiter = flight.subscribe(timeout, work.as_ref());
        if !is_new {
            return waiter;
        }

        let timer = Arc::clone(&flight);
        if let Err(error) = std::thread::Builder::new()
            .name("model-load-deadline".into())
            .spawn(move || timer.watch_deadlines())
        {
            self.inner
                .complete(&key, &flight, Err(Arc::new(error.into())));
            return waiter;
        }
        let inner = Arc::clone(&self.inner);
        let worker_key = key.clone();
        let worker_flight = Arc::clone(&flight);
        if let Err(error) = std::thread::Builder::new()
            .name("model-load".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(producer))
                    .unwrap_or_else(|_| Err(EngineError::internal("native model loader panicked")))
                    .map(Arc::new)
                    .map_err(Arc::new);
                inner.complete(&worker_key, &worker_flight, result);
            })
        {
            self.inner
                .complete(&key, &flight, Err(Arc::new(error.into())));
        }
        waiter
    }

    #[cfg(any(test, feature = "ort"))]
    pub fn peek(&self, key: &K) -> Option<Arc<V>> {
        let entries = self.inner.entries.lock().unwrap();
        match &entries.get(key)?.value {
            EntryValue::Resident(value) => Some(Arc::clone(value)),
            EntryValue::Loading(_) => None,
        }
    }

    /// Swift's idle sweep drops cache ownership, not a consumer's reference.
    /// Active calls keep their Arc until all native outputs have been copied.
    /// Pending loads are never cancelled or detached by a memory warning.
    pub fn evict_idle(&self, except: impl Fn(&K) -> bool) -> Vec<K> {
        let (keys, retired) = {
            let mut entries = self.inner.entries.lock().unwrap();
            let keys = ordered_victims(&entries, except);
            let retired: Vec<_> = keys.iter().filter_map(|key| entries.remove(key)).collect();
            (keys, retired)
        };
        drop(retired);
        keys
    }

    /// Sweep unleased residents even without memory pressure. Unlike
    /// `evict_idle`, this keeps active and queued callers' registry entries
    /// joinable, so a stage change cannot create a second native instance
    /// while a cancelled caller's first native operation is still draining.
    pub fn evict_inactive(&self, except: impl Fn(&K) -> bool) -> Vec<K> {
        let (keys, retired) = {
            let mut entries = self.inner.entries.lock().unwrap();
            let keys = inactive_victims(&entries, except);
            let retired: Vec<_> = keys.iter().filter_map(|key| entries.remove(key)).collect();
            (keys, retired)
        };
        // A destructor may be slow or re-enter the cache. Never retain its lock.
        drop(retired);
        keys
    }

    /// Android no longer sends foreground trim callbacks. Reused sessions can
    /// grow their inference arenas after the last model-load check, so each
    /// inference admission also sheds inactive siblings under the same floor.
    /// A live caller or queued prediction owns an Arc: leave its registry entry
    /// in place so another caller cannot start a duplicate native session.
    pub fn evict_inactive_if_pressure(&self, except: impl Fn(&K) -> bool) -> Vec<K> {
        let available = (self.inner.available_memory)();
        if available == 0 || available >= EVICTION_FLOOR_BYTES {
            return Vec::new();
        }
        self.evict_inactive(except)
    }

    /// A replaced resource invalidates its own versions only. An old native
    /// compiler may finish for its existing waiters, but cannot republish.
    pub fn invalidate(&self, matches: impl Fn(&K) -> bool) -> usize {
        let retired: Vec<_> = {
            let mut entries = self.inner.entries.lock().unwrap();
            let keys: Vec<_> = entries.keys().filter(|key| matches(key)).cloned().collect();
            keys.iter().filter_map(|key| entries.remove(key)).collect()
        };
        let count = retired.len();
        drop(retired);
        count
    }

    pub fn snapshot(&self) -> ModelCacheSnapshot {
        let entries = self.inner.entries.lock().unwrap();
        let mut snapshot = ModelCacheSnapshot::default();
        for entry in entries.values() {
            match &entry.value {
                EntryValue::Resident(_) => snapshot.resident += 1,
                EntryValue::Loading(flight) => {
                    snapshot.loading += 1;
                    snapshot.waiters += flight.state.lock().unwrap().waiters.len();
                }
            }
        }
        snapshot
    }
}

impl<K: Eq + Hash, V> Inner<K, V> {
    fn complete(&self, key: &K, flight: &Arc<Flight<V>>, result: SharedResult<V>) {
        {
            let mut entries = self.entries.lock().unwrap();
            let still_current = matches!(entries.get(key).map(|e| &e.value),
                Some(EntryValue::Loading(current)) if Arc::ptr_eq(current, flight));
            if still_current {
                match &result {
                    Ok(value) => {
                        entries.get_mut(key).unwrap().value =
                            EntryValue::Resident(Arc::clone(value));
                    }
                    Err(_) => {
                        entries.remove(key);
                    }
                }
            }
        }
        // Publication precedes resumption; never wake arbitrary executor
        // code while the cache or flight bookkeeping is locked.
        flight.complete(result);
    }
}

fn inactive_victims<K: Clone + Debug + Eq + Hash, V>(
    entries: &HashMap<K, Entry<V>>,
    except: impl Fn(&K) -> bool,
) -> Vec<K> {
    ordered_victims(entries, except)
        .into_iter()
        .filter(|key| {
            matches!(entries.get(key).map(|entry| &entry.value),
            Some(EntryValue::Resident(value)) if Arc::strong_count(value) == 1)
        })
        .collect()
}

fn ordered_victims<K: Clone + Debug, V>(
    entries: &HashMap<K, Entry<V>>,
    except: impl Fn(&K) -> bool,
) -> Vec<K> {
    let mut victims: Vec<_> = entries
        .iter()
        .filter(|(key, entry)| !except(key) && matches!(entry.value, EntryValue::Resident(_)))
        .collect();
    victims.sort_by(|(ka, a), (kb, b)| {
        b.weight
            .cmp(&a.weight)
            .then_with(|| format!("{ka:?}").cmp(&format!("{kb:?}")))
    });
    victims.into_iter().map(|(key, _)| key.clone()).collect()
}

struct WaiterState<V> {
    result: Option<SharedResult<V>>,
    waker: Option<Waker>,
}

struct Waiter<V> {
    deadline: Option<Instant>,
    state: Mutex<WaiterState<V>>,
}

impl<V> Waiter<V> {
    fn finish(&self, result: SharedResult<V>) {
        let waker = {
            let mut state = self.state.lock().unwrap();
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct FlightState<V> {
    completion: Option<SharedResult<V>>,
    waiters: HashMap<u64, Arc<Waiter<V>>>,
    native_owners: Vec<NativeWorkTicket>,
}

struct Flight<V> {
    description: String,
    state: Mutex<FlightState<V>>,
    changed: Condvar,
    next_waiter: AtomicU64,
}

impl<V> Flight<V> {
    fn new(description: String) -> Self {
        Self {
            description,
            state: Mutex::new(FlightState {
                completion: None,
                waiters: HashMap::new(),
                native_owners: Vec::new(),
            }),
            changed: Condvar::new(),
            next_waiter: AtomicU64::new(0),
        }
    }

    fn subscribe(
        self: &Arc<Self>,
        timeout: Option<Duration>,
        work: Option<&WorkActivity>,
    ) -> LoadWaiter<V> {
        let mut native_owner = work.map(WorkActivity::retain_work);
        let phase = work.map(|work| work.enter_phase(WorkPhase::LoadingModel));
        let id = self.next_waiter.fetch_add(1, Ordering::Relaxed);
        let waiter = Arc::new(Waiter {
            deadline: timeout.map(|timeout| Instant::now() + timeout),
            state: Mutex::new(WaiterState {
                result: None,
                waker: None,
            }),
        });
        let ready = {
            let mut state = self.state.lock().unwrap();
            if let Some(result) = &state.completion {
                Some(result.clone())
            } else {
                state.waiters.insert(id, Arc::clone(&waiter));
                state.native_owners.extend(native_owner.take());
                None
            }
        };
        drop(native_owner);
        if let Some(result) = ready {
            waiter.finish(result);
        }
        self.changed.notify_one();
        LoadWaiter {
            waiter,
            registration: Some((Arc::clone(self), id)),
            phase,
        }
    }

    fn complete(&self, result: SharedResult<V>) {
        let (waiters, native_owners) = {
            let mut state = self.state.lock().unwrap();
            if state.completion.is_some() {
                return;
            }
            state.completion = Some(result.clone());
            (
                std::mem::take(&mut state.waiters),
                std::mem::take(&mut state.native_owners),
            )
        };
        self.changed.notify_one();
        drop(native_owners);
        for waiter in waiters.into_values() {
            waiter.finish(result.clone());
        }
    }

    /// One deadline monitor per native load, not per waiter. A storm of
    /// cancelled or timed-out callers leaves neither registrations nor a
    /// storm of sleeping threads behind the blocked compiler.
    fn watch_deadlines(&self) {
        let mut state = self.state.lock().unwrap();
        loop {
            if state.completion.is_some() {
                return;
            }
            let now = Instant::now();
            let expired: Vec<_> = state
                .waiters
                .iter()
                .filter(|(_, waiter)| waiter.deadline.is_some_and(|deadline| deadline <= now))
                .map(|(&id, _)| id)
                .collect();
            if !expired.is_empty() {
                let waiters: Vec<_> = expired
                    .iter()
                    .filter_map(|id| state.waiters.remove(id))
                    .collect();
                drop(state);
                for waiter in waiters {
                    waiter.finish(Err(Arc::new(EngineError::invalid_config(format!(
                        "model load {} timed out while the single tracked native load continues in the background",
                        self.description,
                    )))));
                }
                state = self.state.lock().unwrap();
                continue;
            }
            state = if let Some(deadline) = state.waiters.values().filter_map(|w| w.deadline).min()
            {
                self.changed
                    .wait_timeout(state, deadline.saturating_duration_since(now))
                    .unwrap()
                    .0
            } else {
                self.changed.wait(state).unwrap()
            };
        }
    }
}

pub(crate) struct LoadWaiter<V> {
    waiter: Arc<Waiter<V>>,
    registration: Option<(Arc<Flight<V>>, u64)>,
    phase: Option<NativeWorkTicket>,
}

impl<V> LoadWaiter<V> {
    fn ready(result: SharedResult<V>) -> Self {
        Self {
            waiter: Arc::new(Waiter {
                deadline: None,
                state: Mutex::new(WaiterState {
                    result: Some(result),
                    waker: None,
                }),
            }),
            registration: None,
            phase: None,
        }
    }
}

impl<V> Future for LoadWaiter<V> {
    type Output = Result<Arc<V>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let result = {
            let mut state = this.waiter.state.lock().unwrap();
            if let Some(result) = &state.result {
                Some(result.as_ref().map(Arc::clone).map_err(|e| copy_error(e)))
            } else {
                state.waker = Some(cx.waker().clone());
                None
            }
        };
        match result {
            Some(result) => {
                this.phase.take();
                Poll::Ready(result)
            }
            None => Poll::Pending,
        }
    }
}

impl<V> Drop for LoadWaiter<V> {
    fn drop(&mut self) {
        if let Some((flight, id)) = &self.registration {
            flight.state.lock().unwrap().waiters.remove(id);
            flight.changed.notify_one();
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn available_memory() -> u64 {
    // Android/Linux expose usable headroom, including reclaimable pages.
    // Hosts without this interface return unknown rather than false pressure.
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let value = line
                    .strip_prefix("MemAvailable:")?
                    .split_whitespace()
                    .next()?;
                value
                    .parse::<u64>()
                    .ok()
                    .and_then(|kb| kb.checked_mul(1024))
            })
        })
        .unwrap_or(u64::MAX)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn available_memory() -> u64 {
    // Safe platform facade; only RAM is queried (no process/CPU/disk scan).
    // Multiple model consumers share a 250 ms sample rather than each asking
    // the kernel on every prediction. Zero means unknown, never pressure.
    use std::sync::OnceLock;
    static PROBE: OnceLock<Mutex<(sysinfo::System, Option<Instant>)>> = OnceLock::new();
    let mut probe = PROBE
        .get_or_init(|| Mutex::new((sysinfo::System::new(), None)))
        .lock()
        .unwrap();
    if probe
        .1
        .is_none_or(|sampled| sampled.elapsed() >= Duration::from_millis(250))
    {
        probe
            .0
            .refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
        probe.1 = Some(Instant::now());
    }
    probe.0.available_memory()
}

// Preserve the producer's typed error for every joined caller; io::Error is
// not Clone, so reconstruct its original OS code or kind/message.
fn copy_error(error: &EngineError) -> EngineError {
    match error {
        EngineError::ModelNotFound(s) => EngineError::ModelNotFound(s.clone()),
        EngineError::ModelOutputMissing { model, key } => EngineError::ModelOutputMissing {
            model: model.clone(),
            key: key.clone(),
        },
        EngineError::InvalidOutput => EngineError::InvalidOutput,
        EngineError::NoAudioData => EngineError::NoAudioData,
        EngineError::AudioTooShort {
            samples,
            min_required,
        } => EngineError::AudioTooShort {
            samples: *samples,
            min_required: *min_required,
        },
        EngineError::InvalidConfiguration(s) => EngineError::InvalidConfiguration(s.clone()),
        EngineError::StftFailed(s) => EngineError::StftFailed(s.clone()),
        EngineError::Decode(s) => EngineError::Decode(s.clone()),
        EngineError::Encode(s) => EngineError::Encode(s.clone()),
        EngineError::Inference(s) => EngineError::Inference(s.clone()),
        EngineError::VideoIo(s) => EngineError::VideoIo(s.clone()),
        EngineError::Models(s) => EngineError::Models(s.clone()),
        EngineError::Io(e) => EngineError::Io(e.raw_os_error().map_or_else(
            || std::io::Error::new(e.kind(), e.to_string()),
            std::io::Error::from_raw_os_error,
        )),
        EngineError::Cancelled => EngineError::Cancelled,
        EngineError::Unsupported(s) => EngineError::Unsupported(s.clone()),
        EngineError::Internal(s) => EngineError::Internal(s.clone()),
    }
}

#[cfg(test)]
mod tests;
