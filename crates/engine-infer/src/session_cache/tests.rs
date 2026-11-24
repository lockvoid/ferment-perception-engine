use super::*;
use futures_executor::block_on;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

type Cache = SessionCache<String, usize>;

fn cache() -> Cache {
    Cache::with_policy(Duration::from_secs(5), || u64::MAX)
}

fn request(cache: &Cache, key: &str, value: usize) -> LoadWaiter<usize> {
    cache.load(key.into(), "graph-v1".into(), value as u64, move || {
        Ok(value)
    })
}

fn eventually(check: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !check() {
        assert!(
            Instant::now() < deadline,
            "bounded wait for worker completion"
        );
        std::thread::yield_now();
    }
}

#[test]
fn completed_load_is_registered_before_the_first_caller_resumes() {
    let cache = cache();
    for i in 0..100 {
        let key = format!("immediate-{i}");
        assert_eq!(*block_on(request(&cache, &key, i)).unwrap(), i);
        assert_eq!(*cache.peek(&key).expect("published before resumption"), i);
        assert_eq!(cache.snapshot().loading, 0);
    }
}

#[test]
fn warmup_and_consumer_share_one_native_load_and_instance() {
    let cache = cache();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let (release, wait) = mpsc::channel();
    let warmup = cache.load("one".into(), "v1".into(), 50, move || {
        counted.fetch_add(1, Ordering::SeqCst);
        wait.recv().unwrap();
        Ok(17)
    });
    let callers: Vec<_> = (0..256)
        .map(|_| {
            cache.load("one".into(), "v1".into(), 50, || {
                panic!("a joined call must not invoke its loader")
            })
        })
        .collect();
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 0,
            loading: 1,
            waiters: 257
        }
    );
    release.send(()).unwrap();
    let model = block_on(warmup).unwrap();
    for caller in callers {
        assert!(Arc::ptr_eq(&model, &block_on(caller).unwrap()));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 1,
            loading: 0,
            waiters: 0
        }
    );
}

#[test]
fn cancellation_storm_removes_every_waiter_and_keeps_the_original_load() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let owner = cache.load("one".into(), "v1".into(), 9, move || {
        wait.recv().unwrap();
        Ok(9)
    });
    let cancelled: Vec<_> = (0..512)
        .map(|_| {
            cache.load("one".into(), "v1".into(), 9, || {
                panic!("no duplicate worker")
            })
        })
        .collect();
    assert_eq!(cache.snapshot().waiters, 513);
    drop(owner);
    drop(cancelled);
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 0,
            loading: 1,
            waiters: 0
        }
    );
    let survivor = cache.load("one".into(), "v1".into(), 9, || {
        panic!("retry joins live worker")
    });
    release.send(()).unwrap();
    assert_eq!(*block_on(survivor).unwrap(), 9);
}

#[test]
fn cancelling_the_first_caller_does_not_cancel_the_survivor() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let first = cache.load("one".into(), "v1".into(), 1, move || {
        wait.recv().unwrap();
        Ok(40)
    });
    let survivor = cache.load("one".into(), "v1".into(), 1, || panic!("joined"));
    drop(first);
    assert_eq!(cache.snapshot().waiters, 1);
    release.send(()).unwrap();
    assert_eq!(*block_on(survivor).unwrap(), 40);
}

#[test]
fn timeout_storm_leaves_no_waiters_and_retries_join_until_the_load_lands() {
    let cache = Cache::with_policy(Duration::from_millis(100), || u64::MAX);
    let (release, wait) = mpsc::channel();
    let first = cache.load("one".into(), "v1".into(), 1, move || {
        wait.recv().unwrap();
        Ok(81)
    });
    let mut timed_out = vec![first];
    timed_out
        .extend((0..128).map(|_| cache.load("one".into(), "v1".into(), 1, || panic!("joined"))));
    for waiter in timed_out {
        let error = block_on(waiter).unwrap_err();
        assert!(matches!(error, EngineError::InvalidConfiguration(_)));
        assert!(
            error
                .to_string()
                .contains("single tracked native load continues")
        );
    }
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 0,
            loading: 1,
            waiters: 0
        }
    );
    // A later caller gets its own full deadline, without starting another load.
    let later = cache.load("one".into(), "v1".into(), 1, || panic!("still joined"));
    assert_eq!(cache.snapshot().waiters, 1);
    release.send(()).unwrap();
    assert_eq!(*block_on(later).unwrap(), 81);
    assert_eq!(
        *block_on(cache.load("one".into(), "v1".into(), 1, || panic!(
            "landed instance must be reused"
        )))
        .unwrap(),
        81
    );
}

#[test]
fn joined_loader_failure_keeps_its_error_kind_and_does_not_poison_retry() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let first = cache.load("one".into(), "v1".into(), 1, move || {
        wait.recv().unwrap();
        Err(EngineError::Io(std::io::Error::from_raw_os_error(13)))
    });
    let second = cache.load("one".into(), "v1".into(), 1, || panic!("joined"));
    release.send(()).unwrap();
    for result in [block_on(first), block_on(second)] {
        match result {
            Err(EngineError::Io(error)) => assert_eq!(error.raw_os_error(), Some(13)),
            _ => panic!("every joined caller receives the producer's original error kind"),
        }
    }
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
    assert_eq!(
        *block_on(cache.load("one".into(), "v1".into(), 1, || Ok(4))).unwrap(),
        4
    );
}

#[test]
fn panicking_loader_finishes_waiters_and_releases_the_key_for_retry() {
    let cache = cache();
    let error = block_on(cache.load("one".into(), "v1".into(), 1, || {
        panic!("native loader failure")
    }))
    .unwrap_err();
    assert!(matches!(error, EngineError::Internal(_)));
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
    assert_eq!(
        *block_on(cache.load("one".into(), "v1".into(), 1, || Ok(6))).unwrap(),
        6
    );
}

#[test]
fn models_and_versions_are_independent_structured_keys() {
    let cache = SessionCache::with_policy(Duration::from_secs(5), || u64::MAX);
    let keys = [("one:main", "v1"), ("one", "main:v1"), ("one", "v2")];
    let mut values = Vec::new();
    for (i, key) in keys.into_iter().enumerate() {
        values.push(
            block_on(cache.load(key, "same-physical-file".into(), 10, move || Ok(i))).unwrap(),
        );
    }
    assert_eq!(cache.snapshot().resident, 3);
    assert_eq!(values.iter().map(|v| **v).collect::<Vec<_>>(), [0, 1, 2]);
}

#[test]
fn separate_cache_contexts_never_reuse_another_backends_session() {
    let first = cache();
    let second = cache();
    assert_eq!(*block_on(request(&first, "one", 1)).unwrap(), 1);
    assert_eq!(*block_on(request(&second, "one", 2)).unwrap(), 2);
    first.evict_idle(|_| false);
    assert_eq!(second.snapshot().resident, 1);
}

#[test]
fn replacing_the_physical_identity_at_the_same_key_reloads_just_that_key() {
    let cache = cache();
    let active = block_on(cache.load("one".into(), "old-file".into(), 10, || Ok(10))).unwrap();
    block_on(request(&cache, "neighbor", 30)).unwrap();
    let replacement = block_on(cache.load("one".into(), "new-file".into(), 10, || Ok(20))).unwrap();
    assert_eq!(
        *active, 10,
        "already accepted native calls retain their old session"
    );
    assert_eq!(*replacement, 20);
    assert!(!Arc::ptr_eq(&active, &replacement));
    assert_eq!(*cache.peek(&"neighbor".into()).unwrap(), 30);
    assert_eq!(cache.snapshot().resident, 2);
}

#[test]
fn invalidated_inflight_load_cannot_overwrite_its_replacement() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let old = cache.load("one".into(), "old".into(), 1, move || {
        wait.recv().unwrap();
        Ok(1)
    });
    block_on(request(&cache, "neighbor", 5)).unwrap();
    assert_eq!(cache.invalidate(|key| key == "one"), 1);
    let new = block_on(cache.load("one".into(), "new".into(), 1, || Ok(2))).unwrap();
    release.send(()).unwrap();
    assert_eq!(*block_on(old).unwrap(), 1);
    assert!(Arc::ptr_eq(&new, &cache.peek(&"one".into()).unwrap()));
    assert_eq!(*cache.peek(&"neighbor".into()).unwrap(), 5);
    assert_eq!(cache.snapshot().resident, 2);
}

#[test]
fn changed_file_midload_does_not_let_the_previous_flight_republish() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let old = cache.load("one".into(), "old".into(), 1, move || {
        wait.recv().unwrap();
        Ok(1)
    });
    let new = block_on(cache.load("one".into(), "new".into(), 1, || Ok(2))).unwrap();
    release.send(()).unwrap();
    assert_eq!(*block_on(old).unwrap(), 1);
    assert!(Arc::ptr_eq(&new, &cache.peek(&"one".into()).unwrap()));
}

#[test]
fn eviction_releases_owned_models_and_an_evicted_model_reloads_on_demand() {
    let cache = cache();
    let model = block_on(request(&cache, "one", 1)).unwrap();
    let weak = Arc::downgrade(&model);
    drop(model);
    assert_eq!(cache.evict_idle(|_| false), ["one"]);
    eventually(|| weak.upgrade().is_none());
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
    assert_eq!(*block_on(request(&cache, "one", 2)).unwrap(), 2);
}

#[test]
fn active_session_lives_through_eviction_then_releases_when_its_call_finishes() {
    let cache = cache();
    let active = block_on(request(&cache, "one", 1)).unwrap();
    let weak = Arc::downgrade(&active);
    assert_eq!(cache.evict_idle(|_| false), ["one"]);
    assert_eq!(cache.snapshot().resident, 0);
    assert_eq!(
        *active, 1,
        "eviction cannot invalidate an active native call"
    );
    assert_eq!(*block_on(request(&cache, "one", 2)).unwrap(), 2);
    drop(active);
    eventually(|| weak.upgrade().is_none());
}

// Direct residents avoid depending on when a model-load worker drops its last
// completion handle. Values and lease counts, not wall-clock timing, drive these
// stage-sweep tests; `load` still grades whether a later caller really reuses it.
fn resident(cache: &Cache, key: &str, weight: usize) -> Arc<usize> {
    let value = Arc::new(weight);
    cache.inner.entries.lock().unwrap().insert(
        key.into(),
        Entry {
            identity: "graph-v1".into(),
            weight: weight as u64,
            value: EntryValue::Resident(value.clone()),
        },
    );
    value
}

#[test]
fn inactive_sweep_keeps_native_and_queued_leases_joinable_until_both_finish() {
    let cache = cache(); // Ample RAM: this sweep is not a pressure-only operation.
    let native = resident(&cache, "one", 19);
    let queued = Arc::clone(&native);
    assert!(cache.evict_inactive(|_| false).is_empty());
    let joined = block_on(cache.load("one".into(), "graph-v1".into(), 19, || {
        panic!("a draining native instance must remain joinable")
    }))
    .unwrap();
    assert!(Arc::ptr_eq(&native, &joined));
    drop(native);
    drop(joined);
    assert!(cache.evict_inactive(|_| false).is_empty());
    assert_eq!(cache.snapshot().resident, 1);
    let weak = Arc::downgrade(&queued);
    drop(queued);
    assert_eq!(cache.evict_inactive(|_| false), ["one"]);
    assert!(weak.upgrade().is_none());
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
    assert_eq!(*block_on(request(&cache, "one", 23)).unwrap(), 23);
}

#[test]
fn inactive_sweep_keeps_exceptions_and_pending_loads_and_orders_unleased_models() {
    let cache = cache();
    for (key, weight) in [("light", 10), ("heavy", 900), ("keep", 50)] {
        drop(resident(&cache, key, weight));
    }
    let active = resident(&cache, "active", 1000);
    let (release, wait) = mpsc::channel();
    let loading = cache.load("loading".into(), "graph-v1".into(), 1200, move || {
        wait.recv().unwrap();
        Ok(1200)
    });
    assert_eq!(
        cache.evict_inactive(|key| key == "keep"),
        ["heavy", "light"]
    );
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 2,
            loading: 1,
            waiters: 1,
        }
    );
    assert!(Arc::ptr_eq(&active, &cache.peek(&"active".into()).unwrap()));
    let joined = cache.load("loading".into(), "graph-v1".into(), 1200, || {
        panic!("stage sweep must not detach a pending load")
    });
    release.send(()).unwrap();
    let loaded = block_on(loading).unwrap();
    assert!(Arc::ptr_eq(&loaded, &block_on(joined).unwrap()));
}

#[test]
fn eviction_orders_actual_resource_weights_and_keeps_excepted_keys_and_loads() {
    let cache = cache();
    for (key, weight) in [
        ("light:main", 10),
        ("medium:main", 300),
        ("heavy:main", 900),
        ("keep", 50),
    ] {
        block_on(cache.load(key.into(), "graph".into(), weight, || Ok(0))).unwrap();
    }
    let (release, wait) = mpsc::channel();
    let loading = cache.load("loading".into(), "graph".into(), 1200, move || {
        wait.recv().unwrap();
        Ok(0)
    });
    assert_eq!(
        cache.evict_idle(|key| key == "keep"),
        ["heavy:main", "medium:main", "light:main"]
    );
    assert_eq!(
        cache.snapshot(),
        ModelCacheSnapshot {
            resident: 1,
            loading: 1,
            waiters: 1
        }
    );
    release.send(()).unwrap();
    block_on(loading).unwrap();
    assert_eq!(cache.snapshot().resident, 2);
}

#[test]
fn preload_eviction_occurs_only_below_the_floor_and_never_on_a_hit() {
    assert_eq!(EVICTION_FLOOR_BYTES, 768 * 1_048_576);
    for (available, expected) in [
        (u64::MAX, 2),
        (EVICTION_FLOOR_BYTES, 2),
        (0, 2),
        (EVICTION_FLOOR_BYTES - 1, 1),
    ] {
        let cache = Cache::with_policy(Duration::from_secs(5), move || available);
        block_on(request(&cache, "a", 1)).unwrap();
        block_on(request(&cache, "b", 2)).unwrap();
        assert_eq!(cache.snapshot().resident, expected, "available={available}");
        assert_eq!(*block_on(request(&cache, "b", 99)).unwrap(), 2);
        assert_eq!(cache.snapshot().resident, expected);
    }
}

#[test]
fn blocked_load_does_not_hold_the_registry_or_prevent_another_model_loading() {
    let cache = cache();
    let (release, wait) = mpsc::channel();
    let blocked = cache.load("blocked".into(), "v1".into(), 1, move || {
        wait.recv().unwrap();
        Ok(1)
    });
    assert_eq!(*block_on(request(&cache, "unrelated", 2)).unwrap(), 2);
    assert_eq!(cache.snapshot().loading, 1);
    release.send(()).unwrap();
    block_on(blocked).unwrap();
}

#[test]
fn preload_pressure_keeps_an_active_sibling_joinable_without_duplicate_load() {
    let cache = Cache::with_policy(Duration::from_secs(5), || EVICTION_FLOOR_BYTES - 1);
    let active = block_on(request(&cache, "active", 1)).unwrap();
    block_on(request(&cache, "new-model", 2)).unwrap();
    let joined = block_on(cache.load("active".into(), "graph-v1".into(), 1, || {
        panic!("loading a new model must not orphan the active sibling's registry entry")
    }))
    .unwrap();
    assert!(Arc::ptr_eq(&active, &joined));
    drop(joined);
    drop(active);
    eventually(|| {
        cache
            .peek(&"active".into())
            .is_some_and(|value| Arc::strong_count(&value) == 2)
    });
    assert!(
        cache
            .evict_inactive_if_pressure(|_| false)
            .contains(&"active".into())
    );
}

// Kill: a pressure check only on cache misses leaves old inference arenas
// resident after the device stops sending foreground memory warnings.
#[test]
fn inference_pressure_reuses_the_requested_hot_graph_and_drops_idle_siblings() {
    let headroom = Arc::new(AtomicU64::new(u64::MAX));
    let probe = Arc::clone(&headroom);
    let cache = Cache::with_policy(Duration::from_secs(5), move || probe.load(Ordering::SeqCst));
    block_on(request(&cache, "old-stage", 31)).unwrap();
    let hot = block_on(request(&cache, "current-stage", 47)).unwrap();
    eventually(|| {
        cache
            .peek(&"old-stage".into())
            .is_some_and(|value| Arc::strong_count(&value) == 2)
    });
    headroom.store(EVICTION_FLOOR_BYTES - 1, Ordering::SeqCst);
    assert_eq!(
        cache.evict_inactive_if_pressure(|key| key == "current-stage"),
        ["old-stage"]
    );
    let reused = block_on(
        cache.load("current-stage".into(), "graph-v1".into(), 47, || {
            panic!("pressure admission must not reload the current graph")
        }),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&hot, &reused));
    assert!(cache.peek(&"old-stage".into()).is_none());
    assert_eq!(cache.snapshot().resident, 1);
}

// Kill: dropping an active graph's registry slot permits a duplicate load
// while its first prediction still holds native weights and arena memory.
#[test]
fn inference_pressure_keeps_active_siblings_registered_until_their_last_lease_ends() {
    let headroom = Arc::new(AtomicU64::new(u64::MAX));
    let probe = Arc::clone(&headroom);
    let cache = Cache::with_policy(Duration::from_secs(5), move || probe.load(Ordering::SeqCst));
    let active = block_on(request(&cache, "other-active", 19)).unwrap();
    headroom.store(EVICTION_FLOOR_BYTES - 1, Ordering::SeqCst);
    assert!(cache.evict_inactive_if_pressure(|_| false).is_empty());
    let joined = block_on(
        cache.load("other-active".into(), "graph-v1".into(), 19, || {
            panic!("a live sibling must remain joinable under pressure")
        }),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&active, &joined));
    drop(joined);
    drop(active);
    eventually(|| {
        cache
            .peek(&"other-active".into())
            .is_some_and(|value| Arc::strong_count(&value) == 2)
    });
    assert_eq!(
        cache.evict_inactive_if_pressure(|_| false),
        ["other-active"]
    );
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
}

// Kill: pressure retirement must neither cancel nor detach the one native
// load; the original waiter and a later consumer must receive the same graph.
#[test]
fn inference_pressure_leaves_pending_loads_and_their_waiters_joinable() {
    let cache = Cache::with_policy(Duration::from_secs(5), || EVICTION_FLOOR_BYTES - 1);
    let (release, wait) = mpsc::channel();
    let first = cache.load("loading".into(), "v1".into(), 11, move || {
        wait.recv().unwrap();
        Ok(11)
    });
    assert!(cache.evict_inactive_if_pressure(|_| false).is_empty());
    let second = cache.load("loading".into(), "v1".into(), 11, || {
        panic!("single producer")
    });
    assert_eq!(cache.snapshot().loading, 1);
    assert_eq!(cache.snapshot().waiters, 2);
    release.send(()).unwrap();
    let a = block_on(first).unwrap();
    let b = block_on(second).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(*a, 11);
}

// Kill: unknown headroom or the exact healthy boundary must not turn every
// repeated prediction into a new model load.
#[test]
fn inference_pressure_respects_unknown_healthy_and_exact_floor_headroom() {
    for available in [0, u64::MAX, EVICTION_FLOOR_BYTES] {
        let cache = Cache::with_policy(Duration::from_secs(5), move || available);
        block_on(request(&cache, "a", 1)).unwrap();
        block_on(request(&cache, "b", 2)).unwrap();
        eventually(|| {
            cache
                .peek(&"a".into())
                .is_some_and(|value| Arc::strong_count(&value) == 2)
        });
        assert!(
            cache.evict_inactive_if_pressure(|_| false).is_empty(),
            "available={available}"
        );
        assert_eq!(cache.snapshot().resident, 2);
    }
}

// Kill: collecting retired handles without dropping them before admission
// still overlaps the next allocation; dropping under the lock can deadlock
// a native destructor that reports its state back to the cache.
#[test]
fn inference_pressure_destroys_retired_graphs_before_return_outside_the_registry_lock() {
    struct NativeLifetime {
        key: &'static str,
        cache: std::sync::Weak<Inner<String, NativeLifetime>>,
        destroyed: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Drop for NativeLifetime {
        fn drop(&mut self) {
            if let Some(cache) = self.cache.upgrade() {
                assert!(
                    cache.entries.try_lock().is_ok(),
                    "native destruction under registry lock"
                );
            }
            self.destroyed.lock().unwrap().push(self.key);
        }
    }
    let headroom = Arc::new(AtomicU64::new(u64::MAX));
    let probe = Arc::clone(&headroom);
    let cache: SessionCache<String, NativeLifetime> =
        SessionCache::with_policy(Duration::from_secs(5), move || probe.load(Ordering::SeqCst));
    let destroyed = Arc::new(Mutex::new(Vec::new()));
    for (key, weight) in [("small", 1), ("large", 3)] {
        let owner = Arc::downgrade(&cache.inner);
        let recorded = Arc::clone(&destroyed);
        block_on(cache.load(key.into(), "v1".into(), weight, move || {
            Ok(NativeLifetime {
                key,
                cache: owner,
                destroyed: recorded,
            })
        }))
        .unwrap();
        eventually(|| {
            cache
                .peek(&key.into())
                .is_some_and(|value| Arc::strong_count(&value) == 2)
        });
    }
    assert!(destroyed.lock().unwrap().is_empty());
    headroom.store(EVICTION_FLOOR_BYTES - 1, Ordering::SeqCst);
    assert_eq!(
        cache.evict_inactive_if_pressure(|_| false),
        ["large", "small"]
    );
    assert_eq!(*destroyed.lock().unwrap(), ["large", "small"]);
    assert_eq!(cache.snapshot(), ModelCacheSnapshot::default());
}

/// iOS WorkPolicy.standard gives standalone opaque model loads ten minutes; kill: the legacy120s cache wait.
#[test]
fn standalone_load_wait_uses_the_standard_ten_minute_opaque_allowance() {
    assert_eq!(LOAD_WAIT_TIMEOUT, Duration::from_secs(600));
}

/// Decode raw events; expected owner/phase counts stay literal at each consumer assertion.
fn live_activity(events: &Mutex<Vec<ferment_engine_core::WorkEvent>>) -> (usize, usize) {
    use ferment_engine_core::WorkEvent;
    let mut live = std::collections::BTreeMap::new();
    for event in events.lock().unwrap().iter() {
        match *event {
            WorkEvent::Started { id, phase } => assert!(live.insert(id, phase).is_none()),
            WorkEvent::Finished { id } => assert!(live.remove(&id).is_some()),
            WorkEvent::Checkpoint => {}
        }
    }
    (
        live.values().filter(|phase| phase.is_none()).count(),
        live.values().filter(|phase| phase.is_some()).count(),
    )
}

/// Kill: let the local wall timer preempt a supervised waiter. Real standalone expiry acknowledges the monitor turn without a settling sleep.
#[test]
fn a_scoped_waiter_defers_the_local_timer_while_standalone_waiters_still_expire() {
    let cache = Cache::with_policy(Duration::ZERO, || u64::MAX);
    let work = ferment_engine_core::WorkActivity::from_callback(|_| {});
    let (release, blocked) = mpsc::channel();
    let scoped = {
        let _scope = work.enter();
        cache.load("one".into(), "v1".into(), 1, move || {
            blocked.recv().unwrap();
            Ok(31)
        })
    };
    let standalone = cache.load("one".into(), "v1".into(), 1, || panic!("must join"));
    let local_error = block_on(standalone).unwrap_err();
    let remaining_waiters = cache.snapshot().waiters;
    release.send(()).unwrap();
    let result = block_on(scoped);
    assert!(matches!(local_error, EngineError::InvalidConfiguration(_)));
    assert_eq!(
        remaining_waiters, 1,
        "only the unscoped local waiter may expire"
    );
    assert_eq!(*result.unwrap(), 31);
}

/// Kill: tie physical ownership to the detachable waiter, or keep its logical phase until native completion.
#[test]
fn a_cancelled_model_waiter_ends_its_phase_but_keeps_native_work_owned_until_completion() {
    use ferment_engine_core::{WorkActivity, WorkEvent};
    let cache = cache();
    let registry = Arc::downgrade(&cache.inner);
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&events);
    let callbacks_unlocked = Arc::new(AtomicBool::new(true));
    let unlocked = Arc::clone(&callbacks_unlocked);
    let (notice, delivered) = mpsc::channel();
    let work = WorkActivity::from_callback(move |event| {
        if let Some(registry) = registry.upgrade()
            && registry.entries.try_lock().is_err()
        {
            unlocked.store(false, Ordering::SeqCst);
        }
        recorded.lock().unwrap().push(event);
        if matches!(event, WorkEvent::Finished { .. }) {
            let _ = notice.send(());
        }
    });
    let (release, blocked) = mpsc::channel();
    let waiter = {
        let _scope = work.enter();
        cache.load("one".into(), "v1".into(), 1, move || {
            blocked.recv().unwrap();
            Ok(47)
        })
    };
    let before_drop = live_activity(&events);
    drop(waiter);
    let after_drop = live_activity(&events);
    let after_drop_waiters = cache.snapshot().waiters;
    let survivor = cache.load("one".into(), "v1".into(), 1, || {
        panic!("must join physical load")
    });
    release.send(()).unwrap();
    let result = block_on(survivor).unwrap();
    while live_activity(&events).0 != 0 {
        delivered.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    assert_eq!(*result, 47);
    assert_eq!(before_drop, (1, 1));
    assert_eq!(after_drop, (1, 0));
    assert_eq!(after_drop_waiters, 0);
    assert_eq!(live_activity(&events), (0, 0));
    assert!(
        callbacks_unlocked.load(Ordering::SeqCst),
        "foreign callbacks must run outside the registry lock"
    );
}

/// Kill: serve a cached graph or evict/admit another graph before checking cancelled-scope admission.
#[test]
fn an_already_cancelled_scope_cannot_hit_evict_or_start_a_native_load() {
    let cache = cache();
    let held = block_on(request(&cache, "hot", 7)).unwrap();
    let before = cache.snapshot();
    let cancelled = Arc::new(AtomicBool::new(true));
    let work =
        ferment_engine_core::WorkActivity::from_callback(|_| {}).with_cancellation(cancelled);
    let calls = Arc::new(AtomicUsize::new(0));
    let hot = {
        let _scope = work.enter();
        request(&cache, "hot", 99)
    };
    let counted = Arc::clone(&calls);
    let cold = {
        let _scope = work.enter();
        cache.load("cold".into(), "v1".into(), 1, move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(13)
        })
    };
    let hot = block_on(hot);
    let cold = block_on(cold);
    assert!(matches!(hot, Err(EngineError::Cancelled)));
    assert!(matches!(cold, Err(EngineError::Cancelled)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(cache.snapshot(), before);
    assert!(Arc::ptr_eq(&held, &cache.peek(&"hot".into()).unwrap()));
}
