//! Current DerivedFileCacheTests.swift, plus deterministic invalidation/error
//! races. Fixture bytes are literal and independently asserted; no model runs.
use ferment_engine_audio::derived_cache::{
    DerivedFileCache, DerivedFileKey, DerivedSource, canonical_json, nonce, request_hash,
};
use ferment_engine_core::{EngineError, Result};
use serde::Serialize;
use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, UNIX_EPOCH};

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("derived-cache-{}", nonce())))
    }
    fn cache(&self, budget: u64) -> DerivedFileCache {
        DerivedFileCache::new(self.0.clone(), budget).unwrap()
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[derive(Serialize)]
struct Key {
    source: DerivedSource,
    rate: u32,
}
impl DerivedFileKey for Key {
    const KIND: &'static str = "test-pcm";
    const VERSION: u32 = 3;
    fn source(&self) -> &DerivedSource {
        &self.source
    }
    fn extension(&self) -> &str {
        "pcm"
    }
}
fn key(source: &str, rate: u32) -> Key {
    Key {
        source: DerivedSource::new(source),
        rate,
    }
}
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    Ok(fs::write(path, bytes)?)
}
fn files(root: &Path) -> Vec<PathBuf> {
    if !root.exists() {
        return Vec::new();
    }
    fs::read_dir(root)
        .unwrap()
        .flat_map(|e| {
            let p = e.unwrap().path();
            if p.is_dir() { files(&p) } else { vec![p] }
        })
        .collect()
}
fn old(path: &Path, seconds: u64) {
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(seconds)))
        .unwrap();
}

#[test]
fn canonical_hash_is_stable_across_field_order_and_instances() {
    #[derive(Serialize)]
    struct Other {
        rate: u32,
        source: DerivedSource,
    }
    impl DerivedFileKey for Other {
        const KIND: &'static str = "test-pcm";
        const VERSION: u32 = 3;
        fn source(&self) -> &DerivedSource {
            &self.source
        }
        fn extension(&self) -> &str {
            "pcm"
        }
    }
    assert_eq!(
        canonical_json(&key("blob/東京", 22050)).unwrap(),
        "{\"rate\":22050,\"source\":{\"id\":\"blob/東京\"}}".as_bytes()
    );
    assert_eq!(
        request_hash(&key("same", 22050)).unwrap(),
        request_hash(&Other {
            rate: 22050,
            source: DerivedSource::new("same")
        })
        .unwrap()
    );
}
#[test]
fn hash_separates_kind_version_and_parameters() {
    #[derive(Serialize)]
    struct V1(Key);
    impl DerivedFileKey for V1 {
        const KIND: &'static str = "test-pcm";
        const VERSION: u32 = 1;
        fn source(&self) -> &DerivedSource {
            &self.0.source
        }
        fn extension(&self) -> &str {
            "pcm"
        }
    }
    #[derive(Serialize)]
    struct Other(Key);
    impl DerivedFileKey for Other {
        const KIND: &'static str = "test-mel";
        const VERSION: u32 = 3;
        fn source(&self) -> &DerivedSource {
            &self.0.source
        }
        fn extension(&self) -> &str {
            "mel"
        }
    }
    let a = request_hash(&key("x", 22050)).unwrap();
    for other in [
        request_hash(&V1(key("x", 22050))).unwrap(),
        request_hash(&Other(key("x", 22050))).unwrap(),
        request_hash(&key("x", 16000)).unwrap(),
    ] {
        assert_ne!(a, other);
    }
}
#[test]
fn source_identity_is_content_not_path_and_observes_three_windows() {
    let root = Root::new();
    fs::create_dir_all(&root.0).unwrap();
    let a = root.0.join("a");
    let b = root.0.join("b");
    let data = vec![0x35_u8; 400_000];
    fs::write(&a, &data).unwrap();
    fs::copy(&a, &b).unwrap();
    let identity = DerivedSource::from_file(&a).unwrap();
    assert_eq!(identity, DerivedSource::from_file(&b).unwrap());
    for offset in [1, 200_000, 399_999] {
        let mut changed = data.clone();
        changed[offset] ^= 1;
        fs::write(&b, changed).unwrap();
        assert_ne!(identity, DerivedSource::from_file(&b).unwrap());
    }
    fs::write(&b, &data[..data.len() - 1]).unwrap();
    assert_ne!(identity, DerivedSource::from_file(&b).unwrap());
}
#[test]
fn miss_publishes_complete_bytes_and_sidecar() {
    let root = Root::new();
    let c = root.cache(100);
    let p = c.fetch(&key("a", 1), |p| write(p, b"complete")).unwrap();
    assert_eq!(fs::read(&p).unwrap(), b"complete");
    assert!(p.with_extension("json").exists());
    assert_eq!(c.total_bytes(), 8);
    assert!(files(&root.0).iter().all(|p| {
        !p.file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(".tmp-")
    }));
}
#[test]
fn hit_touches_without_producing() {
    let root = Root::new();
    let c = root.cache(100);
    let p = c.fetch(&key("a", 1), |p| write(p, b"abc")).unwrap();
    old(&p, 1);
    let hit = c.fetch(&key("a", 1), |_| panic!("cache miss")).unwrap();
    assert_eq!(hit, p);
    assert!(p.metadata().unwrap().modified().unwrap() > UNIX_EPOCH + Duration::from_secs(1));
}
#[test]
fn relaunch_is_a_hit() {
    let root = Root::new();
    let p = root
        .cache(100)
        .fetch(&key("a", 1), |p| write(p, b"abc"))
        .unwrap();
    let c = root.cache(100);
    assert_eq!(c.total_bytes(), 3);
    assert_eq!(
        c.fetch(&key("a", 1), |_| panic!("relaunch miss")).unwrap(),
        p
    );
}
#[test]
fn concurrent_askers_share_one_producer() {
    let root = Root::new();
    let c = Arc::new(root.cache(100));
    let calls = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let (c, calls, gate) = (c.clone(), calls.clone(), gate.clone());
            std::thread::spawn(move || {
                gate.wait();
                c.fetch(&key("a", 1), |p| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    write(p, b"one")
                })
                .unwrap()
            })
        })
        .collect();
    let paths: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert!(paths.iter().all(|p| p == &paths[0]));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[test]
fn failed_producer_cleans_up_and_next_ask_retries() {
    let root = Root::new();
    let c = root.cache(100);
    assert!(matches!(
        c.fetch(&key("a", 1), |p| {
            write(p, b"partial")?;
            Err(EngineError::Cancelled)
        }),
        Err(EngineError::Cancelled)
    ));
    assert!(files(&root.0).is_empty());
    assert_eq!(c.total_bytes(), 0);
    let p = c.fetch(&key("a", 1), |p| write(p, b"retry")).unwrap();
    assert_eq!(fs::read(p).unwrap(), b"retry");
}
#[test]
fn empty_producer_is_an_error() {
    let root = Root::new();
    let c = root.cache(100);
    assert!(c.fetch(&key("a", 1), |_| Ok(())).is_err());
    assert!(c.fetch(&key("a", 1), |p| write(p, b"")).is_err());
    assert_eq!(c.total_bytes(), 0);
    assert!(files(&root.0).is_empty());
}
#[test]
fn lru_evicts_oldest_and_its_sidecar() {
    let root = Root::new();
    let c = root.cache(8);
    let a = c.fetch(&key("a", 1), |p| write(p, b"aaaa")).unwrap();
    let b = c.fetch(&key("b", 1), |p| write(p, b"bbbb")).unwrap();
    old(&a, 1);
    old(&b, 2);
    c.fetch(&key("c", 1), |p| write(p, b"cccc")).unwrap();
    assert!(!a.exists());
    assert!(!a.with_extension("json").exists());
    assert!(b.exists());
    assert_eq!(c.total_bytes(), 8);
}
#[test]
fn oversized_output_is_served_without_evicting_retained_files() {
    let root = Root::new();
    let c = root.cache(4);
    let a = c.fetch(&key("a", 1), |p| write(p, b"abcd")).unwrap();
    let b = c.fetch(&key("b", 1), |p| write(p, b"12345")).unwrap();
    assert!(a.exists());
    assert_eq!(b.parent().unwrap(), root.0.join("scratch"));
    assert_eq!(fs::read(&b).unwrap(), b"12345");
    assert_eq!(c.total_bytes(), 4);
    assert_ne!(b, c.fetch(&key("b", 1), |p| write(p, b"12345")).unwrap());
}
#[test]
fn init_sweeps_partial_and_scratch_recounts_and_trims() {
    let root = Root::new();
    let c = root.cache(100);
    let a = c.fetch(&key("a", 1), |p| write(p, b"aaaa")).unwrap();
    let b = c.fetch(&key("b", 1), |p| write(p, b"bbbb")).unwrap();
    old(&a, 1);
    old(&b, 2);
    fs::write(root.0.join(".tmp-crash"), b"junk").unwrap();
    fs::create_dir_all(root.0.join("scratch")).unwrap();
    fs::write(root.0.join("scratch/junk.pcm"), b"x").unwrap();
    drop(c);
    let c = root.cache(4);
    assert_eq!(c.total_bytes(), 4);
    assert!(!a.exists());
    assert!(b.exists());
    assert!(!root.0.join("scratch").exists());
    assert!(!root.0.join(".tmp-crash").exists());
}
#[test]
fn remove_source_only_removes_its_artifacts_and_remove_all_resets() {
    let root = Root::new();
    let c = root.cache(100);
    let a = c.fetch(&key("a", 1), |p| write(p, b"aa")).unwrap();
    let b = c.fetch(&key("b", 1), |p| write(p, b"bb")).unwrap();
    c.remove_source(&DerivedSource::new("a")).unwrap();
    assert!(!a.exists());
    assert!(b.exists());
    assert_eq!(c.total_bytes(), 2);
    c.remove_all().unwrap();
    assert_eq!(c.total_bytes(), 0);
    assert!(files(&root.0).is_empty());
}
#[test]
fn sidecar_contains_exact_kind_version_and_canonical_request() {
    let root = Root::new();
    let c = root.cache(100);
    let p = c.fetch(&key("x/y", 22050), |p| write(p, b"f32")).unwrap();
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(p.with_extension("json")).unwrap()).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"kind":"test-pcm","version":3,"key":{"source":{"id":"x/y"},"rate":22050}})
    );
}
#[test]
fn invalidation_during_production_cannot_republish_deleted_partial() {
    let root = Root::new();
    let c = Arc::new(root.cache(100));
    let entered = Arc::new(Barrier::new(2));
    let finish = Arc::new(Barrier::new(2));
    let worker = {
        let (c, e, f) = (c.clone(), entered.clone(), finish.clone());
        std::thread::spawn(move || {
            c.fetch(&key("a", 1), |p| {
                write(p, b"partial")?;
                e.wait();
                f.wait();
                Ok(())
            })
        })
    };
    entered.wait();
    c.remove_source(&DerivedSource::new("a")).unwrap();
    finish.wait();
    assert!(worker.join().unwrap().is_err());
    assert_eq!(c.total_bytes(), 0);
    assert!(files(&root.0).is_empty());
}
