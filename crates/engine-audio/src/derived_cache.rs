//! Optional, content-addressed derived files. The filesystem is the index.
//!
//! This is the current `FermentUtils/DerivedFileCache.swift` contract. A
//! producer owns its partial until publication; callers own neither a cached
//! file nor the producer. Independent callers share one production. The
//! synchronous admission is used on the engine's blocking/DSP workers, never
//! the Android main thread, and does not inherit a consumer cancellation token.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, FileTimes};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use ferment_engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivedSource {
    pub id: String,
}

impl DerivedSource {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// Size and the head/middle/tail 64 KiB windows, independent of path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let window = 64 * 1024_u64;
        let offsets = BTreeSet::from([
            0,
            (size / 2).saturating_sub(window / 2),
            size.saturating_sub(window),
        ]);
        let mut digest = Sha256::new();
        let mut bytes = vec![0; window as usize];
        for offset in offsets {
            file.seek(SeekFrom::Start(offset))?;
            let count = usize::try_from((size - offset).min(window)).unwrap();
            file.read_exact(&mut bytes[..count])?;
            digest.update(&bytes[..count]);
        }
        Ok(Self::new(format!("{size}-{:x}", digest.finalize())))
    }
}

pub trait DerivedFileKey: Serialize {
    const KIND: &'static str;
    const VERSION: u32;
    fn source(&self) -> &DerivedSource;
    fn extension(&self) -> &str;
}

/// Sorted keys, literal UTF-8 and slashes, whole-valued numbers without `.0`.
pub fn canonical_json<T: Serialize>(key: &T) -> Result<Vec<u8>> {
    // Serialize first so f32 values retain their own shortest representation.
    let encoded =
        serde_json::to_vec(key).map_err(|e| EngineError::invalid_config(e.to_string()))?;
    let mut value: Value =
        serde_json::from_slice(&encoded).map_err(|e| EngineError::invalid_config(e.to_string()))?;
    fn canonical(value: &mut Value) {
        match value {
            Value::Array(values) => values.iter_mut().for_each(canonical),
            Value::Object(values) => values.values_mut().for_each(canonical),
            Value::Number(number) if number.is_f64() => {
                let n = number.as_f64().unwrap();
                if n.fract() == 0.0 && n >= i64::MIN as f64 && n < i64::MAX as f64 {
                    *number = (n as i64).into();
                }
            }
            _ => {}
        }
    }
    canonical(&mut value);
    serde_json::to_vec(&value).map_err(|e| EngineError::invalid_config(e.to_string()))
}

pub fn request_hash<K: DerivedFileKey>(key: &K) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(K::KIND);
    hash.update([0]);
    hash.update(K::VERSION.to_string());
    hash.update([0]);
    hash.update(canonical_json(key)?);
    Ok(format!("{:x}", hash.finalize()))
}

fn source_hash(source: &DerivedSource) -> String {
    format!("{:x}", Sha256::digest(source.id.as_bytes()))
}

#[derive(Debug)]
struct InFlight {
    result: Mutex<Option<std::result::Result<PathBuf, Arc<EngineError>>>>,
    ready: Condvar,
}

#[derive(Debug, Default)]
struct State {
    total_bytes: u64,
    in_flight: BTreeMap<String, Arc<InFlight>>,
}

#[derive(Debug)]
pub struct DerivedFileCache {
    pub directory: PathBuf,
    pub budget_bytes: u64,
    state: Mutex<State>,
}

impl DerivedFileCache {
    /// Sweep partial/scratch files left by a crash and enforce today's budget.
    pub fn new(directory: PathBuf, budget_bytes: u64) -> Result<Self> {
        fs::create_dir_all(&directory)?;
        let cache = Self {
            directory,
            budget_bytes,
            state: Mutex::new(State::default()),
        };
        let scratch = cache.directory.join("scratch");
        remove_if_present(&scratch)?;
        for path in walk(&cache.directory)? {
            if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(".tmp-"))
            {
                remove_if_present(&path)?;
            }
        }
        let mut state = cache.state.lock().unwrap();
        cache.trim(&mut state, 0)?;
        drop(state);
        Ok(cache)
    }

    pub fn total_bytes(&self) -> u64 {
        self.state.lock().unwrap().total_bytes
    }

    pub fn fetch<K: DerivedFileKey>(
        &self,
        key: &K,
        produce: impl FnOnce(&Path) -> Result<()> + Send,
    ) -> Result<PathBuf> {
        let extension = key.extension();
        if extension.is_empty()
            || !extension
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-')
            || extension == "json"
        {
            return Err(EngineError::invalid_config(
                "derived file extension must be a simple non-sidecar extension",
            ));
        }
        let hash = request_hash(key)?;
        let directory = self.directory.join(source_hash(key.source()));
        let final_path = directory.join(format!("{hash}.{extension}"));
        let mut state = self.state.lock().unwrap();
        if final_path.is_file() {
            // A file's times are written through a handle opened for writing.
            if let Err(error) = File::options()
                .write(true)
                .open(&final_path)
                .and_then(|f| f.set_times(FileTimes::new().set_modified(SystemTime::now())))
            {
                tracing::warn!(
                    file = %final_path.file_name().unwrap_or_default().to_string_lossy(),
                    %error,
                    "derived cache LRU bump failed"
                );
            }
            return Ok(final_path);
        }
        if let Some(running) = state.in_flight.get(&hash).cloned() {
            drop(state);
            let mut result = running.result.lock().unwrap();
            while result.is_none() {
                result = running.ready.wait(result).unwrap();
            }
            return result
                .as_ref()
                .unwrap()
                .clone()
                .map_err(|error| copy_error(&error));
        }
        let running = Arc::new(InFlight {
            result: Mutex::new(None),
            ready: Condvar::new(),
        });
        state.in_flight.insert(hash.clone(), running.clone());
        drop(state);
        // With unwinding enabled, an unexpected producer panic also wakes joiners.
        // Release builds retain the workspace's abort-on-panic policy.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<PathBuf> {
                fs::create_dir_all(&directory)?;
                let partial = directory.join(format!(".tmp-{}", nonce()));
                let owned = PartialFile(partial.clone());
                // A separate worker also permits a decoder to drive its async
                // media adapter while the FFI caller's executor is already active.
                // Scoped ownership joins the writer before any partial is removed.
                std::thread::scope(|scope| scope.spawn(|| produce(&partial)).join())
                    .map_err(|_| EngineError::internal("derived file producer panicked"))??;
                let size = match fs::metadata(&partial) {
                    Ok(metadata) => metadata.len(),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
                    Err(error) => return Err(error.into()),
                };
                if size == 0 {
                    return Err(EngineError::invalid_config(
                        "derived producer wrote nothing",
                    ));
                }
                let mut state = self.state.lock().unwrap();
                if size > self.budget_bytes {
                    let scratch = self.directory.join("scratch");
                    fs::create_dir_all(&scratch)?;
                    let destination = scratch.join(format!("{}.{extension}", nonce()));
                    fs::rename(&partial, &destination)?;
                    return Ok(destination);
                }
                self.trim(&mut state, size)?;
                let value: Value = serde_json::from_slice(&canonical_json(key)?)
                    .map_err(|e| EngineError::invalid_config(e.to_string()))?;
                let sidecar = serde_json::to_vec_pretty(
                    &serde_json::json!({"kind": K::KIND, "version": K::VERSION, "key": value}),
                )
                .map_err(|e| EngineError::invalid_config(e.to_string()))?;
                let sidecar_partial = directory.join(format!(".tmp-{}", nonce()));
                let sidecar_owned = PartialFile(sidecar_partial.clone());
                fs::write(&sidecar_partial, sidecar)?;
                fs::rename(&sidecar_partial, final_path.with_extension("json"))?;
                drop(sidecar_owned);
                if !final_path.exists() {
                    fs::rename(&partial, &final_path)?;
                    state.total_bytes += size;
                }
                drop(owned);
                Ok(final_path)
            }))
            .unwrap_or_else(|_| Err(EngineError::internal("derived file producer panicked")));
        let mut state = self.state.lock().unwrap();
        *running.result.lock().unwrap() = Some(
            result
                .as_ref()
                .map(Clone::clone)
                .map_err(|error| Arc::new(copy_error(error))),
        );
        state.in_flight.remove(&hash);
        running.ready.notify_all();
        result
    }

    pub fn remove_source(&self, source: &DerivedSource) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let path = self.directory.join(source_hash(source));
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
        state.total_bytes = self.artefacts()?.iter().map(|a| a.size).sum();
        Ok(())
    }

    pub fn remove_all(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if self.directory.exists() {
            fs::remove_dir_all(&self.directory)?;
        }
        fs::create_dir_all(&self.directory)?;
        state.total_bytes = 0;
        Ok(())
    }

    fn artefacts(&self) -> Result<Vec<Artefact>> {
        let mut result = Vec::new();
        for path in walk(&self.directory)? {
            if path
                .strip_prefix(&self.directory)
                .unwrap()
                .components()
                .any(|part| {
                    let name = part.as_os_str().to_string_lossy();
                    name.starts_with('.') || name == "scratch"
                })
                || path.extension().is_some_and(|x| x == "json")
            {
                continue;
            }
            let metadata = fs::metadata(&path)?;
            result.push(Artefact {
                path,
                size: metadata.len(),
                modified: metadata.modified().unwrap_or(UNIX_EPOCH),
            });
        }
        Ok(result)
    }

    fn trim(&self, state: &mut State, incoming: u64) -> Result<()> {
        let mut artefacts = self.artefacts()?;
        // A system cache purge is ordinary; recount instead of trusting a stale total.
        state.total_bytes = artefacts.iter().map(|a| a.size).sum();
        artefacts.sort_by_key(|a| a.modified);
        for artefact in artefacts {
            if state.total_bytes.saturating_add(incoming) <= self.budget_bytes {
                break;
            }
            fs::remove_file(&artefact.path)?;
            remove_if_present(&artefact.path.with_extension("json"))?;
            state.total_bytes = state.total_bytes.saturating_sub(artefact.size);
            if let Some(parent) = artefact.path.parent()
                && fs::read_dir(parent)?.next().transpose()?.is_none()
            {
                fs::remove_dir(parent)?;
            }
        }
        Ok(())
    }
}

/// An already-gone cache entry is ordinary; permission and other I/O errors
/// remain the caller's failure, matching FileManager.removeIfPresent.
fn remove_if_present(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let result = if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

struct Artefact {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}
struct PartialFile(PathBuf);
impl Drop for PartialFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn walk(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    if !directory.exists() {
        return Ok(result);
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            result.extend(walk(&entry.path())?);
        } else if kind.is_file() {
            result.push(entry.path());
        }
    }
    Ok(result)
}

pub fn nonce() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

// Every joined caller receives the producer's error kind, including cancellation.
// io::Error is not Clone; reconstruct it while preserving OS code or kind/text.
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
mod concurrency_tests {
    use super::*;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};
    #[derive(Serialize)]
    struct Key {
        source: DerivedSource,
    }
    impl DerivedFileKey for Key {
        const KIND: &'static str = "joined-error";
        const VERSION: u32 = 1;
        fn source(&self) -> &DerivedSource {
            &self.source
        }
        fn extension(&self) -> &str {
            "pcm"
        }
    }
    fn key() -> Key {
        Key {
            source: DerivedSource::new("one"),
        }
    }

    #[test]
    fn joined_caller_receives_the_original_cancellation_kind_and_retry_can_win() {
        let directory = std::env::temp_dir().join(nonce());
        let cache = Arc::new(DerivedFileCache::new(directory.clone(), 100).unwrap());
        let entered = Arc::new(Barrier::new(2));
        let finish = Arc::new(Barrier::new(2));
        let first = {
            let (cache, entered, finish) = (cache.clone(), entered.clone(), finish.clone());
            std::thread::spawn(move || {
                cache.fetch(&key(), |path| {
                    fs::write(path, b"partial")?;
                    entered.wait();
                    finish.wait();
                    Err(EngineError::Cancelled)
                })
            })
        };
        entered.wait();
        let second = {
            let cache = cache.clone();
            std::thread::spawn(move || cache.fetch(&key(), |_| panic!("joiner cannot produce")))
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let joined = cache
                .state
                .lock()
                .unwrap()
                .in_flight
                .values()
                .any(|running| Arc::strong_count(running) >= 3);
            if joined {
                break;
            }
            if Instant::now() > deadline {
                finish.wait();
                panic!("second caller never joined production");
            }
            std::thread::yield_now();
        }
        finish.wait();
        assert!(matches!(first.join().unwrap(), Err(EngineError::Cancelled)));
        assert!(matches!(
            second.join().unwrap(),
            Err(EngineError::Cancelled)
        ));
        assert_eq!(cache.total_bytes(), 0);
        let path = cache
            .fetch(&key(), |p| Ok(fs::write(p, b"complete")?))
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), b"complete");
        fs::remove_dir_all(directory).unwrap();
    }
}
