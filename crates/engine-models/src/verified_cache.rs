//! Process-local verification memo. The digest is still checked on first use
//! and after any filesystem identity/content change. No receipt is trusted.
//! Unix change time catches same-size rewrites even if mtime is restored.
//! Platforms without that change identity keep doing full verification.
use crate::Availability;
use ferment_engine_core::Result;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
pub(crate) struct VerifiedCache {
    entries: Mutex<HashMap<PathBuf, Arc<Mutex<Option<Verified>>>>>,
}
#[derive(Debug)]
struct Verified {
    expected: String,
    stamp: Vec<Stamp>,
}
#[derive(Debug, PartialEq)]
struct Stamp {
    path: PathBuf,
    identity: [u64; 7],
}

impl VerifiedCache {
    pub fn availability(&self, path: &Path, expected: Option<&str>) -> Result<Availability> {
        self.check(path, expected, crate::sha256_path)
    }

    fn check(
        &self,
        path: &Path,
        expected: Option<&str>,
        hash: impl FnOnce(&Path) -> Result<String>,
    ) -> Result<Availability> {
        if !path.exists() {
            return Ok(Availability::Missing);
        }
        let Some(expected) = expected else {
            return Ok(Availability::Unverified);
        };
        let cell = {
            let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
            // The fleet is finite. Bound even callers with arbitrary models;
            // dropping a memo only causes a recheck, never trusts more bytes.
            if entries.len() >= 128 && !entries.contains_key(path) {
                entries.clear();
            }
            entries.entry(path.to_owned()).or_default().clone()
        };
        // Single flight: waiters re-stat, not re-hash a 443 MB encoder.
        let mut held = cell.lock().unwrap_or_else(|p| p.into_inner());
        let before = fingerprint(path)?;
        if let (Some(stamp), Some(verified)) = (&before, held.as_ref())
            && verified.expected == expected
            && &verified.stamp == stamp
        {
            return Ok(Availability::Ready);
        }
        *held = None;
        if hash(path)? != expected {
            return Ok(Availability::Corrupt);
        }
        // Never cache a tree that changed while its digest was being read.
        let after = fingerprint(path)?;
        if before != after {
            return Ok(Availability::Corrupt);
        }
        if let Some(stamp) = after {
            *held = Some(Verified {
                expected: expected.into(),
                stamp,
            });
        }
        Ok(Availability::Ready)
    }
}

#[cfg(unix)]
fn fingerprint(path: &Path) -> Result<Option<Vec<Stamp>>> {
    use std::os::unix::fs::MetadataExt;
    fn walk(path: &Path, out: &mut Vec<Stamp>) -> Result<bool> {
        let m = std::fs::symlink_metadata(path)?;
        // Symlinks and special files are never eligible for memoization.
        if m.file_type().is_symlink() || !(m.is_file() || m.is_dir()) {
            return Ok(false);
        }
        out.push(Stamp {
            path: path.into(),
            identity: [
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime() as u64,
                m.mtime_nsec() as u64,
                m.ctime() as u64,
                m.ctime_nsec() as u64,
            ],
        });
        if m.is_dir() {
            for entry in std::fs::read_dir(path)? {
                if !walk(&entry?.path(), out)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    let mut out = Vec::new();
    if !walk(path, &mut out)? {
        return Ok(None);
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Some(out))
}
#[cfg(not(unix))]
fn fingerprint(_: &Path) -> Result<Option<Vec<Stamp>>> {
    Ok(None)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn verifies_once_but_detects_rewrites_replacement_digest_changes_and_missing_files() {
        let dir = crate::tests::TempRoot::new("verified-cache");
        let path = dir.0.join("model");
        std::fs::create_dir_all(&path).unwrap();
        let weights = path.join("weights");
        std::fs::write(&weights, b"safe").unwrap();
        let digest = crate::sha256_path(&path).unwrap();
        let cache = VerifiedCache::default();
        assert_eq!(
            cache.availability(&path, Some(&digest)).unwrap(),
            Availability::Ready
        );
        assert_eq!(
            cache
                .check(&path, Some(&digest), |_| panic!("hashed an unchanged tree"))
                .unwrap(),
            Availability::Ready
        );
        let old_time = std::fs::metadata(&weights).unwrap().modified().unwrap();
        std::fs::write(&weights, b"evil").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&weights)
            .unwrap()
            .set_modified(old_time)
            .unwrap();
        assert_eq!(
            cache.availability(&path, Some(&digest)).unwrap(),
            Availability::Corrupt
        );
        std::fs::write(&weights, b"safe").unwrap();
        assert_eq!(
            cache.availability(&path, Some(&digest)).unwrap(),
            Availability::Ready
        );
        assert_eq!(
            cache.availability(&path, Some("different digest")).unwrap(),
            Availability::Corrupt
        );
        std::fs::remove_file(&weights).unwrap();
        assert_eq!(
            cache.availability(&weights, Some(&digest)).unwrap(),
            Availability::Missing
        );
        assert_eq!(
            cache.availability(&path, Some(&digest)).unwrap(),
            Availability::Corrupt
        );
    }
}
