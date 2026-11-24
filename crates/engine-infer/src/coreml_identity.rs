//! Metadata, not repeated hashing of model weights, detects bundle replacement.
//! Installation verifies content hashes separately. Visit nested files as an
//! in-place replacement of weights need not touch the root directory's mtime.
use std::fmt::Write;
use std::path::Path;

use ferment_engine_core::{EngineError, Result};

pub(crate) fn bundle_identity(path: &Path) -> Result<(String, u64)> {
    let canonical = crate::model_io::at_path(path, path.canonicalize())?;
    if !canonical.is_dir() {
        return Err(EngineError::invalid_config(
            "Native model must be a compiled bundle",
        ));
    }
    let mut identity = format!("{canonical:?}");
    let mut weight = 0;
    visit(&canonical, &canonical, &mut identity, &mut weight)?;
    Ok((identity, weight))
}

fn visit(root: &Path, path: &Path, identity: &mut String, weight: &mut u64) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let meta = crate::model_io::at_path(path, path.symlink_metadata())?;
    if meta.is_symlink() || (!meta.is_file() && !meta.is_dir()) {
        return Err(EngineError::invalid_config(
            "Native model bundle contains a link or special file",
        ));
    }
    write!(
        identity,
        "|{:?}:{}:{}:{}:{}:{}:{}:{}",
        path.strip_prefix(root).unwrap(),
        meta.dev(),
        meta.ino(),
        meta.len(),
        meta.mtime(),
        meta.mtime_nsec(),
        meta.ctime(),
        meta.ctime_nsec()
    )
    .expect("writing to String cannot fail");
    if meta.is_file() {
        *weight = weight
            .checked_add(meta.len())
            .ok_or_else(|| EngineError::invalid_config("Native model bundle size overflow"))?;
    } else {
        let entries = crate::model_io::at_path(path, std::fs::read_dir(path))?;
        let mut children = crate::model_io::at_path(
            path,
            entries
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>(),
        )?;
        children.sort();
        for child in children {
            visit(root, &child, identity, weight)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            // SystemTime can have coarser resolution than two parallel tests.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "coreml-identity-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn nested_weight_replacement_changes_identity_and_counts_actual_bytes() {
        let scratch = Scratch::new();
        let nested = scratch.0.join("weights");
        std::fs::create_dir(&nested).unwrap();
        let weight = nested.join("weight.bin");
        std::fs::write(&weight, [1, 2, 3, 4]).unwrap();
        let first = bundle_identity(&scratch.0).unwrap();
        assert_eq!(first.1, 4);
        assert_eq!(bundle_identity(&scratch.0).unwrap(), first);
        let replacement = nested.join("replacement.bin");
        std::fs::write(&replacement, [4, 3, 2, 1]).unwrap();
        std::fs::rename(&replacement, &weight).unwrap();
        let replaced = bundle_identity(&scratch.0).unwrap();
        assert_ne!(first.0, replaced.0);
        assert_eq!(first.1, replaced.1);
    }

    #[test]
    fn links_and_non_bundle_files_are_refused() {
        let scratch = Scratch::new();
        let file = scratch.0.join("weight.bin");
        std::fs::write(&file, [0]).unwrap();
        assert!(bundle_identity(&file).is_err());
        std::os::unix::fs::symlink(&file, scratch.0.join("linked.bin")).unwrap();
        assert!(bundle_identity(&scratch.0).is_err());
    }
}
