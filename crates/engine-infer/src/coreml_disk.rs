//! A low-space native Core ML compile can abort on ENOSPC instead of returning
//! an NSError. Refuse known-insufficient scratch capacity before a cold load.
//! This is a preflight, not a reservation against concurrent filesystem writes.
use ferment_engine_core::{EngineError, Result};

const RESERVE: u64 = 1024 * 1024 * 1024;

pub(crate) fn check(bundle_bytes: u64) -> Result<()> {
    // Measure blocks on the actual temporary volume. macOS "important usage"
    // capacity includes purgeable caches and is unsafe for this decision.
    let temporary = std::env::temp_dir();
    let available = crate::model_io::at_path(&temporary, fs4::available_space(&temporary))?;
    admit(bundle_bytes, available)
}

fn admit(bytes: u64, available: u64) -> Result<()> {
    let required = bytes
        .checked_mul(2)
        .and_then(|n| n.checked_add(RESERVE))
        .ok_or_else(|| EngineError::invalid_config("Model preparation disk budget overflow"))?;
    if available < required {
        return Err(EngineError::inference(format!(
            "Not enough disk space to prepare the native model: {:.1} GiB available, at least {:.1} GiB required. Free space and retry; no model or media was deleted.",
            available as f64 / RESERVE as f64,
            required as f64 / RESERVE as f64,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insufficient_compiler_space_is_a_recoverable_error() {
        // Mutation: dropping the reserve/multiplier admits the observed ENOSPC case.
        let error = admit(3 * RESERVE, 2 * RESERVE).unwrap_err().to_string();
        assert!(error.contains("disk space"));
        assert!(error.contains("retry"));
        assert!(admit(3 * RESERVE, 7 * RESERVE - 1).is_err());
        assert!(admit(3 * RESERVE, 7 * RESERVE).is_ok());
        assert!(admit(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn native_space_probe_addresses_the_temporary_volume() {
        let temporary = std::env::temp_dir();
        let capacity = fs4::statvfs(temporary).unwrap();
        assert!(capacity.total_space() > 0);
        assert!(capacity.available_space() <= capacity.total_space());
    }
}
