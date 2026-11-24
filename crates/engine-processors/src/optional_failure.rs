//! Best-effort engine operations retain their fallback, with one report per
//! operation and error type, matching the current Swift engine's Log helper.
use std::any::type_name;
use std::collections::HashSet;
use std::fmt::Display;
use std::io;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

pub(crate) fn report<E: Display>(operation: &'static str, error: &E) {
    static REPORTED: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();
    let first = REPORTED
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((operation, type_name::<E>()));
    if first {
        tracing::error!(operation, error_type = type_name::<E>(), error = %error,
            "optional engine operation failed");
    }
}

pub(crate) fn cleanup<E: Display>(operation: &'static str, result: Result<(), E>) {
    if let Err(error) = result {
        report(operation, &error);
    }
}

/// Absence is the only ignored filesystem refusal, including dangling links.
pub(crate) fn remove_if_present(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
