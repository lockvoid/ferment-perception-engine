//! Preserve artifact identity when filesystem inspection fails before native load.
use std::{io, path::Path};

use ferment_engine_core::{EngineError, Result};

pub(crate) fn at_path<T>(path: &Path, result: io::Result<T>) -> Result<T> {
    result.map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            EngineError::ModelNotFound(path.display().to_string())
        } else {
            EngineError::Io(io::Error::new(
                error.kind(),
                format!("{}: {error}", path.display()),
            ))
        }
    })
}
