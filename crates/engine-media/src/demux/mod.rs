//! Container description.
//!
//! What is left of a module that used to hold two vendored demuxers and an
//! mp3 parser. `libavformat` reads every container now, so this is the one
//! function the rest of the workspace ever called — kept at its own name and
//! path so no consumer had to change.

use std::path::Path;

use crate::{MediaInfo, Result};

/// Describe a media file.
pub fn probe(path: impl AsRef<Path>) -> Result<MediaInfo> {
    crate::platform::libav::probe(path.as_ref())
}
