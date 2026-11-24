//! Worker-only, seekable media input without a pathname or raw file descriptor.

use std::io::{self, Read, Seek};

/// An already authorized, finite byte stream. The owner opens and verifies it
/// before handing it to a decoder. No implementation may treat a resource name
/// or document field alone as permission to read a file.
///
/// All calls (including `validate`) may block and must stay on a media worker,
/// never the UI/audio callback. `byte_len` is a stable bound, not a seek request.
/// `validate` checks cancellation, revocation and source identity even when a
/// decoder can satisfy its next frame entirely from its own buffered packets.
/// Failure permanently retires the libav session; reopening needs a fresh grant.
/// Readers must obey ordinary `Read`/`Seek` contracts and reject changed bytes.
/// Dropping the decoder releases the stream, without creating a media copy.
pub trait MediaInput: Read + Seek + Send {
    fn byte_len(&self) -> u64;
    fn validate(&mut self) -> io::Result<()>;
}
