//! Seekable, read-only libav input using ffmpeg-next's owned StreamIo.
//! Only format setup is unsafe here: secondary URL opens must be disabled
//! BEFORE probing, which the upstream convenience constructor cannot do.

use std::ffi::{c_char, c_int};
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Mutex};

use ffmpeg_next::{Dictionary, ffi, format::context::StreamIo};

use crate::{MediaInput, Result, VideoError};

const BUFFER_BYTES: usize = 32 * 1024;

pub(super) struct Input {
    inner: ffmpeg_next::format::context::Input,
    source: Origin,
}

#[derive(Clone)]
enum Origin {
    Path(PathBuf),
    Stream(Source),
}

impl Origin {
    fn open(self) -> Result<Input> {
        match self {
            Self::Path(path) => Input::path(&path),
            Self::Stream(source) => Input::open(source),
        }
    }
}

impl Input {
    pub(super) fn path(path: &Path) -> Result<Self> {
        super::init()?;
        Ok(Self {
            inner: ffmpeg_next::format::input(path)
                .map_err(|error| VideoError::Demux(format!("{}: {error}", path.display())))?,
            source: Origin::Path(path.to_owned()),
        })
    }

    pub(super) fn stream(source: Box<dyn MediaInput>) -> Result<Self> {
        let len = source.byte_len();
        if len == 0 || len > i64::MAX as u64 {
            return Err(VideoError::InvalidData(
                "media input size is out of bounds".into(),
            ));
        }
        Self::open(Source(Arc::new(Mutex::new(State {
            source,
            failure: None,
        }))))
    }

    fn open(mut source: Source) -> Result<Self> {
        super::init()?;
        source.check()?;
        if source.seek(SeekFrom::Start(0))? != 0 {
            return Err(VideoError::InvalidData("media input did not rewind".into()));
        }
        let mut custom = StreamIo::from_read_seek_with_capacity(source.clone(), BUFFER_BYTES)
            .map_err(|error| VideoError::Demux(format!("media IO: {error}")))?;
        // SAFETY: allocate a fresh context. StreamIo owns its AVIO and callback
        // state and outlives format close, including every failure path below.
        let mut raw = unsafe { ffi::avformat_alloc_context() };
        if raw.is_null() {
            return Err(VideoError::BackendUnavailable(
                "allocating media input".into(),
            ));
        }
        // SAFETY: not yet opened. A media container must not turn its contents
        // into filesystem/network authority (playlists, reference movies, etc).
        unsafe {
            (*raw).pb = custom.as_mut_ptr();
            (*raw).flags |= ffi::AVFMT_FLAG_CUSTOM_IO;
            (*raw).io_open = Some(deny_open);
            (*raw).probesize = 4 * 1024 * 1024;
            (*raw).max_analyze_duration = 5_000_000;
        }
        let mut options = Dictionary::new();
        options.set("protocol_whitelist", "");
        options.set(
            "format_whitelist",
            "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,nut,mp3,wav,flac,ogg,aac,aiff",
        );
        options.set("max_streams", "32");
        // SAFETY: open owns raw on success and frees/nulls it on failure.
        // CUSTOM_IO leaves pb caller-owned. Reclaim unconsumed options.
        let status = unsafe {
            let mut opts = options.disown();
            let status = ffi::avformat_open_input(&mut raw, ptr::null(), ptr::null(), &mut opts);
            let _remaining = Dictionary::own(opts);
            status
        };
        if status < 0 {
            // SAFETY: null is accepted; handles any context left by failure.
            unsafe { ffi::avformat_close_input(&mut raw) };
            source.check()?;
            return Err(VideoError::Demux(format!(
                "media open: {}",
                ffmpeg_next::Error::from(status)
            )));
        }
        // SAFETY: successful open returned an owned context. ffmpeg-next closes
        // the format before dropping custom IO, including its current buffer.
        let mut input = Self {
            inner: unsafe { ffmpeg_next::format::context::Input::wrap_with_custom_io(raw, custom) },
            source: Origin::Stream(source),
        };
        // SAFETY: live input with no per-stream option dictionaries.
        let status = unsafe { ffi::avformat_find_stream_info(input.as_mut_ptr(), ptr::null_mut()) };
        input.check()?;
        super::check_av("avformat_find_stream_info(stream)", status)?;
        Ok(input)
    }

    pub(super) fn check(&mut self) -> Result<()> {
        if let Origin::Stream(source) = &mut self.source {
            source.check()?;
        }
        Ok(())
    }

    /// Probe can seek through priming/padding. Recreate the demuxer over the
    /// SAME authorized stream; don't reopen a path or guess a seek origin.
    pub(super) fn reopen(self) -> Result<Self> {
        let Self { inner, source } = self;
        drop(inner);
        source.open()
    }

    /// Retain only the original input authority for a native setup fallback.
    /// Invoke after dropping the old demuxer; checked inputs share one cursor.
    #[cfg(target_os = "macos")]
    pub(super) fn reopener(&self) -> impl FnOnce() -> Result<Self> + use<> {
        let source = self.source.clone();
        move || source.open()
    }
}

impl Deref for Input {
    type Target = ffmpeg_next::format::context::Input;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Input {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// Shared only between this worker's demuxer callback and pre/post-frame fence.
// No I/O under a UI/owner lock; callbacks never run concurrently or recursively.
#[derive(Clone)]
struct Source(Arc<Mutex<State>>);

struct State {
    source: Box<dyn MediaInput>,
    failure: Option<(io::ErrorKind, String)>,
}

impl Source {
    fn run<T>(&mut self, op: impl FnOnce(&mut dyn MediaInput) -> io::Result<T>) -> io::Result<T> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| io::Error::other("media input poisoned"))?;
        if let Some((kind, message)) = &state.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        // Preserve the original error even if libav swallows it while probing
        // or has already buffered a packet. Never unwind through its C callbacks.
        let result = catch_unwind(AssertUnwindSafe(|| {
            state.source.validate()?;
            let result = op(state.source.as_mut())?;
            state.source.validate()?;
            Ok(result)
        }))
        .unwrap_or_else(|_| Err(io::Error::other("media input callback panicked")));
        if let Err(error) = &result {
            state.failure = Some((error.kind(), error.to_string()));
        }
        result
    }

    fn check(&mut self) -> io::Result<()> {
        self.run(|_| Ok(()))
    }
}

impl Read for Source {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let size = buffer.len().min(BUFFER_BYTES);
        self.run(|source| {
            let count = source.read(&mut buffer[..size])?;
            if count > size {
                return Err(io::Error::other("invalid media read length"));
            }
            Ok(count)
        })
    }
}

impl Seek for Source {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.run(|source| {
            let position = source.seek(from)?;
            if position > i64::MAX as u64 {
                return Err(io::Error::other("media seek overflow"));
            }
            Ok(position)
        })
    }
}

unsafe extern "C" fn deny_open(
    _: *mut ffi::AVFormatContext,
    _: *mut *mut ffi::AVIOContext,
    _: *const c_char,
    _: c_int,
    _: *mut *mut ffi::AVDictionary,
) -> c_int {
    ffi::AVERROR_EXTERNAL
}
