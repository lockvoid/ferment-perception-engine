//! Display timestamps held by an anonymous disk file, never a duration-sized
//! RAM array. Construction has one 6 KiB staging buffer; each lookup reads one
//! exact 24-byte rational MediaTime record. Closing the owner releases all bytes.
use super::super::frame_time;
use ferment_engine_core::{EngineError, Result};
use ferment_engine_media::{MediaTime, TimeBase};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const RECORD_BYTES: usize = 3 * std::mem::size_of::<i64>();
const BUFFER_BYTES: usize = 6 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(super) struct FrameTimeline {
    file: Mutex<File>,
    count: usize,
    first: MediaTime,
}
impl FrameTimeline {
    pub(super) fn len(&self) -> usize {
        self.count
    }
    pub(super) fn first(&self) -> MediaTime {
        self.first
    }
    pub(super) fn time(&self, index: usize) -> Result<MediaTime> {
        if index >= self.count {
            return Err(EngineError::Decode(format!(
                "timestamp {index} is outside {} frames",
                self.count
            )));
        }
        let offset = index
            .checked_mul(RECORD_BYTES)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| EngineError::Decode("timestamp offset overflow".into()))?;
        let mut file = self
            .file
            .lock()
            .map_err(|_| EngineError::Internal("timestamp reader poisoned".into()))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0_u8; RECORD_BYTES];
        file.read_exact(&mut bytes)?;
        let time = MediaTime::new(
            i64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            TimeBase::new_unchecked(
                i64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                i64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            ),
        );
        frame_time::validate(time)?;
        Ok(time)
    }
}

pub(super) struct FrameTimelineWriter {
    file: BufWriter<File>,
    count: usize,
    first: Option<MediaTime>,
    previous: Option<MediaTime>,
}
impl FrameTimelineWriter {
    pub(super) fn new(directory: &Path) -> Result<Self> {
        Ok(Self {
            file: BufWriter::with_capacity(BUFFER_BYTES, anonymous_file(directory)?),
            count: 0,
            first: None,
            previous: None,
        })
    }
    /// Repeated display timestamps collapse; regressions fail with the file
    /// still anonymously owned, so an early error cannot leak a named scratch.
    pub(super) fn append(&mut self, time: MediaTime) -> Result<()> {
        frame_time::validate(time)?;
        if let Some(previous) = self.previous {
            match frame_time::compare(time, previous)? {
                std::cmp::Ordering::Equal => return Ok(()),
                std::cmp::Ordering::Less => {
                    return Err(EngineError::Decode(
                        "Mapped presentation timestamps are not monotonic.".into(),
                    ));
                }
                std::cmp::Ordering::Greater => (),
            }
        }
        let count = self
            .count
            .checked_add(1)
            .ok_or_else(|| EngineError::Decode("frame count overflow".into()))?;
        self.file.write_all(&time.value.to_le_bytes())?;
        self.file
            .write_all(&time.time_base.numerator.to_le_bytes())?;
        self.file
            .write_all(&time.time_base.denominator.to_le_bytes())?;
        self.count = count;
        self.first.get_or_insert(time);
        self.previous = Some(time);
        Ok(())
    }
    pub(super) fn finish(mut self) -> Result<FrameTimeline> {
        let first = self
            .first
            .ok_or_else(|| EngineError::Decode("video decoder produced no frames".into()))?;
        self.file.flush()?;
        let file = self
            .file
            .into_inner()
            .map_err(|error| EngineError::Io(error.into_error()))?;
        Ok(FrameTimeline {
            file: Mutex::new(file),
            count: self.count,
            first,
        })
    }
}

/// A fresh name in `directory`, for a file that is created and opened only once.
#[cfg(any(unix, windows))]
fn candidate(directory: &Path) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    directory.join(format!(
        "ferment-edgetam-timeline-{}-{nonce}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(unix)]
fn anonymous_file(directory: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::create_dir_all(directory)?;
    loop {
        let path = candidate(directory);
        let opened = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path);
        match opened {
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
            Ok(file) => {
                if let Err(error) = std::fs::remove_file(&path) {
                    drop(file);
                    let _ = std::fs::remove_file(path);
                    return Err(error.into());
                }
                return Ok(file);
            }
        }
    }
}

/// As on Unix, the name goes as soon as the file is open: Windows removes it at
/// once on NTFS when every handle allows deletion, and no other process may
/// open the file for reading or writing meanwhile. Delete-on-close keeps the
/// bytes from outliving the handle should the name stay, also when the process
/// ends abnormally.
#[cfg(windows)]
fn anonymous_file(directory: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_DELETE: u32 = 0x0000_0004;
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    std::fs::create_dir_all(directory)?;
    loop {
        let path = candidate(directory);
        let opened = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .share_mode(FILE_SHARE_DELETE)
            .attributes(FILE_ATTRIBUTE_TEMPORARY)
            .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
            .open(&path);
        match opened {
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
            Ok(file) => {
                if let Err(error) = std::fs::remove_file(&path) {
                    drop(file);
                    return Err(error.into());
                }
                return Ok(file);
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn anonymous_file(_: &Path) -> Result<File> {
    Err(EngineError::Unsupported(
        "anonymous video timestamp files need a Unix or Windows file backend".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "ferment-timeline-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn build(path: &Path, values: impl IntoIterator<Item = i64>) -> Result<FrameTimeline> {
        let mut writer = FrameTimelineWriter::new(path)?;
        for value in values {
            writer.append(MediaTime::new(value, TimeBase::MICROSECONDS))?;
        }
        writer.finish()
    }

    /// Kill: keep duplicate samples, drop the origin or sort away a regressing
    /// track; exact display-order identity and the failure boundary both change.
    #[test]
    fn exact_unique_timestamps_survive_random_access_and_bounds() {
        let dir = Scratch::new();
        let index = build(&dir.0, [1_400_000, 1_400_000, 1_433_333, 1_466_667]).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(
            index.first(),
            MediaTime::new(1_400_000, TimeBase::MICROSECONDS)
        );
        assert_eq!(
            index.time(2).unwrap(),
            MediaTime::new(1_466_667, TimeBase::MICROSECONDS)
        );
        assert_eq!(
            index.time(0).unwrap(),
            MediaTime::new(1_400_000, TimeBase::MICROSECONDS)
        );
        assert_eq!(
            index.time(1).unwrap(),
            MediaTime::new(1_433_333, TimeBase::MICROSECONDS)
        );
        assert!(index.time(3).is_err());
        assert!(index.time(usize::MAX).is_err());
        assert!(std::fs::read_dir(&dir.0).unwrap().next().is_none());
    }

    /// Kill: quantize a rational record or compare representations instead of
    /// value; 1/30 and 2/60 are one sample while sub-microsecond neighbors are not.
    #[test]
    fn rational_records_preserve_exact_time_bases_and_collapse_only_equal_values() {
        let dir = Scratch::new();
        let mut writer = FrameTimelineWriter::new(&dir.0).unwrap();
        let times = [
            MediaTime::new(1, TimeBase::new_unchecked(1, 30)),
            MediaTime::new(2, TimeBase::new_unchecked(1, 60)),
            MediaTime::new(3_000_001, TimeBase::new_unchecked(1, 90_000_000)),
        ];
        for time in times {
            writer.append(time).unwrap();
        }
        let index = writer.finish().unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.time(0).unwrap(), times[0]);
        assert_eq!(index.time(1).unwrap(), times[2]);
        assert_eq!(index.file.lock().unwrap().metadata().unwrap().len(), 48);
        let mut invalid = FrameTimelineWriter::new(&dir.0).unwrap();
        assert!(
            invalid
                .append(MediaTime::new(0, TimeBase::new_unchecked(1, 0)))
                .is_err()
        );
    }

    /// Kill: name a surviving scratch or return an empty/regressing timeline;
    /// failure must leave exactly the directory contents from before the call.
    #[test]
    fn malformed_or_empty_timeline_leaves_no_temporary_path() {
        let dir = Scratch::new();
        assert!(build(&dir.0, [33_333, 0]).is_err());
        assert!(build(&dir.0, []).is_err());
        assert!(std::fs::read_dir(&dir.0).unwrap().next().is_none());
    }

    /// Kill: drop buffered records at flush, truncate the disk file or seek to
    /// the wrong record; a ten-hour lazy timeline must remain exactly addressable.
    #[test]
    fn ten_hour_timestamp_index_streams_to_disk_without_retaining_its_records() {
        let dir = Scratch::new();
        let count = 10 * 60 * 60 * 30;
        let mut writer = FrameTimelineWriter::new(&dir.0).unwrap();
        for frame in 0..count {
            writer
                .append(MediaTime::new(frame as i64, TimeBase::new_unchecked(1, 30)))
                .unwrap();
        }
        let index = writer.finish().unwrap();
        assert_eq!(index.len(), count);
        assert_eq!(
            index.file.lock().unwrap().metadata().unwrap().len(),
            (count * RECORD_BYTES) as u64
        );
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        for frame in [0, count / 4, count / 2, count - 1] {
            assert_eq!(
                index.time(frame).unwrap(),
                MediaTime::new(frame as i64, TimeBase::new_unchecked(1, 30))
            );
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
    mod process_memory {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/current_process_memory.rs"
        ));
    }

    /// Kill: keep the ten-hour, 24.7MiB timestamp record set in RAM beside the
    /// anonymous disk file. Same ten-hour/12MiB bound and Mac phys_footprint as iOS.
    #[test]
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
    #[ignore = "physical process-memory gate; run alone to isolate allocator growth"]
    fn a_long_disk_index_has_constant_process_memory() {
        let dir = Scratch::new();
        let baseline = process_memory::footprint_bytes();
        let count = 10 * 60 * 60 * 30;
        let mut writer = FrameTimelineWriter::new(&dir.0).unwrap();
        for frame in 0..count {
            writer
                .append(MediaTime::new(frame as i64, TimeBase::new_unchecked(1, 30)))
                .unwrap();
        }
        let index = writer.finish().unwrap();
        for frame in [0, count / 4, count / 2, count - 1] {
            assert_eq!(
                index.time(frame).unwrap(),
                MediaTime::new(frame as i64, TimeBase::new_unchecked(1, 30))
            );
        }
        let growth = process_memory::footprint_bytes().saturating_sub(baseline);
        assert!(
            growth < 12 * 1_048_576,
            "resident timestamp index grew by {growth} bytes"
        );
        assert_eq!(index.len(), count);
    }
}
