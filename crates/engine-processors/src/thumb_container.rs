//! Packed JPEG thumb-strip container — the port of the iOS `ThumbContainer`
//! format (`FermentCuts/Components/ThumbStrip/Data/ThumbContainer.swift`).
//!
//! One file per video element, written by the encode pass as it decodes and
//! read by the timeline while that encode is still running. There are no
//! locks: a fixed header carries a `valid_count` the writer bumps only after
//! the cell's bytes and its index slot are durable, so a reader's
//! `count → index → data` order can never observe a half-written cell.
//!
//! ```text
//!   [0..4)    magic "PCKT"
//!   [4..8)    version   u32 LE = 2
//!   [8..12)   max_count u32 LE
//!   [12..16)  valid_count u32 LE   <- the publish point
//!   [16..20)  data_offset u32 LE = 24 + max_count * 16
//!   [20..24)  reserved  u32 LE = 0
//!   [24..data_offset)  index slots: (offset u64 LE, size u64 LE) * max_count
//!   [data_offset..EOF) cell bytes, appended
//! ```
//!
//! Concurrency contract, inherited from upstream and unchanged: ONE writer
//! (the encode's ingest thread) and readers that each hold their own handle.
//! Every access is positional, so nothing shares a file cursor.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use ferment_engine_core::{EngineError, Result, ThumbnailFrame};

/// `PCKT`, little-endian on disk exactly as written.
const MAGIC: [u8; 4] = *b"PCKT";
/// Bumping this invalidates every strip on disk; readers reject other values.
const FORMAT_VERSION: u32 = 2;
const HEADER_SIZE: u32 = 24;
const INDEX_SLOT_SIZE: u32 = 16;
const VALID_COUNT_OFFSET: u64 = 12;

/// Cell quality. Upstream's libjpeg-turbo compressor uses 80 on the same
/// 192-pixel cells; a strip is scrub-preview material, not a deliverable.
pub const THUMBNAIL_JPEG_QUALITY: u8 = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    pub offset: u64,
    pub size: u64,
}

/// Reader and appender for one strip file.
#[derive(Debug)]
pub struct ThumbContainer {
    file: File,
    max_count: u32,
    data_offset: u64,
    next_write_offset: u64,
}

impl ThumbContainer {
    /// Byte length of an empty container with `max_count` reserved slots.
    #[must_use]
    pub const fn reserved_len(max_count: u32) -> u64 {
        HEADER_SIZE as u64 + max_count as u64 * INDEX_SLOT_SIZE as u64
    }

    /// Write a fresh empty container at `path`, replacing anything there.
    ///
    /// Pre-creating means a reader can open the strip the instant the element
    /// exists and watch `valid_count` grow from zero as the encode runs.
    pub fn create_empty(path: &Path, max_count: u32) -> Result<()> {
        let data_offset = u32::try_from(Self::reserved_len(max_count)).map_err(|_| {
            EngineError::invalid_config(format!(
                "thumb strip of {max_count} cells overflows the 32-bit index table"
            ))
        })?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::with_capacity(data_offset as usize);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&max_count.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // valid_count
        bytes.extend_from_slice(&data_offset.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
        bytes.resize(data_offset as usize, 0);
        std::fs::write(path, &bytes)?;
        Ok(())
    }

    /// Open an existing container for reading and appending.
    ///
    /// A missing, truncated, wrong-magic, wrong-version or internally
    /// inconsistent file is an error rather than a silently empty strip: a
    /// caller that cannot tell the two apart writes cells nobody can read.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut header = [0u8; HEADER_SIZE as usize];
        read_exact_at(&file, &mut header, 0)?;
        if header[..4] != MAGIC {
            return Err(malformed(path, "bad magic"));
        }
        let version = read_u32(&header, 4);
        if version != FORMAT_VERSION {
            return Err(malformed(
                path,
                &format!("format version {version}, expected {FORMAT_VERSION}"),
            ));
        }
        let max_count = read_u32(&header, 8);
        let valid_count = read_u32(&header, 12);
        let data_offset = read_u32(&header, 16);
        if u64::from(data_offset) != Self::reserved_len(max_count) {
            return Err(malformed(
                path,
                &format!("data offset {data_offset} does not match {max_count} index slots"),
            ));
        }
        if valid_count > max_count {
            return Err(malformed(
                path,
                &format!("valid count {valid_count} exceeds max count {max_count}"),
            ));
        }
        // Resume the append cursor past every published cell, so a container
        // reopened mid-encode keeps its contiguous-append invariant.
        let mut next_write_offset = u64::from(data_offset);
        let mut container = Self {
            file,
            max_count,
            data_offset: u64::from(data_offset),
            next_write_offset,
        };
        for index in 0..valid_count {
            let entry = container.raw_index_entry(index)?;
            next_write_offset = next_write_offset.max(entry.offset + entry.size);
        }
        container.next_write_offset = next_write_offset;
        Ok(container)
    }

    /// Open `path`, creating an empty container with `max_count` slots when it
    /// is absent or its reserved size no longer matches.
    pub fn open_or_create(path: &Path, max_count: u32) -> Result<Self> {
        if let Ok(existing) = Self::open(path)
            && existing.max_count == max_count
        {
            return Ok(existing);
        }
        Self::create_empty(path, max_count)?;
        Self::open(path)
    }

    #[must_use]
    pub const fn max_count(&self) -> u32 {
        self.max_count
    }

    /// Latest published cell count, re-read from the header on every call —
    /// that is how a reader learns about cells the encoder just wrote.
    pub fn valid_count(&self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        read_exact_at(&self.file, &mut bytes, VALID_COUNT_OFFSET)?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// The slot for `index`, or `None` when it is past what is published.
    pub fn index_entry(&self, index: u32) -> Result<Option<IndexEntry>> {
        if index >= self.valid_count()? {
            return Ok(None);
        }
        self.raw_index_entry(index).map(Some)
    }

    /// The raw cell bytes for `index`, or `None` when it is unpublished.
    pub fn cell(&self, index: u32) -> Result<Option<Vec<u8>>> {
        let Some(entry) = self.index_entry(index)? else {
            return Ok(None);
        };
        let size = usize::try_from(entry.size)
            .map_err(|_| EngineError::invalid_config("thumb cell size exceeds usize"))?;
        let mut bytes = vec![0u8; size];
        read_exact_at(&self.file, &mut bytes, entry.offset)?;
        Ok(Some(bytes))
    }

    /// Append one cell and publish it. Returns the new `valid_count`.
    ///
    /// The three writes are ordered bytes → slot → count. A failure part-way
    /// leaves orphaned bytes or an orphaned slot that no reader can see,
    /// because the count did not advance; the cursor only moves on full
    /// success, so the next append overwrites the orphan.
    pub fn append(&mut self, cell: &[u8]) -> Result<u32> {
        let valid_count = self.valid_count()?;
        if valid_count >= self.max_count {
            return Err(EngineError::invalid_config(format!(
                "thumb strip is full at {} cells",
                self.max_count
            )));
        }
        let offset = self.next_write_offset;
        write_all_at(&self.file, cell, offset)?;
        let mut slot = [0u8; INDEX_SLOT_SIZE as usize];
        slot[..8].copy_from_slice(&offset.to_le_bytes());
        slot[8..].copy_from_slice(&(cell.len() as u64).to_le_bytes());
        write_all_at(&self.file, &slot, self.slot_offset(valid_count))?;
        write_all_at(
            &self.file,
            &(valid_count + 1).to_le_bytes(),
            VALID_COUNT_OFFSET,
        )?;
        self.next_write_offset = offset + cell.len() as u64;
        Ok(valid_count + 1)
    }

    /// Republish an empty strip and rewind the cursor.
    ///
    /// A re-run of the encode re-emits every cell from index 0. Without this
    /// the count accumulates run over run and overflows `max_count`. Index
    /// slots are left as they are — the next appends overwrite them.
    pub fn reset(&mut self) -> Result<()> {
        write_all_at(&self.file, &0u32.to_le_bytes(), VALID_COUNT_OFFSET)?;
        self.next_write_offset = self.data_offset;
        Ok(())
    }

    const fn slot_offset(&self, index: u32) -> u64 {
        HEADER_SIZE as u64 + index as u64 * INDEX_SLOT_SIZE as u64
    }

    fn raw_index_entry(&self, index: u32) -> Result<IndexEntry> {
        let mut slot = [0u8; INDEX_SLOT_SIZE as usize];
        read_exact_at(&self.file, &mut slot, self.slot_offset(index))?;
        Ok(IndexEntry {
            offset: u64::from_le_bytes(slot[..8].try_into().expect("8 bytes")),
            size: u64::from_le_bytes(slot[8..].try_into().expect("8 bytes")),
        })
    }
}

/// JPEG-encode one emitted thumbnail into a strip cell.
///
/// Diverges from upstream in one measured way: `image`'s baseline encoder
/// writes 4:4:4, where libjpeg-turbo's `TJSAMP_420` gives iOS ~40% smaller
/// cells at the same quality. The container stores opaque bytes, so this is a
/// size trade, not a format difference.
pub fn encode_thumbnail_jpeg(thumbnail: &ThumbnailFrame) -> Result<Vec<u8>> {
    let frame = &thumbnail.frame;
    if frame.format != ferment_engine_core::PixelFormat::Bgra8 {
        return Err(EngineError::invalid_config(format!(
            "thumb cells are encoded from BGRA8; got {:?}",
            frame.format
        )));
    }
    let pixels = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .ok_or_else(|| EngineError::invalid_config("thumbnail dimensions overflow"))?;
    if frame.data.len() < pixels * 4 {
        return Err(EngineError::invalid_config(format!(
            "thumbnail is {} bytes, short of {}x{} BGRA8",
            frame.data.len(),
            frame.width,
            frame.height
        )));
    }
    let mut rgb = Vec::with_capacity(pixels * 3);
    for pixel in frame.data[..pixels * 4].chunks_exact(4) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, THUMBNAIL_JPEG_QUALITY)
        .encode(
            &rgb,
            frame.width,
            frame.height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|error| EngineError::Encode(error.to_string()))?;
    Ok(jpeg)
}

fn malformed(path: &Path, reason: &str) -> EngineError {
    EngineError::invalid_config(format!(
        "{} is not a v{FORMAT_VERSION} thumb strip: {reason}",
        path.display()
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("4 bytes"))
}

/// Positional read that never touches a shared file cursor.
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut read = 0;
        while read < buffer.len() {
            let count = std::os::windows::fs::FileExt::seek_read(
                file,
                &mut buffer[read..],
                offset + read as u64,
            )?;
            if count == 0 {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            read += count;
        }
        Ok(())
    }
}

fn write_all_at(file: &File, buffer: &[u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::write_all_at(file, buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut written = 0;
        while written < buffer.len() {
            let count = std::os::windows::fs::FileExt::seek_write(
                file,
                &buffer[written..],
                offset + written as u64,
            )?;
            if count == 0 {
                return Err(io::Error::from(io::ErrorKind::WriteZero));
            }
            written += count;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use ferment_engine_core::{PixelFormat, PlaneLayout, VideoFrame};

    use super::*;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct TempStrip(std::path::PathBuf);

    impl TempStrip {
        fn new(label: &str) -> Self {
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "ferment-thumbstrip-{label}-{}-{sequence}.thumbstrip",
                std::process::id()
            )))
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn bytes(&self) -> Vec<u8> {
            std::fs::read(&self.0).unwrap()
        }
    }

    impl Drop for TempStrip {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bgra_thumbnail(width: u32, height: u32, index: usize) -> ThumbnailFrame {
        let stride = width as usize * 4;
        let mut data = Vec::with_capacity(stride * height as usize);
        for y in 0..height {
            for x in 0..width {
                // A gradient, so a decoded cell can be told apart from noise.
                data.extend_from_slice(&[(x * 3) as u8, (y * 5) as u8, 0x20, 0xFF]);
            }
        }
        ThumbnailFrame {
            frame: VideoFrame {
                data: data.into(),
                planes: vec![PlaneLayout {
                    offset: 0,
                    stride,
                    row_bytes: stride,
                    rows: height as usize,
                }],
                width,
                height,
                format: PixelFormat::Bgra8,
                presentation_time: ferment_engine_core::MediaTime::default(),
            },
            index,
            presentation_time: index as f64 * 0.5,
        }
    }

    /// The byte layout is the wire contract with the iOS reader; it is
    /// asserted literally, not through the accessors that produced it.
    #[test]
    fn an_empty_container_is_a_pckt_v2_header_plus_zeroed_slots() {
        let strip = TempStrip::new("empty");
        ThumbContainer::create_empty(strip.path(), 4).unwrap();
        let bytes = strip.bytes();

        assert_eq!(&bytes[0..4], b"PCKT");
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &(24u32 + 4 * 16).to_le_bytes());
        assert_eq!(&bytes[20..24], &0u32.to_le_bytes());
        assert_eq!(bytes.len(), 24 + 4 * 16);
        assert!(bytes[24..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn appending_publishes_bytes_then_slot_then_count() {
        let strip = TempStrip::new("append");
        ThumbContainer::create_empty(strip.path(), 3).unwrap();
        let mut container = ThumbContainer::open(strip.path()).unwrap();
        assert_eq!(container.valid_count().unwrap(), 0);
        assert_eq!(container.index_entry(0).unwrap(), None);
        assert_eq!(container.cell(0).unwrap(), None);

        assert_eq!(container.append(b"first-cell").unwrap(), 1);
        assert_eq!(container.append(b"second").unwrap(), 2);

        let data_offset = ThumbContainer::reserved_len(3);
        assert_eq!(
            container.index_entry(0).unwrap(),
            Some(IndexEntry {
                offset: data_offset,
                size: 10
            })
        );
        assert_eq!(
            container.index_entry(1).unwrap(),
            Some(IndexEntry {
                offset: data_offset + 10,
                size: 6
            })
        );
        assert_eq!(container.cell(0).unwrap().unwrap(), b"first-cell");
        assert_eq!(container.cell(1).unwrap().unwrap(), b"second");
        // Reserved but unpublished.
        assert_eq!(container.index_entry(2).unwrap(), None);

        // The published count is the last thing on disk, and the cells sit
        // exactly where the index says.
        let bytes = strip.bytes();
        assert_eq!(&bytes[12..16], &2u32.to_le_bytes());
        assert_eq!(&bytes[data_offset as usize..][..16], b"first-cellsecond");
    }

    #[test]
    fn a_second_reader_sees_cells_the_writer_has_published() {
        let strip = TempStrip::new("concurrent");
        ThumbContainer::create_empty(strip.path(), 2).unwrap();
        let mut writer = ThumbContainer::open(strip.path()).unwrap();
        let reader = ThumbContainer::open(strip.path()).unwrap();

        assert_eq!(reader.valid_count().unwrap(), 0);
        writer.append(b"cell-zero").unwrap();
        assert_eq!(reader.valid_count().unwrap(), 1);
        assert_eq!(reader.cell(0).unwrap().unwrap(), b"cell-zero");
        assert_eq!(reader.cell(1).unwrap(), None);
    }

    #[test]
    fn a_full_strip_refuses_the_next_cell() {
        let strip = TempStrip::new("full");
        ThumbContainer::create_empty(strip.path(), 1).unwrap();
        let mut container = ThumbContainer::open(strip.path()).unwrap();
        container.append(b"only").unwrap();
        let error = container.append(b"overflow").unwrap_err();
        assert!(
            error.to_string().contains("full at 1 cells"),
            "unexpected error: {error}"
        );
        assert_eq!(container.valid_count().unwrap(), 1);
    }

    /// A re-run of the encode re-emits from index 0. Without the reset the
    /// count accumulates run over run and overflows `max_count`.
    #[test]
    fn reset_rewinds_the_strip_for_a_re_run() {
        let strip = TempStrip::new("reset");
        ThumbContainer::create_empty(strip.path(), 2).unwrap();
        let mut container = ThumbContainer::open(strip.path()).unwrap();
        container.append(b"stale-one").unwrap();
        container.append(b"stale-two").unwrap();

        container.reset().unwrap();
        assert_eq!(container.valid_count().unwrap(), 0);
        assert_eq!(container.append(b"fresh").unwrap(), 1);
        assert_eq!(container.cell(0).unwrap().unwrap(), b"fresh");
        assert_eq!(
            container.index_entry(0).unwrap().unwrap().offset,
            ThumbContainer::reserved_len(2),
            "the cursor rewound to the data offset"
        );
    }

    #[test]
    fn reopening_resumes_the_append_cursor_past_published_cells() {
        let strip = TempStrip::new("resume");
        ThumbContainer::create_empty(strip.path(), 4).unwrap();
        let mut first = ThumbContainer::open(strip.path()).unwrap();
        first.append(b"aaaa").unwrap();
        first.append(b"bbbbbb").unwrap();
        drop(first);

        let mut second = ThumbContainer::open(strip.path()).unwrap();
        assert_eq!(second.append(b"cc").unwrap(), 3);
        assert_eq!(second.cell(0).unwrap().unwrap(), b"aaaa");
        assert_eq!(second.cell(1).unwrap().unwrap(), b"bbbbbb");
        assert_eq!(second.cell(2).unwrap().unwrap(), b"cc");
    }

    #[test]
    fn open_or_create_replaces_a_container_reserved_for_a_different_length() {
        let strip = TempStrip::new("resize");
        let mut original = ThumbContainer::open_or_create(strip.path(), 2).unwrap();
        original.append(b"old").unwrap();
        drop(original);

        let reused = ThumbContainer::open_or_create(strip.path(), 2).unwrap();
        assert_eq!(
            reused.valid_count().unwrap(),
            1,
            "same size reuses in place"
        );
        drop(reused);

        let resized = ThumbContainer::open_or_create(strip.path(), 8).unwrap();
        assert_eq!(resized.max_count(), 8);
        assert_eq!(resized.valid_count().unwrap(), 0);
    }

    #[test]
    fn a_file_that_is_not_a_v2_strip_is_refused_loudly() {
        let strip = TempStrip::new("malformed");
        std::fs::write(strip.path(), b"not a container at all, but long enough").unwrap();
        assert!(ThumbContainer::open(strip.path()).is_err());

        // Right magic, wrong version.
        let mut bytes = vec![0u8; 24];
        bytes[..4].copy_from_slice(b"PCKT");
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&24u32.to_le_bytes());
        std::fs::write(strip.path(), &bytes).unwrap();
        let error = ThumbContainer::open(strip.path()).unwrap_err();
        assert!(
            error.to_string().contains("format version 99"),
            "unexpected error: {error}"
        );

        // Right magic and version, index table that does not match.
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
        std::fs::write(strip.path(), &bytes).unwrap();
        assert!(
            ThumbContainer::open(strip.path())
                .unwrap_err()
                .to_string()
                .contains("data offset")
        );

        // Truncated below the header.
        std::fs::write(strip.path(), b"PCK").unwrap();
        assert!(ThumbContainer::open(strip.path()).is_err());
    }

    #[test]
    fn thumbnail_cells_are_baseline_jpeg_that_decodes_back_to_the_frame_size() {
        let thumbnail = bgra_thumbnail(96, 54, 0);
        let jpeg = encode_thumbnail_jpeg(&thumbnail).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "JPEG SOI");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "JPEG EOI");

        let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (96, 54));

        // Channel order is the trap: the source is BGRA, so byte 0 is BLUE and
        // byte 2 is RED. At (48, 27) the frame holds B=144, G=135, R=0x20, and
        // the decoded RGB pixel must read (0x20, 135, 144) — a swapped encoder
        // would land (144, 135, 0x20).
        let rgb = decoded.to_rgb8();
        let pixel = rgb.get_pixel(48, 27).0;
        for (channel, expected) in pixel.iter().zip([0x20u8, 135, 144]) {
            assert!(
                i32::from(*channel).abs_diff(i32::from(expected)) <= 12,
                "expected ~(0x20, 135, 144) at (48, 27), decoded {pixel:?}"
            );
        }
    }

    #[test]
    fn a_thumbnail_that_is_not_bgra_or_is_short_is_refused() {
        let mut wrong_format = bgra_thumbnail(4, 4, 0);
        wrong_format.frame.format = PixelFormat::Rgba8;
        assert!(encode_thumbnail_jpeg(&wrong_format).is_err());

        let mut truncated = bgra_thumbnail(4, 4, 0);
        truncated.frame.data = truncated.frame.data[..8].into();
        assert!(encode_thumbnail_jpeg(&truncated).is_err());
    }

    /// The whole cell path end to end: encode frames, pack them, read them
    /// back through a fresh handle and decode.
    #[test]
    fn a_strip_round_trips_encoded_thumbnails() {
        let strip = TempStrip::new("roundtrip");
        let mut container = ThumbContainer::open_or_create(strip.path(), 3).unwrap();
        for index in 0..3 {
            let cell = encode_thumbnail_jpeg(&bgra_thumbnail(64, 36, index)).unwrap();
            assert_eq!(container.append(&cell).unwrap(), index as u32 + 1);
        }
        drop(container);

        let reader = ThumbContainer::open(strip.path()).unwrap();
        assert_eq!(reader.valid_count().unwrap(), 3);
        for index in 0..3 {
            let cell = reader.cell(index).unwrap().unwrap();
            let decoded =
                image::load_from_memory_with_format(&cell, image::ImageFormat::Jpeg).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (64, 36));
        }
    }
}
