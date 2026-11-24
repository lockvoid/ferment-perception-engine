//! Model-free file PCM for containers Android's MediaExtractor cannot open.
//!
//! The existing libav decoder owns packet timing and codec priming. Decoding
//! writes bounded interleaved chunks to a caller-owned scratch directory;
//! random reads never retain the full recording in memory. The FFI object
//! owns and removes the scratch file, including every failed-construction path.
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_media::platform::libav::LibavAudioIo;
use ferment_engine_media::{AudioIo, Codec};

use crate::{EngineFfiError, FfiResult};

#[derive(Clone, uniffi::Record)]
pub struct AudioFileMetadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_count: u64,
    pub codec: String,
}

fn failure(error: impl std::fmt::Display) -> EngineFfiError {
    EngineFfiError::new(error.to_string())
}

/// Container metadata only. CAF is read by the same demuxer used for decode,
/// including valid frames from its packet table rather than coded AAC length.
#[uniffi::export]
pub fn probe_audio_file(path: String) -> FfiResult<AudioFileMetadata> {
    let info = futures_executor::block_on(LibavAudioIo.probe(Path::new(&path))).map_err(failure)?;
    let stream = info
        .audio_streams()
        .next()
        .ok_or_else(|| failure("file has no audio stream"))?;
    let rate = stream
        .sample_rate
        .filter(|rate| *rate > 0)
        .ok_or_else(|| failure("audio stream has no sample rate"))?;
    let channels = stream
        .channels
        .filter(|channels| *channels > 0)
        .ok_or_else(|| failure("audio stream has no channels"))?;
    let frames = stream
        .audio_timing
        .and_then(|timing| timing.audible_samples)
        .or_else(|| {
            stream
                .duration
                .map(|time| (time.seconds() * f64::from(rate)).round() as u64)
        })
        .ok_or_else(|| failure("audio stream has no valid duration"))?;
    let codec = match &stream.codec {
        Codec::Aac => "mp4a".to_owned(),
        Codec::Mp3 => ".mp3".to_owned(),
        Codec::Other(name) if name.starts_with("pcm_") => "lpcm".to_owned(),
        Codec::Other(name) => name.clone(),
        _ => return Err(failure("audio stream names a video codec")),
    };
    Ok(AudioFileMetadata {
        sample_rate: rate,
        channels,
        frame_count: frames,
        codec,
    })
}

struct ScratchFile {
    path: PathBuf,
}
impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Bounded, seekable PCM. No model/session/runtime initialization is required.
#[derive(uniffi::Object)]
pub struct DecodedAudioFile {
    metadata: AudioFileMetadata,
    file: Mutex<File>,
    _scratch: ScratchFile,
}

#[uniffi::export]
impl DecodedAudioFile {
    #[uniffi::constructor]
    pub fn new(path: String, temporary_directory: String) -> FfiResult<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let destination = Path::new(&temporary_directory).join(format!(
            "ferment-audio-reader-{}-{nonce}-{}.f32",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        // create_new prevents taking ownership of any pre-existing caller file.
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)
            .map_err(failure)?;
        let scratch = ScratchFile { path: destination };
        let mut writer = BufWriter::new(output);
        let mut metadata = probe_audio_file(path.clone())?;
        let mut count = 0_u64;
        futures_executor::block_on(async {
            let source = Path::new(&path);
            let info = LibavAudioIo.probe(source).await.map_err(failure)?;
            let index = info
                .audio_streams()
                .next()
                .ok_or_else(|| failure("file has no audio stream"))?
                .index;
            let mut decoder = LibavAudioIo
                .open_audio_decoder(source, index)
                .await
                .map_err(failure)?;
            while let Some(frame) = decoder.decode_next().await.map_err(failure)? {
                if frame.sample_rate != metadata.sample_rate || frame.channels != metadata.channels
                {
                    return Err(failure("decoded audio changed its sample rate or channels"));
                }
                count = count
                    .checked_add(frame.frame_count() as u64)
                    .ok_or_else(|| failure("decoded audio is too large"))?;
                for sample in frame.samples.iter() {
                    writer.write_all(&sample.to_le_bytes()).map_err(failure)?;
                }
            }
            Ok::<(), EngineFfiError>(())
        })?;
        writer.flush().map_err(failure)?;
        drop(writer);
        metadata.frame_count = count;
        let file = File::open(&scratch.path).map_err(failure)?;
        Ok(Self {
            metadata,
            file: Mutex::new(file),
            _scratch: scratch,
        })
    }

    pub fn metadata(&self) -> AudioFileMetadata {
        self.metadata.clone()
    }

    /// Interleaved native-rate float32; at most 16,384 frames per FFI call.
    pub fn read_frames(&self, offset: u64, count: u32) -> FfiResult<Vec<f32>> {
        if count > 16_384 {
            return Err(failure("PCM read exceeds 16384 frames"));
        }
        let frames = self
            .metadata
            .frame_count
            .saturating_sub(offset)
            .min(u64::from(count));
        if frames == 0 {
            return Ok(Vec::new());
        }
        let channels = usize::from(self.metadata.channels);
        let mut bytes = vec![0_u8; frames as usize * channels * 4];
        let byte_offset = offset
            .checked_mul(channels as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| failure("PCM read offset overflow"))?;
        let mut file = self.file.lock().map_err(failure)?;
        file.seek(SeekFrom::Start(byte_offset)).map_err(failure)?;
        file.read_exact(&mut bytes).map_err(failure)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}
