//! Processor-side audio ingestion.
//!
//! Portable WAV decoding stays in `engine-audio`. Patent-encumbered AAC is
//! decoded by the operating system through `engine-media`, then downmixed to
//! the mono `AudioFrames` contract consumed by analysis processors.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ferment_engine_audio::derived_cache::{
    DerivedFileCache, DerivedFileKey, DerivedSource, request_hash,
};
use ferment_engine_audio::streaming_resampler::AudioConverter;
use ferment_engine_audio::{AudioFrames, WavStreamReader};
use ferment_engine_core::{EngineError, Result};
use ferment_engine_media::{AudioBackend, MediaError};
use serde::Serialize;

/// Windowed mono-f32 access used by long-running processors.
///
/// The trait itself moved to `engine-core` when [`LiveTranscriptionSession`]
/// — a contract — started reading through it; re-exported here so every
/// existing `ferment_engine_processors::AudioSampleSource` import still names
/// the same trait.
///
/// [`LiveTranscriptionSession`]: ferment_engine_core::LiveTranscriptionSession
pub use ferment_engine_core::AudioSampleSource;

#[derive(Debug, Clone)]
pub struct ArrayAudioSampleSource {
    samples: Arc<[f32]>,
}

impl ArrayAudioSampleSource {
    #[must_use]
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples: samples.into(),
        }
    }
}

impl AudioSampleSource for ArrayAudioSampleSource {
    fn sample_count(&self) -> usize {
        self.samples.len()
    }

    fn copy_samples(&self, destination: &mut [f32], offset: usize, count: usize) -> Result<usize> {
        let available = self.samples.len().saturating_sub(offset);
        let copied = available.min(count).min(destination.len());
        if copied > 0 {
            destination[..copied].copy_from_slice(&self.samples[offset..offset + copied]);
        }
        Ok(copied)
    }
}

/// Owned raw little-endian f32 file with bounded, windowed reads.
///
/// Swift uses `mmap`; this implementation deliberately uses seek/read so the
/// resident set is likewise bounded without adding a native-only dependency.
#[derive(Debug)]
pub struct DiskBackedAudioSampleSource {
    path: PathBuf,
    sample_count: usize,
    cleaned: AtomicBool,
    owned_by_cache: bool,
    file: Mutex<File>,
}

impl DiskBackedAudioSampleSource {
    pub fn from_owned_file(path: PathBuf) -> Result<Self> {
        Self::open(path, false)
    }

    pub fn from_cached_file(path: PathBuf) -> Result<Self> {
        Self::open(path, true)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Transfer temporary-file ownership to an FFI caller, which maps it and
    /// owns deletion. Cache-owned files remain owned by their cache.
    pub fn into_path(mut self) -> PathBuf {
        self.owned_by_cache = true;
        self.path.clone()
    }

    fn open(path: PathBuf, owned_by_cache: bool) -> Result<Self> {
        let byte_length = std::fs::metadata(&path)?.len();
        if byte_length % 4 != 0 {
            return Err(EngineError::Decode(format!(
                "raw f32 source has {byte_length} bytes, not a multiple of four"
            )));
        }
        Ok(Self {
            path: path.clone(),
            sample_count: usize::try_from(byte_length / 4)
                .map_err(|_| EngineError::Decode("raw f32 source is too large".into()))?,
            cleaned: AtomicBool::new(false),
            owned_by_cache,
            file: Mutex::new(File::open(&path)?),
        })
    }

    /// Idempotently remove the owned temporary file.
    pub fn cleanup(&self) -> Result<()> {
        if self.owned_by_cache || self.cleaned.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

impl AudioSampleSource for DiskBackedAudioSampleSource {
    fn sample_count(&self) -> usize {
        self.sample_count
    }

    fn copy_samples(&self, destination: &mut [f32], offset: usize, count: usize) -> Result<usize> {
        let available = self.sample_count.saturating_sub(offset);
        let copied = available.min(count).min(destination.len());
        if copied == 0 {
            return Ok(0);
        }
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start((offset as u64) * 4))?;
        let mut bytes = vec![0_u8; copied * 4];
        file.read_exact(&mut bytes)?;
        for (destination, chunk) in destination.iter_mut().zip(bytes.chunks_exact(4)) {
            *destination = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(copied)
    }
}

impl Drop for DiskBackedAudioSampleSource {
    fn drop(&mut self) {
        crate::optional_failure::cleanup("remove a disk-backed sample file", self.cleanup());
    }
}

#[derive(Debug, Clone)]
pub struct AudioSource {
    media: AudioBackend,
}

impl AudioSource {
    #[must_use]
    pub fn new(media: AudioBackend) -> Self {
        Self { media }
    }

    /// The compressed-audio backend. One on every host: libavcodec decodes,
    /// and reaches the system encoders through the same interface. On Android
    /// MediaCodec decodes on the phone's own audio DSP, and hands back to
    /// libavcodec anything it has no decoder for.
    #[must_use]
    pub fn platform_default() -> Self {
        #[cfg(target_os = "android")]
        let backend: ferment_engine_media::AudioBackend =
            Arc::new(ferment_engine_media::platform::android::AndroidAudioIo);
        #[cfg(not(target_os = "android"))]
        let backend: ferment_engine_media::AudioBackend =
            Arc::new(ferment_engine_media::platform::libav::LibavAudioIo);
        Self::new(backend)
    }

    /// Decode a source into mono floating-point PCM at its native sample rate.
    pub async fn decode(&self, source: &Path) -> Result<AudioFrames> {
        if is_wave_file(source)? {
            return decode_wave_mean(source);
        }

        let info = self.media.probe(source).await.map_err(map_media_error)?;

        let stream_index = info
            .audio_streams()
            .next()
            .map(|stream| stream.index)
            .ok_or(EngineError::NoAudioData)?;
        let mut decoder = self
            .media
            .open_audio_decoder(source, stream_index)
            .await
            .map_err(map_media_error)?;

        let mut sample_rate = None;
        let mut samples = Vec::new();
        while let Some(frame) = decoder.decode_next().await.map_err(map_media_error)? {
            if frame.channels == 0 || frame.sample_rate == 0 {
                return Err(EngineError::NoAudioData);
            }
            if let Some(expected) = sample_rate {
                if frame.sample_rate != expected {
                    return Err(EngineError::Decode(format!(
                        "audio sample rate changed from {expected} to {} mid-stream",
                        frame.sample_rate
                    )));
                }
            } else {
                sample_rate = Some(frame.sample_rate);
            }
            samples.extend(
                frame
                    .samples
                    .chunks_exact(usize::from(frame.channels))
                    .map(mean_channel),
            );
        }

        let sample_rate = sample_rate.ok_or(EngineError::NoAudioData)?;
        if samples.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        Ok(AudioFrames::new(samples, sample_rate))
    }

    /// Decode to an owned raw-f32 temporary file. Native compressed frames are
    /// converted with bounded FIR/polyphase history and written as they arrive.
    pub async fn decode_disk_backed(
        &self,
        source: &Path,
        target_sample_rate: u32,
    ) -> Result<DiskBackedAudioSampleSource> {
        self.decode_disk_backed_in(source, target_sample_rate, &std::env::temp_dir())
            .await
    }

    /// Android app callers provide their own writable cache directory; the
    /// Rust process default (/data/local/tmp) is outside an app's sandbox.
    pub async fn decode_disk_backed_in(
        &self,
        source: &Path,
        target_sample_rate: u32,
        temporary_directory: &Path,
    ) -> Result<DiskBackedAudioSampleSource> {
        self.decode_disk_backed_in_with_cancellation(
            source,
            target_sample_rate,
            temporary_directory,
            || Ok(()),
        )
        .await
    }

    /// The caller's cancellation check runs before and after each bounded
    /// decoder read/conversion. Cancellation removes only this call's partial
    /// PCM artifact; the original input is never owned by the decoder.
    pub async fn decode_disk_backed_in_with_cancellation(
        &self,
        source: &Path,
        target_sample_rate: u32,
        temporary_directory: &Path,
        mut check_cancellation: impl FnMut() -> Result<()>,
    ) -> Result<DiskBackedAudioSampleSource> {
        check_cancellation()?;
        if target_sample_rate == 0 {
            return Err(EngineError::invalid_config(
                "target sample rate must be greater than zero",
            ));
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = temporary_directory.join(format!(
            "ferment-audio-source-{}-{nonce}.f32",
            std::process::id()
        ));
        let result = async {
            self.decode_to_raw_file_checked(
                source,
                target_sample_rate,
                &path,
                &mut check_cancellation,
            )
            .await?;
            check_cancellation()?;
            DiskBackedAudioSampleSource::from_owned_file(path.clone())
        }
        .await;
        if result.is_err() {
            crate::optional_failure::cleanup(
                "remove a partial sample file",
                crate::optional_failure::remove_if_present(&path),
            );
        }
        result
    }

    /// Every path uses this block-streamed conversion, so direct and cached
    /// PCM are bit-identical. The decoder and converter both keep bounded state.
    async fn stream_resampled(
        &self,
        source: &Path,
        target_rate: u32,
        sink: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        self.stream_resampled_checked(source, target_rate, sink, || Ok(()))
            .await
    }

    async fn stream_resampled_checked(
        &self,
        source: &Path,
        target_rate: u32,
        mut sink: impl FnMut(&[f32]) -> Result<()>,
        mut check_cancellation: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        if target_rate == 0 {
            return Err(EngineError::invalid_config(
                "target sample rate must be positive",
            ));
        }
        check_cancellation()?;
        let mut count = 0;
        if is_wave_file(source)? {
            let mut reader = WavStreamReader::open(source)?;
            let channels = usize::from(reader.channels());
            let mut block = vec![0.0; 16_384 * channels];
            let mut converter = AudioConverter::new(reader.sample_rate(), target_rate)?;
            loop {
                check_cancellation()?;
                let frames = reader.read_interleaved(&mut block)?;
                check_cancellation()?;
                if frames == 0 {
                    break;
                }
                let mono: Vec<f32> = block[..frames * channels]
                    .chunks_exact(channels)
                    .map(mean_channel)
                    .collect();
                count += mono.len();
                sink(&converter.process(&mono)?)?;
            }
            check_cancellation()?;
            sink(&converter.finish()?)?;
        } else {
            check_cancellation()?;
            let info = self.media.probe(source).await.map_err(map_media_error)?;
            check_cancellation()?;
            let index = info
                .audio_streams()
                .next()
                .map(|s| s.index)
                .ok_or(EngineError::NoAudioData)?;
            let mut decoder = self
                .media
                .open_audio_decoder(source, index)
                .await
                .map_err(map_media_error)?;
            let mut converter: Option<AudioConverter> = None;
            let mut rate = None;
            loop {
                check_cancellation()?;
                let frame = decoder.decode_next().await.map_err(map_media_error)?;
                check_cancellation()?;
                let Some(frame) = frame else { break };
                if frame.channels == 0 || frame.sample_rate == 0 {
                    return Err(EngineError::NoAudioData);
                }
                if rate.is_some_and(|old| old != frame.sample_rate) {
                    return Err(EngineError::decode("audio sample rate changed mid-stream"));
                }
                rate = Some(frame.sample_rate);
                if converter.is_none() {
                    converter = Some(AudioConverter::new(frame.sample_rate, target_rate)?);
                }
                let mono: Vec<f32> = frame
                    .samples
                    .chunks_exact(usize::from(frame.channels))
                    .map(mean_channel)
                    .collect();
                count += mono.len();
                sink(&converter.as_mut().unwrap().process(&mono)?)?;
            }
            check_cancellation()?;
            if let Some(converter) = &mut converter {
                sink(&converter.finish()?)?;
            }
        }
        check_cancellation()?;
        if count == 0 {
            return Err(EngineError::NoAudioData);
        }
        Ok(())
    }

    async fn decode_to_raw_file(
        &self,
        source: &Path,
        target_rate: u32,
        destination: &Path,
    ) -> Result<()> {
        self.decode_to_raw_file_checked(source, target_rate, destination, || Ok(()))
            .await
    }

    async fn decode_to_raw_file_checked(
        &self,
        source: &Path,
        target_rate: u32,
        destination: &Path,
        mut check_cancellation: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        check_cancellation()?;
        let mut writer = BufWriter::new(File::create(destination)?);
        self.stream_resampled_checked(
            source,
            target_rate,
            |block| write_f32(&mut writer, block),
            &mut check_cancellation,
        )
        .await?;
        check_cancellation()?;
        writer.flush()?;
        Ok(())
    }

    /// Current source identity, shared by every consumer of this source/rate.
    pub async fn resampled(
        &self,
        source: &Path,
        rate: u32,
        identity: Option<DerivedSource>,
        cache: Option<&DerivedFileCache>,
    ) -> Result<AudioFrames> {
        let request = ResampledAudioRequest {
            source: identity.map_or_else(|| DerivedSource::from_file(source), Ok)?,
            sample_rate: rate,
        };
        let identity = request.produced_identity()?;
        let samples = if let Some(cache) = cache {
            let path = cache.fetch(&request, |partial| {
                futures_executor::block_on(self.decode_to_raw_file(source, rate, partial))
            })?;
            read_f32_file(&path)?
        } else {
            let mut samples = Vec::new();
            self.stream_resampled(source, rate, |block| {
                samples.extend_from_slice(block);
                Ok(())
            })
            .await?;
            samples
        };
        Ok(AudioFrames {
            samples,
            sample_rate: rate,
            identity: Some(identity),
        })
    }

    pub async fn cached_source(
        &self,
        source: &Path,
        rate: u32,
        identity: Option<DerivedSource>,
        cache: &DerivedFileCache,
    ) -> Result<DiskBackedAudioSampleSource> {
        let request = ResampledAudioRequest {
            source: identity.map_or_else(|| DerivedSource::from_file(source), Ok)?,
            sample_rate: rate,
        };
        let path = cache.fetch(&request, |partial| {
            futures_executor::block_on(self.decode_to_raw_file(source, rate, partial))
        })?;
        DiskBackedAudioSampleSource::from_cached_file(path)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResampledAudioRequest {
    pub source: DerivedSource,
    pub sample_rate: u32,
}
impl DerivedFileKey for ResampledAudioRequest {
    const KIND: &'static str = "resampled-mono-f32le";
    const VERSION: u32 = 3;
    fn source(&self) -> &DerivedSource {
        &self.source
    }
    fn extension(&self) -> &str {
        "pcm"
    }
}
impl ResampledAudioRequest {
    pub fn produced_identity(&self) -> Result<DerivedSource> {
        Ok(DerivedSource::new(format!("pcm:{}", request_hash(self)?)))
    }
}

/// Current iOS v3: every container averages every channel, never the old
/// first-channel / equal-power split. Anti-phase stereo therefore cancels.
fn mean_channel(frame: &[f32]) -> f32 {
    frame.iter().sum::<f32>() / frame.len().max(1) as f32
}

fn decode_wave_mean(source: &Path) -> Result<AudioFrames> {
    let mut reader = WavStreamReader::open(source)?;
    let channels = usize::from(reader.channels());
    let mut block = vec![0.0; 16_384 * channels];
    let mut samples = Vec::new();
    loop {
        let frames = reader.read_interleaved(&mut block)?;
        if frames == 0 {
            break;
        }
        samples.extend(
            block[..frames * channels]
                .chunks_exact(channels)
                .map(mean_channel),
        );
    }
    if samples.is_empty() {
        return Err(EngineError::NoAudioData);
    }
    Ok(AudioFrames::new(samples, reader.sample_rate()))
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() {
        return Err(EngineError::NoAudioData);
    }
    if bytes.len() % 4 != 0 {
        return Err(EngineError::decode(
            "cached PCM is not whole float32 samples",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Cook artifacts deliberately have no filename extension. Normalization and
/// denoising therefore write RIFF/WAVE bytes to an opaque cook key, and audio
/// ingestion must identify that format by content rather than by path alone.
fn is_wave_file(path: &Path) -> Result<bool> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
    {
        return Ok(true);
    }

    let mut file = File::open(path)?;
    let mut header = [0_u8; 12];
    let count = file.read(&mut header)?;
    Ok(count == header.len()
        && matches!(&header[..4], b"RIFF" | b"RF64")
        && &header[8..12] == b"WAVE")
}

impl Default for AudioSource {
    fn default() -> Self {
        Self::platform_default()
    }
}

fn map_media_error(error: MediaError) -> EngineError {
    match error {
        MediaError::BackendUnavailable(message) => EngineError::Unsupported(message),
        other => EngineError::Decode(other.to_string()),
    }
}

fn write_f32(writer: &mut impl Write, samples: &[f32]) -> Result<()> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod sample_source_tests {
    use super::*;

    // AudioSampleSourceTests.swift

    #[test]
    fn test_array_source_copies_window() {
        let source = ArrayAudioSampleSource::new((0..100).map(|value| value as f32).collect());
        assert_eq!(source.sample_count(), 100);
        let mut destination = vec![-1.0; 10];
        assert_eq!(source.copy_samples(&mut destination, 5, 10).unwrap(), 10);
        assert_eq!(
            destination,
            (5..15).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_array_source_clamps_oversized_window() {
        let source = ArrayAudioSampleSource::new((0..10).map(|value| value as f32).collect());
        let mut destination = vec![-1.0; 50];
        assert_eq!(source.copy_samples(&mut destination, 5, 50).unwrap(), 5);
        assert_eq!(&destination[..5], &[5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(destination[5..].iter().all(|value| *value == -1.0));
    }

    #[test]
    fn test_array_source_offset_past_end_copies_nothing() {
        let source = ArrayAudioSampleSource::new((0..10).map(|value| value as f32).collect());
        let mut destination = vec![-1.0; 5];
        assert_eq!(source.copy_samples(&mut destination, 100, 5).unwrap(), 0);
        assert_eq!(destination, vec![-1.0; 5]);
    }

    #[test]
    fn test_array_source_empty() {
        let source = ArrayAudioSampleSource::new(Vec::new());
        assert_eq!(source.sample_count(), 0);
        let mut destination = vec![-1.0; 5];
        assert_eq!(source.copy_samples(&mut destination, 0, 5).unwrap(), 0);
        assert_eq!(destination, vec![-1.0; 5]);
    }

    fn raw_fixture(name: &str, samples: &[f32]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("ferment-{name}-{}.raw", std::process::id()));
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn test_disk_backed_source_copies_window_from_mapped_file() {
        let samples = (0..1000)
            .map(|value| value as f32 * 0.5)
            .collect::<Vec<_>>();
        let path = raw_fixture("disk-source-test", &samples);
        let source = DiskBackedAudioSampleSource::from_owned_file(path.clone()).unwrap();
        assert_eq!(source.sample_count(), 1000);
        let mut destination = vec![0.0; 16];
        assert_eq!(source.copy_samples(&mut destination, 100, 16).unwrap(), 16);
        assert_eq!(
            destination,
            (100..116)
                .map(|value| value as f32 * 0.5)
                .collect::<Vec<_>>()
        );
        source.cleanup().unwrap();
        source.cleanup().unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_disk_backed_source_cleans_up_in_deinit() {
        let path = raw_fixture("disk-source-deinit", &[1.0, 2.0, 3.0, 4.0]);
        assert!(path.exists());
        {
            let _source = DiskBackedAudioSampleSource::from_owned_file(path.clone()).unwrap();
        }
        assert!(!path.exists());
    }

    #[test]
    fn extensionless_cook_artifact_is_detected_as_wave_by_magic() {
        let path =
            std::env::temp_dir().join(format!("ferment-extensionless-wave-{}", std::process::id()));
        ferment_engine_audio::write_wav_f32(&path, &[0.0, 0.25, -0.25, 0.0], 8_000).unwrap();
        assert!(is_wave_file(&path).unwrap());

        let decoded = futures_executor::block_on(AudioSource::platform_default().decode(&path))
            .expect("extensionless WAVE should bypass the native compressed-media backend");
        assert_eq!(decoded.sample_rate, 8_000);
        assert_eq!(decoded.samples.len(), 4);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn non_wave_magic_does_not_claim_opaque_artifact() {
        let path = std::env::temp_dir().join(format!(
            "ferment-extensionless-non-wave-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"not a wave payload").unwrap();
        assert!(!is_wave_file(&path).unwrap());
        std::fs::remove_file(path).unwrap();
    }
    fn cancel_scratch(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("ferment-{label}-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        directory
    }

    #[test]
    fn cancellation_during_wave_decode_is_bounded_and_removes_partial_pcm() {
        let directory = cancel_scratch("cancel-wave");
        let input = directory.join("input.wav");
        let scratch = directory.join("scratch");
        std::fs::create_dir(&scratch).unwrap();
        ferment_engine_audio::write_wav_f32(&input, &vec![0.125; 48_000 * 8], 48_000).unwrap();
        let original = std::fs::read(&input).unwrap();
        let mut partial_bytes = 0;
        let result = futures_executor::block_on(
            AudioSource::default().decode_disk_backed_in_with_cancellation(
                &input,
                48_000,
                &scratch,
                || {
                    // Ask each file for its length: a directory listing's copy
                    // of it can lag while the file is still open for writing.
                    partial_bytes = std::fs::read_dir(&scratch)
                        .unwrap()
                        .map(|entry| std::fs::metadata(entry.unwrap().path()).unwrap().len())
                        .sum::<u64>();
                    if partial_bytes > 0 {
                        Err(EngineError::Cancelled)
                    } else {
                        Ok(())
                    }
                },
            ),
        );
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert!(
            partial_bytes > 0 && partial_bytes <= 16_384 * 4,
            "cancellation must stop after one bounded block, wrote {partial_bytes}"
        );
        assert_eq!(std::fs::read_dir(&scratch).unwrap().count(), 0);
        assert_eq!(std::fs::read(&input).unwrap(), original);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cancellation_before_decode_creates_no_scratch() {
        let directory = cancel_scratch("pre-cancel");
        let input = directory.join("owned-input.bin");
        std::fs::write(&input, b"caller retains this input").unwrap();
        let result = futures_executor::block_on(
            AudioSource::default().decode_disk_backed_in_with_cancellation(
                &input,
                48_000,
                &directory,
                || Err(EngineError::Cancelled),
            ),
        );
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        assert_eq!(std::fs::read(&input).unwrap(), b"caller retains this input");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[derive(Debug)]
    struct CancellingAudioIo {
        cancelled: Arc<AtomicBool>,
        reads: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[derive(Debug)]
    struct CancellingAudioDecoder {
        cancelled: Arc<AtomicBool>,
        reads: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait(?Send)]
    impl ferment_engine_media::AudioDecodeSession for CancellingAudioDecoder {
        async fn seek(
            &mut self,
            _: ferment_engine_media::MediaTime,
        ) -> ferment_engine_media::Result<()> {
            Ok(())
        }
        async fn decode_next(
            &mut self,
        ) -> ferment_engine_media::Result<Option<ferment_engine_media::PcmAudioFrame>> {
            let reads = self.reads.fetch_add(1, Ordering::SeqCst) + 1;
            if reads == 2 {
                self.cancelled.store(true, Ordering::SeqCst);
            }
            if reads > 100 {
                return Ok(None);
            }
            Ok(Some(ferment_engine_media::PcmAudioFrame::new(
                48_000,
                1,
                ferment_engine_media::MediaTime::ZERO,
                vec![0.125; 1024],
            )?))
        }
    }

    #[async_trait::async_trait(?Send)]
    impl ferment_engine_media::AudioIo for CancellingAudioIo {
        async fn probe(
            &self,
            source: &Path,
        ) -> ferment_engine_media::Result<ferment_engine_media::MediaInfo> {
            ferment_engine_media::AudioIo::probe(&ferment_engine_media::UnavailableAudioIo, source)
                .await
        }
        async fn open_audio_decoder(
            &self,
            _: &Path,
            _: u32,
        ) -> ferment_engine_media::Result<Box<dyn ferment_engine_media::AudioDecodeSession>>
        {
            Ok(Box::new(CancellingAudioDecoder {
                cancelled: self.cancelled.clone(),
                reads: self.reads.clone(),
            }))
        }
        async fn open_audio_encoder(
            &self,
            _: ferment_engine_media::AudioEncoderSettings,
        ) -> ferment_engine_media::Result<Box<dyn ferment_engine_media::AudioEncodeSession>>
        {
            unreachable!("decoding never requests an encoder")
        }
    }

    #[test]
    fn cancellation_during_native_frame_decode_stops_and_removes_partial_pcm() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../engine-media/tests/fixtures/generated/aac-lc-48000-stereo.m4a");
        let original = std::fs::read(&fixture).expect("pinned original native AAC fixture");
        let directory = cancel_scratch("cancel-native");
        let cancelled = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let source = AudioSource::new(Arc::new(CancellingAudioIo {
            cancelled: cancelled.clone(),
            reads: reads.clone(),
        }));
        let result = futures_executor::block_on(source.decode_disk_backed_in_with_cancellation(
            &fixture,
            48_000,
            &directory,
            || {
                if cancelled.load(Ordering::SeqCst) {
                    Err(EngineError::Cancelled)
                } else {
                    Ok(())
                }
            },
        ));
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert_eq!(
            reads.load(Ordering::SeqCst),
            2,
            "no read after the cancellation raised inside decoder"
        );
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        assert_eq!(std::fs::read(&fixture).unwrap(), original);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
