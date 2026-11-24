//! License-clean WAV decoding for native and wasm builds.
//!
//! Compressed formats are intentionally supplied through the platform audio
//! I/O adapter. The source Rust port used Symphonia, whose MPL-2.0 license is
//! outside Studio's shipped dependency allowlist.

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;

use bytes::Bytes;
use ferment_engine_core::{EngineError, Result};
use hound::{SampleFormat, WavReader};

use crate::AudioFrames;

/// Streaming, interleaved WAV decoder used by native playback paths.
///
/// The analysis-oriented [`decode_path`] API deliberately downmixes into one
/// owned mono buffer. Sequencing needs the complementary boundary: preserve
/// channels, seek by sample frame, and decode bounded slices without loading
/// the asset into memory. Keeping both implementations here ensures every WAV
/// consumer uses the same sample normalization policy.
pub struct WavStreamReader {
    reader: WavReader<BufReader<File>>,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    sample_format: SampleFormat,
}

impl std::fmt::Debug for WavStreamReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WavStreamReader")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("bits_per_sample", &self.bits_per_sample)
            .field("sample_format", &self.sample_format)
            .finish_non_exhaustive()
    }
}

impl WavStreamReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let reader =
            WavReader::open(path).map_err(|error| EngineError::decode(error.to_string()))?;
        let spec = reader.spec();
        if spec.sample_rate == 0 || spec.channels == 0 {
            return Err(EngineError::NoAudioData);
        }
        Ok(Self {
            reader,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Seek to a PCM sample frame (not an interleaved sample index).
    pub fn seek(&mut self, frame: u64) -> Result<()> {
        self.reader
            .seek(u32::try_from(frame).unwrap_or(u32::MAX))
            .map_err(|error| EngineError::decode(error.to_string()))
    }

    /// Decode complete interleaved sample frames into `output`.
    pub fn read_interleaved(&mut self, output: &mut [f32]) -> Result<usize> {
        let channels = usize::from(self.channels);
        let requested_samples = output.len() - output.len() % channels;
        let mut written = 0;
        match self.sample_format {
            SampleFormat::Float => {
                for destination in &mut output[..requested_samples] {
                    let Some(sample) = self.reader.samples::<f32>().next() else {
                        break;
                    };
                    *destination =
                        sample.map_err(|error| EngineError::decode(error.to_string()))?;
                    written += 1;
                }
            }
            SampleFormat::Int => {
                let scale = integer_sample_scale(self.bits_per_sample);
                for destination in &mut output[..requested_samples] {
                    let Some(sample) = self.reader.samples::<i32>().next() else {
                        break;
                    };
                    *destination = sample.map_err(|error| EngineError::decode(error.to_string()))?
                        as f32
                        / scale;
                    written += 1;
                }
            }
        }
        Ok(written / channels)
    }
}

pub fn decode_path(path: impl AsRef<Path>) -> Result<AudioFrames> {
    let reader = WavReader::open(path).map_err(|error| EngineError::decode(error.to_string()))?;
    decode_reader(reader)
}

pub fn decode_bytes(bytes: Bytes, extension: Option<&str>) -> Result<AudioFrames> {
    if extension.is_some_and(|extension| !extension.eq_ignore_ascii_case("wav")) {
        return Err(EngineError::Unsupported(format!(
            "compressed audio '.{}' requires a platform AudioIO adapter",
            extension.unwrap_or_default()
        )));
    }
    let reader = WavReader::new(Cursor::new(bytes))
        .map_err(|error| EngineError::decode(error.to_string()))?;
    decode_reader(reader)
}

fn decode_reader<R: Read + Seek>(mut reader: WavReader<R>) -> Result<AudioFrames> {
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 || spec.sample_rate == 0 {
        return Err(EngineError::NoAudioData);
    }

    let interleaved = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| EngineError::decode(error.to_string()))?,
        SampleFormat::Int => {
            let scale = integer_sample_scale(spec.bits_per_sample);
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|sample| sample as f32 / scale)
                        .map_err(|error| EngineError::decode(error.to_string()))
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    if interleaved.is_empty() {
        return Err(EngineError::NoAudioData);
    }
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok(AudioFrames::new(mono, spec.sample_rate))
}

fn integer_sample_scale(bits_per_sample: u16) -> f32 {
    (1_i64 << bits_per_sample.saturating_sub(1)).max(1) as f32
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    fn test_wav() -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for index in 0..16_000 {
                let sample =
                    (i16::MAX as f32 * (2.0 * PI * 440.0 * index as f32 / 16_000.0).sin()) as i16;
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn decodes_and_downmixes_wav_bytes() {
        let decoded = decode_bytes(Bytes::from(test_wav()), Some("wav")).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.samples.len(), 16_000);
        assert!((decoded.duration_secs() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_compressed_format_without_platform_adapter() {
        let error = decode_bytes(Bytes::new(), Some("m4a")).unwrap_err();
        assert!(matches!(error, EngineError::Unsupported(_)));
    }

    #[test]
    fn streaming_reader_preserves_channels_and_seeks_by_frame() {
        let path = std::env::temp_dir().join(format!(
            "ferment_wav_stream_reader_{}.wav",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, test_wav()).unwrap();

        let mut reader = WavStreamReader::open(&path).unwrap();
        assert_eq!(reader.sample_rate(), 16_000);
        assert_eq!(reader.channels(), 2);
        let mut beginning = [0.0; 4];
        assert_eq!(reader.read_interleaved(&mut beginning).unwrap(), 2);
        assert_eq!(beginning[0], beginning[1]);
        assert_eq!(beginning[2], beginning[3]);

        reader.seek(100).unwrap();
        let mut sought = [0.0; 2];
        assert_eq!(reader.read_interleaved(&mut sought).unwrap(), 1);
        assert_eq!(sought[0], sought[1]);
        assert_ne!(sought, beginning[..2]);

        drop(reader);
        let _ = std::fs::remove_file(path);
    }
}
