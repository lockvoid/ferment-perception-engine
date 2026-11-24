//! Minimal WAV writer for 16-bit PCM mono. Mirrors iOS CLI `writeWAV` —
//! reads from `[f32]` samples, clamps to int16 range, writes a standard
//! RIFF/WAVE file.
//!
//! For higher-fidelity intermediate writes (e.g. denoiser output we'll
//! re-process) use `write_wav_f32`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ferment_engine_core::{EngineError, Result};

/// Write mono `f32` samples (clamped to `[-1, 1]`) as 16-bit PCM mono WAV.
pub fn write_wav_pcm16(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref()).map_err(EngineError::Io)?);

    let n = u32::try_from(samples.len())
        .map_err(|_| EngineError::invalid_config("too many samples for WAV"))?;
    let bytes_per_sample = 2u32;
    let byte_rate = sample_rate * bytes_per_sample;
    let block_align: u16 = bytes_per_sample as u16;
    let data_bytes = n * bytes_per_sample;
    let chunk_size = 36u32 + data_bytes;

    writer.write_all(b"RIFF").map_err(EngineError::Io)?;
    writer
        .write_all(&chunk_size.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer.write_all(b"WAVE").map_err(EngineError::Io)?;

    writer.write_all(b"fmt ").map_err(EngineError::Io)?;
    writer
        .write_all(&16u32.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(EngineError::Io)?; // PCM
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(EngineError::Io)?; // mono
    writer
        .write_all(&sample_rate.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&byte_rate.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&block_align.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&16u16.to_le_bytes())
        .map_err(EngineError::Io)?; // bits/sample

    writer.write_all(b"data").map_err(EngineError::Io)?;
    writer
        .write_all(&data_bytes.to_le_bytes())
        .map_err(EngineError::Io)?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let q = (clamped * f32::from(i16::MAX)).round() as i16;
        writer
            .write_all(&q.to_le_bytes())
            .map_err(EngineError::Io)?;
    }

    writer.flush().map_err(EngineError::Io)?;
    Ok(())
}

/// Write mono `f32` samples as 32-bit float WAV (lossless). Used internally
/// when chaining processors so we don't lose precision through int16.
pub fn write_wav_f32(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref()).map_err(EngineError::Io)?);

    let n = u32::try_from(samples.len())
        .map_err(|_| EngineError::invalid_config("too many samples for WAV"))?;
    let bytes_per_sample = 4u32;
    let byte_rate = sample_rate * bytes_per_sample;
    let block_align: u16 = bytes_per_sample as u16;
    let data_bytes = n * bytes_per_sample;
    let chunk_size = 36u32 + data_bytes;

    writer.write_all(b"RIFF").map_err(EngineError::Io)?;
    writer
        .write_all(&chunk_size.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer.write_all(b"WAVE").map_err(EngineError::Io)?;

    writer.write_all(b"fmt ").map_err(EngineError::Io)?;
    writer
        .write_all(&16u32.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&3u16.to_le_bytes())
        .map_err(EngineError::Io)?; // IEEE float
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&sample_rate.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&byte_rate.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&block_align.to_le_bytes())
        .map_err(EngineError::Io)?;
    writer
        .write_all(&32u16.to_le_bytes())
        .map_err(EngineError::Io)?;

    writer.write_all(b"data").map_err(EngineError::Io)?;
    writer
        .write_all(&data_bytes.to_le_bytes())
        .map_err(EngineError::Io)?;

    for &s in samples {
        writer
            .write_all(&s.to_le_bytes())
            .map_err(EngineError::Io)?;
    }

    writer.flush().map_err(EngineError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::loader::decode_path;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ferment_wav_test_{name}_{}.wav",
            std::process::id()
        ))
    }

    fn sine(freq: f32, sr: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn pcm16_round_trips_through_loader() {
        let path = temp_path("pcm16_round_trip");
        let _ = std::fs::remove_file(&path);
        let samples = sine(440.0, 16_000, 16_000);

        write_wav_pcm16(&path, &samples, 16_000).unwrap();
        let loaded = decode_path(&path).unwrap();

        assert_eq!(loaded.sample_rate, 16_000);
        assert_eq!(loaded.samples.len(), samples.len());

        // 16-bit quantization — error bounded by 1 / 32768 ≈ 3e-5
        let mut max_err = 0.0f32;
        for (a, b) in samples.iter().zip(loaded.samples.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "round-trip error {max_err} too large");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn float32_round_trips_lossless() {
        let path = temp_path("float32_round_trip");
        let _ = std::fs::remove_file(&path);
        let samples = sine(440.0, 22_050, 22_050);

        write_wav_f32(&path, &samples, 22_050).unwrap();
        let loaded = decode_path(&path).unwrap();

        assert_eq!(loaded.sample_rate, 22_050);
        assert_eq!(loaded.samples.len(), samples.len());

        for (a, b) in samples.iter().zip(loaded.samples.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clipping_caps_at_pcm_range() {
        let path = temp_path("clipping");
        let _ = std::fs::remove_file(&path);
        let samples = vec![5.0f32, -5.0, 0.5, -0.5, 0.0];
        write_wav_pcm16(&path, &samples, 16_000).unwrap();
        let loaded = decode_path(&path).unwrap();

        // First two should be clamped to ~±1.0 (exactly i16::MAX / i16::MAX = 1.0
        // after our scale, but symphonia normalizes by i16::MAX so we get
        // very close to 1.0).
        assert!(loaded.samples[0] > 0.99);
        assert!(loaded.samples[1] < -0.99);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_samples_writes_valid_header() {
        let path = temp_path("empty");
        let _ = std::fs::remove_file(&path);
        write_wav_pcm16(&path, &[], 16_000).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // Just header, no data
        assert_eq!(bytes.len(), 44);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        let _ = std::fs::remove_file(&path);
    }
}
