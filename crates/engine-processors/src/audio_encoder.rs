use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ferment_engine_audio::resample_sinc;
use ferment_engine_core::{
    AudioEncoding, AudioEncodingPreset, AudioEncodingResult, AudioFileType, EngineError,
    ProgressSink, Result,
};
use ferment_engine_media::{Codec, ContainerFormat};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::audio_source::AudioSource;

#[derive(Debug, Clone, Default)]
pub struct AudioEncoder {
    backends: Option<crate::EncodeBackends>,
}

impl AudioEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the platform seams that let the M4A path re-encode arbitrary
    /// sources to AAC-in-m4a instead of only passing target files through.
    #[must_use]
    pub fn with_backends(mut self, backends: crate::EncodeBackends) -> Self {
        self.backends = Some(backends);
        self
    }

    fn validate_request(
        output: &Path,
        preset: AudioEncodingPreset,
        progress: &ProgressSink,
    ) -> Result<PathBuf> {
        ensure_not_cancelled(progress)?;
        if preset.sample_rate == 0 || !(1..=2).contains(&preset.channels) {
            return Err(EngineError::invalid_config(
                "audio preset requires a positive sample rate and one or two channels",
            ));
        }
        if output.exists() {
            return Err(EngineError::Encode(format!(
                "refusing to overwrite '{}'",
                output.display()
            )));
        }
        let partial = partial_path(output)?;
        if partial.exists() {
            return Err(EngineError::Encode(format!(
                "partial output already exists at '{}'",
                partial.display()
            )));
        }
        Ok(partial)
    }

    async fn encode_wav(
        &self,
        source: &Path,
        output: &Path,
        preset: AudioEncodingPreset,
        progress: ProgressSink,
    ) -> Result<AudioEncodingResult> {
        let partial = Self::validate_request(output, preset, &progress)?;

        progress.report(0.05, "inspecting source").await;
        ensure_not_cancelled(&progress)?;
        let source_spec = WavReaderMetadata::read(source).ok();
        if source_spec.is_some_and(|metadata| {
            metadata.sample_rate == preset.sample_rate
                && metadata.channels == preset.channels
                && metadata.duration <= preset.max_duration
        }) {
            fs::copy(source, &partial)?;
            if let Err(error) = ensure_not_cancelled(&progress) {
                let _ = fs::remove_file(&partial);
                return Err(error);
            }
            fs::rename(&partial, output)?;
            let metadata = fs::metadata(output)?;
            progress.report(1.0, "done").await;
            return Ok(AudioEncodingResult {
                output_path: output.to_path_buf(),
                bytes: metadata.len(),
                duration: source_spec.unwrap().duration,
                did_passthrough: true,
            });
        }

        progress.report(0.20, "decoding").await;
        ensure_not_cancelled(&progress)?;
        let frames = AudioSource::default().decode(source).await?;
        ensure_not_cancelled(&progress)?;
        let mut samples = if frames.sample_rate == preset.sample_rate {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, preset.sample_rate)?
        };
        let max_samples = (preset.max_duration.max(0.0) * f64::from(preset.sample_rate)) as usize;
        samples.truncate(max_samples);
        if samples.is_empty() {
            return Err(EngineError::NoAudioData);
        }

        progress.report(0.55, "encoding wav").await;
        let write_result = (|| -> Result<()> {
            ensure_not_cancelled(&progress)?;
            let spec = WavSpec {
                channels: preset.channels,
                sample_rate: preset.sample_rate,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };
            let mut writer = WavWriter::create(&partial, spec)
                .map_err(|error| EngineError::Encode(error.to_string()))?;
            for (index, sample) in samples.iter().enumerate() {
                if index % 4_096 == 0 {
                    ensure_not_cancelled(&progress)?;
                }
                let quantized = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
                for _ in 0..preset.channels {
                    writer
                        .write_sample(quantized)
                        .map_err(|error| EngineError::Encode(error.to_string()))?;
                }
            }
            writer
                .finalize()
                .map_err(|error| EngineError::Encode(error.to_string()))?;
            Ok(())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&partial);
            return Err(error);
        }
        if let Err(error) = ensure_not_cancelled(&progress) {
            let _ = fs::remove_file(&partial);
            return Err(error);
        }
        fs::rename(&partial, output)?;
        let bytes = fs::metadata(output)?.len();
        progress.report(1.0, "done").await;
        Ok(AudioEncodingResult {
            output_path: output.to_path_buf(),
            bytes,
            duration: samples.len() as f64 / f64::from(preset.sample_rate),
            did_passthrough: false,
        })
    }

    async fn encode_m4a(
        &self,
        source: &Path,
        output: &Path,
        preset: AudioEncodingPreset,
        progress: ProgressSink,
    ) -> Result<AudioEncodingResult> {
        let partial = Self::validate_request(output, preset, &progress)?;
        progress.report(0.05, "inspecting source").await;
        ensure_not_cancelled(&progress)?;

        // Container parsing and AAC identity stay in engine-media. Until that
        // crate exposes mux/finalize-to-path, the only lossless M4A operation
        // available here is the Swift-compatible target-config passthrough.
        if let Ok(info) = ferment_engine_media::demux::probe(source) {
            let audio = info.audio_streams().next();
            let duration = audio
                .and_then(|stream| stream.duration)
                .or(info.duration)
                .map_or(0.0, ferment_engine_media::MediaTime::seconds);
            let is_target = info.container == ContainerFormat::Mp4
                && audio.is_some_and(|stream| {
                    stream.codec == Codec::Aac
                        && stream.sample_rate == Some(preset.sample_rate)
                        && stream.channels == Some(preset.channels)
                })
                && duration > 0.0
                && duration <= preset.max_duration;
            if is_target {
                fs::copy(source, &partial)?;
                if let Err(error) = ensure_not_cancelled(&progress) {
                    let _ = fs::remove_file(&partial);
                    return Err(error);
                }
                fs::rename(&partial, output)?;
                let bytes = fs::metadata(output)?.len();
                progress.report(1.0, "done").await;
                return Ok(AudioEncodingResult {
                    output_path: output.to_path_buf(),
                    bytes,
                    duration,
                    did_passthrough: true,
                });
            }
        }

        let Some(backends) = &self.backends else {
            return Err(EngineError::Unsupported(
                "AAC encode sessions exist in engine-media, but this encoder has no platform backends attached"
                    .into(),
            ));
        };
        let _ = partial;
        let (bytes, duration) = crate::encode_pipeline::transcode_audio_only(
            &backends.audio,
            &backends.muxer,
            source,
            output,
            &progress,
        )
        .await?;
        progress.report(1.0, "done").await;
        Ok(AudioEncodingResult {
            output_path: output.to_path_buf(),
            bytes,
            duration,
            did_passthrough: false,
        })
    }
}

#[async_trait(?Send)]
impl AudioEncoding for AudioEncoder {
    async fn encode(
        &self,
        source_path: &Path,
        output_path: &Path,
        preset: AudioEncodingPreset,
        progress: ProgressSink,
    ) -> Result<AudioEncodingResult> {
        match preset.file_type {
            AudioFileType::Wav => {
                self.encode_wav(source_path, output_path, preset, progress)
                    .await
            }
            AudioFileType::M4a => {
                self.encode_m4a(source_path, output_path, preset, progress)
                    .await
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WavReaderMetadata {
    sample_rate: u32,
    channels: u16,
    duration: f64,
}

impl WavReaderMetadata {
    fn read(path: &Path) -> Result<Self> {
        let reader =
            hound::WavReader::open(path).map_err(|error| EngineError::Decode(error.to_string()))?;
        let spec = reader.spec();
        Ok(Self {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            duration: reader.duration() as f64 / f64::from(spec.sample_rate),
        })
    }
}

fn partial_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| EngineError::invalid_config("output path has no UTF-8 file name"))?;
    Ok(output.with_file_name(format!("{name}.partial")))
}

fn ensure_not_cancelled(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ferment_engine_audio::write_wav_f32;

    use super::*;

    #[test]
    fn wav_passthrough_is_atomic_and_exact() {
        let directory = std::env::temp_dir();
        let source = directory.join(format!("audio-encoder-source-{}.wav", std::process::id()));
        let output = directory.join(format!("audio-encoder-output-{}.wav", std::process::id()));
        let _ = fs::remove_file(&source);
        let _ = fs::remove_file(&output);
        write_wav_f32(&source, &[0.0; 8_000], 8_000).unwrap();
        let preset = AudioEncodingPreset {
            sample_rate: 8_000,
            channels: 1,
            bit_rate: 128_000,
            max_duration: 10.0,
            file_type: AudioFileType::Wav,
        };
        let result = futures_executor::block_on(AudioEncoder::new().encode(
            &source,
            &output,
            preset,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(result.did_passthrough);
        assert_eq!(fs::read(&source).unwrap(), fs::read(&output).unwrap());
        fs::remove_file(source).unwrap();
        fs::remove_file(output).unwrap();
    }
}
