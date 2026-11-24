use std::path::Path;

use async_trait::async_trait;
use ferment_engine_audio::AudioFrames;
use ferment_engine_core::{Result, WaveformExtracting, WaveformResult};

use crate::audio_source::AudioSource;

#[derive(Debug, Clone, Copy, Default)]
pub struct WaveformExtractor;

impl WaveformExtractor {
    pub const SAMPLE_RATE: u32 = 22_050;
    /// Bound display-cache size for long recordings. Metadata always reports
    /// the actual bin width, so the whole source remains correctly timed.
    pub const MAX_BINS: usize = 1_000_000;

    pub async fn extract_with_cache(
        &self,
        path: &Path,
        sample_interval: f64,
        cache: Option<&ferment_engine_audio::derived_cache::DerivedFileCache>,
    ) -> Result<WaveformResult> {
        self.extract_frames(
            AudioSource::default()
                .resampled(path, Self::SAMPLE_RATE, None, cache)
                .await?,
            sample_interval,
        )
    }

    pub fn extract_frames(
        &self,
        frames: AudioFrames,
        sample_interval: f64,
    ) -> Result<WaveformResult> {
        if !sample_interval.is_finite() || sample_interval <= 0.0 {
            return Err(ferment_engine_core::EngineError::invalid_config(
                "sample interval must be finite and greater than zero",
            ));
        }
        let samples = if frames.sample_rate == Self::SAMPLE_RATE {
            frames.samples
        } else {
            ferment_engine_audio::streaming_resampler::resample_current(
                &frames.samples,
                frames.sample_rate,
                Self::SAMPLE_RATE,
            )?
        };
        let duration = samples.len() as f64 / f64::from(Self::SAMPLE_RATE);
        if samples.is_empty() {
            return Ok(WaveformResult {
                duration,
                rms: Vec::new(),
                peaks: Vec::new(),
                sample_rate: Self::SAMPLE_RATE,
                channels: 1,
                samples_per_pixel: 1,
                bits: 8,
            });
        }

        let samples_per_pixel = (sample_interval * f64::from(Self::SAMPLE_RATE))
            .round()
            .clamp(1.0, samples.len() as f64) as usize;
        let samples_per_pixel = samples_per_pixel.max(samples.len().div_ceil(Self::MAX_BINS));
        let output_length = samples.len().div_ceil(samples_per_pixel);
        let mut raw_rms = Vec::with_capacity(output_length);
        let mut peaks = Vec::with_capacity(output_length);
        // Include the final partial bin: dropping it loses a short clip or
        // the transient at its tail. Sanitize non-finite decoder input once.
        for block in samples.chunks(samples_per_pixel) {
            let (energy, peak) = block
                .iter()
                .fold((0.0_f64, 0.0_f32), |(energy, peak), sample| {
                    let value = if sample.is_finite() {
                        sample.abs().min(1.0)
                    } else {
                        0.0
                    };
                    (energy + f64::from(value).powi(2), peak.max(value))
                });
            raw_rms.push((energy / block.len() as f64).sqrt() as f32);
            peaks.push(peak);
        }

        let mut sorted = raw_rms.clone();
        sorted.sort_by(f32::total_cmp);
        let peak = sorted[((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1)];
        let floor = sorted[((sorted.len() as f32 * 0.10) as usize).min(sorted.len() - 1)];
        let range = peak - floor;
        let rms = raw_rms
            .into_iter()
            .map(|value| {
                if range > 0.0 {
                    ((value - floor) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();

        Ok(WaveformResult {
            duration,
            rms,
            peaks,
            sample_rate: Self::SAMPLE_RATE,
            channels: 1,
            samples_per_pixel,
            bits: 8,
        })
    }
}

#[async_trait(?Send)]
impl WaveformExtracting for WaveformExtractor {
    async fn extract(&self, path: &Path, sample_interval: f64) -> Result<WaveformResult> {
        self.extract_with_cache(path, sample_interval, None).await
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn editing_peaks_keep_quiet_audio_and_the_last_partial_bin() {
        let mut samples = vec![0.01; 22_051];
        samples[512] = 1.0;
        samples[22_050] = 0.75;
        let waveform = WaveformExtractor
            .extract_frames(
                AudioFrames::new(samples, WaveformExtractor::SAMPLE_RATE),
                0.001,
            )
            .unwrap();
        assert_eq!(waveform.samples_per_pixel, 22);
        assert_eq!(waveform.peaks.len(), 22_051_usize.div_ceil(22));
        assert_eq!(waveform.peaks[0], 0.01);
        assert_eq!(waveform.peaks[512 / 22], 1.0);
        assert_eq!(waveform.peaks.last(), Some(&0.75));
        assert_eq!(waveform.rms.len(), waveform.peaks.len());
    }

    #[test]
    fn peaks_keep_constant_amplitude_and_reject_non_finite_input() {
        let waveform = WaveformExtractor
            .extract_frames(
                AudioFrames::new(vec![0.25; 2205], WaveformExtractor::SAMPLE_RATE),
                0.001,
            )
            .unwrap();
        assert!(waveform.peaks.iter().all(|p| *p == 0.25));
        let waveform = WaveformExtractor
            .extract_frames(
                AudioFrames::new(
                    vec![f32::NAN, f32::INFINITY, -0.5],
                    WaveformExtractor::SAMPLE_RATE,
                ),
                1.0,
            )
            .unwrap();
        assert_eq!(waveform.peaks, [0.5]);
        assert!(waveform.rms.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn high_detail_requests_keep_the_entire_source_with_bounded_cache_size() {
        let samples = vec![0.25; WaveformExtractor::MAX_BINS + 1];
        let waveform = WaveformExtractor
            .extract_frames(
                AudioFrames::new(samples, WaveformExtractor::SAMPLE_RATE),
                f64::EPSILON,
            )
            .unwrap();
        assert!(waveform.peaks.len() <= WaveformExtractor::MAX_BINS);
        assert_eq!(waveform.samples_per_pixel, 2);
        assert_eq!(waveform.peaks.last(), Some(&0.25));
        assert_eq!(
            waveform.peaks.len(),
            (WaveformExtractor::MAX_BINS + 1).div_ceil(2)
        );
    }

    fn audio_fixture(name: &str) -> Option<PathBuf> {
        let configured = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
            .map(PathBuf::from)
            .map(|root| root.join("Audio").join(name));
        let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures/Audio")
            .join(name);
        configured
            .into_iter()
            .chain([sibling])
            .find(|path| path.is_file())
    }

    // WaveformExtractorTests.swift

    #[test]
    fn test_extract_real_file_produces_normalized_envelope() {
        let path = audio_fixture("techno-1.wav").expect(
            "Required real techno-1.wav is missing: set FERMENT_ENGINE_FIXTURES_DIR to FermentEngine/Tests/Fixtures",
        );
        let waveform = futures_executor::block_on(WaveformExtractor.extract(&path, 0.025))
            .expect("real fixture must decode");
        assert_eq!(waveform.sample_rate, 22_050);
        assert!(waveform.duration > 0.0);
        assert!(!waveform.rms.is_empty());
        assert!(
            waveform.rms.iter().all(|value| (0.0..=1.0).contains(value)),
            "waveform must be normalized to 0...1"
        );
    }

    #[test]
    fn normalized_envelope_is_bounded() {
        let samples = (0..44_100)
            .map(|index| {
                let amplitude = index as f32 / 44_100.0;
                amplitude * (index as f32 * 0.1).sin()
            })
            .collect();
        let waveform = WaveformExtractor
            .extract_frames(AudioFrames::new(samples, 44_100), 0.025)
            .unwrap();
        assert_eq!(waveform.sample_rate, 22_050);
        assert!(!waveform.rms.is_empty());
        assert!(waveform.rms.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn empty_input_keeps_typed_metadata() {
        let waveform = WaveformExtractor
            .extract_frames(AudioFrames::new(Vec::new(), 48_000), 0.025)
            .unwrap();
        assert!(waveform.rms.is_empty());
        assert_eq!(waveform.samples_per_pixel, 1);
    }
}
