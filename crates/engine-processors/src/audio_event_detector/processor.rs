use std::path::Path;

use async_trait::async_trait;
use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{
    AudioEventDetecting, AudioEventDetectionResult, AudioEventFrameProbabilities, EngineError,
    ProgressSink, Result,
};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

use super::Cmvn;
use super::config::{
    FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES, FRAME_SHIFT_SECONDS, INPUT_DIM, MEL_SPEC,
    NUM_EVENT_CLASSES, SAMPLE_RATE, WINDOW_FRAMES, door_windows, frames_for_samples,
    samples_for_frames,
};
use crate::audio_source::{ArrayAudioSampleSource, AudioSampleSource, AudioSource};

/// Backend-neutral FireRedVAD-AED processor over the `fireredvad_aed` door
/// (`ferment-models-export`): `feat [1, 80, 1, 6000]`, `mask [1, 1, 1, 6000]`
/// → `probs [1, 3, 1, 6000]`, the same on every backend.
///
/// The host computes the reference's `AudioFeat` — kaldi fbank
/// ([`MEL_SPEC`]), then the checkpoint's [`Cmvn`] — and runs the door in
/// windows that overlap by the model's receptive field
/// ([`door_windows`]), so every frame is the whole-sequence answer.
#[derive(Debug, Clone)]
pub struct AudioEventDetector {
    backend: Backend,
    version: ModelVersion,
}

impl AudioEventDetector {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    #[must_use]
    pub fn with_version(mut self, version: ModelVersion) -> Self {
        self.version = version;
        self
    }

    /// The door's `feat` for frames `from..from + count` of 16 kHz
    /// unit-scale `samples`: the kaldi fbank of exactly their audio, then the
    /// CMVN — `[count, 80]`, frames major.
    pub fn features(samples: &[f32], from: usize, count: usize) -> Result<Vec<f32>> {
        let start = from * FRAME_SHIFT_SAMPLES;
        let frames = MEL_SPEC.compute(&samples[start..start + samples_for_frames(count)])?;
        let mut values = frames.values;
        if !Cmvn::apply(&mut values, count) {
            return Err(EngineError::InvalidOutput);
        }
        Ok(values)
    }

    pub async fn detect_audio(
        &self,
        audio: &AudioFrames,
        progress: ProgressSink,
    ) -> Result<AudioEventDetectionResult> {
        let samples = prepare_16khz(audio)?;
        self.detect_source(&ArrayAudioSampleSource::new(samples), progress)
            .await
    }

    /// Windowed event detection over mono 16 kHz samples. The published door's
    /// receptive-field overlap and kept frames are identical to detect_audio.
    pub async fn detect_source(
        &self,
        source: &dyn AudioSampleSource,
        progress: ProgressSink,
    ) -> Result<AudioEventDetectionResult> {
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        progress.report(0.05, "loading audio").await;
        if source.sample_count() < FRAME_LENGTH_SAMPLES {
            return Err(EngineError::AudioTooShort {
                samples: source.sample_count(),
                min_required: FRAME_LENGTH_SAMPLES,
            });
        }

        let frame_count = frames_for_samples(source.sample_count());
        let windows = door_windows(frame_count);
        let mut probabilities = vec![0.0_f32; frame_count * NUM_EVENT_CLASSES];
        for (index, window) in windows.iter().enumerate() {
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            let real = (frame_count - window.read_from).min(WINDOW_FRAMES);
            let count = samples_for_frames(real);
            let mut samples = vec![0.; count];
            if source.copy_samples(&mut samples, window.read_from * FRAME_SHIFT_SAMPLES, count)?
                != count
            {
                return Err(EngineError::decode(
                    "Event source ended before its window boundary",
                ));
            }
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            let feat = Self::features(&samples, 0, real)?;
            let answer = self.run_door(&feat, real).await?;
            for frame in window.keep_from..window.keep_to {
                let local = frame - window.read_from;
                for class in 0..NUM_EVENT_CLASSES {
                    probabilities[frame * NUM_EVENT_CLASSES + class] =
                        answer[class * WINDOW_FRAMES + local];
                }
            }
            progress
                .report(
                    0.05 + 0.85 * (index + 1) as f64 / windows.len() as f64,
                    format!("detecting events ({}/{})", index + 1, windows.len()),
                )
                .await;
        }

        let frame_probs = probabilities
            .chunks_exact(NUM_EVENT_CLASSES)
            .map(|row| AudioEventFrameProbabilities {
                speech: row[0],
                singing: row[1],
                music: row[2],
            })
            .collect();
        progress.report(1.0, "done").await;
        Ok(AudioEventDetectionResult {
            duration_seconds: source.sample_count() as f64 / f64::from(SAMPLE_RATE),
            frame_shift_seconds: FRAME_SHIFT_SECONDS,
            frame_probs,
        })
    }

    /// One window through the door: `real` frames of `feat` laid out bands
    /// major and zero-padded to the static frame count, the mask 1 on them —
    /// the answer `[3, 6000]`, classes major.
    async fn run_door(&self, feat: &[f32], real: usize) -> Result<Vec<f32>> {
        let mut bands = vec![0.0_f32; INPUT_DIM * WINDOW_FRAMES];
        for (frame, row) in feat.chunks_exact(INPUT_DIM).enumerate() {
            for (band, value) in row.iter().enumerate() {
                bands[band * WINDOW_FRAMES + frame] = *value;
            }
        }
        let mut mask = vec![0.0_f32; WINDOW_FRAMES];
        mask[..real].fill(1.0);
        let mut inputs = TensorMap::new();
        inputs.insert(
            "feat".into(),
            Tensor::from_f32(vec![1, INPUT_DIM as i64, 1, WINDOW_FRAMES as i64], bands)?,
        );
        inputs.insert(
            "mask".into(),
            Tensor::from_f32(vec![1, 1, 1, WINDOW_FRAMES as i64], mask)?,
        );
        let outputs = self
            .backend
            .infer(ModelId::FireRedVadAed, &self.version, inputs)
            .await?;
        let tensor = outputs
            .get("probs")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::FireRedVadAed.name().into(),
                key: "probs".into(),
            })?;
        let answer = tensor.to_f32()?;
        if answer.len() != NUM_EVENT_CLASSES * WINDOW_FRAMES {
            return Err(EngineError::InvalidOutput);
        }
        Ok(answer)
    }
}

#[async_trait(?Send)]
impl AudioEventDetecting for AudioEventDetector {
    async fn detect(
        &self,
        audio_path: &Path,
        progress: ProgressSink,
    ) -> Result<AudioEventDetectionResult> {
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }
        let audio = AudioSource::default().decode(audio_path).await?;
        self.detect_audio(&audio, progress).await
    }
}

fn prepare_16khz(audio: &AudioFrames) -> Result<Vec<f32>> {
    if audio.sample_rate == 0 {
        return Err(EngineError::invalid_config(
            "audio sample rate must be positive",
        ));
    }
    if audio.samples.is_empty() {
        return Err(EngineError::NoAudioData);
    }
    if audio.sample_rate == SAMPLE_RATE {
        return Ok(audio.samples.clone());
    }

    let expected_length = ((audio.samples.len() as u128 * u128::from(SAMPLE_RATE)
        + u128::from(audio.sample_rate) / 2)
        / u128::from(audio.sample_rate)) as usize;
    let mut samples = resample_sinc(&audio.samples, audio.sample_rate, SAMPLE_RATE)?;
    samples.resize(expected_length, 0.0);
    samples.truncate(expected_length);
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ferment_engine_infer::{InferenceBackend, ModelVersion};

    use super::super::config::{WINDOW_MARGIN, WINDOW_STRIDE};
    use super::*;

    /// Answers every window with each frame's GLOBAL index, recovered from a
    /// marker the test plants in the audio — so a frame kept from the wrong
    /// window, or from the wrong place in it, reads the wrong number.
    #[derive(Debug, Default)]
    struct IndexingBackend {
        calls: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl InferenceBackend for IndexingBackend {
        async fn infer(
            &self,
            model: ModelId,
            _version: &ModelVersion,
            mut inputs: TensorMap,
        ) -> Result<TensorMap> {
            assert_eq!(model, ModelId::FireRedVadAed);
            let feat = inputs.remove("feat").unwrap();
            let mask = inputs.remove("mask").unwrap();
            assert_eq!(feat.shape, vec![1, 80, 1, WINDOW_FRAMES as i64]);
            assert_eq!(mask.shape, vec![1, 1, 1, WINDOW_FRAMES as i64]);
            let (feat, mask) = (feat.to_f32()?, mask.to_f32()?);
            let index = {
                let mut calls = self.calls.lock().unwrap();
                calls.push((feat.clone(), mask.clone()));
                calls.len() - 1
            };
            let mut probs = vec![0.0_f32; 3 * WINDOW_FRAMES];
            for local in 0..WINDOW_FRAMES {
                let global = (index * WINDOW_STRIDE).saturating_sub(WINDOW_MARGIN) + local;
                probs[local] = global as f32;
                probs[WINDOW_FRAMES + local] = mask[local];
                probs[2 * WINDOW_FRAMES + local] = index as f32;
            }
            let mut outputs = TensorMap::new();
            outputs.insert(
                "probs".into(),
                Tensor::from_f32(vec![1, 3, 1, WINDOW_FRAMES as i64], probs)?,
            );
            Ok(outputs)
        }
    }

    #[test]
    fn every_frame_is_kept_from_the_window_that_saw_its_context() {
        let backend = Arc::new(IndexingBackend::default());
        let detector = AudioEventDetector::new(backend.clone());
        let frames = 2 * WINDOW_STRIDE + 700;
        let audio = AudioFrames::new(vec![0.01; samples_for_frames(frames)], SAMPLE_RATE);
        let result =
            futures_executor::block_on(detector.detect_audio(&audio, ProgressSink::noop()))
                .unwrap();
        assert_eq!(result.frame_probs.len(), frames);
        assert_eq!(backend.calls.lock().unwrap().len(), 3);
        for (frame, row) in result.frame_probs.iter().enumerate() {
            assert_eq!(row.speech, frame as f32, "frame {frame}");
            assert_eq!(row.singing, 1.0, "frame {frame} kept from the masked tail");
            assert_eq!(row.music, (frame / WINDOW_STRIDE) as f32, "frame {frame}");
        }
        let calls = backend.calls.lock().unwrap();
        let last_real = frames - (2 * WINDOW_STRIDE - WINDOW_MARGIN);
        assert!(calls[2].1[..last_real].iter().all(|value| *value == 1.0));
        assert!(calls[2].1[last_real..].iter().all(|value| *value == 0.0));
        for band in 0..80 {
            let tail = &calls[2].0[band * WINDOW_FRAMES + last_real..(band + 1) * WINDOW_FRAMES];
            assert!(tail.iter().all(|value| *value == 0.0), "band {band}");
        }
    }

    #[test]
    fn short_audio_is_rejected_before_inference() {
        let backend = Arc::new(IndexingBackend::default());
        let detector = AudioEventDetector::new(backend.clone());
        let audio = AudioFrames::new(vec![0.0; FRAME_LENGTH_SAMPLES - 1], SAMPLE_RATE);
        let error = futures_executor::block_on(detector.detect_audio(&audio, ProgressSink::noop()))
            .unwrap_err();
        assert!(matches!(error, EngineError::AudioTooShort { .. }));
        assert!(backend.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn cancellation_prevents_inference() {
        let backend = Arc::new(IndexingBackend::default());
        let detector = AudioEventDetector::new(backend.clone());
        let audio = AudioFrames::new(vec![0.0; FRAME_LENGTH_SAMPLES], SAMPLE_RATE);
        let progress = ProgressSink::noop();
        progress.cancel();
        let error =
            futures_executor::block_on(detector.detect_audio(&audio, progress)).unwrap_err();
        assert!(matches!(error, EngineError::Cancelled));
        assert!(backend.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn features_are_finite_on_digital_silence() {
        let samples = vec![0.0; FRAME_LENGTH_SAMPLES];
        let feat = AudioEventDetector::features(&samples, 0, 1).unwrap();
        assert_eq!(feat.len(), 80);
        assert!(feat.iter().all(|value| value.is_finite()));
    }
}
