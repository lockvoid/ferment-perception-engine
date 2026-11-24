use crate::{
    Embedding, EmbeddingSpace, GlapLanguage, GlapTokenizer, check_cancelled, encode, token_input,
};
use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, MelSpec, MelSpectrogram, Normalization, Padding, Scale,
    Spectrum, TablePrecision, Window,
};
use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use std::borrow::Cow;

pub const GLAP_SAMPLE_RATE: u32 = 16000;
pub const GLAP_MAX_SAMPLES: usize = 160000;

const MEL: MelSpec = MelSpec {
    sample_rate: 16000.0,
    fft_size: 512,
    hop_length: 160,
    mel_bands: 64,
    f_min: 0.0,
    f_max: 8000.0,
    input_scale: 1.0,
    padding: Padding::Reflect,
    window: Window::HannPeriodic,
    window_length: None,
    dc_removal: false,
    pre_emphasis: None,
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Htk,
    filter_shape: FilterShape::HertzLinear,
    filter_norm: FilterNorm::None,
    table_precision: TablePrecision::Float32,
    compression: Compression::DecibelsAbsolute { amin: 1e-10 },
};

/// Planned, reusable mel frontend. No ten-second padding of short samples:
/// attention would attend to that padding and change the learned embedding.
pub struct GlapFrontend(MelSpectrogram);

impl GlapFrontend {
    pub fn new() -> Result<Self> {
        Ok(Self(MelSpectrogram::new(MEL)?))
    }

    pub fn prepare(&self, mono_16k: &[f32]) -> Result<Tensor> {
        if mono_16k.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        if mono_16k.len() > GLAP_MAX_SAMPLES || mono_16k.iter().any(|s| !s.is_finite()) {
            return Err(EngineError::invalid_config(
                "GLAP requires one finite mono 16 kHz window, at most 10 seconds",
            ));
        }
        let samples = if mono_16k.len() < 512 {
            let mut padded = mono_16k.to_vec();
            padded.resize(512, 0.0);
            Cow::Owned(padded)
        } else {
            Cow::Borrowed(mono_16k)
        };
        let mel = self.0.compute(&samples)?.clamped_below_max_by(120.0);
        Tensor::from_f32(vec![1, 64, mel.frame_count() as i64], mel.bands_major(None))
    }
}

pub struct Glap {
    backend: Backend,
    version: ModelVersion,
    frontend: GlapFrontend,
}

impl Glap {
    pub fn new(backend: Backend, version: ModelVersion) -> Result<Self> {
        Ok(Self {
            backend,
            version,
            frontend: GlapFrontend::new()?,
        })
    }

    /// Caller preserves the timestamp and valid sample count with this window.
    /// Long files are multiple windows, not an undocumented average vector.
    pub async fn audio_window(
        &self,
        mono_16k: &[f32],
        progress: &ProgressSink,
    ) -> Result<Embedding> {
        check_cancelled(progress)?;
        let mel = self.frontend.prepare(mono_16k)?;
        encode(
            &self.backend,
            ModelId::GlapAudio,
            &self.version,
            TensorMap::from([("mel".into(), mel)]),
            EmbeddingSpace::GlapWindowV1,
            progress,
        )
        .await
    }

    pub async fn text(
        &self,
        tokenizer: &GlapTokenizer,
        text: &str,
        language: GlapLanguage,
        progress: &ProgressSink,
    ) -> Result<Embedding> {
        check_cancelled(progress)?;
        encode(
            &self.backend,
            ModelId::GlapText,
            &self.version,
            token_input(tokenizer.encode(text, language)?)?,
            EmbeddingSpace::GlapWindowV1,
            progress,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audio_boundaries_do_not_invent_a_ten_second_tail() {
        let front = GlapFrontend::new().unwrap();
        assert_eq!(front.prepare(&[0.0; 1]).unwrap().shape, [1, 64, 4]);
        assert_eq!(front.prepare(&[0.0; 4000]).unwrap().shape, [1, 64, 26]);
        assert_eq!(
            front.prepare(&vec![0.0; GLAP_MAX_SAMPLES]).unwrap().shape,
            [1, 64, 1001]
        );
        for input in [vec![], vec![f32::NAN], vec![0.0; GLAP_MAX_SAMPLES + 1]] {
            assert!(front.prepare(&input).is_err());
        }
    }
}
