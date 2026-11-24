//! One prediction per fixed window: the checkpoint's front-end on the host
//! ([`features`]), then the door — `mel [1, 1, 72, 200]` in, a 192-d
//! unit-length speaker vector out (`ferment-models-export`,
//! `models/redimnet/model.json`).
//!
//! The L2 normalisation is inside the graph, so this module reads back a
//! direction. Cosine similarity is then a dot product, with no second place
//! that could forget to normalise.

use ferment_engine_audio::mel_spec::{
    Compression, FilterNorm, FilterShape, Normalization, Padding, Scale, Spectrum, TablePrecision,
    Window,
};
use ferment_engine_audio::{AudioFrames, MelSpec, resample_sinc};
use ferment_engine_core::{EngineError, Result, TimeRange};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

use super::windows::{
    self, EMBEDDING_DIMENSION, HOP_LENGTH, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES, WindowPlan,
};

pub const MEL_BANDS: usize = 72;

/// `redimnet2`'s `SpectralFeaturesTF` at the checkpoint's `TFMelBanks`, over
/// [`signal`]'s output: 400-sample symmetric Hamming frames every 160, cut at
/// the edges, in a 512-point transform; power; 72 HTK triangles linear in mel
/// over TensorFlow's bin grid; both clipped to `[ε, 1/ε]`, then `log(x + ε)`.
/// The same value as iOS `ReDimNetSpeakerEncoder.melSpec`.
pub const MEL_SPEC: MelSpec = MelSpec {
    sample_rate: SAMPLE_RATE as f64,
    fft_size: 512,
    hop_length: HOP_LENGTH,
    mel_bands: MEL_BANDS,
    f_min: 20.0,
    f_max: 7_600.0,
    input_scale: 1.0,
    padding: Padding::None,
    window: Window::HammingSymmetric,
    window_length: Some(400),
    dc_removal: false,
    pre_emphasis: None,
    spectrum: Spectrum::Power,
    normalization: Normalization::None,
    mel_scale: Scale::Htk,
    filter_shape: FilterShape::Tensorflow,
    filter_norm: FilterNorm::None,
    table_precision: TablePrecision::Float32,
    compression: Compression::TensorflowLog { epsilon: 1e-8 },
};
pub const SIGNAL_EPSILON: f64 = 1e-8;
pub const PRE_EMPHASIS: f32 = 0.97;

/// `TFMelBanks`' steps before the spectrum: the window normalised to zero mean
/// and unit population std (+ε), pre-emphasised with TensorFlow's reflected
/// first sample (`y[0] = x[0] − 0.97 · x[1]`), zero-padded by half a hop each
/// side.
#[must_use]
pub fn signal(window: &[f32]) -> Vec<f32> {
    let count = window.len() as f64;
    let mean = window.iter().map(|x| f64::from(*x)).sum::<f64>() / count;
    let variance = window
        .iter()
        .map(|x| (f64::from(*x) - mean).powi(2))
        .sum::<f64>()
        / count;
    let scale = 1.0 / (variance.sqrt() + SIGNAL_EPSILON);
    let normalized: Vec<f32> = window
        .iter()
        .map(|x| ((f64::from(*x) - mean) * scale) as f32)
        .collect();
    let pad = HOP_LENGTH / 2;
    let mut out = vec![0.0_f32; window.len() + 2 * pad];
    out[pad] = normalized[0] - PRE_EMPHASIS * normalized[1];
    for index in 1..normalized.len() {
        out[pad + index] = normalized[index] - PRE_EMPHASIS * normalized[index - 1];
    }
    out
}

/// The door's `mel`, bands major `[72 × 200]`: [`MEL_SPEC`] over the window's
/// [`signal`], each band minus its mean over the window.
///
/// # Errors
/// If the spectrogram cannot be computed.
pub fn features(window: &[f32]) -> Result<Vec<f32>> {
    let mel = MEL_SPEC.compute(&signal(window))?.centered_per_band();
    Ok(mel.bands_major(Some(WINDOW_FRAMES)))
}

/// One window's answer, and the instants it describes.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegmentEmbedding {
    pub start_time: f64,
    pub end_time: f64,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ReDimNetSpeakerEncoder {
    backend: Backend,
    version: ModelVersion,
}

impl ReDimNetSpeakerEncoder {
    /// The digest of the checkpoint the graph was exported from — the
    /// contract's `source.files`, pinned by `tests/redimnet_contract.rs`; iOS
    /// `ReDimNetSpeakerEncoder.encoderHash`.
    pub const ENCODER_HASH: &'static str =
        "0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a";

    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    /// The name that rides beside the hash. Both halves of a stored vector's
    /// header come from the model itself, never from a caller's guess.
    #[must_use]
    pub fn encoder_name() -> &'static str {
        ModelId::RedimnetSpeaker.name()
    }

    /// The checkpoint's digest ([`Self::ENCODER_HASH`]). A caller that STORES a
    /// vector stores this beside it: two checkpoints produce incomparable
    /// embeddings, so a reader that cannot name the encoder cannot honestly
    /// compare anything.
    #[must_use]
    pub fn encoder_hash() -> &'static str {
        Self::ENCODER_HASH
    }

    /// Resample to the graph's rate. The tiling and every offset downstream are
    /// in 16 kHz samples, so this has to happen before anything is planned.
    ///
    /// # Errors
    /// If the resampler fails or the audio is empty.
    pub fn to_model_rate(frames: AudioFrames) -> Result<Vec<f32>> {
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        if audio.is_empty() {
            return Err(EngineError::NoAudioData);
        }
        Ok(audio)
    }

    /// Every whole window of `spans`, scored.
    ///
    /// # Errors
    /// If a prediction fails or returns something that is not a finite
    /// [`EMBEDDING_DIMENSION`]-wide vector.
    pub async fn embed(
        &self,
        audio: &[f32],
        spans: &[TimeRange],
    ) -> Result<Vec<SpeakerSegmentEmbedding>> {
        let plans = windows::plan(spans, audio.len());
        let mut embeddings = Vec::with_capacity(plans.len());
        for plan in plans {
            embeddings.push(SpeakerSegmentEmbedding {
                start_time: plan.start_time,
                end_time: plan.end_time,
                vector: self.predict(audio, &plan).await?,
            });
        }
        Ok(embeddings)
    }

    /// One window through the graph.
    async fn predict(&self, audio: &[f32], plan: &WindowPlan) -> Result<Vec<f32>> {
        let window = &audio[plan.sample_offset..plan.sample_offset + WINDOW_SAMPLES];
        self.embedding(&features(window)?).await
    }

    /// The door on one window's bands-major `[72 × 200]` mel.
    ///
    /// # Errors
    /// If the prediction fails or returns something that is not a finite
    /// [`EMBEDDING_DIMENSION`]-wide vector.
    pub async fn embedding(&self, mel: &[f32]) -> Result<Vec<f32>> {
        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(
                vec![1, 1, MEL_BANDS as i64, WINDOW_FRAMES as i64],
                mel.to_vec(),
            )?,
        );
        let outputs = self
            .backend
            .infer(ModelId::RedimnetSpeaker, &self.version, inputs)
            .await?;

        let vector = outputs
            .get("embedding")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::RedimnetSpeaker.name().into(),
                key: "embedding".into(),
            })?
            .to_f32()?;
        if vector.len() != EMBEDDING_DIMENSION || !vector.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidOutput);
        }
        Ok(vector)
    }
}
