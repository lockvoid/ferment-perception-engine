//! One log-mel front-end for the whole fleet — the cross-port of iOS
//! `AudioFrontEnd/MelSpec.swift`, axis for axis.
//!
//! A [`MelSpec`] knows no model: each processor holds the VALUE its model was
//! trained on and calls [`MelSpec::compute`]. Every field that changes the
//! output is here, and the serialized shape is Swift's `Codable` one, so the
//! `frontend.spec` of a model's contract (`ferment-models-export`,
//! `models/<name>/model.json`) decodes into either runtime and is compared to
//! the processor's constant by a test, not by eye.
//!
//! The axes are the ones the estate's front-ends actually differ on: reflect
//! or constant centre padding or none (kaldi `snip_edges`); a periodic or
//! symmetric Hann or kaldi's `povey`, optionally shorter than the transform;
//! per-frame DC removal; pre-emphasis over the signal (NeMo) or per frame
//! (kaldi); magnitude or power, optionally ÷√n; Slaney or HTK mels whose
//! triangles are straight in hertz or in mel, area-normalised or not; tables
//! held at float32 or rounded to bfloat16 (GigaAM-v3); and six compressions. A
//! new axis is a new field, never a new implementation.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};

use ferment_engine_core::{EngineError, Result};

mod range;

/// `torch.stft(center: true)` semantics — `fft_size / 2` samples each side —
/// or none (kaldi `snip_edges`): frames `window_length` long, zero-padded to
/// `fft_size` for the transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Padding {
    Reflect,
    Constant,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Window {
    /// `torch.hann_window(n, periodic: true)` / librosa `fftbins=True`.
    HannPeriodic,
    /// `torch.hann_window(n, periodic: false)` — denominator `n − 1`.
    HannSymmetric,
    /// kaldi `povey`: the symmetric Hann raised to 0.85.
    Povey,
    /// scipy `windows.hamming(n)` / `torch.hamming_window(n, periodic: false)`:
    /// `0.54 − 0.46 · cos(2πi / (n − 1))`.
    HammingSymmetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PreEmphasis {
    /// NeMo / HF: `x[i] − α · x[i − 1]` over the whole signal before framing,
    /// `x[0]` kept.
    Signal {
        #[serde(rename = "_0")]
        alpha: f32,
    },
    /// kaldi: per frame, after its DC removal — `y[0] = x[0] − α · x[0]`, the
    /// frame's own first sample standing in for the one before it.
    Frame {
        #[serde(rename = "_0")]
        alpha: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Spectrum {
    /// |X| — torchaudio `power=1`.
    Magnitude,
    /// |X|² — torchaudio / librosa `power=2`.
    Power,
}

/// Applied to the magnitude before [`Spectrum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Normalization {
    None,
    /// torchaudio `normalized="frame_length"`: ÷ √fft_size.
    FrameLength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Scale {
    /// librosa / torchaudio `mel_scale="slaney"` (linear below 1 kHz).
    Slaney,
    /// torchaudio `mel_scale="htk"`: `2595 · log10(1 + hz / 700)`.
    Htk,
}

/// The triangles' straight lines — between the mel-spaced edges, drawn in
/// hertz or in mel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FilterShape {
    /// torchaudio / librosa: linear in hertz between the edge frequencies.
    HertzLinear,
    /// kaldi `MelBanks`: linear in mel between the edge mels.
    MelLinear,
    /// TensorFlow's `linear_to_mel_weight_matrix` as `redimnet2`'s
    /// `get_filterbanks` draws it: linear in mel, the bins placed at
    /// `linspace(0, sample_rate / 2, fft_size / 2)` — `fft_size / 2` of them, so
    /// the Nyquist bin carries no weight — and the DC bin none either.
    Tensorflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FilterNorm {
    /// torchaudio `norm=None`.
    None,
    /// librosa `norm="slaney"`: each triangle × 2 / (hz_right − hz_left).
    SlaneyArea,
}

/// The precision the window and the filterbank are HELD at. GigaAM-v3's
/// checkpoint carries its torchaudio tables rounded to bfloat16, its exports
/// lift them as they are, and the mel the model runs on is the rounded one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TablePrecision {
    Float32,
    Bfloat16,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Compression {
    /// `log(1 + multiplier · mel)`.
    Log1p { multiplier: f32 },
    /// librosa `power_to_db(ref: max, top_db:)` over the WHOLE spectrogram.
    Decibels {
        #[serde(rename = "topDB")]
        top_db: f32,
    },
    /// NeMo `log(mel + guard)`, the guard `log_zero_guard_value`.
    Log { guard: f32 },
    /// `log(clamp(mel, floor, ceiling))`.
    LogClamped { floor: f32, ceiling: f32 },
    /// kaldi `use_log_fbank`: `log(max(mel, floor))`.
    LogFloored { floor: f32 },
    /// torchaudio `AmplitudeToDB(stype: "power", ref_value: 1)`:
    /// `10 · log10(max(mel, amin))`, no reference — its `top_db` clamp is
    /// against the CALL's max and stays with the caller
    /// ([`MelFrames::clamped_below_max_by`]).
    DecibelsAbsolute { amin: f32 },
    /// `redimnet2`'s `SpectralFeaturesTF` + `TFMelBanks`: the spectrum clipped
    /// to `[ε, 1/ε]` before the projection, the projection clipped again, then
    /// `log(x + ε)`.
    TensorflowLog { epsilon: f32 },
}

/// The parameters of one log-mel front-end — and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MelSpec {
    pub sample_rate: f64,
    pub fft_size: usize,
    pub hop_length: usize,
    pub mel_bands: usize,
    pub f_min: f32,
    pub f_max: f32,
    /// Multiplied into the samples first: 1 for unit-scale audio, 32768 for a
    /// front-end defined on int16 samples (kaldi).
    pub input_scale: f32,
    pub padding: Padding,
    pub window: Window,
    /// The window's own length when shorter than `fft_size`: centred in the
    /// transform with zeros around it under centre padding
    /// (`torch.stft(win_length:)`), the frame itself under [`Padding::None`]
    /// (kaldi `frame_length`). `None` = `fft_size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_length: Option<usize>,
    /// kaldi `remove_dc_offset`: each frame centred on its own mean before
    /// pre-emphasis and the window.
    pub dc_removal: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_emphasis: Option<PreEmphasis>,
    pub spectrum: Spectrum,
    pub normalization: Normalization,
    pub mel_scale: Scale,
    pub filter_shape: FilterShape,
    pub filter_norm: FilterNorm,
    pub table_precision: TablePrecision,
    pub compression: Compression,
}

impl MelSpec {
    /// `1 + n / hop` frames for `n` samples under centre padding;
    /// `(n − frame_length) / hop + 1` without it, the frame `window_length`
    /// (or `fft_size`) long.
    #[must_use]
    pub fn frame_count(&self, samples: usize) -> usize {
        if samples == 0 {
            return 0;
        }
        if self.padding == Padding::None {
            let frame_length = self.window_length.unwrap_or(self.fft_size);
            return if samples >= frame_length {
                (samples - frame_length) / self.hop_length + 1
            } else {
                0
            };
        }
        (samples + 2 * (self.fft_size / 2)).saturating_sub(self.fft_size) / self.hop_length + 1
    }

    /// The spectrogram of `audio` (mono, already at `sample_rate`).
    pub fn compute(&self, audio: &[f32]) -> Result<MelFrames> {
        MelSpectrogram::new(*self)?.compute(audio)
    }

    /// The shared optional cache door; PCM identity includes the SRC version.
    pub fn perform(
        &self,
        audio: &[f32],
        source: Option<&crate::derived_cache::DerivedSource>,
        cache: Option<&crate::derived_cache::DerivedFileCache>,
    ) -> Result<MelFrames> {
        let (Some(source), Some(cache)) = (source, cache) else {
            return self.compute(audio);
        };
        let key = MelRequest {
            source: source.clone(),
            spec: *self,
        };
        let path = cache.fetch(&key, |partial| {
            let frames = self.compute(audio)?;
            let bytes: Vec<u8> = frames.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(partial, bytes)?;
            Ok(())
        })?;
        let bytes = std::fs::read(path)?;
        if self.mel_bands == 0 || bytes.len() % (4 * self.mel_bands) != 0 {
            return Err(EngineError::decode("cached mel is not whole frames"));
        }
        Ok(MelFrames {
            bands: self.mel_bands,
            values: bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MelRequest {
    pub source: crate::derived_cache::DerivedSource,
    pub spec: MelSpec,
}

impl crate::derived_cache::DerivedFileKey for MelRequest {
    const KIND: &'static str = "mel-f32le-frames-major";
    const VERSION: u32 = 3;
    fn source(&self) -> &crate::derived_cache::DerivedSource {
        &self.source
    }
    fn extension(&self) -> &str {
        "mel"
    }
}

/// A log-mel spectrogram: `frame_count × bands`, frames-major, one flat
/// buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFrames {
    pub bands: usize,
    pub values: Vec<f32>,
}

impl MelFrames {
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.values.len() / self.bands
    }

    #[must_use]
    pub fn frame(&self, index: usize) -> &[f32] {
        &self.values[index * self.bands..(index + 1) * self.bands]
    }

    /// The same frames bands-major (`[bands, width]`), the layout the speech
    /// graphs take as `[1, bands, T]`; columns past `frame_count` up to
    /// `padded_frames` are zero.
    #[must_use]
    pub fn bands_major(&self, padded_frames: Option<usize>) -> Vec<f32> {
        let frames = self.frame_count();
        let width = padded_frames.unwrap_or(frames).max(frames);
        let mut out = vec![0.0; self.bands * width];
        for (frame, row) in self.values.chunks_exact(self.bands).enumerate() {
            for (band, value) in row.iter().enumerate() {
                out[band * width + frame] = *value;
            }
        }
        out
    }

    /// torchaudio `AmplitudeToDB(top_db:)`'s clamp: nothing below this call's
    /// maximum minus `top_db` — per call, so on the frames, not in the spec.
    #[must_use]
    pub fn clamped_below_max_by(mut self, top_db: f32) -> Self {
        let Some(peak) = self.values.iter().copied().reduce(f32::max) else {
            return self;
        };
        let floor = peak - top_db;
        for value in &mut self.values {
            *value = value.max(floor);
        }
        self
    }

    /// `redimnet2`'s `TFMelBanks`: each band minus its mean over all the frames
    /// — per call, like the per-band normalisation below.
    #[must_use]
    pub fn centered_per_band(mut self) -> Self {
        let bands = self.bands;
        let frames = self.frame_count();
        if frames == 0 {
            return self;
        }
        for band in 0..bands {
            let mean = (0..frames)
                .map(|frame| self.values[frame * bands + band])
                .sum::<f32>()
                / frames as f32;
            for frame in 0..frames {
                self.values[frame * bands + band] -= mean;
            }
        }
        self
    }

    /// NeMo `normalize="per_feature"` over the first `valid_frames`: each band
    /// centred on its mean over those frames and divided by its
    /// Bessel-corrected std + `epsilon` (HF ADDS the epsilon, never clamps);
    /// frames past `valid_frames` are zeroed — the graph's attention mask is
    /// false there. A per-call operation, so it lives on the frames.
    #[must_use]
    pub fn normalized_per_band(mut self, valid_frames: usize, epsilon: f32) -> Self {
        let bands = self.bands;
        let valid = valid_frames.min(self.frame_count());
        for value in &mut self.values[valid * bands..] {
            *value = 0.0;
        }
        if valid < 2 {
            return self;
        }
        for band in 0..bands {
            let column = || (0..valid).map(|frame| frame * bands + band);
            let mean = column().map(|i| self.values[i]).sum::<f32>() / valid as f32;
            let mut sum_of_squares = 0.0_f32;
            for i in column() {
                self.values[i] -= mean;
                sum_of_squares += self.values[i] * self.values[i];
            }
            let scale = 1.0 / ((sum_of_squares / (valid - 1) as f32).sqrt() + epsilon);
            for i in column() {
                self.values[i] *= scale;
            }
        }
        self
    }
}

/// The engine behind [`MelSpec`] — one implementation for every spec. A frame
/// is `window.len()` samples: `fft_size` under centre padding, the window's
/// own length under [`Padding::None`] — kaldi's frame, zero-padded to the
/// transform. `realfft` is the true DFT at any even length, so GigaAM's 320
/// needs no second path here (iOS takes a direct DFT there only because
/// vDSP's radix-3/5 setups measured wrong).
pub struct MelSpectrogram {
    spec: MelSpec,
    bin_count: usize,
    window: Vec<f32>,
    /// `mel_bands × bin_count`, row-major.
    filterbank: Vec<f32>,
    /// The non-zero span of each row.
    spans: Vec<(usize, usize)>,
    transform: Arc<dyn RealToComplex<f32>>,
}

impl MelSpectrogram {
    pub fn new(spec: MelSpec) -> Result<Self> {
        if spec.fft_size < 2 || !spec.fft_size.is_multiple_of(2) {
            return Err(EngineError::invalid_config("fft_size must be even and ≥ 2"));
        }
        if spec.hop_length == 0 || spec.mel_bands == 0 {
            return Err(EngineError::invalid_config(
                "hop_length and mel_bands must be > 0",
            ));
        }
        let length = spec.window_length.unwrap_or(spec.fft_size);
        if length < 2 || length > spec.fft_size {
            return Err(EngineError::invalid_config(format!(
                "window_length {length} does not fit fft_size {}",
                spec.fft_size
            )));
        }
        let bin_count = spec.fft_size / 2 + 1;
        let window = rounded(make_window(&spec, length), spec.table_precision);
        let filterbank = rounded(make_filterbank(&spec, bin_count), spec.table_precision);
        let spans = filterbank
            .chunks_exact(bin_count)
            .map(|row| {
                let start = row.iter().position(|w| *w != 0.0).unwrap_or(0);
                let end = row.iter().rposition(|w| *w != 0.0).map_or(0, |i| i + 1);
                (start, end.max(start))
            })
            .collect();
        Ok(Self {
            spec,
            bin_count,
            window,
            filterbank,
            spans,
            transform: RealFftPlanner::<f32>::new().plan_fft_forward(spec.fft_size),
        })
    }

    #[must_use]
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    #[must_use]
    pub fn filterbank(&self) -> &[f32] {
        &self.filterbank
    }

    pub fn compute(&self, audio: &[f32]) -> Result<MelFrames> {
        if audio.is_empty() {
            return Ok(MelFrames {
                bands: self.spec.mel_bands,
                values: Vec::new(),
            });
        }
        let padded = self.pad(self.pre_emphasised(self.scaled(audio)));
        self.compute_padded(&padded)
    }

    fn compute_padded(&self, padded: &[f32]) -> Result<MelFrames> {
        let spec = &self.spec;
        let bands = spec.mel_bands;
        let frame_length = self.window.len();
        if padded.len() < frame_length {
            return Ok(MelFrames {
                bands,
                values: Vec::new(),
            });
        }
        let frame_count = (padded.len() - frame_length) / spec.hop_length + 1;

        let mut magnitude_scale = 1.0_f32;
        if spec.normalization == Normalization::FrameLength {
            magnitude_scale /= (spec.fft_size as f32).sqrt();
        }
        let frame_pre_emphasis = match spec.pre_emphasis {
            Some(PreEmphasis::Frame { alpha }) => Some(alpha),
            _ => None,
        };

        let mut input = self.transform.make_input_vec();
        let mut output = self.transform.make_output_vec();
        let mut scratch = self.transform.make_scratch_vec();
        let mut raw = vec![0.0_f32; frame_length];
        let mut bins = vec![0.0_f32; self.bin_count];
        let mut values = vec![0.0_f32; frame_count * bands];

        for (frame, energies) in values.chunks_exact_mut(bands).enumerate() {
            let samples = &padded[frame * spec.hop_length..frame * spec.hop_length + frame_length];
            if spec.dc_removal || frame_pre_emphasis.is_some() {
                // kaldi's `ProcessWindow`: DC removal, pre-emphasis, window.
                raw.copy_from_slice(samples);
                if spec.dc_removal {
                    let mean = raw.iter().sum::<f32>() / frame_length as f32;
                    for value in &mut raw {
                        *value -= mean;
                    }
                }
                if let Some(alpha) = frame_pre_emphasis {
                    for i in (1..frame_length).rev() {
                        raw[i] -= alpha * raw[i - 1];
                    }
                    raw[0] -= alpha * raw[0];
                }
                for ((slot, value), tap) in input.iter_mut().zip(&raw).zip(&self.window) {
                    *slot = value * tap;
                }
            } else {
                for ((slot, value), tap) in input.iter_mut().zip(samples).zip(&self.window) {
                    *slot = value * tap;
                }
            }
            // The transform's zero padding; `realfft` also scribbles on its input.
            input[frame_length..].fill(0.0);
            self.transform
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .map_err(|error| EngineError::StftFailed(error.to_string()))?;
            for (bin, complex) in bins.iter_mut().zip(&output) {
                let magnitude = complex.norm() * magnitude_scale;
                *bin = match spec.spectrum {
                    Spectrum::Magnitude => magnitude,
                    Spectrum::Power => magnitude * magnitude,
                };
                if let Compression::TensorflowLog { epsilon } = spec.compression {
                    *bin = bin.clamp(epsilon, 1.0 / epsilon);
                }
            }
            for (band, energy) in energies.iter_mut().enumerate() {
                let (start, end) = self.spans[band];
                let row = &self.filterbank[band * self.bin_count..(band + 1) * self.bin_count];
                *energy = row[start..end]
                    .iter()
                    .zip(&bins[start..end])
                    .map(|(weight, bin)| weight * bin)
                    .sum();
            }
        }

        self.compress(&mut values);
        Ok(MelFrames { bands, values })
    }

    fn scaled(&self, audio: &[f32]) -> Vec<f32> {
        if self.spec.input_scale == 1.0 {
            return audio.to_vec();
        }
        audio.iter().map(|x| x * self.spec.input_scale).collect()
    }

    /// `Signal`: `y[0] = x[0]`, `y[i] = x[i] − α · x[i − 1]` — on the real
    /// samples, before any padding, so the pad stays zero (HF's order).
    /// `Frame` is applied per frame in `compute`.
    fn pre_emphasised(&self, audio: Vec<f32>) -> Vec<f32> {
        let Some(PreEmphasis::Signal { alpha }) = self.spec.pre_emphasis else {
            return audio;
        };
        if audio.len() < 2 {
            return audio;
        }
        let mut out = Vec::with_capacity(audio.len());
        out.push(audio[0]);
        out.extend(audio.windows(2).map(|pair| pair[1] - alpha * pair[0]));
        out
    }

    fn pad(&self, audio: Vec<f32>) -> Vec<f32> {
        if self.spec.padding == Padding::None {
            return audio;
        }
        let pad = self.spec.fft_size / 2;
        let n = audio.len();
        let mut padded = vec![0.0_f32; n + 2 * pad];
        padded[pad..pad + n].copy_from_slice(&audio);
        if self.spec.padding == Padding::Reflect {
            for i in 0..pad {
                padded[pad - 1 - i] = audio[(i + 1).min(n - 1)];
                padded[pad + n + i] = audio[n.saturating_sub(2 + i)];
            }
        }
        padded
    }

    fn compress(&self, values: &mut [f32]) {
        match self.spec.compression {
            Compression::Log1p { multiplier } => {
                for value in values {
                    *value = (multiplier * *value).ln_1p();
                }
            }
            Compression::Decibels { top_db } => {
                // librosa power_to_db: 10·log10(max(v, amin) / max(ref, amin)),
                // ref = max, clipped at −top_db.
                let amin = 1e-10_f32;
                let reference = values.iter().copied().fold(0.0_f32, f32::max).max(amin);
                let reference_db = 10.0 * reference.log10();
                for value in values {
                    *value = (10.0 * value.max(amin).log10() - reference_db).max(-top_db);
                }
            }
            Compression::Log { guard } => {
                for value in values {
                    *value = (*value + guard).ln();
                }
            }
            Compression::LogClamped { floor, ceiling } => {
                for value in values {
                    *value = value.clamp(floor, ceiling).ln();
                }
            }
            Compression::LogFloored { floor } => {
                for value in values {
                    *value = value.max(floor).ln();
                }
            }
            Compression::DecibelsAbsolute { amin } => {
                for value in values {
                    *value = 10.0 * value.max(amin).log10();
                }
            }
            Compression::TensorflowLog { epsilon } => {
                for value in values {
                    *value = (value.clamp(epsilon, 1.0 / epsilon) + epsilon).ln();
                }
            }
        }
    }
}

/// torch `.bfloat16()`: the top 16 bits, round to nearest even.
fn rounded(table: Vec<f32>, precision: TablePrecision) -> Vec<f32> {
    match precision {
        TablePrecision::Float32 => table,
        TablePrecision::Bfloat16 => table
            .into_iter()
            .map(|value| {
                let bits = value.to_bits();
                let bias = 0x7FFF + ((bits >> 16) & 1);
                f32::from_bits(bits.wrapping_add(bias) & 0xFFFF_0000)
            })
            .collect(),
    }
}

/// The window's taps, `length` long: centred in an `fft_size` frame with zeros
/// around them under centre padding; on their own under [`Padding::None`],
/// where the frame IS the window and the transform's zeros come after it.
fn make_window(spec: &MelSpec, length: usize) -> Vec<f32> {
    let tau = 2.0 * std::f32::consts::PI;
    let hann =
        |i: usize, denominator: usize| 0.5 * (1.0 - (tau * i as f32 / denominator as f32).cos());
    let taps: Vec<f32> = match spec.window {
        Window::HannPeriodic => (0..length).map(|i| hann(i, length)).collect(),
        Window::HannSymmetric => (0..length).map(|i| hann(i, length - 1)).collect(),
        Window::Povey => (0..length)
            .map(|i| hann(i, length - 1).powf(0.85))
            .collect(),
        Window::HammingSymmetric => (0..length)
            .map(|i| {
                (0.54 - 0.46 * (std::f64::consts::TAU * i as f64 / (length - 1) as f64).cos())
                    as f32
            })
            .collect(),
    };
    if length == spec.fft_size || spec.padding == Padding::None {
        return taps;
    }
    let mut window = vec![0.0; spec.fft_size];
    let offset = (spec.fft_size - length) / 2;
    window[offset..offset + length].copy_from_slice(&taps);
    window
}

fn hz_to_mel(hz: f64, scale: Scale) -> f64 {
    match scale {
        Scale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        Scale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            if hz < min_log_hz {
                return hz / f_sp;
            }
            let logstep = 6.4_f64.ln() / 27.0;
            min_log_hz / f_sp + (hz / min_log_hz).ln() / logstep
        }
    }
}

fn mel_to_hz(mel: f64, scale: Scale) -> f64 {
    match scale {
        Scale::Htk => 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0),
        Scale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            if mel < min_log_mel {
                return mel * f_sp;
            }
            let logstep = 6.4_f64.ln() / 27.0;
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        }
    }
}

/// Built in `f64` and narrowed once — what keeps a bfloat16 rounding of the
/// result on the same side as torchaudio's for every entry.
fn make_filterbank(spec: &MelSpec, bin_count: usize) -> Vec<f32> {
    let bands = spec.mel_bands;
    let mel_min = hz_to_mel(f64::from(spec.f_min), spec.mel_scale);
    let mel_max = hz_to_mel(f64::from(spec.f_max), spec.mel_scale);
    let mel_points: Vec<f64> = (0..bands + 2)
        .map(|i| mel_min + i as f64 * (mel_max - mel_min) / (bands + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points
        .iter()
        .map(|m| mel_to_hz(*m, spec.mel_scale))
        .collect();
    // TensorFlow spaces `fft_size / 2` bins over `[0, sample_rate / 2]`.
    let tensorflow = spec.filter_shape == FilterShape::Tensorflow;
    let bin_hz: Vec<f64> = (0..bin_count)
        .map(|k| {
            if tensorflow {
                k as f64 * spec.sample_rate / 2.0 / (spec.fft_size / 2 - 1) as f64
            } else {
                k as f64 * spec.sample_rate / spec.fft_size as f64
            }
        })
        .collect();
    let bin_mel: Vec<f64> = bin_hz
        .iter()
        .map(|hz| hz_to_mel(*hz, spec.mel_scale))
        .collect();
    // The triangle over its three edges, in whichever domain it is straight.
    let (points, bins) = match spec.filter_shape {
        FilterShape::MelLinear | FilterShape::Tensorflow => (&mel_points, &bin_mel),
        FilterShape::HertzLinear => (&hz_points, &bin_hz),
    };
    let weighted = if tensorflow {
        1..spec.fft_size / 2
    } else {
        0..bin_count
    };
    let mut filterbank = vec![0.0_f32; bands * bin_count];
    for band in 0..bands {
        let (left, center, right) = (points[band], points[band + 1], points[band + 2]);
        let norm = match spec.filter_norm {
            FilterNorm::SlaneyArea => 2.0 / (hz_points[band + 2] - hz_points[band]),
            FilterNorm::None => 1.0,
        };
        for (bin, f) in bins.iter().enumerate() {
            if !weighted.contains(&bin) {
                continue;
            }
            let lower = (f - left) / (center - left);
            let upper = (right - f) / (right - center);
            filterbank[band * bin_count + bin] = (norm * lower.min(upper).max(0.0)) as f32;
        }
    }
    filterbank
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    const DASHENG: MelSpec = MelSpec {
        sample_rate: 16_000.0,
        fft_size: 512,
        hop_length: 160,
        mel_bands: 64,
        f_min: 0.0,
        f_max: 8_000.0,
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

    #[test]
    fn ranged_mel_matches_whole_signal_edges_and_global_pre_emphasis() {
        struct Source {
            samples: Vec<f32>,
            maximum: std::sync::atomic::AtomicUsize,
        }
        impl ferment_engine_core::AudioSampleSource for Source {
            fn sample_count(&self) -> usize {
                self.samples.len()
            }
            fn copy_samples(
                &self,
                destination: &mut [f32],
                offset: usize,
                count: usize,
            ) -> Result<usize> {
                self.maximum
                    .fetch_max(count, std::sync::atomic::Ordering::Relaxed);
                let count = count
                    .min(self.samples.len().saturating_sub(offset))
                    .min(destination.len());
                destination[..count].copy_from_slice(&self.samples[offset..offset + count]);
                Ok(count)
            }
        }
        for count in [1, 127, 512, 65_537] {
            let source = Source {
                samples: (0..count).map(|n| (n as f32 * 0.37).sin()).collect(),
                maximum: Default::default(),
            };
            for padding in [Padding::Reflect, Padding::Constant, Padding::None] {
                for pre_emphasis in [
                    None,
                    Some(PreEmphasis::Signal { alpha: 0.97 }),
                    Some(PreEmphasis::Frame { alpha: 0.9 }),
                ] {
                    let spec = MelSpec {
                        padding,
                        pre_emphasis,
                        input_scale: 0.7,
                        ..DASHENG
                    };
                    let mel = MelSpectrogram::new(spec).unwrap();
                    let expected = mel.compute(&source.samples).unwrap();
                    let mut actual = Vec::new();
                    for start in (0..expected.frame_count()).step_by(17) {
                        actual.extend(mel.compute_range(&source, start, 17).unwrap().values);
                    }
                    assert_eq!(
                        actual, expected.values,
                        "samples={count}, {padding:?}, {pre_emphasis:?}"
                    );
                }
            }
            assert!(source.maximum.load(std::sync::atomic::Ordering::Relaxed) <= 17 * 160 + 1024);
        }
    }

    #[test]
    fn decodes_the_shape_swift_encodes() {
        let json = r#"{"sampleRate":16000,"fftSize":512,"hopLength":160,"melBands":64,
            "fMin":0,"fMax":8000,"inputScale":1,"padding":"reflect","window":"hannPeriodic",
            "dcRemoval":false,"spectrum":"power","normalization":"none","melScale":"htk",
            "filterShape":"hertzLinear","filterNorm":"none","tablePrecision":"float32",
            "compression":{"decibelsAbsolute":{"amin":1e-10}}}"#;
        assert_eq!(serde_json::from_str::<MelSpec>(json).unwrap(), DASHENG);
    }

    #[test]
    fn every_associated_value_keeps_swifts_key() {
        let speech = MelSpec {
            window: Window::HannSymmetric,
            window_length: Some(400),
            pre_emphasis: Some(PreEmphasis::Signal { alpha: 0.97 }),
            compression: Compression::Decibels { top_db: 80.0 },
            ..DASHENG
        };
        let json = serde_json::to_value(speech).unwrap();
        assert_eq!(json["preEmphasis"]["signal"]["_0"], 0.97_f32);
        assert_eq!(json["compression"]["decibels"]["topDB"], 80.0);
        assert_eq!(json["windowLength"], 400);
        assert!(
            serde_json::to_value(DASHENG)
                .unwrap()
                .get("windowLength")
                .is_none()
        );
        assert_eq!(serde_json::from_value::<MelSpec>(json).unwrap(), speech);
    }

    #[test]
    fn frame_count_follows_the_padding() {
        assert_eq!(DASHENG.frame_count(160_000), 1001);
        assert_eq!(DASHENG.frame_count(0), 0);
        let kaldi = MelSpec {
            padding: Padding::None,
            window_length: Some(400),
            ..DASHENG
        };
        assert_eq!(kaldi.frame_count(16_000), 98);
        assert_eq!(kaldi.frame_count(399), 0);
        assert_eq!(kaldi.compute(&vec![0.0; 16_000]).unwrap().frame_count(), 98);
    }

    #[test]
    fn silence_is_the_floor_and_a_tone_lands_in_its_band() {
        let silent = DASHENG.compute(&vec![0.0; 16_000]).unwrap();
        assert_eq!(silent.frame_count(), 101);
        assert!(silent.values.iter().all(|v| (*v + 100.0).abs() < 1e-4));

        let tone: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin())
            .collect();
        let frames = DASHENG.compute(&tone).unwrap();
        let middle = frames.frame(50);
        let loudest = (0..64)
            .max_by(|a, b| middle[*a].total_cmp(&middle[*b]))
            .unwrap();
        // 1 kHz on a 64-band HTK scale over 0–8 kHz: mel(1000) / mel(8000) · 65 ≈ 22.9.
        assert!((21..=23).contains(&loudest), "band {loudest}");
    }

    #[test]
    fn the_per_call_steps_live_on_the_frames() {
        let frames = MelFrames {
            bands: 2,
            values: vec![10.0, -200.0, 30.0, -50.0],
        };
        assert_eq!(
            frames.clone().clamped_below_max_by(120.0).values,
            [10.0, -90.0, 30.0, -50.0]
        );
        assert_eq!(
            frames.bands_major(Some(3)),
            [10.0, 30.0, 0.0, -200.0, -50.0, 0.0]
        );

        let normal = MelFrames {
            bands: 1,
            values: vec![1.0, 3.0, 99.0],
        }
        .normalized_per_band(2, 0.0);
        assert_abs_diff_eq!(
            normal.values[0],
            -std::f32::consts::FRAC_1_SQRT_2,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            normal.values[1],
            std::f32::consts::FRAC_1_SQRT_2,
            epsilon = 1e-6
        );
        assert_eq!(normal.values[2], 0.0);
    }

    #[test]
    fn bfloat16_tables_keep_the_top_sixteen_bits() {
        let exact = MelSpectrogram::new(DASHENG).unwrap();
        let coarse = MelSpectrogram::new(MelSpec {
            table_precision: TablePrecision::Bfloat16,
            ..DASHENG
        })
        .unwrap();
        assert!(
            coarse
                .window()
                .iter()
                .all(|tap| tap.to_bits() & 0xFFFF == 0)
        );
        assert!(
            coarse
                .filterbank()
                .iter()
                .all(|w| w.to_bits() & 0xFFFF == 0)
        );
        assert_ne!(exact.window(), coarse.window());
    }
}
