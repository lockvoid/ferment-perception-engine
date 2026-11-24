use ferment_engine_audio::AudioFrames;
use ferment_engine_audio::derived_cache::DerivedFileCache;
use ferment_engine_audio::streaming_resampler::resample_current;
use ferment_engine_core::{BeatLatticeSegment, EnergyStructure, EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::config::{BORDER_SIZE, CHUNK_SIZE, DEFAULT_BEAT_OFFSET, FPS, N_MELS, SAMPLE_RATE};

use super::quantize::{bpm_from_beats, grid_stability, tempo_stability};
use super::{dbn, postprocess};
use crate::energy_structure;

mod streaming;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatAnalysisResult {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub quantized_beats: Vec<f64>,
    pub quantized_downbeats: Vec<f64>,
    pub grid_stability: f64,
    pub tempo_stability: f64,
    pub bpm: Option<f64>,
    pub quantized_bpm: Option<f64>,
    /// The warp lattice the flattened quantized arrays expand from.
    #[serde(default)]
    pub segments: Vec<BeatLatticeSegment>,
    /// RMS in dBFS per BAR of the quantized downbeat lattice. Empty when no
    /// downbeat lattice was published or the detector never held the samples.
    #[serde(default)]
    pub energy_per_bar: Vec<f64>,
    /// The loudness structure segmented over that same column — computed ONCE
    /// here at detection (`with_energy`), never re-derived downstream.
    #[serde(default)]
    pub energy_structure: EnergyStructure,
}

impl BeatAnalysisResult {
    /// A grid with no beats in it — the answer for silence and for audio too
    /// short to hold a pulse.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            beats: Vec::new(),
            downbeats: Vec::new(),
            quantized_beats: Vec::new(),
            quantized_downbeats: Vec::new(),
            grid_stability: 0.0,
            tempo_stability: 0.0,
            bpm: None,
            quantized_bpm: None,
            segments: Vec::new(),
            energy_per_bar: Vec::new(),
            energy_structure: EnergyStructure::default(),
        }
    }

    /// The same measurement with the energy column attached — the detector
    /// computes the grid, then the caller that held the samples fills this in.
    /// Structure rides along: this seam is the ONE place it is computed, for
    /// every detector.
    #[must_use]
    pub fn with_energy(mut self, per_bar: Vec<f64>) -> Self {
        self.energy_structure = energy_structure::detect(&per_bar);
        self.energy_per_bar = per_bar;
        self
    }

    /// The energy column measured over this result's own downbeat lattice.
    #[must_use]
    pub fn with_energy_from(self, samples: &[f32], sample_rate: f64, duration: f64) -> Self {
        let per_bar = energy_structure::energy_per_bar(
            samples,
            sample_rate,
            &self.quantized_downbeats,
            duration,
        );
        self.with_energy(per_bar)
    }
}

/// How the averaged logits become raw beats. `Minimal` is the peak-pick the
/// pipeline shipped with; `Dbn` is the madmom bar-pointer Viterbi, which
/// enforces a continuous bar through material the peak-picker follows
/// literally. Downstream algorithms (offset, quantize, stability, energy) are
/// shared; DBN additionally marks inferred low-support beats as bridges for
/// tempo fitting, while preserving the complete raw decoded path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PostProcessing {
    #[default]
    Minimal,
    Dbn,
}

/// Models in the BeatThis ensemble, in priority order. The device SHIPS the
/// first [`BeatThis::SHIPPING_ENSEMBLE_SIZE`]: ens2 is ear-indistinguishable
/// from ens3 on the adversarial corpus and saves a model download. The third
/// checkpoint stays published so a locally-bundled ens3 A/B still resolves.
const ENSEMBLE: [ModelId; 3] = [ModelId::BeatThis1, ModelId::BeatThis2, ModelId::BeatThis3];

#[derive(Debug, Clone)]
pub struct BeatThis {
    pub(crate) cache: Option<Arc<DerivedFileCache>>,
    backend: Backend,
    version: ModelVersion,
    ensemble_size: usize,
    beat_offset: f64,
    post_processing: PostProcessing,
}

impl BeatThis {
    pub fn with_cache(mut self, cache: Option<Arc<DerivedFileCache>>) -> Self {
        self.cache = cache;
        self
    }

    /// What the device downloads and runs.
    pub const SHIPPING_ENSEMBLE_SIZE: usize = 2;

    pub fn new(backend: Backend) -> Self {
        Self {
            cache: None,
            backend,
            version: ModelVersion::default(),
            ensemble_size: Self::SHIPPING_ENSEMBLE_SIZE,
            beat_offset: DEFAULT_BEAT_OFFSET,
            post_processing: PostProcessing::default(),
        }
    }

    /// `size` clamps to 1…3 — the A/B lane that reaches the unshipped third
    /// checkpoint when it is present locally.
    pub fn with_ensemble_size(mut self, size: usize) -> Self {
        self.ensemble_size = size.clamp(1, ENSEMBLE.len());
        self
    }

    pub fn with_beat_offset(mut self, beat_offset: f64) -> Self {
        self.beat_offset = beat_offset;
        self
    }

    pub fn with_post_processing(mut self, post_processing: PostProcessing) -> Self {
        self.post_processing = post_processing;
        self
    }

    pub fn with_version(mut self, version: ModelVersion) -> Self {
        self.version = version;
        self
    }

    pub fn active_models(&self) -> &'static [ModelId] {
        &ENSEMBLE[..self.ensemble_size]
    }

    /// The shared tail: logits → chosen post-processing → offset → stability.
    /// Public so a caller holding logits can decode without re-running the
    /// models; the energy column is attached by whoever holds the samples.
    #[must_use]
    pub fn analyze_logits(
        &self,
        beat_logits: &[f32],
        downbeat_logits: &[f32],
        fps: f64,
        track_duration: Option<f64>,
    ) -> BeatAnalysisResult {
        let result = match self.post_processing {
            PostProcessing::Minimal => {
                postprocess::process(beat_logits, downbeat_logits, fps, track_duration)
            }
            PostProcessing::Dbn => dbn::process(beat_logits, downbeat_logits, fps, track_duration),
        };
        self.finish(result)
    }
    fn finish(&self, result: postprocess::ProcessResult) -> BeatAnalysisResult {
        let offset = |time: &f64| (time + self.beat_offset).max(0.0);
        let beats = result.beats.iter().map(offset).collect::<Vec<_>>();
        BeatAnalysisResult {
            grid_stability: grid_stability(&beats),
            tempo_stability: tempo_stability(&beats),
            bpm: bpm_from_beats(&beats),
            quantized_bpm: result.quantized_bpm,
            downbeats: result.downbeats.iter().map(offset).collect(),
            quantized_beats: result.quantized_beats.iter().map(offset).collect(),
            quantized_downbeats: result.quantized_downbeats.iter().map(offset).collect(),
            segments: result
                .segments
                .into_iter()
                .map(|segment| BeatLatticeSegment {
                    first: segment.first + self.beat_offset,
                    ..segment
                })
                .collect(),
            beats,
            energy_per_bar: Vec::new(),
            energy_structure: EnergyStructure::default(),
        }
    }

    #[tracing::instrument(skip_all, fields(processor = "beat_this", samples = frames.len(), ensemble = self.ensemble_size, post = ?self.post_processing))]
    pub async fn detect(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<BeatAnalysisResult> {
        progress.report(0.05, "loading audio").await;
        let track_duration = frames.duration_secs();
        let identity = (frames.sample_rate == SAMPLE_RATE)
            .then_some(frames.identity)
            .flatten();
        let audio = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_current(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };

        progress.report(0.20, "computing mel").await;
        let frames =
            super::config::MEL_SPEC.perform(&audio, identity.as_ref(), self.cache.as_deref())?;
        let mel: Vec<Vec<f32>> = frames
            .values
            .chunks_exact(frames.bands)
            .map(<[f32]>::to_vec)
            .collect();
        if mel.is_empty() {
            return Err(EngineError::AudioTooShort {
                samples: audio.len(),
                min_required: super::config::FFT_SIZE,
            });
        }
        let n_frames = mel.len();

        progress.report(0.30, "running ensemble").await;
        let model_ids = self.active_models();
        let mut beat_logits = vec![0.0f32; n_frames];
        let mut downbeat_logits = vec![0.0f32; n_frames];
        let model_count = model_ids.len();
        for (i, &model_id) in model_ids.iter().enumerate() {
            if progress.is_closed() {
                return Err(EngineError::Cancelled);
            }
            let (b, d) = self.run_inference(&mel, model_id).await?;
            for j in 0..n_frames {
                beat_logits[j] += b[j];
                downbeat_logits[j] += d[j];
            }
            let frac = 0.30 + 0.55 * (i + 1) as f64 / model_count as f64;
            progress
                .report(
                    frac,
                    format!("running ensemble ({}/{})", i + 1, model_count),
                )
                .await;
        }
        if model_count > 1 {
            let inv = 1.0 / model_count as f32;
            for v in &mut beat_logits {
                *v *= inv;
            }
            for v in &mut downbeat_logits {
                *v *= inv;
            }
        }

        progress.report(0.90, "post-processing").await;
        let result = self.analyze(&beat_logits, &downbeat_logits, &audio, track_duration);

        progress.report(1.0, "done").await;
        Ok(result)
    }

    /// Logits + the samples they were computed from → the published result.
    /// `detect` is a one-liner over this so the energy attachment cannot
    /// drift away from the grid: running `detect` needs the model fleet, and
    /// a step only `detect` performed would be a step nothing could test.
    fn analyze(
        &self,
        beat_logits: &[f32],
        downbeat_logits: &[f32],
        samples: &[f32],
        track_duration: f64,
    ) -> BeatAnalysisResult {
        self.analyze_logits(beat_logits, downbeat_logits, FPS, Some(track_duration))
            .with_energy_from(samples, f64::from(SAMPLE_RATE), track_duration)
    }

    /// Run chunked inference over [`chunk_plan`]. Returns
    /// `(beat_logits, downbeat_logits)` of length `n_frames`.
    async fn run_inference(
        &self,
        mel: &[Vec<f32>],
        model: ModelId,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = mel.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if CHUNK_SIZE <= 2 * BORDER_SIZE {
            return Err(EngineError::invalid_config(
                "CHUNK_SIZE must be > 2×BORDER_SIZE",
            ));
        }

        // `chunk_plan` provably tiles [0, n), so every element is written.
        // The sentinel is kept anyway because 0.0 is a *plausible* logit
        // (sigmoid 0.5, which the DBN would read as a downbeat region) while
        // -1000 stands out the moment a hole appears.
        let mut beat_logits = vec![-1000.0f32; n];
        let mut downbeat_logits = vec![-1000.0f32; n];
        for entry in chunk_plan(n, CHUNK_SIZE, BORDER_SIZE) {
            let mut chunk = Vec::with_capacity(CHUNK_SIZE);
            for i in 0..CHUNK_SIZE {
                let index = entry.chunk_start + i as isize;
                match usize::try_from(index).ok().and_then(|index| mel.get(index)) {
                    Some(row) => chunk.push(row.clone()),
                    None => chunk.push(vec![0.0f32; N_MELS]),
                }
            }
            let (beat, downbeat) = self.predict_chunk(&chunk, model).await?;
            for i in entry.write_start..entry.write_end {
                let inside = (i as isize - entry.chunk_start) as usize;
                beat_logits[i] = beat[inside];
                downbeat_logits[i] = downbeat[inside];
            }
        }
        Ok((beat_logits, downbeat_logits))
    }

    /// Run a single fixed-size chunk through one model. Returns flat
    /// `(beat[len], downbeat[len])` pairs.
    async fn predict_chunk(
        &self,
        chunk: &[Vec<f32>],
        model: ModelId,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = chunk.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // The door every lowering shares (`model.json`): `mel [1, 1, 128, n]`
        // bands-major in, `logits [1, 2, 1, n]` out — row 0 beat, row 1 downbeat.
        let mut bands_major = vec![0.0f32; N_MELS * n];
        for (frame, row) in chunk.iter().enumerate() {
            for (band, value) in row.iter().take(N_MELS).enumerate() {
                bands_major[band * n + frame] = *value;
            }
        }

        let mut inputs = TensorMap::new();
        inputs.insert(
            "mel".into(),
            Tensor::from_f32(vec![1, 1, N_MELS as i64, n as i64], bands_major)?,
        );
        let outputs = self.backend.infer(model, &self.version, inputs).await?;

        let tensor = outputs
            .get("logits")
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: model.name().into(),
                key: "logits".into(),
            })?;
        let raw = tensor.to_f32()?;
        if raw.len() != n * 2 {
            return Err(EngineError::InvalidOutput);
        }
        let (beat, downbeat) = raw.split_at(n);
        Ok((beat.to_vec(), downbeat.to_vec()))
    }
}

/// One model call: decode `chunk_size` spectrogram rows from `chunk_start`
/// (rows outside `[0, frames)` are zero-padded) and keep the logits rows
/// `[write_start, write_end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlanEntry {
    pub chunk_start: isize,
    pub write_start: usize,
    pub write_end: usize,
}

/// Mirrors `beat_this`'s `split_piece(avoid_short_end=True)` + `keep_first`
/// aggregation: the write windows tile `[0, frames)` EXACTLY, and the
/// realigned final chunk contributes only the rows nobody covered yet.
///
/// Two failure modes live here. Skipping the realigned chunk loses the tail —
/// the loop this replaced stopped one chunk early, so a 186 s track kept 363
/// frames (7.3 s) of `-1000` sentinel and every track longer than a whole
/// number of chunks lost up to 29.7 s of analysis. Letting the realigned chunk
/// overwrite covered rows is `keep_last`, a silent drift from the reference
/// stitcher.
#[must_use]
pub fn chunk_plan(frames: usize, chunk_size: usize, border_size: usize) -> Vec<ChunkPlanEntry> {
    chunk_entries(frames, chunk_size, border_size).collect()
}
fn chunk_entries(
    frames: usize,
    chunk_size: usize,
    border_size: usize,
) -> impl Iterator<Item = ChunkPlanEntry> {
    let step = chunk_size - 2 * border_size;
    let count = if frames <= chunk_size {
        usize::from(frames != 0)
    } else {
        frames.div_ceil(step)
    };
    let mut covered = 0;
    (0..count).filter_map(move |i| {
        if frames <= chunk_size {
            return Some(ChunkPlanEntry {
                chunk_start: 0,
                write_start: 0,
                write_end: frames,
            });
        }
        let border = border_size as isize;
        let start = if i + 1 == count {
            (frames - (chunk_size - border_size)) as isize
        } else {
            -border + (i * step) as isize
        };
        let write_start = ((start + border).max(0) as usize).max(covered);
        let write_end = ((start + (chunk_size - border_size) as isize).max(0) as usize).min(frames);
        if write_end <= write_start {
            return None;
        }
        covered = write_end;
        Some(ChunkPlanEntry {
            chunk_start: start,
            write_start,
            write_end,
        })
    })
}

#[cfg(test)]
mod tests {
    mod streaming;
    use std::sync::Arc;

    use ferment_engine_infer::UnavailableBackend;

    use super::*;

    fn backend() -> Backend {
        Arc::new(UnavailableBackend)
    }

    const CHUNK: usize = 1_500;
    const BORDER: usize = 6;

    /// The invariant the tail-loss bug broke: write windows tile `[0, frames)`
    /// exactly — no gap (a sentinel hole) and no overlap (keep_last drift).
    #[test]
    fn write_windows_tile_exactly() {
        for frames in [
            1, 100, 800, 1_488, 1_489, 1_500, 1_501, 2_976, 2_977, 4_464, 9_291, 60_010,
        ] {
            let plan = chunk_plan(frames, CHUNK, BORDER);
            assert_eq!(
                plan.first().map(|entry| entry.write_start),
                Some(0),
                "head not covered at frames={frames}"
            );
            assert_eq!(
                plan.last().map(|entry| entry.write_end),
                Some(frames),
                "tail lost at frames={frames}"
            );
            let mut covered = 0;
            for entry in &plan {
                assert_eq!(
                    entry.write_start, covered,
                    "gap or overlap at frames={frames}"
                );
                assert!(entry.write_end > entry.write_start);
                assert!(entry.write_start as isize - entry.chunk_start >= 0);
                assert!(entry.write_end as isize - entry.chunk_start <= CHUNK as isize);
                covered = entry.write_end;
            }
            assert_eq!(covered, frames);
        }
    }

    /// Pinned to `beat_this`'s `split_piece(avoid_short_end=True)` +
    /// `keep_first` on the 186 s parity track (9291 frames): the final chunk
    /// realigns to 7797 and contributes only the tail nobody covered.
    #[test]
    fn mirrors_python_split_on_the_parity_track() {
        let plan = chunk_plan(9_291, CHUNK, BORDER);
        assert_eq!(
            plan.iter()
                .map(|entry| entry.chunk_start)
                .collect::<Vec<_>>(),
            vec![-6, 1_482, 2_970, 4_458, 5_946, 7_434, 7_797]
        );
        assert_eq!(
            plan.iter()
                .map(|entry| entry.write_start)
                .collect::<Vec<_>>(),
            vec![0, 1_488, 2_976, 4_464, 5_952, 7_440, 8_928]
        );
        assert_eq!(
            plan.iter().map(|entry| entry.write_end).collect::<Vec<_>>(),
            vec![1_488, 2_976, 4_464, 5_952, 7_440, 8_928, 9_291]
        );
    }

    #[test]
    fn short_piece_is_one_whole_chunk() {
        assert_eq!(
            chunk_plan(800, CHUNK, BORDER),
            vec![ChunkPlanEntry {
                chunk_start: 0,
                write_start: 0,
                write_end: 800
            }]
        );
    }

    #[test]
    fn empty_input_has_no_chunks() {
        assert!(chunk_plan(0, CHUNK, BORDER).is_empty());
    }

    /// `modelNames` is the SHIPPING set (what the device downloads: ens2);
    /// an explicit size can still reach the third checkpoint.
    #[test]
    fn the_shipping_ensemble_is_two_models() {
        let backend = backend();
        assert_eq!(BeatThis::SHIPPING_ENSEMBLE_SIZE, 2);
        let shipping = BeatThis::new(backend.clone()).active_models().to_vec();
        assert_eq!(shipping.len(), 2);
        assert_eq!(
            BeatThis::new(backend.clone())
                .with_ensemble_size(2)
                .active_models(),
            shipping
        );
        assert_eq!(
            BeatThis::new(backend.clone())
                .with_ensemble_size(1)
                .active_models(),
            &[ModelId::BeatThis1]
        );
        assert_eq!(
            BeatThis::new(backend.clone())
                .with_ensemble_size(0)
                .active_models()
                .len(),
            1,
            "size clamps to 1"
        );
        assert_eq!(
            BeatThis::new(backend.clone())
                .with_ensemble_size(9)
                .active_models(),
            &ENSEMBLE,
            "oversize takes every checkpoint"
        );
        assert_eq!(&ENSEMBLE[..2], shipping);
        assert_eq!(
            BeatThis::new(backend).with_ensemble_size(3).active_models(),
            &ENSEMBLE,
            "the full-ensemble request reaches all three checkpoints"
        );
    }

    /// The stored grid is the model's truth — the constructor bakes no lead.
    #[test]
    fn the_default_detector_applies_no_beat_offset() {
        assert_eq!(BeatThis::new(backend()).beat_offset, 0.0);
    }

    /// The whole tail in one pass: DBN logits → lattice segments → energy
    /// column → structure. This is the surface the lead's swap hands to
    /// `crates/project`, so its shape is pinned here rather than left to the
    /// integration round.
    #[test]
    fn the_dbn_lane_publishes_a_lattice_and_an_energy_column() {
        // 120 bpm at 50 fps for 60 s, downbeat every fourth beat.
        let frames = 3_000;
        let mut beat = vec![-8.0f32; frames];
        let mut down = vec![-8.0f32; frames];
        for (index, frame) in (10..frames).step_by(25).enumerate() {
            beat[frame] = 8.0;
            if index.is_multiple_of(4) {
                down[frame] = 8.0;
            }
        }
        // 60 s of a steady full-scale tone under the same grid: one loud
        // plateau, one reading per bar.
        let samples = (0..(60 * SAMPLE_RATE as usize))
            .map(|i| (i as f32 * 0.05).sin())
            .collect::<Vec<_>>();
        let detector = BeatThis::new(backend()).with_post_processing(PostProcessing::Dbn);
        let result = detector.analyze(&beat, &down, &samples, 60.0);

        assert!(result.beats.len() > 110, "{}", result.beats.len());
        assert!(!result.segments.is_empty());
        let covered: usize = result.segments.iter().map(|segment| segment.count).sum();
        assert_eq!(covered, result.quantized_beats.len());
        assert_eq!(
            result.segments[0].meter,
            Some(4),
            "the winning bar IS the meter"
        );
        assert!(
            result.beats[0] >= 0.0 && (result.beats[0] - 0.2).abs() < 0.05,
            "no baked lead: first beat at {}",
            result.beats[0]
        );

        assert_eq!(
            result.energy_per_bar.len(),
            result.quantized_downbeats.len(),
            "one energy reading per bar of the downbeat lattice"
        );
        assert_eq!(
            result.energy_structure.segments.len(),
            1,
            "a steady tone is one plateau: {:?}",
            result.energy_structure.segments
        );
        assert!(result.energy_structure.events.is_empty());
    }

    #[test]
    fn the_dbn_lane_is_opt_in() {
        let backend = backend();
        assert_eq!(
            BeatThis::new(backend.clone()).post_processing,
            PostProcessing::Minimal
        );
        assert_eq!(
            BeatThis::new(backend)
                .with_post_processing(PostProcessing::Dbn)
                .post_processing,
            PostProcessing::Dbn
        );
    }
}
