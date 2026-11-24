//! Classical Queen Mary / Mixxx beat and downbeat analyzer.

pub mod downbeat;
pub mod onset;
pub mod tempo;

use ferment_engine_audio::{AudioFrames, resample_fft};
use ferment_engine_core::{BeatLatticeSegment, EnergyStructure, EngineError, ProgressSink, Result};

use crate::beatthis::{
    BeatAnalysisResult,
    meter::MeterResult,
    quantize::{bpm_from_beats, grid_stability, quantize_beats, tempo_stability},
};
use downbeat::detect as detect_downbeats;
use onset::BeatQueenMaryOnset;
use tempo::{TempoConfig, track};

pub const ANALYSIS_SAMPLE_RATE: u32 = 22_050;

#[derive(Debug, Clone)]
pub struct BeatQueenMary {
    tempo_config: TempoConfig,
    beats_per_bar: Option<usize>,
    beat_offset: f64,
}

impl Default for BeatQueenMary {
    fn default() -> Self {
        Self::new()
    }
}

impl BeatQueenMary {
    pub const MODEL_NAMES: [&str; 0] = [];
    pub const MODEL_FILES: [(&str, &str); 0] = [];

    pub fn new() -> Self {
        Self {
            tempo_config: TempoConfig::default(),
            beats_per_bar: None,
            beat_offset: 0.0,
        }
    }

    pub fn with_options(mut self, beats_per_bar: Option<usize>, beat_offset: f64) -> Self {
        self.beats_per_bar = beats_per_bar;
        self.beat_offset = beat_offset;
        self
    }

    pub fn with_tempo_config(mut self, config: TempoConfig) -> Self {
        self.tempo_config = config;
        self
    }

    /// The DP tracker coasts through silence, so a quiet intro or outro grows
    /// phantom beats where there is no music. The active span runs from the
    /// first to the last envelope frame at or above 2 % of the peak, padded
    /// half a beat period; only those OUTER bounds gate, so a mid-track
    /// breakdown keeps its coasted beats. A silent envelope has no span, and
    /// the tracker's answer stands.
    #[must_use]
    pub fn trim_to_active_span(
        beat_frames: &[f64],
        envelope: &[f32],
        period_frames: f64,
    ) -> Vec<f64> {
        let peak = envelope.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // An empty envelope folds to −inf and an all-NaN one folds to NaN;
        // both mean "no active span", and both land here.
        if peak.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return beat_frames.to_vec();
        }
        let threshold = peak * 0.02;
        let active = |index: &usize| envelope[*index] >= threshold;
        let (Some(first), Some(last)) = (
            (0..envelope.len()).find(active),
            (0..envelope.len()).rev().find(active),
        ) else {
            return beat_frames.to_vec();
        };
        let pad = period_frames * 0.5;
        let (lo, hi) = (first as f64 - pad, last as f64 + pad);
        beat_frames
            .iter()
            .copied()
            .filter(|frame| *frame >= lo && *frame <= hi)
            .collect()
    }

    pub async fn detect(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<BeatAnalysisResult> {
        if frames.sample_rate == 0 {
            return Err(EngineError::invalid_config(
                "audio sample rate must be positive",
            ));
        }
        if self.beats_per_bar == Some(0) {
            return Err(EngineError::invalid_config(
                "beats_per_bar must be positive when specified",
            ));
        }
        progress.report(0.02, "preparing audio").await;
        let duration = frames.duration_secs();
        let audio = if frames.sample_rate == ANALYSIS_SAMPLE_RATE {
            frames.samples
        } else {
            resample_fft(&frames.samples, frames.sample_rate, ANALYSIS_SAMPLE_RATE)?
        };
        if progress.is_closed() {
            return Err(EngineError::Cancelled);
        }

        progress.report(0.20, "computing onset envelope").await;
        let onset = BeatQueenMaryOnset::for_sample_rate(ANALYSIS_SAMPLE_RATE as f64);
        let envelope_rate = ANALYSIS_SAMPLE_RATE as f64 / onset.step_size as f64;
        let envelope = onset.compute(&audio);
        if envelope.is_empty() {
            return Ok(BeatAnalysisResult::empty());
        }

        progress.report(0.55, "estimating tempo").await;
        let tempo = track(&envelope, envelope_rate, &self.tempo_config);
        // Silence gate — see `trim_to_active_span`. Period from the tracked
        // bpm; a dead tempo falls back to the median inter-beat gap.
        let period_frames = if tempo.bpm > 0.0 {
            envelope_rate * 60.0 / tempo.bpm
        } else {
            let mut gaps = tempo
                .beat_frames
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .collect::<Vec<_>>();
            gaps.sort_by(f64::total_cmp);
            gaps.get(gaps.len() / 2).copied().unwrap_or(0.0)
        };
        let active_beat_frames =
            Self::trim_to_active_span(&tempo.beat_frames, &envelope, period_frames);
        let raw_beat_times = active_beat_frames
            .iter()
            .map(|frame| frame / envelope_rate)
            .collect::<Vec<_>>();
        if raw_beat_times.len() < 2 {
            return Ok(BeatAnalysisResult {
                bpm: (tempo.bpm > 0.0).then_some(tempo.bpm),
                beats: raw_beat_times,
                ..BeatAnalysisResult::empty()
            });
        }

        progress.report(0.80, "detecting downbeats").await;
        let beat_sample_frames = active_beat_frames
            .iter()
            .map(|frame| {
                (frame * onset.step_size as f64)
                    .round()
                    .clamp(0.0, audio.len().saturating_sub(1) as f64) as usize
            })
            .collect::<Vec<_>>();
        let downbeat = detect_downbeats(
            &audio,
            ANALYSIS_SAMPLE_RATE as f64,
            &beat_sample_frames,
            self.beats_per_bar,
        );
        let raw_downbeat_times = downbeat
            .downbeat_indices
            .iter()
            .filter_map(|&index| raw_beat_times.get(index).copied())
            .collect::<Vec<_>>();

        progress.report(0.95, "quantizing grid").await;
        let first_downbeat = downbeat.downbeat_indices.first().copied().unwrap_or(0);
        let phase = (downbeat.beats_per_bar - first_downbeat % downbeat.beats_per_bar)
            % downbeat.beats_per_bar;
        let quantized = quantize_beats(
            &raw_beat_times,
            &raw_downbeat_times,
            Some(MeterResult {
                meter: downbeat.beats_per_bar,
                phase,
                raw_meter: downbeat.beats_per_bar,
            }),
            Some(duration),
        );
        let offset = |time: f64| (time + self.beat_offset).max(0.0);
        let beats = raw_beat_times.into_iter().map(offset).collect::<Vec<_>>();
        let downbeats = raw_downbeat_times
            .into_iter()
            .map(offset)
            .collect::<Vec<_>>();
        let quantized_beats = quantized.beats.into_iter().map(offset).collect::<Vec<_>>();
        let quantized_downbeats = quantized
            .downbeats
            .into_iter()
            .map(offset)
            .collect::<Vec<_>>();

        let result = BeatAnalysisResult {
            grid_stability: grid_stability(&beats),
            tempo_stability: tempo_stability(&beats),
            bpm: bpm_from_beats(&beats).or((tempo.bpm > 0.0).then_some(tempo.bpm)),
            quantized_bpm: quantized.bpm,
            beats,
            downbeats,
            quantized_beats,
            quantized_downbeats,
            segments: quantized
                .segments
                .into_iter()
                .map(|segment| BeatLatticeSegment {
                    first: segment.first + self.beat_offset,
                    ..segment
                })
                .collect(),
            energy_per_bar: Vec::new(),
            energy_structure: EnergyStructure::default(),
        }
        // This path holds the samples, so it fills the energy column (and the
        // structure that rides it) exactly as BeatThis does — the two
        // detectors answer the same surface.
        .with_energy_from(&audio, f64::from(ANALYSIS_SAMPLE_RATE), duration);
        progress.report(1.0, "done").await;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use ferment_engine_test_fixtures as fixtures;

    use super::*;

    /// `None` means `ffmpeg` is unavailable, not that the fixture is — the
    /// caller resolves the path first so the two reasons stay distinguishable.
    fn decode_mp3_fixture(path: &Path) -> Option<AudioFrames> {
        let output = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                "22050",
                "pipe:1",
            ])
            .output()
            .ok()?;
        assert!(
            output.status.success(),
            "ffmpeg fixture decode failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let samples = output
            .stdout
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        Some(AudioFrames::new(samples, ANALYSIS_SAMPLE_RATE))
    }

    #[test]
    fn silence_is_total_and_model_free() {
        let analyzer = BeatQueenMary::new();
        let result = futures_executor::block_on(analyzer.detect(
            AudioFrames::new(vec![0.0; 44_100], 22_050),
            ProgressSink::noop(),
        ))
        .expect("silence analysis");
        assert!(result.downbeats.is_empty());
        assert!(result.beats.len() <= 1);
    }

    #[test]
    fn test_detect_on_techno_sample() {
        let Some(path) = fixtures::path("techno-1-30s.mp3") else {
            return;
        };
        let Some(audio) = decode_mp3_fixture(&path) else {
            fixtures::skip("ffmpeg is not on PATH, so the mp3 fixture cannot be decoded");
            return;
        };
        let result =
            futures_executor::block_on(BeatQueenMary::new().detect(audio, ProgressSink::noop()))
                .expect("techno fixture analysis");

        assert!(result.beats.len() > 25, "beats={}", result.beats.len());
        assert!(result.beats.len() < 200, "beats={}", result.beats.len());
        assert!(result.beats.windows(2).all(|pair| pair[1] > pair[0]));
        let bpm = result.bpm.expect("techno BPM");
        assert!(bpm > 40.0, "BPM {bpm} below plausible music tempo");
        assert!(bpm < 220.0, "BPM {bpm} above plausible music tempo");
        let octave_ok = (100.0..=180.0).contains(&bpm)
            || (100.0..=180.0).contains(&(2.0 * bpm))
            || (100.0..=180.0).contains(&(bpm / 2.0));
        assert!(octave_ok, "BPM {bpm} is outside 100–180 at every octave");
        assert!(result.downbeats.len() > 4);
        assert!(result.downbeats.len() < result.beats.len());
        assert!(
            result.grid_stability > 0.6,
            "grid stability={}",
            result.grid_stability
        );
    }

    /// The DP tracker coasts through silence, so a quiet intro/outro grows
    /// phantom beats where there is no music. These pin the trim: beats live
    /// only inside the envelope's active span (padded half a period); silence
    /// in the MIDDLE of a track keeps its coasted beats.
    #[test]
    fn trim_drops_beats_outside_the_active_span() {
        let mut envelope = vec![0.0f32; 700];
        for frame in envelope.iter_mut().take(501).skip(100) {
            *frame = 1.0;
        }
        // A beat every 20 frames across the WHOLE range, silence included.
        let beats = (0..35).map(|i| f64::from(i) * 20.0).collect::<Vec<_>>();

        let trimmed = BeatQueenMary::trim_to_active_span(&beats, &envelope, 20.0);

        assert!(!trimmed.is_empty());
        // Pad is half a period (10 frames): nothing before 90 or after 510.
        assert!(trimmed[0] >= 90.0);
        assert!(trimmed[trimmed.len() - 1] <= 510.0);
        // Every in-span beat survives untouched.
        let expected = beats
            .iter()
            .copied()
            .filter(|beat| (90.0..=510.0).contains(beat))
            .collect::<Vec<_>>();
        assert_eq!(trimmed, expected);

        // The Swift case above cannot see the pad on its own: its beats are
        // multiples of the period, so a padless gate ([100, 500]) admits
        // exactly the same set. On a grid of 10 + 20k, 90 and 510 sit exactly
        // on the padded bounds and outside the unpadded ones — a padless gate
        // answers 110 / 490 instead.
        let offset = (0..35)
            .map(|i| 10.0 + f64::from(i) * 20.0)
            .collect::<Vec<_>>();
        let trimmed = BeatQueenMary::trim_to_active_span(&offset, &envelope, 20.0);
        assert_eq!(
            trimmed.first().copied(),
            Some(90.0),
            "the leading half-period pad must admit the beat at 90"
        );
        assert_eq!(
            trimmed.last().copied(),
            Some(510.0),
            "the trailing half-period pad must admit the beat at 510"
        );
    }

    #[test]
    fn mid_track_quiet_is_not_trimmed() {
        let mut envelope = vec![1.0f32; 600];
        for frame in envelope.iter_mut().take(401).skip(200) {
            *frame = 0.0; // a breakdown, not an edge
        }
        let beats = (0..30).map(|i| f64::from(i) * 20.0).collect::<Vec<_>>();
        assert_eq!(
            BeatQueenMary::trim_to_active_span(&beats, &envelope, 20.0),
            beats,
            "outer-bounds gate must not touch a mid-track breakdown"
        );
    }

    #[test]
    fn a_silent_envelope_passes_through() {
        let beats = [0.0, 20.0, 40.0];
        assert_eq!(
            BeatQueenMary::trim_to_active_span(&beats, &[0.0; 100], 20.0),
            beats,
            "no active span → leave the tracker's answer alone"
        );
        assert_eq!(
            BeatQueenMary::trim_to_active_span(&beats, &[], 20.0),
            beats,
            "no envelope at all → same"
        );
    }

    /// End to end: 3 s of silence on both sides of the techno fixture must not
    /// grow beats. Skips loudly when the fixture or ffmpeg is missing.
    #[test]
    fn padded_silence_grows_no_beats() {
        let Some(path) = fixtures::path("techno-1-30s.mp3") else {
            return;
        };
        let Some(audio) = decode_mp3_fixture(&path) else {
            fixtures::skip("ffmpeg is not on PATH, so the mp3 fixture cannot be decoded");
            return;
        };
        let music_duration = audio.duration_secs();
        let pad = vec![0.0f32; 3 * ANALYSIS_SAMPLE_RATE as usize];
        let padded = [pad.as_slice(), &audio.samples, pad.as_slice()].concat();
        let result = futures_executor::block_on(BeatQueenMary::new().detect(
            AudioFrames::new(padded, ANALYSIS_SAMPLE_RATE),
            ProgressSink::noop(),
        ))
        .expect("padded techno analysis");

        assert!(result.beats.len() > 25, "beats={}", result.beats.len());
        // Half a period of slack at ~2 Hz beats ≈ 0.35 s; give 0.6 s.
        assert!(
            result.beats[0] >= 3.0 - 0.6,
            "phantom beats in the leading silence: first at {}",
            result.beats[0]
        );
        let last = result.beats[result.beats.len() - 1];
        assert!(
            last <= 3.0 + music_duration + 0.6,
            "phantom beats in the trailing silence: last at {last}"
        );
    }

    /// Both detectors answer the same surface: the energy column and the
    /// structure that rides it are filled here, not downstream.
    #[test]
    fn the_detector_fills_the_energy_column() {
        let Some(path) = fixtures::path("techno-1-30s.mp3") else {
            return;
        };
        let Some(audio) = decode_mp3_fixture(&path) else {
            fixtures::skip("ffmpeg is not on PATH, so the mp3 fixture cannot be decoded");
            return;
        };
        let result =
            futures_executor::block_on(BeatQueenMary::new().detect(audio, ProgressSink::noop()))
                .expect("techno fixture analysis");

        assert_eq!(
            result.energy_per_bar.len(),
            result.quantized_downbeats.len(),
            "one energy reading per bar of the downbeat lattice"
        );
        assert!(
            result
                .energy_per_bar
                .iter()
                .all(|db| (-80.0..=0.0).contains(db)),
            "{:?}",
            result.energy_per_bar
        );
        assert!(!result.segments.is_empty(), "the lattice must be published");
        let covered: usize = result.segments.iter().map(|segment| segment.count).sum();
        assert_eq!(
            covered,
            result.quantized_beats.len(),
            "segments expand to exactly the flattened grid"
        );
    }

    #[test]
    fn test_zero_model_files_advertised() {
        assert!(BeatQueenMary::MODEL_NAMES.is_empty());
        assert!(BeatQueenMary::MODEL_FILES.is_empty());
    }
}
