use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    AudioBuffer, EngineService, ImageFormat, MediaTime, MediaTimeRange, ProgressSink, Rect, Result,
    TimeRange, VideoFrame,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioEvent {
    Speech,
    Singing,
    Music,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventSegment {
    pub event: AudioEvent,
    pub start_time: f64,
    pub duration: f64,
    pub confidence: f32,
}

impl AudioEventSegment {
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioEventClassStats {
    pub coverage: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioEventSummary {
    pub speech: AudioEventClassStats,
    pub song: AudioEventClassStats,
    pub music: AudioEventClassStats,
    pub musical: AudioEventClassStats,
    pub vocal: AudioEventClassStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventAnalysis {
    pub segments: Vec<AudioEventSegment>,
    pub summary: AudioEventSummary,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AudioEventFrameProbabilities {
    pub speech: f32,
    pub singing: f32,
    pub music: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEventDetectionResult {
    pub duration_seconds: f64,
    pub frame_shift_seconds: f64,
    pub frame_probs: Vec<AudioEventFrameProbabilities>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribedWord {
    pub id: String,
    pub text: String,
    pub start_time: f64,
    pub duration: f64,
    pub confidence: f32,
}

impl TranscribedWord {
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Caption {
    pub id: String,
    pub text: String,
    pub start_time: f64,
    pub duration: f64,
    pub words: Vec<TranscribedWord>,
}

impl Caption {
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub transcription: String,
    pub language: Option<String>,
    pub captions: Vec<Caption>,
}

/// One stretch of one voice, in seconds of the source file.
///
/// Ported from `Processors/Pyannote/PyannoteSpeakerSegmenter.swift`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub start_time: f64,
    pub end_time: f64,
    /// Another voice was heard over the whole of this stretch. Still this
    /// person speaking — the occurrence is real — but not material to
    /// fingerprint them from, so pooling skips it.
    pub overlapped: bool,
}

impl SpeakerTurn {
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// One voice of ONE element: where it was heard, what it sounds like, and how
/// sure the segmenter was. "Local" because the identity stops at this file —
/// whether this is the same person as a voice in another element is a question
/// about the project, answered elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalVoice {
    pub turns: Vec<SpeakerTurn>,
    /// 192-d, unit length — the mean of the encoder windows tiled across this
    /// voice's CLEAN replies. EMPTY when the voice never filled a whole
    /// 2.005 s window: the honest answer, and the caller marks it.
    pub vector: Vec<f32>,
    /// The segmenter's own certainty over the frames this voice owns, in
    /// `[0, 1]` — the mean winning-class probability, not a similarity.
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenoisedAudio {
    pub audio: AudioBuffer,
}

/// One constant-tempo stretch of the quantized warp lattice: `count` points
/// from `first` at `60 / bpm` seconds apart, every `meter`-th of them from
/// `downbeat_phase` a downbeat. The flattened `quantized_beats` /
/// `quantized_downbeats` arrays are the concatenated expansion of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatLatticeSegment {
    pub bpm: f64,
    pub first: f64,
    pub count: usize,
    /// Lattice index of the first downbeat within THIS segment; `None` when
    /// no meter was found. `meter` is the stride between downbeats.
    pub downbeat_phase: Option<usize>,
    pub meter: Option<usize>,
}

/// Loudness of a bar range, relative to the track's own full power.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnergyLevel {
    Silence,
    Quiet,
    Mid,
    Loud,
}

/// Bars are 1-based and inclusive — the minimap ruler's numbers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergySegment {
    pub from_bar: usize,
    pub to_bar: usize,
    pub level: EnergyLevel,
    pub mean_db: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnergyEventKind {
    Peak,
    Dip,
}

/// A short outlier inside a plateau — a hit or a hole, not a section.
/// `bar` is the event's first bar; `delta_db` its extreme against the
/// hosting segment's mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyEvent {
    pub bar: usize,
    pub kind: EnergyEventKind,
    pub delta_db: f64,
}

/// The track's loudness structure over the bar lattice: piecewise-constant
/// segments plus short outliers as events. Computed once at beat detection
/// and carried by both detectors.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyStructure {
    pub segments: Vec<EnergySegment>,
    pub events: Vec<EnergyEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeatAnalysisResult {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub quantized_beats: Vec<f64>,
    pub quantized_downbeats: Vec<f64>,
    pub grid_stability: f64,
    pub tempo_stability: f64,
    pub bpm: Option<f64>,
    pub quantized_bpm: Option<f64>,
    /// The warp lattice the flattened quantized arrays expand from. Empty
    /// when the detector publishes no lattice.
    #[serde(default)]
    pub segments: Vec<BeatLatticeSegment>,
    /// RMS in dBFS per BAR of the quantized downbeat lattice: entry `k` spans
    /// `quantized_downbeats[k] -> [k + 1]`, the last entry the partial bar
    /// after the final downbeat. Empty when the detector never held samples.
    #[serde(default)]
    pub energy_per_bar: Vec<f64>,
    /// The loudness structure segmented over that same column.
    #[serde(default)]
    pub energy_structure: EnergyStructure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Highlight {
    pub window: u32,
    pub start_time: f32,
    pub duration: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HighlightResult {
    pub saliency: Vec<f32>,
    pub highlights: Vec<Highlight>,
    pub duration: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredTag {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicTagsResult {
    pub genre: Vec<ScoredTag>,
    pub mood: Vec<ScoredTag>,
    pub instrument: Vec<ScoredTag>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisProbabilities {
    pub names: Vec<String>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawTagProbabilities {
    pub genre: AxisProbabilities,
    pub mood: AxisProbabilities,
    pub instrument: AxisProbabilities,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEmbedding {
    pub pooled: Vec<f32>,
    pub chunks: Option<Vec<Vec<f32>>>,
    pub duration_seconds: f64,
    pub window_seconds: u32,
    pub dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoudnessTarget {
    pub target_lufs: f64,
    pub peak_ceiling_dbtp: f64,
    pub enable_leveler: bool,
}

impl LoudnessTarget {
    pub const STANDARD: Self = Self {
        target_lufs: -18.0,
        peak_ceiling_dbtp: -1.0,
        enable_leveler: false,
    };
    pub const SPEECH: Self = Self {
        enable_leveler: true,
        ..Self::STANDARD
    };
}

impl Default for LoudnessTarget {
    fn default() -> Self {
        Self::STANDARD
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoudnessReport {
    pub spec_version: u32,
    pub measured_lufs: f64,
    pub measured_true_peak: f64,
    pub gain_applied_db: f64,
    pub post_limit_lufs: f64,
    pub post_limit_true_peak: f64,
    pub limiter_engaged: bool,
    pub within_clip_leveler_engaged: bool,
    pub qa_pass: bool,
    pub target_lufs: f64,
    pub peak_ceiling_dbtp: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedAudio {
    pub audio: AudioBuffer,
    pub report: LoudnessReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformResult {
    pub duration: f64,
    pub rms: Vec<f32>,
    /// Linear full-scale peaks, one per analysis bin. Unlike the legacy
    /// contrast-normalized RMS envelope these retain quiet sounds/transients.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peaks: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_per_pixel: usize,
    pub bits: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorStats {
    pub pixel_count: usize,
    pub linear_mean: [f64; 3],
    pub neutral_mean: Option<[f64; 3]>,
    pub mean_log_luma: f64,
    pub black_point: f64,
    pub white_point: f64,
    pub skin_centroid: Option<[f64; 2]>,
    pub skin_mass: f64,
    pub chroma_mean: [f64; 2],
}

impl Default for ColorStats {
    fn default() -> Self {
        Self {
            pixel_count: 0,
            linear_mean: [0.0; 3],
            neutral_mean: None,
            mean_log_luma: 0.0,
            black_point: 0.0,
            white_point: 1.0,
            skin_centroid: None,
            skin_mass: 0.0,
            chroma_mean: [0.0; 2],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorStatsKeyframe {
    pub time: f64,
    pub stats: ColorStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFileType {
    M4a,
    Wav,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioEncodingPreset {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_rate: u32,
    pub max_duration: f64,
    pub file_type: AudioFileType,
}

impl Default for AudioEncodingPreset {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 2,
            bit_rate: 128_000,
            max_duration: 600.0,
            file_type: AudioFileType::M4a,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEncodingResult {
    pub output_path: PathBuf,
    pub bytes: u64,
    pub duration: f64,
    pub did_passthrough: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimapCell {
    pub scene_index: usize,
    pub frame_index: usize,
    pub time: f64,
    pub rect: Rect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimapLayout {
    pub cells: Vec<MinimapCell>,
    pub header_rect: Option<Rect>,
    pub barcode_rect: Option<Rect>,
    pub mel_rect: Option<Rect>,
    pub rms_rect: Option<Rect>,
    pub time_axis_rect: Option<Rect>,
    pub plot_left: Option<f32>,
    pub plot_right: Option<f32>,
    pub time_range: TimeRange,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimapResult {
    pub image_data: Vec<u8>,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub layout: MinimapLayout,
    pub total_scenes: Option<usize>,
    pub shown_scenes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoScene {
    pub index: usize,
    pub start_frame: usize,
    pub frame_count: usize,
    pub start_time: f64,
    pub duration: f64,
    pub start_confidence: Option<f32>,
    /// DINOv2 medoid of the scene, 384 floats, unit length. All scenes carry
    /// one or none do (the all-or-nothing rule); absent when the model was
    /// unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoSceneDetectionResult {
    pub scenes: Vec<VideoScene>,
    pub total_frames: usize,
    pub frame_rate: f64,
}

impl VideoSceneDetectionResult {
    pub fn duration(&self) -> f64 {
        self.total_frames as f64 / self.frame_rate
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoFileType {
    Mov,
    Mp4,
}

/// Exactly one rate control per preset. Constant-quality is deliberately NOT
/// here: for H.264/HEVC it is a hardware-encoder privilege, so a
/// quality-driven preset means a capability probe and a platform-dependent
/// file. An average bit rate is the one control every encoder implements;
/// determinism comes from computing it, not from requesting a quality.
///
/// Ported from `VideoEncoding.swift` (`ca4f3e6c2`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoRateControl {
    /// Fixed average in bits/second, independent of output size and frame
    /// rate (the AI feed: its consumers are models, not eyes).
    Average(u64),
    /// Deterministic bits-per-pixel, resolved against the OUTPUT dimensions
    /// and the source frame rate at encode time. The right shape for an
    /// all-intra proxy: per-frame detail is held constant across sizes and
    /// frame rates, and the file's weight is known before the encode starts.
    BitsPerPixel(f64),
}

impl VideoRateControl {
    /// Frame rate used when a track states no usable one. Containers that
    /// report `0` (or a non-finite rate) resolve at the phone standard.
    pub const FALLBACK_FRAME_RATE: f64 = 30.0;

    #[must_use]
    pub fn resolved_bit_rate(self, width: u32, height: u32, frames_per_second: f64) -> u64 {
        match self {
            Self::Average(bit_rate) => bit_rate,
            Self::BitsPerPixel(bits_per_pixel) => {
                let fps = if frames_per_second.is_finite() && frames_per_second > 0.0 {
                    frames_per_second
                } else {
                    Self::FALLBACK_FRAME_RATE
                };
                let pixels = f64::from(width) * f64::from(height);
                (bits_per_pixel * pixels * fps) as u64
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoEncodingPreset {
    pub label: String,
    pub file_type: VideoFileType,
    /// The capped edge in pixels. Which edge it caps is [`Self::caps_short_edge`].
    pub max_width: u32,
    /// `true` caps the SHORT edge (a "1080p proxy" is 1080×1920 for a
    /// portrait 4K — the NLE half-resolution convention); `false` caps the
    /// LONG edge (the AI feed, sized to its model consumers).
    pub caps_short_edge: bool,
    pub rate_control: VideoRateControl,
    /// `0` suppresses frame reordering: delivery order equals presentation
    /// order, which callers that index frames positionally depend on.
    pub b_frames: u8,
    /// Maximum interval between keyframes, in FRAMES. `1` is all-intra.
    pub key_frame_interval: u32,
    pub hdr_eligible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThumbStripOptions {
    pub interval: f64,
    pub max_width: u32,
    pub max_height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThumbnailFrame {
    pub frame: VideoFrame,
    pub index: usize,
    pub presentation_time: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoEncodingResult {
    pub output_path: PathBuf,
    pub bytes: u64,
    pub width: u32,
    pub height: u32,
    pub preset: VideoEncodingPreset,
    pub wall_seconds: f64,
    pub thumbnails_emitted: usize,
}

pub trait ThumbnailSink: EngineService {
    fn emit(&self, thumbnail: ThumbnailFrame);
}

/// One requested output of a shared-decode run.
///
/// Thumbnails belong to the variant that asks for them; the frames tapped are
/// always the shared decoded frames, so attaching a strip to a smaller variant
/// changes the cell cadence and geometry, not which pixels are sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoEncodingVariant {
    pub id: String,
    pub output_path: PathBuf,
    pub preset: VideoEncodingPreset,
    pub thumb_strip: Option<ThumbStripOptions>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiVideoEncodingResult {
    /// Keyed by [`VideoEncodingVariant::id`], which is unique per request.
    pub variants: BTreeMap<String, VideoEncodingResult>,
    pub wall_seconds: f64,
    /// Source frames pulled through the shared decode. Variants derived from
    /// a finished sibling (the HDR split) do not add to this count.
    pub decoded_frames: usize,
}

/// Thumbnail sink for a multi-variant run, where the emitting variant has to
/// be named. The single-output path uses [`ThumbnailSink`].
pub trait VariantThumbnailSink: EngineService {
    fn emit(&self, variant_id: &str, thumbnail: ThumbnailFrame);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionBlurMode {
    Automatic,
    Disabled,
    Enabled { strength: i32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionBlurDecision {
    pub enabled: bool,
    pub strength: i32,
    pub reason: String,
    pub estimated_shutter_angle: Option<f64>,
    pub target_shutter_angle: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoUpscaleMode {
    Enhance,
    Upscale { longest_side: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoRenderResult {
    pub output_path: PathBuf,
    pub frames_written: usize,
    pub output_seconds: f64,
    pub wall_seconds: f64,
    pub output_width: Option<u32>,
    pub output_height: Option<u32>,
    pub motion_blur: Option<MotionBlurDecision>,
    /// True when the selected source soundtrack is retained, including a source
    /// without audio. False when a finished picture was rescued without sound.
    pub audio_preserved: bool,
}

#[async_trait(?Send)]
pub trait SpeechTranscribing: EngineService {
    async fn transcribe(
        &self,
        audio_path: &Path,
        speech_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<TranscriptionResult>;
}

/// Who spoke, when, and what they sound like — for one element.
///
/// Segmentation and embedding are ONE capability, not two, because a
/// segmenter's speaker labels are meaningless outside their own 10 s window:
/// an element's replies do not exist until the embedder has stitched the
/// chunks. Voices come back in the order they are first heard.
#[async_trait(?Send)]
pub trait SpeakerDiarizing: EngineService {
    async fn diarize(&self, audio_path: &Path, progress: ProgressSink) -> Result<Vec<LocalVoice>>;
}

#[async_trait(?Send)]
pub trait SpeechDenoising: EngineService {
    async fn denoise(
        &self,
        audio_path: &Path,
        keep_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<DenoisedAudio>;
}

#[async_trait(?Send)]
pub trait BeatAnalyzing: EngineService {
    async fn detect(&self, audio_path: &Path, progress: ProgressSink)
    -> Result<BeatAnalysisResult>;
}

#[async_trait(?Send)]
pub trait MusicHighlighting: EngineService {
    async fn detect(
        &self,
        audio_path: &Path,
        window_lengths: &[u32],
        progress: ProgressSink,
    ) -> Result<HighlightResult>;
}

#[async_trait(?Send)]
pub trait MusicTagging: EngineService {
    async fn tag(&self, embedding: &[f32], top_k: usize) -> Result<MusicTagsResult>;
}

#[async_trait(?Send)]
pub trait AudioEventDetecting: EngineService {
    async fn detect(
        &self,
        audio_path: &Path,
        progress: ProgressSink,
    ) -> Result<AudioEventDetectionResult>;
}

#[async_trait(?Send)]
pub trait LoudnessNormalizing: EngineService {
    async fn normalize(
        &self,
        audio_path: &Path,
        target: LoudnessTarget,
        progress: ProgressSink,
    ) -> Result<NormalizedAudio>;
}

#[async_trait(?Send)]
pub trait WaveformExtracting: EngineService {
    async fn extract(&self, path: &Path, sample_interval: f64) -> Result<WaveformResult>;
}

#[async_trait(?Send)]
pub trait AudioSemanticEncoding: EngineService {
    async fn encode(
        &self,
        audio_path: &Path,
        return_chunks: bool,
        progress: ProgressSink,
    ) -> Result<AudioEmbedding>;
}

#[async_trait(?Send)]
pub trait VideoSceneDetecting: EngineService {
    async fn detect(
        &self,
        video_path: &Path,
        progress: ProgressSink,
    ) -> Result<VideoSceneDetectionResult>;
}

#[async_trait(?Send)]
pub trait VideoUpscaling: EngineService {
    async fn render(
        &self,
        source_path: &Path,
        output_path: &Path,
        mode: VideoUpscaleMode,
        window: Option<MediaTimeRange>,
        scene_starts: &[MediaTime],
        progress: ProgressSink,
    ) -> Result<VideoRenderResult>;
}

#[async_trait(?Send)]
pub trait VideoNoiseFiltering: EngineService {
    async fn render(
        &self,
        source_path: &Path,
        output_path: &Path,
        strength: f32,
        window: Option<MediaTimeRange>,
        scene_starts: &[MediaTime],
        progress: ProgressSink,
    ) -> Result<VideoRenderResult>;
}

#[async_trait(?Send)]
pub trait VideoEncoding: EngineService {
    async fn encode(
        &self,
        source_path: &Path,
        output_path: &Path,
        preset: &VideoEncodingPreset,
        thumb_strip: Option<ThumbStripOptions>,
        progress: ProgressSink,
        on_thumbnail: Option<&dyn ThumbnailSink>,
    ) -> Result<VideoEncodingResult>;
}

/// Every derivative of one source from a SINGLE decode.
///
/// The decode, not the encode, is the cost on a long 4K import, so the
/// production path asks for all variants at once. HDR sources split: variants
/// that preserve HLG/PQ share the decode, and SDR-only variants are derived
/// from the finished HDR output afterwards.
#[async_trait(?Send)]
pub trait MultiVideoEncoding: EngineService {
    async fn encode_variants(
        &self,
        source_path: &Path,
        variants: &[VideoEncodingVariant],
        progress: ProgressSink,
        on_thumbnail: Option<&dyn VariantThumbnailSink>,
    ) -> Result<MultiVideoEncodingResult>;
}

#[async_trait(?Send)]
pub trait AudioEncoding: EngineService {
    async fn encode(
        &self,
        source_path: &Path,
        output_path: &Path,
        preset: AudioEncodingPreset,
        progress: ProgressSink,
    ) -> Result<AudioEncodingResult>;
}

#[async_trait(?Send)]
pub trait SlowMotionRendering: EngineService {
    #[allow(clippy::too_many_arguments)]
    async fn render(
        &self,
        source_path: &Path,
        output_path: &Path,
        factor: u8,
        window: Option<MediaTimeRange>,
        scene_starts: &[MediaTime],
        motion_blur: MotionBlurMode,
        target_shutter_angle: f64,
        progress: ProgressSink,
    ) -> Result<VideoRenderResult>;
}

pub trait ColorAnalyzing: EngineService {
    fn accumulate(&mut self, frame: &VideoFrame) -> Result<()>;
    fn result(&self) -> ColorStats;
}

#[async_trait(?Send)]
pub trait AudioMinimapComposing: EngineService {
    /// `bar_ruler` = downbeat times in track seconds, iOS's `barRuler:`.
    /// Non-empty, the image gains a BAR RULER — a downbeat vertical through
    /// mel + RMS — the axis a vision LLM reads structure against ("the drop is
    /// bar 35"). An axis, not metadata: the numbers themselves ride in
    /// `MinimapResult::layout`. Empty is the plain render.
    async fn compose(
        &self,
        audio_path: &Path,
        bar_ruler: &[f64],
        progress: ProgressSink,
    ) -> Result<MinimapResult>;
}

#[async_trait(?Send)]
pub trait VideoMinimapComposing: EngineService {
    async fn compose(
        &self,
        video_path: &Path,
        scenes: &VideoSceneDetectionResult,
        progress: ProgressSink,
    ) -> Result<MinimapResult>;
}

#[cfg(test)]
mod video_rate_control_tests {
    use super::VideoRateControl;

    #[test]
    fn bits_per_pixel_multiplies_output_pixels_by_frame_rate() {
        // The 1080-short-edge portrait proxy at 60 fps: ~93 Mbit/s.
        assert_eq!(
            VideoRateControl::BitsPerPixel(0.75).resolved_bit_rate(1_080, 1_920, 60.0),
            (0.75 * 1_080.0 * 1_920.0 * 60.0) as u64
        );
        // Halving the frame rate halves the rate; the per-frame budget holds.
        assert_eq!(
            VideoRateControl::BitsPerPixel(0.75).resolved_bit_rate(1_080, 1_920, 30.0),
            (0.75 * 1_080.0 * 1_920.0 * 30.0) as u64
        );
    }

    #[test]
    fn the_same_inputs_resolve_to_the_same_rate_every_time() {
        let control = VideoRateControl::BitsPerPixel(0.75);
        let first = control.resolved_bit_rate(1_080, 1_920, 59.94);
        for _ in 0..1_000 {
            assert_eq!(control.resolved_bit_rate(1_080, 1_920, 59.94), first);
        }
        assert_eq!(first, 93_218_688);
    }

    #[test]
    fn an_unusable_frame_rate_falls_back_to_thirty() {
        let control = VideoRateControl::BitsPerPixel(0.75);
        let at_thirty = control.resolved_bit_rate(1_080, 1_920, 30.0);
        for unusable in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                control.resolved_bit_rate(1_080, 1_920, unusable),
                at_thirty,
                "frame rate {unusable} must resolve at the {} fallback",
                VideoRateControl::FALLBACK_FRAME_RATE
            );
        }
        // A rate that IS usable, however small, is honoured rather than
        // rounded up to the fallback.
        assert_ne!(control.resolved_bit_rate(1_080, 1_920, 1.0), at_thirty);
    }

    #[test]
    fn an_average_rate_ignores_geometry_and_frame_rate() {
        let control = VideoRateControl::Average(2_500_000);
        assert_eq!(control.resolved_bit_rate(640, 360, 60.0), 2_500_000);
        assert_eq!(control.resolved_bit_rate(3_840, 2_160, 0.0), 2_500_000);
    }
}

/// A frame handed to a saliency or face source: packed RGB24, row-major and
/// tightly packed, so `data.len() == 3 * width * height`. Callers holding a
/// strided buffer copy the rows out first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbFrameRef<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
}

impl<'a> RgbFrameRef<'a> {
    /// `None` when the buffer does not hold exactly `3 * width * height` bytes.
    pub fn new(data: &'a [u8], width: u32, height: u32) -> Option<Self> {
        let expected = 3usize
            .checked_mul(width as usize)?
            .checked_mul(height as usize)?;
        (data.len() == expected && expected > 0).then_some(Self {
            data,
            width,
            height,
        })
    }
}

/// What a heat source measures. iOS ran Apple Vision, whose `attention` mode
/// answers "where a human eye lands"; a salient-object segmenter answers
/// "where the objects are". The two produce differently shaped maps, so the
/// meaning of `VideoSaliencySample::confidence` depends on which one ran —
/// the source declares it rather than the wiring assuming it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaliencyMode {
    Attention,
    Objectness,
}

/// A dense heat map covering the whole frame. Row 0 is the TOP of the frame —
/// it is an image buffer, not a bottom-left bounding box — and the values are
/// the source's own scale, never renormalized: the pedestal under the blob is
/// load-bearing for [`SaliencyHeatSource`] calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct SaliencyHeatMap {
    pub values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// One detected face, normalized to the frame with a TOP-LEFT origin
/// (x right, y down). Sources that work in a bottom-left space flip before
/// returning, and boxes are returned as measured — clipping to the frame is
/// the measurement layer's job.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
}

/// Dense saliency for one frame. One frame at a time, mirroring the Swift
/// `VideoSaliencyMeasure.sample` seam; the analyzer's grab loop is the thing
/// that keeps a long video's frames from piling up.
#[async_trait(?Send)]
pub trait SaliencyHeatSource: EngineService {
    fn mode(&self) -> SaliencyMode;
    async fn heat(&self, frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap>;
}

/// Face rectangles for one frame, strongest-first order not required.
#[async_trait(?Send)]
pub trait FaceRectSource: EngineService {
    async fn faces(&self, frame: RgbFrameRef<'_>) -> Result<Vec<FaceRect>>;
}

// ---------------------------------------------------------------------------
// Live speech transcription
//
// Ported from `Protocols/LiveSpeechTranscribing.swift` and
// `Processors/Nemotron/`.
// ---------------------------------------------------------------------------

/// Windowed mono-f32 access used by long-running processors.
///
/// Ported from `Processors/Shared/AudioSampleSource.swift`. It sits in
/// engine-core rather than engine-processors because [`LiveTranscriptionSession`]
/// — a contract — reads through it, and a contract cannot depend on the crate
/// that implements it.
pub trait AudioSampleSource: Send + Sync {
    fn sample_count(&self) -> usize;

    /// Copy at most `count` samples into `destination`, leaving the unused
    /// suffix untouched. Returns the number copied.
    fn copy_samples(&self, destination: &mut [f32], offset: usize, count: usize) -> Result<usize>;
}

/// What a live session knows mid-take: words it will not revise, and the
/// tail it still might.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveTranscript {
    pub committed: Vec<TranscribedWord>,
    pub volatile: Vec<TranscribedWord>,
}

impl LiveTranscript {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn committed_text(&self) -> String {
        Self::text_of(&self.committed)
    }

    #[must_use]
    pub fn volatile_text(&self) -> String {
        Self::text_of(&self.volatile)
    }

    /// The words joined by single spaces, outer whitespace trimmed.
    ///
    /// `trim` is Unicode-aware exactly as Swift's
    /// `trimmingCharacters(in: .whitespaces)` is; `trim_ascii` would not be.
    #[must_use]
    pub fn text_of(words: &[TranscribedWord]) -> String {
        words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_owned()
    }
}

/// One take's worth of live transcription over a buffer that is still being
/// written. The driver calls [`Self::update`] at its own pace with the same
/// source each time — the session reads whatever prefix exists at that moment
/// and answers with its current hypothesis. [`Self::finish`] runs once, over
/// the whole buffer, and its result is the same one the file path would produce
/// for that audio.
///
/// Sources are mono at the transcriber's own sample rate.
/// `Send` for the same reason the segmentation sessions are: the session is
/// state the caller owns across calls, and iOS's own protocol says
/// `AnyObject, Sendable`.
#[async_trait(?Send)]
pub trait LiveTranscriptionSession: Send {
    async fn update(&mut self, source: &dyn AudioSampleSource) -> Result<LiveTranscript>;
    async fn finish(&mut self, source: &dyn AudioSampleSource) -> Result<TranscriptionResult>;
    async fn cancel(&mut self);
}

/// A transcriber that can also run against a live buffer.
#[async_trait(?Send)]
pub trait LiveSpeechTranscribing: SpeechTranscribing {
    /// Loads whatever the session needs — weights, system assets — so the
    /// first [`LiveTranscriptionSession::update`] is a decode, not a wait.
    async fn make_live_session(&self) -> Result<Box<dyn LiveTranscriptionSession>>;
}
