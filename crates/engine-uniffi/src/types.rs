//! The wire shapes of the capability surface.
//!
//! One record per `engine-core` type the Kotlin façade reads, with the field
//! names the Swift protocols use (`Protocols/*.swift`). They are mirrors, not
//! re-exports, for two reasons: `engine-core` is outside this lane's boundary,
//! and UniFFI has no `usize` — every count crosses as `i32` and every byte
//! count as `i64`, which is what Kotlin's `Int`/`Long` are, and what iOS's
//! `Int` means in each position.
//!
//! Every mirror carries a `From` in the direction the FFI needs it. A record
//! with no `From` is one Kotlin only ever sends.

use ferment_engine_core as core;
use ferment_engine_processors as processors;

use crate::{EngineFfiError, FfiResult};

// MARK: - AudioEventDetecting

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum AudioEvent {
    Speech,
    Singing,
    Music,
}

impl From<core::AudioEvent> for AudioEvent {
    fn from(value: core::AudioEvent) -> Self {
        match value {
            core::AudioEvent::Speech => Self::Speech,
            core::AudioEvent::Singing => Self::Singing,
            core::AudioEvent::Music => Self::Music,
        }
    }
}

impl From<AudioEvent> for core::AudioEvent {
    fn from(value: AudioEvent) -> Self {
        match value {
            AudioEvent::Speech => Self::Speech,
            AudioEvent::Singing => Self::Singing,
            AudioEvent::Music => Self::Music,
        }
    }
}

/// A continuous region where one event class is active. `startTime` +
/// `duration`, the canonical span shape; `endTime` is derived on both sides.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct AudioEventSegment {
    pub event: AudioEvent,
    pub start_time: f64,
    pub duration: f64,
    pub confidence: f32,
}

impl From<core::AudioEventSegment> for AudioEventSegment {
    fn from(value: core::AudioEventSegment) -> Self {
        Self {
            event: value.event.into(),
            start_time: value.start_time,
            duration: value.duration,
            confidence: value.confidence,
        }
    }
}

impl From<&AudioEventSegment> for core::AudioEventSegment {
    fn from(value: &AudioEventSegment) -> Self {
        Self {
            event: value.event.into(),
            start_time: value.start_time,
            duration: value.duration,
            confidence: value.confidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct AudioEventFrameProbabilities {
    pub speech: f32,
    pub singing: f32,
    pub music: f32,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct AudioEventDetectionResult {
    pub duration_seconds: f64,
    pub frame_shift_seconds: f64,
    pub frame_probs: Vec<AudioEventFrameProbabilities>,
}

impl From<core::AudioEventDetectionResult> for AudioEventDetectionResult {
    fn from(value: core::AudioEventDetectionResult) -> Self {
        Self {
            duration_seconds: value.duration_seconds,
            frame_shift_seconds: value.frame_shift_seconds,
            frame_probs: value
                .frame_probs
                .into_iter()
                .map(|probs| AudioEventFrameProbabilities {
                    speech: probs.speech,
                    singing: probs.singing,
                    music: probs.music,
                })
                .collect(),
        }
    }
}

// MARK: - SpeechTranscribing

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct TranscribedWord {
    pub id: String,
    pub text: String,
    pub start_time: f64,
    pub duration: f64,
    pub confidence: f32,
}

impl From<core::TranscribedWord> for TranscribedWord {
    fn from(value: core::TranscribedWord) -> Self {
        Self {
            id: value.id,
            text: value.text,
            start_time: value.start_time,
            duration: value.duration,
            confidence: value.confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct Caption {
    pub id: String,
    pub text: String,
    pub start_time: f64,
    pub duration: f64,
    pub words: Vec<TranscribedWord>,
}

impl From<core::Caption> for Caption {
    fn from(value: core::Caption) -> Self {
        Self {
            id: value.id,
            text: value.text,
            start_time: value.start_time,
            duration: value.duration,
            words: value.words.into_iter().map(TranscribedWord::from).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct TranscriptionResult {
    pub transcription: String,
    pub language: Option<String>,
    pub captions: Vec<Caption>,
}

impl From<core::TranscriptionResult> for TranscriptionResult {
    fn from(value: core::TranscriptionResult) -> Self {
        Self {
            transcription: value.transcription,
            language: value.language,
            captions: value.captions.into_iter().map(Caption::from).collect(),
        }
    }
}

// MARK: - LanguageIdentifying

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct LanguageHypothesis {
    pub code: String,
    pub probability: f32,
}

// MARK: - SpeakerEncoding / diarization

/// One scored window of speech, as a voice fingerprint. `startTime`/`endTime`
/// describe exactly the audio that produced `vector`, not the span it was cut
/// from.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct SpeakerSegmentEmbedding {
    pub start_time: f64,
    pub end_time: f64,
    pub vector: Vec<f32>,
}

/// One stretch of one voice, in seconds of the source file.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct SpeakerTurn {
    pub start_time: f64,
    pub end_time: f64,
    /// Another voice was heard over the whole of this stretch.
    pub overlapped: bool,
}

/// One voice of ONE element: where it was heard, what it sounds like, and how
/// sure the segmenter was.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct LocalVoice {
    pub turns: Vec<SpeakerTurn>,
    /// 192-d, unit length. EMPTY when the voice never filled a whole window.
    pub vector: Vec<f32>,
    pub confidence: f64,
}

impl From<core::LocalVoice> for LocalVoice {
    fn from(value: core::LocalVoice) -> Self {
        Self {
            turns: value
                .turns
                .into_iter()
                .map(|turn| SpeakerTurn {
                    start_time: turn.start_time,
                    end_time: turn.end_time,
                    overlapped: turn.overlapped,
                })
                .collect(),
            vector: value.vector,
            confidence: value.confidence,
        }
    }
}

// MARK: - Audio buffers

/// Mono float32 PCM plus its rate — what `AudioSampleSource` carries on iOS.
///
/// A flat `Vec<f32>` rather than the disk-backed source iOS uses for long
/// files: the FFI copies once into a Kotlin `FloatArray`, and the caller
/// streams from there. Long-file behaviour is §3 of the report.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: f64,
    pub channels: i32,
}

impl From<core::AudioBuffer> for AudioBuffer {
    fn from(value: core::AudioBuffer) -> Self {
        Self {
            samples: value.samples,
            sample_rate: f64::from(value.sample_rate),
            channels: i32::from(value.channels),
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct DenoisedAudio {
    pub audio: AudioBuffer,
}

/// An owned little-endian mono-f32 artifact, transferred without PCM arrays.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FileDenoisedAudio {
    pub path: String,
    pub sample_rate: u32,
    pub sample_count: u64,
}

// MARK: - LoudnessNormalizing

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct LoudnessTarget {
    pub target_lufs: f64,
    pub peak_ceiling_dbtp: f64,
    pub enable_leveler: bool,
}

impl From<LoudnessTarget> for core::LoudnessTarget {
    fn from(value: LoudnessTarget) -> Self {
        Self {
            target_lufs: value.target_lufs,
            peak_ceiling_dbtp: value.peak_ceiling_dbtp,
            enable_leveler: value.enable_leveler,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct LoudnessReport {
    pub spec_version: i32,
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

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct NormalizedAudio {
    pub audio: AudioBuffer,
    pub report: LoudnessReport,
}

impl From<core::NormalizedAudio> for NormalizedAudio {
    fn from(value: core::NormalizedAudio) -> Self {
        Self {
            audio: value.audio.into(),
            report: LoudnessReport {
                spec_version: value.report.spec_version as i32,
                measured_lufs: value.report.measured_lufs,
                measured_true_peak: value.report.measured_true_peak,
                gain_applied_db: value.report.gain_applied_db,
                post_limit_lufs: value.report.post_limit_lufs,
                post_limit_true_peak: value.report.post_limit_true_peak,
                limiter_engaged: value.report.limiter_engaged,
                within_clip_leveler_engaged: value.report.within_clip_leveler_engaged,
                qa_pass: value.report.qa_pass,
                target_lufs: value.report.target_lufs,
                peak_ceiling_dbtp: value.report.peak_ceiling_dbtp,
            },
        }
    }
}

// MARK: - BeatAnalyzing

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct BeatLatticeSegment {
    pub bpm: f64,
    pub first: f64,
    pub count: i32,
    pub downbeat_phase: Option<i32>,
    pub meter: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum EnergyLevel {
    Silence,
    Quiet,
    Mid,
    Loud,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct EnergySegment {
    pub from_bar: i32,
    pub to_bar: i32,
    pub level: EnergyLevel,
    pub mean_db: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum EnergyEventKind {
    Peak,
    Dip,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct EnergyEvent {
    pub bar: i32,
    pub kind: EnergyEventKind,
    pub delta_db: f64,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct EnergyStructure {
    pub segments: Vec<EnergySegment>,
    pub events: Vec<EnergyEvent>,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct BeatAnalysisResult {
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub quantized_beats: Vec<f64>,
    pub quantized_downbeats: Vec<f64>,
    pub grid_stability: f64,
    pub tempo_stability: f64,
    pub bpm: Option<f64>,
    pub quantized_bpm: Option<f64>,
    pub segments: Vec<BeatLatticeSegment>,
    pub energy_per_bar: Vec<f64>,
    pub energy_structure: EnergyStructure,
}

impl From<core::BeatAnalysisResult> for BeatAnalysisResult {
    fn from(value: core::BeatAnalysisResult) -> Self {
        Self {
            beats: value.beats,
            downbeats: value.downbeats,
            quantized_beats: value.quantized_beats,
            quantized_downbeats: value.quantized_downbeats,
            grid_stability: value.grid_stability,
            tempo_stability: value.tempo_stability,
            bpm: value.bpm,
            quantized_bpm: value.quantized_bpm,
            segments: value
                .segments
                .into_iter()
                .map(|segment| BeatLatticeSegment {
                    bpm: segment.bpm,
                    first: segment.first,
                    count: segment.count as i32,
                    downbeat_phase: segment.downbeat_phase.map(|phase| phase as i32),
                    meter: segment.meter.map(|meter| meter as i32),
                })
                .collect(),
            energy_per_bar: value.energy_per_bar,
            energy_structure: EnergyStructure {
                segments: value
                    .energy_structure
                    .segments
                    .into_iter()
                    .map(|segment| EnergySegment {
                        from_bar: segment.from_bar as i32,
                        to_bar: segment.to_bar as i32,
                        level: match segment.level {
                            core::EnergyLevel::Silence => EnergyLevel::Silence,
                            core::EnergyLevel::Quiet => EnergyLevel::Quiet,
                            core::EnergyLevel::Mid => EnergyLevel::Mid,
                            core::EnergyLevel::Loud => EnergyLevel::Loud,
                        },
                        mean_db: segment.mean_db,
                    })
                    .collect(),
                events: value
                    .energy_structure
                    .events
                    .into_iter()
                    .map(|event| EnergyEvent {
                        bar: event.bar as i32,
                        kind: match event.kind {
                            core::EnergyEventKind::Peak => EnergyEventKind::Peak,
                            core::EnergyEventKind::Dip => EnergyEventKind::Dip,
                        },
                        delta_db: event.delta_db,
                    })
                    .collect(),
            },
        }
    }
}

// MARK: - MusicHighlighting / MusicTagging / AudioSemanticEncoding

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct Highlight {
    pub window: i32,
    pub start_time: f32,
    pub duration: f32,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct HighlightResult {
    pub saliency: Vec<f32>,
    pub highlights: Vec<Highlight>,
    pub duration: i32,
}

impl From<core::HighlightResult> for HighlightResult {
    fn from(value: core::HighlightResult) -> Self {
        Self {
            saliency: value.saliency,
            highlights: value
                .highlights
                .into_iter()
                .map(|highlight| Highlight {
                    window: highlight.window as i32,
                    start_time: highlight.start_time,
                    duration: highlight.duration,
                })
                .collect(),
            duration: value.duration as i32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct ScoredTag {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct MusicTagsResult {
    pub genre: Vec<ScoredTag>,
    pub mood: Vec<ScoredTag>,
    pub instrument: Vec<ScoredTag>,
}

impl From<core::MusicTagsResult> for MusicTagsResult {
    fn from(value: core::MusicTagsResult) -> Self {
        let axis = |tags: Vec<core::ScoredTag>| {
            tags.into_iter()
                .map(|tag| ScoredTag {
                    name: tag.name,
                    score: tag.score,
                })
                .collect()
        };
        Self {
            genre: axis(value.genre),
            mood: axis(value.mood),
            instrument: axis(value.instrument),
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct AudioEmbedding {
    pub pooled: Vec<f32>,
    pub chunks: Option<Vec<Vec<f32>>>,
    pub duration_seconds: f64,
    pub window_seconds: i32,
    pub dim: i32,
}

impl From<core::AudioEmbedding> for AudioEmbedding {
    fn from(value: core::AudioEmbedding) -> Self {
        Self {
            pooled: value.pooled,
            chunks: value.chunks,
            duration_seconds: value.duration_seconds,
            window_seconds: value.window_seconds as i32,
            dim: value.dim as i32,
        }
    }
}

// MARK: - WaveformExtracting

/// The `audiowaveform`-compatible shape iOS answers as `[String: Any]`.
///
/// A record, not a JSON blob: the Kotlin side re-encodes it to the same JSON
/// the editor's `WaveformLayerCache` reads, and a typed record is what makes
/// a missing field a compile error rather than a null at read time.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct WaveformResult {
    pub duration: f64,
    pub rms: Vec<f32>,
    pub sample_rate: i32,
    pub channels: i32,
    pub samples_per_pixel: i32,
    pub bits: i32,
}

impl From<core::WaveformResult> for WaveformResult {
    fn from(value: core::WaveformResult) -> Self {
        Self {
            duration: value.duration,
            rms: value.rms,
            sample_rate: value.sample_rate as i32,
            channels: i32::from(value.channels),
            samples_per_pixel: value.samples_per_pixel as i32,
            bits: i32::from(value.bits),
        }
    }
}

// MARK: - ColorAnalyzing

/// Three linear-light components — iOS's `SIMD3<Double>`.
///
/// A record rather than a list: the arity is part of the measurement. A
/// `List<Double>` makes `[0]`/`[1]`/`[2]` a bet the Kotlin reader takes at
/// runtime, and it lets a two-component value cross as if it were three.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct Rgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl From<[f64; 3]> for Rgb {
    fn from([r, g, b]: [f64; 3]) -> Self {
        Self { r, g, b }
    }
}

/// A normalized (Cb,Cr) point in `[-0.5, 0.5]` — iOS's `SIMD2<Double>`.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct Chroma {
    pub cb: f64,
    pub cr: f64,
}

impl From<[f64; 2]> for Chroma {
    fn from([cb, cr]: [f64; 2]) -> Self {
        Self { cb, cr }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct ColorStats {
    pub pixel_count: i64,
    /// Mean linear RGB over ALL pixels.
    pub linear_mean: Rgb,
    /// The trimmed gray-world illuminant estimate; ABSENT when too few neutral
    /// pixels were found, and the caller falls back to `linear_mean`.
    pub neutral_mean: Option<Rgb>,
    pub mean_log_luma: f64,
    pub black_point: f64,
    pub white_point: f64,
    /// Normalized (Cb,Cr) centroid of skin pixels; absent when no skin found.
    pub skin_centroid: Option<Chroma>,
    pub skin_mass: f64,
    /// Normalized (Cb,Cr) centroid over all pixels — overall cast.
    pub chroma_mean: Chroma,
}

impl From<core::ColorStats> for ColorStats {
    fn from(value: core::ColorStats) -> Self {
        Self {
            pixel_count: value.pixel_count as i64,
            linear_mean: value.linear_mean.into(),
            neutral_mean: value.neutral_mean.map(Rgb::from),
            mean_log_luma: value.mean_log_luma,
            black_point: value.black_point,
            white_point: value.white_point,
            skin_centroid: value.skin_centroid.map(Chroma::from),
            skin_mass: value.skin_mass,
            chroma_mean: value.chroma_mean.into(),
        }
    }
}

// MARK: - Geometry and time

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl From<core::Rect> for Rect {
    fn from(value: core::Rect) -> Self {
        Self {
            x: value.x,
            y: value.y,
            width: value.width,
            height: value.height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct TimeRange {
    pub start: f64,
    pub duration: f64,
}

/// `CMTime` — a rational instant. Kept rational across the boundary so a
/// 1001/30000 source timeline survives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct MediaTime {
    pub value: i64,
    pub timescale: i32,
}

impl From<MediaTime> for core::MediaTime {
    fn from(value: MediaTime) -> Self {
        Self::new(value.value, value.timescale)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct MediaTimeRange {
    pub start: MediaTime,
    pub duration: MediaTime,
}

impl From<MediaTimeRange> for core::MediaTimeRange {
    fn from(value: MediaTimeRange) -> Self {
        Self {
            start: value.start.into(),
            duration: value.duration.into(),
        }
    }
}

// MARK: - Minimaps

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Enum)]
pub enum ImageFormat {
    Png,
    Jpeg { quality: f32 },
}

impl From<core::ImageFormat> for ImageFormat {
    fn from(value: core::ImageFormat) -> Self {
        match value {
            core::ImageFormat::Png => Self::Png,
            core::ImageFormat::Jpeg { quality } => Self::Jpeg { quality },
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct MinimapCell {
    pub scene_index: i32,
    pub frame_index: i32,
    pub time: f64,
    pub rect: Rect,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
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

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct MinimapResult {
    pub image_data: Vec<u8>,
    pub format: ImageFormat,
    pub width: i32,
    pub height: i32,
    pub layout: MinimapLayout,
    pub total_scenes: Option<i32>,
    pub shown_scenes: Option<i32>,
}

impl From<core::MinimapResult> for MinimapResult {
    fn from(value: core::MinimapResult) -> Self {
        Self {
            image_data: value.image_data,
            format: value.format.into(),
            width: value.width as i32,
            height: value.height as i32,
            layout: MinimapLayout {
                cells: value
                    .layout
                    .cells
                    .into_iter()
                    .map(|cell| MinimapCell {
                        scene_index: cell.scene_index as i32,
                        frame_index: cell.frame_index as i32,
                        time: cell.time,
                        rect: cell.rect.into(),
                    })
                    .collect(),
                header_rect: value.layout.header_rect.map(Rect::from),
                barcode_rect: value.layout.barcode_rect.map(Rect::from),
                mel_rect: value.layout.mel_rect.map(Rect::from),
                rms_rect: value.layout.rms_rect.map(Rect::from),
                time_axis_rect: value.layout.time_axis_rect.map(Rect::from),
                plot_left: value.layout.plot_left,
                plot_right: value.layout.plot_right,
                time_range: TimeRange {
                    start: value.layout.time_range.start,
                    duration: value.layout.time_range.duration,
                },
            },
            total_scenes: value.total_scenes.map(|count| count as i32),
            shown_scenes: value.shown_scenes.map(|count| count as i32),
        }
    }
}

// MARK: - VideoSceneDetecting

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoScene {
    pub index: i32,
    pub start_frame: i32,
    pub frame_count: i32,
    pub start_time: f64,
    pub duration: f64,
    pub start_confidence: Option<f32>,
    pub embedding: Option<Vec<f32>>,
}

impl From<core::VideoScene> for VideoScene {
    fn from(value: core::VideoScene) -> Self {
        Self {
            index: value.index as i32,
            start_frame: value.start_frame as i32,
            frame_count: value.frame_count as i32,
            start_time: value.start_time,
            duration: value.duration,
            start_confidence: value.start_confidence,
            embedding: value.embedding,
        }
    }
}

impl From<&VideoScene> for core::VideoScene {
    fn from(value: &VideoScene) -> Self {
        Self {
            index: value.index.max(0) as usize,
            start_frame: value.start_frame.max(0) as usize,
            frame_count: value.frame_count.max(0) as usize,
            start_time: value.start_time,
            duration: value.duration,
            start_confidence: value.start_confidence,
            embedding: value.embedding.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoSceneDetectionResult {
    pub scenes: Vec<VideoScene>,
    pub total_frames: i32,
    pub frame_rate: f64,
}

impl From<core::VideoSceneDetectionResult> for VideoSceneDetectionResult {
    fn from(value: core::VideoSceneDetectionResult) -> Self {
        Self {
            scenes: value.scenes.into_iter().map(VideoScene::from).collect(),
            total_frames: value.total_frames as i32,
            frame_rate: value.frame_rate,
        }
    }
}

impl From<&VideoSceneDetectionResult> for core::VideoSceneDetectionResult {
    fn from(value: &VideoSceneDetectionResult) -> Self {
        Self {
            scenes: value.scenes.iter().map(core::VideoScene::from).collect(),
            total_frames: value.total_frames.max(0) as usize,
            frame_rate: value.frame_rate,
        }
    }
}

// MARK: - VideoSaliencyAnalyzing

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum SaliencyMode {
    Attention,
    Objectness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum SaliencySampleSource {
    Face,
    Saliency,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct VideoSaliencySample {
    pub scene_index: i32,
    pub time: f64,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub source: SaliencySampleSource,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct VideoSaliencySceneVerdict {
    pub index: i32,
    pub sample_count: i32,
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    pub face_share: f64,
    pub spread: f64,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoSaliencyResult {
    pub frame_width: i32,
    pub frame_height: i32,
    pub mode: SaliencyMode,
    pub scenes: Vec<VideoSaliencySceneVerdict>,
    pub samples: Vec<VideoSaliencySample>,
}

// MARK: - VideoEncoding

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum VideoFileType {
    Mov,
    Mp4,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Enum)]
pub enum VideoRateControl {
    Average { bits_per_second: i64 },
    BitsPerPixel { bits_per_pixel: f64 },
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoEncodingPreset {
    pub label: String,
    pub file_type: VideoFileType,
    pub max_width: i32,
    pub caps_short_edge: bool,
    pub rate_control: VideoRateControl,
    pub b_frames: i32,
    pub key_frame_interval: i32,
    pub hdr_eligible: bool,
}

/// The crate's preset, or why this one is not a preset.
///
/// This direction used to be a `From`, which meant a clamp: `b_frames: 300`
/// became 255 and `max_width: -1` became 0. The preset is echoed back on
/// [`VideoEncodingResult`], so the caller was then handed a preset it never
/// wrote as the record of what it had encoded. A number the crate cannot hold
/// is refused instead, naming the field and quoting what was sent.
pub(crate) fn core_preset(value: &VideoEncodingPreset) -> FfiResult<core::VideoEncodingPreset> {
    fn field<T: TryFrom<i32>>(name: &str, value: i32) -> FfiResult<T> {
        T::try_from(value).map_err(|_| out_of_range(name, i64::from(value)))
    }
    Ok(core::VideoEncodingPreset {
        label: value.label.clone(),
        file_type: match value.file_type {
            VideoFileType::Mov => core::VideoFileType::Mov,
            VideoFileType::Mp4 => core::VideoFileType::Mp4,
        },
        max_width: field("max_width", value.max_width)?,
        caps_short_edge: value.caps_short_edge,
        rate_control: match value.rate_control {
            VideoRateControl::Average { bits_per_second } => core::VideoRateControl::Average(
                u64::try_from(bits_per_second)
                    .map_err(|_| out_of_range("bits_per_second", bits_per_second))?,
            ),
            VideoRateControl::BitsPerPixel { bits_per_pixel } => {
                core::VideoRateControl::BitsPerPixel(bits_per_pixel)
            }
        },
        b_frames: field("b_frames", value.b_frames)?,
        key_frame_interval: field("key_frame_interval", value.key_frame_interval)?,
        hdr_eligible: value.hdr_eligible,
    })
}

fn out_of_range(field: &str, value: i64) -> EngineFfiError {
    EngineFfiError::new(format!(
        "{field} {value} is out of range for a video preset"
    ))
}

impl From<core::VideoEncodingPreset> for VideoEncodingPreset {
    fn from(value: core::VideoEncodingPreset) -> Self {
        Self {
            label: value.label,
            file_type: match value.file_type {
                core::VideoFileType::Mov => VideoFileType::Mov,
                core::VideoFileType::Mp4 => VideoFileType::Mp4,
            },
            max_width: value.max_width as i32,
            caps_short_edge: value.caps_short_edge,
            rate_control: match value.rate_control {
                core::VideoRateControl::Average(bit_rate) => VideoRateControl::Average {
                    bits_per_second: bit_rate as i64,
                },
                core::VideoRateControl::BitsPerPixel(bits) => VideoRateControl::BitsPerPixel {
                    bits_per_pixel: bits,
                },
            },
            b_frames: i32::from(value.b_frames),
            key_frame_interval: value.key_frame_interval as i32,
            hdr_eligible: value.hdr_eligible,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoEncodingResult {
    pub output_path: String,
    pub bytes: i64,
    pub width: i32,
    pub height: i32,
    pub preset: VideoEncodingPreset,
    pub wall_seconds: f64,
}

impl From<core::VideoEncodingResult> for VideoEncodingResult {
    fn from(value: core::VideoEncodingResult) -> Self {
        Self {
            output_path: value.output_path.to_string_lossy().into_owned(),
            bytes: value.bytes as i64,
            width: value.width as i32,
            height: value.height as i32,
            preset: value.preset.into(),
            wall_seconds: value.wall_seconds,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoEncodingVariant {
    pub id: String,
    pub output_path: String,
    pub preset: VideoEncodingPreset,
    /// Thumbnails belong to the variant that asks for them; the frames tapped
    /// are always the shared decoded frames, so attaching a strip to a smaller
    /// variant changes the cell cadence and geometry, not which pixels are
    /// sampled.
    pub thumb_strip: Option<crate::ThumbStripOptions>,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct MultiVideoEncodingResult {
    pub variants: std::collections::HashMap<String, VideoEncodingResult>,
    pub wall_seconds: f64,
    pub decoded_frames: i32,
}

impl From<core::MultiVideoEncodingResult> for MultiVideoEncodingResult {
    fn from(value: core::MultiVideoEncodingResult) -> Self {
        Self {
            variants: value
                .variants
                .into_iter()
                .map(|(id, result)| (id, result.into()))
                .collect(),
            wall_seconds: value.wall_seconds,
            decoded_frames: value.decoded_frames as i32,
        }
    }
}

// MARK: - Render capabilities (slow motion, upscale, denoise)

/// Caller intent for motion blur on a slow-motion render.
///
/// Swift's three cases, `on(strength:)` included. `engine-core`'s trait takes
/// a strengthless `MotionBlurMode`, so the strength is dropped at the render
/// call and carried only by [`motion_blur_decide`] — which is the one place it
/// can be honoured, because that is where the crate's own policy reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum MotionBlurMode {
    Off,
    Auto,
    /// `strength` is clamped to 1...100 by the policy.
    On {
        strength: i32,
    },
}

impl From<MotionBlurMode> for core::MotionBlurMode {
    fn from(value: MotionBlurMode) -> Self {
        match value {
            MotionBlurMode::Off => Self::Disabled,
            MotionBlurMode::Auto => Self::Automatic,
            MotionBlurMode::On { strength } => Self::Enabled { strength },
        }
    }
}

impl From<MotionBlurMode> for processors::MotionBlurMode {
    fn from(value: MotionBlurMode) -> Self {
        match value {
            MotionBlurMode::Off => Self::Off,
            MotionBlurMode::Auto => Self::Auto,
            MotionBlurMode::On { strength } => Self::On { strength },
        }
    }
}

/// Why the policy decided as it did. The crate's eight cases, typed — where
/// `engine-core`'s render result degrades the same information to a free-form
/// `String` nothing in the engine ever writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum MotionBlurReason {
    ExplicitOff,
    ExplicitOn,
    AlreadyBlurred,
    CrispPreferred,
    FastShutterSource,
    MotionTooSmall,
    Indeterminate,
    Unsupported,
}

impl From<processors::MotionBlurReason> for MotionBlurReason {
    fn from(value: processors::MotionBlurReason) -> Self {
        match value {
            processors::MotionBlurReason::ExplicitOff => Self::ExplicitOff,
            processors::MotionBlurReason::ExplicitOn => Self::ExplicitOn,
            processors::MotionBlurReason::AlreadyBlurred => Self::AlreadyBlurred,
            processors::MotionBlurReason::CrispPreferred => Self::CrispPreferred,
            processors::MotionBlurReason::FastShutterSource => Self::FastShutterSource,
            processors::MotionBlurReason::MotionTooSmall => Self::MotionTooSmall,
            processors::MotionBlurReason::Indeterminate => Self::Indeterminate,
            processors::MotionBlurReason::Unsupported => Self::Unsupported,
        }
    }
}

/// Pre-pass measurement of the source. Displacements are per SOURCE frame
/// pair, as a fraction of frame height.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct SlowMotionAnalysis {
    /// Smear-ratio shutter angle; absent when the clip gave nothing to measure.
    pub source_shutter_angle: Option<f64>,
    pub median_displacement: f64,
    pub p95_displacement: f64,
    pub pairs_sampled: i32,
}

impl From<SlowMotionAnalysis> for processors::SlowMotionAnalysis {
    fn from(value: SlowMotionAnalysis) -> Self {
        Self {
            source_shutter_angle: value.source_shutter_angle,
            median_displacement: value.median_displacement,
            p95_displacement: value.p95_displacement,
            pairs_sampled: value.pairs_sampled.max(0) as usize,
        }
    }
}

/// What the policy decided and why — the five fields Swift's
/// `MotionBlurDecision` carries.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct MotionBlurDecision {
    pub applied: bool,
    pub strength: i32,
    pub source_shutter_angle: f64,
    pub output_smear_angle: f64,
    pub reason: MotionBlurReason,
}

impl From<processors::MotionBlurDecision> for MotionBlurDecision {
    fn from(value: processors::MotionBlurDecision) -> Self {
        Self {
            applied: value.applied,
            strength: value.strength,
            source_shutter_angle: value.source_shutter_angle,
            output_smear_angle: value.output_smear_angle,
            reason: value.reason.into(),
        }
    }
}

/// What a RENDER reported about motion blur.
///
/// `engine-core`'s shape, not Swift's: the trait answers `enabled` +
/// a free-form `reason` and drops `strength` and the smear angle. Named for
/// what it is rather than converted into a [`MotionBlurDecision`] it cannot
/// fill — see the lane report's ruling on `engine-core::MotionBlurDecision`.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct RenderMotionBlur {
    pub enabled: bool,
    pub strength: i32,
    pub reason: String,
    pub estimated_shutter_angle: Option<f64>,
    pub target_shutter_angle: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum VideoUpscaleMode {
    Enhance,
    Upscale { longest_side: u32 },
}

impl From<VideoUpscaleMode> for core::VideoUpscaleMode {
    fn from(value: VideoUpscaleMode) -> Self {
        match value {
            VideoUpscaleMode::Enhance => Self::Enhance,
            VideoUpscaleMode::Upscale { longest_side } => Self::Upscale { longest_side },
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoRenderResult {
    pub output_path: String,
    pub frames_written: i32,
    pub output_seconds: f64,
    pub wall_seconds: f64,
    pub output_width: Option<i32>,
    pub output_height: Option<i32>,
    pub motion_blur: Option<RenderMotionBlur>,
    pub audio_preserved: bool,
}

impl From<core::VideoRenderResult> for VideoRenderResult {
    fn from(value: core::VideoRenderResult) -> Self {
        Self {
            output_path: value.output_path.to_string_lossy().into_owned(),
            frames_written: value.frames_written as i32,
            output_seconds: value.output_seconds,
            wall_seconds: value.wall_seconds,
            output_width: value.output_width.map(|width| width as i32),
            output_height: value.output_height.map(|height| height as i32),
            audio_preserved: value.audio_preserved,
            motion_blur: value.motion_blur.map(|decision| RenderMotionBlur {
                enabled: decision.enabled,
                strength: decision.strength,
                reason: decision.reason,
                estimated_shutter_angle: decision.estimated_shutter_angle,
                target_shutter_angle: decision.target_shutter_angle,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kill: drop the false audio-preservation result while crossing the FFI boundary.
    #[test]
    fn a_silent_rescue_stays_reported_silent_across_the_render_record() {
        for preserved in [false, true] {
            let result = VideoRenderResult::from(core::VideoRenderResult {
                output_path: std::path::PathBuf::from("/render.mp4"),
                frames_written: 15,
                output_seconds: 14.0 / 30.0,
                wall_seconds: 0.25,
                output_width: Some(480),
                output_height: Some(272),
                motion_blur: None,
                audio_preserved: preserved,
            });
            assert_eq!(result.audio_preserved, preserved);
            assert_eq!(result.frames_written, 15);
            assert_eq!(result.output_width, Some(480));
        }
    }

    fn preset() -> VideoEncodingPreset {
        VideoEncodingPreset {
            label: "proxy".to_owned(),
            file_type: VideoFileType::Mp4,
            max_width: 1_080,
            caps_short_edge: true,
            rate_control: VideoRateControl::BitsPerPixel {
                bits_per_pixel: 0.75,
            },
            b_frames: 0,
            key_frame_interval: 1,
            hdr_eligible: false,
        }
    }

    /// A preset crosses as `i32`s and comes back on [`VideoEncodingResult`].
    /// A number the crate cannot hold used to be clamped on the way in —
    /// `b_frames: 300` became 255, `max_width: -1` became 0 — so the caller was
    /// handed, as the record of its own encode, a preset it never wrote.
    ///
    /// Kill: convert with `clamp` / `max(0)` again.
    /// The components cross as NAMED fields in iOS's order, and "no neutral
    /// pixels" is an absent value rather than an empty one — the doc used to
    /// say "empty" while the code answered `None`, which is a null where the
    /// Kotlin reader was told to expect a list.
    ///
    /// Kill: swap r/b (the BGRA order the analyzer reads in), or answer
    /// `Some(default)` where the measurement found nothing.
    #[test]
    fn the_colour_means_cross_as_named_components_in_ios_order() {
        let stats = ColorStats::from(core::ColorStats {
            linear_mean: [0.1, 0.2, 0.3],
            chroma_mean: [0.4, 0.5],
            skin_centroid: Some([0.6, 0.7]),
            ..core::ColorStats::default()
        });
        assert_eq!(
            stats.linear_mean,
            Rgb {
                r: 0.1,
                g: 0.2,
                b: 0.3
            }
        );
        assert_eq!(stats.chroma_mean, Chroma { cb: 0.4, cr: 0.5 });
        assert_eq!(stats.skin_centroid, Some(Chroma { cb: 0.6, cr: 0.7 }));
        assert_eq!(
            ColorStats::from(core::ColorStats::default()).neutral_mean,
            None
        );
    }

    #[test]
    fn a_preset_the_encoder_cannot_take_is_refused_rather_than_rewritten() {
        let refused = [
            VideoEncodingPreset {
                b_frames: 300,
                ..preset()
            },
            VideoEncodingPreset {
                max_width: -1,
                ..preset()
            },
            VideoEncodingPreset {
                key_frame_interval: -5,
                ..preset()
            },
            VideoEncodingPreset {
                rate_control: VideoRateControl::Average {
                    bits_per_second: -1,
                },
                ..preset()
            },
        ];
        for value in refused {
            let error = core_preset(&value)
                .expect_err("a number the crate cannot hold is not a preset")
                .to_string();
            assert!(
                error.ends_with("is out of range for a video preset"),
                "{error}"
            );
        }
        // ...and one the crate CAN take survives the round trip unchanged.
        let sent = preset();
        assert_eq!(VideoEncodingPreset::from(core_preset(&sent).unwrap()), sent);
    }
}
