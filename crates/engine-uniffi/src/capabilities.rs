//! One UniFFI object per iOS capability protocol.
//!
//! `Packages/FermentEngine/Sources/FermentEngine/Protocols/*.swift` is the
//! spec: same member names, same parameter order, same optional semantics
//! (`speechSegments == []` short-circuits without loading the model; `nil`
//! means the whole file). What each object wraps is the perception crate's
//! own processor — nothing about a processor is re-implemented here.
//!
//! **The methods are synchronous.** The crate's capability traits are
//! `#[async_trait(?Send)]`, and UniFFI's async export requires a `Send`
//! future (`uniffi_core::UniffiCompatibleFuture`), so an `async fn` here
//! would not compile against the engine as it exists. Each method blocks on
//! its own future exactly as E0's `infer_smoke` does, and Kotlin calls it
//! from `Dispatchers.IO` — the same thread discipline `FermentEngine.runOnDSP`
//! gives the Swift engine.

use std::path::Path;
use std::sync::Arc;

use ferment_engine_audio::{AudioFrames, resample_sinc};
use ferment_engine_core::{self as core, SaliencyHeatSource as _};
use ferment_engine_processors as processors;
use ferment_engine_processors::audio_source::{ArrayAudioSampleSource, AudioSource};
use ferment_engine_processors::{
    AmberNetLanguageId, AudioEventDetector as CrateAudioEventDetector,
    AudioMinimapComposer as CrateAudioMinimapComposer, AudioSemanticEncoder as CrateAudioEncoder,
    BeatQueenMary as CrateBeatQueenMary, BeatThis as CrateBeatThis, CpuColorAnalyzer,
    DeepFilterNet as CrateDeepFilterNet, GigaAm, LoudnessNormalizer as CrateLoudnessNormalizer,
    MusicHighlighter as CrateMusicHighlighter, MusicTagger as CrateMusicTagger,
    Parakeet as CrateParakeet, PyannoteSpeakerSegmenter, ReDimNetSpeakerEncoder,
    SlowMotionRenderer as CrateSlowMotionRenderer, U2NetpHeatSource,
    VideoEncoder as CrateVideoEncoder, VideoMinimapComposer as CrateVideoMinimapComposer,
    VideoNoiseFilter as CrateVideoNoiseFilter, VideoSaliencyAnalyzer as CrateVideoSaliencyAnalyzer,
    VideoSceneDetector as CrateVideoSceneDetector, VideoUpscaler as CrateVideoUpscaler,
    WaveformExtractor as CrateWaveformExtractor, YuNetFaceSource,
    audio_minimap_composer::AudioMinimapComposerConfig,
    video_minimap_composer::VideoMinimapComposerConfig,
};
use ferment_engine_processors::{ambernet, gigaam, parakeet};

use crate::audio_front_end::DerivedAudioCache;
use crate::progress::{CallProgress, Cancellation, ProgressListener, call_progress};
use crate::service::EngineService;
use crate::types::*;
use crate::{EngineFfiError, FfiResult, LiveTranscriptionSession};

/// The two optional halves every long call takes.
type Listener = Option<Arc<dyn ProgressListener>>;
type Token = Option<Arc<Cancellation>>;

fn block<T>(future: impl std::future::Future<Output = core::Result<T>>) -> FfiResult<T> {
    futures_executor::block_on(future).map_err(EngineFfiError::from)
}

fn core_segments(
    segments: &Option<Vec<AudioEventSegment>>,
) -> Option<Vec<core::AudioEventSegment>> {
    segments
        .as_ref()
        .map(|segments| segments.iter().map(core::AudioEventSegment::from).collect())
}

fn decode(path: &str) -> FfiResult<AudioFrames> {
    block(AudioSource::default().decode(Path::new(path)))
}

// MARK: - AudioEventDetecting

/// `AudioEventDetecting` — FireRedVAD's per-frame sigmoids, no thresholding.
/// Segmentation is `AudioEventSegmenter`'s, which is Kotlin's (pure logic).
#[derive(uniffi::Object)]
pub struct AudioEventDetector(CrateAudioEventDetector);

#[uniffi::export]
impl AudioEventDetector {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateAudioEventDetector::new(service.backend()))
    }

    pub fn detect(
        &self,
        audio_url: String,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<AudioEventDetectionResult> {
        let progress = call_progress(on_progress, cancellation);
        block(core::AudioEventDetecting::detect(
            &self.0,
            Path::new(&audio_url),
            progress.sink(),
        ))
        .map(Into::into)
    }
}

// MARK: - SpeechTranscribing

/// Which transcriber a route resolved to. `TranscriberRouting` on the app side
/// picks; this is the engine half of that choice, and the two Conformers are
/// separate graphs, not one model with a language flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum TranscriberKind {
    /// Parakeet-TDT-0.6B-v3 — 25 languages.
    Parakeet,
    /// GigaAM-v3-e2e-rnnt — Russian.
    GigaAm,
}

/// `SpeechTranscribing`. `speechSegments`: `nil` = the whole file, `[]` = an
/// empty result with no model load, otherwise inference over those regions
/// with absolute timestamps.
#[derive(uniffi::Object)]
pub struct SpeechTranscriber {
    parakeet: CrateParakeet,
    gigaam: GigaAm,
    kind: TranscriberKind,
}

#[uniffi::export]
impl SpeechTranscriber {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>, kind: TranscriberKind) -> Self {
        Self {
            parakeet: CrateParakeet::new(
                service.backend(),
                Path::new(&service.models_directory()).join(parakeet::host_decode::FILE),
            ),
            gigaam: GigaAm::new(service.backend()),
            kind,
        }
    }

    pub fn kind(&self) -> TranscriberKind {
        self.kind
    }

    /// The 25 (Parakeet) or 1 (GigaAM) language codes this transcriber owns —
    /// the ENGINE's knowledge of its own model, which `TranscriberRouting`
    /// reads rather than restating.
    pub fn supported_language_codes(&self) -> Vec<String> {
        let codes: &[&str] = match self.kind {
            TranscriberKind::Parakeet => &CrateParakeet::SUPPORTED_LANGUAGE_CODES,
            TranscriberKind::GigaAm => &GigaAm::SUPPORTED_LANGUAGE_CODES,
        };
        codes.iter().map(|code| (*code).to_owned()).collect()
    }

    pub fn transcribe(
        &self,
        audio_url: String,
        temporary_directory: String,
        speech_segments: Option<Vec<AudioEventSegment>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<TranscriptionResult> {
        let progress = call_progress(on_progress, cancellation);
        let segments = core_segments(&speech_segments);
        match self.kind {
            TranscriberKind::Parakeet => block(self.parakeet.transcribe_file_in(
                Path::new(&audio_url),
                Path::new(&temporary_directory),
                segments.as_deref(),
                progress.sink(),
            ))
            .map(Into::into),
            TranscriberKind::GigaAm => self.transcribe_with_gigaam(&audio_url, segments, &progress),
        }
    }
}

impl SpeechTranscriber {
    /// GigaAM has no path-taking capability impl in the crate (it is Russian's
    /// branch of the router, and the crate's `SpeechTranscribing` is
    /// Parakeet's). The decode and the caption pass are the same ones
    /// `capability_impls.rs` runs for Parakeet, so the two routes answer the
    /// same shape.
    fn transcribe_with_gigaam(
        &self,
        audio_url: &str,
        segments: Option<Vec<core::AudioEventSegment>>,
        progress: &CallProgress,
    ) -> FfiResult<TranscriptionResult> {
        if segments.as_deref().is_some_and(<[_]>::is_empty) {
            progress.report(1.0, "No speech detected");
            return Ok(TranscriptionResult {
                transcription: String::new(),
                language: None,
                captions: Vec::new(),
            });
        }
        let frames = decode(audio_url)?;
        let words = block(
            self.gigaam
                .transcribe(frames, segments.as_deref(), progress.sink()),
        )?;
        let transcription = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let captions = ferment_engine_processors::CaptionSegmenter::captions(&words);
        Ok(TranscriptionResult {
            transcription,
            // GigaAM is the Russian branch of the router; the graph carries no
            // language head, so the model's own declaration is the answer —
            // matched, not restated, the way the provider names are. Empty
            // output names no language, matching the explicit no-speech path.
            language: (!words.is_empty()).then(|| gigaam::LANGUAGE.to_owned()),
            captions: captions.into_iter().map(Caption::from).collect(),
        })
    }
}

// MARK: - LanguageIdentifying

/// `LanguageIdentifying` — AmberNet over VoxLingua107, hypotheses sorted by
/// probability descending.
#[derive(uniffi::Object)]
pub struct LanguageIdentifier(AmberNetLanguageId);

#[uniffi::export]
impl LanguageIdentifier {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(AmberNetLanguageId::new(service.backend()))
    }

    pub fn identify(
        &self,
        audio_url: String,
        speech_segments: Option<Vec<AudioEventSegment>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<Vec<LanguageHypothesis>> {
        let progress = call_progress(on_progress, cancellation);
        let segments = core_segments(&speech_segments);
        if segments.as_deref().is_some_and(<[_]>::is_empty) {
            progress.report(1.0, "No speech detected");
            return Ok(Vec::new());
        }
        let source = language_id_source(decode(&audio_url)?)?;
        let result = block(
            self.0
                .identify(&source, segments.as_deref(), progress.sink()),
        )?;
        Ok(result
            .hypotheses
            .into_iter()
            .map(|hypothesis| LanguageHypothesis {
                code: hypothesis.code,
                probability: hypothesis.probability,
            })
            .collect())
    }
}

/// Decoded audio as the source AmberNet reads, at AmberNet's own rate.
///
/// `AudioSampleSource` carries samples and nothing else, and `build_window`
/// converts every bound with [`ambernet::SAMPLE_RATE`] — so the rate is an
/// invariant of what is handed over, not of what is read. `decode` answers at
/// the file's native rate. This is `ReDimNetSpeakerEncoder::to_model_rate`
/// applied to the other source-taking processor, and for the same reason its
/// doc gives: every offset downstream is in model samples.
fn language_id_source(frames: AudioFrames) -> FfiResult<ArrayAudioSampleSource> {
    let samples = if frames.sample_rate == ambernet::SAMPLE_RATE {
        frames.samples
    } else {
        resample_sinc(&frames.samples, frames.sample_rate, ambernet::SAMPLE_RATE)?
    };
    Ok(ArrayAudioSampleSource::new(samples))
}

// MARK: - SpeakerEncoding + diarization

/// `SpeakerEncoding` — ReDimNet2-B2. An embedder, not a diarizer: spans in,
/// L2-normalised vectors out, in time order. Segments shorter than the model's
/// window are skipped rather than padded.
#[derive(uniffi::Object)]
pub struct SpeakerEncoder(ReDimNetSpeakerEncoder);

#[uniffi::export]
impl SpeakerEncoder {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(ReDimNetSpeakerEncoder::new(service.backend()))
    }

    pub fn embed(
        &self,
        audio_url: String,
        segments: Vec<AudioEventSegment>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<Vec<SpeakerSegmentEmbedding>> {
        let progress = call_progress(on_progress, cancellation);
        if segments.is_empty() {
            progress.report(1.0, "No speech to embed");
            return Ok(Vec::new());
        }
        progress.report(0.1, "loading audio");
        let audio = ReDimNetSpeakerEncoder::to_model_rate(decode(&audio_url)?)
            .map_err(EngineFfiError::from)?;
        let spans: Vec<core::TimeRange> = segments
            .iter()
            .map(|segment| core::TimeRange::new(segment.start_time, segment.duration))
            .collect();
        let embeddings = block(self.0.embed(&audio, &spans))?;
        progress.report(1.0, "embedded");
        Ok(embeddings
            .into_iter()
            .map(|embedding| SpeakerSegmentEmbedding {
                start_time: embedding.start_time,
                end_time: embedding.end_time,
                vector: embedding.vector,
            })
            .collect())
    }
}

/// Who spoke, when, and what they sound like — pyannote's segmentation and
/// ReDimNet's fingerprints as one capability, because a segmenter's labels are
/// meaningless outside their own 10 s window.
#[derive(uniffi::Object)]
pub struct SpeakerDiarizer(PyannoteSpeakerSegmenter);

#[uniffi::export]
impl SpeakerDiarizer {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(PyannoteSpeakerSegmenter::new(service.backend()))
    }

    pub fn diarize(
        &self,
        audio_url: String,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<Vec<LocalVoice>> {
        let progress = call_progress(on_progress, cancellation);
        let voices = block(core::SpeakerDiarizing::diarize(
            &self.0,
            Path::new(&audio_url),
            progress.sink(),
        ))?;
        Ok(voices.into_iter().map(LocalVoice::from).collect())
    }

    /// The name that rides beside a stored vector, iOS's `encoderName`.
    pub fn encoder_name(&self) -> String {
        ReDimNetSpeakerEncoder::encoder_name().to_owned()
    }

    /// The encoder checkpoint's digest, iOS's `encoderHash` — the contract's
    /// (`ReDimNetSpeakerEncoder::ENCODER_HASH`).
    ///
    /// A caller that stores a speaker vector stores this beside it: two
    /// checkpoints produce incomparable embeddings, so a vector whose encoder
    /// cannot be named must not be folded into a pool.
    pub fn encoder_hash(&self) -> FfiResult<String> {
        Ok(ReDimNetSpeakerEncoder::encoder_hash().to_owned())
    }

    /// The segmenter half's digest. A voice is the product of BOTH graphs.
    pub fn segmenter_hash(&self) -> FfiResult<String> {
        Ok(PyannoteSpeakerSegmenter::segmenter_hash().to_owned())
    }
}

// MARK: - SpeechDenoising

/// `SpeechDenoising` — DeepFilterNet. `keepSegments` gates the OUTPUT with
/// raised-cosine edges; the model runs over the whole input either way.
#[derive(uniffi::Object)]
pub struct SpeechDenoiser(CrateDeepFilterNet);

#[uniffi::export]
impl SpeechDenoiser {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateDeepFilterNet::new(service.backend()))
    }

    pub fn denoise(
        &self,
        audio_url: String,
        keep_segments: Option<Vec<AudioEventSegment>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<DenoisedAudio> {
        let progress = call_progress(on_progress, cancellation);
        let segments = core_segments(&keep_segments);
        let result = block(core::SpeechDenoising::denoise(
            &self.0,
            Path::new(&audio_url),
            segments.as_deref(),
            progress.sink(),
        ))?;
        Ok(DenoisedAudio {
            audio: result.audio.into(),
        })
    }

    /// The Android file-mode entry point. The caller owns the returned file
    /// and must delete it after closing its sample source. Errors transfer no
    /// file and leave the original input untouched.
    pub fn denoise_to_file(
        &self,
        audio_url: String,
        temporary_directory: String,
        keep_segments: Option<Vec<AudioEventSegment>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<FileDenoisedAudio> {
        use ferment_engine_core::AudioSampleSource;
        let progress = call_progress(on_progress, cancellation);
        let segments = core_segments(&keep_segments);
        let source = block(self.0.denoise_to_file(
            Path::new(&audio_url),
            Path::new(&temporary_directory),
            segments.as_deref(),
            progress.sink(),
        ))?;
        let sample_count = source.sample_count() as u64;
        Ok(FileDenoisedAudio {
            path: source.into_path().to_string_lossy().into_owned(),
            sample_rate: 48_000,
            sample_count,
        })
    }
}

// MARK: - BeatAnalyzing

/// Which beat detector ran. iOS forks on `ChipGeneration.atLeast(BeatThis
/// .minChipGeneration)`; on Android the fork is the caller's, and the DSP-only
/// Queen Mary detector needs no model at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum BeatDetectorKind {
    /// The transformer ensemble — two graphs, the primary lane.
    BeatThis,
    /// Pure DSP, no model. The fallback below the chip floor on iOS, and the
    /// answer here for a device the ensemble is too heavy for.
    QueenMary,
}

/// `BeatAnalyzing` — beats, downbeats, the quantized warp lattice and the
/// energy structure, all in one measurement.
#[derive(uniffi::Object)]
pub struct BeatAnalyzer {
    beat_this: CrateBeatThis,
    queen_mary: CrateBeatQueenMary,
    kind: BeatDetectorKind,
}

#[uniffi::export]
impl BeatAnalyzer {
    #[uniffi::constructor]
    pub fn with_cache(
        service: Arc<EngineService>,
        kind: BeatDetectorKind,
        cache: Option<Arc<DerivedAudioCache>>,
    ) -> Self {
        let mut value = Self::new(service, kind);
        value.beat_this = value.beat_this.with_cache(cache.map(|c| c.inner.clone()));
        value
    }

    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>, kind: BeatDetectorKind) -> Self {
        Self {
            beat_this: CrateBeatThis::new(service.backend()),
            queen_mary: CrateBeatQueenMary::new(),
            kind,
        }
    }

    pub fn kind(&self) -> BeatDetectorKind {
        self.kind
    }

    pub fn detect(
        &self,
        audio_url: String,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<BeatAnalysisResult> {
        let progress = call_progress(on_progress, cancellation);
        let path = Path::new(&audio_url);
        match self.kind {
            BeatDetectorKind::BeatThis => block(core::BeatAnalyzing::detect(
                &self.beat_this,
                path,
                progress.sink(),
            )),
            BeatDetectorKind::QueenMary => block(core::BeatAnalyzing::detect(
                &self.queen_mary,
                path,
                progress.sink(),
            )),
        }
        .map(Into::into)
    }
}

// MARK: - MusicHighlighting / MusicTagging / AudioSemanticEncoding

/// `MusicHighlighting` — per-window saliency and the best cut points.
#[derive(uniffi::Object)]
pub struct MusicHighlighter(CrateMusicHighlighter);

#[uniffi::export]
impl MusicHighlighter {
    #[uniffi::constructor]
    pub fn with_cache(service: Arc<EngineService>, cache: Option<Arc<DerivedAudioCache>>) -> Self {
        Self(
            Self::new(service)
                .0
                .with_cache(cache.map(|c| c.inner.clone())),
        )
    }

    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateMusicHighlighter::new(service.backend()))
    }

    pub fn detect(
        &self,
        audio_url: String,
        window_lengths: Vec<i32>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<HighlightResult> {
        let progress = call_progress(on_progress, cancellation);
        let windows: Vec<u32> = window_lengths
            .into_iter()
            .map(|length| length.max(0) as u32)
            .collect();
        block(core::MusicHighlighting::detect(
            &self.0,
            Path::new(&audio_url),
            &windows,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// `MusicTagging` — three MLP heads on top of a Dasheng embedding.
#[derive(uniffi::Object)]
pub struct MusicTagger(CrateMusicTagger);

#[uniffi::export]
impl MusicTagger {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateMusicTagger::new(service.backend()))
    }

    pub fn tag(&self, embedding: Vec<f32>, top_k: i32) -> FfiResult<MusicTagsResult> {
        block(core::MusicTagging::tag(
            &self.0,
            &embedding,
            top_k.max(0) as usize,
        ))
        .map(Into::into)
    }
}

/// `AudioSemanticEncoding` — Dasheng-Base, 768-dim, task-agnostic.
#[derive(uniffi::Object)]
pub struct AudioSemanticEncoder(CrateAudioEncoder);

#[uniffi::export]
impl AudioSemanticEncoder {
    #[uniffi::constructor]
    pub fn with_cache(service: Arc<EngineService>, cache: Option<Arc<DerivedAudioCache>>) -> Self {
        Self(
            Self::new(service)
                .0
                .with_cache(cache.map(|c| c.inner.clone())),
        )
    }

    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateAudioEncoder::new(service.backend()))
    }

    pub fn encode(
        &self,
        audio_url: String,
        return_chunks: bool,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<AudioEmbedding> {
        let progress = call_progress(on_progress, cancellation);
        block(core::AudioSemanticEncoding::encode(
            &self.0,
            Path::new(&audio_url),
            return_chunks,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

// MARK: - LoudnessNormalizing / WaveformExtracting

/// `LoudnessNormalizing` — two-pass BS.1770-4 with an audit report.
#[derive(uniffi::Object)]
pub struct LoudnessNormalizer(CrateLoudnessNormalizer);

impl Default for LoudnessNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[uniffi::export]
impl LoudnessNormalizer {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self(CrateLoudnessNormalizer)
    }

    pub fn normalize(
        &self,
        audio_url: String,
        target: LoudnessTarget,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<NormalizedAudio> {
        let progress = call_progress(on_progress, cancellation);
        block(core::LoudnessNormalizing::normalize(
            &self.0,
            Path::new(&audio_url),
            target.into(),
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// `WaveformExtracting` — the `audiowaveform`-compatible RMS envelope.
#[derive(uniffi::Object, Default)]
pub struct WaveformExtractor(
    CrateWaveformExtractor,
    Option<Arc<ferment_engine_audio::derived_cache::DerivedFileCache>>,
);

#[uniffi::export]
impl WaveformExtractor {
    #[uniffi::constructor]
    pub fn with_cache(cache: Option<Arc<DerivedAudioCache>>) -> Self {
        Self(CrateWaveformExtractor, cache.map(|c| c.inner.clone()))
    }

    #[uniffi::constructor]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extract(&self, url: String, sample_interval: f64) -> FfiResult<WaveformResult> {
        block(
            self.0
                .extract_with_cache(Path::new(&url), sample_interval, self.1.as_deref()),
        )
        .map(Into::into)
    }
}

// MARK: - Minimaps

/// `AudioMinimapComposing` — mel + RMS + a time axis, for a vision LLM.
#[derive(uniffi::Object)]
pub struct AudioMinimapComposer(CrateAudioMinimapComposer);

impl Default for AudioMinimapComposer {
    fn default() -> Self {
        Self::new()
    }
}

#[uniffi::export]
impl AudioMinimapComposer {
    #[uniffi::constructor]
    pub fn with_cache(cache: Option<Arc<DerivedAudioCache>>) -> Self {
        Self(Self::new().0.with_cache(cache.map(|c| c.inner.clone())))
    }

    #[uniffi::constructor]
    pub fn new() -> Self {
        Self(CrateAudioMinimapComposer::new(
            AudioMinimapComposerConfig::default(),
        ))
    }

    /// `bar_ruler` = downbeat times in track seconds, iOS's `barRuler:`.
    /// Empty is the plain render; non-empty adds the downbeat verticals the
    /// v3 minimap reads structure against.
    pub fn compose(
        &self,
        audio_url: String,
        bar_ruler: Vec<f64>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<MinimapResult> {
        let progress = call_progress(on_progress, cancellation);
        block(core::AudioMinimapComposing::compose(
            &self.0,
            Path::new(&audio_url),
            &bar_ruler,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// `VideoMinimapComposing` — one contact sheet of the scenes.
#[derive(uniffi::Object)]
pub struct VideoMinimapComposer(CrateVideoMinimapComposer);

impl Default for VideoMinimapComposer {
    fn default() -> Self {
        Self::new()
    }
}

#[uniffi::export]
impl VideoMinimapComposer {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self(CrateVideoMinimapComposer::new(
            VideoMinimapComposerConfig::default(),
        ))
    }

    pub fn compose(
        &self,
        video_url: String,
        scenes: VideoSceneDetectionResult,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<MinimapResult> {
        let progress = call_progress(on_progress, cancellation);
        let scenes = core::VideoSceneDetectionResult::from(&scenes);
        block(core::VideoMinimapComposing::compose(
            &self.0,
            Path::new(&video_url),
            &scenes,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

// MARK: - VideoSceneDetecting / saliency / colour

/// `VideoSceneDetecting` — AutoShot boundaries, the jump-cut merge, and the
/// DINOv2 medoid per scene.
#[derive(uniffi::Object)]
pub struct VideoSceneDetector(CrateVideoSceneDetector);

#[uniffi::export]
impl VideoSceneDetector {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(CrateVideoSceneDetector::new(service.backend()))
    }

    pub fn detect(
        &self,
        video_url: String,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoSceneDetectionResult> {
        let progress = call_progress(on_progress, cancellation);
        block(core::VideoSceneDetecting::detect(
            &self.0,
            Path::new(&video_url),
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// `VideoSaliencyAnalyzing` — where the subject sits, sparsely, per scene.
/// Apple Vision has no off-Apple twin: U^2-Netp gives the heat map and YuNet
/// the face boxes (redshift ruling 8).
#[derive(uniffi::Object)]
pub struct VideoSaliencyAnalyzer {
    analyzer: CrateVideoSaliencyAnalyzer,
    mode: core::SaliencyMode,
}

#[uniffi::export]
impl VideoSaliencyAnalyzer {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        let backend = service.backend();
        let heat = U2NetpHeatSource::new(Arc::clone(&backend));
        let mode = heat.mode();
        Self {
            analyzer: CrateVideoSaliencyAnalyzer::new(
                Arc::new(heat),
                Arc::new(YuNetFaceSource::new(backend)),
            ),
            mode,
        }
    }

    /// The mode the heat source answers in. `objectness` here, where iOS's
    /// Vision default is `attention` — the substitute is a salient-object
    /// segmenter, and the config says so rather than the wiring assuming it.
    pub fn mode(&self) -> SaliencyMode {
        match self.mode {
            core::SaliencyMode::Attention => SaliencyMode::Attention,
            core::SaliencyMode::Objectness => SaliencyMode::Objectness,
        }
    }

    pub fn analyze(
        &self,
        video_url: String,
        scenes: VideoSceneDetectionResult,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoSaliencyResult> {
        let progress = call_progress(on_progress, cancellation);
        let scenes = core::VideoSceneDetectionResult::from(&scenes);
        let result = block(self.analyzer.analyze(
            Path::new(&video_url),
            &scenes.scenes,
            progress.sink(),
        ))?;
        Ok(VideoSaliencyResult {
            frame_width: result.frame_width as i32,
            frame_height: result.frame_height as i32,
            mode: match result.mode {
                core::SaliencyMode::Attention => SaliencyMode::Attention,
                core::SaliencyMode::Objectness => SaliencyMode::Objectness,
            },
            scenes: result
                .scenes
                .into_iter()
                .map(|verdict| VideoSaliencySceneVerdict {
                    index: verdict.index as i32,
                    sample_count: verdict.sample_count as i32,
                    cx: verdict.cx,
                    cy: verdict.cy,
                    w: verdict.w,
                    h: verdict.h,
                    confidence: verdict.confidence,
                    face_share: verdict.face_share,
                    spread: verdict.spread,
                })
                .collect(),
            samples: result
                .samples
                .into_iter()
                .map(|sample| VideoSaliencySample {
                    scene_index: sample.scene_index as i32,
                    time: sample.time,
                    cx: sample.cx,
                    cy: sample.cy,
                    w: sample.w,
                    h: sample.h,
                    confidence: sample.confidence,
                    source: match sample.source {
                        ferment_engine_processors::video_saliency::SampleSource::Face => {
                            SaliencySampleSource::Face
                        }
                        ferment_engine_processors::video_saliency::SampleSource::Saliency => {
                            SaliencySampleSource::Saliency
                        }
                    },
                })
                .collect(),
        })
    }
}

/// `ColorAnalyzing` — a streaming reducer. `AnyObject` on iOS so a GPU
/// implementation can replace it without touching callers; an object here for
/// the same reason, and because it is the one capability that is fed frames
/// rather than a URL.
#[derive(uniffi::Object)]
pub struct ColorAnalyzer(std::sync::Mutex<CpuColorAnalyzer>);

impl Default for ColorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[uniffi::export]
impl ColorAnalyzer {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self(std::sync::Mutex::new(CpuColorAnalyzer::default()))
    }

    /// Accumulate one 32BGRA buffer, row-major with `bytes_per_row` stride —
    /// what `ImageReader`'s `Image.Plane` hands out.
    ///
    /// The geometry has to describe the bytes. `CpuColorAnalyzer` bounds-checks
    /// every pixel it reads, so a buffer that is too short does not fault — it
    /// yields statistics over whatever prefix fitted, which is a wrong answer
    /// with no signal attached. It is refused here instead, quoting the numbers
    /// the caller sent, the way an unknown execution provider is.
    ///
    /// The requirement is `bytes_per_row * (height - 1) + width * 4`, not
    /// `bytes_per_row * height`: an `ImageReader` plane's final row carries no
    /// trailing padding, so the stricter form would refuse every real frame.
    pub fn accumulate(
        &self,
        bgra: Vec<u8>,
        width: i32,
        height: i32,
        bytes_per_row: i32,
    ) -> FfiResult<()> {
        let Some(required) = frame_bytes(width, height, bytes_per_row)? else {
            // An empty frame accumulates nothing, exactly as iOS's loops do.
            return Ok(());
        };
        if bgra.len() < required {
            return Err(EngineFfiError::new(format!(
                "a {width}x{height} BGRA frame at {bytes_per_row} bytes per row needs {required} bytes; {} were sent",
                bgra.len()
            )));
        }
        let frame = core::VideoFrame {
            data: bgra.into(),
            planes: vec![core::PlaneLayout {
                offset: 0,
                stride: bytes_per_row as usize,
                row_bytes: width as usize * 4,
                rows: height as usize,
            }],
            width: width as u32,
            height: height as u32,
            format: core::PixelFormat::Bgra8,
            presentation_time: core::MediaTime::default(),
        };
        let mut analyzer = self.0.lock().unwrap_or_else(|error| error.into_inner());
        core::ColorAnalyzing::accumulate(&mut *analyzer, &frame).map_err(EngineFfiError::from)
    }

    pub fn result(&self) -> ColorStats {
        let analyzer = self.0.lock().unwrap_or_else(|error| error.into_inner());
        core::ColorAnalyzing::result(&*analyzer).into()
    }
}

/// How many bytes a frame of this geometry occupies, or `None` when it is
/// EMPTY, or an error when the geometry itself is impossible.
///
/// Three outcomes, because there are three cases and iOS only has two of them:
///
/// * **Empty** — zero or negative dimensions. `CVPixelBuffer` cannot be handed
///   one, and iOS's reducer simply runs no loop iterations, so the port must
///   accumulate nothing and say nothing. `None`.
/// * **Impossible** — a stride under the row, or arithmetic that overflows.
///   iOS cannot express this: a `CVPixelBuffer` carries its own geometry, so
///   the pair cannot disagree. Here Kotlin supplies the two separately, so a
///   disagreement is a caller bug, and measuring whatever prefix fitted would
///   answer statistics for a frame nobody sent. `Err`.
/// * **A frame** — the byte count, which is `stride * (height - 1) + width * 4`
///   and deliberately NOT `stride * height`: an `ImageReader` plane carries no
///   trailing padding after its last row, so the strict form would reject
///   every real Android frame.
///
/// Every one of these is a number Kotlin chose and this crate aborts on panic,
/// so the arithmetic is checked rather than trusted: `width * 4` alone
/// overflows `i32` at a width Kotlin can express.
fn frame_bytes(width: i32, height: i32, bytes_per_row: i32) -> FfiResult<Option<usize>> {
    let invalid = || {
        EngineFfiError::new(format!(
            "{width}x{height} at {bytes_per_row} bytes per row is not a frame"
        ))
    };
    if width <= 0 || height <= 0 {
        return Ok(None);
    }
    let (Ok(width), Ok(height), Ok(stride)) = (
        usize::try_from(width),
        usize::try_from(height),
        usize::try_from(bytes_per_row),
    ) else {
        return Err(invalid());
    };
    let row_bytes = width.checked_mul(4).ok_or_else(invalid)?;
    if stride < row_bytes {
        return Err(invalid());
    }
    stride
        .checked_mul(height - 1)
        .and_then(|body| body.checked_add(row_bytes))
        .ok_or_else(invalid)
        .map(Some)
}

/// The automatic motion-blur decision, from the ONE implementation of it.
///
/// Pure math — no engine, no models, no decode — so it is a free function
/// rather than a capability. It exists because the alternative is a SECOND
/// implementation of the same research-backed thresholds on the Kotlin side,
/// which is exactly what the engine charter chose the Rust engine to avoid:
/// "One numerical truth. A Kotlin engine would be a third."
///
/// `blur_available` is the caller's: on iOS it is whether VideoToolbox offers
/// the stage, here whether the renderer has one at all.
#[uniffi::export]
pub fn motion_blur_decide(
    mode: MotionBlurMode,
    analysis: Option<SlowMotionAnalysis>,
    factor: i32,
    target_shutter_angle: f64,
    blur_available: bool,
) -> MotionBlurDecision {
    let analysis = analysis.map(processors::SlowMotionAnalysis::from);
    processors::MotionBlurPolicy::decide(
        mode.into(),
        analysis.as_ref(),
        factor.clamp(0, i32::from(u8::MAX)) as u8,
        target_shutter_angle,
        blur_available,
    )
    .into()
}

/// The strength the policy asks for to fill toward a target smear.
///
/// The engine's formula, for the same reason [`motion_blur_decide`] is: the
/// rounding, the divide by the factor and the 1..=100 clamp are the numerical
/// truth, and a Kotlin copy of them would be a second one.
#[uniffi::export]
pub fn motion_blur_strength(fill_to: f64, output_smear_angle: f64, factor: i32) -> i32 {
    processors::MotionBlurPolicy::strength(
        fill_to,
        output_smear_angle,
        factor.clamp(0, i32::from(u8::MAX)) as u8,
    )
}

/// The reference frame every colour-family smoke runs on, as packed BGRA8.
///
/// The SAME `selfie_640.png` E0's `infer_smoke` compiles in, decoded here so
/// the device and the Mac physically cannot differ in their input: Kotlin never
/// decodes it, it feeds these bytes straight to [`ColorAnalyzer::accumulate`].
/// Row stride equals `4 * width`, which is what the record promises.
#[derive(Debug, uniffi::Record)]
pub struct SmokeFrame {
    pub bgra: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub bytes_per_row: i32,
}

/// The compiled-in reference frame, decoded to BGRA8.
#[uniffi::export]
pub fn color_smoke_frame() -> FfiResult<SmokeFrame> {
    let frame = image::load_from_memory(crate::SMOKE_FRAME_PNG)
        .map_err(|error| EngineFfiError::new(format!("reference frame: {error}")))?
        .to_rgb8();
    let (width, height) = (frame.width(), frame.height());
    let mut bgra = Vec::with_capacity((width * height * 4) as usize);
    for pixel in frame.pixels() {
        bgra.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
    }
    Ok(SmokeFrame {
        bgra,
        width: width as i32,
        height: height as i32,
        bytes_per_row: (width * 4) as i32,
    })
}

// MARK: - VideoEncoding and the render capabilities

/// Configures inline thumbnail generation during an encode.
///
/// iOS's `ThumbStripOptions`: the encoder taps every Nth decoded frame
/// (PTS-based interval), scales it into `max_width` × `max_height` preserving
/// aspect, and hands it to [`ThumbnailListener`]. No file I/O happens in the
/// engine — the caller decides whether to encode JPEG, write to disk,
/// accumulate atlases or feed a live preview surface.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct ThumbStripOptions {
    /// Seconds between emitted thumbnails.
    pub interval: f64,
    pub max_width: u32,
    pub max_height: u32,
}

/// One thumbnail emitted by the encoder, as packed BGRA8.
///
/// iOS hands back a `CVPixelBuffer` whose default layout is
/// `kCVPixelFormatType_32BGRA`, and the crate's encode pipeline already
/// produces exactly that — so the pixels cross unconverted, with the producer's
/// own `bytes_per_row`. `variant_id` is empty for the single-output `encode`
/// and names the emitting variant for `encode_variants` — iOS's two callback
/// shapes (`(ThumbnailFrame) -> Void` and `(String, ThumbnailFrame) -> Void`)
/// as one.
#[derive(Debug, uniffi::Record)]
pub struct ThumbnailFrame {
    pub variant_id: String,
    pub bgra: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub index: u64,
    pub presentation_time: f64,
}

/// The Kotlin side of iOS's `onThumbnail`.
///
/// A callback interface for the same reason [`ProgressListener`] is one: the
/// encoder calls it many times per operation, from the thread the encode runs
/// on. A strip requested without one is refused by the crate before it opens
/// the source, which is why the option and the listener travel together.
#[uniffi::export(with_foreign)]
pub trait ThumbnailListener: Send + Sync {
    fn on_thumbnail(&self, thumbnail: ThumbnailFrame);
}

/// Adapts a foreign [`ThumbnailListener`] onto the crate's two sink traits.
struct ThumbnailBridge(Arc<dyn ThumbnailListener>);

impl ThumbnailBridge {
    fn emit(&self, variant_id: &str, thumbnail: core::ThumbnailFrame) {
        let frame = thumbnail.frame;
        // The encode pipeline emits one BGRA8 plane. A frame that is not that
        // is dropped rather than reinterpreted: a strip is a derivative, and
        // guessing a layout would hand Kotlin garbage that renders.
        let (core::PixelFormat::Bgra8, Some(plane)) = (frame.format, frame.planes.first()) else {
            return;
        };
        self.0.on_thumbnail(ThumbnailFrame {
            variant_id: variant_id.to_owned(),
            bgra: frame.data.as_ref().to_vec(),
            width: frame.width,
            height: frame.height,
            bytes_per_row: plane.row_bytes as u32,
            index: thumbnail.index as u64,
            presentation_time: thumbnail.presentation_time,
        });
    }
}

impl core::ThumbnailSink for ThumbnailBridge {
    fn emit(&self, thumbnail: core::ThumbnailFrame) {
        Self::emit(self, "", thumbnail);
    }
}

impl core::VariantThumbnailSink for ThumbnailBridge {
    fn emit(&self, variant_id: &str, thumbnail: core::ThumbnailFrame) {
        Self::emit(self, variant_id, thumbnail);
    }
}

fn core_thumb_strip(options: Option<ThumbStripOptions>) -> Option<core::ThumbStripOptions> {
    options.map(|options| core::ThumbStripOptions {
        interval: options.interval,
        max_width: options.max_width,
        max_height: options.max_height,
    })
}

/// `VideoEncoding` — one preset over one source.
#[derive(uniffi::Object, Default)]
pub struct VideoEncoder(CrateVideoEncoder);

#[uniffi::export]
impl VideoEncoder {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self(
            CrateVideoEncoder::new()
                .with_backends(ferment_engine_processors::platform_encode_backends()),
        )
    }

    /// `thumb_strip` and `on_thumbnail` travel together: the crate refuses a
    /// strip with no sink before it opens the source, so passing one without
    /// the other is an error rather than a silently thumbnail-less encode.
    ///
    /// Eight parameters because iOS's `encode(sourceURL:outputURL:preset:
    /// thumbStrip:onProgress:onThumbnail:)` has six and this boundary adds the
    /// cancellation token every long call takes; grouping them into a record
    /// would stop the two signatures reading as one.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        source_url: String,
        output_url: String,
        preset: VideoEncodingPreset,
        thumb_strip: Option<ThumbStripOptions>,
        on_thumbnail: Option<Arc<dyn ThumbnailListener>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoEncodingResult> {
        if thumb_strip.is_some() != on_thumbnail.is_some() {
            return Err(EngineFfiError::new(
                "a thumb strip needs its onThumbnail listener, and a listener needs a strip",
            ));
        }
        let progress = call_progress(on_progress, cancellation);
        let preset = core_preset(&preset)?;
        let bridge = on_thumbnail.map(ThumbnailBridge);
        block(core::VideoEncoding::encode(
            &self.0,
            Path::new(&source_url),
            Path::new(&output_url),
            &preset,
            core_thumb_strip(thumb_strip).as_ref().copied(),
            progress.sink(),
            bridge
                .as_ref()
                .map(|bridge| bridge as &dyn core::ThumbnailSink),
        ))
        .map(Into::into)
    }

    /// Every derivative of one source from a SINGLE decode — the shape the
    /// import path uses, because the decode, not the encode, is the cost.
    pub fn encode_variants(
        &self,
        source_url: String,
        variants: Vec<VideoEncodingVariant>,
        on_thumbnail: Option<Arc<dyn ThumbnailListener>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<MultiVideoEncodingResult> {
        let progress = call_progress(on_progress, cancellation);
        let variants = variants
            .iter()
            .map(|variant| {
                Ok(core::VideoEncodingVariant {
                    id: variant.id.clone(),
                    output_path: std::path::PathBuf::from(&variant.output_path),
                    preset: core_preset(&variant.preset)?,
                    thumb_strip: core_thumb_strip(variant.thumb_strip),
                })
            })
            .collect::<FfiResult<Vec<_>>>()?;
        if variants.iter().any(|variant| variant.thumb_strip.is_some()) != on_thumbnail.is_some() {
            return Err(EngineFfiError::new(
                "a thumb strip needs its onThumbnail listener, and a listener needs a strip",
            ));
        }
        let bridge = on_thumbnail.map(ThumbnailBridge);
        block(core::MultiVideoEncoding::encode_variants(
            &self.0,
            Path::new(&source_url),
            &variants,
            progress.sink(),
            bridge
                .as_ref()
                .map(|bridge| bridge as &dyn core::VariantThumbnailSink),
        ))
        .map(Into::into)
    }
}

/// `SlowMotionRendering` — `factor − 1` synthesized frames between every pair,
/// the timeline stretched by `factor`. `sceneStarts` are held, never
/// interpolated.
#[derive(uniffi::Object, Default)]
pub struct SlowMotionRenderer(CrateSlowMotionRenderer);

#[uniffi::export]
impl SlowMotionRenderer {
    #[uniffi::constructor]
    pub fn new(engine: Arc<EngineService>) -> Self {
        Self(CrateSlowMotionRenderer::new().with_backend(engine.backend()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        source_url: String,
        output_url: String,
        factor: i32,
        window: Option<MediaTimeRange>,
        scenes: Vec<MediaTime>,
        motion_blur: MotionBlurMode,
        target_shutter_angle: f64,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoRenderResult> {
        let progress = call_progress(on_progress, cancellation);
        let scenes: Vec<core::MediaTime> = scenes.into_iter().map(Into::into).collect();
        block(core::SlowMotionRendering::render(
            &self.0,
            Path::new(&source_url),
            Path::new(&output_url),
            slow_motion_factor(factor)?,
            window.map(Into::into),
            &scenes,
            core::MotionBlurMode::from(motion_blur),
            target_shutter_angle,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// The crate's `u8` factor, refusing rather than clamping.
///
/// The range inside the byte (2..=8) is the crate's rule and the crate's
/// message. What must not happen here is a clamp: it hands the crate a number
/// the caller never sent, and the crate then quotes THAT back — "factor 255 is
/// out of range" to a caller who wrote 1000.
fn slow_motion_factor(factor: i32) -> FfiResult<u8> {
    u8::try_from(factor)
        .map_err(|_| EngineFfiError::new(format!("slow-motion factor {factor} is out of range")))
}

/// `VideoUpscaling` — Real-ESRGAN Compact, recurrent on the previous output,
/// so each scene's first frame announces a sequence restart.
#[derive(uniffi::Object, Default)]
pub struct VideoUpscaler(CrateVideoUpscaler);

#[uniffi::export]
impl VideoUpscaler {
    #[uniffi::constructor]
    pub fn new(engine: Arc<EngineService>) -> Self {
        Self(CrateVideoUpscaler::new().with_backend(engine.backend()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        source_url: String,
        output_url: String,
        mode: VideoUpscaleMode,
        window: Option<MediaTimeRange>,
        scenes: Vec<MediaTime>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoRenderResult> {
        let progress = call_progress(on_progress, cancellation);
        let scenes: Vec<core::MediaTime> = scenes.into_iter().map(Into::into).collect();
        block(core::VideoUpscaling::render(
            &self.0,
            Path::new(&source_url),
            Path::new(&output_url),
            mode.into(),
            window.map(Into::into),
            &scenes,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

/// `VideoNoiseFiltering` — FastDVDnet's five-frame temporal window, reset at
/// every scene boundary.
/// The orchestrator is stateless — `FastDvdnetTemporalFilter` is what holds
/// the backend, and it is constructed per render inside the crate.
#[derive(uniffi::Object, Default)]
pub struct VideoNoiseFilter(CrateVideoNoiseFilter);

#[uniffi::export]
impl VideoNoiseFilter {
    #[uniffi::constructor]
    pub fn new(engine: Arc<EngineService>) -> Self {
        Self(CrateVideoNoiseFilter::new().with_backend(engine.backend()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        source_url: String,
        output_url: String,
        strength: f32,
        window: Option<MediaTimeRange>,
        scenes: Vec<MediaTime>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<VideoRenderResult> {
        let progress = call_progress(on_progress, cancellation);
        let scenes: Vec<core::MediaTime> = scenes.into_iter().map(Into::into).collect();
        block(core::VideoNoiseFiltering::render(
            &self.0,
            Path::new(&source_url),
            Path::new(&output_url),
            strength,
            window.map(Into::into),
            &scenes,
            progress.sink(),
        ))
        .map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use ferment_engine_processors::AudioSampleSource as _;

    use super::*;

    /// Every argument here is a number Kotlin chose, and this crate is
    /// `panic = "abort"` in release: an arithmetic overflow in a guard is not
    /// an exception the app can catch, it is the process dying. `width * 4` is
    /// an `i32` multiply on a caller-supplied `i32`.
    ///
    /// Kill: compute the row width in `i32` again — the guard overflows before
    /// it can reject anything.
    #[test]
    fn a_hostile_geometry_cannot_take_the_process_down() {
        let analyzer = ColorAnalyzer::new();
        assert!(
            analyzer
                .accumulate(vec![0; 16], i32::MAX, i32::MAX, i32::MAX)
                .is_err()
        );
        assert_eq!(analyzer.result().pixel_count, 0);
    }

    /// A buffer that does not hold the frame it claims to is refused, not
    /// half-read. `CpuColorAnalyzer` bounds-checks every pixel, so a short
    /// buffer produces statistics for whatever prefix happened to fit — a
    /// wrong answer with no signal at all, which is the one outcome a
    /// measurement API must never give.
    ///
    /// Kill: drop the length check — 64 bytes are accepted as a 64x64 frame.
    #[test]
    fn a_buffer_shorter_than_its_geometry_is_refused() {
        let analyzer = ColorAnalyzer::new();
        let error = analyzer.accumulate(vec![255; 64], 64, 64, 256).unwrap_err();
        assert!(error.to_string().contains("16384"), "{error}");
        assert_eq!(analyzer.result().pixel_count, 0);
    }

    /// ...and the length Android actually hands out is accepted. An
    /// `ImageReader` plane's buffer is `rowStride * (height - 1) + width *
    /// pixelStride` — the final row carries no trailing padding — so a check
    /// written as `stride * height` would reject every real frame.
    ///
    /// Kill: require `stride * height` bytes.
    #[test]
    fn an_image_reader_buffer_with_no_trailing_padding_is_accepted() {
        let (width, height, stride) = (64_usize, 64_usize, 320_usize);
        let analyzer = ColorAnalyzer::new();
        analyzer
            .accumulate(
                vec![128; stride * (height - 1) + width * 4],
                width as i32,
                height as i32,
                stride as i32,
            )
            .expect("the shape ImageReader hands out");
        assert!(analyzer.result().pixel_count > 0);
    }

    /// A factor outside the crate's 2..=8 range is refused by the crate, which
    /// quotes the number in its message. The façade must not have altered that
    /// number on the way in: `clamp(0, 255)` turns "you asked for 1000" into
    /// "you asked for 255", which sends the reader looking for a 255 nobody
    /// wrote. Same rule as `an_unknown_provider_is_rejected_verbatim`.
    ///
    /// Kill: clamp the factor into `u8` instead of refusing what does not fit.
    #[test]
    fn a_slow_motion_factor_is_reported_as_the_caller_sent_it() {
        let error = SlowMotionRenderer::new(Arc::new(
            EngineService::new("/models".to_owned()).expect("service"),
        ))
        .render(
            "/nonexistent/source.mov".to_owned(),
            "/nonexistent/output.mp4".to_owned(),
            1_000,
            None,
            Vec::new(),
            MotionBlurMode::Off,
            180.0,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("1000"), "{error}");
    }

    /// The languages a transcriber claims are the MODEL's own declaration,
    /// matched rather than restated — the rule `provider_names_round_trip`
    /// holds for the execution providers. A second spelling here is a second
    /// source of truth for what the router is allowed to route to.
    ///
    /// Kill: write the code list out again in the façade.
    #[test]
    fn a_transcriber_reports_the_languages_its_model_declares() {
        let service = Arc::new(EngineService::new("/models".to_owned()).expect("service"));
        for (kind, declared) in [
            (
                TranscriberKind::Parakeet,
                &CrateParakeet::SUPPORTED_LANGUAGE_CODES[..],
            ),
            (
                TranscriberKind::GigaAm,
                &GigaAm::SUPPORTED_LANGUAGE_CODES[..],
            ),
        ] {
            let transcriber = SpeechTranscriber::new(Arc::clone(&service), kind);
            assert_eq!(transcriber.supported_language_codes(), declared);
            assert_eq!(transcriber.kind(), kind);
        }
        // The one language GigaAM answers with is the one it says it takes.
        assert_eq!(
            GigaAm::SUPPORTED_LANGUAGE_CODES,
            [ferment_engine_processors::gigaam::LANGUAGE]
        );
    }

    /// AmberNet's window is measured in ITS OWN samples: `build_window` converts
    /// every bound with `ambernet::SAMPLE_RATE` and the graph is compiled for
    /// exactly 30 s of them. `AudioSampleSource` carries no rate, and the decode
    /// hands back the file's NATIVE one — so handing the decode straight over
    /// makes the model read a 48 kHz file as if it were 16 kHz: three times the
    /// audio, at three times the speed, with every segment offset landing in the
    /// wrong place. Silently: there is no shape for it to fail on.
    ///
    /// Kill: drop the resample and hand `frames.samples` over as they arrive.
    #[test]
    fn the_language_window_is_built_at_the_models_own_rate() {
        let source = language_id_source(AudioFrames::new(vec![0.0; 48_000 * 3], 48_000))
            .expect("a decodable buffer");
        let seconds = source.sample_count() as f64 / f64::from(ambernet::SAMPLE_RATE);
        assert!(
            (seconds - 3.0).abs() < 0.05,
            "AmberNet will read {seconds:.3} s out of three seconds of audio"
        );
    }

    /// The other side of the colour seam. Prints the answer the device's
    /// `EngineFacadeSmokeTest` is graded against, on the Mac, from the same
    /// compiled-in frame through the same reducer:
    ///
    /// ```text
    /// cargo test -p ferment-engine-uniffi --lib -- --ignored --nocapture \
    ///   prints_the_mac_color_answer_for_the_device_suite
    /// ```
    ///
    /// Kill: change the BGRA channel order in [`color_smoke_frame`], or the
    /// skin/neutral thresholds in `CpuColorAnalyzer` — the numbers move, which
    /// is exactly what the device assertion then catches.
    #[test]
    #[ignore = "prints the golden the device suite is graded against"]
    fn prints_the_mac_color_answer_for_the_device_suite() {
        let frame = color_smoke_frame().expect("reference frame");
        let analyzer = ColorAnalyzer::new();
        analyzer
            .accumulate(frame.bgra, frame.width, frame.height, frame.bytes_per_row)
            .expect("accumulate");
        let stats = analyzer.result();
        eprintln!("pixel_count   = {}", stats.pixel_count);
        eprintln!("linear_mean   = {:?}", stats.linear_mean);
        eprintln!("neutral_mean  = {:?}", stats.neutral_mean);
        eprintln!("mean_log_luma = {}", stats.mean_log_luma);
        eprintln!("black_point   = {}", stats.black_point);
        eprintln!("white_point   = {}", stats.white_point);
        eprintln!("skin_centroid = {:?}", stats.skin_centroid);
        eprintln!("skin_mass     = {}", stats.skin_mass);
        eprintln!("chroma_mean   = {:?}", stats.chroma_mean);
    }

    /// The reference frame is packed BGRA8 with no row padding — the shape
    /// [`ColorAnalyzer::accumulate`] documents and the only one Kotlin sends.
    ///
    /// Kill: emit RGBA instead of BGRA and the blue/red means swap on both
    /// sides at once, which no round-trip could see; this pins the byte order
    /// against the PNG's own first pixel.
    #[test]
    fn the_reference_frame_is_packed_bgra() {
        let frame = color_smoke_frame().expect("reference frame");
        assert_eq!(frame.bytes_per_row, frame.width * 4);
        assert_eq!(frame.bgra.len() as i32, frame.bytes_per_row * frame.height,);
        let rgb = image::load_from_memory(crate::SMOKE_FRAME_PNG)
            .expect("png")
            .to_rgb8();
        let first = rgb.get_pixel(0, 0);
        assert_eq!(&frame.bgra[..4], &[first[2], first[1], first[0], 255]);
    }
}

// MARK: - ObjectSegmenting and LiveSpeechTranscribing (ruling 39's two contracts)

/// One prompt token pair, as the interactive surface sends it.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct NormalizedVideoPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum VideoObjectPointLabel {
    Background,
    Foreground,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct VideoObjectPointPrompt {
    pub point: NormalizedVideoPoint,
    pub label: VideoObjectPointLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct NormalizedVideoBox {
    pub top_left: NormalizedVideoPoint,
    pub bottom_right: NormalizedVideoPoint,
}

/// EdgeTAM encodes a box as two prompt tokens before any correction points.
/// The combined case preserves that exact ordering — a `box` with `points` is
/// NOT the same request as the points alone.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum VideoObjectPrompt {
    Points {
        points: Vec<VideoObjectPointPrompt>,
    },
    Box {
        bounds: NormalizedVideoBox,
    },
    BoxAndPoints {
        bounds: NormalizedVideoBox,
        points: Vec<VideoObjectPointPrompt>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum VideoObjectPropagationDirection {
    Forward,
    Reverse,
}

/// An immutable, tightly packed binary mask: row-major, exactly `width` bytes
/// per row, background `0` and foreground `255`.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct VideoObjectMask {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

/// One object's mask at one source-video frame.
#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct VideoObjectMaskFrame {
    pub frame_index: u32,
    pub presentation_time: f64,
    pub object_id: String,
    pub mask: VideoObjectMask,
    pub predicted_iou: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct VideoObjectSegmentationSourceInfo {
    pub frame_count: u32,
    pub frame_rate: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Record)]
pub struct ImageObjectSegmentationSourceInfo {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct ImageObjectSegmentationResult {
    pub object_id: String,
    pub mask: VideoObjectMask,
    pub predicted_iou: Option<f32>,
}

/// Where propagated masks land, one at a time, as they are produced.
///
/// iOS's `AsyncThrowingStream` with `bufferingOldest(0)`: the producer computes
/// at most one frame ahead and waits. Answering `false` stops the propagation —
/// the twin of the iOS continuation's `.terminated`.
#[uniffi::export(with_foreign)]
pub trait VideoObjectMaskListener: Send + Sync {
    fn on_mask(&self, frame: VideoObjectMaskFrame) -> bool;
}

struct MaskBridge(Arc<dyn VideoObjectMaskListener>);

impl core::VideoObjectMaskSink for MaskBridge {
    fn emit(&self, frame: core::VideoObjectMaskFrame) -> bool {
        self.0.on_mask(VideoObjectMaskFrame::from(frame))
    }
}

impl From<core::VideoObjectMask> for VideoObjectMask {
    fn from(mask: core::VideoObjectMask) -> Self {
        Self {
            width: mask.width() as u32,
            height: mask.height() as u32,
            bytes: mask.into_bytes(),
        }
    }
}

impl From<core::VideoObjectMaskFrame> for VideoObjectMaskFrame {
    fn from(frame: core::VideoObjectMaskFrame) -> Self {
        Self {
            frame_index: frame.frame_index as u32,
            presentation_time: frame.presentation_time,
            object_id: frame.object_id.0,
            mask: frame.mask.into(),
            predicted_iou: frame.predicted_iou,
        }
    }
}

fn core_point(point: NormalizedVideoPoint) -> core::NormalizedVideoPoint {
    core::NormalizedVideoPoint::new(point.x, point.y)
}

fn core_points(points: &[VideoObjectPointPrompt]) -> Vec<core::VideoObjectPointPrompt> {
    points
        .iter()
        .map(|prompt| {
            core::VideoObjectPointPrompt::new(
                core_point(prompt.point),
                match prompt.label {
                    VideoObjectPointLabel::Background => core::VideoObjectPointLabel::Background,
                    VideoObjectPointLabel::Foreground => core::VideoObjectPointLabel::Foreground,
                },
            )
        })
        .collect()
}

fn core_box(bounds: NormalizedVideoBox) -> core::NormalizedVideoBox {
    core::NormalizedVideoBox::new(core_point(bounds.top_left), core_point(bounds.bottom_right))
}

fn core_prompt(prompt: &VideoObjectPrompt) -> core::VideoObjectPrompt {
    match prompt {
        VideoObjectPrompt::Points { points } => {
            core::VideoObjectPrompt::Points(core_points(points))
        }
        VideoObjectPrompt::Box { bounds } => core::VideoObjectPrompt::Box(core_box(*bounds)),
        VideoObjectPrompt::BoxAndPoints { bounds, points } => {
            core::VideoObjectPrompt::BoxAndPoints {
                r#box: core_box(*bounds),
                points: core_points(points),
            }
        }
    }
}

/// `ImageObjectSegmenting` / `VideoObjectSegmenting` — EdgeTAM V1.
///
/// One tracked object per video session (the V1 reference profile), at most 32
/// prompted frames, and 32 active objects in a still-image session.
#[derive(uniffi::Object)]
pub struct ObjectSegmenter(processors::EdgeTam);

#[uniffi::export]
impl ObjectSegmenter {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(processors::EdgeTam::new(
            service.backend(),
            processors::video_backend(),
        ))
    }

    /// Encode one immutable image and retain its features for every prompt and
    /// correction made through the returned session.
    ///
    /// `rgb` is packed RGB24 with `3 * width * height` bytes, already in the
    /// intended display orientation — `CGImage`'s contract, which carries no
    /// EXIF orientation either.
    pub fn make_image_session(
        &self,
        rgb: Vec<u8>,
        width: u32,
        height: u32,
    ) -> FfiResult<Arc<ImageObjectSegmentationSession>> {
        let frame = core::RgbFrameRef::new(&rgb, width, height).ok_or_else(|| {
            EngineFfiError::new(format!(
                "an RGB24 image of {width}x{height} needs {} bytes, got {}",
                3 * width as usize * height as usize,
                rgb.len()
            ))
        })?;
        let session = block(core::ImageObjectSegmenting::make_session(&self.0, frame))?;
        Ok(Arc::new(ImageObjectSegmentationSession(
            std::sync::Mutex::new(session),
        )))
    }

    pub fn make_video_session(
        &self,
        video_url: String,
        temporary_directory: String,
        cancellation: Token,
    ) -> FfiResult<Arc<VideoObjectSegmentationSession>> {
        let segmenter = self
            .0
            .clone()
            .with_temporary_directory(std::path::PathBuf::from(temporary_directory));
        Ok(Arc::new(VideoObjectSegmentationSession(
            SessionThread::spawn(move || {
                let progress = call_progress(None, cancellation);
                futures_executor::block_on(
                    core::VideoObjectSegmenting::make_session_with_cancellation(
                        &segmenter,
                        Path::new(&video_url),
                        progress.sink(),
                    ),
                )
            })?,
        )))
    }
}

/// Mutable prompt state for one fixed image.
///
/// The `Mutex` is iOS's actor: one operation at a time, so corrections and
/// prior-mask reuse can never race.
#[derive(uniffi::Object)]
pub struct ImageObjectSegmentationSession(
    std::sync::Mutex<Box<dyn core::ImageObjectSegmentationSession>>,
);

#[uniffi::export]
impl ImageObjectSegmentationSession {
    pub fn source_info(&self) -> ImageObjectSegmentationSourceInfo {
        let session = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let info = session.source_info();
        ImageObjectSegmentationSourceInfo {
            width: info.width as u32,
            height: info.height as u32,
        }
    }

    pub fn maximum_active_object_count(&self) -> u32 {
        let session = self.0.lock().unwrap_or_else(|error| error.into_inner());
        session.maximum_active_object_count() as u32
    }

    /// Set `replacing_existing_prompts` to false to append point clicks to the
    /// object's existing prompt. A box can only start or replace one.
    pub fn apply(
        &self,
        prompt: VideoObjectPrompt,
        object_id: String,
        replacing_existing_prompts: bool,
        cancellation: Token,
    ) -> FfiResult<ImageObjectSegmentationResult> {
        let progress = call_progress(None, cancellation);
        let mut session = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let result = block(session.apply_with_cancellation(
            &core_prompt(&prompt),
            &core::VideoObjectId::new(object_id),
            replacing_existing_prompts,
            progress.sink(),
        ))?;
        Ok(ImageObjectSegmentationResult {
            object_id: result.object_id.0,
            mask: result.mask.into(),
            predicted_iou: result.predicted_iou,
        })
    }

    pub fn remove(&self, object_id: String) {
        let mut session = self.0.lock().unwrap_or_else(|error| error.into_inner());
        futures_executor::block_on(session.remove(&core::VideoObjectId::new(object_id)));
    }

    pub fn reset(&self) {
        let mut session = self.0.lock().unwrap_or_else(|error| error.into_inner());
        futures_executor::block_on(session.reset());
    }
}

/// One thread that owns a video segmentation session for its whole life, and
/// the channel calls travel down.
///
/// The session is NOT `Send`: it owns a platform decoder holding raw codec
/// pointers (libav's `AVCodecContext`, VideoToolbox's session), so it never
/// leaves the thread that opened it. The UniFFI object holding it must be
/// `Send + Sync`, so the object is this handle and the session stays put —
/// which is also what iOS does, where the actor's executor is a thread the
/// session never leaves. One thread per OPEN session; the editor has one.
struct SessionThread {
    jobs: std::sync::mpsc::Sender<Job>,
    thread: Option<std::thread::JoinHandle<()>>,
}

type Session = Box<dyn core::VideoObjectSegmentationSession>;
type Job = Box<dyn FnOnce(&mut Session) + Send>;

impl SessionThread {
    /// `open` runs ON the worker thread, so the session it builds is created
    /// and dropped there and never crosses.
    fn spawn(open: impl FnOnce() -> core::Result<Session> + Send + 'static) -> FfiResult<Self> {
        let (jobs, receiver) = std::sync::mpsc::channel::<Job>();
        let (opened, opened_rx) = std::sync::mpsc::sync_channel::<core::Result<()>>(1);
        let thread = std::thread::Builder::new()
            .name("edgetam-session".into())
            .spawn(move || {
                let mut session = match open() {
                    Ok(session) => {
                        let _ = opened.send(Ok(()));
                        session
                    }
                    Err(error) => {
                        let _ = opened.send(Err(error));
                        return;
                    }
                };
                while let Ok(job) = receiver.recv() {
                    job(&mut session);
                }
            })
            .map_err(|error| EngineFfiError::new(format!("session thread: {error}")))?;
        match opened_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                jobs,
                thread: Some(thread),
            }),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(EngineFfiError::new("the session thread died while opening")),
        }
    }

    fn call<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut Session) -> T + Send + 'static,
    ) -> FfiResult<T> {
        let (answer, answer_rx) = std::sync::mpsc::sync_channel::<T>(1);
        self.jobs
            .send(Box::new(move |session| {
                let _ = answer.send(job(session));
            }))
            .map_err(|_| EngineFfiError::new("the session thread is gone"))?;
        answer_rx
            .recv()
            .map_err(|_| EngineFfiError::new("the session thread died mid-call"))
    }
}

impl Drop for SessionThread {
    fn drop(&mut self) {
        // Dropping the sender ends the loop; joining makes the decoder's
        // release happen before this object's owner returns.
        let (dead, _) = std::sync::mpsc::channel();
        let jobs = std::mem::replace(&mut self.jobs, dead);
        drop(jobs);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Mutable inference state for one video: prompts, temporal memory and
/// propagation order, serialized on the session's own thread.
#[derive(uniffi::Object)]
pub struct VideoObjectSegmentationSession(SessionThread);

#[uniffi::export]
impl VideoObjectSegmentationSession {
    pub fn source_info(&self) -> FfiResult<VideoObjectSegmentationSourceInfo> {
        self.0.call(|session| {
            let info = session.source_info();
            VideoObjectSegmentationSourceInfo {
                frame_count: info.frame_count as u32,
                frame_rate: info.frame_rate,
                width: info.width as u32,
                height: info.height as u32,
            }
        })
    }

    pub fn apply(
        &self,
        prompt: VideoObjectPrompt,
        object_id: String,
        frame_index: u32,
        replacing_existing_prompts: bool,
        cancellation: Token,
    ) -> FfiResult<VideoObjectMaskFrame> {
        let prompt = core_prompt(&prompt);
        let object_id = core::VideoObjectId::new(object_id);
        self.0
            .call(move |session| {
                let progress = call_progress(None, cancellation);
                futures_executor::block_on(session.apply_with_cancellation(
                    &prompt,
                    &object_id,
                    frame_index as usize,
                    replacing_existing_prompts,
                    progress.sink(),
                ))
            })?
            .map(Into::into)
            .map_err(EngineFfiError::from)
    }

    /// Propagate every registered object from `frame_index` to the end in the
    /// selected direction. The prompt frame itself is included, and the
    /// listener answering `false` stops the walk.
    pub fn propagate(
        &self,
        frame_index: u32,
        direction: VideoObjectPropagationDirection,
        on_mask: Arc<dyn VideoObjectMaskListener>,
        cancellation: Token,
    ) -> FfiResult<()> {
        let direction = match direction {
            VideoObjectPropagationDirection::Forward => {
                core::VideoObjectPropagationDirection::Forward
            }
            VideoObjectPropagationDirection::Reverse => {
                core::VideoObjectPropagationDirection::Reverse
            }
        };
        self.0
            .call(move |session| {
                let bridge = MaskBridge(on_mask);
                let progress = call_progress(None, cancellation);
                futures_executor::block_on(session.propagate_with_cancellation(
                    frame_index as usize,
                    direction,
                    &bridge,
                    progress.sink(),
                ))
            })?
            .map_err(EngineFfiError::from)
    }

    pub fn reset(&self) -> FfiResult<()> {
        self.0
            .call(|session| futures_executor::block_on(session.reset()))
    }
}

/// What a live session knows mid-take: words it will not revise, and the tail
/// it still might.
#[derive(Debug, Clone, Default, PartialEq, uniffi::Record)]
pub struct LiveTranscript {
    pub committed: Vec<TranscribedWord>,
    pub volatile: Vec<TranscribedWord>,
}

/// `LiveSpeechTranscribing` — Nemotron 3.5 ASR streaming, 560 ms tier.
///
/// The dictation lane's model: it emits text while the microphone is still
/// open, which no full-attention encoder can. The caption lane keeps the batch
/// Conformers.
#[derive(uniffi::Object)]
pub struct LiveSpeechTranscriber(processors::NemotronTranscriber);

#[uniffi::export]
impl LiveSpeechTranscriber {
    #[uniffi::constructor]
    pub fn new(service: Arc<EngineService>) -> Self {
        Self(processors::NemotronTranscriber::new(
            service.backend(),
            service.models_directory(),
        ))
    }

    pub fn transcribe(
        &self,
        audio_url: String,
        speech_segments: Option<Vec<AudioEventSegment>>,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<TranscriptionResult> {
        let progress = call_progress(on_progress, cancellation);
        let segments = core_segments(&speech_segments);
        block(core::SpeechTranscribing::transcribe(
            &self.0,
            Path::new(&audio_url),
            segments.as_deref(),
            progress.sink(),
        ))
        .map(Into::into)
    }

    /// Loads whatever the session needs so the first `update` is a decode, not
    /// a wait.
    pub fn make_live_session(&self) -> FfiResult<Arc<LiveTranscriptionSession>> {
        let session = block(core::LiveSpeechTranscribing::make_live_session(&self.0))?;
        Ok(Arc::new(LiveTranscriptionSession::new(session)))
    }
}

/// Single still using the same super-resolution graph and output geometry as video.
#[derive(uniffi::Object)]
pub struct ImageUpscaler(Arc<EngineService>);

#[derive(uniffi::Record)]
pub struct ImageUpscaleResult {
    pub width: u32,
    pub height: u32,
}

#[uniffi::export]
impl ImageUpscaler {
    #[uniffi::constructor]
    pub fn new(engine: Arc<EngineService>) -> Self {
        Self(engine)
    }

    pub fn render(
        &self,
        source_url: String,
        output_url: String,
        mode: VideoUpscaleMode,
        on_progress: Listener,
        cancellation: Token,
    ) -> FfiResult<ImageUpscaleResult> {
        let progress = call_progress(on_progress, cancellation);
        let mode = match mode {
            VideoUpscaleMode::Enhance => {
                ferment_engine_processors::video_upscaler::UpscaleMode::Enhance
            }
            VideoUpscaleMode::Upscale { longest_side } => {
                ferment_engine_processors::video_upscaler::UpscaleMode::Upscale { longest_side }
            }
        };
        let result = block(ferment_engine_processors::render_pipeline::upscale_image(
            self.0.backend(),
            Path::new(&source_url),
            Path::new(&output_url),
            mode,
            progress.sink(),
        ))?;
        Ok(ImageUpscaleResult {
            width: result.width,
            height: result.height,
        })
    }
}
