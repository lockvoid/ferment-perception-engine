//! FermentEngine processor families. Modules keep the Swift family boundaries
//! while sharing backend-neutral audio and inference contracts.

pub mod ambernet;
pub mod audio_encoder;
pub mod audio_event_detector;
pub mod audio_minimap_composer;
pub mod audio_source;
pub mod beat_queen_mary;
pub mod beatthis;
mod capability_impls;
pub mod caption_segmenter;
pub mod color_analyzer;
pub mod dasheng;
pub mod deepfilternet;
pub mod dinov2;
pub mod edgetam;
pub mod encode_pipeline;
pub mod energy_structure;
mod frame_tensors;
pub mod gigaam;
pub mod loop_bpm;
pub mod loudness_normalizer;
pub mod musichighlighter;
pub mod musictagger;
pub mod nemotron;
mod optional_failure;
pub mod parakeet;
pub mod pyannote;
pub mod redimnet;
pub mod render_pipeline;
mod rgb_hq;
pub mod slow_motion_renderer;
pub mod thumb_container;
pub mod token_seam_merge;
pub mod transcriber_routing;
pub mod video_encoder;
pub mod video_minimap_composer;
pub mod video_noise_filter;
pub mod video_saliency;
pub mod video_scene_detector;
mod video_source;
pub mod video_upscaler;
pub mod waveform_extractor;

pub use ambernet::AmberNetLanguageId;
pub use audio_encoder::AudioEncoder;
pub use audio_event_detector::AudioEventDetector;
pub use audio_minimap_composer::AudioMinimapComposer;
pub use audio_source::{
    ArrayAudioSampleSource, AudioSampleSource, AudioSource, DiskBackedAudioSampleSource,
};
pub use beat_queen_mary::BeatQueenMary;
pub use beatthis::BeatThis;
pub use caption_segmenter::CaptionSegmenter;
pub use color_analyzer::CpuColorAnalyzer;
pub use dasheng::AudioSemanticEncoder;
pub use deepfilternet::DeepFilterNet;
pub use dinov2::{Dinov2SemanticEncoder, FrameSnapshot};
pub use edgetam::EdgeTam;
pub use encode_pipeline::{EncodeBackends, platform_encode_backends};
pub use gigaam::GigaAm;
pub use loudness_normalizer::LoudnessNormalizer;
pub use musichighlighter::MusicHighlighter;
pub use musictagger::MusicTagger;
pub use nemotron::NemotronTranscriber;
pub use parakeet::Parakeet;
pub use pyannote::PyannoteSpeakerSegmenter;
pub use redimnet::ReDimNetSpeakerEncoder;
pub use slow_motion_renderer::{
    BlockMotionEstimator, LumaPlane, MotionBlurDecision, MotionBlurMode, MotionBlurPolicy,
    MotionBlurReason, MotionField, MotionVector, RifeFrameInterpolator, ShutterAngleEstimator,
    SlowMotionAnalysis, SlowMotionAnalyzer, SlowMotionRenderer,
};
pub use thumb_container::{IndexEntry, ThumbContainer, encode_thumbnail_jpeg};
pub use video_encoder::{
    AtomicFileDelivery, HdrTransfer, SourceVideo, VideoCodecSettings, VideoEncoder,
    VideoEncodingError, VideoProfile,
};
pub use video_minimap_composer::VideoMinimapComposer;
pub use video_noise_filter::{FastDvdnetTemporalFilter, VideoNoiseFilter};
pub use video_saliency::{U2NetpHeatSource, VideoSaliencyAnalyzer, YuNetFaceSource};
pub use video_scene_detector::VideoSceneDetector;
pub use video_source::video_backend;
pub use video_upscaler::{CompactFrameUpscaler, VideoUpscaler};
pub use waveform_extractor::WaveformExtractor;
