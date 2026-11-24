//! Strongly typed model identifiers — single source of truth for the
//! names processors use to address inference models. Every iOS processor
//! gets a variant here, plus the encoder/decoder/joint splits.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    // BeatThis ensemble
    BeatThis1,
    BeatThis2,
    BeatThis3,

    // Parakeet TDT: the encoder. The mel and the decode are the host's.
    ParakeetEncoder,

    // DeepFilterNet
    Dfnet,

    // MusicHighlighter
    MusicHighlighter,

    // Dasheng audio semantic encoder
    DashengBase,

    // Paired retrieval spaces. Dasheng/DINO features are NOT interchangeable
    // with these text-aligned, versioned unit vectors.
    GlapAudio,
    GlapText,
    Siglip2Image,
    Siglip2Text,

    // MERT-v1 music features: distinct checkpoints and output dimensions.
    Mert95M,
    Mert330M,

    // Music tagger heads (genre/mood/instrument)
    MusicTaggerHeads,

    // Scene detector (autoshot)
    Autoshot,

    // DINOv2 ViT-S/14 visual semantic encoder (shot identity + scene embeddings)
    Dinov2Vits14,

    // FireRedVAD audio-event detector
    FireRedVadAed,

    // GigaAM-v3-e2e-rnnt Russian ASR (SberDevices, MIT)
    GigaAmPreprocessor,
    GigaAmEncoder,
    GigaAmDecoder,
    GigaAmJoint,

    // AmberNet spoken-language identification (NVIDIA NGC, VoxLingua107)
    AmberNetLangId,

    // 016 enhancement fleet (third-party ONNX; macOS tier-1 stays VideoToolbox)
    // Real-ESRGAN Compact super-resolution (SRVGGNetCompact, BSD-3-Clause)
    RealesrGeneralX4,
    RealesrGeneralWdnX4,
    RealesrAnimeVideo3,
    // FastDVDnet stateless 5-frame temporal denoiser (MIT)
    Fastdvdnet,
    // Practical-RIFE v4.25 frame interpolation, vs-mlrt v2 interface (MIT)
    RifeV425,

    // Speaker diarization: pyannote/segmentation-3.0 (MIT) says who speaks
    // when, ReDimNet2-B2 (MIT) says who they are. One measurement, two graphs.
    PyannoteSegmentation,
    RedimnetSpeaker,

    // Video saliency substitute backings (Apple Vision has no off-Apple twin):
    // U^2-Netp (Apache-2.0) gives the heat map, YuNet (MIT) the face boxes.
    U2Netp,
    YuNetFace,

    // EdgeTAM V1 promptable segmentation (Apache-2.0). iOS ships four Core ML
    // assets; the fourth is a MULTIFUNCTION package with sixteen specialized
    // point-count/mask variants. ONNX has no multifunction container, so the
    // decoder is ONE graph with a dynamic token axis and the SAM2 export's
    // `has_mask_input` flag — see `edgetam::model`.
    EdgetamImageEncoder,
    EdgetamPromptMaskDecoder,
    EdgetamMemoryEncoder,
    EdgetamMemoryAttention,

    // Nemotron 3.5 ASR streaming 0.6b multilingual, 560 ms tier
    // (nvidia, OpenMDW-1.1). The dictation lane's model: it emits text while
    // the microphone is still open, which no full-attention encoder can.
    NemotronPreprocessor,
    NemotronEncoder,
    NemotronDecoderJoint,
}

impl ModelId {
    pub const ALL: [Self; 38] = [
        Self::BeatThis1,
        Self::BeatThis2,
        Self::BeatThis3,
        Self::ParakeetEncoder,
        Self::Dfnet,
        Self::MusicHighlighter,
        Self::DashengBase,
        Self::GlapAudio,
        Self::GlapText,
        Self::Siglip2Image,
        Self::Siglip2Text,
        Self::Mert95M,
        Self::Mert330M,
        Self::MusicTaggerHeads,
        Self::Autoshot,
        Self::Dinov2Vits14,
        Self::FireRedVadAed,
        Self::GigaAmPreprocessor,
        Self::GigaAmEncoder,
        Self::GigaAmDecoder,
        Self::GigaAmJoint,
        Self::AmberNetLangId,
        Self::RealesrGeneralX4,
        Self::RealesrGeneralWdnX4,
        Self::RealesrAnimeVideo3,
        Self::Fastdvdnet,
        Self::RifeV425,
        Self::PyannoteSegmentation,
        Self::RedimnetSpeaker,
        Self::U2Netp,
        Self::YuNetFace,
        Self::EdgetamImageEncoder,
        Self::EdgetamPromptMaskDecoder,
        Self::EdgetamMemoryEncoder,
        Self::EdgetamMemoryAttention,
        Self::NemotronPreprocessor,
        Self::NemotronEncoder,
        Self::NemotronDecoderJoint,
    ];

    /// inference model repo name. Must match `models/<name>/config.pbtxt`.
    pub fn name(self) -> &'static str {
        match self {
            Self::BeatThis1 => "beat_this_1",
            Self::BeatThis2 => "beat_this_2",
            Self::BeatThis3 => "beat_this_3",
            Self::ParakeetEncoder => "parakeet_encoder",
            Self::Dfnet => "dfnet",
            Self::MusicHighlighter => "music_highlighter",
            Self::DashengBase => "dasheng_base_mel",
            Self::GlapAudio => "glap_audio_mel",
            Self::GlapText => "glap_text",
            Self::Siglip2Image => "siglip2_base_224_image",
            Self::Siglip2Text => "siglip2_base_224_text",
            Self::Mert95M => "mert_v1_95m",
            Self::Mert330M => "mert_v1_330m",
            Self::MusicTaggerHeads => "music_tagger_heads",
            Self::Autoshot => "autoshot",
            Self::Dinov2Vits14 => "dinov2_vits14",
            Self::FireRedVadAed => "fireredvad_aed",
            Self::GigaAmPreprocessor => "gigaam_preprocessor",
            Self::GigaAmEncoder => "gigaam_encoder",
            Self::GigaAmDecoder => "gigaam_decoder",
            Self::GigaAmJoint => "gigaam_joint",
            Self::AmberNetLangId => "ambernet_langid",
            Self::RealesrGeneralX4 => "realesr_general_x4v3",
            Self::RealesrGeneralWdnX4 => "realesr_general_wdn_x4v3",
            Self::RealesrAnimeVideo3 => "realesr_animevideov3",
            Self::Fastdvdnet => "fastdvdnet",
            Self::RifeV425 => "rife_v4_25",
            Self::PyannoteSegmentation => "pyannote_segmentation",
            Self::RedimnetSpeaker => "redimnet_speaker",
            Self::U2Netp => "u2netp",
            Self::YuNetFace => "yunet_face",
            Self::EdgetamImageEncoder => "edgetam_v1_image_encoder",
            Self::EdgetamPromptMaskDecoder => "edgetam_v1_prompt_mask_decoder",
            Self::EdgetamMemoryEncoder => "edgetam_v1_memory_encoder",
            Self::EdgetamMemoryAttention => "edgetam_v1_memory_attention",
            Self::NemotronPreprocessor => "nemotron_preprocessor",
            Self::NemotronEncoder => "nemotron_encoder",
            Self::NemotronDecoderJoint => "nemotron_decoder_joint",
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ModelVersion {
    /// "Latest" — sent as empty string to inference.
    #[default]
    Latest,
    /// Pinned version (e.g. "1", "2").
    Pinned(String),
}

impl ModelVersion {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Latest => "",
            Self::Pinned(v) => v.as_str(),
        }
    }

    pub fn pinned(v: impl Into<String>) -> Self {
        Self::Pinned(v.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_variant_has_unique_name() {
        let names = ModelId::ALL
            .iter()
            .map(|m| m.name())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(names.len(), ModelId::ALL.len());
    }

    #[test]
    fn version_latest_is_empty_string() {
        assert_eq!(ModelVersion::default().as_str(), "");
        assert_eq!(ModelVersion::pinned("3").as_str(), "3");
    }

    #[test]
    fn display_matches_name() {
        assert_eq!(ModelId::ParakeetEncoder.to_string(), "parakeet_encoder");
    }
}
