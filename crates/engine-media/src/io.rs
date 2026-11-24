use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    Codec, ColorInfo, DecoderConfiguration, EncodedPacket, MediaInfo, MediaTime, Result, TimeBase,
    VideoError, VideoFrame,
};

/// Android's `hdr-static-info`: descriptor id 0 and twelve little-endian
/// u16 fields, carried byte-for-byte from the source. The API owns no inferred
/// display or content-luminance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrStaticMetadata([u8; 25]);

impl HdrStaticMetadata {
    pub fn from_cta861_3(bytes: &[u8]) -> Result<Self> {
        let value: [u8; 25] = bytes.try_into().map_err(|_| {
            VideoError::InvalidData("HDR static descriptor must contain exactly 25 bytes".into())
        })?;
        if value[0] != 0 {
            return Err(VideoError::InvalidData(
                "unsupported HDR static descriptor id".into(),
            ));
        }
        Ok(Self(value))
    }

    pub const fn as_bytes(&self) -> &[u8; 25] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderSettings {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    /// Timestamp units, independent of the nominal picture cadence.
    pub time_base: TimeBase,
    /// Optional nominal frames-per-second numerator/denominator. This is a
    /// codec rate-control hint only: variable-rate input keeps its supplied
    /// PTS and durations. It must never be reconstructed from time_base.
    pub frame_rate: Option<(u32, u32)>,
    pub bit_rate: Option<u64>,
    pub profile: Option<String>,
    pub max_keyframe_interval: Option<u32>,
    /// `false` forbids B-frames, so delivery order equals presentation order.
    /// Stated rather than left to the encoder: VideoToolbox documents its
    /// default as `true`, so a bounded-GOP preset would silently regain
    /// reordering and break callers that index frames by delivery position.
    pub allow_frame_reordering: bool,
    /// Colorimetry the compressed stream declares. `None` leaves the encoder's
    /// inference in place; stating it keeps the tags off the input buffers.
    pub color: Option<ColorInfo>,
    /// Request the platform's automatic HDR metadata policy. Preserve actual
    /// source mastering metadata when available; HLG and PQ without a source
    /// descriptor must not acquire invented mastering values. Output samples
    /// and color tags must still preserve the requested HDR signal.
    pub insert_hdr_metadata: bool,
    /// Original CTA-861.3 descriptor, when the source actually declares one.
    /// HLG has no mastering descriptor. Never synthesize PQ mastering values
    /// from primaries/transfer tags; those do not describe a mastering display.
    pub hdr_static_metadata: Option<HdrStaticMetadata>,
}

impl EncoderSettings {
    /// Baseline settings for `codec` at `width`×`height`: no rate control, no
    /// profile pin, encoder-chosen GOP, reordering allowed, inferred color.
    /// Callers override the fields they mean to state.
    #[must_use]
    pub fn new(codec: Codec, width: u32, height: u32, time_base: TimeBase) -> Self {
        Self {
            codec,
            width,
            height,
            time_base,
            frame_rate: None,
            bit_rate: None,
            profile: None,
            max_keyframe_interval: None,
            allow_frame_reordering: true,
            color: None,
            insert_hdr_metadata: false,
            hdr_static_metadata: None,
        }
    }

    /// Initial rate-control hint for encoders that require one (MediaCodec).
    /// Unknown/zero nominal metadata uses the same 30 fps fallback as source
    /// bitrate planning and motion analysis. This does not prescribe a CFR
    /// timeline: submitted buffer timestamps remain authoritative for VFR.
    #[must_use]
    pub fn frame_rate_hint(&self) -> f64 {
        let (numerator, denominator) = self
            .frame_rate
            .filter(|(numerator, denominator)| *numerator > 0 && *denominator > 0)
            .unwrap_or((30, 1));
        f64::from(numerator) / f64::from(denominator)
    }

    /// MediaCodec expresses GOP intent in seconds, not picture counts.
    /// A one-picture GOP means every picture is a keyframe; other bounds must
    /// retain their fractional seconds instead of truncating to all-intra.
    #[must_use]
    pub fn keyframe_interval_seconds(&self) -> f64 {
        match self.max_keyframe_interval {
            None => 1.0,
            Some(0 | 1) => 0.0,
            Some(frames) => f64::from(frames) / self.frame_rate_hint(),
        }
    }
}

/// Backend policy is independent of the output's size and residency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecodeAcceleration {
    /// Use the platform decode path, with the backend's ordinary fallback.
    #[default]
    Automatic,
    /// Explicit CPU decode, e.g. a measured small-thumbnail workload.
    Software,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecodeOutput {
    /// Keep the backend's native frame representation until explicit readback.
    #[default]
    Native,
    /// Materialize an eight-bit BGRA CPU frame in SDR BT.709. A PQ or HLG
    /// source is tone-mapped to it (BT.2390 onto the BT.2408 reference white).
    CpuBgra8,
    /// Preserve float32 components on the CPU. Refuses integer sources and resize.
    CpuRgba32Float,
    /// A PQ or HLG source as the light of an HDR project: float32 RGBA,
    /// linear BT.709 primaries with extended range, 1.0 the BT.2408 reference
    /// white (203 cd/m²), highlights above it; no tone mapping or clipping.
    /// Refuses SDR sources, which keep eight bits.
    CpuSceneLinear,
    /// Planes a GPU reads in place, never copied to the CPU: on macOS an
    /// IOSurface-backed, Metal-compatible bi-planar Y'CbCr `CVPixelBuffer`
    /// (`FrameBuffer::MacOS`) at the stream's own depth, chroma and range —
    /// `420v`/`420f` for eight bits, `x420`/`xf20` deeper, `x422`/`xf22` for
    /// 4:2:2 ProRes — which VideoToolbox decodes into. The colour is the
    /// stream's, HDR included, unconverted. No resize. Where no such decoder
    /// takes the stream (another platform or codec, ProRes 4444 with its
    /// alpha), opening reports `BackendUnavailable` and the caller decodes to
    /// the CPU instead.
    GpuPlanes,
    /// A PQ or HLG source's own Y'CbCr codes in memory (`FrameBuffer::Ycbcr`)
    /// for a GPU to convert as it does `GpuPlanes`: a software decoder's
    /// planes at the stream's depth and chroma, unconverted and unscaled.
    /// Where a hardware decoder would take the stream, `GpuPlanes` is the
    /// better ask; this is what every other HDR source can still give.
    CpuPlanes,
}

/// What a caller may ask of a decoder beyond the stream itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// None preserves the source extent; Some requests an exact resize.
    /// This does not choose CPU output or software decoding.
    pub output_size: Option<(u32, u32)>,
    pub acceleration: DecodeAcceleration,
    pub output: DecodeOutput,
}

impl DecodeOptions {
    /// The measured small CPU-thumbnail policy, explicit at its consumer.
    pub fn software_scaled(size: (u32, u32)) -> Self {
        Self {
            output_size: Some(size),
            acceleration: DecodeAcceleration::Software,
            output: DecodeOutput::Native,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self
            .output_size
            .is_some_and(|(w, h)| w == 0 || h == 0 || w > i32::MAX as u32 || h > i32::MAX as u32)
        {
            return Err(VideoError::InvalidData(
                "Invalid decoder output size".into(),
            ));
        }
        if matches!(
            self.output,
            DecodeOutput::GpuPlanes | DecodeOutput::CpuPlanes
        ) && self.output_size.is_some()
        {
            return Err(VideoError::InvalidData(
                "Y'CbCr planes keep the source extent; the GPU scales them".into(),
            ));
        }
        if self.output == DecodeOutput::GpuPlanes
            && self.acceleration == DecodeAcceleration::Software
        {
            return Err(VideoError::BackendUnavailable(
                "GPU planes come from the platform's hardware decoder, not software".into(),
            ));
        }
        Ok(())
    }

    /// CPU BGRA8 keeps float sources out: they have their float pipeline. A
    /// backend that cannot tone-map (`converts_hdr` false) also refuses HDR
    /// and high-bit-depth sources, so the caller falls back to libav.
    pub(crate) fn validate_color(self, color: ColorInfo, converts_hdr: bool) -> Result<()> {
        if self.output == DecodeOutput::CpuSceneLinear
            && (!converts_hdr || !crate::hdr::is_hdr(color.transfer))
        {
            return Err(VideoError::UnsupportedCodec(
                "Scene-linear decode takes a PQ or HLG source on a backend that converts HDR"
                    .into(),
            ));
        }
        if self.output == DecodeOutput::CpuPlanes
            && (!crate::hdr::is_hdr(color.transfer) || color.bits_per_channel == Some(32))
        {
            return Err(VideoError::UnsupportedCodec(
                "CPU Y'CbCr planes are for a PQ or HLG Y'CbCr source".into(),
            ));
        }
        if self.output == DecodeOutput::GpuPlanes && color.bits_per_channel == Some(32) {
            return Err(VideoError::UnsupportedCodec(
                "A float source keeps its float pipeline, not Y'CbCr GPU planes".into(),
            ));
        }
        if self.output == DecodeOutput::CpuBgra8 && color.bits_per_channel == Some(32) {
            return Err(VideoError::UnsupportedCodec(
                "A float source keeps its float pipeline, not CPU BGRA8".into(),
            ));
        }
        if self.output == DecodeOutput::CpuBgra8
            && !converts_hdr
            && (color.bits_per_channel.is_some_and(|bits| bits > 8)
                || crate::hdr::is_hdr(color.transfer))
        {
            return Err(VideoError::UnsupportedCodec(
                "HDR/high-bit-depth decode requires an explicit color pipeline, not CPU BGRA8"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EncodedVideo {
    pub configuration: DecoderConfiguration,
    pub packets: Vec<EncodedPacket>,
}

#[async_trait(?Send)]
pub trait DecodeSession: Debug {
    /// Reset at the preceding sync sample without discarding leading pictures.
    /// Frame-accurate consumers select the interval using native presentation
    /// timestamps. Unsupported backends must reject, never emulate `seek`.
    async fn seek_sync(&mut self, _target: MediaTime) -> Result<()> {
        Err(crate::VideoError::BackendUnavailable(
            "Sync-sample seek is not supported by this decoder".into(),
        ))
    }
    /// Reset codec state, seek the demuxer to the previous sync sample, and
    /// discard decoded leading pictures until `target` is reached.
    async fn seek(&mut self, target: MediaTime) -> Result<()>;

    /// Return the next picture in presentation order.
    async fn decode_next(&mut self) -> Result<Option<VideoFrame>>;
}

#[async_trait(?Send)]
pub trait EncodeSession: Debug {
    async fn push_frame(&mut self, frame: VideoFrame) -> Result<()>;
    async fn finalize(self: Box<Self>) -> Result<EncodedVideo>;
}

#[async_trait(?Send)]
pub trait VideoIo: Debug + Send + Sync {
    async fn probe(&self, source: &Path) -> Result<MediaInfo>;

    async fn open_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>>;

    /// Coded-orientation pictures: the container display matrix is NOT applied.
    /// This lets a display-aware consumer apply it exactly once. Current
    /// VideoToolbox/libav readers already satisfy this; Android overrides this
    /// door to disable MediaCodec surface auto-rotation, including after seek.
    /// Other decoder doors preserve their existing platform behavior.
    async fn open_coded_decoder(
        &self,
        source: &Path,
        stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        self.open_decoder(source, stream_index).await
    }

    /// `open_decoder` with [`DecodeOptions`]. A backend must explicitly
    /// implement nondefault requests; silently ignoring residency/size or
    /// acceleration would violate the consumer's frame contract.
    async fn open_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        options.validate()?;
        if options != DecodeOptions::default() {
            return Err(VideoError::BackendUnavailable(
                "This decoder does not support explicit decode options".into(),
            ));
        }
        self.open_decoder(source, stream_index).await
    }

    /// Coded pictures with an optional decoder resize. Default backends
    /// already return coded pixels, so preserve their existing resize support.
    /// Surface backends override this door to suppress automatic transforms.
    async fn open_coded_decoder_with(
        &self,
        source: &Path,
        stream_index: u32,
        options: DecodeOptions,
    ) -> Result<Box<dyn DecodeSession>> {
        self.open_decoder_with(source, stream_index, options).await
    }

    /// Original static HDR data, not guessed from color tags. The Android
    /// extractor exposes the exact CTA descriptor consumed by MediaCodec.
    async fn hdr_static_metadata(
        &self,
        _source: &Path,
        _stream_index: u32,
    ) -> Result<Option<HdrStaticMetadata>> {
        Ok(None)
    }

    /// Coded pictures for an encoder's target color. Native macOS pixel
    /// transfer/encoding already performs the conversion. Android requests
    /// and verifies decoder tone mapping for an SDR-only derivative pass.
    async fn open_coded_decoder_for_color(
        &self,
        source: &Path,
        stream_index: u32,
        _color: ColorInfo,
    ) -> Result<Box<dyn DecodeSession>> {
        self.open_coded_decoder(source, stream_index).await
    }

    async fn open_encoder(&self, settings: EncoderSettings) -> Result<Box<dyn EncodeSession>>;
}

pub type VideoBackend = Arc<dyn VideoIo>;

#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableVideoIo;

#[async_trait(?Send)]
impl VideoIo for UnavailableVideoIo {
    async fn probe(&self, source: &Path) -> Result<MediaInfo> {
        crate::demux::probe(source)
    }

    async fn open_decoder(
        &self,
        _source: &Path,
        _stream_index: u32,
    ) -> Result<Box<dyn DecodeSession>> {
        Err(unavailable())
    }

    async fn open_encoder(&self, _settings: EncoderSettings) -> Result<Box<dyn EncodeSession>> {
        Err(unavailable())
    }
}

fn unavailable() -> VideoError {
    VideoError::BackendUnavailable("no platform VideoIO backend configured".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated/h264-gop4.mp4")
    }

    #[test]
    fn unavailable_backend_still_probes_and_rejects_codec_sessions() {
        futures_executor::block_on(async {
            let backend = UnavailableVideoIo;
            assert_eq!(
                backend
                    .probe(&fixture())
                    .await
                    .unwrap()
                    .video_streams()
                    .count(),
                1
            );

            let decoder_error = backend.open_decoder(&fixture(), 0).await.unwrap_err();
            assert!(matches!(decoder_error, VideoError::BackendUnavailable(_)));

            let encoder_error = backend
                .open_encoder(EncoderSettings::new(
                    Codec::H264,
                    16,
                    16,
                    TimeBase::new_unchecked(1, 30),
                ))
                .await
                .unwrap_err();
            assert!(matches!(encoder_error, VideoError::BackendUnavailable(_)));
        });
    }
    // Kill: use a codec clock (15360/90000/1e6 ticks per second) as the
    // number of pictures per second, or change it to match a nominal rate.
    #[test]
    fn cadence_is_independent_of_the_timestamp_clock() {
        for denominator in [30, 15360, 90000, 1_000_000] {
            let clock = TimeBase::new_unchecked(1, denominator);
            let mut settings = EncoderSettings::new(Codec::Hevc, 1440, 2560, clock);
            settings.frame_rate = Some((30, 1));
            assert_eq!(settings.frame_rate_hint(), 30.0);
            assert_eq!(settings.time_base, clock);
        }
    }

    // Kill: integer rounding or the old 240 fps clamp. NTSC and rates above
    // that arbitrary cap must reach the actual codec's admission policy.
    #[test]
    fn cadence_preserves_ntsc_and_does_not_clamp_supported_rates() {
        let mut settings =
            EncoderSettings::new(Codec::H264, 64, 48, TimeBase::new_unchecked(1, 90000));
        for (numerator, denominator) in [(30000, 1001), (60000, 1001), (300, 1)] {
            settings.frame_rate = Some((numerator, denominator));
            let expected = f64::from(numerator) / f64::from(denominator);
            assert_eq!(settings.frame_rate_hint(), expected);
            assert!((f64::from(settings.frame_rate_hint() as f32) - expected).abs() < 0.00001);
        }
        assert_eq!(settings.frame_rate_hint(), 300.0);
    }

    // Kill: reject a valid VFR file solely because it has no nominal rate,
    // divide by zero, or resurrect the clock-derived cadence as its fallback.
    #[test]
    fn unknown_or_unusable_cadence_uses_only_a_rate_control_hint() {
        let clock = TimeBase::new_unchecked(1, 1_000_000);
        let mut settings = EncoderSettings::new(Codec::Hevc, 64, 48, clock);
        for unknown in [None, Some((0, 1)), Some((30, 0)), Some((0, 0))] {
            settings.frame_rate = unknown;
            assert_eq!(settings.frame_rate_hint(), 30.0);
            assert_eq!(settings.frame_rate, unknown);
            assert_eq!(settings.time_base, clock);
        }
    }

    // Kill: convert frame-count GOP bounds by integer division, producing
    // all-intra or an incorrect interval for fractional-rate input.
    #[test]
    fn fractional_gop_uses_the_explicit_nominal_cadence() {
        let mut settings =
            EncoderSettings::new(Codec::H264, 64, 48, TimeBase::new_unchecked(1, 90000));
        settings.frame_rate = Some((30, 1));
        settings.max_keyframe_interval = Some(12);
        assert_eq!(settings.keyframe_interval_seconds(), 0.4);
        settings.frame_rate = Some((30000, 1001));
        settings.max_keyframe_interval = Some(30);
        assert!((settings.keyframe_interval_seconds() - 1.001).abs() < 1e-12);
    }

    // Kill: reinterpret an all-intra preset as merely a short GOP, or alter
    // the previous encoder-chosen one-second default.
    #[test]
    fn all_intra_and_default_gop_intents_are_preserved() {
        let mut settings =
            EncoderSettings::new(Codec::H264, 64, 48, TimeBase::new_unchecked(1, 15360));
        settings.frame_rate = Some((30000, 1001));
        assert_eq!(settings.keyframe_interval_seconds(), 1.0);
        for frames in [0, 1] {
            settings.max_keyframe_interval = Some(frames);
            assert_eq!(settings.keyframe_interval_seconds(), 0.0);
        }
    }
}
