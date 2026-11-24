use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, MultiVideoEncoding, MultiVideoEncodingResult, ProgressSink, Result,
    ThumbStripOptions, ThumbnailSink, VariantThumbnailSink, VideoEncoding, VideoEncodingPreset,
    VideoEncodingResult, VideoEncodingVariant, VideoFileType, VideoRateControl,
};
use ferment_engine_media::{
    Codec, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, StreamInfo, TransferCharacteristic,
};

/// Media-independent planning and validation for the VideoEncoder family,
/// plus the real proxy/AI encode when platform backends are attached.
///
/// Platform decode, codec sessions, and container muxing stay in
/// `engine-media`; this type pins the Swift encoder's preset, sizing, rate
/// control, HDR, and atomic-delivery contracts. `with_backends` turns
/// `VideoEncoding::encode` and `MultiVideoEncoding::encode_variants` into the
/// full decode → scale → encode → mux pass; without backends they keep
/// reporting the boundary.
#[derive(Debug, Clone, Default)]
pub struct VideoEncoder {
    backends: Option<crate::EncodeBackends>,
}

impl VideoEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_backends(mut self, backends: crate::EncodeBackends) -> Self {
        self.backends = Some(backends);
        self
    }

    fn backends(&self) -> Result<&crate::EncodeBackends> {
        self.backends
            .as_ref()
            .ok_or_else(|| EngineError::Unsupported(Self::MUX_UNAVAILABLE.into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrTransfer {
    Hlg,
    Pq,
}

impl HdrTransfer {
    const fn characteristic(self) -> TransferCharacteristic {
        match self {
            Self::Hlg => TransferCharacteristic::Hlg,
            Self::Pq => TransferCharacteristic::Pq,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoProfile {
    H264HighAutoLevel,
    HevcMain10AutoLevel,
}

/// The source facts a plan needs: coded geometry, the frame rate a
/// bits-per-pixel rate control resolves against, and the colorimetry that
/// decides the HDR branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceVideo {
    pub width: u32,
    pub height: u32,
    pub frames_per_second: f64,
    pub color: ColorInfo,
}

impl SourceVideo {
    /// Read the plan inputs off a probed video stream. A stream that states no
    /// usable frame rate resolves at [`VideoRateControl::FALLBACK_FRAME_RATE`].
    pub fn from_stream(stream: &StreamInfo) -> Result<Self> {
        let width = stream
            .width
            .ok_or_else(|| EngineError::Decode("video stream has no width".into()))?;
        let height = stream
            .height
            .ok_or_else(|| EngineError::Decode("video stream has no height".into()))?;
        let frames_per_second = stream
            .frame_rate
            .filter(|(_, denominator)| *denominator != 0)
            .map_or(0.0, |(numerator, denominator)| {
                f64::from(numerator) / f64::from(denominator)
            });
        Ok(Self {
            width,
            height,
            frames_per_second,
            color: stream.color,
        })
    }
}

/// One encoder session's full configuration — the port of the Swift
/// `sdrVideoSettings` / `hdrVideoSettings` dictionaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoCodecSettings {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub bit_rate: u64,
    pub profile: VideoProfile,
    pub max_key_frame_interval: Option<u32>,
    pub allow_frame_reordering: bool,
    /// Colorimetry the output states. Always stated, never inferred from the
    /// input buffers: an SDR variant of an HDR source would otherwise inherit
    /// BT.2020/HLG tags onto BT.709 pixels.
    pub color: ColorInfo,
    pub insert_hdr_metadata: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoEncodingError {
    ThumbStripWithoutHandler,
    NoVariants,
    DuplicateVariantId(String),
}

impl fmt::Display for VideoEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThumbStripWithoutHandler => formatter.write_str(
                "VideoEncoder: thumbStrip options provided but onThumbnail handler is nil",
            ),
            Self::NoVariants => {
                formatter.write_str("VideoEncoder: encodeVariants called with no variants")
            }
            Self::DuplicateVariantId(id) => {
                write!(formatter, "VideoEncoder: duplicate variant id {id}")
            }
        }
    }
}

impl std::error::Error for VideoEncodingError {}

impl VideoEncoder {
    pub const MUX_UNAVAILABLE: &'static str = "VideoEncoder: engine-media exposes encoded video packets but no container mux/finalize-to-path API";

    /// Proxy: 1080 SHORT-edge all-intra at 0.75 bits/pixel, HDR-preserving.
    ///
    /// Short edge because a portrait 4K proxied on its long axis lands at
    /// 606 wide — softer than the display it upscales onto. 0.75 bpp is where
    /// three independent calibrations converge for all-intra 1080p (Instagram
    /// Edits proxies measure 0.70–0.76 by storage delta, ProRes 422 Proxy is
    /// 0.72); the fixed 8 Mbps it replaced is ~0.2 bpp at 60 fps and reads as
    /// mush. Every frame is a keyframe so scrubbing seeks land instantly.
    pub fn proxy_preset() -> VideoEncodingPreset {
        VideoEncodingPreset {
            label: "proxy".into(),
            file_type: VideoFileType::Mov,
            max_width: 1_080,
            caps_short_edge: true,
            rate_control: VideoRateControl::BitsPerPixel(0.75),
            b_frames: 0,
            key_frame_interval: 1,
            hdr_eligible: true,
        }
    }

    /// AI: the perception feed. 640 on the LONG axis, aspect preserved,
    /// H.264 at a fixed 2.5 Mbps, SDR, never uploaded.
    ///
    /// 640 is the size every current consumer reaches by downscaling.
    /// `b_frames: 0` is the load-bearing half: it suppresses frame reordering
    /// so delivery order equals presentation order, which the scene detectors
    /// rely on because they index frames positionally.
    /// `key_frame_interval: 60` is a bound, not a contract — the unit is
    /// FRAMES, so it lands between 1.0 s (60 fps) and 2.5 s (24 fps). It only
    /// stops the encoder choosing an unbounded GOP; deliberately not all-intra,
    /// which would multiply the file for seekability nothing here uses.
    pub fn ai_preset() -> VideoEncodingPreset {
        VideoEncodingPreset {
            label: "ai".into(),
            file_type: VideoFileType::Mp4,
            max_width: 640,
            caps_short_edge: false,
            rate_control: VideoRateControl::Average(2_500_000),
            b_frames: 0,
            key_frame_interval: 60,
            hdr_eligible: false,
        }
    }

    pub fn detect_hdr_transfer(color: ColorInfo) -> Option<HdrTransfer> {
        match color.transfer {
            TransferCharacteristic::Hlg => Some(HdrTransfer::Hlg),
            TransferCharacteristic::Pq => Some(HdrTransfer::Pq),
            _ => None,
        }
    }

    /// Aspect-preserving fit of `natural_width`×`natural_height` with one edge
    /// capped at `max_edge`, floored to even dimensions.
    ///
    /// `caps_short_edge` decides WHICH edge is capped: the proxy caps the
    /// short one (portrait 4K → 1080×1920), the AI feed the long one
    /// (portrait 4K → 360×640). Never upscales.
    pub fn scaled_size(
        natural_width: u32,
        natural_height: u32,
        max_edge: u32,
        caps_short_edge: bool,
    ) -> (u32, u32) {
        if natural_width == 0 || natural_height == 0 {
            return (max_edge, max_edge);
        }
        let capped_edge = if caps_short_edge {
            natural_width.min(natural_height)
        } else {
            natural_width.max(natural_height)
        };
        let scale = (f64::from(max_edge) / f64::from(capped_edge)).min(1.0);
        let width = ((f64::from(natural_width) * scale) as u32) & !1;
        let height = ((f64::from(natural_height) * scale) as u32) & !1;
        // Upstream returns the raw floor; a sub-2-pixel edge would hand the
        // platform encoder a zero dimension, so it is clamped here.
        (width.max(2), height.max(2))
    }

    /// GOP + reordering, shared by both colorimetries.
    ///
    /// `b_frames == 0` means all-intra or an I/P-only bounded GOP, and it
    /// states `allow_frame_reordering = false` — VideoToolbox documents its
    /// default as `true`, so the AI preset's bounded GOP would otherwise
    /// silently regain reordering, which the scene detectors cannot survive
    /// because they index frames by delivery position.
    fn group_of_pictures(b_frames: u8, key_frame_interval: u32) -> (Option<u32>, bool) {
        if b_frames == 0 {
            (
                (key_frame_interval > 0).then_some(key_frame_interval),
                false,
            )
        } else {
            (None, true)
        }
    }

    pub fn hdr_video_settings(
        width: u32,
        height: u32,
        bit_rate: u64,
        b_frames: u8,
        key_frame_interval: u32,
        transfer: HdrTransfer,
    ) -> VideoCodecSettings {
        let (max_key_frame_interval, allow_frame_reordering) =
            Self::group_of_pictures(b_frames, key_frame_interval);
        VideoCodecSettings {
            codec: Codec::Hevc,
            width,
            height,
            bit_rate,
            profile: VideoProfile::HevcMain10AutoLevel,
            max_key_frame_interval,
            allow_frame_reordering,
            color: ColorInfo {
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt2020,
                transfer: transfer.characteristic(),
                matrix: ColorMatrix::Bt2020NonConstant,
                bits_per_channel: Some(10),
            },
            insert_hdr_metadata: true,
        }
    }

    pub fn sdr_video_settings(
        width: u32,
        height: u32,
        bit_rate: u64,
        b_frames: u8,
        key_frame_interval: u32,
    ) -> VideoCodecSettings {
        let (max_key_frame_interval, allow_frame_reordering) =
            Self::group_of_pictures(b_frames, key_frame_interval);
        VideoCodecSettings {
            codec: Codec::H264,
            width,
            height,
            bit_rate,
            profile: VideoProfile::H264HighAutoLevel,
            max_key_frame_interval,
            allow_frame_reordering,
            color: ColorInfo {
                range: ColorRange::Limited,
                primaries: ColorPrimaries::Bt709,
                transfer: TransferCharacteristic::Bt709,
                matrix: ColorMatrix::Bt709,
                bits_per_channel: Some(8),
            },
            insert_hdr_metadata: false,
        }
    }

    /// Resolve `preset` against one source into the exact encoder session.
    ///
    /// HDR pass-through is gated on BOTH an HLG/PQ source AND a preset that
    /// opts in, so an SDR-only preset always emits BT.709-tagged H.264.
    #[must_use]
    pub fn plan(source: SourceVideo, preset: &VideoEncodingPreset) -> VideoCodecSettings {
        let (width, height) = Self::scaled_size(
            source.width,
            source.height,
            preset.max_width,
            preset.caps_short_edge,
        );
        let bit_rate =
            preset
                .rate_control
                .resolved_bit_rate(width, height, source.frames_per_second);
        match Self::detect_hdr_transfer(source.color).filter(|_| preset.hdr_eligible) {
            Some(transfer) => Self::hdr_video_settings(
                width,
                height,
                bit_rate,
                preset.b_frames,
                preset.key_frame_interval,
                transfer,
            ),
            None => Self::sdr_video_settings(
                width,
                height,
                bit_rate,
                preset.b_frames,
                preset.key_frame_interval,
            ),
        }
    }

    pub fn validate_thumb_strip(
        thumb_strip: Option<ThumbStripOptions>,
        has_thumbnail_handler: bool,
    ) -> std::result::Result<(), VideoEncodingError> {
        if thumb_strip.is_some() && !has_thumbnail_handler {
            Err(VideoEncodingError::ThumbStripWithoutHandler)
        } else {
            Ok(())
        }
    }

    /// A multi-variant request is well-formed: non-empty, unique ids, and a
    /// handler for whichever variants want thumbnails.
    pub fn validate_variants(
        variants: &[VideoEncodingVariant],
        has_thumbnail_handler: bool,
    ) -> std::result::Result<(), VideoEncodingError> {
        if variants.is_empty() {
            return Err(VideoEncodingError::NoVariants);
        }
        for (index, variant) in variants.iter().enumerate() {
            if variants[..index].iter().any(|seen| seen.id == variant.id) {
                return Err(VideoEncodingError::DuplicateVariantId(variant.id.clone()));
            }
            Self::validate_thumb_strip(variant.thumb_strip, has_thumbnail_handler)?;
        }
        Ok(())
    }
}

fn invalid_config(error: VideoEncodingError) -> EngineError {
    EngineError::invalid_config(error.to_string())
}

#[async_trait(?Send)]
impl VideoEncoding for VideoEncoder {
    async fn encode(
        &self,
        source_path: &Path,
        output_path: &Path,
        preset: &VideoEncodingPreset,
        thumb_strip: Option<ThumbStripOptions>,
        progress: ProgressSink,
        on_thumbnail: Option<&dyn ThumbnailSink>,
    ) -> Result<VideoEncodingResult> {
        Self::validate_thumb_strip(thumb_strip, on_thumbnail.is_some()).map_err(invalid_config)?;
        let backends = self.backends()?;
        let started = std::time::Instant::now();
        let mut result = crate::encode_pipeline::transcode_video(
            backends,
            source_path,
            output_path,
            preset,
            thumb_strip,
            &progress,
            on_thumbnail,
        )
        .await?;
        progress.report(1.0, "done").await;
        result.wall_seconds = started.elapsed().as_secs_f64();
        Ok(result)
    }
}

#[async_trait(?Send)]
impl MultiVideoEncoding for VideoEncoder {
    async fn encode_variants(
        &self,
        source_path: &Path,
        variants: &[VideoEncodingVariant],
        progress: ProgressSink,
        on_thumbnail: Option<&dyn VariantThumbnailSink>,
    ) -> Result<MultiVideoEncodingResult> {
        Self::validate_variants(variants, on_thumbnail.is_some()).map_err(invalid_config)?;
        let backends = self.backends()?;
        let started = std::time::Instant::now();
        let mut result = crate::encode_pipeline::transcode_variants(
            backends,
            source_path,
            variants,
            &progress,
            on_thumbnail,
        )
        .await?;
        progress.report(1.0, "done").await;
        result.wall_seconds = started.elapsed().as_secs_f64();
        Ok(result)
    }
}

/// Sibling-file staging used by streaming encoders.
///
/// The platform writer receives [`Self::partial_path`]. Dropping without
/// `commit` cleans the partial, while `commit` makes the final path visible in
/// one same-directory rename.
#[derive(Debug)]
pub struct AtomicFileDelivery {
    final_path: PathBuf,
    partial_path: PathBuf,
    committed: bool,
}

impl AtomicFileDelivery {
    pub fn begin(final_path: &Path) -> Result<Self> {
        ferment_engine_core::WorkActivity::check_current()?;
        if let Some(parent) = final_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let partial_path = Self::reserve_partial(final_path)?;
        Ok(Self {
            final_path: final_path.to_owned(),
            partial_path,
            committed: false,
        })
    }

    fn reserve_partial(final_path: &Path) -> Result<PathBuf> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let file_name = final_path.file_name().ok_or_else(|| {
            ferment_engine_core::EngineError::invalid_config("output path has no file name")
        })?;
        let stem = match final_path.file_stem() {
            Some(stem) if final_path.extension().is_some() => stem.to_os_string(),
            _ => file_name.to_os_string(),
        };
        loop {
            let mut partial_name = std::ffi::OsString::from(".");
            partial_name.push(&stem);
            partial_name.push(format!(
                ".partial-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            if let Some(extension) = final_path.extension() {
                partial_name.push(".");
                partial_name.push(extension);
            }
            let path = final_path.with_file_name(partial_name);
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub fn partial_path(&self) -> &Path {
        &self.partial_path
    }

    pub fn commit(mut self) -> Result<()> {
        ferment_engine_core::WorkActivity::check_current()?;
        let _phase = ferment_engine_core::WorkActivity::current()
            .map(|work| work.enter_phase(ferment_engine_core::WorkPhase::Finalizing));
        ferment_engine_core::WorkActivity::check_current()?;
        fs::rename(&self.partial_path, &self.final_path)?;
        self.committed = true;
        Ok(())
    }

    pub fn write<T>(final_path: &Path, perform: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
        let delivery = Self::begin(final_path)?;
        let result = perform(delivery.partial_path())?;
        delivery.commit()?;
        Ok(result)
    }
}

impl Drop for AtomicFileDelivery {
    fn drop(&mut self) {
        if !self.committed {
            crate::optional_failure::cleanup(
                "remove a partial video file",
                crate::optional_failure::remove_if_present(&self.partial_path),
            );
        }
    }
}
