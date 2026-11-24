//! Real encode pipelines over the engine-media backends.
//!
//! This is the machinery the 020-deferred `aiVideo` / `proxyVideo` / `aiAudio`
//! operators were waiting on: platform decode → CPU scale → platform encode →
//! the 021 pure-Rust mux, with the inline thumbnail tap the proxy pass emits
//! during its single decode. Codec sessions stay platform work; this module
//! owns sequencing, interleave, progress, and atomic file delivery.
//!
//! A single SDR output without thumbnails can bound its decode extent. Native
//! frames keep their backend's scaling path; CPU byte frames preserve their
//! channels through the shared encoder-input allocator. Secondary CPU scales
//! still use `CpuFrame::scaled_to`; color/precision conversion stays with its
//! existing owner.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferment_engine_core::{
    EngineError, MultiVideoEncodingResult, PixelFormat, PlaneLayout, ProgressSink, Result,
    ThumbStripOptions, ThumbnailFrame, ThumbnailSink, VariantThumbnailSink, VideoEncodingPreset,
    VideoEncodingResult, VideoEncodingVariant, VideoFileType, WorkPhase,
};
use ferment_engine_media::{
    AudioBackend, AudioEncoderSettings, AudioMuxTrack, AudioTiming, CpuFrame, DecodeOptions,
    EncodedAudio, EncodedPacket, EncoderSettings, MediaInfo, MuxBackend, MuxFormat, MuxSettings,
    StreamInfo, VideoBackend, VideoError, VideoMuxTrack,
};

use crate::video_encoder::{SourceVideo, VideoCodecSettings, VideoEncoder, VideoProfile};

/// The platform seams a real encode needs. Hosts construct these once (the
/// same backends the poster/preview paths already use) and hand them to the
/// encoder family.
#[derive(Clone)]
pub struct EncodeBackends {
    pub video: VideoBackend,
    pub audio: AudioBackend,
    pub muxer: MuxBackend,
}

impl std::fmt::Debug for EncodeBackends {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EncodeBackends")
            .finish_non_exhaustive()
    }
}

/// Platform codecs and the shared container muxer used by production exports.
pub fn platform_encode_backends() -> EncodeBackends {
    EncodeBackends {
        video: crate::video_backend(),
        audio: std::sync::Arc::new(ferment_engine_media::platform::libav::LibavAudioIo),
        muxer: std::sync::Arc::new(ferment_engine_media::platform::libav::LibavMuxer),
    }
}

/// AAC bitrate for muxed video variants and the standalone `aiAudio` m4a.
pub const AAC_BIT_RATE: u32 = 128_000;

/// Share of a split HDR run's progress owned by the shared decode. The SDR
/// derivatives re-encode a short proxy, so they are the cheap tail.
const HDR_SHARED_SPAN: f64 = 0.85;

pub(crate) fn media_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::Decode(error.to_string())
}

fn ensure_live(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

pub(crate) fn first_stream(
    info: &MediaInfo,
    kind: ferment_engine_media::MediaKind,
) -> Option<&StreamInfo> {
    info.streams.iter().find(|stream| stream.kind == kind)
}

const fn profile_name(profile: VideoProfile) -> &'static str {
    match profile {
        VideoProfile::H264HighAutoLevel => "high",
        VideoProfile::HevcMain10AutoLevel => "main10",
    }
}

fn thumbnail_frame(small: &CpuFrame, index: usize, presentation_time: f64) -> ThumbnailFrame {
    let bgra = small.to_bgra8();
    let stride = bgra.width as usize * 4;
    ThumbnailFrame {
        frame: ferment_engine_core::VideoFrame {
            data: bgra.data,
            planes: vec![PlaneLayout {
                offset: 0,
                stride,
                row_bytes: stride,
                rows: bgra.height as usize,
            }],
            width: bgra.width,
            height: bgra.height,
            format: PixelFormat::Bgra8,
            presentation_time: ferment_engine_core::MediaTime::default(),
        },
        index,
        presentation_time,
    }
}

/// Decode the source's first audio stream and re-encode it as AAC at the
/// source's rate/channel layout. Returns `None` when the source has no audio.
pub(crate) async fn encode_aac_lane(
    audio: &AudioBackend,
    source: &Path,
    info: &MediaInfo,
    progress: &ProgressSink,
) -> Result<Option<(EncodedAudio, u32, u16)>> {
    let Some(stream) = first_stream(info, ferment_engine_media::MediaKind::Audio) else {
        return Ok(None);
    };
    let sample_rate = stream.sample_rate.unwrap_or(44_100);
    let channels = stream.channels.unwrap_or(2).max(1);
    let mut decoder = audio
        .open_audio_decoder(source, stream.index)
        .await
        .map_err(media_error)?;
    let mut encoder = audio
        .open_audio_encoder(AudioEncoderSettings {
            bit_rate: AAC_BIT_RATE,
            sample_rate,
            channels,
        })
        .await
        .map_err(media_error)?;
    let processing = progress.enter_phase(WorkPhase::Processing);
    loop {
        ensure_live(progress)?;
        let Some(frame) = decoder.decode_next().await.map_err(media_error)? else {
            break;
        };
        ensure_live(progress)?;
        encoder.push_pcm(frame).await.map_err(media_error)?;
        progress.checkpoint();
    }
    drop(processing);
    let encoded = {
        let _phase = progress.enter_phase(WorkPhase::Finalizing);
        ensure_live(progress)?;
        let encoded = encoder.finalize().await.map_err(media_error)?;
        ensure_live(progress)?;
        progress.checkpoint();
        encoded
    };
    Ok(Some((encoded, sample_rate, channels)))
}

fn packet_order_key(packet: &EncodedPacket) -> f64 {
    packet
        .dts
        .or(packet.pts)
        .map_or(f64::MAX, ferment_engine_media::MediaTime::seconds)
}

/// Interleave two packet lanes by decode order for sane container layout.
pub(crate) fn interleaved_lanes(
    video: Vec<EncodedPacket>,
    audio: Vec<EncodedPacket>,
) -> Vec<(bool, EncodedPacket)> {
    let mut output = Vec::with_capacity(video.len() + audio.len());
    let mut video = video.into_iter().peekable();
    let mut audio = audio.into_iter().peekable();
    loop {
        match (video.peek(), audio.peek()) {
            (Some(next_video), Some(next_audio)) => {
                if packet_order_key(next_video) <= packet_order_key(next_audio) {
                    output.push((true, video.next().expect("peeked")));
                } else {
                    output.push((false, audio.next().expect("peeked")));
                }
            }
            (Some(_), None) => output.push((true, video.next().expect("peeked"))),
            (None, Some(_)) => output.push((false, audio.next().expect("peeked"))),
            (None, None) => break,
        }
    }
    output
}

/// The slice of the caller's 0..1 progress that one phase owns. The HDR split
/// runs two phases back to back, so each has to scale its own reports.
#[derive(Debug, Clone, Copy)]
struct ProgressWindow {
    base: f64,
    span: f64,
}

impl ProgressWindow {
    const FULL: Self = Self {
        base: 0.0,
        span: 1.0,
    };

    fn at(self, fraction: f64) -> f64 {
        self.base + fraction * self.span
    }
}

/// All branches finish in owned sibling files before any caller destination
/// changes. HDR derivatives read their lead's stage. Commit keeps old bytes in
/// owned backups until every rename and the final cancellation check succeeds.
#[derive(Default)]
struct CompletedVariantOutputs {
    files: Vec<StagedVariantOutput>,
    keep: bool,
}
struct StagedVariantOutput {
    final_path: PathBuf,
    staged_path: PathBuf,
    backup_path: Option<PathBuf>,
    installed: bool,
}
impl CompletedVariantOutputs {
    fn sibling(final_path: &Path, label: &str) -> Result<PathBuf> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        if let Some(parent) = final_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let name = final_path
            .file_name()
            .ok_or_else(|| EngineError::invalid_config("output path has no file name"))?;
        loop {
            let mut owned_name = std::ffi::OsString::from(".");
            owned_name.push(name);
            owned_name.push(format!(
                ".{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let path = final_path.with_file_name(owned_name);
            match std::fs::OpenOptions::new()
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
    fn stage(&mut self, final_path: &Path) -> Result<PathBuf> {
        let staged_path = Self::sibling(final_path, "encode-stage")?;
        self.files.push(StagedVariantOutput {
            final_path: final_path.to_owned(),
            staged_path: staged_path.clone(),
            backup_path: None,
            installed: false,
        });
        Ok(staged_path)
    }
    fn input_path(&self, final_path: &Path) -> PathBuf {
        self.files
            .iter()
            .rev()
            .find(|file| file.final_path == final_path)
            .map_or_else(|| final_path.to_owned(), |file| file.staged_path.clone())
    }
    fn backup(final_path: &Path, progress: &ProgressSink) -> Result<PathBuf> {
        use std::io::{Read, Write};
        let path = Self::sibling(final_path, "encode-backup")?;
        let copied: Result<()> = (|| {
            let mut input = std::fs::File::open(final_path)?;
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&path)?;
            let mut block = [0_u8; 64 * 1024];
            let processing = progress.enter_phase(WorkPhase::Processing);
            loop {
                ensure_live(progress)?;
                let count = input.read(&mut block)?;
                if count == 0 {
                    break;
                }
                output.write_all(&block[..count])?;
                progress.checkpoint();
            }
            drop(processing);
            output.flush()?;
            std::fs::set_permissions(&path, input.metadata()?.permissions())?;
            ensure_live(progress)
        })();
        if let Err(error) = copied {
            crate::optional_failure::cleanup(
                "remove a partial variant backup",
                crate::optional_failure::remove_if_present(&path),
            );
            return Err(error);
        }
        Ok(path)
    }
    fn commit(&mut self, progress: &ProgressSink) -> Result<()> {
        for file in &mut self.files {
            ensure_live(progress)?;
            if file.final_path.exists() {
                if !file.final_path.is_file() {
                    return Err(EngineError::invalid_config(
                        "output destination is not a regular file",
                    ));
                }
                // Copy through a bounded cancellable buffer: the old final
                // remains visible and byte-for-byte intact until the atomic
                // replacement, even while a backup is being prepared.
                file.backup_path = Some(Self::backup(&file.final_path, progress)?);
            }
            ensure_live(progress)?;
            {
                let _phase = progress.enter_phase(WorkPhase::Finalizing);
                ensure_live(progress)?;
                std::fs::rename(&file.staged_path, &file.final_path)?;
            }
            file.installed = true;
        }
        ensure_live(progress)?;
        self.keep = true;
        for file in &self.files {
            if let Some(backup) = &file.backup_path {
                crate::optional_failure::cleanup(
                    "remove a delivered variant backup",
                    crate::optional_failure::remove_if_present(backup),
                );
            }
        }
        Ok(())
    }
}
impl Drop for CompletedVariantOutputs {
    fn drop(&mut self) {
        if !self.keep {
            for file in self.files.iter().rev() {
                if let Some(backup) = &file.backup_path {
                    // If restoration itself fails, leave the owned backup
                    // intact. Its bytes must never be discarded on an error.
                    crate::optional_failure::cleanup(
                        "restore a half-delivered variant",
                        std::fs::rename(backup, &file.final_path),
                    );
                } else if file.installed {
                    crate::optional_failure::cleanup(
                        "sweep a half-delivered variant",
                        crate::optional_failure::remove_if_present(&file.final_path),
                    );
                }
                crate::optional_failure::cleanup(
                    "sweep a half-delivered variant",
                    crate::optional_failure::remove_if_present(&file.staged_path),
                );
            }
        }
    }
}

/// Current Swift makeThumbInfo: round the matrix's angle to a quarter-turn,
/// fit the DISPLAYED natural size to even cell dimensions, then rotate the
/// small picture. The output file separately retains all nine matrix words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThumbnailGeometry {
    width: u32,
    height: u32,
    clockwise_turns: u8,
}
impl ThumbnailGeometry {
    fn new(width: u32, height: u32, matrix: Option<[i32; 9]>, options: ThumbStripOptions) -> Self {
        let matrix = matrix.unwrap_or([65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824]);
        let degrees = f64::from(matrix[1])
            .atan2(f64::from(matrix[0]))
            .to_degrees()
            .round() as i32;
        let clockwise_turns = (((degrees.rem_euclid(360) + 45) / 90) % 4) as u8;
        let (display_width, display_height) = if clockwise_turns % 2 == 1 {
            (height, width)
        } else {
            (width, height)
        };
        let scale = (f64::from(options.max_width) / f64::from(display_width.max(1)))
            .min(f64::from(options.max_height) / f64::from(display_height.max(1)));
        Self {
            width: ((f64::from(display_width) * scale).round() as u32 & !1).max(2),
            height: ((f64::from(display_height) * scale).round() as u32 & !1).max(2),
            clockwise_turns,
        }
    }
    fn picture(self, frame: &ferment_engine_media::VideoFrame) -> Result<CpuFrame> {
        let (width, height) = if self.clockwise_turns % 2 == 1 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        };
        // Bound readback by the requested thumbnail. Do not download/rotate a
        // full-resolution movie picture merely to emit one small strip cell.
        let small = frame.scaled_cpu_rgb(width, height).map_err(media_error)?;
        if self.clockwise_turns == 0 {
            return Ok(small);
        }
        let picture = ferment_engine_media::VideoFrame {
            width,
            height,
            pts: frame.pts,
            duration: frame.duration,
            color: frame.color,
            buffer: ferment_engine_media::FrameBuffer::Cpu(small),
        }
        .rotated_clockwise(self.clockwise_turns)
        .map_err(media_error)?;
        picture.to_cpu_rgb().map_err(media_error)
    }
}

/// One variant's live state inside a shared decode.
struct VariantLane {
    id: String,
    output: PathBuf,
    preset: VideoEncodingPreset,
    settings: VideoCodecSettings,
    encoder: Box<dyn ferment_engine_media::EncodeSession>,
    thumb_strip: Option<ThumbStripOptions>,
    thumbnails_emitted: usize,
    next_thumbnail_at: f64,
    display_matrix: Option<[i32; 9]>,
    thumbnail_geometry: Option<ThumbnailGeometry>,
}

impl VariantLane {
    async fn open(
        backends: &EncodeBackends,
        variant: &VideoEncodingVariant,
        source: SourceVideo,
        time_base: ferment_engine_media::TimeBase,
        frame_rate: Option<(u32, u32)>,
        display_matrix: Option<[i32; 9]>,
        hdr_static_metadata: Option<ferment_engine_media::HdrStaticMetadata>,
    ) -> Result<Self> {
        let settings = VideoEncoder::plan(source, &variant.preset);
        let encoder = backends
            .video
            .open_encoder(EncoderSettings {
                codec: settings.codec.clone(),
                width: settings.width,
                height: settings.height,
                time_base,
                frame_rate,
                bit_rate: Some(settings.bit_rate),
                profile: Some(profile_name(settings.profile).to_owned()),
                max_keyframe_interval: settings.max_key_frame_interval,
                allow_frame_reordering: settings.allow_frame_reordering,
                color: Some(settings.color),
                insert_hdr_metadata: settings.insert_hdr_metadata,
                hdr_static_metadata: if settings.insert_hdr_metadata {
                    hdr_static_metadata
                } else {
                    None
                },
            })
            .await
            .map_err(media_error)?;
        Ok(Self {
            id: variant.id.clone(),
            output: variant.output_path.clone(),
            preset: variant.preset.clone(),
            settings,
            encoder,
            thumb_strip: variant.thumb_strip,
            thumbnails_emitted: 0,
            next_thumbnail_at: 0.0,
            display_matrix,
            thumbnail_geometry: variant.thumb_strip.map(|options| {
                ThumbnailGeometry::new(source.width, source.height, display_matrix, options)
            }),
        })
    }

    /// Tap one decoded frame for this lane's strip, if it wants one and the
    /// PTS has reached the next cell.
    fn tap_thumbnail(
        &mut self,
        frame: &ferment_engine_media::VideoFrame,
        pts_seconds: f64,
        sink: Option<&dyn VariantThumbnailSink>,
    ) -> Result<()> {
        let (Some(options), Some(sink)) = (self.thumb_strip, sink) else {
            return Ok(());
        };
        if pts_seconds + 1e-9 < self.next_thumbnail_at {
            return Ok(());
        }
        let geometry = self
            .thumbnail_geometry
            .ok_or_else(|| EngineError::internal("thumbnail geometry missing"))?;
        let small = geometry.picture(frame)?;
        sink.emit(
            &self.id,
            thumbnail_frame(&small, self.thumbnails_emitted, pts_seconds),
        );
        self.thumbnails_emitted += 1;
        self.next_thumbnail_at += options.interval.max(1e-3);
        Ok(())
    }

    async fn push(&mut self, frame: &ferment_engine_media::VideoFrame) -> Result<()> {
        // A native frame stays native: at size it is the decoder's own
        // buffer, otherwise the platform scales it in hardware. Only a
        // CPU-resident frame takes the download/resample/upload route.
        use ferment_engine_media::{CpuPixelFormat, FrameBuffer, VideoFrame};
        let encodable = match &frame.buffer {
            // Keep byte channels until the encoder's own conversion. The
            // shared allocator still creates native buffers where required.
            FrameBuffer::Cpu(cpu)
                if matches!(
                    cpu.format,
                    CpuPixelFormat::Rgb24 | CpuPixelFormat::Rgba8 | CpuPixelFormat::Bgra8
                ) && VideoEncoder::detect_hdr_transfer(frame.color).is_none()
                    && frame.color.bits_per_channel.is_none_or(|bits| bits <= 8) =>
            {
                VideoFrame::encodable_from_cpu(
                    cpu,
                    self.settings.width,
                    self.settings.height,
                    frame.pts,
                    frame.duration,
                    self.settings.color,
                )
            }
            _ => frame.encodable_scaled(
                self.settings.width,
                self.settings.height,
                self.settings.color,
            ),
        }
        .map_err(media_error)?;
        self.encoder
            .push_frame(encodable)
            .await
            .map_err(media_error)
    }

    /// Finalize the encoder and write this lane's container.
    async fn deliver(
        self,
        muxer: &MuxBackend,
        time_base: ferment_engine_media::TimeBase,
        audio_lane: Option<&(EncodedAudio, u32, u16)>,
        wall_seconds: f64,
        progress: &ProgressSink,
        completed: &mut CompletedVariantOutputs,
    ) -> Result<VideoEncodingResult> {
        ensure_live(progress)?;
        let encoded_video = {
            let _phase = progress.enter_phase(WorkPhase::Finalizing);
            let encoded = self.encoder.finalize().await.map_err(media_error)?;
            progress.checkpoint();
            encoded
        };
        ensure_live(progress)?;
        if encoded_video.packets.is_empty() {
            return Err(EngineError::Decode(
                "video encode produced no packets".into(),
            ));
        }
        let mux_format = match self.preset.file_type {
            VideoFileType::Mov => MuxFormat::Mov,
            VideoFileType::Mp4 => MuxFormat::Mp4,
        };
        let video_track = VideoMuxTrack {
            display_matrix: self.display_matrix,
            configuration: encoded_video.configuration.clone(),
            width: self.settings.width,
            height: self.settings.height,
            time_base,
            // The OUTPUT colorimetry, never the source's: an SDR variant of an
            // HDR source must not inherit BT.2020/HLG tags onto BT.709 pixels.
            color: self.settings.color,
            duration: None,
        };
        let muxer = muxer.clone();
        let staged_path = completed.stage(&self.output)?;
        crate::AtomicFileDelivery::write(&staged_path, |partial| {
            ensure_live(progress)?;
            let mut session = muxer
                .open(
                    partial,
                    MuxSettings {
                        format: mux_format,
                        ..MuxSettings::default()
                    },
                )
                .map_err(media_error)?;
            let video_id = session.add_video_track(video_track).map_err(media_error)?;
            let processing = progress.enter_phase(WorkPhase::Processing);
            match audio_lane {
                Some((encoded_audio, sample_rate, channels)) => {
                    let audio_id = session
                        .add_audio_track(AudioMuxTrack {
                            configuration: encoded_audio.configuration.clone(),
                            sample_rate: *sample_rate,
                            channels: *channels,
                            timing: encoded_audio.timing,
                        })
                        .map_err(media_error)?;
                    for (is_video, packet) in
                        interleaved_lanes(encoded_video.packets, encoded_audio.packets.clone())
                    {
                        ensure_live(progress)?;
                        session
                            .write_packet(if is_video { video_id } else { audio_id }, packet)
                            .map_err(media_error)?;
                        progress.checkpoint();
                    }
                }
                None => {
                    for packet in encoded_video.packets {
                        ensure_live(progress)?;
                        session
                            .write_packet(video_id, packet)
                            .map_err(media_error)?;
                        progress.checkpoint();
                    }
                }
            }
            drop(processing);
            {
                let _phase = progress.enter_phase(WorkPhase::Finalizing);
                session.finalize().map_err(media_error)?;
                progress.checkpoint();
            }
            // Finalization may complete after cancellation. Never promote its
            // now-complete partial when the caller has abandoned the export.
            ensure_live(progress)
        })?;
        Ok(VideoEncodingResult {
            bytes: std::fs::metadata(&staged_path)?.len(),
            output_path: self.output,
            width: self.settings.width,
            height: self.settings.height,
            preset: self.preset,
            wall_seconds,
            thumbnails_emitted: self.thumbnails_emitted,
        })
    }
}

/// One SDR output can resize during decode instead of materializing the full
/// source on the CPU. Independent variants/thumbnails retain the original
/// extent to avoid repeated resizing; HDR/precision keeps its color owner.
fn decode_output_size(source: SourceVideo, lanes: &[VariantLane]) -> Option<(u32, u32)> {
    let [lane] = lanes else { return None };
    if lane.thumbnail_geometry.is_some()
        || VideoEncoder::detect_hdr_transfer(source.color).is_some()
        || source.color.bits_per_channel.is_some_and(|bits| bits > 8)
    {
        return None;
    }
    let size = (
        lane.settings.width.min(source.width),
        lane.settings.height.min(source.height),
    );
    (size.0 > 0 && size.1 > 0 && size != (source.width, source.height)).then_some(size)
}

/// One decode of `source` feeding every lane in `variants`.
///
/// Independent variants and thumbnails scale from the full-resolution frame;
/// a single output may instead resize once in the decoder. This base case
/// does no HDR splitting, so one SDR output of HDR is normalized by its reader.
async fn shared_decode_encode(
    backends: &EncodeBackends,
    source: &Path,
    variants: &[VideoEncodingVariant],
    progress: &ProgressSink,
    window: ProgressWindow,
    on_thumbnail: Option<&dyn VariantThumbnailSink>,
    completed: &mut CompletedVariantOutputs,
) -> Result<(BTreeMap<String, VideoEncodingResult>, usize)> {
    let started = std::time::Instant::now();
    let info = backends.video.probe(source).await.map_err(media_error)?;
    let stream = first_stream(&info, ferment_engine_media::MediaKind::Video)
        .ok_or_else(|| EngineError::Decode("source has no video stream".into()))?
        .clone();
    let source_video = SourceVideo::from_stream(&stream)?;
    let duration_seconds = stream
        .duration
        .or(info.duration)
        .map(ferment_engine_media::MediaTime::seconds)
        .filter(|seconds| *seconds > 0.0);

    let hdr_static_metadata = if VideoEncoder::detect_hdr_transfer(stream.color).is_some() {
        backends
            .video
            .hdr_static_metadata(source, stream.index)
            .await
            .map_err(media_error)?
    } else {
        None
    };
    let mut lanes = Vec::with_capacity(variants.len());
    for variant in variants {
        ensure_live(progress)?;
        lanes.push(
            VariantLane::open(
                backends,
                variant,
                source_video,
                stream.time_base,
                stream.frame_rate,
                stream.display_matrix,
                hdr_static_metadata,
            )
            .await?,
        );
    }

    // transcode_variants_inner already splits HDR lanes from SDR derivatives.
    // A tone-map request must never affect the decoder feeding the HDR proxy.
    let target_color = lanes
        .first()
        .map(|lane| lane.settings.color)
        .unwrap_or(stream.color);
    if lanes.iter().any(|lane| lane.settings.color != target_color) {
        return Err(EngineError::invalid_config(
            "one decode pass cannot mix HDR and SDR color targets",
        ));
    }
    let mut decoder = if let Some(size) = decode_output_size(source_video, &lanes) {
        match backends
            .video
            .open_coded_decoder_with(
                source,
                stream.index,
                DecodeOptions {
                    output_size: Some(size),
                    ..DecodeOptions::default()
                },
            )
            .await
        {
            Err(VideoError::BackendUnavailable(error)) => {
                tracing::debug!(%error, "Decoder resize unavailable; preparing original frames");
                backends
                    .video
                    .open_coded_decoder_for_color(source, stream.index, target_color)
                    .await
            }
            result => result,
        }
    } else {
        backends
            .video
            .open_coded_decoder_for_color(source, stream.index, target_color)
            .await
    }
    .map_err(media_error)?;
    let mut decoded_frames = 0usize;
    let processing = progress.enter_phase(WorkPhase::Processing);
    while let Some(frame) = decoder.decode_next().await.map_err(media_error)? {
        ensure_live(progress)?;
        let pts_seconds = frame.pts.seconds();
        for lane in &mut lanes {
            lane.tap_thumbnail(&frame, pts_seconds, on_thumbnail)?;
            lane.push(&frame).await?;
        }
        decoded_frames += 1;
        progress.checkpoint();
        if let Some(total) = duration_seconds {
            progress
                .report(
                    window.at((pts_seconds / total).clamp(0.0, 0.85)),
                    "Processing video",
                )
                .await;
        }
    }
    drop(decoder);
    drop(processing);
    ensure_live(progress)?;

    let audio_lane = encode_aac_lane(&backends.audio, source, &info, progress).await?;
    progress.report(window.at(0.92), "Writing container").await;

    let wall_seconds = started.elapsed().as_secs_f64();
    let mut results = BTreeMap::new();
    for lane in lanes {
        let id = lane.id.clone();
        let result = lane
            .deliver(
                &backends.muxer,
                stream.time_base,
                audio_lane.as_ref(),
                wall_seconds,
                progress,
                completed,
            )
            .await?;
        results.insert(id, result);
    }
    Ok((results, decoded_frames))
}

// A result ID owns one destination. Resolve lexical aliases before any
// source probe, codec construction, staging or rename; no destination needs to
// exist for this validation. Filesystem/symlink identity is not guessed.
fn validate_variant_output_paths(variants: &[VideoEncodingVariant]) -> Result<()> {
    let directory = std::env::current_dir()?;
    let mut seen = std::collections::HashSet::new();
    for variant in variants {
        let absolute = if variant.output_path.is_absolute() {
            variant.output_path.clone()
        } else {
            directory.join(&variant.output_path)
        };
        let mut normalized = PathBuf::new();
        for component in absolute.components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    normalized.pop();
                }
                other => normalized.push(other.as_os_str()),
            }
        }
        if !seen.insert(normalized) {
            return Err(EngineError::invalid_config(
                "duplicate video output destination",
            ));
        }
    }
    Ok(())
}

/// Every derivative of one source, from one decode where the colorimetry
/// allows it.
///
/// HDR split: an HLG/PQ source shares its decode only among variants that
/// PRESERVE it. SDR-only variants are derived AFTERWARDS from the largest
/// finished HDR output — the expensive source decode runs once and the
/// normalize rides the small proxy instead of a second full-source pass.
/// Second-generation quality is the accepted price. With no HDR variant in
/// the request there is no lead to borrow from, so those variants decode the
/// source directly and are normalized at the reader.
pub(crate) async fn transcode_variants(
    backends: &EncodeBackends,
    source: &Path,
    variants: &[VideoEncodingVariant],
    progress: &ProgressSink,
    on_thumbnail: Option<&dyn VariantThumbnailSink>,
) -> Result<MultiVideoEncodingResult> {
    validate_variant_output_paths(variants)?;
    ensure_live(progress)?;
    let mut completed = CompletedVariantOutputs::default();
    let result = transcode_variants_inner(
        backends,
        source,
        variants,
        progress,
        on_thumbnail,
        &mut completed,
    )
    .await?;
    ensure_live(progress)?;
    completed.commit(progress)?;
    Ok(result)
}

async fn transcode_variants_inner(
    backends: &EncodeBackends,
    source: &Path,
    variants: &[VideoEncodingVariant],
    progress: &ProgressSink,
    on_thumbnail: Option<&dyn VariantThumbnailSink>,
    completed: &mut CompletedVariantOutputs,
) -> Result<MultiVideoEncodingResult> {
    let started = std::time::Instant::now();
    let info = backends.video.probe(source).await.map_err(media_error)?;
    let source_is_hdr = first_stream(&info, ferment_engine_media::MediaKind::Video)
        .is_some_and(|stream| VideoEncoder::detect_hdr_transfer(stream.color).is_some());
    let needs_split = source_is_hdr && variants.iter().any(|variant| !variant.preset.hdr_eligible);
    if !needs_split {
        let (variants, decoded_frames) = shared_decode_encode(
            backends,
            source,
            variants,
            progress,
            ProgressWindow::FULL,
            on_thumbnail,
            completed,
        )
        .await?;
        return Ok(MultiVideoEncodingResult {
            variants,
            wall_seconds: started.elapsed().as_secs_f64(),
            decoded_frames,
        });
    }

    let hdr: Vec<VideoEncodingVariant> = variants
        .iter()
        .filter(|variant| variant.preset.hdr_eligible)
        .cloned()
        .collect();
    let sdr: Vec<VideoEncodingVariant> = variants
        .iter()
        .filter(|variant| !variant.preset.hdr_eligible)
        .cloned()
        .collect();
    let shared_span = if hdr.is_empty() { 0.0 } else { HDR_SHARED_SPAN };

    let mut results = BTreeMap::new();
    let mut decoded_frames = 0;
    if !hdr.is_empty() {
        let (shared, decoded) = shared_decode_encode(
            backends,
            source,
            &hdr,
            progress,
            ProgressWindow {
                base: 0.0,
                span: shared_span,
            },
            on_thumbnail,
            completed,
        )
        .await?;
        results = shared;
        decoded_frames = decoded;
    }

    // The largest finished HDR output feeds the derivatives.
    let derivative_source: PathBuf = hdr
        .iter()
        .max_by_key(|variant| variant.preset.max_width)
        .and_then(|variant| results.get(&variant.id))
        .map_or_else(
            || source.to_path_buf(),
            |result| completed.input_path(&result.output_path),
        );

    let slice = (1.0 - shared_span) / sdr.len() as f64;
    for (offset, variant) in sdr.iter().enumerate() {
        let (derived, _) = shared_decode_encode(
            backends,
            &derivative_source,
            std::slice::from_ref(variant),
            progress,
            ProgressWindow {
                base: shared_span + slice * offset as f64,
                span: slice,
            },
            on_thumbnail,
            completed,
        )
        .await?;
        results.extend(derived);
    }

    Ok(MultiVideoEncodingResult {
        variants: results,
        wall_seconds: started.elapsed().as_secs_f64(),
        decoded_frames,
    })
}

/// The single-output pass: one variant through the shared decode, with the
/// caller's unscoped thumbnail sink adapted to it.
pub(crate) async fn transcode_video(
    backends: &EncodeBackends,
    source: &Path,
    output: &Path,
    preset: &VideoEncodingPreset,
    thumb_strip: Option<ThumbStripOptions>,
    progress: &ProgressSink,
    on_thumbnail: Option<&dyn ThumbnailSink>,
) -> Result<VideoEncodingResult> {
    let variant = VideoEncodingVariant {
        id: preset.label.clone(),
        output_path: output.to_path_buf(),
        preset: preset.clone(),
        thumb_strip,
    };
    let adapter = on_thumbnail.map(SingleVariantSink);
    ensure_live(progress)?;
    let mut completed = CompletedVariantOutputs::default();
    let (mut results, _) = shared_decode_encode(
        backends,
        source,
        std::slice::from_ref(&variant),
        progress,
        ProgressWindow::FULL,
        adapter
            .as_ref()
            .map(|sink| sink as &dyn VariantThumbnailSink),
        &mut completed,
    )
    .await?;
    ensure_live(progress)?;
    completed.commit(progress)?;
    Ok(results
        .remove(&variant.id)
        .expect("the single requested variant is always delivered"))
}

/// Drops the variant id for the single-output `VideoEncoding::encode` path,
/// where there is only ever one emitter.
struct SingleVariantSink<'a>(&'a dyn ThumbnailSink);

impl std::fmt::Debug for SingleVariantSink<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SingleVariantSink")
    }
}

impl VariantThumbnailSink for SingleVariantSink<'_> {
    fn emit(&self, _variant_id: &str, thumbnail: ThumbnailFrame) {
        self.0.emit(thumbnail);
    }
}

/// The standalone `aiAudio` pass: AAC-in-m4a from the source's audio stream.
pub(crate) async fn transcode_audio_only(
    audio: &AudioBackend,
    muxer: &MuxBackend,
    source: &Path,
    output: &Path,
    progress: &ProgressSink,
) -> Result<(u64, f64)> {
    ensure_live(progress)?;
    let info = audio.probe(source).await.map_err(media_error)?;
    let duration = info
        .audio_streams()
        .next()
        .and_then(|stream| stream.duration)
        .or(info.duration)
        .map_or(0.0, ferment_engine_media::MediaTime::seconds);
    let Some((encoded, sample_rate, channels)) =
        encode_aac_lane(audio, source, &info, progress).await?
    else {
        return Err(EngineError::NoAudioData);
    };
    if encoded.packets.is_empty() {
        return Err(EngineError::Decode(
            "audio encode produced no packets".into(),
        ));
    }
    progress.report(0.92, "Writing container").await;
    let timing = encoded.timing;
    let muxer = muxer.clone();
    crate::AtomicFileDelivery::write(output, |partial| {
        ensure_live(progress)?;
        let mut session = muxer
            .open(partial, MuxSettings::default())
            .map_err(media_error)?;
        let track = session
            .add_audio_track(AudioMuxTrack {
                configuration: encoded.configuration.clone(),
                sample_rate,
                channels,
                timing,
            })
            .map_err(media_error)?;
        let processing = progress.enter_phase(WorkPhase::Processing);
        for packet in encoded.packets.clone() {
            ensure_live(progress)?;
            session.write_packet(track, packet).map_err(media_error)?;
            progress.checkpoint();
        }
        drop(processing);
        {
            let _phase = progress.enter_phase(WorkPhase::Finalizing);
            ensure_live(progress)?;
            session.finalize().map_err(media_error)?;
            progress.checkpoint();
        }
        ensure_live(progress)
    })?;
    let bytes = std::fs::metadata(output)?.len();
    Ok((bytes, duration))
}

// `AudioTiming` is carried through unchanged: the platform AAC encoder
// reports priming/padding and the 021 mux writes the matching edit list —
// the gapless invariant the 010/021 work pinned.
const _: fn(AudioTiming) = |_| {};

#[cfg(test)]
mod tests {
    use ferment_engine_media::{MediaTime, Packetization, TimeBase};

    use super::*;

    fn packet(seconds: f64, sync: bool) -> EncodedPacket {
        EncodedPacket {
            stream_index: 0,
            pts: Some(MediaTime::new(
                (seconds * 600.0) as i64,
                TimeBase::new_unchecked(1, 600),
            )),
            dts: None,
            duration: None,
            is_sync: sync,
            data: Vec::new(),
            packetization: Packetization::Elementary,
        }
    }

    #[test]
    fn interleave_orders_lanes_by_decode_time() {
        let video = vec![packet(0.0, true), packet(1.0, false), packet(2.0, false)];
        let audio = vec![packet(0.5, true), packet(1.5, true)];
        let order: Vec<(bool, f64)> = interleaved_lanes(video, audio)
            .into_iter()
            .map(|(is_video, packet)| (is_video, packet.pts.unwrap().seconds()))
            .collect();
        assert_eq!(
            order,
            vec![
                (true, 0.0),
                (false, 0.5),
                (true, 1.0),
                (false, 1.5),
                (true, 2.0)
            ]
        );
    }

    #[test]
    fn thumbnail_frames_are_tightly_packed_bgra() {
        let small = CpuFrame::new(
            2,
            1,
            6,
            ferment_engine_media::CpuPixelFormat::Rgb24,
            vec![255, 0, 0, 0, 255, 0],
        )
        .unwrap();
        let thumb = thumbnail_frame(&small, 3, 1.5);
        assert_eq!(thumb.index, 3);
        assert_eq!(thumb.presentation_time, 1.5);
        assert_eq!(thumb.frame.format, PixelFormat::Bgra8);
        assert_eq!(&thumb.frame.data[..], &[0, 0, 255, 255, 0, 255, 0, 255]);
    }
}

#[cfg(test)]
#[path = "encode_pipeline_transform_tests.rs"]
mod transform_tests;

#[cfg(test)]
#[path = "encode_pipeline_decode_tests.rs"]
mod decode_tests;
