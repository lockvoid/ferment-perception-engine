//! Swift VideoUpscaler / ImageUpscaler: source pixels → Compact SR → target
//! resample → platform encode and an atomic file commit.

use crate::encode_pipeline::{interleaved_lanes, media_error};
use crate::video_encoder::AtomicFileDelivery;
use crate::video_upscaler::{CompactFrameUpscaler, Dimensions, FrameUpscaling, UpscaleMode};
use ferment_engine_core::{self as core, EngineError, ProgressSink, Result};
use ferment_engine_infer::Backend;
use ferment_engine_media::{
    Codec, ColorInfo, CpuFrame, CpuPixelFormat, EncoderSettings, FrameBuffer, MediaTime,
    MuxSettings, TimeBase, VideoFrame, VideoMuxTrack,
};
use std::{path::Path, time::Instant};

fn live(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn time(value: core::MediaTime) -> Result<MediaTime> {
    Ok(MediaTime::new(
        value.value,
        TimeBase::new(1, i64::from(value.timescale)).map_err(media_error)?,
    ))
}

fn resized(frame: &VideoFrame, size: Dimensions) -> Result<VideoFrame> {
    if frame.width == size.width && frame.height == size.height {
        return Ok(frame.clone());
    }
    let rgb = frame.to_cpu_rgb().map_err(media_error)?;
    let packed = rgb.to_rgb24();
    let image = image::RgbImage::from_raw(packed.width, packed.height, packed.data.to_vec())
        .ok_or(EngineError::InvalidOutput)?;
    let resized = image::imageops::resize(
        &image,
        size.width,
        size.height,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(VideoFrame {
        width: size.width,
        height: size.height,
        pts: frame.pts,
        duration: frame.duration,
        color: frame.color,
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(
                size.width,
                size.height,
                size.width as usize * 3,
                CpuPixelFormat::Rgb24,
                resized.into_raw(),
            )
            .map_err(media_error)?,
        ),
    })
}

fn oriented(frame: VideoFrame, matrix: Option<[i32; 9]>) -> Result<VideoFrame> {
    let rotation = crate::video_saliency::rotation_degrees(matrix);
    if rotation == 0 {
        return Ok(frame);
    }
    frame
        .rotated_clockwise((rotation / 90) as u8)
        .map_err(media_error)
}

fn declared_color(color: ColorInfo) -> Option<ColorInfo> {
    use ferment_engine_media::{ColorMatrix, ColorPrimaries, TransferCharacteristic};
    (color.primaries != ColorPrimaries::Unspecified
        && color.matrix != ColorMatrix::Unspecified
        && color.transfer != TransferCharacteristic::Unspecified)
        .then_some(color)
}

fn checked_plan(
    mode: UpscaleMode,
    width: u32,
    height: u32,
) -> Result<crate::video_upscaler::UpscalePlan> {
    checked_plan_for(mode, width, height, 4, Some(Dimensions::new(1440, 1080)))
}

fn checked_plan_for(
    mode: UpscaleMode,
    width: u32,
    height: u32,
    scale_factor: u32,
    input_cap: Option<Dimensions>,
) -> Result<crate::video_upscaler::UpscalePlan> {
    if scale_factor == 0 {
        return Err(EngineError::invalid_config(
            "upscale factor must be positive",
        ));
    }
    let plan = mode.plan(Dimensions::new(width, height), scale_factor, input_cap);
    if plan.input_size.width < 2
        || plan.input_size.height < 2
        || plan.output_size.width < 2
        || plan.output_size.height < 2
    {
        return Err(EngineError::invalid_config(
            "super-resolution target must have two positive even dimensions",
        ));
    }
    Ok(plan)
}

pub async fn upscale_image(
    backend: Backend,
    source: &Path,
    output: &Path,
    mode: UpscaleMode,
    progress: ProgressSink,
) -> Result<Dimensions> {
    live(&progress)?;
    let mut reader = image::ImageReader::open(source)?.with_guessed_format()?;
    reader.limits(image::Limits::default());
    let mut decoder = reader.into_decoder().map_err(media_error)?;
    use image::ImageDecoder;
    let orientation = decoder.orientation().map_err(media_error)?;
    let mut decoded = image::DynamicImage::from_decoder(decoder).map_err(media_error)?;
    decoded.apply_orientation(orientation);
    let rgb = decoded.into_rgb8();
    let plan = checked_plan(mode, rgb.width(), rgb.height())?;
    let source_frame = VideoFrame {
        width: rgb.width(),
        height: rgb.height(),
        pts: MediaTime::ZERO,
        duration: None,
        color: ColorInfo::default(),
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(
                rgb.width(),
                rgb.height(),
                rgb.width() as usize * 3,
                CpuPixelFormat::Rgb24,
                rgb.into_raw(),
            )
            .map_err(media_error)?,
        ),
    };
    let input = resized(&source_frame, plan.input_size)?;
    let mut upscaler = CompactFrameUpscaler::general(backend).with_progress(progress.clone());
    upscaler.start_session(plan.input_size, false)?;
    let result = upscaler.upscale(&input).await;
    upscaler.end_session();
    let scaled = VideoFrame {
        width: plan.ml_size().width,
        height: plan.ml_size().height,
        buffer: result?,
        ..input
    };
    live(&progress)?;
    let final_frame = resized(&scaled, plan.output_size)?
        .to_cpu_rgb()
        .map_err(media_error)?;
    let delivery = AtomicFileDelivery::begin(output)?;
    let mut file = std::fs::File::create(delivery.partial_path())?;
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, 95)
        .encode(
            &final_frame.data,
            final_frame.width,
            final_frame.height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(media_error)?;
    file.sync_all()?;
    live(&progress)?;
    delivery.commit()?;
    progress.report(1.0, "Complete").await;
    Ok(plan.output_size)
}

pub async fn upscale_video(
    backend: Backend,
    source: &Path,
    output: &Path,
    mode: UpscaleMode,
    window: Option<core::MediaTimeRange>,
    _scenes: &[core::MediaTime],
    progress: ProgressSink,
) -> Result<core::VideoRenderResult> {
    upscale_video_with_backends(
        backend,
        crate::platform_encode_backends(),
        source,
        output,
        mode,
        window,
        progress,
    )
    .await
}

// One file-render path; tests inject only the mux failure at its real backend boundary.
async fn upscale_video_with_backends(
    backend: Backend,
    backends: crate::EncodeBackends,
    source: &Path,
    output: &Path,
    mode: UpscaleMode,
    window: Option<core::MediaTimeRange>,
    progress: ProgressSink,
) -> Result<core::VideoRenderResult> {
    let mut upscaler = CompactFrameUpscaler::general(backend).with_progress(progress.clone());
    upscale_video_with_processor(
        backends,
        source,
        output,
        mode,
        window,
        progress,
        &mut upscaler,
        Some(Dimensions::new(1440, 1080)),
    )
    .await
}

// The current Swift tests inject only the same frame-upscaler seam. File decode,
// pre/post resampling, encode, soundtrack and atomic delivery stay in production.
#[allow(clippy::too_many_arguments)]
async fn upscale_video_with_processor(
    backends: crate::EncodeBackends,
    source: &Path,
    output: &Path,
    mode: UpscaleMode,
    window: Option<core::MediaTimeRange>,
    progress: ProgressSink,
    upscaler: &mut dyn FrameUpscaling,
    input_cap: Option<Dimensions>,
) -> Result<core::VideoRenderResult> {
    let started = Instant::now();
    live(&progress)?;
    let info = backends.video.probe(source).await.map_err(media_error)?;
    let stream = info
        .video_streams()
        .next()
        .ok_or_else(|| EngineError::Decode("source has no video track".into()))?;
    let duration = window
        .map(|w| time(w.duration))
        .transpose()?
        .or(stream.duration)
        .or(info.duration)
        .ok_or_else(|| EngineError::Decode("video duration is missing".into()))?;
    let start = window
        .map(|w| time(w.start))
        .transpose()?
        .unwrap_or(MediaTime::ZERO);
    if duration.value <= 0 || start.value < 0 {
        return Err(EngineError::invalid_config(
            "video-upscale window must be positive",
        ));
    }
    let (width, height) = crate::video_saliency::display_size(
        stream.width.unwrap_or(0),
        stream.height.unwrap_or(0),
        stream.display_matrix,
    );
    let plan = checked_plan_for(
        mode,
        width,
        height,
        upscaler.native_scale_factor(),
        input_cap,
    )?;
    let mut decoder = backends
        .video
        .open_coded_decoder(source, stream.index)
        .await
        .map_err(media_error)?;
    if start.value > 0 {
        decoder.seek(start).await.map_err(media_error)?;
    }
    let mut settings = EncoderSettings::new(
        Codec::Hevc,
        plan.output_size.width,
        plan.output_size.height,
        stream.time_base,
    );
    settings.frame_rate = stream.frame_rate;
    settings.allow_frame_reordering = false;
    settings.bit_rate = Some(crate::video_upscaler::scaled_bitrate(
        stream.bit_rate.unwrap_or(8_000_000),
        u64::from(stream.width.unwrap_or(0)) * u64::from(stream.height.unwrap_or(0)),
        plan.output_size.area(),
    ));
    settings.color = declared_color(stream.color);
    let mut encoder = backends
        .video
        .open_encoder(settings)
        .await
        .map_err(media_error)?;
    upscaler.start_session(plan.input_size, false)?;
    let mut frames = 0;
    let mut last_pts = 0.0;
    let processing = async {
        while let Some(mut frame) = decoder.decode_next().await.map_err(media_error)? {
            live(&progress)?;
            if frame.pts.seconds() < start.seconds() {
                continue;
            }
            if frame.pts.seconds() >= start.seconds() + duration.seconds() {
                break;
            }
            frame.pts.value -= start.rescale(frame.pts.time_base).value;
            let frame = oriented(frame, stream.display_matrix)?;
            let input = resized(&frame, plan.input_size)?;
            let buffer = upscaler.upscale(&input).await?;
            let scaled = VideoFrame {
                width: plan.ml_size().width,
                height: plan.ml_size().height,
                buffer,
                ..input
            };
            let delivered = resized(&scaled, plan.output_size)?;
            last_pts = delivered.pts.seconds();
            encoder
                .push_frame(
                    delivered
                        .encodable_scaled(delivered.width, delivered.height, delivered.color)
                        .map_err(media_error)?,
                )
                .await
                .map_err(media_error)?;
            frames += 1;
            progress
                .report(
                    (last_pts / duration.seconds()).min(0.95),
                    "Super resolution",
                )
                .await;
        }
        if frames == 0 {
            return Err(EngineError::Decode(
                "video window produced no frames".into(),
            ));
        }
        Ok(())
    }
    .await;
    upscaler.end_session();
    processing?;
    live(&progress)?;
    let encoded = encoder.finalize().await.map_err(media_error)?;
    let video_track = VideoMuxTrack {
        display_matrix: None,
        configuration: encoded.configuration.clone(),
        width: plan.output_size.width,
        height: plan.output_size.height,
        time_base: stream.time_base,
        color: stream.color,
        duration: Some(duration),
    };
    let audio_preserved = finish_upscaled_picture(
        &backends,
        &info,
        encoded,
        video_track,
        start,
        duration,
        FileRenderContext {
            source,
            output,
            progress: &progress,
        },
    )
    .await?;
    progress.report(1.0, "Complete").await;
    Ok(core::VideoRenderResult {
        output_path: output.to_owned(),
        frames_written: frames,
        output_seconds: last_pts,
        wall_seconds: started.elapsed().as_secs_f64(),
        output_width: Some(plan.output_size.width),
        output_height: Some(plan.output_size.height),
        audio_preserved,
        motion_blur: None,
    })
}

/// Complete the picture container first. Soundtrack preparation/export may fail
/// after that point without destroying a valid render, matching VideoAudioMuxer.
/// Only one packet is cloned at a time while writing the silent safety copy;
/// the audio attempt consumes the original packets, not a second movie-sized Vec.
struct FileRenderContext<'a> {
    source: &'a Path,
    output: &'a Path,
    progress: &'a ProgressSink,
}

async fn finish_upscaled_picture(
    backends: &crate::EncodeBackends,
    info: &ferment_engine_media::MediaInfo,
    encoded: ferment_engine_media::EncodedVideo,
    video_track: VideoMuxTrack,
    start: MediaTime,
    duration: MediaTime,
    context: FileRenderContext<'_>,
) -> Result<bool> {
    let FileRenderContext {
        source,
        output,
        progress,
    } = context;
    live(progress)?;
    let delivery = AtomicFileDelivery::begin(output)?;
    let mut picture = backends
        .muxer
        .open(delivery.partial_path(), MuxSettings::default())
        .map_err(media_error)?;
    let video_id = picture
        .add_video_track(video_track.clone())
        .map_err(media_error)?;
    for packet in &encoded.packets {
        live(progress)?;
        picture
            .write_packet(video_id, packet.clone())
            .map_err(media_error)?;
    }
    picture.finalize().map_err(media_error)?;
    live(progress)?;
    if info.audio_streams().next().is_none() {
        delivery.commit()?;
        return Ok(true);
    }

    let audio_attempt: Result<Option<AtomicFileDelivery>> = async {
        // Match the current Swift composition: the source window cannot
        // outlive the completed picture. Probe its real container duration.
        let picture_info = backends
            .video
            .probe(delivery.partial_path())
            .await
            .map_err(media_error)?;
        let picture_duration = picture_info
            .video_streams()
            .next()
            .and_then(|stream| stream.duration)
            .or(picture_info.duration)
            .ok_or_else(|| EngineError::Decode("finished picture duration is missing".into()))?;
        let duration = if picture_duration.seconds() < duration.seconds() {
            picture_duration
        } else {
            duration
        };
        let checked = || {
            if progress.is_closed() {
                Err(ferment_engine_media::VideoError::InvalidData(
                    "audio passthrough cancelled".into(),
                ))
            } else {
                Ok(())
            }
        };
        let mut readers = Vec::new();
        for stream in info.audio_streams() {
            live(progress)?;
            let Some(mut reader) =
                ferment_engine_media::AudioPassthroughReader::open(source, stream, start, duration)
                    .map_err(media_error)?
            else {
                continue;
            };
            let first = reader
                .next_packet_checked(checked)
                .map_err(media_error)?
                .ok_or_else(|| {
                    EngineError::Decode("overlapping source soundtrack has no packets".into())
                })?;
            readers.push((reader, Some(first)));
        }
        if readers.is_empty() {
            // Audio tracks entirely outside the selected picture are omitted.
            return Ok(None);
        }
        live(progress)?;
        let audio_destination = delivery.partial_path().with_extension("audio.mp4");
        let scratch = AtomicFileDelivery::begin(&audio_destination)?;
        let mut mux = backends
            .muxer
            .open(scratch.partial_path(), MuxSettings::default())
            .map_err(media_error)?;
        let video_id = mux.add_video_track(video_track).map_err(media_error)?;
        let mut ids = Vec::with_capacity(readers.len());
        for (reader, _) in &readers {
            ids.push(
                mux.add_passthrough_audio_track(reader.track())
                    .map_err(media_error)?,
            );
        }
        // Only one pending packet per soundtrack. Interleaving at this door
        // also prevents the native muxer from buffering an entire audio lane.
        let mut pictures = encoded.packets.into_iter().peekable();
        loop {
            live(progress)?;
            let audio_next = readers
                .iter()
                .enumerate()
                .filter_map(|(index, (_, packet))| {
                    packet.as_ref().map(|packet| {
                        (
                            index,
                            packet
                                .dts()
                                .or(packet.pts())
                                .unwrap_or(MediaTime::ZERO)
                                .seconds(),
                        )
                    })
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let video_time = pictures.peek().map(|packet| {
                packet
                    .dts
                    .or(packet.pts)
                    .unwrap_or(MediaTime::ZERO)
                    .seconds()
            });
            match (video_time, audio_next) {
                (Some(video), Some((_, audio))) if video <= audio => {
                    mux.write_packet(video_id, pictures.next().unwrap())
                        .map_err(media_error)?;
                }
                (_, Some((index, _))) => {
                    let (reader, packet) = &mut readers[index];
                    mux.write_passthrough_audio_packet(ids[index], packet.take().unwrap())
                        .map_err(media_error)?;
                    *packet = reader.next_packet_checked(checked).map_err(media_error)?;
                }
                (Some(_), None) => mux
                    .write_packet(video_id, pictures.next().unwrap())
                    .map_err(media_error)?,
                (None, None) => break,
            }
        }
        mux.finalize().map_err(media_error)?;
        Ok(Some(scratch))
    }
    .await;

    let audio_preserved = match audio_attempt {
        Ok(Some(scratch)) => {
            live(progress)?;
            // This is delivery, not a soundtrack failure: propagate rename errors.
            std::fs::rename(scratch.partial_path(), delivery.partial_path())?;
            true
        }
        Ok(None) => true,
        Err(EngineError::Cancelled) => return Err(EngineError::Cancelled),
        Err(_) => {
            live(progress)?;
            false
        }
    };
    live(progress)?;
    delivery.commit()?;
    Ok(audio_preserved)
}

struct SourceWindow {
    backends: crate::EncodeBackends,
    info: ferment_engine_media::MediaInfo,
    stream: ferment_engine_media::StreamInfo,
    decoder: Box<dyn ferment_engine_media::DecodeSession>,
    matrix: Option<[i32; 9]>,
    start: MediaTime,
    duration: MediaTime,
}

impl SourceWindow {
    async fn open(source: &Path, window: Option<core::MediaTimeRange>) -> Result<Self> {
        Self::open_with_backends(crate::platform_encode_backends(), source, window).await
    }

    async fn open_with_backends(
        backends: crate::EncodeBackends,
        source: &Path,
        window: Option<core::MediaTimeRange>,
    ) -> Result<Self> {
        let info = backends.video.probe(source).await.map_err(media_error)?;
        let mut stream = info
            .video_streams()
            .next()
            .cloned()
            .ok_or_else(|| EngineError::Decode("source has no video track".into()))?;
        let start = window
            .map(|w| time(w.start))
            .transpose()?
            .unwrap_or(MediaTime::ZERO);
        let duration = window
            .map(|w| time(w.duration))
            .transpose()?
            .or(stream.duration)
            .or(info.duration)
            .ok_or_else(|| EngineError::Decode("video duration is missing".into()))?;
        if duration.value <= 0 || start.value < 0 {
            return Err(EngineError::invalid_config(
                "render window must be positive",
            ));
        }
        let mut decoder = backends
            .video
            .open_coded_decoder(source, stream.index)
            .await
            .map_err(media_error)?;
        if start.value > 0 {
            decoder.seek(start).await.map_err(media_error)?;
        }
        let matrix = stream.display_matrix;
        let (width, height) = crate::video_saliency::display_size(
            stream.width.unwrap_or(0),
            stream.height.unwrap_or(0),
            matrix,
        );
        stream.width = Some(width);
        stream.height = Some(height);
        stream.display_matrix = None;
        Ok(Self {
            backends,
            info,
            stream,
            decoder,
            matrix,
            start,
            duration,
        })
    }

    async fn next(&mut self, progress: &ProgressSink) -> Result<Option<VideoFrame>> {
        loop {
            live(progress)?;
            let Some(mut frame) = self.decoder.decode_next().await.map_err(media_error)? else {
                return Ok(None);
            };
            if frame.pts.seconds() < self.start.seconds() {
                continue;
            }
            if frame.pts.seconds() >= self.start.seconds() + self.duration.seconds() {
                return Ok(None);
            }
            frame.pts.value -= self.start.rescale(frame.pts.time_base).value;
            return oriented(frame, self.matrix).map(Some);
        }
    }
}

struct VideoOutput {
    encoder: Box<dyn ferment_engine_media::EncodeSession>,
    frames: usize,
    last_pts: f64,
}

impl VideoOutput {
    async fn open(source: &SourceWindow) -> Result<Self> {
        let mut settings = EncoderSettings::new(
            Codec::Hevc,
            source.stream.width.unwrap_or(0),
            source.stream.height.unwrap_or(0),
            source.stream.time_base,
        );
        // Slow motion stretches duration while preserving source cadence.
        // Factor affects scheduled PTS, never this nominal rate-control hint.
        settings.frame_rate = source.stream.frame_rate;
        settings.allow_frame_reordering = false;
        settings.bit_rate = Some(source.stream.bit_rate.unwrap_or(4_000_000).max(4_000_000));
        settings.color = declared_color(source.stream.color);
        Ok(Self {
            encoder: source
                .backends
                .video
                .open_encoder(settings)
                .await
                .map_err(media_error)?,
            frames: 0,
            last_pts: 0.0,
        })
    }

    async fn append(&mut self, frame: VideoFrame) -> Result<()> {
        self.last_pts = frame.pts.seconds();
        self.encoder
            .push_frame(
                frame
                    .encodable_scaled(frame.width, frame.height, frame.color)
                    .map_err(media_error)?,
            )
            .await
            .map_err(media_error)?;
        self.frames += 1;
        Ok(())
    }

    async fn finish(
        self,
        source: &SourceWindow,
        keep_audio: bool,
        duration: MediaTime,
        started: Instant,
        context: FileRenderContext<'_>,
    ) -> Result<core::VideoRenderResult> {
        let FileRenderContext {
            source: source_path,
            output,
            progress,
        } = context;
        if self.frames == 0 {
            return Err(EngineError::Decode(
                "video window produced no frames".into(),
            ));
        }
        live(progress)?;
        let encoded = self.encoder.finalize().await.map_err(media_error)?;
        let delivery = AtomicFileDelivery::begin(output)?;
        let mut mux = source
            .backends
            .muxer
            .open(delivery.partial_path(), MuxSettings::default())
            .map_err(media_error)?;
        let video = mux
            .add_video_track(VideoMuxTrack {
                display_matrix: None,
                configuration: encoded.configuration,
                width: source.stream.width.unwrap_or(0),
                height: source.stream.height.unwrap_or(0),
                time_base: source.stream.time_base,
                color: source.stream.color,
                duration: Some(duration),
            })
            .map_err(media_error)?;
        let audio = if keep_audio {
            audio_window(
                &source.backends,
                &source.info,
                source.start,
                source.duration,
                source_path,
                progress,
            )
            .await?
        } else {
            None
        };
        let (audio_id, packets) = if let Some((encoded, rate, channels)) = audio {
            let id = mux
                .add_audio_track(ferment_engine_media::AudioMuxTrack {
                    configuration: encoded.configuration,
                    sample_rate: rate,
                    channels,
                    timing: encoded.timing,
                })
                .map_err(media_error)?;
            (Some(id), encoded.packets)
        } else {
            (None, Vec::new())
        };
        for (is_video, packet) in interleaved_lanes(encoded.packets, packets) {
            live(progress)?;
            mux.write_packet(
                if is_video {
                    video
                } else {
                    audio_id.expect("audio packets own a track")
                },
                packet,
            )
            .map_err(media_error)?;
        }
        mux.finalize().map_err(media_error)?;
        live(progress)?;
        delivery.commit()?;
        progress.report(1.0, "Complete").await;
        Ok(core::VideoRenderResult {
            output_path: output.to_owned(),
            frames_written: self.frames,
            output_seconds: self.last_pts,
            wall_seconds: started.elapsed().as_secs_f64(),
            output_width: source.stream.width,
            output_height: source.stream.height,
            audio_preserved: keep_audio || source.info.audio_streams().next().is_none(),
            motion_blur: None,
        })
    }
}

async fn audio_window(
    backends: &crate::EncodeBackends,
    info: &ferment_engine_media::MediaInfo,
    start_time: MediaTime,
    duration: MediaTime,
    path: &Path,
    progress: &ProgressSink,
) -> Result<Option<(ferment_engine_media::EncodedAudio, u32, u16)>> {
    let Some(stream) = info.audio_streams().next() else {
        return Ok(None);
    };
    let rate = stream.sample_rate.unwrap_or(48_000);
    let channels = stream.channels.unwrap_or(1);
    let base = TimeBase::new_unchecked(1, i64::from(rate));
    let start = start_time.rescale(base).value;
    let end = start + duration.rescale(base).value;
    let mut decoder = backends
        .audio
        .open_audio_decoder(path, stream.index)
        .await
        .map_err(media_error)?;
    if start > 0 {
        decoder.seek(start_time).await.map_err(media_error)?;
    }
    let mut encoder = backends
        .audio
        .open_audio_encoder(ferment_engine_media::AudioEncoderSettings {
            bit_rate: 128_000,
            sample_rate: rate,
            channels,
        })
        .await
        .map_err(media_error)?;
    let mut written = 0;
    while let Some(frame) = decoder.decode_next().await.map_err(media_error)? {
        live(progress)?;
        let frame_start = frame.start.rescale(base).value;
        if frame_start >= end {
            break;
        }
        let first = (start - frame_start).max(0).min(frame.frame_count() as i64) as usize;
        let last = (end - frame_start).max(0).min(frame.frame_count() as i64) as usize;
        if last <= first {
            continue;
        }
        let pcm = ferment_engine_media::PcmAudioFrame::new(
            rate,
            channels,
            MediaTime::new(frame_start + first as i64 - start, base),
            frame.samples[first * usize::from(channels)..last * usize::from(channels)].to_vec(),
        )
        .map_err(media_error)?;
        encoder.push_pcm(pcm).await.map_err(media_error)?;
        written += last - first;
    }
    if written == 0 {
        return Ok(None);
    }
    Ok(Some((
        encoder.finalize().await.map_err(media_error)?,
        rate,
        channels,
    )))
}

pub async fn denoise_video(
    backend: Backend,
    source_path: &Path,
    output: &Path,
    strength: f32,
    window: Option<core::MediaTimeRange>,
    scenes: &[core::MediaTime],
    progress: ProgressSink,
) -> Result<core::VideoRenderResult> {
    let mut filter = crate::video_noise_filter::FastDvdnetTemporalFilter::new(backend);
    denoise_video_with_filter(
        source_path,
        output,
        strength,
        window,
        scenes,
        progress,
        &mut filter,
    )
    .await
}

/// Keep the current source's temporal-filter seam on the actual file pipeline.
/// Public callers still construct the same FastDVDnet backend.
async fn denoise_video_with_filter(
    source_path: &Path,
    output: &Path,
    strength: f32,
    window: Option<core::MediaTimeRange>,
    scenes: &[core::MediaTime],
    progress: ProgressSink,
    filter: &mut dyn crate::video_noise_filter::TemporalNoiseFiltering,
) -> Result<core::VideoRenderResult> {
    use crate::video_noise_filter::VideoNoiseFilter;
    use std::collections::VecDeque;
    VideoNoiseFilter::validate_strength(strength).map_err(media_error)?;
    let started = Instant::now();
    let mut source = SourceWindow::open(source_path, window).await?;
    let mut writer = VideoOutput::open(&source).await?;
    let cuts: Vec<f64> = scenes
        .iter()
        .map(|cut| cut.value as f64 / f64::from(cut.timescale) - source.start.seconds())
        .collect();
    filter
        .start_session(
            source.stream.width.unwrap_or(0),
            source.stream.height.unwrap_or(0),
            false,
        )
        .map_err(media_error)?;
    let result: Result<()> = async {
        let mut previous: VecDeque<VideoFrame> = VecDeque::new();
        let mut pending: VecDeque<VideoFrame> = VecDeque::new();
        let mut next_scene = None;
        let mut scene_end = false;
        let mut ended = false;
        loop {
            while pending.len() < filter.next_frame_count().saturating_add(1)
                && !scene_end
                && !ended
            {
                match source.next(&progress).await? {
                    Some(frame) => {
                        let prior = pending.back().or(previous.back());
                        if prior.is_some_and(|prior| {
                            cuts.iter().any(|cut| {
                                *cut > prior.pts.seconds() && *cut <= frame.pts.seconds()
                            })
                        }) {
                            next_scene = Some(frame);
                            scene_end = true;
                        } else {
                            pending.push_back(frame);
                        }
                    }
                    None => ended = true,
                }
            }
            let Some(mut frame) = pending.pop_front() else {
                break;
            };
            live(&progress)?;
            let sharp = frame.clone();
            if !previous.is_empty() || !pending.is_empty() {
                frame.buffer = filter
                    .filter(
                        &frame,
                        &previous.iter().cloned().collect::<Vec<_>>(),
                        &pending.iter().cloned().collect::<Vec<_>>(),
                        strength,
                        previous.is_empty(),
                    )
                    .await
                    .map_err(media_error)?;
            }
            let seconds = frame.pts.seconds();
            writer.append(frame).await?;
            previous.push_back(sharp);
            if previous.len() > filter.previous_frame_count() {
                previous.pop_front();
            }
            if pending.is_empty() && scene_end {
                previous.clear();
                if let Some(frame) = next_scene.take() {
                    pending.push_back(frame);
                }
                scene_end = false;
            }
            progress
                .report(
                    (seconds / source.duration.seconds()).min(0.95),
                    "Removing noise",
                )
                .await;
        }
        Ok(())
    }
    .await;
    filter.end_session();
    result?;
    writer
        .finish(
            &source,
            true,
            source.duration,
            started,
            FileRenderContext {
                source: source_path,
                output,
                progress: &progress,
            },
        )
        .await
}

fn slow_processing_size(width: u32, height: u32) -> (u32, u32) {
    (width.max(height), width.min(height))
}

/// The existing synchronous frame scheduler owns cuts/PTS; one worker owns its
/// decoder, RIFE requests, optional exposure integration and encoder.
#[allow(clippy::too_many_arguments)]
pub fn slow_motion_video(
    backend: Backend,
    source_path: &Path,
    output: &Path,
    factor: u8,
    window: Option<core::MediaTimeRange>,
    scenes: &[core::MediaTime],
    mode: core::MotionBlurMode,
    target_shutter_angle: f64,
    progress: ProgressSink,
) -> Result<core::VideoRenderResult> {
    use crate::slow_motion_renderer as slow;
    let mut interpolator = slow::RifeFrameInterpolator::new(backend.clone());
    let mut blurrer = RifeExposureBlurrer {
        interpolator: slow::RifeFrameInterpolator::new(backend),
        progress: progress.clone(),
    };
    slow_motion_video_with_processors(
        source_path,
        output,
        factor,
        window,
        scenes,
        mode,
        target_shutter_angle,
        progress,
        &mut interpolator,
        &mut blurrer,
    )
}

// The file renderer is shared by production and the current source's recorder
// tests. Only interpolation/blur are seams; decoding, analysis, encoding, muxing
// and final delivery remain the production path in every case.
#[allow(clippy::too_many_arguments)]
fn slow_motion_video_with_processors(
    source_path: &Path,
    output: &Path,
    factor: u8,
    window: Option<core::MediaTimeRange>,
    scenes: &[core::MediaTime],
    mode: core::MotionBlurMode,
    target_shutter_angle: f64,
    progress: ProgressSink,
    interpolator: &mut dyn crate::slow_motion_renderer::FrameInterpolator<VideoFrame>,
    blurrer: &mut dyn crate::slow_motion_renderer::FrameBlurrer<VideoFrame>,
) -> Result<core::VideoRenderResult> {
    use crate::slow_motion_renderer as slow;
    use futures_executor::block_on;
    if !(2..=8).contains(&factor) {
        return Err(EngineError::invalid_config(
            "slow-motion factor must be 2 through 8",
        ));
    }
    live(&progress)?;
    let started = Instant::now();
    let mut source = block_on(SourceWindow::open(source_path, window))?;
    let mode = match mode {
        core::MotionBlurMode::Disabled => slow::MotionBlurMode::Off,
        core::MotionBlurMode::Automatic => slow::MotionBlurMode::Auto,
        core::MotionBlurMode::Enabled { strength } => slow::MotionBlurMode::On { strength },
    };
    let analysis = if matches!(mode, slow::MotionBlurMode::Auto) {
        match analyze_motion(&source, source_path, &progress) {
            Ok(analysis) => Some(analysis),
            Err(EngineError::Cancelled) => return Err(EngineError::Cancelled),
            Err(error) => {
                tracing::warn!("motion analysis failed; blur is indeterminate: {error}");
                None
            }
        }
    } else {
        None
    };
    // Cancellation racing a failed optional analysis must not open a writer.
    live(&progress)?;
    let decision =
        slow::MotionBlurPolicy::decide(mode, analysis.as_ref(), factor, target_shutter_angle, true);
    let original_size = (
        source.stream.width.unwrap_or(0),
        source.stream.height.unwrap_or(0),
    );
    let portrait = slow_processing_size(original_size.0, original_size.1) != original_size;
    let mut writer = block_on(VideoOutput::open(&source))?;
    let cuts: Vec<f64> = scenes
        .iter()
        .map(|cut| cut.value as f64 / f64::from(cut.timescale) - source.start.seconds())
        .collect();
    {
        let mut adapter = SlowOutput {
            writer: &mut writer,
            progress: &progress,
            restore_portrait: portrait,
        };
        let mut scheduler = slow::orchestrator::Scheduler::new(factor, interpolator, &mut adapter);
        let mut previous: Option<(slow::TimedFrame<VideoFrame>, bool)> = None;
        let mut current: Option<(slow::TimedFrame<VideoFrame>, bool)> = None;
        while let Some(frame) = block_on(source.next(&progress))? {
            let frame = if portrait {
                frame.rotated_clockwise(1).map_err(media_error)?
            } else {
                frame
            };
            let cut = current.as_ref().is_some_and(|(last, _)| {
                cuts.iter()
                    .any(|cut| *cut > last.pts.seconds() && *cut <= frame.pts.seconds())
            });
            let next = slow::TimedFrame {
                pts: slow::MediaTime::new(
                    frame.pts.value * frame.pts.time_base.numerator,
                    u32::try_from(frame.pts.time_base.denominator)
                        .map_err(|_| EngineError::invalid_config("video time base exceeds u32"))?,
                ),
                frame,
            };
            if let Some((sharp, current_cut)) = current.take() {
                let mut anchor = sharp.clone();
                if decision.applied {
                    anchor.frame = blurrer
                        .blur(
                            &sharp,
                            if current_cut {
                                None
                            } else {
                                previous.as_ref().map(|(frame, _)| frame)
                            },
                            if cut { None } else { Some(&next) },
                            decision.strength,
                        )
                        .map_err(|e| render_error(e, &progress))?;
                }
                scheduler
                    .emit_anchor(anchor, current_cut)
                    .map_err(|e| render_error(e, &progress))?;
                previous = Some((sharp, current_cut));
            }
            let seconds = next.pts.seconds();
            current = Some((next, cut));
            block_on(progress.report(
                (seconds / source.duration.seconds()).min(0.95),
                "Interpolating frames",
            ));
        }
        if let Some((sharp, cut)) = current {
            let mut anchor = sharp.clone();
            if decision.applied {
                anchor.frame = blurrer
                    .blur(
                        &sharp,
                        if cut {
                            None
                        } else {
                            previous.as_ref().map(|(frame, _)| frame)
                        },
                        None,
                        decision.strength,
                    )
                    .map_err(|e| render_error(e, &progress))?;
            }
            scheduler
                .emit_anchor(anchor, cut)
                .map_err(|e| render_error(e, &progress))?;
        }
        scheduler.finish().map_err(|e| render_error(e, &progress))?;
    }
    let duration = MediaTime::new(
        source.duration.value * i64::from(factor),
        source.duration.time_base,
    );
    let mut result = block_on(writer.finish(
        &source,
        false,
        duration,
        started,
        FileRenderContext {
            source: source_path,
            output,
            progress: &progress,
        },
    ))?;
    use slow::MotionBlurReason as Reason;
    result.motion_blur = Some(core::MotionBlurDecision {
        enabled: decision.applied,
        strength: decision.strength,
        reason: match decision.reason {
            Reason::ExplicitOff => "explicitOff",
            Reason::ExplicitOn => "explicitOn",
            Reason::AlreadyBlurred => "alreadyBlurred",
            Reason::CrispPreferred => "crispPreferred",
            Reason::FastShutterSource => "fastShutterSource",
            Reason::MotionTooSmall => "motionTooSmall",
            Reason::Indeterminate => "indeterminate",
            Reason::Unsupported => "unsupported",
        }
        .into(),
        estimated_shutter_angle: Some(decision.source_shutter_angle),
        target_shutter_angle,
    });
    Ok(result)
}

struct SlowOutput<'a> {
    writer: &'a mut VideoOutput,
    progress: &'a ProgressSink,
    restore_portrait: bool,
}
impl crate::slow_motion_renderer::FrameOutput<VideoFrame> for SlowOutput<'_> {
    fn begin(&mut self) -> std::result::Result<(), String> {
        Ok(())
    }
    fn append(
        &mut self,
        mut frame: VideoFrame,
        pts: crate::slow_motion_renderer::MediaTime,
    ) -> std::result::Result<(), String> {
        live(self.progress).map_err(|e| e.to_string())?;
        frame.pts = MediaTime::new(
            pts.value,
            TimeBase::new_unchecked(1, i64::from(pts.timescale)),
        );
        if self.restore_portrait {
            frame = frame.rotated_clockwise(3).map_err(|e| e.to_string())?;
        }
        futures_executor::block_on(self.writer.append(frame)).map_err(|e| e.to_string())
    }
    fn commit(&mut self) -> std::result::Result<(), String> {
        Ok(())
    }
    fn abort(&mut self) {}
}

struct RifeExposureBlurrer {
    interpolator: crate::slow_motion_renderer::RifeFrameInterpolator,
    progress: ProgressSink,
}
impl crate::slow_motion_renderer::FrameBlurrer<VideoFrame> for RifeExposureBlurrer {
    fn blur(
        &mut self,
        frame: &crate::slow_motion_renderer::TimedFrame<VideoFrame>,
        previous: Option<&crate::slow_motion_renderer::TimedFrame<VideoFrame>>,
        next: Option<&crate::slow_motion_renderer::TimedFrame<VideoFrame>>,
        strength: i32,
    ) -> std::result::Result<VideoFrame, String> {
        use crate::slow_motion_renderer::FrameInterpolator;
        live(&self.progress).map_err(|e| e.to_string())?;
        let span = strength.clamp(1, 100) as f32 / 200.0;
        let mut exposure = vec![frame.frame.clone()];
        if let Some(previous) = previous {
            exposure.extend(self.interpolator.interpolate(
                previous,
                frame,
                &[1.0 - span, 1.0 - span * 0.5],
                true,
            )?);
        }
        live(&self.progress).map_err(|e| e.to_string())?;
        if let Some(next) = next {
            exposure.extend(self.interpolator.interpolate(
                frame,
                next,
                &[span * 0.5, span],
                true,
            )?);
        }
        let mut sum = vec![0u32; frame.frame.width as usize * frame.frame.height as usize * 3];
        for sample in &exposure {
            let rgb = sample.to_cpu_rgb().map_err(|e| e.to_string())?.to_rgb24();
            for (sum, byte) in sum.iter_mut().zip(rgb.data.iter()) {
                *sum += u32::from(*byte);
            }
        }
        let count = exposure.len() as u32;
        let data: Vec<u8> = sum
            .into_iter()
            .map(|value| ((value + count / 2) / count) as u8)
            .collect();
        let mut result = frame.frame.clone();
        result.buffer = FrameBuffer::Cpu(
            CpuFrame::new(
                result.width,
                result.height,
                result.width as usize * 3,
                CpuPixelFormat::Rgb24,
                data,
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(result)
    }
}

fn analyze_motion(
    source: &SourceWindow,
    path: &Path,
    progress: &ProgressSink,
) -> Result<crate::slow_motion_renderer::SlowMotionAnalysis> {
    use crate::slow_motion_renderer::{LumaPlane, SlowMotionAnalyzer};
    use futures_executor::block_on;
    let mut decoder = block_on(
        source
            .backends
            .video
            .open_coded_decoder(path, source.stream.index),
    )
    .map_err(media_error)?;
    let fps = source
        .stream
        .frame_rate
        .map(|(n, d)| f64::from(n) / f64::from(d))
        .unwrap_or(30.0);
    let mut pairs = Vec::new();
    let luma = |frame: VideoFrame| -> Result<LumaPlane> {
        let rgb = frame.to_cpu_rgb().map_err(media_error)?.to_rgb24();
        let plane = LumaPlane {
            width: rgb.width as usize,
            height: rgb.height as usize,
            bytes_per_row: rgb.width as usize,
            data: rgb
                .data
                .chunks_exact(3)
                .map(|p| {
                    ((u32::from(p[0]) * 54 + u32::from(p[1]) * 183 + u32::from(p[2]) * 19) >> 8)
                        as u8
                })
                .collect(),
        };
        SlowMotionAnalyzer::downscaled_luma(&plane).ok_or(EngineError::InvalidOutput)
    };
    for seconds in
        SlowMotionAnalyzer::sample_times(source.start.seconds(), source.duration.seconds(), fps)
    {
        live(progress)?;
        block_on(decoder.seek(MediaTime::from_seconds(seconds, source.stream.time_base)))
            .map_err(media_error)?;
        let Some(a) = block_on(decoder.decode_next()).map_err(media_error)? else {
            continue;
        };
        let Some(b) = block_on(decoder.decode_next()).map_err(media_error)? else {
            continue;
        };
        pairs.push((luma(a)?, luma(b)?));
    }
    Ok(SlowMotionAnalyzer::analyze_pairs(&pairs))
}

fn render_error(error: impl std::fmt::Display, progress: &ProgressSink) -> EngineError {
    if progress.is_closed() {
        EngineError::Cancelled
    } else {
        media_error(error)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "render_pipeline_current_tests.rs"]
mod current_slowmotion_real_assets;

#[cfg(all(test, feature = "ort", not(target_arch = "wasm32")))]
#[path = "render_pipeline_audio_tests.rs"]
mod audio_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "render_pipeline_slowmotion_file_tests.rs"]
mod current_slowmotion_file_contract;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "render_pipeline_noise_source_tests.rs"]
mod current_noise_source_tests;

#[cfg(test)]
#[path = "render_pipeline_cadence_tests.rs"]
mod cadence_tests;

#[cfg(test)]
#[path = "render_pipeline_orientation_tests.rs"]
mod orientation_tests;
