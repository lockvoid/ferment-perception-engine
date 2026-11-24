//! Public `engine-core` capability adapters for the lifted processor families.
//!
//! The reusable processors continue to accept decoded `AudioFrames`; these
//! adapters preserve the Swift path-based contracts at the crate boundary.

use std::path::Path;

use async_trait::async_trait;
use ferment_engine_core as core;

use crate::audio_minimap_composer;
use crate::{
    AudioMinimapComposer, AudioSemanticEncoder, BeatQueenMary, BeatThis, CaptionSegmenter,
    DeepFilterNet, MusicHighlighter, MusicTagger, Parakeet, SlowMotionRenderer,
    VideoMinimapComposer, VideoSceneDetector, VideoUpscaler, audio_source::AudioSource,
};

const VIDEO_UPSCALER_UNAVAILABLE: &str = "video upscaling requires platform VideoIO decode, a frame-upscaling backend, and playable encode/mux output";
const SLOW_MOTION_UNAVAILABLE: &str = "slow-motion rendering requires platform VideoIO decode, a frame-interpolation backend, and playable encode/mux output";

fn validate_render_window(family: &str, window: Option<core::MediaTimeRange>) -> core::Result<()> {
    if window.is_some_and(|range| range.duration.timescale <= 0 || range.duration.value <= 0) {
        return Err(core::EngineError::invalid_config(format!(
            "{family} window duration must be positive"
        )));
    }
    Ok(())
}

#[async_trait(?Send)]
impl core::VideoUpscaling for VideoUpscaler {
    async fn render(
        &self,
        source_path: &Path,
        output_path: &Path,
        mode: core::VideoUpscaleMode,
        window: Option<core::MediaTimeRange>,
        scene_starts: &[core::MediaTime],
        progress: core::ProgressSink,
    ) -> core::Result<core::VideoRenderResult> {
        validate_render_window("video-upscale", window)?;
        let backend = self
            .backend
            .clone()
            .ok_or_else(|| core::EngineError::Unsupported(VIDEO_UPSCALER_UNAVAILABLE.into()))?;
        let mode = match mode {
            core::VideoUpscaleMode::Enhance => crate::video_upscaler::UpscaleMode::Enhance,
            core::VideoUpscaleMode::Upscale { longest_side } => {
                crate::video_upscaler::UpscaleMode::Upscale { longest_side }
            }
        };
        crate::render_pipeline::upscale_video(
            backend,
            source_path,
            output_path,
            mode,
            window,
            scene_starts,
            progress,
        )
        .await
    }
}

#[async_trait(?Send)]
impl core::SlowMotionRendering for SlowMotionRenderer {
    async fn render(
        &self,
        source_path: &Path,
        output_path: &Path,
        factor: u8,
        window: Option<core::MediaTimeRange>,
        scene_starts: &[core::MediaTime],
        motion_blur: core::MotionBlurMode,
        target_shutter_angle: f64,
        progress: core::ProgressSink,
    ) -> core::Result<core::VideoRenderResult> {
        if !(2..=8).contains(&factor) {
            return Err(core::EngineError::invalid_config(format!(
                "slow-motion factor {factor} is out of range (2-8)"
            )));
        }
        validate_render_window("slow-motion", window)?;
        let backend = self
            .backend
            .clone()
            .ok_or_else(|| core::EngineError::Unsupported(SLOW_MOTION_UNAVAILABLE.into()))?;
        std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    crate::render_pipeline::slow_motion_video(
                        backend,
                        source_path,
                        output_path,
                        factor,
                        window,
                        scene_starts,
                        motion_blur,
                        target_shutter_angle,
                        progress,
                    )
                })
                .join()
                .map_err(|_| core::EngineError::inference("slow-motion worker panicked"))?
        })
    }
}

#[cfg(test)]
mod wave4_capability_tests {
    use super::*;

    fn assert_video_upscaling<T: core::VideoUpscaling>() {}
    fn assert_slow_motion_rendering<T: core::SlowMotionRendering>() {}

    fn window(duration_value: i64) -> core::MediaTimeRange {
        core::MediaTimeRange {
            start: core::MediaTime::new(0, 600),
            duration: core::MediaTime::new(duration_value, 600),
        }
    }

    async fn upscale(
        window: Option<core::MediaTimeRange>,
    ) -> core::Result<core::VideoRenderResult> {
        core::VideoUpscaling::render(
            &VideoUpscaler::new(),
            Path::new("source.mov"),
            Path::new("output.mp4"),
            core::VideoUpscaleMode::Upscale { longest_side: 3840 },
            window,
            &[],
            core::ProgressSink::noop(),
        )
        .await
    }

    async fn slow_motion(
        factor: u8,
        window: Option<core::MediaTimeRange>,
    ) -> core::Result<core::VideoRenderResult> {
        core::SlowMotionRendering::render(
            &SlowMotionRenderer::new(),
            Path::new("source.mov"),
            Path::new("output.mp4"),
            factor,
            window,
            &[],
            core::MotionBlurMode::Disabled,
            180.0,
            core::ProgressSink::noop(),
        )
        .await
    }

    #[test]
    fn wave4_services_implement_public_capability_traits() {
        assert_video_upscaling::<VideoUpscaler>();
        assert_slow_motion_rendering::<SlowMotionRenderer>();
    }

    #[test]
    fn video_upscaler_reports_the_platform_processing_and_mux_boundary() {
        let error = futures_executor::block_on(upscale(None)).unwrap_err();
        match error {
            core::EngineError::Unsupported(message) => {
                assert_eq!(message, VIDEO_UPSCALER_UNAVAILABLE);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn slow_motion_renderer_reports_the_platform_processing_and_mux_boundary() {
        let error = futures_executor::block_on(slow_motion(2, None)).unwrap_err();
        match error {
            core::EngineError::Unsupported(message) => {
                assert_eq!(message, SLOW_MOTION_UNAVAILABLE);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn slow_motion_renderer_validates_factor_before_platform_capabilities() {
        let error = futures_executor::block_on(slow_motion(1, None)).unwrap_err();
        match error {
            core::EngineError::InvalidConfiguration(message) => {
                assert_eq!(message, "slow-motion factor 1 is out of range (2-8)");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn wave4_services_validate_empty_windows_before_platform_capabilities() {
        let upscale_error = futures_executor::block_on(upscale(Some(window(0)))).unwrap_err();
        match upscale_error {
            core::EngineError::InvalidConfiguration(message) => {
                assert_eq!(message, "video-upscale window duration must be positive");
            }
            other => panic!("unexpected error: {other}"),
        }

        let slow_motion_error =
            futures_executor::block_on(slow_motion(2, Some(window(-1)))).unwrap_err();
        match slow_motion_error {
            core::EngineError::InvalidConfiguration(message) => {
                assert_eq!(message, "slow-motion window duration must be positive");
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

#[async_trait(?Send)]
impl core::VideoSceneDetecting for VideoSceneDetector {
    async fn detect(
        &self,
        video_path: &Path,
        progress: core::ProgressSink,
    ) -> core::Result<core::VideoSceneDetectionResult> {
        let result = self.detect_streaming(video_path, progress).await?;
        Ok(core::VideoSceneDetectionResult {
            scenes: result
                .scenes
                .into_iter()
                .enumerate()
                .map(|(index, scene)| core::VideoScene {
                    index,
                    start_frame: scene.start_frame,
                    frame_count: scene.frame_count,
                    start_time: scene.start_time,
                    duration: scene.duration,
                    start_confidence: scene.start_confidence,
                    embedding: scene.embedding,
                })
                .collect(),
            total_frames: result.total_frames,
            frame_rate: result.frame_rate,
        })
    }
}

#[async_trait(?Send)]
impl core::VideoMinimapComposing for VideoMinimapComposer {
    async fn compose(
        &self,
        video_path: &Path,
        scenes: &core::VideoSceneDetectionResult,
        progress: core::ProgressSink,
    ) -> core::Result<core::MinimapResult> {
        let source_scenes = scenes
            .scenes
            .iter()
            .map(|scene| crate::video_minimap_composer::Scene {
                start_frame: scene.start_frame,
                frame_count: scene.frame_count,
                start_time: scene.start_time,
                duration: scene.duration,
            })
            .collect::<Vec<_>>();
        progress.report(0.05, "decoding video").await;
        let result = self
            .compose_file(
                video_path,
                &source_scenes,
                scenes.total_frames,
                scenes.frame_rate,
                progress,
            )
            .await?;
        let duration = scenes.duration();
        let shown_scenes = result
            .cells
            .iter()
            .map(|cell| cell.scene_index)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        Ok(core::MinimapResult {
            image_data: result.image_data,
            format: match result.format {
                crate::video_minimap_composer::ImageFormat::Png => core::ImageFormat::Png,
                crate::video_minimap_composer::ImageFormat::Jpeg { quality } => {
                    core::ImageFormat::Jpeg {
                        quality: f32::from(quality) / 100.0,
                    }
                }
            },
            width: result.width,
            height: result.height,
            layout: core::MinimapLayout {
                cells: result
                    .cells
                    .into_iter()
                    .map(|cell| core::MinimapCell {
                        scene_index: cell.scene_index,
                        frame_index: cell.frame_index,
                        time: cell.time,
                        rect: core::Rect {
                            x: f64::from(cell.x),
                            y: f64::from(cell.y),
                            width: f64::from(cell.width),
                            height: f64::from(cell.height),
                        },
                    })
                    .collect(),
                header_rect: Some(result.layout.header_rect),
                barcode_rect: Some(result.layout.barcode_rect),
                mel_rect: None,
                rms_rect: None,
                time_axis_rect: Some(result.layout.time_axis_rect),
                plot_left: None,
                plot_right: None,
                time_range: core::TimeRange::new(0.0, duration),
            },
            total_scenes: Some(source_scenes.len()),
            shown_scenes: Some(shown_scenes),
        })
    }
}

#[async_trait(?Send)]
impl core::BeatAnalyzing for BeatThis {
    async fn detect(
        &self,
        audio_path: &Path,
        progress: core::ProgressSink,
    ) -> core::Result<core::BeatAnalysisResult> {
        let result = BeatThis::detect(
            self,
            AudioSource::default()
                .resampled(audio_path, 22050, None, self.cache.as_deref())
                .await?,
            progress,
        )
        .await?;
        Ok(map_beat_result(result))
    }
}

#[async_trait(?Send)]
impl core::BeatAnalyzing for BeatQueenMary {
    async fn detect(
        &self,
        audio_path: &Path,
        progress: core::ProgressSink,
    ) -> core::Result<core::BeatAnalysisResult> {
        let result = BeatQueenMary::detect(
            self,
            AudioSource::default().decode(audio_path).await?,
            progress,
        )
        .await?;
        Ok(map_beat_result(result))
    }
}

fn map_beat_result(result: crate::beatthis::BeatAnalysisResult) -> core::BeatAnalysisResult {
    core::BeatAnalysisResult {
        beats: result.beats,
        downbeats: result.downbeats,
        quantized_beats: result.quantized_beats,
        quantized_downbeats: result.quantized_downbeats,
        grid_stability: result.grid_stability,
        tempo_stability: result.tempo_stability,
        bpm: result.bpm,
        quantized_bpm: result.quantized_bpm,
        segments: result.segments,
        energy_per_bar: result.energy_per_bar,
        energy_structure: result.energy_structure,
    }
}

#[async_trait(?Send)]
impl core::AudioSemanticEncoding for AudioSemanticEncoder {
    async fn encode(
        &self,
        audio_path: &Path,
        return_chunks: bool,
        progress: core::ProgressSink,
    ) -> core::Result<core::AudioEmbedding> {
        let result = AudioSemanticEncoder::encode(
            self,
            AudioSource::default()
                .resampled(audio_path, 16000, None, self.cache.as_deref())
                .await?,
            return_chunks,
            progress,
        )
        .await?;
        Ok(core::AudioEmbedding {
            pooled: result.pooled,
            chunks: result.chunks,
            duration_seconds: result.duration_seconds,
            window_seconds: result.window_seconds as u32,
            dim: result.dim,
        })
    }
}

#[async_trait(?Send)]
impl core::SpeechDenoising for DeepFilterNet {
    async fn denoise(
        &self,
        audio_path: &Path,
        keep_segments: Option<&[core::AudioEventSegment]>,
        progress: core::ProgressSink,
    ) -> core::Result<core::DenoisedAudio> {
        let frames = AudioSource::default().decode(audio_path).await?;
        let output = match keep_segments {
            Some(segments) => {
                self.denoise_with_keep_segments(frames, segments, progress)
                    .await?
            }
            None => DeepFilterNet::denoise(self, frames, progress).await?,
        };
        Ok(core::DenoisedAudio {
            audio: core::AudioBuffer::mono(output.samples, output.sample_rate),
        })
    }
}

#[async_trait(?Send)]
impl core::MusicHighlighting for MusicHighlighter {
    async fn detect(
        &self,
        audio_path: &Path,
        window_lengths: &[u32],
        progress: core::ProgressSink,
    ) -> core::Result<core::HighlightResult> {
        let result = MusicHighlighter::detect(
            self,
            AudioSource::default()
                .resampled(audio_path, 22050, None, self.cache.as_deref())
                .await?,
            window_lengths.iter().map(|&value| value as usize).collect(),
            progress,
        )
        .await?;
        Ok(core::HighlightResult {
            saliency: result.saliency,
            highlights: result
                .highlights
                .into_iter()
                .map(|highlight| core::Highlight {
                    window: highlight.window as u32,
                    start_time: highlight.start,
                    duration: highlight.duration,
                })
                .collect(),
            duration: result.duration as u32,
        })
    }
}

#[async_trait(?Send)]
impl core::MusicTagging for MusicTagger {
    async fn tag(&self, embedding: &[f32], top_k: usize) -> core::Result<core::MusicTagsResult> {
        let result = MusicTagger::tag(self, embedding, top_k).await?;
        let map_axis = |tags: Vec<crate::musictagger::ScoredTag>| {
            tags.into_iter()
                .map(|tag| core::ScoredTag {
                    name: tag.name,
                    score: tag.score,
                })
                .collect()
        };
        Ok(core::MusicTagsResult {
            genre: map_axis(result.genre),
            mood: map_axis(result.mood),
            instrument: map_axis(result.instrument),
        })
    }
}

#[async_trait(?Send)]
impl core::SpeechTranscribing for Parakeet {
    async fn transcribe(
        &self,
        audio_path: &Path,
        speech_segments: Option<&[core::AudioEventSegment]>,
        progress: core::ProgressSink,
    ) -> core::Result<core::TranscriptionResult> {
        self.transcribe_file_in(audio_path, &std::env::temp_dir(), speech_segments, progress)
            .await
    }
}

impl Parakeet {
    /// File-based transcription owns one resampled PCM scratch until this
    /// call returns. Android supplies a directory in its own UID; neither the
    /// input nor another call's files are ever adopted. Cleanup runs on error,
    /// cancellation and success. In-memory processor callers remain supported.
    pub async fn transcribe_file_in(
        &self,
        audio_path: &Path,
        temporary_directory: &Path,
        speech_segments: Option<&[core::AudioEventSegment]>,
        progress: core::ProgressSink,
    ) -> core::Result<core::TranscriptionResult> {
        use crate::parakeet::config::SAMPLE_RATE;

        if speech_segments.is_some_and(<[core::AudioEventSegment]>::is_empty) {
            progress.report(1.0, "No speech detected").await;
            return Ok(core::TranscriptionResult {
                transcription: String::new(),
                language: None,
                captions: Vec::new(),
            });
        }
        let check_cancellation = || {
            if progress.is_closed() {
                Err(core::EngineError::Cancelled)
            } else {
                Ok(())
            }
        };
        check_cancellation()?;
        progress.report(0.02, "loading audio").await;
        check_cancellation()?;
        let source = AudioSource::default()
            .decode_disk_backed_in_with_cancellation(
                audio_path,
                SAMPLE_RATE,
                temporary_directory,
                check_cancellation,
            )
            .await?;
        check_cancellation()?;
        let result = self
            .transcribe_source_inner(&source, speech_segments, &progress)
            .await?;
        drop(source);
        progress.report(1.0, "done").await;
        check_cancellation()?;
        Ok(result)
    }

    /// The host can reuse its verified, leased PCM cache. Samples must be
    /// mono 16 kHz; only the current encoder window is copied into memory.
    pub async fn transcribe_source(
        &self,
        source: &dyn core::AudioSampleSource,
        speech_segments: Option<&[core::AudioEventSegment]>,
        progress: core::ProgressSink,
    ) -> core::Result<core::TranscriptionResult> {
        let result = self
            .transcribe_source_inner(source, speech_segments, &progress)
            .await?;
        progress.report(1.0, "done").await;
        if progress.is_closed() {
            return Err(core::EngineError::Cancelled);
        }
        Ok(result)
    }

    async fn transcribe_source_inner(
        &self,
        source: &dyn core::AudioSampleSource,
        speech_segments: Option<&[core::AudioEventSegment]>,
        progress: &core::ProgressSink,
    ) -> core::Result<core::TranscriptionResult> {
        use crate::parakeet::config::SAMPLE_RATE;
        let check_cancellation = || {
            if progress.is_closed() {
                Err(core::EngineError::Cancelled)
            } else {
                Ok(())
            }
        };
        check_cancellation()?;
        if speech_segments.is_some_and(<[core::AudioEventSegment]>::is_empty) {
            return Ok(core::TranscriptionResult {
                transcription: String::new(),
                language: None,
                captions: Vec::new(),
            });
        }
        let duration = source.sample_count() as f64 / f64::from(SAMPLE_RATE);
        let runs = speech_segments.map(|segments| coalesce_segments(segments, 1.0, duration));
        let mut words = Vec::new();
        let mut language = None;

        match runs {
            None => {
                let result = self
                    .transcribe_source_range(source, 0..source.sample_count(), progress, 0.10, 0.95)
                    .await?;
                language.clone_from(&result.language);
                append_words(&mut words, result, 0.0);
            }
            Some(runs) => {
                let budget = 0.85 / runs.len().max(1) as f64;
                for (index, run) in runs.into_iter().enumerate() {
                    check_cancellation()?;
                    let start =
                        seconds_to_sample(run.start_time, SAMPLE_RATE).min(source.sample_count());
                    let end =
                        seconds_to_sample(run.end_time(), SAMPLE_RATE).min(source.sample_count());
                    if end <= start {
                        continue;
                    }
                    let result = self
                        .transcribe_source_range(
                            source,
                            start..end,
                            progress,
                            0.10 + index as f64 * budget,
                            0.10 + (index + 1) as f64 * budget,
                        )
                        .await?;
                    if language.is_none() {
                        language.clone_from(&result.language);
                    }
                    append_words(&mut words, result, run.start_time);
                }
            }
        }
        check_cancellation()?;
        let transcription = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let captions = CaptionSegmenter::captions(&words);
        Ok(core::TranscriptionResult {
            transcription,
            language,
            captions,
        })
    }
}

#[async_trait(?Send)]
impl core::AudioMinimapComposing for AudioMinimapComposer {
    async fn compose(
        &self,
        audio_path: &Path,
        bar_ruler: &[f64],
        progress: core::ProgressSink,
    ) -> core::Result<core::MinimapResult> {
        let result = AudioMinimapComposer::compose_with_bar_ruler(
            self,
            AudioSource::default()
                .resampled(audio_path, 22050, None, self.cache.as_deref())
                .await?,
            bar_ruler,
            progress,
        )
        .await?;
        let geometry = result.layout;
        Ok(core::MinimapResult {
            image_data: result.image_data,
            format: match result.format {
                audio_minimap_composer::ImageFormat::Png => core::ImageFormat::Png,
                audio_minimap_composer::ImageFormat::Jpeg { quality } => core::ImageFormat::Jpeg {
                    quality: f32::from(quality) / 100.0,
                },
            },
            width: result.width,
            height: result.height,
            layout: core::MinimapLayout {
                cells: Vec::new(),
                header_rect: Some(map_rect(geometry.header_rect)),
                barcode_rect: None,
                mel_rect: Some(map_rect(geometry.mel_rect)),
                rms_rect: Some(map_rect(geometry.rms_rect)),
                time_axis_rect: Some(map_rect(geometry.time_axis_rect)),
                plot_left: Some(geometry.plot_left),
                plot_right: Some(geometry.plot_right),
                time_range: core::TimeRange::new(
                    geometry.time_min,
                    geometry.time_max - geometry.time_min,
                ),
            },
            total_scenes: None,
            shown_scenes: None,
        })
    }
}

fn append_words(
    output: &mut Vec<core::TranscribedWord>,
    result: crate::parakeet::TranscriptionResult,
    time_offset: f64,
) {
    for word in result
        .captions
        .into_iter()
        .flat_map(|caption| caption.words)
    {
        output.push(core::TranscribedWord {
            id: ferment_utils::id::short(),
            text: word.text,
            start_time: word.start_time + time_offset,
            duration: (word.end_time - word.start_time).max(0.0),
            confidence: word.confidence,
        });
    }
}

fn seconds_to_sample(seconds: f64, sample_rate: u32) -> usize {
    (seconds.max(0.0) * f64::from(sample_rate)) as usize
}

fn coalesce_segments(
    segments: &[core::AudioEventSegment],
    pad_seconds: f64,
    duration_seconds: f64,
) -> Vec<core::AudioEventSegment> {
    let pad = pad_seconds.max(0.0);
    let upper = duration_seconds.max(0.0);
    let mut padded = segments
        .iter()
        .filter_map(|segment| {
            let start = (segment.start_time - pad).clamp(0.0, upper);
            let end = (segment.end_time() + pad).clamp(start, upper);
            (end > start).then_some(core::AudioEventSegment {
                event: segment.event,
                start_time: start,
                duration: end - start,
                confidence: segment.confidence,
            })
        })
        .collect::<Vec<_>>();
    padded.sort_by(|left, right| left.start_time.total_cmp(&right.start_time));

    let mut output: Vec<core::AudioEventSegment> = Vec::new();
    for next in padded {
        if let Some(current) = output.last_mut()
            && next.start_time <= current.end_time()
        {
            let end = current.end_time().max(next.end_time());
            current.duration = end - current.start_time;
            current.confidence = current.confidence.max(next.confidence);
            continue;
        }
        output.push(next);
    }
    output
}

fn map_rect(rect: audio_minimap_composer::layout::Rect) -> core::Rect {
    core::Rect {
        x: f64::from(rect.x),
        y: f64::from(rect.y),
        width: f64::from(rect.width),
        height: f64::from(rect.height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_core::{AudioEvent, AudioEventSegment};

    fn segment(start: f64, duration: f64, confidence: f32) -> AudioEventSegment {
        AudioEventSegment {
            event: AudioEvent::Speech,
            start_time: start,
            duration,
            confidence,
        }
    }

    #[test]
    fn segment_coalescing_pads_merges_and_clamps() {
        let result = coalesce_segments(&[segment(3.0, 1.0, 0.4), segment(0.5, 1.0, 0.8)], 1.0, 5.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_time, 0.0);
        assert_eq!(result[0].end_time(), 5.0);
        assert_eq!(result[0].confidence, 0.8);
    }
}
