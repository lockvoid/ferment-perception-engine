//! Observe the settings at the actual shared encoder admission boundary.
//! This deliberately stops before a codec opens; file/pixel qualification
//! remains the original platform tests and Android portrait/quality gates.
use super::*;
use async_trait::async_trait;
use ferment_engine_core::VideoEncoding;
use ferment_engine_media::{
    ContainerFormat, DecodeSession, DecoderConfiguration, EncodeSession, MediaInfo, MediaKind,
    StreamInfo, UnavailableAudioIo, VideoError, VideoIo,
};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct CapturingVideo {
    info: MediaInfo,
    settings: Mutex<Vec<EncoderSettings>>,
}

#[derive(Debug)]
struct EmptyDecoder;

#[async_trait(?Send)]
impl DecodeSession for EmptyDecoder {
    async fn seek(&mut self, _target: MediaTime) -> ferment_engine_media::Result<()> {
        Ok(())
    }
    async fn decode_next(&mut self) -> ferment_engine_media::Result<Option<VideoFrame>> {
        Ok(None)
    }
}

#[async_trait(?Send)]
impl VideoIo for CapturingVideo {
    async fn probe(&self, _source: &Path) -> ferment_engine_media::Result<MediaInfo> {
        Ok(self.info.clone())
    }
    async fn open_decoder(
        &self,
        _source: &Path,
        _stream_index: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        Ok(Box::new(EmptyDecoder))
    }
    async fn open_encoder(
        &self,
        settings: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        self.settings.lock().unwrap().push(settings);
        Err(VideoError::BackendUnavailable(
            "cadence admission recorded".into(),
        ))
    }
}

fn fixture(rate: Option<(u32, u32)>) -> (crate::EncodeBackends, Arc<CapturingVideo>) {
    let clock = TimeBase::new_unchecked(1, 90000);
    let duration = Some(MediaTime::new(180000, clock));
    let stream = StreamInfo {
        index: 0,
        kind: MediaKind::Video,
        codec: Codec::H264,
        time_base: clock,
        start_time: Some(MediaTime::new(0, clock)),
        duration,
        width: Some(1440),
        height: Some(2560),
        sample_rate: None,
        channels: None,
        frame_rate: rate,
        display_matrix: None,
        bit_rate: Some(100_053_440),
        language: None,
        color: ColorInfo::default(),
        audio_timing: None,
        decoder_configuration: DecoderConfiguration::from_codec_private(Codec::H264, Vec::new())
            .unwrap(),
    };
    let video = Arc::new(CapturingVideo {
        info: MediaInfo {
            container: ContainerFormat::Mp4,
            duration,
            streams: vec![stream],
        },
        settings: Mutex::new(Vec::new()),
    });
    let backends = crate::EncodeBackends {
        video: video.clone(),
        audio: Arc::new(UnavailableAudioIo),
        muxer: Arc::new(ferment_engine_media::platform::libav::LibavMuxer),
    };
    (backends, video)
}

fn admitted(video: &CapturingVideo, rate: Option<(u32, u32)>) -> EncoderSettings {
    let mut all = video.settings.lock().unwrap();
    assert_eq!(
        all.len(),
        1,
        "the intended production encoder door must be reached exactly once"
    );
    let settings = all.pop().unwrap();
    assert_eq!(settings.time_base, TimeBase::new_unchecked(1, 90000));
    assert_eq!(
        settings.frame_rate, rate,
        "do not reconstruct or round the source rational"
    );
    let hint = settings.frame_rate_hint();
    let expected = rate
        .filter(|(n, d)| *n > 0 && *d > 0)
        .map_or(30.0, |(n, d)| f64::from(n) / f64::from(d));
    assert_eq!(hint, expected);
    // Keep the already admitted source-planning fallback aligned with this
    // configure-only hint. Neither is a replacement timeline for VFR input.
    assert_eq!(
        ferment_engine_core::VideoRateControl::FALLBACK_FRAME_RATE,
        30.0
    );
    settings
}

// Kill: drop nominal metadata in the public VideoEncoder -> VariantLane
// path, including unknown-rate VFR inputs or fractional-rate sources.
#[test]
fn variant_admission_preserves_source_cadence_and_timestamp_clock() {
    futures_executor::block_on(async {
        for rate in [Some((30, 1)), Some((30000, 1001)), None, Some((0, 0))] {
            let (backends, video) = fixture(rate);
            let error = crate::VideoEncoder::new()
                .with_backends(backends)
                .encode(
                    Path::new("cadence-source.mp4"),
                    Path::new("cadence-output.mp4"),
                    &crate::VideoEncoder::ai_preset(),
                    None,
                    ProgressSink::noop(),
                    None,
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("cadence admission recorded"));
            admitted(&video, rate);
        }
    });
}

// Kill: only fix VideoEncoder while VideoUpscaler still derives cadence from
// its 90 kHz clock, or reject absent metadata before reaching the codec.
#[test]
fn upscale_admission_preserves_fractional_or_unknown_source_cadence() {
    futures_executor::block_on(async {
        for rate in [Some((30, 1)), Some((30000, 1001)), None, Some((0, 0))] {
            let (backends, video) = fixture(rate);
            let error = upscale_video_with_processor(
                backends,
                Path::new("cadence-source.mp4"),
                Path::new("cadence-output.mp4"),
                UpscaleMode::Enhance,
                None,
                ProgressSink::noop(),
                &mut crate::video_upscaler::PassthroughFrameUpscaler,
                None,
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("cadence admission recorded"));
            let settings = admitted(&video, rate);
            assert_eq!(settings.codec, Codec::Hevc);
            assert_eq!((settings.width, settings.height), (1440, 2560));
        }
    });
}

// Kill: use the slow-motion factor or timestamp clock as frame cadence at the
// shared denoise/slow-motion writer; the source geometry/bitrate stay exact.
#[test]
fn shared_render_admission_keeps_source_cadence_geometry_and_bitrate() {
    futures_executor::block_on(async {
        for rate in [Some((30, 1)), Some((30000, 1001)), None, Some((0, 0))] {
            let (backends, video) = fixture(rate);
            let stream = video.info.streams[0].clone();
            let source = SourceWindow {
                backends,
                info: video.info.clone(),
                stream: stream.clone(),
                decoder: Box::new(EmptyDecoder),
                matrix: None,
                start: stream.start_time.unwrap(),
                duration: stream.duration.unwrap(),
            };
            let error = VideoOutput::open(&source)
                .await
                .err()
                .expect("capture must stop before codec setup");
            assert!(error.to_string().contains("cadence admission recorded"));
            let settings = admitted(&video, rate);
            assert_eq!(settings.codec, Codec::Hevc);
            assert_eq!((settings.width, settings.height), (1440, 2560));
            assert_eq!(settings.bit_rate, Some(100_053_440));
            assert!(!settings.allow_frame_reordering);
        }
    });
}
