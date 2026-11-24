//! Observe the production pass at its decoder boundary, before any pixels or
//! output files are allocated. Native scene/codec tests cover the delivered movie.
use super::*;
use async_trait::async_trait;
use ferment_engine_media::{
    Codec, ContainerFormat, DecodeOptions, DecodeSession, DecoderConfiguration, EncodeSession,
    EncodedVideo, MediaKind, TimeBase, VideoError, VideoFrame, VideoIo,
};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct Observed {
    color: ferment_engine_media::ColorInfo,
    requests: Mutex<Vec<DecodeOptions>>,
    resize_available: bool,
}
fn stopped<T>() -> ferment_engine_media::Result<T> {
    Err(VideoError::InvalidData(
        "decoder observation complete".into(),
    ))
}
#[derive(Debug)]
struct UnusedEncoder;
#[async_trait(?Send)]
impl EncodeSession for UnusedEncoder {
    async fn push_frame(&mut self, _: VideoFrame) -> ferment_engine_media::Result<()> {
        panic!("observation stops before decode");
    }
    async fn finalize(self: Box<Self>) -> ferment_engine_media::Result<EncodedVideo> {
        panic!("observation stops before decode");
    }
}
#[async_trait(?Send)]
impl VideoIo for Observed {
    async fn probe(&self, _: &Path) -> ferment_engine_media::Result<MediaInfo> {
        Ok(MediaInfo {
            container: ContainerFormat::Mp4,
            duration: None,
            streams: vec![StreamInfo {
                index: 0,
                kind: MediaKind::Video,
                codec: Codec::H264,
                time_base: TimeBase::MICROSECONDS,
                start_time: None,
                duration: None,
                width: Some(1920),
                height: Some(1080),
                sample_rate: None,
                channels: None,
                frame_rate: Some((30, 1)),
                display_matrix: None,
                bit_rate: None,
                language: None,
                color: self.color,
                audio_timing: None,
                decoder_configuration: DecoderConfiguration::from_codec_private(
                    Codec::H264,
                    vec![],
                )?,
            }],
        })
    }
    async fn open_decoder(
        &self,
        _: &Path,
        _: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.requests.lock().unwrap().push(DecodeOptions::default());
        stopped()
    }
    async fn open_decoder_with(
        &self,
        _: &Path,
        _: u32,
        options: DecodeOptions,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.requests.lock().unwrap().push(options);
        if !self.resize_available {
            return Err(VideoError::BackendUnavailable(
                "resize not implemented by test backend".into(),
            ));
        }
        stopped()
    }
    async fn open_encoder(
        &self,
        _: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        Ok(Box::new(UnusedEncoder))
    }
}
fn requests(
    presets: &[(
        ferment_engine_core::VideoEncodingPreset,
        Option<ThumbStripOptions>,
    )],
    hdr: bool,
    resize_available: bool,
) -> Vec<DecodeOptions> {
    let mut color = ferment_engine_media::ColorInfo::default();
    if hdr {
        color.transfer = ferment_engine_media::TransferCharacteristic::Hlg;
        color.bits_per_channel = Some(10);
    }
    let observed = Arc::new(Observed {
        color,
        requests: Mutex::new(vec![]),
        resize_available,
    });
    let mut backends = platform_encode_backends();
    backends.video = observed.clone();
    let variants: Vec<_> = presets
        .iter()
        .enumerate()
        .map(|(n, (preset, thumb_strip))| VideoEncodingVariant {
            id: n.to_string(),
            output_path: format!("unused-{n}.mp4").into(),
            preset: preset.clone(),
            thumb_strip: *thumb_strip,
        })
        .collect();
    let result = futures_executor::block_on(shared_decode_encode(
        &backends,
        Path::new("observed.mp4"),
        &variants,
        &ProgressSink::noop(),
        ProgressWindow::FULL,
        None,
        &mut CompletedVariantOutputs::default(),
    ));
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("decoder observation complete")
    );
    observed.requests.lock().unwrap().clone()
}
#[test]
fn sdr_feed_requests_only_its_coded_decode_extent() {
    let requested = requests(&[(VideoEncoder::ai_preset(), None)], false, true);
    assert_eq!(requested[0].output_size, Some((640, 360)));
    assert_eq!(
        requested[0].acceleration,
        ferment_engine_media::DecodeAcceleration::Automatic
    );
}
#[test]
fn independent_variants_and_thumbnails_keep_the_original_decode_extent() {
    let mut proxy = VideoEncoder::proxy_preset();
    proxy.max_width = 720;
    let requested = requests(
        &[(VideoEncoder::ai_preset(), None), (proxy, None)],
        false,
        true,
    );
    assert_eq!(requested, vec![DecodeOptions::default()]);
    let requested = requests(
        &[(
            VideoEncoder::ai_preset(),
            Some(ThumbStripOptions {
                interval: 1.0,
                max_width: 1000,
                max_height: 1000,
            }),
        )],
        false,
        true,
    );
    assert_eq!(requested, vec![DecodeOptions::default()]);
}
#[test]
fn full_size_and_hdr_passes_keep_the_existing_color_decoder() {
    for (preset, hdr) in [
        (VideoEncoder::proxy_preset(), false),
        (VideoEncoder::ai_preset(), true),
    ] {
        assert_eq!(
            requests(&[(preset, None)], hdr, true),
            vec![DecodeOptions::default()]
        );
    }
}
#[test]
fn optional_decoder_resize_can_fall_back_without_changing_acceleration_policy() {
    let requested = requests(&[(VideoEncoder::ai_preset(), None)], false, false);
    assert_eq!(requested.len(), 2);
    assert_eq!(requested[0].output_size, Some((640, 360)));
    assert_eq!(requested[1], DecodeOptions::default());
}
