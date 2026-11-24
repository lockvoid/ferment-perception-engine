use super::*;
use async_trait::async_trait;
use ferment_engine_core::VideoObjectPropagationDirection::{Forward, Reverse};
use ferment_engine_media::{
    ColorInfo, CpuFrame, CpuPixelFormat, FrameBuffer, MediaTime, VideoFrame,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct State {
    cancellation: ProgressSink,
    reads: AtomicUsize,
    seeks: AtomicUsize,
    drops: AtomicUsize,
    cancel_seek: bool,
    cancel_read: usize,
}
#[derive(Debug)]
struct Decoder {
    state: Arc<State>,
    next: usize,
}
impl Drop for Decoder {
    fn drop(&mut self) {
        self.state.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[async_trait(?Send)]
impl DecodeSession for Decoder {
    async fn seek(&mut self, target: MediaTime) -> ferment_engine_media::Result<()> {
        self.state.seeks.fetch_add(1, Ordering::SeqCst);
        self.next = (target
            .rescale(ferment_engine_media::TimeBase::MICROSECONDS)
            .value
            / 33_333) as usize;
        if self.state.cancel_seek {
            self.state.cancellation.cancel();
        }
        Ok(())
    }
    async fn decode_next(&mut self) -> ferment_engine_media::Result<Option<VideoFrame>> {
        let read = self.state.reads.fetch_add(1, Ordering::SeqCst) + 1;
        if read == self.state.cancel_read {
            self.state.cancellation.cancel();
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(VideoFrame {
            width: 1,
            height: 1,
            pts: MediaTime::new(
                index as i64 * 33_333,
                ferment_engine_media::TimeBase::MICROSECONDS,
            ),
            duration: None,
            color: ColorInfo::default(),
            buffer: FrameBuffer::Cpu(CpuFrame::new(
                1,
                1,
                3,
                CpuPixelFormat::Rgb24,
                vec![index as u8; 3],
            )?),
        }))
    }
}
#[derive(Debug)]
struct Backend(Arc<State>);
#[async_trait(?Send)]
impl ferment_engine_media::VideoIo for Backend {
    async fn probe(
        &self,
        _: &Path,
    ) -> ferment_engine_media::Result<ferment_engine_media::MediaInfo> {
        unreachable!("timeline already indexed")
    }
    async fn open_decoder(
        &self,
        _: &Path,
        _: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        Ok(Box::new(Decoder {
            state: self.0.clone(),
            next: 0,
        }))
    }
    async fn open_encoder(
        &self,
        _: ferment_engine_media::EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn ferment_engine_media::EncodeSession>> {
        unreachable!("decode only")
    }
}
fn source(cancel_seek: bool, cancel_read: usize) -> (EdgeTamFrameSource, Arc<State>) {
    let state = Arc::new(State {
        cancellation: ProgressSink::noop(),
        reads: AtomicUsize::new(0),
        seeks: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        cancel_seek,
        cancel_read,
    });
    let mut timeline = FrameTimelineWriter::new(&std::env::temp_dir()).unwrap();
    for index in 0..3 {
        timeline
            .append(MediaTime::new(
                index * 33_333,
                ferment_engine_media::TimeBase::MICROSECONDS,
            ))
            .unwrap();
    }
    (
        EdgeTamFrameSource {
            backend: Arc::new(Backend(state.clone())),
            path: "controlled-decoder".into(),
            stream_index: 0,
            source_info: VideoObjectSegmentationSourceInfo {
                frame_count: 3,
                frame_rate: 30.0,
                width: 1,
                height: 1,
            },
            timeline: timeline.finish().unwrap(),
            orientation: super::super::frame_orientation::DisplayOrientation::new(1, 1, None)
                .unwrap(),
            reverse_frames: Vec::new(),
            decoder: None,
            cursor: None,
            origin: MediaTime::ZERO,
            seeks: 0,
        },
        state,
    )
}
fn assert_cancelled(result: Result<EdgeTamRgbFrame>, source: &EdgeTamFrameSource) {
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(!source.has_decoder_resources());
    assert_eq!(source.cached_reverse_byte_count(), 0);
}
/// Kill: admit work before checking its already-cancelled token.
#[test]
fn an_already_cancelled_read_does_not_open_or_seek_the_decoder() {
    let (mut source, state) = source(false, usize::MAX);
    state.cancellation.cancel();
    let result =
        futures_executor::block_on(source.frame_with_cancellation(1, state.cancellation.clone()));
    assert_cancelled(result, &source);
    assert_eq!(state.seeks.load(Ordering::SeqCst), 0);
    assert_eq!(state.reads.load(Ordering::SeqCst), 0);
    assert_eq!(state.drops.load(Ordering::SeqCst), 0);
}
/// Kill: begin decoding after a cancellation that arrived during a native seek.
#[test]
fn cancellation_during_seek_drains_and_drops_without_reading_a_frame() {
    let (mut source, state) = source(true, usize::MAX);
    let result =
        futures_executor::block_on(source.frame_with_cancellation(1, state.cancellation.clone()));
    assert_cancelled(result, &source);
    assert_eq!(state.seeks.load(Ordering::SeqCst), 1);
    assert_eq!(state.reads.load(Ordering::SeqCst), 0);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}
/// Kill: return a frame whose native decode completed after cancellation.
#[test]
fn cancellation_during_a_random_or_forward_read_discards_the_decoded_frame() {
    for sequential in [false, true] {
        let (mut source, state) = source(false, 1);
        let result = if sequential {
            futures_executor::block_on(source.sequential_frame_with_cancellation(
                1,
                Forward,
                state.cancellation.clone(),
            ))
        } else {
            futures_executor::block_on(
                source.frame_with_cancellation(1, state.cancellation.clone()),
            )
        };
        assert_cancelled(result, &source);
        assert_eq!(state.reads.load(Ordering::SeqCst), 1);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    }
}
/// Kill: keep decoding a reverse batch, retain a partial batch, or return its stale tail.
#[test]
fn cancellation_partway_through_a_reverse_batch_releases_every_owned_frame() {
    let (mut source, state) = source(false, 2);
    let result = futures_executor::block_on(source.sequential_frame_with_cancellation(
        2,
        Reverse,
        state.cancellation.clone(),
    ));
    assert_cancelled(result, &source);
    assert_eq!(state.reads.load(Ordering::SeqCst), 2);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}
/// Kill: check only decoder reads while allowing a cancelled cache hit to return a frame.
#[test]
fn cancellation_discards_an_already_decoded_reverse_batch() {
    let (mut source, state) = source(false, usize::MAX);
    let first = futures_executor::block_on(source.sequential_frame_with_cancellation(
        2,
        Reverse,
        state.cancellation.clone(),
    ))
    .unwrap();
    assert_eq!(first.pixels, [2, 2, 2]);
    assert_eq!(source.cached_reverse_byte_count(), 6);
    state.cancellation.cancel();
    let result = futures_executor::block_on(source.sequential_frame_with_cancellation(
        1,
        Reverse,
        state.cancellation.clone(),
    ));
    assert_cancelled(result, &source);
    assert_eq!(state.reads.load(Ordering::SeqCst), 3);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}
