use super::*;
use async_trait::async_trait;
use ferment_engine_core::VideoObjectPropagationDirection::Forward;
use ferment_engine_media::{EncodeSession, EncoderSettings, MediaInfo, MediaKind, VideoIo};
use futures_executor::block_on;
use std::{path::PathBuf, sync::Arc};

fn fixture(name: &str) -> PathBuf {
    let p = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
            .expect("required original current fixtures"),
    )
    .join(name);
    assert!(p.is_file(), "missing original fixture {}", p.display());
    p
}
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p =
            std::env::temp_dir().join(format!("edgetam-boundary-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Kill: create a video session for audio, invent a missing sample, accept an
/// out-of-range index, or leave a named temporary index after decode failure.
#[test]
#[ignore = "requires original audio/one-frame fixtures and the actual platform decoder"]
fn audio_only_one_frame_invalid_indices_and_missing_mdat_fail_at_the_real_boundary() {
    let scratch = Scratch::new();
    let audio = fixture("Audio/synthetic-speech-1.wav");
    let error = block_on(EdgeTamFrameSource::new_in(
        crate::video_backend(),
        &audio,
        &scratch.0,
    ))
    .unwrap_err();
    assert!(
        matches!(&error,EngineError::Decode(message) if message.contains("no video stream")),
        "{error}"
    );
    let one = fixture("EdgeTAM/one-frame-16x16.mp4");
    let original = std::fs::read(&one).unwrap();
    let mut source = block_on(EdgeTamFrameSource::new_in(
        crate::video_backend(),
        &one,
        &scratch.0,
    ))
    .unwrap();
    assert_eq!(source.source_info().frame_count, 1);
    assert_eq!(source.presentation_time(0).unwrap(), 0.);
    let frame = block_on(source.frame(0)).unwrap();
    assert_eq!((frame.frame_index, frame.width, frame.height), (0, 16, 16));
    assert_eq!(frame.pixels.len(), 16 * 16 * 3);
    // Rust's public index is unsigned; usize::MAX is the refused boundary
    // corresponding to a negative index arriving through an unchecked cast.
    assert!(source.presentation_time(usize::MAX).is_err());
    let error = block_on(source.frame(1)).unwrap_err();
    assert_eq!(
        error.to_string(),
        EngineError::from(super::super::error::EdgeTamError::InvalidFrameIndex {
            index: 1,
            frame_count: 1
        })
        .to_string()
    );
    assert!(!source.has_decoder_resources());
    drop(source);
    let marker = original
        .windows(4)
        .position(|w| w == b"mdat")
        .expect("original fixture's mdat marker")
        + 4;
    let corrupt = scratch.0.join("missing-mdat.mp4");
    std::fs::write(&corrupt, &original[..marker]).unwrap();
    // Native's exact timeline indexes presented decoded samples, so absent
    // media is refused during construction. Swift indexes container samples
    // and refuses at frame(0). Both refuse; native must not fabricate a sample
    // merely to postpone the same error until the Apple API's stage.
    let error = block_on(EdgeTamFrameSource::new_in(
        crate::video_backend(),
        &corrupt,
        &scratch.0,
    ))
    .unwrap_err();
    assert!(
        matches!(error, EngineError::Decode(_) | EngineError::VideoIo(_)),
        "{error}"
    );
    let remaining: Vec<_> = std::fs::read_dir(&scratch.0)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(remaining, [std::ffi::OsString::from("missing-mdat.mp4")]);
    assert_eq!(std::fs::read(&one).unwrap(), original);
}

#[derive(Debug)]
struct MatrixVideo {
    inner: VideoBackend,
    matrix: Option<[i32; 9]>,
}
#[async_trait(?Send)]
impl VideoIo for MatrixVideo {
    async fn probe(&self, path: &Path) -> ferment_engine_media::Result<MediaInfo> {
        let mut info = self.inner.probe(path).await?;
        for stream in &mut info.streams {
            if stream.kind == MediaKind::Video {
                stream.display_matrix = self.matrix;
            }
        }
        Ok(info)
    }
    async fn open_decoder(
        &self,
        path: &Path,
        index: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.inner.open_decoder(path, index).await
    }
    async fn open_coded_decoder(
        &self,
        path: &Path,
        index: u32,
    ) -> ferment_engine_media::Result<Box<dyn DecodeSession>> {
        self.inner.open_coded_decoder(path, index).await
    }
    async fn open_encoder(
        &self,
        settings: EncoderSettings,
    ) -> ferment_engine_media::Result<Box<dyn EncodeSession>> {
        self.inner.open_encoder(settings).await
    }
}
fn open_with_matrix(path: &Path, matrix: Option<[i32; 9]>) -> EdgeTamFrameSource {
    block_on(EdgeTamFrameSource::new(
        Arc::new(MatrixVideo {
            inner: crate::video_backend(),
            matrix,
        }),
        path,
    ))
    .unwrap()
}

/// Kill: classify a transform correctly but omit it in the real decoder path,
/// reuse stale RGB across a forward walk, or retain a duration-sized RGB batch.
#[test]
#[ignore = "requires the original rotated-48x32 fixture and actual platform decode"]
fn actual_forward_decode_applies_display_coordinates_and_retains_no_rgb_batch() {
    let path = fixture("EdgeTAM/rotated-48x32.mp4");
    let mut raw = open_with_matrix(&path, None);
    let identity_info = raw.source_info();
    assert_eq!(
        (
            identity_info.width,
            identity_info.height,
            identity_info.frame_count
        ),
        (48, 32, 30)
    );
    for (matrix, turns, mirror) in [
        (
            [0, -65536, 0, 65536, 0, 0, 48 * 65536, 0, 1 << 30],
            1,
            false,
        ),
        (
            [-65536, 0, 0, 0, -65536, 0, 48 * 65536, 32 * 65536, 1 << 30],
            2,
            false,
        ),
        (
            [0, 65536, 0, -65536, 0, 0, 0, 32 * 65536, 1 << 30],
            3,
            false,
        ),
        ([-65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30], 0, true),
    ] {
        let mut displayed = open_with_matrix(&path, Some(matrix));
        let (width, height) = if turns % 2 == 1 { (32, 48) } else { (48, 32) };
        assert_eq!(
            (
                displayed.source_info().width,
                displayed.source_info().height
            ),
            (width, height)
        );
        for index in 0..30 {
            let reference = block_on(raw.frame(index)).unwrap();
            let actual = block_on(displayed.sequential_frame(index, Forward)).unwrap();
            assert_eq!((actual.width, actual.height), (width, height));
            assert_eq!(
                actual.presentation_timestamp,
                reference.presentation_timestamp
            );
            for y in 0..32 {
                for x in 0..48 {
                    let (dx, dy) = if mirror {
                        (47 - x, y)
                    } else {
                        match turns {
                            1 => (y, 47 - x),
                            2 => (47 - x, 31 - y),
                            3 => (31 - y, x),
                            _ => unreachable!(),
                        }
                    };
                    assert_eq!(
                        &actual.pixels[(dy * width + dx) * 3..(dy * width + dx) * 3 + 3],
                        &reference.pixels[(y * 48 + x) * 3..(y * 48 + x) * 3 + 3]
                    );
                }
            }
            assert_eq!(displayed.cached_reverse_byte_count(), 0);
        }
        assert_eq!(
            displayed.seek_count(),
            1,
            "forward pull must reuse one positioned decoder"
        );
        displayed.cancel_decoder_resources();
        assert!(!displayed.has_decoder_resources());
    }
}
