//! Regression for a saved live recording that imported but could not replay:
//! the shipped whitelist included Matroska without its lossless FFV1 decoder.
use ferment_engine_media::platform::libav::LibavVideoIo;
use ferment_engine_media::{DecodeOptions, DecodeOutput, FrameBuffer, MediaInput, VideoIo};
use futures_executor::block_on;
use std::io::{self, Cursor, Read, Seek, SeekFrom};

struct FixtureBytes(Cursor<Vec<u8>>);
impl Read for FixtureBytes {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.0.read(bytes)
    }
}
impl Seek for FixtureBytes {
    fn seek(&mut self, at: SeekFrom) -> io::Result<u64> {
        self.0.seek(at)
    }
}
impl MediaInput for FixtureBytes {
    fn byte_len(&self) -> u64 {
        self.0.get_ref().len() as u64
    }
    fn validate(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Kill: remove `ffv1` from provision-ffmpeg's decoder whitelist. Opening the
/// independent CLI-encoded fixture then fails with AV_CODEC_ID_FFV1. Dropping
/// alpha or swapping RGB channels also fails the literal pixel comparison.
#[test]
fn saved_lossless_rgba_recording_decodes_exact_color_and_alpha() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ffv1-rgba-8x6.mkv");
    let mut decoder = block_on(LibavVideoIo.open_decoder_with(
        &path,
        0,
        DecodeOptions {
            output: DecodeOutput::CpuBgra8,
            ..Default::default()
        },
    ))
    .expect("the shipped decoder opens a saved FFV1 recording");
    for _ in 0..2 {
        let frame = block_on(decoder.decode_next()).unwrap().unwrap();
        assert_eq!((frame.width, frame.height), (8, 6));
        let FrameBuffer::Cpu(cpu) = frame.buffer else {
            panic!("explicit CPU BGRA output was ignored");
        };
        let pixels = cpu.packed_rgba8().unwrap();
        for (i, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
            let expected = match (i % 8 < 4, i / 8 < 3) {
                (true, true) => [255, 0, 0, 255],
                (false, true) => [0, 255, 0, 128],
                (true, false) => [0, 0, 255, 64],
                (false, false) => [17, 33, 65, 0],
            };
            assert_eq!(*pixel, expected, "pixel {i}");
        }
    }
    assert!(block_on(decoder.decode_next()).unwrap().is_none());
}

/// Kill: omit AVI from the build or checked-stream whitelist, or quantize PTS.
/// Literal frame-index colors and alpha were independently CLI-encoded;
/// both exact rational timestamps and bytes must survive the decoder.
#[test]
fn lossless_cfr_recordings_keep_their_exact_integer_and_fractional_frame_grid() {
    for (numerator, denominator) in [(60, 1), (30_000, 1001)] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "tests/fixtures/ffv1-clock-{numerator}-{denominator}.avi"
        ));
        for checked_stream in [false, true] {
            let options = DecodeOptions {
                output: DecodeOutput::CpuBgra8,
                ..Default::default()
            };
            let mut decoder = if checked_stream {
                LibavVideoIo.open_stream(
                    Box::new(FixtureBytes(Cursor::new(std::fs::read(&path).unwrap()))),
                    0,
                    options,
                )
            } else {
                block_on(LibavVideoIo.open_decoder_with(&path, 0, options))
            }
            .expect("the shipped decoder opens the exact-clock FFV1 recording");
            for index in 0..31 {
                let frame = block_on(decoder.decode_next()).unwrap().unwrap();
                assert_eq!((frame.width, frame.height), (16, 8));
                assert_eq!(
                    i128::from(frame.pts.value)
                        * i128::from(frame.pts.time_base.numerator)
                        * i128::from(numerator),
                    i128::from(index)
                        * i128::from(denominator)
                        * i128::from(frame.pts.time_base.denominator),
                    "frame {index} at {numerator}/{denominator} FPS"
                );
                let FrameBuffer::Cpu(cpu) = frame.buffer else {
                    panic!("explicit CPU BGRA output was ignored");
                };
                for (pixel, rgba) in cpu
                    .packed_rgba8()
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                {
                    assert_eq!(
                        *rgba,
                        [
                            index as u8 * 7,
                            255 - index as u8 * 7,
                            index as u8 * 3,
                            (pixel % 4) as u8 * 85
                        ],
                        "frame {index}, pixel {pixel}"
                    );
                }
            }
            assert!(block_on(decoder.decode_next()).unwrap().is_none());
        }
    }
}
