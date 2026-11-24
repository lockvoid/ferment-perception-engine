//! Literal planes encoded by the independent FFmpeg CLI, not our writer.
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

/// Kill: omit NUT/rawvideo support or materialize source float pixels as BGRA8.
/// Values include negative/HDR RGB and hidden color at exactly zero alpha.
#[test]
fn native_float_recording_retains_literal_channels_and_rational_pts() {
    for (fixture, count) in [
        ("linear-rgba32f-16x8.nut", 31),
        ("linear-rgba32f-one.nut", 1),
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        for checked_stream in [false, true] {
            let input = || {
                Box::new(FixtureBytes(Cursor::new(std::fs::read(&path).unwrap())))
                    as Box<dyn MediaInput>
            };
            let info = if checked_stream {
                ferment_engine_media::platform::libav::probe_stream(input())
            } else {
                block_on(LibavVideoIo.probe(&path))
            }
            .unwrap();
            let duration = info.duration.expect("complete presentation extent");
            assert_eq!(
                i128::from(duration.value) * i128::from(duration.time_base.numerator) * 30000,
                i128::from(count) * 1001 * i128::from(duration.time_base.denominator),
                "last frame duration was omitted"
            );
            let mut decoder = if checked_stream {
                LibavVideoIo.open_stream(
                    input(),
                    0,
                    DecodeOptions {
                        output: DecodeOutput::CpuRgba32Float,
                        ..Default::default()
                    },
                )
            } else {
                block_on(LibavVideoIo.open_decoder_with(&path, 0, DecodeOptions::default()))
            }
            .expect("native float recording opens");
            for index in 0..count {
                let frame = block_on(decoder.decode_next()).unwrap().unwrap();
                assert_eq!((frame.width, frame.height), (16, 8));
                assert_eq!(
                    i128::from(frame.pts.value) * i128::from(frame.pts.time_base.numerator) * 30000,
                    i128::from(index) * 1001 * i128::from(frame.pts.time_base.denominator)
                );
                let FrameBuffer::Cpu(cpu) = frame.buffer else {
                    panic!("float source must remain CPU float");
                };
                assert_eq!(cpu.format.bytes_per_pixel(), 16, "source was quantized");
                for y in 0..8 {
                    for x in 0..16 {
                        let expected: [f32; 4] = match (x + y + index) % 4 {
                            0 => [2., -0.5, 8., 0.25],
                            1 => [-2., 4., 0.125, 0.],
                            2 => [0., 0.25, 16., 1.],
                            _ => [0.03125, 3., -0.25, 0.5],
                        };
                        for (c, value) in expected.iter().enumerate() {
                            let at = y as usize * cpu.stride + x as usize * 16 + c * 4;
                            assert_eq!(
                                u32::from_le_bytes(cpu.data[at..at + 4].try_into().unwrap()),
                                value.to_bits(),
                                "frame {index}, ({x},{y}), channel {c}"
                            );
                        }
                    }
                }
            }
            assert!(block_on(decoder.decode_next()).unwrap().is_none());
        }
    }
}
