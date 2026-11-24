//! `platform::android` on a real device, graded against the Mac.
//!
//! Two halves, one file, because the seam is between them:
//!
//! * [`print_the_mac_reference`] runs on the Mac and prints what the libav
//!   backend decoded out of each fixture. It is `#[ignore]`d, and its output is
//!   pasted below as `MAC_*` literals with the command that produced it.
//! * everything under `#[cfg(target_os = "android")]` runs on the device
//!   through `native/perception-engine/device-test.sh`, on the SAME fixture
//!   bytes, and asserts against those literals.
//!
//! Nothing here derives an expected value through the code under test. The
//! fixtures are 932 KB of committed files; the Mac decoded them with FFmpeg and
//! the phone decodes them with MediaCodec, and the two answers have to agree.
//!
//! The laws are `tests/libav_decode.rs`'s and `tests/libav_audio.rs`'s, which
//! are the laws `Packages/FermentUtils/Tests` grades on iOS and
//! `packages/utils/src/androidTest` grades in Kotlin: every frame exactly once,
//! in presentation order, with the tail of the last GOP drained; a seek that
//! lands at or after its target and returns the same pixels a sequential decode
//! did; PCM that is contiguous and stops where the container says the program
//! does.

#![allow(clippy::uninlined_format_args)]

// -------------------------------------------------------------------------
// The Mac half: the numbers the device is graded against

/// The reference, printed by FFmpeg on the Mac.
///
/// ```text
/// cargo test -p ferment-engine-media --test android_media \
///     print_the_mac_reference -- --ignored --nocapture
/// ```
///
/// Re-run it and paste the output when a fixture changes. It is `#[ignore]`d
/// because it is a printer, not an assertion — the assertions are on the
/// device, against what it printed.
#[cfg(not(target_os = "android"))]
#[test]
#[ignore]
fn print_the_mac_reference() {
    use ferment_engine_media::platform::libav::{LibavAudioIo, LibavVideoIo};
    use ferment_engine_media::{AudioIo, MediaKind, VideoIo};
    use futures_executor::block_on;

    for name in VIDEO_FIXTURES.iter().map(|row| row.0) {
        let path = fixture(name);
        let info = block_on(LibavVideoIo.probe(&path)).unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = block_on(LibavVideoIo.open_decoder(&path, stream.index)).unwrap();
        let mut micros = Vec::new();
        while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
            micros.push((frame.pts.seconds() * 1_000_000.0).round() as i64);
        }
        println!(
            "VIDEO {name}: {} frames, {}x{}, pts_us {micros:?}",
            micros.len(),
            stream.width.unwrap(),
            stream.height.unwrap(),
        );
    }
    for name in AUDIO_FIXTURES.iter().map(|row| row.0) {
        let path = fixture(name);
        let info = block_on(LibavAudioIo.probe(&path)).unwrap();
        let stream = info
            .streams
            .iter()
            .find(|stream| stream.kind == MediaKind::Audio)
            .unwrap();
        let mut decoder = block_on(LibavAudioIo.open_audio_decoder(&path, stream.index)).unwrap();
        let mut total = 0usize;
        let mut peak = 0.0f32;
        while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
            total += frame.frame_count();
            for sample in frame.samples.iter() {
                peak = peak.max(sample.abs());
            }
        }
        println!(
            "AUDIO {name}: {total} samples, {} Hz, {} ch, peak {peak:.4}, timing {:?}",
            stream.sample_rate.unwrap(),
            stream.channels.unwrap(),
            stream.audio_timing,
        );
    }
}

/// Every video fixture, with the frame count and presentation timestamps the
/// Mac's libav decoder produced — `(name, frames, pts_us)`.
///
/// Verbatim from `print_the_mac_reference`, 2026-09-04, FFmpeg 8.1.2 on
/// aarch64-apple-darwin. Every clip is 12 frames of 64x48 at 12 fps; the MP4
/// and MOV families are timed in 1/12000 and round to a third of a microsecond,
/// the Matroska ones in milliseconds and do not.
const VIDEO_FIXTURES: &[(&str, usize, &[i64])] = &[
    (
        "h264-gop4.mp4",
        12,
        &[
            0, 83333, 166667, 250000, 333333, 416667, 500000, 583333, 666667, 750000, 833333,
            916667,
        ],
    ),
    (
        "h264-gop5.mov",
        12,
        &[
            0, 83333, 166667, 250000, 333333, 416667, 500000, 583333, 666667, 750000, 833333,
            916667,
        ],
    ),
    (
        "h264-gop6.mkv",
        12,
        &[
            0, 83000, 167000, 250000, 333000, 417000, 500000, 583000, 667000, 750000, 833000,
            917000,
        ],
    ),
    (
        "hevc-gop4.mp4",
        12,
        &[
            0, 83333, 166667, 250000, 333333, 416667, 500000, 583333, 666667, 750000, 833333,
            916667,
        ],
    ),
    (
        "hevc-gop5.mov",
        12,
        &[
            0, 83333, 166667, 250000, 333333, 416667, 500000, 583333, 666667, 750000, 833333,
            916667,
        ],
    ),
    (
        "hevc-gop6.mkv",
        12,
        &[
            0, 83000, 167000, 250000, 333000, 417000, 500000, 583000, 667000, 750000, 833000,
            917000,
        ],
    ),
    (
        "prores-standard.mov",
        12,
        &[
            0, 83333, 166667, 250000, 333333, 416667, 500000, 583333, 666667, 750000, 833333,
            916667,
        ],
    ),
];

/// Every audio fixture — `(name, mac_samples, android_samples, rate, channels,
/// peak)` — as the Mac's libav decoder produced it, from the same run.
///
/// Seven of the nine are the same number on both platforms, and the assertion
/// on those is exact equality. Two are not, and neither expected value is
/// derived through the code under test:
///
/// * `aac-lc-48000-stereo.mkv` — 49 008. **A regression pin, not a law, and
///   weaker than the rows around it: the number is `audible_samples` from the
///   probe, and the decoder clamps to that same field, so the assertion only
///   proves MediaCodec supplied at least that many samples.** libavcodec
///   applies Matroska's `CodecDelay` and hands back coded-minus-priming
///   (48 128); MediaExtractor carries the delay into no trim at all. Both are
///   longer than the one-second tone the fixture contains, so neither platform
///   is right here — Android keeps 1 024 samples of priming at the head.
/// * the two HE-AAC files — 44 100 is one second at the SBR output rate, which
///   is what these fixtures are: every fixture in this corpus is a one-second
///   tone, and the LC files at 44.1 kHz decode to exactly 44 100 on both
///   platforms. MediaCodec's HE decoder applies both trims; libavcodec applies
///   the head one and the MOV demuxer emits no end discard, so the Mac keeps
///   4 988 samples of the encoder's padding. **Android's answer is the correct
///   one here**, and the Mac's is the one carrying padding.
const AUDIO_FIXTURES: &[(&str, usize, usize, u32, u16, f32)] = &[
    ("aac-lc-48000-stereo.m4a", 48_000, 48_000, 48_000, 2, 0.7475),
    ("aac-lc-48000-mono.mp4", 48_000, 48_000, 48_000, 1, 0.4798),
    ("aac-lc-44100-stereo.mov", 44_100, 44_100, 44_100, 2, 0.5408),
    ("aac-lc-48000-stereo.mkv", 48_128, 49_008, 48_000, 2, 0.7475),
    (
        "mp3-cbr-44100-stereo.mp3",
        46_080,
        46_080,
        44_100,
        2,
        0.4286,
    ),
    (
        "mp3-vbr-44100-stereo.mp3",
        44_100,
        44_100,
        44_100,
        2,
        0.4572,
    ),
    ("aac-he-44100-stereo.m4a", 49_088, 44_100, 44_100, 2, 0.4946),
    (
        "aac-he-v2-44100-stereo.m4a",
        49_088,
        44_100,
        44_100,
        2,
        0.4550,
    ),
    ("aac-lc-44100-mono.mp4", 44_100, 44_100, 44_100, 1, 0.5132),
];

/// A fixture's path.
///
/// `CARGO_MANIFEST_DIR` does not exist on a phone, so the device suite is told
/// where its fixtures are; `device-test.sh` exports it after pushing them.
fn fixture(name: &str) -> std::path::PathBuf {
    match std::env::var("FERMENT_MEDIA_FIXTURES") {
        Ok(directory) => std::path::PathBuf::from(directory).join(name),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated")
            .join(name),
    }
}

// -------------------------------------------------------------------------
// The device half

#[cfg(target_os = "android")]
mod device {
    use super::{AUDIO_FIXTURES, VIDEO_FIXTURES, fixture};

    use ferment_engine_media::platform::android::{AndroidAudioIo, AndroidMuxer, AndroidVideoIo};
    use ferment_engine_media::platform::libav::PacketReader;
    use ferment_engine_media::{
        AudioEncoderSettings, AudioIo, Codec, DecodeOptions, DecodeSession, EncoderSettings,
        FrameBuffer, MediaKind, MediaTime, MuxFormat, MuxSettings, Muxer, PcmAudioFrame, TimeBase,
        VideoFrame, VideoIo, VideoMuxTrack, demux,
    };
    use futures_executor::block_on;
    use std::path::{Path, PathBuf};

    /// The Mac's answer for one fixture.
    fn reference(name: &str) -> (usize, &'static [i64]) {
        VIDEO_FIXTURES
            .iter()
            .find(|row| row.0 == name)
            .map(|row| (row.1, row.2))
            .unwrap_or_else(|| panic!("{name} has no Mac reference"))
    }

    fn decode_all(path: &Path, options: DecodeOptions) -> Vec<VideoFrame> {
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(path))
            .unwrap_or_else(|error| panic!("probe {}: {error}", path.display()));
        let stream = info
            .video_streams()
            .next()
            .unwrap_or_else(|| panic!("no video stream in {}", path.display()))
            .index;
        let mut decoder = block_on(backend.open_decoder_with(path, stream, options))
            .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
        drain(&mut *decoder)
    }

    fn drain(decoder: &mut dyn DecodeSession) -> Vec<VideoFrame> {
        let mut frames = Vec::new();
        while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
            frames.push(frame);
        }
        frames
    }

    fn micros(frame: &VideoFrame) -> i64 {
        (frame.pts.seconds() * 1_000_000.0).round() as i64
    }

    fn rgb_bytes(frame: &VideoFrame) -> Vec<u8> {
        frame.to_cpu_rgb().unwrap().data.as_ref().to_vec()
    }

    /// A scratch path that removes itself.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(suffix: &str) -> Self {
            Self(std::env::temp_dir().join(format!("ferment-e1-{}-{suffix}", std::process::id())))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    // ---------------------------------------------------------------------
    // Decode

    /// THE law: every frame the Mac decoded, in the order and at the times the
    /// Mac decoded it — including the twelfth, which only comes out if the
    /// decoder is drained at end of stream. A backend that forgets the drain
    /// loses the tail of every GOP in every file, silently.
    ///
    /// H.264 and HEVC go through MediaCodec; ProRes has no Android decoder and
    /// takes the libav fallback, which is the point of
    /// [`the_libav_fallback_decodes_what_mediacodec_refuses`] below and is
    /// asserted here only as "the file still decodes".
    ///
    /// KILL: drop the end-of-stream flag in `MediaCodecDecoder::feed` — the
    /// decoder never drains and the pass ends in the stall error rather than
    /// at twelve frames. Dropping the released-timing queue is NOT a kill on
    /// the happy path: it holds exactly one entry, pushed and popped with
    /// nothing in between, so removing it is the identity.
    #[test]
    fn decodes_the_fixture_matrix_at_the_macs_own_timestamps() {
        for (name, frames_expected, pts_expected) in VIDEO_FIXTURES {
            let frames = decode_all(&fixture(name), DecodeOptions::default());
            assert_eq!(frames.len(), *frames_expected, "{name}: frame count");
            let pts: Vec<i64> = frames.iter().map(micros).collect();
            // Within one microsecond, not equal to the digit: a 12 fps frame
            // in a 1/12000 container is 166 666.67 µs, and MediaExtractor
            // truncates the conversion where libav rounds it. Both are the
            // same instant at the resolution the platform's own unit has, and
            // this still catches everything worth catching — a seek that
            // landed on the wrong frame differs by 83 333.
            assert_eq!(pts.len(), pts_expected.len(), "{name}");
            for (index, (measured, expected)) in pts.iter().zip(pts_expected.iter()).enumerate() {
                assert!(
                    (measured - expected).abs() <= 1,
                    "{name} frame {index}: {measured} µs against the Mac's {expected} µs \
                     (whole list {pts:?})",
                );
            }
            assert!(
                pts.windows(2).all(|pair| pair[0] < pair[1]),
                "{name} is out of presentation order: {pts:?}",
            );
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.width == 64 && frame.height == 48),
                "{name}: {:?}",
                frames
                    .iter()
                    .map(|frame| (frame.width, frame.height))
                    .collect::<Vec<_>>(),
            );
            for frame in &frames {
                let cpu = frame.to_cpu_rgb().expect("download");
                assert_eq!(cpu.data.len(), 64 * 48 * 3, "{name}");
            }
        }
    }

    /// A decoded picture stays in graphics memory: the `AImage` MediaCodec
    /// rendered, holding an `AHardwareBuffer`. A `FrameBuffer::Cpu` for a
    /// consumer that releases each frame means every frame is paying a
    /// full-resolution readback — the same law
    /// `libav_decode.rs::macos_frames_stay_on_the_gpu` states for
    /// VideoToolbox.
    ///
    /// This decodes the way the workspace's own consumer does
    /// (`VideoSource::decode_rgb`): take a frame, use it, drop it. Only the
    /// MediaCodec fixtures — ProRes decodes through libavcodec, which has no
    /// Android hardware frame to hand back.
    ///
    /// KILL: pass a null surface to `AMediaCodec_configure` — every frame
    /// arrives as a CPU buffer and the hardware buffer is gone.
    #[test]
    fn frames_stay_in_graphics_memory_for_a_consumer_that_releases_them() {
        for (name, expected, ..) in VIDEO_FIXTURES
            .iter()
            .filter(|row| !row.0.contains("prores"))
        {
            let path = fixture(name);
            let backend = AndroidVideoIo;
            let info = block_on(backend.probe(&path)).unwrap();
            let stream = info.video_streams().next().unwrap().index;
            let mut decoder = block_on(backend.open_decoder(&path, stream)).unwrap();
            let mut seen = 0usize;
            while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
                let FrameBuffer::Android(image) = &frame.buffer else {
                    panic!("{name} frame {seen} is a CPU buffer — the image reader was not used");
                };
                assert!(
                    image.hardware_buffer().is_some(),
                    "{name} frame {seen}: the image carries no AHardwareBuffer",
                );
                seen += 1;
            }
            assert_eq!(seen, *expected, "{name}");
        }
    }

    /// A consumer that holds every frame gets pixels instead of an error.
    ///
    /// `AImageReader`'s pool is four deep and `acquireNextImage` refuses past
    /// it, while `decode_next` hands out an owned frame with no lifetime on
    /// it. The backend never gives away the last slot: once the caller is
    /// holding the pool, the picture is downloaded and the image released. The
    /// residency changes, the decode does not.
    ///
    /// KILL: hand out `FrameBuffer::Android` unconditionally. The fifth frame
    /// of every clip fails with
    /// `Platform { operation: "AImageReader_acquireNextImage", status: -30002 }`
    /// — which is the defect this test was written after, found on `cuts-36`.
    #[test]
    fn a_consumer_that_holds_every_frame_gets_pixels_rather_than_an_exhausted_pool() {
        let name = "h264-gop4.mp4";
        let (expected, _) = reference(name);
        let hoarded = decode_all(&fixture(name), DecodeOptions::default());
        assert_eq!(
            hoarded.len(),
            expected,
            "{name}: the hoarding pass is short"
        );

        let hardware = hoarded
            .iter()
            .filter(|frame| matches!(frame.buffer, FrameBuffer::Android(_)))
            .count();
        println!(
            "MEASURED hoarding pass: {hardware}/{} frames in graphics memory",
            hoarded.len()
        );
        assert!(
            hardware < hoarded.len(),
            "the pool is only four deep; a 12-frame hoard cannot hold every image",
        );

        // And the pixels are the same either way, which is the whole claim:
        // decoding one frame at a time (every frame hardware-resident) gives
        // the same picture as hoarding (most of them downloaded).
        let path = fixture(name);
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info.video_streams().next().unwrap().index;
        let mut decoder = block_on(backend.open_decoder(&path, stream)).unwrap();
        let mut index = 0usize;
        while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
            assert_eq!(
                rgb_bytes(&frame),
                rgb_bytes(&hoarded[index]),
                "frame {index} differs between a prompt and a hoarding consumer",
            );
            index += 1;
        }
        assert_eq!(index, expected);
    }

    /// What MediaCodec refuses still decodes. No Android device has ever
    /// shipped a ProRes decoder, and the six `libav*.so` are already in the
    /// process — rejecting the file would be a choice, not a limit.
    ///
    /// KILL: make `AndroidVideoIo::open_decoder_with` propagate
    /// `UnsupportedCodec` instead of handing it to `LibavVideoIo`. Giving
    /// `mime_for` a `"video/prores"` arm is NOT a kill — `MediaCodec::decoder`
    /// maps a null codec back to `UnsupportedCodec`, so the fallback still
    /// fires and this stays green.
    #[test]
    fn the_libav_fallback_decodes_what_mediacodec_refuses() {
        let name = "prores-standard.mov";
        let (expected, pts_expected) = reference(name);
        let frames = decode_all(&fixture(name), DecodeOptions::default());
        assert_eq!(frames.len(), expected, "{name}");
        let pts: Vec<i64> = frames.iter().map(micros).collect();
        assert_eq!(pts, pts_expected.to_vec(), "{name}");
        assert!(
            frames
                .iter()
                .all(|frame| matches!(frame.buffer, FrameBuffer::Cpu(_))),
            "the libav fallback has no Android hardware frame to hand back",
        );
    }

    /// The seek contract: land on the target, and land on the same pixels every
    /// time. Byte equality against a sequential decode is the only assertion
    /// that catches a seek which lands one frame early and returns a picture
    /// that is merely plausible.
    ///
    /// KILL: seek the extractor without flushing the codec — the first picture
    /// back is whatever was still in the pipeline from the old position, and
    /// its bytes differ.
    #[test]
    fn exact_seeks_are_byte_deterministic_against_sequential_decode() {
        let path = fixture("h264-gop4.mp4");
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = block_on(backend.open_decoder(&path, stream.index)).unwrap();
        let sequential = drain(&mut *decoder);
        assert_eq!(sequential.len(), 12);
        let reference: Vec<(MediaTime, Vec<u8>)> = sequential
            .iter()
            .map(|frame| (frame.pts, rgb_bytes(frame)))
            .collect();

        // Deliberately unordered, and revisiting frames: a decoder that only
        // ever seeks forward, or that caches its last answer, passes an
        // in-order walk.
        for index in [7usize, 1, 10, 4, 0, 9, 2, 11, 5, 8] {
            let (target, expected) = &reference[index];
            block_on(decoder.seek(*target)).unwrap();
            let first = block_on(decoder.decode_next())
                .unwrap()
                .unwrap_or_else(|| panic!("no frame after seeking to {index}"));
            assert_eq!(
                (first.pts.seconds() * 1_000_000.0).round() as i64,
                (target.seconds() * 1_000_000.0).round() as i64,
                "seek to frame {index} landed elsewhere",
            );
            let bytes = rgb_bytes(&first);
            assert_eq!(
                &bytes, expected,
                "seek to frame {index} decoded different pixels than a sequential decode",
            );

            block_on(decoder.seek(*target)).unwrap();
            let repeated = block_on(decoder.decode_next()).unwrap().unwrap();
            assert_eq!(bytes, rgb_bytes(&repeated), "repeated seek {index}");
        }
    }

    /// A seek never lands before its target. The frame source chases a playhead
    /// with these, and an early frame shows the wrong picture rather than
    /// failing.
    ///
    /// KILL: drop the `seek_target` discard in `decode_next` — the decoder
    /// starts at the sync sample the extractor landed on, which for a GOP of 6
    /// is up to five frames early.
    #[test]
    fn a_seek_never_lands_before_its_target() {
        let path = fixture("h264-gop6.mkv");
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info.video_streams().next().unwrap();
        let mut decoder = block_on(backend.open_decoder(&path, stream.index)).unwrap();
        let reference = drain(&mut *decoder);
        assert_eq!(reference.len(), 12);

        for frame in &reference {
            block_on(decoder.seek(frame.pts)).unwrap();
            let landed = block_on(decoder.decode_next())
                .unwrap()
                .expect("a frame after seeking");
            assert!(
                landed.pts.seconds() >= frame.pts.seconds() - 1e-9,
                "seek to {} landed at {}",
                frame.pts.seconds(),
                landed.pts.seconds(),
            );
        }
    }

    /// `DecodeOptions::output_size` asks the decoder for that exact size. The
    /// trait allows a backend that cannot to deliver the source size instead —
    /// consumers read `VideoFrame::width` — so what is asserted is the
    /// contract, not the wish: the frame is at one of the two sizes, and the
    /// picture it downloads matches the size it declares.
    ///
    /// The line it prints is the measurement the report carries: which of the
    /// two this device does.
    ///
    /// KILL: report `self.options.output_size` as the frame's dimensions
    /// instead of the image's — the declared size and the downloaded picture
    /// stop agreeing.
    #[test]
    fn output_size_is_honoured_or_declined_but_never_misdeclared() {
        let path = fixture("h264-gop4.mp4");
        let frames = decode_all(
            &path,
            DecodeOptions {
                output_size: Some((32, 32)),
                ..Default::default()
            },
        );
        assert_eq!(frames.len(), 12);
        let first = &frames[0];
        println!(
            "MEASURED output_size: asked 32x32, got {}x{}",
            first.width, first.height,
        );
        assert!(
            (first.width, first.height) == (32, 32) || (first.width, first.height) == (64, 48),
            "the decoder delivered {}x{}, which is neither the request nor the source",
            first.width,
            first.height,
        );
        for frame in &frames {
            let cpu = frame.to_cpu_rgb().unwrap();
            assert_eq!(
                (cpu.width, cpu.height),
                (frame.width, frame.height),
                "the frame declares a size its own pixels do not have",
            );
        }
        // Not merely present — a decoded picture. The fixtures are never a flat
        // field, so a decoder that produced grey would pass a length check.
        let cpu = frames[0].to_cpu_rgb().unwrap();
        let first_byte = cpu.data[0];
        assert!(
            cpu.data.iter().any(|byte| *byte != first_byte),
            "the decoded frame is a single flat colour",
        );
    }

    /// Opening a stream that is not video is a refusal, not a panic or an
    /// empty decode — `MediaKind` is the caller's to check, and the backend
    /// says so.
    #[test]
    fn opening_an_audio_stream_as_video_is_refused() {
        let path = fixture("synthetic-multitrack.mp4");
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let audio = info.audio_streams().next().expect("a multitrack fixture");
        let error = block_on(backend.open_decoder(&path, audio.index)).unwrap_err();
        assert!(
            error.to_string().contains("not video"),
            "unexpected error: {error}",
        );
    }

    /// Every stream of a multitrack file opens as its own track.
    ///
    /// What this does NOT prove is the ordinal mapping's kill. libavformat and
    /// `AMediaExtractor` number tracks independently, and `track_for_stream`
    /// matches on the ordinal WITHIN a kind rather than on the raw index —
    /// but for ISO-BMFF and Matroska both walk the container's own track list,
    /// so the two orders coincide and no fixture in this corpus can tell the
    /// safe mapping from the unsafe one. Said here rather than claimed as a
    /// kill line that does not exist (E0's D5).
    ///
    /// What it does prove: four streams, two of each kind, each opening as the
    /// kind the probe says it is, and both video tracks decoding.
    #[test]
    fn every_stream_of_a_multitrack_file_opens_as_its_own_track() {
        let path = fixture("synthetic-multitrack.mp4");
        let info = demux::probe(&path).unwrap();
        assert_eq!(info.video_streams().count(), 2, "fixture precondition");
        assert_eq!(info.audio_streams().count(), 2, "fixture precondition");

        let backend = AndroidVideoIo;
        for video in info.video_streams() {
            let mut decoder = block_on(backend.open_decoder(&path, video.index))
                .unwrap_or_else(|error| panic!("video stream {}: {error}", video.index));
            let frames = drain(&mut *decoder);
            assert!(
                !frames.is_empty(),
                "video stream {} decoded nothing",
                video.index
            );
            assert_eq!(
                (frames[0].width, frames[0].height),
                (video.width.unwrap(), video.height.unwrap()),
                "video stream {} came back at another track's size",
                video.index,
            );
        }
        for audio in info.audio_streams() {
            let mut decoder = block_on(AndroidAudioIo.open_audio_decoder(&path, audio.index))
                .unwrap_or_else(|error| panic!("audio stream {}: {error}", audio.index));
            let frame = block_on(decoder.decode_next())
                .expect("decode")
                .unwrap_or_else(|| panic!("audio stream {} decoded nothing", audio.index));
            assert_eq!(frame.sample_rate, audio.sample_rate.unwrap());
            // And the video door refuses it, rather than decoding noise.
            assert!(block_on(backend.open_decoder(&path, audio.index)).is_err());
        }
    }

    /// HDR colorimetry survives the decode, and the report learns whether this
    /// device produced 10-bit pictures for it.
    ///
    /// `synthetic-hdr-nclx.mp4` is HEVC tagged BT.2020 / PQ with NO declared
    /// bit depth, which is why the backend reads the transfer curve as well as
    /// the depth when it asks for a reader format. The AVD has no 10-bit
    /// decoder at all (`android-emulator-media-ceiling`: a Main10 decode there
    /// takes the process down with a native SIGSEGV), so what is asserted is
    /// what holds on any device — the file decodes and the frame carries the
    /// container's colour, unaltered — and what the device did with the depth
    /// is PRINTED for the report's table rather than asserted into a green
    /// that would be a lie on the emulator.
    ///
    /// KILL: stamp `ColorInfo::default()` on the frame instead of the stream's
    /// colour — an HDR import then plays back as washed-out SDR, which is the
    /// failure `libav_mux.rs::colour_tags_survive_the_container` exists for on
    /// the writing side.
    #[test]
    fn hdr_colour_survives_the_decode() {
        let path = fixture("synthetic-hdr-nclx.mp4");
        let info = demux::probe(&path).expect("probe");
        let stream = info.video_streams().next().expect("a video stream");
        assert_eq!(
            stream.color.transfer,
            ferment_engine_media::TransferCharacteristic::Pq,
            "fixture precondition: the source must be tagged PQ",
        );

        let frames = decode_all(&path, DecodeOptions::default());
        assert!(!frames.is_empty(), "the HDR fixture decoded nothing");
        for frame in &frames {
            assert_eq!(
                frame.color, stream.color,
                "the decoder replaced the container's colorimetry",
            );
        }
        let residency = frames
            .iter()
            .map(|frame| match &frame.buffer {
                FrameBuffer::Android(image) => format!("AImage 0x{:x}", image.format()),
                FrameBuffer::Cpu(_) => "CPU".to_string(),
                FrameBuffer::Yuv420(_) => "owned YUV420".to_string(),
            })
            .next()
            .unwrap_or_default();
        println!(
            "MEASURED HDR: {} frames, first as {residency}, colour {:?}",
            frames.len(),
            stream.color,
        );
    }

    /// A seek is the FIRST thing a scrub or a thumbnail does, and it must
    /// return the frame it asks for.
    ///
    /// Two defects lived here, both found by driving the edge rather than by
    /// reading: `AMediaCodec_flush` on a codec that has produced no output
    /// wedges it — the decoder swallowed all twelve samples and emitted
    /// nothing but end-of-stream, so `decode_next` returned `None` for the
    /// whole clip — and `AMediaExtractor_seekTo(0)` lands on the first sample
    /// whose PRESENTATION time is at or after zero, which on an MP4 with an
    /// edit list skips the priming sample the decoder would have burned, so
    /// the program's first coded frame went to the codec's start-up instead of
    /// to the caller.
    ///
    /// KILL: delete the `!self.touched` guard in either `seek` — the video
    /// half reds here, and the audio half reds in
    /// `a_seek_to_the_start_returns_the_whole_program`.
    #[test]
    fn a_seek_before_any_decode_returns_the_frame_it_asks_for() {
        let path = fixture("h264-gop4.mp4");
        let backend = AndroidVideoIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info.video_streams().next().unwrap().index;

        for target_us in [0i64, -1_000_000, 250_000] {
            let mut decoder = block_on(backend.open_decoder(&path, stream)).unwrap();
            let target = MediaTime::new(target_us, TimeBase::MICROSECONDS);
            block_on(decoder.seek(target)).expect("seek");
            let frame = block_on(decoder.decode_next())
                .expect("decode after seek")
                .unwrap_or_else(|| {
                    panic!(
                        "seek to {target_us} us as the first operation \
                                           returned no frame at all"
                    )
                });
            assert!(
                micros(&frame) >= target_us.max(0),
                "seek to {target_us} landed at {}",
                micros(&frame),
            );
        }
    }

    /// A scrub to the very start returns the program from its very start, all
    /// of it. This is the audio half of the defect above: before the fix a
    /// seek to sample 0 came back at 1 024 and the drain after it was 1 024
    /// samples short of the program.
    ///
    /// KILL: delete the `!self.touched` guard in `MediaCodecAudioDecoder::seek`
    /// — the rebuilt codec consumes the program's first coded frame and both
    /// assertions red.
    #[test]
    fn a_seek_to_the_start_returns_the_whole_program() {
        let name = "aac-lc-48000-stereo.m4a";
        let expected: usize = AUDIO_FIXTURES
            .iter()
            .find(|row| row.0 == name)
            .map(|row| row.2)
            .expect("a reference row");
        let path = fixture(name);
        let backend = AndroidAudioIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info
            .streams
            .iter()
            .find(|stream| stream.kind == MediaKind::Audio)
            .unwrap();
        let rate = stream.sample_rate.unwrap();

        let mut decoder = block_on(backend.open_audio_decoder(&path, stream.index)).unwrap();
        block_on(decoder.seek(MediaTime::new(
            0,
            TimeBase::new_unchecked(1, i64::from(rate)),
        )))
        .expect("seek");
        let first = block_on(decoder.decode_next())
            .expect("decode")
            .expect("a frame after seeking to the start");
        assert_eq!(first.start.value, 0, "the program does not start at zero");
        let mut decoded = first.frame_count();
        while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
            decoded += frame.frame_count();
        }
        assert_eq!(
            decoded,
            expected,
            "a seek to the start lost {} samples of the program",
            expected as i64 - decoded as i64,
        );
    }

    /// Damaged media is refused or decoded, never a crash and never a hang.
    ///
    /// The corpus carries four broken files on purpose. What each one does is
    /// the platform's business — `corrupt-bitflipped.mp4` decodes all twelve
    /// frames because a decoder is meant to survive bit errors, and
    /// `corrupt-bitflipped.mkv` is refused at the probe — so what is asserted
    /// is the invariant that holds for all of them: the call returns.
    ///
    /// KILL: remove the `check_media` on `AMediaExtractor_setDataSourceFd` and
    /// a file the extractor rejects goes on to be decoded from an unset source.
    #[test]
    fn damaged_and_missing_media_is_refused_rather_than_fatal() {
        let backend = AndroidVideoIo;

        let missing = std::env::temp_dir().join("ferment-e1-not-here.mp4");
        let _ = std::fs::remove_file(&missing);
        let error = block_on(backend.probe(&missing)).unwrap_err();
        assert!(
            matches!(error, ferment_engine_media::MediaError::Io(_)),
            "a missing file should be an IO error, got {error}",
        );

        let empty = std::env::temp_dir().join("ferment-e1-empty.mp4");
        std::fs::write(&empty, b"").unwrap();
        assert!(
            block_on(backend.probe(&empty)).is_err(),
            "an empty file probed as media",
        );
        let _ = std::fs::remove_file(&empty);

        for name in [
            "corrupt-truncated.mp4",
            "corrupt-bitflipped.mp4",
            "corrupt-truncated.mkv",
            "corrupt-bitflipped.mkv",
        ] {
            let path = fixture(name);
            let Ok(info) = block_on(backend.probe(&path)) else {
                continue;
            };
            let Some(stream) = info.video_streams().next() else {
                continue;
            };
            let Ok(mut decoder) = block_on(backend.open_decoder(&path, stream.index)) else {
                continue;
            };
            let mut frames = 0usize;
            while let Ok(Some(_)) = block_on(decoder.decode_next()) {
                frames += 1;
                assert!(frames <= 1_000, "{name} decoded without end");
            }
            println!("MEASURED {name}: {frames} frames before it stopped");
        }
    }

    /// Two decoders alive at once, interleaved. A backend that keeps
    /// process-wide state — one image reader, one codec, one extractor
    /// position — passes every test in this file that opens a single session
    /// and fails the moment the app previews one clip while exporting another.
    ///
    /// KILL: make `ImageReader::open` return a shared reader instead of a new
    /// one; the two sessions then take each other's pictures.
    #[test]
    fn two_decoders_run_at_once_without_touching_each_other() {
        let backend = AndroidVideoIo;
        let one_path = fixture("h264-gop4.mp4");
        let two_path = fixture("hevc-gop4.mp4");
        let mut one = block_on(backend.open_decoder(&one_path, 0)).unwrap();
        let mut two = block_on(backend.open_decoder(&two_path, 0)).unwrap();

        let (_, one_expected) = reference("h264-gop4.mp4");
        let (_, two_expected) = reference("hevc-gop4.mp4");
        for index in 0..12 {
            let first = block_on(one.decode_next())
                .expect("decode")
                .expect("a frame");
            let second = block_on(two.decode_next())
                .expect("decode")
                .expect("a frame");
            // Within one microsecond, for the reason
            // `decodes_the_fixture_matrix_at_the_macs_own_timestamps` gives:
            // MediaExtractor truncates the container's ticks where libav rounds
            // them.
            assert!(
                (micros(&first) - one_expected[index]).abs() <= 1,
                "interleaved h264 frame {index}: {} against the Mac's {}",
                micros(&first),
                one_expected[index],
            );
            assert!(
                (micros(&second) - two_expected[index]).abs() <= 1,
                "interleaved hevc frame {index}: {} against the Mac's {}",
                micros(&second),
                two_expected[index],
            );
        }
    }

    /// An output size with a zero in it is a caller error, not a size to
    /// deliver. The macOS twin refuses it in
    /// `destination_image_buffer_attributes`; before this, Android silently
    /// ignored it and handed back the source size.
    ///
    /// KILL: delete the guard at the top of `MediaCodecDecoder::open` — the
    /// call succeeds and returns 64x48.
    #[test]
    fn a_zero_output_size_is_refused() {
        let path = fixture("h264-gop4.mp4");
        for size in [(0u32, 32u32), (32, 0), (0, 0)] {
            let error = block_on(AndroidVideoIo.open_decoder_with(
                &path,
                0,
                DecodeOptions {
                    output_size: Some(size),
                    ..Default::default()
                },
            ))
            .err()
            .unwrap_or_else(|| panic!("{size:?} was accepted"));
            assert!(
                error.to_string().contains("non-zero"),
                "unexpected error for {size:?}: {error}",
            );
        }
    }

    // ---------------------------------------------------------------------
    // Audio

    fn decode_audio(name: &str) -> Vec<PcmAudioFrame> {
        let path = fixture(name);
        let backend = AndroidAudioIo;
        let info = block_on(backend.probe(&path)).unwrap();
        let stream = info
            .streams
            .iter()
            .find(|stream| stream.kind == MediaKind::Audio)
            .unwrap_or_else(|| panic!("no audio stream in {name}"));
        let mut decoder = block_on(backend.open_audio_decoder(&path, stream.index))
            .unwrap_or_else(|error| panic!("open {name}: {error}"));
        let mut frames = Vec::new();
        while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
            frames.push(frame);
        }
        frames
    }

    /// Every container the app opens, decoded to contiguous interleaved PCM at
    /// the source rate, in the amount the Mac decoded. Contiguity is the
    /// assertion that matters: a caller stitching frames into a buffer relies
    /// on frame N ending exactly where frame N+1 starts, and a gap becomes
    /// silence in the middle of a clip.
    ///
    /// The sample count is the Mac's, not a tolerance: both sides drop the
    /// container's declared priming and stop at its declared end, so a
    /// MediaCodec decoder that hands back the encoder's delay as program is a
    /// count that does not match.
    ///
    /// KILL: delete the `audible_limit` truncation in `decode_next` — six of
    /// the nine rows go long (the m4a to 48 128, the mkv to 49 152). Dropping a
    /// leading trim is NOT a kill for this test: `decode_next` never applies
    /// one, because the platform already has.
    #[test]
    fn decodes_the_fixture_matrix_to_contiguous_pcm_at_the_macs_own_length() {
        for (name, mac_samples, samples_expected, rate, channels, peak_expected) in AUDIO_FIXTURES {
            let frames = decode_audio(name);
            assert!(!frames.is_empty(), "{name} decoded nothing");
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.sample_rate == *rate && frame.channels == *channels),
                "{name}: expected {rate} Hz / {channels} ch, got {} Hz / {}",
                frames[0].sample_rate,
                frames[0].channels,
            );
            assert!(
                frames.windows(2).all(|pair| {
                    pair[0].start.value + pair[0].frame_count() as i64 == pair[1].start.value
                }),
                "{name}: decoded frames are not contiguous",
            );
            let decoded: usize = frames.iter().map(|frame| frame.frame_count()).sum();
            assert_eq!(
                decoded, *samples_expected,
                "{name}: decoded length (the Mac's libav gives {mac_samples})",
            );
            let peak = frames
                .iter()
                .flat_map(|frame| frame.samples.iter())
                .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
            // A real signal at the level the Mac measured. The tolerance is
            // the two decoders' own arithmetic, not a licence: 0.01 is 1 % of
            // full scale, where a decoder that lost SBR or picked the wrong
            // channel differs by an order of magnitude.
            assert!(
                (peak - peak_expected).abs() < 0.01,
                "{name}: peak {peak} against the Mac's {peak_expected}",
            );
        }
    }

    /// A seek lands on the requested sample and returns the same audio a
    /// sequential decode produced there. Anything less and scrubbing plays
    /// something other than what playback would.
    ///
    /// Two things are graded, and the second is why this is a search and not a
    /// subtraction: the frame must REPORT the requested sample, and the audio
    /// it carries must be the audio at that sample. The alignment between the
    /// two is the platform's, and it is not always zero — MediaCodec's MP3
    /// decoder ramps its polyphase filter up again after every
    /// `AMediaCodec_flush`, so the samples come back a few positions out of
    /// step with a decode that never flushed. Measured on `cuts-36`: AAC is
    /// exact, MP3 sits about four samples off (0.10 on a 0.43 tone, which is
    /// 0.23 rad at 440 Hz).
    ///
    /// So the offset is measured and bounded rather than assumed away. Eight
    /// samples is 0.2 ms; a seek that landed on the wrong FRAME would be 1 024
    /// (AAC) or 1 152 (MP3) out and no offset in this window would fit it.
    ///
    /// KILL: seek straight to the target instead of landing before it — the
    /// first frame back is the one after the flush, and no alignment in the
    /// window brings it under the tolerance. Or drop the rounding in the
    /// post-seek origin: the frame then reports the sample it was asked for
    /// while carrying its neighbour's audio, and the measured offset moves off
    /// zero for AAC too.
    #[test]
    fn seeks_return_the_same_audio_as_a_sequential_decode() {
        const WINDOW: i64 = 8;
        // The expected offset per fixture and target, measured on `cuts-36`
        // and pinned rather than absorbed into a tolerance. AAC is exact
        // everywhere. MP3 is exact at the start — where the seek is a no-op on
        // an untouched decoder, so nothing restarts — and two samples out
        // mid-stream, which is MediaCodec's polyphase filter resuming after the
        // codec is rebuilt.
        for (name, at_start, mid_stream) in [
            ("aac-lc-48000-stereo.m4a", 0i64, 0i64),
            ("mp3-cbr-44100-stereo.mp3", 0i64, 2i64),
        ] {
            let path = fixture(name);
            let backend = AndroidAudioIo;
            let info = block_on(backend.probe(&path)).unwrap();
            let stream = info
                .streams
                .iter()
                .find(|stream| stream.kind == MediaKind::Audio)
                .unwrap();
            let sequential: Vec<f32> = decode_audio(name)
                .iter()
                .flat_map(|frame| frame.samples.iter().copied())
                .collect();
            let channels = usize::from(stream.channels.expect("a channel count"));
            let rate = stream.sample_rate.expect("a sample rate");

            let mut decoder = block_on(backend.open_audio_decoder(&path, stream.index)).unwrap();
            for (target_sample, expected_offset) in
                [(0i64, at_start), (i64::from(rate) / 4, mid_stream)]
            {
                let target =
                    MediaTime::new(target_sample, TimeBase::new_unchecked(1, i64::from(rate)));
                block_on(decoder.seek(target)).expect("seek");
                let first = block_on(decoder.decode_next())
                    .expect("decode after seek")
                    .expect("a frame after seek");

                assert_eq!(
                    first.start.value, target_sample,
                    "{name}: the frame reports {} rather than the {target_sample} it was asked for",
                    first.start.value,
                );

                // Several frames' worth: the first frame after a seek is the tail
                // of the coded frame the target landed inside, so on its own it can
                // be a few hundred samples and prove very little.
                let mut seeked: Vec<f32> = first.samples.to_vec();
                let mut start = first.start.value + first.frame_count() as i64;
                while seeked.len() / channels < 4_096 {
                    let Some(frame) = block_on(decoder.decode_next()).expect("decode") else {
                        break;
                    };
                    assert_eq!(
                        frame.start.value, start,
                        "{name}: frames after a seek are not contiguous",
                    );
                    start += frame.frame_count() as i64;
                    seeked.extend_from_slice(&frame.samples);
                }

                // The best alignment in the window, and how bad it is there.
                let compared = 4_096.min(seeked.len() / channels) as i64;
                assert!(
                    compared > 512,
                    "{name}: only {compared} sample frames after the seek",
                );
                let mut best = (f32::MAX, i64::MAX);
                for offset in -WINDOW..=WINDOW {
                    let at = target_sample + offset;
                    if at < 0 || (at + compared) as usize * channels > sequential.len() {
                        continue;
                    }
                    let worst = (0..(compared as usize * channels))
                        .map(|index| {
                            (seeked[index] - sequential[at as usize * channels + index]).abs()
                        })
                        .fold(0.0f32, f32::max);
                    if worst < best.0 {
                        best = (worst, offset);
                    }
                }
                println!(
                    "MEASURED {name} @{target_sample}: best alignment {} samples, worst difference {:.6}",
                    best.1, best.0,
                );
                assert!(
                    best.0 < 1.0e-2,
                    "{name}: the audio after a seek matches nothing within {WINDOW} samples \
                 (best {:.6} at offset {})",
                    best.0,
                    best.1,
                );
                // The offset the search FOUND, asserted — not merely that it lies
                // inside the window it was drawn from, which is what
                // `best.1.abs() <= WINDOW` said and which no implementation could
                // fail. Without this the search hides the thing it exists to
                // expose: a decoder consistently a few samples late reports the
                // position it was asked for, delivers misaligned audio, and passes.
                assert_eq!(
                    best.1, expected_offset,
                    "{name} @{target_sample}: the audio sits {} samples from where the frame \
                 says it does",
                    best.1,
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Encode

    /// Twelve frames in, twelve packets out — which only holds if `finalize`
    /// flushes the encoder. A lookahead that is never drained silently drops
    /// the tail of every clip.
    ///
    /// KILL: return from `finalize` without queueing the end-of-stream buffer;
    /// the count comes back short and the last frames of the clip are gone.
    #[test]
    fn every_frame_comes_back_as_a_packet_after_the_flush() {
        let encoded = encode_twelve();
        assert_eq!(encoded.packets.len(), 12);
        assert!(encoded.packets.iter().all(|packet| !packet.data.is_empty()));
        assert!(
            encoded.packets.iter().any(|packet| packet.is_sync),
            "no keyframe: nothing could start decoding this",
        );
    }

    /// The `EncodedVideo` contract: a codec description, and packets whose
    /// declared form matches it. MediaCodec states its parameter sets as
    /// Annex-B, so both sides say Annex-B — a consumer converts payloads using
    /// the form the configuration declares, and a disagreement is a stream
    /// nothing can read.
    ///
    /// KILL: declare `Packetization::LengthPrefixed` on the packets while the
    /// csd stays Annex-B; the muxer then writes length prefixes over start
    /// codes.
    #[test]
    fn the_configuration_and_the_packets_declare_the_same_form() {
        let encoded = encode_twelve();
        assert_eq!(encoded.configuration.codec, Codec::H264);
        assert!(
            !encoded.configuration.description.is_empty(),
            "no csd: nothing could decode this",
        );
        assert!(
            encoded
                .packets
                .iter()
                .all(|packet| packet.packetization == encoded.configuration.packetization),
            "packets declare a different form than the configuration ({:?})",
            encoded.configuration.packetization,
        );
        assert!(
            encoded.configuration.description.starts_with(&[0, 0, 0, 1])
                || encoded.configuration.description.starts_with(&[0, 0, 1]),
            "MediaCodec's csd is Annex-B; this one has no start code",
        );
    }

    /// A moving block, so successive frames are not identical — an encoder
    /// handed twelve copies of one picture produces a stream that says nothing
    /// about whether it works.
    fn encode_twelve() -> ferment_engine_media::EncodedVideo {
        use ferment_engine_media::{ColorInfo, CpuFrame, CpuPixelFormat};

        let time_base = TimeBase::new_unchecked(1, 12);
        let settings = EncoderSettings {
            frame_rate: Some((12, 1)),
            bit_rate: Some(400_000),
            max_keyframe_interval: Some(4),
            allow_frame_reordering: false,
            ..EncoderSettings::new(Codec::H264, 64, 48, time_base)
        };
        let mut encoder = block_on(AndroidVideoIo.open_encoder(settings)).expect("open");
        for index in 0..12i64 {
            let (width, height) = (64usize, 48usize);
            let mut pixels = vec![16u8; width * height * 3];
            let x0 = (index as usize * 4) % (width - 8);
            for y in 8..24 {
                for x in x0..x0 + 8 {
                    let at = (y * width + x) * 3;
                    pixels[at..at + 3].copy_from_slice(&[220, 180, 90]);
                }
            }
            let cpu = CpuFrame::new(
                width as u32,
                height as u32,
                width * 3,
                CpuPixelFormat::Rgb24,
                pixels,
            )
            .unwrap();
            let frame = VideoFrame::encodable_from_cpu(
                &cpu,
                width as u32,
                height as u32,
                MediaTime::new(index, time_base),
                Some(MediaTime::new(1, time_base)),
                ColorInfo::default(),
            )
            .unwrap();
            block_on(encoder.push_frame(frame)).expect("push");
        }
        block_on(encoder.finalize()).expect("finalize")
    }

    /// The AAC encoder produces a stream with an AudioSpecificConfig and
    /// reports the priming a decoder will have to drop. Without the number the
    /// muxer writes no edit list and every round trip gains silence at the
    /// head.
    #[test]
    fn encodes_aac_and_reports_its_priming() {
        let settings = AudioEncoderSettings {
            bit_rate: 128_000,
            sample_rate: 48_000,
            channels: 2,
        };
        let mut encoder = block_on(AndroidAudioIo.open_audio_encoder(settings)).expect("open");
        let base = TimeBase::new_unchecked(1, 48_000);
        let total = 48_000usize;
        let mut samples = Vec::with_capacity(total * 2);
        for index in 0..total {
            let value = (index as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.4;
            samples.push(value);
            samples.push(value);
        }
        block_on(
            encoder
                .push_pcm(PcmAudioFrame::new(48_000, 2, MediaTime::new(0, base), samples).unwrap()),
        )
        .expect("push");
        let encoded = block_on(encoder.finalize()).expect("finalize");

        assert_eq!(encoded.configuration.codec, Codec::Aac);
        assert!(
            !encoded.configuration.description.is_empty(),
            "no AudioSpecificConfig: nothing could decode this",
        );
        assert!(!encoded.packets.is_empty());
        assert!(encoded.packets.iter().all(|packet| !packet.data.is_empty()));
        assert_eq!(
            encoded.timing.audible_samples,
            Some(total as u64),
            "the audible length should be what was pushed",
        );
        assert!(
            encoded.timing.leading_samples > 0,
            "the encoder reported no priming",
        );
    }

    /// `allow_frame_reordering: false` means delivery order IS presentation
    /// order. The scene detectors index frames positionally, so a stream that
    /// reorders mis-associates every result rather than failing.
    ///
    /// This exists because the setting was a silent no-op: the backend spelled
    /// the key `max-b-frames` and the platform's is `max-bframes`
    /// (`android.media.MediaFormat.KEY_MAX_B_FRAMES`, read off the platform
    /// jar). MediaCodec ignores a key it does not know, without a word.
    ///
    /// KILL: put the hyphen back in `KEY_MAX_B_FRAMES`. On a device whose
    /// encoder emits B-frames by default the timestamps come back reordered;
    /// on one that does not, the assertion below still holds and the test
    /// PRINTS the measured order so the report can say which kind of device
    /// graded it.
    #[test]
    fn reordering_off_delivers_presentation_order() {
        let encoded = encode_twelve();
        let pts: Vec<i64> = encoded
            .packets
            .iter()
            .filter_map(|packet| packet.pts.map(|pts| pts.value))
            .collect();
        println!("MEASURED encoder delivery order: {pts:?}");
        assert_eq!(pts.len(), 12, "not every frame came back with a timestamp");
        assert!(
            pts.windows(2).all(|pair| pair[0] < pair[1]),
            "delivery order is not presentation order: {pts:?}",
        );
    }

    // ---------------------------------------------------------------------
    // Mux

    /// The round trip through the platform container writer: every packet of a
    /// real fixture read out and written back, then read again. This exercises
    /// the codec description, the timestamps and the trailer at once, and the
    /// source file is the expected answer.
    ///
    /// KILL: hand the avcC record to `csd-0` instead of its parameter sets —
    /// the output's `avcC` is malformed and the re-probe reports a different
    /// codec description.
    #[test]
    fn a_remuxed_fixture_reads_back_as_the_same_media() {
        for name in ["h264-gop4.mp4", "hevc-gop4.mp4"] {
            let source = fixture(name);
            let scratch = Scratch::new(&format!("remux-{name}"));
            let info = demux::probe(&source).expect("probe the source");
            let video = info.video_streams().next().expect("a video stream");

            let mut reader = PacketReader::open(&source, video.index).expect("packet reader");
            let mut session = AndroidMuxer
                .open(
                    &scratch.0,
                    MuxSettings {
                        format: MuxFormat::Mp4,
                        fast_start: false,
                        movie_timescale: None,
                    },
                )
                .expect("open the muxer");
            let track = session
                .add_video_track(VideoMuxTrack {
                    display_matrix: None,
                    configuration: video.decoder_configuration.clone(),
                    width: video.width.unwrap(),
                    height: video.height.unwrap(),
                    time_base: video.time_base,
                    color: video.color,
                    duration: video.duration,
                })
                .expect("add the video track");
            let mut written = 0usize;
            while let Some(packet) = reader.next_packet().expect("read a packet") {
                session.write_packet(track, packet).expect("write a packet");
                written += 1;
            }
            let output = session.finalize().expect("finalize");
            assert_eq!(written, 12, "{name}");
            assert_eq!(output.track_count, 1, "{name}");
            assert!(
                output.bytes.is_none(),
                "a path-backed session writes a file"
            );

            let after = demux::probe(&scratch.0)
                .unwrap_or_else(|error| panic!("{name}: the remuxed file does not probe: {error}"));
            let remuxed = after
                .video_streams()
                .next()
                .unwrap_or_else(|| panic!("{name}: the remuxed file has no video stream"));
            assert_eq!(remuxed.codec, video.codec, "{name}");
            assert_eq!(remuxed.width, video.width, "{name}");
            assert_eq!(remuxed.height, video.height, "{name}");

            // And it decodes: the strongest statement about a container is that
            // something can read the media back out of it.
            let frames = decode_all(&scratch.0, DecodeOptions::default());
            assert_eq!(frames.len(), 12, "{name}: the remuxed file decodes short");
        }
    }

    /// What `AMediaMuxer` cannot do is refused, not approximated. All three
    /// refusals are load-bearing: a MOV asked for and an MP4 written is the
    /// silent relabel the libav muxer refuses for ProRes, and an export that
    /// quietly lost `faststart` is discovered by a player seeking to the end of
    /// a file meant to stream.
    ///
    /// KILL: map `MuxFormat::Mov` to `AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4`, or
    /// drop the `fast_start` guard — both make this pass an error where it
    /// expects one.
    #[test]
    fn the_platform_muxer_refuses_what_it_cannot_do() {
        let scratch = Scratch::new("refusals");
        for (settings, expected) in [
            (
                MuxSettings {
                    format: MuxFormat::Mov,
                    fast_start: false,
                    movie_timescale: None,
                },
                "MPEG-4",
            ),
            (
                MuxSettings {
                    format: MuxFormat::Mp4,
                    fast_start: true,
                    movie_timescale: None,
                },
                "fast-start",
            ),
            (
                MuxSettings {
                    format: MuxFormat::Mp4,
                    fast_start: false,
                    movie_timescale: Some(600),
                },
                "movie-timescale",
            ),
        ] {
            let error = AndroidMuxer
                .open(&scratch.0, settings)
                .err()
                .unwrap_or_else(|| panic!("{settings:?} should have been refused"));
            assert!(
                error.to_string().contains(expected),
                "unexpected error for {settings:?}: {error}",
            );
        }
        // The default settings ARE fast-start, so the platform muxer refuses
        // them. That is the fact the report carries: an export that needs
        // fast-start uses LibavMuxer, which is in this same binary.
        assert!(
            AndroidMuxer
                .open(&scratch.0, MuxSettings::default())
                .is_err(),
            "MuxSettings::default() asks for fast-start",
        );
    }
}
