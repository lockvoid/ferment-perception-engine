//! Video decode through `LibavVideoIo`.
//!
//! This is `tests/videotoolbox.rs`'s decode half, rewritten against the libav
//! backend (ruling 12) — same fixtures, same assertions. What it pins is the
//! behaviour a caller depends on, not how libavcodec reaches it: the frame
//! count including the tail of the last GOP, presentation order, the platform
//! buffer a frame arrives in, byte-for-byte seek determinism, and the exact
//! size `DecodeOptions` promises.

use std::path::{Path, PathBuf};

use ferment_engine_media::platform::libav::LibavVideoIo;
use ferment_engine_media::{DecodeOptions, DecodeSession, FrameBuffer, VideoFrame, VideoIo};
use futures_executor::block_on;

/// Every container and codec the app opens: three containers × three codecs.
/// Each fixture is 12 frames of 64×48.
const FIXTURES: &[&str] = &[
    "h264-gop4.mp4",
    "h264-gop5.mov",
    "h264-gop6.mkv",
    "hevc-gop4.mp4",
    "hevc-gop5.mov",
    "hevc-gop6.mkv",
    "prores-standard.mp4",
    "prores-standard.mov",
    "prores-standard.mkv",
];

#[cfg(target_os = "macos")]
#[test]
fn hdr_metadata_and_native_residency_survive_while_explicit_bgra_is_tone_mapped_sdr() {
    use ferment_engine_media::{DecodeOutput, TransferCharacteristic};
    use objc2_core_video::{CVPixelBufferGetPixelFormatType, kCVPixelFormatType_32BGRA};
    let root = std::env::temp_dir().join(format!("ferment-decode-hdr-{}", std::process::id()));
    std::fs::create_dir(&root).expect("new owned HDR fixture directory");
    for (tag, transfer) in [
        ("arib-std-b67", TransferCharacteristic::Hlg),
        ("smpte2084", TransferCharacteristic::Pq),
    ] {
        let path = root.join(format!("{tag}.mp4"));
        let parameters = format!(
            "pools=none:frame-threads=1:log-level=error:colorprim=bt2020:transfer={tag}:colormatrix=bt2020nc:range=limited"
        );
        assert!(
            std::process::Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x96:rate=30:duration=0.1",
                    "-c:v",
                    "libx265",
                    "-x265-params",
                    &parameters,
                    "-pix_fmt",
                    "yuv420p10le",
                    "-colorspace",
                    "bt2020nc",
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    tag,
                    "-color_range",
                    "tv",
                    "-frames:v",
                    "1"
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        let mut decoder = block_on(LibavVideoIo.open_decoder(&path, 0)).unwrap();
        let frame = block_on(decoder.decode_next()).unwrap().unwrap();
        assert_eq!(frame.color.transfer, transfer);
        assert_eq!(frame.color.bits_per_channel, Some(10));
        let FrameBuffer::MacOS(buffer) = &frame.buffer else {
            panic!("native HDR was downloaded");
        };
        // Native allocations may use an opaque/tiled format such as p420.
        // This boundary test is not a full HDR display/encode qualification.
        let format = CVPixelBufferGetPixelFormatType(buffer.as_cv_pixel_buffer());
        assert_ne!(
            format, kCVPixelFormatType_32BGRA,
            "implicit BGRA8 conversion"
        );
        let mut sdr = block_on(LibavVideoIo.open_decoder_with(
            &path,
            0,
            DecodeOptions {
                output: DecodeOutput::CpuBgra8,
                ..Default::default()
            },
        ))
        .unwrap();
        let frame = block_on(sdr.decode_next()).unwrap().unwrap();
        assert_eq!(frame.color.transfer, TransferCharacteristic::Bt709);
        assert!(matches!(
            &frame.buffer,
            FrameBuffer::Cpu(cpu) if cpu.format == ferment_engine_media::CpuPixelFormat::Bgra8
        ));
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// An HDR source drawn on the CPU is SDR BT.709. Black, 18% grey, HDR
/// reference white (203 cd/m²) and the 1000 cd/m² peak read the same BT.2390
/// levels in HLG and PQ: at the source extent, resized, and through the YUV
/// samples a native decode keeps.
#[test]
fn hdr_sources_read_as_sdr_at_the_same_levels_on_every_cpu_path() {
    use ferment_engine_media::{DecodeAcceleration, DecodeOutput, TransferCharacteristic};
    let root = std::env::temp_dir().join(format!("ferment-decode-sdr-{}", std::process::id()));
    std::fs::create_dir(&root).expect("new owned HDR fixture directory");
    for (tag, [black, grey, white, peak]) in [
        ("arib-std-b67", [64, 397, 721, 940]),
        ("smpte2084", [64, 398, 573, 723]),
    ] {
        let path = root.join(format!("{tag}.mp4"));
        let bands = format!(
            "format=yuv420p10le,geq=lum='if(lt(X\\,32)\\,{black}\\,if(lt(X\\,64)\\,{grey}\\,if(lt(X\\,96)\\,{white}\\,{peak})))':cb=512:cr=512"
        );
        let parameters = format!(
            "lossless=1:log-level=error:colorprim=bt2020:transfer={tag}:colormatrix=bt2020nc:range=limited"
        );
        assert!(
            std::process::Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:size=128x96:rate=10:duration=0.1",
                    "-vf",
                    &bands,
                    "-c:v",
                    "libx265",
                    "-x265-params",
                    &parameters,
                    "-pix_fmt",
                    "yuv420p10le",
                    "-colorspace",
                    "bt2020nc",
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    tag,
                    "-color_range",
                    "tv",
                ])
                .arg(&path)
                .status()
                .unwrap()
                .success()
        );
        for (options, width) in [
            (
                DecodeOptions {
                    output: DecodeOutput::CpuBgra8,
                    ..Default::default()
                },
                128,
            ),
            (
                DecodeOptions {
                    output: DecodeOutput::CpuBgra8,
                    output_size: Some((64, 48)),
                    ..Default::default()
                },
                64,
            ),
            (
                DecodeOptions {
                    acceleration: DecodeAcceleration::Software,
                    ..Default::default()
                },
                128,
            ),
        ] {
            let mut decoder = block_on(LibavVideoIo.open_decoder_with(&path, 0, options)).unwrap();
            let frame = block_on(decoder.decode_next()).unwrap().unwrap();
            let rgb = frame.to_cpu_rgb().unwrap();
            assert_eq!(rgb.width, width, "{tag} {options:?}");
            if matches!(frame.buffer, FrameBuffer::Cpu(_)) {
                assert_eq!(frame.color.transfer, TransferCharacteristic::Bt709);
            }
            for (band, level) in [0u8, 109, 230, 255].into_iter().enumerate() {
                let x = band * width as usize / 4 + width as usize / 8;
                let at = rgb.height as usize / 2 * rgb.stride + x * 3;
                let pixel = &rgb.data[at..at + 3];
                assert!(
                    pixel.iter().all(|value| value.abs_diff(level) <= 1),
                    "{tag} {options:?} band {band}: {pixel:?}, expected {level}"
                );
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// GPU planes are the picture as decoded, laid out for Metal textures over
/// its two planes: the stream's depth, chroma and range, uncompressed, and a
/// ten-bit code in the top ten bits of its word — the layout the GPU
/// conversion reads. Where no hardware decoder takes the stream, the caller
/// is told to decode to the CPU instead.
#[cfg(target_os = "macos")]
#[test]
fn gpu_planes_hold_the_stream_s_own_codes_in_an_uncompressed_bi_planar_layout() {
    use ferment_engine_media::{DecodeAcceleration, DecodeOutput, VideoError};
    use objc2_core_video::{
        CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
        CVPixelBufferGetPixelFormatType, CVPixelBufferGetPlaneCount, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
    };
    let planes = DecodeOptions {
        output: DecodeOutput::GpuPlanes,
        ..Default::default()
    };
    let open = |path: &Path, options| block_on(LibavVideoIo.open_decoder_with(path, 0, options));
    for (name, format) in [
        (
            "h264-gop4.mp4",
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ),
        (
            "hevc-gop5.mov",
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ),
        (
            "prores-standard.mov",
            kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
        ),
    ] {
        let mut decoder = open(&fixture_path(name), planes).unwrap();
        let frame = block_on(decoder.decode_next()).unwrap().unwrap();
        let FrameBuffer::MacOS(buffer) = &frame.buffer else {
            panic!("{name}: not GPU planes");
        };
        let buffer = buffer.as_cv_pixel_buffer();
        assert_eq!(CVPixelBufferGetPixelFormatType(buffer), format, "{name}");
        assert_eq!(CVPixelBufferGetPlaneCount(buffer), 2, "{name}");
    }

    let root = std::env::temp_dir().join(format!("ferment-gpu-planes-{}", std::process::id()));
    std::fs::create_dir(&root).expect("new owned HDR fixture directory");
    let path = root.join("hlg.mp4");
    let codes = [64u16, 397, 721, 940];
    let bands = format!(
        "format=yuv420p10le,geq=lum='if(lt(X\\,32)\\,{}\\,if(lt(X\\,64)\\,{}\\,if(lt(X\\,96)\\,{}\\,{})))':cb=512:cr=512",
        codes[0], codes[1], codes[2], codes[3]
    );
    assert!(
        std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=128x96:rate=10:duration=0.1",
                "-vf",
                &bands,
                "-c:v",
                "libx265",
                "-x265-params",
                "lossless=1:log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
                "-pix_fmt",
                "yuv420p10le",
                "-colorspace",
                "bt2020nc",
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-color_range",
                "tv",
            ])
            .arg(&path)
            .status()
            .unwrap()
            .success()
    );
    let mut decoder = open(&path, planes).unwrap();
    let frame = block_on(decoder.decode_next()).unwrap().unwrap();
    let FrameBuffer::MacOS(buffer) = &frame.buffer else {
        panic!("HLG: not GPU planes");
    };
    let buffer = buffer.as_cv_pixel_buffer();
    assert_eq!(
        CVPixelBufferGetPixelFormatType(buffer),
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
    );
    // SAFETY: the buffer stays retained by `frame` through the read lock, and
    // the plane rows are only read before the matching unlock.
    let words: Vec<u16> = unsafe {
        assert_eq!(
            CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly),
            0
        );
        let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0).cast::<u8>();
        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
        let words = (0..4)
            .map(|band| {
                let at = 48 * stride + (band * 32 + 16) * 2;
                u16::from_le_bytes([*base.add(at), *base.add(at + 1)])
            })
            .collect();
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags::ReadOnly);
        words
    };
    assert_eq!(words, codes.map(|code| code << 6), "{words:x?}");
    // The explicit CPU read narrows the same codes to eight bits (Y/4) and
    // takes them through the matrix as they are, without tone mapping.
    let rgb = frame.to_cpu_rgb().unwrap();
    for (band, level) in [0u8, 97, 191, 255].into_iter().enumerate() {
        let at = 48 * rgb.stride + (band * 32 + 16) * 3;
        let pixel = &rgb.data[at..at + 3];
        assert!(
            pixel.iter().all(|value| value.abs_diff(level) <= 1),
            "band {band}: {pixel:?}, expected {level}"
        );
    }

    // No hardware decoder, or software asked for: the CPU path, said so.
    for (path, options) in [
        (
            path.clone(),
            DecodeOptions {
                acceleration: DecodeAcceleration::Software,
                ..planes
            },
        ),
        (
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ffv1-rgba-8x6.mkv"),
            planes,
        ),
    ] {
        assert!(
            matches!(open(&path, options), Err(VideoError::BackendUnavailable(_))),
            "{path:?} {options:?}"
        );
    }
    assert!(matches!(
        open(
            &path,
            DecodeOptions {
                output_size: Some((64, 48)),
                ..planes
            }
        ),
        Err(VideoError::InvalidData(_))
    ));
    std::fs::remove_dir_all(root).unwrap();
}

/// Twelve frames, not eleven. The twelfth only comes out if the decoder is
/// drained at end of file — `avcodec_send_packet(NULL)` — and a backend that
/// forgets to loses the tail of every GOP in every file, silently.
#[test]
fn decodes_the_full_fixture_matrix_and_drains_the_last_gop() {
    for name in FIXTURES {
        let frames = decode(&fixture_path(name));
        assert_eq!(frames.len(), 12, "{name}");
        assert!(
            frames
                .windows(2)
                .all(|pair| pair[0].pts.value <= pair[1].pts.value),
            "{name} is out of presentation order: {:?}",
            frames
                .iter()
                .map(|frame| frame.pts.value)
                .collect::<Vec<_>>()
        );
        assert!(
            frames
                .iter()
                .all(|frame| frame.width == 64 && frame.height == 48),
            "{name}"
        );
        for frame in &frames {
            let cpu = frame.to_cpu_rgb().unwrap();
            assert_eq!(cpu.data.len(), 64 * 48 * 3, "{name}");
        }
    }
}

/// On macOS a decoded frame must not touch the CPU: a VideoToolbox frame IS a
/// `CVPixelBuffer`, and that is what wgpu imports through IOSurface. A
/// `FrameBuffer::Cpu` here means every frame is paying for a full-resolution
/// readback.
///
/// Under amendment A1 this is the VideoToolbox session's own output, which is
/// why the fixture corpus can assert it. It could not when full-size decode
/// went through libavcodec's hwaccel: that negotiates a pixel format per
/// stream and does not offer `AV_PIX_FMT_VIDEOTOOLBOX` at 64x48 — measured
/// 2026-09-02, a 64x48 clip came back `AV_PIX_FMT_YUV420P` where a 96x64 one
/// came back `AV_PIX_FMT_VIDEOTOOLBOX`. Our session has no such threshold.
#[cfg(target_os = "macos")]
#[test]
fn macos_frames_stay_on_the_gpu() {
    for name in FIXTURES {
        let frames = decode(&fixture_path(name));
        assert!(
            frames
                .iter()
                .all(|frame| matches!(frame.buffer, FrameBuffer::MacOS(_))),
            "{name} decoded to CPU buffers — the hardware decode path was not taken",
        );
    }
}

/// The software path still decodes, and still decodes correctly.
///
/// It is reached two ways: `FERMENT_LIBAV_SOFTWARE` forces it, and a scaled
/// request takes it by design (a small CPU-side picture is cheaper decoded in
/// software than decoded on the GPU, scaled there and read back). This
/// asserts the second, because it is the one a user actually hits — the
/// colour pass takes it on every import.
#[test]
fn a_scaled_request_decodes_in_software_and_lands_on_the_cpu() {
    let path = fixture_path("h264-gop4.mp4");
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().unwrap();
    let options = DecodeOptions::software_scaled((32, 32));
    let mut decoder = block_on(backend.open_decoder_with(&path, stream.index, options)).unwrap();
    let frames = decode_remaining(&mut *decoder);

    assert_eq!(frames.len(), 12);
    assert!(
        frames
            .iter()
            .all(|frame| matches!(frame.buffer, FrameBuffer::Cpu(_))),
        "a scaled request should come back on the CPU, ready to read",
    );
    // Not merely present — a decoded picture. The fixtures are never a flat
    // field, so a decoder that produced grey would pass a length check.
    let cpu = frames[0].to_cpu_rgb().unwrap();
    let first = cpu.data[0];
    assert!(
        cpu.data.iter().any(|byte| *byte != first),
        "the decoded frame is a single flat colour",
    );
}

/// The seek contract: land on the target, and land on the same pixels every
/// time. Byte equality against a sequential decode is the only assertion that
/// catches a seek which lands one frame early and returns a picture that is
/// merely plausible.
#[test]
fn exact_seeks_are_byte_deterministic_against_sequential_decode() {
    let path = fixture_path("h264-gop4.mp4");
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().unwrap();
    let mut decoder = block_on(backend.open_decoder(&path, stream.index)).unwrap();
    let reference = decode_remaining(&mut *decoder);
    assert_eq!(reference.len(), 12);

    // Deliberately unordered, and revisiting frames: a decoder that only ever
    // seeks forward, or that caches its last answer, passes an in-order walk.
    for index in [7usize, 1, 10, 4, 0, 9, 2, 11, 5, 8] {
        let target = reference[index].pts;
        block_on(decoder.seek(target)).unwrap();
        let first = block_on(decoder.decode_next()).unwrap().unwrap();
        assert_eq!(first.pts, reference[index].pts, "seek to frame {index}");
        let first_bytes = rgb_bytes(&first);
        assert_eq!(
            first_bytes,
            rgb_bytes(&reference[index]),
            "seek to frame {index} decoded different pixels than a sequential decode",
        );

        block_on(decoder.seek(target)).unwrap();
        let repeated = block_on(decoder.decode_next()).unwrap().unwrap();
        assert_eq!(first_bytes, rgb_bytes(&repeated), "repeated seek {index}");
    }
}

/// A seek never lands before its target. The frame source chases a playhead
/// with these, and an early frame would show the wrong picture rather than
/// fail.
#[test]
fn a_seek_never_lands_before_its_target() {
    let path = fixture_path("h264-gop6.mkv");
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().unwrap();
    let mut decoder = block_on(backend.open_decoder(&path, stream.index)).unwrap();
    let reference = decode_remaining(&mut *decoder);

    for frame in &reference {
        block_on(decoder.seek(frame.pts)).unwrap();
        let landed = block_on(decoder.decode_next()).unwrap().unwrap();
        assert!(
            landed.pts.seconds() >= frame.pts.seconds() - 1e-9,
            "seek to {} landed at {}",
            frame.pts.seconds(),
            landed.pts.seconds(),
        );
    }
}

/// `DecodeOptions::output_size` is what keeps the colour pass off
/// full-resolution frames: it promises that EXACT size, aspect not preserved,
/// scaled on the way out of the decoder rather than afterwards.
#[test]
fn output_size_delivers_that_exact_size() {
    let path = fixture_path("h264-gop4.mp4");
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().unwrap();

    // 192×192 is the colour analyser's, and it is deliberately not 4:3 like
    // the 64×48 source — the contract is the caller's size, not a fit.
    let options = DecodeOptions {
        output_size: Some((192, 192)),
        ..Default::default()
    };
    let mut decoder = block_on(backend.open_decoder_with(&path, stream.index, options)).unwrap();
    let frames = decode_remaining(&mut *decoder);

    assert_eq!(frames.len(), 12);
    for frame in &frames {
        assert_eq!((frame.width, frame.height), (192, 192));
        let cpu = frame.to_cpu_rgb().unwrap();
        assert_eq!((cpu.width, cpu.height), (192, 192));
        assert_eq!(cpu.data.len(), 192 * 192 * 3);
    }
}

/// Opening a stream that is not video is a refusal, not a panic or an empty
/// decode — `MediaKind` is the caller's to check, and the backend says so.
#[test]
fn opening_an_audio_stream_as_video_is_refused() {
    let path = fixture_path("synthetic-multitrack.mp4");
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let audio = info.audio_streams().next().expect("a multitrack fixture");
    let error = block_on(backend.open_decoder(&path, audio.index)).unwrap_err();
    assert!(
        error.to_string().contains("not video"),
        "unexpected error: {error}",
    );
}

// -------------------------------------------------------------------------

fn decode(path: &Path) -> Vec<VideoFrame> {
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(path))
        .unwrap_or_else(|error| panic!("probe failed for {}: {error}", path.display()));
    let stream_index = info
        .video_streams()
        .next()
        .unwrap_or_else(|| panic!("no video stream in {}", path.display()))
        .index;
    let mut decoder = block_on(backend.open_decoder(path, stream_index))
        .unwrap_or_else(|error| panic!("open failed for {}: {error}", path.display()));
    decode_remaining(&mut *decoder)
}

fn decode_remaining(decoder: &mut dyn DecodeSession) -> Vec<VideoFrame> {
    let mut frames = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
        frames.push(frame);
    }
    frames
}

fn rgb_bytes(frame: &VideoFrame) -> Vec<u8> {
    frame.to_cpu_rgb().unwrap().data.as_ref().to_vec()
}

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name)
}

/// `FERMENT_BENCH_CLIP=/path/to.mp4 cargo test -p ferment-engine-media
/// --test libav_decode bench_decode -- --ignored --nocapture`
///
/// Ruling 15's decode floor: how long a clip takes with frames left on the
/// GPU. `FERMENT_BENCH_DECODE_SIZE=192x192` asks for the colour pass's scaled
/// output instead, which is the other number that ruling budgets.
#[test]
#[ignore]
fn bench_decode_on_env_clip() {
    let Ok(clip) = std::env::var("FERMENT_BENCH_CLIP") else {
        eprintln!("FERMENT_BENCH_CLIP not set");
        return;
    };
    let output_size = std::env::var("FERMENT_BENCH_DECODE_SIZE")
        .ok()
        .and_then(|size| {
            let (width, height) = size.split_once('x')?;
            Some((width.parse().ok()?, height.parse().ok()?))
        });

    let path = PathBuf::from(&clip);
    let backend = LibavVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().expect("a video stream").index;
    let mut decoder = block_on(backend.open_decoder_with(
        &path,
        stream,
        DecodeOptions {
            output_size,
            ..Default::default()
        },
    ))
    .unwrap();

    let started = std::time::Instant::now();
    let mut frames = 0usize;
    let mut on_gpu = 0usize;
    while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
        frames += 1;
        if !matches!(frame.buffer, FrameBuffer::Cpu(_)) {
            on_gpu += 1;
        }
    }
    eprintln!(
        "BENCH libav decode only [{output_size:?}]: {:.2}s wall, {frames} frames, \
         {on_gpu} left on the GPU",
        started.elapsed().as_secs_f64(),
    );
}
