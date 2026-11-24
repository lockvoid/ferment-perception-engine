#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ferment_engine_media::platform::macos::VideoToolboxVideoIo;
use ferment_engine_media::{
    Codec, DecodeSession, EncoderSettings, FrameBuffer, ProResProfile, VideoFrame, VideoIo,
};
use futures_executor::block_on;

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

#[test]
fn decodes_the_full_fixture_matrix_to_native_buffers() {
    for name in FIXTURES {
        let frames = decode_native(&fixture_path(name));
        assert_eq!(frames.len(), 12, "{name}");
        assert!(
            frames
                .windows(2)
                .all(|pair| pair[0].pts.value <= pair[1].pts.value),
            "{name}: {:?}",
            frames
                .iter()
                .map(|frame| frame.pts.value)
                .collect::<Vec<_>>()
        );
        assert!(frames.iter().all(|frame| {
            frame.width == 64 && frame.height == 48 && matches!(frame.buffer, FrameBuffer::MacOS(_))
        }));
        for frame in frames {
            let cpu = frame.to_cpu_rgb().unwrap();
            assert_eq!(cpu.data.len(), 64 * 48 * 3, "{name}");
        }
    }
}

#[test]
fn exact_seeks_are_byte_deterministic_against_sequential_decode() {
    let path = fixture_path("h264-gop4.mp4");
    let backend = VideoToolboxVideoIo;
    let info = block_on(backend.probe(&path)).unwrap();
    let stream = info.video_streams().next().unwrap();
    let mut decoder = block_on(backend.open_decoder(&path, stream.index)).unwrap();
    let reference = decode_remaining(&mut *decoder, Some(&path));
    let order = [7usize, 1, 10, 4, 0, 9, 2, 11, 5, 8];

    for index in order {
        let target = reference[index].pts;
        block_on(decoder.seek(target)).unwrap();
        let first = block_on(decoder.decode_next()).unwrap().unwrap();
        let first_bytes = first.to_cpu_rgb().unwrap().data;
        assert_eq!(first.pts, reference[index].pts, "seek {index}");
        assert_eq!(
            first_bytes.as_ref(),
            rgb_bytes(&reference[index]).as_slice()
        );

        block_on(decoder.seek(target)).unwrap();
        let repeated = block_on(decoder.decode_next()).unwrap().unwrap();
        assert_eq!(
            first_bytes.as_ref(),
            repeated.to_cpu_rgb().unwrap().data.as_ref(),
            "repeat seek {index}"
        );
    }
}

#[test]
fn encodes_h264_hevc_and_prores_from_native_frames() {
    let path = fixture_path("h264-gop4.mp4");
    let frames = decode_native(&path);
    let info = block_on(VideoToolboxVideoIo.probe(&path)).unwrap();
    let frame_rate = info.video_streams().next().unwrap().frame_rate;
    for codec in [
        Codec::H264,
        Codec::Hevc,
        Codec::ProRes(ProResProfile::Standard),
    ] {
        let settings = EncoderSettings {
            frame_rate,
            bit_rate: Some(500_000),
            max_keyframe_interval: Some(4),
            ..EncoderSettings::new(codec.clone(), 64, 48, frames[0].pts.time_base)
        };
        let backend = VideoToolboxVideoIo;
        let mut encoder = block_on(backend.open_encoder(settings)).unwrap();
        for frame in frames.iter().cloned() {
            block_on(encoder.push_frame(frame)).unwrap();
        }
        let encoded = block_on(encoder.finalize()).unwrap();
        assert_eq!(encoded.configuration.codec, codec);
        assert_eq!(encoded.packets.len(), frames.len());
        assert!(encoded.packets.iter().all(|packet| !packet.data.is_empty()));
        assert!(encoded.packets.iter().any(|packet| packet.is_sync));
    }
}

fn decode_native(path: &Path) -> Vec<VideoFrame> {
    decode_with(&VideoToolboxVideoIo, path)
}

fn decode_with(backend: &dyn VideoIo, path: &Path) -> Vec<VideoFrame> {
    let info = block_on(backend.probe(path))
        .unwrap_or_else(|error| panic!("probe failed for {}: {error}", path.display()));
    let stream_index = info.video_streams().next().unwrap().index;
    let mut decoder = block_on(backend.open_decoder(path, stream_index))
        .unwrap_or_else(|error| panic!("open failed for {}: {error}", path.display()));
    decode_remaining(&mut *decoder, Some(path))
}

fn decode_remaining(decoder: &mut dyn DecodeSession, path: Option<&Path>) -> Vec<VideoFrame> {
    let mut frames = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).unwrap_or_else(|error| {
        panic!(
            "decode failed for {}: {error}",
            path.map_or_else(|| "seek session".into(), |path| path.display().to_string())
        )
    }) {
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
