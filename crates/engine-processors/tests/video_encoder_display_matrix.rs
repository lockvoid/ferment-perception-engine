//! Actual coded-size encodes of the admitted preferred-rotation movie.
//! The file observer reads tkhd independently, and FFmpeg drains every frame
//! with autorotation disabled. No fake codec or post-encode metadata repair.
#![cfg(any(target_os = "macos", target_os = "linux"))]
use ferment_engine_core::{
    MultiVideoEncoding, ProgressSink, ThumbStripOptions, ThumbnailFrame, ThumbnailSink,
    VariantThumbnailSink, VideoEncoding, VideoEncodingPreset, VideoEncodingVariant, VideoFileType,
    VideoRateControl,
};
use ferment_engine_processors::{VideoEncoder, platform_encode_backends};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

const MATRIX: [i32; 9] = [0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824];
const SOURCE_SHA: &str = "386bfe397ba1bf7ea8eabb1e6ebb680542d58f87c5ae448ebda52a12cf242c08";
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        loop {
            let p = std::env::temp_dir().join(format!(
                "ferment-encode-matrix-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&p) {
                Ok(()) => return Self(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("owned scratch admission failed: {e}"),
            }
        }
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-media/tests/fixtures/generated/synthetic-rotation.mov");
    assert_eq!(
        ferment_engine_models::sha256_bytes(&std::fs::read(&p).unwrap()),
        SOURCE_SHA
    );
    p
}
fn options() -> ThumbStripOptions {
    ThumbStripOptions {
        interval: 0.25,
        max_width: 96,
        max_height: 96,
    }
}
fn preset(width: u32, file_type: VideoFileType) -> VideoEncodingPreset {
    let mut p = VideoEncoder::ai_preset();
    p.max_width = width;
    p.caps_short_edge = false;
    p.rate_control = VideoRateControl::Average(2_500_000);
    p.file_type = file_type;
    p
}
#[derive(Debug, Default)]
struct Sink(Mutex<BTreeMap<String, Vec<ThumbnailFrame>>>);
impl ThumbnailSink for Sink {
    fn emit(&self, frame: ThumbnailFrame) {
        self.0
            .lock()
            .unwrap()
            .entry("single".into())
            .or_default()
            .push(frame);
    }
}
impl VariantThumbnailSink for Sink {
    fn emit(&self, id: &str, frame: ThumbnailFrame) {
        self.0
            .lock()
            .unwrap()
            .entry(id.into())
            .or_default()
            .push(frame);
    }
}
fn child(data: &[u8], wanted: [u8; 4]) -> &[u8] {
    let mut cursor = 0;
    while cursor + 8 <= data.len() {
        let small = u32::from_be_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        let (size, head) = if small == 1 {
            assert!(cursor + 16 <= data.len());
            (
                usize::try_from(u64::from_be_bytes(
                    data[cursor + 8..cursor + 16].try_into().unwrap(),
                ))
                .unwrap(),
                16,
            )
        } else if small == 0 {
            (data.len() - cursor, 8)
        } else {
            (small, 8)
        };
        assert!(size >= head && size <= data.len() - cursor);
        if data[cursor + 4..cursor + 8] == wanted {
            return &data[cursor + head..cursor + size];
        }
        cursor += size;
    }
    panic!("missing atom {:?}", wanted);
}
fn matrix_and_size(path: &Path) -> ([i32; 9], (u32, u32)) {
    let data = std::fs::read(path).unwrap();
    let header = child(child(child(&data, *b"moov"), *b"trak"), *b"tkhd");
    let offset = match header[0] {
        0 => 40,
        1 => 52,
        other => panic!("unexpected tkhd version {other}"),
    };
    assert!(header.len() >= offset + 44);
    let mut m = [0; 9];
    for (word, bytes) in m
        .iter_mut()
        .zip(header[offset..offset + 36].chunks_exact(4))
    {
        *word = i32::from_be_bytes(bytes.try_into().unwrap());
    }
    let w = u32::from_be_bytes(header[offset + 36..offset + 40].try_into().unwrap());
    let h = u32::from_be_bytes(header[offset + 40..offset + 44].try_into().unwrap());
    assert_eq!((w & 0xffff, h & 0xffff), (0, 0));
    (m, (w >> 16, h >> 16))
}
fn probe(path: &Path) -> Value {
    let r = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("independent ffprobe required");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    serde_json::from_slice(&r.stdout).unwrap()
}
fn decode(path: &Path, filter: &str) -> Vec<u8> {
    let r = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-noautorotate", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            filter,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .output()
        .expect("independent full-EOS FFmpeg decode required");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    r.stdout
}
fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(x.abs_diff(*y)))
        .sum::<f64>()
        / a.len() as f64
}
fn verify_file(source: &Path, path: &Path, width: u32, height: u32) {
    assert_eq!(
        matrix_and_size(path),
        (MATRIX, (width, height)),
        "coded dimensions change, preferred transform does not"
    );
    let facts = probe(path);
    let stream = &facts["streams"][0];
    assert_eq!(
        (stream["width"].as_u64(), stream["height"].as_u64()),
        (Some(width.into()), Some(height.into()))
    );
    assert_eq!(stream["codec_name"], "h264");
    let frames = facts["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 12, "full original frame count");
    for (index, frame) in frames.iter().enumerate() {
        let actual: f64 = frame["best_effort_timestamp_time"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            (actual - index as f64 / 12.0).abs() <= 0.000001,
            "original source cadence/origin changed: {index}/{actual}"
        );
    }
    let decoded = decode(path, "format=rgb24");
    assert_eq!(
        decoded.len(),
        12 * width as usize * height as usize * 3,
        "decoder must reach the full final frame"
    );
    let expected = decode(
        source,
        &format!("scale={width}:{height}:flags=bilinear,format=rgb24"),
    );
    // Re-encoding and platform scaling are lossy. This checks actual coded
    // pixels; the exact transform/geometry assertions above are lossless.
    assert!(
        mae(&decoded, &expected) < 20.0,
        "coded picture changed orientation or content: {}",
        mae(&decoded, &expected)
    );
}
fn verify_thumbs(source: &Path, sink: &Sink, ids: &[&str]) {
    let displayed = decode(
        source,
        "transpose=cclock,scale=72:96:flags=bilinear,format=rgb24",
    );
    let unrotated = decode(source, "scale=72:96:flags=bilinear,format=rgb24");
    let stride = 72 * 96 * 3;
    assert_eq!(displayed.len(), 12 * stride);
    assert_eq!(unrotated.len(), 12 * stride);
    let entries = sink.0.lock().unwrap();
    assert_eq!(entries.keys().map(String::as_str).collect::<Vec<_>>(), ids);
    for id in ids {
        let cells = &entries[*id];
        assert_eq!(cells.len(), 4);
        for (index, frame) in cells.iter().enumerate() {
            assert_eq!(
                (frame.index, frame.frame.width, frame.frame.height),
                (index, 72, 96)
            );
            assert!((frame.presentation_time - index as f64 * 0.25).abs() <= 0.000001);
            let plane = &frame.frame.planes[0];
            let mut actual = Vec::with_capacity(stride);
            for y in 0..96 {
                for x in 0..72 {
                    let at = plane.offset + y * plane.stride + x * 4;
                    let p = &frame.frame.data[at..at + 4];
                    assert_eq!(p[3], 255);
                    actual.extend_from_slice(&[p[2], p[1], p[0]]);
                }
            }
            let offset = index * 3 * stride;
            let correct = mae(&actual, &displayed[offset..offset + stride]);
            let absent = mae(&actual, &unrotated[offset..offset + stride]);
            assert!(
                correct < 20.0,
                "displayed thumbnail pixels disagree: {correct}"
            );
            assert!(
                correct + 10.0 < absent,
                "orientation observer does not distinguish an absent transform: {correct} vs {absent}"
            );
        }
    }
}
// Kill: resize coded frames but drop the source matrix, or hand the strip the
// coded landscape picture; inspect the actual single-encode file through EOS.
#[test]
#[ignore = "real platform H264 codec and independent FFmpeg decode; run explicitly"]
fn single_encode_keeps_exact_source_transform_and_emits_displayed_thumbnail_pixels() {
    let source = fixture();
    assert_eq!(matrix_and_size(&source), (MATRIX, (64, 48)));
    let scratch = Scratch::new();
    let output = scratch.file("single.mov");
    let sink = Sink::default();
    let result = futures_executor::block_on(
        VideoEncoder::new()
            .with_backends(platform_encode_backends())
            .encode(
                &source,
                &output,
                &preset(32, VideoFileType::Mov),
                Some(options()),
                ProgressSink::noop(),
                Some(&sink),
            ),
    )
    .unwrap();
    assert_eq!(
        (result.width, result.height, result.thumbnails_emitted),
        (32, 24, 4)
    );
    assert_eq!(result.output_path, output);
    assert_eq!(result.bytes, std::fs::metadata(&output).unwrap().len());
    verify_file(&source, &output, 32, 24);
    verify_thumbs(&source, &sink, &["single"]);
    fixture();
}
// Kill: keep metadata only on the single door, normalize matrix translation
// during one variant's resize, or rotate the shared decoded picture twice.
#[test]
#[ignore = "real platform H264 codecs and independent FFmpeg decode; run explicitly"]
fn shared_decode_keeps_each_resized_variant_transform_and_its_displayed_strip() {
    let source = fixture();
    let scratch = Scratch::new();
    let sink = Sink::default();
    let variants = vec![
        VideoEncodingVariant {
            id: "large".into(),
            output_path: scratch.file("large.mov"),
            preset: preset(48, VideoFileType::Mov),
            thumb_strip: Some(options()),
        },
        VideoEncodingVariant {
            id: "small".into(),
            output_path: scratch.file("small.mp4"),
            preset: preset(32, VideoFileType::Mp4),
            thumb_strip: Some(options()),
        },
    ];
    let result = futures_executor::block_on(
        VideoEncoder::new()
            .with_backends(platform_encode_backends())
            .encode_variants(&source, &variants, ProgressSink::noop(), Some(&sink)),
    )
    .unwrap();
    assert_eq!(result.decoded_frames, 12);
    assert_eq!(result.variants.len(), 2);
    for (variant, size) in variants.iter().zip([(48, 36), (32, 24)]) {
        let row = &result.variants[&variant.id];
        assert_eq!(
            (row.width, row.height, row.thumbnails_emitted),
            (size.0, size.1, 4)
        );
        assert_eq!(row.output_path, variant.output_path);
        verify_file(&source, &variant.output_path, size.0, size.1);
    }
    verify_thumbs(&source, &sink, &["large", "small"]);
    fixture();
}
