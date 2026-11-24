//! Real muxer metadata and packet tests, independent of the encoder transform
//! implementation. The original files are remuxed; no external CLI encodes,
//! replacement goldens, pixel-orientation inference or packet mutation.
use ferment_engine_media::platform::libav::{LibavMuxer, PacketReader};
use ferment_engine_media::{
    Codec, EncodedPacket, MediaTime, MuxFormat, MuxSettings, Muxer, TransferCharacteristic,
    VideoMuxTrack, demux,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const IDENTITY: [i32; 9] = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];
const ROTATED: [i32; 9] = [0, 65536, 0, -65536, 0, 0, 125829120, 851968, 1073741824];
const MIRRORED: [i32; 9] = [0, -65536, 0, -65536, 0, 0, 125829120, 70778880, 1073741824];
const AFFINE: [i32; 9] = [
    98304, 16384, 0, -32768, 131072, 0, 125829120, -458752, 1073741824,
];
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Scratch {
    fn new(format: MuxFormat) -> Self {
        let extension = if format == MuxFormat::Mov {
            "mov"
        } else {
            "mp4"
        };
        Self(std::env::temp_dir().join(format!(
            "ferment-matrix-{}-{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            extension
        )))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name)
}
fn packets(path: &Path) -> Vec<EncodedPacket> {
    let info = demux::probe(path).unwrap();
    let index = info.video_streams().next().unwrap().index;
    let mut reader = PacketReader::open(path, index).unwrap();
    let mut packets = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        packets.push(packet);
    }
    packets
}
fn same_time(left: Option<MediaTime>, right: Option<MediaTime>) {
    match (left, right) {
        (None, None) => (),
        (Some(a), Some(b)) => assert_eq!(
            i128::from(a.value)
                * i128::from(a.time_base.numerator)
                * i128::from(b.time_base.denominator),
            i128::from(b.value)
                * i128::from(b.time_base.numerator)
                * i128::from(a.time_base.denominator),
            "exact rational timestamp changed: {a:?} != {b:?}"
        ),
        _ => panic!("timestamp presence changed: {left:?} != {right:?}"),
    }
}
fn same_packets(expected: &[EncodedPacket], actual: &[EncodedPacket]) {
    assert!(!expected.is_empty());
    assert_eq!(expected.len(), actual.len());
    for (a, b) in expected.iter().zip(actual) {
        assert_eq!(a.data, b.data, "compressed sample bytes changed");
        assert_eq!(a.packetization, b.packetization);
        assert_eq!(a.is_sync, b.is_sync);
        same_time(a.pts, b.pts);
        same_time(a.dts, b.dts);
        same_time(a.duration, b.duration);
    }
}
fn remux(source: &Path, matrix: Option<[i32; 9]>, format: MuxFormat) -> Scratch {
    let info = demux::probe(source).unwrap();
    let video = info.video_streams().next().unwrap();
    let output = Scratch::new(format);
    let mut session = LibavMuxer
        .open(
            &output.0,
            MuxSettings {
                format,
                ..MuxSettings::default()
            },
        )
        .unwrap();
    let track = session
        .add_video_track(VideoMuxTrack {
            configuration: video.decoder_configuration.clone(),
            width: video.width.unwrap(),
            height: video.height.unwrap(),
            time_base: video.time_base,
            color: video.color,
            duration: video.duration,
            display_matrix: matrix,
        })
        .unwrap();
    let mut reader = PacketReader::open(source, video.index).unwrap();
    while let Some(packet) = reader.next_packet().unwrap() {
        session.write_packet(track, packet).unwrap();
    }
    session.finalize().unwrap();
    output
}

// A bounded independent ISO-BMFF atom observer. The production writer/probe
// cannot satisfy matrix equality by agreeing on the same metadata mistake.
fn child(data: &[u8], wanted: [u8; 4]) -> &[u8] {
    let mut cursor = 0;
    while cursor + 8 <= data.len() {
        let small = u32::from_be_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        let (size, header) = if small == 1 {
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
        assert!(
            size >= header && size <= data.len() - cursor,
            "invalid atom extent"
        );
        if data[cursor + 4..cursor + 8] == wanted {
            return &data[cursor + header..cursor + size];
        }
        cursor += size;
    }
    panic!("missing atom {}", String::from_utf8_lossy(&wanted));
}
fn track_header(path: &Path) -> ([i32; 9], (u32, u32)) {
    let data = std::fs::read(path).unwrap();
    let track = child(child(&data, *b"moov"), *b"trak");
    let header = child(track, *b"tkhd");
    let offset = match header[0] {
        0 => 40,
        1 => 52,
        version => panic!("unexpected tkhd version{version}"),
    };
    assert!(header.len() >= offset + 44);
    let mut matrix = [0i32; 9];
    for (word, bytes) in matrix
        .iter_mut()
        .zip(header[offset..offset + 36].chunks_exact(4))
    {
        *word = i32::from_be_bytes(bytes.try_into().unwrap());
    }
    let width = u32::from_be_bytes(header[offset + 36..offset + 40].try_into().unwrap());
    let height = u32::from_be_bytes(header[offset + 40..offset + 44].try_into().unwrap());
    (matrix, (width, height))
}
fn sample_entry(path: &Path) -> [u8; 4] {
    let bytes = std::fs::read(path).unwrap();
    let mut node = bytes.as_slice();
    for name in [*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"stsd"] {
        node = child(node, name);
    }
    assert!(node.len() >= 16 && u32::from_be_bytes(node[4..8].try_into().unwrap()) == 1);
    node[12..16].try_into().unwrap()
}

/// Kill: write only a rotation angle, normalize translation to output size,
/// drop mirrors/shear, or bake metadata by changing compressed packets/clocks.
#[test]
fn mp4_and_mov_preserve_the_complete_matrix_and_original_video_samples() {
    let source = fixture("h264-gop4.mp4");
    let original = std::fs::read(&source).unwrap();
    let before = demux::probe(&source).unwrap();
    let video = before.video_streams().next().unwrap();
    let expected = packets(&source);
    assert_eq!(expected.len(), 12);
    for format in [MuxFormat::Mp4, MuxFormat::Mov] {
        for matrix in [ROTATED, MIRRORED, AFFINE] {
            let output = remux(&source, Some(matrix), format);
            let after = demux::probe(&output.0).unwrap();
            let track = after.video_streams().next().unwrap();
            assert_eq!(track.display_matrix, Some(matrix));
            assert_eq!((track.width, track.height), (video.width, video.height));
            assert_eq!(track.color, video.color);
            assert_eq!(track.decoder_configuration, video.decoder_configuration);
            assert_eq!(
                track_header(&output.0),
                (
                    matrix,
                    (video.width.unwrap() * 65536, video.height.unwrap() * 65536)
                )
            );
            same_packets(&expected, &packets(&output.0));
        }
    }
    assert_eq!(std::fs::read(&source).unwrap(), original);
}

/// Kill: add a default display transform to every caller or alter the previous
/// no-matrix container/packet contract while implementing the optional field.
#[test]
fn absent_matrix_keeps_identity_container_and_unchanged_samples() {
    let source = fixture("h264-gop4.mp4");
    let expected = packets(&source);
    for format in [MuxFormat::Mp4, MuxFormat::Mov] {
        let output = remux(&source, None, format);
        let info = demux::probe(&output.0).unwrap();
        let track = info.video_streams().next().unwrap();
        assert_eq!(track.display_matrix, None);
        assert_eq!(track_header(&output.0).0, IDENTITY);
        same_packets(&expected, &packets(&output.0));
    }
}

/// Kill: attaching the matrix overwrites HDR colorimetry, hvc1 codec-private
/// normalization, or sample clocks/bytes. The existing real PQ-tagged HEVC
/// original supplies all media; compare the same mux with/without the matrix.
#[test]
fn matrix_metadata_preserves_real_hevc_hdr_description_and_packet_timing() {
    let source = fixture("synthetic-hdr-nclx.mp4");
    let original = std::fs::read(&source).unwrap();
    let before = demux::probe(&source).unwrap();
    let video = before.video_streams().next().unwrap();
    assert_eq!(video.codec, Codec::Hevc);
    assert_eq!(video.color.transfer, TransferCharacteristic::Pq);
    let plain = remux(&source, None, MuxFormat::Mp4);
    let rotated = remux(&source, Some(ROTATED), MuxFormat::Mp4);
    let plain_info = demux::probe(&plain.0).unwrap();
    let rotated_info = demux::probe(&rotated.0).unwrap();
    let plain_video = plain_info.video_streams().next().unwrap();
    let rotated_video = rotated_info.video_streams().next().unwrap();
    assert_eq!(plain_video.color, video.color);
    assert_eq!(rotated_video.color, video.color);
    assert_eq!(
        rotated_video.decoder_configuration,
        plain_video.decoder_configuration
    );
    assert_eq!(sample_entry(&rotated.0), *b"hvc1");
    assert_eq!(track_header(&rotated.0).0, ROTATED);
    same_time(rotated_video.duration, plain_video.duration);
    same_packets(&packets(&plain.0), &packets(&rotated.0));
    assert_eq!(std::fs::read(&source).unwrap(), original);
}
