//! Real compressed fixtures across the hvc1 configuration/sample boundary.
//! No encoder substitute: every successful remux must retain the picture NALs,
//! timing and all twelve decoded frames. These tests do not qualify Android's
//! hardware encoder; the Annex-B inputs reproduce its public mux contract.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_media::platform::libav::{LibavMuxer, LibavVideoIo, PacketReader};
use ferment_engine_media::{
    Codec, CpuFrame, DecoderConfiguration, EncodedPacket, MediaTime, MuxSettings, Muxer,
    Packetization, VideoError, VideoIo, VideoMuxTrack, demux,
};
use futures_executor::block_on;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferment-hvc1-mux-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&directory).expect("new owned mux scratch directory");
        Self(directory)
    }
    fn file(&self) -> PathBuf {
        self.0.join("output.mp4")
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("HVC1_MUX_FAILURE_ARTIFACT {}", self.0.display());
        } else {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

struct Fixture {
    path: PathBuf,
    track: VideoMuxTrack,
    packets: Vec<EncodedPacket>,
}
fn fixture(name: &str) -> Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    let info = demux::probe(&path).expect("required real fixture");
    let video = info.video_streams().next().expect("fixture video");
    let track = VideoMuxTrack {
        display_matrix: None,
        configuration: video.decoder_configuration.clone(),
        width: video.width.unwrap(),
        height: video.height.unwrap(),
        time_base: video.time_base,
        color: video.color,
        duration: video.duration,
    };
    let packets = packets(&path);
    assert_eq!(
        packets.len(),
        12,
        "the admitted original fixture has twelve pictures"
    );
    Fixture {
        path,
        track,
        packets,
    }
}
fn packets(path: &Path) -> Vec<EncodedPacket> {
    let info = demux::probe(path).unwrap();
    let mut reader = PacketReader::open(path, info.video_streams().next().unwrap().index).unwrap();
    let mut result = Vec::new();
    while let Some(packet) = reader.next_packet().unwrap() {
        result.push(packet);
    }
    result
}

#[derive(Clone)]
struct Array {
    kind: u8,
    complete: bool,
    nals: Vec<Vec<u8>>,
}
fn u16_at(bytes: &[u8], cursor: &mut usize) -> usize {
    let end = cursor.checked_add(2).unwrap();
    let value = u16::from_be_bytes(bytes[*cursor..end].try_into().unwrap());
    *cursor = end;
    usize::from(value)
}
/// Independently inspect the actual hvcC bytes, including the completeness
/// bits that DecoderConfiguration intentionally does not expose.
fn arrays(description: &[u8]) -> Vec<Array> {
    assert!(description.len() >= 23 && description[0] == 1);
    let mut cursor = 23;
    let mut result = Vec::new();
    for _ in 0..description[22] {
        let header = description[cursor];
        cursor += 1;
        let count = u16_at(description, &mut cursor);
        let mut nals = Vec::new();
        for _ in 0..count {
            let size = u16_at(description, &mut cursor);
            let end = cursor.checked_add(size).unwrap();
            nals.push(description[cursor..end].to_vec());
            cursor = end;
        }
        result.push(Array {
            kind: header & 63,
            complete: header & 128 != 0,
            nals,
        });
    }
    assert_eq!(cursor, description.len(), "unparsed hvcC bytes");
    result
}
fn hvcc(original: &[u8], groups: &[Array], width: u8) -> Vec<u8> {
    assert!([1, 2, 4].contains(&width));
    let mut result = original[..23].to_vec();
    result[21] = (result[21] & !3) | (width - 1);
    result[22] = u8::try_from(groups.len()).unwrap();
    for group in groups {
        result.push(group.kind | if group.complete { 128 } else { 0 });
        result.extend_from_slice(&u16::try_from(group.nals.len()).unwrap().to_be_bytes());
        for nal in &group.nals {
            result.extend_from_slice(&u16::try_from(nal.len()).unwrap().to_be_bytes());
            result.extend_from_slice(nal);
        }
    }
    result
}
fn kind(nal: &[u8]) -> u8 {
    assert!(nal.len() >= 2, "HEVC NAL requires its two-byte header");
    (nal[0] >> 1) & 63
}
fn is_ps(nal: &[u8]) -> bool {
    matches!(kind(nal), 32..=34)
}
fn length_units(packet: &EncodedPacket) -> Vec<Vec<u8>> {
    let Packetization::LengthPrefixed { nal_length_size } = packet.packetization else {
        panic!("an actual MP4 fixture/output must declare length-prefixed samples");
    };
    let width = usize::from(nal_length_size);
    assert!((1..=4).contains(&width));
    let mut cursor = 0;
    let mut result = Vec::new();
    while cursor < packet.data.len() {
        let end = cursor + width;
        let size = packet.data[cursor..end]
            .iter()
            .fold(0usize, |n, b| n * 256 + usize::from(*b));
        assert!(size > 0, "empty NAL in an actual output sample");
        cursor = end;
        let end = cursor.checked_add(size).unwrap();
        result.push(packet.data[cursor..end].to_vec());
        cursor = end;
    }
    assert!(!result.is_empty(), "empty video sample");
    result
}
fn serialize(nals: &[Vec<u8>], framing: Packetization) -> Vec<u8> {
    let mut bytes = Vec::new();
    for nal in nals {
        match framing {
            Packetization::AnnexB => bytes.extend_from_slice(&[0, 0, 0, 1]),
            Packetization::LengthPrefixed { nal_length_size } => {
                let width = usize::from(nal_length_size);
                assert!((1..=4).contains(&width));
                let length = u32::try_from(nal.len()).unwrap();
                assert!(u64::from(length) < (1u64 << (8 * width)));
                bytes.extend_from_slice(&length.to_be_bytes()[4 - width..]);
            }
            Packetization::Elementary => panic!("NAL fixture needs explicit framing"),
        }
        bytes.extend_from_slice(nal);
    }
    bytes
}
fn reframed(packet: &EncodedPacket, nals: &[Vec<u8>], framing: Packetization) -> EncodedPacket {
    EncodedPacket {
        data: serialize(nals, framing),
        packetization: framing,
        ..packet.clone()
    }
}
fn configuration(source: &DecoderConfiguration, framing: Packetization) -> DecoderConfiguration {
    let groups = arrays(&source.description);
    let description = match framing {
        Packetization::AnnexB => serialize(
            &groups
                .iter()
                .flat_map(|group| group.nals.clone())
                .collect::<Vec<_>>(),
            framing,
        ),
        Packetization::LengthPrefixed { nal_length_size } => {
            hvcc(&source.description, &groups, nal_length_size)
        }
        Packetization::Elementary => unreachable!(),
    };
    DecoderConfiguration::from_codec_private(Codec::Hevc, description).unwrap()
}
fn same_time(actual: Option<MediaTime>, expected: Option<MediaTime>, label: &str) {
    match (actual, expected) {
        (Some(a), Some(b)) => assert_eq!(
            i128::from(a.value)
                * i128::from(a.time_base.numerator)
                * i128::from(b.time_base.denominator),
            i128::from(b.value)
                * i128::from(b.time_base.numerator)
                * i128::from(a.time_base.denominator),
            "{label}: exact rational time changed",
        ),
        _ => assert_eq!(actual, expected, "{label}: timestamp presence changed"),
    }
}
fn compare_packets(actual: &[EncodedPacket], expected: &[EncodedPacket], hevc: bool) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "a picture was lost or an empty PS sample was manufactured"
    );
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        let a_units = length_units(a);
        let mut b_units = length_units(b);
        if hevc {
            assert!(
                a_units.iter().all(|nal| !is_ps(nal)),
                "picture {index} retained inline parameter sets"
            );
            b_units.retain(|nal| !is_ps(nal));
        }
        assert_eq!(
            a_units, b_units,
            "picture {index} changed non-parameter NAL bytes/order"
        );
        same_time(a.pts, b.pts, "PTS");
        same_time(a.dts, b.dts, "DTS");
        same_time(a.duration, b.duration, "duration");
        assert_eq!(
            a.is_sync, b.is_sync,
            "picture {index} changed its random-access flag"
        );
    }
}
fn decoded(path: &Path) -> Vec<(MediaTime, CpuFrame)> {
    block_on(async {
        let backend = LibavVideoIo;
        let info = backend.probe(path).await.unwrap();
        let mut decoder = backend
            .open_decoder(path, info.video_streams().next().unwrap().index)
            .await
            .unwrap();
        let mut frames = Vec::new();
        while let Some(frame) = decoder.decode_next().await.unwrap() {
            frames.push((frame.pts, frame.to_cpu_rgb().unwrap()));
        }
        assert_eq!(frames.len(), 12, "full decode must drain the last GOP");
        frames
    })
}
fn compare_decode(output: &Path, source: &Path) {
    let expected = decoded(source);
    let actual = decoded(output);
    for ((at, a), (bt, b)) in actual.iter().zip(&expected) {
        same_time(Some(*at), Some(*bt), "decoded PTS");
        assert_eq!(a, b, "muxing changed the actual decoded picture");
    }
}
fn assert_hvc1(path: &Path) {
    let input = ffmpeg_next::format::input(path).unwrap();
    let video = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .unwrap();
    let parameters = video.parameters();
    // SAFETY: parameters owns a live AVCodecParameters for this input stream.
    let tag = unsafe { (*parameters.as_ptr()).codec_tag };
    assert_eq!(
        tag.to_le_bytes(),
        *b"hvc1",
        "wrong actual MP4 sample-entry tag"
    );
    let info = demux::probe(path).unwrap();
    let config = &info.video_streams().next().unwrap().decoder_configuration;
    let groups = arrays(&config.description);
    for expected in [32, 33, 34] {
        let group = groups
            .iter()
            .find(|array| array.kind == expected)
            .expect("complete VPS/SPS/PPS array");
        assert!(
            group.complete && !group.nals.is_empty(),
            "hvc1 has an incomplete parameter array"
        );
        assert!(group.nals.iter().all(|nal| kind(nal) == expected));
    }
    assert!(
        groups
            .iter()
            .filter(|a| !matches!(a.kind, 32..=34))
            .all(|a| !a.complete)
    );
}
fn success(framing: Packetization, ps_only: bool) {
    let source = fixture("hevc-gop4.mp4");
    let groups = arrays(&source.track.configuration.description);
    let sets = groups
        .iter()
        .filter(|a| matches!(a.kind, 32..=34))
        .flat_map(|a| a.nals.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        sets.len(),
        3,
        "the admitted fixture supplies one VPS, SPS and PPS"
    );
    let sei = groups.iter().find(|a| a.kind == 39).unwrap().nals[0].clone();
    let mut expected = source.packets.clone();
    // Keep a real prefix-SEI payload and a valid AUD in-band to kill filters
    // that accidentally remove every non-VCL NAL rather than only 32/33/34.
    let mut first = vec![vec![35 << 1, 1, 0x10], sei];
    first.extend(length_units(&expected[0]));
    expected[0] = reframed(&expected[0], &first, expected[0].packetization);
    let mut track = source.track.clone();
    track.configuration = configuration(&track.configuration, framing);
    if ps_only {
        track.duration = None;
    }
    let scratch = Scratch::new();
    let output = scratch.file();
    let mut mux = LibavMuxer.open(&output, MuxSettings::default()).unwrap();
    let id = mux.add_video_track(track).unwrap();
    if ps_only {
        let mut packet = reframed(&expected[0], &sets, framing);
        // A skipped configuration buffer must not contribute a fake extent.
        packet.duration = Some(MediaTime::new(90_000, source.track.time_base));
        mux.write_packet(id, packet).unwrap();
    }
    for (index, packet) in expected.iter().enumerate() {
        let mut nals = if index == 0 { sets.clone() } else { Vec::new() };
        nals.extend(length_units(packet));
        mux.write_packet(id, reframed(packet, &nals, framing))
            .unwrap();
    }
    let delivered = mux.finalize().unwrap();
    same_time(
        Some(delivered.duration),
        source.track.duration,
        "delivered video extent",
    );
    assert_hvc1(&output);
    let actual = packets(&output);
    compare_packets(&actual, &expected, true);
    if let Packetization::LengthPrefixed { nal_length_size } = framing {
        let expected_framing = Packetization::LengthPrefixed { nal_length_size };
        assert!(actual.iter().all(|p| p.packetization == expected_framing));
    }
    compare_decode(&output, &source.path);
}

/// Kill: merely retag hev1 while leaving duplicate parameter sets in samples.
#[test]
fn hvcc_samples_become_complete_hvc1_without_changing_pictures_or_timing() {
    success(Packetization::LengthPrefixed { nal_length_size: 4 }, false);
}
/// Kill: silently force four-byte NAL sizes while hvcC still declares two.
#[test]
fn hvcc_two_byte_lengths_remain_consistent_with_the_actual_samples() {
    success(Packetization::LengthPrefixed { nal_length_size: 2 }, false);
}
/// Kill: preconvert Android packets and make movenc convert them a second time.
#[test]
fn android_annex_b_configuration_and_samples_are_converted_exactly_once() {
    success(Packetization::AnnexB, false);
}
/// Kill: create an empty picture or alter extent from a pure configuration buffer.
#[test]
fn parameter_only_packets_do_not_create_a_picture_or_extend_the_video() {
    for framing in [
        Packetization::AnnexB,
        Packetization::LengthPrefixed { nal_length_size: 4 },
    ] {
        success(framing, true);
    }
}
fn assert_bad_configuration(source: &Fixture, description: Vec<u8>) {
    let scratch = Scratch::new();
    let mut mux = LibavMuxer
        .open(&scratch.file(), MuxSettings::default())
        .unwrap();
    let mut track = source.track.clone();
    // Keep the old cached fields on purpose: the public boundary must parse
    // the actual description and cannot trust a contradictory cached vector.
    track.configuration.description = description;
    let error = mux
        .add_video_track(track)
        .expect_err("invalid out-of-band HEVC configuration was admitted");
    assert!(
        matches!(error, VideoError::InvalidData(_)),
        "unexpected refusal: {error:?}"
    );
}
/// Kill: claim hvc1 completeness despite a missing VPS, SPS or PPS.
#[test]
fn a_missing_out_of_band_parameter_kind_is_refused_in_both_representations() {
    let source = fixture("hevc-gop4.mp4");
    let all = arrays(&source.track.configuration.description);
    for missing in [32, 33, 34] {
        let groups = all
            .iter()
            .filter(|a| a.kind != missing)
            .cloned()
            .collect::<Vec<_>>();
        assert_bad_configuration(
            &source,
            hvcc(&source.track.configuration.description, &groups, 4),
        );
        let nals = groups.into_iter().flat_map(|a| a.nals).collect::<Vec<_>>();
        assert_bad_configuration(&source, serialize(&nals, Packetization::AnnexB));
    }
}
/// Kill: trust hvcC labels/cached sets, accept an empty NAL, or defer a malformed
/// configuration until movenc silently discards its hvcC writer error.
#[test]
fn malformed_or_mislabelled_out_of_band_parameter_sets_are_refused() {
    let source = fixture("hevc-gop4.mp4");
    let all = arrays(&source.track.configuration.description);
    let mut wrong_kind = all.clone();
    wrong_kind[0].nals[0][0] = 33 << 1;
    assert_bad_configuration(
        &source,
        hvcc(&source.track.configuration.description, &wrong_kind, 4),
    );
    let mut empty = all.clone();
    empty[0].nals[0].clear();
    assert_bad_configuration(
        &source,
        hvcc(&source.track.configuration.description, &empty, 4),
    );
    assert_bad_configuration(
        &source,
        source.track.configuration.description[..24].to_vec(),
    );
    let mut truncated = vec![vec![32 << 1]];
    truncated.extend(all.iter().skip(1).flat_map(|a| a.nals.clone()));
    assert_bad_configuration(&source, serialize(&truncated, Packetization::AnnexB));
}
fn assert_bad_packet(source: &Fixture, framing: Packetization, bad: EncodedPacket) {
    let scratch = Scratch::new();
    let mut mux = LibavMuxer
        .open(&scratch.file(), MuxSettings::default())
        .unwrap();
    let mut track = source.track.clone();
    track.configuration = configuration(&track.configuration, framing);
    let id = mux.add_video_track(track).unwrap();
    let first = &source.packets[0];
    mux.write_packet(id, reframed(first, &length_units(first), framing))
        .unwrap();
    let error = mux
        .write_packet(id, bad)
        .expect_err("invalid later HEVC packet was accepted");
    assert!(
        matches!(error, VideoError::InvalidData(_)),
        "unexpected refusal: {error:?}"
    );
}
/// Kill: silently strip a newly introduced parameter set the hvcC cannot supply.
#[test]
fn an_unannounced_parameter_set_after_admission_is_refused() {
    let source = fixture("hevc-gop4.mp4");
    let mut vps = arrays(&source.track.configuration.description)
        .into_iter()
        .find(|a| a.kind == 32)
        .unwrap()
        .nals
        .remove(0);
    // The first four RBSP bits are vps_video_parameter_set_id; changing that
    // ID creates a different, syntactically formed VPS with the same payload.
    vps[2] ^= 0x10;
    for framing in [
        Packetization::AnnexB,
        Packetization::LengthPrefixed { nal_length_size: 4 },
    ] {
        let mut nals = vec![vps.clone()];
        nals.extend(length_units(&source.packets[1]));
        assert_bad_packet(
            &source,
            framing,
            reframed(&source.packets[1], &nals, framing),
        );
    }
}
/// Kill: truncate, overrun, or accept empty NALs at the actual mux boundary.
#[test]
fn malformed_hevc_packet_lengths_and_headers_are_refused() {
    let source = fixture("hevc-gop4.mp4");
    for (framing, data) in [
        (
            Packetization::LengthPrefixed { nal_length_size: 4 },
            vec![0, 0, 0, 5, 0x26, 1],
        ),
        (
            Packetization::LengthPrefixed { nal_length_size: 4 },
            vec![0, 0, 0, 0],
        ),
        (Packetization::AnnexB, vec![0, 0, 0, 1, 32 << 1]),
        (Packetization::AnnexB, vec![0, 0, 1]),
    ] {
        assert_bad_packet(
            &source,
            framing,
            EncodedPacket {
                data,
                packetization: framing,
                ..source.packets[1].clone()
            },
        );
    }
}
/// Kill: fix only HEVC but leave the same Annex-B double conversion in AVC.
#[test]
fn h264_annex_b_configuration_and_samples_retain_every_packet_and_decoded_frame() {
    let source = fixture("h264-gop4.mp4");
    let nals = source
        .track
        .configuration
        .parameter_sets
        .iter()
        .map(|p| p.bytes.clone())
        .collect::<Vec<_>>();
    assert!(!nals.is_empty());
    let mut track = source.track.clone();
    track.configuration = DecoderConfiguration::from_codec_private(
        Codec::H264,
        serialize(&nals, Packetization::AnnexB),
    )
    .unwrap();
    let scratch = Scratch::new();
    let output = scratch.file();
    let mut mux = LibavMuxer.open(&output, MuxSettings::default()).unwrap();
    let id = mux.add_video_track(track).unwrap();
    for packet in &source.packets {
        mux.write_packet(
            id,
            reframed(packet, &length_units(packet), Packetization::AnnexB),
        )
        .unwrap();
    }
    mux.finalize().unwrap();
    compare_packets(&packets(&output), &source.packets, false);
    compare_decode(&output, &source.path);
}
