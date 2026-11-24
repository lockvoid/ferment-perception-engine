//! Container output through `LibavMuxer`.
//!
//! This is `src/mux.rs`'s test module rewritten against the libav muxer
//! (ruling 12), and rewritten in the way that ruling asks for: where the old
//! tests reached into atom offsets and sample tables, these assert what a
//! reader can see. The strongest of them is a REMUX — every packet of a real
//! fixture read out and written back — because that exercises the codec
//! description, the timestamps, the interleave and the trailer at once, and
//! the source file is the expected answer.

use std::path::{Path, PathBuf};

use ferment_engine_media::platform::libav::{LibavMuxer, PacketReader};
use ferment_engine_media::{
    AudioMuxTrack, AudioTiming, Codec, DecoderConfiguration, EncodedPacket, MediaKind, MediaTime,
    MuxFormat, MuxSettings, Muxer, Packetization, TimeBase, VideoMuxTrack, demux,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name)
}

/// A scratch path that removes itself.
struct Scratch(PathBuf);

impl Scratch {
    fn new(suffix: &str) -> Self {
        Self(
            std::env::temp_dir().join(format!("ferment-libav-mux-{}-{suffix}", std::process::id())),
        )
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Read every packet of `source`'s video stream, write them into a new
/// container, and describe the result.
fn remux(source: &Path, output: &Path, settings: MuxSettings) -> usize {
    let info = demux::probe(source).expect("probe the source");
    let video = info
        .video_streams()
        .next()
        .unwrap_or_else(|| panic!("no video stream in {}", source.display()));

    let mut reader = PacketReader::open(source, video.index).expect("open the packet reader");
    let mut session = LibavMuxer.open(output, settings).expect("open the muxer");
    let track = session
        .add_video_track(VideoMuxTrack {
            display_matrix: None,
            configuration: video.decoder_configuration.clone(),
            width: video.width.expect("a video width"),
            height: video.height.expect("a video height"),
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
    session.finalize().expect("finalize");
    written
}

/// The round trip. Twelve packets in, twelve out, and the description a
/// reader recovers matches the source's — codec, dimensions, frame count and
/// duration all survive the trip.
#[test]
fn a_remuxed_fixture_reads_back_as_the_same_media() {
    // ProRes goes to MOV, which is the only container that carries it — see
    // `prores_in_mp4_is_refused` below.
    for (name, format) in [
        ("h264-gop4.mp4", MuxFormat::Mp4),
        ("hevc-gop4.mp4", MuxFormat::Mp4),
        ("prores-standard.mov", MuxFormat::Mov),
    ] {
        let source = fixture(name);
        let scratch = Scratch::new(&format!("remux-{name}"));
        let written = remux(
            &source,
            &scratch.0,
            MuxSettings {
                format,
                ..MuxSettings::default()
            },
        );
        assert_eq!(written, 12, "{name}");

        let before = demux::probe(&source).unwrap();
        let after = demux::probe(&scratch.0).unwrap();
        let source_video = before.video_streams().next().unwrap();
        let output_video = after
            .video_streams()
            .next()
            .unwrap_or_else(|| panic!("{name}: the remuxed file has no video stream"));

        assert_eq!(output_video.codec, source_video.codec, "{name}");
        assert_eq!(output_video.width, source_video.width, "{name}");
        assert_eq!(output_video.height, source_video.height, "{name}");
        let mut expected_description = source_video.decoder_configuration.description.clone();
        if source_video.codec == Codec::Hevc {
            // hvc1 makes the three parameter-set arrays complete. The source
            // fixture is hev1; every other codec-description byte must survive.
            assert!(expected_description.len() >= 23 && expected_description[0] == 1);
            let mut cursor = 23;
            let mut present = [false; 3];
            for _ in 0..expected_description[22] {
                let kind = expected_description[cursor] & 63;
                let count = u16::from_be_bytes(
                    expected_description[cursor + 1..cursor + 3]
                        .try_into()
                        .unwrap(),
                );
                if (32..=34).contains(&kind) {
                    assert!(count > 0);
                    expected_description[cursor] |= 128;
                    present[(kind - 32) as usize] = true;
                }
                cursor += 3;
                for _ in 0..count {
                    let bytes = u16::from_be_bytes(
                        expected_description[cursor..cursor + 2].try_into().unwrap(),
                    ) as usize;
                    cursor += 2 + bytes;
                    assert!(cursor <= expected_description.len());
                }
            }
            assert_eq!(present, [true; 3]);
            assert_eq!(cursor, expected_description.len());
        }
        assert_eq!(
            output_video.decoder_configuration.description, expected_description,
            "{name}: codec bytes changed beyond the required hvc1 completeness bits",
        );

        let (before_seconds, after_seconds) = (
            source_video.duration.unwrap().seconds(),
            output_video.duration.unwrap().seconds(),
        );
        assert!(
            (before_seconds - after_seconds).abs() < 0.001,
            "{name}: duration {before_seconds} became {after_seconds}",
        );
    }
}

/// ProRes in MP4 is refused, loudly.
///
/// The hand-written muxer wrote it — the corpus still carries a
/// `prores-standard.mp4` from those days — but no other tool accepts the
/// result, so a silent success was the worse outcome. Nothing in the pipeline
/// asks for it: the proxy preset, the only one that reaches for ProRes, is
/// `VideoFileType::Mov`.
#[test]
fn prores_in_mp4_is_refused() {
    let scratch = Scratch::new("prores.mp4");
    let info = demux::probe(fixture("prores-standard.mov")).unwrap();
    let video = info.video_streams().next().unwrap();
    let mut session = LibavMuxer.open(&scratch.0, MuxSettings::default()).unwrap();
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
        .unwrap();
    let mut reader = PacketReader::open(&fixture("prores-standard.mov"), video.index).unwrap();
    let packet = reader.next_packet().unwrap().unwrap();
    assert!(
        session.write_packet(track, packet).is_err(),
        "ProRes in MP4 should be refused rather than written",
    );
}

/// Colour tags reach the container. This is the `colr` atom, and getting it
/// wrong makes an HDR export play back as washed-out SDR — a failure a
/// structural test catches and an eyeball does not, until it is shipped.
#[test]
fn colour_tags_survive_the_container() {
    let source = fixture("synthetic-hdr-nclx.mp4");
    let scratch = Scratch::new("hdr");
    remux(&source, &scratch.0, MuxSettings::default());

    let before = demux::probe(&source).unwrap();
    let after = demux::probe(&scratch.0).unwrap();
    let source_color = before.video_streams().next().unwrap().color;
    let output_color = after.video_streams().next().unwrap().color;

    assert_eq!(output_color.primaries, source_color.primaries);
    assert_eq!(output_color.transfer, source_color.transfer);
    assert_eq!(output_color.matrix, source_color.matrix);
    assert_eq!(output_color.range, source_color.range);
}

/// `MuxFormat` picks the flavour, and a reader can tell which it got.
#[test]
fn both_container_flavours_are_written_and_readable() {
    for (format, suffix) in [(MuxFormat::Mp4, "mp4"), (MuxFormat::Mov, "mov")] {
        let scratch = Scratch::new(&format!("flavour.{suffix}"));
        remux(
            &fixture("h264-gop4.mp4"),
            &scratch.0,
            MuxSettings {
                format,
                ..MuxSettings::default()
            },
        );
        let info = demux::probe(&scratch.0)
            .unwrap_or_else(|error| panic!("{format:?} output did not probe: {error}"));
        assert_eq!(info.video_streams().count(), 1, "{format:?}");
    }
}

/// `fast_start` moves `moov` ahead of `mdat` in a second pass. The observable
/// is the byte order of the two atoms: a player that has to seek to the end
/// of the file before it can start is the thing this setting exists to avoid.
#[test]
fn fast_start_puts_the_movie_header_before_the_media_data() {
    for fast_start in [true, false] {
        let scratch = Scratch::new(&format!("faststart-{fast_start}"));
        remux(
            &fixture("h264-gop4.mp4"),
            &scratch.0,
            MuxSettings {
                fast_start,
                ..MuxSettings::default()
            },
        );
        let bytes = std::fs::read(&scratch.0).unwrap();
        let moov = find_atom(&bytes, b"moov").expect("a moov atom");
        let mdat = find_atom(&bytes, b"mdat").expect("an mdat atom");
        if fast_start {
            assert!(
                moov < mdat,
                "faststart left moov at {moov}, after mdat at {mdat}"
            );
        } else {
            assert!(mdat < moov, "without faststart moov should trail the media");
        }
    }
}

/// AAC encoder priming survives as an edit list.
///
/// This is the gapless-playback contract: the decoder must be told to drop
/// the first 2112 samples, and the container is where that instruction lives.
/// The hand-written muxer wrote the `edts` and `iTunSMPB` by hand; libavformat
/// writes them from `initial_padding`, and the number has to come back.
#[test]
fn aac_priming_survives_as_an_edit_list() {
    let scratch = Scratch::new("gapless.m4a");
    let timing = AudioTiming {
        leading_samples: 2_112,
        trailing_samples: 0,
        audible_samples: Some(5_389),
    };
    let mut session = LibavMuxer.open(&scratch.0, MuxSettings::default()).unwrap();
    let track = session
        .add_audio_track(AudioMuxTrack {
            // A real AudioSpecificConfig: object type 2, 48 kHz, stereo.
            configuration: DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x11, 0x90])
                .unwrap(),
            sample_rate: 48_000,
            channels: 2,
            timing,
        })
        .unwrap();

    let base = TimeBase::new_unchecked(1, 48_000);
    for index in 0..8_i64 {
        session
            .write_packet(
                track,
                EncodedPacket {
                    stream_index: 0,
                    pts: Some(MediaTime::new(index * 1024 - 2_112, base)),
                    dts: Some(MediaTime::new(index * 1024 - 2_112, base)),
                    duration: Some(MediaTime::new(1024, base)),
                    is_sync: true,
                    data: vec![0x21, index as u8],
                    packetization: Packetization::Elementary,
                },
            )
            .unwrap();
    }
    let output = session.finalize().unwrap();
    assert_eq!(output.track_count, 1);
    assert!(
        output.bytes.is_none(),
        "a path-backed session writes a file"
    );

    let info = demux::probe(&scratch.0).unwrap();
    let audio = info
        .streams
        .iter()
        .find(|stream| stream.kind == MediaKind::Audio)
        .expect("an audio stream");
    assert_eq!(audio.sample_rate, Some(48_000));
    assert_eq!(audio.channels, Some(2));
    assert_eq!(
        audio.audio_timing.unwrap().leading_samples,
        2_112,
        "the encoder delay did not survive the container",
    );
}

/// A session with no tracks is a caller error, not an empty file.
#[test]
fn a_container_with_no_tracks_is_refused() {
    let scratch = Scratch::new("empty.mp4");
    let session = LibavMuxer.open(&scratch.0, MuxSettings::default()).unwrap();
    assert!(session.finalize().is_err());
}

/// Adding a track after writing has begun would need a header that is already
/// on disk. Refused, rather than silently dropped.
#[test]
fn a_track_added_after_the_first_packet_is_refused() {
    let scratch = Scratch::new("late-track.mp4");
    let info = demux::probe(fixture("h264-gop4.mp4")).unwrap();
    let video = info.video_streams().next().unwrap();
    let mut session = LibavMuxer.open(&scratch.0, MuxSettings::default()).unwrap();
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
        .unwrap();

    let mut reader = PacketReader::open(&fixture("h264-gop4.mp4"), video.index).unwrap();
    let packet = reader.next_packet().unwrap().unwrap();
    session.write_packet(track, packet).unwrap();

    let error = session
        .add_audio_track(AudioMuxTrack {
            configuration: DecoderConfiguration::from_codec_private(Codec::Aac, vec![0x11, 0x90])
                .unwrap(),
            sample_rate: 48_000,
            channels: 2,
            timing: AudioTiming::default(),
        })
        .unwrap_err();
    assert!(
        error.to_string().contains("after the first packet"),
        "unexpected error: {error}",
    );
}

/// Byte offset of a top-level ISO-BMFF atom, walking the box tree rather than
/// searching for the tag — `moov` appears inside `mdat` payloads by chance
/// often enough that a substring search is not an answer.
fn find_atom(bytes: &[u8], tag: &[u8; 4]) -> Option<usize> {
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        if &bytes[offset + 4..offset + 8] == tag {
            return Some(offset);
        }
        // A 64-bit size lives in the eight bytes after the tag; size 0 means
        // "to end of file". Either way there is nothing after it to find.
        offset += match size {
            0 => return None,
            1 => u64::from_be_bytes(bytes.get(offset + 8..offset + 16)?.try_into().ok()?) as usize,
            size => size,
        };
    }
    None
}

/// The in-memory session: same muxer, bytes instead of a file. The sequencer's
/// export path uses it to hand a finished container straight on without
/// touching disk.
#[test]
fn an_in_memory_session_returns_the_container_as_bytes() {
    let info = demux::probe(fixture("h264-gop4.mp4")).unwrap();
    let video = info.video_streams().next().unwrap();
    let settings = MuxSettings {
        // `faststart` needs a re-readable output; see below.
        fast_start: false,
        ..MuxSettings::default()
    };
    let mut session = LibavMuxer::memory(settings).expect("open an in-memory session");
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
        .unwrap();
    let mut reader = PacketReader::open(&fixture("h264-gop4.mp4"), video.index).unwrap();
    while let Some(packet) = reader.next_packet().unwrap() {
        session.write_packet(track, packet).unwrap();
    }
    let output = session.finalize().unwrap();

    assert!(output.path.is_none(), "an in-memory session names no file");
    let bytes = output.bytes.expect("bytes");
    assert!(bytes.len() > 1_000, "only {} bytes", bytes.len());
    assert_eq!(&bytes[4..8], b"ftyp", "not an ISO-BMFF container");

    // And it is readable: write it out and probe it back.
    let scratch = Scratch::new("memory.mp4");
    std::fs::write(&scratch.0, &bytes).unwrap();
    let probed = demux::probe(&scratch.0).expect("the in-memory output probes");
    assert_eq!(probed.video_streams().count(), 1);
}

/// `fast_start` in memory is refused rather than ignored. Its second pass
/// reopens the output for reading, which a dynamic buffer cannot do — and an
/// export that silently lost faststart is found by a player, not by us.
#[test]
fn an_in_memory_session_refuses_fast_start() {
    let error = LibavMuxer::memory(MuxSettings {
        fast_start: true,
        ..MuxSettings::default()
    })
    .unwrap_err();
    assert!(
        error.to_string().contains("fast-start"),
        "unexpected error: {error}",
    );
}
