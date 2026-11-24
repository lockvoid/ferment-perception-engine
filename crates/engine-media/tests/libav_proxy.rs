//! Editing proxies through the shipped libav: ProRes 422 Proxy with every
//! picture a key frame, the short edge capped, and the source's own
//! timestamps, colour tags and orientation.
use ferment_engine_media::platform::libav::{LibavVideoIo, PacketReader, probe, write_proxy};
use ferment_engine_media::{
    Codec, DecodeAcceleration, DecodeOptions, MediaInput, ProResProfile, TransferCharacteristic,
    VideoIo,
};
use futures_executor::block_on;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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

/// A fresh folder of this test's own, removed when it ends.
struct Folder(PathBuf);
impl Folder {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("ferment-proxy-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn input(path: &Path) -> Box<dyn MediaInput> {
    Box::new(FixtureBytes(Cursor::new(std::fs::read(path).unwrap())))
}

/// 2 s of 640×360 HLG at 30 fps in 10-bit HEVC: one-second GOPs with
/// B-frames, the shape of a phone clip. A stream-copy remux then gives it a
/// 90° display matrix over the same coded pictures.
fn long_gop_hlg(root: &Path) -> PathBuf {
    let coded = root.join("coded.mov");
    let path = root.join("source.mov");
    let ffmpeg = |args: &[&str]| {
        assert!(
            std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-nostdin"])
                .args(args)
                .status()
                .unwrap()
                .success(),
            "{args:?}"
        )
    };
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=2",
        "-c:v",
        "libx265",
        "-x265-params",
        "log-level=error:keyint=30:min-keyint=30:bframes=3:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
        "-pix_fmt",
        "yuv420p10le",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "arib-std-b67",
        "-colorspace",
        "bt2020nc",
        "-color_range",
        "tv",
        coded.to_str().unwrap(),
    ]);
    ffmpeg(&[
        "-display_rotation:v:0",
        "90",
        "-i",
        coded.to_str().unwrap(),
        "-c",
        "copy",
        path.to_str().unwrap(),
    ]);
    path
}

fn timestamps(path: &Path) -> Vec<f64> {
    let mut decoder = block_on(LibavVideoIo.open_decoder_with(
        path,
        0,
        DecodeOptions {
            acceleration: DecodeAcceleration::Software,
            ..Default::default()
        },
    ))
    .unwrap();
    let mut seconds = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
        seconds.push(frame.pts.seconds());
    }
    seconds
}

/// Kill: a proxy with inter frames, a re-timed or dropped picture, lost
/// HLG/BT.2020 tags or display matrix, a raised extent, or a progress that
/// does not reach the end.
#[test]
fn a_long_gop_hlg_clip_becomes_an_intra_prores_proxy_with_its_own_timing_and_colour() {
    let root = Folder::new("timing");
    let source = long_gop_hlg(root.path());
    let output = root.path().join("proxy.mov");
    let mut done = Vec::new();
    let report = write_proxy(input(&source), 0, &output, 180, &mut |share| {
        done.push(share);
        true
    })
    .unwrap();
    assert_eq!((report.width, report.height), (320, 180), "{report:?}");
    assert_eq!(report.pictures, 60);
    assert!(done.windows(2).all(|pair| pair[0] <= pair[1]), "{done:?}");
    assert!(done.last().is_some_and(|share| *share > 0.95), "{done:?}");

    let original = probe(&source).unwrap();
    let original = original.video_streams().next().unwrap();
    let info = probe(&output).unwrap();
    let proxy = info.video_streams().next().unwrap();
    assert_eq!(proxy.codec, Codec::ProRes(ProResProfile::Proxy));
    assert_eq!((proxy.width, proxy.height), (Some(320), Some(180)));
    assert_eq!(proxy.color.transfer, TransferCharacteristic::Hlg);
    assert_eq!(proxy.color.primaries, original.color.primaries);
    assert_eq!(proxy.color.matrix, original.color.matrix);
    assert!(original.display_matrix.is_some());
    assert_eq!(proxy.display_matrix, original.display_matrix);

    let mut packets = PacketReader::open(&output, 0).unwrap();
    let mut count = 0;
    while let Some(packet) = packets.next_packet().unwrap() {
        assert!(packet.is_sync, "packet {count} is not a key frame");
        count += 1;
    }
    assert_eq!(count, 60);
    let (source, proxy) = (timestamps(&source), timestamps(&output));
    assert_eq!(source.len(), 60);
    for (a, b) in source.iter().zip(&proxy) {
        assert!((a - b).abs() < 1e-6, "{source:?} {proxy:?}");
    }
}

/// Kill: a cancelled proxy that still reports success.
#[test]
fn a_proxy_stops_when_its_progress_says_so() {
    let root = Folder::new("cancel");
    let source = long_gop_hlg(root.path());
    let mut calls = 0;
    let stopped = write_proxy(
        input(&source),
        0,
        &root.path().join("proxy.mov"),
        180,
        &mut |_| {
            calls += 1;
            calls < 5
        },
    );
    assert!(stopped.is_err());
    assert_eq!(calls, 5);
}
