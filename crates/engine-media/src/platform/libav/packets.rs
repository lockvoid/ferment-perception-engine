//! Compressed packets out of `libavformat`, for a decoder that is not
//! libavcodec's.
//!
//! macOS keeps its own VideoToolbox decompression session: FFmpeg's
//! VideoToolbox hwaccel waits for every frame it submits
//! (`libavcodec/videotoolbox.c` — `VTDecompressionSessionDecodeFrame`
//! immediately followed by `VTDecompressionSessionWaitForAsynchronousFrames`),
//! which serializes a decoder built to pipeline and measures 2.8× slower than
//! the async session we already have. What that session should NOT keep is a
//! hand-written demuxer, so it reads its packets from here instead.
//!
//! Everything else — Windows, and any software fallback — decodes through
//! `libavcodec` in [`super::decode`] and never touches this.

use std::path::Path;

use ffmpeg_next::codec::packet::traits::{Mut as PacketMut, Ref as PacketRef};
use ffmpeg_next::ffi;

use super::{check_av, init, probe};
use crate::{EncodedPacket, MediaInfo, MediaTime, Packetization, Result, TimeBase, VideoError};

/// One stream's packets, in demux order, from an open `AVFormatContext`.
pub struct PacketReader {
    input: super::input::Input,
    info: MediaInfo,
    stream_index: u32,
    time_base: TimeBase,
    packetization: Packetization,
}

impl std::fmt::Debug for PacketReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PacketReader")
            .field("stream_index", &self.stream_index)
            .field("time_base", &self.time_base)
            .finish_non_exhaustive()
    }
}

impl PacketReader {
    pub fn open(path: &Path, stream_index: u32) -> Result<Self> {
        init()?;
        let info = probe::probe(path)?;
        Self::from_input(super::input::Input::path(path)?, info, stream_index)
    }

    pub fn open_stream(source: Box<dyn crate::MediaInput>, stream_index: u32) -> Result<Self> {
        let mut input = super::input::Input::stream(source)?;
        let info = probe::describe(&mut input);
        input.check()?;
        let info = info?;
        Self::from_input(input.reopen()?, info, stream_index)
    }

    pub fn validate(&mut self) -> Result<()> {
        self.input.check()
    }

    pub(super) fn from_input(
        input: super::input::Input,
        info: MediaInfo,
        stream_index: u32,
    ) -> Result<Self> {
        let stream = info
            .stream(stream_index)
            .ok_or_else(|| VideoError::InvalidData(format!("no stream {stream_index}")))?;
        let time_base = stream.time_base;
        let packetization = stream.decoder_configuration.packetization;
        Ok(Self {
            input,
            info,
            stream_index,
            time_base,
            packetization,
        })
    }

    pub fn info(&self) -> &MediaInfo {
        &self.info
    }

    /// The next packet belonging to this reader's stream, or `None` at end of
    /// file. Packets for other streams are skipped here rather than handed up.
    pub fn next_packet(&mut self) -> Result<Option<EncodedPacket>> {
        self.next_packet_with_discard()
            .map(|packet| packet.map(|(packet, _)| packet))
    }

    /// Decode-only packets must still reach the codec for reference pictures,
    /// but edit-list preroll/tail samples must not become displayed frames.
    /// The boolean is libav's AV_PKT_FLAG_DISCARD, not a seek-time guess.
    pub fn next_packet_with_discard(&mut self) -> Result<Option<(EncodedPacket, bool)>> {
        self.input.check()?;
        loop {
            // A skipped stream's packet must be unreferenced before another
            // av_read_frame overwrites it. The existing Packet Drop owns that.
            let mut packet = ffmpeg_next::codec::packet::Packet::empty();
            // SAFETY: both pointers are live for the call.
            let status =
                unsafe { ffi::av_read_frame(self.input.as_mut_ptr(), packet.as_mut_ptr()) };
            self.input.check()?;
            if status == ffi::AVERROR_EOF {
                return Ok(None);
            }
            check_av("av_read_frame", status)?;
            if packet.stream() != self.stream_index as usize {
                continue;
            }
            return Ok(Some(self.convert(&packet)));
        }
    }

    /// Seek to the sync sample at or before `target`.
    ///
    /// Returns where the demuxer actually landed, which is the caller's cue
    /// for how much decoded output to discard — `AVSEEK_FLAG_BACKWARD` lands
    /// on a keyframe, and that is the only place a decoder can start.
    pub fn seek(&mut self, target: MediaTime) -> Result<MediaTime> {
        self.input.check()?;
        let target = target.rescale(self.time_base);
        // SAFETY: the context is live and the stream index is one of its own.
        check_av("av_seek_frame", unsafe {
            ffi::av_seek_frame(
                self.input.as_mut_ptr(),
                self.stream_index as i32,
                target.value,
                ffi::AVSEEK_FLAG_BACKWARD,
            )
        })?;
        self.input.check()?;
        Ok(target)
    }

    fn convert(&self, packet: &ffmpeg_next::codec::packet::Packet) -> (EncodedPacket, bool) {
        // SAFETY: a live packet; every read below is a plain field.
        let raw = unsafe { &*packet.as_ptr() };
        let at = |value: i64| {
            (value != ffi::AV_NOPTS_VALUE).then(|| MediaTime::new(value, self.time_base))
        };
        (
            EncodedPacket {
                stream_index: self.stream_index,
                pts: at(raw.pts),
                dts: at(raw.dts),
                duration: (raw.duration > 0).then(|| MediaTime::new(raw.duration, self.time_base)),
                is_sync: raw.flags & ffi::AV_PKT_FLAG_KEY != 0,
                data: packet.data().unwrap_or_default().to_vec(),
                packetization: self.packetization,
            },
            raw.flags & ffi::AV_PKT_FLAG_DISCARD != 0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MediaKind;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/generated")
            .join(name)
    }

    fn video_stream(path: &Path) -> u32 {
        probe::probe(path)
            .unwrap()
            .streams
            .iter()
            .find(|stream| stream.kind == MediaKind::Video)
            .unwrap()
            .index
    }

    /// Kill: drop DISCARD metadata or skip dependency packets before the decoder sees them.
    #[test]
    #[ignore = "original edited current iOS movies; FERMENT_ENGINE_FIXTURES_DIR required"]
    fn edited_movies_retain_decode_dependencies_and_mark_exact_visible_samples() {
        let fixtures = std::path::PathBuf::from(
            std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
                .expect("original current iOS fixtures are required"),
        );
        for (name, packet_count, discard_count, visible_count, expected_hash) in [
            (
                "formula1-trackside.mov",
                1354,
                341,
                1013,
                0xbe16_799e_8461_388e_u64,
            ),
            (
                "japanese-rain-garden.mov",
                2994,
                40,
                2954,
                0xb1ea_1113_a9b9_76a0_u64,
            ),
            (
                "dhh-interview.mp4",
                7104,
                0,
                7104,
                0x45cf_1621_3f1e_cc08_u64,
            ),
        ] {
            let path = fixtures.join("Video").join(name);
            assert!(path.is_file(), "missing original {}", path.display());
            let mut reader = PacketReader::open(&path, video_stream(&path)).unwrap();
            let mut all = 0;
            let mut discarded = 0;
            let mut visible = Vec::new();
            while let Some((packet, discard)) = reader.next_packet_with_discard().unwrap() {
                all += 1;
                assert!(
                    !packet.data.is_empty(),
                    "reference packets must retain their compressed bytes"
                );
                if discard {
                    discarded += 1;
                } else {
                    visible.push(packet.pts.unwrap());
                }
            }
            visible.sort_by(|a, b| a.seconds().total_cmp(&b.seconds()));
            assert_eq!(all, packet_count, "{name}");
            assert_eq!(discarded, discard_count, "{name}");
            assert_eq!(visible.len(), visible_count, "{name}");
            let origin = visible[0].value;
            let hash = visible
                .iter()
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, time| {
                    let relative = MediaTime::new(time.value - origin, time.time_base).seconds();
                    (hash ^ relative.to_bits()).wrapping_mul(0x100_0000_01b3)
                });
            assert_eq!(hash, expected_hash, "{name} current Swift exact timeline");
        }
    }

    /// Twelve packets, one stream, timestamps present and keyframes marked —
    /// the contract the VideoToolbox session feeds on.
    #[test]
    fn reads_one_streams_packets_in_demux_order() {
        for name in ["h264-gop4.mp4", "hevc-gop5.mov", "prores-standard.mkv"] {
            let path = fixture(name);
            let mut reader = PacketReader::open(&path, video_stream(&path)).unwrap();
            let mut packets = Vec::new();
            while let Some(packet) = reader.next_packet().unwrap() {
                packets.push(packet);
            }
            assert_eq!(packets.len(), 12, "{name}");
            assert!(
                packets.iter().all(|packet| packet.pts.is_some()),
                "{name}: a packet arrived without a presentation timestamp",
            );
            assert!(
                packets.iter().any(|packet| packet.is_sync),
                "{name}: no keyframe — a decoder would have nowhere to start",
            );
            assert!(
                packets.iter().all(|packet| !packet.data.is_empty()),
                "{name}: an empty packet",
            );
            // Demux order is decode order, so DTS is what must be monotonic.
            let dts: Vec<i64> = packets
                .iter()
                .filter_map(|packet| packet.dts.map(|dts| dts.value))
                .collect();
            assert!(
                dts.windows(2).all(|pair| pair[0] <= pair[1]),
                "{name}: decode timestamps went backwards: {dts:?}",
            );
        }
    }

    /// A seek rewinds the reader, and what comes back starts at a keyframe at
    /// or before the target. Both halves matter: landing after it would make
    /// the decoder skip content, and landing on a non-keyframe would make it
    /// decode garbage.
    #[test]
    fn a_seek_rewinds_to_a_keyframe_at_or_before_the_target() {
        let path = fixture("h264-gop4.mp4");
        let mut reader = PacketReader::open(&path, video_stream(&path)).unwrap();
        let all: Vec<EncodedPacket> = std::iter::from_fn(|| reader.next_packet().transpose())
            .collect::<Result<_>>()
            .unwrap();

        let target = all[7].pts.unwrap();
        let landed = reader.seek(target).unwrap();
        assert!(landed.seconds() <= target.seconds() + 1e-9);
        let first = reader
            .next_packet()
            .unwrap()
            .expect("a packet after seeking");
        assert!(first.is_sync, "a seek landed on a non-keyframe");
        assert!(
            first.pts.unwrap().seconds() <= target.seconds() + 1e-9,
            "a seek landed after its target, so content would be skipped",
        );
    }
}
