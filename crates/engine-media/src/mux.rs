use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{
    AudioTiming, ColorInfo, DecoderConfiguration, EncodedPacket, MediaTime, Result, TimeBase,
    VideoError,
};

/// ISO-BMFF flavor written by the pure-Rust muxer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MuxFormat {
    #[default]
    Mp4,
    Mov,
}

/// Container-level policy. Classic (non-fragmented) fast-start output is the
/// export default; fragmented output is deliberately not part of v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuxSettings {
    pub format: MuxFormat,
    pub fast_start: bool,
    /// `None` chooses the least common multiple of track timescales when it is
    /// representable, preserving exact frame/sample durations.
    pub movie_timescale: Option<u32>,
}

impl Default for MuxSettings {
    fn default() -> Self {
        Self {
            format: MuxFormat::Mp4,
            fast_start: true,
            movie_timescale: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MuxTrackId(u32);

impl MuxTrackId {
    /// Mint an id. Only a `MuxSession` implementation calls this — the id is
    /// meaningless outside the session that issued it.
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoMuxTrack {
    /// Exact container preferred transform (ISO-BMFF fixed-point words).
    /// Encoded pictures remain in coded orientation; None writes identity.
    /// Preserve translation even when the coded dimensions were scaled,
    /// matching AVAssetWriterInput.transform rather than normalizing it.
    pub display_matrix: Option<[i32; 9]>,
    pub configuration: DecoderConfiguration,
    pub width: u32,
    pub height: u32,
    pub time_base: TimeBase,
    pub color: ColorInfo,
    /// Exact visible duration selected by the video edit list. When absent,
    /// the muxer derives it from packet presentation timestamps. Supplying it
    /// lets an encoder/remux caller trim a terminal packet whose coded sample
    /// duration extends beyond the frozen export interval.
    pub duration: Option<MediaTime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioMuxTrack {
    pub configuration: DecoderConfiguration,
    pub sample_rate: u32,
    pub channels: u16,
    pub timing: AudioTiming,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuxOutput {
    pub path: Option<PathBuf>,
    /// Present for an in-memory session and absent for a path-backed session.
    pub bytes: Option<Vec<u8>>,
    pub duration: MediaTime,
    pub track_count: usize,
}

/// Packet-oriented container session. Codec sessions remain responsible only
/// for compression; this boundary owns interleave, sample tables, edits, and
/// final file layout.
pub trait MuxSession: Debug + Send {
    fn add_video_track(&mut self, track: VideoMuxTrack) -> Result<MuxTrackId>;
    fn add_audio_track(&mut self, track: AudioMuxTrack) -> Result<MuxTrackId>;
    fn write_packet(&mut self, track: MuxTrackId, packet: EncodedPacket) -> Result<()>;
    /// Preserve the source codec parameters rather than inventing an AAC
    /// description. A backend without this lane must refuse, never transcode.
    fn add_passthrough_audio_track(
        &mut self,
        _track: &crate::AudioPassthroughTrack,
    ) -> Result<MuxTrackId> {
        Err(VideoError::BackendUnavailable(
            "encoded audio passthrough is unavailable".into(),
        ))
    }
    /// Packet payload and side data are owned together until the container
    /// accepts them. The decoded/encoded lane above remains unchanged.
    fn write_passthrough_audio_packet(
        &mut self,
        _track: MuxTrackId,
        _packet: crate::AudioPassthroughPacket,
    ) -> Result<()> {
        Err(VideoError::BackendUnavailable(
            "encoded audio passthrough is unavailable".into(),
        ))
    }
    fn finalize(self: Box<Self>) -> Result<MuxOutput>;
}

pub trait Muxer: Debug + Send + Sync {
    fn open(&self, output: &Path, settings: MuxSettings) -> Result<Box<dyn MuxSession>>;
}

pub type MuxBackend = Arc<dyn Muxer>;

// The one piece of the hand-written writer that outlived it: the libav
// muxer converts Annex-B payloads with it before handing them to
// libavformat's `mov` muxer, which takes length-prefixed NAL units.

pub(crate) fn annex_b_to_length_prefixed(data: &[u8]) -> Result<Vec<u8>> {
    let units = annex_b_nal_units(data);
    if units.is_empty() {
        return Err(VideoError::InvalidData(
            "Annex-B packet contains no NAL units".into(),
        ));
    }
    let mut output = Vec::with_capacity(data.len().saturating_add(units.len() * 4));
    for unit in units {
        let length = u32::try_from(unit.len())
            .map_err(|_| VideoError::InvalidData("NAL unit exceeds MP4 u32".into()))?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(&unit);
    }
    Ok(output)
}

fn annex_b_nal_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut cursor = 0usize;
    while cursor + 3 <= data.len() {
        let width = if data.get(cursor..cursor + 4) == Some(&[0, 0, 0, 1]) {
            4
        } else if data.get(cursor..cursor + 3) == Some(&[0, 0, 1]) {
            3
        } else {
            cursor += 1;
            continue;
        };
        starts.push((cursor, width));
        cursor += width;
    }
    starts
        .iter()
        .enumerate()
        .filter_map(|(index, (start, width))| {
            let payload_start = start + width;
            let payload_end = starts.get(index + 1).map_or(data.len(), |(next, _)| *next);
            (payload_start < payload_end).then(|| data[payload_start..payload_end].to_vec())
        })
        .collect()
}

// AMediaMuxer has no exact nine-word transform API. Kept pure so the limited
// backend's refusal policy is executable on host as well as Android.
#[cfg(any(target_os = "android", test))]
pub(crate) fn require_identity_display_matrix(matrix: Option<[i32; 9]>) -> Result<()> {
    const IDENTITY: [i32; 9] = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];
    if matrix.is_some_and(|matrix| matrix != IDENTITY) {
        return Err(VideoError::UnsupportedContainer(
            "AMediaMuxer cannot preserve a nonidentity preferred transform; use LibavMuxer".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod display_matrix_tests {
    use super::*;
    // Kill: accepting a rotated/mirrored/translated/scaled track while the
    // limited platform muxer silently writes identity instead.
    #[test]
    fn limited_muxer_refuses_every_nonidentity_matrix_before_track_admission() {
        const IDENTITY: [i32; 9] = [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824];
        require_identity_display_matrix(None).unwrap();
        require_identity_display_matrix(Some(IDENTITY)).unwrap();
        for matrix in [
            [0, 65536, 0, -65536, 0, 0, 4194304, 0, 1073741824],
            [-65536, 0, 0, 0, 65536, 0, 8388608, 0, 1073741824],
            [65536, 0, 0, 0, 65536, 0, 1, 0, 1073741824],
            [65537, 0, 0, 0, 65536, 0, 0, 0, 1073741824],
        ] {
            assert!(matches!(require_identity_display_matrix(Some(matrix)),
                Err(VideoError::UnsupportedContainer(message)) if message.contains("preferred transform")));
        }
    }
}
