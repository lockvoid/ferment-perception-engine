//! NUT's index stores max PTS, not the end of the last presentation interval.
//! Observe the tail's actual packet duration rather than adding an FPS epsilon.
use super::*;
use std::collections::BTreeMap;

pub(super) fn durations(
    input: &mut ffmpeg_next::format::context::Input,
) -> Result<BTreeMap<usize, MediaTime>> {
    let mut durations = BTreeMap::new();
    let duration = input.duration();
    if duration == ffi::AV_NOPTS_VALUE || duration < 0 {
        return Ok(durations);
    }
    // SAFETY: this reads a scalar field of the live demuxer.
    let origin = unsafe { (*input.as_ptr()).start_time };
    let origin = if origin == ffi::AV_NOPTS_VALUE {
        0
    } else {
        origin
    };
    let target = duration
        .checked_add(origin)
        .ok_or_else(|| VideoError::InvalidData("NUT tail timestamp overflow".into()))?;
    input
        .seek(target, ..target)
        .map_err(|e| VideoError::Demux(format!("NUT presentation tail: {e}")))?;
    let mut bytes = 0u64;
    for _ in 0..4096 {
        let mut packet = ffmpeg_next::Packet::empty();
        match packet.read(input) {
            Ok(()) => (),
            Err(ffmpeg_next::Error::Eof) => return Ok(durations),
            Err(error) => return Err(VideoError::Demux(format!("NUT presentation tail: {error}"))),
        }
        bytes = bytes.saturating_add(packet.size() as u64);
        if bytes > 128 * 1024 * 1024 {
            return Err(VideoError::InvalidData(
                "NUT presentation tail exceeds 128 MiB".into(),
            ));
        }
        let stream = input
            .stream(packet.stream())
            .ok_or_else(|| VideoError::InvalidData("NUT packet has no stream".into()))?;
        if stream.parameters().medium() != ffmpeg_next::media::Type::Video {
            continue;
        }
        let Some(pts) = packet.pts() else {
            continue;
        };
        if packet.duration() <= 0 {
            return Err(VideoError::InvalidData(
                "NUT video packet has no presentation duration".into(),
            ));
        }
        let start = stream.start_time();
        if start == ffi::AV_NOPTS_VALUE {
            continue;
        }
        let value = pts
            .checked_add(packet.duration())
            .and_then(|end| end.checked_sub(start))
            .filter(|&n| n > 0)
            .ok_or_else(|| VideoError::InvalidData("Invalid NUT presentation extent".into()))?;
        let end = MediaTime::new(value, rational_time_base(stream.time_base())?);
        durations
            .entry(stream.index())
            .and_modify(|old: &mut MediaTime| {
                if old.value < end.value {
                    *old = end;
                }
            })
            .or_insert(end);
    }
    Err(VideoError::InvalidData(
        "NUT presentation tail exceeds 4096 packets".into(),
    ))
}
