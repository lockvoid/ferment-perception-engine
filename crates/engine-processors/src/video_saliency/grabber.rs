//! Sparse grabs: seek to the sync sample before each planned time and decode
//! forward, instead of paying decode for every frame. Port of the Swift
//! `VideoSaliencyFrameGrabber` intent — the scene detector reads every frame
//! because it needs every frame, we do not.
//!
//! Frames come back display-oriented (the container's display matrix applied)
//! and capped at the configured long edge, one at a time: a long video's grabs
//! never pile up in memory.

use std::path::Path;

use ferment_engine_core::{EngineError, Result, RgbFrameRef};
use ferment_engine_media::{
    CpuPixelFormat, DecodeSession, MediaError, MediaTime, VideoBackend, VideoFrame,
};
use image::{RgbImage, imageops};

use super::sample_plan::SampleEntry;

/// Quarter-turns the container asks for, from the ISO-BMFF display matrix.
/// Mirrors `crates/project/src/engine.rs`'s poster path so a video's poster and
/// its saliency measurement agree on which way is up.
pub(crate) fn rotation_degrees(matrix: Option<[i32; 9]>) -> i32 {
    let Some([a, b, _, c, d, _, _, _, _]) = matrix else {
        return 0;
    };
    if a == 0 && d == 0 {
        if b.signum() > 0 && c.signum() < 0 {
            return 90;
        }
        if b.signum() < 0 && c.signum() > 0 {
            return 270;
        }
    }
    if a.signum() < 0 && d.signum() < 0 {
        return 180;
    }
    0
}

/// Display-oriented source size — coded size through the display matrix.
pub fn display_size(width: u32, height: u32, matrix: Option<[i32; 9]>) -> (u32, u32) {
    match rotation_degrees(matrix) {
        90 | 270 => (height, width),
        _ => (width, height),
    }
}

/// Grab cap that puts the long edge at `max_long_edge`. Never asks for an
/// upscale — a source smaller than the cap is measured at its own size.
pub fn maximum_size(display: (u32, u32), max_long_edge: u32) -> (u32, u32) {
    let (width, height) = display;
    let long_edge = width.max(height);
    if long_edge == 0 {
        return display;
    }
    let scale = (f64::from(max_long_edge) / f64::from(long_edge)).min(1.0);
    (
        (f64::from(width) * scale).round() as u32,
        (f64::from(height) * scale).round() as u32,
    )
}

/// One open decode session plus the geometry every grab is normalized to.
pub(crate) struct SparseGrabber {
    decoder: Box<dyn DecodeSession>,
    time_base: ferment_engine_media::TimeBase,
    rotation: i32,
    maximum: (u32, u32),
}

impl SparseGrabber {
    /// Opens `path`, returning the grabber and the display-oriented source
    /// size the result reports.
    pub(crate) async fn open(
        backend: &VideoBackend,
        path: &Path,
        max_long_edge: u32,
    ) -> Result<(Self, (u32, u32))> {
        let info = backend.probe(path).await.map_err(map_media_error)?;
        let stream =
            info.video_streams().next().cloned().ok_or_else(|| {
                EngineError::Decode(format!("no video track in {}", path.display()))
            })?;
        let coded = (stream.width.unwrap_or(0), stream.height.unwrap_or(0));
        let display = display_size(coded.0, coded.1, stream.display_matrix);
        let decoder = backend
            .open_coded_decoder(path, stream.index)
            .await
            .map_err(map_media_error)?;
        Ok((
            Self {
                decoder,
                time_base: stream.time_base,
                rotation: rotation_degrees(stream.display_matrix),
                maximum: maximum_size(display, max_long_edge),
            },
            display,
        ))
    }

    /// Seek to the sync sample before `entry.time - entry.tolerance` and decode
    /// forward to the first frame at or after it — the earliest frame the plan
    /// accepts, which is what a tolerance-carrying seek is for. `None` when the
    /// file ends before that time.
    pub(crate) async fn grab(&mut self, entry: &SampleEntry) -> Result<Option<RgbImage>> {
        let earliest = (entry.time - entry.tolerance).max(0.0);
        self.decoder
            .seek(MediaTime::from_seconds(earliest, self.time_base))
            .await
            .map_err(map_media_error)?;
        while let Some(frame) = self.decoder.decode_next().await.map_err(map_media_error)? {
            if frame.pts.seconds() + f64::EPSILON >= earliest {
                return self.prepare(&frame).map(Some);
            }
        }
        Ok(None)
    }

    /// Decoded frame -> display-oriented, long-edge-capped RGB24.
    fn prepare(&self, frame: &VideoFrame) -> Result<RgbImage> {
        let cpu = frame.to_cpu_rgb().map_err(map_media_error)?;
        debug_assert_eq!(cpu.format, CpuPixelFormat::Rgb24);
        let row_bytes = cpu.width as usize * 3;
        let mut packed = Vec::with_capacity(row_bytes * cpu.height as usize);
        for row in 0..cpu.height as usize {
            let start = row * cpu.stride;
            packed.extend_from_slice(&cpu.data[start..start + row_bytes]);
        }
        let image = RgbImage::from_raw(cpu.width, cpu.height, packed)
            .ok_or_else(|| EngineError::Decode("decoded frame is not packed RGB24".into()))?;
        let oriented = match self.rotation {
            90 => imageops::rotate90(&image),
            180 => imageops::rotate180(&image),
            270 => imageops::rotate270(&image),
            _ => image,
        };
        if (oriented.width(), oriented.height()) == self.maximum {
            return Ok(oriented);
        }
        Ok(imageops::resize(
            &oriented,
            self.maximum.0,
            self.maximum.1,
            imageops::FilterType::Lanczos3,
        ))
    }
}

/// Borrow an `RgbImage` as the frame view the sources take.
pub(crate) fn frame_ref(image: &RgbImage) -> Result<RgbFrameRef<'_>> {
    RgbFrameRef::new(image.as_raw(), image.width(), image.height())
        .ok_or_else(|| EngineError::internal("grabbed frame is not tightly packed RGB24"))
}

fn map_media_error(error: MediaError) -> EngineError {
    match error {
        MediaError::BackendUnavailable(message) => EngineError::Unsupported(message),
        other => EngineError::Decode(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    //! Ported from `VideoSaliencyOrientationTests.swift`'s pure cases: the
    //! generator cap keeps the long edge at `max_long_edge` and never asks for
    //! an upscale, and the reported frame size is the display-oriented one.

    use super::*;

    /// 90 degrees clockwise in ISO-BMFF 16.16 fixed point.
    const ROTATE_90: [i32; 9] = [0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30];
    const ROTATE_180: [i32; 9] = [-(1 << 16), 0, 0, 0, -(1 << 16), 0, 0, 0, 1 << 30];
    const IDENTITY: [i32; 9] = [1 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30];

    #[test]
    fn generator_cap_scales_the_long_edge_and_never_upscales() {
        assert_eq!(maximum_size((1920, 1080), 512), (512, 288));
        assert_eq!(maximum_size((608, 1080), 512), (288, 512));
        assert_eq!(maximum_size((480, 270), 512), (480, 270));
        assert_eq!(maximum_size((0, 0), 512), (0, 0));
    }

    #[test]
    fn display_size_applies_the_track_transform() {
        assert_eq!(display_size(1920, 1080, Some(ROTATE_90)), (1080, 1920));
        assert_eq!(display_size(640, 360, Some(IDENTITY)), (640, 360));
        assert_eq!(display_size(640, 360, Some(ROTATE_180)), (640, 360));
        assert_eq!(display_size(640, 360, None), (640, 360));
    }

    #[test]
    fn rotation_reads_every_quarter_turn() {
        assert_eq!(rotation_degrees(Some(ROTATE_90)), 90);
        assert_eq!(rotation_degrees(Some(ROTATE_180)), 180);
        assert_eq!(
            rotation_degrees(Some([0, -(1 << 16), 0, 1 << 16, 0, 0, 0, 0, 1 << 30])),
            270
        );
        assert_eq!(rotation_degrees(Some(IDENTITY)), 0);
        assert_eq!(rotation_degrees(None), 0);
    }
}
