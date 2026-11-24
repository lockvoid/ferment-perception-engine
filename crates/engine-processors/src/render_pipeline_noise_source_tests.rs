//! Current source file-pipeline assertions. Only the same explicitly injected
//! temporal filter as Swift is a spy; decode, encode, container and output are real.
use super::*;
use crate::video_noise_filter::{TemporalNoiseFiltering, VideoNoiseFilterError};
use std::{path::PathBuf, process::Command};
#[derive(Debug)]
struct Call {
    pts: f64,
    previous: Vec<f64>,
    next: usize,
    strength: f32,
    discontinuity: bool,
}
#[derive(Default)]
struct Spy {
    calls: Vec<Call>,
    started: usize,
    ended: usize,
    reject_empty: bool,
}
#[async_trait::async_trait(?Send)]
impl TemporalNoiseFiltering for Spy {
    fn previous_frame_count(&self) -> usize {
        2
    }
    fn next_frame_count(&self) -> usize {
        2
    }
    fn start_session(
        &mut self,
        _w: u32,
        _h: u32,
        _hdr: bool,
    ) -> std::result::Result<(), VideoNoiseFilterError> {
        self.started += 1;
        Ok(())
    }
    async fn filter(
        &mut self,
        frame: &VideoFrame,
        previous: &[VideoFrame],
        next: &[VideoFrame],
        strength: f32,
        discontinuity: bool,
    ) -> std::result::Result<FrameBuffer, VideoNoiseFilterError> {
        if self.reject_empty && previous.is_empty() && next.is_empty() {
            return Err(VideoNoiseFilterError::EmptyWindow);
        }
        self.calls.push(Call {
            pts: frame.pts.seconds(),
            previous: previous.iter().map(|v| v.pts.seconds()).collect(),
            next: next.len(),
            strength,
            discontinuity,
        });
        Ok(frame.buffer.clone())
    }
    fn end_session(&mut self) {
        self.ended += 1;
    }
}
struct Fixture {
    root: PathBuf,
    source: PathBuf,
    output: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
fn run(tool: &str, args: &[&str]) -> Vec<u8> {
    let r = Command::new(tool)
        .args(args)
        .output()
        .expect("real media tools required");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    r.stdout
}
fn fixture(frames: usize, cut: Option<usize>) -> Fixture {
    let root = std::env::temp_dir().join(format!(
        "ferment-current-noise-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let f = Fixture {
        source: root.join("source.mov"),
        output: root.join("output.mp4"),
        root,
    };
    // The assertions concern temporal windows and PTS. This real encoded
    // 480x272 moving-box picture has a hard photometric cut at the source cut.
    let n = frames.to_string();
    let filter = match cut {
        Some(c) => format!("negate=enable='gte(n,{c})'"),
        None => "null".into(),
    };
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=480x272:rate=30",
            "-vf",
            &filter,
            "-frames:v",
            &n,
            "-an",
            "-c:v",
            "libx264",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            f.source.to_str().unwrap(),
        ],
    );
    f
}
fn delivered(f: &Fixture, frames: usize) {
    let report: serde_json::Value = serde_json::from_slice(&run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            f.output.to_str().unwrap(),
        ],
    ))
    .unwrap();
    assert_eq!(
        report["streams"][0]["nb_read_frames"]
            .as_str()
            .unwrap()
            .parse::<usize>()
            .unwrap(),
        frames
    );
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            f.output.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ],
    );
    assert_eq!(
        std::fs::read_dir(&f.root).unwrap().count(),
        2,
        "no owned partial may remain"
    );
}
/// Kill: assemble temporal neighbours only in the in-memory helper, cross a
/// real decoded cut, reorder prior PTS, or lose an encoded output frame.
#[test]
#[ignore = "real platform codecs and container inspection; run explicitly"]
fn the_file_filter_observes_original_scene_windows_and_discontinuities() {
    let f = fixture(24, Some(12));
    let original = std::fs::read(&f.source).unwrap();
    let mut spy = Spy::default();
    let r = futures_executor::block_on(denoise_video_with_filter(
        &f.source,
        &f.output,
        0.7,
        None,
        &[core::MediaTime::new(12, 30)],
        ProgressSink::noop(),
        &mut spy,
    ))
    .unwrap();
    assert_eq!(r.frames_written, 24);
    assert_eq!(spy.calls.len(), 24);
    assert_eq!((spy.started, spy.ended), (1, 1));
    assert!(spy.calls.iter().all(|c| (c.strength - 0.7).abs() < 1e-5));
    let call = |frame: usize| {
        spy.calls
            .iter()
            .find(|c| (c.pts - frame as f64 / 30.0).abs() < 0.001)
            .unwrap()
    };
    assert_eq!(
        spy.calls
            .iter()
            .filter(|c| c.discontinuity)
            .map(|c| c.pts)
            .collect::<Vec<_>>(),
        vec![0.0, 12.0 / 30.0]
    );
    assert_eq!(call(6).previous.len(), 2);
    assert_eq!(call(6).next, 2);
    assert!(call(6).previous.windows(2).all(|p| p[0] <= p[1]));
    assert_eq!(call(11).next, 0);
    assert_eq!(call(12).previous.len(), 0);
    assert_eq!(call(13).previous.len(), 1);
    assert_eq!(call(0).previous.len(), 0);
    assert_eq!(call(23).next, 0);
    delivered(&f, 24);
    assert_eq!(std::fs::read(&f.source).unwrap(), original);
}
/// Kill: pass the final one-frame scene to a rejecting real-pipeline filter.
#[test]
#[ignore = "real platform codecs and container inspection; run explicitly"]
fn the_file_filter_passes_a_lone_scene_through_without_filtering_it() {
    let f = fixture(4, None);
    let mut spy = Spy {
        reject_empty: true,
        ..Spy::default()
    };
    let r = futures_executor::block_on(denoise_video_with_filter(
        &f.source,
        &f.output,
        0.7,
        None,
        &[core::MediaTime::new(3, 30)],
        ProgressSink::noop(),
        &mut spy,
    ))
    .unwrap();
    assert_eq!(r.frames_written, 4);
    assert_eq!(spy.calls.len(), 3);
    assert!(!spy.calls.iter().any(|c| (c.pts - 3.0 / 30.0).abs() < 0.001));
    assert_eq!((spy.started, spy.ended), (1, 1));
    delivered(&f, 4);
}
