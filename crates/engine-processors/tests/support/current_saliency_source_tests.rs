//! Current Swift SyntheticSubjectFixture values, carried through a real encoded
//! all-I file and the actual U2Netp/YuNet provider. No fake heat maps or faces.
use super::*;
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::Mutex,
};
struct Fixture {
    root: PathBuf,
    clip: PathBuf,
    scenes: Vec<VideoScene>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
// (frames, from-x, from-y, to-x, to-y). Current Swift defaults are 480×270,
// 30fps, background120, disc250, radius=0.07*width; radius0 is a uniform field.
fn fixture(spec: &[(usize, f64, f64, f64, f64)], radius: f64) -> Fixture {
    let root = std::env::temp_dir().join(format!(
        "ferment-current-saliency-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let clip = root.join("source.mp4");
    let mut f = Fixture {
        root,
        clip,
        scenes: Vec::new(),
    };
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
            "480x270", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-b:v", "20M", "-g",
            "1", "-pix_fmt", "yuv420p",
        ])
        .arg(&f.clip)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("fixture encoder required");
    let mut input = encoder.stdin.take().unwrap();
    let mut start = 0;
    for (index, &(frames, x0, y0, x1, y1)) in spec.iter().enumerate() {
        f.scenes.push(VideoScene {
            index,
            start_frame: start,
            frame_count: frames,
            start_time: start as f64 / 30.0,
            duration: frames as f64 / 30.0,
            start_confidence: (index > 0).then_some(0.99),
            embedding: None,
        });
        for frame in 0..frames {
            let t = frame as f64 / (frames - 1).max(1) as f64;
            let cx = (x0 + (x1 - x0) * t) * 480.0;
            let cy = (y0 + (y1 - y0) * t) * 270.0;
            let r2 = (radius * 480.0).powi(2);
            let mut pixels = vec![120u8; 480 * 270 * 3];
            if radius > 0.0 {
                for y in 0..270 {
                    for x in 0..480 {
                        if (x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2) <= r2 {
                            pixels[(y * 480 + x) * 3..(y * 480 + x) * 3 + 3].fill(250);
                        }
                    }
                }
            }
            input.write_all(&pixels).unwrap();
        }
        start += frames;
    }
    drop(input);
    let encoded = encoder.wait_with_output().unwrap();
    assert!(
        encoded.status.success(),
        "{}",
        String::from_utf8_lossy(&encoded.stderr)
    );
    f
}
fn analyzer() -> VideoSaliencyAnalyzer {
    let model = backend();
    VideoSaliencyAnalyzer::new(
        Arc::new(U2NetpHeatSource::new(model.clone())),
        Arc::new(YuNetFaceSource::new(model)),
    )
}
fn static_spec(frames: usize, x: f64, y: f64) -> (usize, f64, f64, f64, f64) {
    (frames, x, y, x, y)
}

/// Kill: report only terminal completion or lose the original stage label/order.
#[test]
#[ignore = "actual original-shape encoded fixture, U2Netp and YuNet; run explicitly"]
fn every_original_sample_reports_monotonic_progress_and_saliency_done() {
    let f = fixture(
        &[static_spec(120, 0.3, 0.5), (180, 0.15, 0.5, 0.85, 0.5)],
        0.07,
    );
    let entries = Arc::new(Mutex::new(Vec::new()));
    let capture = entries.clone();
    let sink = ProgressSink::from_callback(move |e| capture.lock().unwrap().push(e));
    let result = futures_executor::block_on(analyzer().analyze(&f.clip, &f.scenes, sink)).unwrap();
    assert_eq!(result.samples.len(), 6);
    let entries = entries.lock().unwrap();
    assert!(entries.windows(2).all(|v| v[0].progress <= v[1].progress));
    assert_eq!(entries.last().unwrap().progress, 1.0);
    assert_eq!(entries.last().unwrap().stage, "saliency done");
    for n in 1..=6 {
        assert_eq!(
            entries
                .iter()
                .filter(|e| e.stage == format!("saliency {n}/6"))
                .count(),
            1
        );
    }
}
/// Kill: treat a featureless field's model artifact as an off-centre subject.
#[test]
#[ignore = "actual original-shape encoded fixture, U2Netp and YuNet; run explicitly"]
fn the_original_uniform_field_reads_as_centre_with_real_models() {
    let f = fixture(&[static_spec(120, 0.5, 0.5)], 0.0);
    let result =
        futures_executor::block_on(analyzer().analyze(&f.clip, &f.scenes, ProgressSink::noop()))
            .unwrap();
    assert_eq!(result.scenes.len(), 1);
    let v = result.scenes[0];
    assert!((v.cx - 0.5).abs() <= 0.1, "cx={}", v.cx);
    assert!((v.cy - 0.5).abs() <= 0.1, "cy={}", v.cy);
}
/// Kill: consume all 24 planned actual samples after the first one cancels.
#[test]
#[ignore = "actual original-shape encoded fixture, U2Netp and YuNet; run explicitly"]
fn cancellation_after_first_actual_sample_stops_before_twenty_four() {
    let spec: Vec<_> = (0..8)
        .map(|i| static_spec(60, 0.2 + 0.08 * i as f64, 0.5))
        .collect();
    let f = fixture(&spec, 0.07);
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    let holder = Arc::new(Mutex::new(None::<ProgressSink>));
    let cancel = holder.clone();
    let sink = ProgressSink::from_callback(move |e| {
        let first = e.stage.starts_with("saliency 1/");
        capture.lock().unwrap().push(e);
        if first {
            cancel.lock().unwrap().as_ref().unwrap().cancel();
        }
    });
    *holder.lock().unwrap() = Some(sink.clone());
    let result = futures_executor::block_on(analyzer().analyze(&f.clip, &f.scenes, sink));
    holder.lock().unwrap().take();
    assert!(matches!(
        result,
        Err(ferment_engine_core::EngineError::Cancelled)
    ));
    let events = events.lock().unwrap();
    let samples = events
        .iter()
        .filter(|e| e.stage.starts_with("saliency ") && e.stage != "saliency done")
        .count();
    assert!(samples > 0 && samples < 24);
    assert!(!events.iter().any(|e| e.stage == "saliency done"));
}
fn quadrant(x: f64, y: f64) -> ferment_engine_processors::video_saliency::VideoSaliencySample {
    let f = fixture(&[static_spec(60, x, y)], 0.07);
    // A 0.1s scene at the source's midpoint asks for the same t=1.0 grab as
    // the Swift single-frame measure case, without bypassing the real reader.
    let scene = VideoScene {
        index: 0,
        start_frame: 28,
        frame_count: 3,
        start_time: 0.95,
        duration: 0.1,
        start_confidence: None,
        embedding: None,
    };
    let result =
        futures_executor::block_on(analyzer().analyze(&f.clip, &[scene], ProgressSink::noop()))
            .unwrap();
    assert_eq!(result.samples.len(), 1);
    result.samples[0]
}
/// Kill: mirror the actual model's top-left coordinates or return an oversized union.
#[test]
#[ignore = "actual original-shape encoded fixture, U2Netp and YuNet; run explicitly"]
fn the_original_top_left_disc_measures_top_left() {
    let s = quadrant(0.25, 0.25);
    assert!(s.cx < 0.35, "cx={}", s.cx);
    assert!(s.cy < 0.35, "cy={}", s.cy);
    assert!(s.w > 0.0 && s.w < 0.6);
    assert!(s.h > 0.0 && s.h < 0.6);
    assert_eq!(s.source, SampleSource::Saliency);
}
/// Kill: invert the real model's bottom-right frame orientation.
#[test]
#[ignore = "actual original-shape encoded fixture, U2Netp and YuNet; run explicitly"]
fn the_original_bottom_right_disc_measures_bottom_right() {
    let s = quadrant(0.75, 0.75);
    assert!(s.cx > 0.65, "cx={}", s.cx);
    assert!(s.cy > 0.60, "cy={}", s.cy);
    assert!(s.w > 0.0 && s.w < 0.6);
    assert!(s.h > 0.0 && s.h < 0.6);
}
