//! Current iOS SlowMotionAnalyzerTests through actual encoded assets and the
//! production file decoder. The known pan is synthesized before the codec;
//! no analyzer/decoder/model seam or measured result is substituted.
use super::*;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

const WIDTH: usize = 480;
const HEIGHT: usize = 272;
const FRAMES: usize = 50;
static NEXT: AtomicU64 = AtomicU64::new(0);

struct VideoFixture {
    directory: PathBuf,
    path: PathBuf,
}
impl Drop for VideoFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

// Ported from the source oracle's SyntheticLuma + SyntheticVideoFixture:
// seed 42, two 3x3 smooths, a symmetric horizontal trail, then sliding crops.
fn fixture(name: &str, displacement: usize, trail_fraction: f64) -> VideoFixture {
    let directory = std::env::temp_dir().join(format!(
        "ferment-current-motion-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let fixture = VideoFixture {
        path: directory.join("source.mp4"),
        directory,
    };
    let strip_width = WIDTH + displacement * (FRAMES - 1) + 2;
    let mut state = 42_u64;
    let mut strip = vec![0_u8; strip_width * HEIGHT];
    for value in &mut strip {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *value = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8;
    }
    for _ in 0..2 {
        let mut next = strip.clone();
        for y in 1..HEIGHT - 1 {
            for x in 1..strip_width - 1 {
                let mut sum = 0_u32;
                for sy in y - 1..=y + 1 {
                    for sx in x - 1..=x + 1 {
                        sum += u32::from(strip[sy * strip_width + sx]);
                    }
                }
                next[y * strip_width + x] = (sum / 9) as u8;
            }
        }
        strip = next;
    }
    let trail = displacement as f64 * trail_fraction;
    if trail > 0.0 {
        let taps = (trail.ceil() as usize).max(3) | 1;
        let mut blurred = strip.clone();
        for y in 0..HEIGHT {
            for x in 0..strip_width {
                let mut sum = 0.0;
                for tap in 0..taps {
                    let offset = (tap as f64 / (taps - 1) as f64 - 0.5) * trail;
                    let position = (x as f64 + offset).clamp(0.0, (strip_width - 1) as f64);
                    let first = position as usize;
                    let second = (first + 1).min(strip_width - 1);
                    let mix = position - first as f64;
                    sum += f64::from(strip[y * strip_width + first]) * (1.0 - mix)
                        + f64::from(strip[y * strip_width + second]) * mix;
                }
                blurred[y * strip_width + x] = (sum / taps as f64).round().clamp(0.0, 255.0) as u8;
            }
        }
        strip = blurred;
    }
    let raw = fixture.directory.join("source.gray");
    let mut writer = BufWriter::new(File::create(&raw).unwrap());
    for frame in 0..FRAMES {
        for y in 0..HEIGHT {
            let begin = y * strip_width + frame * displacement;
            writer.write_all(&strip[begin..begin + WIDTH]).unwrap();
        }
    }
    writer.flush().unwrap();
    drop(writer);
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-video_size",
            "480x272",
            "-framerate",
            "30",
            "-i",
        ])
        .arg(&raw)
        .args([
            "-an", "-c:v", "libx264", "-b:v", "20000000", "-g", "12", "-pix_fmt", "yuv420p",
        ])
        .arg(&fixture.path)
        .output()
        .expect("the real-file gate requires ffmpeg");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_file(raw).unwrap();
    fixture
}

fn analyze(
    fixture: &VideoFixture,
    window: Option<core::MediaTimeRange>,
) -> crate::slow_motion_renderer::SlowMotionAnalysis {
    let source = futures_executor::block_on(SourceWindow::open(&fixture.path, window)).unwrap();
    analyze_motion(&source, &fixture.path, &ProgressSink::noop()).unwrap()
}

/// Kill: skip real decode/downscale or read nonconsecutive frames; the known
/// 12px-per-frame pan no longer has the oracle's angle and displacement.
#[test]
#[ignore = "requires the real platform decoder and ffmpeg"]
fn crisp_pan_reads_fast_shutter_and_displacement_through_a_real_codec() {
    let fixture = fixture("crisp", 12, 0.0);
    let measured = analyze(&fixture, None);
    assert!(
        measured.source_shutter_angle.unwrap() < 45.0,
        "{measured:?}"
    );
    assert!(
        (measured.median_displacement - 0.044).abs() <= 0.02,
        "{measured:?}"
    );
    assert!(measured.pairs_sampled >= 6);
}

/// Kill: discard the source trail in file analysis and the 180-degree source
/// is incorrectly offered synthetic motion blur.
#[test]
#[ignore = "requires the real platform decoder and ffmpeg"]
fn normal_shutter_pan_reads_wide_through_a_real_codec() {
    let fixture = fixture("normal", 12, 0.5);
    let measured = analyze(&fixture, None);
    assert!(
        measured.source_shutter_angle.unwrap() > 100.0,
        "{measured:?}"
    );
}

/// Kill: infer shutter without actual motion; the static encoded clip earns
/// a fabricated angle or displacement.
#[test]
#[ignore = "requires the real platform decoder and ffmpeg"]
fn a_static_encoded_clip_is_inconclusive() {
    let fixture = fixture("static", 0, 0.0);
    let measured = analyze(&fixture, None);
    assert!(measured.source_shutter_angle.is_none(), "{measured:?}");
    assert!(measured.median_displacement < 0.005, "{measured:?}");
}

/// Kill: fail to decode the requested nonzero interval; the narrow window
/// cannot produce four actual measured pairs and an angle.
#[test]
#[ignore = "requires the real platform decoder and ffmpeg"]
fn a_nonzero_encoded_window_is_analyzed() {
    let fixture = fixture("window", 12, 0.0);
    let measured = analyze(
        &fixture,
        Some(core::MediaTimeRange {
            start: core::MediaTime::new(4, 10),
            duration: core::MediaTime::new(8, 10),
        }),
    );
    assert!(measured.source_shutter_angle.is_some(), "{measured:?}");
    assert!(measured.pairs_sampled >= 4);
}

/// Kill: turn the bounded real-file pre-pass into a whole-video decoder pass
/// or repeat expensive analysis; the source oracle's three-second budget fails.
#[test]
#[ignore = "requires the real platform decoder and ffmpeg; run release mode"]
fn encoded_prepass_stays_inside_the_three_second_budget() {
    let fixture = fixture("budget", 12, 0.0);
    let started = Instant::now();
    let _ = analyze(&fixture, None);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
}
