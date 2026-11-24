//! Exact current Swift file-render contracts. Recorder seams replace only the
//! same interpolation/blur objects as Swift; real files cross the production
//! reader, motion analysis, scheduler, encoder and atomic mux delivery.
use super::*;
use crate::slow_motion_renderer::{FrameBlurrer, FrameInterpolator, TimedFrame};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

const WIDTH: usize = 480;
const HEIGHT: usize = 272;
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    directory: PathBuf,
    source: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    fn output(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.mp4"))
    }
    fn no_partials(&self) {
        for entry in std::fs::read_dir(&self.directory).unwrap() {
            assert!(
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains("partial")
            );
        }
    }
}
fn run(tool: &str, args: &[&str]) -> Vec<u8> {
    let output = Command::new(tool)
        .args(args)
        .output()
        .expect("real codec oracle required");
    assert!(
        output.status.success(),
        "{tool}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

// SyntheticVideoFixture.Spec, including both original scene seeds and the
// exact 12-pixel pan, two smoothing passes, trail and source frame counts.
fn fixture(frames: usize, trail_fraction: f64, cut_at: Option<usize>) -> Fixture {
    fixture_at_size(frames, trail_fraction, cut_at, WIDTH, HEIGHT)
}

fn fixture_at_size(
    frames: usize,
    trail_fraction: f64,
    cut_at: Option<usize>,
    width: usize,
    height: usize,
) -> Fixture {
    let directory = std::env::temp_dir().join(format!(
        "ferment-current-slow-file-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let f = Fixture {
        source: directory.join("source.mov"),
        directory,
    };
    let strip_width = width + 12 * (frames - 1) + 2;
    let make_strip = |mut state: u64| {
        let mut strip = vec![0_u8; strip_width * height];
        for value in &mut strip {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            *value = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8;
        }
        for _ in 0..2 {
            let mut next = strip.clone();
            for y in 1..height - 1 {
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
        let trail = 12.0 * trail_fraction;
        if trail > 0.0 {
            let taps = (trail.ceil() as usize).max(3) | 1;
            let mut blurred = strip.clone();
            for y in 0..height {
                for x in 0..strip_width {
                    let mut sum = 0.0;
                    for tap in 0..taps {
                        let offset = (tap as f64 / (taps - 1) as f64 - 0.5) * trail;
                        let p = (x as f64 + offset).clamp(0.0, (strip_width - 1) as f64);
                        let first = p as usize;
                        let second = (first + 1).min(strip_width - 1);
                        let mix = p - first as f64;
                        sum += f64::from(strip[y * strip_width + first]) * (1.0 - mix)
                            + f64::from(strip[y * strip_width + second]) * mix;
                    }
                    blurred[y * strip_width + x] =
                        (sum / taps as f64).round().clamp(0.0, 255.0) as u8;
                }
            }
            strip = blurred;
        }
        strip
    };
    let first = make_strip(42);
    let second = cut_at.map(|_| make_strip(777));
    let raw = f.directory.join("source.gray");
    let mut writer = BufWriter::new(File::create(&raw).unwrap());
    for frame in 0..frames {
        let strip = if cut_at.is_some_and(|cut| frame >= cut) {
            second.as_ref().unwrap()
        } else {
            &first
        };
        for y in 0..height {
            let begin = y * strip_width + frame * 12;
            writer.write_all(&strip[begin..begin + width]).unwrap();
        }
    }
    writer.flush().unwrap();
    drop(writer);
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            "30",
            "-i",
            raw.to_str().unwrap(),
            "-an",
            "-c:v",
            "libx264",
            "-b:v",
            "20000000",
            "-g",
            "12",
            "-pix_fmt",
            "yuv420p",
            f.source.to_str().unwrap(),
        ],
    );
    std::fs::remove_file(raw).unwrap();
    f
}

#[derive(Debug)]
struct InterpolationCall {
    from: f64,
    phases: Vec<f32>,
    restart: bool,
}
#[derive(Default)]
struct SpyInterpolator {
    calls: Vec<InterpolationCall>,
}
impl FrameInterpolator<VideoFrame> for SpyInterpolator {
    fn interpolate(
        &mut self,
        from: &TimedFrame<VideoFrame>,
        _to: &TimedFrame<VideoFrame>,
        phases: &[f32],
        restart: bool,
    ) -> std::result::Result<Vec<VideoFrame>, String> {
        self.calls.push(InterpolationCall {
            from: from.pts.seconds(),
            phases: phases.to_vec(),
            restart,
        });
        Ok(vec![from.frame.clone(); phases.len()])
    }
}
impl SpyInterpolator {
    fn at(&self, frame: usize) -> Option<&InterpolationCall> {
        self.calls
            .iter()
            .find(|call| (call.from - frame as f64 / 30.0).abs() < 0.001)
    }
    fn restarts(&self) -> Vec<f64> {
        self.calls
            .iter()
            .filter(|c| c.restart)
            .map(|c| c.from)
            .collect()
    }
}
#[derive(Debug)]
struct BlurCall {
    pts: f64,
    previous: bool,
    next: bool,
    strength: i32,
}
#[derive(Default)]
struct SpyBlurrer {
    calls: Vec<BlurCall>,
    mark: bool,
}
impl FrameBlurrer<VideoFrame> for SpyBlurrer {
    fn blur(
        &mut self,
        frame: &TimedFrame<VideoFrame>,
        previous: Option<&TimedFrame<VideoFrame>>,
        next: Option<&TimedFrame<VideoFrame>>,
        strength: i32,
    ) -> std::result::Result<VideoFrame, String> {
        self.calls.push(BlurCall {
            pts: frame.pts.seconds(),
            previous: previous.is_some(),
            next: next.is_some(),
            strength,
        });
        let mut result = frame.frame.clone();
        if self.mark {
            // Neutral RGB214 encodes to the Swift marker's video-range Y200.
            result.buffer = FrameBuffer::Cpu(
                CpuFrame::new(
                    result.width,
                    result.height,
                    result.width as usize * 3,
                    CpuPixelFormat::Rgb24,
                    vec![214; result.width as usize * result.height as usize * 3],
                )
                .map_err(|e| e.to_string())?,
            );
        }
        Ok(result)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Fixture helper mirrors the public render request and its two observed processors"
)]
fn render(
    f: &Fixture,
    output: &Path,
    factor: u8,
    window: Option<core::MediaTimeRange>,
    cuts: &[usize],
    mode: core::MotionBlurMode,
    spy: &mut SpyInterpolator,
    blur: &mut SpyBlurrer,
) -> core::VideoRenderResult {
    let source = std::fs::read(&f.source).unwrap();
    let cuts: Vec<_> = cuts
        .iter()
        .map(|c| core::MediaTime::new(*c as i64, 30))
        .collect();
    let result = slow_motion_video_with_processors(
        &f.source,
        output,
        factor,
        window,
        &cuts,
        mode,
        180.0,
        ProgressSink::noop(),
        spy,
        blur,
    )
    .unwrap();
    assert_eq!(std::fs::read(&f.source).unwrap(), source);
    assert!(output.exists());
    f.no_partials();
    result
}
fn luma_frames(path: &Path) -> Vec<Vec<u8>> {
    let raw = run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ],
    );
    let frame_bytes = WIDTH * HEIGHT * 3 / 2;
    assert_eq!(raw.len() % frame_bytes, 0);
    raw.chunks_exact(frame_bytes)
        .map(|frame| frame[..WIDTH * HEIGHT].to_vec())
        .collect()
}
fn mean_difference(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut total = 0_u64;
    let mut count = 0;
    for y in (0..HEIGHT).step_by(4) {
        for x in (0..WIDTH).step_by(4) {
            total += u64::from(a[y * WIDTH + x].abs_diff(b[y * WIDTH + x]));
            count += 1;
        }
    }
    total as f64 / count as f64
}
fn off() -> core::MotionBlurMode {
    core::MotionBlurMode::Disabled
}
fn on(strength: i32) -> core::MotionBlurMode {
    core::MotionBlurMode::Enabled { strength }
}

/// Kill: ignore the blur result, skip an anchor, or cross a clip endpoint with a neighbor.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn explicit_on_blurs_every_actual_source_anchor_and_writes_the_marker() {
    let f = fixture(24, 0.0, None);
    let output = f.output("on");
    let mut spy = SpyInterpolator::default();
    let mut blur = SpyBlurrer {
        mark: true,
        ..Default::default()
    };
    let result = render(&f, &output, 2, None, &[], on(42), &mut spy, &mut blur);
    let decision = result.motion_blur.unwrap();
    assert!(decision.enabled);
    assert_eq!(decision.reason, "explicitOn");
    assert_eq!(decision.strength, 42);
    assert_eq!(blur.calls.len(), 24);
    assert!(blur.calls.windows(2).all(|pair| pair[0].pts <= pair[1].pts));
    assert!(blur.calls.iter().all(|call| call.strength == 42));
    assert!(!blur.calls[0].previous && blur.calls[0].next);
    assert!(blur.calls[23].previous && !blur.calls[23].next);
    assert!(
        blur.calls[1..23]
            .iter()
            .all(|call| call.previous && call.next)
    );
    assert_eq!(result.frames_written, 47);
    let decoded = luma_frames(&output);
    assert_eq!(decoded.len(), 47);
    let middle = &decoded[23];
    let mean = middle.iter().map(|v| f64::from(*v)).sum::<f64>() / middle.len() as f64;
    assert!(
        (mean - 200.0).abs() <= 6.0,
        "actual encoded marker luma {mean}"
    );
}

/// Kill: call the blur seam when the user disables blur.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn off_never_touches_the_file_blur_seam() {
    let f = fixture(24, 0.0, None);
    let output = f.output("off");
    let mut spy = SpyInterpolator::default();
    let mut blur = SpyBlurrer::default();
    let result = render(&f, &output, 2, None, &[], off(), &mut spy, &mut blur);
    assert!(blur.calls.is_empty());
    let decision = result.motion_blur.unwrap();
    assert!(!decision.enabled);
    assert_eq!(decision.reason, "explicitOff");
    assert_eq!(result.frames_written, 47);
    assert_eq!(luma_frames(&output).len(), 47);
}

/// Kill: drop the one-frame drain, invent a neighbor, or interpolate outside the selected window.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn a_twenty_millisecond_window_blurs_one_solo_frame() {
    let f = fixture(24, 0.0, None);
    let output = f.output("solo");
    let mut spy = SpyInterpolator::default();
    let mut blur = SpyBlurrer::default();
    let result = render(
        &f,
        &output,
        2,
        Some(core::MediaTimeRange {
            start: core::MediaTime::new(0, 600),
            duration: core::MediaTime::new(12, 600),
        }),
        &[],
        on(30),
        &mut spy,
        &mut blur,
    );
    assert_eq!(blur.calls.len(), 1);
    assert!(!blur.calls[0].previous && !blur.calls[0].next);
    assert_eq!(result.frames_written, 1);
    assert!(spy.calls.is_empty());
    assert_eq!(luma_frames(&output).len(), 1);
}

/// Kill: bypass actual file motion analysis or fail to deliver its blur strength to every source anchor.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn auto_applies_on_the_actual_forty_frame_crisp_pan() {
    let f = fixture(40, 0.0, None);
    let mut spy = SpyInterpolator::default();
    let mut blur = SpyBlurrer::default();
    let result = render(
        &f,
        &f.output("auto-crisp"),
        2,
        None,
        &[],
        core::MotionBlurMode::Automatic,
        &mut spy,
        &mut blur,
    );
    let decision = result.motion_blur.unwrap();
    assert!(decision.enabled);
    assert_eq!(decision.reason, "fastShutterSource");
    assert!(decision.estimated_shutter_angle.unwrap() < 45.0);
    assert!(decision.strength >= 1);
    assert!(!blur.calls.is_empty());
    assert!(
        blur.calls
            .iter()
            .all(|call| call.strength == decision.strength)
    );
}

/// Kill: treat the real 180-degree source as a fast-shutter source and blur it again.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn auto_keeps_the_actual_normal_shutter_pan_crisp() {
    let f = fixture(40, 0.5, None);
    let mut spy = SpyInterpolator::default();
    let mut blur = SpyBlurrer::default();
    let result = render(
        &f,
        &f.output("auto-normal"),
        2,
        None,
        &[],
        core::MotionBlurMode::Automatic,
        &mut spy,
        &mut blur,
    );
    let decision = result.motion_blur.unwrap();
    assert!(!decision.enabled);
    assert!(blur.calls.is_empty());
    assert_ne!(decision.reason, "unsupported");
}

/// Kill: interpolate across source frame12, omit either neighbor, or fail to reset after the cut.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn a_real_cut_pair_is_not_interpolated_and_announces_two_restarts() {
    let f = fixture(24, 0.0, Some(12));
    let mut spy = SpyInterpolator::default();
    let result = render(
        &f,
        &f.output("cut"),
        2,
        None,
        &[12],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 47);
    assert_eq!(spy.calls.len(), 22);
    assert!(spy.at(11).is_none());
    assert!(spy.at(10).is_some() && spy.at(12).is_some());
    let restarts = spy.restarts();
    assert_eq!(restarts.len(), 2);
    assert!(restarts[0].abs() < 0.001);
    assert!((restarts[1] - 0.4).abs() < 0.001);
}

/// Kill: freeze the outgoing pair instead of stretching the first incoming pair.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn the_incoming_real_pair_stretches_across_the_cut_slots() {
    let f = fixture(24, 0.0, Some(12));
    let mut spy = SpyInterpolator::default();
    let result = render(
        &f,
        &f.output("stretch"),
        2,
        None,
        &[12],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 47);
    let phases = &spy.at(12).unwrap().phases;
    assert_eq!(phases.len(), 2);
    assert!((phases[0] - 1.0 / 3.0).abs() < 1e-4 && (phases[1] - 2.0 / 3.0).abs() < 1e-4);
    assert_eq!(spy.at(6).unwrap().phases, [0.5]);
}

/// Kill: place outgoing-scene pixels in any of the three immediate-cut output slots.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn the_delivered_cut_is_immediate_in_actual_decoded_pixels() {
    let f = fixture(24, 0.0, Some(12));
    let output = f.output("pixels");
    render(
        &f,
        &output,
        2,
        None,
        &[12],
        off(),
        &mut SpyInterpolator::default(),
        &mut SpyBlurrer::default(),
    );
    let decoded = luma_frames(&output);
    assert_eq!(decoded.len(), 47);
    for slot in 23..=25 {
        assert!(
            mean_difference(&decoded[slot], &decoded[22]) > 5.0,
            "outgoing pixels froze slot {slot}"
        );
    }
}

/// Kill: interpolate or drop the last pair when the file ends on the first incoming frame.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn a_cut_on_the_final_real_pair_keeps_the_incoming_frame() {
    let f = fixture(24, 0.0, Some(23));
    let mut spy = SpyInterpolator::default();
    let result = render(
        &f,
        &f.output("last"),
        2,
        None,
        &[23],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 47);
    assert!(spy.at(22).is_none());
}

/// Kill: send either consecutive boundary through interpolation or forget the following restart/stretch.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn a_single_frame_scene_copies_then_stretches_in_the_real_file_path() {
    let f = fixture(24, 0.0, Some(12));
    let mut spy = SpyInterpolator::default();
    let result = render(
        &f,
        &f.output("single-scene"),
        2,
        None,
        &[12, 13],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 47);
    assert!(spy.at(11).is_none() && spy.at(12).is_none());
    assert_eq!(spy.at(13).unwrap().phases.len(), 2);
    assert!(spy.at(13).unwrap().restart);
}

/// Kill: compare source-asset cut time with window-relative decoded frame time.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn a_real_nonzero_window_rebases_its_scene_boundary() {
    let f = fixture(24, 0.0, Some(12));
    let mut spy = SpyInterpolator::default();
    let window = core::MediaTimeRange {
        start: core::MediaTime::new(6, 30),
        duration: core::MediaTime::new(18, 30),
    };
    let result = render(
        &f,
        &f.output("window"),
        2,
        Some(window),
        &[12],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 35);
    assert_eq!(spy.calls.len(), 16);
    assert!(spy.at(5).is_none());
}

/// Kill: dump three orphaned ticks into one incoming pair instead of spreading them across three pairs.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn factor_four_spreads_the_real_cut_deficit_over_three_pairs() {
    let f = fixture(24, 0.0, Some(12));
    let mut spy = SpyInterpolator::default();
    let result = render(
        &f,
        &f.output("factor-four"),
        4,
        None,
        &[12],
        off(),
        &mut spy,
        &mut SpyBlurrer::default(),
    );
    assert_eq!(result.frames_written, 93);
    for frame in [12, 13, 14] {
        let phases = &spy.at(frame).unwrap().phases;
        assert_eq!(phases.len(), 4);
        assert!((phases[0] - 0.2).abs() < 1e-4);
    }
    assert_eq!(spy.at(15).unwrap().phases.len(), 3);
    assert_eq!(spy.restarts().len(), 2);
}

/// Kill: feed blur pixels from the other side of a real source scene boundary.
#[test]
#[ignore = "actual platform codecs and ffmpeg; run serially"]
fn blur_neighbors_treat_the_real_cut_as_a_clip_edge() {
    let f = fixture(24, 0.0, Some(12));
    let mut blur = SpyBlurrer::default();
    render(
        &f,
        &f.output("blur-cut"),
        2,
        None,
        &[12],
        on(30),
        &mut SpyInterpolator::default(),
        &mut blur,
    );
    assert_eq!(blur.calls.len(), 24);
    let at = |frame| {
        blur.calls
            .iter()
            .find(|c| (c.pts - frame as f64 / 30.0).abs() < 0.001)
            .unwrap()
    };
    assert!(at(11).previous && !at(11).next);
    assert!(!at(12).previous && at(12).next);
    assert!(at(6).previous && at(6).next);
}

#[cfg(feature = "ort")]
fn rife_backend() -> Backend {
    use ferment_engine_infer::{
        ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };
    let master = ferment_engine_test_fixtures::export::artifact("rife_v4_25.onnx")
        .expect("current file-quality gate requires the pinned real RIFE master");
    let paths = ModelPaths::new().with_path(ModelId::RifeV425, ModelVersion::Latest, master);
    std::sync::Arc::new(OrtBackend::with_execution_provider(
        std::sync::Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}
#[cfg(feature = "ort")]
fn real_render(
    f: &Fixture,
    name: &str,
    factor: u8,
    cuts: &[usize],
    mode: core::MotionBlurMode,
) -> core::VideoRenderResult {
    let cuts: Vec<_> = cuts
        .iter()
        .map(|c| core::MediaTime::new(*c as i64, 30))
        .collect();
    let result = slow_motion_video(
        rife_backend(),
        &f.source,
        &f.output(name),
        factor,
        None,
        &cuts,
        mode,
        180.0,
        ProgressSink::noop(),
    )
    .unwrap();
    f.no_partials();
    result
}

/// Kill: the real exposure integration leaves the delivered pan sharp or changes its frame count.
#[cfg(feature = "ort")]
#[test]
#[ignore = "actual RIFE master/platform codecs/ffmpeg; run serially"]
fn actual_rife_blur_bakes_a_trail_below_the_original_energy_bound() {
    let f = fixture(30, 0.0, None);
    let blurred = real_render(&f, "blurred", 2, &[], on(60));
    let sharp = real_render(&f, "sharp", 2, &[], off());
    assert!(blurred.motion_blur.as_ref().unwrap().enabled);
    assert_eq!(blurred.motion_blur.as_ref().unwrap().reason, "explicitOn");
    assert!(!sharp.motion_blur.unwrap().enabled);
    assert_eq!(blurred.frames_written, sharp.frames_written);
    let energy = |pixels: &[u8]| {
        let mut total = 0.0;
        for y in (2..HEIGHT - 2).step_by(2) {
            for x in 2..WIDTH - 2 {
                let g = f64::from(pixels[y * WIDTH + x + 1]) - f64::from(pixels[y * WIDTH + x - 1]);
                total += g * g;
            }
        }
        total
    };
    let a = luma_frames(&f.output("blurred"));
    let b = luma_frames(&f.output("sharp"));
    assert_eq!(a.len(), b.len());
    let blurred_energy = energy(&a[24]);
    let sharp_energy = energy(&b[24]);
    assert!(
        blurred_energy < sharp_energy * 0.8,
        "strength60 x-gradient energy {blurred_energy} vs {sharp_energy}"
    );
}

/// Kill: stop the automatic policy before the real RIFE blur/encode/delivery path.
#[cfg(feature = "ort")]
#[test]
#[ignore = "actual RIFE master/platform codecs/ffmpeg; run serially"]
fn automatic_analysis_drives_actual_rife_blur_and_file_delivery() {
    let f = fixture(40, 0.0, None);
    let result = real_render(&f, "automatic", 2, &[], core::MotionBlurMode::Automatic);
    let decision = result.motion_blur.unwrap();
    assert!(decision.enabled);
    assert_eq!(decision.reason, "fastShutterSource");
    assert!(decision.estimated_shutter_angle.unwrap() < 45.0);
    assert!(f.output("automatic").exists());
}

/// Kill: retain outgoing-scene flow/pixels in the first real interpolated pair after a cut.
#[cfg(feature = "ort")]
#[test]
#[ignore = "actual RIFE master/platform codecs/ffmpeg; run serially"]
fn the_first_actual_rife_pair_after_a_cut_belongs_to_its_own_scene() {
    let f = fixture(24, 0.0, Some(12));
    real_render(&f, "rife-cut", 4, &[12], off());
    let source = luma_frames(&f.source);
    let output = luma_frames(&f.output("rife-cut"));
    for slot in 49..=51 {
        let own = mean_difference(&output[slot], &output[48]);
        let outgoing = mean_difference(&output[slot], &source[11]);
        assert!(
            own < outgoing,
            "slot {slot}: own {own}, outgoing {outgoing}"
        );
    }
}

#[path = "render_pipeline_remaining_file_tests.rs"]
mod remaining_file_contract;
