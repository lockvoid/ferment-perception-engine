//! Current iOS long-file assertions through real published CPU ONNX models.
//! Run serially with pinned model/fixture/runtime inputs. RSS is process
//! resident memory, not Apple's physical_footprint and not Android evidence.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use ferment_engine_core::{AudioEventDetecting, AudioSampleSource, ProgressSink};
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::{AudioEventDetector, DeepFilterNet};
use ferment_engine_test_fixtures::export;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

fn backend(model: ModelId, name: &str) -> Arc<OrtBackend> {
    let path =
        export::artifact(name).expect("pinned published model is required; missing is not a skip");
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(model, ModelVersion::Latest, path)),
        OrtExecutionProvider::Cpu,
    ))
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("audio-current-long-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn tone(&self, name: &str, seconds: usize, rate: u32) -> PathBuf {
        let path = self.0.join(name);
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for i in 0..seconds * rate as usize {
            let value = 0.05 * (std::f64::consts::TAU * 440.0 * i as f64 / rate as f64).sin();
            writer.write_sample(value as f32).unwrap();
        }
        writer.finalize().unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").unwrap();
        let kib = text
            .lines()
            .find(|line| line.starts_with("VmRSS:"))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u64>()
            .unwrap();
        kib * 1024
    }
    #[cfg(not(target_os = "linux"))]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .unwrap();
        assert!(output.status.success());
        std::str::from_utf8(&output.stdout)
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap()
            * 1024
    }
}

struct ResidentProbe {
    baseline: u64,
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl ResidentProbe {
    fn start() -> Self {
        let baseline = rss_bytes();
        let peak = Arc::new(AtomicU64::new(baseline));
        let stop = Arc::new(AtomicBool::new(false));
        let p = peak.clone();
        let s = stop.clone();
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(rss_bytes(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            baseline,
            peak,
            stop,
            thread: Some(thread),
        }
    }
    fn finish(mut self, label: &str, maximum_mb: f64) {
        let after = rss_bytes();
        self.peak.fetch_max(after, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        let delta = (after as i128 - self.baseline as i128) as f64 / 1_048_576.0;
        eprintln!(
            "AUDIO_MEMORY {label} baseline_rss={} retained_rss={after} peak_sampled_rss={} delta_mb={delta} interval_ms=50 criterion_mb={maximum_mb}",
            self.baseline,
            self.peak.load(Ordering::Relaxed)
        );
        // Exact existing iOS criterion: retained process growth after the
        // warmed real-model file operation, with its result still alive.
        // Peak is also recorded; it is sampled and is not a peak guarantee.
        assert!(delta < maximum_mb, "{label}: {delta} MB >= {maximum_mb} MB");
    }
}
impl Drop for ResidentProbe {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
    }
}

fn denoise_memory(seconds: usize, maximum_mb: f64) {
    let scratch = Scratch::new();
    let input = scratch.tone("original.wav", seconds, 48_000);
    let before_input = std::fs::metadata(&input).unwrap().len();
    let warm = scratch.tone("warm.wav", 2, 48_000);
    let model = DeepFilterNet::new(backend(ModelId::Dfnet, "dfnet.onnx"));
    drop(
        futures_executor::block_on(model.denoise_to_file(
            &warm,
            &scratch.0,
            None,
            ProgressSink::noop(),
        ))
        .unwrap(),
    );
    let probe = ResidentProbe::start();
    let output = futures_executor::block_on(model.denoise_to_file(
        &input,
        &scratch.0,
        None,
        ProgressSink::noop(),
    ))
    .unwrap();
    probe.finish(&format!("dfnet-{seconds}s"), maximum_mb);
    assert!(output.sample_count().abs_diff(seconds * 48_000) < 2 * 480);
    let output_path = output.path().to_owned();
    let mut window = [0.0; 4096];
    for offset in (0..output.sample_count()).step_by(window.len()) {
        let copied = output.copy_samples(&mut window, offset, 4096).unwrap();
        assert!(copied > 0);
        assert!(window[..copied].iter().all(|v| v.is_finite()));
    }
    output.cleanup().unwrap();
    drop(output);
    assert!(!output_path.exists());
    assert_eq!(std::fs::metadata(&input).unwrap().len(), before_input);
    assert_eq!(
        std::fs::read_dir(&scratch.0).unwrap().count(),
        2,
        "only original and warmup survive"
    );
}

#[test]
#[ignore = "requires pinned ORT and dfnet; execute serially in current audio long-file gate"]
fn real_dfnet_five_minute_file_stays_under_300_mb_and_returns_every_sample() {
    denoise_memory(300, 300.0);
}
#[test]
#[ignore = "requires pinned ORT and dfnet; execute serially in current audio long-file gate"]
fn real_dfnet_ten_minute_file_stays_under_200_mb_and_returns_every_sample() {
    denoise_memory(600, 200.0);
}

#[test]
#[ignore = "requires pinned ORT and fireredvad_aed; execute serially in current audio long-file gate"]
fn real_aed_one_second_and_window_tail_and_ten_minutes_are_complete_with_bounded_memory() {
    let scratch = Scratch::new();
    let model = AudioEventDetector::new(backend(ModelId::FireRedVadAed, "fireredvad_aed.onnx"));
    let warm = scratch.tone("warm.wav", 2, 16_000);
    futures_executor::block_on(AudioEventDetecting::detect(
        &model,
        Path::new(&warm),
        ProgressSink::noop(),
    ))
    .unwrap();
    for seconds in [1, 61, 600] {
        let input = scratch.tone(&format!("original-{seconds}.wav"), seconds, 16_000);
        let probe = ResidentProbe::start();
        let result = futures_executor::block_on(AudioEventDetecting::detect(
            &model,
            &input,
            ProgressSink::noop(),
        ))
        .unwrap();
        probe.finish(&format!("aed-{seconds}s"), 300.0);
        assert!((result.duration_seconds - seconds as f64).abs() < 0.01);
        assert_eq!(result.frame_probs.len(), (seconds * 16_000 - 400) / 160 + 1);
        assert!(
            result
                .frame_probs
                .iter()
                .all(|p| p.speech.is_finite() && p.singing.is_finite() && p.music.is_finite())
        );
        assert!(input.exists());
    }
}
