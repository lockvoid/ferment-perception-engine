//! Portable parts of current CoreAI parity tests: exact original PyTorch
//! fixture/grid and repeated real graph lifetime. ORT replaces Apple providers;
//! Apple .aimodel specialization itself is not an Android requirement.
#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]
use ferment_engine_infer::{
    InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
    TensorMap,
};
use ferment_engine_processors::{
    DeepFilterNet,
    beatthis::peaks::{deduplicate_peaks, find_peaks},
};
use ferment_engine_test_fixtures::{self as fixtures, export};
use futures_executor::block_on;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

fn backend(id: ModelId, name: &str) -> Arc<OrtBackend> {
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "required admitted runtime"
    );
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(ModelPaths::new().with_path(
            id,
            ModelVersion::Latest,
            export::artifact(name).expect("required graph"),
        )),
        OrtExecutionProvider::Cpu,
    ))
}
fn beat_input() -> TensorMap {
    assert!(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").is_some(),
        "required frozen original Swift fixtures"
    );
    let frames_major = fixtures::require_f32_le("model_input_mel.bin");
    assert_eq!(frames_major.len(), 1500 * 128);
    let mut bands_major = vec![0.0; frames_major.len()];
    for t in 0..1500 {
        for b in 0..128 {
            bands_major[b * 1500 + t] = frames_major[t * 128 + b];
        }
    }
    let mut inputs = TensorMap::new();
    inputs.insert(
        "mel".into(),
        Tensor::from_f32(vec![1, 1, 128, 1500], bands_major).unwrap(),
    );
    inputs
}
fn beat_logits(backend: &OrtBackend, input: TensorMap) -> Vec<f32> {
    let out = block_on(backend.infer(ModelId::BeatThis1, &ModelVersion::Latest, input)).unwrap();
    assert_eq!(out["logits"].shape, vec![1, 2, 1, 1500]);
    out["logits"].to_f32().unwrap()
}
fn grid(logits: &[f32]) -> Vec<f64> {
    deduplicate_peaks(&find_peaks(logits, 7))
}

#[test]
#[ignore = "requires admitted ORT, beat_this_1 and the exact current Swift model-stage fixture"]
fn current_beat_logits_and_grid_answer_the_original_pytorch_fixture() {
    let backend = backend(ModelId::BeatThis1, "beat_this_1.onnx");
    let actual = beat_logits(&backend, beat_input());
    let reference = fixtures::require_f32_le("model_output_logits.bin");
    assert_eq!(reference.len(), 3000);
    for axis in 0..2 {
        let expected = (0..1500)
            .map(|t| reference[t * 2 + axis])
            .collect::<Vec<_>>();
        let ours = &actual[axis * 1500..(axis + 1) * 1500];
        assert!(ours.iter().all(|v| v.is_finite()));
        let error = ours
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Same bound as current palettized CoreAI source (CoreML's is looser,
        // 0.5). The existing fp32 export contract keeps its stronger 0.02.
        assert!(error < 0.25, "axis{axis}: max delta {error}");
        let a = grid(ours);
        let b = grid(&expected);
        assert_eq!(a.len(), b.len(), "axis{axis}: a beat gained or lost");
        let drift = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        // Android uses the fp32 independent grid directly; it has no second
        // Apple provider whose error could enlarge this 0.5-frame floor.
        assert!(drift <= 0.5, "axis{axis}: grid moved {drift} frames");
    }
}

fn rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").unwrap();
        text.lines()
            .find(|s| s.starts_with("VmRSS:"))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u64>()
            .unwrap()
            * 1024
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
struct Probe {
    baseline: u64,
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Probe {
    fn new() -> Self {
        let baseline = rss_bytes();
        let peak = Arc::new(AtomicU64::new(baseline));
        let stop = Arc::new(AtomicBool::new(false));
        let p = peak.clone();
        let s = stop.clone();
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(rss_bytes(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Self {
            baseline,
            peak,
            stop,
            thread: Some(thread),
        }
    }
    fn finish(mut self, label: &str) {
        let retained = rss_bytes();
        self.peak.fetch_max(retained, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap();
        let delta = (retained as i128 - self.baseline as i128) as f64 / 1_048_576.0;
        eprintln!(
            "AUDIO_REPEATED_MEMORY {label} baseline_rss={} retained_rss={retained} sampled_peak_rss={} sample_ms=50 retained_delta_mb={delta} criterion_mb=32",
            self.baseline,
            self.peak.load(Ordering::Relaxed)
        );
        assert!(delta < 32.0, "{label}: retained growth {delta} MB");
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.join().unwrap();
        }
    }
}

#[test]
#[ignore = "requires admitted ORT and beat_this_1; run serially for memory measurement"]
fn current_eight_beat_chunks_do_not_accumulate_32_mb() {
    let backend = backend(ModelId::BeatThis1, "beat_this_1.onnx");
    let input = beat_input();
    drop(beat_logits(&backend, input.clone()));
    let probe = Probe::new();
    for iteration in 1..=8 {
        let output = beat_logits(&backend, input.clone());
        assert_eq!(output.len(), 3000);
        drop(output);
        let resident = backend.cache_snapshot().resident;
        eprintln!(
            "AUDIO_REPEATED_STEP beat-this-8 iteration={iteration} rss={} resident_sessions={resident}",
            rss_bytes(),
        );
        assert_eq!(resident, 1, "a chunk must reuse the resident model session");
    }
    probe.finish("beat-this-8");
    assert_eq!(backend.cache_snapshot().resident, 1);
    assert_eq!(
        backend.selected_provider_for(ModelId::BeatThis1),
        Some(OrtExecutionProvider::Cpu)
    );
}
#[test]
#[ignore = "requires admitted ORT and dfnet; run serially for memory measurement"]
fn current_twelve_denoiser_chunks_do_not_accumulate_32_mb() {
    let backend = backend(ModelId::Dfnet, "dfnet.onnx");
    let model = DeepFilterNet::new(backend.clone());
    let erb = export::require_f32_le("dfnet/goldens/dfnet_feat_erb_in.f32");
    let spec = export::require_f32_le("dfnet/goldens/dfnet_feat_spec_in.f32");
    assert_eq!(erb.len(), 1500 * 32);
    assert_eq!(spec.len(), 2 * 1500 * 96);
    drop(block_on(model.run_chunk(&erb, &spec, 1500)).unwrap());
    let probe = Probe::new();
    for _ in 0..12 {
        let result = block_on(model.run_chunk(&erb, &spec, 1500)).unwrap();
        assert_eq!(result.lsnr.len(), 1500);
        drop(result);
    }
    probe.finish("dfnet-12");
    assert_eq!(backend.cache_snapshot().resident, 1);
    assert_eq!(
        backend.selected_provider_for(ModelId::Dfnet),
        Some(OrtExecutionProvider::Cpu)
    );
}
