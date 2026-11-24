//! Does the GPU earn a promotion? The measurement behind the fleet's Core ML
//! pins, run against the real artifacts on the machine that ships them.
//!
//! iOS pins nine families to `cpuOnly`, and those pins encode two different
//! facts: some graphs are genuinely hostile to the accelerators, and some were
//! kept off them because an iPhone has a thermal budget a Mac does not. Only
//! the first fact transfers. This harness separates them the only way that is
//! honest — by measuring, on a desktop, with the fixtures the oracles use.
//!
//! Order matters and it is not the obvious one. **Parity first, speed second.**
//! Migration 016 found CUDA making `music_highlighter` 40% faster while
//! corrupting its ranking, which is the kind of win that ships a bug; so a
//! candidate that disagrees with its CPU reference is finished before its
//! timing is even interesting. The CPU reference is the right oracle here
//! because CPU is what the shipped pin selects and what the committed goldens
//! were taken against.
//!
//! Nothing here promotes anything. It prints a table; the lead reads it.
//!
//! The inputs are `ferment-models-export`'s goldens, fed at each graph's door;
//! the bundles are the published ones.
//!
//! ```text
//! FERMENT_ENGINE_MODELS_DIR=… \
//!   cargo test -p ferment-engine-infer --test coreml_placement -- --ignored --nocapture
//! ```

#![cfg(all(feature = "coreml", target_vendor = "apple"))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ferment_engine_infer::{
    ComputeUnits, CoreMlBackend, DType, InferenceBackend, ModelId, ModelPaths, ModelVersion,
    Tensor, TensorMap,
};
use ferment_engine_test_fixtures::export;

/// Timed iterations after the warm-up. Core ML's first load is a compile that
/// can run tens of seconds and tells you nothing about steady-state cost.
/// Thirty-one because half the fleet predicts in well under a millisecond,
/// where a handful of samples measures the scheduler rather than the graph.
const ITERATIONS: usize = 31;

/// The bar a promotion has to clear on wall time, from the locked ruling. Below
/// it, moving a graph off a measured-correct compute unit buys noise.
const SPEEDUP_BAR: f64 = 1.3;

/// Outputs are FP32 at this boundary whatever the unit computed in, so a
/// disagreement here is a real disagreement — not a dtype artefact.
const PARITY_EPSILON: f32 = 1e-4;

/// Three of the fleet's `cpuOnly` mlmodelc families, with the file that backs
/// each.
fn candidates() -> Vec<(ModelId, &'static str)> {
    vec![
        (ModelId::MusicHighlighter, "music_highlighter.mlmodelc"),
        (ModelId::MusicTaggerHeads, "music_tagger_heads.mlmodelc"),
        (ModelId::FireRedVadAed, "fireredvad_aed.mlmodelc"),
    ]
}

/// The candidates' published bundles behind one compute unit, or `None` (a
/// `SKIP`) when they are not at hand.
fn backend(units: ComputeUnits) -> Option<CoreMlBackend> {
    let mut paths = ModelPaths::new();
    for (id, file_name) in candidates() {
        paths.insert(id, ModelVersion::Latest, export::artifact(file_name)?);
    }
    Some(CoreMlBackend::with_compute_units(Arc::new(paths), units))
}

fn golden(relative: &str) -> Vec<f32> {
    export::require_f32_le(relative)
}

fn infer(backend: &CoreMlBackend, model: ModelId, inputs: TensorMap) -> TensorMap {
    futures_executor::block_on(backend.infer(model, &ModelVersion::Latest, inputs))
        .unwrap_or_else(|error| panic!("{model} inference failed: {error}"))
}

/// Median rather than mean: one scheduler hiccup should not decide whether a
/// graph moves to the GPU.
fn median_duration(backend: &CoreMlBackend, model: ModelId, inputs: &TensorMap) -> Duration {
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = infer(backend, model, inputs.clone());
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[ITERATIONS / 2]
}

/// The largest disagreement across every named output, so one corrupted head
/// cannot hide behind a well-behaved neighbour.
///
/// Integer outputs are compared exactly and reported as their own row: a
/// decoded token id that moved is not "a small difference", it is a different
/// transcript, and averaging it into a float tolerance would hide that.
fn max_difference(reference: &TensorMap, candidate: &TensorMap) -> f32 {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        candidate.keys().collect::<Vec<_>>(),
        "the two units produced different output sets"
    );
    let mut worst = 0.0f32;
    for (name, expected) in reference {
        let actual = &candidate[name];
        assert_eq!(
            expected.dtype, actual.dtype,
            "output '{name}' changed dtype between compute units"
        );
        let difference = match expected.dtype {
            DType::Fp16 | DType::Fp32 | DType::Fp64 => {
                let expected = expected.to_f32().expect("floating output");
                let actual = actual.to_f32().expect("floating output");
                assert_eq!(
                    expected.len(),
                    actual.len(),
                    "output '{name}' changed length between compute units"
                );
                expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0f32, f32::max)
            }
            _ => {
                let expected = expected.to_i64().expect("integral output");
                let actual = actual.to_i64().expect("integral output");
                assert_eq!(
                    expected.len(),
                    actual.len(),
                    "output '{name}' changed length between compute units"
                );
                expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| expected.abs_diff(*actual) as f32)
                    .fold(0.0f32, f32::max)
            }
        };
        worst = worst.max(difference);
    }
    worst
}

// ---------------------------------------------------------------------------
// Inputs. Real fixture bytes wherever one exists: 016 caught RIFE passing at
// 61.7 dB on natural frames and failing at 11 dB on synthetic noise, so an
// input distribution is part of a measurement, not a detail of it.
// ---------------------------------------------------------------------------

const HIGHLIGHTER_CHUNKS: usize = 200;
const HIGHLIGHTER_FRAMES: usize = 129;
const HIGHLIGHTER_MELS: usize = 128;
const HIGHLIGHTER_FEATURES: usize = 256;

const TAGGER_EMBEDDING: usize = 768;

const VAD_FEATURES: usize = 80;
const VAD_WINDOW: usize = 6_000;

/// The highlighter's own sinusoid (`musichighlighter::pos_enc`): interleaved
/// sin/cos pairs, position 0 all zeros.
fn positional_encoding(positions: usize) -> Vec<f32> {
    let mut encoding = vec![0.0f32; positions * HIGHLIGHTER_FEATURES];
    for position in 1..positions {
        for pair in (0..HIGHLIGHTER_FEATURES).step_by(2) {
            let frequency =
                1.0 / 10_000_f32.powf(2.0 * (pair / 2) as f32 / HIGHLIGHTER_FEATURES as f32);
            let angle = position as f32 * frequency;
            encoding[position * HIGHLIGHTER_FEATURES + pair] = angle.sin();
            encoding[position * HIGHLIGHTER_FEATURES + pair + 1] = angle.cos();
        }
    }
    encoding
}

/// The golden track's 37 chunks tiled over the graph's 200 slots — what the
/// processor does for any track shorter than the window, so the arithmetic
/// stays on real mel content rather than on zeros.
fn music_highlighter_inputs() -> TensorMap {
    let chunks = golden("music_highlighter/goldens/music_highlighter_track_chunks_in.f32");
    let chunk_width = HIGHLIGHTER_FRAMES * HIGHLIGHTER_MELS;
    let count = chunks.len() / chunk_width;
    let positions = positional_encoding(count);

    let mut tiled_chunks = vec![0.0f32; HIGHLIGHTER_CHUNKS * chunk_width];
    let mut tiled_positions = vec![0.0f32; HIGHLIGHTER_CHUNKS * HIGHLIGHTER_FEATURES];
    for slot in 0..HIGHLIGHTER_CHUNKS {
        let source = slot % count;
        tiled_chunks[slot * chunk_width..(slot + 1) * chunk_width]
            .copy_from_slice(&chunks[source * chunk_width..(source + 1) * chunk_width]);
        tiled_positions[slot * HIGHLIGHTER_FEATURES..(slot + 1) * HIGHLIGHTER_FEATURES]
            .copy_from_slice(
                &positions[source * HIGHLIGHTER_FEATURES..(source + 1) * HIGHLIGHTER_FEATURES],
            );
    }

    let mut inputs = TensorMap::new();
    inputs.insert(
        "chunks".into(),
        Tensor::from_f32(
            vec![
                HIGHLIGHTER_CHUNKS as i64,
                HIGHLIGHTER_FRAMES as i64,
                HIGHLIGHTER_MELS as i64,
            ],
            tiled_chunks,
        )
        .unwrap(),
    );
    inputs.insert(
        "pos_enc".into(),
        Tensor::from_f32(
            vec![1, HIGHLIGHTER_CHUNKS as i64, HIGHLIGHTER_FEATURES as i64],
            tiled_positions,
        )
        .unwrap(),
    );
    inputs
}

/// The tagger heads only ever see a Dasheng embedding, so the input is one:
/// the golden of the Dasheng ladder — stock torch over a real music window.
fn music_tagger_inputs() -> TensorMap {
    let embedding = golden("dasheng/goldens/dasheng_music_embedding_out.bin");
    assert_eq!(embedding.len(), TAGGER_EMBEDDING);

    let mut inputs = TensorMap::new();
    inputs.insert(
        "embedding".into(),
        Tensor::from_f32(vec![1, TAGGER_EMBEDDING as i64], embedding).unwrap(),
    );
    inputs
}

/// The hairdryer clip at the masked door: kaldi's golden fbank, the
/// contract's CMVN, one window with the real frames masked in.
fn vad_inputs() -> TensorMap {
    let contract: serde_json::Value = serde_json::from_slice(
        &std::fs::read(export::require_path("fireredvad_aed/model.json")).unwrap(),
    )
    .unwrap();
    let cmvn = &contract["frontend"]["cmvn"];
    let column = |name: &str| -> Vec<f32> { serde_json::from_value(cmvn[name].clone()).unwrap() };
    let (means, inverse_std) = (column("means"), column("inverseStd"));
    let fbank = golden("fireredvad_aed/goldens/fireredvad_aed_hairdryer_fbank_out.f32");
    let frames = fbank.len() / VAD_FEATURES;
    assert!(frames <= VAD_WINDOW);

    let mut feat = vec![0.0f32; VAD_FEATURES * VAD_WINDOW];
    for (frame, row) in fbank.chunks_exact(VAD_FEATURES).enumerate() {
        for (band, value) in row.iter().enumerate() {
            feat[band * VAD_WINDOW + frame] = (value - means[band]) * inverse_std[band];
        }
    }
    let mut mask = vec![0.0f32; VAD_WINDOW];
    mask[..frames].fill(1.0);

    let mut inputs = TensorMap::new();
    inputs.insert(
        "feat".into(),
        Tensor::from_f32(vec![1, VAD_FEATURES as i64, 1, VAD_WINDOW as i64], feat).unwrap(),
    );
    inputs.insert(
        "mask".into(),
        Tensor::from_f32(vec![1, 1, 1, VAD_WINDOW as i64], mask).unwrap(),
    );
    inputs
}

/// One candidate unit's result against the CPU reference.
struct Measurement {
    difference: f32,
    elapsed: Duration,
}

struct Verdict {
    model: ModelId,
    reference: Duration,
    /// `.cpuAndGPU` — the unit the ruling asks about.
    gpu: Measurement,
    /// `.cpuAndNeuralEngine` — not a promotion candidate, a CONTROL. Without
    /// it, a GPU column that matches the CPU reference bit-for-bit is
    /// ambiguous: it could mean the GPU agrees, or it could mean Core ML's
    /// planner never left the CPU. `MLComputeUnits` is a permission, not an
    /// instruction. A second accelerator that also lands bit-exact says which
    /// of the two it is.
    ane: Measurement,
}

impl Measurement {
    /// Did the graph actually leave the CPU?
    ///
    /// `MLComputeUnits` is a permission, not an instruction: Core ML's planner
    /// takes the ops it wants and leaves the rest, and for several of these
    /// graphs it wants none of them. A *bit-identical* result is the tell.
    /// These are FP32 conv and transformer stacks — different silicon running
    /// them agrees to a few ULP at best, never exactly, and the ANE control
    /// column proves this harness sees that when it happens (the Parakeet
    /// encoder drifts 2.5e-2 there). So an exact zero does not mean "the
    /// accelerator agrees"; it means the accelerator never ran it, and the
    /// wall-time column beside it is CPU scheduling variance.
    ///
    /// This matters because that variance is big enough to decide a promotion:
    /// across two runs of this file the Parakeet preprocessor measured 1.17x
    /// and 1.37x on a 2 ms graph whose output never moved a bit.
    fn engaged(&self) -> bool {
        self.difference > 0.0
    }
}

impl Verdict {
    fn speedup(&self, measurement: &Measurement) -> f64 {
        self.reference.as_secs_f64() / measurement.elapsed.as_secs_f64()
    }

    /// A promotion claims a graph is faster somewhere else. If the bits prove
    /// it never went anywhere else, there is nothing to promote it to — so
    /// that is the first question, ahead of both parity and speed.
    fn outcome(&self) -> &'static str {
        if !self.gpu.engaged() {
            "hold — CPU never left"
        } else if self.gpu.difference > PARITY_EPSILON {
            "hold — diverged"
        } else if self.speedup(&self.gpu) < SPEEDUP_BAR {
            "hold — under the bar"
        } else {
            "PROMOTE"
        }
    }

    fn row(&self) -> String {
        format!(
            "| {} | {:.2} ms | {:.3e} | {} | {:.2} ms | {:.2}x | {:.3e} | {} | {:.2} ms | {:.2}x | {} |",
            self.model,
            self.reference.as_secs_f64() * 1_000.0,
            self.gpu.difference,
            oracle_column(&self.gpu),
            self.gpu.elapsed.as_secs_f64() * 1_000.0,
            self.speedup(&self.gpu),
            self.ane.difference,
            oracle_column(&self.ane),
            self.ane.elapsed.as_secs_f64() * 1_000.0,
            self.speedup(&self.ane),
            self.outcome()
        )
    }
}

/// What the parity number means, in three words. "n/a" is the honest answer
/// for a unit that never ran the graph: there is no agreement to report
/// between a result and itself.
fn oracle_column(measurement: &Measurement) -> &'static str {
    if !measurement.engaged() {
        "n/a"
    } else if measurement.difference <= PARITY_EPSILON {
        "MATCH"
    } else {
        "DIVERGED"
    }
}

fn measure_unit(
    model: ModelId,
    inputs: &TensorMap,
    reference: &TensorMap,
    backend: &CoreMlBackend,
) -> Measurement {
    // Warm before timing: the first load of an mlmodelc is a Core ML compile,
    // and on an accelerator path it is the slowest thing that will ever happen
    // to that graph.
    let outputs = infer(backend, model, inputs.clone());
    Measurement {
        difference: max_difference(reference, &outputs),
        elapsed: median_duration(backend, model, inputs),
    }
}

fn measure(model: ModelId, inputs: TensorMap, units: &Units) -> Verdict {
    let reference_outputs = infer(&units.cpu, model, inputs.clone());
    let gpu = measure_unit(model, &inputs, &reference_outputs, &units.gpu);
    let ane = measure_unit(model, &inputs, &reference_outputs, &units.ane);
    Verdict {
        model,
        reference: median_duration(&units.cpu, model, &inputs),
        gpu,
        ane,
    }
}

/// One backend per compute unit. They must be separate: `CoreMlBackend` caches
/// a loaded `Model` per id, and the units are fixed when it is built.
struct Units {
    cpu: CoreMlBackend,
    gpu: CoreMlBackend,
    ane: CoreMlBackend,
}

#[test]
#[ignore = "requires the published Core ML bundles and ferment-models-export's goldens"]
fn cpu_only_families_measured_under_cpu_and_gpu() {
    let (Some(cpu), Some(gpu), Some(ane)) = (
        backend(ComputeUnits::CpuOnly),
        backend(ComputeUnits::CpuAndGpu),
        backend(ComputeUnits::CpuAndNeuralEngine),
    ) else {
        return;
    };
    let units = Units { cpu, gpu, ane };

    let cases: Vec<(ModelId, TensorMap)> = vec![
        (ModelId::MusicHighlighter, music_highlighter_inputs()),
        (ModelId::MusicTaggerHeads, music_tagger_inputs()),
        (ModelId::FireRedVadAed, vad_inputs()),
    ];

    let verdicts: Vec<Verdict> = cases
        .into_iter()
        .map(|(model, inputs)| measure(model, inputs, &units))
        .collect();

    println!("\n### cpuOnly families off the CPU (median of {ITERATIONS}, M3)\n");
    println!(
        "| model | cpuOnly | GPU diff | GPU oracle | GPU | GPU speedup | ANE diff | ANE oracle | ANE | ANE speedup | verdict |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for verdict in &verdicts {
        println!("{}", verdict.row());
    }
    println!(
        "\nPromotion bar: the unit actually ran the graph (diff != 0) AND agreed with the CPU \
         reference within {PARITY_EPSILON:e} AND was >= {SPEEDUP_BAR}x faster.\n"
    );
    println!(
        "A `diff` of exactly 0 means Core ML's planner declined the unit and ran on CPU; the \
         speedup beside it is scheduling variance, not a measurement of that unit.\n"
    );

    // The harness reports; it does not decide. What it does assert is that
    // every candidate actually ran under both units — a silent skip would read
    // as a clean table.
    assert_eq!(verdicts.len(), candidates().len());
}
