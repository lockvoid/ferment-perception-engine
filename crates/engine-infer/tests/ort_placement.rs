//! The one real ANE gap between Studio and iOS, measured.
//!
//! iOS runs DINOv2 on the Neural Engine from a compiled `.mlmodelc`. Studio has
//! no Core ML DINOv2 artifact at all — its visual encoder is the ONNX master on
//! every platform — so the only route to the same hardware is ONNX Runtime's
//! Core ML execution provider. This asks whether that route is worth taking.
//!
//! The metric is cosine distance, not max-abs-diff, because that is what the
//! consumer computes: `Dinov2SemanticEncoder::cosine_distance` is what decides
//! whether two frames are the same shot. An absolute drift that leaves the unit
//! vector pointing the same way costs nothing; one that rotates it moves scene
//! boundaries.
//!
//! ```text
//! ORT_DYLIB_PATH=… cargo test -p ferment-engine-infer --features ort \
//!   --test ort_placement -- --ignored --nocapture
//! ```

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferment_engine_infer::{
    InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider, Tensor,
    TensorMap,
};
use ferment_engine_test_fixtures::export;

const ITERATIONS: usize = 21;
const SPEEDUP_BAR: f64 = 1.3;
/// Two embeddings this close describe the same frame for every consumer in the
/// pipeline: the shot-identity threshold is orders of magnitude coarser.
const COSINE_EPSILON: f32 = 1e-4;

const WIDTH: usize = 224;
const HEIGHT: usize = 126;
const EMBEDDING: usize = 384;

/// A real frame at model resolution: one of the export's golden frames of
/// `strobe-cuts.mp4`, planar `[3, H, W]` in 0..255 — the layout and range
/// `Dinov2SemanticEncoder` hands the backend; the graph does its own
/// normalization. Real content matters here — 016 caught RIFE passing at
/// 61.7 dB on natural frames and failing at 11 dB on synthetic noise.
fn frame_input() -> Option<TensorMap> {
    let path = export::path("dinov2/goldens/dinov2_f037_image_in.u8")?;
    let planar = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
    assert_eq!(planar.len(), 3 * WIDTH * HEIGHT);
    let mut inputs = TensorMap::new();
    inputs.insert(
        "image".into(),
        Tensor::from_f32(
            vec![1, 3, HEIGHT as i64, WIDTH as i64],
            planar.into_iter().map(f32::from).collect(),
        )
        .unwrap(),
    );
    Some(inputs)
}

/// The published fp32 master, on one provider.
fn backend(master: &Path, provider: OrtExecutionProvider) -> OrtBackend {
    let paths = ModelPaths::new().with_path(
        ModelId::Dinov2Vits14,
        ModelVersion::Latest,
        master.to_path_buf(),
    );
    // A pinned single-provider chain, so an unavailable provider errors instead
    // of quietly handing back CPU numbers under a Core ML label — the exact
    // trap 016 fell into on the Python side.
    OrtBackend::with_execution_provider(Arc::new(paths), provider)
}

fn embed(backend: &OrtBackend, frame: &TensorMap) -> Vec<f32> {
    let outputs = futures_executor::block_on(backend.infer(
        ModelId::Dinov2Vits14,
        &ModelVersion::Latest,
        frame.clone(),
    ))
    .unwrap_or_else(|error| panic!("DINOv2 inference failed: {error}"));
    let embedding = outputs["embedding"].to_f32().unwrap();
    assert_eq!(embedding.len(), EMBEDDING);
    embedding
}

fn median_duration(backend: &OrtBackend, frame: &TensorMap) -> Duration {
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = embed(backend, frame);
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    samples[ITERATIONS / 2]
}

/// Cosine distance between the raw (un-normalized) embeddings — the same
/// quantity `Dinov2SemanticEncoder::cosine_distance` computes once they are
/// unit vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let norm = |values: &[f32]| values.iter().map(|value| value * value).sum::<f32>().sqrt();
    1.0 - dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "needs ORT_DYLIB_PATH and the published DINOv2 master"]
fn dinov2_measured_on_the_ort_coreml_provider() {
    let (Some(master), Some(frame)) = (export::artifact("dinov2_vits14.onnx"), frame_input())
    else {
        return;
    };
    let cpu = backend(&master, OrtExecutionProvider::Cpu);
    let reference = embed(&cpu, &frame);
    let cpu_elapsed = median_duration(&cpu, &frame);

    let coreml = backend(&master, OrtExecutionProvider::CoreMl);
    // The provider is registered `error_on_failure`, so an ORT build without
    // the Core ML EP fails here rather than reporting CPU numbers twice.
    let candidate = embed(&coreml, &frame);
    let coreml_elapsed = median_duration(&coreml, &frame);

    let cosine = cosine_distance(&reference, &candidate);
    let absolute = reference
        .iter()
        .zip(&candidate)
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0f32, f32::max);
    let speedup = cpu_elapsed.as_secs_f64() / coreml_elapsed.as_secs_f64();

    println!("\n### DINOv2 on the ORT Core ML provider (median of {ITERATIONS}, M3)\n");
    println!(
        "| model | cosine distance | max abs diff | oracle | CPU EP | CoreML EP | speedup | verdict |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    println!(
        "| dinov2_vits14 | {cosine:.3e} | {absolute:.3e} | {} | {:.2} ms | {:.2} ms | {speedup:.2}x | {} |",
        if cosine <= COSINE_EPSILON {
            "MATCH"
        } else {
            "DIVERGED"
        },
        cpu_elapsed.as_secs_f64() * 1_000.0,
        coreml_elapsed.as_secs_f64() * 1_000.0,
        if cosine <= COSINE_EPSILON && speedup >= SPEEDUP_BAR {
            "PROMOTE"
        } else {
            "hold"
        }
    );
    println!(
        "\nPromotion bar: cosine distance within {COSINE_EPSILON:e} of the CPU reference AND \
         >= {SPEEDUP_BAR}x faster.\n"
    );

    // The harness reports; the lead signs. What it asserts is that both
    // providers genuinely produced an embedding.
    assert!(reference.iter().any(|value| *value != 0.0));
    assert!(candidate.iter().any(|value| *value != 0.0));
}
