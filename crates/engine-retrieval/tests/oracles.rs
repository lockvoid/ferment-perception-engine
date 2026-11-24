//! Real end-to-end encoders: original PCM/RGB/text → Rust frontend → model.
//! Never grades only precomputed mel/tokens while bypassing the public API.
#![cfg(any(
    feature = "ort",
    all(feature = "coreml", target_vendor = "apple"),
    all(feature = "coreai", target_os = "macos")
))]

use ferment_engine_core::{ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelPaths, ModelVersion};
use ferment_engine_retrieval::{
    Embedding, EmbeddingSpace, Glap, GlapLanguage, GlapTokenizer, Siglip2, Siglip2Tokenizer,
};
use ferment_engine_test_fixtures::export;
use std::{future::Future, sync::Arc, time::Instant};

#[cfg(any(feature = "ort", all(feature = "coreml", target_vendor = "apple")))]
fn paths(extension: &str) -> ModelPaths {
    paths_for(
        extension,
        &[
            ModelId::GlapAudio,
            ModelId::GlapText,
            ModelId::Siglip2Image,
            ModelId::Siglip2Text,
        ],
    )
}

fn paths_for(extension: &str, models: &[ModelId]) -> ModelPaths {
    let mut paths = ModelPaths::new();
    for &id in models {
        let file = format!("{}_fp16.{extension}", id.name());
        let (family, kind) = match id {
            ModelId::GlapAudio => ("glap", "audio"),
            ModelId::GlapText => ("glap", "text"),
            ModelId::Siglip2Image => ("siglip2", "image"),
            ModelId::Siglip2Text => ("siglip2", "text"),
            _ => unreachable!("retrieval model"),
        };
        let backend = if extension.ends_with("aimodelc") {
            "coreai"
        } else if extension == "mlmodelc" {
            "coreml"
        } else {
            "onnx"
        };
        let relative = if backend == "coreai" {
            format!("aot/{file}")
        } else {
            file.clone()
        };
        let path = export::artifact(&relative).unwrap_or_else(|| panic!("missing {relative}"));
        let contract: serde_json::Value =
            serde_json::from_slice(&export::require_bytes(&format!("{family}/model.json")))
                .unwrap();
        let pin = contract["members"][kind]["artifacts"][backend]["pins"]
            .as_array()
            .unwrap()
            .iter()
            .find(|pin| pin["object"] == file)
            .expect("qualified artifact pin");
        assert_eq!(
            ferment_engine_models::sha256_path(&path).unwrap(),
            pin["tree_sha256"].as_str().unwrap()
        );
        paths.insert(id, ModelVersion::Latest, path);
    }
    paths
}

fn compare(actual: &Embedding, family: &str, name: &str) {
    let golden = export::require_f32_le(&format!("{family}/goldens/{name}.embedding.f32"));
    let expected = Embedding::from_unit_values(actual.space(), golden).unwrap();
    let cosine = actual.cosine(&expected).unwrap();
    let max_abs = actual
        .values()
        .iter()
        .zip(expected.values())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("[retrieval-parity] {family}/{name} cosine={cosine:.8} max_abs={max_abs:e}");
    assert!(cosine >= 0.999, "{family}/{name}: cosine {cosine}");
    assert!(max_abs <= 0.01, "{family}/{name}: max_abs {max_abs}");
}

fn rank(query: &Embedding, corpus: &[Embedding]) -> usize {
    corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, query.cosine(v).unwrap()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

// Timings compare identical inputs, not a long cold sample with a short warm
// one. "Cold" means a new backend session; OS/Core ML disk caches may be warm.
async fn measured<F, Fut>(name: &str, first_door: bool, mut run: F) -> Embedding
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Embedding>>,
{
    let start = Instant::now();
    let first = run().await.unwrap();
    eprintln!(
        "[retrieval-time] {name} first_ms={:.2}",
        start.elapsed().as_secs_f64() * 1000.0
    );
    if first_door {
        for repeat in 1..=3 {
            let start = Instant::now();
            let next = run().await.unwrap();
            eprintln!(
                "[retrieval-time] {name} warm={repeat} elapsed_ms={:.2}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            assert!(first.cosine(&next).unwrap() >= 0.99999);
            assert!(
                first
                    .values()
                    .iter()
                    .zip(next.values())
                    .all(|(a, b)| (a - b).abs() <= 0.0001),
                "same-input repeat drifted: {name}"
            );
        }
    }
    first
}

async fn glap_ladder(backend: Backend) {
    let progress = ProgressSink::noop();
    let glap = Glap::new(backend.clone(), ModelVersion::Latest).unwrap();
    let glap_tokens = GlapTokenizer::from_path(
        &export::artifact("glap_sentencepiece.model").expect("GLAP tokenizer"),
    )
    .unwrap();
    let mut audio = vec![];
    let manifest: serde_json::Value =
        serde_json::from_slice(&export::require_bytes("glap/goldens/cases.json")).unwrap();
    assert_eq!(
        manifest["audio"].as_array().unwrap().len(),
        19,
        "extended musical corpus is required"
    );
    for case in manifest["audio"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let samples = export::require_f32_le(&format!("glap/goldens/{name}.audio.f32"));
        let embedding = measured(&format!("glap/{name}"), name == "music", || {
            glap.audio_window(&samples, &progress)
        })
        .await;
        compare(&embedding, "glap", name);
        audio.push(embedding);
    }
    for (i, case) in manifest["tokenization"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let language: GlapLanguage = serde_json::from_value(case["language"].clone()).unwrap();
        let embedding = measured(&format!("glap/text-{i}"), i == 0, || {
            glap.text(
                &glap_tokens,
                case["text"].as_str().unwrap(),
                language,
                &progress,
            )
        })
        .await;
        compare(&embedding, "glap", &format!("text-{i}"));
        if i < 4 {
            assert_eq!(
                rank(&embedding, &audio[..2]),
                i % 2,
                "GLAP RU/EN retrieval {i}"
            );
        }
    }
}

async fn siglip_ladder(backend: Backend) {
    let progress = ProgressSink::noop();
    let siglip = Siglip2::new(backend, ModelVersion::Latest);
    let tokens = Siglip2Tokenizer::from_path(
        &export::artifact("siglip2_tokenizer.json").expect("SigLIP tokenizer"),
    )
    .unwrap();
    let mut images = vec![];
    let manifest: serde_json::Value =
        serde_json::from_slice(&export::require_bytes("siglip2/goldens/cases.json")).unwrap();
    for (i, case) in manifest["image"].as_array().unwrap().iter().enumerate() {
        let name = case["name"].as_str().unwrap();
        let w = case["width"].as_u64().unwrap() as u32;
        let h = case["height"].as_u64().unwrap() as u32;
        let rgb = export::require_bytes(&format!("siglip2/goldens/{name}.rgb.u8"));
        let embedding = measured(&format!("siglip2/{name}"), i == 0, || {
            siglip.image_rgb(&rgb, w, h, &progress)
        })
        .await;
        compare(&embedding, "siglip2", name);
        images.push(embedding);
    }
    for (i, case) in manifest["tokenization"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let embedding = measured(&format!("siglip2/text-{i}"), i == 0, || {
            siglip.text(&tokens, case["text"].as_str().unwrap(), &progress)
        })
        .await;
        compare(&embedding, "siglip2", &format!("text-{i}"));
        if i < 3 {
            assert_eq!(
                rank(&embedding, &images[..3]),
                i,
                "SigLIP color retrieval {i}"
            );
        }
        if i >= 7 {
            assert_eq!(
                rank(&embedding, &images[4..]),
                (i - 7) % 3,
                "SigLIP real-video RU/EN retrieval {i}"
            );
        }
        assert_eq!(embedding.space(), EmbeddingSpace::Siglip2Base224V1);
    }
}

#[cfg(any(feature = "ort", all(feature = "coreml", target_vendor = "apple")))]
async fn ladder(backend: Backend) {
    glap_ladder(backend.clone()).await;
    siglip_ladder(backend).await;
}

async fn reload_after_eviction(backend: Backend) {
    let rgb = export::require_bytes("siglip2/goldens/red.rgb.u8");
    let manifest: serde_json::Value =
        serde_json::from_slice(&export::require_bytes("siglip2/goldens/cases.json")).unwrap();
    let case = manifest["image"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "red")
        .unwrap();
    let siglip = Siglip2::new(backend, ModelVersion::Latest);
    let result = siglip
        .image_rgb(
            &rgb,
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
            &ProgressSink::noop(),
        )
        .await
        .unwrap();
    compare(&result, "siglip2", "red");
}

#[cfg(feature = "ort")]
#[test]
#[ignore = "requires exported ONNX models, tokenizers, and ORT_DYLIB_PATH"]
fn ort_retrieval_end_to_end() {
    eprintln!("[retrieval-backend] ORT CPU");
    let backend = Arc::new(ferment_engine_infer::OrtBackend::new(Arc::new(paths(
        "onnx",
    ))));
    futures_executor::block_on(ladder(backend.clone()));
    backend.evict_idle_models(&[]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    futures_executor::block_on(reload_after_eviction(backend.clone()));
    let cache = backend.cache_snapshot();
    assert_eq!([cache.resident, cache.loading, cache.waiters], [1, 0, 0]);
}

/// Kill: CPU fallback, a broken Rust frontend, FP16 graph drift, or stale
/// outputs after eviction fails against independent upstream PCM/RGB/text.
#[cfg(all(feature = "ort", target_os = "windows"))]
#[test]
#[ignore = "requires the shipped Windows ORT DLLs, GPU, exports and tokenizers"]
fn ort_directml_retrieval_end_to_end() {
    use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};
    eprintln!("[retrieval-backend] ORT DirectML; no provider fallback");
    let backend = Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths("onnx")),
        OrtExecutionProvider::DirectMl,
    ));
    futures_executor::block_on(ladder(backend.clone()));
    for id in [
        ModelId::GlapAudio,
        ModelId::GlapText,
        ModelId::Siglip2Image,
        ModelId::Siglip2Text,
    ] {
        assert_eq!(
            backend.selected_provider_for(id),
            Some(OrtExecutionProvider::DirectMl)
        );
    }
    backend.evict_idle_models(&[]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    futures_executor::block_on(reload_after_eviction(backend.clone()));
    assert_eq!(
        backend.selected_provider_for(ModelId::Siglip2Image),
        Some(OrtExecutionProvider::DirectMl)
    );
}

/// Kill: a missing catalog registration, broken graph-assignment ABI, or
/// implicit CPU fallback fails even if the embeddings happen to be correct.
/// Also kills ORT's default MaxEfficiency stealing a CPU-pinned model after
/// another backend registered a process-wide accelerator.
#[cfg(all(feature = "ort", target_os = "windows"))]
#[test]
#[ignore = "requires Windows ML, GPU, exports and tokenizers"]
fn windows_ml_retrieval_and_cpu_pin_end_to_end() {
    use ferment_engine_infer::{OrtBackend, OrtExecutionProvider};
    let locator = Arc::new(paths("onnx"));
    let backend = Arc::new(OrtBackend::windows_default(locator.clone()));
    futures_executor::block_on(ladder(backend.clone()));
    let check = |id| {
        assert_eq!(
            backend.selected_provider_for(id),
            Some(OrtExecutionProvider::WindowsMl)
        );
        let assignments = backend
            .graph_assignment_for(id)
            .expect("actual graph assignment");
        let expected = std::env::var("FERMENT_EXPECT_EP").ok();
        let mut accelerated = 0;
        for graph in assignments {
            eprintln!("{id:?}: {} {} nodes", graph.provider, graph.nodes.len());
            if graph.provider == "CPUExecutionProvider" {
                assert!(
                    graph.nodes.iter().all(|node| !matches!(
                        node.operator.as_str(),
                        "Conv" | "ConvTranspose" | "MatMul" | "Gemm"
                    )),
                    "weighted CPU fallback: {graph:?}"
                );
            } else {
                if let Some(expected) = &expected {
                    assert_eq!(&graph.provider, expected);
                }
                accelerated += graph.nodes.len();
            }
        }
        assert!(accelerated > 0);
    };
    for id in [
        ModelId::GlapAudio,
        ModelId::GlapText,
        ModelId::Siglip2Image,
        ModelId::Siglip2Text,
    ] {
        check(id);
    }
    backend.evict_idle_models(&[]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    futures_executor::block_on(reload_after_eviction(backend.clone()));
    check(ModelId::Siglip2Image);

    let cpu = Arc::new(OrtBackend::new(locator));
    futures_executor::block_on(reload_after_eviction(cpu.clone()));
    let assignment = cpu
        .graph_assignment_for(ModelId::Siglip2Image)
        .expect("CPU assignment");
    assert!(!assignment.is_empty());
    assert!(
        assignment
            .iter()
            .all(|graph| graph.provider == "CPUExecutionProvider")
    );
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
#[test]
#[ignore = "requires exported Core ML models and tokenizers on macOS"]
fn coreml_retrieval_end_to_end() {
    eprintln!("[retrieval-backend] Core ML CPU_ONLY");
    let backend = Arc::new(ferment_engine_infer::CoreMlBackend::with_compute_units(
        Arc::new(paths("mlmodelc")),
        ferment_engine_infer::ComputeUnits::CpuOnly,
    ));
    futures_executor::block_on(ladder(backend.clone()));
    backend.evict_idle_models(&[]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    futures_executor::block_on(reload_after_eviction(backend.clone()));
    let cache = backend.cache_snapshot();
    assert_eq!([cache.resident, cache.loading, cache.waiters], [1, 0, 0]);
}

/// Kill: bypassing the GLAP short-function dispatch fails 4..7-frame inputs;
/// a strided/interleaved copy bug fails independent upstream embeddings.
#[cfg(all(feature = "coreai", target_os = "macos"))]
#[test]
#[ignore = "requires macOS 27, qualified compiled Core AI models and tokenizers"]
fn coreai_glap_end_to_end() {
    let arch = ferment_engine_infer::coreai_device_architecture().expect("Core AI is unavailable");
    let backend = Arc::new(ferment_engine_infer::CoreAiBackend::new(Arc::new(
        paths_for(
            &format!("macos.{arch}.aimodelc"),
            &[ModelId::GlapAudio, ModelId::GlapText],
        ),
    )));
    futures_executor::block_on(glap_ladder(backend.clone()));
    backend.evict_inactive_models(&[]);
    let cache = backend.cache_snapshot();
    assert_eq!([cache.resident, cache.loading, cache.waiters], [0, 0, 0]);
}

/// Kill: dropping model-owned outputs before materialization or choosing a
/// different checkpoint fails the independent image/text/retrieval goldens.
#[cfg(all(feature = "coreai", target_os = "macos"))]
#[test]
#[ignore = "requires macOS 27, qualified compiled Core AI models and tokenizers"]
fn coreai_siglip_end_to_end() {
    let arch = ferment_engine_infer::coreai_device_architecture().expect("Core AI is unavailable");
    let backend = Arc::new(ferment_engine_infer::CoreAiBackend::new(Arc::new(
        paths_for(
            &format!("macos.{arch}.aimodelc"),
            &[ModelId::Siglip2Image, ModelId::Siglip2Text],
        ),
    )));
    futures_executor::block_on(siglip_ladder(backend.clone()));
    backend.evict_inactive_models(&[]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    futures_executor::block_on(reload_after_eviction(backend.clone()));
    let cache = backend.cache_snapshot();
    assert_eq!([cache.resident, cache.loading, cache.waiters], [1, 0, 0]);
}

/// Kill: dropping a caller releases the native permit/instance before Core AI
/// returns, or leaves a cancelled waiter queued. The next real text embedding
/// must still match the independent checkpoint golden after native eviction.
#[cfg(all(feature = "coreai", target_os = "macos"))]
#[test]
#[ignore = "requires macOS 27 and the qualified GLAP text Core AI bundle"]
fn coreai_cancellation_keeps_native_ownership_until_completion() {
    use ferment_engine_infer::InferenceBackend;
    use futures_executor::block_on;
    use std::task::{Context, Waker};

    let arch = ferment_engine_infer::coreai_device_architecture().expect("Core AI is unavailable");
    let backend = Arc::new(ferment_engine_infer::CoreAiBackend::new(Arc::new(
        paths_for(&format!("macos.{arch}.aimodelc"), &[ModelId::GlapText]),
    )));
    let version = ModelVersion::Latest;
    block_on(backend.warmup(&[ModelId::GlapText])).unwrap();
    let glap = Glap::new(backend.clone(), version.clone()).unwrap();
    let tokens =
        GlapTokenizer::from_path(&export::artifact("glap_sentencepiece.model").unwrap()).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&export::require_bytes("glap/goldens/cases.json")).unwrap();
    let case = &cases["tokenization"][0];
    let text = case["text"].as_str().unwrap();
    let language: GlapLanguage = serde_json::from_value(case["language"].clone()).unwrap();
    let progress = ProgressSink::noop();
    let mut cx = Context::from_waker(Waker::noop());
    let mut active = Box::pin(glap.text(&tokens, text, language, &progress));
    assert!(active.as_mut().poll(&mut cx).is_pending());
    let state = block_on(backend.prediction_snapshot(ModelId::GlapText, &version)).unwrap();
    assert_eq!(
        (state.active, state.completed),
        (1, 0),
        "must observe the real cold native call"
    );
    let mut queued = Box::pin(glap.text(&tokens, text, language, &progress));
    assert!(queued.as_mut().poll(&mut cx).is_pending());
    assert_eq!(
        block_on(backend.prediction_snapshot(ModelId::GlapText, &version))
            .unwrap()
            .waiting,
        1
    );
    drop(queued);
    drop(active);
    let state = block_on(backend.prediction_snapshot(ModelId::GlapText, &version)).unwrap();
    assert_eq!((state.active, state.waiting, state.completed), (1, 0, 0));
    assert!(
        backend.evict_inactive_models(&[]).is_empty(),
        "cancelled reply must not retire the active native model"
    );
    assert_eq!(backend.cache_snapshot().resident, 1);

    let output = block_on(glap.text(&tokens, text, language, &progress)).unwrap();
    let state = block_on(backend.prediction_snapshot(ModelId::GlapText, &version)).unwrap();
    assert_eq!(
        (state.active, state.waiting, state.completed, state.failed),
        (0, 0, 2, 0)
    );
    assert_eq!(state.peak_active, 1);
    assert_eq!(state.peak_waiting, 1);
    assert_eq!(backend.evict_inactive_models(&[]), [ModelId::GlapText]);
    assert_eq!(backend.cache_snapshot().resident, 0);
    drop(glap);
    drop(backend);
    compare(&output, "glap", "text-0");
    eprintln!(
        "[coreai-cancellation] cancelled reply and waiter; native completed=2 failed=0 peak_active=1; output survives eviction"
    );
}
