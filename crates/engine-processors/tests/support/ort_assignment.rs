//! Native evidence: a requested policy is not proof of GPU execution.
use ferment_engine_infer::{ModelId, OrtBackend};

/// Kill: CPU fallback after a failed catalog/EP load must fail GPU qualification.
pub fn require_gpu(backend: &OrtBackend, model: ModelId) {
    if std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() != Ok("windows") {
        return;
    }
    #[cfg(windows)]
    {
        let assignment = backend
            .graph_assignment_for(model)
            .expect("native graph assignment");
        let expected = std::env::var("FERMENT_EXPECT_EP").ok();
        let mut accelerated = 0;
        for graph in assignment {
            eprintln!("{model:?}: {} {} nodes", graph.provider, graph.nodes.len());
            if graph.provider == "CPUExecutionProvider" {
                assert!(
                    graph.nodes.iter().all(|node| !matches!(
                        node.operator.as_str(),
                        "Conv" | "ConvTranspose" | "MatMul" | "Gemm"
                    )),
                    "weighted operators fell back to CPU: {graph:?}"
                );
            } else {
                if let Some(expected) = &expected {
                    assert_eq!(&graph.provider, expected);
                }
                accelerated += graph.nodes.len();
            }
        }
        assert!(
            accelerated > 0,
            "no nodes executed on the requested accelerator"
        );
    }
    #[cfg(not(windows))]
    {
        let _ = (backend, model);
        panic!("Windows ML qualification requires Windows");
    }
}
