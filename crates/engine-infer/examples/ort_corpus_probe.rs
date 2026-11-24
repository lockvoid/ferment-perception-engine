//! Reuse one native session across an independent corpus. The external runner
//! grades raw owned outputs and actual graph placement, including CPU fallback.
//! Usage: ort_corpus_probe MANIFEST.json
use ferment_engine_infer::{
    DType, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    Tensor, TensorMap,
};
use std::{fs, path::Path, sync::Arc};

fn text<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args().nth(1).ok_or("expected manifest")?;
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(file)?)?;
    let model = ModelId::ALL
        .into_iter()
        .find(|id| Some(id.name()) == manifest["model"].as_str())
        .ok_or("unknown model")?;
    let provider = match text(&manifest, "provider")? {
        "winml" => OrtExecutionProvider::WindowsMl,
        "cpu" => OrtExecutionProvider::Cpu,
        _ => return Err("expected winml or cpu".into()),
    };
    let mut paths = ModelPaths::new();
    paths.insert(model, ModelVersion::Latest, text(&manifest, "path")?);
    let backend = OrtBackend::with_execution_provider(Arc::new(paths), provider);
    let dtype = DType::from_wire(text(&manifest, "dtype")?)?;
    for case in manifest["cases"].as_array().ok_or("missing cases")? {
        let shape = case["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|n| n.as_i64().ok_or("invalid dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        let input = Tensor::new(shape, dtype, fs::read(text(case, "input")?)?.into())?;
        let started = std::time::Instant::now();
        let outputs = futures_executor::block_on(backend.infer(
            model,
            &ModelVersion::Latest,
            TensorMap::from([(text(&manifest, "input_name")?.into(), input)]),
        ))?;
        let out = Path::new(text(case, "output")?);
        fs::create_dir_all(out)?;
        for (name, value) in outputs {
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err("invalid output name".into());
            }
            fs::write(out.join(format!("{name}.bin")), &value.data)?;
            fs::write(
                out.join(format!("{name}.shape")),
                value
                    .shape
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )?;
            fs::write(out.join(format!("{name}.dtype")), value.dtype.as_str())?;
        }
        println!(
            "{model}/{}: {:.3}s",
            text(case, "name")?,
            started.elapsed().as_secs_f64()
        );
    }
    if backend.selected_provider_for(model) != Some(provider) {
        return Err("requested policy was not used".into());
    }
    #[cfg(windows)]
    {
        let graphs = backend
            .graph_assignment_for(model)
            .ok_or("no graph assignment")?;
        let rows: Vec<_> = graphs
            .iter()
            .map(|graph| {
                serde_json::json!({
                    "provider": graph.provider,
                    "nodes": graph.nodes.iter().map(|node| serde_json::json!({
                        "name": node.name, "domain": node.domain, "operator": node.operator
                    })).collect::<Vec<_>>()
                })
            })
            .collect();
        fs::write(
            text(&manifest, "assignment")?,
            serde_json::to_vec_pretty(&rows)?,
        )?;
    }
    Ok(())
}
