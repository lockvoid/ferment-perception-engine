//! Real ORT backend probe; an external runner compares owned output bytes to
//! independent upstream goldens. A requested provider may never fall back.
//! Usage: ort_model_probe PROVIDER MODEL_ID MODEL INPUT_NAME DTYPE SHAPE INPUT OUT
use ferment_engine_infer::{
    DType, InferenceBackend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    Tensor, TensorMap,
};
use std::{env, fs, path::Path, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 9 {
        return Err("expected PROVIDER MODEL_ID MODEL INPUT_NAME DTYPE SHAPE INPUT OUT".into());
    }
    let provider = match args[1].as_str() {
        "cpu" => OrtExecutionProvider::Cpu,
        "dml" => OrtExecutionProvider::DirectMl,
        "cuda" => OrtExecutionProvider::Cuda,
        "winml" => OrtExecutionProvider::WindowsMl,
        _ => return Err("provider must be cpu, dml, cuda or winml".into()),
    };
    let model = ModelId::ALL
        .into_iter()
        .find(|id| id.name() == args[2])
        .ok_or("unknown model ID")?;
    let dtype = DType::from_wire(&args[5])?;
    let shape = args[6]
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let input = Tensor::new(shape, dtype, fs::read(&args[7])?.into())?;
    let outputs = {
        let mut paths = ModelPaths::new();
        paths.insert(model, ModelVersion::Latest, args[3].clone());
        let backend = OrtBackend::with_execution_provider(Arc::new(paths), provider);
        let inputs = TensorMap::from([(args[4].clone(), input)]);
        let outputs =
            futures_executor::block_on(backend.infer(model, &ModelVersion::Latest, inputs))?;
        if backend.selected_provider_for(model) != Some(provider) {
            return Err("requested provider was not used".into());
        }
        #[cfg(windows)]
        {
            let assignment = backend
                .graph_assignment_for(model)
                .ok_or("no graph assignment")?;
            let graphs: Vec<_> = assignment
                .iter()
                .map(|graph| {
                    serde_json::json!({
                        "provider": graph.provider,
                        "nodes": graph.nodes.iter().map(|node| serde_json::json!({
                            "name": node.name, "domain": node.domain, "operator": node.operator,
                        })).collect::<Vec<_>>()
                    })
                })
                .collect();
            fs::create_dir_all(&args[8])?;
            fs::write(
                Path::new(&args[8]).join("assignment.json"),
                serde_json::to_vec_pretty(&graphs)?,
            )?;
        }
        outputs
    }; // Native session/input die before owned output bytes are read.
    let root = Path::new(&args[8]);
    fs::create_dir_all(root)?;
    for (name, output) in outputs {
        if name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            return Err("unexpected output name".into());
        }
        fs::write(root.join(format!("{name}.bin")), &output.data)?;
        fs::write(
            root.join(format!("{name}.shape")),
            output
                .shape
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )?;
        fs::write(root.join(format!("{name}.dtype")), output.dtype.as_str())?;
        println!(
            "{provider:?} {model}: {name} {:?} {:?}",
            output.dtype, output.shape
        );
    }
    Ok(())
}
