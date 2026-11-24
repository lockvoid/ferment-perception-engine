//! Run a real model through the public Rust facade and dump owned output bytes.
//! Compare these with independent upstream fixtures, outside this program.
//! Usage: model_probe MODEL INPUT_NAME f32|f16|i32|i64 SHAPE_CSV INPUT OUTPUT_DIR [FUNCTION]
use coreai_native::{DType, Model, Tensor};
use std::{env, fs, io, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if !(7..=8).contains(&args.len()) {
        return Err("expected MODEL INPUT_NAME DTYPE SHAPE_CSV INPUT OUTPUT_DIR [FUNCTION]".into());
    }
    let dtype = match args[3].as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        _ => return Err("unknown dtype".into()),
    };
    let input = Tensor {
        name: args[2].clone(),
        shape: args[4]
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        dtype,
        data: fs::read(&args[5])?,
    };
    let started = Instant::now();
    let outputs = {
        let model = Model::load(Path::new(&args[1])).map_err(io::Error::other)?;
        let outputs = model
            .predict(args.get(7).map_or("main", String::as_str), &[input])
            .map_err(io::Error::other)?;
        // Model and input die here. Later reads must still see valid owned bytes.
        outputs
    };
    fs::create_dir_all(&args[6])?;
    for output in outputs {
        if output.name.is_empty()
            || !output
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err("unexpected output name".into());
        }
        let root = Path::new(&args[6]);
        fs::write(root.join(format!("{}.bin", output.name)), &output.data)?;
        fs::write(
            root.join(format!("{}.shape", output.name)),
            output
                .shape
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )?;
        println!(
            "{} {:?} {:?}: {} bytes",
            output.name,
            output.dtype,
            output.shape,
            output.data.len()
        );
    }
    println!(
        "architecture={:?}, elapsed={:?}",
        coreai_native::device_architecture(),
        started.elapsed()
    );
    Ok(())
}
