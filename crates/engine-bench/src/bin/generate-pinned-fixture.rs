use std::path::PathBuf;

use ferment_engine_bench::stereo_48k;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/engine-bench/pinned-60s-mono-48k.wav"));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create fixture directory");
    }
    let (samples, _) = stereo_48k();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&output, spec).expect("create WAV fixture");
    for sample in samples {
        writer.write_sample(sample).expect("write sample");
    }
    writer.finalize().expect("finalize WAV fixture");
    println!("{}", output.display());
}
