//! Dump the actual processor audio path for a cross-runtime PCM comparison.
//! Usage: decode_audio INPUT OUTPUT SAMPLE_RATE [PREFIX_SAMPLES]

use std::io::Write;
use std::path::Path;

use ferment_engine_processors::audio_source::AudioSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("decode_audio INPUT OUTPUT SAMPLE_RATE [PREFIX_SAMPLES]".into());
    }
    let rate: u32 = args[3].parse()?;
    let audio = futures_executor::block_on(AudioSource::default().decode(Path::new(&args[1])))?;
    let source_samples = audio.samples.len();
    let samples = ferment_engine_audio::resample_sinc(&audio.samples, audio.sample_rate, rate)?;
    let output_samples = samples.len();
    let prefix = if let Some(value) = args.get(4) {
        value.parse::<usize>()?.min(output_samples)
    } else {
        output_samples
    };
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    for value in samples.iter().take(prefix) {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    println!(
        "{}",
        serde_json::json!({"source_rate": audio.sample_rate, "source_samples": source_samples,
        "output_rate": rate, "output_samples": output_samples, "written_samples": prefix})
    );
    Ok(())
}
