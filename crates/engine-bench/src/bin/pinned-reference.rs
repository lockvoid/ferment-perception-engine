use ferment_engine_audio::AudioFrames;
use ferment_engine_bench::stereo_48k;
use ferment_engine_core::ProgressSink;
use ferment_engine_processors::beat_queen_mary::BeatQueenMary;

fn main() {
    let (samples, _) = stereo_48k();
    let result = futures_executor::block_on(
        BeatQueenMary::new().detect(AudioFrames::new(samples, 48_000), ProgressSink::noop()),
    )
    .expect("run pinned BeatQueenMary workload");
    println!(
        "bpm={:?} beats={} downbeats={} quantized_beats={} quantized_downbeats={}",
        result.bpm,
        result.beats.len(),
        result.downbeats.len(),
        result.quantized_beats.len(),
        result.quantized_downbeats.len()
    );
}
