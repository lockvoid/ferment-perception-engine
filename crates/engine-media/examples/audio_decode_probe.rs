//! Decode a file's first audio stream through the libav audio backend and
//! report the resolved contract: profile, rates, continuity, sample totals.
//!
//!     cargo run -p ferment-engine-media --example audio_decode_probe -- <file>

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: audio_decode_probe <file>");
    run(&path);
}

fn run(path: &str) {
    use ferment_engine_media::platform::libav::LibavAudioIo;
    use ferment_engine_media::{AudioIo, MediaKind};
    use futures_executor::block_on;

    let backend = LibavAudioIo;
    let info = block_on(backend.probe(path.as_ref())).expect("probe");
    let stream = info
        .streams
        .iter()
        .find(|stream| stream.kind == MediaKind::Audio)
        .expect("no audio stream");
    println!(
        "stream {}: codec {:?}, declared rate {:?}, channels {:?}, profile {:?}, effective {:?}",
        stream.index,
        stream.codec,
        stream.sample_rate,
        stream.channels,
        stream.decoder_configuration.aac_profile(),
        stream.decoder_configuration.aac_effective_sample_rate(),
    );
    println!("timing {:?}", stream.audio_timing);

    let mut decoder =
        block_on(backend.open_audio_decoder(path.as_ref(), stream.index)).expect("open decoder");
    let mut frames = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
        frames.push(frame);
    }
    assert!(!frames.is_empty(), "no PCM produced");
    let contiguous = frames
        .windows(2)
        .all(|pair| pair[0].start.value + pair[0].frame_count() as i64 == pair[1].start.value);
    let total: usize = frames.iter().map(|frame| frame.frame_count()).sum();
    let rate = frames[0].sample_rate;
    let peak = frames
        .iter()
        .flat_map(|frame| frame.samples.iter())
        .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
    println!(
        "decoded {} frames in {} chunks at {} Hz x{}ch, first start {}, contiguous {}, \
         {:.4}s, peak {:.3}",
        total,
        frames.len(),
        rate,
        frames[0].channels,
        frames[0].start.value,
        contiguous,
        total as f64 / f64::from(rate),
        peak,
    );
}
