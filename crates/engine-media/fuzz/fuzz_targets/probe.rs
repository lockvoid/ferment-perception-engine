#![no_main]

//! Fuzz the container probe.
//!
//! The two demuxers this used to fuzz are gone; `libavformat` reads every
//! container now, and it is fuzzed far harder upstream than we could. What is
//! still ours — and still worth fuzzing — is the mapping on top of it: the
//! `AVFormatContext` walk in `platform::libav::probe`, the codec and colour
//! translations, the timestamp arithmetic, and the `DecoderConfiguration`
//! parsing of whatever bytes a file claims are an avcC record.
//!
//! The property is the same as before: a malformed file is an `Err`, never a
//! panic and never a hang.

use std::fs;

use ferment_engine_media::{MediaTime, demux};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_STREAMS: usize = 32;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "ferment-engine-media-{}-probe-fuzz",
        std::process::id()
    ));
    if fs::write(&path, data).is_err() {
        return;
    }
    if let Ok(info) = demux::probe(&path) {
        // Touch every field a consumer reads, so a value that only explodes on
        // use is caught here rather than in an import.
        let _ = info.duration.map(MediaTime::seconds);
        for stream in info.streams.iter().take(MAX_STREAMS) {
            let _ = stream.start_time.map(MediaTime::seconds);
            let _ = stream.duration.map(MediaTime::seconds);
            let _ = stream.decoder_configuration.aac_profile();
            let _ = stream.decoder_configuration.aac_output_channels();
            let _ = stream.decoder_configuration.aac_effective_sample_rate();
            let _ = stream.audio_timing;
            let _ = stream.display_matrix;
        }
    }
    let _ = fs::remove_file(&path);
});
