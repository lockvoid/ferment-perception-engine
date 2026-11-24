use std::hint::black_box;
use std::time::Instant;

use ferment_engine_audio::{
    MelFilterbank, MelNorm, MelOptions, MelScale, Padding, PortableStftMelFrontend, SpectrumMode,
    StftConfig, StftMelFrontend, resample_sinc, window,
};
use ferment_engine_bench::signal;
use ferment_engine_processors::loudness_normalizer::{LoudnessMeter, TruePeakMeter};

const SANITY_SECONDS: usize = 5;
const ITERATIONS: usize = 5;

fn measure(name: &str, mut workload: impl FnMut() -> f64) {
    black_box(workload());
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERATIONS {
        checksum += workload();
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
    println!("{name}\t{elapsed_ms:.3} ms/iteration\tchecksum={checksum}");
}

fn main() {
    let audio_48k = signal(48_000, SANITY_SECONDS, 0);
    let audio_44100 = signal(44_100, SANITY_SECONDS, 0);
    let config = StftConfig::new(2_048, 512, window::hann_periodic(2_048)).unwrap();
    let filterbank = MelFilterbank::new(MelOptions {
        n_mels: 128,
        n_fft: 2_048,
        sample_rate: 48_000,
        f_min: 20.0,
        f_max: 24_000.0,
        scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
    })
    .unwrap();
    let mut frontend =
        PortableStftMelFrontend::new(config, filterbank, SpectrumMode::Power, 1.0).unwrap();

    measure("stft_mel_5s", || {
        let output = frontend.analyze(&audio_48k, Padding::Reflect).unwrap();
        black_box(&output);
        f64::from(output[output.len() / 2][64])
    });
    measure("resample_5s", || {
        let output = resample_sinc(&audio_44100, 44_100, 48_000).unwrap();
        black_box(&output);
        f64::from(output[output.len() / 2])
    });
    measure("loudness_true_peak_5s", || {
        LoudnessMeter::measure_lufs(&audio_48k, 48_000.0) + TruePeakMeter::measure_dbtp(&audio_48k)
    });
}
