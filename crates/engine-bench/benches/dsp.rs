use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ferment_engine_audio::{
    AudioFrames, MelFilterbank, MelNorm, MelOptions, MelScale, Padding, PortableStftMelFrontend,
    SpectrumMode, StftConfig, StftMelFrontend, resample_fft, resample_sinc, stft_centered_with,
    window,
};
use ferment_engine_bench::{PINNED_SECONDS, mono_22050, mono_44100, stereo_48k};
use ferment_engine_core::{LoudnessTarget, ProgressSink};
use ferment_engine_processors::audio_minimap_composer::{mel as minimap_mel, rms as minimap_rms};
use ferment_engine_processors::beat_queen_mary::{
    ANALYSIS_SAMPLE_RATE, BeatQueenMary, downbeat,
    onset::BeatQueenMaryOnset,
    tempo::{self, TempoConfig},
};
use ferment_engine_processors::loudness_normalizer::{
    LoudnessKWeighting, LoudnessMeter, LoudnessNormalizer, TruePeakMeter,
};
use ferment_engine_processors::waveform_extractor::WaveformExtractor;
use rustfft::num_complex::Complex;

const SAMPLE_RATE: u32 = 48_000;
const MEL_FFT_SIZE: usize = 2_048;
const MEL_HOP_SIZE: usize = 512;
const MEL_BANDS: usize = 128;

fn stft_config(fft_size: usize) -> StftConfig {
    StftConfig::new(fft_size, fft_size / 4, window::hann_periodic(fft_size)).unwrap()
}

fn frontend_config() -> (StftConfig, MelFilterbank) {
    let config = StftConfig::new(
        MEL_FFT_SIZE,
        MEL_HOP_SIZE,
        window::hann_periodic(MEL_FFT_SIZE),
    )
    .unwrap();
    let filterbank = MelFilterbank::new(MelOptions {
        n_mels: MEL_BANDS,
        n_fft: MEL_FFT_SIZE,
        sample_rate: SAMPLE_RATE,
        f_min: 20.0,
        f_max: SAMPLE_RATE as f32 / 2.0,
        scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
    })
    .unwrap();
    (config, filterbank)
}

fn power_spectra(spectra: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
    spectra
        .iter()
        .map(|frame| frame.iter().map(Complex::norm_sqr).collect())
        .collect()
}

fn mel_projection(filterbank: &MelFilterbank, power: &[Vec<f32>]) -> Vec<Vec<f32>> {
    power
        .iter()
        .map(|frame| filterbank.apply(frame).unwrap())
        .collect()
}

fn split_complex_roundtrip(spectra: &[Vec<Complex<f32>>]) -> Vec<Vec<Complex<f32>>> {
    spectra
        .iter()
        .map(|frame| {
            let mut real = Vec::with_capacity(frame.len());
            let mut imaginary = Vec::with_capacity(frame.len());
            for value in frame {
                real.push(value.re);
                imaginary.push(value.im);
            }
            real.into_iter()
                .zip(imaginary)
                .map(|(re, im)| Complex::new(re, im))
                .collect()
        })
        .collect()
}

fn bench_stft(c: &mut Criterion) {
    let (left, right) = stereo_48k();
    let mut group = c.benchmark_group("stft_60s_stereo_48k");
    group.throughput(Throughput::Elements((left.len() + right.len()) as u64));
    for fft_size in [1_024, 2_048, 4_096] {
        let config = stft_config(fft_size);
        group.bench_with_input(
            BenchmarkId::from_parameter(fft_size),
            &config,
            |b, config| {
                b.iter(|| {
                    let left_frames =
                        stft_centered_with(black_box(&left), config, Padding::Reflect).unwrap();
                    let right_frames =
                        stft_centered_with(black_box(&right), config, Padding::Reflect).unwrap();
                    black_box((left_frames, right_frames));
                });
            },
        );
    }
    group.finish();
}

fn bench_stft_mel_frontend(c: &mut Criterion) {
    let (left, _) = stereo_48k();
    let (config, filterbank) = frontend_config();
    let spectra = stft_centered_with(&left, &config, Padding::Reflect).unwrap();
    let power = power_spectra(&spectra);
    let mut frontend =
        PortableStftMelFrontend::new(config.clone(), filterbank.clone(), SpectrumMode::Power, 1.0)
            .unwrap();

    let mut group = c.benchmark_group("stft_frontend_end_to_end_60s_mono_48k_2048_mel128");
    group.throughput(Throughput::Elements(left.len() as u64));
    group.bench_function("portable", |b| {
        b.iter(|| {
            black_box(
                frontend
                    .analyze(black_box(&left), Padding::Reflect)
                    .unwrap(),
            )
        });
    });
    group.bench_function("stft_only", |b| {
        b.iter(|| {
            black_box(stft_centered_with(black_box(&left), &config, Padding::Reflect).unwrap())
        });
    });
    group.bench_function("magnitude_power_only", |b| {
        b.iter(|| black_box(power_spectra(black_box(&spectra))));
    });
    group.bench_function("mel_projection_only", |b| {
        b.iter(|| black_box(mel_projection(&filterbank, black_box(&power))));
    });
    group.bench_function("split_complex_roundtrip_only", |b| {
        b.iter(|| black_box(split_complex_roundtrip(black_box(&spectra))));
    });
    #[cfg(all(target_os = "macos", feature = "accel-fastpath"))]
    {
        let mut bridge =
            ferment_engine_bench::accelerate::SplitComplexRoundtrip::new(MEL_FFT_SIZE / 2 + 1);
        group.bench_function("accelerate_ctoz_ztoc_roundtrip_only", |b| {
            b.iter(|| black_box(bridge.run(black_box(&spectra))));
        });
    }
    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let input = mono_44100();
    let mut group = c.benchmark_group("resample_60s_mono_44100_to_48000");
    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_function("rubato_sinc", |b| {
        b.iter(|| black_box(resample_sinc(black_box(&input), 44_100, 48_000).unwrap()));
    });
    group.finish();
}

fn bench_biquad(c: &mut Criterion) {
    let (left, right) = stereo_48k();
    let mut group = c.benchmark_group("biquad_k_weighting_60s_stereo_48k");
    group.throughput(Throughput::Elements((left.len() + right.len()) as u64));
    group.bench_function("two_stage", |b| {
        b.iter_batched(
            || (left.clone(), right.clone()),
            |(mut left, mut right)| {
                let mut left_filter = LoudnessKWeighting::new(SAMPLE_RATE.into());
                let mut right_filter = LoudnessKWeighting::new(SAMPLE_RATE.into());
                left_filter.process(&mut left);
                right_filter.process(&mut right);
                black_box((left, right));
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_loudness(c: &mut Criterion) {
    let (left, right) = stereo_48k();
    let mut group = c.benchmark_group("ebu_r128_true_peak_60s_stereo_48k");
    group.throughput(Throughput::Elements((left.len() + right.len()) as u64));
    group.bench_function("loudness", |b| {
        b.iter(|| {
            black_box((
                LoudnessMeter::measure_lufs(black_box(&left), SAMPLE_RATE.into()),
                LoudnessMeter::measure_lufs(black_box(&right), SAMPLE_RATE.into()),
            ))
        });
    });
    group.bench_function("true_peak", |b| {
        b.iter(|| {
            black_box((
                TruePeakMeter::measure_dbtp(black_box(&left)),
                TruePeakMeter::measure_dbtp(black_box(&right)),
            ))
        });
    });
    group.bench_function("combined", |b| {
        b.iter(|| {
            black_box((
                LoudnessMeter::measure_lufs(black_box(&left), SAMPLE_RATE.into()),
                LoudnessMeter::measure_lufs(black_box(&right), SAMPLE_RATE.into()),
                TruePeakMeter::measure_dbtp(black_box(&left)),
                TruePeakMeter::measure_dbtp(black_box(&right)),
            ))
        });
    });
    group.finish();
}

fn bench_waveform_and_minimap(c: &mut Criterion) {
    let (waveform, _) = stereo_48k();
    let minimap = mono_22050();
    let mut group = c.benchmark_group("waveform_minimap_60s");
    group.bench_function("waveform_pipeline_48k", |b| {
        b.iter_batched(
            || waveform.clone(),
            |samples| {
                black_box(
                    WaveformExtractor
                        .extract_frames(AudioFrames::new(samples, SAMPLE_RATE), 0.025)
                        .unwrap(),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function("waveform_resample_48k_to_22050", |b| {
        b.iter(|| black_box(resample_sinc(black_box(&waveform), 48_000, 22_050).unwrap()));
    });
    group.bench_function("waveform_decimation_only_22050", |b| {
        b.iter_batched(
            || minimap.clone(),
            |samples| {
                black_box(
                    WaveformExtractor
                        .extract_frames(AudioFrames::new(samples, 22_050), 0.025)
                        .unwrap(),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function("minimap_mel128", |b| {
        b.iter(|| black_box(minimap_mel::compute_mel_db(black_box(&minimap)).unwrap()));
    });
    group.bench_function("minimap_rms", |b| {
        b.iter(|| black_box(minimap_rms::compute(black_box(&minimap))));
    });
    group.bench_function("minimap_mel128_and_rms", |b| {
        b.iter(|| {
            black_box((
                minimap_mel::compute_mel_db(black_box(&minimap)).unwrap(),
                minimap_rms::compute(black_box(&minimap)),
            ))
        });
    });
    group.finish();
}

fn bench_swift_reference_pipelines(c: &mut Criterion) {
    let (input, _) = stereo_48k();
    let mut group = c.benchmark_group("swift_cli_reference_60s_mono_48k");
    group.bench_function("beat_queen_mary", |b| {
        b.iter_batched(
            || input.clone(),
            |samples| {
                black_box(
                    futures_executor::block_on(
                        BeatQueenMary::new()
                            .detect(AudioFrames::new(samples, SAMPLE_RATE), ProgressSink::noop()),
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function("loudness_normalizer", |b| {
        b.iter_batched(
            || input.clone(),
            |samples| {
                black_box(
                    futures_executor::block_on(LoudnessNormalizer.normalize_frames(
                        AudioFrames::new(samples, SAMPLE_RATE),
                        LoudnessTarget {
                            target_lufs: -18.0,
                            peak_ceiling_dbtp: -1.0,
                            enable_leveler: false,
                        },
                        ProgressSink::noop(),
                    ))
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_beat_queen_mary_components(c: &mut Criterion) {
    let (input, _) = stereo_48k();
    let resampled = resample_sinc(&input, SAMPLE_RATE, ANALYSIS_SAMPLE_RATE).unwrap();
    let onset = BeatQueenMaryOnset::for_sample_rate(f64::from(ANALYSIS_SAMPLE_RATE));
    let envelope = onset.compute(&resampled);
    let envelope_rate = f64::from(ANALYSIS_SAMPLE_RATE) / onset.step_size as f64;
    let tempo = tempo::track(&envelope, envelope_rate, &TempoConfig::default());
    let beat_frames = tempo
        .beat_frames
        .iter()
        .map(|frame| {
            (frame * onset.step_size as f64)
                .round()
                .clamp(0.0, resampled.len().saturating_sub(1) as f64) as usize
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("beat_queen_mary_components_60s_mono");
    group.bench_function("resample_sinc_48k_to_22050", |b| {
        b.iter(|| {
            black_box(resample_sinc(black_box(&input), SAMPLE_RATE, ANALYSIS_SAMPLE_RATE).unwrap())
        });
    });
    group.bench_function("resample_fft_48k_to_22050", |b| {
        b.iter(|| {
            black_box(resample_fft(black_box(&input), SAMPLE_RATE, ANALYSIS_SAMPLE_RATE).unwrap())
        });
    });
    group.bench_function("onset", |b| {
        b.iter(|| black_box(onset.compute(black_box(&resampled))));
    });
    group.bench_function("tempo", |b| {
        b.iter(|| {
            black_box(tempo::track(
                black_box(&envelope),
                envelope_rate,
                &TempoConfig::default(),
            ))
        });
    });
    group.bench_function("downbeat", |b| {
        b.iter(|| {
            black_box(downbeat::detect(
                black_box(&resampled),
                f64::from(ANALYSIS_SAMPLE_RATE),
                &beat_frames,
                None,
            ))
        });
    });
    group.finish();
}

fn criterion_config() -> Criterion {
    assert_eq!(PINNED_SECONDS, 60);
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(12))
        .noise_threshold(0.03)
}

criterion_group! {
    name = dsp;
    config = criterion_config();
    targets = bench_stft,
        bench_stft_mel_frontend,
        bench_resample,
        bench_biquad,
        bench_loudness,
        bench_waveform_and_minimap,
        bench_swift_reference_pipelines,
        bench_beat_queen_mary_components
}
criterion_main!(dsp);
