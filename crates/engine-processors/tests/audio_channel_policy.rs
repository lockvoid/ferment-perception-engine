//! Current Sept25 AudioReader/ResampledAudioRequest v3: mono is the mean of
//! every channel in every container. The Sept13 first-channel/equal-power
//! PCM files remain archived fixtures; they are intentionally not this oracle.
use ferment_engine_audio::derived_cache::nonce;
use ferment_engine_processors::{AudioSampleSource, AudioSource};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ios_audio_channel_policy")
        .join(name)
}
fn tone_amplitude(samples: &[f32], hz: f64) -> f64 {
    let skip = 200;
    let count = samples.len() - 2 * skip;
    let (mut real, mut imaginary) = (0., 0.);
    for (index, sample) in samples.iter().enumerate().skip(skip).take(count) {
        let phase = std::f64::consts::TAU * hz * index as f64 / 16000.;
        real += f64::from(*sample) * phase.cos();
        imaginary += f64::from(*sample) * phase.sin();
    }
    2. * real.hypot(imaginary) / count as f64
}
fn wave(channels: u16, interleaved: &[f32]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("channel-mean-{}.wav", nonce()));
    let mut w = hound::WavWriter::create(
        &p,
        hound::WavSpec {
            channels,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for x in interleaved {
        w.write_sample(*x).unwrap();
    }
    w.finalize().unwrap();
    p
}

#[test]
fn stereo_anti_phase_cancels_in_both_flat_and_disk_doors() {
    let p = wave(2, &[1., -1., 0.25, -0.25, -0.75, 0.75]);
    let source = AudioSource::default();
    let plain = futures_executor::block_on(source.resampled(&p, 16000, None, None)).unwrap();
    assert_eq!(plain.samples, [0., 0., 0.]);
    let disk = futures_executor::block_on(source.decode_disk_backed(&p, 16000)).unwrap();
    let mut out = vec![1.; 3];
    disk.copy_samples(&mut out, 0, 3).unwrap();
    assert_eq!(out, [0., 0., 0.]);
    std::fs::remove_file(p).unwrap();
}
#[test]
fn every_channel_contributes_with_reciprocal_channel_count() {
    let p = wave(3, &[0.75, -0.25, 0.25, 0., 0.75, 0., -0.75, 0.75, 0.]);
    let samples = futures_executor::block_on(AudioSource::default().decode(&p))
        .unwrap()
        .samples;
    assert_eq!(samples, [0.25, 0.25, 0.]);
    std::fs::remove_file(p).unwrap();
}
#[test]
fn integer_pcm_is_divided_by_its_signed_power_of_two() {
    let p = std::env::temp_dir().join(format!("pcm-scale-{}.wav", nonce()));
    let mut w = hound::WavWriter::create(
        &p,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for v in [i16::MIN, -16384, 0, 16384, i16::MAX] {
        w.write_sample(v).unwrap();
    }
    w.finalize().unwrap();
    let x = futures_executor::block_on(AudioSource::default().decode(&p)).unwrap();
    assert_eq!(x.samples, [-1., -0.5, 0., 0.5, 32767. / 32768.]);
    std::fs::remove_file(p).unwrap();
}
#[test]
fn audio_and_video_containers_both_use_the_arithmetic_mean() {
    futures_executor::block_on(async {
        for (name, gain) in [("stereo.wav", 1.), ("stereo.mp3", 0.95), ("stereo.mp4", 1.)] {
            let source = AudioSource::default();
            let audio = source
                .resampled(&fixture(name), 16000, None, None)
                .await
                .unwrap();
            assert!((audio.duration_secs() - 1.).abs() < 0.02);
            // The committed independent generator: L=14000*sin(997Hz),
            // R=10000*cos(233Hz), int16. Mean weights are 1/2, including video.
            for (hz, amplitude) in [(997., 14000. / 32768.), (233., 10000. / 32768.)] {
                let want = amplitude * 0.5 * gain;
                let got = tone_amplitude(&audio.samples, hz);
                assert!(
                    (got - want).abs() < 0.015,
                    "{name} {hz}Hz: {got} vs mean {want}"
                );
            }
            let disk = source
                .decode_disk_backed(&fixture(name), 16000)
                .await
                .unwrap();
            let mut samples = vec![0.; disk.sample_count()];
            disk.copy_samples(&mut samples, 0, disk.sample_count())
                .unwrap();
            assert_eq!(samples, audio.samples);
        }
    });
}
