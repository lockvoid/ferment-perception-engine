use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_audio::{resample_linear, write_wav_f32};
use ferment_engine_processors::AudioSource;
use ferment_engine_processors::beatthis::{mel::compute_log_mel, postprocess::process};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ferment-beat-pipeline-{label}-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn mel_spectrogram_output() {
    let sample_rate = 22_050usize;
    let audio = (0..sample_rate)
        .map(|index| (std::f32::consts::TAU * 440.0 * index as f32 / sample_rate as f32).sin())
        .collect::<Vec<_>>();
    let result = compute_log_mel(&audio).unwrap();

    assert!(result.len() > 40);
    assert!(result.len() < 60);
    assert_eq!(result[0].len(), 128);
    for frame in result {
        for value in frame {
            assert!(value >= 0.0);
        }
    }
}

fn probability_to_logit(probability: f32) -> f32 {
    (probability / (1.0 - probability)).ln()
}

#[test]
fn dbn_beat_tracker_with_synthetic_input() {
    let num_frames = 200usize;
    let mut beat_probability = vec![0.1f32; num_frames];
    let mut downbeat_probability = vec![0.05f32; num_frames];
    for index in (0..num_frames).step_by(25) {
        beat_probability[index] = 0.9;
        if index % 100 == 0 {
            downbeat_probability[index] = 0.8;
        }
    }
    let beat_logits = beat_probability
        .into_iter()
        .map(probability_to_logit)
        .collect::<Vec<_>>();
    let downbeat_logits = downbeat_probability
        .into_iter()
        .map(probability_to_logit)
        .collect::<Vec<_>>();
    let result = process(&beat_logits, &downbeat_logits, 50.0, Some(4.0));

    assert!(!result.beats.is_empty(), "should detect beats");
    assert!(!result.downbeats.is_empty(), "should detect downbeats");
    let duration = num_frames as f64 / 50.0;
    for beat in result.beats {
        assert!(beat >= 0.0);
        assert!(beat <= duration);
    }
}

#[test]
fn load_audio_file() {
    let directory = TestDirectory::new("load");
    let fixture = directory.join("phonk-1-lossless-equivalent.wav");
    let sample_rate = 44_100u32;
    let samples = (0..sample_rate)
        .map(|index| {
            (std::f32::consts::TAU * 110.0 * index as f32 / sample_rate as f32).sin() * 0.25
        })
        .collect::<Vec<_>>();
    write_wav_f32(&fixture, &samples, sample_rate).unwrap();
    eprintln!(
        "fixture substitution: phonk-1.mp3 is blocked by the shipped compressed-decoder policy; exercising the same load contract with generated lossless PCM WAV"
    );

    let loaded = futures_executor::block_on(AudioSource::platform_default().decode(&fixture))
        .expect("lossless audio fixture should load");
    assert!(!loaded.samples.is_empty(), "should have audio samples");
    assert!(loaded.sample_rate > 0, "should have a valid sample rate");
}

#[test]
fn resample_audio() {
    let input = vec![1.0f32; 44_100];
    let output = resample_linear(&input, 44_100, 22_050);
    assert_eq!(output.len(), 22_050);
}
