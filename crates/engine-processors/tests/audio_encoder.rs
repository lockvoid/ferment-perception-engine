use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ferment_engine_audio::write_wav_f32;
use ferment_engine_core::{
    AudioEncoding, AudioEncodingPreset, AudioFileType, EngineError, ProgressSink,
};
use ferment_engine_processors::AudioEncoder;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ferment-audio-encoder-{label}-{}-{sequence}",
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

fn tone_wav(path: &Path, sample_rate: u32, seconds: usize) {
    let samples = (0..sample_rate as usize * seconds)
        .map(|index| {
            (std::f32::consts::TAU * 440.0 * index as f32 / sample_rate as f32).sin() * 0.25
        })
        .collect::<Vec<_>>();
    write_wav_f32(path, &samples, sample_rate).unwrap();
}

fn wav_preset(max_duration: f64) -> AudioEncodingPreset {
    AudioEncodingPreset {
        sample_rate: 44_100,
        channels: 2,
        bit_rate: 128_000,
        max_duration,
        file_type: AudioFileType::Wav,
    }
}

#[test]
fn max_duration_clamp() {
    let directory = TestDirectory::new("duration");
    let source = directory.join("source.wav");
    let output = directory.join("clamped.wav");
    tone_wav(&source, 44_100, 3);

    let result = futures_executor::block_on(AudioEncoder::new().encode(
        &source,
        &output,
        wav_preset(1.0),
        ProgressSink::noop(),
    ))
    .unwrap();

    assert!(!result.did_passthrough);
    assert!((result.duration - 1.0).abs() <= 0.05);
    let reader = hound::WavReader::open(output).unwrap();
    assert_eq!(reader.spec().sample_rate, 44_100);
    assert_eq!(reader.spec().channels, 2);
    assert!((reader.duration() as f64 / 44_100.0 - 1.0).abs() <= 0.05);
}

#[test]
fn no_audio_track_throws() {
    let directory = TestDirectory::new("no-audio");
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-media/tests/fixtures/generated/h264-gop4.mp4");
    let output = directory.join("none.wav");

    let error = futures_executor::block_on(AudioEncoder::new().encode(
        &source,
        &output,
        wav_preset(10.0),
        ProgressSink::noop(),
    ))
    .unwrap_err();

    assert!(matches!(error, EngineError::NoAudioData));
    assert!(!output.exists());
}

#[test]
fn cancelling_mid_encode_does_not_deliver_output() {
    let directory = TestDirectory::new("cancel");
    let source = directory.join("source.wav");
    let output = directory.join("original.wav");
    let partial = directory.join("original.wav.partial");
    tone_wav(&source, 44_100, 2);
    let progress_slot = Arc::new(Mutex::new(None::<ProgressSink>));
    let callback_slot = Arc::clone(&progress_slot);
    let progress = ProgressSink::from_callback(move |event| {
        if event.stage == "encoding wav" {
            callback_slot.lock().unwrap().as_ref().unwrap().cancel();
        }
    });
    *progress_slot.lock().unwrap() = Some(progress.clone());

    let error = futures_executor::block_on(AudioEncoder::new().encode(
        &source,
        &output,
        wav_preset(1.0),
        progress,
    ))
    .unwrap_err();

    assert!(matches!(error, EngineError::Cancelled));
    assert!(!output.exists(), "cancelled encode delivered output");
    assert!(!partial.exists(), "cancelled encode left a stale partial");
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod native_aac {
    use std::sync::Arc;

    use ferment_engine_media::{
        AudioBackend, AudioEncoderSettings, Codec, EncodedAudio, MediaTime, PcmAudioFrame, TimeBase,
    };
    use ferment_engine_processors::AudioSource;

    use super::*;

    fn aac_fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../engine-media/tests/fixtures/generated/aac-lc-44100-stereo.m4a")
    }

    /// Both platforms encode and decode AAC with libav's own codecs, which the
    /// pinned build carries everywhere; no system codec is asked for.
    fn platform_backend() -> AudioBackend {
        Arc::new(ferment_engine_media::platform::libav::LibavAudioIo)
    }

    async fn encode_pcm(samples: &[f32], sample_rate: u32) -> EncodedAudio {
        let backend = platform_backend();
        let mut encoder = backend
            .open_audio_encoder(AudioEncoderSettings {
                bit_rate: 128_000,
                sample_rate,
                channels: 2,
            })
            .await
            .unwrap();
        let stereo = samples
            .iter()
            .flat_map(|sample| [*sample, *sample])
            .collect::<Vec<_>>();
        encoder
            .push_pcm(
                PcmAudioFrame::new(
                    sample_rate,
                    2,
                    MediaTime::new(0, TimeBase::new_unchecked(1, i64::from(sample_rate))),
                    stereo,
                )
                .unwrap(),
            )
            .await
            .unwrap();
        encoder.finalize().await.unwrap()
    }

    fn argmax_abs(samples: &[f32]) -> usize {
        samples
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map_or(0, |(index, _)| index)
    }

    fn assert_is_aac(path: &Path, sample_rate: u32, channels: u16) {
        let info = ferment_engine_media::demux::probe(path).unwrap();
        let stream = info
            .audio_streams()
            .next()
            .expect("output has no audio track");
        assert_eq!(stream.codec, Codec::Aac);
        assert_eq!(stream.sample_rate, Some(sample_rate));
        assert_eq!(stream.channels, Some(channels));
    }

    #[test]
    #[ignore = "blocked: engine-media has AAC packets but no M4A mux/finalize-to-path API"]
    fn encode_wav_produces_valid_aac() {
        let directory = TestDirectory::new("wav-to-aac");
        let source = directory.join("source.wav");
        let output = directory.join("original.m4a");
        tone_wav(&source, 44_100, 1);

        let result = futures_executor::block_on(AudioEncoder::new().encode(
            &source,
            &output,
            AudioEncodingPreset::default(),
            ProgressSink::noop(),
        ))
        .unwrap();

        assert!(output.exists());
        assert_eq!(result.output_path, output);
        assert!(!result.did_passthrough);
        assert!(result.bytes > 1_024);
        assert!(result.duration > 0.0);
        assert_is_aac(&result.output_path, 44_100, 2);
    }

    #[test]
    fn passthrough_when_already_target_aac() {
        let directory = TestDirectory::new("aac-passthrough");
        let source = aac_fixture();
        let output = directory.join("passed.m4a");

        let result = futures_executor::block_on(AudioEncoder::new().encode(
            &source,
            &output,
            AudioEncodingPreset::default(),
            ProgressSink::noop(),
        ))
        .unwrap();

        assert!(result.did_passthrough);
        assert!(output.exists());
        assert!(result.bytes > 1_024);
        assert_eq!(
            std::fs::read(source).unwrap(),
            std::fs::read(output).unwrap()
        );
    }

    #[test]
    fn encode_aac_and_wav_source_reencoded_impulse_aligns() {
        futures_executor::block_on(async {
            let directory = TestDirectory::new("priming");
            let aac_source = aac_fixture();
            let wav_source = directory.join("source.wav");
            let source = AudioSource::platform_default()
                .decode(&aac_source)
                .await
                .unwrap();
            write_wav_f32(&wav_source, &source.samples, source.sample_rate).unwrap();
            let wav = AudioSource::platform_default()
                .decode(&wav_source)
                .await
                .unwrap();

            let wav_encoded = encode_pcm(&wav.samples, wav.sample_rate).await;
            let aac_encoded = encode_pcm(&source.samples, source.sample_rate).await;

            assert!(!wav.samples.is_empty());
            assert!(!source.samples.is_empty());
            assert!(argmax_abs(&source.samples).abs_diff(argmax_abs(&wav.samples)) <= 48);
            assert_eq!(aac_encoded.timing, wav_encoded.timing);
            assert_eq!(
                aac_encoded.timing.audible_samples,
                Some(source.samples.len() as u64)
            );
            assert!(aac_encoded.timing.leading_samples > 0);
            assert_eq!(aac_encoded.configuration.codec, Codec::Aac);
            assert_eq!(wav_encoded.configuration.codec, Codec::Aac);
            assert!(!aac_encoded.packets.is_empty());
            assert!(!wav_encoded.packets.is_empty());
        });
    }
}
