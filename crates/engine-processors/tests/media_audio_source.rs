#![cfg(any(target_os = "macos", target_os = "windows"))]

use std::f32::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use ferment_engine_audio::{resample_sinc, write_wav_f32};
use ferment_engine_media::{AudioBackend, MediaKind};
use ferment_engine_processors::{AudioSampleSource, AudioSource, WaveformExtractor};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-media/tests/fixtures/generated")
        .join(name)
}

fn common_max_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn common_mean_difference(left: &[f32], right: &[f32]) -> f32 {
    let count = left.len().min(right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / count.max(1) as f32
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn impulse_pair(name: &str) -> Option<(PathBuf, PathBuf)> {
    let wav_path = std::env::temp_dir().join(format!("ferment-{name}-{}.wav", std::process::id()));
    let aac_path = wav_path.with_extension("m4a");
    let mut samples = vec![0.0; 48_000];
    samples[24_000] = 1.0;
    write_wav_f32(&wav_path, &samples, 48_000).unwrap();
    let encode = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&wav_path)
        .args(["-c:a", "aac", "-profile:a", "aac_low", "-b:a", "96k"])
        .arg(&aac_path)
        .status();
    let status = match encode {
        Ok(status) => status,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::remove_file(&wav_path).unwrap();
            return None;
        }
        Err(error) => panic!("failed to launch ffmpeg: {error}"),
    };
    assert!(status.success(), "ffmpeg failed to encode AAC oracle");
    Some((wav_path, aac_path))
}

// WaveformExtractorTests.swift

#[test]
fn test_extract_aac_and_lpcm_peak_in_same_window() {
    futures_executor::block_on(async {
        let source = AudioSource::platform_default();
        let extractor = WaveformExtractor;
        let Some((wav_path, aac_path)) = impulse_pair("waveform-priming-oracle") else {
            eprintln!(
                "SKIP test_extract_aac_and_lpcm_peak_in_same_window: ffmpeg is required to create the dev-only AAC oracle"
            );
            return;
        };

        let aac = source.decode(&aac_path).await.unwrap();
        let wav = source.decode(&wav_path).await.unwrap();
        fs::remove_file(&wav_path).unwrap();
        fs::remove_file(&aac_path).unwrap();
        let aac_waveform = extractor.extract_frames(aac, 0.025).unwrap().rms;
        let wav_waveform = extractor.extract_frames(wav, 0.025).unwrap().rms;

        assert!(
            argmax(&aac_waveform).abs_diff(argmax(&wav_waveform)) <= 1,
            "priming-trimmed AAC energy must land within one waveform window of LPCM"
        );
    });
}

// AudioDecodeAlignmentTests.swift

#[test]
fn test_load_resampled_audio_aac_impulse_aligns_with_lpcm_ground_truth() {
    futures_executor::block_on(async {
        let Some((wav_path, aac_path)) = impulse_pair("resample-alignment-oracle") else {
            eprintln!(
                "SKIP test_load_resampled_audio_aac_impulse_aligns_with_lpcm_ground_truth: ffmpeg is required to create the dev-only AAC oracle"
            );
            return;
        };
        let source = AudioSource::platform_default();
        let wav = source.decode(&wav_path).await.unwrap();
        let aac = source.decode(&aac_path).await.unwrap();
        fs::remove_file(&wav_path).unwrap();
        fs::remove_file(&aac_path).unwrap();
        let wav = resample_sinc(&wav.samples, wav.sample_rate, 22_050).unwrap();
        let aac = resample_sinc(&aac.samples, aac.sample_rate, 22_050).unwrap();
        assert!(
            argmax(&aac).abs_diff(argmax(&wav)) <= 16,
            "AAC impulse must align with LPCM after gapless decode + resample"
        );
    });
}

#[test]
fn test_load_resampled_audio_aac_decodes_valid_length() {
    futures_executor::block_on(async {
        let Some((wav_path, aac_path)) = impulse_pair("valid-length-oracle") else {
            eprintln!(
                "SKIP test_load_resampled_audio_aac_decodes_valid_length: ffmpeg is required to create the dev-only AAC oracle"
            );
            return;
        };
        let decoded = AudioSource::platform_default()
            .decode(&aac_path)
            .await
            .unwrap();
        fs::remove_file(&wav_path).unwrap();
        fs::remove_file(&aac_path).unwrap();
        assert!(
            decoded.samples.len().abs_diff(48_000) <= 128,
            "AAC valid length is {}, expected 48,000 ±128",
            decoded.samples.len()
        );
    });
}

#[test]
fn test_load_audio_file_video_container_decodes_valid_length() {
    futures_executor::block_on(async {
        let decoded = AudioSource::platform_default()
            .decode(&fixture("synthetic-multitrack.mp4"))
            .await
            .unwrap();
        assert!(
            decoded.samples.len().abs_diff(decoded.sample_rate as usize) <= 4096,
            "video-container audio decoded {} frames at {} Hz, expected about one second",
            decoded.samples.len(),
            decoded.sample_rate
        );
    });
}

fn read_all(source: &impl ferment_engine_processors::AudioSampleSource) -> Vec<f32> {
    let mut samples = vec![0.0; source.sample_count()];
    let count = samples.len();
    source.copy_samples(&mut samples, 0, count).unwrap();
    samples
}

// AudioDecoderStreamingTests.swift

#[test]
fn test_make_disk_backed_source_aac_impulse_aligns_with_in_memory() {
    futures_executor::block_on(async {
        let Some((wav_path, aac_path)) = impulse_pair("disk-aac-alignment-oracle") else {
            eprintln!(
                "SKIP test_make_disk_backed_source_aac_impulse_aligns_with_in_memory: ffmpeg is required to create the dev-only AAC oracle"
            );
            return;
        };
        let decoder = AudioSource::platform_default();
        let in_memory = decoder.decode(&aac_path).await.unwrap();
        let disk = decoder.decode_disk_backed(&aac_path, 48_000).await.unwrap();
        let disk_samples = read_all(&disk);
        disk.cleanup().unwrap();
        fs::remove_file(&wav_path).unwrap();
        fs::remove_file(&aac_path).unwrap();
        assert!(disk_samples.len().abs_diff(in_memory.samples.len()) <= 64);
        assert!(argmax(&disk_samples).abs_diff(argmax(&in_memory.samples)) <= 8);
    });
}

#[test]
fn test_make_disk_backed_source_wav_is_untouched() {
    futures_executor::block_on(async {
        let Some((wav_path, aac_path)) = impulse_pair("disk-wav-oracle") else {
            eprintln!(
                "SKIP test_make_disk_backed_source_wav_is_untouched: ffmpeg is required to create the paired dev oracle"
            );
            return;
        };
        let disk = AudioSource::platform_default()
            .decode_disk_backed(&wav_path, 48_000)
            .await
            .unwrap();
        assert!(disk.sample_count().abs_diff(48_000) <= 64);
        let peak = argmax(&read_all(&disk));
        assert!((24_000..24_128).contains(&peak));
        disk.cleanup().unwrap();
        fs::remove_file(&aac_path).unwrap();
    });
}

#[test]
fn waveform_matches_wav_for_aac_in_m4a_mp4_mov_and_mkv() {
    futures_executor::block_on(async {
        let source = AudioSource::platform_default();
        let extractor = WaveformExtractor;

        // Decode first, then compare over the SHORTEST common length.
        //
        // The four files hold the same one-second tone, but they do not all
        // declare the encoder's trailing padding: MP4 carries it in the edit
        // list, and this Matroska fixture carries no `discard_padding` at all
        // (`ffprobe -show_packets` shows one leading skip of 1024 and nothing
        // else). So Matroska decodes 128 samples longer, correctly — there is
        // nothing in it that says those samples are padding. The envelope
        // extractor normalizes, so an unequal length would change every value
        // and turn a 2.7 ms declaration difference into a total mismatch.
        // Trimming to the common length compares what this test is about:
        // that the four containers hold the same audio.
        let mut decoded_pcm = Vec::new();
        for file_name in [
            "aac-lc-48000-mono.m4a",
            "aac-lc-48000-mono.mp4",
            "aac-lc-48000-mono.mov",
            "aac-lc-48000-mono.mkv",
        ] {
            decoded_pcm.push((file_name, source.decode(&fixture(file_name)).await.unwrap()));
        }
        let common = decoded_pcm
            .iter()
            .map(|(_, frames)| frames.samples.len())
            .min()
            .expect("four decodes");
        let mut container_waveforms = Vec::new();
        for (file_name, frames) in decoded_pcm {
            let trimmed = ferment_engine_audio::AudioFrames::new(
                frames.samples[..common].to_vec(),
                frames.sample_rate,
            );
            container_waveforms.push((
                file_name,
                extractor.extract_frames(trimmed, 0.01).unwrap().rms,
            ));
        }
        for (name, waveform) in &container_waveforms[1..] {
            assert!(
                container_waveforms[0].1.len().abs_diff(waveform.len()) <= 3,
                "{} and {name} differ by more than three 10 ms windows: {} vs {}",
                container_waveforms[0].0,
                container_waveforms[0].1.len(),
                waveform.len()
            );
            let difference = common_max_difference(&container_waveforms[0].1, waveform);
            assert!(
                difference <= 0.08,
                "{} and {name} waveform envelopes differ by {difference}, above 0.08",
                container_waveforms[0].0
            );
        }

        let wav_path = std::env::temp_dir().join(format!(
            "ferment-audio-source-oracle-{}.wav",
            std::process::id()
        ));
        let samples = (0..48_000)
            .map(|index| {
                let time = index as f32 / 48_000.0;
                0.35 * (TAU * 997.0 * time).sin() + 0.1 * (TAU * 233.0 * time).cos()
            })
            .collect::<Vec<_>>();
        write_wav_f32(&wav_path, &samples, 48_000).unwrap();
        let wav = source.decode(&wav_path).await.unwrap();
        fs::remove_file(&wav_path).unwrap();
        let wav_waveform = extractor.extract_frames(wav, 0.01).unwrap().rms;
        let aac_waveform = &container_waveforms[0].1;
        assert!(
            aac_waveform.len().abs_diff(wav_waveform.len()) <= 3,
            "AAC/WAV waveform lengths differ by more than three 10 ms windows: {} vs {}",
            aac_waveform.len(),
            wav_waveform.len()
        );
        let aac_wav_difference = common_max_difference(aac_waveform, &wav_waveform);
        let aac_wav_mean_difference = common_mean_difference(aac_waveform, &wav_waveform);
        assert!(
            aac_wav_mean_difference <= 0.11 && aac_wav_difference <= 0.75,
            "AAC/WAV waveform envelopes differ by mean {aac_wav_mean_difference} / max {aac_wav_difference}, above the 0.11 / 0.75 lossy-codec tolerances"
        );
    });
}

// AudioTrimProbeTests.swift — diagnostic probes, deliberately assertion-free.

#[derive(Debug)]
struct TrimDecodeReport {
    samples_48k: Vec<f32>,
    source_rate: u32,
    decoded_source_frames: usize,
    first_pts_seconds: Option<f64>,
}

#[derive(Debug, Default)]
struct ProbeInfo {
    sample_rate: Option<u32>,
    duration: Option<f64>,
}

fn trim_log(message: impl std::fmt::Display) {
    eprintln!("[TRIM-PROBE] {message}");
}

fn trim_fixture_root() -> PathBuf {
    std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures")
        })
}

fn trim_backend() -> AudioBackend {
    #[cfg(target_os = "macos")]
    return Arc::new(ferment_engine_media::platform::libav::LibavAudioIo);
    #[cfg(target_os = "windows")]
    return Arc::new(ferment_engine_media::platform::libav::LibavAudioIo);
}

async fn trim_decode_platform(path: &Path, label: &str) -> Option<TrimDecodeReport> {
    let backend = trim_backend();
    let info = match backend.probe(path).await {
        Ok(info) => info,
        Err(error) => {
            trim_log(format!(
                "{label}: platform probe unavailable ({error}); raw trim attachments=N/A because Rust AudioDecodeSession does not expose CMSampleBuffer attachments"
            ));
            return None;
        }
    };
    let Some(stream) = info
        .streams
        .iter()
        .find(|stream| stream.kind == MediaKind::Audio)
    else {
        trim_log(format!("{label}: no audio track"));
        return None;
    };
    match stream.audio_timing {
        Some(timing) => trim_log(format!(
            "{label} container timing: priming={} remainder={} validFrames={}",
            timing.leading_samples,
            timing.trailing_samples,
            timing
                .audible_samples
                .map_or_else(|| "unknown".into(), |samples| samples.to_string())
        )),
        None => trim_log(format!("{label} container timing: <none readable>")),
    }
    trim_log(format!(
        "{label} AudioFile packet-table property: N/A — engine-media exposes normalized StreamInfo::audio_timing instead of the Apple AudioFileID property"
    ));
    let mut decoder = match backend.open_audio_decoder(path, stream.index).await {
        Ok(decoder) => decoder,
        Err(error) => {
            trim_log(format!("{label}: platform decoder open failed: {error}"));
            return None;
        }
    };
    let mut mono = Vec::new();
    let mut source_rate = stream.sample_rate.unwrap_or(48_000);
    let mut source_frames = 0_usize;
    let mut first_pts = None;
    let mut buffer_index = 0_usize;
    loop {
        let frame = match decoder.decode_next().await {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(error) => {
                trim_log(format!("{label}: decode stopped with error: {error}"));
                return None;
            }
        };
        source_rate = frame.sample_rate;
        first_pts.get_or_insert_with(|| frame.start.seconds());
        if buffer_index < 3 {
            trim_log(format!(
                "{label} buf#{buffer_index}: numSamples={} pts={}s start[CMGetAttachment]=N/A start[array]=N/A end[array]=N/A (platform abstraction returns already-trimmed PCM)",
                frame.frame_count(),
                frame.start.seconds()
            ));
        }
        let channels = usize::from(frame.channels);
        mono.extend(
            frame
                .samples
                .chunks_exact(channels)
                .map(|values| values.iter().sum::<f32>() / channels as f32),
        );
        source_frames += frame.frame_count();
        buffer_index += 1;
    }
    let samples_48k = if source_rate == 48_000 {
        mono
    } else {
        match resample_sinc(&mono, source_rate, 48_000) {
            Ok(samples) => samples,
            Err(error) => {
                trim_log(format!(
                    "{label}: 48 kHz diagnostic resample failed: {error}"
                ));
                return None;
            }
        }
    };
    Some(TrimDecodeReport {
        samples_48k,
        source_rate,
        decoded_source_frames: source_frames,
        first_pts_seconds: first_pts,
    })
}

fn ffprobe_info(path: &Path) -> Option<ProbeInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let mut info = ProbeInfo::default();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Some(value) = line.strip_prefix("sample_rate=") {
            info.sample_rate = value.parse().ok();
        } else if let Some(value) = line.strip_prefix("duration=") {
            info.duration = value.parse().ok();
        }
    }
    Some(info)
}

fn ffmpeg_decode_48k(path: &Path) -> Option<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        output
            .stdout
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect(),
    )
}

fn trim_argmax_abs(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map_or(0, |(index, _)| index)
}

fn remove_probe_file(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        trim_log(format!(
            "could not remove temporary {}: {error}",
            path.display()
        ));
    }
}

#[test]
#[ignore = "diagnostic probe: logs host AAC priming exposure and trim metadata"]
fn test_probe_aac_priming_exposure() {
    futures_executor::block_on(async {
        const SAMPLE_RATE: u32 = 48_000;
        const IMPULSE_FRAME: usize = 24_000;
        let wav_path =
            std::env::temp_dir().join(format!("trim-probe-{}-{}.wav", std::process::id(), "lpcm"));
        let aac_path = wav_path.with_extension("m4a");
        let mut samples = vec![0.0_f32; SAMPLE_RATE as usize];
        for index in 0..128 {
            let window = 0.5 - 0.5 * (TAU * index as f32 / 127.0).cos();
            samples[IMPULSE_FRAME + index] =
                0.9 * window * (TAU * 4_000.0 * index as f32 / SAMPLE_RATE as f32).sin();
        }
        if let Err(error) = write_wav_f32(&wav_path, &samples, SAMPLE_RATE) {
            trim_log(format!("LPCM synthesis failed: {error}"));
            return;
        }
        let lpcm_peak = trim_argmax_abs(&samples);
        trim_log(format!(
            "LPCM impulse peak at frame {lpcm_peak} (expected {IMPULSE_FRAME}); decoded {} frames",
            samples.len()
        ));
        trim_log(
            "AAC-in-CAF encode: N/A — engine-media has an elementary AAC encoder but no CAF muxer; using dev ffmpeg M4A to probe the platform decoder",
        );
        let encode = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&wav_path)
            .args(["-c:a", "aac", "-profile:a", "aac_low", "-b:a", "128k"])
            .arg(&aac_path)
            .status();
        let encoded = encode.is_ok_and(|status| status.success());
        if !encoded {
            trim_log(
                "AAC encode FAILED on this host — probing committed dark-ambient-1.m4a if available",
            );
            remove_probe_file(&wav_path);
            let fallback = trim_fixture_root().join("Audio/dark-ambient-1.m4a");
            if fallback.is_file() {
                let _ = trim_decode_platform(&fallback, "committed-m4a").await;
            } else {
                trim_log("no committed m4a fixture found");
            }
            return;
        }
        if let Some(decoded) = trim_decode_platform(&aac_path, "AAC").await {
            let aac_peak = trim_argmax_abs(&decoded.samples_48k);
            trim_log(format!(
                "AAC impulse peak at frame {aac_peak}; decoded {} frames",
                decoded.samples_48k.len()
            ));
            trim_log(format!(
                ">>> PRIMING SHIFT (AAC peak - LPCM peak) = {} frames (2112 ~= Apple AAC priming)",
                aac_peak as isize - lpcm_peak as isize
            ));
        }
        remove_probe_file(&wav_path);
        remove_probe_file(&aac_path);
    });
}

#[test]
#[ignore = "diagnostic probe: logs committed audio-container priming behavior"]
fn test_probe_committed_fixture_formats() {
    futures_executor::block_on(async {
        for (name, extension) in [
            ("ferment-gen-deephouse", "m4a"),
            ("dark-ambient-1", "m4a"),
            ("phonk-1", "mp3"),
            ("classical-piano", "mp3"),
            ("techno-1", "wav"),
        ] {
            let path = trim_fixture_root().join(format!("Audio/{name}.{extension}"));
            if !path.is_file() {
                trim_log(format!(
                    "{name}.{extension}: NOT FOUND in fixture directory"
                ));
                continue;
            }
            let probe = ffprobe_info(&path);
            let decoded = trim_decode_platform(&path, name).await;
            let (decoded_frames, source_rate, route) = if let Some(decoded) = decoded {
                (
                    decoded.samples_48k.len(),
                    Some(decoded.source_rate),
                    "engine-media platform decoder",
                )
            } else if let Some(samples) = ffmpeg_decode_48k(&path) {
                trim_log(format!(
                    "{name}.{extension}: using dev ffmpeg fallback because engine-media cannot probe this container; trim attachments and normalized packet timing=N/A"
                ));
                (
                    samples.len(),
                    probe.as_ref().and_then(|info| info.sample_rate),
                    "dev ffmpeg",
                )
            } else {
                trim_log(format!("{name}.{extension}: no decoder available"));
                continue;
            };
            let nominal = probe
                .as_ref()
                .and_then(|info| info.duration)
                .map(|duration| (duration * 48_000.0).round() as usize);
            let delta = nominal.map(|nominal| decoded_frames as isize - nominal as isize);
            trim_log(format!(
                "{name}.{extension}: nativeRate={} | decoded={decoded_frames} nominal@48k={} delta={} | route={route}",
                source_rate.map_or_else(|| "N/A".into(), |rate| rate.to_string()),
                nominal.map_or_else(|| "N/A".into(), |frames| frames.to_string()),
                delta.map_or_else(|| "N/A".into(), |frames| frames.to_string())
            ));
        }
    });
}

#[test]
#[ignore = "diagnostic probe: logs video-container audio timing and decoded frame counts"]
fn test_probe_video_container_priming() {
    futures_executor::block_on(async {
        for (name, extension) in [
            ("electronic-music-festival", "mp4"),
            ("formula1-onboard", "mov"),
            ("street-traffic", "mp4"),
        ] {
            let path = trim_fixture_root().join(format!("Video/{name}.{extension}"));
            if !path.is_file() {
                trim_log(format!("{name}.{extension}: NOT FOUND"));
                continue;
            }
            trim_log(format!(
                "{name}.{extension}: AudioFileOpenURL status=N/A — Rust does not expose the Apple AudioFileID API"
            ));
            let backend = trim_backend();
            match backend.probe(&path).await {
                Ok(info) => {
                    let Some(stream) = info.audio_streams().next() else {
                        trim_log(format!("{name}.{extension}: no audio track"));
                        continue;
                    };
                    trim_log(format!(
                        "{name}.{extension}: raw AVAssetTrack segments/timeMapping=N/A — engine-media exposes normalized start/duration/timing only"
                    ));
                    trim_log(format!(
                        "  normalized stream: start={}s duration={}s timing={:?}",
                        stream.start_time.map_or(0.0, |time| time.seconds()),
                        stream.duration.map_or(0.0, |time| time.seconds()),
                        stream.audio_timing
                    ));
                }
                Err(error) => {
                    trim_log(format!(
                        "{name}.{extension}: platform probe failed: {error}"
                    ));
                    continue;
                }
            }
            if let Some(decoded) = trim_decode_platform(&path, name).await {
                let nominal = ffprobe_info(&path)
                    .and_then(|info| info.duration)
                    .map(|duration| (duration * 48_000.0).round() as usize);
                trim_log(format!(
                    "{name}.{extension}: firstBufferPTS={} decoded={} nominal@48k={} delta={}",
                    decoded
                        .first_pts_seconds
                        .map_or_else(|| "n/a".into(), |seconds| format!("{seconds}s")),
                    decoded.samples_48k.len(),
                    nominal.map_or_else(|| "N/A".into(), |frames| frames.to_string()),
                    nominal.map_or_else(
                        || "N/A".into(),
                        |frames| (decoded.samples_48k.len() as isize - frames as isize).to_string()
                    )
                ));
                trim_log(format!(
                    "{name}.{extension}: source-rate decoded frames={} at {} Hz",
                    decoded.decoded_source_frames, decoded.source_rate
                ));
            }
        }
    });
}
