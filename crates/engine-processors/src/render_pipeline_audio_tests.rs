//! Current VideoUpscaler soundtrack contract through the actual Compact graph,
//! platform codecs and muxer. Faults are injected only after the real mux has
//! finalized; FFmpeg independently decodes the delivered container and audio.
use super::*;
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_media::{
    AudioMuxTrack, EncodedPacket, MuxBackend, MuxOutput, MuxSession, MuxTrackId, Muxer,
};
use ferment_engine_test_fixtures::export;
use std::{
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    directory: PathBuf,
    source: PathBuf,
    output: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
fn run(tool: &str, args: &[&str]) -> Vec<u8> {
    let result = Command::new(tool)
        .args(args)
        .output()
        .expect("real media oracle must be installed");
    assert!(
        result.status.success(),
        "{tool}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}
fn fixture(name: &str, audio: Option<bool>) -> Fixture {
    let directory = std::env::temp_dir().join(format!(
        "ferment-current-upscale-audio-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let fixture = Fixture {
        source: directory.join("source.mov"),
        output: directory.join("output.mp4"),
        directory,
    };
    // Same current source: 480x272, 60 frames at 30fps; the selected 1.1..1.6s
    // window must contain the second (880Hz) half, never the earlier 440Hz tone.
    let tone = fixture.directory.join("tone.wav");
    if audio.is_some() {
        let mut writer = hound::WavWriter::create(
            &tone,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for index in 0..96_000 {
            let frequency = if index < 48_000 { 440.0 } else { 880.0 };
            writer
                .write_sample(
                    (0.5 * (2.0 * std::f64::consts::PI * frequency * index as f64 / 48_000.0).sin())
                        as f32,
                )
                .unwrap();
        }
        writer.finalize().unwrap();
    }
    let mut args = vec![
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=480x272:rate=30:duration=2",
    ];
    if let Some(pcm) = audio {
        args.extend([
            "-i",
            tone.to_str().unwrap(),
            "-c:a",
            if pcm { "pcm_f32le" } else { "aac" },
        ]);
        if !pcm {
            // Hound marks a one-channel Float32 WAV as front-left. Give AAC
            // the canonical mono layout by mapping the same channel directly;
            // the sample values and the source's 440/880Hz halves stay intact.
            args.extend([
                "-af",
                "channelmap=map=0-FC:channel_layout=mono",
                "-b:a",
                "192k",
            ]);
        }
    } else {
        args.push("-an");
    }
    args.extend([
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        "60",
        fixture.source.to_str().unwrap(),
    ]);
    run("ffmpeg", &args);
    fixture
}
fn backend() -> Backend {
    let master = export::artifact("realesr_general_x4v3.onnx")
        .expect("current upscale audio gate requires the actual Compact master");
    let paths =
        ModelPaths::new().with_path(ModelId::RealesrGeneralX4, ModelVersion::Latest, master);
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}
fn window() -> core::MediaTimeRange {
    core::MediaTimeRange {
        start: core::MediaTime::new(52_800, 48_000),
        duration: core::MediaTime::new(24_000, 48_000),
    }
}
fn probe(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            path.to_str().unwrap(),
        ],
    ))
    .unwrap()
}
fn assert_no_partials(f: &Fixture) {
    for path in std::fs::read_dir(&f.directory).unwrap() {
        let name = path.unwrap().file_name();
        assert!(
            !name.to_string_lossy().contains("partial"),
            "owned scratch leaked: {name:?}"
        );
    }
}
fn assert_picture(f: &Fixture, audio: bool, width: u32, height: u32) {
    let json = probe(&f.output);
    let streams = json["streams"].as_array().unwrap();
    let video: Vec<_> = streams
        .iter()
        .filter(|s| s["codec_type"] == "video")
        .collect();
    assert_eq!(video.len(), 1);
    assert_eq!(video[0]["width"], width);
    assert_eq!(video[0]["height"], height);
    assert_eq!(video[0]["nb_read_frames"], "15");
    assert_eq!(
        streams
            .iter()
            .filter(|s| s["codec_type"] == "audio")
            .count(),
        usize::from(audio)
    );
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            f.output.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ],
    );
    assert_no_partials(f);
}
#[derive(Clone, Copy, Debug)]
enum Fault {
    Picture,
    Audio,
    CancelAudio,
}
#[derive(Debug)]
struct FaultMuxer {
    inner: MuxBackend,
    fault: Fault,
    opens: Arc<AtomicUsize>,
    audio_attempts: Arc<AtomicUsize>,
    progress: ProgressSink,
    output: PathBuf,
}
#[derive(Debug)]
struct FaultSession {
    inner: Box<dyn MuxSession>,
    fault: Fault,
    lane: usize,
    audio_attempts: Arc<AtomicUsize>,
    progress: ProgressSink,
    output: PathBuf,
}
impl Muxer for FaultMuxer {
    fn open(
        &self,
        output: &Path,
        settings: MuxSettings,
    ) -> ferment_engine_media::Result<Box<dyn MuxSession>> {
        assert!(
            !self.output.exists(),
            "destination visible before completed delivery"
        );
        let lane = self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FaultSession {
            inner: self.inner.open(output, settings)?,
            fault: self.fault,
            lane,
            audio_attempts: self.audio_attempts.clone(),
            progress: self.progress.clone(),
            output: self.output.clone(),
        }))
    }
}
impl MuxSession for FaultSession {
    fn add_video_track(
        &mut self,
        track: VideoMuxTrack,
    ) -> ferment_engine_media::Result<MuxTrackId> {
        self.inner.add_video_track(track)
    }
    fn add_audio_track(
        &mut self,
        track: AudioMuxTrack,
    ) -> ferment_engine_media::Result<MuxTrackId> {
        self.audio_attempts.fetch_add(1, Ordering::SeqCst);
        self.inner.add_audio_track(track)
    }
    fn write_packet(
        &mut self,
        track: MuxTrackId,
        packet: EncodedPacket,
    ) -> ferment_engine_media::Result<()> {
        self.inner.write_packet(track, packet)
    }
    fn add_passthrough_audio_track(
        &mut self,
        track: &ferment_engine_media::AudioPassthroughTrack,
    ) -> ferment_engine_media::Result<MuxTrackId> {
        self.audio_attempts.fetch_add(1, Ordering::SeqCst);
        self.inner.add_passthrough_audio_track(track)
    }
    fn write_passthrough_audio_packet(
        &mut self,
        track: MuxTrackId,
        packet: ferment_engine_media::AudioPassthroughPacket,
    ) -> ferment_engine_media::Result<()> {
        self.inner.write_passthrough_audio_packet(track, packet)
    }
    fn finalize(self: Box<Self>) -> ferment_engine_media::Result<MuxOutput> {
        let Self {
            inner,
            fault,
            lane,
            progress,
            output,
            ..
        } = *self;
        assert!(
            !output.exists(),
            "destination visible while mux is still finishing"
        );
        let complete = inner.finalize()?;
        let fail = match fault {
            Fault::Picture => lane == 0,
            Fault::Audio | Fault::CancelAudio => lane == 1,
        };
        if fail {
            if matches!(fault, Fault::CancelAudio) {
                progress.cancel();
            }
            return Err(ferment_engine_media::VideoError::InvalidData(
                "injected failure after actual container finalization".into(),
            ));
        }
        Ok(complete)
    }
}
fn render_fault(f: &Fixture, fault: Fault) -> (Result<core::VideoRenderResult>, usize, usize) {
    let mut backends = crate::platform_encode_backends();
    let opens = Arc::new(AtomicUsize::new(0));
    let audio_attempts = Arc::new(AtomicUsize::new(0));
    let progress = ProgressSink::noop();
    backends.muxer = Arc::new(FaultMuxer {
        inner: backends.muxer.clone(),
        fault,
        opens: opens.clone(),
        audio_attempts: audio_attempts.clone(),
        progress: progress.clone(),
        output: f.output.clone(),
    });
    let result = futures_executor::block_on(upscale_video_with_backends(
        backend(),
        backends,
        &f.source,
        &f.output,
        UpscaleMode::Enhance,
        Some(window()),
        progress,
    ));
    (
        result,
        opens.load(Ordering::SeqCst),
        audio_attempts.load(Ordering::SeqCst),
    )
}

/// Kill: propagate an audio export failure or discard the valid completed picture.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn failed_audio_mux_delivers_the_finished_silent_picture_atomically() {
    let f = fixture("rescue", Some(false));
    let original = std::fs::read(&f.source).unwrap();
    let (result, opens, audio_attempts) = render_fault(&f, Fault::Audio);
    let result = result.unwrap();
    assert_eq!(opens, 2);
    assert_eq!(
        audio_attempts, 1,
        "the injected soundtrack mux must have run exactly once"
    );
    assert!(!result.audio_preserved);
    assert_eq!(result.frames_written, 15);
    assert_picture(&f, false, 480, 272);
    assert_eq!(std::fs::read(&f.source).unwrap(), original);
}

/// Kill: catch cancellation as an ordinary audio loss and commit the rescued picture.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn cancellation_during_audio_mux_never_delivers_a_silent_rescue() {
    let f = fixture("cancel", Some(false));
    let (result, _, audio_attempts) = render_fault(&f, Fault::CancelAudio);
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert_eq!(audio_attempts, 1);
    assert!(!f.output.exists());
    assert_no_partials(&f);
}

/// Kill: treat a failure finishing the picture itself as an optional soundtrack failure.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn picture_finalization_failure_remains_an_error_and_does_not_try_audio() {
    let f = fixture("picture-error", Some(false));
    let (result, opens, audio_attempts) = render_fault(&f, Fault::Picture);
    match result {
        Err(EngineError::Decode(reason)) => {
            assert!(reason.contains("injected failure after actual container finalization"))
        }
        other => panic!("picture-finalization error must propagate: {other:?}"),
    }
    assert_eq!(opens, 1);
    assert_eq!(audio_attempts, 0);
    assert!(!f.output.exists());
    assert_no_partials(&f);
}

/// Kill: report a source without a soundtrack as lost audio.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn a_source_without_audio_counts_as_preserved() {
    let f = fixture("no-audio", None);
    let result = futures_executor::block_on(upscale_video(
        backend(),
        &f.source,
        &f.output,
        UpscaleMode::Enhance,
        Some(window()),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.audio_preserved);
    assert_picture(&f, false, 480, 272);
}

fn assert_audio_window(name: &str, pcm: bool, mode: UpscaleMode, width: u32, height: u32) {
    let f = fixture(name, Some(pcm));
    let original = std::fs::read(&f.source).unwrap();
    let result = futures_executor::block_on(upscale_video(
        backend(),
        &f.source,
        &f.output,
        mode,
        Some(window()),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.audio_preserved);
    assert_picture(&f, true, width, height);
    let json = probe(&f.output);
    let audio = json["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["codec_type"] == "audio")
        .unwrap();
    let start = audio["start_time"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let duration = audio["duration"].as_str().unwrap().parse::<f64>().unwrap();
    assert!(start.abs() <= 0.03, "audio start {start}");
    assert!((duration - 0.5).abs() <= 0.04, "audio duration {duration}");
    let bytes = run(
        "ffmpeg",
        &[
            "-nostdin",
            "-v",
            "error",
            "-i",
            f.output.to_str().unwrap(),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ],
    );
    assert_eq!(bytes.len() % 4, 0);
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(samples.len() > 20_000);
    let rms = (samples.iter().map(|v| f64::from(v * v)).sum::<f64>() / samples.len() as f64).sqrt();
    assert!(rms > 0.15, "silent or attenuated soundtrack RMS {rms}");
    let crossings = samples
        .windows(2)
        .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
        .count();
    let frequency = crossings as f64 * 48_000.0 / samples.len() as f64;
    assert!(
        (frequency - 880.0).abs() <= 15.0,
        "selected 880Hz window, actual {frequency}"
    );
    assert_eq!(
        std::fs::read(&f.source).unwrap(),
        original,
        "the original file must remain byte-identical"
    );
}

/// Kill: retain the full AAC track or fill the selected window with silent samples.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn enhance_preserves_the_selected_aac_window_audibly() {
    assert_audio_window("aac-enhance", false, UpscaleMode::Enhance, 480, 272);
}

/// Kill: omit uncompressed camera audio or select its earlier 440Hz region.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn enhance_preserves_the_selected_uncompressed_camera_audio() {
    assert_audio_window("pcm-enhance", true, UpscaleMode::Enhance, 480, 272);
}

/// Kill: preserve audio only in enhance mode, losing it when delivering an upscale.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn upscale_preserves_the_selected_aac_window_audibly() {
    assert_audio_window(
        "aac-upscale",
        false,
        UpscaleMode::Upscale { longest_side: 960 },
        960,
        544,
    );
}

#[path = "render_pipeline_passthrough_tests.rs"]
mod passthrough;
