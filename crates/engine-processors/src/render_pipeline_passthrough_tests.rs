//! Actual encoded packet and selected PCM byte assertions. The CLI creates and
//! independently inspects media only in tests; production uses shipped libav.
use super::*;
use serde_json::Value;

fn tracks(path: &Path) -> Vec<Value> {
    let value: Value = serde_json::from_slice(&run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_streams",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            path.to_str().unwrap(),
        ],
    ))
    .unwrap();
    value["streams"].as_array().unwrap().clone()
}
fn packets(path: &Path, track: usize) -> Vec<Value> {
    let selector = format!("a:{track}");
    let value: Value = serde_json::from_slice(&run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            &selector,
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            path.to_str().unwrap(),
        ],
    ))
    .unwrap();
    value["packets"].as_array().unwrap().clone()
}
fn seconds(value: &Value, field: &str) -> f64 {
    value[field].as_str().unwrap().parse().unwrap()
}
fn multitrack() -> Fixture {
    let f = fixture("multitrack", None);
    run(
        "ffmpeg",
        &[
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-itsoffset",
            "0.375",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=44100:duration=1.5",
            "-itsoffset",
            "1.75",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=32000:duration=0.25",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=fra",
            "-metadata:s:a:2",
            "language=deu",
            f.source.to_str().unwrap(),
        ],
    );
    f
}
fn crop() -> core::MediaTimeRange {
    core::MediaTimeRange {
        start: core::MediaTime::new(3, 10),
        duration: core::MediaTime::new(6, 5),
    }
}

/// Kill: select only the first soundtrack, re-encode AAC, reset a delayed track
/// to zero, retain an out-of-window track, or keep packets beyond the picture.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn all_overlapping_aac_tracks_keep_original_packets_and_relative_time() {
    let f = multitrack();
    let source_bytes = std::fs::read(&f.source).unwrap();
    let result = futures_executor::block_on(upscale_video(
        backend(),
        &f.source,
        &f.output,
        UpscaleMode::Enhance,
        Some(crop()),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.audio_preserved);
    let original = tracks(&f.source);
    let actual = tracks(&f.output);
    assert_eq!(original.len(), 3);
    assert_eq!(actual.len(), 2);
    for index in 0..2 {
        for field in ["codec_name", "sample_rate", "channels", "extradata_hash"] {
            assert!(!original[index][field].is_null(), "fixture field {field}");
            assert_eq!(
                actual[index][field], original[index][field],
                "track {index} {field}"
            );
        }
        assert_eq!(
            actual[index]["tags"]["language"],
            original[index]["tags"]["language"]
        );
        let source = packets(&f.source, index);
        let expected: Vec<_> = source
            .iter()
            .filter(|p| {
                seconds(p, "pts_time") + seconds(p, "duration_time") > 0.3
                    && seconds(p, "pts_time") < 1.5
            })
            .collect();
        let delivered = packets(&f.output, index);
        assert!(expected.len() > 5);
        assert_eq!(
            delivered.len(),
            expected.len(),
            "every selected encoded access unit, track {index}"
        );
        for (got, want) in delivered.iter().zip(expected.iter()) {
            assert_eq!(
                got["data_hash"], want["data_hash"],
                "encoded bytes changed in track {index}"
            );
            assert!(
                (seconds(got, "pts_time") - (seconds(want, "pts_time") - 0.3)).abs() <= 0.0011,
                "relative packet PTS changed: {got:?} vs {want:?}"
            );
        }
        let start = seconds(&actual[index], "start_time");
        let length = seconds(&actual[index], "duration");
        assert!(
            start + length <= 1.202,
            "audio must end within the actual picture: {start}+{length}"
        );
        // The unfiltered ffmpeg CLI emits complete AAC frames at a delayed
        // track's tail, including for the independent original-iOS mux output
        // (51200 instead of50636 in this fixture). Use the SHIPPED decoder's
        // presentation-aware PCM here; packet bytes/time remain independently
        // checked above, and the selected ±1-sample budget is unchanged.
        let io = ferment_engine_media::platform::libav::LibavAudioIo;
        let info = futures_executor::block_on(ferment_engine_media::AudioIo::probe(&io, &f.output))
            .unwrap();
        let audio_stream = info
            .streams
            .iter()
            .filter(|stream| stream.kind == ferment_engine_media::MediaKind::Audio)
            .nth(index)
            .unwrap();
        let mut decoder = futures_executor::block_on(
            ferment_engine_media::AudioIo::open_audio_decoder(&io, &f.output, audio_stream.index),
        )
        .unwrap();
        let mut decoded = Vec::new();
        let mut next_start = None;
        while let Some(frame) = futures_executor::block_on(decoder.decode_next()).unwrap() {
            assert_eq!(frame.channels, 1, "pinned source is mono per track");
            if next_start.is_none() && index == 0 {
                assert_eq!(
                    frame.start.value, 0,
                    "partial AAC head trim must also advance PCM PTS"
                );
            }
            if let Some(next) = next_start {
                assert_eq!(frame.start.value, next, "PCM must be contiguous");
            }
            next_start = Some(frame.start.value + frame.frame_count() as i64);
            decoded.extend_from_slice(&frame.samples);
        }
        let rate: f64 = original[index]["sample_rate"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let source_start = seconds(&original[index], "start_time").max(0.3);
        let expected_frames = ((1.5 - source_start) * rate).round() as i64;
        let frames = decoded.len() as i64;
        assert!(
            (frames - expected_frames).abs() <= 1,
            "track {index}: audible frames {frames}, selected source frames {expected_frames}"
        );
        let tail = &decoded[decoded.len() - 256..];
        let rms = (tail.iter().map(|v| f64::from(v * v)).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(
            rms > 0.01,
            "selected tail must remain audible, track {index}: {rms}"
        );
    }
    assert!(
        seconds(&actual[1], "start_time") > 0.04,
        "delayed soundtrack must not be moved to zero"
    );
    assert_eq!(std::fs::read(&f.source).unwrap(), source_bytes);
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
    assert_no_partials(&f);
}

/// Kill: transcode camera PCM to AAC or change a selected sample's bit pattern.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn camera_pcm_preserves_exact_selected_sample_bytes_and_codec() {
    let f = fixture("pcm-byte-identity", Some(true));
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
    let before = tracks(&f.source);
    let after = tracks(&f.output);
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    for field in ["codec_name", "sample_rate", "channels", "bits_per_sample"] {
        assert_eq!(after[0][field], before[0][field], "{field}");
    }
    let raw = |path: &Path| {
        run(
            "ffmpeg",
            &[
                "-nostdin",
                "-v",
                "error",
                "-i",
                path.to_str().unwrap(),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                "-f",
                "f32le",
                "-",
            ],
        )
    };
    let original = raw(&f.source);
    let delivered = raw(&f.output);
    assert_eq!(
        delivered.as_slice(),
        &original[52_800 * 4..76_800 * 4],
        "passthrough must retain exact PCM frame bytes"
    );
    assert_no_partials(&f);
}

/// Kill: omit original codec-delay packets when copying an entire soundtrack.
#[test]
#[ignore = "actual Compact model/platform codecs/ffmpeg; run serially"]
fn a_full_aac_track_keeps_its_original_priming_packet_and_all_access_units() {
    let f = fixture("aac-full-packets", Some(false));
    let result = futures_executor::block_on(upscale_video(
        backend(),
        &f.source,
        &f.output,
        UpscaleMode::Enhance,
        None,
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert!(result.audio_preserved);
    let source = packets(&f.source, 0);
    let output = packets(&f.output, 0);
    assert!(
        seconds(&source[0], "pts_time") < 0.0,
        "fixture carries encoder priming"
    );
    let hash_list = |list: &[Value]| {
        list.iter()
            .map(|p| p["data_hash"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(hash_list(&output), hash_list(&source));
    let trim = |packet: &Value| {
        packet["side_data_list"]
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| item["side_data_type"] == "Skip Samples")
            })
            .map(|item| {
                (
                    item["skip_samples"].as_u64().unwrap_or(0),
                    item["discard_padding"].as_u64().unwrap_or(0),
                )
            })
            .unwrap_or((0, 0))
    };
    assert!(
        trim(&source[0]).0 > 0,
        "fixture must carry real encoder delay"
    );
    assert_eq!(
        trim(&output[0]),
        trim(&source[0]),
        "first packet codec-delay metadata"
    );
    assert_eq!(
        trim(output.last().unwrap()),
        trim(source.last().unwrap()),
        "last packet discard-padding metadata"
    );
    assert!(
        (seconds(&output[0], "pts_time") - seconds(&source[0], "pts_time")).abs() <= 1.0 / 48_000.0
    );
    assert_no_partials(&f);
}

/// Kill: defer cancellation until all source audio packets have been read.
#[test]
#[ignore = "actual file container/ffmpeg; run explicitly"]
fn cancellation_is_checked_after_each_native_packet_read() {
    let f = fixture("packet-read-cancel", Some(false));
    let original = std::fs::read(&f.source).unwrap();
    let info = ferment_engine_media::platform::libav::probe(&f.source).unwrap();
    let stream = info.audio_streams().next().unwrap();
    let mut reader = ferment_engine_media::AudioPassthroughReader::open(
        &f.source,
        stream,
        MediaTime::ZERO,
        MediaTime::new(2, TimeBase::SECONDS),
    )
    .unwrap()
    .unwrap();
    let mut checkpoints = 0;
    let result = reader.next_packet_checked(|| {
        checkpoints += 1;
        if checkpoints == 3 {
            Err(ferment_engine_media::VideoError::InvalidData(
                "cancel exactly after first native read".into(),
            ))
        } else {
            Ok(())
        }
    });
    assert!(result.unwrap_err().to_string().contains("cancel exactly"));
    assert_eq!(checkpoints, 3);
    drop(reader);
    assert_eq!(std::fs::read(&f.source).unwrap(), original);
    assert!(!f.output.exists());
    assert_no_partials(&f);
}
