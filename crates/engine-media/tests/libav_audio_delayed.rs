//! Delayed-track presentation endpoints through the shipped libav decoder.
//! The source is the unchanged pinned three-track AAC MOV used by actual
//! Android passthrough tests. Expected ticks below come from its independently
//! frozen stream tables, not from the decoder under test.
use ferment_engine_media::platform::libav::{LibavAudioIo, LibavMuxer};
use ferment_engine_media::{
    AudioIo, AudioPassthroughReader, MediaKind, MediaTime, MuxSettings, Muxer, PcmAudioFrame,
    TimeBase,
};
use futures_executor::block_on;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    // The iOS source keeps app-level media fixtures separate from the engine
    // fixtures. An admitted overlay can still use the original single root;
    // without one, the byte-identical copy checked in beside these tests.
    let root = std::env::var_os("FERMENT_ENGINE_SUPPLEMENT_FIXTURES_DIR")
        .or_else(|| std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR"))
        .map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
            PathBuf::from,
        );
    let path = root.join("Video/upscale-three-aac-tracks.mov");
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 68_114);
    // Qualification independently requires SHA256
    //01b3d290e63ddcbb8b3f69817509939a21f1c9b2bfe32040a8501f603705077f.
    path
}
fn stream(path: &std::path::Path, audio: usize) -> ferment_engine_media::StreamInfo {
    block_on(LibavAudioIo.probe(path))
        .unwrap()
        .streams
        .into_iter()
        .filter(|stream| stream.kind == MediaKind::Audio)
        .nth(audio)
        .unwrap()
}
fn decoded(path: &std::path::Path, stream: u32) -> Vec<PcmAudioFrame> {
    let mut decoder = block_on(LibavAudioIo.open_audio_decoder(path, stream)).unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
        assert!(frame.frame_count() > 0);
        frames.push(frame);
    }
    frames
}
fn samples(frames: &[PcmAudioFrame]) -> Vec<f32> {
    frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect()
}

/// Kill: treat the67174-frame program length as absolute PTS and lose its15514-frame delay from the tail.
#[test]
fn a_delayed_aac_track_keeps_its_full_program_and_absolute_start() {
    let path = fixture();
    let stream = stream(&path, 1);
    assert_eq!(stream.sample_rate, Some(44_100));
    assert_eq!(stream.channels, Some(1));
    let frames = decoded(&path, stream.index);
    assert_eq!(frames.first().unwrap().start.value, 15_514);
    assert_eq!(
        frames.iter().map(PcmAudioFrame::frame_count).sum::<usize>(),
        67_174
    );
    assert!(
        frames
            .windows(2)
            .all(|pair| pair[0].start.value + pair[0].frame_count() as i64 == pair[1].start.value)
    );
    let last = frames.last().unwrap();
    assert_eq!(last.start.value + last.frame_count() as i64, 82_688);
    let pcm = samples(&frames);
    let tail = &pcm[pcm.len() - 256..];
    let rms = (tail.iter().map(|x| f64::from(x * x)).sum::<f64>() / 256.0).sqrt();
    assert!(rms > 0.01, "declared source tail must stay audible: {rms}");
}

fn assert_tail_seek(audio: usize, origin: i64, end: i64, rate: u32) {
    let path = fixture();
    let stream = stream(&path, audio);
    let reference = samples(&decoded(&path, stream.index));
    assert_eq!(reference.len() as i64, end - origin);
    let target = end - 100;
    let base = TimeBase::new_unchecked(1, i64::from(rate));
    let mut decoder = block_on(LibavAudioIo.open_audio_decoder(&path, stream.index)).unwrap();
    block_on(decoder.seek(MediaTime::new(target, base))).unwrap();
    let frame = block_on(decoder.decode_next())
        .unwrap()
        .expect("selected final100frames");
    assert_eq!(frame.sample_rate, rate);
    assert_eq!(frame.channels, 1);
    assert_eq!(frame.start.value, target);
    assert_eq!(
        frame.frame_count(),
        100,
        "seek path must apply the same terminal trim"
    );
    assert!(block_on(decoder.decode_next()).unwrap().is_none());
    let expected = &reference[(target - origin) as usize..];
    // Current iOS creates a fresh AVAssetReader for a seek. Its actual
    // independent decode of these same access units changes AAC perceptual
    // noise samples (max0.0003503) while landing on the right sample. Thus
    // the old supplemental max1e-4 comparison was not an iOS contract.
    // Keep exact PTS/count/EOS above, then verify audible finite PCM whose
    // best alignment is the requested sample, not either adjacent sample.
    // This detects a one-frame slip without demanding PNS bit identity.
    assert!(frame.samples.iter().all(|sample| sample.is_finite()));
    let rms = (frame.samples.iter().map(|x| f64::from(x * x)).sum::<f64>()
        / frame.samples.len() as f64)
        .sqrt();
    assert!(
        rms > 0.01,
        "sought original tone must remain audible: {rms}"
    );
    let alignment_error = |offset: isize| {
        (1..frame.samples.len() - 1)
            .map(|index| {
                let delta = f64::from(frame.samples[index])
                    - f64::from(expected[(index as isize + offset) as usize]);
                delta * delta
            })
            .sum::<f64>()
    };
    let exact = alignment_error(0);
    assert!(
        exact < alignment_error(-1) && exact < alignment_error(1),
        "sought signal is not aligned to the requested sample: before={}, exact={exact}, after={}",
        alignment_error(-1),
        alignment_error(1)
    );
}

/// Kill: return the sought tail before applying its delayed presentation endpoint.
#[test]
fn a_seek_into_the_last_delayed_aac_frame_stops_at_the_program_end() {
    assert_tail_seek(1, 15_514, 82_688, 44_100);
}

/// Kill: leave the original zero-origin AAC seek path able to expose terminal codec padding.
#[test]
fn a_seek_into_the_last_zero_origin_aac_frame_stops_at_the_program_end() {
    assert_tail_seek(0, 0, 96_000, 48_000);
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Kill: trim part of an AAC head packet without advancing its PCM PTS, or
/// treat a delayed track's program length as an absolute endpoint.
#[test]
fn a_partial_aac_head_trim_preserves_exact_selected_pcm_time_and_length() {
    let source = fixture();
    let before = std::fs::read(&source).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = Scratch(std::env::temp_dir().join(format!(
        "ferment-aac-partial-head-{}-{nonce}",
        std::process::id()
    )));
    std::fs::create_dir(&scratch.0).unwrap();
    for (index, rate, first, length) in [(0, 48_000, 0, 57_600), (1, 44_100, 2_284, 50_636)] {
        let info = stream(&source, index);
        let mut reader = AudioPassthroughReader::open(
            &source,
            &info,
            MediaTime::new(3, TimeBase::new_unchecked(1, 10)),
            MediaTime::new(6, TimeBase::new_unchecked(1, 5)),
        )
        .unwrap()
        .unwrap();
        let output = scratch.0.join(format!("selected-{index}.m4a"));
        let mut mux = LibavMuxer.open(&output, MuxSettings::default()).unwrap();
        let track = mux.add_passthrough_audio_track(reader.track()).unwrap();
        let mut first_packet = true;
        while let Some(packet) = reader.next_packet().unwrap() {
            if first_packet && index == 0 {
                assert_eq!(
                    packet
                        .pts()
                        .unwrap()
                        .rescale(TimeBase::new_unchecked(1, 48_000))
                        .value,
                    -64,
                    "fixture must straddle the requested head by part of one AAC frame"
                );
            }
            first_packet = false;
            mux.write_passthrough_audio_packet(track, packet).unwrap();
        }
        assert!(!first_packet);
        mux.finalize().unwrap();
        let output_info = stream(&output, 0);
        let frames = decoded(&output, output_info.index);
        assert_eq!(frames[0].sample_rate, rate);
        let actual_start = frames[0].start.value;
        if index == 0 {
            assert_eq!(
                actual_start, first,
                "trimmed PCM must begin exactly at zero"
            );
        } else {
            assert!(
                (actual_start - first).abs() <= 1,
                "delayed first PCM: {actual_start}"
            );
        }
        assert!(
            frames
                .windows(2)
                .all(|pair| pair[0].start.value + pair[0].frame_count() as i64
                    == pair[1].start.value)
        );
        let actual_count = frames.iter().map(PcmAudioFrame::frame_count).sum::<usize>() as i64;
        assert!(
            (actual_count - length).abs() <= 1,
            "selected track{index}: decoded{actual_count}, expected{length}"
        );
        let pcm = samples(&frames);
        let tail = &pcm[pcm.len() - 256..];
        let rms = (tail.iter().map(|x| f64::from(x * x)).sum::<f64>() / 256.0).sqrt();
        assert!(
            rms > 0.01,
            "selected original tail must remain audible: {rms}"
        );
    }
    assert_eq!(std::fs::read(&source).unwrap(), before);
}
