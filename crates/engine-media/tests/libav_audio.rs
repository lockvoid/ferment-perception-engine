//! Compressed audio through `LibavAudioIo`.
//!
//! `tests/audiotoolbox.rs` rewritten against the libav backend (ruling 12).
//! What these pin is sample counts and sample identity, because that is where
//! an audio backend goes wrong invisibly: a decoder that drops the encoder's
//! priming, or resamples because the output device wanted 48 kHz, produces
//! audio that sounds right and drifts out of sync with the picture.

use std::path::{Path, PathBuf};

use ferment_engine_media::platform::libav::LibavAudioIo;
use ferment_engine_media::{
    AacProfile, AudioEncoderSettings, AudioIo, Codec, MediaKind, MediaTime, PcmAudioFrame, TimeBase,
};
use futures_executor::block_on;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name)
}

fn audio_stream(path: &Path) -> ferment_engine_media::StreamInfo {
    block_on(LibavAudioIo.probe(path))
        .unwrap_or_else(|error| panic!("probe {}: {error}", path.display()))
        .streams
        .into_iter()
        .find(|stream| stream.kind == MediaKind::Audio)
        .unwrap_or_else(|| panic!("no audio stream in {}", path.display()))
}

fn decode(path: &Path) -> Vec<PcmAudioFrame> {
    let stream = audio_stream(path);
    let mut decoder = block_on(LibavAudioIo.open_audio_decoder(path, stream.index))
        .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut frames = Vec::new();
    while let Some(frame) = block_on(decoder.decode_next()).expect("decode") {
        frames.push(frame);
    }
    frames
}

fn samples_of(frames: &[PcmAudioFrame]) -> Vec<f32> {
    frames
        .iter()
        .flat_map(|frame| frame.samples.iter().copied())
        .collect()
}

/// Every container the app opens, decoded to contiguous interleaved PCM at
/// the source rate. Contiguity is the assertion that matters: a caller
/// stitching frames into a buffer relies on frame N ending exactly where
/// frame N+1 starts, and a gap becomes silence in the middle of a clip.
#[test]
fn decodes_the_fixture_matrix_to_contiguous_pcm() {
    for (name, rate, channels) in [
        ("aac-lc-48000-stereo.m4a", 48_000u32, 2u16),
        ("aac-lc-48000-mono.mp4", 48_000, 1),
        ("aac-lc-44100-stereo.mov", 44_100, 2),
        ("aac-lc-48000-stereo.mkv", 48_000, 2),
        ("mp3-cbr-44100-stereo.mp3", 44_100, 2),
        ("mp3-vbr-44100-stereo.mp3", 44_100, 2),
    ] {
        let frames = decode(&fixture(name));
        assert!(!frames.is_empty(), "{name} decoded nothing");
        assert!(
            frames
                .iter()
                .all(|frame| frame.sample_rate == rate && frame.channels == channels),
            "{name}: expected {rate} Hz / {channels} ch",
        );
        assert!(
            frames.windows(2).all(|pair| {
                pair[0].start.value + pair[0].frame_count() as i64 == pair[1].start.value
            }),
            "{name}: decoded frames are not contiguous",
        );
        // A real signal, not silence: these fixtures are tones.
        let peak = samples_of(&frames)
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!(peak > 0.01, "{name}: decoded silence (peak {peak})");
    }
}

/// HE-AAC decodes at the SBR output rate, not the core rate the container
/// leads with — the whole point of the profile is that 22050 Hz of coded
/// audio becomes 44100 Hz of PCM.
#[test]
fn he_aac_decodes_at_the_sbr_effective_rate() {
    for (name, profile) in [
        ("aac-he-44100-stereo.m4a", AacProfile::HighEfficiency),
        ("aac-he-v2-44100-stereo.m4a", AacProfile::HighEfficiencyV2),
        ("aac-he-44100-stereo.mp4", AacProfile::HighEfficiency),
    ] {
        let stream = audio_stream(&fixture(name));
        assert_eq!(stream.codec, Codec::Aac, "{name}");
        assert_eq!(
            stream.decoder_configuration.aac_profile().unwrap(),
            Some(profile),
            "{name}",
        );
        let frames = decode(&fixture(name));
        assert!(
            frames.iter().all(|frame| frame.sample_rate == 44_100),
            "{name}: decoded below the SBR rate",
        );
    }
}

/// The decoded length matches what the container declared audible. This is
/// the gapless contract: priming dropped, padding dropped, and the number
/// that remains is the program.
#[test]
fn the_decoded_length_matches_the_containers_audible_declaration() {
    for name in [
        "aac-lc-48000-stereo.m4a",
        "aac-lc-44100-mono.mp4",
        "mp3-vbr-44100-stereo.mp3",
    ] {
        let stream = audio_stream(&fixture(name));
        let Some(audible) = stream
            .audio_timing
            .and_then(|timing| timing.audible_samples)
        else {
            panic!("{name}: the container declared no audible length");
        };
        let decoded: u64 = decode(&fixture(name))
            .iter()
            .map(|frame| frame.frame_count() as u64)
            .sum();
        // Exactly, not approximately. Both trims are applied — libavcodec
        // consumes the leading skip itself, and the decoder truncates to the
        // declared end because the MOV demuxer emits no end discard — so the
        // decoded length IS the audible program.
        assert_eq!(
            decoded, audible,
            "{name}: decoded {decoded} against a declared {audible}",
        );
    }
}

/// A seek lands on the requested sample and returns the same audio a
/// sequential decode produced there. Anything less and scrubbing plays
/// something other than what playback would.
///
/// Compared numerically, not bit-exactly (ruling 12 allows this for audio).
/// libavcodec's AAC decoder converges on slightly different floats depending
/// on how much pre-roll preceded a frame — the observed difference is around
/// 3e-6, roughly -110 dBFS and an order of magnitude below 16-bit
/// quantization. The tolerance below is still three orders tighter than a
/// seek that landed on the wrong frame, which would differ by O(0.1).
#[test]
fn seeks_return_the_same_audio_as_a_sequential_decode() {
    for name in ["aac-lc-48000-stereo.m4a", "mp3-cbr-44100-stereo.mp3"] {
        let path = fixture(name);
        let stream = audio_stream(&path);
        let rate = stream.sample_rate.expect("a sample rate");
        let reference = samples_of(&decode(&path));
        let channels = usize::from(stream.channels.expect("a channel count"));

        let mut decoder = block_on(LibavAudioIo.open_audio_decoder(&path, stream.index)).unwrap();
        // A quarter of a second in, well past any pre-roll.
        let target_sample = i64::from(rate) / 4;
        let target = MediaTime::new(target_sample, TimeBase::new_unchecked(1, i64::from(rate)));
        block_on(decoder.seek(target)).expect("seek");
        let frame = block_on(decoder.decode_next())
            .expect("decode after seek")
            .expect("a frame after seek");

        assert_eq!(
            frame.start.value, target_sample,
            "{name}: seek landed on {} rather than {target_sample}",
            frame.start.value,
        );
        let at = target_sample as usize * channels;
        let take = frame.samples.len().min(reference.len() - at);
        let worst = frame.samples[..take]
            .iter()
            .zip(&reference[at..at + take])
            .map(|(seeked, sequential)| (seeked - sequential).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1.0e-4,
            "{name}: the samples after a seek differ from a sequential decode by {worst}",
        );
    }
}

#[test]
fn repeated_seeks_preserve_start_priming_and_vbr_reservoir() {
    for name in [
        "aac-lc-44100-stereo.m4a",
        "mp3-vbr-44100-stereo.mp3",
        "mp3-cbr-44100-stereo.mp3",
    ] {
        let path = fixture(name);
        let mut decoder = block_on(LibavAudioIo.open_audio_decoder(&path, 0)).unwrap();
        let mut reference = Vec::new();
        let mut origin = None;
        let base = TimeBase::new(1, 44100).unwrap();
        while let Some(frame) = block_on(decoder.decode_next()).unwrap() {
            origin.get_or_insert(frame.start.rescale(base).value);
            reference.extend_from_slice(&frame.samples);
        }
        for at in [0, 3240, 6000, 11025, 20000, 35000, 0, 17, 35000, 1023] {
            block_on(decoder.seek(MediaTime::new(at + origin.unwrap(), base))).unwrap();
            let mut data = Vec::new();
            while data.len() < 6000 {
                let Some(frame) = block_on(decoder.decode_next()).unwrap() else {
                    break;
                };
                data.extend_from_slice(&frame.samples);
            }
            assert!(!data.is_empty(), "{name}: missing seek output at {at}");
            let worst = data
                .iter()
                .zip(&reference[at as usize * 2..])
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(worst < 0.0001, "{name}: seek {at} differs by {worst}");
        }
    }
}

/// Encode, and report the priming the decoder will have to drop. The AAC
/// encoder's delay is not optional information — without it the muxer writes
/// no edit list and every round trip gains silence at the head.
#[test]
fn encodes_aac_and_reports_its_priming() {
    let settings = AudioEncoderSettings {
        bit_rate: 128_000,
        sample_rate: 48_000,
        channels: 2,
    };
    let mut encoder = block_on(LibavAudioIo.open_audio_encoder(settings)).expect("open");

    // One second of a 440 Hz tone, in the layout the encoder takes.
    let base = TimeBase::new_unchecked(1, 48_000);
    let total = 48_000usize;
    let mut samples = Vec::with_capacity(total * 2);
    for index in 0..total {
        let value = (index as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.4;
        samples.push(value);
        samples.push(value);
    }
    block_on(
        encoder.push_pcm(PcmAudioFrame::new(48_000, 2, MediaTime::new(0, base), samples).unwrap()),
    )
    .expect("push");
    let encoded = block_on(encoder.finalize()).expect("finalize");

    assert_eq!(encoded.configuration.codec, Codec::Aac);
    assert!(
        !encoded.configuration.description.is_empty(),
        "no AudioSpecificConfig: nothing could decode this",
    );
    assert_eq!(
        encoded.configuration.aac_profile().unwrap(),
        Some(AacProfile::LowComplexity),
    );
    assert!(!encoded.packets.is_empty());
    assert!(encoded.packets.iter().all(|packet| !packet.data.is_empty()));
    assert_eq!(
        encoded.timing.audible_samples,
        Some(total as u64),
        "the audible length should be what was pushed",
    );
    // AAC-LC's encoder delay is one coded frame; every encoder here reports
    // it, and a zero would mean the edit list never gets written.
    assert!(
        encoded.timing.leading_samples > 0,
        "the encoder reported no priming",
    );
}

/// Asking the encoder for a layout it was not opened with is a refusal, not
/// a silent resample — a caller that mixes rates has a bug upstream.
#[test]
fn pcm_that_does_not_match_the_encoder_is_refused() {
    let mut encoder = block_on(LibavAudioIo.open_audio_encoder(AudioEncoderSettings {
        bit_rate: 128_000,
        sample_rate: 48_000,
        channels: 2,
    }))
    .unwrap();
    let base = TimeBase::new_unchecked(1, 44_100);
    let error = block_on(encoder.push_pcm(
        PcmAudioFrame::new(44_100, 2, MediaTime::new(0, base), vec![0.0; 2048]).unwrap(),
    ))
    .unwrap_err();
    assert!(
        error.to_string().contains("48000 Hz"),
        "unexpected error: {error}",
    );
}
