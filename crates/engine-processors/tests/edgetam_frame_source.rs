//! `EdgeTamFrameSource` against the synthetic EdgeTAM video fixtures.
//!
//! The frame source is the one part of the video session that needs no model,
//! so it is graded now while the memory-attention export is still blocked. The
//! fixtures are the three files `EdgeTAMFrameSourceTests` uses, and their
//! geometry is a fact of the FILES (`ffprobe`), not of this code:
//!
//! ```text
//! one-frame-16x16.mp4        16x16, 30 fps, 1 frame
//! rotated-48x32.mp4          48x32, 30 fps, 30 frames, every frame different
//! ten-minute-30fps-16x16.mp4 16x16, 30 fps, 18 000 frames, all black
//! ```
//!
//! **What these tests must NOT assert.** `EdgeTamRgbFrame::frame_index` is a
//! value `take_frame_at` copies from its own ARGUMENT, so `frame(i).frame_index
//! == i` is a restatement of the code and holds however wrong the decode is —
//! measured: four separate mutations survived a suite built on it. What links
//! the frame asked for to the frame returned is the TIMESTAMP, which comes off
//! the decoder, and the PIXELS. Both are used below.

use std::path::PathBuf;

use ferment_engine_core::VideoObjectPropagationDirection::{Forward, Reverse};
use ferment_engine_processors::edgetam::frame_source::EdgeTamFrameSource;
use ferment_engine_processors::edgetam::preprocessing::EdgeTamRgbFrame;
use ferment_engine_processors::video_backend;

/// A copy of a fixture whose first presentation timestamp is NOT zero.
///
/// Every committed EdgeTAM fixture starts at pts 0, so none of them can see
/// whether the source subtracts its own origin — which is exactly how that bug
/// survived. `-output_ts_offset` remuxes without re-encoding, so the pictures
/// are byte-identical to the original and only the clock moves.
fn offset_fixture(test: &str, seconds: &str) -> Option<PathBuf> {
    derived(
        test,
        "offset",
        &["-c", "copy", "-output_ts_offset", seconds],
    )
}

/// A copy re-encoded at a rate that is NOT the fixtures' 30 fps.
///
/// All three committed fixtures are exactly 30 fps, so a source that simply
/// hard-coded 30 would satisfy every one of them — measured: that mutation
/// survived until this fixture existed.
fn rate_fixture(test: &str, fps: &str) -> Option<PathBuf> {
    derived(
        test,
        fps,
        &[
            "-filter:v",
            &format!("fps={fps}"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ],
    )
}

fn derived(test: &str, tag: &str, args: &[&str]) -> Option<PathBuf> {
    let source = fixture(test, "rotated-48x32.mp4")?;
    let out = std::env::temp_dir().join(format!("ferment-e3-{tag}-{}.mp4", std::process::id()));
    let status = std::process::Command::new("ffmpeg")
        .args(["-loglevel", "error", "-y", "-i"])
        .arg(&source)
        .args(args)
        .arg(&out)
        .status();
    match status {
        Ok(status) if status.success() => Some(out),
        _ => {
            println!("SKIP {test}: ffmpeg could not build the {tag} fixture");
            None
        }
    }
}

fn fixture(test: &str, name: &str) -> Option<PathBuf> {
    let directory = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(|root| PathBuf::from(root).join("EdgeTAM"))
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/edgetam")
        });
    let path = directory.join(name);
    if path.exists() {
        Some(path)
    } else {
        println!("SKIP {test}: no fixture at {}", path.display());
        None
    }
}

fn open(path: &std::path::Path) -> EdgeTamFrameSource {
    futures_executor::block_on(EdgeTamFrameSource::new(video_backend(), path))
        .expect("open the frame source")
}

fn frame(source: &mut EdgeTamFrameSource, index: usize) -> EdgeTamRgbFrame {
    futures_executor::block_on(source.frame(index))
        .unwrap_or_else(|error| panic!("frame {index}: {error}"))
}

/// Every frame the timeline names decodes to the picture at THAT timestamp,
/// whether it is reached by seeking or by walking.
///
/// The two paths are independent — `frame` seeks to the timestamp and skips
/// forward, `sequential_frame` reuses the open decoder — so making them agree,
/// on 30 frames that are all pixel-different, is the check. Neither side is
/// derived from the other.
///
/// kill: make `sequential_frame`'s cursor reuse off by one — the walked
/// pictures shift and stop matching the sought ones.
///
/// NOT kills, both measured, both UNFALSIFIABLE on this backend: removing the
/// `micros < target` skip, and removing the `micros != target` overshoot
/// refusal. `DecodeSession::seek`'s own contract says it "seeks to the previous
/// sync sample and discards decoded leading pictures until `target`", and
/// VideoToolbox honours it exactly — so on the Mac neither branch ever fires.
/// They are iOS's two guards (`EdgeTAMFrameSource.swift:544-553`, which throws
/// "Random reader skipped the indexed timestamp …") kept for MediaCodec, the
/// backend this lane cannot run. A fixture that makes them fire needs a
/// long-GOP source whose seek lands off-target; the three committed EdgeTAM
/// fixtures are short and all-keyframe.
#[test]
fn seeking_to_a_frame_and_walking_to_it_return_the_same_picture() {
    let Some(path) = fixture(
        "seeking_to_a_frame_and_walking_to_it_return_the_same_picture",
        "rotated-48x32.mp4",
    ) else {
        return;
    };

    // The walk first: one open decoder, every frame in order.
    let mut source = open(&path);
    let count = source.source_info().frame_count;
    assert_eq!(count, 30);
    let walked: Vec<EdgeTamRgbFrame> = (0..count)
        .map(|index| {
            futures_executor::block_on(source.sequential_frame(index, Forward))
                .unwrap_or_else(|error| panic!("sequential frame {index}: {error}"))
        })
        .collect();

    // Only frames whose picture is UNIQUE in the walk can catch a wrong
    // picture; this fixture repeats one near the end (28 and 29 are the same
    // frame). Selecting them is not weakening the test — it is refusing to
    // claim discrimination the fixture cannot provide.
    let discriminating: Vec<usize> = (0..count)
        .filter(|index| {
            walked
                .iter()
                .enumerate()
                .all(|(other, frame)| other == *index || frame.pixels != walked[*index].pixels)
        })
        .collect();
    assert!(
        discriminating.len() >= 20,
        "only {} of {count} frames are distinguishable by pixels; the fixture cannot grade a \
         wrong picture",
        discriminating.len()
    );

    // Now the seek path, out of order so no accidental sequentiality helps.
    let mut source = open(&path);
    for &index in discriminating.iter().rev().chain(discriminating.iter()) {
        let sought = frame(&mut source, index);
        assert_eq!(
            sought.presentation_time_micros(),
            source.presentation_time_micros(index).expect("timeline"),
            "frame {index} decoded at a timestamp the timeline does not name"
        );
        assert_eq!(
            sought.pixels, walked[index].pixels,
            "seeking to frame {index} returned a different picture than walking to it"
        );
    }
}

/// The REVERSE walk has no fast path: every step seeks, and every step must
/// land on its own frame.
///
/// kill: let `sequential_frame` take the forward fast path for `Reverse` — the
/// decoder advances while the caller retreats, and the timestamps stop matching
/// the timeline.
#[test]
fn a_reverse_walk_lands_on_each_frames_own_timestamp() {
    let Some(path) = fixture(
        "a_reverse_walk_lands_on_each_frames_own_timestamp",
        "rotated-48x32.mp4",
    ) else {
        return;
    };
    let mut source = open(&path);
    let count = source.source_info().frame_count;
    for index in (0..count).rev() {
        let frame = futures_executor::block_on(source.sequential_frame(index, Reverse))
            .unwrap_or_else(|error| panic!("reverse frame {index}: {error}"));
        assert_eq!(
            frame.presentation_time_micros(),
            source.presentation_time_micros(index).expect("timeline"),
            "reverse step {index} landed on the wrong frame"
        );
    }
}

/// A one-frame video is a real case, not a degenerate one: it is what a
/// still-image import looks like to the video path.
///
/// kill: drop the last decoded timestamp in `new` (an off-by-one in the
/// timeline loop) and the count is zero, so the session reports an empty video.
#[test]
fn a_one_frame_video_has_exactly_one_frame() {
    let Some(path) = fixture(
        "a_one_frame_video_has_exactly_one_frame",
        "one-frame-16x16.mp4",
    ) else {
        return;
    };
    let mut source = open(&path);
    let info = source.source_info();
    assert_eq!(info.frame_count, 1);
    assert_eq!((info.width, info.height), (16, 16));

    let only = frame(&mut source, 0);
    assert_eq!(only.pixels.len(), 16 * 16 * 3);
    assert_eq!(only.presentation_time_micros(), 0);

    // One past the end is an error, not a panic and not frame 0 again.
    assert!(
        futures_executor::block_on(source.frame(1)).is_err(),
        "frame 1 of a one-frame video must be refused"
    );
    assert!(source.presentation_time_micros(1).is_err());
}

/// A ten-minute source: 18 000 frames indexed, the last one reachable, and the
/// timeline strictly increasing across all of them.
///
/// This fixture is all black, so pixels cannot grade it — the timestamps are
/// the whole signal, which is also the point: eight bytes a frame is what lets
/// a ten-minute source be indexed at all (report D8).
///
/// kill: index the timeline in DECODE order (drop the monotonic guard and the
/// `continue` that skips a repeat) and the strict increase below fails.
#[test]
fn a_ten_minute_sources_timeline_is_strictly_increasing() {
    let Some(path) = fixture(
        "a_ten_minute_sources_timeline_is_strictly_increasing",
        "ten-minute-30fps-16x16.mp4",
    ) else {
        return;
    };
    let mut source = open(&path);
    let info = source.source_info();
    assert_eq!(info.frame_count, 18_000);
    assert!(
        (info.duration() - 600.0).abs() < 1.0,
        "a ten-minute source measured {} s",
        info.duration()
    );

    let mut previous = i64::MIN;
    for index in 0..info.frame_count {
        let time = source.presentation_time_micros(index).expect("timeline");
        assert!(
            time > previous,
            "timeline entry {index} ({time}) does not follow {previous}"
        );
        previous = time;
    }

    // The last frame decodes at the timestamp the timeline names, after a seek
    // across ten minutes — and the first is still reachable afterwards.
    let last = frame(&mut source, info.frame_count - 1);
    assert_eq!(
        last.presentation_time_micros(),
        source.presentation_time_micros(17_999).expect("timeline")
    );
    assert!(last.presentation_time() > 599.0);
    assert_eq!(frame(&mut source, 0).presentation_time_micros(), 0);
}

/// `cancel_decoder_resources` releases the decoder and the next access reopens
/// it — the session calls this at every reset and at the end of a propagation.
///
/// kill: make it a no-op and `has_decoder_resources` keeps answering true, so
/// the session's memory snapshot lies about an open decoder.
#[test]
fn releasing_the_decoder_leaves_the_source_usable() {
    let Some(path) = fixture(
        "releasing_the_decoder_leaves_the_source_usable",
        "rotated-48x32.mp4",
    ) else {
        return;
    };
    let mut source = open(&path);
    assert!(
        !source.has_decoder_resources(),
        "nothing is open before the first frame"
    );

    let before = frame(&mut source, 7);
    assert!(source.has_decoder_resources());

    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());

    let after = frame(&mut source, 7);
    assert_eq!(
        before.presentation_time_micros(),
        after.presentation_time_micros()
    );
    assert_eq!(
        before.pixels, after.pixels,
        "the same frame must decode the same"
    );
}

/// Kill: seek every reverse frame or reuse a stale batch after a direction
/// change. Both directions preserve decoded timestamps while the seek count
/// proves bounded batching rather than an output-equivalent slow fallback.
#[test]
fn forward_reuses_the_decoder_and_reverse_reuses_a_bounded_batch() {
    let path = fixture(
        "forward_reuses_the_decoder_and_reverse_reuses_a_bounded_batch",
        "rotated-48x32.mp4",
    )
    .expect("the current frame-source gate requires the real rotated fixture");
    let mut source = open(&path);
    let count = source.source_info().frame_count;
    let mut forward = Vec::new();
    for index in 0..count {
        forward.push(futures_executor::block_on(source.sequential_frame(index, Forward)).unwrap());
    }
    assert_eq!(source.seek_count(), 1);
    let before_reverse = source.seek_count();
    for index in (0..count).rev() {
        let actual = futures_executor::block_on(source.sequential_frame(index, Reverse)).unwrap();
        assert_eq!(
            actual.presentation_time_micros(),
            forward[index].presentation_time_micros()
        );
        assert_eq!(
            actual.pixels, forward[index].pixels,
            "reverse frame {index} must contain its own pixels"
        );
        assert!(source.cached_reverse_byte_count() <= 64 * 1_048_576);
    }
    assert_eq!(source.seek_count() - before_reverse, 1);
    let switched = futures_executor::block_on(source.sequential_frame(12, Forward)).unwrap();
    assert_eq!(switched.pixels, forward[12].pixels);
    let reversed = futures_executor::block_on(source.sequential_frame(10, Reverse)).unwrap();
    assert_eq!(reversed.pixels, forward[10].pixels);
    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());
    assert_eq!(source.cached_reverse_byte_count(), 0);
}

/// Kill: exceed the current iOS decoded-RGB budget, or let malformed geometry
/// overflow into a massive allocation.
#[test]
fn reverse_batch_budgets_match_current_small_hd_and_4k_limits() {
    assert_eq!(EdgeTamFrameSource::reverse_reader_batch_size(16, 16), 1_024);
    assert_eq!(
        EdgeTamFrameSource::reverse_reader_batch_size(1_920, 1_080),
        10
    );
    assert_eq!(
        EdgeTamFrameSource::reverse_reader_batch_size(3_840, 2_160),
        2
    );
    assert_eq!(EdgeTamFrameSource::reverse_reader_batch_size(0, 1), 1);
    assert_eq!(
        EdgeTamFrameSource::reverse_reader_batch_size(usize::MAX, 2),
        1
    );
}

/// Kill: retain old reverse batches, miss a batch boundary, or seek once per
/// picture. The 18,000 actual decoded timestamps cross 18 bounded batches.
#[test]
fn a_ten_minute_reverse_walk_has_exact_timestamps_and_bounded_batches() {
    let path = fixture(
        "a_ten_minute_reverse_walk_has_exact_timestamps_and_bounded_batches",
        "ten-minute-30fps-16x16.mp4",
    )
    .expect("the current frame-source gate requires the real ten-minute fixture");
    let mut source = open(&path);
    assert_eq!(source.source_info().frame_count, 18_000);
    for index in (0..18_000).rev() {
        let expected = source.presentation_time_micros(index).unwrap();
        let actual = futures_executor::block_on(source.sequential_frame(index, Reverse)).unwrap();
        assert_eq!(actual.presentation_time_micros(), expected);
        assert!((actual.presentation_time() - index as f64 / 30.0).abs() < 1e-9);
        assert!(source.cached_reverse_byte_count() <= 1_024 * 16 * 16 * 3);
    }
    assert_eq!(source.seek_count(), 18_000_usize.div_ceil(1_024));
    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());
    assert_eq!(source.cached_reverse_byte_count(), 0);
}

/// A source whose first frame is not at zero still reports its frames from
/// ZERO — the timeline is origin-relative, as iOS's is.
///
/// A container's first retained sample is video time zero to a decoder even
/// when the asset timeline places it later; MPEG-TS conventionally starts near
/// 1.4 s. Without the subtraction every mask this session emits is late by the
/// origin, and every committed fixture starts at 0, so nothing else can see it.
///
/// kill: stop subtracting `origin` in `take_frame_at` or in
/// `presentation_time_micros` — frame 0 then reports 1 400 000 µs.
#[test]
fn a_source_that_starts_late_still_reports_from_zero() {
    let Some(offset) = offset_fixture("a_source_that_starts_late_still_reports_from_zero", "1.4")
    else {
        return;
    };
    let plain = fixture(
        "a_source_that_starts_late_still_reports_from_zero",
        "rotated-48x32.mp4",
    )
    .expect("the plain fixture");

    let mut shifted = open(&offset);
    let mut original = open(&plain);
    assert_eq!(
        shifted.source_info().frame_count,
        original.source_info().frame_count,
        "the remux must not change the frame count"
    );

    for index in [0_usize, 1, 15, 29] {
        assert_eq!(
            shifted.presentation_time_micros(index).expect("timeline"),
            original.presentation_time_micros(index).expect("timeline"),
            "frame {index} of the shifted source must report the same time as the original"
        );
        let from_shifted = frame(&mut shifted, index);
        let from_original = frame(&mut original, index);
        assert_eq!(
            from_shifted.presentation_time_micros(),
            from_original.presentation_time_micros(),
            "frame {index} decoded at a shifted timestamp"
        );
        // The remux copied the pictures, so they must be identical too — which
        // is what proves the seek still found the RIGHT frame in the shifted
        // clock while reporting it in the relative one.
        assert_eq!(from_shifted.pixels, from_original.pixels);
    }
    assert_eq!(frame(&mut shifted, 0).presentation_time_micros(), 0);
    let _ = std::fs::remove_file(&offset);
}

/// The frame rate is MEASURED over the indexed frames, not read off the
/// container's nominal rate — iOS's rule (`EdgeTAMFrameSource.swift:241-244`),
/// and the one that is right for variable-frame-rate sources.
///
/// kill: hard-code 30.0, or prefer the container's declared rate. Both were the
/// port's behaviour and both survived every other test here, because all three
/// committed fixtures are exactly 30 fps.
#[test]
fn the_frame_rate_is_measured_not_declared() {
    let Some(path) = rate_fixture("the_frame_rate_is_measured_not_declared", "15") else {
        return;
    };
    let source = open(&path);
    let info = source.source_info();
    assert_eq!(info.frame_count, 15);
    assert!(
        (info.frame_rate - 15.0).abs() < 0.5,
        "a 15 fps source measured {} fps",
        info.frame_rate
    );
    assert!(
        (info.duration() - 1.0).abs() < 0.1,
        "a one-second source measured {} s",
        info.duration()
    );
    let _ = std::fs::remove_file(&path);
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
#[path = "support/current_process_memory.rs"]
mod process_memory;

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
fn current_resident_bytes() -> u64 {
    process_memory::footprint_bytes()
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
fn ten_minute_decode_memory(direction: ferment_engine_core::VideoObjectPropagationDirection) {
    let path = fixture("ten_minute_decode_memory", "ten-minute-30fps-16x16.mp4")
        .expect("the memory gate requires the actual ten-minute clip");
    let mut source = open(&path);
    let count = source.source_info().frame_count;
    assert_eq!(count, 18_000);
    let index = |processed| {
        if direction == Forward {
            processed
        } else {
            count - 1 - processed
        }
    };
    let consume = |source: &mut EdgeTamFrameSource, at| {
        let frame = futures_executor::block_on(source.sequential_frame(at, direction)).unwrap();
        assert_eq!(
            frame.presentation_time_micros(),
            source.presentation_time_micros(at).unwrap()
        );
        frame.pixels[0] // No frame survives one iteration or a consumer pause.
    };
    for processed in 0..512 {
        let _ = consume(&mut source, index(processed));
    }
    let baseline = current_resident_bytes();
    let mut peak = baseline;
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for processed in 512..count {
        checksum ^= u64::from(consume(&mut source, index(processed)));
        checksum = checksum.wrapping_mul(0x100_0000_01b3);
        if processed % 4_500 == 0 || processed == count - 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            peak = peak.max(current_resident_bytes());
        }
    }
    source.cancel_decoder_resources();
    assert!(!source.has_decoder_resources());
    assert_eq!(source.cached_reverse_byte_count(), 0);
    let growth = peak.saturating_sub(baseline);
    println!(
        "current EdgeTAM decode {direction:?}: baseline={baseline} peak={peak} checksum={checksum:x}"
    );
    assert!(
        growth < 32 * 1_048_576,
        "duration-sized decoder retention grew by {growth} bytes"
    );
}

/// Kill: keep a decoded frame or codec surface per request; pauses must not
/// turn a ten-minute forward consumer into duration-sized retained memory.
#[test]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
#[ignore = "real ten-minute decoder process-memory gate; run serially"]
fn current_ten_minute_forward_decode_stays_bounded() {
    ten_minute_decode_memory(Forward);
}

/// Kill: accumulate every prior reverse batch instead of releasing it; the
/// current source's 32 MiB sustained-growth bound must hold across 18 batches.
#[test]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
#[ignore = "real ten-minute decoder process-memory gate; run serially"]
fn current_ten_minute_reverse_decode_stays_bounded() {
    ten_minute_decode_memory(Reverse);
}
