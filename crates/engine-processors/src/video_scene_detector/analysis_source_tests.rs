use super::*;
use crate::video_source::scene_analysis_size;

/// Swift rounds and then forces even dimensions only when shrinking; a small
/// odd-sized source is kept intact. The platform may honor or ignore the hint.
/// Kill: shrink with the wrong rounding/evenness or retain full pictures when a codec declines scaling.
#[test]
fn scene_decode_requests_the_current_even_size_and_enforces_declined_hints() {
    let shapes = [
        ((3840, 2160), Some((256, 144))),
        ((1080, 1920), Some((256, 454))),
        ((641, 359), Some((256, 142))),
        ((257, 1), Some((256, 2))),
        ((256, 145), None),
        ((255, 143), None),
        ((1, 1), None),
    ];
    for (size, requested) in shapes {
        assert_eq!(scene_analysis_size(size.0, size.1), requested);
        for honor in [false, true] {
            let mut state = State::new(2);
            state.source_size = size;
            state.honor_scaling = honor;
            let state = Arc::new(state);
            let source = VideoSource::new(Arc::new(Video(state.clone())));
            futures_executor::block_on(async {
                let mut stream = source
                    .open_scene_rgb(path(), &state.progress)
                    .await
                    .unwrap();
                assert_eq!(*state.requested_sizes.lock().unwrap(), vec![requested]);
                for index in 0..2 {
                    let frame = stream.next(&state.progress).await.unwrap().unwrap();
                    let expected = requested.unwrap_or(size);
                    assert_eq!(
                        (frame.width, frame.height),
                        expected,
                        "source={size:?}, honor={honor}"
                    );
                    assert_eq!(
                        frame.data.len(),
                        expected.0 as usize * expected.1 as usize * 3
                    );
                    assert!(frame.data.iter().all(|byte| *byte == index));
                    assert!((frame.presentation_time - f64::from(index) / 30.0).abs() < 1e-9);
                    let luma = jump_cut::analysis_luma(&frame).unwrap();
                    assert_eq!(
                        (luma.width, luma.height),
                        (expected.0 as usize, expected.1 as usize)
                    );
                }
                assert!(stream.next(&state.progress).await.unwrap().is_none());
                assert_eq!(state.drops.load(Ordering::SeqCst), 1);
            });
        }
    }
}

/// Unknown geometry cannot allow the first decoded large picture to escape
/// into the snapshot queue; its actual dimensions supply the same policy.
/// Kill: allow unknown probed geometry to bypass the analysis-size bound.
#[test]
fn scene_decode_enforces_analysis_size_when_probe_dimensions_are_absent() {
    let mut state = State::new(1);
    state.source_size = (1920, 1080);
    state.metadata_size = false;
    let state = Arc::new(state);
    let source = VideoSource::new(Arc::new(Video(state.clone())));
    futures_executor::block_on(async {
        let mut stream = source
            .open_scene_rgb(path(), &state.progress)
            .await
            .unwrap();
        assert_eq!(*state.requested_sizes.lock().unwrap(), vec![None]);
        let frame = stream.next(&state.progress).await.unwrap().unwrap();
        assert_eq!(
            (frame.width, frame.height, frame.data.len()),
            (256, 144, 256 * 144 * 3)
        );
    });
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// A backend can report the right width while delivering a different height.
/// Rechecking width alone would retain a frame unlike the one iOS requested.
/// Kill: accept a codec width-only resize with the wrong retained height.
#[test]
fn a_partly_honored_decode_hint_is_still_the_exact_requested_picture() {
    let mut state = State::new(1);
    state.source_size = (641, 359);
    state.decoded_size = Some((256, 145));
    let state = Arc::new(state);
    let source = VideoSource::new(Arc::new(Video(state.clone())));
    futures_executor::block_on(async {
        let mut stream = source
            .open_scene_rgb(path(), &state.progress)
            .await
            .unwrap();
        let frame = stream.next(&state.progress).await.unwrap().unwrap();
        assert_eq!(
            *state.requested_sizes.lock().unwrap(),
            vec![Some((256, 142))]
        );
        assert_eq!(
            (frame.width, frame.height, frame.data.len()),
            (256, 142, 256 * 142 * 3)
        );
    });
}

/// Current Swift refuses unknown cadence before opening AVAssetReader. Neither
/// production streaming nor the buffered profiler may invent 30fps identities.
/// Kill: invent30 FPS or start decoder/model work for an invalid scene cadence.
#[test]
fn both_scene_paths_refuse_unknown_or_zero_cadence_before_decoder_or_model_admission() {
    for rate in [None, Some((0, 1)), Some((30, 0)), Some((0, 0))] {
        for streaming in [false, true] {
            let mut state = State::new(120);
            state.declared_rate = rate;
            state.allow_identity = true;
            let state = Arc::new(state);
            let processor = detector(state.clone());
            let error = futures_executor::block_on(async {
                if streaming {
                    processor
                        .detect_streaming(path(), state.progress.clone())
                        .await
                        .unwrap_err()
                } else {
                    processor.decode_for_detection(path()).await.unwrap_err()
                }
            });
            assert!(
                matches!(error, EngineError::InvalidConfiguration(ref message)
                if message == "Video has zero/unknown frame rate")
            );
            assert_eq!(state.probes.load(Ordering::SeqCst), 1);
            assert_eq!(state.opens.load(Ordering::SeqCst), 0);
            assert_eq!(state.reads.load(Ordering::SeqCst), 0);
            assert_eq!(state.models.load(Ordering::SeqCst), 0);
        }
    }
}

/// The pre-extracted scene entry cannot emit infinite or NaN scene timestamps.
/// Kill: accept zero, negative, infinite or NaN cadence in the preextracted-frame entry.
#[test]
fn preextracted_scene_frames_require_a_positive_finite_cadence() {
    for rate in [0.0, -30.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let state = Arc::new(State::new(1));
        let error = futures_executor::block_on(detector(state.clone()).detect(
            &frame_bytes(0),
            1,
            rate,
            state.progress.clone(),
        ))
        .unwrap_err();
        assert!(
            matches!(error, EngineError::InvalidConfiguration(ref message)
            if message == "Video has zero/unknown frame rate")
        );
        assert_eq!(state.models.load(Ordering::SeqCst), 0);
    }
}

/// Generic video readers are used by unrelated processors. The scene policy
/// must not shrink their pixels or replace their timestamp-derived fallback.
/// Kill: leak the scene-only size/cadence restriction into unrelated video consumers.
#[test]
fn other_rgb_consumers_keep_full_size_and_measured_cadence() {
    let mut state = State::new(3);
    state.source_size = (640, 360);
    state.declared_rate = None;
    state.pts_rate = 24;
    let state = Arc::new(state);
    let source = VideoSource::new(Arc::new(Video(state.clone())));
    futures_executor::block_on(async {
        let mut stream = source.open_rgb(path(), &state.progress).await.unwrap();
        assert_eq!(stream.frame_rate(), 30.0);
        for index in 0..3 {
            let frame = stream.next(&state.progress).await.unwrap().unwrap();
            assert_eq!(
                (frame.width, frame.height, frame.data.len()),
                (640, 360, 640 * 360 * 3)
            );
            assert!(frame.data.iter().all(|byte| usize::from(*byte) == index));
        }
        assert!(stream.next(&state.progress).await.unwrap().is_none());
        assert_eq!(stream.frame_rate(), 24.0);
    });
    assert!(state.requested_sizes.lock().unwrap().is_empty());
    assert_eq!(state.opens.load(Ordering::SeqCst), 1);
    assert_eq!(state.drops.load(Ordering::SeqCst), 1);
}

/// Both high-level paths must use the new scene reader. Exact frame order and
/// declared fractional cadence still determine every cut and scene timestamp.
/// Kill: skip frames or reconstruct a fractional source cadence as30 FPS.
#[test]
fn both_scene_entries_keep_every_frame_and_the_declared_fractional_cadence() {
    let mut state = State::new(121);
    state.source_size = (641, 359);
    state.declared_rate = Some((30_000, 1001));
    state.allow_identity = true;
    let state = Arc::new(state);
    let processor = detector(state.clone());
    futures_executor::block_on(async {
        let streamed = processor
            .detect_streaming_with_options(path(), false, false, state.progress.clone())
            .await
            .unwrap();
        let decoded = processor.decode_for_detection(path()).await.unwrap();
        assert_eq!(
            *state.requested_sizes.lock().unwrap(),
            vec![Some((256, 142)); 2]
        );
        assert_eq!(decoded.frame_count, 121);
        assert_eq!(
            decoded.frames,
            (0..121).flat_map(frame_bytes).collect::<Vec<_>>()
        );
        assert_eq!(decoded.frame_rate, 30_000.0 / 1001.0);
        let buffered = processor
            .detect(
                &decoded.frames,
                decoded.frame_count,
                decoded.frame_rate,
                ProgressSink::noop(),
            )
            .await
            .unwrap();
        assert_eq!(streamed.total_frames, 121);
        assert_eq!(streamed.frame_rate, buffered.frame_rate);
        assert_eq!(streamed.scenes, buffered.scenes);
        assert_eq!(
            streamed
                .cuts
                .iter()
                .map(|cut| (cut.frame, cut.time, cut.confidence))
                .collect::<Vec<_>>(),
            buffered
                .cuts
                .iter()
                .map(|cut| (cut.frame, cut.time, cut.confidence))
                .collect::<Vec<_>>()
        );
        assert_eq!(state.reads.load(Ordering::SeqCst), 244);
        assert_eq!(state.drops.load(Ordering::SeqCst), 2);
    });
}
