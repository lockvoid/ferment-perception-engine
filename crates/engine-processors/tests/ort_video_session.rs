//! `EdgeTamSession` on a real video through the real graphs — the four ONNX
//! masters `ferment-models-export` publishes (`models/edgetam`).
//!
//! The FIRST prompt on a session needs the image encoder, the prompt/mask
//! decoder and the memory encoder; memory attention only conditions a TRACKING
//! frame, which propagation walks. This is the only thing that executes
//! `session.rs`, the video preprocessing resize, and `store_candidate` at all.
//!
//! ```text
//! FERMENT_ENGINE_ONNX_DIR      the directory holding edgetam_v1_*.onnx
//! FERMENT_ENGINE_FIXTURES_DIR  FermentEngine/Tests/Fixtures
//! ORT_DYLIB_PATH               the ONNX Runtime
//! ```

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_core::{
    NormalizedVideoPoint, VideoObjectId, VideoObjectPointLabel, VideoObjectPointPrompt,
    VideoObjectPrompt, VideoObjectSegmentationSession, VideoObjectSegmenting,
};
use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_processors::EdgeTam;
use ferment_engine_processors::edgetam::model::EdgeTamModel;
use ferment_engine_processors::video_backend;

const FIXTURE: &str = "rotated-48x32.mp4";
/// The picture as it is shown: the fixture is coded 48×32 with a 90° display
/// rotation, and the frame source works in display orientation.
const WIDTH: usize = 32;
const HEIGHT: usize = 48;
const FRAMES: usize = 30;

/// The model behind the segmenter, for the tests that need the CONCRETE
/// session rather than the trait object, with every graph but `left_out`.
fn model_without(left_out: Option<ModelId>) -> EdgeTamModel {
    let dir = std::env::var_os("FERMENT_ENGINE_ONNX_DIR")
        .or_else(|| std::env::var_os("FERMENT_ENGINE_MODELS_DIR"))
        .map(PathBuf::from)
        .expect("checked by setup");
    let mut paths = ModelPaths::new();
    for id in EdgeTamModel::MODEL_FILES
        .into_iter()
        .filter(|id| Some(*id) != left_out)
    {
        paths.insert(
            id,
            ModelVersion::Latest,
            dir.join(format!("{}.onnx", id.name())),
        );
    }
    EdgeTamModel::new(Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    )))
}

fn setup(test: &str) -> Option<(EdgeTam, PathBuf)> {
    let models = std::env::var_os("FERMENT_ENGINE_ONNX_DIR")
        .or_else(|| std::env::var_os("FERMENT_ENGINE_MODELS_DIR"))
        .map(PathBuf::from)?;
    let missing: Vec<String> = EdgeTamModel::MODEL_FILES
        .iter()
        .map(|id| format!("{}.onnx", id.name()))
        .filter(|name| !models.join(name).exists())
        .collect();
    if !missing.is_empty() {
        println!(
            "SKIP {test}: {} lacks {}",
            models.display(),
            missing.join(", ")
        );
        return None;
    }
    let Some(fixtures) = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").map(PathBuf::from) else {
        println!("SKIP {test}: set FERMENT_ENGINE_FIXTURES_DIR");
        return None;
    };
    let video = fixtures.join("EdgeTAM").join(FIXTURE);
    if !video.exists() {
        println!("SKIP {test}: no fixture at {}", video.display());
        return None;
    }

    let mut paths = ModelPaths::new();
    for id in EdgeTamModel::MODEL_FILES {
        paths.insert(
            id,
            ModelVersion::Latest,
            models.join(format!("{}.onnx", id.name())),
        );
    }
    Some((
        EdgeTam::new(
            Arc::new(OrtBackend::with_execution_provider(
                Arc::new(paths),
                OrtExecutionProvider::Cpu,
            )),
            video_backend(),
        ),
        video,
    ))
}

fn click(x: f32, y: f32) -> VideoObjectPrompt {
    VideoObjectPrompt::Points(vec![VideoObjectPointPrompt::new(
        NormalizedVideoPoint::new(x, y),
        VideoObjectPointLabel::Foreground,
    )])
}

async fn session(segmenter: &EdgeTam, video: &Path) -> Box<dyn VideoObjectSegmentationSession> {
    VideoObjectSegmenting::make_session(segmenter, video)
        .await
        .expect("video session")
}

/// A prompt on a video frame returns a mask AT THAT FRAME, at the source's own
/// dimensions and presentation time.
///
/// The whole prompt path runs here for the first time: decode → the video
/// resize to 1024² → the image encoder → the prompt/mask decoder → candidate
/// selection → the memory encoder → the bilinear resize back to 32×48. The
/// three things asserted are all facts of the FIXTURE (ffprobe: 48×32 rotated
/// 90°, so 32×48 on screen; 30 fps, 30 frames), not of this code.
///
/// kill: pass `sourceInfo.height`/`width` to `MaskWorkspace` in the wrong order
/// and the mask comes back 48×32; drop the `presentation_time` off the encoded
/// frame and every mask claims frame 0's timestamp.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn a_prompt_on_a_video_frame_answers_at_that_frame() {
    let Some((segmenter, video)) = setup("a_prompt_on_a_video_frame_answers_at_that_frame") else {
        return;
    };
    futures_executor::block_on(async {
        let info = session(&segmenter, &video).await.source_info();
        assert_eq!(info.frame_count, FRAMES);
        assert_eq!((info.width, info.height), (WIDTH, HEIGHT));

        for frame_index in [0_usize, 7, 29] {
            // A fresh session per frame: the FIRST prompt is the path that
            // needs no memory attention, so each frame gets its own.
            let mut session = session(&segmenter, &video).await;
            let frame = session
                .apply(
                    &click(0.5, 0.5),
                    &VideoObjectId::new("subject"),
                    frame_index,
                    true,
                )
                .await
                .unwrap_or_else(|error| panic!("apply at {frame_index}: {error}"));

            assert_eq!(frame.frame_index, frame_index);
            assert_eq!(
                (frame.mask.width(), frame.mask.height()),
                (WIDTH, HEIGHT),
                "the mask must be the SOURCE's size, not the model's"
            );
            assert_eq!(frame.mask.bytes().len(), WIDTH * HEIGHT);
            assert!(frame.mask.bytes().iter().all(|b| *b == 0 || *b == 255));
            // 30 fps: frame i is at i/30 s. The fixture's own rate.
            let expected = frame_index as f64 / 30.0;
            assert!(
                (frame.presentation_time - expected).abs() < 0.02,
                "frame {frame_index} claims {} s, the fixture puts it at {expected} s",
                frame.presentation_time
            );
            assert!(frame.predicted_iou.is_some_and(f32::is_finite));
        }
    });
}

/// EdgeTAM V1 tracks ONE object per video session; a second id is refused with
/// the reason, not silently tracked or silently ignored.
///
/// kill: drop the `objects.keys().next().is_some_and(...)` guard in
/// `EdgeTamSession::apply` and the second object is accepted, so a session
/// carries two temporal states through a bank sized for one.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn a_second_object_is_refused_by_the_v1_profile() {
    let Some((segmenter, video)) = setup("a_second_object_is_refused_by_the_v1_profile") else {
        return;
    };
    futures_executor::block_on(async {
        let mut session = session(&segmenter, &video).await;
        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("first"), 0, true)
            .await
            .expect("the first object is accepted");

        let error = session
            .apply(&click(0.25, 0.25), &VideoObjectId::new("second"), 0, true)
            .await
            .expect_err("a second object must be refused")
            .to_string();
        assert!(error.contains("second"), "{error}");
        assert!(error.contains("one tracked object"), "{error}");

        // The FIRST object is still usable — the refusal must not have
        // corrupted the session.
        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("first"), 0, true)
            .await
            .expect("the first object still works after the refusal");
    });
}

/// A frame index outside the video is refused, and refused BEFORE the
/// session's other preconditions are tested.
///
/// kill: drop `EdgeTamSession::validate` from `propagate` — an out-of-range
/// frame on an unprompted session then reports "apply at least one prompt"
/// instead of naming the range, so the caller fixes the wrong thing.
///
/// NOT a kill, measured: dropping it from `apply`. `EdgeTamFrameSource::validate`
/// raises the SAME `InvalidFrameIndex` with the same range, so the two guards
/// are observationally identical there — `apply`'s is a cheap early-out and
/// iOS's own `validate(frameIndex:)`, not a second contract.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn a_frame_outside_the_video_is_refused_with_the_range() {
    let Some((segmenter, video)) = setup("a_frame_outside_the_video_is_refused_with_the_range")
    else {
        return;
    };
    futures_executor::block_on(async {
        let mut session = session(&segmenter, &video).await;
        for index in [FRAMES, FRAMES + 1, usize::MAX] {
            let error = session
                .apply(
                    &click(0.5, 0.5),
                    &VideoObjectId::new("subject"),
                    index,
                    true,
                )
                .await
                .expect_err("an out-of-range frame must be refused")
                .to_string();
            assert!(
                error.contains(&format!("0..<{FRAMES}")),
                "the refusal must name the range: {error}"
            );
        }
        struct Sink;
        impl ferment_engine_core::VideoObjectMaskSink for Sink {
            fn emit(&self, _: ferment_engine_core::VideoObjectMaskFrame) -> bool {
                panic!("nothing should be emitted");
            }
        }
        use ferment_engine_core::VideoObjectPropagationDirection::Forward;

        // Propagation with nothing prompted is its own refusal…
        let error = session
            .propagate(0, Forward, &Sink)
            .await
            .expect_err("propagation with no prompted object must be refused")
            .to_string();
        assert!(error.contains("at least one"), "{error}");

        // …but an out-of-range frame is caught FIRST, so the caller is told the
        // thing that is actually wrong. This is what the session's own
        // `validate` buys, and the only place it is observable.
        let error = session
            .propagate(FRAMES, Forward, &Sink)
            .await
            .expect_err("an out-of-range propagation must be refused")
            .to_string();
        assert!(
            error.contains(&format!("0..<{FRAMES}")),
            "the range must be named before the missing prompt: {error}"
        );
    });
}

/// `reset` clears the prompts and the object map, and the session is reusable
/// afterwards — including for a DIFFERENT object id, which the V1 one-object
/// rule would otherwise refuse forever.
///
/// kill: make `reset` leave `objects` populated and the second id below is
/// refused by the V1 guard, which is exactly the bug a caller would report as
/// "I can never re-prompt after resetting".
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn reset_frees_the_session_for_a_different_object() {
    let Some((segmenter, video)) = setup("reset_frees_the_session_for_a_different_object") else {
        return;
    };
    futures_executor::block_on(async {
        let mut session = session(&segmenter, &video).await;
        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("first"), 0, true)
            .await
            .expect("first");
        assert!(
            session
                .apply(&click(0.5, 0.5), &VideoObjectId::new("second"), 0, true)
                .await
                .is_err(),
            "the V1 guard must be active before the reset"
        );

        session.reset().await;

        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("second"), 0, true)
            .await
            .expect("after a reset a different object is accepted");
    });
}

/// A propagation that FAILS still releases the decoder.
///
/// iOS frees it after the do/catch, so success, cancellation and error all
/// release (`EdgeTAMSession.swift:468-474`). Every `?` inside the walk used to
/// return with the decoder and its codec context still open until the session
/// itself dropped — invisible to any test that only checks the error.
///
/// The failure is forced by leaving memory attention out of the session's
/// fleet: a tracking frame calls it, so a propagation off the prompted frame
/// fails on the first step, which is exactly the error path being graded.
///
/// kill: put `cancel_decoder_resources` back inside the walk (on the sink-stop
/// and loop-end paths only) and the decoder is still open after this error.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn a_failed_propagation_still_releases_the_decoder() {
    let Some((_segmenter, video)) = setup("a_failed_propagation_still_releases_the_decoder") else {
        return;
    };
    struct Sink;
    impl ferment_engine_core::VideoObjectMaskSink for Sink {
        fn emit(&self, _: ferment_engine_core::VideoObjectMaskFrame) -> bool {
            true
        }
    }
    // The CONCRETE session, so `memory_snapshot` — iOS's own runtime evidence
    // hook, and until now dead accounting with no caller — can answer.
    futures_executor::block_on(async {
        let source = ferment_engine_processors::edgetam::frame_source::EdgeTamFrameSource::new(
            video_backend(),
            &video,
        )
        .await
        .expect("frame source");
        let mut session = ferment_engine_processors::edgetam::session::EdgeTamSession::new(
            model_without(Some(ModelId::EdgetamMemoryAttention)),
            source,
        );
        assert!(!session.memory_snapshot().has_decoder_resources);

        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("subject"), 0, true)
            .await
            .expect("the prompt");
        let after_prompt = session.memory_snapshot();
        assert_eq!(after_prompt.prompted_frame_count, 1);
        assert_eq!(after_prompt.stored_frame_count, 1);
        assert_eq!(after_prompt.prompt_frame_budget_remaining, 31);
        assert!(
            after_prompt.has_decoder_resources,
            "the prompt opened a decoder"
        );

        let error = session
            .propagate(
                0,
                ferment_engine_core::VideoObjectPropagationDirection::Forward,
                &Sink,
            )
            .await
            .expect_err("propagation needs memory attention, which this fleet lacks");
        // The frame AFTER the prompted one is the first tracking frame, so the
        // walk got as far as opening the decoder before it failed.
        assert!(
            error.to_string().contains("edgetam_v1_memory_attention"),
            "the failure must be the missing graph, not something else: {error}"
        );
        assert!(
            !session.memory_snapshot().has_decoder_resources,
            "a failed propagation must still release the decoder"
        );
    });
}

/// A prompted object is tracked through every later frame: one mask a frame,
/// in order, at the source's size — the memory encoder and memory attention
/// in the loop.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH), the EdgeTAM masters and FERMENT_ENGINE_FIXTURES_DIR"]
fn propagation_tracks_the_object_through_every_later_frame() {
    let Some((segmenter, video)) = setup("propagation_tracks_the_object_through_every_later_frame")
    else {
        return;
    };
    struct Frames(std::sync::Mutex<Vec<(usize, usize, usize)>>);
    impl ferment_engine_core::VideoObjectMaskSink for Frames {
        fn emit(&self, frame: ferment_engine_core::VideoObjectMaskFrame) -> bool {
            self.0.lock().unwrap().push((
                frame.frame_index,
                frame.mask.width(),
                frame.mask.height(),
            ));
            true
        }
    }
    let frames = Frames(std::sync::Mutex::new(Vec::new()));
    futures_executor::block_on(async {
        let mut session = session(&segmenter, &video).await;
        session
            .apply(&click(0.5, 0.5), &VideoObjectId::new("subject"), 0, true)
            .await
            .expect("the prompt");
        session
            .propagate(
                0,
                ferment_engine_core::VideoObjectPropagationDirection::Forward,
                &frames,
            )
            .await
            .expect("propagation");
    });
    let frames = frames.0.into_inner().unwrap();
    assert_eq!(
        frames.iter().map(|frame| frame.0).collect::<Vec<_>>(),
        (0..FRAMES).collect::<Vec<_>>()
    );
    assert!(
        frames
            .iter()
            .all(|frame| (frame.1, frame.2) == (WIDTH, HEIGHT))
    );
}

/// Kill: quantize rational time in cached encoded features, prompt correction,
/// stored temporal outputs, or public delivery after reverse-state reuse.
#[cfg(not(target_os = "android"))]
#[test]
#[ignore = "real ORT video session and a backend exposing original rational timestamps"]
fn rational_time_survives_real_prompt_correction_tracking_and_reverse_delivery() {
    use ferment_engine_core::{
        VideoObjectMaskFrame, VideoObjectMaskSink,
        VideoObjectPropagationDirection::{Forward, Reverse},
    };
    struct Times(std::sync::Mutex<Vec<(usize, f64)>>);
    impl VideoObjectMaskSink for Times {
        fn emit(&self, frame: VideoObjectMaskFrame) -> bool {
            let mut frames = self.0.lock().unwrap();
            frames.push((frame.frame_index, frame.presentation_time));
            frames.len() < 2
        }
    }
    let (segmenter, video) = setup("rational-time-real-session")
        .expect("all four real EdgeTAM masters and rotated fixture are required");
    futures_executor::block_on(async {
        let mut session = session(&segmenter, &video).await;
        let id = VideoObjectId::new("exact-time");
        let initial = session.apply(&click(0.5, 0.5), &id, 7, true).await.unwrap();
        let corrected = session
            .apply(&click(0.6, 0.6), &id, 7, false)
            .await
            .unwrap();
        for frame in [initial, corrected] {
            assert!((frame.presentation_time - 7.0 / 30.0).abs() < 1e-9);
        }
        let forward = Times(std::sync::Mutex::new(Vec::new()));
        session.propagate(7, Forward, &forward).await.unwrap();
        assert_eq!(
            forward
                .0
                .lock()
                .unwrap()
                .iter()
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            [7, 8]
        );
        let reverse = Times(std::sync::Mutex::new(Vec::new()));
        session.propagate(8, Reverse, &reverse).await.unwrap();
        assert_eq!(
            reverse
                .0
                .lock()
                .unwrap()
                .iter()
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            [8, 7]
        );
        for records in [&forward, &reverse] {
            for &(index, time) in records.0.lock().unwrap().iter() {
                assert!(
                    (time - index as f64 / 30.0).abs() < 1e-9,
                    "frame {index}: {time:.12}"
                );
            }
        }
        session.reset().await;
    });
}
