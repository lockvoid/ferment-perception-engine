//! The current four-graph lifecycle assertion, through the real session driver.
use super::*;
use crate::{edgetam::EdgeTam, video_backend};
use ferment_engine_core::{
    ImageObjectSegmentationResult, ImageObjectSegmenting, NormalizedVideoPoint, RgbFrameRef,
    VideoObjectMask, VideoObjectPointLabel, VideoObjectPointPrompt,
};
use ferment_engine_infer::{
    Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
};
use futures_executor::block_on;
use std::{
    path::PathBuf,
    sync::{Arc, Barrier, Mutex},
    time::{Duration, Instant},
};

fn fixture(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").expect("required original fixtures"),
    )
    .join(name);
    assert!(
        path.is_file(),
        "required original fixture {}",
        path.display()
    );
    path
}
fn backend() -> Arc<OrtBackend> {
    let root = PathBuf::from(
        std::env::var_os("FERMENT_ENGINE_ONNX_DIR").expect("required pinned EdgeTAM ONNX graphs"),
    );
    let mut paths = ModelPaths::new();
    for model in EdgeTamModel::MODEL_FILES {
        let path = root.join(format!("{}.onnx", model.name()));
        assert!(path.is_file(), "required pinned model {}", path.display());
        paths = paths.with_path(model, ModelVersion::Latest, path);
    }
    Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ))
}
fn prompt(x: f32, y: f32, label: VideoObjectPointLabel) -> VideoObjectPrompt {
    VideoObjectPrompt::Points(vec![VideoObjectPointPrompt::new(
        NormalizedVideoPoint::new(x, y),
        label,
    )])
}
fn valid(mask: &VideoObjectMask, iou: Option<f32>, width: usize, height: usize) {
    assert_eq!((mask.width(), mask.height()), (width, height));
    assert_eq!(mask.bytes().len(), width * height);
    assert!(mask.bytes().iter().all(|b| matches!(b, 0 | 255)));
    if let Some(iou) = iou {
        assert!(iou.is_finite());
    }
}
#[derive(Default)]
struct FirstTwo(Mutex<Vec<VideoObjectMaskFrame>>);
impl VideoObjectMaskSink for FirstTwo {
    fn emit(&self, frame: VideoObjectMaskFrame) -> bool {
        let mut values = self.0.lock().unwrap();
        values.push(frame);
        values.len() < 2
    }
}

/// Kill: leak session state/decoder on reset, change the first propagated mask,
/// bypass the shared native permit, or admit concurrent predictions on one graph.
#[test]
#[ignore = "requires real four pinned ONNX graphs, original sec1, and platform video decode"]
fn real_video_and_four_concurrent_image_sessions_release_owned_state() {
    let backend = backend();
    let model = EdgeTamModel::new(backend.clone() as Backend);
    let source = block_on(EdgeTamFrameSource::new(
        video_backend(),
        &fixture("reel-1/sec1.mp4"),
    ))
    .unwrap();
    let mut session = EdgeTamSession::new(model, source);
    let info = session.source_info();
    let object = VideoObjectId::new("smoke-object");
    let fg = prompt(0.5, 0.5, VideoObjectPointLabel::Foreground);
    let prompted = block_on(session.apply(&fg, &object, 0, true)).unwrap();
    valid(
        &prompted.mask,
        prompted.predicted_iou,
        info.width,
        info.height,
    );
    assert!(prompted.presentation_time.is_finite());
    let after_prompt = session.memory_snapshot();
    assert!(after_prompt.preprocessing_workspace_byte_count > 0);
    let output = FirstTwo::default();
    // Inspect the real driver before the public callback wrapper performs its
    // immediate final drain. This is the same ownership point as Swift's
    // retained stream after its consumer stops at frame 1; native's public
    // callback API otherwise releases the reader earlier, on sink=false.
    block_on(session.run_propagation(
        0,
        VideoObjectPropagationDirection::Forward,
        &output,
        &ProgressSink::noop(),
    ))
    .unwrap();
    let before = session.memory_snapshot();
    assert_eq!(before.prompted_frame_count, 1);
    assert!(before.stored_frame_count >= 2);
    assert_eq!(before.prompt_frame_budget_remaining, 31);
    assert!(
        before.has_cached_encoded_features
            && before.has_temporal_tensor_bank
            && before.has_decoder_resources
    );
    assert_eq!(
        before.preprocessing_workspace_byte_count,
        after_prompt.preprocessing_workspace_byte_count
    );
    let start = Instant::now();
    block_on(session.reset());
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(
        session.memory_snapshot(),
        MemorySnapshot {
            prompted_frame_count: 0,
            stored_frame_count: 0,
            prompt_frame_budget_remaining: 32,
            has_cached_encoded_features: false,
            has_temporal_tensor_bank: false,
            preprocessing_workspace_byte_count: before.preprocessing_workspace_byte_count,
            has_decoder_resources: false,
        }
    );
    let propagated = output.0.lock().unwrap();
    assert_eq!(
        propagated.iter().map(|f| f.frame_index).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(propagated[0], prompted);
    for frame in propagated.iter() {
        valid(&frame.mask, frame.predicted_iou, info.width, info.height);
        assert!(frame.presentation_time.is_finite());
    }
    drop(propagated);
    drop(session);

    let (width, height) = (96_u32, 64_u32);
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    (x * 255 / (width - 1)) as u8,
                    (y * 255 / (height - 1)) as u8,
                    ((x + y) % 256) as u8,
                ]
            })
        })
        .collect();
    let processor = EdgeTam::new(backend.clone() as Backend, video_backend());
    let mut image = block_on(ImageObjectSegmenting::make_session(
        &processor,
        RgbFrameRef::new(&pixels, width, height).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        (image.source_info().width, image.source_info().height),
        (96, 64)
    );
    let id = VideoObjectId::new("image-smoke-object");
    let first = block_on(image.apply(&fg, &id, true)).unwrap();
    valid(&first.mask, first.predicted_iou, 96, 64);
    let corrected = block_on(image.apply(
        &prompt(0.15, 0.15, VideoObjectPointLabel::Background),
        &id,
        false,
    ))
    .unwrap();
    valid(&corrected.mask, corrected.predicted_iou, 96, 64);
    block_on(image.reset());
    drop(image);
    let version = ModelVersion::Latest;
    let encoder_before =
        block_on(backend.prediction_snapshot(ModelId::EdgetamImageEncoder, &version)).unwrap();
    let prompt_before =
        block_on(backend.prediction_snapshot(ModelId::EdgetamPromptMaskDecoder, &version)).unwrap();
    let start = Barrier::new(4);
    let masks: Vec<ImageObjectSegmentationResult> = std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for index in 0..4 {
            let (start, processor, pixels) = (&start, &processor, &pixels);
            joins.push(scope.spawn(move || {
                start.wait();
                block_on(async {
                    let mut session = ImageObjectSegmenting::make_session(
                        processor,
                        RgbFrameRef::new(pixels, width, height).unwrap(),
                    )
                    .await
                    .unwrap();
                    let result = session
                        .apply(
                            &prompt(0.5, 0.5, VideoObjectPointLabel::Foreground),
                            &VideoObjectId::new(format!("concurrent-{index}")),
                            true,
                        )
                        .await
                        .unwrap();
                    session.reset().await;
                    result
                })
            }));
        }
        joins.into_iter().map(|join| join.join().unwrap()).collect()
    });
    assert_eq!(masks.len(), 4);
    for result in &masks {
        valid(&result.mask, result.predicted_iou, 96, 64);
    }
    let encoder_after =
        block_on(backend.prediction_snapshot(ModelId::EdgetamImageEncoder, &version)).unwrap();
    let prompt_after =
        block_on(backend.prediction_snapshot(ModelId::EdgetamPromptMaskDecoder, &version)).unwrap();
    assert_eq!(encoder_after.completed - encoder_before.completed, 4);
    assert_eq!(prompt_after.completed - prompt_before.completed, 4);
    for (before, after) in [
        (encoder_before, encoder_after),
        (prompt_before, prompt_after),
    ] {
        assert_eq!(after.peak_active, 1);
        assert_eq!((after.active, after.waiting), (0, 0));
        assert_eq!(after.failed, before.failed);
    }
    assert!(
        encoder_after.peak_waiting > 0,
        "start barrier must drive real overlapping encoder calls"
    );
    println!("EDGETAM_CONCURRENCY sessions=4 encoder={encoder_after:?} prompt={prompt_after:?}");
}
