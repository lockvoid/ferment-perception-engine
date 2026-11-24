//! Current image-session composition/cancellation assertions against the real
//! session state machine and a recording decoder, matching the Swift test seam.
use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, ImageObjectId, ImageObjectSegmentationSession, ImageObjectSegmentationSourceInfo,
    NormalizedVideoPoint, ProgressSink, Result, VideoObjectPointLabel, VideoObjectPointPrompt,
    VideoObjectPrompt,
};
use ferment_engine_processors::edgetam::{
    EdgeTamImageSession,
    model::{EdgeTamDecodeOutput, EdgeTamImageFeatures, EdgeTamPromptDecoding, shape},
    processing::PromptTensor,
};
use futures_executor::block_on;
use half::f16;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Call {
    prompt: PromptTensor,
    no_memory: bool,
    score_gate: bool,
    previous: Option<Vec<f16>>,
}

#[derive(Clone, Default)]
struct Decoder {
    calls: Arc<Mutex<Vec<Call>>>,
    cancel: Arc<Mutex<Option<(usize, ProgressSink)>>>,
}

#[async_trait(?Send)]
impl EdgeTamPromptDecoding for Decoder {
    async fn decode(
        &self,
        _: &EdgeTamImageFeatures,
        prompt: &PromptTensor,
        no_memory: bool,
        score_gate: bool,
        previous: Option<&[f16]>,
    ) -> Result<EdgeTamDecodeOutput> {
        let mut calls = self.calls.lock().unwrap();
        calls.push(Call {
            prompt: prompt.clone(),
            no_memory,
            score_gate,
            previous: previous.map(<[f16]>::to_vec),
        });
        let call = calls.len();
        drop(calls);
        // The cancellation occurs after admission, while decode owns the call,
        // before it returns a valid answer which must never be committed.
        if let Some((at, sink)) = &*self.cancel.lock().unwrap()
            && *at == call
        {
            sink.cancel();
        }
        EdgeTamDecodeOutput::new(
            (0..4)
                .flat_map(|candidate| vec![f16::from_f32((call * 4 + candidate) as f32); 256 * 256])
                .collect(),
            [0.625, 0.5, 0.875, 0.75].map(f16::from_f32).to_vec(),
            vec![f16::ONE; 4 * 256],
            vec![f16::ONE],
        )
    }
}

fn session(decoder: Decoder) -> EdgeTamImageSession<Decoder> {
    let zero = |shape: &[i64]| vec![f16::ZERO; shape::element_count(shape)];
    EdgeTamImageSession::new(
        decoder,
        EdgeTamImageFeatures::new(
            zero(&shape::VISION_FEATURES),
            zero(&shape::HIGH_RESOLUTION_FEATURES_0),
            zero(&shape::HIGH_RESOLUTION_FEATURES_1),
        )
        .unwrap(),
        ImageObjectSegmentationSourceInfo {
            width: 3,
            height: 2,
        },
    )
}

fn point(x: f32, y: f32, label: VideoObjectPointLabel) -> VideoObjectPrompt {
    VideoObjectPrompt::Points(vec![VideoObjectPointPrompt::new(
        NormalizedVideoPoint::new(x, y),
        label,
    )])
}

#[test]
fn cancellation_during_a_decode_does_not_commit_a_new_object() {
    let decoder = Decoder::default();
    let sink = ProgressSink::noop();
    *decoder.cancel.lock().unwrap() = Some((1, sink.clone()));
    let mut session = session(decoder.clone());
    let error = block_on(session.apply_with_cancellation(
        &point(0.5, 0.5, VideoObjectPointLabel::Foreground),
        &ImageObjectId::new("cancelled"),
        true,
        sink,
    ))
    .unwrap_err();
    assert!(matches!(error, EngineError::Cancelled));
    assert_eq!(decoder.calls.lock().unwrap().len(), 1);
    assert_eq!(session.lifecycle_snapshot().active_object_count, 0);
    assert_eq!(session.lifecycle_snapshot().retained_mask_byte_count, 0);
}

#[test]
fn cancellation_during_a_correction_preserves_the_previous_prompt_and_mask() {
    let decoder = Decoder::default();
    let mut session = session(decoder.clone());
    let id = ImageObjectId::new("subject");
    block_on(session.apply(
        &point(0.25, 0.75, VideoObjectPointLabel::Foreground),
        &id,
        true,
    ))
    .unwrap();
    let sink = ProgressSink::noop();
    *decoder.cancel.lock().unwrap() = Some((2, sink.clone()));
    assert!(matches!(
        block_on(session.apply_with_cancellation(
            &point(0.5, 0.5, VideoObjectPointLabel::Background),
            &id,
            false,
            sink
        )),
        Err(EngineError::Cancelled)
    ));
    assert_eq!(session.lifecycle_snapshot().active_object_count, 1);
    block_on(session.apply(
        &point(0.75, 0.25, VideoObjectPointLabel::Background),
        &id,
        false,
    ))
    .unwrap();
    let calls = decoder.calls.lock().unwrap();
    assert_eq!(calls[2].prompt.coordinates, [256.0, 768.0, 768.0, 256.0]);
    assert_eq!(calls[2].prompt.labels, [1, 0]);
    assert_eq!(
        calls[2].previous.as_ref().unwrap(),
        &vec![f16::from_f32(6.0); 256 * 256]
    );
}

#[test]
fn a_pre_cancelled_call_never_enters_the_decoder() {
    let decoder = Decoder::default();
    let mut session = session(decoder.clone());
    let sink = ProgressSink::noop();
    sink.cancel();
    assert!(matches!(
        block_on(session.apply_with_cancellation(
            &point(0.5, 0.5, VideoObjectPointLabel::Foreground),
            &ImageObjectId::new("cancelled"),
            true,
            sink
        )),
        Err(EngineError::Cancelled)
    ));
    assert!(decoder.calls.lock().unwrap().is_empty());
    assert_eq!(session.lifecycle_snapshot().active_object_count, 0);
}

#[test]
fn image_corrections_use_stock_flags_candidate_zero_and_the_selected_previous_mask() {
    let decoder = Decoder::default();
    let mut session = session(decoder.clone());
    let id = ImageObjectId::new("subject");
    let initial = block_on(session.apply(
        &point(0.25, 0.75, VideoObjectPointLabel::Foreground),
        &id,
        true,
    ))
    .unwrap();
    let corrected = block_on(session.apply(
        &point(0.5, 0.125, VideoObjectPointLabel::Background),
        &id,
        false,
    ))
    .unwrap();
    let calls = decoder.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].prompt.coordinates, [256.0, 768.0]);
    assert_eq!(calls[0].prompt.labels, [1]);
    assert!(calls[0].previous.is_none());
    assert_eq!(calls[1].prompt.coordinates, [256.0, 768.0, 512.0, 128.0]);
    assert_eq!(calls[1].prompt.labels, [1, 0]);
    assert_eq!(
        calls[1].previous.as_ref().unwrap(),
        &vec![f16::from_f32(6.0); 256 * 256]
    );
    assert!(calls.iter().all(|call| call.no_memory && !call.score_gate));
    assert_eq!(initial.object_id, id);
    assert_eq!(corrected.object_id, id);
    assert_eq!((initial.mask.width(), initial.mask.height()), (3, 2));
    assert_eq!(initial.predicted_iou, Some(0.875));
    assert_eq!(corrected.predicted_iou, Some(0.625));
}

#[test]
fn object_ids_own_independent_prompts_and_masks_and_reset_keeps_the_image_reusable() {
    let decoder = Decoder::default();
    let mut session = session(decoder.clone());
    let first = ImageObjectId::new("first");
    let second = ImageObjectId::new("second");
    block_on(session.apply(
        &point(0.125, 0.25, VideoObjectPointLabel::Foreground),
        &first,
        true,
    ))
    .unwrap();
    block_on(session.apply(
        &point(0.75, 0.875, VideoObjectPointLabel::Foreground),
        &second,
        false,
    ))
    .unwrap();
    block_on(session.apply(
        &point(0.5, 0.625, VideoObjectPointLabel::Background),
        &first,
        false,
    ))
    .unwrap();
    {
        let calls = decoder.calls.lock().unwrap();
        assert!(calls[0].previous.is_none() && calls[1].previous.is_none());
        assert_eq!(calls[1].prompt.coordinates, [768.0, 896.0]);
        assert_eq!(calls[2].prompt.coordinates, [128.0, 256.0, 512.0, 640.0]);
        assert_eq!(calls[2].prompt.labels, [1, 0]);
        assert_eq!(
            calls[2].previous.as_ref().unwrap(),
            &vec![f16::from_f32(6.0); 256 * 256]
        );
        assert_ne!(
            calls[2].previous.as_ref().unwrap()[100],
            f16::from_f32(10.0)
        );
    }
    block_on(session.reset());
    assert_eq!(session.lifecycle_snapshot().active_object_count, 0);
    let result = block_on(session.apply(
        &point(0.875, 0.125, VideoObjectPointLabel::Background),
        &first,
        false,
    ))
    .unwrap();
    let calls = decoder.calls.lock().unwrap();
    assert!(calls[3].previous.is_none());
    assert_eq!(calls[3].prompt.coordinates, [896.0, 128.0]);
    assert_eq!(calls[3].prompt.labels, [0]);
    assert!(calls[3].no_memory && !calls[3].score_gate);
    assert_eq!(result.object_id, first);
    assert_eq!(result.predicted_iou, Some(0.875));
    assert_eq!(
        session.source_info(),
        ImageObjectSegmentationSourceInfo {
            width: 3,
            height: 2
        }
    );
    assert_eq!(session.lifecycle_snapshot().active_object_count, 1);
    assert_eq!(
        session.lifecycle_snapshot().retained_mask_byte_count,
        256 * 256 * 2
    );
}

#[test]
fn active_object_budget_is_enforced_before_inference_and_remove_releases_one_slot() {
    let decoder = Decoder::default();
    let mut session = session(decoder.clone());
    let prompt = point(0.5, 0.5, VideoObjectPointLabel::Foreground);
    for index in 0..32 {
        block_on(session.apply(
            &prompt,
            &ImageObjectId::new(format!("object-{index}")),
            true,
        ))
        .unwrap();
    }
    assert_eq!(session.maximum_active_object_count(), 32);
    assert_eq!(session.lifecycle_snapshot().active_object_count, 32);
    assert_eq!(
        session.lifecycle_snapshot().retained_mask_byte_count,
        4 * 1024 * 1024
    );
    let error =
        block_on(session.apply(&prompt, &ImageObjectId::new("overflow"), true)).unwrap_err();
    assert!(error.to_string().contains("33"));
    assert_eq!(decoder.calls.lock().unwrap().len(), 32);
    block_on(session.remove(&ImageObjectId::new("object-0")));
    assert_eq!(session.lifecycle_snapshot().active_object_count, 31);
    assert_eq!(
        session.lifecycle_snapshot().retained_mask_byte_count,
        31 * 256 * 256 * 2
    );
    block_on(session.apply(&prompt, &ImageObjectId::new("replacement"), true)).unwrap();
    assert_eq!(session.lifecycle_snapshot().active_object_count, 32);
}
