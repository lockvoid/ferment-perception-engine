//! EdgeTAM's still-image capability, through the real two-graph export.
//!
//! The gesture is the source (FLEET_LAW §1): a foreground click lands on an
//! object, and the mask that comes back must be THAT object — measured against
//! pixels this test wrote, never against the model's own output. The image
//! session needs only the image encoder and the prompt/mask decoder, which is
//! why this can run while the video path's two memory graphs cannot (report §8).
//!
//! ```text
//! FERMENT_ENGINE_ONNX_DIR   the directory holding edgetam_v1_*.onnx
//! ORT_DYLIB_PATH            the ONNX Runtime
//! ```

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferment_engine_core::{
    ImageObjectSegmenting, NormalizedVideoPoint, RgbFrameRef, VideoObjectId, VideoObjectMask,
    VideoObjectPointLabel, VideoObjectPointPrompt, VideoObjectPrompt,
};
use ferment_engine_infer::{ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_media::UnavailableVideoIo;
use ferment_engine_processors::EdgeTam;
use ferment_engine_processors::edgetam::model::EdgeTamModel;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
/// The disc's centre and radius, in pixels. Everything outside it is background.
const CENTRE: (f64, f64) = (0.35 * WIDTH as f64, 0.55 * HEIGHT as f64);
const RADIUS: f64 = 60.0;

fn models(test: &str) -> Option<PathBuf> {
    let directory = std::env::var_os("FERMENT_ENGINE_ONNX_DIR")
        .or_else(|| std::env::var_os("FERMENT_ENGINE_MODELS_DIR"))
        .map(PathBuf::from)?;
    let missing: Vec<String> = EdgeTamModel::MODEL_FILES
        .iter()
        .take(2)
        .map(|id| format!("{}.onnx", id.name()))
        .filter(|name| !directory.join(name).exists())
        .collect();
    if missing.is_empty() {
        Some(directory)
    } else {
        println!(
            "SKIP {test}: {} lacks {}",
            directory.display(),
            missing.join(", ")
        );
        None
    }
}

fn edgetam_model(directory: &Path) -> EdgeTamModel {
    let mut paths = ModelPaths::new();
    for id in EdgeTamModel::MODEL_FILES {
        paths.insert(
            id,
            ModelVersion::Latest,
            directory.join(format!("{}.onnx", id.name())),
        );
    }
    EdgeTamModel::new(Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    )))
}

fn segmenter(directory: &Path) -> EdgeTam {
    let mut paths = ModelPaths::new();
    for id in EdgeTamModel::MODEL_FILES {
        paths.insert(
            id,
            ModelVersion::Latest,
            directory.join(format!("{}.onnx", id.name())),
        );
    }
    EdgeTam::new(
        Arc::new(OrtBackend::with_execution_provider(
            Arc::new(paths),
            OrtExecutionProvider::Cpu,
        )),
        Arc::new(UnavailableVideoIo),
    )
}

/// A bright disc on a dark field, in packed RGB24. The TEST owns every pixel,
/// so "inside the disc" is a fact rather than the model's opinion.
fn disc_image() -> Vec<u8> {
    let mut rgb = vec![0_u8; WIDTH * HEIGHT * 3];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let inside = inside_disc(x, y);
            let offset = (y * WIDTH + x) * 3;
            let (r, g, b) = if inside { (245, 180, 40) } else { (18, 22, 30) };
            rgb[offset] = r;
            rgb[offset + 1] = g;
            rgb[offset + 2] = b;
        }
    }
    rgb
}

fn inside_disc(x: usize, y: usize) -> bool {
    let dx = x as f64 + 0.5 - CENTRE.0;
    let dy = y as f64 + 0.5 - CENTRE.1;
    dx * dx + dy * dy <= RADIUS * RADIUS
}

/// The fraction of the disc the mask covers, and the fraction of the background
/// it wrongly claims — both against the TEST's own geometry.
fn coverage(mask: &VideoObjectMask) -> (f64, f64) {
    assert_eq!((mask.width(), mask.height()), (WIDTH, HEIGHT));
    let (mut inside, mut inside_hit, mut outside, mut outside_hit) = (0, 0, 0, 0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let foreground = mask.bytes()[y * WIDTH + x] == 255;
            if inside_disc(x, y) {
                inside += 1;
                inside_hit += usize::from(foreground);
            } else {
                outside += 1;
                outside_hit += usize::from(foreground);
            }
        }
    }
    (
        inside_hit as f64 / inside as f64,
        outside_hit as f64 / outside as f64,
    )
}

fn click(x: f64, y: f64, label: VideoObjectPointLabel) -> VideoObjectPointPrompt {
    VideoObjectPointPrompt::new(
        NormalizedVideoPoint::new((x / WIDTH as f64) as f32, (y / HEIGHT as f64) as f32),
        label,
    )
}

/// A foreground click inside the object returns the object.
///
/// The assertion is the GESTURE's own: what the finger landed on comes back,
/// and what it did not stays out. Both numbers are measured against the disc
/// this test drew, so a model that answered "everything" or "nothing" fails —
/// which is what an all-foreground or all-background mask is.
///
/// kill: drop the `add_no_memory_embedding` flag to 0 (the video path's value)
/// and the image predictor's features are wrong; drop the point's normalization
/// by `MODEL_IMAGE_SIZE` in `processing::model_point` and the click lands off
/// the object.
#[test]
#[ignore = "needs the EdgeTAM image-path export"]
fn a_foreground_click_returns_the_object_it_landed_on() {
    let Some(directory) = models("a_foreground_click_returns_the_object_it_landed_on") else {
        return;
    };
    let rgb = disc_image();
    let frame = RgbFrameRef::new(&rgb, WIDTH as u32, HEIGHT as u32).expect("rgb frame");
    let segmenter = segmenter(&directory);

    let result = futures_executor::block_on(async {
        let mut session = segmenter.make_session(frame).await.expect("image session");
        session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    CENTRE.0,
                    CENTRE.1,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("disc"),
                true,
            )
            .await
            .expect("apply")
    });

    let (covered, leaked) = coverage(&result.mask);
    println!("[EDGETAM] disc covered {covered:.3}, background claimed {leaked:.3}");
    assert!(
        covered > 0.90,
        "the click's own object is only {covered:.3} covered"
    );
    assert!(
        leaked < 0.05,
        "the mask claims {leaked:.3} of the background"
    );
    // The model's own confidence, and a real assertion: 0.994 on this disc.
    //
    // It read 0.0000416 while the decoder's token axis was STATIC at eight and
    // the host padded short prompts with `-1` labels. That was written off as
    // "a synthetic disc is far from the training distribution" — wrong. The
    // padding embeds as `not_a_point_embed`, so a one-click prompt reached the
    // transformer as nine sparse tokens instead of two and the IoU head
    // collapsed. A near-zero IoU beside a perfect mask is the symptom, and
    // asserting it back is what stops the padding returning.
    let iou = result.predicted_iou.expect("a candidate carries its IOU");
    println!("[EDGETAM] predicted IOU {iou}");
    assert!(
        iou.is_finite() && iou > 0.5,
        "the model should be confident about a mask it got exactly right: {iou}"
    );
}

/// A click somewhere ELSE does not return the object the first click found —
/// and what it does return is measured, not assumed.
///
/// kill: stop sending the click's own coordinates to the decoder (map them all
/// to the image centre in `EdgeTamModel::decode`) — both prompts then answer
/// with the disc.
///
/// NOT a kill, measured: collapsing every object id to one key in
/// `EdgeTamImageSession`. The second `apply` REPLACES the prompt, so a shared
/// key still answers the corner click with the corner's region; per-object
/// keying is graded by `the_thirty_third_object_is_refused`, where the count is
/// the observable.
///
/// The corner mask is NOT empty — measured, it claims 98% of the background,
/// which is a coherent answer to "what is at this corner" for a two-region
/// image. So `corner_covered < disc_covered - 0.30` alone would also pass for
/// an implementation that returned everything-but-the-click; the leak is
/// asserted too, in both directions, which is what makes the pair a real
/// discriminator.
#[test]
#[ignore = "needs the EdgeTAM image-path export"]
fn a_background_click_does_not_return_the_object() {
    let Some(directory) = models("a_background_click_does_not_return_the_object") else {
        return;
    };
    let rgb = disc_image();
    let frame = RgbFrameRef::new(&rgb, WIDTH as u32, HEIGHT as u32).expect("rgb frame");
    let segmenter = segmenter(&directory);

    let (foreground, background) = futures_executor::block_on(async {
        let mut session = segmenter.make_session(frame).await.expect("image session");
        let foreground = session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    CENTRE.0,
                    CENTRE.1,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("disc"),
                true,
            )
            .await
            .expect("foreground click");
        // A far corner, well outside the disc, as the ONLY prompt of a second
        // object — the session keeps them apart.
        let background = session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    WIDTH as f64 - 8.0,
                    8.0,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("corner"),
                true,
            )
            .await
            .expect("corner click");
        (foreground, background)
    });

    let (disc_covered, disc_leak) = coverage(&foreground.mask);
    let (corner_covered, corner_leak) = coverage(&background.mask);
    println!(
        "[EDGETAM] disc {disc_covered:.3}/{disc_leak:.3} vs corner-prompt \
         {corner_covered:.3}/{corner_leak:.3}"
    );
    assert!(disc_covered > 0.90, "the disc click must find the disc");
    assert!(
        disc_leak < 0.05,
        "the disc click must not claim the background"
    );
    assert!(
        corner_covered < disc_covered - 0.30,
        "a click in the corner returned the disc: {corner_covered:.3} vs {disc_covered:.3}"
    );
    // The two masks must be DIFFERENT ANSWERS, not one answer twice: the disc
    // prompt owns the disc and not the background, the corner prompt the
    // reverse. Asserting only the first would pass for a session that returned
    // the same mask for every object id.
    assert!(
        corner_leak > disc_leak + 0.30,
        "the corner prompt returned the same region as the disc prompt: \
         leak {corner_leak:.3} vs {disc_leak:.3}"
    );
}

/// The session's own budget: 32 active objects, and the 33rd is refused rather
/// than silently evicting one.
///
/// kill: drop the `TooManyImageObjects` check and the 33rd object is accepted,
/// so a caller's correction masks grow without bound.
#[test]
#[ignore = "needs the EdgeTAM image-path export"]
fn the_thirty_third_object_is_refused() {
    let Some(directory) = models("the_thirty_third_object_is_refused") else {
        return;
    };
    let rgb = disc_image();
    let frame = RgbFrameRef::new(&rgb, WIDTH as u32, HEIGHT as u32).expect("rgb frame");
    let segmenter = segmenter(&directory);

    futures_executor::block_on(async {
        let mut session = segmenter.make_session(frame).await.expect("image session");
        assert_eq!(session.maximum_active_object_count(), 32);
        for index in 0..32 {
            session
                .apply(
                    &VideoObjectPrompt::Points(vec![click(
                        CENTRE.0,
                        CENTRE.1,
                        VideoObjectPointLabel::Foreground,
                    )]),
                    &VideoObjectId::new(format!("object-{index}")),
                    true,
                )
                .await
                .unwrap_or_else(|error| panic!("object {index}: {error}"));
        }
        let refused = session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    CENTRE.0,
                    CENTRE.1,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("object-32"),
                true,
            )
            .await;
        let error = refused
            .expect_err("the 33rd object must be refused")
            .to_string();
        assert!(error.contains("33"), "{error}");

        // Removing one frees EXACTLY one slot. Asserting only that the next
        // object is accepted would pass for a `remove` that wiped the whole
        // session — measured: that mutation survived until the count below and
        // the second refusal were added.
        session.remove(&VideoObjectId::new("object-0")).await;
        session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    CENTRE.0,
                    CENTRE.1,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("object-32"),
                true,
            )
            .await
            .expect("the freed slot takes the next object");
        let error = session
            .apply(
                &VideoObjectPrompt::Points(vec![click(
                    CENTRE.0,
                    CENTRE.1,
                    VideoObjectPointLabel::Foreground,
                )]),
                &VideoObjectId::new("object-33"),
                true,
            )
            .await
            .expect_err("the budget is full again after ONE object took the freed slot")
            .to_string();
        assert!(error.contains("33"), "{error}");
    });
}

/// The memory encoder's answer DEPENDS ON THE MASK.
///
/// A graph that ignored its mask input would still return a well-shaped
/// `[1, 512, 64]` and every shape assertion would pass — and the video session
/// would then carry the same memory for every frame, which is the same as
/// carrying none. So the property asserted is the one that cannot be faked:
/// two different masks over the SAME features produce two different memories,
/// and the same mask twice produces the same one.
///
/// kill: drop `selected_low_resolution_mask` from the graph's inputs (export it
/// as a constant) and the two memories become identical. The symmetric kill —
/// ignoring the FEATURES — is covered by the third comparison, which was added
/// after the first version passed with an all-zero feature map.
#[test]
#[ignore = "needs the EdgeTAM memory-encoder export"]
fn the_memory_a_frame_contributes_depends_on_its_mask() {
    let Some(directory) = models("the_memory_a_frame_contributes_depends_on_its_mask") else {
        return;
    };
    if !directory
        .join(format!(
            "{}.onnx",
            ferment_engine_infer::ModelId::EdgetamMemoryEncoder.name()
        ))
        .exists()
    {
        println!("SKIP the_memory_a_frame_contributes_depends_on_its_mask: no memory encoder");
        return;
    }
    let rgb = disc_image();
    let frame = RgbFrameRef::new(&rgb, WIDTH as u32, HEIGHT as u32).expect("rgb frame");
    let model = edgetam_model(&directory);

    let (memory_a, memory_b, memory_a_again, memory_zero_features) =
        futures_executor::block_on(async {
            let normalized =
                ferment_engine_processors::edgetam::image_preprocessing::normalized_image(
                    &ferment_engine_processors::edgetam::preprocessing::EdgeTamRgbFrame::new(
                        0,
                        ferment_engine_media::MediaTime::new(
                            0,
                            ferment_engine_media::TimeBase::MICROSECONDS,
                        ),
                        WIDTH,
                        HEIGHT,
                        rgb.clone(),
                    )
                    .expect("source frame"),
                )
                .expect("normalize");
            let features = model.encode_image(&normalized).await.expect("encode");

            // Two masks the TEST writes: everything, and nothing.
            let count = 256 * 256;
            let all = vec![half::f16::from_f32(16.0); count];
            let none = vec![half::f16::from_f32(-16.0); count];
            let blank = vec![half::f16::ZERO; features.vision_features.len()];
            (
                model
                    .encode_memory(&features.vision_features, &all)
                    .await
                    .expect("memory a"),
                model
                    .encode_memory(&features.vision_features, &none)
                    .await
                    .expect("memory b"),
                model
                    .encode_memory(&features.vision_features, &all)
                    .await
                    .expect("memory a again"),
                // The symmetric half: same mask, DIFFERENT frame. A graph that
                // ignored its features would pass every assertion above.
                model
                    .encode_memory(&blank, &all)
                    .await
                    .expect("memory with blank features"),
            )
        });
    let _ = frame;

    assert!(memory_a.iter().all(|value| value.to_f32().is_finite()));
    assert_eq!(
        memory_a, memory_a_again,
        "the same features and the same mask must give the same memory"
    );
    assert_ne!(
        memory_a, memory_b,
        "the memory encoder ignored its MASK: an all-foreground and an \
         all-background mask produced the same memory"
    );
    assert_ne!(
        memory_a, memory_zero_features,
        "the memory encoder ignored its FEATURES: a real frame and a blank one \
         produced the same memory"
    );
}

/// A CORRECTION runs end to end, and the gesture works in BOTH directions.
///
/// This is the path an editor's second tap takes and nothing else runs it:
/// `prompt_inference::prompt_tensor`'s append branch, `previous_mask_input`'s
/// clamp, and `EdgeTamImageSession`'s reuse of the retained mask.
///
/// What is asserted is what the gesture means, measured against the disc THIS
/// test drew: appending a FOREGROUND click keeps the object, appending a
/// BACKGROUND click inside it takes the object away. The second number is
/// degenerate on purpose — a flat uniform disc has no sub-region to keep — so
/// it is paired with the first, and the two must differ by most of the object.
/// A pipeline that answered "nothing" or "everything" to both fails.
///
/// The append-vs-replace SEMANTICS are not graded here: both send a valid
/// prompt and this fixture cannot tell the two masks apart. They are graded
/// exactly, on the tensor, by `prompt_inference::tests` — which is where the
/// difference actually lives.
///
/// kill: drop the `previous_mask` argument in `EdgeTamImageSession::apply`, so
/// the correction stops being a correction — the foreground append no longer
/// holds the object.
#[test]
#[ignore = "needs the EdgeTAM image-path export"]
fn a_correction_can_both_keep_and_remove_the_object() {
    let Some(directory) = models("a_correction_can_both_keep_and_remove_the_object") else {
        return;
    };
    let rgb = disc_image();
    let segmenter = segmenter(&directory);
    let object = VideoObjectId::new("disc");

    let correct = |second: VideoObjectPointPrompt| {
        let frame = RgbFrameRef::new(&rgb, WIDTH as u32, HEIGHT as u32).expect("rgb frame");
        futures_executor::block_on(async {
            let mut session = segmenter.make_session(frame).await.expect("image session");
            let first = session
                .apply(
                    &VideoObjectPrompt::Points(vec![click(
                        CENTRE.0,
                        CENTRE.1,
                        VideoObjectPointLabel::Foreground,
                    )]),
                    &object,
                    true,
                )
                .await
                .expect("the first click");
            let corrected = session
                .apply(&VideoObjectPrompt::Points(vec![second]), &object, false)
                .await
                .expect("the correction");
            (
                coverage(&first.mask),
                coverage(&corrected.mask),
                (corrected.mask.width(), corrected.mask.height()),
                corrected.mask.bytes().iter().all(|b| *b == 0 || *b == 255),
            )
        })
    };

    let ((first_covered, _), (kept, kept_leak), size, binary) = correct(click(
        CENTRE.0 + RADIUS * 0.5,
        CENTRE.1,
        VideoObjectPointLabel::Foreground,
    ));
    println!("[EDGETAM] append fg: {first_covered:.3} -> {kept:.3} (background {kept_leak:.3})");
    assert!(first_covered > 0.90, "the opening click must find the disc");
    assert!(
        kept > 0.90,
        "a foreground correction must keep the object: {kept:.3}"
    );
    assert!(kept_leak < 0.05);
    assert_eq!(size, (WIDTH, HEIGHT));
    assert!(binary);

    let (_, (removed, _), _, _) = correct(click(
        CENTRE.0 + RADIUS * 0.75,
        CENTRE.1,
        VideoObjectPointLabel::Background,
    ));
    println!("[EDGETAM] append bg: {first_covered:.3} -> {removed:.3}");
    assert!(
        removed < 0.10,
        "a background correction inside the object must take it away: {removed:.3}"
    );
    assert!(
        kept - removed > 0.80,
        "the two corrections must differ: kept {kept:.3}, removed {removed:.3}"
    );
}
