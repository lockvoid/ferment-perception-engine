//! The 016 enhancement backings against the goldens `ferment-models-export`
//! owns (`models/realesr`, `models/fastdvdnet`, `models/rife`): real frames
//! of the pinned clips through the processor as the app calls it — u8 frame
//! in, the published master through the real ONNX Runtime backend, u8 frame
//! out — against the checkpoint's own answer rendered the way a host shows it
//! (clamp to [0, 1], × 255, round).
//!
//! Select the execution provider with `FERMENT_ENGINE_ORT_EP`
//! (cpu | directml | cuda | windows).

#![cfg(all(feature = "ort", not(target_arch = "wasm32")))]

#[path = "support/ort_assignment.rs"]
mod ort_assignment;

use std::sync::Arc;

use ferment_engine_infer::{ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider};
use ferment_engine_media::{
    ColorInfo, CpuFrame, CpuPixelFormat, FrameBuffer, MediaTime, TimeBase, VideoFrame,
};
use ferment_engine_processors::slow_motion_renderer::{
    FrameInterpolator, MediaTime as RenderTime, TimedFrame,
};
use ferment_engine_processors::video_noise_filter::TemporalNoiseFiltering;
use ferment_engine_processors::video_upscaler::{Dimensions, FrameUpscaling};
use ferment_engine_processors::{
    CompactFrameUpscaler, FastDvdnetTemporalFilter, RifeFrameInterpolator,
};
use ferment_engine_test_fixtures::export;

/// Cross-runtime tolerance in u8 space: the export measured every fp32
/// master's rendered frame within one count of its checkpoint's, and another
/// ONNX Runtime build's kernels may move a rounding boundary.
const MAX_ABS_U8: i16 = 3;

/// GPU execution providers drift further from the CPU-generated goldens (real
/// fp16 kernels). Until per-EP tolerances are signed off, runs may widen the
/// bound explicitly via `FERMENT_ENHANCE_MAX_U8`; every run prints the measured
/// diff regardless, so the recorded numbers — not the bound — are the result.
fn tolerance() -> i16 {
    std::env::var("FERMENT_ENHANCE_MAX_U8")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(MAX_ABS_U8)
}

fn backend(model: ModelId, file_name: &str) -> Option<Arc<OrtBackend>> {
    let paths =
        ModelPaths::new().with_path(model, ModelVersion::Latest, export::artifact(file_name)?);
    let locator = Arc::new(paths);
    Some(Arc::new(
        match std::env::var("FERMENT_ENGINE_ORT_EP").as_deref() {
            Ok("directml") => {
                OrtBackend::with_execution_provider(locator, OrtExecutionProvider::DirectMl)
            }
            Ok("cuda") => OrtBackend::with_execution_provider(locator, OrtExecutionProvider::Cuda),
            Ok("windows") => OrtBackend::windows_default(locator),
            Ok("cpu") | Err(_) => {
                OrtBackend::with_execution_provider(locator, OrtExecutionProvider::Cpu)
            }
            Ok(other) => panic!("unsupported FERMENT_ENGINE_ORT_EP: {other}"),
        },
    ))
}

fn frame(width: u32, height: u32, index: i64, rgb: Vec<u8>) -> VideoFrame {
    VideoFrame {
        width,
        height,
        pts: MediaTime::new(index, TimeBase::new_unchecked(1, 30)),
        duration: None,
        color: ColorInfo::default(),
        buffer: FrameBuffer::Cpu(
            CpuFrame::new(
                width,
                height,
                width as usize * 3,
                CpuPixelFormat::Rgb24,
                rgb,
            )
            .unwrap(),
        ),
    }
}

/// Packed RGB24 of `planes` `[3, height, width]` holding u8 / 255 — the doors'
/// own inputs, which are a decoded frame's pixels exactly.
fn packed_from_unit_planes(planes: &[f32]) -> Vec<u8> {
    let pixels = planes.len() / 3;
    (0..pixels * 3)
        .map(|index| (planes[(index % 3) * pixels + index / 3] * 255.0).round() as u8)
        .collect()
}

/// A checkpoint's `[3, height, width]` answer as the host shows it, packed.
fn shown(planes: &[f32]) -> Vec<u8> {
    let pixels = planes.len() / 3;
    (0..pixels * 3)
        .map(|index| {
            (planes[(index % 3) * pixels + index / 3].clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect()
}

fn golden(relative: &str) -> Vec<f32> {
    export::require_f32_le(relative)
}

fn golden_bytes(relative: &str) -> Vec<u8> {
    std::fs::read(export::require_path(relative)).unwrap()
}

fn contract(model: &str) -> Option<serde_json::Value> {
    let path = export::path(&format!("{model}/model.json"))?;
    Some(serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
}

fn assert_close(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut worst = 0i16;
    let mut beyond_one = 0usize;
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = (i16::from(a) - i16::from(e)).abs();
        worst = worst.max(diff);
        if diff > 1 {
            beyond_one += 1;
        }
    }
    let bound = tolerance();
    eprintln!("{label}: max u8 diff {worst}, {beyond_one} bytes beyond ±1 (bound ±{bound})");
    assert!(
        worst <= bound,
        "{label}: max u8 diff {worst} exceeds ±{bound}"
    );
}

fn cpu_bytes(buffer: FrameBuffer) -> Vec<u8> {
    let FrameBuffer::Cpu(frame) = buffer else {
        panic!("expected a CPU buffer");
    };
    frame.data.to_vec()
}

/// PSNR gate for flow-warped output: RIFE's GPU drift concentrates in
/// single-pixel outliers at motion edges, so a max-u8 bound misjudges it while
/// PSNR tracks perceptual fidelity. 45 dB is the 016 shipping gate, the floor
/// the export holds the rendered frame to; override with
/// `FERMENT_ENHANCE_MIN_PSNR` pending lead sign-off.
fn assert_psnr(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let (mut sum_sq, mut worst) = (0.0f64, 0i16);
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = i16::from(a) - i16::from(e);
        worst = worst.max(diff.abs());
        sum_sq += f64::from(diff) * f64::from(diff);
    }
    let mse = sum_sq / actual.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    let bound = std::env::var("FERMENT_ENHANCE_MIN_PSNR")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(45.0f64);
    eprintln!("{label}: psnr {psnr:.2} dB, max u8 diff {worst} (bound >= {bound} dB)");
    assert!(psnr >= bound, "{label}: psnr {psnr:.2} dB below {bound} dB");
}

/// Each member on each clip: a 96 × 96 crop of a decoded frame in, four
/// times its size out.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the realesr masters"]
fn every_upscaler_answers_its_checkpoint() {
    let Some(contract) = contract("realesr") else {
        return;
    };
    let crop = contract["goldens"]["crop"].as_u64().unwrap() as u32;
    for (model, file_name, upscaler) in [
        (
            ModelId::RealesrGeneralX4,
            "realesr_general_x4v3.onnx",
            CompactFrameUpscaler::general as fn(_) -> CompactFrameUpscaler,
        ),
        (
            ModelId::RealesrGeneralWdnX4,
            "realesr_general_wdn_x4v3.onnx",
            CompactFrameUpscaler::general_denoise,
        ),
        (
            ModelId::RealesrAnimeVideo3,
            "realesr_animevideov3.onnx",
            CompactFrameUpscaler::anime_video,
        ),
    ] {
        let Some(backend) = backend(model, file_name) else {
            return;
        };
        let mut upscaler = upscaler(backend.clone());
        upscaler
            .start_session(Dimensions::new(crop, crop), false)
            .unwrap();
        for clip in ["street", "castle"] {
            let low = packed_from_unit_planes(&golden(&format!(
                "realesr/goldens/realesr_{clip}_lr_in.f32"
            )));
            let output =
                futures_executor::block_on(upscaler.upscale(&frame(crop, crop, 0, low))).unwrap();
            assert_close(
                &cpu_bytes(output),
                &shown(&golden(&format!(
                    "realesr/goldens/{}_{clip}_sr_out.f32",
                    model.name()
                ))),
                &format!("{} {clip} {crop}² → {}²", model.name(), crop * 4),
            );
            ort_assignment::require_gpu(&backend, model);
        }
    }
}

/// The centre of a five-frame window at the clip's strength — the noise map
/// the filter packs is the strength's share of FastDVDnet's top sigma.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the fastdvdnet master"]
fn the_temporal_filter_answers_its_checkpoint() {
    let Some(contract) = contract("fastdvdnet") else {
        return;
    };
    let Some(backend) = backend(ModelId::Fastdvdnet, "fastdvdnet.onnx") else {
        return;
    };
    let crop = contract["goldens"]["crop"].as_u64().unwrap() as u32;
    let mut filter = FastDvdnetTemporalFilter::new(backend.clone());
    filter.start_session(crop, crop, false).unwrap();
    for (clip, facts) in contract["goldens"]["clips"].as_object().unwrap() {
        let window = golden(&format!(
            "fastdvdnet/goldens/fastdvdnet_{clip}_frames_in.f32"
        ));
        let plane = (crop * crop) as usize;
        let frames: Vec<VideoFrame> = window
            .chunks_exact(3 * plane)
            .enumerate()
            .map(|(index, planes)| frame(crop, crop, index as i64, packed_from_unit_planes(planes)))
            .collect();
        assert_eq!(frames.len(), 5);
        let strength = facts["strength"].as_f64().unwrap() as f32;
        let output = futures_executor::block_on(filter.filter(
            &frames[2],
            &frames[..2],
            &frames[3..],
            strength,
            false,
        ))
        .unwrap();
        assert_close(
            &cpu_bytes(output),
            &shown(&golden(&format!(
                "fastdvdnet/goldens/fastdvdnet_{clip}_denoised_out.f32"
            ))),
            &format!("fastdvdnet {clip} at strength {strength}"),
        );
        ort_assignment::require_gpu(&backend, ModelId::Fastdvdnet);
    }
}

/// A frame whose sides are not multiples of four. FastDVDnet's block
/// downsamples twice and shuffles back up twice, so the graph's own skip
/// addition mismatches and the session dies in `/temp1_2/Add` — measured on
/// the export's master at 854 × 480 (standard 480p widescreen) and 1280 × 718.
/// The filter pads and crops back, so what this asserts is that a caller may
/// hand it any size and get that size back.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the fastdvdnet master"]
fn fastdvdnet_denoises_a_frame_whose_sides_are_not_a_multiple_of_four() {
    let Some(backend) = backend(ModelId::Fastdvdnet, "fastdvdnet.onnx") else {
        return;
    };
    let mut filter = FastDvdnetTemporalFilter::new(backend.clone());
    // Kill: a profile fixed to the first resolution, an opt=1 TensorRT
    // profile falling back to CPU, or broken host alignment/cropping.
    for (width, height) in [(34, 22), (4, 4), (64, 36)] {
        filter.start_session(width, height, false).unwrap();
        let frames: Vec<VideoFrame> = (0..5)
            .map(|index| {
                let rgb = (0..width * height * 3)
                    .map(|value| ((value * 7 + index * 29) % 256) as u8)
                    .collect();
                frame(width, height, index as i64, rgb)
            })
            .collect();
        let output = futures_executor::block_on(filter.filter(
            &frames[2],
            &frames[..2],
            &frames[3..],
            0.5,
            false,
        ))
        .expect("a changing frame size denoises");
        assert_eq!(
            cpu_bytes(output).len(),
            (width * height * 3) as usize,
            "the frame's own size"
        );
        ort_assignment::require_gpu(&backend, ModelId::Fastdvdnet);
    }
}

fn rife_pair(
    clip: &str,
    width: u32,
    height: u32,
) -> (TimedFrame<VideoFrame>, TimedFrame<VideoFrame>) {
    let crop = |rgb: Vec<u8>| -> Vec<u8> {
        (0..height as usize)
            .flat_map(|y| rgb[y * 256 * 3..y * 256 * 3 + width as usize * 3].to_vec())
            .collect()
    };
    let timed = |suffix: &str, index: i64| TimedFrame {
        frame: frame(
            width,
            height,
            index,
            crop(golden_bytes(&format!(
                "rife/goldens/rife_v4_25_{clip}_{suffix}_in.u8"
            ))),
        ),
        pts: RenderTime::new(index, 30),
    };
    (timed("img0", 0), timed("img1", 1))
}

/// The 016 shipping gate for slow-mo: real consecutive video frames (onboard
/// racing, a DJ reel, a kitchen close-up) at the midpoint, against the fp32
/// master's frame rendered as a runtime renders it. Synthetic patterns are
/// out of distribution for flow estimation and earn no golden.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the rife master"]
fn the_interpolator_answers_its_checkpoint_on_real_frames() {
    let Some(contract) = contract("rife") else {
        return;
    };
    let Some(backend) = backend(ModelId::RifeV425, "rife_v4_25.onnx") else {
        return;
    };
    let mut interpolator = RifeFrameInterpolator::new(backend.clone());
    for clip in contract["goldens"]["clips"].as_object().unwrap().keys() {
        let (from, to) = rife_pair(clip, 256, 256);
        let frames = interpolator.interpolate(&from, &to, &[0.5], false).unwrap();
        assert_psnr(
            &cpu_bytes(frames.into_iter().next().unwrap().buffer),
            &golden_bytes(&format!("rife/goldens/rife_v4_25_{clip}_frame_out.u8")),
            &format!("rife '{clip}' midpoint"),
        );
        ort_assignment::require_gpu(&backend, ModelId::RifeV425);
    }
}

/// A frame whose sides are not multiples of 32: the interpolator
/// edge-replicates up to the door's alignment and crops the answer back, so
/// the caller gets its own size.
#[test]
#[ignore = "needs ONNX Runtime (ORT_DYLIB_PATH) and the rife master"]
fn rife_interpolates_a_frame_whose_sides_are_not_a_multiple_of_32() {
    if contract("rife").is_none() {
        return;
    }
    let Some(backend) = backend(ModelId::RifeV425, "rife_v4_25.onnx") else {
        return;
    };
    let mut interpolator = RifeFrameInterpolator::new(backend.clone());
    let (from, to) = rife_pair("formula1", 48, 40);
    let frames = interpolator.interpolate(&from, &to, &[0.5], false).unwrap();
    assert_eq!(frames.len(), 1);
    let frame = frames.into_iter().next().unwrap();
    assert_eq!((frame.width, frame.height), (48, 40));
    assert_eq!(cpu_bytes(frame.buffer).len(), 48 * 40 * 3);
    ort_assignment::require_gpu(&backend, ModelId::RifeV425);
}
