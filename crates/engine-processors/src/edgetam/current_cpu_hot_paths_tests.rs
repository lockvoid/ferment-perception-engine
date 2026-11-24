//! Original HD shapes, byte laws, reusable storage, and diagnostic timings.
use super::{
    image_preprocessing,
    preprocessing::{self, EdgeTamRgbFrame, Workspace},
    processing,
    prompt_inference::MaskWorkspace,
};
use ferment_engine_media::{CpuFrame, CpuPixelFormat, MediaTime};
use half::f16;
use std::{hint::black_box, time::Instant};

fn timed(mut body: impl FnMut()) -> f64 {
    let start = Instant::now();
    body();
    start.elapsed().as_secs_f64() * 1000.
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn pair(name: &str, iterations: usize, mut reference: impl FnMut(), mut optimized: impl FnMut()) {
    for _ in 0..2 {
        reference();
        optimized();
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..iterations {
        if i % 2 == 0 {
            a.push(timed(&mut reference));
            b.push(timed(&mut optimized));
        } else {
            b.push(timed(&mut optimized));
            a.push(timed(&mut reference));
        }
    }
    println!(
        "EDGETAM_BENCH_PAIR {name} reference_ms={} optimized_ms={} iterations={iterations}",
        median(&mut a),
        median(&mut b)
    );
}
fn bench(name: &str, iterations: usize, mut body: impl FnMut()) {
    for _ in 0..2 {
        body();
    }
    let mut samples: Vec<_> = (0..iterations).map(|_| timed(&mut body)).collect();
    let middle = median(&mut samples);
    println!(
        "EDGETAM_BENCH {name} median_ms={middle} min_ms={} iterations={iterations}",
        samples[0]
    );
}
fn scalar_rgb(source: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(source.len() / 4 * 3);
    for i in 0..source.len() / 4 {
        for c in 0..3 {
            rgb.push(source[i * 4 + c]);
        }
    }
    rgb
}

fn assert_exact_bytes(actual: &[u8], expected: &[u8], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: byte count");
    let mut different = 0;
    let mut first = None;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a != e {
            different += 1;
            if first.is_none() {
                first = Some((index, a, e));
            }
        }
    }
    assert_eq!(
        different, 0,
        "{label}: differing bytes, first (index, actual, expected) {first:?}"
    );
}

/// Kill: channel/stride errors at HD, workspace buffer growth, half-normalization
/// drift, or fused resize/threshold output diverging from the scalar display law.
#[test]
#[ignore = "original HD hot-path diagnostics; run in its own release process"]
fn hd_cpu_hot_paths_keep_bytes_and_workspace_storage_across_original_iterations() {
    let (w, h) = (1920, 1080);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i * 29 + 17) as u8).collect();
    let source = EdgeTamRgbFrame::new(0, MediaTime::ZERO, w, h, pixels).unwrap();
    let rgba: Vec<u8> = (0..w * h)
        .flat_map(|i| [(i * 13) as u8, (i * 7) as u8, (i * 3) as u8, 255])
        .collect();
    let image = CpuFrame::new(
        w as u32,
        h as u32,
        w * 4,
        CpuPixelFormat::Rgba8,
        rgba.clone(),
    )
    .unwrap();
    assert_eq!(&*image.to_rgb24().data, scalar_rgb(&rgba));
    pair(
        "rasterize-rgba-to-rgb-1920x1080",
        8,
        || {
            black_box(scalar_rgb(&rgba));
        },
        || {
            black_box(image.to_rgb24());
        },
    );
    let oracle = std::path::PathBuf::from(
        std::env::var_os("FERMENT_EDGETAM_HD_ORACLE_DIR")
            .expect("requires frozen independent current Swift HD vImage capture"),
    );
    let original = std::fs::read(oracle.join("source.rgb")).unwrap();
    assert_exact_bytes(&source.pixels, &original, "original HD input formula");
    let reference_square = std::fs::read(oracle.join("square.rgb")).unwrap();
    let allocated = preprocessing::square_model_frame(&source).unwrap();
    assert_exact_bytes(
        &allocated,
        &reference_square,
        "original vImage HQ square RGB",
    );
    // The independent Apple scalar result is captured above. Keep the original
    // eight production timings without labelling a second native implementation
    // as an independent Apple execution.
    bench("video-square-rgb-1920x1080-to-1024", 8, || {
        black_box(preprocessing::square_model_frame(&source).unwrap());
    });
    let mut workspace = Workspace::new(w, h).unwrap();
    assert_eq!(
        workspace.square_model_frame(&source).unwrap().pixels,
        allocated
    );
    let retained = workspace.retained_byte_count();
    assert!(retained >= 1024 * 1024 * 3);
    pair(
        "video-square-reusable-workspace-1920x1080-to-1024",
        8,
        || {
            black_box(preprocessing::square_model_frame(&source).unwrap());
        },
        || {
            black_box(workspace.square_model_frame(&source).unwrap());
        },
    );
    assert_eq!(workspace.retained_byte_count(), retained);
    println!("EDGETAM_MEMORY preprocessing-workspace-retained-bytes={retained}");
    let normalized =
        preprocessing::normalized_image_from_square_model_frame(1024, 1024, &allocated).unwrap();
    assert_eq!(normalized.len(), 3 * 1024 * 1024);
    let reference_normalized = std::fs::read(oracle.join("normalized.fp16")).unwrap();
    let normalized_bytes: Vec<_> = normalized
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    assert_exact_bytes(
        &normalized_bytes,
        &reference_normalized,
        "original video FP16 normalization",
    );
    for channel in 0..3 {
        for pixel in 0..1024 * 1024 {
            let unit = allocated[pixel * 3 + channel] as f32 / 255.;
            let expected = f16::from_f32(
                (unit - [0.485, 0.456, 0.406][channel]) / [0.229, 0.224, 0.225][channel],
            );
            assert_eq!(
                normalized[channel * 1024 * 1024 + pixel].to_bits(),
                expected.to_bits()
            );
        }
    }
    bench("video-normalize-rgb-1024", 8, || {
        black_box(
            preprocessing::normalized_image_from_square_model_frame(1024, 1024, &allocated)
                .unwrap(),
        );
    });
    bench("image-exact-preprocess-1920x1080-to-1024", 5, || {
        let value = image_preprocessing::normalized_image(&source).unwrap();
        assert_eq!(value.len(), 3 * 1024 * 1024);
        assert!(value.iter().all(|v| v.is_finite()));
        black_box(value);
    });
    let low: Vec<_> = (0..256 * 256)
        .map(|i| f16::from_f32(((i % 257) - 128) as f32 / 17.))
        .collect();
    let float: Vec<_> = low.iter().map(|v| v.to_f32()).collect();
    let logits = processing::display_logits(&float, w, h).unwrap();
    let reference = processing::threshold(&logits, w, h).unwrap();
    let mut masks = MaskWorkspace::new(w, h).unwrap();
    assert_eq!(masks.mask(&low).unwrap(), reference);
    let mask_bytes = masks.retained_byte_count();
    assert!(mask_bytes > 0);
    pair(
        "mask-current-allocation-vs-session-workspace",
        8,
        || {
            black_box(processing::thresholded_display_mask(&float, w, h).unwrap());
        },
        || {
            black_box(masks.mask(&low).unwrap());
        },
    );
    assert_eq!(masks.retained_byte_count(), mask_bytes);
    println!("EDGETAM_MEMORY mask-workspace-retained-bytes={mask_bytes}");
}
