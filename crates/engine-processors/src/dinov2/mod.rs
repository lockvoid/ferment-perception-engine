//! Visual Semantic Encoder — DINOv2 ViT-S/14 (Apache-2.0, 22 M), the CLS token
//! as a 384-dim unit vector. Task-agnostic: "is this the same picture / shot /
//! object?" by cosine distance — the scene detector's flash veto and per-scene
//! embeddings ride it today; import dedup, poster picking (a scene's medoid
//! frame) and visual search over scenes can ride the same vector.
//!
//! Port of iOS `Processors/DINOv2/DINOv2SemanticEncoder.swift`. The graph is
//! `ferment-models-export`'s (`models/dinov2`, `model.json` the contract): one
//! 224×126 landscape frame — 16×9 patches, the analysis aspect; squashing to a
//! square cost a third of the measured same-shot/other-shot gap upstream — with
//! ImageNet's normalisation inside, per channel, so this side hands over plain
//! 0..255 RGB planes. A portrait picture is rotated to landscape here, so two
//! portrait pictures compare exactly as two landscape ones do.
//!
//! **Deviation from iOS, recorded:** iOS carries pictures as BGRA because
//! vImage and CoreML want it; the decode path here already produces packed RGB
//! and the ONNX graph takes RGB, so [`FrameSnapshot`] is RGB and there is no
//! channel swizzle to get wrong.

use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

mod resample;

/// One decoded frame kept for shot identification: packed 8-bit RGB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSnapshot {
    pub width: usize,
    pub height: usize,
    pub bytes_per_row: usize,
    pub rgb: Vec<u8>,
}

impl FrameSnapshot {
    /// A packed snapshot (`bytes_per_row == width * 3`).
    #[must_use]
    pub fn packed(width: usize, height: usize, rgb: Vec<u8>) -> Self {
        Self {
            width,
            height,
            bytes_per_row: width * 3,
            rgb,
        }
    }

    fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.bytes_per_row >= self.width * 3
            && self.rgb.len() >= self.bytes_per_row * self.height
    }
}

/// DINOv2 ViT-S/14 behind the inference backend.
#[derive(Clone)]
pub struct Dinov2SemanticEncoder {
    backend: Backend,
    version: ModelVersion,
}

impl std::fmt::Debug for Dinov2SemanticEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Dinov2SemanticEncoder")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl Dinov2SemanticEncoder {
    pub const EMBEDDING_SIZE: usize = 384;
    pub const INPUT_WIDTH: usize = 224;
    pub const INPUT_HEIGHT: usize = 126;

    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }

    /// A picture of any size, either orientation → unit vector. Empty when the
    /// picture cannot be prepared, inference fails, or the model answers no
    /// signal — every caller treats "no identity" as "abstain", so a hard error
    /// here would only turn an abstention into a failed cook.
    #[must_use]
    pub async fn embedding(&self, frame: &FrameSnapshot) -> Vec<f32> {
        let Some(prepared) = prepare(frame) else {
            return Vec::new();
        };
        let Ok(input) = Tensor::from_f32(
            vec![1, 3, Self::INPUT_HEIGHT as i64, Self::INPUT_WIDTH as i64],
            prepared,
        ) else {
            return Vec::new();
        };
        let mut inputs = TensorMap::new();
        inputs.insert("image".into(), input);
        let outputs = match self
            .backend
            .infer(ModelId::Dinov2Vits14, &self.version, inputs)
            .await
        {
            Ok(outputs) => outputs,
            // Abstaining is the right behaviour, but abstaining *silently* is
            // not: a backend that cannot serve this model makes every frame
            // abstain, which disables the flash veto and moves scene
            // boundaries with no other symptom.
            Err(error) => {
                crate::optional_failure::report(
                    "load the shot encoder — scene identity abstains",
                    &error,
                );
                return Vec::new();
            }
        };
        let Some(Ok(vector)) = outputs.get("embedding").map(Tensor::to_f32) else {
            return Vec::new();
        };
        if vector.len() != Self::EMBEDDING_SIZE
            || !vector
                .iter()
                .any(|value| *value != 0.0 && value.is_finite())
        {
            return Vec::new();
        }
        vector
    }

    /// Cosine distance of two unit vectors: 0 = the same picture, 1 =
    /// orthogonal. (Euclidean distance on unit vectors is √(2·cosine).)
    #[must_use]
    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 1.0;
        }
        1.0 - a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>()
    }
}

/// RGB picture → landscape → 224×126 → planar `[3, 126, 224]` in 0..255.
///
/// The high-quality resample follows the documented five-lobe Lanczos kernel
/// and the independently probed vImage axis/phase convention. Original iOS
/// pixel goldens and tolerances cover the complete preparation path.
fn prepare(frame: &FrameSnapshot) -> Option<Vec<f32>> {
    if !frame.is_valid() {
        return None;
    }
    let (width, height, rgb) = if frame.height > frame.width {
        rotate_landscape(frame)
    } else {
        let mut packed = Vec::with_capacity(frame.width * frame.height * 3);
        for y in 0..frame.height {
            let row = y * frame.bytes_per_row;
            packed.extend_from_slice(&frame.rgb[row..row + frame.width * 3]);
        }
        (frame.width, frame.height, packed)
    };

    u32::try_from(width).ok()?;
    u32::try_from(height).ok()?;
    let scaled = resample::resize_rgb(
        rgb,
        width,
        height,
        Dinov2SemanticEncoder::INPUT_WIDTH,
        Dinov2SemanticEncoder::INPUT_HEIGHT,
    );

    let pixels = Dinov2SemanticEncoder::INPUT_WIDTH * Dinov2SemanticEncoder::INPUT_HEIGHT;
    let mut planar = vec![0.0_f32; 3 * pixels];
    for (index, pixel) in scaled.chunks_exact(3).enumerate() {
        planar[index] = f32::from(pixel[0]);
        planar[pixels + index] = f32::from(pixel[1]);
        planar[2 * pixels + index] = f32::from(pixel[2]);
    }
    Some(planar)
}

/// 90° clockwise, matching iOS's `vImageRotate90_ARGB8888(…, 3, …)` — the
/// direction only matters for cross-platform comparison of portrait content,
/// and that is exactly what the shared embedding column needs.
fn rotate_landscape(frame: &FrameSnapshot) -> (usize, usize, Vec<u8>) {
    let (width, height) = (frame.height, frame.width);
    let mut rotated = vec![0_u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            // dst[y][x] = src[H−1−x][y]
            let source = (frame.height - 1 - x) * frame.bytes_per_row + y * 3;
            let target = (y * width + x) * 3;
            rotated[target..target + 3].copy_from_slice(&frame.rgb[source..source + 3]);
        }
    }
    (width, height, rotated)
}

/// The committed resampler goldens, produced once by
/// `scripts/exports/dinov2/vimage_reference.swift`.
#[cfg(test)]
fn golden(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/exports/dinov2/goldens")
        .join(name);
    std::fs::read(&path)
        .unwrap_or_else(|error| panic!("in-repo golden {} did not read: {error}", path.display()))
}

#[cfg(test)]
mod resampler_tests {
    use super::*;

    const SOURCE_WIDTH: usize = 640;
    const SOURCE_HEIGHT: usize = 360;

    /// Original port regression bounds, retained unchanged. The stricter
    /// current_pixel_contract below separately enforces the original iOS
    /// maximum/mean error instead of granting these legacy drift allowances.
    const MEAN_ABS_DRIFT: f64 = 2.0;
    const MAX_ABS_DRIFT: u8 = 24;
    /// Share of channel samples that must land within ±4 of vImage.
    const NEAR_EXACT_SHARE: f64 = 0.85;

    fn source() -> FrameSnapshot {
        FrameSnapshot::packed(
            SOURCE_WIDTH,
            SOURCE_HEIGHT,
            golden("resample_source_640x360.rgb"),
        )
    }

    /// Production output as 8-bit RGB, in the same layout vImage wrote.
    fn lanczos() -> Vec<u8> {
        let planar = prepare(&source()).expect("the golden source prepares");
        let pixels = Dinov2SemanticEncoder::INPUT_WIDTH * Dinov2SemanticEncoder::INPUT_HEIGHT;
        (0..pixels)
            .flat_map(|index| {
                [
                    planar[index] as u8,
                    planar[pixels + index] as u8,
                    planar[2 * pixels + index] as u8,
                ]
            })
            .collect()
    }

    /// Writes the Rust resample where `scripts/exports/dinov2/measure_drift.py`
    /// can pick it up, so the reported EMBEDDING drift is measured on the code
    /// that ships rather than on a Python re-implementation of it. Ignored:
    /// this is a measurement, not a gate.
    #[test]
    #[ignore = "measurement helper for scripts/exports/dinov2/measure_drift.py"]
    fn dump_lanczos3_resample_for_the_drift_measurement() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/dinov2_lanczos3_224x126.rgb");
        std::fs::write(&path, lanczos()).expect("write the measurement dump");
        println!("wrote {}", path.display());
    }

    #[test]
    fn lanczos3_tracks_vimages_high_quality_resample() {
        let ours = lanczos();
        let theirs = golden("resample_vimage_224x126.rgb");
        assert_eq!(ours.len(), theirs.len());

        let deltas: Vec<u8> = ours
            .iter()
            .zip(&theirs)
            .map(|(ours, theirs)| ours.abs_diff(*theirs))
            .collect();
        let mean = deltas.iter().map(|delta| f64::from(*delta)).sum::<f64>() / deltas.len() as f64;
        let max = deltas.iter().copied().max().expect("non-empty");
        let near = deltas.iter().filter(|delta| **delta <= 4).count() as f64 / deltas.len() as f64;

        println!(
            "resampler drift vs vImage: mean {mean:.3} max {max} within±4 {:.1}%",
            near * 100.0
        );
        assert!(mean <= MEAN_ABS_DRIFT, "mean channel drift {mean:.3}");
        assert!(max <= MAX_ABS_DRIFT, "max channel drift {max}");
        assert!(
            near >= NEAR_EXACT_SHARE,
            "only {:.1}% within ±4",
            near * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: usize, height: usize) -> FrameSnapshot {
        let mut rgb = vec![0_u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) * 3;
                rgb[index] = u8::try_from(x * 255 / width.max(1)).unwrap_or(255);
                rgb[index + 1] = u8::try_from(y * 255 / height.max(1)).unwrap_or(255);
                rgb[index + 2] = 40;
            }
        }
        FrameSnapshot::packed(width, height, rgb)
    }

    #[test]
    fn preparation_always_yields_the_models_static_shape() {
        for (width, height) in [(640, 360), (256, 144), (144, 256), (224, 126), (17, 9)] {
            let planar = prepare(&gradient(width, height)).expect("prepared");
            assert_eq!(
                planar.len(),
                3 * Dinov2SemanticEncoder::INPUT_WIDTH * Dinov2SemanticEncoder::INPUT_HEIGHT,
                "{width}x{height}"
            );
            assert!(planar.iter().all(|value| (0.0..=255.0).contains(value)));
        }
    }

    #[test]
    fn preparation_rejects_a_snapshot_whose_bytes_do_not_match_its_geometry() {
        assert_eq!(prepare(&FrameSnapshot::packed(64, 64, vec![0; 10])), None);
        assert_eq!(prepare(&FrameSnapshot::packed(0, 0, Vec::new())), None);
    }

    #[test]
    fn preparation_honours_row_padding() {
        let packed = gradient(40, 24);
        let mut padded = FrameSnapshot {
            width: 40,
            height: 24,
            bytes_per_row: 160,
            rgb: vec![0; 160 * 24],
        };
        for y in 0..24 {
            padded.rgb[y * 160..y * 160 + 120].copy_from_slice(&packed.rgb[y * 120..y * 120 + 120]);
        }
        assert_eq!(prepare(&packed), prepare(&padded));
    }

    /// A portrait picture and the same picture already rotated to landscape
    /// must prepare to the same tensor — that is the whole point of rotating
    /// instead of squashing.
    #[test]
    fn portrait_rotates_to_the_landscape_it_would_have_been() {
        let portrait = gradient(60, 100);
        let (width, height, rgb) = rotate_landscape(&portrait);
        let landscape = FrameSnapshot::packed(width, height, rgb);
        assert_eq!((width, height), (100, 60));
        assert_eq!(prepare(&portrait), prepare(&landscape));
    }

    #[test]
    fn rotation_is_clockwise() {
        // 2×3 portrait, distinct pixels by red channel:
        //   0 1        4 2 0
        //   2 3   -->  5 3 1
        //   4 5
        let mut rgb = vec![0_u8; 2 * 3 * 3];
        for index in 0..6 {
            rgb[index * 3] = u8::try_from(index).expect("small");
        }
        let (width, height, rotated) = rotate_landscape(&FrameSnapshot::packed(2, 3, rgb));
        assert_eq!((width, height), (3, 2));
        assert_eq!(
            rotated
                .chunks_exact(3)
                .map(|pixel| pixel[0])
                .collect::<Vec<_>>(),
            vec![4, 2, 0, 5, 3, 1]
        );
    }

    #[test]
    fn cosine_distance_is_zero_for_a_vector_against_itself() {
        let unit = {
            let mut vector = vec![0.0_f32; Dinov2SemanticEncoder::EMBEDDING_SIZE];
            vector[7] = 1.0;
            vector
        };
        assert!(Dinov2SemanticEncoder::cosine_distance(&unit, &unit).abs() < 1e-6);
        assert_eq!(Dinov2SemanticEncoder::cosine_distance(&unit, &[]), 1.0);
        assert_eq!(Dinov2SemanticEncoder::cosine_distance(&[], &[]), 1.0);
    }
}

#[cfg(test)]
mod current_pixel_contract {
    use super::*;
    fn reference(name: &str) -> Vec<u8> {
        let file =
            ferment_engine_test_fixtures::export::require_path(&format!("dinov2/goldens/{name}"));
        std::fs::read(file).expect("independent original DINO pixels required")
    }
    /// Kill: alter an already prepared image's channels or identity geometry.
    #[test]
    fn the_current_prepared_frame_is_the_original_doors_planes() {
        let golden = reference("dinov2_f118_image_in.u8");
        let plane = 224 * 126;
        assert_eq!(golden.len(), plane * 3);
        let rgb: Vec<_> = (0..plane)
            .flat_map(|i| [golden[i], golden[plane + i], golden[2 * plane + i]])
            .collect();
        let actual = prepare(&FrameSnapshot::packed(224, 126, rgb)).unwrap();
        let worst = actual
            .iter()
            .zip(golden)
            .map(|(a, b)| (*a - f32::from(b)).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1.0, "original identity pixel bound: {worst}");
    }
    /// Kill: choose a lower-quality kernel or loosen the original vImage pixel bound.
    #[test]
    fn the_current_synthetic_frame_meets_the_original_vimage_pixel_budget() {
        let rgb = reference("dinov2_synthetic_source_640x360.rgb");
        assert_eq!(rgb.len(), 640 * 360 * 3);
        let golden = reference("dinov2_synthetic_image_in.u8");
        let actual = prepare(&FrameSnapshot::packed(640, 360, rgb)).unwrap();
        assert_eq!(actual.len(), golden.len());
        let deltas: Vec<_> = actual
            .iter()
            .zip(golden)
            .map(|(a, b)| (*a - f32::from(b)).abs())
            .collect();
        let worst = deltas.iter().copied().fold(0.0f32, f32::max);
        let mean = deltas.iter().map(|v| f64::from(*v)).sum::<f64>() / deltas.len() as f64;
        println!("CURRENT_DINO_PIXELS meanAbs={mean} maxAbs={worst}");
        assert!(worst <= 2.0, "original maximum pixel error: {worst}");
        assert!(mean < 0.1, "original mean pixel error: {mean}");
    }
}
