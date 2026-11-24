//! YuNet face-rectangle backing for [`FaceRectSource`].
//!
//! YuNet (MIT, Shiqi Yu) replaces Vision's `VNDetectFaceRectanglesRequest`. The
//! decode below is transcribed from OpenCV's own `FaceDetectorYNImpl`
//! (`modules/objdetect/src/face_detect.cpp`): anchor-free, three strides, score
//! = sqrt(cls * obj), box = (cell + offset) * stride with exponential extent.
//!
//! Pre-processing is OpenCV's `blobFromImage` defaults — raw 0..255 floats in
//! **BGR** channel order, no mean subtraction, no scaling. The fleet's graph
//! (`face_detection_yunet_2026may`) carries SYMBOLIC height/width — the
//! `2023mar` weights re-exported for ONNX Runtime engines — but this source
//! still letterboxes every frame to 640x640: the score threshold below was
//! measured at that size, and identical weights at an identical size give
//! identical scores. A frame is scaled uniformly to fit and padded at the
//! RIGHT and BOTTOM, matching OpenCV's `padWithDivisor`; detections divide
//! the scale back out and never see the padding. Other input sizes are now a
//! choice (re-measure the threshold before taking one), not a constraint.

use async_trait::async_trait;
use ferment_engine_core::{EngineError, FaceRect, FaceRectSource, Result, RgbFrameRef};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use image::{Rgb, RgbImage, imageops};

/// The measured input size. The `2023mar` graph was static at 640x640; the
/// `2026may` re-export is dynamic, and 640 stays the size the threshold and
/// the goldens were taken at.
pub const YUNET_INPUT_EDGE: u32 = 640;

/// Feature-map strides the graph emits, in output-name order.
const STRIDES: [u32; 3] = [8, 16, 32];

/// OpenCV Zoo ships 0.9 (`demo.py --conf_threshold`), kept here as the
/// upstream reference. Measured on the FermentEngine video fixtures
/// (`scripts/exports/video_saliency/measure_face_threshold.py`), YuNet's scores
/// separate cleanly: true faces land at 0.900-0.935 and the worst false
/// positive (a DJ's head seen from behind) at 0.602. 0.9 therefore sits
/// exactly ON two of the true positives, and a sub-pixel resize difference is
/// enough to drop them -- it did, on `djs-walking-city` at 1.86 s.
const OPENCV_ZOO_SCORE_THRESHOLD: f64 = 0.9;

/// The midpoint of that measured gap: 0.148 of margin above the worst false
/// positive, 0.150 below the weakest true face. A false face is expensive here
/// -- a face wins over the heat map outright, so a spurious box moves the whole
/// scene verdict -- and so is a miss, which drops the scene into the
/// differently scaled saliency lane.
pub const SCORE_THRESHOLD: f64 = (OPENCV_ZOO_SCORE_THRESHOLD + WORST_MEASURED_FALSE_POSITIVE) / 2.0;

/// The highest score any non-face reached across the fixture sweep.
const WORST_MEASURED_FALSE_POSITIVE: f64 = 0.6;

/// OpenCV Zoo's shipped NMS IoU (`demo.py --nms_threshold`).
const NMS_IOU_THRESHOLD: f64 = 0.3;

/// OpenCV's `top_k` guard: cap the candidate list before the quadratic NMS.
const TOP_K: usize = 5_000;

const INPUT: &str = "input";

#[derive(Debug, Clone)]
pub struct YuNetFaceSource {
    backend: Backend,
    version: ModelVersion,
}

impl YuNetFaceSource {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }
}

#[async_trait(?Send)]
impl FaceRectSource for YuNetFaceSource {
    async fn faces(&self, frame: RgbFrameRef<'_>) -> Result<Vec<FaceRect>> {
        let (values, scale) = preprocess(frame);
        let mut inputs = TensorMap::new();
        inputs.insert(
            INPUT.into(),
            Tensor::from_f32(
                vec![
                    1,
                    3,
                    i64::from(YUNET_INPUT_EDGE),
                    i64::from(YUNET_INPUT_EDGE),
                ],
                values,
            )?,
        );
        let outputs = self
            .backend
            .infer(ModelId::YuNetFace, &self.version, inputs)
            .await?;

        let mut candidates = Vec::new();
        for stride in STRIDES {
            let cls = output(&outputs, &format!("cls_{stride}"))?;
            let obj = output(&outputs, &format!("obj_{stride}"))?;
            let bbox = output(&outputs, &format!("bbox_{stride}"))?;
            decode_stride(&cls, &obj, &bbox, stride, SCORE_THRESHOLD, &mut candidates);
        }
        Ok(finish(candidates, scale, frame.width, frame.height))
    }
}

/// One detection in padded-input pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Detection {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub score: f64,
}

/// Frame -> planar `[1, 3, 640, 640]` f32 BGR, plus the uniform scale applied.
pub(crate) fn preprocess(frame: RgbFrameRef<'_>) -> (Vec<f32>, f64) {
    let edge = YUNET_INPUT_EDGE;
    let scale = f64::from(edge) / f64::from(frame.width.max(frame.height));
    let width = ((f64::from(frame.width) * scale).round() as u32).clamp(1, edge);
    let height = ((f64::from(frame.height) * scale).round() as u32).clamp(1, edge);
    let source = RgbImage::from_raw(frame.width, frame.height, frame.data.to_vec())
        .expect("RgbFrameRef guarantees a packed RGB24 buffer");
    let fitted = if (frame.width, frame.height) == (width, height) {
        source
    } else {
        imageops::resize(&source, width, height, imageops::FilterType::Lanczos3)
    };
    let mut padded = RgbImage::from_pixel(edge, edge, Rgb([0, 0, 0]));
    imageops::replace(&mut padded, &fitted, 0, 0);

    let plane = (edge * edge) as usize;
    let mut values = vec![0.0f32; 3 * plane];
    for (index, pixel) in padded.pixels().enumerate() {
        values[index] = f32::from(pixel[2]); // B
        values[plane + index] = f32::from(pixel[1]); // G
        values[2 * plane + index] = f32::from(pixel[0]); // R
    }
    (values, scale)
}

/// OpenCV's `postProcess` inner loop for one stride.
pub(crate) fn decode_stride(
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    stride: u32,
    score_threshold: f64,
    out: &mut Vec<Detection>,
) {
    let columns = (YUNET_INPUT_EDGE / stride) as usize;
    let rows = columns;
    let stride = f64::from(stride);
    for row in 0..rows {
        for column in 0..columns {
            let index = row * columns + column;
            let (Some(&cls), Some(&obj)) = (cls.get(index), obj.get(index)) else {
                return;
            };
            let score = (f64::from(cls).clamp(0.0, 1.0) * f64::from(obj).clamp(0.0, 1.0)).sqrt();
            if score < score_threshold {
                continue;
            }
            let Some(offsets) = bbox.get(index * 4..index * 4 + 4) else {
                return;
            };
            let cx = (column as f64 + f64::from(offsets[0])) * stride;
            let cy = (row as f64 + f64::from(offsets[1])) * stride;
            let width = f64::from(offsets[2]).exp() * stride;
            let height = f64::from(offsets[3]).exp() * stride;
            out.push(Detection {
                x: cx - width / 2.0,
                y: cy - height / 2.0,
                width,
                height,
                score,
            });
        }
    }
}

/// Greedy NMS, then padded-input pixels -> frame-normalized, top-left-origin
/// boxes. OpenCV rounds the boxes to integers before its IoU; we keep them in
/// floats, which cannot move a decision at an IoU threshold of 0.3.
pub(crate) fn finish(
    mut candidates: Vec<Detection>,
    scale: f64,
    frame_width: u32,
    frame_height: u32,
) -> Vec<FaceRect> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(TOP_K);
    let kept = non_maximum_suppression(&candidates, NMS_IOU_THRESHOLD);
    let normalize_x = 1.0 / (f64::from(frame_width) * scale);
    let normalize_y = 1.0 / (f64::from(frame_height) * scale);
    kept.into_iter()
        .map(|face| FaceRect {
            x: face.x * normalize_x,
            y: face.y * normalize_y,
            width: face.width * normalize_x,
            height: face.height * normalize_y,
            confidence: face.score,
        })
        .collect()
}

pub(crate) fn non_maximum_suppression(
    candidates: &[Detection],
    iou_threshold: f64,
) -> Vec<Detection> {
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in candidates {
        if kept
            .iter()
            .all(|keeper| intersection_over_union(candidate, keeper) < iou_threshold)
        {
            kept.push(*candidate);
        }
    }
    kept
}

fn intersection_over_union(a: &Detection, b: &Detection) -> f64 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let union = a.width * a.height + b.width * b.height - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn output(outputs: &TensorMap, key: &str) -> Result<Vec<f32>> {
    outputs
        .get(key)
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: ModelId::YuNetFace.name().into(),
            key: key.into(),
        })?
        .to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(x: f64, y: f64, size: f64, score: f64) -> Detection {
        Detection {
            x,
            y,
            width: size,
            height: size,
            score,
        }
    }

    #[test]
    fn letterbox_scales_uniformly_and_pads_bottom_right() {
        let data = vec![255u8; 3 * 512 * 288];
        let (values, scale) = preprocess(RgbFrameRef::new(&data, 512, 288).unwrap());
        assert!((scale - 640.0 / 512.0).abs() < 1e-12);
        let plane = (YUNET_INPUT_EDGE * YUNET_INPUT_EDGE) as usize;
        assert_eq!(values.len(), 3 * plane);
        // 288 * 1.25 = 360 rows of content, the rest is zero padding.
        let row = |y: usize| values[y * YUNET_INPUT_EDGE as usize];
        assert!((row(0) - 255.0).abs() < 1e-6);
        assert!((row(359) - 255.0).abs() < 1e-6);
        assert_eq!(row(360), 0.0);
        assert_eq!(row(639), 0.0);
    }

    /// OpenCV feeds BGR because `blobFromImage` does not swap channels. A frame
    /// packed RGB must arrive at the graph reversed.
    #[test]
    fn channels_reach_the_graph_in_bgr_order() {
        let mut data = vec![0u8; 3 * 640 * 640];
        for pixel in data.chunks_exact_mut(3) {
            pixel.copy_from_slice(&[10, 20, 30]);
        }
        let (values, _) = preprocess(RgbFrameRef::new(&data, 640, 640).unwrap());
        let plane = (YUNET_INPUT_EDGE * YUNET_INPUT_EDGE) as usize;
        assert_eq!(values[0], 30.0);
        assert_eq!(values[plane], 20.0);
        assert_eq!(values[2 * plane], 10.0);
    }

    /// One anchor at row 1, column 2 of the stride-32 grid, decoded by hand
    /// from OpenCV's formula: cx = (2 + 0.5) * 32, w = exp(1) * 32.
    #[test]
    fn stride_decode_matches_the_opencv_formula() {
        let cells = (YUNET_INPUT_EDGE / 32) as usize;
        let mut cls = vec![0.0f32; cells * cells];
        let mut obj = vec![0.0f32; cells * cells];
        let mut bbox = vec![0.0f32; cells * cells * 4];
        let (row, column) = (1usize, 2usize);
        let index = row * cells + column;
        cls[index] = 1.0;
        obj[index] = 0.81;
        bbox[index * 4..index * 4 + 4].copy_from_slice(&[0.5, 0.25, 1.0, 0.0]);

        let mut out = Vec::new();
        decode_stride(&cls, &obj, &bbox, 32, 0.5, &mut out);
        assert_eq!(out.len(), 1);
        let face = out[0];
        assert!((face.score - 0.9).abs() < 1e-6);
        let width = std::f64::consts::E * 32.0;
        assert!((face.width - width).abs() < 1e-9);
        assert!((face.height - 32.0).abs() < 1e-9);
        assert!((face.x - ((column as f64 + 0.5) * 32.0 - width / 2.0)).abs() < 1e-9);
        assert!((face.y - ((row as f64 + 0.25) * 32.0 - 16.0)).abs() < 1e-9);
    }

    #[test]
    fn the_score_is_the_geometric_mean_of_cls_and_obj_and_clamps() {
        let cells = (YUNET_INPUT_EDGE / 32) as usize;
        let mut cls = vec![0.0f32; cells * cells];
        let mut obj = vec![0.0f32; cells * cells];
        let bbox = vec![0.0f32; cells * cells * 4];
        cls[0] = 4.0; // out of range: clamps to 1
        obj[0] = 0.25;
        let mut out = Vec::new();
        decode_stride(&cls, &obj, &bbox, 32, 0.4, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn nms_keeps_the_strongest_of_an_overlapping_pair() {
        let candidates = vec![
            detection(0.0, 0.0, 100.0, 0.99),
            detection(10.0, 10.0, 100.0, 0.95),
            detection(300.0, 300.0, 100.0, 0.93),
        ];
        let kept = non_maximum_suppression(&candidates, NMS_IOU_THRESHOLD);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.99).abs() < 1e-12);
        assert!((kept[1].score - 0.93).abs() < 1e-12);
    }

    /// The scale divides back out: a box covering the whole letterboxed
    /// content area is the whole frame in normalized coordinates.
    #[test]
    fn detections_normalize_back_onto_the_source_frame() {
        let scale = 640.0 / 512.0;
        let faces = finish(
            vec![Detection {
                x: 0.0,
                y: 0.0,
                width: 512.0 * scale,
                height: 288.0 * scale,
                score: 0.95,
            }],
            scale,
            512,
            288,
        );
        assert_eq!(faces.len(), 1);
        assert!((faces[0].width - 1.0).abs() < 1e-12);
        assert!((faces[0].height - 1.0).abs() < 1e-12);
        assert!((faces[0].confidence - 0.95).abs() < 1e-12);
    }
}
