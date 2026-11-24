//! One grabbed frame -> one [`VideoSaliencySample`]. Port of Swift
//! `VideoSaliencyMeasure`.
//!
//! A face wins when there is one — for the framing question a face IS the
//! subject, and the detector's box is tighter than any heat map; with several
//! faces the dominant (largest) one is the subject. Otherwise the heat map
//! decides.

use ferment_engine_core::{
    EngineError, FaceRect, FaceRectSource, Rect, Result, RgbFrameRef, SaliencyHeatSource,
};

use super::config::VideoSaliencyConfig;
use super::{SampleSource, VideoSaliencySample};

/// Half-width of the confidence window on each axis: the share of heat-map
/// mass inside centroid +- this is the "is there one subject" number.
pub const CONFIDENCE_WINDOW: f64 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatMeasurement {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
}

impl HeatMeasurement {
    /// A map with no mass says nothing: centred, full frame, zero confidence.
    const EMPTY: Self = Self {
        cx: 0.5,
        cy: 0.5,
        w: 1.0,
        h: 1.0,
        confidence: 0.0,
    };
}

/// Face first, heat map otherwise — exactly the Swift order.
pub async fn measure(
    heat_source: &dyn SaliencyHeatSource,
    face_source: &dyn FaceRectSource,
    frame: RgbFrameRef<'_>,
    scene_index: usize,
    time: f64,
    config: &VideoSaliencyConfig,
) -> Result<VideoSaliencySample> {
    let faces = face_source.faces(frame).await?;
    if let Some(face) = dominant_face(&faces) {
        // A detector extrapolates the box of a face that leaves the frame —
        // Vision's A13 model shipped w=1.135 on a real clip, outside the unit
        // rect the wire contract promises. The subject the consumers reason
        // about is the VISIBLE part: intersect with the frame.
        let clipped = unit_clipped(Rect {
            x: face.x,
            y: face.y,
            width: face.width,
            height: face.height,
        });
        return Ok(VideoSaliencySample {
            scene_index,
            time,
            cx: clipped.x + clipped.width / 2.0,
            cy: clipped.y + clipped.height / 2.0,
            w: clipped.width,
            h: clipped.height,
            confidence: face.confidence,
            source: SampleSource::Face,
        });
    }

    let heat = heat_source.heat(frame).await?;
    if heat.values.len() != heat.width * heat.height {
        return Err(EngineError::inference(format!(
            "heat source returned {} values for a {}x{} map",
            heat.values.len(),
            heat.width,
            heat.height
        )));
    }
    let measurement = heat_measurement(&heat.values, heat.width, heat.height, config.heat_floor);
    Ok(VideoSaliencySample {
        scene_index,
        time,
        cx: measurement.cx,
        cy: measurement.cy,
        w: measurement.w,
        h: measurement.h,
        confidence: measurement.confidence,
        source: SampleSource::Saliency,
    })
}

/// A measured box confined to the frame it was measured in. A box already
/// inside comes back UNTOUCHED — intersecting grinds float dust into
/// coordinates it does not change, and the common path's numbers should be the
/// detector's own. A box entirely outside collapses to zero extent at the
/// clamped corner (Swift's `CGRect.intersection` answers `.null` there; no
/// real detection reaches it).
pub fn unit_clipped(rect: Rect) -> Rect {
    let inside =
        rect.x >= 0.0 && rect.y >= 0.0 && rect.x + rect.width <= 1.0 && rect.y + rect.height <= 1.0;
    if inside {
        return rect;
    }
    let left = rect.x.max(0.0);
    let top = rect.y.max(0.0);
    let right = (rect.x + rect.width).min(1.0);
    let bottom = (rect.y + rect.height).min(1.0);
    Rect {
        x: left,
        y: top,
        width: (right - left).max(0.0),
        height: (bottom - top).max(0.0),
    }
}

/// The dominant face — largest area — is the subject. Not the union: on a
/// crowd plus one foreground face the union spans the frame and its centre
/// lands on nobody; not the first: detector order is not prominence. Ties keep
/// the earlier face, matching Swift's `max(by:)`.
pub fn dominant_face(faces: &[FaceRect]) -> Option<&FaceRect> {
    let area = |face: &FaceRect| face.width * face.height;
    let mut best = faces.first()?;
    for face in &faces[1..] {
        if area(face) > area(best) {
            best = face;
        }
    }
    Some(best)
}

/// Pure: intensity-weighted centroid, +-2 sigma extent clamped to the frame,
/// and the mass share inside the confidence window.
///
/// `floor` (a fraction of the map's peak) is subtracted from every cell first:
/// the pedestal under the blob is wide enough to own most of the mass, and
/// without it the centroid reports "centre" for any frame.
///
/// Row 0 is the TOP of the frame — the map is an image buffer, not a
/// bottom-left bounding box.
///
/// # Panics
/// If `map` is shorter than `width * height`. [`measure`] rejects a source that
/// answers with a mis-shaped map before reaching here.
pub fn heat_measurement(map: &[f32], width: usize, height: usize, floor: f64) -> HeatMeasurement {
    assert!(
        map.len() >= width * height,
        "heat map is shorter than its own dimensions"
    );
    let peak = map.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let peak = if peak.is_finite() {
        f64::from(peak)
    } else {
        0.0
    };
    let cutoff = peak * floor;
    let floored = |x: usize, y: usize| (f64::from(map[y * width + x]) - cutoff).max(0.0);

    let mut mass = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for y in 0..height {
        let ny = (y as f64 + 0.5) / height as f64;
        for x in 0..width {
            let value = floored(x, y);
            mass += value;
            sum_x += value * (x as f64 + 0.5) / width as f64;
            sum_y += value * ny;
        }
    }
    if mass <= 0.0 {
        return HeatMeasurement::EMPTY;
    }

    let cx = sum_x / mass;
    let cy = sum_y / mass;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    let mut window_mass = 0.0;
    for y in 0..height {
        let ny = (y as f64 + 0.5) / height as f64;
        for x in 0..width {
            let value = floored(x, y);
            let nx = (x as f64 + 0.5) / width as f64;
            variance_x += value * (nx - cx) * (nx - cx);
            variance_y += value * (ny - cy) * (ny - cy);
            if (nx - cx).abs() <= CONFIDENCE_WINDOW && (ny - cy).abs() <= CONFIDENCE_WINDOW {
                window_mass += value;
            }
        }
    }
    let sigma_x = (variance_x / mass).sqrt();
    let sigma_y = (variance_y / mass).sqrt();
    HeatMeasurement {
        cx,
        cy,
        w: (cx + 2.0 * sigma_x).min(1.0) - (cx - 2.0 * sigma_x).max(0.0),
        h: (cy + 2.0 * sigma_y).min(1.0) - (cy - 2.0 * sigma_y).max(0.0),
        confidence: window_mass / mass,
    }
}

#[cfg(test)]
mod heat_math_tests {
    //! Ported from `VideoSaliencyMeasureMathTests.swift`: the heat-map moments
    //! on hand-built maps, no video, no model. Pins the background floor that
    //! the real maps made necessary — without it the pedestal under the blob
    //! owns most of the mass and the centroid reports "centre" for every frame.

    use super::*;

    fn map(width: usize, height: usize, value: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut cells = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                cells.push(value(x, y));
            }
        }
        cells
    }

    /// One hot cell at column 2, row 7 of a 10x10 map: centroid at its centre
    /// (0.25, 0.75) — x from the column, y from the ROW, top-left origin.
    #[test]
    fn a_single_hot_cell_is_the_centroid() {
        let heat = map(10, 10, |x, y| if x == 2 && y == 7 { 1.0 } else { 0.0 });
        let measurement = heat_measurement(&heat, 10, 10, 0.3);
        assert!((measurement.cx - 0.25).abs() < 1e-12);
        assert!((measurement.cy - 0.75).abs() < 1e-12);
        assert!(measurement.w.abs() < 1e-12);
        assert!(measurement.h.abs() < 1e-12);
        assert!((measurement.confidence - 1.0).abs() < 1e-12);
    }

    /// A flat map has nothing to say: centre, full frame, and a confidence
    /// equal to the +-0.15 window's share of the area — 6 of 20 columns x 6 of
    /// 20 rows = 0.09. A 20-wide map keeps every cell clear of the window
    /// edge; at 10 wide the cells at 0.35 and 0.65 sit exactly ON it and
    /// floating point picks the winner.
    #[test]
    fn a_flat_map_is_centred_and_full_frame() {
        let heat = map(20, 20, |_, _| 1.0);
        let measurement = heat_measurement(&heat, 20, 20, 0.3);
        assert!((measurement.cx - 0.5).abs() < 1e-12);
        assert!((measurement.cy - 0.5).abs() < 1e-12);
        assert!((measurement.w - 1.0).abs() < 1e-12);
        assert!((measurement.h - 1.0).abs() < 1e-12);
        assert!((measurement.confidence - 0.09).abs() < 1e-12);
    }

    #[test]
    fn a_map_with_no_mass_is_the_centred_full_frame_default() {
        let measurement = heat_measurement(&map(8, 8, |_, _| 0.0), 8, 8, 0.3);
        assert_eq!(measurement, HeatMeasurement::EMPTY);
    }

    /// The ruling of this round, pinned deterministically. A 4x4 blob of 1.0
    /// centred at (0.2, 0.2) on a 0.25 pedestal covering the other 384 cells:
    /// the pedestal carries 96 of the 112 total mass, so the raw centroid sits
    /// at 0.468 — near frame centre, nowhere near the blob. Subtracting
    /// 0.3 x peak deletes the pedestal outright and the centroid lands on the
    /// subject.
    #[test]
    fn the_floor_discards_the_pedestal_that_would_drag_the_centroid() {
        let heat = map(20, 20, |x, y| {
            if (2..=5).contains(&x) && (2..=5).contains(&y) {
                1.0
            } else {
                0.25
            }
        });

        let raw = heat_measurement(&heat, 20, 20, 0.0);
        assert!((raw.cx - 0.4679).abs() < 1e-4, "raw cx {}", raw.cx);
        assert!((raw.cy - 0.4679).abs() < 1e-4, "raw cy {}", raw.cy);

        let floored = heat_measurement(&heat, 20, 20, 0.3);
        assert!((floored.cx - 0.2).abs() < 1e-12);
        assert!((floored.cy - 0.2).abs() < 1e-12);
        assert!((floored.confidence - 1.0).abs() < 1e-12);
        assert!(floored.w < 0.25);
    }

    /// The floor is a fraction of the map's own peak, not an absolute value:
    /// the same shape at a tenth of the amplitude must measure identically.
    #[test]
    fn the_floor_scales_with_the_maps_peak() {
        let shape = |scale: f32| {
            map(20, 20, move |x, y| {
                if (2..=5).contains(&x) && (2..=5).contains(&y) {
                    scale
                } else {
                    0.25 * scale
                }
            })
        };
        let bright = heat_measurement(&shape(1.0), 20, 20, 0.3);
        let dim = heat_measurement(&shape(0.1), 20, 20, 0.3);
        assert!((bright.cx - dim.cx).abs() < 1e-9);
        assert!((bright.w - dim.w).abs() < 1e-9);
        assert!((bright.confidence - dim.confidence).abs() < 1e-9);
    }
}

#[cfg(test)]
mod source_contract_tests {
    //! A heat source that answers with a map that does not match its own
    //! dimensions is a bug in that source, and it has to say so out loud — the
    //! alternative is a centred full-frame verdict that looks like a real
    //! measurement.

    use async_trait::async_trait;
    use ferment_engine_core::{SaliencyHeatMap, SaliencyMode};

    use super::*;
    use crate::video_saliency::config::VideoSaliencyConfig;

    #[derive(Debug)]
    struct MisShapedHeat;

    #[async_trait(?Send)]
    impl SaliencyHeatSource for MisShapedHeat {
        fn mode(&self) -> SaliencyMode {
            SaliencyMode::Objectness
        }

        async fn heat(&self, _frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap> {
            Ok(SaliencyHeatMap {
                values: vec![0.5; 9],
                width: 8,
                height: 8,
            })
        }
    }

    #[derive(Debug)]
    struct NoFaces;

    #[async_trait(?Send)]
    impl FaceRectSource for NoFaces {
        async fn faces(&self, _frame: RgbFrameRef<'_>) -> Result<Vec<FaceRect>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn a_mis_shaped_heat_map_is_an_error_not_a_centred_default() {
        let pixels = vec![0u8; 3 * 4 * 4];
        let frame = RgbFrameRef::new(&pixels, 4, 4).unwrap();
        let error = futures_executor::block_on(measure(
            &MisShapedHeat,
            &NoFaces,
            frame,
            0,
            0.0,
            &VideoSaliencyConfig::default(),
        ))
        .expect_err("a 9-value 8x8 map must not measure");
        assert!(
            error.to_string().contains("9 values for a 8x8 map"),
            "unhelpful error: {error}"
        );
    }
}

#[cfg(test)]
mod dominant_face_tests {
    //! Ported from `VideoSaliencyDominantFaceTests.swift`. Several faces in one
    //! frame: the subject is the DOMINANT face (largest area), never the union.
    //! Real footage showed the union spanning a crowd plus one foreground face
    //! with the centroid landing on the sky between them.

    use super::*;

    fn rect(x: f64, y: f64, width: f64, height: f64) -> Rect {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    fn face(x: f64, y: f64, width: f64, height: f64) -> FaceRect {
        FaceRect {
            x,
            y,
            width,
            height,
            confidence: 0.9,
        }
    }

    /// The A13 payload the server refused (2026-08-20 smoke): the face lane
    /// extrapolates the box of a face that leaves the frame, and two scenes
    /// shipped w=1.111 / w=1.135 — outside the unit rect the wire contract
    /// promises. The measurement must be the box's INTERSECTION with the frame.
    #[test]
    fn face_box_is_clipped_to_the_unit_frame() {
        let hanging = rect(0.4, 0.1, 1.135, 0.7);
        let clipped = unit_clipped(hanging);
        assert!((clipped.width - 0.6).abs() < 1e-9);
        assert!((clipped.x - 0.4).abs() < 1e-9);

        let inside = rect(0.2, 0.2, 0.5, 0.5);
        assert_eq!(
            unit_clipped(inside),
            inside,
            "a box already inside the frame must come back untouched"
        );

        let overshoot_both = rect(-0.1, -0.2, 1.4, 1.5);
        assert_eq!(unit_clipped(overshoot_both), rect(0.0, 0.0, 1.0, 1.0));
    }

    /// The escape is not only rightward: a face leaving the top-left clips its
    /// origin, and the A13 payload's other scene (w=1.111) clips its width.
    #[test]
    fn clipping_handles_every_edge_the_a13_payload_crossed() {
        assert_eq!(
            unit_clipped(rect(-0.05, -0.05, 0.3, 0.3)),
            rect(0.0, 0.0, 0.25, 0.25)
        );
        let wide = unit_clipped(rect(0.0, 0.2, 1.111, 0.4));
        assert!((wide.width - 1.0).abs() < 1e-9);
        assert!((wide.height - 0.4).abs() < 1e-9);
        let below = unit_clipped(rect(0.3, 0.8, 0.2, 0.5));
        assert!((below.height - 0.2).abs() < 1e-9);
    }

    #[test]
    fn largest_area_face_wins_regardless_of_order() {
        let small = face(0.05, 0.70, 0.05, 0.08);
        let large = face(0.60, 0.30, 0.20, 0.30);
        let medium = face(0.40, 0.50, 0.10, 0.15);
        assert_eq!(dominant_face(&[small, large, medium]), Some(&large));
        assert_eq!(dominant_face(&[large, small]), Some(&large));
        assert_eq!(dominant_face(&[medium]), Some(&medium));
        assert_eq!(dominant_face(&[]), None);
    }

    /// Equal areas keep the earlier face, matching Swift's `max(by:)` — a
    /// detector that reorders its output must not flip the verdict.
    #[test]
    fn equal_areas_keep_the_first_face() {
        let first = face(0.1, 0.1, 0.2, 0.2);
        let second = face(0.6, 0.6, 0.2, 0.2);
        assert_eq!(dominant_face(&[first, second]), Some(&first));
    }
}
