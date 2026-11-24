//! Planar-tensor packing shared by the ORT-backed enhancement backings
//! (Real-ESRGAN Compact, FastDVDnet, RIFE).
//!
//! All three models consume planar `[N, C, H, W]` f32 in `[0, 1]` and emit the
//! same layout back. Frames cross this boundary as packed RGB24 `CpuFrame`s
//! (the guaranteed output of `VideoFrame::to_cpu_rgb`).

use ferment_engine_media::{CpuFrame, CpuPixelFormat};

/// Interleaved RGB24 → planar `[3, H, W]` f32 in `[0, 1]`, appended to `out`.
pub(crate) fn append_planar_rgb(frame: &CpuFrame, out: &mut Vec<f32>) {
    append_planar_rgb_padded(frame, frame.width as usize, frame.height as usize, out);
}

/// Like [`append_planar_rgb`], but edge-replicates to `padded_w` × `padded_h`
/// (RIFE requires spatial dims aligned to 32; the caller crops the output).
pub(crate) fn append_planar_rgb_padded(
    frame: &CpuFrame,
    padded_w: usize,
    padded_h: usize,
    out: &mut Vec<f32>,
) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    debug_assert!(padded_w >= width && padded_h >= height);
    let plane = padded_w * padded_h;
    let base = out.len();
    out.resize(base + 3 * plane, 0.0);
    let (reds, rest) = out[base..].split_at_mut(plane);
    let (greens, blues) = rest.split_at_mut(plane);
    for y in 0..padded_h {
        let source_y = y.min(height - 1);
        let row = &frame.data[source_y * frame.stride..];
        for x in 0..padded_w {
            let source_x = x.min(width - 1);
            let index = y * padded_w + x;
            reds[index] = f32::from(row[source_x * 3]) / 255.0;
            greens[index] = f32::from(row[source_x * 3 + 1]) / 255.0;
            blues[index] = f32::from(row[source_x * 3 + 2]) / 255.0;
        }
    }
}

/// Planar `[3, H, W]` f32 (top-left `width`×`height` crop of a possibly larger
/// `source_w`×`source_h` plane) → packed RGB24 frame, clamped to `[0, 1]`.
pub(crate) fn planar_rgb_frame(
    values: &[f32],
    source_w: usize,
    source_h: usize,
    width: u32,
    height: u32,
) -> Option<CpuFrame> {
    let plane = source_w * source_h;
    let (crop_w, crop_h) = (width as usize, height as usize);
    if values.len() < 3 * plane || crop_w > source_w || crop_h > source_h {
        return None;
    }
    let quantize = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut data = vec![0u8; 3 * crop_w * crop_h];
    for y in 0..crop_h {
        for x in 0..crop_w {
            let source = y * source_w + x;
            let target = (y * crop_w + x) * 3;
            data[target] = quantize(values[source]);
            data[target + 1] = quantize(values[plane + source]);
            data[target + 2] = quantize(values[2 * plane + source]);
        }
    }
    CpuFrame::new(width, height, crop_w * 3, CpuPixelFormat::Rgb24, data).ok()
}

#[cfg(test)]
pub(crate) mod testing {
    //! A capturing mock `InferenceBackend` for enhancement-backing tests.

    use std::sync::Mutex;

    use async_trait::async_trait;
    use ferment_engine_core::Result;
    use ferment_engine_infer::{InferenceBackend, ModelId, ModelVersion, Tensor, TensorMap};

    type Responder = Box<dyn Fn(ModelId, &TensorMap) -> Result<TensorMap> + Send + Sync>;

    pub(crate) struct MockBackend {
        pub captured: Mutex<Vec<(ModelId, TensorMap)>>,
        responder: Responder,
    }

    impl std::fmt::Debug for MockBackend {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("MockBackend")
                .finish_non_exhaustive()
        }
    }

    impl MockBackend {
        pub(crate) fn new(responder: Responder) -> Self {
            Self {
                captured: Mutex::new(Vec::new()),
                responder,
            }
        }

        /// Backend that answers every call with one constant-filled f32 tensor.
        pub(crate) fn constant(name: &'static str, shape: Vec<i64>, value: f32) -> Self {
            Self::new(Box::new(move |_, _| {
                let count = shape.iter().product::<i64>() as usize;
                let mut outputs = TensorMap::new();
                outputs.insert(
                    name.to_owned(),
                    Tensor::from_f32(shape.clone(), vec![value; count]).unwrap(),
                );
                Ok(outputs)
            }))
        }
    }

    #[async_trait]
    impl InferenceBackend for MockBackend {
        async fn infer(
            &self,
            model: ModelId,
            _version: &ModelVersion,
            inputs: TensorMap,
        ) -> Result<TensorMap> {
            let response = (self.responder)(model, &inputs);
            self.captured.lock().unwrap().push((model, inputs));
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_frame(width: u32, height: u32) -> CpuFrame {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for (index, value) in data.iter_mut().enumerate() {
            *value = (index % 251) as u8;
        }
        CpuFrame::new(
            width,
            height,
            width as usize * 3,
            CpuPixelFormat::Rgb24,
            data,
        )
        .unwrap()
    }

    #[test]
    fn planar_round_trip_preserves_pixels() {
        let frame = gradient_frame(5, 4);
        let mut values = Vec::new();
        append_planar_rgb(&frame, &mut values);
        assert_eq!(values.len(), 3 * 5 * 4);
        let back = planar_rgb_frame(&values, 5, 4, 5, 4).unwrap();
        assert_eq!(back.data, frame.data);
    }

    #[test]
    fn padded_packing_replicates_edges_and_crop_recovers() {
        let frame = gradient_frame(3, 2);
        let mut values = Vec::new();
        append_planar_rgb_padded(&frame, 4, 4, &mut values);
        assert_eq!(values.len(), 3 * 4 * 4);
        // Rightmost padded column replicates x=2; bottom rows replicate y=1.
        let red_plane = &values[..16];
        assert_eq!(red_plane[3], red_plane[2]);
        assert_eq!(red_plane[3 * 4], red_plane[4]);
        let cropped = planar_rgb_frame(&values, 4, 4, 3, 2).unwrap();
        assert_eq!(cropped.data, frame.data);
    }

    #[test]
    fn quantization_clamps_out_of_range_values() {
        let values = vec![-0.5, 2.0, 0.5];
        let frame = planar_rgb_frame(&values, 1, 1, 1, 1).unwrap();
        assert_eq!(&frame.data[..], &[0, 255, 128]);
    }
}
