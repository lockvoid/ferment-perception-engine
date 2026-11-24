//! U^2-Netp salient-object backing for [`SaliencyHeatSource`].
//!
//! Apple Vision has no off-Apple equivalent, so the heat map comes from
//! U^2-Netp (Apache-2.0, ~4.6 MB), a salient-OBJECT segmenter. Its map is
//! shaped nothing like Vision's attention map — near-binary rather than a
//! broad blob on a low pedestal — which is why
//! [`super::config::U2NETP_HEAT_FLOOR`] is re-measured rather than inherited.
//!
//! Pre-processing is the upstream `ToTensorLab(flag: 0)` recipe, quirks
//! included: the image is divided by ITS OWN maximum channel value, not by
//! 255, before the ImageNet shift/scale. Dividing by 255 instead would move
//! every activation on any frame that never reaches full white.

use async_trait::async_trait;
use ferment_engine_core::{
    EngineError, Result, RgbFrameRef, SaliencyHeatMap, SaliencyHeatSource, SaliencyMode,
};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use image::{RgbImage, imageops};

/// The exported graph is static at 320x320 — upstream's stated requirement for
/// the released weights.
pub const U2NETP_INPUT_EDGE: u32 = 320;

/// ImageNet channel statistics, as `ToTensorLab(flag: 0)` applies them.
const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const STD: [f64; 3] = [0.229, 0.224, 0.225];

/// Guard from upstream's `max(np.max(image), 1e-6)`: an all-black frame must
/// not divide by zero.
const MIN_IMAGE_PEAK: f64 = 1e-6;

const INPUT: &str = "image";
const OUTPUT: &str = "saliency";

#[derive(Debug, Clone)]
pub struct U2NetpHeatSource {
    backend: Backend,
    version: ModelVersion,
}

impl U2NetpHeatSource {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            version: ModelVersion::default(),
        }
    }
}

#[async_trait(?Send)]
impl SaliencyHeatSource for U2NetpHeatSource {
    fn mode(&self) -> SaliencyMode {
        SaliencyMode::Objectness
    }

    async fn heat(&self, frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap> {
        let mut inputs = TensorMap::new();
        inputs.insert(
            INPUT.into(),
            Tensor::from_f32(
                vec![
                    1,
                    3,
                    i64::from(U2NETP_INPUT_EDGE),
                    i64::from(U2NETP_INPUT_EDGE),
                ],
                preprocess(frame),
            )?,
        );
        let outputs = self
            .backend
            .infer(ModelId::U2Netp, &self.version, inputs)
            .await?;
        let mut values = outputs
            .get(OUTPUT)
            .ok_or_else(|| EngineError::ModelOutputMissing {
                model: ModelId::U2Netp.name().into(),
                key: OUTPUT.into(),
            })?
            .to_f32()?;
        let edge = U2NETP_INPUT_EDGE as usize;
        if values.len() != edge * edge {
            return Err(EngineError::inference(format!(
                "u2netp returned {} values, expected {}",
                values.len(),
                edge * edge
            )));
        }
        // A spatially constant image has no salient object. Convolutional
        // padding can still produce an asymmetric border response. The
        // relative heat centroid would treat that response as a subject.
        // Keep inference and output validation above (including model errors),
        // then express the absence of spatial evidence as an empty map. Use
        // exact RGB equality so no textured or low-contrast subject is erased.
        let first = &frame.data[..3];
        if frame.data.chunks_exact(3).all(|pixel| pixel == first) {
            values.fill(0.0);
        }
        Ok(SaliencyHeatMap {
            values,
            width: edge,
            height: edge,
        })
    }
}

/// Frame -> planar `[1, 3, 320, 320]` f32, upstream's normalization exactly.
/// Public to the crate so the ORT oracle can feed the same tensor the shipping
/// path builds.
pub(crate) fn preprocess(frame: RgbFrameRef<'_>) -> Vec<f32> {
    let resized = resize_to_model(frame);
    let peak = f64::from(resized.as_raw().iter().copied().max().unwrap_or(0)) / 255.0;
    let peak = peak.max(MIN_IMAGE_PEAK);
    let plane = (U2NETP_INPUT_EDGE * U2NETP_INPUT_EDGE) as usize;
    let mut values = vec![0.0f32; 3 * plane];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            let scaled = f64::from(pixel[channel]) / 255.0 / peak;
            values[channel * plane + index] = ((scaled - MEAN[channel]) / STD[channel]) as f32;
        }
    }
    values
}

fn resize_to_model(frame: RgbFrameRef<'_>) -> RgbImage {
    let image = RgbImage::from_raw(frame.width, frame.height, frame.data.to_vec())
        .expect("RgbFrameRef guarantees a packed RGB24 buffer");
    if frame.width == U2NETP_INPUT_EDGE && frame.height == U2NETP_INPUT_EDGE {
        return image;
    }
    imageops::resize(
        &image,
        U2NETP_INPUT_EDGE,
        U2NETP_INPUT_EDGE,
        imageops::FilterType::Lanczos3,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(value: u8, width: u32, height: u32) -> Vec<u8> {
        vec![value; (3 * width * height) as usize]
    }

    /// The `/ max` quirk: a mid-grey field and a white field normalize to the
    /// SAME tensor, because each is divided by its own peak. Dividing by 255
    /// would separate them — this is the assertion that catches that slip.
    #[test]
    fn normalization_divides_by_the_images_own_peak() {
        let grey = flat_frame(128, 320, 320);
        let white = flat_frame(255, 320, 320);
        let grey = preprocess(RgbFrameRef::new(&grey, 320, 320).unwrap());
        let white = preprocess(RgbFrameRef::new(&white, 320, 320).unwrap());
        assert_eq!(grey, white);
        // (1.0 - 0.485) / 0.229
        assert!((f64::from(grey[0]) - 2.248_908_296_943_23).abs() < 1e-6);
    }

    #[test]
    fn an_all_black_frame_does_not_divide_by_zero() {
        let black = flat_frame(0, 320, 320);
        let values = preprocess(RgbFrameRef::new(&black, 320, 320).unwrap());
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((f64::from(values[0]) + MEAN[0] / STD[0]).abs() < 1e-6);
    }

    #[test]
    fn channels_are_planar_and_shifted_per_channel() {
        let mut data = vec![0u8; 3 * 320 * 320];
        for pixel in data.chunks_exact_mut(3) {
            pixel.copy_from_slice(&[255, 128, 0]);
        }
        let values = preprocess(RgbFrameRef::new(&data, 320, 320).unwrap());
        let plane = 320 * 320;
        let expect = |raw: f64, channel: usize| (raw - MEAN[channel]) / STD[channel];
        assert!((f64::from(values[0]) - expect(1.0, 0)).abs() < 1e-6);
        assert!((f64::from(values[plane]) - expect(128.0 / 255.0, 1)).abs() < 1e-6);
        assert!((f64::from(values[2 * plane]) - expect(0.0, 2)).abs() < 1e-6);
    }

    #[test]
    fn a_non_square_frame_is_stretched_to_the_model_grid() {
        let data = flat_frame(200, 512, 288);
        let values = preprocess(RgbFrameRef::new(&data, 512, 288).unwrap());
        assert_eq!(values.len(), 3 * 320 * 320);
    }

    #[derive(Debug)]
    struct Provider {
        values: Option<Vec<f32>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait)]
    impl ferment_engine_infer::InferenceBackend for Provider {
        async fn infer(
            &self,
            model: ModelId,
            _version: &ModelVersion,
            _inputs: TensorMap,
        ) -> Result<TensorMap> {
            assert_eq!(model, ModelId::U2Netp);
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut result = TensorMap::new();
            if let Some(values) = &self.values {
                result.insert(
                    OUTPUT.into(),
                    Tensor::from_f32(vec![values.len() as i64], values.clone())?,
                );
            }
            Ok(result)
        }
    }

    // The provider deliberately returns border-biased heat. Every exact flat
    // colour must still execute it, then carry no localized object evidence.
    #[test]
    fn spatially_uniform_colours_still_run_the_provider_without_inventing_a_subject() {
        let provider = std::sync::Arc::new(Provider {
            values: Some(
                (0..320 * 320)
                    .map(|n| n as f32 / (320 * 320) as f32)
                    .collect(),
            ),
            calls: Default::default(),
        });
        let source = U2NetpHeatSource::new(provider.clone());
        for colour in [[0, 0, 0], [120, 120, 120], [255, 12, 79]] {
            let pixels = colour.repeat(6);
            let heat =
                futures_executor::block_on(source.heat(RgbFrameRef::new(&pixels, 3, 2).unwrap()))
                    .unwrap();
            assert_eq!((heat.width, heat.height), (320, 320));
            assert!(heat.values.iter().all(|&value| value == 0.0));
        }
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    // A single one-level RGB change is sufficient spatial evidence. This
    // guards against turning the exact-flat rule into a low-contrast filter.
    #[test]
    fn even_one_changed_colour_channel_preserves_every_provider_value() {
        let values: Vec<_> = (0..320 * 320)
            .map(|n| n as f32 / (320 * 320) as f32)
            .collect();
        let provider = std::sync::Arc::new(Provider {
            values: Some(values.clone()),
            calls: Default::default(),
        });
        let source = U2NetpHeatSource::new(provider.clone());
        let mut pixels = vec![120; 3 * 3 * 2];
        pixels[13] = 121;
        let heat =
            futures_executor::block_on(source.heat(RgbFrameRef::new(&pixels, 3, 2).unwrap()))
                .unwrap();
        assert_eq!(heat.values, values);
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    // Flat input cannot make missing model outputs or malformed tensor lengths
    // look like a successful centred answer.
    #[test]
    fn a_uniform_image_does_not_hide_missing_or_malformed_model_outputs() {
        let pixels = vec![120; 3 * 3 * 2];
        for values in [None, Some(vec![0.5; 3])] {
            let provider = std::sync::Arc::new(Provider {
                values,
                calls: Default::default(),
            });
            let source = U2NetpHeatSource::new(provider.clone());
            assert!(
                futures_executor::block_on(source.heat(RgbFrameRef::new(&pixels, 3, 2).unwrap()))
                    .is_err()
            );
            assert_eq!(provider.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }
}
