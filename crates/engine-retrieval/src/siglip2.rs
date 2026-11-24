use crate::{Embedding, EmbeddingSpace, Siglip2Tokenizer, check_cancelled, encode, token_input};
use ferment_engine_core::{EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};

pub const SIGLIP2_IMAGE_SIZE: u32 = 224;

/// RGB8, no crop. Callers normalize orientation/color when decoding media.
pub fn siglip2_pixels(rgb: &[u8], width: u32, height: u32) -> Result<Tensor> {
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(3));
    if width == 0 || height == 0 || expected != Some(rgb.len() as u64) {
        return Err(EngineError::invalid_config(
            "SigLIP 2 expects packed, nonempty RGB8",
        ));
    }
    if width > 32768 || height > 32768 || rgb.len() > 300_000_000 {
        return Err(EngineError::invalid_config(
            "SigLIP 2 input exceeds 100 MP or 32768 pixels per side",
        ));
    }
    let resized = crate::resize::rgb_bilinear(
        rgb,
        width as usize,
        height as usize,
        SIGLIP2_IMAGE_SIZE as usize,
    );
    let count = (SIGLIP2_IMAGE_SIZE * SIGLIP2_IMAGE_SIZE) as usize;
    let mut pixels = vec![0.0; count * 3];
    for (index, pixel) in resized.chunks_exact(3).enumerate() {
        for channel in 0..3 {
            pixels[channel * count + index] = (f32::from(pixel[channel]) / 255.0 - 0.5) / 0.5;
        }
    }
    Tensor::from_f32(vec![1, 3, 224, 224], pixels)
}

pub struct Siglip2 {
    backend: Backend,
    version: ModelVersion,
}

impl Siglip2 {
    pub fn new(backend: Backend, version: ModelVersion) -> Self {
        Self { backend, version }
    }

    pub async fn image_rgb(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        progress: &ProgressSink,
    ) -> Result<Embedding> {
        check_cancelled(progress)?;
        encode(
            &self.backend,
            ModelId::Siglip2Image,
            &self.version,
            TensorMap::from([("pixels".into(), siglip2_pixels(rgb, width, height)?)]),
            EmbeddingSpace::Siglip2Base224V1,
            progress,
        )
        .await
    }

    pub async fn text(
        &self,
        tokenizer: &Siglip2Tokenizer,
        text: &str,
        progress: &ProgressSink,
    ) -> Result<Embedding> {
        check_cancelled(progress)?;
        encode(
            &self.backend,
            ModelId::Siglip2Text,
            &self.version,
            token_input(tokenizer.encode(text)?)?,
            EmbeddingSpace::Siglip2Base224V1,
            progress,
        )
        .await
    }
}
