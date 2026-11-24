//! Codec discovery is a supported Java MediaCodecList operation on API33–35.
//! The NDK type-only factory can choose an8-bit first encoder even if a later
//! codec supports Main10. A single context-free platform resolver is installed
//! at engine startup; no operation/frame/model lifetime is captured here.
use crate::{Result, VideoError};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone)]
pub struct EncoderQuery {
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub bit_rate: u32,
    pub frame_rate: f64,
    pub profile: Option<i32>,
    pub color_format: i32,
    pub color_standard: Option<i32>,
    pub color_transfer: Option<i32>,
    pub color_range: Option<i32>,
}

pub trait EncoderSelector: Send + Sync {
    fn select(&self, query: EncoderQuery) -> Result<Vec<String>>;
}
static SELECTOR: OnceLock<Arc<dyn EncoderSelector>> = OnceLock::new();

/// Repeated engine starts do not accumulate foreign callbacks. The installed
/// resolver owns only platform discovery behavior for this process.
pub fn install_encoder_selector(selector: Arc<dyn EncoderSelector>) {
    let _ = SELECTOR.set(selector);
}

pub(crate) fn encoder_names(query: EncoderQuery) -> Result<Option<Vec<String>>> {
    let Some(selector) = SELECTOR.get() else {
        return Ok(None);
    };
    let names = selector.select(query)?;
    if names.len() > 256
        || names
            .iter()
            .any(|name| name.is_empty() || name.len() > 256 || name.contains('\0'))
    {
        return Err(VideoError::InvalidData(
            "codec resolver returned invalid names/count".into(),
        ));
    }
    let mut unique = Vec::with_capacity(names.len());
    for name in names {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    Ok(Some(unique))
}
