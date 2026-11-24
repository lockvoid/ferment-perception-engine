//! Supported Android codec discovery. This API is exported on the HOST too,
//! because the Kotlin binding is generated from that library. Only Android
//! uses it; this never loads a model, retains an Activity, or carries pixels.
use std::sync::Arc;

#[derive(Debug, Clone, uniffi::Record)]
pub struct AndroidEncoderQuery {
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

#[derive(Debug, Clone, uniffi::Record)]
pub struct AndroidEncoderSelection {
    pub codec_names: Vec<String>,
    pub failure: Option<String>,
}

/// Foreign implementations catch every throwable into `failure`. An
/// unexpected exception from a foreign callback is not a recoverable codec
/// refusal in UniFFI, so this contract deliberately returns a value.
#[uniffi::export(with_foreign)]
pub trait AndroidEncoderSelector: Send + Sync {
    fn select(&self, query: AndroidEncoderQuery) -> AndroidEncoderSelection;
}

#[uniffi::export]
pub fn install_android_encoder_selector(selector: Arc<dyn AndroidEncoderSelector>) {
    #[cfg(target_os = "android")]
    {
        struct Adapter(Arc<dyn AndroidEncoderSelector>);
        impl ferment_engine_media::platform::android_codec_selection::EncoderSelector for Adapter {
            fn select(
                &self,
                query: ferment_engine_media::platform::android_codec_selection::EncoderQuery,
            ) -> ferment_engine_media::Result<Vec<String>> {
                let result = self.0.select(AndroidEncoderQuery {
                    mime: query.mime,
                    width: query.width,
                    height: query.height,
                    bit_rate: query.bit_rate,
                    frame_rate: query.frame_rate,
                    profile: query.profile,
                    color_format: query.color_format,
                    color_standard: query.color_standard,
                    color_transfer: query.color_transfer,
                    color_range: query.color_range,
                });
                if let Some(failure) = result.failure {
                    return Err(ferment_engine_media::VideoError::UnsupportedCodec(format!(
                        "Android codec discovery failed: {failure}"
                    )));
                }
                Ok(result.codec_names)
            }
        }
        ferment_engine_media::platform::android_codec_selection::install_encoder_selector(
            Arc::new(Adapter(selector)),
        );
    }
    #[cfg(not(target_os = "android"))]
    let _ = selector;
}
