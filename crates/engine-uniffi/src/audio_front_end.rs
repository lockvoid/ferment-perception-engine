//! Optional derived PCM/mel cache and the same decoder used by processors.
use std::path::Path;
use std::sync::Arc;

use crate::{EngineFfiError, FfiResult};
use ferment_engine_audio::MelSpec;
use ferment_engine_audio::derived_cache::{DerivedFileCache, DerivedSource};
use ferment_engine_processors::audio_source::{AudioSampleSource, AudioSource};

#[derive(uniffi::Object)]
pub struct DerivedAudioCache {
    pub(crate) inner: Arc<DerivedFileCache>,
}

#[uniffi::export]
impl DerivedAudioCache {
    #[uniffi::constructor]
    pub fn new(directory: String, budget_bytes: u64) -> FfiResult<Self> {
        Ok(Self {
            inner: Arc::new(DerivedFileCache::new(directory.into(), budget_bytes)?),
        })
    }
    pub fn directory(&self) -> String {
        self.inner.directory.to_string_lossy().into_owned()
    }
    pub fn total_bytes(&self) -> u64 {
        self.inner.total_bytes()
    }
    pub fn remove_source(&self, source_id: String) -> FfiResult<()> {
        Ok(self.inner.remove_source(&DerivedSource::new(source_id))?)
    }
    pub fn remove_all(&self) -> FfiResult<()> {
        Ok(self.inner.remove_all()?)
    }
}

#[derive(uniffi::Record)]
pub struct ResampledPcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub identity: String,
}

#[derive(uniffi::Record)]
pub struct PreparedPcm {
    pub samples: Vec<f32>,
    pub path: Option<String>,
    pub owned_by_cache: bool,
}

#[derive(uniffi::Record)]
pub struct MelFrameBuffer {
    pub values: Vec<f32>,
    pub bands: u32,
}

/// This object loads no model and does not require an ORT environment.
#[derive(uniffi::Object)]
pub struct AudioFrontEnd {
    cache: Option<Arc<DerivedAudioCache>>,
}

#[uniffi::export]
impl AudioFrontEnd {
    #[uniffi::constructor]
    pub fn new(cache: Option<Arc<DerivedAudioCache>>) -> Self {
        Self { cache }
    }

    pub fn source_identity(&self, path: String) -> FfiResult<String> {
        Ok(DerivedSource::from_file(Path::new(&path))?.id)
    }

    pub fn load(
        &self,
        path: String,
        sample_rate: u32,
        source_id: Option<String>,
    ) -> FfiResult<ResampledPcm> {
        let audio = futures_executor::block_on(AudioSource::default().resampled(
            Path::new(&path),
            sample_rate,
            source_id.map(DerivedSource::new),
            self.cache.as_ref().map(|c| c.inner.as_ref()),
        ))?;
        Ok(ResampledPcm {
            samples: audio.samples,
            sample_rate: audio.sample_rate,
            identity: audio.identity.expect("resampled request identity").id,
        })
    }

    /// Cache hits are always mapped by the caller. Without a cache, short
    /// sources become arrays and long sources transfer an owned temporary.
    pub fn prepare_source(
        &self,
        path: String,
        sample_rate: u32,
        source_id: Option<String>,
        memory_limit_seconds: f64,
        temporary_directory: String,
    ) -> FfiResult<PreparedPcm> {
        if !memory_limit_seconds.is_finite() || memory_limit_seconds < 0.0 {
            return Err(EngineFfiError::new(
                "memory limit must be finite and nonnegative",
            ));
        }
        if let Some(cache) = &self.cache {
            let source = futures_executor::block_on(AudioSource::default().cached_source(
                Path::new(&path),
                sample_rate,
                source_id.map(DerivedSource::new),
                &cache.inner,
            ))?;
            return Ok(PreparedPcm {
                samples: Vec::new(),
                path: Some(source.path().to_string_lossy().into_owned()),
                owned_by_cache: true,
            });
        }
        let source = futures_executor::block_on(AudioSource::default().decode_disk_backed_in(
            Path::new(&path),
            sample_rate,
            Path::new(&temporary_directory),
        ))?;
        if source.sample_count() as f64 <= memory_limit_seconds * f64::from(sample_rate) {
            let mut samples = vec![0.0; source.sample_count()];
            source.copy_samples(&mut samples, 0, source.sample_count())?;
            Ok(PreparedPcm {
                samples,
                path: None,
                owned_by_cache: false,
            })
        } else {
            Ok(PreparedPcm {
                samples: Vec::new(),
                path: Some(source.into_path().to_string_lossy().into_owned()),
                owned_by_cache: false,
            })
        }
    }

    /// Spec JSON is the native/current Swift Codable contract, not a second
    /// restatement of the numerical front-end in Kotlin.
    pub fn mel(
        &self,
        spec_json: String,
        samples: Vec<f32>,
        source_id: Option<String>,
    ) -> FfiResult<MelFrameBuffer> {
        let spec: MelSpec = serde_json::from_str(&spec_json)
            .map_err(|e| EngineFfiError::new(format!("invalid mel spec: {e}")))?;
        let source = source_id.map(DerivedSource::new);
        let frames = spec.perform(
            &samples,
            source.as_ref(),
            self.cache.as_ref().map(|c| c.inner.as_ref()),
        )?;
        Ok(MelFrameBuffer {
            values: frames.values,
            bands: frames.bands as u32,
        })
    }
}
