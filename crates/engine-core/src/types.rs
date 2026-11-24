use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub extension: String,
}

impl ModelFile {
    pub fn new(name: impl Into<String>, extension: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            extension: extension.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: f64,
    pub duration: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaTime {
    pub value: i64,
    pub timescale: i32,
}

impl MediaTime {
    pub fn new(value: i64, timescale: i32) -> Self {
        Self { value, timescale }
    }

    pub fn seconds(self) -> f64 {
        if self.timescale == 0 {
            0.0
        } else {
            self.value as f64 / f64::from(self.timescale)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaTimeRange {
    pub start: MediaTime,
    pub duration: MediaTime,
}

impl TimeRange {
    pub fn new(start: f64, duration: f64) -> Self {
        Self { start, duration }
    }

    pub fn end(self) -> f64 {
        self.start + self.duration
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Bgra8,
    Nv12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaneLayout {
    pub offset: usize,
    pub stride: usize,
    pub row_bytes: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoFrame {
    pub data: Arc<[u8]>,
    pub planes: Vec<PlaneLayout>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub presentation_time: MediaTime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            self.samples.len() as f64 / f64::from(self.sample_rate) / f64::from(self.channels)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaArtifact {
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ImageFormat {
    Png,
    Jpeg { quality: f32 },
}

impl ImageFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg { .. } => "jpg",
        }
    }

    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg { .. } => "image/jpeg",
        }
    }
}
