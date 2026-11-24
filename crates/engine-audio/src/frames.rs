//! Decoded-audio container passed between layers.

/// Mono PCM samples + sample rate. Owned to keep async boundaries simple.
#[derive(Debug, Clone)]
pub struct AudioFrames {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// The bytes of a particular decode/SRC request, never its source path.
    pub identity: Option<crate::derived_cache::DerivedSource>,
}

impl AudioFrames {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            identity: None,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / f64::from(self.sample_rate)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_matches_sample_count() {
        let f = AudioFrames::new(vec![0.0; 16_000], 16_000);
        assert!((f.duration_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn duration_is_zero_for_zero_rate() {
        let f = AudioFrames::new(vec![0.0; 100], 0);
        assert_eq!(f.duration_secs(), 0.0);
    }
}
