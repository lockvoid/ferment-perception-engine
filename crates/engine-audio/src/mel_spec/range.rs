//! Global STFT coordinates over bounded source reads. Padding and signal
//! pre-emphasis belong to the original signal, never to an interior tile.
use super::{Compression, MelFrames, MelSpectrogram, Padding, PreEmphasis};
use ferment_engine_core::{AudioSampleSource, EngineError, Result};

impl MelSpectrogram {
    /// Compute up to 4096 consecutive spectrogram frames without materializing
    /// the whole signal. `start` is an absolute STFT frame, not a sample offset.
    /// The source is mono at the recipe's sample rate. A source read failure or
    /// a short read is an error; unavailable samples are never padded as silence.
    /// Global-maximum-relative decibels require a separate reduction and are
    /// deliberately unsupported here. All frame-local compressions are exact.
    pub fn compute_range(
        &self,
        source: &dyn AudioSampleSource,
        start: usize,
        count: usize,
    ) -> Result<MelFrames> {
        let spec = &self.spec;
        if count > 4096 || matches!(spec.compression, Compression::Decibels { .. }) {
            return Err(EngineError::invalid_config(
                "Mel ranges require at most 4096 frames and frame-local compression",
            ));
        }
        let n = source.sample_count();
        if n > isize::MAX as usize - spec.fft_size {
            return Err(EngineError::invalid_config(
                "Audio sample count exceeds range",
            ));
        }
        let count = count.min(spec.frame_count(n).saturating_sub(start));
        if count == 0 {
            return Ok(MelFrames {
                bands: spec.mel_bands,
                values: Vec::new(),
            });
        }
        let pad = if spec.padding == Padding::None {
            0
        } else {
            spec.fft_size / 2
        };
        let first = (start as i128) * spec.hop_length as i128 - pad as i128;
        let length = (count - 1)
            .checked_mul(spec.hop_length)
            .and_then(|len| len.checked_add(self.window.len()))
            .ok_or_else(|| EngineError::invalid_config("Mel range overflow"))?;
        let index = |at: usize| -> Option<usize> {
            let i = first + at as i128;
            if (0..n as i128).contains(&i) {
                Some(i as usize)
            } else if spec.padding == Padding::Reflect {
                // Exactly the existing whole-signal pad, including its
                // saturating edge rule when the entire source is very short.
                Some(if i < 0 {
                    (-i).min((n - 1) as i128) as usize
                } else {
                    (2 * n as i128 - 2 - i).max(0) as usize
                })
            } else {
                None
            }
        };
        let mut lo = n;
        let mut hi = 0;
        for i in (0..length).filter_map(index) {
            lo = lo.min(i);
            hi = hi.max(i);
        }
        let signal_alpha = match spec.pre_emphasis {
            Some(PreEmphasis::Signal { alpha }) => Some(alpha),
            _ => None,
        };
        if signal_alpha.is_some() {
            lo = lo.saturating_sub(1);
        }
        let mut raw = vec![0.; hi + 1 - lo];
        let size = raw.len();
        if source.copy_samples(&mut raw, lo, size)? != size {
            return Err(EngineError::invalid_config("Incomplete PCM source range"));
        }
        if spec.input_scale != 1.0 {
            for value in &mut raw {
                *value *= spec.input_scale;
            }
        }
        let mut padded = vec![0.; length];
        for (at, value) in padded.iter_mut().enumerate() {
            if let Some(i) = index(at) {
                *value = raw[i - lo];
                if let Some(alpha) = signal_alpha.filter(|_| i != 0) {
                    *value -= alpha * raw[i - lo - 1];
                }
            }
        }
        self.compute_padded(&padded)
    }
}
