//! Bounded-source BeatThis. Windows retain the original global STFT/model
//! coordinates; a disk-backed exact DBN sees the complete sequence of logits.
use super::*;
use crate::beatthis::{config::MEL_SPEC, storage::Logits};
use ferment_engine_audio::MelSpectrogram;
use ferment_engine_core::AudioSampleSource;

impl BeatThis {
    /// Analyze mono PCM already at 22050 Hz. The host supplies empty temporary
    /// scratch files and owns their lifetime/disk admission. PCM/mel/activations
    /// and Viterbi history never become whole-track resident arrays. The result
    /// has an explicit event quota, checked before quantized expansion; exceeding
    /// it is an error, not a partial/fabricated beat grid. No native callback I/O.
    pub async fn detect_source(
        &self,
        source: &dyn AudioSampleSource,
        scratch: &mut dyn FnMut() -> Result<std::fs::File>,
        max_events: usize,
        progress: ProgressSink,
    ) -> Result<BeatAnalysisResult> {
        let check = || {
            if progress.is_closed() {
                Err(EngineError::Cancelled)
            } else {
                Ok(())
            }
        };
        check()?;
        if max_events == 0 {
            return Err(EngineError::invalid_config("Empty beat result quota"));
        }
        let n = MEL_SPEC.frame_count(source.sample_count());
        if n == 0 {
            return Err(EngineError::AudioTooShort {
                samples: 0,
                min_required: super::super::config::FFT_SIZE,
            });
        }
        let duration = source.sample_count() as f64 / f64::from(SAMPLE_RATE);
        let transform = MelSpectrogram::new(MEL_SPEC)?;
        let mut logits = Logits::new(scratch()?)?;
        for entry in chunk_entries(n, CHUNK_SIZE, BORDER_SIZE) {
            check()?;
            let start = entry.chunk_start.max(0) as usize;
            let end = ((entry.chunk_start + CHUNK_SIZE as isize) as usize).min(n);
            let mel = transform.compute_range(source, start, end - start)?;
            let mut chunk = vec![vec![0.; N_MELS]; CHUNK_SIZE];
            let padding = (start as isize - entry.chunk_start) as usize;
            for (row, values) in chunk
                .iter_mut()
                .skip(padding)
                .zip(mel.values.chunks_exact(N_MELS))
            {
                row.copy_from_slice(values);
            }
            let mut beat = vec![0.; CHUNK_SIZE];
            let mut down = vec![0.; CHUNK_SIZE];
            for &model in self.active_models() {
                check()?;
                let (b, d) = self.predict_chunk(&chunk, model).await?;
                // Preserve ensemble summation order and float32 arithmetic.
                for i in 0..CHUNK_SIZE {
                    beat[i] += b[i];
                    down[i] += d[i];
                }
            }
            if self.ensemble_size > 1 {
                let inv = 1. / self.ensemble_size as f32;
                for value in beat.iter_mut().chain(&mut down) {
                    *value *= inv;
                }
            }
            let lo = (entry.write_start as isize - entry.chunk_start) as usize;
            let hi = (entry.write_end as isize - entry.chunk_start) as usize;
            if logits.len() != entry.write_start {
                return Err(EngineError::internal("Noncontiguous model windows"));
            }
            logits.append(&beat[lo..hi], &down[lo..hi])?;
            progress
                .report(
                    0.1 + 0.7 * entry.write_end as f64 / n as f64,
                    "Analyzing audio windows",
                )
                .await;
        }
        check()?;
        progress.report(0.85, "Tracking the global beat grid").await;
        let result = match self.post_processing {
            PostProcessing::Minimal => {
                postprocess::process_paged(&mut logits, FPS, Some(duration), max_events, &check)?
            }
            PostProcessing::Dbn => dbn::process_paged(
                &mut logits,
                scratch,
                FPS,
                Some(duration),
                max_events,
                &check,
            )?,
        };
        let result = self.finish(result);
        progress.report(0.95, "Measuring bar energy").await;
        let energy = energy_structure::energy_per_bar_source(
            source,
            f64::from(SAMPLE_RATE),
            &result.quantized_downbeats,
            duration,
            &check,
        )?;
        check()?;
        let result = result.with_energy(energy);
        progress.report(1., "done").await;
        Ok(result)
    }
}
