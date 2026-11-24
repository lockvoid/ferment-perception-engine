//! Top-level orchestrator, iOS's `DeepFilterNet.denoise` chunk for chunk:
//! libdf's analysis streamed over the input, every 1500-frame chunk's
//! features, door calls and enhancement computed from that chunk alone, and
//! the inverse STFT streamed back out — nothing held but the chunk in flight.
//! The synthesis's first `FFT_SIZE - HOP_SIZE` samples are its warm-up and are
//! trimmed. Given keep segments, a chunk whose output overlaps none of them is
//! not enhanced at all (`canSkipChunk`).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ferment_engine_audio::{
    self as audio, AudioFrames, StftConfig, StreamingIstft, StreamingStft, resample_sinc,
};
use ferment_engine_core::{AudioEventSegment, EngineError, ProgressSink, Result};
use ferment_engine_infer::{Backend, ModelId, ModelVersion, Tensor, TensorMap};
use rustfft::num_complex::Complex;

use super::config::{
    CHUNK_FRAMES, DF_LOOKAHEAD, DF_ORDER, FFT_SIZE, HOP_SIZE, N_FREQ, NB_DF, NB_ERB, SAMPLE_RATE,
    STREAM_FRAMES, STREAM_STATE,
};
use super::enhance::{apply_deep_filtering, apply_erb_mask, merge};
use super::events_mask::apply_events_mask_window;
use super::features::chunk_features;
use crate::audio_source::{
    ArrayAudioSampleSource, AudioSampleSource, AudioSource, DiskBackedAudioSampleSource,
};

#[derive(Debug, Clone)]
pub struct DenoiseOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// The door's answer for one chunk, in the flat layouts the DSP indexes.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkOutputs {
    /// `[T, 32]`.
    pub erb_mask: Vec<f32>,
    /// `[5, T, 96, 2]` — plane `k` is tap `k` of the deep filter.
    pub df_coefs: Vec<f32>,
    /// `[T]`.
    pub lsnr: Vec<f32>,
}

/// libdf's spectrum scale, `2·hop / fft²`: its analysis multiplies by it and
/// its synthesis divides it back out.
const WNORM: f32 = (2 * HOP_SIZE) as f32 / (FFT_SIZE * FFT_SIZE) as f32;

#[derive(Debug, Clone)]
pub struct DeepFilterNet {
    backend: Backend,
    version: ModelVersion,
    max_frames_per_chunk: usize,
}

impl DeepFilterNet {
    pub fn new(backend: Backend) -> Self {
        Self::with_version(backend, ModelVersion::default())
    }

    pub fn with_version(backend: Backend, version: ModelVersion) -> Self {
        Self {
            backend,
            version,
            max_frames_per_chunk: CHUNK_FRAMES,
        }
    }

    /// Frames per chunk — where the features and the door start over. Tests
    /// pass smaller values to put the seams where they can see them.
    #[must_use]
    pub fn with_max_frames_per_chunk(mut self, frames: usize) -> Self {
        self.max_frames_per_chunk = frames.clamp(DF_LOOKAHEAD + 1, CHUNK_FRAMES);
        self
    }

    /// Denoise and apply the AED keep mask as one orchestration operation. A
    /// chunk whose output overlaps no keep segment is skipped as iOS skips it:
    /// silence runs through the analysis and the synthesis in its place — so
    /// the chunk after it starts from silenced memories — and its samples are
    /// zeros, which the mask would have made them anyway.
    pub async fn denoise_with_keep_segments(
        &self,
        frames: AudioFrames,
        keep_segments: &[AudioEventSegment],
        progress: ProgressSink,
    ) -> Result<DenoiseOutput> {
        self.run(frames, Some(keep_segments), progress).await
    }

    /// Run full denoise pipeline on `frames`. Returns mono samples at 48 kHz.
    pub async fn denoise(
        &self,
        frames: AudioFrames,
        progress: ProgressSink,
    ) -> Result<DenoiseOutput> {
        self.run(frames, None, progress).await
    }

    #[tracing::instrument(skip_all, fields(processor = "deepfilternet", samples = frames.len(), sr = frames.sample_rate))]
    async fn run(
        &self,
        frames: AudioFrames,
        keep_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<DenoiseOutput> {
        check_cancellation(&progress)?;
        progress.report(0.02, "preparing audio").await;
        let samples = if frames.sample_rate == SAMPLE_RATE {
            frames.samples
        } else {
            resample_sinc(&frames.samples, frames.sample_rate, SAMPLE_RATE)?
        };
        let source = ArrayAudioSampleSource::new(samples);
        let mut output = Vec::with_capacity(source.sample_count());
        self.run_windowed(&source, keep_segments, progress, |block| {
            output.extend_from_slice(block);
            Ok(())
        })
        .await?;
        Ok(DenoiseOutput {
            samples: output,
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Production file path. Only the decoded source and returned raw-f32
    /// artifact grow with duration; neither crosses FFI as a sample array.
    /// Both are owned temporaries under the caller's writable directory.
    pub async fn denoise_to_file(
        &self,
        audio_path: &Path,
        temporary_directory: &Path,
        keep_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
    ) -> Result<DiskBackedAudioSampleSource> {
        check_cancellation(&progress)?;
        progress.report(0.02, "preparing audio").await;
        check_cancellation(&progress)?;
        let source = AudioSource::default()
            .decode_disk_backed_in_with_cancellation(
                audio_path,
                SAMPLE_RATE,
                temporary_directory,
                || check_cancellation(&progress),
            )
            .await?;
        check_cancellation(&progress)?;
        let mut output = OwnedPcmSink::new(temporary_directory)?;
        self.run_windowed_with_sync(&source, keep_segments, progress.clone(), |block, sync| {
            output.append(block)?;
            if sync {
                output.synchronize()?;
            }
            Ok(())
        })
        .await?;
        check_cancellation(&progress)?;
        output.finish()
    }

    /// Common chunk pipeline for the existing array API and the file API.
    /// Input reads, spectrum, model outputs, synthesis and mask are bounded
    /// by CHUNK_FRAMES. Sample coordinates remain absolute across chunks.
    pub(super) async fn run_windowed(
        &self,
        source: &dyn AudioSampleSource,
        keep_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
        mut sink: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        self.run_windowed_with_sync(source, keep_segments, progress, |block, _| sink(block))
            .await
    }

    async fn run_windowed_with_sync(
        &self,
        source: &dyn AudioSampleSource,
        keep_segments: Option<&[AudioEventSegment]>,
        progress: ProgressSink,
        mut sink: impl FnMut(&[f32], bool) -> Result<()>,
    ) -> Result<()> {
        check_cancellation(&progress)?;
        let total = source.sample_count();
        if total < HOP_SIZE {
            return Err(EngineError::AudioTooShort {
                samples: total,
                min_required: HOP_SIZE,
            });
        }
        let frame_count = total
            .checked_add(FFT_SIZE)
            .ok_or_else(|| EngineError::invalid_config("audio sample count overflow"))?
            / HOP_SIZE;
        let chunk_frames = self.max_frames_per_chunk;
        let chunks = frame_count.div_ceil(chunk_frames);
        let keep = keep_segments.map(|segments| {
            segments
                .iter()
                .map(|segment| sample_at(segment.start_time)..sample_at(segment.end_time()))
                .collect::<Vec<_>>()
        });
        let mut analysis = Analysis::new()?;
        let mut synthesis = Synthesis::new()?;
        let delay = FFT_SIZE - HOP_SIZE;
        for chunk in 0..chunks {
            check_cancellation(&progress)?;
            let first = chunk * chunk_frames;
            let frames = (frame_count - first).min(chunk_frames);
            let kept = output_range(first, frames, total);
            let skipped = keep.as_ref().is_some_and(|ranges| {
                !kept.is_empty()
                    && !ranges
                        .iter()
                        .any(|range| range.start < kept.end && range.end > kept.start)
            });
            let mut output = Vec::with_capacity(frames * HOP_SIZE);
            if skipped {
                synthesis.silence(&analysis.silence(frames)?, &mut output)?;
            } else {
                let spectrum = analysis.next_source(source, frames)?;
                let enhanced = self.enhance(&spectrum, frames, &progress).await?;
                synthesis.push(&enhanced, &mut output)?;
            }
            check_cancellation(&progress)?;
            let raw_start = first * HOP_SIZE;
            let from = delay.saturating_sub(raw_start).min(output.len());
            let to = (delay + total).saturating_sub(raw_start).min(output.len());
            let block = &mut output[from..to];
            if let Some(segments) = keep_segments {
                apply_events_mask_window(block, kept.start, total, segments, SAMPLE_RATE);
            }
            // Swift synchronizes enhanced output at zero-based chunk indices
            // 5, 10, ...; fully masked chunks bypass that synchronization.
            // A real write/flush/sync failure remains the processing failure.
            sink(block, !skipped && chunk > 0 && chunk % 5 == 0)?;
            progress
                .report(
                    0.05 + 0.94 * (chunk + 1) as f64 / chunks as f64,
                    format!("removing noise ({}/{chunks})", chunk + 1),
                )
                .await;
        }
        check_cancellation(&progress)?;
        progress.report(1.0, "done").await;
        check_cancellation(&progress)?;
        Ok(())
    }

    /// One chunk as iOS's `processSpec`: the chunk's own features, the door,
    /// then the mask, the deep filter and the merge over the chunk's frames.
    /// lsnr is not read: nothing after the merge gates on it.
    async fn enhance(
        &self,
        spectrum: &[f32],
        frames: usize,
        progress: &ProgressSink,
    ) -> Result<Vec<f32>> {
        let (feat_erb, feat_spec) = chunk_features(spectrum, frames);
        let answer = self
            .run_chunk_with_progress(&feat_erb, &feat_spec, frames, Some(progress))
            .await?;
        let masked = apply_erb_mask(spectrum, &answer.erb_mask, frames);
        let filtered = apply_deep_filtering(spectrum, &answer.df_coefs, frames);
        Ok(merge(&filtered, &masked, frames))
    }

    /// One chunk of `frames` lookahead-shifted features through the streaming
    /// door (`ferment-models-export`, `models/dfnet/model.json`): `feat_erb`
    /// `[T, 32]` and `feat_spec` `[2, T, 96]`, each channel `T * 96` long.
    ///
    /// `ceil(frames / 100)` calls. The state starts from zeros and every
    /// `<state>_out` is the next call's `<state>`; the last call is
    /// zero-padded to the door's width and trimmed on the read side — the
    /// graph is causal, so the padding cannot reach a valid frame.
    pub async fn run_chunk(
        &self,
        feat_erb: &[f32],
        feat_spec: &[f32],
        frames: usize,
    ) -> Result<ChunkOutputs> {
        self.run_chunk_with_progress(feat_erb, feat_spec, frames, None)
            .await
    }

    async fn run_chunk_with_progress(
        &self,
        feat_erb: &[f32],
        feat_spec: &[f32],
        frames: usize,
        progress: Option<&ProgressSink>,
    ) -> Result<ChunkOutputs> {
        if frames == 0
            || frames > CHUNK_FRAMES
            || feat_erb.len() != frames * NB_ERB
            || feat_spec.len() != 2 * frames * NB_DF
        {
            return Err(EngineError::invalid_config(format!(
                "a DeepFilterNet chunk is 1..={CHUNK_FRAMES} frames of [T, {NB_ERB}] and \
                 [2, T, {NB_DF}] features; got {frames} frames with {} and {} values",
                feat_erb.len(),
                feat_spec.len()
            )));
        }
        let row = NB_DF * 2;
        let mut answer = ChunkOutputs {
            erb_mask: Vec::with_capacity(frames * NB_ERB),
            df_coefs: vec![0.0; DF_ORDER * frames * row],
            lsnr: Vec::with_capacity(frames),
        };
        let mut state = STREAM_STATE
            .iter()
            .map(|tensor| {
                Tensor::from_f32(tensor.shape.to_vec(), vec![0.0; elements(tensor.shape)])
            })
            .collect::<Result<Vec<_>>>()?;

        for start in (0..frames).step_by(STREAM_FRAMES) {
            if let Some(progress) = progress {
                check_cancellation(progress)?;
            }
            let valid = (frames - start).min(STREAM_FRAMES);
            let mut erb = vec![0.0; STREAM_FRAMES * NB_ERB];
            erb[..valid * NB_ERB]
                .copy_from_slice(&feat_erb[start * NB_ERB..(start + valid) * NB_ERB]);
            let mut spec = vec![0.0; 2 * STREAM_FRAMES * NB_DF];
            for channel in 0..2 {
                let from = channel * frames * NB_DF + start * NB_DF;
                let to = channel * STREAM_FRAMES * NB_DF;
                spec[to..to + valid * NB_DF]
                    .copy_from_slice(&feat_spec[from..from + valid * NB_DF]);
            }

            let mut inputs = TensorMap::new();
            inputs.insert(
                "feat_erb".into(),
                Tensor::from_f32(vec![1, 1, STREAM_FRAMES as i64, NB_ERB as i64], erb)?,
            );
            inputs.insert(
                "feat_spec".into(),
                Tensor::from_f32(vec![1, 2, STREAM_FRAMES as i64, NB_DF as i64], spec)?,
            );
            for (tensor, value) in STREAM_STATE.iter().zip(state) {
                inputs.insert(tensor.input.into(), value);
            }
            let mut outputs = self
                .backend
                .infer(ModelId::Dfnet, &self.version, inputs)
                .await?;

            let mask = door_output(&mut outputs, "erb_mask", STREAM_FRAMES * NB_ERB)?;
            answer.erb_mask.extend_from_slice(&mask[..valid * NB_ERB]);
            let coefs = door_output(&mut outputs, "df_coefs", DF_ORDER * STREAM_FRAMES * row)?;
            for plane in 0..DF_ORDER {
                let to = plane * frames * row + start * row;
                let from = plane * STREAM_FRAMES * row;
                answer.df_coefs[to..to + valid * row]
                    .copy_from_slice(&coefs[from..from + valid * row]);
            }
            let lsnr = door_output(&mut outputs, "lsnr", STREAM_FRAMES)?;
            answer.lsnr.extend_from_slice(&lsnr[..valid]);
            state = STREAM_STATE
                .iter()
                .map(|tensor| {
                    let values = door_output(&mut outputs, tensor.output, elements(tensor.shape))?;
                    Tensor::from_f32(tensor.shape.to_vec(), values)
                })
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(answer)
    }
}

fn elements(shape: &[i64]) -> usize {
    shape.iter().product::<i64>() as usize
}

/// One output of the door, `values` long in its row-major order.
fn door_output(outputs: &mut TensorMap, name: &str, values: usize) -> Result<Vec<f32>> {
    let tensor = outputs
        .remove(name)
        .ok_or_else(|| EngineError::ModelOutputMissing {
            model: ModelId::Dfnet.name().into(),
            key: name.into(),
        })?
        .to_f32()?;
    if tensor.len() == values {
        Ok(tensor)
    } else {
        Err(EngineError::InvalidOutput)
    }
}

fn stft_config() -> Result<StftConfig> {
    StftConfig::new(FFT_SIZE, HOP_SIZE, audio::window::vorbis(FFT_SIZE))
}

/// A keep segment's edge in samples, as iOS truncates it.
fn sample_at(seconds: f64) -> usize {
    (seconds * f64::from(SAMPLE_RATE)).max(0.0) as usize
}

/// Where the chunk of `frames` frames from `first` lands in the input's own
/// timeline: the synthesis runs `FFT_SIZE - HOP_SIZE` samples behind it, and
/// nothing past the input's end is kept.
fn output_range(first: usize, frames: usize, total: usize) -> Range<usize> {
    let delay = FFT_SIZE - HOP_SIZE;
    let start = (first * HOP_SIZE).saturating_sub(delay).min(total);
    let end = ((first + frames) * HOP_SIZE)
        .saturating_sub(delay)
        .min(total);
    start..end
}

/// libdf's analysis, a hop at a time: the Vorbis window over the last
/// `FFT_SIZE` samples, scaled by `WNORM`. Its memory runs on from chunk to
/// chunk — only the features and the door start over.
pub(super) struct Analysis {
    stft: StreamingStft,
    frame: usize,
}

impl Analysis {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            stft: StreamingStft::new(stft_config()?)?,
            frame: 0,
        })
    }

    /// The next `frames` frames of `samples`, zero past its end, as `[T, 481,
    /// 2]`.
    #[cfg(test)]
    pub(super) fn next(&mut self, samples: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.analyse(Some(samples), frames)
    }

    /// Read only this chunk; a short copy from a declared source is an error,
    /// while the expected final FFT tail is explicitly zero-padded.
    fn next_source(&mut self, source: &dyn AudioSampleSource, frames: usize) -> Result<Vec<f32>> {
        let start = self.frame * HOP_SIZE;
        let mut samples = vec![0.0; frames * HOP_SIZE];
        let expected = source
            .sample_count()
            .saturating_sub(start)
            .min(samples.len());
        let copied = source.copy_samples(&mut samples, start, expected)?;
        if copied != expected {
            return Err(EngineError::decode(format!(
                "audio source ended early: {copied} of {expected} samples"
            )));
        }
        self.analyse_window(Some(&samples), frames, start)
    }

    /// The next `frames` frames with silence in place of the input — what a
    /// skipped chunk leaves in the memory (iOS's `stftFwd.process(zeroAudio)`).
    pub(super) fn silence(&mut self, frames: usize) -> Result<Vec<f32>> {
        self.analyse(None, frames)
    }

    fn analyse(&mut self, samples: Option<&[f32]>, frames: usize) -> Result<Vec<f32>> {
        self.analyse_window(samples, frames, 0)
    }

    fn analyse_window(
        &mut self,
        samples: Option<&[f32]>,
        frames: usize,
        base: usize,
    ) -> Result<Vec<f32>> {
        let mut spectrum = Vec::with_capacity(frames * N_FREQ * 2);
        for _ in 0..frames {
            let start = self.frame * HOP_SIZE - base;
            let hop = samples.map_or(&[][..], |samples| {
                &samples[start.min(samples.len())..(start + HOP_SIZE).min(samples.len())]
            });
            for bin in self.stft.frame(hop)? {
                spectrum.push(bin.re * WNORM);
                spectrum.push(bin.im * WNORM);
            }
            self.frame += 1;
        }
        Ok(spectrum)
    }
}

/// libdf's synthesis, a frame at a time: `WNORM` divided back out, the
/// inverse FFT windowed and overlap-added onto the previous frame's tail.
pub(super) struct Synthesis {
    istft: StreamingIstft,
    column: Vec<Complex<f32>>,
}

impl Synthesis {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            istft: StreamingIstft::new(stft_config()?)?,
            column: vec![Complex::default(); N_FREQ],
        })
    }

    /// `[T, 481, 2]` in, `T · HOP_SIZE` samples onto `output`.
    pub(super) fn push(&mut self, spectrum: &[f32], output: &mut Vec<f32>) -> Result<()> {
        for frame in spectrum.chunks_exact(N_FREQ * 2) {
            output.extend(self.frame(frame)?);
        }
        Ok(())
    }

    /// A skipped chunk: `spectrum`, the analysis of the silence in its place,
    /// runs through to move the overlap memory on as iOS's does, and the
    /// chunk's `T · HOP_SIZE` samples onto `output` are zeros.
    pub(super) fn silence(&mut self, spectrum: &[f32], output: &mut Vec<f32>) -> Result<()> {
        for frame in spectrum.chunks_exact(N_FREQ * 2) {
            self.frame(frame)?;
            output.resize(output.len() + HOP_SIZE, 0.0);
        }
        Ok(())
    }

    fn frame(&mut self, bins: &[f32]) -> Result<Vec<f32>> {
        let unscale = 1.0 / WNORM;
        for (bin, pair) in self.column.iter_mut().zip(bins.chunks_exact(2)) {
            *bin = Complex::new(pair[0] * unscale, pair[1] * unscale);
        }
        self.istft.frame(&self.column)
    }
}

fn check_cancellation(progress: &ProgressSink) -> Result<()> {
    if progress.is_closed() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

/// Partial output stays owned until a readable source takes it. Dropping a
/// failed or cancelled future closes the writer and removes only this file.
struct OwnedPcmSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    transferred: bool,
}

impl OwnedPcmSink {
    fn new(directory: &Path) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for _ in 0..8 {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "ferment-denoised-{}-{time}-{id}.f32",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        writer: Some(BufWriter::new(file)),
                        transferred: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(EngineError::invalid_config(
            "could not allocate a unique denoised output",
        ))
    }

    fn append(&mut self, samples: &[f32]) -> Result<()> {
        let writer = self.writer.as_mut().expect("live output writer");
        for sample in samples {
            writer.write_all(&sample.to_le_bytes())?;
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        let writer = self.writer.as_mut().expect("live output writer");
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(())
    }

    fn finish(mut self) -> Result<DiskBackedAudioSampleSource> {
        self.synchronize()?;
        let writer = self.writer.take().expect("live output writer");
        drop(writer);
        let source = DiskBackedAudioSampleSource::from_owned_file(self.path.clone())?;
        self.transferred = true;
        Ok(source)
    }
}

impl Drop for OwnedPcmSink {
    fn drop(&mut self) {
        drop(self.writer.take());
        if !self.transferred {
            crate::optional_failure::cleanup(
                "remove a partial sample file",
                crate::optional_failure::remove_if_present(&self.path),
            );
        }
    }
}
