//! Beat / downbeat / tempo analysis using the BeatThis ensemble (3 ONNX
//! models) plus DBN-style post-processing (peak picking + Viterbi meter
//! detection + linear-fit beat-grid quantization).
//!
//! Pipeline:
//! 1. Decode audio → resample 22.05 kHz mono
//! 2. Mel spectrogram (custom: reflect padding, magnitude, ÷√N, log1p·1000)
//! 3. Chunked inference (chunk=1500 frames, border=6) — averaged over ensemble
//! 4. Post-processing (`PostProcessing`): either the minimal peak-pick +
//!    Viterbi meter detection (3/4/5/6/7), or the madmom bar-pointer DBN
//!    (`dbn`), which enforces a continuous bar through material the
//!    peak-picker follows literally
//! 5. Warp-lattice quantization (`lattice` + `quantize`) → segments, snapped
//!    beats, BPM and stability metrics
//! 6. Energy column + loudness structure over the downbeat lattice

pub mod config;
pub mod dbn;
pub mod lattice;
pub mod mel;
pub mod meter;
pub mod peaks;
pub mod postprocess;
pub mod quantize;
pub mod transient;

mod processor;
mod storage;

pub use processor::{BeatAnalysisResult, BeatThis, PostProcessing, chunk_plan};

#[cfg(test)]
mod pipeline_tests {
    use ferment_engine_test_fixtures as fixtures;

    use super::{
        mel::compute_log_mel,
        meter::detect_meter,
        peaks::{deduplicate_peaks, find_peaks},
        postprocess::process,
    };

    const FPS: f64 = 50.0;

    #[test]
    fn test_stage_1_beat_this_mel_spectrogram() {
        let Some(audio) = fixtures::f32_le("pipeline_test_audio.bin") else {
            return;
        };
        let reference = fixtures::require_f32_le("pipeline_ref_mel.bin");
        let reference_frames = reference.len() / 128;
        let actual = compute_log_mel(&audio).expect("BeatThis mel");
        assert!(actual.len().abs_diff(reference_frames) <= 1);

        if actual.len() > 10 && reference_frames > 10 {
            let actual_sum = actual[10].iter().sum::<f32>();
            let reference_sum = reference[10 * 128..11 * 128].iter().sum::<f32>();
            assert!((actual_sum - reference_sum).abs() <= 0.1);
        }
        let max_difference = actual
            .iter()
            .flatten()
            .zip(reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_difference < 0.01, "max difference={max_difference}");
    }

    #[test]
    fn test_stage_2_peak_picking() {
        let logits = [
            -5.0, -3.0, -1.0, 3.09, 2.5, -2.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
            -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
            -5.0, 2.0, 1.5, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
        ];
        let peaks = find_peaks(&logits, 7);
        let frames = deduplicate_peaks(&peaks);
        let times = frames.iter().map(|frame| frame / FPS).collect::<Vec<_>>();
        assert!(peaks.contains(&3));
        assert!((times[0] - 0.06).abs() < 0.001);
    }

    #[test]
    fn test_stage_3_meter_detection() {
        let scores = [
            2.95, 0.44, -1.38, -1.02, 4.30, -0.82, -1.92, -2.66, 0.37, -4.52,
        ];
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 4);
        assert_eq!(result.phase, 0);
    }

    #[test]
    fn test_stage_3_meter_detection_with_real_scores() {
        let scores = [
            2.954_356_7,
            0.443_672_54,
            -1.379_210_5,
            -1.015_685_4,
            4.298_330_3,
            -0.824_31,
            -1.923_1,
            -2.658_9,
            0.373_5,
            -4.520_6,
        ];
        let result = detect_meter(&scores).expect("meter");
        assert_eq!(result.meter, 4);
        assert_eq!(result.phase, 0);
    }

    #[test]
    fn test_stage_4_full_pipeline() {
        let Some(audio) = fixtures::f32_le("pipeline_test_audio.bin") else {
            return;
        };
        let spectrogram = compute_log_mel(&audio).expect("BeatThis mel");
        assert!(!spectrogram.is_empty());
        let mut beat_logits = vec![-5.0; spectrogram.len()];
        let mut downbeat_logits = vec![-5.0; spectrogram.len()];
        let positions = [3, 30, 55, 77, 102, 127, 152, 177, 202, 227];
        for (index, &position) in positions.iter().enumerate() {
            if position < spectrogram.len() {
                beat_logits[position] = 5.0;
                downbeat_logits[position] = if index.is_multiple_of(4) { 10.0 } else { -10.0 };
            }
        }
        let result = process(&beat_logits, &downbeat_logits, FPS, None);
        assert_eq!(result.beats.len(), positions.len());
        assert!(!result.downbeats.is_empty());
        assert!(result.downbeats.iter().all(|downbeat| {
            result
                .beats
                .iter()
                .any(|beat| (beat - downbeat).abs() < 0.001)
        }));
        if result.downbeats.len() >= 2 {
            let ratio =
                (result.downbeats[1] - result.downbeats[0]) / (result.beats[1] - result.beats[0]);
            assert!((ratio - 4.0).abs() <= 0.5, "ratio={ratio}");
        }
    }

    #[test]
    fn test_beat_this_mel_spectrogram_python_reference_match() {
        let Some(audio) = fixtures::f32_le("test_audio.bin") else {
            return;
        };
        let reference = fixtures::require_f32_le("test_mel_reference.bin");
        let reference_frames = reference.len() / 128;
        let actual = compute_log_mel(&audio).expect("BeatThis mel");
        assert_eq!(actual.len(), reference_frames);
        let max_difference = actual
            .iter()
            .flatten()
            .zip(reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_difference < 0.001, "max difference={max_difference}");
    }
}
