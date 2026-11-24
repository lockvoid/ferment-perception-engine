//! The current iOS PipelineTraceTests' original 102-frame libdf ladder.
//! Each stage consumes the independently frozen input for that stage. A
//! correct final waveform cannot conceal compensating intermediate errors.

use super::config::{FFT_SIZE, HOP_SIZE, N_FREQ, NB_DF, NB_ERB};
use super::enhance::{apply_deep_filtering, apply_erb_mask, merge};
use super::features::{erb_features, lookahead_pad, spec_features};
use super::processor::{Analysis, Synthesis};
use ferment_engine_test_fixtures as fixtures;

const FRAMES: usize = 102;

fn trace(name: &str) -> Vec<f32> {
    // Missing input is always a failure, including an unconfigured fixture
    // tree. These tests may not report success by returning early.
    fixtures::require_f32_le(&format!("SpeechDenoising/trace_{name}.bin"))
}

fn close(label: &str, actual: &[f32], expected: &[f32], bound: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: full output length");
    assert!(
        actual.iter().all(|sample| sample.is_finite()),
        "{label}: finite output"
    );
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "[dfnet-current-trace] {label}: {} values, max |delta| {worst}",
        actual.len()
    );
    assert!(worst < bound, "{label}: {worst} >= {bound}");
}

#[test]
fn stage1_stft_matches_libdf() {
    let audio = trace("audio_padded");
    assert_eq!(audio.len(), 48_960);
    let spectrum = Analysis::new().unwrap().next(&audio, FRAMES).unwrap();
    assert_eq!(spectrum.len(), FRAMES * N_FREQ * 2);
    close("STFT", &spectrum, &trace("stft"), 1e-5);
}

#[test]
fn stage2_erb_features_match_libdf() {
    close(
        "ERB",
        &erb_features(&trace("stft"), FRAMES),
        &trace("erb_feat"),
        1e-4,
    );
}

#[test]
fn stage3_spec_features_match_libdf() {
    close(
        "spec features",
        &spec_features(&trace("stft"), FRAMES),
        &trace("spec_feat"),
        1e-3,
    );
}

#[test]
fn stage4_erb_lookahead_matches_libdf() {
    close(
        "ERB lookahead",
        &lookahead_pad(&trace("erb_feat"), FRAMES, NB_ERB),
        &trace("erb_feat_padded"),
        1e-5,
    );
}

#[test]
fn stage4_spec_lookahead_matches_libdf() {
    close(
        "spec lookahead",
        &lookahead_pad(&trace("spec_feat"), FRAMES, NB_DF),
        &trace("spec_feat_padded"),
        1e-5,
    );
}

#[test]
fn stage7_erb_mask_matches_libdf() {
    close(
        "ERB mask",
        &apply_erb_mask(&trace("stft"), &trace("erb_mask"), FRAMES),
        &trace("spec_masked"),
        1e-5,
    );
}

#[test]
fn stage8_deep_filtering_matches_libdf_low_bins() {
    let actual = apply_deep_filtering(&trace("stft"), &trace("df_coefs"), FRAMES);
    let expected = trace("spec_filtered");
    assert_eq!(actual.len(), FRAMES * N_FREQ * 2);
    assert_eq!(expected.len(), actual.len());
    let low_bins = |s: &[f32]| {
        s.chunks_exact(N_FREQ * 2)
            .flat_map(|frame| frame[..NB_DF * 2].iter().copied())
            .collect::<Vec<_>>()
    };
    close(
        "deep filter low bins",
        &low_bins(&actual),
        &low_bins(&expected),
        1e-4,
    );
}

#[test]
fn stage8_merge_matches_libdf() {
    close(
        "merge",
        &merge(&trace("spec_filtered"), &trace("spec_masked"), FRAMES),
        &trace("spec_enhanced"),
        1e-5,
    );
}

#[test]
fn stage9_istft_matches_libdf() {
    let mut output = Vec::new();
    Synthesis::new()
        .unwrap()
        .push(&trace("spec_enhanced"), &mut output)
        .unwrap();
    assert_eq!(output.len(), FRAMES * HOP_SIZE);
    let delay = FFT_SIZE - HOP_SIZE;
    close(
        "ISTFT",
        &output[delay..delay + 48_000],
        &trace("audio_out"),
        1e-4,
    );
}

#[cfg(all(feature = "ort", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires pinned ORT and the published dfnet model; run explicitly in the current audio gate"]
fn stage5_6_real_model_outputs_match_libdf() {
    use ferment_engine_infer::{
        ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };
    use ferment_engine_test_fixtures::export;
    use std::sync::Arc;
    let artifact =
        export::artifact("dfnet.onnx").expect("the pinned dfnet ONNX artifact is required");
    let paths = ModelPaths::new().with_path(ModelId::Dfnet, ModelVersion::Latest, artifact);
    let backend = Arc::new(OrtBackend::with_execution_provider(
        Arc::new(paths),
        OrtExecutionProvider::Cpu,
    ));
    let answer = futures_executor::block_on(super::DeepFilterNet::new(backend).run_chunk(
        &trace("erb_feat_padded"),
        &trace("spec_feat_padded"),
        FRAMES,
    ))
    .unwrap();
    // Preserve the current source's asserted bounds. The stronger fp32
    // 1500-frame checkpoint gate remains separate in dfnet_contract.
    close("model ERB mask", &answer.erb_mask, &trace("erb_mask"), 0.5);
    close(
        "model coefficients",
        &answer.df_coefs,
        &trace("df_coefs"),
        0.1,
    );
    assert_eq!(answer.lsnr.len(), FRAMES);
    assert!(answer.lsnr.iter().all(|v| v.is_finite()));
    // The original frozen trace has no trace_lsnr.bin; its iOS assertion is
    // conditional. dfnet_contract always grades lsnr on the full golden chunk.
}
