//! The current iOS FIR/polyphase design, graded against the captured Python
//! gate files, never goldens recomputed from this implementation.
use ferment_engine_audio::streaming_resampler::{StreamingResampler, resample_current};

fn stream(input: &[f32], from: u32, to: u32, blocks: &[usize]) -> Vec<f32> {
    let mut r = StreamingResampler::new(from, to).unwrap();
    let (mut result, mut offset, mut n) = (Vec::new(), 0, 0);
    while offset < input.len() {
        let end = (offset + blocks[n % blocks.len()]).min(input.len());
        result.extend(r.process(&input[offset..end]).unwrap());
        assert!(
            r.retained_samples() < 1000,
            "history is bounded after each block"
        );
        offset = end;
        n += 1;
    }
    result.extend(r.finish().unwrap());
    result
}
fn fixture(name: &str) -> Vec<f32> {
    assert!(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").is_some(),
        "explicit immutable fixture root required"
    );
    ferment_engine_test_fixtures::require_f32_le(&format!("Resampler/{name}.f32"))
}
fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn tone(hz: f64, rate: u32) -> Vec<f32> {
    (0..rate * 2)
        .map(|i| (std::f64::consts::TAU * hz * f64::from(i) / f64::from(rate)).sin() as f32)
        .collect()
}
fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

#[test]
fn identity_empty_tiny_and_ceil_counts() {
    for (from, to) in [
        (44100, 22050),
        (44100, 16000),
        (48000, 22050),
        (22050, 44100),
        (48000, 16000),
    ] {
        for n in [0, 1, 5, 441, 1001] {
            let y = stream(&vec![0.25; n], from, to, &[2, 1, 7]);
            assert_eq!(y.len(), (n * to as usize).div_ceil(from as usize));
        }
    }
    assert_eq!(stream(&[1., 2., 3.], 44100, 44100, &[1]), [1., 2., 3.]);
    assert!(StreamingResampler::new(44100, 16001).is_none());
    assert!(StreamingResampler::new(0, 16000).is_none());
}
#[test]
fn input_after_finish_and_second_finish_are_errors() {
    let mut r = StreamingResampler::new(44100, 16000).unwrap();
    r.finish().unwrap();
    assert!(r.process(&[1.]).is_err());
    assert!(r.finish().is_err());
}
#[test]
fn passband_levels_and_stopband_rejection_match_current_design() {
    for (to, pass, stop, range) in [
        (22050, 5000., 11500., 2000..42000),
        (16000, 3000., 9000., 2000..30000),
    ] {
        let y = resample_current(&tone(pass, 44100), 44100, to).unwrap();
        let want = tone(pass, to);
        let db = 20. * (rms(&y[range.clone()]) / rms(&want[range.clone()])).log10();
        assert!(db.abs() < 0.1, "passband {to} {db}dB");
        let y = resample_current(&tone(stop, 44100), 44100, to).unwrap();
        let db = 20. * (rms(&y[range]) / std::f64::consts::FRAC_1_SQRT_2).log10();
        assert!(db < -90., "stopband {to} {db}dB");
    }
}
#[test]
fn impulses_keep_their_absolute_time() {
    for (from, to, input, expected) in [
        (44100, 22050, 30000, 15000),
        (44100, 16000, 44100, 16000),
        (48000, 16000, 48000, 16000),
    ] {
        let mut x = vec![0.; from as usize * 2];
        x[input] = 1.;
        let y = resample_current(&x, from, to).unwrap();
        let peak = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, expected);
    }
}
#[test]
#[ignore = "requires pinned iOS Resampler golden files"]
fn fir_matches_independent_python_excerpt_and_filter() {
    let input = fixture("kaiser257_44k_in");
    let expected = fixture("kaiser257_22k_out");
    assert_eq!(input.len(), 220500);
    let out = resample_current(&input, 44100, 22050).unwrap();
    assert!(worst(&out, &expected) < 5e-5);
    let taps = fixture("kaiser257_taps");
    assert_eq!(taps.len(), 257);
    let mut x = vec![0.; 600];
    x[256] = 1.;
    let actual = resample_current(&x, 44100, 22050).unwrap();
    for i in 0..129 {
        assert!((actual[64 + i] - taps[i * 2]).abs() < 1e-6);
    }
}
#[test]
#[ignore = "requires pinned iOS Resampler golden files"]
fn polyphase_matches_both_independent_python_excerpts() {
    for (name, from, to, count) in [
        ("poly441_160", 44100, 16000, 48000),
        ("poly320_147", 48000, 22050, 66150),
    ] {
        let out = resample_current(&fixture(&format!("{name}_in")), from, to).unwrap();
        assert_eq!(out.len(), count);
        assert!(worst(&out, &fixture(&format!("{name}_out"))) < 5e-5);
    }
}
#[test]
#[ignore = "requires pinned iOS Resampler golden files"]
fn every_block_schedule_is_bit_identical_to_the_flat_door() {
    for (name, from, to) in [
        ("kaiser257_44k_in", 44100, 22050),
        ("poly441_160_in", 44100, 16000),
        ("poly320_147_in", 48000, 22050),
    ] {
        let input = fixture(name);
        let flat = resample_current(&input, from, to).unwrap();
        for schedule in [
            vec![16384],
            vec![1],
            vec![3, 7, 1000, 2, 16384, 5],
            vec![1 << 20],
        ] {
            assert_eq!(flat, stream(&input, from, to, &schedule));
        }
    }
}
