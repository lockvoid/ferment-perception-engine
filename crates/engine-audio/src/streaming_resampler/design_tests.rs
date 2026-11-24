//! Exact current iOS design laws; numerical excerpt gates are separate.
use super::*;

#[test]
#[ignore = "requires immutable captured Python filter taps"]
fn the_two_to_one_kernel_is_the_independent_python_filter() {
    let root = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .expect("explicit immutable fixture tree required");
    let bytes =
        std::fs::read(std::path::PathBuf::from(root).join("Resampler/kaiser257_taps.f32")).unwrap();
    assert_eq!(bytes.len(), 257 * 4);
    let python = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let resampler = StreamingResampler::new(44100, 22050).unwrap();
    let Kernel::Decimate { factor, taps } = resampler.kernel else {
        panic!("not an integer decimator")
    };
    assert_eq!(factor, 2);
    assert_eq!(taps.len(), 257);
    assert!((taps[128] - 0.475).abs() < 1e-6);
    assert!((taps.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(taps.iter().zip(python).all(|(a, b)| (a - b).abs() < 1e-6));
}

#[test]
fn three_to_one_follows_the_current_kaiser_design_rule() {
    let resampler = StreamingResampler::new(48000, 16000).unwrap();
    let Kernel::Decimate { factor, taps } = resampler.kernel else {
        panic!("not an integer decimator")
    };
    assert_eq!(factor, 3);
    assert_eq!(48000 / factor, 16000);
    assert_eq!(taps.len() % 2, 1);
    assert!(taps.len().abs_diff(385) <= 4);
    assert!((taps.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn rational_phases_keep_the_current_prototype_size_and_integer_delay() {
    for (input, output, expected_up, expected_down, count) in [
        (44100, 16000, 160, 441, 57_331),
        (48000, 22050, 147, 320, 41_601),
    ] {
        let resampler = StreamingResampler::new(input, output).unwrap();
        let Kernel::Polyphase {
            up,
            down,
            delay,
            phases,
            ..
        } = resampler.kernel
        else {
            panic!("not a rational converter")
        };
        assert_eq!((up, down), (expected_up, expected_down));
        assert_eq!(delay, 65);
        assert_eq!(phases.iter().map(Vec::len).sum::<usize>(), count);
        // Phases partition the complete prototype and carry its upsampling
        // gain. Undo that gain to grade the design's unit DC normalization.
        let prototype_sum = phases
            .iter()
            .flatten()
            .map(|v| f64::from(*v) / up as f64)
            .sum::<f64>();
        assert!((prototype_sum - 1.0).abs() < 1e-4);
    }
}
