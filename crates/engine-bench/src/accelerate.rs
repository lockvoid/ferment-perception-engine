//! Exact Accelerate split/interleaved conversion tax for the tier-4 gate.

use rustfft::num_complex::Complex;

#[repr(C)]
struct DspComplex {
    real: f32,
    imaginary: f32,
}

#[repr(C)]
struct DspSplitComplex {
    real: *mut f32,
    imaginary: *mut f32,
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_ctoz(
        input: *const DspComplex,
        input_stride: isize,
        output: *mut DspSplitComplex,
        output_stride: isize,
        length: usize,
    );
    fn vDSP_ztoc(
        input: *const DspSplitComplex,
        input_stride: isize,
        output: *mut DspComplex,
        output_stride: isize,
        length: usize,
    );
}

pub struct SplitComplexRoundtrip {
    real: Vec<f32>,
    imaginary: Vec<f32>,
    interleaved: Vec<Complex<f32>>,
}

impl SplitComplexRoundtrip {
    #[must_use]
    pub fn new(complex_count: usize) -> Self {
        assert_eq!(
            size_of::<Complex<f32>>(),
            size_of::<DspComplex>(),
            "num_complex and Accelerate complex layouts must agree"
        );
        assert_eq!(
            align_of::<Complex<f32>>(),
            align_of::<DspComplex>(),
            "num_complex and Accelerate complex alignment must agree"
        );
        Self {
            real: vec![0.0; complex_count],
            imaginary: vec![0.0; complex_count],
            interleaved: vec![Complex::new(0.0, 0.0); complex_count],
        }
    }

    pub fn run(&mut self, frames: &[Vec<Complex<f32>>]) -> f32 {
        let mut checksum = 0.0;
        for frame in frames {
            assert_eq!(frame.len(), self.interleaved.len());
            let mut split = DspSplitComplex {
                real: self.real.as_mut_ptr(),
                imaginary: self.imaginary.as_mut_ptr(),
            };
            // SAFETY: Complex<f32> is C-layout-compatible with two adjacent
            // f32 values (asserted above), every buffer holds `frame.len()`
            // elements, interleaved strides are two scalar lanes, split
            // strides are one, and Accelerate does not retain them.
            unsafe {
                vDSP_ctoz(frame.as_ptr().cast(), 2, &mut split, 1, frame.len());
                vDSP_ztoc(
                    &split,
                    1,
                    self.interleaved.as_mut_ptr().cast(),
                    2,
                    frame.len(),
                );
            }
            checksum += self.interleaved[frame.len() / 2].re;
        }
        checksum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerate_roundtrip_preserves_interleaved_complex_values() {
        let input = vec![vec![
            Complex::new(1.0, -2.0),
            Complex::new(3.0, 4.0),
            Complex::new(-5.0, 6.0),
        ]];
        let mut bridge = SplitComplexRoundtrip::new(3);
        bridge.run(&input);
        assert_eq!(bridge.interleaved, input[0]);
    }
}
