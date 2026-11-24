use ferment_engine_core::{ColorAnalyzing, ColorStats, PixelFormat, Result, VideoFrame};

const MAX_SAMPLES_PER_FRAME: usize = 50_000;
const NEUTRAL_CHROMA_THRESHOLD: f64 = 18.0;

#[derive(Debug, Clone)]
pub struct CpuColorAnalyzer {
    sum_rgb: [f64; 3],
    trimmed_rgb: [f64; 3],
    pixel_count: usize,
    trimmed_count: usize,
    sum_log_luma: f64,
    luma_count: usize,
    luma_histogram: [usize; 256],
    skin_chroma: [f64; 2],
    skin_count: usize,
    chroma: [f64; 2],
}

impl Default for CpuColorAnalyzer {
    fn default() -> Self {
        Self {
            sum_rgb: [0.0; 3],
            trimmed_rgb: [0.0; 3],
            pixel_count: 0,
            trimmed_count: 0,
            sum_log_luma: 0.0,
            luma_count: 0,
            luma_histogram: [0; 256],
            skin_chroma: [0.0; 2],
            skin_count: 0,
            chroma: [0.0; 2],
        }
    }
}

impl CpuColorAnalyzer {
    fn reduce(&mut self, blue: u8, green: u8, red: u8) {
        let linear = [linear(red), linear(green), linear(blue)];
        for (sum, value) in self.sum_rgb.iter_mut().zip(linear) {
            *sum += value;
        }
        self.pixel_count += 1;

        let luma = 0.2627 * linear[0] + 0.6780 * linear[1] + 0.0593 * linear[2];
        if luma > 1e-5 {
            self.sum_log_luma += luma.ln();
            self.luma_count += 1;
        }
        self.luma_histogram[(luma * 255.0).clamp(0.0, 255.0) as usize] += 1;

        let red = f64::from(red);
        let green = f64::from(green);
        let blue = f64::from(blue);
        let cb = -0.168_736 * red - 0.331_264 * green + 0.5 * blue + 128.0;
        let cr = 0.5 * red - 0.418_688 * green - 0.081_312 * blue + 128.0;
        let y = 0.299 * red + 0.587 * green + 0.114 * blue;
        self.chroma[0] += cb;
        self.chroma[1] += cr;

        let is_skin = y > 40.0 && (77.0..=127.0).contains(&cb) && (133.0..=173.0).contains(&cr);
        if is_skin {
            self.skin_chroma[0] += cb;
            self.skin_chroma[1] += cr;
            self.skin_count += 1;
        }
        if !is_skin && (cb - 128.0).abs().max((cr - 128.0).abs()) < NEUTRAL_CHROMA_THRESHOLD {
            for (sum, value) in self.trimmed_rgb.iter_mut().zip(linear) {
                *sum += value;
            }
            self.trimmed_count += 1;
        }
    }

    fn percentile(&self, percentile: f64) -> f64 {
        let target = (self.pixel_count as f64 * percentile) as usize;
        let mut cumulative = 0;
        for (bin, count) in self.luma_histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return bin as f64 / 255.0;
            }
        }
        1.0
    }
}

impl ColorAnalyzing for CpuColorAnalyzer {
    fn accumulate(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != PixelFormat::Bgra8 || frame.width == 0 || frame.height == 0 {
            return Ok(());
        }
        let Some(plane) = frame.planes.first() else {
            return Ok(());
        };
        let total = frame.width as usize * frame.height as usize;
        let step = ((total as f64 / MAX_SAMPLES_PER_FRAME as f64).sqrt() as usize).max(1);
        for y in (0..frame.height as usize).step_by(step) {
            let row = plane.offset + y * plane.stride;
            for x in (0..frame.width as usize).step_by(step) {
                let offset = row + x * 4;
                if let Some(pixel) = frame.data.get(offset..offset + 4) {
                    self.reduce(pixel[0], pixel[1], pixel[2]);
                }
            }
        }
        Ok(())
    }

    fn result(&self) -> ColorStats {
        if self.pixel_count == 0 {
            return ColorStats::default();
        }
        let count = self.pixel_count as f64;
        let neutral_mean = (self.trimmed_count >= 64.max(self.pixel_count / 50)).then(|| {
            let count = self.trimmed_count as f64;
            self.trimmed_rgb.map(|sum| sum / count)
        });
        ColorStats {
            pixel_count: self.pixel_count,
            linear_mean: self.sum_rgb.map(|sum| sum / count),
            neutral_mean,
            mean_log_luma: if self.luma_count == 0 {
                0.0
            } else {
                self.sum_log_luma / self.luma_count as f64
            },
            black_point: self.percentile(0.01),
            white_point: self.percentile(0.99),
            skin_centroid: (self.skin_count > 0).then(|| {
                let count = self.skin_count as f64;
                [
                    (self.skin_chroma[0] / count - 128.0) / 255.0,
                    (self.skin_chroma[1] / count - 128.0) / 255.0,
                ]
            }),
            skin_mass: self.skin_count as f64 / count,
            chroma_mean: [
                (self.chroma[0] / count - 128.0) / 255.0,
                (self.chroma[1] / count - 128.0) / 255.0,
            ],
        }
    }
}

fn linear(channel: u8) -> f64 {
    let value = f64::from(channel) / 255.0;
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ferment_engine_core::{MediaTime, PlaneLayout};

    use super::*;

    fn solid_frame(blue: u8, green: u8, red: u8) -> VideoFrame {
        let mut data = Vec::with_capacity(64 * 64 * 4);
        for _ in 0..64 * 64 {
            data.extend_from_slice(&[blue, green, red, 255]);
        }
        VideoFrame {
            data: Arc::from(data),
            planes: vec![PlaneLayout {
                offset: 0,
                stride: 64 * 4,
                row_bytes: 64 * 4,
                rows: 64,
            }],
            width: 64,
            height: 64,
            format: PixelFormat::Bgra8,
            presentation_time: MediaTime::default(),
        }
    }

    #[test]
    fn test_uniform_grey_stats() {
        let mut analyzer = CpuColorAnalyzer::default();
        analyzer.accumulate(&solid_frame(128, 128, 128)).unwrap();
        let stats = analyzer.result();
        // Independent sRGB reference; do not grade linear() against itself.
        let expected = 0.215_860_500_113_899_26_f64;
        assert!(stats.pixel_count > 0);
        for mean in stats.linear_mean {
            assert!((mean - expected).abs() < 0.01);
        }
        assert!(stats.neutral_mean.is_some());
        assert!(stats.skin_mass < 0.001);
        assert!(stats.chroma_mean.iter().all(|value| value.abs() < 0.01));
        assert!((stats.mean_log_luma - expected.ln()).abs() < 0.05);
    }

    #[test]
    fn test_color_cast_is_not_neutral() {
        let mut analyzer = CpuColorAnalyzer::default();
        analyzer.accumulate(&solid_frame(220, 110, 90)).unwrap();
        let stats = analyzer.result();
        assert!(stats.neutral_mean.is_none());
        assert!(stats.linear_mean[2] > stats.linear_mean[0]);
    }

    #[test]
    fn test_empty_result() {
        assert_eq!(CpuColorAnalyzer::default().result(), ColorStats::default());
    }
}
