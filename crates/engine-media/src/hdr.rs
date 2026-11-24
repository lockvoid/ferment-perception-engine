//! HDR pictures (BT.2100 PQ and HLG) as eight-bit SDR BT.709: the one
//! conversion the engine applies wherever such a source becomes RGB8. An HDR
//! project instead takes the same light unclipped and unmapped, as linear
//! float (`to_scene_linear`).
//!
//! Display light per BT.2100: the PQ EOTF, or the HLG inverse OETF and OOTF
//! at its 1000 cd/m² nominal peak. BT.2020 (or Display P3) primaries become
//! BT.709 by the simple conversion of ITU-R Report BT.2407: the matrix, then
//! negative components set to zero, so saturated colours stay saturated (as
//! Apple's own conversion draws BT.2020 primaries). The BT.2390 EETF on
//! maxRGB maps the 1000 cd/m² peak onto SDR white, which stands for the
//! BT.2408 HDR reference white of 203 cd/m²: mid-tones keep their light and
//! highlights roll off instead of clipping. The inverse BT.1886 EOTF (gamma
//! 2.4) encodes the result like any SDR BT.709 master.
//!
//! Every pixel takes the exact chain in f64. Video frames do not come here:
//! a software decoder hands their planes to the GPU (`DecodeOutput::
//! CpuPlanes`), whose shader runs the same chain; this serves pictures read
//! on the CPU — thumbnails, explicit readback, hosts without a GPU. (A 65³
//! table stood here before, 26 levels off on saturated colours at the gamut
//! boundary, 19.11.)

use crate::{
    ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuFrame, CpuPixelFormat, Result,
    TransferCharacteristic, VideoError,
};

/// HDR reference white, the light SDR white stands for (ITU-R BT.2408).
const WHITE: f64 = 203.0;
/// HLG nominal peak, and the PQ light the EETF maps onto SDR white.
const PEAK: f64 = 1000.0;
const ENCODING: f64 = 1.0 / 2.4;

/// What the scene-linear pixels are: linear BT.709 primaries, extended range,
/// 1.0 the HDR reference white, float32.
pub(crate) const SCENE_LINEAR: ColorInfo = ColorInfo {
    range: ColorRange::Full,
    primaries: ColorPrimaries::Bt709,
    transfer: TransferCharacteristic::Linear,
    matrix: ColorMatrix::Identity,
    bits_per_channel: Some(32),
};

/// What the converted pixels are: BT.709 SDR, eight bits, full-range RGB.
pub(crate) const SDR: ColorInfo = ColorInfo {
    range: ColorRange::Full,
    primaries: ColorPrimaries::Bt709,
    transfer: TransferCharacteristic::Bt709,
    matrix: ColorMatrix::Bt709,
    bits_per_channel: Some(8),
};

pub(crate) fn is_hdr(transfer: TransferCharacteristic) -> bool {
    matches!(
        transfer,
        TransferCharacteristic::Hlg | TransferCharacteristic::Pq
    )
}

/// One YCbCr component as laid out in memory.
#[derive(Clone, Copy)]
pub(crate) struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    /// Bytes from one sample to the next in a row.
    pub step: usize,
    pub offset: usize,
    /// A sample is a little-endian word shifted right by this (P010: 6),
    /// or one byte when the picture is eight bits deep.
    pub shift: u32,
    /// log2 of the horizontal and vertical subsampling.
    pub subsampling: (u32, u32),
}

/// A YCbCr picture of 8–16 bit codes.
pub(crate) struct Picture<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Y, Cb, Cr.
    pub planes: [Plane<'a>; 3],
}

/// `picture` (an HDR source, tagged by `color`) as SDR BT.709 pixels in
/// `format`, tightly packed: Rgb24, Rgba8 or Bgra8, alpha opaque.
pub(crate) fn to_sdr(
    picture: &Picture,
    color: ColorInfo,
    format: CpuPixelFormat,
) -> Result<CpuFrame> {
    let order = match format {
        CpuPixelFormat::Rgb24 | CpuPixelFormat::Rgba8 => [0, 1, 2],
        CpuPixelFormat::Bgra8 => [2, 1, 0],
        CpuPixelFormat::Rgba32Float => {
            return Err(VideoError::UnsupportedCodec(
                "HDR to SDR writes eight-bit pixels".into(),
            ));
        }
    };
    let channels = format.bytes_per_pixel();
    let data = each_pixel(picture, color, channels, |light, pixel| {
        let sdr = tone(light);
        for (channel, value) in order.into_iter().zip(sdr) {
            // `as` saturates; +0.5 rounds the 0..=1 level.
            pixel[channel] = (value * 255.0 + 0.5) as u8;
        }
        if channels == 4 {
            pixel[3] = u8::MAX;
        }
    })?;
    CpuFrame::new(
        picture.width,
        picture.height,
        picture.width as usize * channels,
        format,
        data,
    )
}

/// `picture` (an HDR source, tagged by `color`) as the light of an HDR
/// project: linear BT.709 primaries with extended range (a colour outside
/// BT.709 has a negative component), 1.0 the BT.2408 reference white of
/// 203 cd/m², highlights kept above it. No EETF and no clipping: the same
/// light `to_sdr` maps, before its bends. Rgba32Float, alpha 1.
pub(crate) fn to_scene_linear(picture: &Picture, color: ColorInfo) -> Result<CpuFrame> {
    let format = CpuPixelFormat::Rgba32Float;
    let data = each_pixel(picture, color, format.bytes_per_pixel(), |light, pixel| {
        let rgba = [light[0] as f32, light[1] as f32, light[2] as f32, 1.0];
        for (bytes, value) in pixel.chunks_exact_mut(4).zip(rgba) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    })?;
    CpuFrame::new(
        picture.width,
        picture.height,
        picture.width as usize * format.bytes_per_pixel(),
        format,
        data,
    )
}

/// The shared pass: every pixel's BT.709 light, relative to SDR white and
/// unclipped, handed to `write` with its `pixel_bytes` of output.
fn each_pixel(
    picture: &Picture,
    color: ColorInfo,
    pixel_bytes: usize,
    write: impl Fn([f64; 3], &mut [u8]) + Sync,
) -> Result<Vec<u8>> {
    if !is_hdr(color.transfer) || !(8..=16).contains(&picture.depth) {
        return Err(VideoError::UnsupportedCodec(format!(
            "HDR conversion needs a PQ or HLG picture of 8-16 bits, not {:?} at {}",
            color.transfer, picture.depth
        )));
    }
    let gamut = match color.primaries {
        ColorPrimaries::Bt709 => Gamut::Bt709,
        ColorPrimaries::DisplayP3 => Gamut::DisplayP3,
        // BT.2100 is BT.2020; an untagged HDR source is too.
        _ => Gamut::Bt2020,
    };
    let transfer = color.transfer;
    let codes = Codes::new(color, picture.depth)?;
    let (width, height) = (picture.width as usize, picture.height as usize);
    if width == 0 || height == 0 {
        return Err(VideoError::InvalidData("empty HDR picture".into()));
    }
    let wide = picture.depth > 8;
    for plane in &picture.planes {
        plane.check(width, height, wide)?;
    }
    let mask = ((1u32 << picture.depth) - 1) as u16;
    let row_bytes = width * pixel_bytes;
    let mut data = vec![0u8; row_bytes * height];
    let [y_plane, cb_plane, cr_plane] = picture.planes;
    let rows = |first: usize, chunk: &mut [u8]| {
        for (index, row) in chunk.chunks_exact_mut(row_bytes).enumerate() {
            let y = first + index;
            for (x, pixel) in row.chunks_exact_mut(pixel_bytes).enumerate() {
                let signal = codes.signal(
                    y_plane.sample(x, y, wide, mask),
                    cb_plane.sample(x, y, wide, mask),
                    cr_plane.sample(x, y, wide, mask),
                );
                write(light(signal.map(f64::from), transfer, gamut), pixel);
            }
        }
    };
    // Below a megapixel threads cost more than they save.
    let threads = if width * height >= 1 << 20 {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
    } else {
        1
    };
    if threads == 1 {
        rows(0, &mut data);
    } else {
        let per_thread = height.div_ceil(threads);
        let rows = &rows;
        std::thread::scope(|scope| {
            for (index, chunk) in data.chunks_mut(per_thread * row_bytes).enumerate() {
                scope.spawn(move || rows(index * per_thread, chunk));
            }
        });
    }
    Ok(data)
}

impl Plane<'_> {
    #[inline]
    fn sample(&self, x: usize, y: usize, wide: bool, mask: u16) -> f32 {
        let at = (y >> self.subsampling.1) * self.stride
            + (x >> self.subsampling.0) * self.step
            + self.offset;
        if wide {
            f32::from((u16::from_le_bytes([self.data[at], self.data[at + 1]]) >> self.shift) & mask)
        } else {
            f32::from(self.data[at])
        }
    }

    /// The first `samples` of this plane's row covering luma row `y`, as
    /// codes: bytes, or little-endian words with the code in the low bits.
    /// A layout that already is that (planar, unshifted) is copied as is.
    pub(crate) fn copy_row(&self, y: usize, samples: usize, wide: bool, mask: u16, out: &mut [u8]) {
        let start = (y >> self.subsampling.1) * self.stride + self.offset;
        let bytes = if wide { 2 } else { 1 };
        if self.step == bytes && self.shift == 0 {
            out[..samples * bytes].copy_from_slice(&self.data[start..start + samples * bytes]);
            if wide {
                for word in out[..samples * 2].chunks_exact_mut(2) {
                    let code = u16::from_le_bytes([word[0], word[1]]) & mask;
                    word.copy_from_slice(&code.to_le_bytes());
                }
            }
        } else if wide {
            for (index, word) in out.chunks_exact_mut(2).take(samples).enumerate() {
                let at = start + index * self.step;
                let code =
                    (u16::from_le_bytes([self.data[at], self.data[at + 1]]) >> self.shift) & mask;
                word.copy_from_slice(&code.to_le_bytes());
            }
        } else {
            for (index, byte) in out.iter_mut().take(samples).enumerate() {
                *byte = self.data[start + index * self.step];
            }
        }
    }

    pub(crate) fn check(&self, width: usize, height: usize, wide: bool) -> Result<()> {
        let columns = width.div_ceil(1 << self.subsampling.0);
        let rows = height.div_ceil(1 << self.subsampling.1);
        let end = (rows - 1)
            .checked_mul(self.stride)
            .and_then(|start| start.checked_add((columns - 1).checked_mul(self.step)?))
            .and_then(|start| start.checked_add(self.offset + if wide { 2 } else { 1 }));
        if self.step == 0 || end.is_none_or(|end| end > self.data.len()) {
            return Err(VideoError::InvalidData(
                "HDR picture plane is shorter than its extent".into(),
            ));
        }
        Ok(())
    }
}

/// YCbCr codes of one depth, matrix and range to R'G'B' signal.
#[derive(Clone, Copy)]
struct Codes {
    black: f32,
    luma: f32,
    zero: f32,
    chroma: f32,
    cr_r: f32,
    cb_g: f32,
    cr_g: f32,
    cb_b: f32,
}

impl Codes {
    fn new(color: ColorInfo, depth: u32) -> Result<Self> {
        let (kr, kb) = match color.matrix {
            // Constant luminance decodes approximately through the same
            // coefficients; an untagged HDR source is BT.2100.
            ColorMatrix::Bt2020NonConstant
            | ColorMatrix::Bt2020Constant
            | ColorMatrix::Unspecified => (0.2627, 0.0593),
            ColorMatrix::Bt709 => (0.2126, 0.0722),
            ColorMatrix::Bt470Bg | ColorMatrix::Smpte170M => (0.299, 0.114),
            other => {
                return Err(VideoError::UnsupportedCodec(format!(
                    "HDR to SDR from the {other:?} matrix"
                )));
            }
        };
        let kg = 1.0 - kr - kb;
        // BT.2100 Table 9; an untagged range is the video default.
        let unit = f64::from(1u32 << (depth - 8));
        let max = f64::from((1u32 << depth) - 1);
        let (black, luma, zero, chroma) = if color.range == ColorRange::Full {
            (0.0, max, f64::from(1u32 << (depth - 1)), max)
        } else {
            (16.0 * unit, 219.0 * unit, 128.0 * unit, 224.0 * unit)
        };
        Ok(Self {
            black: black as f32,
            luma: (1.0 / luma) as f32,
            zero: zero as f32,
            chroma: (1.0 / chroma) as f32,
            cr_r: (2.0 * (1.0 - kr)) as f32,
            cb_g: (2.0 * kb * (1.0 - kb) / kg) as f32,
            cr_g: (2.0 * kr * (1.0 - kr) / kg) as f32,
            cb_b: (2.0 * (1.0 - kb)) as f32,
        })
    }

    #[inline]
    fn signal(&self, y: f32, cb: f32, cr: f32) -> [f32; 3] {
        let y = (y - self.black) * self.luma;
        let cb = (cb - self.zero) * self.chroma;
        let cr = (cr - self.zero) * self.chroma;
        [
            y + self.cr_r * cr,
            y - self.cb_g * cb - self.cr_g * cr,
            y + self.cb_b * cb,
        ]
        .map(|value| value.clamp(0.0, 1.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gamut {
    Bt709,
    Bt2020,
    DisplayP3,
}

impl Gamut {
    /// Linear RGB to BT.709, from the primaries' chromaticities at D65.
    const fn to_bt709(self) -> Option<[[f64; 3]; 3]> {
        match self {
            Self::Bt709 => None,
            Self::Bt2020 => Some([
                [1.660_491_002_108, -0.587_641_138_789, -0.072_849_863_320],
                [-0.124_550_474_522, 1.132_899_897_126, -0.008_349_422_604],
                [-0.018_150_763_355, -0.100_578_898_008, 1.118_729_661_363],
            ]),
            Self::DisplayP3 => Some([
                [1.224_940_176_281, -0.224_940_176_281, 0.0],
                [-0.042_056_954_710, 1.042_056_954_710, 0.0],
                [-0.019_637_554_590, -0.078_636_045_551, 1.098_273_600_141],
            ]),
        }
    }

    const fn luminance(self) -> [f64; 3] {
        match self {
            Self::Bt709 => [0.212_639_005_872, 0.715_168_678_768, 0.072_192_315_361],
            Self::Bt2020 => [0.262_700_212_011, 0.677_998_071_519, 0.059_301_716_470],
            Self::DisplayP3 => [0.228_974_564_070, 0.691_738_521_837, 0.079_286_914_094],
        }
    }
}

/// The whole chain for one source R'G'B': SDR R'G'B', 0..=1.
#[cfg(test)]
fn sdr(signal: [f64; 3], transfer: TransferCharacteristic, gamut: Gamut) -> [f64; 3] {
    tone(light(signal, transfer, gamut))
}

/// BT.709 light (relative to SDR white) as SDR R'G'B', 0..=1: negatives
/// clipped, the EETF on the brightest component, BT.1886 encoding.
fn tone(light: [f64; 3]) -> [f64; 3] {
    let rgb = light.map(|value| value.max(0.0));
    let gain = gain(rgb[0].max(rgb[1]).max(rgb[2]));
    rgb.map(|value| (value * gain).clamp(0.0, 1.0).powf(ENCODING))
}

/// BT.709 light relative to SDR white, before clipping.
fn light(signal: [f64; 3], transfer: TransferCharacteristic, gamut: Gamut) -> [f64; 3] {
    let light = display_light(signal, transfer, gamut);
    let light = match gamut.to_bt709() {
        None => light,
        Some(matrix) => matrix.map(|row| row[0] * light[0] + row[1] * light[1] + row[2] * light[2]),
    };
    light.map(|value| value / WHITE)
}

/// What the EETF multiplies `brightest` (relative to SDR white) by: 1 up to
/// its knee, then the roll-off onto SDR white.
fn gain(brightest: f64) -> f64 {
    if brightest <= 0.0 {
        return 1.0;
    }
    eetf(brightest * WHITE) / WHITE / brightest
}

/// cd/m² in the source primaries.
fn display_light(signal: [f64; 3], transfer: TransferCharacteristic, gamut: Gamut) -> [f64; 3] {
    let signal = signal.map(|value| value.clamp(0.0, 1.0));
    if transfer == TransferCharacteristic::Pq {
        return signal.map(pq_eotf);
    }
    let scene = signal.map(hlg_scene);
    let [kr, kg, kb] = gamut.luminance();
    let luminance = kr * scene[0] + kg * scene[1] + kb * scene[2];
    // OOTF: γ = 1.2 at the 1000 cd/m² nominal peak, black at zero.
    let gain = PEAK * luminance.powf(0.2);
    scene.map(|value| value * gain)
}

/// ITU-R BT.2390 EETF from PEAK onto WHITE in the PQ domain, cd/m².
fn eetf(light: f64) -> f64 {
    let source = pq_signal(PEAK);
    let level = (pq_signal(light) / source).min(1.0);
    let top = pq_signal(WHITE) / source;
    let knee = 1.5 * top - 0.5;
    if level <= knee {
        return light;
    }
    let t = (level - knee) / (1.0 - knee);
    let (t2, t3) = (t * t, t * t * t);
    let level = (2.0 * t3 - 3.0 * t2 + 1.0) * knee
        + (t3 - 2.0 * t2 + t) * (1.0 - knee)
        + (3.0 * t2 - 2.0 * t3) * top;
    pq_eotf(level * source).min(WHITE)
}

const M1: f64 = 2610.0 / 16384.0;
const M2: f64 = 2523.0 / 4096.0 * 128.0;
const C1: f64 = 3424.0 / 4096.0;
const C2: f64 = 2413.0 / 4096.0 * 32.0;
const C3: f64 = 2392.0 / 4096.0 * 32.0;

fn pq_eotf(signal: f64) -> f64 {
    let power = signal.clamp(0.0, 1.0).powf(1.0 / M2);
    10_000.0 * ((power - C1).max(0.0) / (C2 - C3 * power)).powf(1.0 / M1)
}

fn pq_signal(light: f64) -> f64 {
    let power = (light / 10_000.0).clamp(0.0, 1.0).powf(M1);
    ((C1 + C2 * power) / (1.0 + C3 * power)).powf(M2)
}

/// HLG inverse OETF: scene light, 0..=1.
fn hlg_scene(signal: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    let b = 1.0 - 4.0 * A;
    let c = 0.5 - A * (4.0 * A).ln();
    if signal <= 0.5 {
        signal * signal / 3.0
    } else {
        (((signal - c) / A).exp() + b) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HLG: TransferCharacteristic = TransferCharacteristic::Hlg;
    const PQ: TransferCharacteristic = TransferCharacteristic::Pq;

    fn grey(signal: f64, transfer: TransferCharacteristic) -> f64 {
        let [r, g, b] = sdr([signal; 3], transfer, Gamut::Bt2020);
        assert!((r - g).abs() < 1e-9 && (g - b).abs() < 1e-9, "{r} {g} {b}");
        r
    }

    /// Kill: per-channel HLG gamma instead of the luminance OOTF, a reference
    /// white other than 203 cd/m², or a PQ path on another scale.
    #[test]
    fn the_same_light_reads_the_same_level_in_hlg_and_pq() {
        for (hlg, nits, level) in [
            (0.38, 26.24, 0.4264),
            (0.75, 203.15, 0.9034),
            (1.0, 1000.0, 1.0),
        ] {
            let (from_hlg, from_pq) = (grey(hlg, HLG), grey(pq_signal(nits), PQ));
            assert!((from_hlg - level).abs() < 5e-4, "HLG {hlg}: {from_hlg}");
            assert!((from_pq - level).abs() < 5e-4, "PQ {nits}: {from_pq}");
        }
        assert_eq!(grey(0.0, HLG), 0.0);
        assert_eq!(grey(0.0, PQ), 0.0);
    }

    /// Kill: clipping at reference white, or a curve that is not monotonic.
    #[test]
    fn highlights_roll_off_up_to_the_peak_instead_of_clipping() {
        let ramp: Vec<f64> = (0..=100).map(|i| grey(f64::from(i) / 100.0, HLG)).collect();
        assert!(ramp.windows(2).all(|pair| pair[0] < pair[1]), "{ramp:?}");
        // Above reference white (75%) every step still changes the picture.
        assert!(
            ramp[85] < 0.99 && ramp[95] < 1.0,
            "{} {}",
            ramp[85],
            ramp[95]
        );
        // Below the knee the light is the source's own.
        let dark = grey(0.25, HLG);
        let nits = 1000.0 * hlg_scene(0.25).powf(1.2);
        assert!((dark - (nits / WHITE).powf(1.0 / 2.4)).abs() < 1e-9);
        // PQ above the 1000 cd/m² it maps from is SDR white.
        assert!((grey(pq_signal(4000.0), PQ) - 1.0).abs() < 1e-9);
    }

    /// Kill: dropping the gamut conversion (BT.2020 green drawn as BT.709
    /// green), a soft mapping that tints saturated colours, or one that moves
    /// colours already inside BT.709. BT.2020 primaries at full signal land on
    /// the pure BT.709 primaries, as Apple's own conversion draws them.
    #[test]
    fn bt2020_primaries_clip_to_pure_bt709_and_inside_colours_stay() {
        for (channel, transfer) in [(0, HLG), (1, HLG), (2, HLG), (0, PQ), (1, PQ), (2, PQ)] {
            let mut signal = [0.0; 3];
            signal[channel] = if transfer == PQ {
                pq_signal(1000.0)
            } else {
                1.0
            };
            let encoded = sdr(signal, transfer, Gamut::Bt2020);
            for (index, value) in encoded.iter().enumerate() {
                // HLG's OOTF gives a blue primary a little less light.
                let pure = if index == channel {
                    *value > 0.99
                } else {
                    *value == 0.0
                };
                assert!(pure, "{transfer:?} {signal:?}: {encoded:?}");
            }
        }
        // A BT.709 colour inside its gamut comes back as itself.
        let inside = [0.4, 0.2, 0.12];
        let bt2020 = [
            [0.627_404, 0.329_283, 0.043_313],
            [0.069_097, 0.919_540, 0.011_362],
            [0.016_391, 0.088_013, 0.895_595],
        ]
        .map(|row| row[0] * inside[0] + row[1] * inside[1] + row[2] * inside[2]);
        let matrix = Gamut::Bt2020.to_bt709().unwrap();
        let back = matrix.map(|row| row[0] * bt2020[0] + row[1] * bt2020[1] + row[2] * bt2020[2]);
        for (value, expected) in back.iter().zip(inside) {
            assert!((value - expected).abs() < 1e-5, "{back:?}");
        }
    }

    /// Saturated cyan at the gamut boundary from a real HLG clip
    /// (`docs/dev/evidence/gpu-decode-20261119/exact.log` in Hypr): the red
    /// component just above zero is where the old 65³ table was 26 levels
    /// off. Through the real pass on Y'CbCr codes, each reads the exact
    /// chain. Kill: a table or curve back in the pass.
    #[test]
    fn saturated_colours_at_the_gamut_boundary_read_the_exact_chain() {
        let color = ColorInfo {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Hlg,
            matrix: ColorMatrix::Bt2020NonConstant,
            bits_per_channel: Some(10),
        };
        for ([y, cb, cr], exact) in [
            ([600u16, 564, 421], [28u8, 196, 220]),
            ([612, 556, 422], [34, 204, 219]),
            ([608, 597, 425], [24, 178, 239]),
        ] {
            fn read(data: &[u8]) -> Plane<'_> {
                Plane {
                    data,
                    stride: 2,
                    step: 2,
                    offset: 0,
                    shift: 0,
                    subsampling: (0, 0),
                }
            }
            let plane = |code: u16| code.to_le_bytes().to_vec();
            let (y, cb, cr) = (plane(y), plane(cb), plane(cr));
            let picture = Picture {
                width: 1,
                height: 1,
                depth: 10,
                planes: [read(&y), read(&cb), read(&cr)],
            };
            let frame = to_sdr(&picture, color, CpuPixelFormat::Rgb24).unwrap();
            assert_eq!(frame.data[..3], exact, "codes {y:?} {cb:?} {cr:?}");
        }
    }

    /// Kill: P010's high-aligned codes read unshifted, interleaved chroma
    /// read from the wrong offset, or BGRA written as RGBA.
    #[test]
    fn interleaved_ten_bit_chroma_and_bgra_order_are_read_as_laid_out() {
        // 2×2 P010: reference white luma; chroma pushed towards red (Cr up).
        let luma = [721u16 << 6; 4].map(u16::to_le_bytes).concat();
        let chroma = [512u16 << 6, 600 << 6].map(u16::to_le_bytes).concat();
        let picture = Picture {
            width: 2,
            height: 2,
            depth: 10,
            planes: [
                Plane {
                    data: &luma,
                    stride: 4,
                    step: 2,
                    offset: 0,
                    shift: 6,
                    subsampling: (0, 0),
                },
                Plane {
                    data: &chroma,
                    stride: 4,
                    step: 4,
                    offset: 0,
                    shift: 6,
                    subsampling: (1, 1),
                },
                Plane {
                    data: &chroma,
                    stride: 4,
                    step: 4,
                    offset: 2,
                    shift: 6,
                    subsampling: (1, 1),
                },
            ],
        };
        let color = ColorInfo {
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt2020,
            transfer: HLG,
            matrix: ColorMatrix::Bt2020NonConstant,
            bits_per_channel: Some(10),
        };
        let frame = to_sdr(&picture, color, CpuPixelFormat::Bgra8).unwrap();
        let &[b, g, r, a] = &frame.data[..4] else {
            unreachable!()
        };
        assert!(r > g && r > b && a == 255, "{:?}", &frame.data[..4]);
        assert!(
            frame
                .data
                .chunks_exact(4)
                .all(|pixel| pixel == &frame.data[..4])
        );
        let codes = Codes::new(color, 10).unwrap();
        let exact = sdr(
            codes.signal(721.0, 512.0, 600.0).map(f64::from),
            HLG,
            Gamut::Bt2020,
        );
        for (value, exact) in [r, g, b].iter().zip(exact) {
            assert!((f64::from(*value) - exact * 255.0).abs() <= 1.0, "{exact}");
        }
        // The same picture as scene light: float32 RGBA, the exact chain's
        // light before clipping, alpha 1.
        let frame = to_scene_linear(&picture, color).unwrap();
        assert_eq!(frame.format, CpuPixelFormat::Rgba32Float);
        let pixel: Vec<f32> = frame.data[..16]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let exact = light(
            codes.signal(721.0, 512.0, 600.0).map(f64::from),
            HLG,
            Gamut::Bt2020,
        );
        for (value, exact) in pixel.iter().zip(exact) {
            assert!(
                (f64::from(*value) - exact).abs() < 2e-3 * exact.abs().max(1.0),
                "{exact}"
            );
        }
        assert_eq!(pixel[3], 1.0);
    }

    fn scene(signal: [f64; 3], transfer: TransferCharacteristic) -> [f64; 3] {
        light(signal, transfer, Gamut::Bt2020)
    }

    /// Kill: an HDR project clipped at reference white, mapped by the EETF,
    /// on another scale than 203 cd/m² = 1.0, or BT.2020 colours clipped to
    /// BT.709 instead of carried as negative components.
    #[test]
    fn scene_light_keeps_reference_white_highlights_and_wide_colours() {
        for (hlg, nits) in [(0.38, 26.24), (0.75, 203.15), (1.0, 1000.0)] {
            let level = nits / WHITE;
            for [r, g, b] in [scene([hlg; 3], HLG), scene([pq_signal(nits); 3], PQ)] {
                assert!((r - g).abs() < 1e-3 && (g - b).abs() < 1e-3, "{r} {g} {b}");
                assert!(
                    (r - level).abs() < 2e-3 * level.max(1.0),
                    "{nits} cd/m²: {r}"
                );
            }
        }
        // 4000 cd/m² PQ stays far above white, not rolled off to it.
        let [bright, ..] = scene([pq_signal(4000.0); 3], PQ);
        assert!((bright - 4000.0 / WHITE).abs() < 0.05, "{bright}");
        // Pure BT.2020 green is outside BT.709: red and blue go negative.
        let [r, g, b] = scene([0.0, 0.75, 0.0], HLG);
        let exact = light([0.0, 0.75, 0.0], HLG, Gamut::Bt2020);
        assert!(r < 0.0 && b < 0.0 && g > 0.0, "{r} {g} {b}");
        for (value, exact) in [r, g, b].into_iter().zip(exact) {
            assert!(
                (value - exact).abs() < 2e-3 * exact.abs().max(1.0),
                "{value} {exact}"
            );
        }
    }
}
