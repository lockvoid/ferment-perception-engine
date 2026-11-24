use std::cmp::Ordering;
use std::fmt;

use super::{MotionBlurDecision, MotionBlurMode, MotionBlurPolicy, SlowMotionAnalysis};

/// Portable rational media time. Multiplication preserves the source
/// timescale, matching CoreMedia's slow-motion PTS contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTime {
    pub value: i64,
    pub timescale: u32,
}

impl MediaTime {
    #[must_use]
    pub const fn new(value: i64, timescale: u32) -> Self {
        Self { value, timescale }
    }

    #[must_use]
    pub fn seconds(self) -> f64 {
        self.value as f64 / f64::from(self.timescale)
    }

    fn multiply(self, multiplier: i64) -> Self {
        Self::new(self.value.saturating_mul(multiplier), self.timescale)
    }

    fn add(self, other: Self) -> Self {
        let timescale = common_timescale(self.timescale, other.timescale);
        let lhs = converted_value(self, timescale);
        let rhs = converted_value(other, timescale);
        Self::new(lhs.saturating_add(rhs), timescale)
    }

    fn subtract(self, other: Self) -> Self {
        let timescale = common_timescale(self.timescale, other.timescale);
        let lhs = converted_value(self, timescale);
        let rhs = converted_value(other, timescale);
        Self::new(lhs.saturating_sub(rhs), timescale)
    }

    fn multiplied_by_ratio(self, numerator: usize, denominator: usize) -> Self {
        debug_assert_ne!(denominator, 0);
        let numerator = i64::try_from(numerator).unwrap_or(i64::MAX);
        let denominator = u32::try_from(denominator).unwrap_or(u32::MAX);
        let raw = self.value.saturating_mul(numerator);
        let divisor = gcd_u64(raw.unsigned_abs(), u64::from(denominator));
        let reduced_denominator = u64::from(denominator) / divisor;
        let timescale = u64::from(self.timescale).saturating_mul(reduced_denominator);
        let value = i128::from(raw) / i128::from(divisor);
        Self::new(
            value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64,
            u32::try_from(timescale).unwrap_or(u32::MAX),
        )
    }
}

impl PartialOrd for MediaTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MediaTime {
    fn cmp(&self, other: &Self) -> Ordering {
        (i128::from(self.value) * i128::from(other.timescale))
            .cmp(&(i128::from(other.value) * i128::from(self.timescale)))
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.max(1)
}

fn common_timescale(left: u32, right: u32) -> u32 {
    let gcd = gcd_u64(u64::from(left), u64::from(right));
    let lcm = u64::from(left) / gcd * u64::from(right);
    u32::try_from(lcm).unwrap_or(u32::MAX)
}

fn converted_value(time: MediaTime, timescale: u32) -> i64 {
    let numerator = i128::from(time.value) * i128::from(timescale);
    let value = numerator / i128::from(time.timescale);
    value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// A decoded frame and its source presentation timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedFrame<F> {
    pub frame: F,
    pub pts: MediaTime,
}

/// Frame synthesis seam. Platform optical-flow processors and deterministic
/// test doubles implement the same ordering/restart contract.
pub trait FrameInterpolator<F> {
    fn interpolate(
        &mut self,
        from: &TimedFrame<F>,
        to: &TimedFrame<F>,
        phases: &[f32],
        starts_new_sequence: bool,
    ) -> Result<Vec<F>, String>;
}

/// Source-anchor blur seam. Neighbors are absent at clip and scene edges.
pub trait FrameBlurrer<F> {
    fn blur(
        &mut self,
        frame: &TimedFrame<F>,
        previous: Option<&TimedFrame<F>>,
        next: Option<&TimedFrame<F>>,
        strength: i32,
    ) -> Result<F, String>;
}

/// Transactional output seam. Platform muxers own partial-file creation and
/// atomic delivery; the engine owns begin/append/commit/abort ordering.
pub trait FrameOutput<F> {
    fn begin(&mut self) -> Result<(), String>;
    fn append(&mut self, frame: F, pts: MediaTime) -> Result<(), String>;
    fn commit(&mut self) -> Result<(), String>;
    fn abort(&mut self);
}

#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub factor: u8,
    pub window_start: MediaTime,
    pub window_duration: MediaTime,
    pub scene_cuts: Vec<MediaTime>,
    pub motion_blur: MotionBlurMode,
    pub target_shutter_angle: f64,
    pub analysis: Option<SlowMotionAnalysis>,
}

impl RenderConfig {
    #[must_use]
    pub fn new(factor: u8, window_duration: MediaTime) -> Self {
        Self {
            factor,
            window_start: MediaTime::new(0, window_duration.timescale),
            window_duration,
            scene_cuts: Vec::new(),
            motion_blur: MotionBlurMode::Auto,
            target_shutter_angle: 180.0,
            analysis: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlowMotionRenderResult {
    pub frames_written: usize,
    pub output_pts: MediaTime,
    pub motion_blur: MotionBlurDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlowMotionRenderError {
    InvalidFactor(u8),
    Cancelled,
    Interpolator(String),
    Blurrer(String),
    Output(String),
    OutputCount { expected: usize, actual: usize },
    Unsupported(String),
}

impl fmt::Display for SlowMotionRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFactor(factor) => write!(formatter, "invalid slow-motion factor {factor}"),
            Self::Cancelled => formatter.write_str("slow-motion render cancelled"),
            Self::Interpolator(message) => write!(formatter, "interpolator: {message}"),
            Self::Blurrer(message) => write!(formatter, "blurrer: {message}"),
            Self::Output(message) => write!(formatter, "output: {message}"),
            Self::OutputCount { expected, actual } => {
                write!(
                    formatter,
                    "interpolator returned {actual} frames, expected {expected}"
                )
            }
            Self::Unsupported(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for SlowMotionRenderError {}

/// In-between phases for a normal source pair.
#[must_use]
pub fn interpolation_phases(factor: u8) -> Vec<f32> {
    (1..factor)
        .map(|phase| f32::from(phase) / f32::from(factor))
        .collect()
}

/// Scale a source PTS without degrading its timescale.
#[must_use]
pub fn output_pts(source: MediaTime, factor: u8) -> MediaTime {
    source.multiply(i64::from(factor))
}

/// A cut in `(from, to]` belongs to the pair arriving at the new scene.
#[must_use]
pub fn pair_crosses_cut(from: MediaTime, to: MediaTime, cuts: &[MediaTime]) -> bool {
    cuts.iter().any(|cut| *cut > from && *cut <= to)
}

pub const PIXEL_FORMAT_420V: u32 = u32::from_be_bytes(*b"420v");
pub const PIXEL_FORMAT_X420: u32 = u32::from_be_bytes(*b"x420");
pub const TRANSFER_HLG: &str = "ITU_R_2100_HLG";
pub const TRANSFER_PQ: &str = "SMPTE_ST_2084_PQ";

/// Resolve an interpolator's advertised formats to the one scalar format a
/// reader needs. `None` and an empty list use the platform-neutral defaults.
#[must_use]
pub fn reader_pixel_format(advertised: Option<&[u32]>, prefer_hdr: bool) -> u32 {
    let default = if prefer_hdr {
        PIXEL_FORMAT_X420
    } else {
        PIXEL_FORMAT_420V
    };
    let Some(formats) = advertised else {
        return default;
    };
    if prefer_hdr && formats.contains(&PIXEL_FORMAT_X420) {
        PIXEL_FORMAT_X420
    } else {
        formats.first().copied().unwrap_or(default)
    }
}

/// Whether the chosen reader format is accepted by the blur processor.
#[must_use]
pub fn format_compatible(chosen: Option<u32>, blur_formats: Option<&[u32]>) -> bool {
    let Some(chosen) = chosen else { return true };
    blur_formats.is_none_or(|formats| formats.contains(&chosen))
}

#[must_use]
pub fn is_hdr_transfer(transfer: Option<&str>) -> bool {
    matches!(transfer, Some(TRANSFER_HLG | TRANSFER_PQ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderCapability {
    VideoDecode,
    VideoEncode,
    PlayableMuxedOutput,
    VideoToolboxFrameInterpolation,
    VideoToolboxMotionBlur,
    VideoToolboxCapabilityFactory,
    PrivatePixelEffectVerification,
}

/// Explicit gate for exact Swift oracles that require media I/O or private
/// VideoToolbox effects. Ignored tests call this so forcing them fails loudly.
pub fn require_render_capabilities(
    required: &[RenderCapability],
) -> Result<(), SlowMotionRenderError> {
    Err(SlowMotionRenderError::Unsupported(format!(
        "slow-motion render unavailable; missing capabilities: {required:?}"
    )))
}

/// Run the I/O-free render orchestration over already decoded frames.
///
/// Decoding and muxing remain outside this module. The injected output is a
/// transaction, so cancellation and every backend failure abort partial work.
pub fn render_frames<F: Clone>(
    source: &[TimedFrame<F>],
    config: &RenderConfig,
    interpolator: &mut dyn FrameInterpolator<F>,
    mut blurrer: Option<&mut dyn FrameBlurrer<F>>,
    output: &mut dyn FrameOutput<F>,
    mut is_cancelled: impl FnMut() -> bool,
    mut on_progress: impl FnMut(f64),
) -> Result<SlowMotionRenderResult, SlowMotionRenderError> {
    if !(2..=8).contains(&config.factor) {
        return Err(SlowMotionRenderError::InvalidFactor(config.factor));
    }

    let decision = MotionBlurPolicy::decide(
        config.motion_blur,
        config.analysis.as_ref(),
        config.factor,
        config.target_shutter_angle,
        blurrer.is_some(),
    );
    output.begin().map_err(SlowMotionRenderError::Output)?;

    let rendered = (|| {
        let cuts = config
            .scene_cuts
            .iter()
            .map(|cut| cut.subtract(config.window_start))
            .collect::<Vec<_>>();
        let relative = source
            .iter()
            .map(|frame| TimedFrame {
                frame: frame.frame.clone(),
                pts: frame.pts.subtract(config.window_start),
            })
            .collect::<Vec<_>>();
        let cut_before = relative
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                index > 0 && pair_crosses_cut(relative[index - 1].pts, frame.pts, &cuts)
            })
            .collect::<Vec<_>>();

        let mut anchors = Vec::with_capacity(relative.len());
        for (index, frame) in relative.iter().enumerate() {
            if is_cancelled() {
                return Err(SlowMotionRenderError::Cancelled);
            }
            let value = if decision.applied {
                let previous = (index > 0 && !cut_before[index]).then(|| &relative[index - 1]);
                let next = (index + 1 < relative.len() && !cut_before[index + 1])
                    .then(|| &relative[index + 1]);
                blurrer
                    .as_deref_mut()
                    .expect("an applied decision requires an available blurrer")
                    .blur(frame, previous, next, decision.strength)
                    .map_err(SlowMotionRenderError::Blurrer)?
            } else {
                frame.frame.clone()
            };
            anchors.push((
                TimedFrame {
                    frame: value,
                    pts: frame.pts,
                },
                cut_before[index],
            ));
            let progress = if config.window_duration.value > 0 {
                (frame.pts.seconds() / config.window_duration.seconds()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            on_progress(progress);
        }

        let mut scheduler = Scheduler::new(config.factor, interpolator, output);
        for (anchor, is_cut) in anchors {
            scheduler.emit_anchor(anchor, is_cut)?;
        }
        scheduler.finish()
    })();

    match rendered {
        Ok((frames_written, last_pts)) => {
            if let Err(error) = output.commit() {
                output.abort();
                return Err(SlowMotionRenderError::Output(error));
            }
            on_progress(1.0);
            Ok(SlowMotionRenderResult {
                frames_written,
                output_pts: last_pts,
                motion_blur: decision,
            })
        }
        Err(error) => {
            output.abort();
            Err(error)
        }
    }
}

struct PendingCut<F> {
    a_last: TimedFrame<F>,
    a_last_shift: usize,
    b_first: TimedFrame<F>,
}

pub(crate) struct Scheduler<'a, F> {
    factor: u8,
    interpolator: &'a mut dyn FrameInterpolator<F>,
    output: &'a mut dyn FrameOutput<F>,
    previous_anchor: Option<TimedFrame<F>>,
    pending_cut: Option<PendingCut<F>>,
    absorb_ticks: usize,
    needs_sequence_restart: bool,
    last_tick: MediaTime,
    frames_written: usize,
    last_output_pts: MediaTime,
}

impl<'a, F: Clone> Scheduler<'a, F> {
    pub(crate) fn new(
        factor: u8,
        interpolator: &'a mut dyn FrameInterpolator<F>,
        output: &'a mut dyn FrameOutput<F>,
    ) -> Self {
        Self {
            factor,
            interpolator,
            output,
            previous_anchor: None,
            pending_cut: None,
            absorb_ticks: 0,
            needs_sequence_restart: true,
            last_tick: MediaTime::new(1, 30),
            frames_written: 0,
            last_output_pts: MediaTime::new(0, 1),
        }
    }

    fn shifted_output(&self, source: MediaTime, ticks: usize, tick: MediaTime) -> MediaTime {
        output_pts(source, self.factor).subtract(tick.multiply(ticks as i64))
    }

    fn append(&mut self, frame: F, pts: MediaTime) -> Result<(), SlowMotionRenderError> {
        self.output
            .append(frame, pts)
            .map_err(SlowMotionRenderError::Output)?;
        self.frames_written += 1;
        self.last_output_pts = pts;
        Ok(())
    }

    fn write_fill(
        &mut self,
        frames: Vec<F>,
        start: MediaTime,
        end: MediaTime,
    ) -> Result<(), SlowMotionRenderError> {
        let count = frames.len();
        let span = end.subtract(start);
        for (index, frame) in frames.into_iter().enumerate() {
            let pts = start.add(span.multiplied_by_ratio(index + 1, count + 1));
            self.append(frame, pts)?;
        }
        Ok(())
    }

    fn interpolate(
        &mut self,
        from: &TimedFrame<F>,
        to: &TimedFrame<F>,
        phase_count: usize,
        starts_new_sequence: bool,
    ) -> Result<Vec<F>, SlowMotionRenderError> {
        let phases = uniform_phases(phase_count);
        let frames = self
            .interpolator
            .interpolate(from, to, &phases, starts_new_sequence)
            .map_err(SlowMotionRenderError::Interpolator)?;
        if frames.len() != phases.len() {
            return Err(SlowMotionRenderError::OutputCount {
                expected: phases.len(),
                actual: frames.len(),
            });
        }
        Ok(frames)
    }

    pub(crate) fn emit_anchor(
        &mut self,
        anchor: TimedFrame<F>,
        cut_before: bool,
    ) -> Result<(), SlowMotionRenderError> {
        if let Some(cut) = self.pending_cut.take() {
            let tick = cut.b_first.pts.subtract(cut.a_last.pts);
            self.last_tick = tick;
            let a_out = self.shifted_output(cut.a_last.pts, cut.a_last_shift, tick);
            self.append(cut.a_last.frame, a_out)?;
            let deficit = usize::from(self.factor) - 1 + cut.a_last_shift;

            if cut_before {
                self.previous_anchor = Some(anchor.clone());
                self.pending_cut = Some(PendingCut {
                    a_last: cut.b_first,
                    a_last_shift: deficit,
                    b_first: anchor,
                });
                return Ok(());
            }

            let b_out = a_out.add(tick);
            self.append(cut.b_first.frame.clone(), b_out)?;
            let next_shift = deficit.saturating_sub(1);
            let pair_tick = anchor.pts.subtract(cut.b_first.pts);
            let anchor_out = self.shifted_output(anchor.pts, next_shift, pair_tick);
            let phase_count = usize::from(self.factor) - 1 + deficit - next_shift;
            let in_betweens = self.interpolate(&cut.b_first, &anchor, phase_count, true)?;
            self.needs_sequence_restart = false;
            self.write_fill(in_betweens, b_out, anchor_out)?;
            self.previous_anchor = Some(anchor);
            self.absorb_ticks = next_shift;
            return Ok(());
        }

        if cut_before && let Some(previous) = self.previous_anchor.take() {
            self.pending_cut = Some(PendingCut {
                a_last: previous,
                a_last_shift: self.absorb_ticks,
                b_first: anchor.clone(),
            });
            self.absorb_ticks = 0;
            self.previous_anchor = Some(anchor);
            return Ok(());
        }

        if let Some(previous) = self.previous_anchor.take() {
            let tick = anchor.pts.subtract(previous.pts);
            self.last_tick = tick;
            let previous_out = self.shifted_output(previous.pts, self.absorb_ticks, tick);
            self.append(previous.frame.clone(), previous_out)?;
            let next_shift = self.absorb_ticks.saturating_sub(1);
            let anchor_out = self.shifted_output(anchor.pts, next_shift, tick);
            let phase_count = usize::from(self.factor) - 1 + self.absorb_ticks - next_shift;
            let starts_new_sequence = self.needs_sequence_restart;
            let in_betweens =
                self.interpolate(&previous, &anchor, phase_count, starts_new_sequence)?;
            self.needs_sequence_restart = false;
            self.write_fill(in_betweens, previous_out, anchor_out)?;
            self.absorb_ticks = next_shift;
        }
        self.previous_anchor = Some(anchor);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(usize, MediaTime), SlowMotionRenderError> {
        if let Some(cut) = self.pending_cut.take() {
            let tick = cut.b_first.pts.subtract(cut.a_last.pts);
            let a_out = self.shifted_output(cut.a_last.pts, cut.a_last_shift, tick);
            self.append(cut.a_last.frame, a_out)?;
            let b_out = a_out.add(tick);
            self.append(cut.b_first.frame.clone(), b_out)?;
            let owed = usize::from(self.factor) - 1 + cut.a_last_shift;
            if owed > 0 {
                let end = b_out.add(tick.multiply((owed + 1) as i64));
                self.write_fill(vec![cut.b_first.frame; owed], b_out, end)?;
            }
            self.previous_anchor = None;
        }

        if let Some(anchor) = self.previous_anchor.take() {
            let anchor_out = self.shifted_output(anchor.pts, self.absorb_ticks, self.last_tick);
            self.append(anchor.frame.clone(), anchor_out)?;
            if self.absorb_ticks > 0 {
                let end = anchor_out.add(
                    self.last_tick
                        .multiply((self.absorb_ticks.saturating_add(1)) as i64),
                );
                self.write_fill(vec![anchor.frame; self.absorb_ticks], anchor_out, end)?;
            }
        }
        Ok((self.frames_written, self.last_output_pts))
    }
}

fn uniform_phases(count: usize) -> Vec<f32> {
    (1..=count)
        .map(|phase| phase as f32 / (count + 1) as f32)
        .collect()
}
