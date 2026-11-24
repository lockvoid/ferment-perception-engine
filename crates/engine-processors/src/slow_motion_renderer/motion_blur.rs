/// Caller intent for motion blur on a slow-motion render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionBlurMode {
    Off,
    Auto,
    On { strength: i32 },
}

/// Pure pre-pass measurements. Displacements are per source-frame pair as a
/// fraction of frame height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlowMotionAnalysis {
    pub source_shutter_angle: Option<f64>,
    pub median_displacement: f64,
    pub p95_displacement: f64,
    pub pairs_sampled: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionBlurReason {
    ExplicitOff,
    ExplicitOn,
    AlreadyBlurred,
    CrispPreferred,
    FastShutterSource,
    MotionTooSmall,
    Indeterminate,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionBlurDecision {
    pub applied: bool,
    pub strength: i32,
    pub source_shutter_angle: f64,
    pub output_smear_angle: f64,
    pub reason: MotionBlurReason,
}

/// Research-backed automatic motion-blur decision math.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionBlurPolicy;

impl MotionBlurPolicy {
    pub const OFF_THRESHOLD: f64 = 120.0;
    pub const ON_THRESHOLD: f64 = 90.0;
    pub const FILL_CEILING: f64 = 120.0;
    pub const STROBE_DISPLACEMENT_FLOOR: f64 = 0.003;

    #[must_use]
    pub fn output_smear_angle(source_shutter_angle: f64, factor: u8) -> f64 {
        source_shutter_angle * f64::from(factor)
    }

    /// Calculate the VideoToolbox-style strength needed to fill toward a
    /// target smear. A zero factor is treated as one for totality; render
    /// validation should reject it before reaching this layer.
    #[must_use]
    pub fn strength(fill_to: f64, output_smear_angle: f64, factor: u8) -> i32 {
        let factor = factor.max(1);
        let raw = (100.0 * (fill_to - output_smear_angle) / 360.0 / f64::from(factor)).round();
        (raw as i32).clamp(1, 100)
    }

    #[must_use]
    pub fn decide(
        mode: MotionBlurMode,
        analysis: Option<&SlowMotionAnalysis>,
        factor: u8,
        target_shutter_angle: f64,
        blur_available: bool,
    ) -> MotionBlurDecision {
        let off = |reason, shutter, smear| MotionBlurDecision {
            applied: false,
            strength: 0,
            source_shutter_angle: shutter,
            output_smear_angle: smear,
            reason,
        };

        match mode {
            MotionBlurMode::Off => off(MotionBlurReason::ExplicitOff, 0.0, 0.0),
            MotionBlurMode::On { strength } => {
                if !blur_available {
                    return off(MotionBlurReason::Unsupported, 0.0, 0.0);
                }
                MotionBlurDecision {
                    applied: true,
                    strength: strength.clamp(1, 100),
                    source_shutter_angle: 0.0,
                    output_smear_angle: 0.0,
                    reason: MotionBlurReason::ExplicitOn,
                }
            }
            MotionBlurMode::Auto => {
                if !blur_available {
                    return off(MotionBlurReason::Unsupported, 0.0, 0.0);
                }
                let Some(shutter) = analysis.and_then(|value| value.source_shutter_angle) else {
                    let assumed = 180.0;
                    return off(
                        MotionBlurReason::Indeterminate,
                        assumed,
                        Self::output_smear_angle(assumed, factor),
                    );
                };

                let smear = Self::output_smear_angle(shutter, factor);
                if smear >= Self::OFF_THRESHOLD {
                    return off(MotionBlurReason::AlreadyBlurred, shutter, smear);
                }
                if smear >= Self::ON_THRESHOLD {
                    return off(MotionBlurReason::CrispPreferred, shutter, smear);
                }

                let output_displacement = analysis
                    .expect("shutter came from analysis")
                    .median_displacement
                    / f64::from(factor.max(1));
                if output_displacement < Self::STROBE_DISPLACEMENT_FLOOR {
                    return off(MotionBlurReason::MotionTooSmall, shutter, smear);
                }
                let fill = target_shutter_angle.min(Self::FILL_CEILING);
                if fill <= smear {
                    return off(MotionBlurReason::CrispPreferred, shutter, smear);
                }

                MotionBlurDecision {
                    applied: true,
                    strength: Self::strength(fill, smear, factor),
                    source_shutter_angle: shutter,
                    output_smear_angle: smear,
                    reason: MotionBlurReason::FastShutterSource,
                }
            }
        }
    }
}
