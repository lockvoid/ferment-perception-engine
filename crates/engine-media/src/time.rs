use crate::{Result, VideoError};

/// Seconds per timestamp tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeBase {
    pub numerator: i64,
    pub denominator: i64,
}

impl TimeBase {
    pub const SECONDS: Self = Self::new_unchecked(1, 1);
    pub const MILLISECONDS: Self = Self::new_unchecked(1, 1_000);
    pub const MICROSECONDS: Self = Self::new_unchecked(1, 1_000_000);
    pub const NANOSECONDS: Self = Self::new_unchecked(1, 1_000_000_000);

    pub fn new(numerator: i64, denominator: i64) -> Result<Self> {
        if numerator <= 0 || denominator <= 0 {
            return Err(VideoError::InvalidData(format!(
                "time base must be positive, got {numerator}/{denominator}"
            )));
        }
        Ok(Self::new_unchecked(numerator, denominator))
    }

    pub const fn new_unchecked(numerator: i64, denominator: i64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn seconds_for(self, ticks: i64) -> f64 {
        ticks as f64 * self.numerator as f64 / self.denominator as f64
    }

    pub fn ticks_for(self, seconds: f64) -> i64 {
        if !seconds.is_finite() {
            return 0;
        }
        let ticks = seconds * self.denominator as f64 / self.numerator as f64;
        ticks.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    }

    pub fn rescale(self, ticks: i64, destination: Self) -> i64 {
        let numerator = i128::from(ticks)
            .saturating_mul(i128::from(self.numerator))
            .saturating_mul(i128::from(destination.denominator));
        let denominator =
            i128::from(self.denominator).saturating_mul(i128::from(destination.numerator));
        if denominator == 0 {
            return 0;
        }
        let value = numerator / denominator;
        value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
    }
}

/// A media timestamp with its exact time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MediaTime {
    pub value: i64,
    pub time_base: TimeBase,
}

impl MediaTime {
    pub const ZERO: Self = Self::new(0, TimeBase::SECONDS);

    pub const fn new(value: i64, time_base: TimeBase) -> Self {
        Self { value, time_base }
    }

    pub fn from_seconds(seconds: f64, time_base: TimeBase) -> Self {
        Self::new(time_base.ticks_for(seconds), time_base)
    }

    pub fn seconds(self) -> f64 {
        self.time_base.seconds_for(self.value)
    }

    pub fn rescale(self, destination: TimeBase) -> Self {
        Self::new(self.time_base.rescale(self.value, destination), destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_rescaling_is_exact_for_common_video_bases() {
        let source = TimeBase::new(1, 90_000).unwrap();
        let destination = TimeBase::new(1, 1_000).unwrap();
        assert_eq!(source.rescale(225_000, destination), 2_500);
    }

    #[test]
    fn rejects_invalid_time_bases() {
        assert!(TimeBase::new(1, 0).is_err());
        assert!(TimeBase::new(-1, 1_000).is_err());
    }
}
