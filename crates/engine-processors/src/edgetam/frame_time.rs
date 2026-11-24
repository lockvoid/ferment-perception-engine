//! Exact rational timestamp operations. Rounding to a fixed tick unit before
//! indexing aliases distinct samples and changes the public Double timestamps.
use ferment_engine_core::{EngineError, Result};
use ferment_engine_media::{MediaTime, TimeBase};
use std::cmp::Ordering;

pub(super) fn validate(time: MediaTime) -> Result<()> {
    if time.time_base.numerator <= 0 || time.time_base.denominator <= 0 {
        return Err(EngineError::Decode("invalid presentation time base".into()));
    }
    Ok(())
}

fn fraction(time: MediaTime) -> Result<(i128, i128)> {
    validate(time)?;
    let numerator = i128::from(time.value) * i128::from(time.time_base.numerator);
    let denominator = i128::from(time.time_base.denominator);
    let divisor = gcd(numerator.unsigned_abs(), denominator as u128) as i128;
    Ok((numerator / divisor, denominator / divisor))
}
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}
fn overflow() -> EngineError {
    EngineError::Decode("presentation time arithmetic overflow".into())
}

pub(super) fn compare(left: MediaTime, right: MediaTime) -> Result<Ordering> {
    let (ln, ld) = fraction(left)?;
    let (rn, rd) = fraction(right)?;
    let common = gcd(ld as u128, rd as u128) as i128;
    let left = ln.checked_mul(rd / common).ok_or_else(overflow)?;
    let right = rn.checked_mul(ld / common).ok_or_else(overflow)?;
    Ok(left.cmp(&right))
}

pub(super) fn subtract(time: MediaTime, origin: MediaTime) -> Result<MediaTime> {
    validate(time)?;
    validate(origin)?;
    if time.time_base == origin.time_base {
        return Ok(MediaTime::new(
            time.value.checked_sub(origin.value).ok_or_else(overflow)?,
            time.time_base,
        ));
    }
    let (tn, td) = fraction(time)?;
    let (on, od) = fraction(origin)?;
    let common = gcd(td as u128, od as u128) as i128;
    let denominator = td.checked_mul(od / common).ok_or_else(overflow)?;
    let numerator = tn
        .checked_mul(od / common)
        .ok_or_else(overflow)?
        .checked_sub(on.checked_mul(td / common).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let divisor = gcd(numerator.unsigned_abs(), denominator as u128) as i128;
    let numerator = i64::try_from(numerator / divisor).map_err(|_| overflow())?;
    let denominator = i64::try_from(denominator / divisor).map_err(|_| overflow())?;
    Ok(MediaTime::new(
        numerator,
        TimeBase::new_unchecked(1, denominator),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn time(value: i64, denominator: i64) -> MediaTime {
        MediaTime::new(value, TimeBase::new_unchecked(1, denominator))
    }
    /// Kill: quantize distinct sub-microsecond samples or reject equivalent time bases.
    #[test]
    fn rational_identity_does_not_depend_on_a_fixed_tick_unit() {
        assert_eq!(compare(time(1, 30), time(2, 60)).unwrap(), Ordering::Equal);
        assert_eq!(
            compare(time(1, 90_000_000), time(2, 90_000_000)).unwrap(),
            Ordering::Less
        );
        assert_eq!(compare(time(-1, 30), time(0, 1)).unwrap(), Ordering::Less);
        assert!(compare(time(1, 0), time(0, 1)).is_err());
    }
    /// Kill: subtract rounded origins or convert to Double before relative-time arithmetic.
    #[test]
    fn relative_times_preserve_thirtieths_across_different_bases() {
        for frame in [0, 1, 248, 249, 17_999] {
            let relative = subtract(time(42 + frame, 30), time(14, 10)).unwrap();
            assert!((relative.seconds() - frame as f64 / 30.0).abs() < 1e-12);
        }
        assert!(subtract(time(i64::MAX, 1), time(i64::MIN, 1)).is_err());
    }
}
