//! One place where a libav status becomes a [`crate::VideoError`].
//!
//! libav reports failure as a negative `AVERROR` code, and the code alone is
//! useless in a log ("avcodec_send_packet returned -1094995529"). Every call
//! site in this backend goes through [`check_av`], which names the call and
//! attaches `av_strerror`'s text — so the error a user sees says
//! `avformat_open_input: Invalid data found when processing input`, and the
//! numeric code is still there for a bug report.

use std::ffi::{CStr, c_char, c_int};

use ffmpeg_next::ffi;

use crate::{Result, VideoError};

/// `status` when the call succeeded, a [`VideoError::Libav`] naming `call`
/// otherwise. libav's convention is that negative is failure and any
/// non-negative value is a result (a byte count, a stream index, zero).
pub fn check_av(call: &'static str, status: c_int) -> Result<c_int> {
    if status >= 0 {
        return Ok(status);
    }
    Err(VideoError::Libav {
        code: status,
        call,
        message: strerror(status),
    })
}

/// libav's own description of an AVERROR code.
fn strerror(code: c_int) -> String {
    // AV_ERROR_MAX_STRING_SIZE.
    const SIZE: usize = 64;
    let mut buffer = [0 as c_char; SIZE];
    // SAFETY: `buffer` is a live, writable array of exactly `SIZE` bytes;
    // av_strerror writes at most that many and NUL-terminates.
    let written = unsafe { ffi::av_strerror(code, buffer.as_mut_ptr(), SIZE) };
    if written < 0 {
        return format!("unknown libav error {code}");
    }
    // SAFETY: on success av_strerror NUL-terminates within `SIZE`.
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    String::from_utf8_lossy(text.to_bytes()).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_negative_status_is_the_value_itself() {
        assert_eq!(check_av("avcodec_send_packet", 0).unwrap(), 0);
        assert_eq!(check_av("av_read_frame", 7).unwrap(), 7);
    }

    /// The point of the helper: the message a caller reads names the call and
    /// carries libav's own words, not a bare number.
    #[test]
    fn a_negative_status_carries_the_call_and_av_strerror_text() {
        super::super::init().unwrap();
        // AVERROR(EINVAL) — the code libav returns for a rejected argument.
        let error = check_av("avformat_open_input", -22).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("avformat_open_input"), "{rendered}");
        assert!(rendered.contains("-22"), "{rendered}");
        assert!(
            rendered.to_lowercase().contains("invalid argument"),
            "expected av_strerror's text, got {rendered}"
        );
        assert!(matches!(error, VideoError::Libav { code: -22, .. }));
    }
}
