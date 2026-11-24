//! libav's log stream, routed into the `log` facade.
//!
//! Left alone, libavformat and libavcodec write straight to stderr — a
//! "moov atom not found" would appear in the terminal with no level, no
//! target, and no path into the file sink `tauri-plugin-log` owns. This
//! module takes the callback and re-emits every line under the target
//! `ferment_engine_media::libav`, so a media problem is filterable with the
//! same `RUST_LOG` the rest of the app uses.

use std::ffi::{CStr, c_char, c_int, c_void};

use ffmpeg_next::ffi;

/// libav's own level ladder (`libavutil/log.h`). PANIC (0) and FATAL (8) sit
/// below ERROR and need no constant of their own — the mapping is a ladder of
/// upper bounds, so they fall into ERROR's rung.
const AV_LOG_ERROR: c_int = 16;
const AV_LOG_WARNING: c_int = 24;
const AV_LOG_INFO: c_int = 32;
const AV_LOG_VERBOSE: c_int = 40;

/// One formatted libav line. 1024 is `av_log_default_callback`'s own buffer;
/// anything longer is a hex dump we would truncate anyway.
const LINE: usize = 1024;

pub(super) fn install() {
    // An escape hatch for diagnosing libav itself. Our callback routes lines
    // into the `log` facade, and a test binary installs no logger — so
    // libavformat's explanation of a refusal goes nowhere exactly when it is
    // most wanted. With this set, libav keeps its own stderr logging.
    if std::env::var_os("FERMENT_LIBAV_STDERR").is_some() {
        // SAFETY: a process-global setter, called once from `init()`.
        unsafe { ffi::av_log_set_level(AV_LOG_VERBOSE) };
        return;
    }
    // INFO and below is per-packet chatter ("Using non-standard frame rate")
    // that would flood a release log. Callers who want it raise the level
    // themselves; the callback maps it either way.
    // SAFETY: both are process-global setters with no aliasing requirement,
    // called from `init()`'s `OnceLock` before any decode exists.
    unsafe {
        ffi::av_log_set_level(AV_LOG_WARNING);
        ffi::av_log_set_callback(Some(callback));
    }
}

/// # Safety
///
/// Called by libav on its own threads with a printf format string and its
/// `va_list`. The pair is only ever handed straight back to
/// `av_log_format_line2`, which is the function libav documents for exactly
/// this: formatting a callback's arguments without interpreting them here.
unsafe extern "C" fn callback(
    context: *mut c_void,
    level: c_int,
    format: *const c_char,
    arguments: ffi::va_list,
) {
    // `print_prefix` is an in/out flag libav uses to continue a partial line.
    // A fresh 1 per call means every line carries its own "[h264 @ 0x…]"
    // component prefix, which is the part that says WHICH decoder spoke.
    let mut print_prefix: c_int = 1;
    let mut line = [0 as c_char; LINE];
    // SAFETY: `line` is a live buffer of exactly `LINE` bytes; `format` and
    // `arguments` are libav's own and valid for this call.
    let written = unsafe {
        ffi::av_log_format_line2(
            context,
            level,
            format,
            arguments,
            line.as_mut_ptr(),
            LINE as c_int,
            &mut print_prefix,
        )
    };
    if written <= 0 {
        return;
    }
    // SAFETY: on success `av_log_format_line2` NUL-terminates within `LINE`.
    let message = unsafe { CStr::from_ptr(line.as_ptr()) };
    let message = String::from_utf8_lossy(message.to_bytes());
    let message = message.trim_end();
    if message.is_empty() {
        return;
    }

    // PANIC and FATAL sit below ERROR on libav's ladder and have no louder
    // level to land on; the facade tops out at `error`. DEBUG and TRACE both
    // land on `trace` — libav's DEBUG is per-frame, not a developer aside.
    let level = if level <= AV_LOG_ERROR {
        ::log::Level::Error
    } else if level <= AV_LOG_WARNING {
        ::log::Level::Warn
    } else if level <= AV_LOG_INFO {
        ::log::Level::Info
    } else if level <= AV_LOG_VERBOSE {
        ::log::Level::Debug
    } else {
        ::log::Level::Trace
    };
    ::log::log!(target: "ferment_engine_media::libav", level, "{message}");
}
