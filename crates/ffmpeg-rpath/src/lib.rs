//! One line of build-script glue: tell the binary being linked where the
//! vendored FFmpeg dylibs are.
//!
//! `scripts/provision-ffmpeg` installs them with `@rpath/libavcodec.62.dylib`
//! install names, because the shipped macOS app resolves them out of
//! `Contents/Frameworks`. A `cargo test` or `cargo run` binary has no bundle,
//! so without an rpath of its own dyld searches `/usr/local/lib`, finds
//! nothing, and aborts the process before `main` — a failure that reads as a
//! crash rather than as missing provisioning.
//!
//! An rpath is a property of the executable, and no cargo directive
//! propagates a link argument from a dependency, so every crate in the
//! workspace that produces a binary linking libav calls this from its own
//! `build.rs`:
//!
//! ```text
//! // build.rs
//! fn main() {
//!     ferment_ffmpeg_rpath::emit();
//! }
//! ```
//!
//! `text` rather than a doctest: the snippet is the whole of a `build.rs`,
//! which is not a context rustdoc can compile it in, and building it as one
//! hung `rustdoc` — and with it every `cargo test` in the workspace.

/// Emit the dev-time rpath for `$FFMPEG_DIR/lib`, if that variable is set.
///
/// Silent when it is not: a target that does not link FFmpeg (the wasm
/// checks, a bare `cargo doc`) must not fail here. Windows needs nothing —
/// its loader takes DLLs from the executable's own directory and `PATH`,
/// which is what `just provision-ffmpeg-win` and the bundle both arrange.
/// Android needs nothing either — its loader takes the libraries from the
/// APK's `lib/<abi>`, and an rpath would only bake the build machine's path
/// into the shipped `.so`. Both read the TARGET: a build script runs on the
/// host, so `cfg!(windows)` would answer for the wrong machine.
pub fn emit() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" || target_os == "android" {
        return;
    }
    let Ok(ffmpeg_dir) = std::env::var("FFMPEG_DIR") else {
        return;
    };
    // `.cargo/config.toml` resolves FFMPEG_DIR to an absolute path, but its
    // last component is the `host` symlink; canonicalizing hands dyld a real
    // directory rather than a link to resolve at load time.
    let lib = std::path::Path::new(&ffmpeg_dir).join("lib");
    let lib = std::fs::canonicalize(&lib).unwrap_or(lib);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
