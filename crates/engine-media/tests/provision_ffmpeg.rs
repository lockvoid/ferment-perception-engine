//! `scripts/provision-ffmpeg`'s download path, end to end.
//!
//! Every machine and every CI runner gets its libav\* libraries through that
//! path, and a mistake in it does not fail loudly — it leaves a half-unpacked
//! vendor tree that the next `cargo build` reports as a missing header. So the
//! test builds its own mirror out of the libraries this very binary is linked
//! against, serves it over `file://`, and provisions from it into a scratch
//! directory: fetch, digest check, unpack and the tree verification, exactly
//! as the CDN path runs them.
//!
//! Nothing here touches `vendor/ffmpeg` —
//! `FERMENT_FFMPEG_VENDOR` redirects the script, because wiping that tree
//! would pull the dylibs out from under the running test.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/engine-media has a repository root")
        .to_path_buf()
}

/// A default `scripts/provision-ffmpeg` pins (`NAME="${ENV:-default}"`),
/// read from the script so the test cannot drift away from it.
fn pinned(root: &Path, assignment: &str) -> String {
    let script = std::fs::read_to_string(root.join("scripts/provision-ffmpeg"))
        .expect("scripts/provision-ffmpeg is readable");
    script
        .lines()
        .find_map(|line| line.strip_prefix(assignment))
        .and_then(|rest| rest.split('}').next())
        .unwrap_or_else(|| panic!("the script pins {assignment}"))
        .to_owned()
}

/// The FFmpeg release the script pins.
fn pinned_version(root: &Path) -> String {
    pinned(root, "VERSION=\"${FFMPEG_VERSION:-")
}

/// The build number beside it: the same release compiled with a different
/// whitelist is a different artifact, and the archive is named by both.
fn pinned_build(root: &Path) -> String {
    pinned(root, "BUILD=\"${FFMPEG_BUILD:-")
}

/// A command in the script's own environment. The script is a bash program
/// and the justfile runs every recipe under bash (`set shell := ["bash",
/// "-uc"]`), so its `tar`, `zstd` and `curl` are the ones the test uses too,
/// not whichever an executable search finds first: on Windows that `tar` is
/// System32's bsdtar, which stalls behind an external compressor.
fn bash() -> Command {
    Command::new("bash")
}

fn have(tool: &str) -> bool {
    bash()
        .args(["-c", r#""$0" --version"#, tool])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[test]
fn the_download_path_fetches_verifies_and_unpacks_a_mirror() {
    let root = repository_root();
    let version = pinned_version(&root);
    let build = pinned_build(&root);
    let target = current_target();
    let vendor = root.join("vendor/ffmpeg");
    let provisioned = vendor.join(&target);

    if !provisioned.join("include/libavcodec/avcodec.h").exists() {
        println!("SKIP the_download_path…: no provisioned {target} tree to mirror");
        return;
    }
    if !have("zstd") || !have("tar") {
        println!("SKIP the_download_path…: tar + zstd are needed to build a mirror");
        return;
    }

    let scratch =
        std::env::temp_dir().join(format!("ferment-ffmpeg-mirror-{}", std::process::id()));
    let mirror = scratch.join("mirror");
    let into = scratch.join("vendor");
    std::fs::create_dir_all(&mirror).expect("scratch mirror directory");
    std::fs::create_dir_all(&into).expect("scratch vendor directory");
    struct Scrub(PathBuf);
    impl Drop for Scrub {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _scrub = Scrub(scratch.clone());

    // The mirror: the same archive shape `--package` produces, so the script
    // under test cannot tell this apart from the CDN. The archive is named
    // relative to the mirror: GNU tar reads a drive letter's colon as a host.
    let name = format!("ffmpeg-{version}-{build}-{target}-lgpl-shared.tar.zst");
    let packed = bash()
        .args([
            "-c",
            r#"tar --use-compress-program "zstd -1" -cf "$0" -C "$1" "$2""#,
        ])
        .arg(&name)
        .arg(&vendor)
        .arg(&target)
        .current_dir(&mirror)
        .status()
        .expect("bash runs");
    assert!(packed.success(), "packing the mirror archive failed");

    let digest = sha256_hex(&mirror.join(&name));
    std::fs::write(
        mirror.join(format!("{name}.sha256")),
        format!("{digest}  {name}\n"),
    )
    .expect("mirror sidecar");

    let run = bash()
        .arg(root.join("scripts/provision-ffmpeg"))
        .arg(&target)
        .env(
            "FERMENT_FFMPEG_MIRROR",
            format!("file://{}", mirror.display()),
        )
        .env("FERMENT_FFMPEG_VENDOR", &into)
        .env("FFMPEG_VERSION", &version)
        .env("FFMPEG_BUILD", &build)
        .output()
        .expect("scripts/provision-ffmpeg runs");
    assert!(
        run.status.success(),
        "provisioning from the mirror failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    assert!(
        into.join(&target)
            .join("include/libavcodec/avcodec.h")
            .exists(),
        "the provisioned tree has no libavcodec headers",
    );

    // A corrupted download must fail rather than leave a broken tree behind.
    // This is the case the digest exists for, so it gets asserted rather than
    // assumed: same archive, a sidecar that no longer describes it.
    std::fs::write(
        mirror.join(format!("{name}.sha256")),
        format!("{}  {name}\n", "0".repeat(64)),
    )
    .expect("tampered sidecar");
    let tampered = bash()
        .arg(root.join("scripts/provision-ffmpeg"))
        .arg(&target)
        .env(
            "FERMENT_FFMPEG_MIRROR",
            format!("file://{}", mirror.display()),
        )
        .env("FERMENT_FFMPEG_VENDOR", &into)
        .env("FFMPEG_VERSION", &version)
        .env("FFMPEG_BUILD", &build)
        .output()
        .expect("scripts/provision-ffmpeg runs");
    assert!(
        !tampered.status.success(),
        "a digest mismatch was accepted: {}",
        String::from_utf8_lossy(&tampered.stdout),
    );
    assert!(
        String::from_utf8_lossy(&tampered.stderr).contains("digest mismatch"),
        "the failure did not name the digest: {}",
        String::from_utf8_lossy(&tampered.stderr),
    );
}

fn current_target() -> String {
    // The script's own `host_target` mapping, for the platforms we build.
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        ("windows", _) => "x86_64-pc-windows-msvc",
        (os, arch) => panic!("unsupported host {os}/{arch}"),
    }
    .to_owned()
}

fn sha256_hex(path: &Path) -> String {
    let archive = std::fs::read(path).expect("the mirror archive is readable");
    Sha256::digest(archive)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// -------------------------------------------------------------------------
// The dependency gate, host side.
//
// `scripts/provision-ffmpeg` runs this after every build and every download.
// It runs here too, because the vendor tree is what `cargo test` and the dev
// binary actually load, and because a bad dependency is invisible on the
// machine that produced it — it fails on a user's machine, at first import,
// as a dyld abort with no stack.

/// The dependencies a shipped library is allowed to name, by platform.
///
/// macOS: our own siblings through `@rpath`, plus the two prefixes every mac
/// is guaranteed to have. A `/opt/homebrew`, `/usr/local` or `/opt/local`
/// path is a library that exists only on the build box.
///
/// Only the macOS half of this file reads it — `otool` is the reader, and the
/// Android tree is gated by `assert_android_dependencies` in the script itself
/// (the NDK's `llvm-readelf`), not from here.
#[cfg(target_os = "macos")]
fn is_system_dependency(path: &str) -> bool {
    path.starts_with("@rpath/")
        || path.starts_with("/System/Library/")
        || path.starts_with("/usr/lib/")
}

/// Names `otool -L` prints for `library`, excluding its own install-name line.
#[cfg(target_os = "macos")]
fn linked_libraries(library: &Path) -> Vec<String> {
    let output = Command::new("otool")
        .arg("-L")
        .arg(library)
        .output()
        .expect("otool runs on macOS");
    assert!(
        output.status.success(),
        "otool failed on {}",
        library.display()
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
fn the_vendored_libraries_link_nothing_outside_the_system() {
    let root = repository_root();
    let lib = root
        .join("vendor/ffmpeg")
        .join(current_target())
        .join("lib");
    if !lib.is_dir() {
        println!("SKIP the_vendored_libraries…: run `just provision-ffmpeg-mac` first");
        return;
    }

    let mut checked = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&lib).expect("the vendored lib directory") {
        let path = entry.expect("a directory entry").path();
        // Skip the unversioned symlinks; each names the same real file.
        if path.extension().and_then(|e| e.to_str()) != Some("dylib")
            || std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink())
        {
            continue;
        }
        checked += 1;
        for dependency in linked_libraries(&path) {
            if !is_system_dependency(&dependency) {
                offenders.push(format!("{}: {dependency}", path.display()));
            }
        }
    }
    assert!(
        checked >= 6,
        "expected the six libav* dylibs, saw {checked}"
    );
    assert!(
        offenders.is_empty(),
        "the vendored libraries link outside the system — an artifact built \
         with configure's autodetect on, which loads only on the machine that \
         made it:\n  {}",
        offenders.join("\n  "),
    );
}

/// The predicate above only means something if it REJECTS something. Homebrew
/// is the exact shape of the failure this gate exists for — its FFmpeg links
/// forty libraries out of /opt/homebrew — so when it is installed, it stands
/// in as the negative case.
#[cfg(target_os = "macos")]
#[test]
fn the_dependency_predicate_rejects_a_homebrew_built_library() {
    // Pure cases first: these hold with or without Homebrew on the box.
    assert!(is_system_dependency("@rpath/libavutil.60.dylib"));
    assert!(is_system_dependency("/usr/lib/libSystem.B.dylib"));
    assert!(is_system_dependency(
        "/System/Library/Frameworks/VideoToolbox.framework/Versions/A/VideoToolbox"
    ));
    assert!(!is_system_dependency(
        "/opt/homebrew/opt/libx11/lib/libX11.6.dylib"
    ));
    assert!(!is_system_dependency("/usr/local/lib/libx264.164.dylib"));

    let brewed = Path::new("/opt/homebrew/opt/ffmpeg/lib/libavcodec.dylib");
    if !brewed.exists() {
        println!("SKIP the real-library half: Homebrew's ffmpeg is not installed");
        return;
    }
    let rejected: Vec<String> = linked_libraries(brewed)
        .into_iter()
        .filter(|dependency| !is_system_dependency(dependency))
        .collect();
    assert!(
        !rejected.is_empty(),
        "Homebrew's libavcodec links only system paths, which cannot be right — \
         the predicate has stopped rejecting anything",
    );
}
