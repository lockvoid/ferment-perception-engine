use std::{env, path::PathBuf, process::Command};

fn output(args: &[&str]) -> String {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .expect("Xcode is required");
    assert!(
        output.status.success(),
        "xcrun {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8 tool output")
        .trim()
        .to_owned()
}

fn main() {
    println!("cargo:rerun-if-changed=src/Bridge.swift");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let sdk = output(&["--sdk", "macosx", "--show-sdk-path"]);
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported Mac architecture {other}"),
    };
    let deployment = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".into());
    let target = format!("{arch}-apple-macos{deployment}");
    let archive = out.join("libferment_coreai.a");
    let header = out.join("FermentCoreAI.h");
    let status = Command::new("xcrun")
        .args([
            "--sdk",
            "macosx",
            "swiftc",
            "-parse-as-library",
            "-emit-library",
            "-static",
            "-O",
            "-swift-version",
            "6",
            "-module-name",
            "FermentCoreAI",
            "-target",
            &target,
            "-sdk",
            &sdk,
            "-emit-objc-header-path",
        ])
        .arg(&header)
        .args(["src/Bridge.swift", "-o"])
        .arg(&archive)
        .status()
        .unwrap();
    assert!(status.success(), "CoreAI Swift bridge compilation failed");
    // Swift emits the C declaration of every @c entry point. Generate Rust
    // from that declaration: Int is ptrdiff_t/isize, not size_t/usize.
    bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_args(["-x", "c", "-target", &target, "-isysroot", &sdk])
        .allowlist_function("ferment_coreai_.*")
        .layout_tests(false)
        .generate()
        .expect("CoreAI generated C ABI could not be read")
        .write_to_file(out.join("bindings.rs"))
        .expect("CoreAI Rust ABI could not be written");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ferment_coreai");
    // Swift object autolink records name the runtime libraries. Apple's clang
    // linker processes these records; provide the SDK and toolchain search paths.
    let swiftc = PathBuf::from(output(&["--find", "swiftc"]));
    let swift_lib = swiftc
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("lib/swift/macosx");
    println!("cargo:rustc-link-search=native={}", swift_lib.display());
    println!("cargo:rustc-link-search=native={sdk}/usr/lib/swift");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
