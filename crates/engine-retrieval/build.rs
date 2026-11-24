fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var_os("CARGO_FEATURE_COREAI").is_some()
    {
        // Link arguments on coreai-native do not propagate to this crate's
        // final test/example binaries. Swift's runtime ships with macOS.
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
