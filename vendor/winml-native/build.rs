fn main() {
    println!("cargo:rerun-if-changed=include");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    bindgen::Builder::default()
        .header("include/onnxruntime_c_api.h")
        .allowlist_type("OrtApi.*")
        .allowlist_var("ORT_API_VERSION")
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("read the pinned Windows ML C ABI")
        .write_to_file(
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("ort.rs"),
        )
        .expect("write Windows ML C ABI bindings");
}
