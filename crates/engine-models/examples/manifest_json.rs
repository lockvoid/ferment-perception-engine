//! Dump the default fleet manifest as JSON — the machine-readable source of
//! truth for tooling (`scripts/models-publish` reads it to stage CDN uploads).

fn main() {
    let manifest = ferment_engine_models::default_fleet_manifest();
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest).expect("manifest serializes")
    );
}
