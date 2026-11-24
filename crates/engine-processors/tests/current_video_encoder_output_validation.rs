//! Destination identity is validated before a source can be probed. Existing
//! destination bytes and nonexistent path components remain untouched.
use ferment_engine_core::{MultiVideoEncoding, ProgressSink, VideoEncodingVariant};
use ferment_engine_processors::{VideoEncoder, platform_encode_backends};
use std::path::{Path, PathBuf};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferment-encoder-output-validation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn rejected(d: &Directory, first: &Path, second: &Path) {
    std::fs::write(first, b"original caller-owned bytes").unwrap();
    let variants = [
        VideoEncodingVariant {
            id: "large-hdr".into(),
            output_path: first.to_owned(),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: None,
        },
        VideoEncodingVariant {
            id: "small-hdr".into(),
            output_path: second.to_owned(),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: None,
        },
    ];
    let result = futures_executor::block_on(
        VideoEncoder::new()
            .with_backends(platform_encode_backends())
            .encode_variants(
                &d.0.join("SOURCE_MUST_NEVER_BE_PROBED.mp4"),
                &variants,
                ProgressSink::noop(),
                None,
            ),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("duplicate video output destination")
    );
    assert_eq!(
        std::fs::read(first).unwrap(),
        b"original caller-owned bytes"
    );
    assert_eq!(
        std::fs::read_dir(&d.0).unwrap().count(),
        1,
        "preflight created a file or directory"
    );
}

/// Kill: accept different result IDs for one destination and let HDR lead lookup choose another variant.
#[test]
fn duplicate_destinations_fail_before_probe_or_output_mutation() {
    let d = Directory::new();
    let path = d.0.join("same.mov");
    rejected(&d, &path, &path);
}

/// Kill: compare only spelling, allowing a nonexistent-parent/../ alias to evade destination ownership.
#[test]
fn lexical_destination_aliases_fail_without_creating_the_missing_component() {
    let d = Directory::new();
    let first = d.0.join("same.mov");
    let second =
        d.0.join("not-created")
            .join("..")
            .join(".")
            .join("same.mov");
    rejected(&d, &first, &second);
    assert!(!d.0.join("not-created").exists());
}
