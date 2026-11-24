//! Fixture resolution — and honest skipping — for the engine oracle suites.
//!
//! The parity suites read binary fixtures produced by the iOS engine's Swift
//! tests. Those fixtures live outside this repository, so every suite has to
//! cope with them being absent. There are exactly three ways that can go, and
//! this crate makes all three loud:
//!
//! * the fixture resolves — the test runs;
//! * no fixture root is configured and there is no sibling `ferment-cuts-ios`
//!   checkout — print `SKIP <test>: <reason>` and return `None`, so the run
//!   states what it did not check;
//! * a root **is** configured but the fixture is not under it — panic.
//!
//! The third case is the whole point. Before this crate the suites hard-coded
//! the sibling checkout path, never read `FERMENT_ENGINE_FIXTURES_DIR` at all,
//! and returned early on a failed read — so `just engine-oracle` set a variable
//! nothing consumed and roughly thirty parity tests reported success by doing
//! nothing. Pointing at a fixture root is a promise that the fixtures are
//! there; breaking that promise has to fail the build, not skip quietly.

use std::path::{Path, PathBuf};

/// Set by `just engine-oracle` to `FermentEngine/Tests/Fixtures`.
pub const ROOT_VAR: &str = "FERMENT_ENGINE_FIXTURES_DIR";

/// Checkout layout the whole fleet assumes: `ferment-redshift` and
/// `ferment-cuts-ios` as siblings. Resolved against this crate's manifest
/// directory so it does not change with the calling crate.
const SIBLING_ROOT: &str = "../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures";

/// The configured root, if `FERMENT_ENGINE_FIXTURES_DIR` names one.
///
/// # Panics
/// If the variable is set but does not name a directory — a typo in the
/// variable is otherwise indistinguishable from "fixtures unavailable", which
/// is exactly the confusion this crate exists to remove.
fn configured_root() -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os(ROOT_VAR)?);
    assert!(
        root.is_dir(),
        "{ROOT_VAR} is set to {}, which is not a directory",
        root.display()
    );
    Some(root)
}

/// The sibling `ferment-cuts-ios` fixture tree, if that checkout exists.
fn sibling_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(SIBLING_ROOT);
    root.is_dir().then_some(root)
}

/// The fixture root in force: configured wins, sibling is the fallback.
fn root() -> Option<PathBuf> {
    configured_root().or_else(sibling_root)
}

/// Name of the running test, for `SKIP` lines.
///
/// libtest names each test thread after the test itself. Under
/// `--test-threads=1` there is no such thread, hence the fallback.
fn test_name() -> String {
    std::thread::current()
        .name()
        .filter(|name| *name != "main")
        .unwrap_or("<test>")
        .to_owned()
}

/// Print the one line that distinguishes "did not run" from "passed".
///
/// Prefer [`path`] and the `read_*` helpers, which call this for you; reach for
/// it directly when a test's precondition is something other than a file (an
/// `ffmpeg` binary, a decoder, a model).
pub fn skip(reason: &str) {
    eprintln!("SKIP {}: {reason}", test_name());
}

/// What to do about a fixture, decided from the roots alone.
///
/// Split out from [`path`] so all three outcomes are unit-testable without
/// touching `FERMENT_ENGINE_FIXTURES_DIR` — a process-global that every other
/// test in this binary reads, and that none of them can safely change while
/// they run in parallel. The `Fail` arm is this crate's entire reason to exist,
/// so it needs a test rather than a comment: with the arm deleted, every one of
/// the ~700 tests in the workspace still passed.
#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Use(PathBuf),
    Skip(String),
    Fail(String),
}

/// One fixture tree: the variable that configures it, and what to tell
/// whoever is missing it.
struct Tree {
    var: &'static str,
    checkout: &'static str,
    hint: &'static str,
}

const IOS: Tree = Tree {
    var: ROOT_VAR,
    checkout: "ferment-cuts-ios",
    hint: "FermentEngine/Tests/Fixtures, or check out ferment-cuts-ios next to this repository",
};

fn decide(configured: Option<&Path>, sibling: Option<&Path>, relative: &str) -> Decision {
    decide_in(&IOS, configured, sibling, relative)
}

fn decide_in(
    tree: &Tree,
    configured: Option<&Path>,
    sibling: Option<&Path>,
    relative: &str,
) -> Decision {
    let var = tree.var;
    if let Some(root) = configured {
        // Configured wins outright — no fallback to the sibling. Falling back
        // would resolve against a different set of bytes than the operator
        // asked for, which is the confusion this crate exists to remove.
        let candidate = root.join(relative);
        return if candidate.exists() {
            Decision::Use(candidate)
        } else {
            Decision::Fail(format!(
                "{var} is set to {} but fixture {relative} is missing from it — \
                 the fixture tree is stale or the variable points at the wrong directory",
                root.display()
            ))
        };
    }

    if let Some(root) = sibling {
        let candidate = root.join(relative);
        if candidate.exists() {
            return Decision::Use(candidate);
        }
        // Nobody promised the sibling checkout was complete, so a gap there is
        // a skip rather than a failure.
        return Decision::Skip(format!(
            "fixture {relative} is missing from the sibling checkout at {} \
             (stale {}?); set {var} to a complete tree",
            root.display(),
            tree.checkout
        ));
    }

    Decision::Skip(format!(
        "fixture {relative} unavailable — set {var} to {}",
        tree.hint
    ))
}

/// Resolve `relative` under the fixture root.
///
/// `None` means the test should return early; the reason has already been
/// printed.
///
/// # Panics
/// If `FERMENT_ENGINE_FIXTURES_DIR` is set and the fixture is not under it.
#[must_use]
pub fn path(relative: &str) -> Option<PathBuf> {
    match decide(
        configured_root().as_deref(),
        sibling_root().as_deref(),
        relative,
    ) {
        Decision::Use(path) => Some(path),
        Decision::Skip(reason) => {
            skip(&reason);
            None
        }
        Decision::Fail(message) => panic!("{message}"),
    }
}

/// Resolve `relative`, failing the test if it is unavailable.
///
/// For the *second* and later fixtures of a test that has already decided to
/// run: once the first one resolves, a missing companion is a broken tree, not
/// a reason to skip.
///
/// # Panics
/// If the fixture cannot be resolved.
#[must_use]
pub fn require_path(relative: &str) -> PathBuf {
    let root = root().unwrap_or_else(|| {
        panic!("fixture {relative} is required here but no fixture root is available")
    });
    let candidate = root.join(relative);
    assert!(
        candidate.exists(),
        "required fixture {relative} is missing from {}",
        root.display()
    );
    candidate
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| {
        panic!(
            "fixture {} exists but did not read: {error}",
            path.display()
        )
    })
}

/// Raw bytes of an optional fixture.
#[must_use]
pub fn bytes(relative: &str) -> Option<Vec<u8>> {
    path(relative).as_deref().map(read)
}

/// Raw bytes of a fixture the test already committed to.
///
/// # Panics
/// If the fixture cannot be resolved or read.
#[must_use]
pub fn require_bytes(relative: &str) -> Vec<u8> {
    read(&require_path(relative))
}

fn as_f32_le(bytes: &[u8], label: &str) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "fixture {label} is {} bytes, not a whole number of little-endian f32",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Note the absence of a `/ 32768`: the audio-event fbank oracles follow
/// Kaldi, which windows raw int16 magnitudes rather than unit-scaled samples.
/// Normalising here would shift every mel bin and quietly break parity.
fn as_i16_le_as_f32(bytes: &[u8], label: &str) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(2),
        "fixture {label} is {} bytes, not a whole number of little-endian i16",
        bytes.len()
    );
    bytes
        .chunks_exact(2)
        .map(|chunk| f32::from(i16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

/// Little-endian `f32` samples from an optional fixture.
#[must_use]
pub fn f32_le(relative: &str) -> Option<Vec<f32>> {
    bytes(relative).map(|raw| as_f32_le(&raw, relative))
}

/// Little-endian `f32` samples from a fixture the test already committed to.
///
/// # Panics
/// If the fixture cannot be resolved or read.
#[must_use]
pub fn require_f32_le(relative: &str) -> Vec<f32> {
    as_f32_le(&require_bytes(relative), relative)
}

/// Little-endian `i16` samples from an optional fixture, at raw int16 magnitude.
#[must_use]
pub fn i16_le_as_f32(relative: &str) -> Option<Vec<f32>> {
    bytes(relative).map(|raw| as_i16_le_as_f32(&raw, relative))
}

/// The golden ladders and contracts `ferment-models-export` owns.
///
/// A model's goldens are an export product with two consumers — this engine
/// and the iOS one — so they are read where they are made, under
/// `models/<name>/`, with the same three outcomes as the iOS tree: resolves,
/// `SKIP`, or (a configured root that cannot serve it) a hard failure.
pub mod export {
    use std::path::{Path, PathBuf};

    use super::{Decision, Tree, as_f32_le, decide_in, read, skip};

    /// Set to `ferment-models-export/models`.
    pub const ROOT_VAR: &str = "FERMENT_MODELS_EXPORT_DIR";

    const SIBLING_ROOT: &str = "../../../ferment-models-export/models";

    const TREE: Tree = Tree {
        var: ROOT_VAR,
        checkout: "ferment-models-export",
        hint: "ferment-models-export/models, or check out ferment-models-export next to this repository",
    };

    /// Raw bytes from an export fixture already required by this test.
    /// Panics if the tree or companion file is missing.
    #[must_use]
    pub fn require_bytes(relative: &str) -> Vec<u8> {
        read(&require_path(relative))
    }

    fn configured_root() -> Option<PathBuf> {
        let root = PathBuf::from(std::env::var_os(ROOT_VAR)?);
        assert!(
            root.is_dir(),
            "{ROOT_VAR} is set to {}, which is not a directory",
            root.display()
        );
        Some(root)
    }

    fn sibling_root() -> Option<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(SIBLING_ROOT);
        root.is_dir().then_some(root)
    }

    /// Resolve `relative` (`<model>/<file>`) under the export tree. `None`
    /// means the test should return early; the reason has been printed.
    ///
    /// # Panics
    /// If `FERMENT_MODELS_EXPORT_DIR` is set and the file is not under it.
    #[must_use]
    pub fn path(relative: &str) -> Option<PathBuf> {
        match decide_in(
            &TREE,
            configured_root().as_deref(),
            sibling_root().as_deref(),
            relative,
        ) {
            Decision::Use(path) => Some(path),
            Decision::Skip(reason) => {
                skip(&reason);
                None
            }
            Decision::Fail(message) => panic!("{message}"),
        }
    }

    /// A companion of a file that already resolved: missing now is a broken
    /// tree, not a reason to skip.
    ///
    /// # Panics
    /// If the file cannot be resolved.
    #[must_use]
    pub fn require_path(relative: &str) -> PathBuf {
        let root = configured_root().or_else(sibling_root).unwrap_or_else(|| {
            panic!("{relative} is required here but no export tree is available")
        });
        let candidate = root.join(relative);
        assert!(
            candidate.exists(),
            "required {relative} is missing from {}",
            root.display()
        );
        candidate
    }

    /// Set by `just engine-oracle`: a flat directory of the published objects
    /// under their own names (`<name>.onnx`, `<name>.mlmodelc`, a resource).
    pub const MODELS_VAR: &str = "FERMENT_ENGINE_MODELS_DIR";

    /// A lowered artifact (`<name>.onnx`, `<name>.mlmodelc`) by its published
    /// name: from `FERMENT_ENGINE_MODELS_DIR` when the oracle names one — where
    /// it must be — else from the export repository's `artifacts/`, which is
    /// emptied after every upload, so absence there is a `SKIP`.
    ///
    /// # Panics
    /// If `FERMENT_ENGINE_MODELS_DIR` is set and the artifact is not in it.
    #[must_use]
    pub fn artifact(file_name: &str) -> Option<PathBuf> {
        if let Some(models) = std::env::var_os(MODELS_VAR).map(PathBuf::from) {
            let candidate = models.join(file_name);
            assert!(
                candidate.exists(),
                "{MODELS_VAR} is set to {} but {file_name} is missing from it",
                models.display()
            );
            return Some(candidate);
        }
        let candidate = configured_root()
            .or_else(sibling_root)?
            .parent()?
            .join("artifacts")
            .join(file_name);
        if candidate.exists() {
            return Some(candidate);
        }
        skip(&format!(
            "{file_name} is neither in {MODELS_VAR} nor in ferment-models-export/artifacts"
        ));
        None
    }

    /// Little-endian `f32` values of an optional golden.
    #[must_use]
    pub fn f32_le(relative: &str) -> Option<Vec<f32>> {
        path(relative).map(|path| as_f32_le(&read(&path), relative))
    }

    /// Little-endian `f32` values of a golden the test already committed to.
    ///
    /// # Panics
    /// If the golden cannot be resolved or read.
    #[must_use]
    pub fn require_f32_le(relative: &str) -> Vec<f32> {
        as_f32_le(&read(&require_path(relative)), relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entries every consumer of the fixture tree depends on. Checked by the
    /// canary so a root pointing at the wrong directory fails once, with a useful
    /// message, instead of thirty times with confusing ones.
    const CANARY_ENTRIES: &[&str] = &[
        "Audio/techno-1.wav",
        "AudioEventDetector/hairdryer/audio_int16.bin",
        "AudioEventDetector/sec5/probs.bin",
        "BeatThis",
        "BeatThis/dbn-parity-expected.json",
        "BeatThis/dbn-parity-logits.json",
        "MusicHighlighter/test_audio.bin",
        "MusicHighlighter/test_pos_enc.bin",
        "MusicHighlighter/test_saliency.bin",
        "Pyannote/one-speaker/reference_probabilities.bin",
        "RegressionTests/VideoSceneDetector/strobe-cuts.mp4",
        // RedimNet's current embeddings are exporter goldens. The removed
        // Swift speaker-encoder embedding belonged to the retired model;
        // keep checking its real input corpus, not that obsolete output.
        "SpeakerEncoder/speaker-b-1.wav",
        "SpeakerEncoder/speaker-a-1.wav",
        "SpeechDenoising/trace_audio_padded.bin",
        "SpeechDenoising/trace_stft.bin",
        "Video/cloudy_9x16.mp4",
        "Video/dhh-interview.mp4",
        "Video/djs-walking-city.mp4",
        "VideoSceneDetector/test_frames.bin",
        "pipeline_test_audio.bin",
        "techno-1-30s.mp3",
        "test_audio.bin",
    ];

    /// A scratch tree holding exactly `entries`, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn with(label: &str, entries: &[&str]) -> Self {
            let root = std::env::temp_dir().join(format!(
                "ferment-fixture-decide-{}-{label}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).expect("scratch root");
            for entry in entries {
                let path = root.join(entry);
                std::fs::create_dir_all(path.parent().expect("entry has a parent"))
                    .expect("scratch parent");
                std::fs::write(&path, [0_u8; 4]).expect("scratch fixture");
            }
            Self(root)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// THE contract. Deleting the `Fail` arm left all ~700 workspace tests
    /// green, which made this crate's central guarantee a comment.
    #[test]
    fn a_configured_root_that_cannot_serve_a_fixture_is_a_hard_failure() {
        let configured = Scratch::with("configured-empty", &[]);
        match decide(Some(&configured.0), None, "Audio/techno-1.wav") {
            Decision::Fail(message) => {
                assert!(message.contains(ROOT_VAR), "{message}");
                assert!(message.contains("Audio/techno-1.wav"), "{message}");
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    /// A configured root must never fall back to the sibling checkout — that
    /// would silently resolve against different bytes than the operator asked
    /// for, and the run would look green while checking the wrong thing.
    #[test]
    fn a_configured_root_never_falls_back_to_the_sibling() {
        let configured = Scratch::with("configured-bare", &[]);
        let sibling = Scratch::with("sibling-full", &["Audio/techno-1.wav"]);
        assert!(matches!(
            decide(Some(&configured.0), Some(&sibling.0), "Audio/techno-1.wav"),
            Decision::Fail(_)
        ));
    }

    #[test]
    fn a_configured_root_holding_the_fixture_resolves_to_it() {
        let configured = Scratch::with("configured-full", &["Audio/techno-1.wav"]);
        assert_eq!(
            decide(Some(&configured.0), None, "Audio/techno-1.wav"),
            Decision::Use(configured.0.join("Audio/techno-1.wav"))
        );
    }

    #[test]
    fn a_sibling_checkout_missing_a_fixture_skips_rather_than_fails() {
        let sibling = Scratch::with("sibling-bare", &[]);
        assert!(matches!(
            decide(None, Some(&sibling.0), "Audio/techno-1.wav"),
            Decision::Skip(_)
        ));
    }

    #[test]
    fn no_roots_at_all_skips_and_says_how_to_fix_it() {
        match decide(None, None, "Audio/techno-1.wav") {
            Decision::Skip(reason) => assert!(reason.contains(ROOT_VAR), "{reason}"),
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    /// The canary. A configured fixture root that cannot serve the tree is a
    /// broken gate, and a broken gate has to fail rather than let ~30 parity
    /// tests skip their way to green.
    #[test]
    fn configured_root_serves_the_whole_fixture_tree() {
        let Some(root) = configured_root() else {
            skip(&format!(
                "{ROOT_VAR} is not set, so there is no configured root to verify"
            ));
            return;
        };
        let missing: Vec<&str> = CANARY_ENTRIES
            .iter()
            .copied()
            .filter(|entry| !root.join(entry).exists())
            .collect();
        assert!(
            missing.is_empty(),
            "{ROOT_VAR} points at {}, which is missing {} of {} expected fixtures: {missing:?}",
            root.display(),
            missing.len(),
            CANARY_ENTRIES.len()
        );
    }

    #[test]
    fn sibling_checkout_when_present_serves_the_whole_fixture_tree() {
        let Some(root) = sibling_root() else {
            skip("no sibling ferment-cuts-ios checkout to verify");
            return;
        };
        let missing: Vec<&str> = CANARY_ENTRIES
            .iter()
            .copied()
            .filter(|entry| !root.join(entry).exists())
            .collect();
        assert!(
            missing.is_empty(),
            "sibling checkout at {} is stale — missing {missing:?}",
            root.display()
        );
    }

    #[test]
    fn f32_decoding_is_little_endian() {
        let raw = 1.5f32.to_le_bytes().to_vec();
        assert_eq!(as_f32_le(&raw, "unit"), vec![1.5]);
    }

    /// Kaldi magnitude, not unit scale — see `as_i16_le_as_f32`.
    #[test]
    fn i16_decoding_preserves_raw_magnitude() {
        let raw = [i16::MIN.to_le_bytes(), 1234i16.to_le_bytes()].concat();
        assert_eq!(as_i16_le_as_f32(&raw, "unit"), vec![-32_768.0, 1234.0]);
    }

    #[test]
    #[should_panic(expected = "not a whole number of little-endian f32")]
    fn ragged_f32_fixture_is_rejected() {
        as_f32_le(&[0, 1, 2], "unit");
    }
}
