# Common dev tasks. Run `just` to list, `just <target>` to invoke.
# Requires: https://github.com/casey/just

set shell := ["bash", "-uc"]

# default — list available recipes
default:
    @just --list --unsorted

# `ort` deadlocks — it does not fail — when its dynamic library cannot be
# loaded, so the ORT tests need `ORT_DYLIB_PATH` naming a file before they run.
#
# clippy + tests + fmt + license gate — what CI will run
check:
    @test -f "${ORT_DYLIB_PATH:?set ORT_DYLIB_PATH to the ONNX Runtime library (just provision-ort-mac)}" || { echo "ORT_DYLIB_PATH=$ORT_DYLIB_PATH is not a file"; exit 1; }
    cargo clippy --all-targets -- -D warnings
    cargo nextest run --workspace --exclude ferment-engine-bench
    cargo fmt -- --check
    cargo deny check licenses

# The whole Rust suite, every test in its own process, all binaries at once.
# `cargo test` runs one test BINARY at a time; nextest runs them in parallel and
# is ~40x here. Doctests are not run at all: every crate is `publish = false`,
# so there is no public API for an example to keep honest. `--ignored` benches
# still use plain `cargo test`.
test:
    cargo nextest run --workspace --exclude ferment-engine-bench

# Delete the build artifacts that make macOS media tests crawl.
#
# CoreMedia scans the running binary's own directory once per process when a
# VideoToolbox compression session is created (`FigRegistryAddSearchPathWithOptions`
# -> readdir). `target/debug/deps` had 631 861 entries on 2026-09-02 and that
# scan cost 18.7 s PER TEST BINARY — a 12-frame encode test took 18.7 s where
# the same binary run from a clean directory took 0.17 s. Cargo rebuilds what
# it needs; nothing here is source.
clean-stale:
    #!/usr/bin/env bash
    set -euo pipefail
    count=$(ls target/debug/deps 2>/dev/null | wc -l | tr -d ' ')
    echo "target/debug/deps holds $count entries"
    if [[ "$count" -lt 20000 ]]; then echo "nothing to do"; exit 0; fi
    mkdir -p /tmp/ferment-trash
    mv target/debug/deps target/debug/.fingerprint target/debug/incremental /tmp/ferment-trash/ 2>/dev/null || true
    echo "moved aside; deleting in the background — cargo can build again now"
    (rm -rf /tmp/ferment-trash &)

# Cross-compile the portable engine for Android arm64 (the E0 gate).
#
# Needs the project NDK on the path — `source
# ../ferment-cuts-android/scripts/env` — and, before the release build,
# `scripts/provision-ffmpeg aarch64-linux-android --build`: `cargo ndk check`
# is satisfied by the HOST FFmpeg headers under the NDK sysroot, but the link
# is not, and it is the link that says whether the engine runs on a phone.
#
# `-P 33` because cargo-ndk defaults to NDK platform 21, below this project's
# minSdk. `--no-default-features --features ort` where a crate defaults to
# `coreml`, which does not exist off Apple.
android-check:
    cargo ndk -t arm64-v8a -P 33 clippy --all-targets -p ferment-engine-core -p ferment-engine-audio -p ferment-engine-models -p ferment-engine-media -- -D warnings
    cargo ndk -t arm64-v8a -P 33 check --all-targets -p ferment-engine-infer --no-default-features --features ort
    cargo ndk -t arm64-v8a -P 33 check --all-targets -p ferment-engine-processors --no-default-features --features ort
    FFMPEG_DIR="${FFMPEG_DIR:-$PWD/vendor/ffmpeg/aarch64-linux-android}" cargo ndk -t arm64-v8a -P 33 build --release --locked -p ferment-engine-uniffi
    # The device suite has to LINK, which `check` never proves: engine-media's
    # Android module names libmediandk and libandroid, and a missing symbol
    # there is invisible until a phone tries to load it.
    FFMPEG_DIR="${FFMPEG_DIR:-$PWD/vendor/ffmpeg/aarch64-linux-android}" cargo ndk -t arm64-v8a -P 33 test --release --locked --no-run -p ferment-engine-media

# Run the engine-media device suites on a connected phone, or on cuts-36.
#
# Takes the machine-wide device lock, pushes the test binaries, the six
# libav*.so they link and the fixture corpus, runs both suites and prints the
# count read from the log file. Needs the Android checkout beside this one:
#   PERCEPTION_REPO=$PWD just android-device-test
android-device-test *args='':
    ../ferment-cuts-android/native/perception-engine/device-test.sh {{args}}

# Format the Rust workspace
fmt:
    cargo fmt

# Fetch the ONNX Runtime this repo's ort backend loads (arm64, CoreML EP) into
# the vendor path. Pins onnxruntime 1.24.x to match `ort 2.0.0-rc.12`
# (api-24). Overrides: ORT_DYLIB_PATH copies your own library instead;
# ORT_VERSION pins a different onnxruntime release.
provision-ort-mac:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="${ORT_VERSION:-1.24.4}"
    DEST="vendor/onnxruntime/aarch64-apple-darwin"
    OUT="$DEST/libonnxruntime.dylib"
    mkdir -p "$DEST"
    if [[ -n "${ORT_DYLIB_PATH:-}" ]]; then
      echo "provision-ort-mac: copying \$ORT_DYLIB_PATH ($ORT_DYLIB_PATH)"
      cp -f "$ORT_DYLIB_PATH" "$OUT"
    else
      echo "provision-ort-mac: fetching onnxruntime==$VERSION (host arm64 wheel, CoreML EP built in)"
      TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
      python3 -m pip download "onnxruntime==$VERSION" --no-deps -d "$TMP"
      WHEEL="$(ls "$TMP"/onnxruntime-*.whl | head -1)"
      unzip -o -q "$WHEEL" -d "$TMP/x"
      DYLIB="$(ls "$TMP"/x/onnxruntime/capi/libonnxruntime.*.dylib | head -1)"
      cp -f "$DYLIB" "$OUT"
    fi
    ARCH="$(lipo -archs "$OUT" 2>/dev/null || echo unknown)"
    # Apple-Silicon only (031 Part D): the bundled dylib must be arm64, not fat.
    if [[ "$ARCH" != "arm64" ]]; then
      echo "ERROR: bundled dylib must be arm64-only, got '$ARCH'"; exit 1
    fi
    echo "provision-ort-mac: $OUT ($ARCH, $(du -h "$OUT" | cut -f1))"

# Fetch the ONNX Runtime + DirectML DLLs for the Windows x64 target into the
# vendor path. Runs on the Windows box. Pins onnxruntime-directml 1.24.x
# (api-24); ORT_VERSION overrides.
#
# Invoke the .ps1 directly — do NOT make this a `#!/usr/bin/env pwsh` shebang
# recipe: just writes shebang bodies to an EXTENSIONLESS temp file, and
# PowerShell's -File rejects anything not ending in .ps1 ("failed because the
# file does not have a '.ps1' extension").
#
# Use `pwsh` (7), NOT `powershell` (5.1): CI already runs the step under pwsh 7,
# and spawning 5.1 from it leaks PowerShell 7's PSModulePath, so 5.1 can't load
# its own Microsoft.PowerShell.Security ("Get-ExecutionPolicy ... could not be
# loaded") and the uv bootstrap dies.
provision-ort-win:
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/exports/provision_ort_win.ps1

# Fetch the FFmpeg libav* shared libraries `ferment-engine-media` links against
# into the vendor path, and point `vendor/ffmpeg/host` at this machine's
# target. Run once before any cargo command — `.cargo/config.toml` sets
# FFMPEG_DIR to that path, so a missing vendor tree fails the build.
#
# LGPL-2.1-or-later, built --disable-gpl --disable-nonfree, linked dynamically.
# `--build` rebuilds the artifact from the pinned FFmpeg source instead of
# downloading it; that is how the published artifacts are made.
provision-ffmpeg-mac:
    scripts/provision-ffmpeg aarch64-apple-darwin

provision-ffmpeg-win:
    scripts/provision-ffmpeg x86_64-pc-windows-msvc

provision-ffmpeg-linux:
    scripts/provision-ffmpeg x86_64-unknown-linux-gnu

# Android arm64. `--build` only: no artifact is published to the CDN yet, so
# there is nothing to download. Cross-built natively with the project's NDK
# (no docker), which `ANDROID_NDK_HOME` has to name.
provision-ffmpeg-android:
    scripts/provision-ffmpeg aarch64-linux-android --build

# Rebuild every publishable FFmpeg artifact from source and stage it with its
# sha256 under dist/ffmpeg-publish/. macOS builds natively; Windows and Linux
# cross-build in a container. `scripts/ffmpeg-publish` uploads the result.
provision-ffmpeg-build:
    scripts/provision-ffmpeg --build --package aarch64-apple-darwin
    scripts/provision-ffmpeg --build --package x86_64-pc-windows-msvc
    scripts/provision-ffmpeg --build --package x86_64-unknown-linux-gnu

# Full Engine oracle. Large models/fixtures remain external by policy.
engine-oracle:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${FERMENT_ENGINE_MODELS_DIR:?set FERMENT_ENGINE_MODELS_DIR to the CoreML/model fleet}"
    : "${FERMENT_ENGINE_FIXTURES_DIR:?set FERMENT_ENGINE_FIXTURES_DIR to FermentEngine/Tests/Fixtures}"
    : "${ORT_DYLIB_PATH:?set ORT_DYLIB_PATH to the ONNX Runtime dynamic library}"
    export FERMENT_ENGINE_ONNX_DIR="${FERMENT_ENGINE_ONNX_DIR:-$FERMENT_ENGINE_MODELS_DIR}"
    export FERMENT_ENGINE_MODELS_DIR="$(cd "$FERMENT_ENGINE_MODELS_DIR" && pwd -P)"
    export FERMENT_ENGINE_ONNX_DIR="$(cd "$FERMENT_ENGINE_ONNX_DIR" && pwd -P)"
    export FERMENT_ENGINE_FIXTURES_DIR="$(cd "$FERMENT_ENGINE_FIXTURES_DIR" && pwd -P)"
    export ORT_DYLIB_PATH="$(cd "$(dirname "$ORT_DYLIB_PATH")" && pwd -P)/$(basename "$ORT_DYLIB_PATH")"
    # The fixture canary runs first: a mis-set FERMENT_ENGINE_FIXTURES_DIR should
    # fail here once, not thirty times further down.
    cargo test -p ferment-engine-test-fixtures
    cargo test -p ferment-engine-core -p ferment-engine-audio -p ferment-engine-models
    cargo test -p ferment-engine-infer --features ort
    cargo test -p ferment-engine-processors --features ort
    cargo test -p ferment-engine-processors --test media_audio_source --test current_beat_holdout -- --ignored
    # Every family against its contract and the goldens ferment-models-export
    # owns: the lowered halves — ORT, and on a Mac Core ML on the lanes the
    # fleet pins — on the published objects in FERMENT_ENGINE_MODELS_DIR.
    cargo test -p ferment-engine-processors --features ort \
      --test ambernet_contract --test autoshot_contract --test beat_this_contract \
      --test dasheng_contract --test dfnet_contract --test dinov2_contract \
      --test edgetam_contract --test fireredvad_aed_contract --test gigaam_contract \
      --test music_highlighter_contract --test music_tagger_contract \
      --test parakeet_contract --test pyannote_contract --test redimnet_contract \
      -- --ignored
    if [[ "$(uname -s)" == Darwin ]]; then
      cargo test -p ferment-engine-infer --test coreml_fleet -- --ignored
      cargo test -p ferment-engine-processors --test coreml_processors -- --ignored --skip test_twenty_minute_synthetic_audio_completes_within_memory_budget
    fi
    # Scene detection end to end: decode -> AutoShot -> jump cuts -> flash veto
    # -> per-scene embeddings, against the upstream strobe-cuts.mp4 ground truth.
    cargo test -p ferment-engine-processors --features ort --test scene_detection_regression -- --ignored
    # The enhancement backings on the export's natural-frame goldens.
    cargo test -p ferment-engine-processors --features ort --test ort_enhance -- --ignored
    # Video-saliency backings: tensor parity against the export's goldens plus
    # the end-to-end measurement over the committed synthetic clip. The three
    # iOS-fixture cases inside print SKIP when the fixture tree is absent.
    cargo test -p ferment-engine-processors --features ort --test ort_saliency -- --ignored
    # Diarization behaviour on the iOS speaker fixtures, both published masters.
    cargo test -p ferment-engine-processors --features ort --test ort_diarization -- --ignored
    # The streaming recogniser and the object segmenter's image and video
    # sessions, on their published masters.
    cargo test -p ferment-engine-processors --features ort --test ort_live_speech -- --ignored
    cargo test -p ferment-engine-processors --features ort --test ort_segmentation -- --ignored
    cargo test -p ferment-engine-processors --features ort --test ort_video_session -- --ignored
    # Cross-EP behavioural agreement. Degenerate (CPU vs CPU) unless
    # FERMENT_ENGINE_ORT_EP names a GPU provider; the suite says so out loud.
    cargo test -p ferment-engine-processors --features ort --test ort_behavioral -- --ignored
    # The CUDA leg runs offline from fleet_ep_probe.py dumps -- no ORT, any platform.
    if [[ -n "${FERMENT_EP_DUMPS:-}" ]]; then
      cargo test -p ferment-engine-processors --test behavioral_from_dumps -- --ignored
    else
      echo "SKIP behavioral_from_dumps: set FERMENT_EP_DUMPS to a fleet_ep_probe.py dump directory"
    fi

# Pinned 60-second DSP workloads. Pass a Criterion filter after `--`.
engine-bench *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    features=()
    if [[ "$(uname -s)" == Darwin ]]; then
      features=(--features accel-fastpath)
    fi
    cargo bench -p ferment-engine-bench --bench dsp "${features[@]}" -- {{args}}

# Compile the benchmark harness without running measurements.
engine-bench-check:
    cargo bench -p ferment-engine-bench --bench dsp --no-run

# Relative wasm SIMD sanity numbers under Node (not comparable to native).
engine-bench-wasm:
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1 -p ferment-engine-bench --bin wasm-sanity
    node crates/engine-bench/wasm/run.mjs target/wasm32-wasip1/release/wasm-sanity.wasm
