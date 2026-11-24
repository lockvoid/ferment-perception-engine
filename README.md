# Ferment Perception Engine

A Rust library for turning audio and video into data an editor can use. An application can draw a waveform, snap a cut to a beat, split footage at a scene change, or find audio and images that match a text query. The engine handles media decoding, signal processing and model inference. The application decides what to do with the results.

## Find the right crate

| If you want to… | Start here |
| --- | --- |
| Decode media, read audio frames or resample a signal | [Media](crates/engine-media) and [audio](crates/engine-audio) |
| Detect beats or scenes, transcribe speech, or generate a waveform | [Processors](crates/engine-processors/src/lib.rs) |
| Compare a text query with audio or images | [Retrieval](crates/engine-retrieval/src/lib.rs) pairs GLAP audio and text embeddings, and SigLIP 2 image and text embeddings. |
| Load a verified model or select an inference backend | [Model storage](crates/engine-models) and [inference](crates/engine-infer) |
| Call the engine from Kotlin on Android | [UniFFI bindings](crates/engine-uniffi) |

## Run a first test

From the repository root, use the [pinned Rust toolchain](rust-toolchain.toml) to run a small audio test. It generates a signal, writes a PCM WAV, reads it back, and checks the decoded samples. No model weights or media downloads are needed.

```sh
cargo test --locked -p ferment-engine-audio --lib wav::tests::pcm16_round_trips_through_loader
```

For the full processor library on Apple Silicon, install the Xcode command line tools and `just`, then provision FFmpeg and build with the default Core ML backend.

```sh
just provision-ffmpeg-mac
cargo build --locked -p ferment-engine-processors
```

FFmpeg is linked dynamically. The provisioning recipe sets up `vendor/ffmpeg/host`, the path used by Cargo. For an ONNX Runtime build, provision FFmpeg for the host platform and select the `ort` feature. Running ONNX inference also requires a compatible ONNX Runtime library.

```sh
cargo build --locked -p ferment-engine-processors --no-default-features --features ort
```

## Backends and model assets

Core ML is the default for processors on Apple platforms. The inference and retrieval crates also expose Core AI on macOS for models that supply a compatible compiled artifact. Windows uses ONNX Runtime with Windows ML for selected models. Android uses ONNX Runtime through the Kotlin bindings. The model and platform determine which providers can run a graph. [Windows ML qualification](docs/provenance/2026-11-16_windows-ml_report.md) · [Android build and integration notes](docs/provenance/2026-10-06_engine-consolidation.md)

Model manifests describe the required artifacts and verify their hashes. The model weights are distributed separately from this source tree. For native Android builds, use `just provision-ffmpeg-android` and `just android-check` with the Android NDK and `cargo-ndk`. The [Justfile](Justfile) also lists the Windows and Linux FFmpeg recipes and the benchmark commands.

## The oracle

For the routine workspace checks on Apple Silicon, install `cargo-nextest` and `cargo-deny`. The ORT provisioning recipe needs a compatible arm64 Python wheel (CPython 3.11–3.14 on macOS 14+ for the pinned version) or an existing runtime library.

```sh
just provision-ort-mac
ORT_DYLIB_PATH="$PWD/vendor/onnxruntime/aarch64-apple-darwin/libonnxruntime.dylib" just check
```

For full comparisons with the reference implementations, run `just engine-oracle` with `FERMENT_ENGINE_MODELS_DIR` pointing to the verified model objects, `FERMENT_ENGINE_FIXTURES_DIR` to the reference media, `FERMENT_MODELS_EXPORT_DIR` to the model contracts and reference tensors, and `ORT_DYLIB_PATH` to the runtime library. Supply the external oracle inputs separately. The test output reports why an optional fixture or provider was skipped.

The Ballroom holdout reads `Ballroom/Albums-Latin_Jam5-01.beats` from the reference media directory. The GigaAM speech oracle reads `Speech/speech-ru.m4a` from the same directory. [Ballroom provenance](crates/engine-processors/tests/fixtures/ballroom/PROVENANCE.md)

## Licensing

Perception's original source code is available under the [MIT License](LICENSE). Third-party materials and separately distributed model weights have their own terms. [Third-party notices](THIRD_PARTY_NOTICES.md)
