# Shared perception engine consolidation

Scope 1 locally qualified, 2026-10-07 (test runs on October 6 UTC). This report
distinguishes recovered source from newly executed acceptance. New GLAP/SigLIP 2
artifacts are ready for the user's CDN handoff; no upload was performed.
Scope 2 application/index integration and scope 3 review remain separate work.

## Source receipts

- Main Rust baseline: `f380afa490c7bf1144b4233b5ea22ddb1782e8ee`.
- Preserved pre-existing model-contract migration: `f485afb`.
- Android source archive: `ferment-cuts-android/native/source-snapshots/20260927/perception.tar.gz`,
  17,643,563 bytes, SHA-256
  `f92c989cbcd961649148b5675c312d98ce62c40caffb2225dc7a9f3433e0f6ca`.
- Its raw binary-safe `head-to-worktree.patch`: 21,841,686 bytes, SHA-256
  `47c4f74d621973dda537ce3485212688eb54fa4ec6541f90ff26a42f391d6dff`.
- Reconstructed on a detached worktree at the recorded base, after checking
  archive paths and refusing links. **9,293 file digests and 43 expected
  deletions matched**, no mismatches. Source checkpoint `958944d`, branch
  `archive/android-perception-20260927`; caches excluded.
- Android consumer inspected at `d265d9d9fb15d63c59ec277d4b5e5d19817b119f`,
  with existing uncommitted Kotlin/editor changes left intact.
- Swift behavioral reference: iOS checkout `0e68fab7ce3a42272db49e63a510f7c3d5dc6bc5`;
  last engine commit `ac30d07d`. Its unrelated dirty working tree is not changed.

The snapshot and main baseline share an ancestor. The three-way merge was
reviewed against `f485afb`, not applied as a directory overwrite. Before host
adaptations, the resolved tree matched the recovered snapshot: the main
checkpoint's model-contract changes were already contained in that snapshot.

## Port map

| Plane | Main checkpoint | Recovered Android source | Consolidation action |
| --- | --- | --- | --- |
| Model contracts/pins | September 23 fleet, host mel, current vocabularies | Same contracts plus stronger assertions | Preserve; do not restore retired graphs or old tensor names |
| Audio frontend | Host `MelSpec`, buffered SRC | Current FIR/polyphase streaming SRC, channel policy, shared derived PCM/mel cache, bounded file sources | Recover common Rust implementation and its tests |
| ORT lifecycle | Registry mutex spans load/infer | Shared asynchronous loads, detachable waiters, per-instance prediction permits, explicit invalidation/eviction | Recover; test cancellation, late completion, replacement, concurrency |
| Core ML lifecycle | Native Core ML binding, current units and multifunction doors | Unchanged by Android port | Audit separately; ORT acceptance does not prove Core ML lifecycle parity |
| Processors | Current model doors, several buffered paths | Bounded denoise/ASR/minimap/scene paths, cleanup, cancellation, EdgeTAM time/ownership fixes | Recover without UI or document dependencies |
| Media | libav + platform adapters | Original-depth HDR handling, orientation, recovery/timing, audio remux | Recover; keep platform-specific code gated and run host/device checks separately |
| Android integration | New Kotlin bindings/binary already present | Source formerly selected through a now-missing temporary checkout | Build from the common sibling, record new source/binary/binding identity |
| New retrieval models | No GLAP or SigLIP 2 pair | No pair | Implement in `ferment-models-export`, then narrow Rust runtime APIs |
| Core AI | Absent in Rust | Absent in Rust | Intentional; desktop macOS uses Core ML, portable targets ORT |

The pre-existing Android binary exposes the new frontend/cache APIs; this is
not proof that it was built byte-for-byte from the archive. The fresh native
build and binding comparison are recorded in the final qualification below.

## Host integration corrections

- Preserve the executable bit of `scripts/provision-ffmpeg`; the source archive
  had accidentally cleared it.
- Gate non-macOS YUV encoder helpers and Android tone mapping at their actual
  platform boundary while compiling their pure tests on the host.
- Apply workspace formatting and mechanical strict-clippy fixes. Remove the
  unused whole-movie RGB buffer wrapper and obsolete `fit_within` helper/test;
  production streaming/geometry paths and their tests remain.
- Share a private file-render context at completion boundaries, rather than
  extending positional argument lists. Scope a test mutex guard before await.
- Keep the existing stricter Pyannote probability (0.04) and decisive-margin
  (0.02) gates alongside the incoming Swift-derived decision-count gate.
- Remove the fixture-tree canary's dependency on the retired Swift speaker
  encoder output. Current RedimNet outputs are exporter goldens; real speaker
  WAV inputs remain required. Delayed AAC tests accept an explicit supplement
  root because iOS keeps app media fixtures outside its engine fixture tree.

## Recovered baseline verification

- All-target/all-feature workspace clippy with `-D warnings`: **passed**.
- Full all-feature workspace nextest excluding the bench crate: **1,553
  reported passed, 322 skipped** (15.962 s execution). This is the default
  workspace gate, not execution of the ignored model-oracle/device suites.
- `cargo fmt --all -- --check` and `cargo deny check licenses`: **passed**.
- The initial run had 36 failures: exporter root mistakenly omitted `/models`,
  mandatory media fixture roots were unset, and the canary named the retired
  embedding. No numerical tolerance was relaxed to fix those failures.
- Reproduction environment: `FERMENT_MODELS_EXPORT_DIR=../ferment-models-export/models`,
  `FERMENT_ENGINE_FIXTURES_DIR=../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures`,
  `FERMENT_ENGINE_SUPPLEMENT_FIXTURES_DIR=../ferment-cuts-ios/Tests/Fixtures`,
  and `ORT_DYLIB_PATH` points to this checkout's provisioned ORT dylib.
  The admitted AAC fixture matches its expected SHA-256
  `01b3d290e63ddcbb8b3f69817509939a21f1c9b2bfe32040a8501f603705077f`.
  Full output: `target/consolidation-nextest.log` (local, not committed).
- This baseline run preceded the new retrieval models. Their later acceptance
  is recorded below; native Android acceptance remains a separate gate.

## Storage and provenance rules

Production model weights, downloaded runtimes and Python/build caches stay
untracked. `vendor/coreml-native` is a small source-only patch of the licensed
binding (plus its tiny linear-model test fixtures), not a production-model or
runtime distribution. Its compute-unit correction was part of `f485afb`.

The recovered archive and both checkpoints remain available in Git. No user
project, media library, original iOS implementation or prior Android dirty
change is deleted by this consolidation.

## Core ML lifecycle follow-up

The recovered implementation is committed as `0dfd459`. Core ML now uses the
same bounded-wait, shared-load cache as ORT, including physical bundle identity,
explicit invalidation and eviction. Repinning compute units invalidates the old
specialization. A caller's cancellation cannot clear a native load still in
progress or let its late completion overwrite a replacement.

Each loaded Core ML instance owns one bounded prediction worker and FIFO
admission queue. Native prediction does not occupy the calling async executor.
Dropping a reply future leaves the worker owning its model, input buffers and
permit until the native call finishes; the next inference cannot race it.
FP16 inputs retain their actual bit patterns at the native boundary.

Available RAM on macOS/Windows is sampled through `sysinfo`'s safe RAM-only API,
at most once per 250 ms across consumers. The existing 768 MiB pressure floor
and unknown-memory handling remain; Linux/Android retain `/proc/meminfo`.
No additional unsafe code is introduced into engine-infer.

Verification: engine-infer all-feature strict clippy passed; all-feature nextest
reported **75 passed, 12 skipped** (0.730 s execution), including real native
linear-model load/prediction, eight concurrent consumers, cache eviction/reload,
placement invalidation, half-input representation and cancellation ownership.
Production-model parity and target-device acceptance are separate gates.

## Paired retrieval runtime

Exporter checkpoint `45c1047` adds GLAP and SigLIP 2 Base/16 224. Engine
checkpoint `082e17a` adds `ferment-engine-retrieval`: exact shared mel and RGB
frontends, verified reusable tokenizers, unit-vector/space validation, bounded
audio windows, cancellation and sink backpressure. No UI, database or project
dependency enters this crate. See the exporter's
[contracts and CDN handoff](../../../ferment-models-export/docs/retrieval-models.md).

The model registry now has one canonical `ModelId::ALL` list (36 entries).
Four new graphs and two tokenizer resources have the actual exported pins.
Existing artifact/version pins are unchanged. The additions are lazy, on CPU;
Android retains its existing provider rules for old families and explicitly
does not try XNNPACK first for the four new, CPU-qualified graphs.

Real public Rust API acceptance, `target/retrieval-complete.log`:

- **13 passed, 0 failed, 0 ignored**, including both production-backend oracles.
- Four GLAP audio inputs, six text inputs, seven SigLIP images and thirteen
  text inputs run through original PCM/RGB/text, not just precomputed tensors.
- Pixel-for-pixel Pillow frontend parity, tokenizer IDs including Unicode and
  truncation, unit-vector numerical parity and EN/RU retrieval rankings pass.
- Three real video frames are timestamped/pinned in exporter goldens. This is
  a small regression corpus, not a broad semantic-search quality benchmark.
- Same-input warm repeats and actual cache eviction/reload pass on both
  production backends. Separate cache tests cover eight concurrent consumers,
  replaced/missing models and load cancellation; a memory-pressure regression
  prevents evicting an active sibling and creating a duplicate native instance.
- A ten-minute synthetic sample source yields 61 bounded windows including an
  exact 17-sample tail through a controlled backend. Sink failure and cancellation
  stop further delivery; an in-flight pending inference detaches safely. This
  proves streaming ownership/bounds, not ten minutes of production-model speed.

### Same-input latency snapshot

Machine: Mac15,9, arm64, 48 GiB RAM. Cargo optimized test profile, ORT CPU
1.24.4 and Core ML `CPU_ONLY`. Times include the Rust frontend, measured at the
public encoder API. Cold = a new backend session; OS/Core ML disk caches were
not purged. Warm = median of three immediate repeats of the identical input.
These are single local observations under shared-machine load, not an SLO.

| Door / input | Core ML first / warm ms | ORT first / warm ms |
| --- | ---: | ---: |
| GLAP audio / ten-second music | 1132.69 / 67.39 | 398.68 / 80.26 |
| GLAP text / first English music query | 12027.50 / 41.89 | 2507.10 / 27.83 |
| SigLIP image / red image fixture | 989.51 / 29.72 | 287.00 / 53.26 |
| SigLIP text / first English red query | 2954.88 / 11.56 | 665.64 / 17.69 |

`/usr/bin/time -l` around the whole Cargo test command reports 41.60 s elapsed
(includes build/harness) and maximum resident set size **6,566,477,824 bytes**.
This is the command's reported maximum, not steady-state memory or a per-model
allocation measurement. The two production oracles themselves took 26.56 s.
In particular, GLAP's FP32 text weights are large: background lazy loading,
shared sessions and explicit release matter; this is not live-show readiness.

The `082e17a` all-feature workspace gate passed **1,574 tests / 325 skipped**
in 16.442 s, with strict all-target/all-feature clippy, fmt and license gate.
ORT-only retrieval compilation passed. The first integration run found an
obsolete three-resource UniFFI expectation; adding the two real tokenizer
resources fixed it without loosening any numerical threshold. A subsequent
Android live-PCM compatibility correction requires its own final gate below.

### Android source compatibility discovered during regeneration

The user's current, pre-existing Kotlin changes already send only appended
little-endian f32 bytes for live dictation. The September 27 Rust archive still
took `Vec<f32>` containing the whole take. Regenerating bindings blindly would
have overwritten that newer consumer contract. The first native build was
intentionally interrupted before accepting its outputs.

The shared engine now owns the incremental UniFFI adapter in a separate
`live_transcription` module: validate/append new PCM, lend one growing source
to the existing processor without whole-take copies, keep accepted samples on
a decoder error, and release take/processor ownership on finish or cancel.
Kotlin's existing application edits are preserved. Dedicated boundary tests
grade endian/delta/tail semantics, malformed-input atomicity, retries and
terminal ownership. This corrects the current consumer seam; it is not a
rewrite of Nemotron's decoder or a new inference model.

## Final qualification and Android integration

Runtime source checkpoint: **`2ad7eff`**, including `6c481da`'s incremental
PCM seam and `69f7a77`'s unused Android decoder-wrapper cleanup. Later report
and README edits do not change that runtime. The existing Kotlin incremental
adapter and its two tests are preserved, not replaced by the older snapshot's
whole-take API. Generated declarations move because the Rust object now has
its own module; UniFFI's checksums are regenerated with it.
Android integration checkpoint: **`9fe06a1`**. Only related engine/binding/
build changes are included; the user's editor and `PORT_TODO.md` edits remain
outside that commit. Exporter checkpoint remains **`45c1047`**.

Android consumer build corrections:

- Default to the main shared sibling; no implicit `Tmp` environment sourcing.
- Record engine/utils working-source digests, compiler and output lengths/
  SHA-256; reject source drift during a build. Temporary binding tests keep
  separate receipts rather than overwriting the packaged binding's proof.
- Bound binding-regeneration process lifetime before reading its output.
- Pass the actual FFmpeg bundle to the host loader probe. Its old missing-env
  expansion could exit through a bash 3.2 trap as false success; now it fails
  explicitly, with a regression test.
- Preserve the old Android FFmpeg directory as
  `vendor/ffmpeg/aarch64-linux-android.pre-scope1-20261006`; rebuild policy
  **8.1.2-2** with required CAF/AIFF/WAV and AAC/ALAC/PCM support. Six LGPL
  libraries remain dynamically linked. All seven shipped native libraries
  pass API-33/16-KB alignment and dependency checks; the host actually dlopens.
- The explicitly selected, model-free `EngineFacadeSmokeTest` no longer
  packages unrelated EdgeTAM graphs. Default/mixed/unknown/regex selections
  retain the staging integrity gate. Three actual Gradle task-plan probes
  confirm default and regex include staging, smoke omits it; the real smoke
  APK contains no stale staged ONNX graphs. Seven existing asset-integrity
  tests still pass, including refusal of missing/corrupt graphs.

The first actual facade run was **9 passed / 1 failed**: physical model
identity inspection returned a bare I/O error, losing the missing EdgeTAM
filename. This was fixed in both inference backends, not by weakening the
device assertion. Missing artifacts now return `ModelNotFound` with their
path; other I/O errors preserve their kind and path. Two real-filesystem
warmup regressions cover ONNX and Core ML before native load.

### Final gates (logs read, not inferred from compilation)

| Gate | Result | Local evidence |
| --- | --- | --- |
| All-feature workspace, bench excluded | **1,579 passed, 325 skipped**, 16.432 s execution | `target/scope1-nextest.log` |
| All-target/all-feature clippy `-D warnings`, fmt, licenses | Passed | `target/scope1-clippy.log`, `target/scope1-licenses.log` |
| Retrieval with ignored production oracles explicitly included | **13 passed, 0 failed, 0 ignored** on Core ML CPU_ONLY and ORT CPU | `target/scope1-retrieval-final.log` |
| Android arm64 ORT-only library clippy (UniFFI + retrieval) | Passed, `-D warnings`, NDK/API 33 | `target/scope1-android-clippy.log` |
| Inference contract without either native feature | Compiles | `target/scope1-infer-portable-contract.log` |
| Kotlin host: binding / Android payload policy / engine facade | **2 + 8 + 263 passed**, zero skipped or missing cases | Android `Tmp/verification/host-20261006-235758-8686391d/summary.json` |
| Android facade | **10 passed**, zero skipped or missing cases | Android `Tmp/verification/device-20261006-235812-9402a392/summary.json` |
| Android native load + real YuNet CPU/XNNPACK + cache lifecycle | **5 passed**, zero skipped or missing cases | Android `Tmp/verification/device-20261006-235901-7758fd95/summary.json` |
| Source/binary receipt regressions | **3 passed** | Android `Tmp/scope1-source-receipt-tests.log` |
| Exporter tests including payload contract integrity | **7 passed** | Exporter `artifacts/logs/retrieval-integrity.log` |
| Final upload staging vs contracts | **10 objects verified**, 8,351,611,504 payload bytes | Exporter `artifacts/logs/retrieval-staged-integrity.log` |

The three final Kotlin/device runs have the same source/native-input digest
`6bf45be732984ad0cc3b37021e13a4eb52307b017a99c09fe8188c5a8483e5aa`, no
source drift. Both device runs use a fresh `ferment-testing`, arm64/API 36;
all selected cases execute, logcat gates pass, and owned emulator cleanup
succeeds. Manual `ferment` data is not used or reset. Existing Kotlin/compiler
deprecation warnings remain; this is not a warning-free full Android app build.

Host and device commands from the Android checkout:

```sh
source scripts/env
export PATH=/opt/homebrew/bin:$PATH  # this machine's Python 3.14; pyenv default is 3.7
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
native/perception-engine/build.sh host android bindings
scripts/test --module perception-engine --module perception-engine-android --module engine --timeout 900
scripts/device-suite :engine:connectedDebugAndroidTest --fresh-emulator --timeout 900 \
  -Pandroid.testInstrumentationRunnerArguments.class=run.ferment.engine.EngineFacadeSmokeTest
scripts/device-suite :perception-engine-android:connectedDebugAndroidTest --fresh-emulator --timeout 900
```

The new binding regenerates byte-identically and loads with its real host
library. Native receipts live in Android `native/perception-engine/build/receipts/`:

- Engine source digest: `e1d7efa8b7d8f58a1565ecfc6c426f85451cfaffcfee29539e8ec966f56b0ca1`.
- Host library: 6,260,160 bytes, `bcd2e54c8cc52f55d0251676f7e58c17685ecaf279d88b1de6a5c253a3f86350`.
- Android library: 7,195,344 bytes, `db1821465e37eedf9d61ca8ae5d8e1fc3f79317b8615d363cafd945c49be6194`.
- Kotlin binding: 731,991 bytes, `5601ca42880698980f85a8f1c38c0553e3c1c9b3d81b4a9e5675891a84d3a06c`.

YuNet device CPU and XNNPACK results match the host's 24 faces with max
absolute difference **0.0**. The host inference output was also compared to
the exporter's independent decoded-box goldens: all five fields of 24 faces,
max error `1.11e-16`; compiled-in PNG pixels equal the original RGB fixture.
Evidence: `target/android-smoke-host-origin.log`,
`target/android-yunet-origin-comparison.log`; color origin independently ran
in `target/android-color-host-origin.log` and matches the device constants.

### Limits and handoff

- This is selected engine/consumer qualification, not a full Android app,
  physical-phone or every legacy production-model oracle run. The generic
  saliency oracle also requires the absent published U2Netp artifact even for
  its YuNet case; it did not pass here. The explicit host-output/golden
  comparison above does not claim to run that whole suite.
- Native Linux/Windows, GPU providers and the new large GLAP/SigLIP graphs on
  Android hardware were not run. Retrieval's ORT-only Android code cross-
  compiles; Android retrieval UI/FFI integration is not claimed by that check.
- Cold/warm, concurrency, eviction, cancellation, bounded ten-minute sources
  and maximum-RSS evidence are above. No steady-state memory SLO, broad search
  quality benchmark or live-show readiness is implied. GLAP text must load
  lazily off the interactive/realtime path.
- The user's original iOS checkout, projects and unrelated Android editor
  edits remain intact. Disk checks finished at about **37 GiB available**.
  No unrelated cache cleanup or remote publication was performed.
- CDN handoff is [the exporter's ten-object staging](../../../ferment-models-export/docs/retrieval-models.md#cdn-handoff),
  with both tokenizers and all manifests/checksums. Verify served bytes after
  the user uploads; scope 2 adds the application jobs/index/preview integration.

## Desktop low disk preflight added during scope 2

The Hypr simultaneous AV/indexing probe reproduced a native Core ML compiler
abort, `LLVM ERROR: IO failure on output stream: No space left on device`,
during the first GLAP text load. The macOS backend now checks available blocks
on the actual temporary volume before a cold session load. The conservative
budget is twice the bundle's bytes plus 1 GiB; insufficient space returns a
recoverable inference error with a retry instruction. Warm resident sessions
remain usable. This is a preflight, not a reservation against concurrent writes
or a substitute for future native compiler isolation. No iOS, Android, model
preprocessing or numerical output contract changed.

`fs4 1.1.0` is macOS-only and optional under `coreml` (MIT/Apache-2.0); it avoids
counting macOS purgeable capacity as immediately writable scratch space.
`target/scope2-coreml-disk-gate2.log` passed strict clippy, **1,582 tests with
323 skipped**, formatting and licenses. The first invocation lacked the
mandatory supplement fixture environment and failed four delayed-AAC tests;
the successful run used both fixture roots listed above. No test was weakened.
`target/scope2-coreml-disk-ort.log` passed the ORT-only `engine-infer` check.
The two new regressions cover low-space/overflow rejection and the native
temporary-volume capacity probe. Hypr's subsequent real AV/index report is
`target/scope2-av-index-report5.json` in that consumer checkout.
