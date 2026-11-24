# Track H report — Oracle & gate hardening

**Executed by the lead, 2026-08-31, against `main @ 5b97e84c`.**
Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track H".

All six items are done. Two of them turned out to be worse than the doc
described, and one CI blocker surfaced that the doc did not anticipate. Details
below, then the deviations, then what B–G need to know.

---

## Zero-trust pass: all six claims re-verified before coding

| Claim | Verdict |
|---|---|
| H.1 — 19 orphaned oracles | **Confirmed exactly.** `ort_semantic` 7, `ort_enhance` 4, `ort_behavioral` 4, `behavioral_from_dumps` 4. |
| H.2 — ~31 vacuous fixture tests | **Confirmed, and worse** — see below. |
| H.3 — CI runs no `cargo test` | **Confirmed.** `ci.yml` ran fmt/clippy/wasm/deny/`pnpm test:unit` and nothing else. The browser Vitest project was also unrun. |
| H.4 — loudness normalizer has no reference coverage | **Confirmed.** 906 LOC, 30 unit tests, zero published vectors. |
| H.5 — `fleet.rs:46` says "17 identifiers" | **Confirmed.** `ModelId` has 22. |
| H.6 — `next_short_id` is a process-global counter | **Confirmed, and the defect is the opposite of the one recorded** — see below. |

---

## H.1 — the 19 orphaned oracles now run

`Justfile:engine-oracle` gained `ort_semantic`, `ort_enhance`, `ort_behavioral`
(unconditional — the env guards the recipe already asserts are exactly what they
need) and `behavioral_from_dumps` (gated on `FERMENT_EP_DUMPS`, printing
`SKIP behavioral_from_dumps: …` when unset).

**Judgment call.** `ort_behavioral` compares a CPU backend against
`FERMENT_ENGINE_ORT_EP`. Unset, that is CPU against CPU — a green cross-EP gate
that proves nothing, which is precisely the failure mode Track H exists to kill.
It now prints a `DEGENERATE ort_behavioral: …` banner once per run in that case
(`tests/ort_behavioral.rs`, `gpu_provider()`). The suite still runs, because it
exercises the shipping post-processors either way; it just no longer reads as a
passed GPU gate on a Mac.

---

## H.2 — the vacuous tests were also *unreachable*

The doc said ~31 tests silently `return` when `../ferment-cuts-ios` is absent.
True — 31 sites, after subtracting four false positives (two production
`let-else`s in `sequencer-audio/src/driver.rs`, one already-loud `trim_log` skip
in `tests/media_audio_source.rs`, one already-loud site in
`waveform_extractor.rs`).

**The part the doc missed:** almost all of them hard-coded
`../../../ferment-cuts-ios/…` and **never read `FERMENT_ENGINE_FIXTURES_DIR` at
all**. Only 7 files in the repo referenced that variable. So `just engine-oracle`
was exporting a variable that most of the parity suites ignored — pointing it at
a real fixture tree could not make them run, and pointing it at a broken one
could not make them fail.

**Fix: a new crate, `crates/engine-test-fixtures` (`ferment-engine-test-fixtures`).**
One resolver, three outcomes, all loud:

- resolves → the test runs;
- no configured root **and** no sibling checkout → prints `SKIP <test>: <reason>`
  and returns `None`;
- a root **is** configured but the fixture is not under it → **panics**.

That third rule makes every fixture-backed test its own canary, which is
strictly stronger than the one canary the doc asked for. The canary itself also
exists (`configured_root_serves_the_whole_fixture_tree`), checking a 13-entry
sentinel list so a root pointing at the wrong directory fails *once* with a
useful message instead of thirty times with confusing ones. It is wired as the
first line of `just engine-oracle`.

Test names come from the libtest thread name, so a renamed test cannot leave a
stale string behind.

**Review round caught the obvious hole: the contract was unenforced.** Mutation
testing found that deleting the configured-but-missing panic — this crate's
entire reason to exist — left *all ~700 workspace tests green*. It had only ever
been demonstrated by hand in a shell. Fixed by splitting the decision out of
`path()` into a pure `decide(configured, sibling, relative) -> Use | Skip | Fail`,
which is unit-testable without mutating the process-global env var that every
other test in the binary reads. Five tests now cover all three arms, including
"a configured root never falls back to the sibling". Re-ran the same mutation:
two tests go red.

**Proof it works** (both demonstrated, not asserted):

```
$ FERMENT_ENGINE_FIXTURES_DIR=/tmp/bogus-fixtures cargo test -p ferment-engine-test-fixtures configured_root
FERMENT_ENGINE_FIXTURES_DIR points at /tmp/bogus-fixtures, which is missing 13 of 13 expected fixtures: [...]

$ FERMENT_ENGINE_FIXTURES_DIR=/tmp/partial-fixtures cargo test -p ferment-engine-processors --lib -- fbank::tests::test_fbank_sec5
FERMENT_ENGINE_FIXTURES_DIR is set to /tmp/partial-fixtures but fixture
AudioEventDetector/sec5/audio_int16.bin is missing from it — the fixture tree is
stale or the variable points at the wrong directory
```

**Near-miss worth recording.** The shared reader initially normalised int16 to
`-1.0..1.0`. The AED fbank oracles follow Kaldi and window **raw int16
magnitudes**; that helper would have shifted every mel bin while all eight fbank
tests still "passed" against goldens read through the same wrong scale. Caught
by reading the local helper it replaced. `as_i16_le_as_f32` now carries the
comment explaining why there is no `/ 32768`.

Migrated: `engine-audio/stft.rs`; `audio_event_detector/{cmvn,fbank,segmenter,oracle_tests}.rs`;
`beat_queen_mary/mod.rs`; `beatthis/{mel,mod}.rs`; `deepfilternet/features.rs`;
`musichighlighter/{mel,oracle_tests,pos_enc,window_search}.rs`;
`video_scene_detector/histograms.rs`. `sequencer-audio/src/host.rs`'s
device-absent skip was reworded onto the same `SKIP <name>:` convention.

---

## H.3 — CI runs the tests now, and the gate is not flaky

`.github/workflows/ci.yml` gained a `cargo test --workspace` step and the WebKit
browser suite (`pnpm exec playwright install webkit --with-deps` +
`pnpm test:browser`). `just check` gained the identical pair, so the two stay in
step — the workflow header claims it mirrors `just check`, and now it does.

Exclusions, each with its reason in the workflow file: `ferment-studio` (the
Tauri shell — its gate is the frontend build), `ferment-sequencer-video` (wgpu
tests `.expect()` a real adapter), `ferment-gpu-spike` (pulls wgpu for one pure
packing test), `ferment-engine-bench` (Criterion), and the two **vendored**
crates `matroska-demuxer` / `re_mp4`. It is an exclude list, not an include
list, so new crates are covered by default.

**Two things the local run caught that a "looks right" CI push would not have:**

1. **Vendored crates are workspace members.** A path dependency of a member
   joins the workspace automatically, so `--workspace` ran
   `matroska-demuxer`'s upstream suite, which wants Matroska conformance `.mkv`
   files we do not ship: 8 hard failures. Hence the two vendor exclusions.
2. **`ferment-storage`'s `redb_warmup_preserves_lru_order_across_restart` was
   flaky** — 2 failures in 5 full-workspace runs, but 12/12 green in isolation.
   Eviction runs off-thread on a 20 ms debounce and the test bet 80 ms of
   `sleep` on scheduler latency, a bet it loses when every other crate's tests
   share the machine. Replaced both fixed sleeps in `crates/storage/tests/store.rs`
   with `await_eviction`, a 10 s bounded poll. **4/4 full-suite runs green
   afterwards.** Left unfixed, the new gate would have been ~40% red for
   reasons having nothing to do with the change under test.

**Caveat, stated plainly: this was qualified on macOS, not Linux.** I started a
Docker `rust:1.95.0-bookworm` run to reproduce the Linux leg (per the standing
"CI lints Linux, the app ships mac+win" rule) and killed it when the working
tree became unstable — see *Working-tree collision* below. The first CI run is
therefore the Linux qualification step. If `ferment-engine-media` or
`ferment-sequencer-audio` misbehave headless, narrow to the doc's stated minimum
(`-p ferment-engine-processors -p ferment-sequencer -p processorman -p ferment-project`).

---

## H.4 — EBU R128 conformance vectors, and the meter passes them

`mod ebu_r128_conformance_tests` in `loudness_normalizer.rs`. EBU Tech 3341 §2.6
tests 1–5 plus a true-peak vector. **All six pass**, so the existing
implementation is Tech 3341-compliant within the spec's own ±0.1 LU — that is
now proven rather than assumed.

Two adaptations, both forced and both documented in the module comment:

- Tech 3341 states its signals as **stereo** and BS.1770 sums channel mean
  squares; `measure_lufs` is single-channel, so a mono signal must sit 3.01 dB
  higher to read the same LUFS. `amplitude_for_lufs` does that conversion and
  nothing else.
- **Tests 6–8 (5.1 channel weighting) and 9 (short-term window) are not ported.**
  This meter has neither a channel matrix nor a momentary/short-term lane, so
  there is nothing to point them at. Left out deliberately rather than faked.

What each vector actually pins: test 3 the relative gate (a meter without it
reads −24.2), test 5 the plain energy mean (log-domain averaging reads −23.8).

**Correction from the review round.** This report previously claimed test 4 pins
the absolute gate. It does not. Mutation testing showed the absolute gate can be
deleted outright with all six EBU tests still green: test 4's −72 LUFS shoulders
are also more than 10 LU below the mean, so the relative gate discards them
either way. Added `content_below_the_absolute_threshold_integrates_to_silence` —
content entirely at −75 LUFS must integrate to −∞ — which isolates the gate.
Re-ran the mutation: red. The signal generator is a
single phase-continuous 1 kHz tone whose amplitude steps at whole seconds —
1 kHz at 48 kHz is exactly 48 samples per cycle, so every step lands on a zero
crossing and the K-weighting filter has nothing to ring on.

**Gap left open:** Tech 3341's true-peak compliance table needs the spec's
supplied WAVs, which are not in-repo. The one TP vector here is derivable from
theory (a full-scale 997 Hz sine has a true peak of exactly 0 dBTP; Tech 3341
allows ±0.4 dB) and passes.

---

## H.5 — the stale count is gone, and cannot come back

`fleet.rs` no longer names a number. `validate_complete_fleet` already walks
`all_model_ids()`, and `default_manifest_is_complete_and_valid` already asserts
22 — the comment now points at those instead of duplicating them, so the round
that adds DINOv2/AmberNet/GigaAM/Pyannote/ReDimNet updates one assertion and no
prose.

---

## H.6 — the ID defect is the opposite of the one recorded

The doc recorded "caption/word IDs are not deterministic per re-cook". Reading
`FermentID.swift` first: **iOS mints these randomly too**, so re-cook
determinism is not a property iOS has or relies on. That framing pointed at the
wrong bug.

**The real defect:** `next_short_id` was a process-global `AtomicU64` starting at
1, so *every app launch reminted `0000000001`, `0000000002`, …*. Two elements
captioned in two different sessions produced **identical** caption and word ids.
Those ids "live inside ONE document" by design — which is exactly where a
collision hurts. Not non-determinism: systematic collision.

**Per your call mid-session, `FermentID` is now fully ported** into a new shared
crate `crates/utils` (`ferment-utils`), mirroring iOS `Packages/FermentUtils`.
Complete port of all three forms — `long` (UUIDv7 in Crockford base32),
`short` (random base62) and `short_seeded` (SHA-256 → first 8 bytes big-endian →
mod 62^n), plus `compact`, `resolve`/`Ambiguous`, `of_uuid`, `to_uuid`,
`timestamp_ms`, `is_valid`, and the `outro` / `ferment-outro` reserved
allowlist.

The Swift suite was ported case for case, **including the cross-platform vectors
that `test/unit/lib/ferment/id_test.rb` pins**, so all three implementations now
fail on one shared example if any drifts:

```
short_seeded("ferment")   == "h1nT4Qi2DJ"
short_seeded("ferment:1") == "q1kVBirBnG"
short_seeded("seed:2")    == "1qCxxiFVKB"
of_uuid("01983265-89db-7d2a-8b0b-3ce68b2a5cee") == "01k0s6b2evfmn8p2swwt5jmq7e"
compact("01k0s6b2evfmn8p2swwt5jmq7e")           == "5jmq7e"
of_uuid("ffffffff-…-ffffffffffff")              == "7zzzzzzzzzzzzzzzzzzzzzzzzz"
```

All 30 pass, including the all-zero/all-one boundary pair that exercises the
64-bit carry seam in `encode`/`decode`.

`caption_segmenter` and `capability_impls` now call `id::short()`; the counter,
the local base62 encoder and the atomic are deleted. A regression test names the
old bug so it cannot return quietly.

**One deliberate deviation from Swift:** `of_uuid` requires every character to
be an ASCII hex digit. The Swift twin inherits `UInt64(_:radix:)`'s tolerance of
a leading `+`. No real UUID reaches that branch and the stricter rule cannot
turn a valid id away.

**Open decision for the integration pass — yours, not mine.** `short_seeded` is
ported and free to use. If caption/word ids should survive a re-cook (which
matters more once ReplicaMan keys CRDT maps by them), seeding from stable
content makes re-cooks idempotent and would put us *ahead* of iOS rather than
level with it. I kept the faithful port (random, as iOS does) because changing
the semantics is a wire-contract decision, not a port decision.

---

## Working-tree collision (needs your attention)

Mid-session the ReplicaMan track began scaffolding in this same working tree —
`docs/fleet/2026-08-31_replicaman-port.md`, then `crates/ferment-doc`,
`crates/replicaman`, `crates/replica-models`, and edits to
`crates/processorman/src/v5/`. For a while `crates/ferment-doc` held fixtures but
no `Cargo.toml`, which made the `crates/*` glob fail **every** cargo command
repo-wide. I added a one-line documented `exclude`, and it had been removed again
(by that track's own edit to `Cargo.toml`) by the time `ferment-doc` gained its
manifest — the current state is correct and my two `[workspace.dependencies]`
entries survived. I touched none of their files.

Two things that are theirs, not mine, both currently red:

- **`cargo deny check licenses` FAILS.** Four crates rejected, all under
  `loro`: `bitmaps 2.1.0` (MPL-2.0+), `im 15.1.0` (MPL-2.0),
  `sized-chunks 0.6.5` (MPL-2.0), `xxhash-rust 0.8.18` (BSL-1.0). The
  cross-port doc's rule is "stop and report, do not vendor around it" — so:
  reported. My additions (`rand`, `sha2`) produce zero findings.
- **`ferment-replicaman` does not compile** — `error[E0505]: cannot move out of
  'guard' because it is borrowed` in its lib test. The new CI `cargo test` step
  will go red on it until that lands.
- **`processorman` has 5 failing V5 tests.** The only diff in that crate is a
  3-line deletion in `src/v5/runtime_state.rs` removing the "skip if the state is
  unchanged" guard, and the three failing integration tests are
  `a_fully_settled_project_opens_with_zero_writes`,
  `a_cold_import_writes_once_then_reopens_quietly` and
  `a_mid_session_server_settle_redrives_once_without_a_storm` — all three about
  not writing when nothing changed. In-progress work by that track; I did not
  touch it.

Both will block CI regardless of Track H. Numbers below therefore exclude
`ferment-replicaman` and `ferment-doc`.

---

## Test evidence

| Suite | Result |
|---|---|
| `ferment-utils` | **30 passed** + **2 malformed-input tests**, 0 failed |
| `ferment-engine-test-fixtures` | **10 passed**, 0 failed (5 of them the new contract tests) |
| `ferment-engine-processors --lib` | **644 passed**, 0 failed, 27 ignored |
| EBU R128 conformance | **7 passed**, 0 failed |
| Full CI command (parallel-track crates excluded) | **1384 passed, 0 failed**, 76 ignored |
| Full CI command, repeated ×4 | **4/4 green** (was 2/5 red before the flake fix) |
| LRU test ×40 under 8× CPU load | **40/40 green** |
| `cargo fmt --check` on every touched file | clean |
| `cargo clippy --all-targets -D warnings` on every touched crate | clean |
| `cargo deny check licenses` | **FAILS on the loro tree — not Track H** |

### Mutation testing — every test here was proven to fail when the code is wrong

| Mutation | Result | Verdict |
|---|---|---|
| `id::short()` returns a constant | 2 red | real |
| int16 fixture read normalised to ±1.0 (the Kaldi trap) | 5 fbank red | real |
| `compact()` guard weakened to a length check | 2 red (char-boundary panic) | real |
| relative gate disabled | 2 red | real |
| log-domain instead of power-domain mean | test 5 red | real |
| K-weighting offset zeroed | 5 red | real |
| stft golden perturbed by 0.25 | red, max difference exactly 0.25 | real |
| `pos_enc` output perturbed by 0.5 | red | real |
| eviction disabled entirely | red in 10 s, "evictor did not settle" | poll cannot mask a regression |
| **absolute gate disabled** | **was 6 green → now red** | **was garbage, now fixed** |
| **resolver `Fail` arm deleted** | **was ~700 green → now 2 red** | **was garbage, now fixed** |

Seeded-id vectors were independently recomputed in Python (`hashlib`) rather than
trusted from the Rust or the Swift: all three agree on `h1nT4Qi2DJ`,
`q1kVBirBnG`, `1qCxxiFVKB`.

Edges exercised by running, not by assertion: a directory where a fixture file
belongs (loud panic, names the path); ~20 malformed strings through every public
`id` entry point — empty, NUL, newline, multi-byte, 1000 chars, `\u{10FFFF}`,
26-char strings with one multi-byte character (no panics); an incomplete fixture
root (loud panic naming the missing fixture); a fixture root pointing at the
wrong directory (canary fails once, listing 13 missing entries).

## Simplified during review

- `path()`'s nested match collapsed into the pure `decide()` above — one place
  decides, one place acts on the decision.
- Deleted `require_i16_le_as_f32` (no callers) and `is_short_shaped_with_length`
  (no callers); `is_short_shaped` lost a redundant `is_ascii()` check, since
  every base62 byte is ASCII and a multi-byte character cannot pass the byte scan.
- `root`, `configured_root` and `sibling_root` are private now — they were public
  with zero external callers and became implementation detail once `decide`
  existed.

Behaviour held: full suite re-run green after each cut.

## Lead review of the B–G fleet (2026-08-31, after Track H)

All six tracks landed and were judged by running, not by reading. Every
track's central claim was mutation-tested by the lead independently of its own
self-report.

| Track | Mutation applied by the lead | Result |
|---|---|---|
| B | `OBSERVATION_LAMBDA` 16→4 | RED — madmom parity bites |
| B | `OBSERVATION_LAMBDA` 16→15, `TRANSITION_LAMBDA` 100→99 | **green** — the gate is real but insensitive to ±1 on this fixture |
| C | texture judged on the normalized plane instead of raw luma | RED |
| D | contiguous merge tier deleted | RED |
| E | `.exp()` dropped from the log-probability decode | RED |
| F | `dominant_face` tie-break first→last maximum | RED |
| G | `allow_frame_reordering` forced true | RED at the settings builder |

Three defects the self-reports did not contain, found and fixed by the lead:

1. **`media_video_paths::scene_detection_and_minimap_consume_sequential_video_io`
   was red.** Track C made scene detection request `Dinov2Vits14` in addition to
   `Autoshot`, and that test's stub backend hard-asserts the model id. Stub now
   serves both; an all-zero embedding is a legitimate abstention, so the test
   stays about video IO sequencing.
2. **DINOv2 inference failure was silent.** Abstaining on failure is the right
   policy and is documented, but there was no signal distinguishing "abstained"
   from "the backend cannot serve this model" — the second case disables the
   flash veto and moves scene boundaries with no other symptom. Added a
   `tracing::warn!` on the inference-error branch only.
3. **`ferment-sequencer-video` did not compile.** Track G added three fields to
   the shared `EncoderSettings`; two test-only struct literals in
   `sequencer-video/src/offline.rs` were never updated. G missed it because that
   crate is excluded from the `cargo test` step — but CI's
   `clippy --all-targets` compiles it, so CI would have gone red. Fixed with
   `allow_frame_reordering: false` (offline export indexes frames positionally).

Also resolved: the shared `resources.len()` assertion in `fleet.rs`, stale at 3
once Track D added `gigaam_vocab` and `ambernet_langs`. Bumped once, by the
lead, rather than four times by four agents.

**Final state, run after every fix:** workspace **2346 passed, 0 failed**, 71
ignored (excluding the Tauri shell, the two GPU crates, benches and the two
vendored crates); `cargo fmt --all --check` clean; `cargo clippy --workspace
--all-targets -D warnings` **0 errors**.

**Operational error by the lead, recorded so it is not repeated.** The CPU
contention check used to qualify the storage flake fix spawned eight busy-loops
and tried to reap them with `kill $(jobs -p)` — `jobs -p` is empty in a
non-interactive zsh, so nothing was killed and eight loops ran for roughly two
hours at ~58% of a core each, degrading all six agents. Track B found and
reaped them. Never background load generators without recording their PIDs
directly (`$!`).

## Files

**New:** `crates/utils/{Cargo.toml,src/lib.rs,src/id.rs}`,
`crates/engine-test-fixtures/{Cargo.toml,src/lib.rs}`.

**Changed:** `.github/workflows/ci.yml`, `Justfile`, `Cargo.toml`,
`crates/engine-models/src/fleet.rs`, `crates/engine-audio/src/stft.rs`,
`crates/storage/tests/store.rs`, `crates/sequencer-audio/src/host.rs`,
`crates/engine-processors/tests/ort_behavioral.rs`, and the 13
fixture-consuming modules in `crates/engine-processors/src/` listed under H.2.

## What B–G inherit

- **Fixture-backed tests: use `ferment-engine-test-fixtures`.** Add
  `ferment-engine-test-fixtures.workspace = true` under `[dev-dependencies]` and
  call `fixtures::f32_le(…)` / `require_f32_le(…)` / `bytes(…)` / `path(…)` /
  `skip(…)`. Do not hand-roll a fixture path — that is the bug this track
  removed. Paths are relative to `Tests/Fixtures` (e.g.
  `"Pyannote/one-speaker/…"`). Track E's committed goldens and Track C's
  `strobe-cuts.mp4` regression both go through this.
- **New sentinel entries.** When a track lands fixtures its suites cannot run
  without, add them to `CANARY_ENTRIES` in `crates/engine-test-fixtures/src/lib.rs`.
- **New oracle suites must be added to `just engine-oracle`.** That is how this
  round's 19 got orphaned; a `--test` target nothing invokes is not a gate.
- **Ids: use `ferment_utils::id`.** `short()` for anything document-scoped,
  `long()` for anything the server stores under an `id` column,
  `short_seeded(seed)` when a retry must land on the same id. Do not write a
  third base62 encoder.
- **CI now runs `cargo test`**, so a broken test in your track fails the PR
  rather than sitting green until someone runs the suite by hand.
