# Track C report — Scene detection + DINOv2

**Executed 2026-08-31 against `main @ 5b97e84c` in the shared working tree.**
Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track C".
iOS reference: `ferment-cuts-ios @ 3d1c2c180`.

All of C.1, C.2 and C.3 are done, plus the `strobe-cuts.mp4` regression the
acceptance criteria called optional — the fixture was obtainable and the
regression **passes**: scene starts 39 / 189 / 337.

---

## Zero-trust pass: every constant re-read from the Swift

| Doc claim | Source | Verdict |
|---|---|---|
| `baselineWindow 24`, `minimumBaselineSamples 8`, `impulseFactor 2.5`, `neighbourFactor 1.6`, `diffFloor 4.0`, `revertRatio 0.5`, `tornFractionFloor 0.12`, `minLockedBlocks 8`, `compactBodyBlocks 4`, `compactShare 0.2`, `normalizedResidual 12`, `minGapFrames 12` | `JumpCutDetector.swift:45-74` | **all exact** |
| `settleFrames 2`, `maxRunFrames 25` (+ the transient scanner's own `baselineWindow 24`, `minimumBaselineSamples 8`, `impulseFactor 2.5`, `diffFloor 4.0`) | `PhotometricTransientScanner.swift:35-41` | **all exact** |
| `structureLockShare 0.6`, `cutMargin 0.1`, nil verdict = model's boundaries stand | `TransientVerdict.swift:32-36,50-52` | **all exact** |
| `sameShotDistance 0.72` / `similarShotDistance 0.80` | `EncoderShotIdentity.swift:9-10` | **exact** |
| `refreshSeconds 10`, `minGapSeconds 0.3`, EMA gate reusing the transient constants | `SceneEmbeddingTap.swift:23-29,53-64` | **exact** (the EMA weight is 0.9/0.1, which the doc did not state) |
| DINOv2 input 224×126 non-square, positional grid baked, ImageNet norm inside | `Scripts/visual_semantic_encoder/convert.py:30-34,50-52,99-107` | **exact** |
| fp16 ≈ 41 MB | our fp16 export is 43.2 MB | close enough to be the same thing |

**Three things the doc did not say, which the source does — and which change the port:**

1. **`PhotometricNormalizer` also carries `windowSide 15`, `scale 32`, `varianceFloor 25`**
   (`PhotometricNormalizer.swift:12-18`). Without the variance floor a flat wall
   under sensor noise normalises to full-contrast texture and every block looks
   matchable. Ported, with its own test.
2. **The transient scanner's baseline ring holds only pairs judged *calm*** — an
   impulse pair never enters it, and an abandoned run retro-feeds its pairs into
   the ring (`PhotometricTransientScanner.swift:82-88`). Reading only the doc
   would have produced a rolling median over all pairs, which drifts upward
   during a strobe train and stops finding runs.
3. **`SceneEmbeddingTap`'s change gate uses an EMA baseline, not the scanner's
   rolling median** (`SceneEmbeddingTap.swift:55`). Different smoothing, same
   constants.

**One place the doc's summary is looser than the code.** The doc says the pair
`(t−1, t)` is "decided at t+2". The source decides it when frame `t+2` has been
*pushed*, i.e. `t = frame − 3` with four planes held — so the verdict for the
pair `(t−1, t)` lands three pushes later, not two. Ported from the code.

---

## C.1 — Jump-cut rewrite

`crates/engine-processors/src/video_scene_detector/`:

* **`photometric.rs` (new)** — `normalize` + `mean_absolute_difference`. iOS runs
  the two box passes through `vImageConvolve_PlanarF` with `kvImageEdgeExtend`;
  this is the same separable 15-tap uniform convolution with edge replication,
  written out. Deliberately *not* a running-sum box blur: the next step subtracts
  `E[x]²` from `E[x²]`, and f32 drift accumulated across a row is exactly the
  cancellation `VARIANCE_FLOOR` exists to bound.
* **`jump_cut.rs` (rewritten)** — the four rules, and the streaming `Scanner`
  holding 4 raw + 4 normalised planes. The old file was the pre-rewrite
  bimodality-only test with `DIFF_FLOOR 2.0` / `DIFF_BASELINE_FACTOR 2.0` and no
  photometric normalisation, no impulse rule, no flash revert and no compact
  body; all four are now in.
* `largest_body` is the 4-connected component walk over the torn blocks'
  **grid** coordinates.

**Deviations, deliberate:**

* `JumpCutDetector::verdict` returns `Option<f32>` (the tear fraction) rather
  than a `Detection { frame: -1, … }` sentinel. The scan owns the frame index;
  a `-1` that means "ask someone else" is a magic value.
* `Detection::frame` is `usize`, not `isize`. `t ≥ 1` always, so the negative
  case did not exist; deleting it also deleted a `usize::try_from` in the merge.
* `traceFrames` / `PhotometricTransient.trace` (upstream's CLI `--trace-*`
  stdout narration) are **not** ported. They are debugging scaffolding for a CLI
  this repo does not have.

## C.2 — Transient scanner, verdict, block motion

* **`transient.rs` (new)** — `Run`, the streaming `Scanner`, `scan`. Runs on
  **raw** luma, as upstream does: normalising would remove exactly the signal it
  is hunting.
* **`transient_verdict.rs` (new)** — `Verdict`, `judge`, `distance`,
  `structure_lock`, `touches`. `judge` returning `None` means "no verdict": the
  model's boundaries stand.
* **`shot_identity.rs` (new)** — `ShotIdentifying`, `EncoderShotIdentity`
  (0.72 / 0.80), `TransientCollector`.
* **`slow_motion_renderer/block_motion.rs`** — `MotionField` gains
  `torn_blocks`; `estimate_matching(from, to, texture_reference,
  max_mean_residual)` sits beside the existing `estimate(from, to)`, which now
  delegates to it. **Texture is judged on `texture_reference` (RAW luma) while
  matching runs on `from`/`to` (NORMALIZED).** That distinction is the one the
  doc warned about; mutation 4 below proves the suite catches conflating them.

**Deviation: `providesSceneEmbeddings` is gone.** Upstream needs it because it
has two identities (DINOv2 and Apple's Vision feature print) whose vectors must
never mix in the server's column. Vision is Apple-only and is not ported
(locked #9), so there is exactly one identity here and the flag has no case to
distinguish. The `ShotIdentifying` trait stays, because the veto must work with
no identity at all and because the verdict suite pins its rules against a stub.

**Deviation: `apply_transients` walks verdicts in index order.** Upstream
iterates a Swift `Dictionary`, whose order is undefined; when two verdicts
resolve to the same boundary frame, which one wins is therefore unspecified
upstream. Deciding in run order costs nothing and makes the output reproducible.

**Bug found and fixed in the merge order.** `merge_jump_cuts` is now
`merge_boundaries(result, jump_cuts, verdicts)` — one entry point that runs the
flash veto **before** the jump-cut merge, because the veto only ever inspects
boundaries and a jump cut merged first would sit inside a transient the veto is
about to collapse without the veto ever seeing it. Splitting the two into
separate public functions would have left that ordering as a trap for the
integration pass.

## C.3 — DINOv2

**Export** (`scripts/exports/dinov2/`, dasheng/parakeet pattern):

* `pyproject.toml` — pinned deps (`torch>=2.8,<2.9`, `numpy<2`,
  `onnx>=1.16,<1.20`, `onnxruntime>=1.18`, `pillow>=10`).
* `export_dinov2.py` — the ViT-S/14 forward written out statically with the
  positional grid interpolated once, input `image [1,3,126,224]` fp32 in 0..255,
  output `embedding [1,384]` L2-normalised, opset 17. Self-checks ONNX-vs-torch
  parity and exits non-zero above 1e-4 (measured **5.66e-07**).
* `gen_goldens.py` — torch reference for the committed 224×126 frame →
  `scripts/exports/rust_goldens/dinov2_{image_in,embedding_out}.bin`.
* `measure_drift.py` — the five measurements in the table below, reproducible.
* `vimage_reference.swift` / `coreml_reference.swift` — the two macOS-only tools
  that produce the iOS side of the comparison.

**`torch.hub.load("facebookresearch/dinov2", …)` does not work here.** It fetches
the repo through the GitHub *API*, which rate-limits from a shared address
(`HTTP 403: rate limit exceeded`). The export loads a pinned local clone
(`repo/`, `7764ea0f912e53c92e82eb78a2a1631e92725fc8`) over plain git instead;
weights still come from `dl.fbaipublicfiles.com` via the repo's own `hubconf`.
The clone is gitignored with the one-line command to recreate it.

**`inspect_onnx.py` verdict:** opset 17, 597 nodes, no GridSample, no custom
domains, no disqualifiers, `portable_directml_webgpu_qnn: true`.

**Normalisation: iOS's CoreML quirk is reproduced on purpose.** CoreML's
`ImageType` takes one scalar `scale` with a per-channel `bias`, so
`convert.py:97` uses `scale = mean(1/(255·std))` for all three channels rather
than the exact per-channel gain. Reproducing the textbook normalisation would
have put Studio's vectors in a slightly different space from iOS's, and both
land in the same server-side column. `--exact-imagenet-norm` exports the
textbook variant for the measurement (row 3 below); the shipping master does not
use it.

**Port** (`crates/engine-processors/src/dinov2/mod.rs`):
`FrameSnapshot`, `Dinov2SemanticEncoder` (`embedding`, `cosine_distance`),
portrait→landscape rotation (90° clockwise, matching
`vImageRotate90_ARGB8888(…, 3, …)`), Lanczos3 resample to 224×126, planar
`[3,126,224]` in 0..255.

**Deviation: `FrameSnapshot` carries RGB, not BGRA.** iOS uses BGRA because
vImage and CoreML want it. The decode path here already produces packed RGB and
the ONNX graph takes RGB, so there is no channel swizzle to get wrong.

`SceneEmbeddingTap` + `Scene::embedding` are in `embedding_tap.rs` /
`processor.rs`. `scene_embeddings` returns `Option<Vec<Vec<f32>>>` — `None` when
the pass produced no samples at all. That types the all-or-nothing rule: when
there are samples, every scene gets a vector (medoid, or the nearest sample for
a sliver the gate never sampled), so a half-filled column cannot reach the
server. `Scene::embedding` is `Option<Vec<f32>>`, `#[serde(default,
skip_serializing_if = "Option::is_none")]`.

---

## THE KNOWN RISK, MEASURED: resampler and embedding drift

iOS resamples with `vImageScale_ARGB8888(…, kvImageHighQualityResampling)`,
whose taps Apple does not publish. This port uses `image` 0.25's `Lanczos3` — no
new dependency, no `cargo deny` surface. Rather than argue about it, the whole
chain was measured against the artifacts iOS actually ships.

`vimage_reference.swift` builds a 640×360 frame designed to be hard (a smooth
two-axis ramp, a 16 px hard-edged checkerboard, and a band of value noise) and
writes both it and vImage's 224×126 answer. `coreml_reference.swift` runs the
**shipped `dinov2_vits14_identity_fp32.mlmodelc`** from the iOS `LocalModels`
tree on either resample. All numbers from
`uv run --project scripts/exports/dinov2 python scripts/exports/dinov2/measure_drift.py`:

| Measurement | Cosine distance |
|---|---|
| **ONNX master vs the shipped iOS `.mlmodelc`, identical pixels** | **5.96e-08** |
| **Studio end to end (Lanczos3 + ONNX) vs iOS end to end (vImage + CoreML)** | **6.78e-04** |
| resampler alone, both frames through iOS's CoreML model | 6.78e-04 |
| resampler alone, both frames through the ONNX master | 6.77e-04 |
| fp16 master vs fp32 master, identical pixels | 2.4e-05 |
| iOS's scalar-mean gain vs exact per-channel ImageNet gain | 9.5e-05 |

Pixel level, Lanczos3 vs vImage on that frame: **mean |Δ| 1.52/255, max 15/255,
90.6 % of channel samples within ±4.**

**Reading, plainly.** The graph is not the risk — 5.96e-08 says the ONNX master
*is* the iOS model, and that reproducing CoreML's scalar-mean gain was the right
call. The entire measured drift is the resampler: **6.78e-04 cosine, 0.0368
Euclidean on unit vectors.**

**Where that sits against the gates, honestly.** `EncoderShotIdentity` gates at
0.72 / 0.80 **Euclidean**, and the upstream measurement it comes from is "same
shot across a flash ≤ 0.67, a real cut across a flash ≥ 0.82". So the decision
band is 0.15 wide, with 0.05 of margin below the first threshold and 0.02 above
the second. A 0.0368 drift is 25 % of the band and larger than the upper margin.
For an average pair it is nothing; for a pair already sitting within 0.037 of a
threshold it could flip the verdict. **Recommendation for the lead:** if the
strobe regression ever gets noisy, or before the veto is trusted on a large
library, re-measure the two distances on a Studio-labelled clip rather than
inheriting iOS's. Nothing in this port needs to change for that — the two
constants are `EncoderShotIdentity::{SAME,SIMILAR}_SHOT_DISTANCE`.

`dinov2::resampler_tests::lanczos3_tracks_vimages_high_quality_resample` pins
the pixel-level agreement (mean ≤ 2.0, max ≤ 24, ≥ 85 % within ±4) against the
committed vImage golden, so a silent change of resizer fails a test rather than
quietly moving every embedding in the fleet.

### fp32 is the shipping master, not fp16

iOS ships fp16 for the ANE. Studio's qualified baseline is the CPU EP
(locked #10), where ORT itself reports `Could not find a CPU kernel and hence
can't constant fold Sqrt` for the fp16 graph — it inserts casts. **The CPU-EP
timing could not be measured honestly on this box** (six agents sharing it: four
interleaved runs gave fp16 137/237/349 ms and fp32 166/265/326 ms per frame —
the spread is larger than the difference). Rather than trade accuracy for an
unmeasurable win, the manifest points at fp32; 86 MB is nowhere near a size
problem next to the 432 MB parakeet encoder already in the fleet. The fp16
master is built under `scripts/exports/dinov2/artifacts/` and covered by its own
oracle, so the Windows/DirectML qualification round can adopt it by changing one
digest and one file name in `fleet.rs`. It is deliberately **not** staged for the
CDN: nothing in the manifest points at it yet.

---

## Model delivery

| File | sha256 | Bytes |
|---|---|---|
| `dinov2_vits14.onnx` (**wired**) | `c6d32f3846e411366be5497198ba11e6dc8eb2dd83135029b2e7e190d85763bc` | 86 403 653 |
| `dinov2_vits14.fp16.onnx` (built, not in the manifest, not staged) | `7954e7fe2aa44dd6470f5bf97fd9d230212c37346b85109cbaec973c981e7ebe` | 43 232 159 |

* `ModelId::Dinov2Vits14` → `"dinov2_vits14"`.
* `fleet.rs`: `third_party_ort(…, "Apache-2.0",
  "https://github.com/facebookresearch/dinov2", <coreml gap reason>)`. The Core
  ML gap is deliberate and is not a missing export — iOS compiles its own
  `.mlmodelc` from the same weights and Studio's backing on every platform is
  this ONNX master.
* Staged: `dist/models-publish/dinov2_vits14.onnx`, digest verified against the
  manifest. `scripts/models-publish` finds it automatically
  (`scripts/exports/*/artifacts/` is one of its master locations), so no change
  to that script was needed.
* `cargo deny check licenses`: **no new findings from this track** — the port
  adds no Rust dependency at all (`image` and `async-trait` were already
  workspace deps). The pre-existing loro/MPL-2.0 failure recorded in the Track H
  report is untouched and is not mine.
* `MODEL_FLEET_VERSION` **not bumped** — per the round's rule it bumps once, in
  the last-landing model track.

---

## Test evidence

| Suite | Result |
|---|---|
| `ferment-engine-processors --lib video_scene_detector` | **97 passed**, 0 failed (99 listed; the 2 ignored are the pre-existing Core ML parity tests) |
| `ferment-engine-processors --lib dinov2` | **7 passed**, 0 failed, 1 ignored (the drift-measurement dump) |
| `ferment-engine-processors --lib` (whole crate) | **942 passed**, 28 ignored; the only failure is `thumb_container::…`, another track's in-flight file |
| `ferment-engine-processors --lib slow_motion` (the `block_motion` change) | **58 passed**, 0 failed, 6 ignored |
| `--test ort_semantic` (3 new DINOv2 oracles, `--ignored`) | **3 passed** |
| `--test scene_detection_regression` (`--ignored`) | **3 passed** |
| `ferment-engine-test-fixtures` with a real fixture root | **10 passed** (canary includes the new entry) |
| `ferment-engine-infer` | **20 passed** |
| `rustfmt --check` on every file I touched | clean |
| `cargo clippy -p ferment-engine-{processors,models,infer} --all-targets --all-features` | **zero findings in my files** |

### The strobe regression, running

```
running 3 tests
test every_final_scene_carries_a_unit_embedding ... ok
test strobe_footage_yields_only_the_real_cuts ... ok
test the_model_alone_still_fires_on_the_flashes ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 55.88s
```

Real decode (VideoToolbox) → AutoShot through ORT → jump cuts → flash veto with
DINOv2 → per-scene embeddings, on the upstream `strobe-cuts.mp4`. Scene starts
land within ±1 of 39 / 189 / 337, exactly as the Swift gate demands. The second
test is the anti-vacuity check the Swift suite carries: with the veto off, the
model alone still fires on the flashes at 110 / 120 / 138 and produces more than
six boundaries — so the first test is pinning a regression the model really
does have.

### The DINOv2 oracles, running

```
dinov2 embedding [1, 384]: observed maximum difference 0.00000057742
test dinov2_embedding_matches_python_origin ... ok
dinov2 fp16 embedding [1, 384]: observed maximum difference 0.00033459067
test dinov2_fp16_master_still_answers_the_origin ... ok
dinov2: same picture under light 0.06979418, different content 0.15557194
test dinov2_encoder_puts_light_nearer_than_content ... ok
```

The third one runs the **whole** encoder — snapshot preparation, rotation,
resample, tensor plumbing — through the real ORT backend, which the tensor-level
parity test cannot see, because it starts from an already-prepared tensor.

### Resampler gate, running

```
resampler drift vs vImage: mean 1.522 max 15 within±4 90.6%
test dinov2::resampler_tests::lanczos3_tracks_vimages_high_quality_resample ... ok
```

---

## Mutation testing

Fourteen mutations, each applied to the shipping code, run, and reverted. Two
came back **GREEN — worthless coverage** — and both were fixed with a new test
before this report was written; the last column is the re-run.

| # | Mutation | Result | Verdict |
|---|---|---|---|
| 1 | flash-revert rule deleted (`ahead1.min(ahead2) < REVERT_RATIO·d`) | **was GREEN → now RED** | **was garbage, now fixed** |
| 2 | compact-body rule deleted | RED | real |
| 3 | neighbour rule deleted (a run of high pairs becomes an edit) | RED | real |
| 4 | texture judged on the NORMALIZED plane instead of the raw one | **was GREEN → now RED** | **was garbage, now fixed** |
| 5 | photometric normalization bypassed (fixed pedestal instead of the local z-score) | RED | real |
| 6 | transient settle look-ahead ignored (a step opens a run) | RED | real |
| 7 | `CUT_MARGIN` → 0 (a tie becomes a placement) | RED | real |
| 8 | structure lock ignored inside the identity band | RED | real |
| 9 | flash veto runs AFTER the jump-cut merge | RED | real |
| 10 | medoid replaced by a mean | RED | real |
| 11 | embeddings attached even with zero samples (all-or-nothing broken) | RED | real |
| 12 | resampler swapped `Lanczos3` → `Triangle` | RED | real |
| 13 | portrait rotation direction flipped | RED | real |
| 14 | torn blocks reported in pixel instead of grid coordinates | RED | real |

**The two that were garbage, and why.**

*Mutation 1.* Every flash test in the suite used a **brightness** flash, and
photometric normalisation already kills those before the impulse test ever runs
— so the revert rule had no test of its own. What it actually guards is a
**structural** interruption: the subject is replaced at frame 20, holds at 21,
and the picture is back at 22. That pair passes the impulse test *and* the
neighbour test, and the block matcher calls it a jump cut; only "frame 19 meets
frame 22 again" rejects it.
`a_structural_interruption_that_reverts_is_not_an_edit_at_its_start` builds
exactly that and pins the boundary at 21; with the rule deleted it moves to 19,
where nothing was edited.

*Mutation 4.* This is the exact trap the dispatch warned about — "texture judged
on RAW luma, matching on normalized… it is easy to conflate" — and my suite did
not catch it, because every synthetic plane in it is textured edge to edge.
`texture_is_judged_on_the_raw_plane_not_the_normalized_one` builds a shot that
is half real texture and half flat wall with ±2 luma of sensor noise, and
measures both ways on the *same* static pair:

```
RAW texture reference:        textured 42, locked 42, torn  0  -> verdict None
NORMALIZED texture reference: textured 78, locked 42, torn 36  -> verdict Some(0.46)
```

Judge texture on the normalized plane and a wall's sensor noise — z-scored up to
full contrast — becomes 36 "textured" blocks that match nothing, and a static
shot invents a 46 %-tear jump cut. The test asserts both halves, including that
the wrong reference really does invent a boundary, so it cannot quietly stop
discriminating.


### Harness incident, disclosed

The mutation harness used a single `/tmp/mut.bak` and I briefly had two runs of
it in flight at once. The two runs interleaved backup and restore, which left
five source files holding each other's contents and destroyed `embedding_tap.rs`
outright. Nothing shipped in that state — it happened between test runs — but it
is worth recording because `embedding_tap.rs` in the tree is a **reconstruction**
rather than the original bytes. Recovery and verification:

* four files were a clean permutation and were moved back
  (`jump_cut.rs`, `photometric.rs`, `dinov2/mod.rs`, `block_motion.rs`);
* one leftover mutation (the transient settle look-ahead) was still applied and
  was caught by three red transient tests, not by inspection;
* `embedding_tap.rs` was rewritten and is byte-checked by its own 12 tests, the
  medoid/assignment assertions, and `rustfmt`/clippy;
* the whole module tree was re-enumerated afterwards — the same test names as
  before, and every one of the seven load-bearing lines the mutations target
  greps back to exactly one occurrence.

The harness now writes one backup file per mutation
(`/tmp/mut2-<mangled-path>.bak`) and is never run concurrently with itself.

---

## A real defect this work surfaced: nested `block_on`

The first run of the strobe regression failed with

```
cannot execute `LocalPool` executor from within another executor: EnterError
```

`Dinov2SemanticEncoder::embedding` originally used
`futures_executor::block_on` around `Backend::infer`, following
`slow_motion_renderer/rife.rs`. That precedent is safe only because the
slow-motion capability is `Unsupported` and RIFE is therefore never driven from
inside an executor. The scene detector's producer *is* inside one, so the
sync-over-async seam was a guaranteed panic under `futures_executor` and a
blocked worker under tokio.

Fixed structurally rather than papered over: `ShotIdentifying::embedding` and
`Dinov2SemanticEncoder::embedding` are `async`; **everything downstream of the
model stays synchronous and pure** — `judge`, the placement walk, the medoid,
`apply_transients`, `scene_embeddings`. `TransientCollector` fetches embeddings
in two rounds (`before`/`after`, then the run's own frames only when the verdict
is a shot change), so a same-shot run — a flash, the common case — still costs
exactly two encoder passes, the same as the lazy Swift version. `judge` is a
pure function of the embeddings handed to it, so calling it twice on the rare
shot-change run costs one repeated block match.

There is now **no `block_on` anywhere in the shipping scene-detection path**.

---

## Scope

Everything landed inside the boundary: `video_scene_detector/`, a new `dinov2/`
module, `engine-models`/`engine-infer` entries, `scripts/exports/dinov2/`. Plus:

* `slow_motion_renderer/block_motion.rs` — required by C.2 ("`BlockMotionEstimator`
  gains `textureReference:`/`maxMeanResidual:` + `tornBlocks`"). Not claimed by
  any other track.
* `capability_impls.rs` — **one call site**, updated for the new `Decoded`
  struct and the renamed merge. Track B owns the *hybrid beat* impl in that
  file; I did not touch it.
* `lib.rs` — `pub mod dinov2;` + one re-export line.
* `crates/engine-test-fixtures/src/lib.rs` — one `CANARY_ENTRIES` line for
  `strobe-cuts.mp4`, as the Track H report instructs.
* `Justfile` — the new `--test scene_detection_regression` target added to
  `just engine-oracle`, as the Track H report instructs.
* `.gitignore` — the two >10 MB masters and the pinned upstream clone.

`crates/project`, `apps/`, `packages/` and all graph/cook wiring: **untouched**.

### The one seam left for the lead

`ferment_engine_core::VideoScene` (the capability-level DTO in
`crates/engine-core/src/capabilities.rs`) has **no `embedding` field**, so the
per-scene vectors stop at `video_scene_detector::Scene` and cannot cross
`VideoSceneDetecting::detect`. I left `capabilities.rs` alone on purpose: three
other tracks are adding traits to that file in this same working tree, and the
DTO's shape is a wire-contract decision the integration pass owns. The change is
two lines:

```rust
// crates/engine-core/src/capabilities.rs, in `struct VideoScene`
    /// DINOv2 medoid of the scene, 384 floats. All scenes carry one or none do.
    pub embedding: Option<Vec<f32>>,
```

```rust
// crates/engine-processors/src/capability_impls.rs, in the VideoScene map
                    embedding: scene.embedding,
```

---

## What the lead needs for the wiring pass

**New public types** (all under `ferment_engine_processors`):

| Type | Where | Shape |
|---|---|---|
| `video_scene_detector::Decoded` | `processor.rs` | `{ frames: Vec<u8>, frame_count: usize, frame_rate: f64, jump_cuts: Vec<JumpCut>, verdicts: Vec<Verdict>, samples: Vec<Sample> }` — one decode, all three taps |
| `Scene.embedding` | `processor.rs` | `Option<Vec<f32>>` (384, unit) |
| `JumpCut` (`jump_cut::Detection`) | `jump_cut.rs` | `{ frame: usize, confidence: f32 }` |
| `TransientVerdict` (`transient_verdict::Verdict`) | `transient_verdict.rs` | `{ run: Run, same_shot: bool, cut_start: Option<usize> }` |
| `TransientRun` (`transient::Run`) | `transient.rs` | `{ first, last }` + `before()` / `after()` |
| `Sample` | `embedding_tap.rs` | `{ frame: usize, vector: Vec<f32> }` |
| `EncoderShotIdentity` | `shot_identity.rs` | wraps `Dinov2SemanticEncoder`; 0.72 / 0.80 |
| `Dinov2SemanticEncoder` | `dinov2/mod.rs` | `async embedding(&FrameSnapshot) -> Vec<f32>`, `cosine_distance` |
| `FrameSnapshot` | `dinov2/mod.rs` | packed RGB + `bytes_per_row` |

**Call order — this is load-bearing:**

```rust
let decoded = detector.decode_for_detection(path).await?;                  // one decode
let result  = detector.detect(&decoded.frames, decoded.frame_count,
                              decoded.frame_rate, progress).await?;        // AutoShot
let result  = VideoSceneDetector::merge_boundaries(result,
                  &decoded.jump_cuts, &decoded.verdicts);                  // veto, then merge
let result  = VideoSceneDetector::attach_embeddings(result, &decoded.samples); // last
```

`capability_impls.rs` already does exactly this; the sequence is repeated here
because `attach_embeddings` **must** be last (samples carry frame stamps, not
scene indices, so they can only be assigned once the boundaries have settled)
and because `merge_boundaries` encodes the veto-before-merge order internally so
it cannot be got wrong from outside.

**Behaviour when the DINOv2 model is absent from the local fleet:** every
`embedding` call returns empty, `judge` returns `None`, no verdicts are
produced, the tap collects no samples, and `Scene::embedding` stays `None`
everywhere. The veto abstains and the model's boundaries stand — which is the
upstream contract, reached here without a separate `isInstalled` probe.

**640-wide input.** The processor's input surface is unchanged. Frames now
arrive at whatever width the caller's feed produces; `jump_cut::analysis_luma`
reduces them to the 256-wide analysis plane the scanners share, and the encoder
gets the *decoded* frame rather than a second-generation copy of it — one
high-quality resample to 224×126 instead of two, and exactly the geometry the
resampler gate measures. Both the jump-cut law and the flash law have explicit
640×360 tests
(`the_same_jump_cut_is_found_on_a_640_wide_feed`,
`the_flash_law_holds_on_a_640_wide_feed`).

---

## Files

**New**

```
crates/engine-processors/src/dinov2/mod.rs
crates/engine-processors/src/video_scene_detector/photometric.rs
crates/engine-processors/src/video_scene_detector/transient.rs
crates/engine-processors/src/video_scene_detector/transient_verdict.rs
crates/engine-processors/src/video_scene_detector/shot_identity.rs
crates/engine-processors/src/video_scene_detector/embedding_tap.rs
crates/engine-processors/src/video_scene_detector/synthetic.rs   (#[cfg(test)])
crates/engine-processors/tests/scene_detection_regression.rs
scripts/exports/dinov2/{pyproject.toml,export_dinov2.py,gen_goldens.py,measure_drift.py}
scripts/exports/dinov2/{vimage_reference.swift,coreml_reference.swift}
scripts/exports/dinov2/goldens/{resample_source_640x360.rgb,resample_vimage_224x126.rgb}
scripts/exports/rust_goldens/{dinov2_image_in.bin,dinov2_embedding_out.bin}
```

**Changed**

```
crates/engine-processors/src/video_scene_detector/{jump_cut.rs,mod.rs,processor.rs,scenes.rs}
crates/engine-processors/src/slow_motion_renderer/block_motion.rs
crates/engine-processors/src/{lib.rs,capability_impls.rs}
crates/engine-processors/tests/ort_semantic.rs
crates/engine-infer/src/model.rs
crates/engine-models/src/{fleet.rs,lib.rs}
crates/engine-test-fixtures/src/lib.rs      (one CANARY_ENTRIES line)
Justfile                                    (one engine-oracle target)
.gitignore                                  (the two masters + the pinned clone)
```

Untracked/modified paths in this tree that look adjacent but are **not mine**:
`video_scene_detector/histograms.rs` (Track H's fixture migration),
`crates/engine-test-fixtures/` as a whole (Track H's crate),
`scripts/exports/.gitignore` (Track E).

---

## Not run / not mine

* **`cargo test --workspace` was not run to completion.** Other tracks are live
  in this tree and were intermittently red while I worked (`beatthis/lattice.rs`
  and `video_saliency/measure.rs` both broke the lib-test build for stretches).
  I ran my own crates repeatedly instead.
* **`ferment-engine-models --lib` has one failing test that is not mine:**
  `default_manifest_is_complete_and_valid` asserts `resources.len() == 3` and
  there are now 5 — two `third_party_resource` entries added by another track
  (the AmberNet language table and the GigaAM vocab) without updating the
  assertion. `validate_complete_fleet()` passes, and the models count (32)
  including `dinov2_vits14` passes.
* **`cargo deny check licenses` still fails on the loro tree** (MPL-2.0 /
  BSL-1.0), exactly as the Track H report recorded. Nothing from this track
  contributes to it.
* **fp16 CPU-EP timing** — see above; the box was too loaded to measure.
* **GPU EPs** — not qualified, per locked #10.
* **`pnpm test`** — untouched; this track changes no JavaScript.
