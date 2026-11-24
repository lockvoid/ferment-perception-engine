# 2026-08-31 — Engine cross-port: FermentEngine drift + new families (FLEET)

**Status: ready to dispatch.** Prepared by the lead from a verified three-way sweep
(iOS engine at HEAD, redshift ports, server contract). Tracks B–H below are
independently dispatchable; dependencies are stated per track.

**SCOPE (hard boundary).** This round ports **processors and protocols only** —
pure, freestanding, fully tested modules in the engine crates:
`crates/engine-processors`, `crates/engine-models`, `crates/engine-infer`,
`crates/engine-core` (new capability traits), `scripts/exports/`, plus
Justfile/CI for Track H.
**Do NOT touch** `crates/project`, `apps/`, `packages/`, or any network/cook/graph
wiring — element-network integration (operator keys, cook kinds, blob versioning,
processor versions, engine.rs) is the **lead's** job and lives in
`2026-08-31_engine-cross-port-integration.md`. If your port seems to require a
`crates/project` change, stop, leave the seam clean (a trait + a TODO-free doc note),
and record it in your report. We are porting independent parts of the app;
ProcessorMan/ReplicaMan integration comes in a later round.

**Executor protocol (every track):**
- Zero-trust flows down: every constant, signature and key in this doc was read from
  source on 2026-08-31, but you re-verify against the referenced file/SHA before
  coding. If the source disagrees with this doc, the source wins — say so in your report.
- Everything you build is tested. New oracle/parity tests follow the established
  pattern (`gen_goldens.py` from the stock origin → `scripts/exports/rust_goldens/` →
  `#[ignore]` ORT oracle). Fixture-gated tests must **print `SKIP <name>: <reason>`**
  when the fixture is absent — never silently pass (see Track H for why).
- Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` (pinned 1.95.0),
  crate tests green, root `pnpm test` untouched. No Prettier, ever.
- Large model artifacts are **gitignored** (>10 MB rule) and staged for the CDN;
  goldens are small and committed.
- Finish by writing **`docs/fleet/2026-08-31_engine-cross-port-<track>_report.md`**
  (e.g. `-b_report.md`): what you did, every deviation from this doc, judgment calls,
  test evidence (names + counts), and everything the lead needs for the wiring pass
  (new public types, trait impls, output shapes, model IDs). A track without a
  report is not done.

---

## Baselines and authority

| What | Ref |
|---|---|
| iOS baseline the July port tracks | `ferment-cuts-ios @ 0bd5d679c` (2026-07-23) |
| iOS current | `ferment-cuts-ios @ 3d1c2c180` (2026-08-31) — 26 engine commits in between |
| redshift | `main @ 5b97e84c` |
| iOS engine sources (implementation reference) | `ferment-cuts-ios/Packages/FermentEngine/Sources/FermentEngine/` |

Wire-contract questions (operator keys, cook kinds) are **out of your scope**; where
this doc mentions a cook payload shape it is so your Rust output types carry the
right fields — the lead maps them onto the wire.

## What changed upstream (verified summary)

**New families since the freeze:** AmberNet (language ID, gate of the speech lane),
GigaAM (Russian ASR), Pyannote + ReDimNet (speaker diarization + embedding),
DINOv2 (visual semantic encoder inside scene detection), VideoSaliencyAnalyzer
(Apple Vision upstream — ported here with a substitute ONNX backing, Track F),
EdgeTAM (object segmentation — **unplugged upstream, deferred**), AppleSpeech
(Apple-only — **excluded**).

**Drift in ported families:** beats got a full re-cut (madmom-parity DBN decoder,
tempo lattice segmentation, energy structure, offset 0, anti-aliased SRC;
`HybridBeatDetector` **deleted upstream**); scene detector got a jump-cut rewrite +
flash veto + DINOv2 identity + per-scene embeddings; video encoder moved to
deterministic bpp rate control; caption segmenter gained an abbreviation guard;
audio minimap gained a bar ruler.

## Locked decisions

1. **The DBN + lattice path replaces `HybridBeatDetector`.** Build the new path
   complete and tested **alongside** the existing module; do not delete or rewire
   hybrid — the lead swaps consumers and deletes it in the integration pass.
2. **Beat grid offset = 0.** The stored grid is the model's truth; audition lead is a
   consumer concern. Do not bake −0.010.
3. **BeatThis ensemble = 2** (`shippingEnsembleSize = 2` upstream). ONNX masters
   `beat_this_fp16_{1,2}`; master 3 stays published but unused.
4. **`pal8` palettization is CoreML-only.** The ONNX fleet stays fp16/fp32. Never try
   to reproduce palettization in ORT.
5. **Desktop speech routing has no Apple lane:** GigaAM(`ru`) → Parakeet(its 25) →
   unsupported. Legacy-tag modernization `iw→he`, `jw→jv`. Top-3 hypothesis walk
   (first of the top 3 any transcriber supports). No chip axis anywhere.
6. **EdgeTAM: do not port.** Unplugged upstream, absent from model delivery.
   Re-evaluate when iOS wires it.
7. **AppleSpeech: do not port.**
8. **Saliency IS ported with a substitute backing** (Dimitri's call, 2026-08-31):
   pure math exactly from iOS + an ONNX salient-object model + face detector behind
   traits. See Track F.
9. **Apple plumbing is never ported:** ANE waterfalls, prewarm scheduling,
   `MLComputeUnits` tables, `minChipGeneration` floors, VT probes. Where iOS pinned a
   model `.cpuOnly` for ANE-economics reasons, the ORT default EP chain is fine unless
   a track says otherwise.
10. **New-model EP policy:** new models ship with CPU-EP as the qualified baseline;
    DirectML/CUDA qualification happens later on the Windows box (016 pattern).
    Do not block on GPU EPs.
11. **Speaker project-level fold is out of scope** (it is replica-side on iOS; the
    replica round comes later). Diarization ends at the typed result.

---

## Track B — Beats overhaul (`crates/engine-processors`)

The biggest behavioral drift. iOS sources under
`Packages/FermentEngine/Sources/FermentEngine/Processors/BeatThis/` + commits
`df4b1b29e`, `9a40897b0`, `1be9614f5`.
**Boundary:** everything lands inside `beatthis/`, `beat_queen_mary/`,
`audio_minimap_composer/` and a new shared `energy_structure` module. Leave
`hybrid_beat_detector/`, `capability_impls.rs`'s hybrid impl, and `crates/project`
untouched (locked #1) — but DO add/extend the `BeatAnalyzing`-facing surface on the
new path so the lead's swap is a constructor change, not a redesign.

**B.1 Port `BeatThisDBN.swift` (406 LOC)** — madmom `DBNDownBeatTrackingProcessor`
over BeatThis logits. All math in `f64`; parity gate is **exact frame indices**.
Constants (re-verify in source): `barLengths [3,4]`, `minBPM 55`, `maxBPM 215`,
`numTempi 60`, `transitionLambda 100`, `observationLambda 16`,
`activationThreshold 0.05`, `correct = true`, `transitionFloor = np.spacing(1)`.
Input massaging: sigmoid in double → squeeze into `(ε/2, 1−ε/2)`, `ε = 1e-5` → Böck
two-column form → `threshold_activations` trim. The winning HMM's bar length IS the
meter. Fallback to the existing minimal postprocess when the decode yields < 2 beats.
**Parity fixtures are committed upstream** (force-added):
`Tests/Fixtures/BeatThis/dbn-parity-{logits,expected}.json` — copy them in and match
exactly.

**B.2 Port `BeatLatticeSegmentation.swift` (237 LOC)** — piecewise-constant tempo:
ornament gate (conjunction: `|ibi − runningMedian(17)| > 3σ̂` AND metrical ratio
`< 0.75 || ≥ 1.5`; `σ̂ = max(MAD·1.4826, quantizationSigma, 1e-9)`,
`quantizationSigma = median(nonzero |Δ|)/√12`) → PELT (L2, `β = 2σ̂² ln n`, BIC,
bridge intervals masked) → per-segment linear-fit period + circular-mean anchor +
downbeat phase by votes. Output type: `segments[{bpm, first, count, downbeatPhase?,
meter?}]` alongside the flattened quantized grid in `BeatAnalysisResult`.

**B.3 Port `EnergyStructure` + `energyPerBar`** (`1be9614f5`,
`Protocols/EnergyStructure.swift`) — computed once at detection, carried by BOTH
detectors (`BeatAnalysisResult.withEnergy(perBar:)`). RMS per bar in dBFS (floor −80,
one decimal). `detect`: `.empty` below 8 bars; partition (optimal partitioning,
penalty `2σ̂² ln n`) → merge (closest-pair-first under `jndDb 1.0` /
`maxSegments 24` — **the doc-comment at `EnergyStructure.swift:70-74` is stale,
port the code not the comment**) → collapseEvents (`phraseBars 2`, collapsed bars
enter a `spent` set) → leveled (headroom below the track's own full power;
`silenceDb −45`). Result fields: 1-based inclusive bars,
`level ∈ silence|quiet|mid|loud`, events `peak|dip`.

**B.4 Grid + input corrections:** `defaultBeatOffset → 0` (locked #2).
**Anti-aliased SRC:** iOS moved from linear interpolation to polyphase resampling at
the decoder and the 22.05 kHz mel moved by mean |Δ| 0.68 — audit redshift's resample
path (`audio_source.rs` / engine-audio); if it is linear, replace with a proper
polyphase/windowed-sinc resampler (license must clear `cargo deny`) and re-run the
BeatThis mel fixture tests.
**BeatQueenMary silence trim:** `trimToActiveSpan` — active span = first..last
envelope frame ≥ 2% of peak, padded half a beat period, outer bounds only.
**B.5 AudioMinimap bar ruler** (`1be9614f5`): new `compose` overload taking downbeat
times; yellow `rgb(1.0, 0.82, 0.12)` verticals; `labelStride` starts 4, doubles until
`barWidth×stride ≥ 40px`; `lineStride` starts 1, doubles until `≥ 3px`; labelled
alpha 0.9/width 2, unlabelled 0.35/1; labels white over black +1px offset. The
"tolerant join" (composing plain without beats) is wiring — just make the no-ruler
path the default overload.

**Acceptance:** DBN parity exact on the committed fixtures; lattice + energy tests
ported from `BeatThisDBNTests`/`BeatThisChunkPlanTests` + the Swift energy suites;
`BeatAnalysisResult` carries `segments` + `energyPerBar` + `energyStructure`; both
detectors populate energy; hybrid untouched and workspace still compiles.

---

## Track C — Scene detection + DINOv2

iOS sources: `Processors/VideoSceneDetector/` + `Processors/DINOv2/`; commits
`d690ce01f`, `5bc1921f4`, `1f18a842a`, `11b25b977`.
**Boundary:** all inside `video_scene_detector/` + a new `dinov2/` module +
`engine-models`/`engine-infer` entries + `scripts/exports/dinov2/`. The "scenes now
read the 640-wide ai feed" change is wiring (lead's pass) — your processor keeps its
current input surface but must accept 640-wide frames and the test suite covers that
geometry.

**C.1 Jump-cut rewrite** (pure luma math, fully portable): `JumpCutDetector` — a jump
cut is 4 things at once: photometrically-normalized structure match
(`PhotometricNormalizer`), temporal impulse with flash-revert, compact tear
(4-connected torn-block component), locked background. Streaming scanner holds 4 raw
+ 4 normalized planes; pair (t−1, t) decided at t+2. Constants (re-verify):
`baselineWindow 24`, `minimumBaselineSamples 8`, `impulseFactor 2.5`,
`neighbourFactor 1.6`, `diffFloor 4.0`, `revertRatio 0.5`, `tornFractionFloor 0.12`,
`minLockedBlocks 8`, `compactBodyBlocks 4`, `compactShare 0.2`,
`normalizedResidual 12`, `minGapFrames 12`.
**C.2 `PhotometricTransientScanner`** (flash runs on raw luma; `settleFrames 2`,
`maxRunFrames 25`) and **`TransientVerdict`** (identity + structure lock
`structureLockShare 0.6`; `cutMargin 0.1`; `nil` verdict = model's boundaries stand).
`BlockMotionEstimator` gains `textureReference:`/`maxMeanResidual:` + `tornBlocks`
(texture judged on RAW luma, matching on normalized).
**C.3 DINOv2:** export ViT-S/14 to ONNX (Apache-2.0; iOS converter decisions in
`Scripts/visual_semantic_encoder/convert.py`: static forward, positional grid baked,
**input 224×126 non-square**, ImageNet norm inside the graph; fp16 ≈ 41 MB). Port
`DINOv2SemanticEncoder` pre-processing (portrait→landscape rotate, high-quality
resample — pick a Lanczos-class Rust resizer and pin it with fixture tests; vImage
parity is the known risk, measure the embedding drift and record it),
`EncoderShotIdentity` (cosine `sameShotDistance 0.72` / `similarShotDistance 0.80`),
`SceneEmbeddingTap` (EMA change gate reusing transient constants,
`refreshSeconds 10`, `minGapSeconds 0.3`), and `Scene.embedding` = **medoid**
assigned after merge + veto (`embedding: Option<Vec<f32>>` on the scene type,
all-or-nothing across the result).

**Acceptance:** transient/jump-cut/verdict unit tests ported from the Swift suites;
DINOv2 ONNX oracle (golden = fp32 CPU reference outputs on committed small images,
pyannote-pattern, Track E note); if the upstream `strobe-cuts.mp4` regression
fixture (scene starts 39/189/337) is obtainable from the iOS fixtures archive, port
that regression; result type carries embeddings.

---

## Track D — Speech system: AmberNet + GigaAM + routing

iOS: `Processors/AmberNet/`, `Processors/GigaAM/`, `TranscriberRouting.swift`,
`Helpers/TokenSeamMerge.swift`; commits `88ac816ec`, `24bcd9d38`, `3dd704d80`,
`c23c77d1e`, `c7ab8d6b2`.
**Boundary:** new `ambernet/` + `gigaam/` modules, a shared `token_seam_merge`
module, a pure `transcriber_routing` module (engine-side — it is a pure function),
`caption_segmenter.rs` edits, exports + models entries. The
`DownloadTranscriberModel` graph node and the languageId cook plumbing are wiring
(lead) — your routing module just exposes `route(hypotheses) -> Route` and
`model_files(route)`.

**D.1 AmberNet (224 LOC Swift):** ONNX export from NVIDIA NGC `langid_ambernet`
v1.12.0 (`Scripts/language_id/convert.py` documents the source; mel frontend baked,
inputs `audio[1,480000] f32` + `num_frames[1] i32`, output 107 logits). The 107
language tags live in model metadata upstream — bake them into a committed
`ambernet_langs.json` resource. Port `buildWindow` (speech segments → 30 s
zero-padded window, `frameCount = filled/160`, min 2 frames → error) and
`hypotheses` (max-subtracted softmax, sorted desc). Output type:
`{hypotheses: [{code, probability}], speech_seconds}`. **No coverage threshold** —
"trust AED".
**D.2 GigaAM (10 files, 1205 LOC):** Russian ASR, 4 graphs
(preprocessor/encoder/decoder/joint + 1024-piece vocab). Check SberDevices for an
official ONNX export first; else torch→ONNX mirroring `Scripts/gigaam/convert.py`
(fp16 encoder like parakeet if >2 GB; `keep_io_types`). Port the chunk pipeline —
constants pinned in `GigaAMConfig.swift` (15 s chunks / `maxChunkSamples 240_000`,
`overlapFrames 50`, `samplesPerEncoderFrame 640`, `blankTokenId 1024`,
`maxSymbolsPerStep 10`, `maxTokensPerChunk 300`; mel frames `= (n−320)/160 + 1`,
encoder frames = two rounds of `(L−1)/2 + 1`; **the `length` input is
load-bearing** — the preprocessor always sees a full 240k buffer), decoder state,
joint, postprocessing.
**D.3 `TokenSeamMerge`:** extract the shared three-tier merge
(contiguous → LCS → midpoint) exactly as upstream did — parakeet's TDT `duration`
lane rides through, byte-identical to existing parakeet goldens (they gate the
refactor).
**D.4 Routing (locked #5):** pure function; hypotheses top-3 walk; `iw→he`/`jw→jv`;
Parakeet declares its 25 codes (list in `88ac816ec`:
`bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro ru sk sl es sv uk`).
**D.5 Caption segmenter guard** (`c7ab8d6b2`): `endsSentence` — a sentence-final
period does not end a caption when the word starts with a capital/digit, doesn't end
in `".."`, and has < 3 chars outside capitals/digits/periods (`Dr.`, `U.S.`, `No.9`).
Mirror in `caption_segmenter.rs`; the Ruby twin `Captions::Segmenter` in ferment-api
is the cross-check.
**D.6 Note, don't implement:** upstream buried the transcript coverage gate
(`TranscribeSpeech` v4 — speech skip now happens at the languageId node). If
redshift's engine-side transcribe path carries a `CoverageGate`, remove it there;
if the gate lives in `crates/project`, leave it and flag for the lead.

**Acceptance:** AmberNet + GigaAM ORT oracles with committed goldens (semantic
pattern: tensor-level in/out pairs from the stock origin); routing table tests incl.
the be↔ru neighbour-confusion case (Russian IDed «be 0.885 / ru 0.113» must still
transcribe); seam-merge parity via existing parakeet goldens; segmenter guard tests
mirroring the Ruby twin's; a Russian E2E smoke (speech fixture → GigaAM transcript)
gated like `parakeet_transcribes_speech_end_to_end`.

---

## Track E — Diarization: Pyannote + ReDimNet

iOS: `Processors/Pyannote/` (938 LOC), `Processors/ReDimNet/` (269 LOC); commits
`8b825e5e6`, `d0a690c86`. **Cleanest ports in the round — no Accelerate, committed
goldens, documented parity.** FluidAudio (Apache-2.0) credit rides along — mirror
the license notice. **Boundary:** new `pyannote/` + `redimnet/` modules + a
diarization capability trait in `engine-core` (shape it after `SpeechTranscribing`);
ends at the typed result (locked #11 — no project fold, no cook glue).

**E.1 Pyannote model:** the ungated ONNX already exists —
`onnx-community/pyannote-segmentation-3.0` (the same weights the iOS parity script
scores against; `stage.py` guards output `[1,589,7]` — the community-1 repo ships an
8-class impostor under a similar name; keep that guard). No export work: qualify,
pin sha, stage for CDN.
**E.2 Port the pure logic:** `SegmentationWindows` (`sampleRate 16000`,
`windowSamples 160_000`, `windowFrames 589`, `frameStride 270`, `frameSamples 991`,
`stepFrames 294`; zero-pad last chunk, pad frames dropped by count; overlapping
frame times are deliberate — `end` is not the next `start`), `PowersetDecoder`
(7-class table `[[],[0],[1],[2],[0,1],[0,2],[1,2]]`; activity/marginals/confidence
views; graph outputs **log**-probabilities — exponentiate), `TurnAssembler`
(`minimumTurnSeconds 0.20`, `minimumGapSeconds 0.20`; gaps close first, then short
runs drop; never close leading/trailing; overlap flag smoothed first),
`LocalSpeakerStitcher` (average-linkage cosine, threshold **0.45**; vectorless
candidates attached by shared seconds; per-frame marginal/coverage ≥ 0.5; final
fingerprint = one pool over ALL clean speech per voice; two-phase embedding:
per-candidate probe then per-voice pool).
**E.3 ReDimNet:** export `b2-vox2-lm` (MIT, PalabraAI/redimnet2) to ONNX — trace the
pinned upstream repo; **mel frontend must stay fp32** (fp16 overflows on ordinary
speech: the mel tensor needs 8.9e13 of range vs fp16's 1.1e12); simplest correct
choice: whole-graph fp32 (~16 MB) unless fp16-with-fp32-mel is easy. **Fixed
32,080-sample window (2.005 s), never padded**; port `windows(for:)`
(round-not-truncate; never cross segment boundaries; sort result by offset;
`minimumSpanSeconds = 2.005`).
**E.4 Goldens:** copy the committed upstream fixtures —
`Tests/Fixtures/Pyannote/{one-speaker,turn-taking,overlap}/` (ONNX fp32 references;
parity bar: ≥ 99.9% frame decisions) and `Tests/Fixtures/SpeakerEncoder/*.wav` +
`.embedding.bin` (torch goldens; **cosine ≥ 0.999 per window**, window count exact,
times to 1e-9). WAV only — that is deliberate (decode-path isolation).

**Acceptance:** both parity suites green through the ORT backend at the upstream
bars; stitcher/assembler/decoder unit suites ported; the diarization result type
documents its shape in the report for the lead's cook mapping.

---

## Track F — VideoSaliency: pure math + substitute ONNX backing

Upstream is Apple Vision — no neural core of ours. **Dimitri's ruling: we port it
for real**, with a substitute backing. Vision's exact numbers are unreachable; the
structure, coordinates and reduction math are ported exactly, the heat/face sources
are swapped, and every constant that was calibrated against Vision gets re-measured.
**Boundary:** new `video_saliency/` module + traits (`SaliencyHeatSource`,
`FaceRectSource` — engine-core) + ONNX backings + exports. No graph wiring.

**F.1 Port the pure math exactly** (iOS `Processors/VideoSaliencyAnalyzer/`, 455 LOC):
`VideoSaliencySamplePlan` (`samplesPerSecond 0.5`, `minSamplesPerScene 3`,
`maxSamplesPerScene 32`, `interior 0.1`, seek tolerance `min(0.25,
interior·sceneDuration)`; scenes < 0.4 s get one midpoint grab), heat measurement
(`heatFloor 0.3×peak` pedestal subtraction → intensity-weighted centroid → ±2σ
extent clamped → confidence = mass share in `centroid ± 0.15`; empty map →
`(0.5, 0.5, 1, 1, 0)`), `unitClipped` (intersect ONLY on escape — a box already
inside comes back untouched), `dominantFace` (largest area; not union, not first),
`VideoSaliencyReduce` (lower-median per scene over `cx, cy, w, h, confidence`;
`faceShare = faces/count`; `spread`; all-failed scene → centered full-frame zero).
Coordinates: normalized, **top-left origin**, row 0 = top. `maxLongEdge 512` for
the frame handed to the heat source; result `frameWidth/Height` = display-oriented
source dimensions.
**F.2 Backings:** salient-object model — default candidate **U²-Netp**
(Apache-2.0, ~4.7 MB); face detector — default candidate **YuNet** (MIT, ONNX
available). You may substitute better-qualified models of compatible license —
justify in the report. Export/qualify via the standard harness
(`inspect_onnx.py` opset/portability gate, goldens, CPU-EP baseline per locked #10).
**F.3 Recalibration:** `heatFloor 0.3` and the confidence window were measured
against Vision's pedestal (~0.06 mean vs 0.85 peak). Measure the substitute model's
pedestal on a small committed frame set; if the pedestal differs materially, derive
the equivalent floor, keep the iOS constant as the documented reference, and put the
measurement table in your report — the lead reviews the final constants.

**Acceptance:** math unit suites ported from the 9 Swift test files (754 LOC, incl.
the A13 escape-clip cases `VideoSaliencyDominantFaceTests`); ONNX oracles for both
backings (reference outputs committed); an end-to-end test over a synthetic clip
(known bright subject on flat background → centroid/extent within tolerance);
pedestal measurement recorded.

---

## Track G — Video encoder: deterministic rate control + variants engine side

iOS commits `ca4f3e6c2`, `1d98ba700`, `203deb5d9`, `11b25b977`;
`Protocols/VideoEncoding.swift`.
**Boundary:** `video_encoder.rs`, `encode_pipeline.rs`, engine-media platform
encoder settings. The `videoVariants` operator consolidation and artifact naming
are wiring (lead); you deliver the encode capabilities it needs.

- **`VideoRateControl { Average(u64), BitsPerPixel(f64) }`**;
  `resolved_bit_rate = bpp × w × h × fps`, fps fallback **30** when the track
  reports 0/non-finite. Presets: **proxy = bpp 0.75, short-edge cap 1080
  (`caps_short_edge`), bFrames 0, keyFrameInterval 1 (all-intra), HDR-eligible**;
  **ai = average 2.5 Mbps, maxWidth 640, keyFrameInterval 60, SDR, mp4, frame
  reordering OFF explicitly** (the bounded-GOP preset would silently regain
  reordering, which the scene detectors cannot survive). No quality-mode probes —
  determinism comes from computing the rate, not requesting a quality.
- **HDR split:** HDR-preserving variants (proxy + thumbstrip) share one decode; SDR
  variants encode **from the finished proxy** (second-generation is accepted);
  single-variant SDR-from-HDR tone-maps at the reader.
- **Thumbstrip:** the multi-variant encode composes it from the same decoded frames
  (`withThumbStrip` upstream) — this resolves the July "thumbstrip container format
  not ported" divergence; port the container format from
  `EncodeVideoProcessor`/FermentEditor.
- Map the presets onto AudioToolbox/MF settings and document any knob a platform
  can't honor (in the report, with the measured fallback).

**Acceptance:** rate-control resolution unit tests (incl. fps fallback), preset
geometry tests (portrait 4K → short-edge 1080, was 606-wide under the long-axis
bug), reordering-off asserted in encoder-settings tests, thumbstrip format tests;
`encode_pipeline_e2e` extended to the multi-variant shape.

---

## Track H — Oracle & gate hardening — independent, dispatch anytime (ideally first)

From the lead's zero-trust pass over the existing ports (verified 2026-08-31):

1. **19 orphaned oracles:** `just engine-oracle` never runs `--test ort_semantic`
   (7 parakeet/dasheng/tagger parity), `--test ort_enhance` (4), `--test
   ort_behavioral` (4), `--test behavioral_from_dumps` (4). Add them (semantic +
   enhance unconditionally under the env guards; behavioral where dumps exist).
2. **~31 vacuous fixture tests** silently pass when `../ferment-cuts-ios` is absent
   (`let Some(x) = read_fixture(…) else { return; }`). Convert to the
   `waveform_extractor.rs:111` SKIP-print pattern, and add one canary test that
   **fails** when `FERMENT_ENGINE_FIXTURES_DIR` is set but any fixture read failed.
3. **CI runs no `cargo test`.** Add a workspace `cargo test` job (excluding
   `#[ignore]`/env-gated suites) to `.github/workflows/ci.yml`, and the browser
   Vitest project while in there. Watch CI minutes; if the workspace build is too
   heavy, at minimum `-p ferment-engine-processors -p ferment-sequencer
   -p ferment-processorman -p ferment-project`.
4. **Loudness normalizer has zero reference coverage** (906 LOC): add EBU R128
   conformance-vector tests (the spec ships test signals).
5. Stale doc: `fleet.rs:46` "all 17 inference identifiers" (there are 22, and this
   round adds more). Fix while touching the manifest.
6. `caption_segmenter::next_short_id` is a process-global counter → caption/word IDs
   are not deterministic per re-cook. iOS moved to `FermentID.short()`. Investigate
   what shape/stability the IDs need, align the generator, report the finding.

**Acceptance:** `just engine-oracle` green locally with all suites listed; CI green;
canary demonstrably fails when a fixture path is broken.

---

## Model delivery (shared reference for C/D/E/F)

- New ModelId variants + `fleet.rs` definitions with `license` + `source_url`
  (validation enforces both): `dinov2_vits14` (Apache-2.0), `ambernet_langid`
  (NGC/NVIDIA — record the NGC terms), `gigaam_*` ×4 + vocab (verify SberDevices
  license), `pyannote_segmentation` (MIT weights; FluidAudio Apache-2.0 credit),
  `redimnet_speaker` (MIT), saliency + face models (Track F). `cargo deny check
  licenses` must stay green — if a license fails the policy, stop and report
  (do not vendor around it).
- Export pipelines live under `scripts/exports/<family>/` following the
  dasheng/parakeet pattern (`export*.py` + `gen_goldens.py` + pinned deps);
  run artifacts through `inspect_onnx.py` (opset ≤ 20, portability gate) and
  `to_fp16.py` where fp16 is chosen.
- Masters stage under `dist/models-publish/` with files + shas listed in the report —
  the CDN upload to `https://public.lockvoid.com/ferment/studio/models` is Dimitri's
  step.
- `MODEL_FLEET_VERSION` bumps once, in the last-landing track that adds models.

## Dispatch order

```
H (gates) ─ first, so B–G land against honest gates
B (beats) · C (scenes+DINOv2) · D (speech) · E (diarization) · F (saliency) · G (encoder)
   — mutually independent, parallel-safe (disjoint modules; C and G share no files)
```

Every track gets a real lead review; after review, the lead runs the integration
pass (`2026-08-31_engine-cross-port-integration.md`) that wires the landed pieces
into `crates/project`.
