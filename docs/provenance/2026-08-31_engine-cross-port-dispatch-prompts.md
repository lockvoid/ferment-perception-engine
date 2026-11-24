# Dispatch prompts — engine cross-port tracks B–G

Track H (gates) is landed and reviewed; B–G now land against honest gates.
Each block below is self-contained — paste one per agent. They are
**mutually independent and parallel-safe** (disjoint modules, no shared files).

Every prompt already carries the two things a fresh agent cannot infer: the
zero-trust rule, and what Track H changed under them.

---

## Shared preamble (already inlined in each prompt — repeated here for reference)

> Read `docs/fleet/2026-08-31_engine-cross-port.md` in full first: the SCOPE
> boundary and the Executor protocol bind you. Then read your track's section.
>
> **Zero-trust:** every constant and signature in that doc was read from source
> on 2026-08-31, but re-verify against the referenced file before coding. If the
> source disagrees, the source wins — say so in your report.
>
> **Track H landed (`docs/fleet/2026-08-31_engine-cross-port-h_report.md`) — read
> its "What B–G inherit" section.** In particular:
> - Fixture-backed tests go through `ferment-engine-test-fixtures`
>   (`ferment-engine-test-fixtures.workspace = true` under `[dev-dependencies]`;
>   `fixtures::f32_le/require_f32_le/bytes/path/skip`, paths relative to
>   `Tests/Fixtures`). Never hand-roll a fixture path — a configured
>   `FERMENT_ENGINE_FIXTURES_DIR` that cannot serve a fixture now panics, on
>   purpose. Add fixtures your suite cannot run without to `CANARY_ENTRIES`.
> - Any new `--test` oracle target must be added to `just engine-oracle`, or it
>   is not a gate.
> - Ids come from `ferment_utils::id` (`short`, `long`, `short_seeded`). Do not
>   write another base62 encoder.
> - CI now runs `cargo test --workspace` plus the WebKit browser suite, so a
>   broken test fails the PR.
>
> **Do NOT touch** `crates/project`, `apps/`, `packages/`, or any
> network/cook/graph wiring — that is the lead's integration pass. If your port
> seems to need a `crates/project` change, stop, leave the seam clean, and record
> it in your report.
>
> **Heads-up:** the ReplicaMan track is working in the same tree
> (`crates/{replicaman,ferment-doc,replica-models}`, `crates/processorman/src/v5/`).
> Those are not yours. `cargo deny check licenses` and `ferment-replicaman`'s
> build are currently red for reasons that belong to that track — do not try to
> fix them, and do not let them block you.
>
> Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` (1.95.0),
> your crate's tests green, root `pnpm test` untouched. No Prettier.
>
> Finish by writing `docs/fleet/2026-08-31_engine-cross-port-<track>_report.md`:
> what you did, every deviation, judgment calls, test evidence (names + counts),
> and everything the lead needs for wiring (new public types, trait impls, output
> shapes, model IDs). A track without a report is not done.

---

## Track B — Beats overhaul

```
Execute Track B of docs/fleet/2026-08-31_engine-cross-port.md (Beats overhaul,
crates/engine-processors).

[paste shared preamble]

Your section: "Track B — Beats overhaul". B.1 DBN (madmom parity, exact frame
indices, committed fixtures at Tests/Fixtures/BeatThis/dbn-parity-{logits,expected}.json),
B.2 lattice segmentation, B.3 EnergyStructure + energyPerBar, B.4 grid/offset-0 +
anti-aliased SRC audit + BeatQueenMary silence trim, B.5 AudioMinimap bar ruler.

Locked, do not relitigate: the DBN+lattice path is built ALONGSIDE
HybridBeatDetector — do not delete or rewire hybrid (the lead swaps consumers
later). Beat grid offset = 0. BeatThis ensemble = 2.

Note EnergyStructure.swift:70-74's doc-comment is stale — port the code, not the
comment.

Biggest risk: B.4's resampler. If redshift's resample path is linear
interpolation, replacing it with a polyphase/windowed-sinc resampler changes mel
inputs — re-run the BeatThis mel fixture tests and report the measured delta.
License must clear `cargo deny`.
```

## Track C — Scene detection + DINOv2

```
Execute Track C of docs/fleet/2026-08-31_engine-cross-port.md (Scene detection +
DINOv2).

[paste shared preamble]

Your section: "Track C". C.1 jump-cut rewrite (pure luma), C.2
PhotometricTransientScanner + TransientVerdict + BlockMotionEstimator changes,
C.3 DINOv2 ViT-S/14 export + encoder + EncoderShotIdentity + SceneEmbeddingTap +
per-scene medoid embedding.

Your processor keeps its current input surface but must accept 640-wide frames,
and the tests must cover that geometry — the "scenes read the 640-wide ai feed"
change itself is the lead's wiring.

Known risk to measure and report, not to hide: DINOv2 pre-processing uses a
high-quality resample that was vImage on iOS. Pick a Lanczos-class Rust resizer,
pin it with fixture tests, and report the measured embedding drift versus the
iOS reference. Input is 224x126 non-square with ImageNet norm baked into the
graph.

If the strobe-cuts.mp4 regression fixture (scene starts 39/189/337) is obtainable
from the iOS fixtures archive, port that regression too.
```

## Track D — Speech: AmberNet + GigaAM + routing

```
Execute Track D of docs/fleet/2026-08-31_engine-cross-port.md (Speech system).

[paste shared preamble]

Your section: "Track D". D.1 AmberNet language ID (+ committed
ambernet_langs.json), D.2 GigaAM Russian ASR (check SberDevices for an official
ONNX export before converting yourself), D.3 shared TokenSeamMerge extraction,
D.4 routing, D.5 caption segmenter abbreviation guard.

Locked routing (#5): GigaAM(ru) -> Parakeet(its 25) -> unsupported. No Apple
lane, no chip axis. Legacy tags iw->he, jw->jv. Top-3 hypothesis walk. AmberNet
has NO coverage threshold — trust AED.

D.3 is a refactor under existing goldens: parakeet's TDT `duration` lane must
ride through byte-identical to the current parakeet goldens. Those goldens are
the gate on the refactor — if they move, you broke it.

D.5 has a Ruby twin (Captions::Segmenter in ferment-api) — cross-check against
it. Note caption ids now come from ferment_utils::id::short() (Track H).

D.6 is investigate-and-report only: if an engine-side CoverageGate exists in the
transcribe path, remove it; if it lives in crates/project, leave it and flag it.
```

## Track E — Diarization: Pyannote + ReDimNet

```
Execute Track E of docs/fleet/2026-08-31_engine-cross-port.md (Diarization).

[paste shared preamble]

Your section: "Track E". The cleanest port in the round: no Accelerate, committed
goldens, documented parity bars.

E.1 qualify onnx-community/pyannote-segmentation-3.0 (KEEP the [1,589,7] output
guard — the community-1 repo ships an 8-class impostor under a similar name).
E.2 port SegmentationWindows / PowersetDecoder (graph emits LOG-probabilities,
exponentiate) / TurnAssembler / LocalSpeakerStitcher. E.3 export ReDimNet
b2-vox2-lm — the mel frontend MUST stay fp32 (fp16 overflows on ordinary speech);
whole-graph fp32 (~16 MB) is the simplest correct choice.

Parity bars are not negotiable: Pyannote >= 99.9% frame decisions; ReDimNet
cosine >= 0.999 per window, window count exact, times to 1e-9.

The upstream fixtures (Tests/Fixtures/Pyannote/{one-speaker,turn-taking,overlap}/
and Tests/Fixtures/SpeakerEncoder/*.wav + .embedding.bin) are present in the
sibling checkout — read them through ferment-engine-test-fixtures and add them to
CANARY_ENTRIES. WAV-only is deliberate (decode-path isolation) — do not "improve"
it to mp3.

Ends at the typed result (locked #11): no project fold, no cook glue. Add the
diarization capability trait in engine-core shaped after SpeechTranscribing, and
document the result shape in your report for the lead's cook mapping. Mirror the
FluidAudio (Apache-2.0) credit.
```

## Track F — VideoSaliency

```
Execute Track F of docs/fleet/2026-08-31_engine-cross-port.md (VideoSaliency).

[paste shared preamble]

Your section: "Track F". Upstream is Apple Vision; Dimitri's ruling is that we
port it for real with a substitute backing. The structure, coordinates and
reduction math are ported EXACTLY; the heat/face sources are swapped behind
traits (SaliencyHeatSource, FaceRectSource in engine-core).

F.1 port the pure math exactly — sample plan, heat measurement, unitClipped
(intersect ONLY on escape), dominantFace (largest area, not union, not first),
VideoSaliencyReduce (lower-median). Coordinates are normalized with a TOP-LEFT
origin, row 0 = top.

F.2 backings: U2-Netp (Apache-2.0) and YuNet (MIT) are the default candidates.
You may substitute better-qualified models of compatible license — justify in the
report. CPU-EP baseline per locked #10; do not block on GPU EPs.

F.3 is the part that needs judgment: heatFloor 0.3 and the confidence window were
calibrated against Vision's pedestal (~0.06 mean vs 0.85 peak). Measure your
model's pedestal on a small committed frame set. If it differs materially, derive
the equivalent floor, KEEP the iOS constant as the documented reference, and put
the measurement table in your report — the lead signs off the final constants.

Port all 9 Swift test files (754 LOC), including the A13 escape-clip cases in
VideoSaliencyDominantFaceTests.
```

## Track G — Video encoder

```
Execute Track G of docs/fleet/2026-08-31_engine-cross-port.md (Video encoder:
deterministic rate control + variants).

[paste shared preamble]

Your section: "Track G". VideoRateControl { Average(u64), BitsPerPixel(f64) },
resolved_bit_rate = bpp * w * h * fps with fps fallback 30 when the track reports
0/non-finite.

Presets: proxy = bpp 0.75, short-edge cap 1080 (caps_short_edge — note the July
long-axis bug made portrait 4K come out 606 wide; the geometry test must cover
it), bFrames 0, keyFrameInterval 1 (all-intra), HDR-eligible. ai = average
2.5 Mbps, maxWidth 640, keyFrameInterval 60, SDR, mp4, and frame reordering OFF
**explicitly** — the bounded-GOP preset silently regains reordering, which the
scene detectors cannot survive. Assert reordering-off in the encoder-settings
tests.

Determinism comes from computing the rate, never from requesting a quality mode.
No quality-mode probes.

HDR split: HDR-preserving variants (proxy + thumbstrip) share one decode; SDR
variants encode FROM the finished proxy (second-generation accepted);
single-variant SDR-from-HDR tone-maps at the reader.

Thumbstrip: compose it from the same decoded frames (withThumbStrip upstream).
This resolves the July "thumbstrip container format not ported" divergence — port
the container format from EncodeVideoProcessor/FermentEditor.

Document any knob a platform cannot honor, with the measured fallback, in your
report. The videoVariants operator consolidation and artifact naming are the
lead's wiring — you deliver the capabilities.
```

---

## Dispatch order

H is done. B · C · D · E · F · G are parallel-safe and can all go at once.
Each gets a real lead review before the integration pass
(`2026-08-31_engine-cross-port-integration.md`).
