# 2026-09-01_02 — W5 executor report: per-model compute placement

Three staged commits, all gates green, and one finding the dispatch doc did not
anticipate: **the GPU question answers itself, because on this hardware Core ML
will not take these graphs at any permission level.** No promotions to sign.

| stage | commit | what landed |
|---|---|---|
| 1 | `c27f00b9` | the placement axis on the manifest; `CoreMlBackend` per model; iOS pins + the 016 EP matrix as data |
| 2 | `b9fb0061` | per-model ORT provider chains; Windows off blanket-CPU; the DINOv2 experiment seam |
| 3 | `0f51aab2` | the M3 measurement, both lanes, and the harness rule that keeps it honest |

---

## 1. What the axis is

`ModelArtifact` grew one field: `placement`. Core ML compute units on the
mlmodelc lane, execution-provider preference on the ORT lane. Two validation
rules, both mirroring what the integrity fields already do — a shipped artifact
**must** declare a placement, a `Missing` one **must not**, and a placement from
the wrong lane is refused.

Placement never moves a model between lanes. `RoutingBackend` still picks the
lane by asking which backend has a real artifact; placement only decides where
the graph runs once it is there.

The reason it is per *artifact* rather than per model is BeatThis, which proves
the case on its own: iOS pins the device build (palettized fp16) to the ANE and
the Simulator build (fp32) to the GPU, because on that family the units are a
property of the weights. Two artifacts, two pins, one model.

## 2. The Core ML baseline — iOS, verbatim

Every family Studio carries a real `.mlmodelc` for, at the platform a Mac
selects. Pinned as data in `fleet.rs`'s tests, so moving a graph off its
measured unit is a deliberate edit someone reviews.

| model | units | model | units |
|---|---|---|---|
| beat_this_1/2/3 | cpuAndNeuralEngine | music_highlighter | cpuOnly |
| parakeet_preprocessor | cpuOnly | dasheng_base | cpuAndNeuralEngine |
| parakeet_encoder | cpuOnly | music_tagger_heads | cpuOnly |
| parakeet_decoder | cpuOnly | autoshot | cpuAndNeuralEngine |
| parakeet_joint | cpuOnly | fireredvad_aed | cpuOnly |
| dfnet | cpuOnly | | |

Plus the Simulator lane, where BeatThis keeps its own `cpuAndGpu` pin.

**Two deliberate divergences**, both commented where they bite:

- **`ChipGeneration` is not ported**, per ruling 1. iOS drops AutoShot and
  BeatThis off the ANE below chip 14; Studio is measured on an M3 (chip 17), so
  every gate evaluates to the same branch. A gate nothing can exercise is worse
  than no gate — it reads as coverage. The comment in `fleet.rs` says so and
  says what would bring it back.
- **Only families with a real `.mlmodelc` carry units.** iOS pins DINOv2 to the
  ANE; Studio has no Core ML DINOv2 artifact at all — its visual encoder is the
  ONNX master on every platform. The pin has nowhere to land, which is exactly
  why DINOv2 became stage 3's ORT experiment instead.

## 3. The Windows half — 016's matrix, finally encoded

`migration/016-report.md` §4.1/§4.2 as data. DirectML only where 016 measured it
and it passed; CPU everywhere else, including for every family 016 never put on
a GPU box at all.

| ORT placement | models |
|---|---|
| `DirectMl` | beat_this ×3, autoshot, fireredvad_aed, realesr ×3, fastdvdnet, rife_v4_25 |
| `Cpu` | everything else — 21 masters |

The five first-party promotions are the behavioural verdicts, not the raw ones:
beat_this's raw drift on DirectML is 0.752 against any sane logit tolerance, and
its beat grid is still identical to CPU's within one analysis frame. That was
016's headline and it is what the pin encodes.

**One thing the lead should look at.** Ruling 3 says "CPU where CUDA/GPU
corrupted (`music_highlighter`) or hard-failed (`dfnet`)", and that is what I
encoded. The source report is narrower: §4.2 gives `music_highlighter` a
DirectML behavioural **pass** ("window + top-chunk identical (live)") and
`dfnet` a DirectML **pass** outright, and concludes "on the CUDA-primary chain,
`music_highlighter` and `dfnet` pin to DirectML/CPU". So 016 would allow both on
DirectML and bar only CUDA; the ruling bars both accelerators.

I followed the ruling, and I think it is the better call anyway —
`music_highlighter`'s DirectML raw drift (0.0074) is 5× its own tolerance and
the pass rests entirely on a behavioural check of the top chunk — but it is a
tighter pin than the report it cites, and that is the lead's to confirm. If you
want the looser reading it is a two-word edit in `fleet.rs`.

Also worth recording: **nothing in this workspace can select CUDA today**. The
placement vocabulary has no CUDA variant, because 016 found no model that CUDA
both ran and ran correctly, and the Rust `ort` + CUDA-DLL hang is still open.
CUDA remains implemented in `OrtBackend` and unreachable from the manifest.

## 4. The measurement

`crates/engine-infer/tests/coreml_placement.rs` and `ort_placement.rs`, both
`#[ignore]`d, both against the real artifacts on the M3 Max, both under the
env-var contract the other oracles use.

Inputs are fixture bytes wherever one exists — the Swift DeepFilterNet trace's
102 real frames at the head of the graph's fixed 1500, the highlighter's mel
chunks tiled the way the processor tiles them, real 16 kHz speech PCM for the
Parakeet chain and the VAD, and for the tagger heads a real Dasheng embedding
computed from that same PCM. 016's RIFE finding — 61.7 dB on natural frames,
11 dB on synthetic noise, same model, same provider — is why.

### 4.1 The eight cpuOnly families under `.cpuAndGPU` (median of 31)

`.cpuAndNeuralEngine` sits beside it as a **control**, not a candidate.

| model | cpuOnly | GPU diff | GPU | GPU speedup | ANE diff | ANE | ANE speedup | verdict |
|---|---|---|---|---|---|---|---|---|
| parakeet_preprocessor | 2.11 ms | 0 | 2.20 ms | 0.96x | 0 | 2.17 ms | 0.97x | hold — CPU never left |
| parakeet_encoder | 34.64 ms | 0 | 34.59 ms | 1.00x | **2.5e-2** | 27.16 ms | **1.28x** | hold — CPU never left |
| parakeet_decoder | 0.17 ms | 0 | 0.24 ms | 0.73x | 0 | 0.16 ms | 1.05x | hold — CPU never left |
| parakeet_joint | 0.31 ms | 0 | 0.33 ms | 0.95x | 0 | 0.38 ms | 0.83x | hold — CPU never left |
| dfnet | 350.92 ms | 0 | 347.47 ms | 1.01x | 0 | 358.36 ms | 0.98x | hold — CPU never left |
| music_highlighter | 4.43 ms | 0 | 4.15 ms | 1.07x | 0 | 5.26 ms | 0.84x | hold — CPU never left |
| music_tagger_heads | 0.06 ms | 0 | 0.06 ms | 1.02x | 0 | 0.06 ms | 1.00x | hold — CPU never left |
| fireredvad_aed | 4.79 ms | 0 | 5.25 ms | 0.91x | 0 | 4.98 ms | 0.96x | hold — CPU never left |

**Every GPU column is bit-identical to CPU.** That is not "the GPU agrees".
`MLComputeUnits` is a permission, not an instruction — Core ML's planner takes
the operators it wants and leaves the rest — and these are FP32 conv and
transformer stacks that no two devices would ever agree on *exactly*. An exact
zero means the planner declined the unit and ran on CPU, and the wall-time
figure beside it is CPU scheduling variance.

The control is what licenses that reading. On the ANE the Parakeet encoder
drifts 2.5e-2 and gains 1.28x — the harness sees engagement when it happens. So
zero really does mean zero.

**This distinction decides the whole stage, and it caught a wrong answer.** The
first version of this harness (5 iterations, no engagement rule) promoted
`parakeet_decoder` at 1.92x and `parakeet_joint` at 1.30x. At 21 iterations
those fell to 1.30x and 0.94x, and `parakeet_preprocessor` measured 1.17x on one
run and 1.37x — over the bar — on the next. All of it was sub-millisecond
scheduling noise on graphs whose output never moved a bit. The promotion rule
now asks "did the graph go anywhere else" *before* it asks about parity or
speed, and the verdict column has been stable across every run since.

### 4.2 DINOv2 on the ORT Core ML provider

The one real ANE gap against iOS. Scored on cosine distance, because
`Dinov2SemanticEncoder::cosine_distance` is what decides whether two frames are
the same shot; an absolute drift that leaves the unit vector pointing the same
way costs nothing.

| model | cosine distance | max abs diff | oracle | CPU EP | CoreML EP | speedup | verdict |
|---|---|---|---|---|---|---|---|
| dinov2_vits14 | 1.192e-7 | 2.906e-7 | MATCH | 11.9–12.2 ms | 13.1–19.1 ms | 0.62–0.94x | hold |

Numerically clean and reproducible to the digit across three runs. Slower every
time — the provider *is* engaged (the drift is small but non-zero, and the wall
time moves), it just does not pay for the transfer on a graph this size. The
placement stays CPU.

For the record, this also retires a blocker: 016 §5.6b recorded Rust `ort`
hanging on macOS with the Python-wheel dylib. It does not hang now — the CPU and
Core ML providers both ran clean against the vendored 1.24.4 dylib.

### 4.3 The verdict

**No promotions.** iOS's cpuOnly pins hold on desktop silicon — for a reason iOS
could not have known. The dispatch doc expected those pins to encode two things,
ANE-hostility (transfers) and iPhone thermals (does not). The measurement says
neither is the operative fact here: Core ML's planner will not take these graphs
on this hardware at all, so there is no thermal question to transfer. The pins
are right, and now they are right for a recorded reason.

The one door left ajar is the **Parakeet encoder on the ANE**: 1.28x for 2.5e-2
of drift on encoder states spanning roughly ±1.6. That is outside this track's
question (the ruling asks about `.cpuAndGPU`) and it is not obviously a good
trade — iOS's own note on the sibling GigaAM encoder is that ANE specialization
"costs more than it repays at this predict rate". Recorded, not acted on.

## 5. Gates

| gate | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --all-targets` | clean |
| `cargo clippy` over the three touched crates with `ort` | clean |
| full workspace tests (CI's exclusion set) | green |
| `coreml_fleet` --ignored | 1 passed |
| `coreml_processors` --ignored | 20 passed |
| `beat_analyzing_pipeline` --ignored | 3 passed |
| `ort_fleet` --ignored | 3 passed |
| `ort_processors` --ignored | 2 passed |
| `ort_semantic` --ignored | 16 passed |
| `scene_detection_regression` --ignored | 3 passed |
| `ort_diarization` --ignored | 7 passed |

New tests: 8 in `engine-models` (the pinned baseline tables, the Simulator
lane, the gap rule, three validation rules), 4 in `engine-infer`, 2 in
`project` (the backend the runtime actually builds, both lanes).

## 6. Two things found on the way

**`ort_saliency` does not compile — pre-existing, outside this boundary.**
`cargo test -p ferment-engine-processors --features ort --test ort_saliency`
fails with five `missing field 'embedding' in initializer of VideoScene`.
Verified against a clean stash: it is broken at `HEAD` before any of these
commits. CI does not catch it because the workspace clippy run does not enable
`ort`, but `just engine-oracle` runs that suite, so the recipe is red today for
a reason unrelated to placement. `crates/engine-processors` is outside W5's
boundary; flagging rather than fixing.

**The Core ML oracles do not exercise the pins.** `coreml_processors.rs` builds
its backend with `CoreMlBackend::new(...)`, which is `ComputeUnits::All` — so
after stage 1, production runs the pinned units and the golden oracle runs the
planner's choice. On today's numbers this is harmless (§4.1 shows the planner
lands on CPU for the cpuOnly families either way, and `All` is what those
goldens were taken under), but it is a genuine coverage gap: a future pin change
would not be caught by the suite that exists to catch it. The fix is one line —
have that harness read `default_fleet_manifest().placements(...)` the way
`runtime.rs` does — and it lives in `crates/engine-processors`, outside this
boundary. Recommend it as the first item of whatever track owns that crate next.
