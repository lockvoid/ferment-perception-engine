# 016 — Enhancement Model Fleet: Final Execution Report

**For:** Dimitri (lead audit) · **From:** Claude (executing agent) · **Date:** 2026-07-19
**Companion:** `016-notes.md` is the chronological execution log with every command and raw number; this report is the synthesis you read start-to-finish.

> **Lead audit (2026-07-20): PASS — committed.** Line-level review of frame_tensors, all three backings, the manifest provenance, and both behavioral oracles; gates re-run clean (fmt, clippy incl. `--features ort`, 699 crate / 1365 workspace tests). **Sign-offs granted:** behavioral gate values (beat ≤ 20 ms = one 50 fps frame; scene-cut lists exact; VAD ≥ 0.995; highlight top-chunk + window exact) · RIFE natural gate PSNR ≥ 45 dB · GPU enhance-golden bound ±12 u8 **as provisional** via `FERMENT_ENHANCE_MAX_U8` with printed diffs as the record · synthetic-RIFE scoped CPU-only (superseded on GPU by the natural gate). **Audit deltas:** export intermediates (source `.7z`, extracted fp32s, the unwired v1 RIFE fp16) gitignored — the in-repo set is the five wired masters + goldens + natural clips (~34 MB). **Follow-ups recorded:** encode the per-model × per-EP ship matrix as EP allowlists in the dump oracles (music_highlighter × CUDA is *expected*-fail today); shape-assert enhance output tensors; the Rust `ort`+CUDA-DLL hang stays an open engineering item (python + offline behavioral cover the column).

---

## 1. Executive summary

016 set out to give Windows and web the enhancement capabilities macOS gets from VideoToolbox: super-resolution, temporal denoise, and slow-motion, backed by three third-party ONNX models behind the engine's existing capability seams.

**The track is executed to completion of everything that can execute on today's infrastructure.** All three models are exported, license-verified, golden-validated on CPU and DirectML, registered in the manifest with enforced provenance, and wired behind the three capability seams with the Rust integration path proven byte-perfect. The ten-master fleet has a fresh, fully-measured CPU + DirectML matrix. The CUDA execution provider is implemented with a fail-loud fallback chain — and empirically pinned as *unusable on the current box's GPU generation* (a hardware fact, not a code gap).

What remains is exclusively **decisions and hardware**: a Turing+ GPU instance if you want the CUDA column, per-EP tolerance policy for the four DirectML-gated first-party models, natural-frame goldens for RIFE, and the held models awaiting your D-decisions and weights outreach.

Nothing in this report is estimated or assumed: every number below came from an executed run, and the two places where a "pass" would have been easy to fake (CUDA on Pascal, RIFE on synthetic input) are reported as the failures they are.

---

## 2. What shipped (code, all uncommitted for your audit except §2.1)

### 2.1 Already folded into your commits (segment 1)
- **Manifest provenance** (`engine-models`): `license` + `source_url` on `ModelArtifact` and `ResourceDefinition`; `validate_provenance()` rejects any shipped artifact missing either — the "no third-party artifact without provenance" rule is enforced by the type system + validation, not convention. All ten first-party masters backfilled (`proprietary-internal` + `internal:scripts/engine_export_onnx.py`).
- **CUDA EP + fail-loud chain** (`engine-infer`): `OrtExecutionProvider::Cuda`; `windows_default()` = CUDA → DirectML → CPU; per-provider `error_on_failure`; `selected_provider()` telemetry; forced single provider never falls back. **Found and fixed a latent bug:** the pre-existing DirectML branch registered without `error_on_failure` — despite its comment, ort's default would have silently fallen back to CPU.
- **SR seam** (`engine-processors::video_upscaler`): `FrameUpscaling` trait (the previously-missing model seam, mirroring `TemporalNoiseFiltering`) + `PassthroughFrameUpscaler`.

### 2.2 In the working tree (segments 3–5, pending audit)
| Area | Files | Content |
|---|---|---|
| Model IDs | `engine-infer/src/model.rs` | 5 variants: `RealesrGeneralX4`, `RealesrGeneralWdnX4`, `RealesrAnimeVideo3`, `Fastdvdnet`, `RifeV425` (fleet 17→22) |
| Manifest | `engine-models/src/{fleet,lib}.rs` | `third_party_ort()` — each model registered with real SPDX license, upstream URL, sha256, and an explicit `Missing` Core ML entry ("macOS backing is VideoToolbox by design") |
| Shared packing | `engine-processors/src/frame_tensors.rs` (new) | RGB24 ↔ planar-f32 (plus edge-padded variant), quantization, and the capturing `MockBackend` test double |
| SR backing | `video_upscaler.rs` | `CompactFrameUpscaler` (`general` / `general_denoise` / `anime_video`, native ×4) |
| Denoise backing | `video_noise_filter.rs` | `FastDvdnetTemporalFilter` (2+2 window, upstream edge-replication, `FASTDVDNET_MAX_SIGMA = 55/255` strength mapping) |
| Slow-mo backing | `slow_motion_renderer/rife.rs` (new) | `RifeFrameInterpolator` (v2 interface, pad-to-32 + crop-back, one inference per phase, stateless) |
| Integration goldens | `tests/ort_enhance.rs` (new) | End-to-end u8 goldens through the real `OrtBackend` (CPU/DirectML/CUDA selectable) |
| Export harness | `scripts/exports/` | Reusable: `inspect_onnx.py` (golden-rule gate), `to_fp16.py` (with the Resize-scales fix), `onnx_cpu_smoke.py`, `ep_probe.py` (fail-loud EP replay incl. `disable_fallback`), `gen_rust_goldens.py`, per-model export dirs, box provisioning/runner scripts |

**Gates (all green):** `cargo fmt`; `clippy --all-targets -D warnings` on all three crates (with `ort`); `cargo check --workspace` (coexists cleanly with your in-flight 008 work); tests **engine-processors 631/631** (+12 new mock tests), **engine-infer 20/20**, **engine-models 14/14**.

---

## 3. The artifacts (all exported, hashed, golden-validated)

| Artifact | License (verified) | sha256 (fp16 ONNX) | Size |
|---|---|---|---|
| `realesr-general-x4v3.fp16.onnx` | BSD-3-Clause | `4852c5de…a9584` | 2.44 MB |
| `realesr-general-wdn-x4v3.fp16.onnx` | BSD-3-Clause | `02ea7f72…d7db3` | 2.44 MB |
| `realesr-animevideov3.fp16.onnx` | BSD-3-Clause | `649d6939…e60b45` | 1.25 MB |
| `fastdvdnet.fp16.onnx` | **MIT — LICENSE verified in-tree** (sha256 `a74a120e…`, vendored; confirms the research report's "GPL" was wrong) | `e7a80a50…36890` | 5.00 MB |
| `rife_v4.25_v2.fp16.onnx` | MIT (vs-mlrt / Practical-RIFE) | `91325975…5fb89` | 11.42 MB |

Export env (uv, pinned): Python 3.12, torch 2.9.0, onnx 1.22.0, onnxruntime 1.23.0, numpy 1.26.0.

**"×2" note:** no official Compact ×2 checkpoint exists; the ×2 product tier is the ×4 model + the seam's post-resample. Recorded so nobody hunts a phantom weight.

**RIFE interface decision:** I exported both vs-mlrt variants and wired the **v2** interface after *empirically proving* its semantics (`input[1,7,H,W] = img0+img1+timestep`: identity pair → 0.003; t=0 → img0; t=1 → img1). v2 also turned out dramatically more fp16-stable than v1 (79.4 dB vs 37 dB on CPU; 61.7 dB vs 39.6 dB on DirectML) — the semantic-certainty choice was also the quality choice.

---

## 4. The numbers

### 4.1 Enhancement models × EP (max abs diff / PSNR vs fp32 reference)

Final baseline measured on the **RTX 5000 (Turing) box**; the P4000 DirectML numbers are retained in
`016-notes.md` for the generation comparison (everything shifted up on Turing).

| Model | CPU | DirectML (RTX 5000, ort-dml 1.24.4) | CUDA (RTX 5000, ort-gpu 1.23 + cuDNN 9.8 pin) |
|---|---|---|---|
| realesr-general-x4v3 | 0.0017 / **68.9 dB** | 0.0100 / **54.2 dB** ✅ | 0.0038 / **64.0 dB** ✅ |
| realesr-general-wdn-x4v3 | 0.0090 / **60.8 dB** | 0.0356 / **46.7 dB** ✅ | 0.0145 / **56.5 dB** ✅ |
| realesr-animevideov3 | 0.0018 / **70.0 dB** | 0.0171 / **54.4 dB** ✅ | 0.0043 / **63.6 dB** ✅ |
| fastdvdnet | 0.0005 / **79.0 dB** | 0.0014 / **69.6 dB** ✅ | 0.0008 / **73.9 dB** ✅ |
| rife_v4.25 (v1) | runs; input-sensitive | 0.1597 / 36.0 | 0.1364 / 37.7 |
| rife_v4.25 **v2 (wired)** | **79.4 dB** | 0.0093 / **66.2 dB** ✅ | 0.0088 / **66.1 dB** ✅ |

CUDA sits consistently closest to the fp32 reference; DirectML deltas are fp16-GPU-kernel physics.

**RIFE natural-frame goldens (the §6.3 shipping gate — measured):** three real consecutive-frame
pairs (onboard racing / heavy camera pan / kitchen close-up, 256×256, native fps, t=0.5). PSNR vs
fp32: fp16-CPU 56.6–62.7 dB; **CUDA and DirectML agree within 0.05 dB of each other**, worst case
47.6 dB under the heaviest pan — broadcast-transparent. Derived gate: **per-clip PSNR ≥ 45 dB**
(max-u8 is the wrong metric for flow warps; single-pixel outliers at motion edges dominate it).

### 4.2 Ten-master fleet × EP — raw drift AND behavioral verdicts (RTX 5000 baseline)

Raw max-abs-diff vs the Swift oracle (Rust harness + the python mirror, cross-validated to 4+
digits on shared EPs; P4000 numbers retained in the notes):

| Suite | CPU (tol) | DirectML raw | CUDA raw | DML **behavioral** | CUDA **behavioral** |
|---|---|---|---|---|---|
| beat_this_1 | ✅ 0.0488 (0.075) | ❌ 0.752 | ❌ 0.084 | ✅ **grid identical, worst 1 frame** | ✅ **identical** |
| autoshot | ✅ 0.0147 (0.05) | ✅ **0.0420 — Turing clears the 012 gate** | ✅ 0.0131 | ✅ cuts identical | ✅ cuts identical |
| music_highlighter | ✅ 0.00147 (0.0015) | ❌ 0.0074 | ❌ **0.213** | ✅ **window + top-chunk identical (live)** | ❌ **ranking corrupted** (top 5→9) |
| fireredvad_aed | ✅ (0.001) | ❌ 0.0056 | ❌ 0.0037 | ✅ ≥99.97% frames | ✅ 100% |
| dfnet fused + split | ✅ 0.0018/0.0028 (0.05) | ✅ PASS | ❌ **cuDNN EP_FAIL** (asymmetric-pad conv) | ✅ | ⛔ gated |

Two headline verdicts. First, **behavioral gating vindicates itself**: beat_this's raw 0.75 on
DirectML collapses to a beat grid identical within one analysis frame — raw-logit tolerances were
the wrong gate for GPU EPs. Second, **"CUDA clears the gated four" is refuted as a generalization**:
CUDA fixes autoshot and beat_this but corrupts music_highlighter's ranking (fp16 attention) and
hard-fails dfnet at the cuDNN level. The per-model × per-EP capability gating the traits were
designed for is exactly what shipping needs: on the CUDA-primary chain, `music_highlighter` and
`dfnet` pin to DirectML/CPU; the other three run anywhere.

### 4.3 Rust-path integration goldens (backings through the real `OrtBackend`)

| Backing | CPU | DirectML |
|---|---|---|
| CompactFrameUpscaler | **max u8 diff 1 — byte-perfect** ✅ | diff 7 (≈ its known 46 dB GPU drift; test tolerance ±3 was CPU-calibrated) |
| FastDvdnetTemporalFilter | **diff 0** ✅ | **diff 1** ✅ |
| RifeFrameInterpolator | **diff 1 — pad/crop path proven** ✅ | diff 118 on the synthetic pattern — the input-sensitivity phenomenon (see §5.6) |

The CPU row is the load-bearing one: it proves the pack → infer → unpack pipeline in Rust is equivalent to the Python reference for all three backings.

---

## 5. Verdicts and findings

### 5.1 Spike 3 (Compact + harness) — **PASS**
Export harness built and proven; all three Compact checkpoints portable and validated on CPU + DirectML. Landmine found and fixed: the ORT fp16 converter corrupts `Resize` scales (retypes to fp16 without a cast); `force_resize_scales_fp32()` repairs it while keeping the data path fp16 — this fix is load-bearing for every model with a resize/upsample.

### 5.2 Spike 1 (RIFE on DirectML) — **PASS on correctness; perf unmeasured**
The vs-mlrt-era "RIFE poorly supported on ORT-DML" flag **does not reproduce** on ort-dml 1.24.4: GridSample executes, numerics are sane, and the wired v2 master hits 61.7 dB. Caveats: ORT reported some nodes CPU-assigned within the session (a throughput question, acceptable under D1's sub-realtime stance), and real-frame goldens are still owed (§6.3).

### 5.3 Spike 2 (GridSample on ort-web WebGPU) — **op-availability GREEN; correctness deferred**
onnxruntime-web 1.27.0's WebGPU backend ships a real GridSample kernel (verified in the shipped bundles) — the "unsupported → server-cook" default is disproven. The numerical-correctness run needs a browser+GPU harness; until then web slow-mo remains unpromised, per the brief.

### 5.4 CUDA — **a two-tier cuDNN floor (corrected by the lead's Turing finding, then validated)**
The original Pascal verdict stands: `onnxruntime-gpu 1.27` is a CUDA-13 build (CUDA 13 removed Pascal outright), and the CUDA-12 builds (≥1.22, required by our `ort` api level) hard-require cuDNN 9, which dropped Pascal. **But the floor is two-tiered:** the same `CUDNN_BACKEND_API_FAILED` signature reproduced on **Turing** with cuDNN 9.24 and passed with **cuDNN 9.8.0.87** (lead-discovered, then used for every CUDA number in §4). Ship-matrix wording: **CUDA on Turing requires the cuDNN ≤9.10 pin; Ampere+ runs latest; Pascal is unsupported at any modern version.** The RTX 5000 box's `ort-cuda-env` carries the pinned combo and must not be upgraded casually.

### 5.5 Fail-loud, three times over
(a) The Rust DirectML branch's silent-CPU-fallback bug — fixed with `error_on_failure` for both GPU EPs. (b) ORT's *Python* wrapper silently recreated a failed CUDA session on CPU and nearly reported CPU numbers under a CUDA label — caught, and `ep_probe.py` now calls `disable_fallback()`. (c) `Start-Process -File` on the box silently never starts scripts (the pre-crash fleet run genuinely never ran); the `-Command '&'` form works. All three are now guarded or documented.

### 5.6 RIFE input sensitivity — characterized, not resolved
Flow estimation amplifies fp16 error on out-of-distribution input: 79 dB on realistic input vs 11 dB on noise (CPU), and diff-118 on my synthetic mod-256 pattern (DirectML) while the same model+EP passes at 61.7 dB on in-distribution input. Conclusion: RIFE ships only with **natural-frame-pair goldens** and a per-EP tolerance derived from them (§6.3). The backing code itself is proven correct (byte-perfect on CPU).

### 5.6b Rust `ort` + CUDA-build onnxruntime DLLs — hard hang (open engineering item)
`ort` 2.0-rc.12 (load-dynamic) hangs at session creation (0 CPU) with the **CUDA-build** onnxruntime DLLs at every tested version (GPU 1.23.0 and 1.24.1), while the **DirectML build (1.24.4) works** and Python loads the same CUDA DLLs fine. Rust-side CUDA execution is therefore blocked pending an `ort`-crate investigation (suspect: the provider shared-library loading path). It did **not** block the CUDA columns: model-level CUDA validation ran in Python (fail-loud, `disable_fallback`, cross-checked against the Rust harness to 4+ digits on shared EPs), and the behavioral gate ran the dumped CUDA outputs through the same shipping Rust pure functions offline. Also for the record: the Rust path crashes natively on autoshot-CUDA (Python runs it fine) — same investigation bucket.

### 5.7 Infrastructure incidents (for the record)
The Paperspace 204 episode ("machine not responding") interrupted segment 3 and killed the first fleet attempt; the box returned with all provisioning intact. Mutagen's two-way sync propagated a box-side `.venv` deletion back to the Mac before my gitignore landed, corrupting the export env (rebuilt; envs now never sync). macOS `ort` load-dynamic + the Python-wheel dylib hangs (3/3 tests, 0 CPU) — not pursued since macOS's real EP is CoreML; box covers the ORT columns.

---

## 6. Decisions on your desk

1. ~~CUDA column hardware~~ — **done** (RTX 5000 box; CUDA columns in §4, cuDNN pin in §5.4).
2. **Per-EP tolerance policy** for the gated four — the behavioral suite you directed is implemented (`tests/ort_behavioral.rs`): CPU and GPU EPs post-processed through the shipping pure functions, asserting on beat grids (count equal, |Δt| ≤ 20 ms), scene-cut lists (equal), VAD frame decisions (≥ 99.5% agreement), and highlight windows (equal). Cross-EP results land in `016-notes.md` from the final box run; your sign-off converts the provisional gates into shipped tolerances.
3. **RIFE shipping goldens** — built and measured (three natural pairs; §4.1). Proposed gate: per-clip **PSNR ≥ 45 dB** (worst measured 47.6 on both GPU EPs). Encoded in `ort_enhance.rs` (`FERMENT_ENHANCE_MIN_PSNR`); awaiting your sign-off.
4. **Standing D-decisions, unchanged:** D1 TensorRT (deferred), D2 BasicVSR++, D3 GMFSS, SPAN/BSVD weights outreach.
5. **Ops handover item:** the mutagen session still targets the dead P4000 (`172.83.14.88`) — session lifecycle is yours; recreate it against `184.105.180.26` when convenient (the box tree currently mirrors my validated Mac tree via tarball; two-way sync will reconcile cleanly).

---

## 7. Audit guide

Suggested order: (1) `engine-models/src/fleet.rs` — `third_party_ort()` + provenance tests; (2) `engine-processors/src/frame_tensors.rs` — the shared packing everything rides on; (3) the three backings (`video_upscaler.rs`, `video_noise_filter.rs`, `slow_motion_renderer/rife.rs`) against their seams; (4) `tests/ort_enhance.rs` + `scripts/exports/gen_rust_goldens.py` as a pair (the math must mirror); (5) `scripts/exports/` harness scripts; (6) `016-notes.md` for any number you want to trace to its run.

Working tree: my deltas are the 8 modified + 3 new Rust files listed in §2.2 plus `scripts/exports/` and the two migration docs; your in-flight 008 files are untouched by me. **No commits were made.** Weights/venvs are gitignored; the small fp16 masters + goldens are in-repo by design (they sync to the box and back the manifest's `LocalMaster` entries — say the word if you'd rather they move to `ferment-models` and I'll re-point the manifest).

---

## 8. Bottom line

Windows now has a **validated GPU enhancement path** (DirectML) for super-resolution, temporal denoise, and frame interpolation — the "eyes upgraded to match the Mac" promise — with correct, tested Rust backings behind the same capability traits the macOS tier-1 path implements, a provenance-enforcing manifest, and a reusable export/validation harness that turns the next model port (SPAN, BSVD, when cleared) into a day's work instead of a track.
