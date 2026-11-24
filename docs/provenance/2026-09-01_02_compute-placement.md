# 2026-09-01_02 — W5: per-model compute placement — iOS's pins come to Studio (FLEET)

**One executor, three staged commits + report
(`2026-09-01_02_compute-placement_report.md`).**

iOS pins compute units PER MODEL (`ModelPlacement`, tuned by hand — a day of
measurement lives in those values). Studio has no placement axis: every
mlmodelc gets `ComputeUnits::All`. Port the axis; iOS pins are the baseline.

## LOCKED rulings (Dimitri, 2026-09-01)

1. **Studio mirrors the iOS pins exactly** as the starting values (verified at
   HEAD): cpuOnly = Parakeet×4, GigaAM×4, AmberNet, Pyannote, ReDimNet, DFNet,
   MusicHighlighter, MusicTagger, FireRedVAD-AED; cpuAndNeuralEngine =
   Dasheng, DINOv2, BeatThis; Autoshot = **cpuAndNeuralEngine, NO chip gate**
   — the `ChipGeneration.clears` machinery is deliberately not ported (we
   test on modern M3; gates come later — leave a comment saying so).
2. **The axis lives in the fleet manifest** (`crates/engine-models`): a
   `placement` per artifact/model — Core ML units for the mlmodelc lane, EP
   preference for the ORT lane. `CoreMlBackend` reads it per model
   (`with_compute_units` exists; make it per-model, not per-backend).
   `RoutingBackend` stays the lane chooser; placement is within-lane.
3. **The Windows half is the 016 ship matrix, finally encoded**: DirectML
   where qualified, CPU where CUDA/GPU corrupted (music_highlighter) or
   hard-failed (dfnet) — from `migration/016-report.md`, into the same
   manifest axis. This closes Track H's "EP matrix still prose-only" leftover.
4. **The GPU question is answered by measurement, not blanket promotion.**
   iOS's cpuOnly pins encode two things — ANE-hostility (transfers to
   desktop) and iPhone thermals (does NOT transfer). So stage 3 measures, on
   the M3, each cpuOnly family under `.cpuAndGPU`: oracle output parity FIRST
   (016 proved GPU numerics can corrupt — CUDA broke music_highlighter),
   then wall time. Promote a model only when outputs match its oracle AND
   wall time improves ≥1.3x. Same pass measures DINOv2 via the ORT CoreML EP
   (the one real ANE gap vs iOS). Report the full table; the lead signs
   promotions.

## Stages

1. Manifest axis + CoreMlBackend per-model units + iOS baseline values +
   Windows EP matrix encoding. Tests: placement resolves per model; the
   baseline table pinned.
2. Plumb ORT EP preference (CPU default; DirectML on Windows where matrix
   says; CoreML-EP hook for the DINOv2 experiment, off by default).
3. The M3 measurement pass: `just engine-oracle` + benches per candidate
   under cpuAndGPU; the promotion table in the report. No pin changes land
   without the oracle column green.

## Boundary

`crates/engine-models`, `crates/engine-infer`, `crates/project` (backend
construction only), `migration/016-report.md` read-only. Gates: full
workspace + oracles for every touched family.
