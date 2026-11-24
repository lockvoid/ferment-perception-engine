# Track B report — Beats overhaul

Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track B". Executed against
`main @ 5b97e84c` in the shared working tree, with tracks C–G live in the same
crate.

All five items (B.1–B.5) are done. The zero-trust pass found one item of the
doc to be **already satisfied** (B.4's resampler), one stale line reference, and
one defect the doc did not know about — a tail-loss bug in redshift's own
BeatThis chunking that silently discarded up to 29.7 s of analysis from every
track. Details below.

---

## Zero-trust pass: every constant re-read from source

`ferment-cuts-ios @ 3d1c2c180`, files
`Packages/FermentEngine/Sources/FermentEngine/Processors/BeatThis/{BeatThisDBN,
BeatLatticeSegmentation,BeatThisProcessing,BeatThis}.swift`,
`Protocols/{BeatAnalyzing,EnergyStructure}.swift`,
`Processors/BeatQueenMary/BeatQueenMary.swift`,
`Processors/AudioMinimapComposer/AudioMinimapRenderer.swift`;
commits `df4b1b29e`, `9a40897b0`, `1be9614f5`.

| Doc claim | Source verdict |
|---|---|
| `barLengths [3,4]`, `minBPM 55`, `maxBPM 215`, `numTempi 60`, `transitionLambda 100`, `observationLambda 16`, `activationThreshold 0.05`, `correct = true`, `transitionFloor = np.spacing(1)` | **All confirmed verbatim** (`BeatThisDBN.swift:16-24`). `np.spacing(1) == f64::EPSILON == 2.220446049250313e-16`. |
| Input massaging: sigmoid in double → squeeze into (ε/2, 1−ε/2), ε = 1e-5 → Böck two-column → `threshold_activations` trim | Confirmed (`BeatThisDBN.swift:86-110`). |
| "The winning HMM's bar length IS the meter" | Confirmed (`BeatThisDBN.swift:54-58`). |
| Ornament gate: `|ibi − runningMedian(17)| > 3σ̂` **AND** ratio `< 0.75 \|\| ≥ 1.5`; `σ̂ = max(MAD·1.4826, quantizationSigma, 1e-9)`; `quantizationSigma = median(nonzero \|Δ\|)/√12` | Confirmed (`BeatLatticeSegmentation.swift:44-99`). |
| PELT: L2, `β = 2 ln n`, bridges masked | **Partly off.** The doc says `β = 2σ̂² ln n`. The *lattice* PELT normalises the cost by σ̂² and its penalty is **`2·ln n`** (`BeatLatticeSegmentation.swift:124`) — algebraically the same objective, but the constant in the source is `2 ln n`. `2σ̂²·ln n` is the *energy* partition's penalty (`EnergyStructure.swift:115`), which does **not** normalise its cost. Ported each as written. |
| Doc did not mention: PELT **winsorizes** the series to `runningMedian ± 3σ̂`, then z-merges neighbouring segments at 3 standard errors | Both present in source and ported (`BeatLatticeSegmentation.swift:126-230`). Without the winsorization a steady track shatters — mutation-checked below. |
| `EnergyStructure`: `jndDb 1.0`, `phraseBars 2`, `silenceDb −45`, `maxSegments 24`, `.empty` below 8 bars, RMS dBFS floor −80 one decimal | Confirmed. |
| "the doc-comment at `EnergyStructure.swift:70-74` is stale" | **Confirmed, at 73-77.** Line 70 is `static let phraseBars = 2`; the stale sentence ("The penalty escalates (×2) until the result fits") is lines 75-76, inside the `maxSegments` doc comment. Ported the code (weakest-first closest-pair merge), not the comment. |
| `defaultBeatOffset → 0` | Confirmed (`BeatThis.swift:74`). |
| `shippingEnsembleSize = 2` | Confirmed (`BeatThis.swift:59`). |
| `trimToActiveSpan`: 2 % of peak, padded half a beat period, outer bounds only | Confirmed (`BeatQueenMary.swift:69-81`). |
| Bar ruler: yellow `rgb(1.0, 0.82, 0.12)`, `labelStride` from 4 doubling to ≥ 40 px, `lineStride` from 1 doubling to ≥ 3 px, labelled 0.9/2 and unlabelled 0.35/1, bar 1 always labelled | Confirmed (`AudioMinimapRenderer.swift:344-393`). The *doc comment* in that function says "numbered every 8 bars" while the code starts the stride at 4 — a second stale upstream comment; the code is 4. |
| Parity fixtures committed upstream at `Tests/Fixtures/BeatThis/dbn-parity-{logits,expected}.json` | Confirmed, but the path is **`Packages/FermentEngine/Tests/Fixtures/BeatThis/…`**, i.e. relative `"BeatThis/dbn-parity-*.json"` through the fixture crate. 9291 frames, fps 50, 185.808 s; expected 458 beats / 115 downbeats. |
| B.4 "audit redshift's resample path … if it is linear, replace" | **Already polyphase.** See B.4. |

---

## B.1 — `BeatThisDBN.swift` ported, parity exact

`crates/engine-processors/src/beatthis/dbn.rs` (678 lines including its tests).
All arithmetic in `f64`. Public surface: `dbn::decode(beat_logits,
downbeat_logits, fps) -> Option<Decoded>` and `dbn::process(...) ->
ProcessResult`, shaped exactly like `postprocess::process` so the two lanes are
interchangeable downstream.

The bar-pointer HMM keeps the iOS memory shape: back-pointers only for the
first state of each beat (the one place with a choice), `u16` ids, chain states
resolved as `state − 1`. At 50 fps that is 5796 states × 168 slots for the
4-beat bar; the 186 s parity track decodes both bar lengths in **0.58 s** for
the whole `dbn` suite under `[profile.test] opt-level = 2`.

**Independent verification before trusting my own Rust.** I re-implemented the
Swift in NumPy (`/tmp/dbn_ref.py`, throwaway) straight from
`BeatThisDBN.swift`, not from the Rust, and ran it against the committed
fixture:

```
  bar=3 logP=-27552.630
  bar=4 logP=-24571.113
bar_length 4 logP -24571.113168014326
beats 458 expected 458
downbeats 115 expected 115
BEATS MATCH True
DOWNBEATS MATCH True
```

So the fixture is reproducible and my reading of the Swift is right. The Rust
then matches the same fixture:

```
running 8 tests
test beatthis::dbn::tests::empty_logits_decode_to_nothing ... ok
test beatthis::dbn::tests::silence_decodes_to_nothing ... ok
test beatthis::dbn::tests::shipping_intervals_are_linear_and_a_wide_range_is_log_spaced ... ok
test beatthis::dbn::tests::process_falls_back_to_the_minimal_postprocess_when_the_decode_is_too_thin ... ok
test beatthis::dbn::tests::the_bar_pointer_hmm_walks_a_continuous_bar_across_a_gap ... ok
test beatthis::dbn::tests::decodes_steady_four_four ... ok
test beatthis::dbn::tests::picks_three_beat_bar ... ok
test beatthis::dbn::tests::matches_madmom_on_real_logits ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 936 filtered out; finished in 0.58s
```

`matches_madmom_on_real_logits` asserts count-equality plus every beat and
every downbeat within 1e-6 — the same bar the Swift suite uses.

**Fallback.** `< 2` decoded beats hands the *whole* answer back to
`postprocess::process` (asserted field-for-field, not just the beat list). Two
arms are covered: one isolated spike (decode yields exactly 1 beat) and silence
(`decode` answers `None`).

**Judgment calls.**

- **madmom's log-spaced tempo branch is ported even though 50 fps never reaches
  it.** `beat_intervals` is linear 14…55 at the shipping frame rate (42 < 60
  tempi), so the `numTempi` branch is dead in production. `fps` is a public
  parameter, though, and dropping the branch would silently build a 200-tempo
  state space at another rate. It is 12 lines and has its own test that
  exercises it directly at 200 fps (60 distinct intervals, 56…218).
- **`2.0_f64.powf(x)` rather than `x.exp2()`** in that branch, to match Swift's
  `pow(2, ·)` bit-for-bit.
- The observation-pointer test `positions[s] % 1.0 < border` reproduces
  `np.linspace(0, 1, i, endpoint=False)` as `k · (1/i)`, not `k/i`, exactly as
  the Swift comment demands — that multiplication is what makes the border
  comparison match.

---

## B.2 — `BeatLatticeSegmentation.swift` ported; the quantizer is now the lattice

New `crates/engine-processors/src/beatthis/lattice.rs` (ornament gate, PELT,
winsorization, z-merge). `crates/engine-processors/src/beatthis/quantize.rs`'s
`quantize_beats` was **rewritten in place** to the upstream lattice form, and
the dead single-global-fit machinery it replaced was deleted
(`stable_beat_range`, `linear_fit`).

**Why in place rather than alongside:** upstream has exactly one
`quantizeBeats`, and all three callers (minimal post-process, DBN, BeatQueenMary)
share it. Keeping two quantizers would have meant two behaviours for the same
question. `HybridBeatDetector` does **not** call it, so locked decision #1 is
untouched. The three committed BeatQuantizer regression fixtures
(`j8VwTEV3HD`, `v8H5hd4SnF`, `sdOAq0IPSq`) — which upstream kept and whose
comments the new code still cites — pass unchanged against the new quantizer.

`QuantizationResult` and `postprocess::ProcessResult` both gained `segments:
Vec<BeatLatticeSegment>`.

**Ported from `BeatLatticeSegmentationTests.swift`**, using the same seeded LCG
so the synthetic series are identical on both sides (`lattice::jitter`, shared
with the quantizer suite):
`mad_sigma_matches_the_canonical_scale`,
`drum_fill_ibis_are_ornaments_and_steady_ones_are_not`,
`discretized_tracker_steps_are_not_ornaments`,
`frame_alternation_on_a_steady_kick_is_not_an_ornament`,
`steady_track_earns_exactly_one_segment`,
`tempo_change_is_cut_within_two_beats`,
`slow_drift_segments_bound_the_lattice_error`,
`bridge_gaps_do_not_split_agreeing_tempo`,
`sparse_skipped_beat_spikes_do_not_shatter_the_segmentation`,
`short_and_empty_series_do_not_crash`, plus the quantizer-level
`segment_joins_lose_no_beats` (the Ballroom partition defect).

**Added beyond the Swift suite**, because mutation testing showed the ported
tests did not isolate the property:
`a_burst_of_bridges_is_spanned_not_segmented`,
`segments_expand_to_the_flattened_grid`,
`a_drum_fill_does_not_bend_the_grid`.

**Not ported: `testWorstBallroomHoldoutCaseKeepsItsBeats`.** It reads
`/tmp/BallroomAnnotations/Albums-Latin_Jam5-01.beats`, a dataset that is not in
the fixture tree and not obtainable here; the Swift test itself `XCTSkip`s
without it. Inventing an env var for a dataset nobody has would be a gate that
never runs. The property it pins (≥95 % coverage, worst orphan < 0.30 s at a
segment join) is pinned by `segment_joins_lose_no_beats` on a synthetic
150-beat/150-beat tempo change at ≥97 % coverage and < 0.25 s — a strictly
tighter bar on synthetic data. **If the lead can get the Ballroom annotations,
this is the one upstream test still missing.**

**Faithful-but-worth-flagging:** `inter_beat_intervals` filters `Δ > 0`, so a
non-monotone beat list would misalign the `flagged` index set against `beats`.
That is upstream's own shape (`BeatThisProcessing.swift:487`) and both peak
pickers emit strictly increasing beats; ported as written rather than
"fixed" behind the lead's back.

---

## B.3 — `EnergyStructure` + `energyPerBar`, carried by BOTH detectors

New shared module `crates/engine-processors/src/energy_structure.rs`:
`detect(&[f64]) -> EnergyStructure` and `energy_per_bar(samples, sample_rate,
downbeats, duration) -> Vec<f64>`.

Ported the **code**, not the stale comment: the ceiling is closest-pair-first
weakest-boundary merging under `gap < jndDb || runs.len() > maxSegments`, with
no penalty escalation anywhere.

All 13 tests of `EnergyStructureTests.swift` and both of `BeatEnergyTests.swift`
are ported, plus `energy_past_the_end_of_the_samples_floors` (the quantized
downbeat lattice legitimately extends half a period past the last sample, and
Rust slicing is less forgiving than Swift's `min`).

**Bug caught in my own port before it shipped.** The event-depth "extreme" fold
initially seeded from `f64::NAN`; `x > NaN` is false, so the fold would never
replace and every event would have reported a NaN depth. Caught by reading the
Swift `max(by:)` semantics (first maximum wins) rather than trusting the
translation. It now seeds from the window's own head.

Both detectors attach the column: `BeatThis::detect` (which holds the 22.05 kHz
mono buffer) and `BeatQueenMary::detect` (22.05 kHz), through
`BeatAnalysisResult::with_energy_from(samples, sample_rate, duration)` →
`with_energy(per_bar)` → `energy_structure::detect`. One home, exactly as
upstream's `withEnergy(perBar:)`.

`HybridBeatDetector` honestly emits none — attack snap moves individual beats
off any constant-tempo grid, so there is no lattice and no bar column to
measure. That is upstream's own verdict (`9a40897b0`: "Hybrid honestly emits
none — attackSnap breaks the lattice").

---

## B.4 — offset 0, the resampler audit, and a tail-loss bug the doc did not know about

### `defaultBeatOffset → 0`

`config::DEFAULT_BEAT_OFFSET` is `0.0`, with the upstream reasoning in the doc
comment. `BeatThis::new` no longer bakes `−0.010`. Pinned by
`the_stored_grid_carries_no_attack_lead` and by
`the_default_detector_applies_no_beat_offset`.

`HybridBeatDetector`'s own `BeatAlignment::FixedOffset { seconds: -0.010 }`
default is **untouched** — it is hybrid's constant, and hybrid is locked.

### The resampler is already anti-aliased — no change needed

**The doc's premise does not hold for redshift.** Every production path that
changes rate already uses `ferment_engine_audio::resample_sinc`, which is
rubato `SincFixedIn` — a polyphase windowed-sinc resampler:

```
sinc_len 256, f_cutoff 0.95, oversampling_factor 256,
window BlackmanHarris2, interpolation Linear (between the 256× oversampled
polyphase taps, not between input samples)
```

Call sites verified: `beatthis/processor.rs:72`, `audio_source.rs:237` and
`:283` (both decode paths), `audio_minimap_composer/processor.rs:86`,
`hybrid_beat_detector`. `resample_linear` exists but has **exactly one caller in
the whole workspace, a test** (`tests/beat_analyzing_pipeline.rs:210`); it is
not in any production path. `BeatQueenMary` uses `resample_fft` (also
band-limited).

So there is nothing to replace, no new dependency, and no licence question:
rubato 0.16.2 is already in `Cargo.lock` and already clears `cargo deny`.

**Measured delta** — see "Measurements" below. I measured what redshift would
have suffered had it been linear, so the lead has the number the iOS commit
quotes (mean |Δ| 0.68 on the 22.05 kHz mel).

### The chunk plan lost the tail of every track

Not in the dispatch doc; found while porting `BeatThisChunkPlanTests`.
`beatthis/processor.rs::run_inference` drove its own `loop` over chunk starts
and **never ran the realigned final chunk**. Simulated against the iOS
`chunkPlan` over the same frame counts the Swift suite uses:

| frames | redshift covered | should cover | lost |
|---|---|---|---|
| 1501 | 1488 | 1501 | 13 frames |
| 2977 | 2976 | 2977 | 1 frame |
| **9291** (the 186 s parity track) | 8928 | 9291 | **363 frames = 7.3 s** |
| 60010 | 59520 | 60010 | **490 frames = 9.8 s** |

Those frames kept their `-1000.0` sentinel, so the last stretch of every track
whose frame count was not a whole number of chunks produced no beats at all.
Worst case is one full effective chunk, `1488 / 50 = 29.7 s`.

Fixed by extracting the pure `chunk_plan(frames, chunk_size, border_size) ->
Vec<ChunkPlanEntry>` — a direct port of `BeatThis.chunkPlan` — and driving the
loop from it. The four `BeatThisChunkPlanTests` cases are ported, including the
exact Python `split_piece(avoid_short_end=True)` + `keep_first` expectation on
9291 frames (`chunk_start [-6, 1482, 2970, 4458, 5946, 7434, 7797]`). The
sentinel is gone too: the plan tiles `[0, frames)` exactly, so the buffers
start at `0.0` rather than at a value that would survive a gap unnoticed.
`pad_to_chunk` became dead and was deleted.

### `BeatQueenMary::trim_to_active_span`

Ported verbatim, wired into `detect` with the same period fallback (tracked bpm,
else the median inter-beat gap). Its three pure-span tests are ported from
`BeatQueenMarySilenceTrimTests`, plus the end-to-end
`padded_silence_grows_no_beats` (3 s of digital silence either side of the
techno fixture) — which runs here, since the fixture is in the tree.

---

## B.5 — AudioMinimap bar ruler

`AudioMinimapComposer::compose_with_bar_ruler(frames, bar_ruler, progress)` is
the new overload; `compose(frames, progress)` is the default and now delegates
with an empty ruler, so the `core::AudioMinimapComposing` impl is unchanged and
the "tolerant join" stays the lead's wiring.

The thinning math is a pure, tested function —
`Layout::bar_ruler_ticks(downbeats, duration) -> Vec<BarRulerTick>` — with the
upstream constants (`labelStride` from 4 doubling to ≥ 40 px, `lineStride` from
1 doubling to ≥ 3 px, bar 1 always labelled). The renderer strokes the
verticals through mel + RMS in `Color::from_rgba(1.0, 0.82, 0.12, α)` with
α 0.9 / width 2 when labelled and 0.35 / 1 otherwise.

**Known divergence: no printed numbers.** This renderer has no font — the module
has documented that gap since it landed ("Text labels are deliberately omitted
— full text rendering needs a bundled font and ab_glyph composition"), and the
mel Hz labels, dB labels, header and time-axis numbers are all missing for the
same reason. Adding a font just for the ruler would leave the image
inconsistent and pull in a new dependency plus a binary asset. Instead the
ticks — `{bar, x, labeled}` — ride out on `LayoutGeometry::bar_ruler`, which is
exactly the JSON sidecar iOS says the numbers belong in. **The lead should
decide whether to bundle a font**; until then the ruler is an axis with an
external legend rather than a labelled axis.

---

## Measurements

### Linear vs polyphase on the BeatThis mel

The doc asks for a measured delta. Since nothing was replaced, the delta *of
this change* is exactly zero. The useful number is what redshift would have
suffered had the path been linear, so I measured that: `techno-1-30s.mp3`
decoded at 44.1 kHz, resampled to 22.05 kHz three ways, then through
`beatthis::mel::compute_log_mel` (128 bins × 1500 frames).

The three resamplers carry different group delays (`SincFixedIn`'s is
`sinc_len/2` input samples), so each candidate was first aligned to the sinc
output by the integer shift minimising time-domain error — otherwise the
comparison measures the delay rather than the aliasing.

```
MEASURE alignment: linear +1 samples, fft +129 samples
MEASURE 44100->22050 on techno-1-30s (1323000 source samples, 1500 mel frames of 128 bins)
MEASURE sinc vs linear: mean |delta| = 0.0133, max |delta| = 2.2320
MEASURE sinc vs fft   : mean |delta| = 0.0033, max |delta| = 1.4777
MEASURE output lengths: sinc 661439 linear 661499 fft 661375
```

Read it this way: the two **band-limited** resamplers agree with each other
**4× more closely** than either agrees with linear interpolation. Linear is the
outlier, which is the signature of the folded high band — exactly what the iOS
commit describes ("linear resampler folded HF into the mel"). Our absolute
numbers are smaller than the iOS commit's mean |Δ| 0.68 and are **not directly
comparable**: different track, and this mel is `log1p(x·1000)`-compressed,
which squashes the difference.

The measurement harness was a throwaway integration test, run and deleted; it
is not committed.

The `BeatThis` mel fixture parity tests (`test_beat_this_mel_spectrogram_python_reference_match`,
max |Δ| < 0.001 against the Python reference, and
`test_stage_1_beat_this_mel_spectrogram`) pass unchanged — they read
already-22.05 kHz fixtures, so the resampler is not in their path at all. Worth
knowing: **no committed test covers the decode-path resampler**, which is why
the doc's premise could not be checked from the test suite alone.

---

## Test evidence

Everything below was run on this machine. `-- <prefix>` runs are scoped to the
four module trees Track B owns.

| Suite | Result |
|---|---|
| `beatthis::` + `beat_queen_mary::` + `energy_structure::` + `audio_minimap_composer::` | **211 passed, 0 failed** |
| `ferment-engine-processors --lib` (whole crate) | **944 passed, 0 failed**, 28 ignored |
| `ferment-engine-core` | **18 passed, 0 failed** |
| `ferment-engine-test-fixtures` | **10 passed, 0 failed** |
| `cargo check --workspace` | **Finished** — the workspace compiles |
| `rustfmt --edition 2024 --check` on all 18 files I touched | clean |
| `cargo clippy -p ferment-engine-processors --all-targets` | **zero diagnostics in any file I touched** (10 found and fixed: `needless_range_loop` ×6, `int_plus_one`, `manual_range_contains`, `neg_cmp_op_on_partial_ord`, plus two in tests) |
| `cargo clippy -p ferment-engine-core -p ferment-engine-test-fixtures --all-targets` | clean |
| Root `pnpm test` | untouched — no JS/TS/JSON file in this track |

Crate lib count before this track: **644 passed / 27 ignored**. The +300 is
Track B plus tracks C–G landing in the same crate over the same session.

**61 new tests**, by module:

| Module | Passing | New here |
|---|---|---|
| `beatthis::dbn` | 9 | 9 (incl. the madmom parity gate) |
| `beatthis::lattice` | 12 | 12 |
| `energy_structure` | 18 | 18 |
| `beatthis::processor` | 8 | 8 (chunk plan, ensemble, offset, DBN end-to-end) |
| `beatthis::quantize` | 31 | 4 new, 1 deleted with `stable_beat_range` |
| `beatthis::config` | 2 | 1 |
| `beat_queen_mary` (`mod.rs`) | 8 | 5 (trim ×3, padded silence, energy column) |
| `audio_minimap_composer::layout` | 7 | 3 |
| `audio_minimap_composer::renderer` | 3 | 1 |

The DBN parity suite, verbatim:

```
running 8 tests
test beatthis::dbn::tests::empty_logits_decode_to_nothing ... ok
test beatthis::dbn::tests::silence_decodes_to_nothing ... ok
test beatthis::dbn::tests::shipping_intervals_are_linear_and_a_wide_range_is_log_spaced ... ok
test beatthis::dbn::tests::process_falls_back_to_the_minimal_postprocess_when_the_decode_is_too_thin ... ok
test beatthis::dbn::tests::the_bar_pointer_hmm_walks_a_continuous_bar_across_a_gap ... ok
test beatthis::dbn::tests::decodes_steady_four_four ... ok
test beatthis::dbn::tests::picks_three_beat_bar ... ok
test beatthis::dbn::tests::matches_madmom_on_real_logits ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 936 filtered out; finished in 0.58s
```

(`the_tempo_range_reaches_its_own_edges` was added afterwards — the suite is 9
now, and the scoped run reports **211 passed; 0 failed**.)

---

## Mutation testing

Four rounds, 36 mutations, each applied to the source, run against one named
test, then reverted. Nothing here is asserted from reading the code.

The harness itself needed two fixes before its output could be trusted, and
both are worth recording because either would have produced a confident wrong
answer:

1. **Its verdict classifier read a failing test as a compile error.** `cargo
   test` emits `error: test failed …` and a panic line naming the source file,
   so the "did it fail to build?" check matched first. Ten round-1 verdicts
   were unusable until the RED check was moved ahead of it.
2. **Killing the harness mid-mutation left the mutation applied**, and the
   restarted run adopted the corrupted file as its baseline — so the *next*
   mutation stacked on top and the "revert" restored the corruption. Caught by
   spot-checking two anchors and finding **both** absent, which is impossible
   for a single-mutation run. `dbn.rs` was repaired, the parity test re-run
   green, and the harness gained a baseline-hash guard plus SIGTERM/SIGINT
   handlers that restore every file.

### 32 mutations turn a named test RED

| Mutation | Test that catches it |
|---|---|
| DBN `correct=true` peak snap disabled (region start wins) | `matches_madmom_on_real_logits` |
| DBN observation border `1/16 → 1/8` | `matches_madmom_on_real_logits` |
| DBN threshold-trim offset dropped from the beat times | `matches_madmom_on_real_logits` |
| DBN transition λ `100 → 10` | `matches_madmom_on_real_logits` |
| DBN two-column Böck form loses `max(·, ε/2)` | `matches_madmom_on_real_logits` |
| DBN bar length fixed at 4 (no 3-vs-4 contest) | `picks_three_beat_bar` |
| DBN tempo range `55/215 → 60/200` | `the_tempo_range_reaches_its_own_edges` |
| DBN winning bar length not carried into the meter | `the_dbn_lane_publishes_a_lattice_and_an_energy_column` |
| `BeatThis` never attaches the energy column at detection | `the_dbn_lane_publishes_a_lattice_and_an_energy_column` |
| PELT winsorization removed | `sparse_skipped_beat_spikes_do_not_shatter_the_segmentation` |
| PELT bridges ignored at every site (cost, period, snapping) | `a_burst_of_bridges_is_spanned_not_segmented` |
| Ornament gate: metrical conjunction dropped | `discretized_tracker_steps_are_not_ornaments` |
| Ornament gate never reaches the quantizer | `a_drum_fill_does_not_bend_the_grid` |
| Emission windows clamp instead of partitioning | `segment_joins_lose_no_beats` |
| Segments stop expanding to the flattened grid (phase dropped) | `segments_expand_to_the_flattened_grid` |
| Energy: collapsed bars folded back into the host mean | `identical_spikes_report_identical_depths` |
| Energy: `maxSegments` ceiling removed | `a_long_set_of_equally_contrasted_phrases_coarsens_instead_of_collapsing` |
| Energy: track-adaptive σ̂ replaced by a fixed 1 dB | `the_same_step_splits_over_a_calm_floor_and_holds_over_a_loud_one` |
| Energy: phrase floor `2 → 4` bars | `three_bar_break_is_a_section_not_an_event` |
| Energy: phrase floor dropped from `leveled`'s sustained filter | `a_one_bar_segment_is_not_the_tracks_loudness_reference` |
| Energy column: `20·log10 → 10·log10` | `energy_follows_the_bars` |
| **chunk_plan: the last chunk is never emitted (the bug this track fixed)** | `write_windows_tile_exactly` |
| chunk_plan: final chunk not realigned (`avoid_short_end` dropped) | `mirrors_python_split_on_the_parity_track` |
| chunk_plan: `keep_last` instead of `keep_first` | `mirrors_python_split_on_the_parity_track` |
| Bar ruler: label-stride doubling removed | `a_long_mix_thins_both_strides_instead_of_drawing_a_picket_fence` |
| Bar ruler: line-stride doubling removed | `a_long_mix_thins_both_strides_instead_of_drawing_a_picket_fence` |
| Bar ruler: verticals never painted | `a_bar_ruler_paints_yellow_verticals_the_plain_render_does_not_have` |
| QM trim: gate applied per beat instead of outer bounds only | `mid_track_quiet_is_not_trimmed` |
| QM trim: half-period pad removed | `trim_drops_beats_outside_the_active_span` |
| QM: energy column never filled | `the_detector_fills_the_energy_column` |
| Beat offset back to the baked `−0.010` lead | `the_stored_grid_carries_no_attack_lead` |
| Ensemble back to 3 shipping models | `the_shipping_ensemble_is_two_models` |

### Three mutations came back GREEN and were fixed by adding a test

| Mutation | Why the suite missed it | Fix | New verdict |
|---|---|---|---|
| DBN tempo range `55/215 → 60/200` | the parity fixture is a mid-tempo track; every interval it uses survives a narrower range | `the_tempo_range_reaches_its_own_edges` — click tracks at 54 and 14 frames per beat, the slowest and fastest lattices 55/215 can represent at 50 fps | **RED** |
| QM trim: half-period pad removed | the ported Swift case puts beats on multiples of the period, so the padded gate `[90, 510]` and the padless gate `[100, 500]` admit **the same set** | a second grid at `10 + 20k`, where 90 and 510 sit exactly on the padded bounds | **RED** |
| `BeatThis` never attaches the energy column | the test called `analyze_logits(…).with_energy_from(…)` directly; the mutation was in `detect`, which needs the model fleet and is unrunnable here | collapsed `detect`'s tail into one private `analyze()` seam that both `detect` and the test call, so the two cannot drift | **RED** |

### Four mutations stay GREEN — recorded, not papered over

| Mutation | Finding |
|---|---|
| DBN transition λ `100 → 99` | The parity fixture is insensitive to a 1 % change in λ; the winning Viterbi path is identical. λ **is** pinned — `100 → 10` is RED — just not to one part in a hundred. I judged a tighter gate not worth manufacturing: the fixture's job is the decode, not each constant's last digit. |
| DBN activation trim removed (`first` forced to 0) | **Cannot be caught by construction.** The trim's offset cancels: including the 9 leading inactive frames moves `first` to 0 and every frame index up by 9, and `(frame + first)/fps` is invariant. The arithmetic that *could* break — dropping `+ first` — is RED. What the trim buys is decode cost (9291 vs 9206 frames here), not a different answer. |
| Ornament/PELT quantization-σ̂ floor (`median(nonzero \|Δ\|)/√12`) dropped | **No test isolates it, at either call site, and I could not construct one.** In the ornament gate the metrical conjunction already rejects the cases where MAD collapses. In `segment` the floor's absence drives σ̂ to the 1e-9 guard — but the winsorization then clamps the series to `runningMedian ± 3e-9`, i.e. to the running median itself, whose RSS is 0 on a discretized series, so PELT still answers one segment. Ported faithfully from upstream; belt-and-braces on every input I could build. |
| PELT z-merge of agreeing neighbours removed | Same shape: winsorization already prevents the spike-cluster fragmentation the z-merge exists to clean up, on the series the Swift suite exercises. Ported faithfully, unisolated. |

Three things checked by running rather than asserting: an independent NumPy
re-implementation of the Swift decoder reproducing the madmom fixture exactly
(458 beats / 115 downbeats, bar 4, logP −24571.113 vs −27552.630 for bar 3);
the chunk plans of both implementations simulated side by side over the twelve
frame counts the Swift suite uses, which is how the tail loss surfaced; and the
resampler comparison above, re-run with delay alignment after the first
un-aligned numbers proved to be measuring group delay.

---

## What the lead needs for the wiring pass

### The swap itself is one constructor line

```rust
// crates/engine-processors/src/capability_impls.rs — the lead's pass
BeatThis::new(backend).with_post_processing(PostProcessing::Dbn)
```

`BeatThis` already implements `core::BeatAnalyzing`, and `map_beat_result` now
carries the lattice and the energy column through. `PostProcessing::Minimal`
stays the type default, matching upstream's own `BeatThis.init` default — iOS's
shipping caller (`AnalyzeBeatsProcessor`) opts in the same way.

### New public types — `ferment_engine_core` (`capabilities.rs`)

```rust
pub struct BeatLatticeSegment { bpm: f64, first: f64, count: usize,
                                downbeat_phase: Option<usize>, meter: Option<usize> }
pub enum   EnergyLevel { Silence, Quiet, Mid, Loud }          // serde lowercase
pub struct EnergySegment { from_bar: usize, to_bar: usize,     // 1-based inclusive
                           level: EnergyLevel, mean_db: f64 }
pub enum   EnergyEventKind { Peak, Dip }                       // serde lowercase
pub struct EnergyEvent { bar: usize, kind: EnergyEventKind, delta_db: f64 }
pub struct EnergyStructure { segments: Vec<EnergySegment>, events: Vec<EnergyEvent> }
```

`core::BeatAnalysisResult` gained three fields, **all `#[serde(default)]`** so
stored payloads keep decoding:

```rust
pub segments: Vec<BeatLatticeSegment>,
pub energy_per_bar: Vec<f64>,     // one entry per quantized downbeat; last = partial tail
pub energy_structure: EnergyStructure,
```

`crates/project` only *reads* `core::BeatAnalysisResult` (`processors.rs:536`),
never constructs one, so nothing there needed a change and nothing there broke.

### Output shapes the lead maps onto the wire

- **`segments` is the wire shape.** `quantized_beats` / `quantized_downbeats`
  are its concatenated expansion: segment `k` contributes `count` points from
  `first` at `60/bpm`, and every `meter`-th of those from `downbeat_phase` is a
  downbeat. `segments_expand_to_the_flattened_grid` pins that identity to 1e-9,
  so a consumer may store the segments alone and rebuild the arrays.
- **`energy_per_bar[k]` spans `quantized_downbeats[k] → [k+1]`**, the last entry
  the partial bar after the final downbeat. dBFS, floor −80, one decimal.
- **`energy_structure` bars are 1-based inclusive** and index that same column —
  the same vocabulary as the minimap ruler's numbers and the `bar:N` source
  addresses.
- **`LayoutGeometry::bar_ruler: Vec<BarRulerTick { bar, x, labeled }>`** is
  populated only when a ruler was requested. `core::MinimapLayout` has no field
  for it yet — deliberately, since `core::AudioMinimapComposing::compose` also
  has no downbeats parameter. Extending both is one wiring change:
  `compose(audio_path, downbeats, progress)` →
  `compose_with_bar_ruler` → map the ticks onto `MinimapLayout`.

### Signature changes that will break naive call sites

| Before | After |
|---|---|
| `BeatThis::with_options(ensemble: bool, beat_offset: f64)` | `with_ensemble_size(usize)` / `with_beat_offset(f64)` / `with_post_processing(PostProcessing)` |
| `BeatThis::new` ran 3 models | runs `SHIPPING_ENSEMBLE_SIZE = 2`; `with_ensemble_size(3)` reaches the unshipped third checkpoint |
| `DEFAULT_BEAT_OFFSET = -0.010` | `0.0` |
| `Layout::geometry(duration)` | `geometry(duration, bar_ruler: &[f64])` |
| `RenderInput { mel_db, rms_db, layout }` | `+ bar_ruler: &[f64], duration_secs: f64` |
| `QuantizationResult { beats, downbeats, bpm }` | `+ segments` |
| `postprocess::ProcessResult { … }` | `+ segments` |
| `quantize::stable_beat_range`, `quantize::linear_fit`, `processor::pad_to_chunk` | **deleted** (no callers after the rewrite) |

### New public API — `ferment_engine_processors`

```
beatthis::dbn::{decode, process, Decoded}
beatthis::lattice::{Segment, segment, ornament_ibis, mad_sigma, quantization_sigma}
beatthis::quantize::{quantize_beats, indexed_linear_fit, QuantizationResult}
beatthis::{PostProcessing, chunk_plan, ChunkPlanEntry}
BeatThis::{SHIPPING_ENSEMBLE_SIZE, with_ensemble_size, with_beat_offset,
           with_post_processing, active_models, analyze_logits}
BeatAnalysisResult::{empty, with_energy, with_energy_from}
energy_structure::{detect, energy_per_bar}
BeatQueenMary::trim_to_active_span
AudioMinimapComposer::compose_with_bar_ruler
audio_minimap_composer::layout::{BarRulerTick, Layout::bar_ruler_ticks}
```

`analyze_logits(beat_logits, downbeat_logits, fps, track_duration)` is the
public logits seam — the equivalent of iOS's `detect(logits:)`, so a caller that
already has logits (the hybrid seam, a benchmark lane) can decode without
re-running the models.

---

## Deviations from the dispatch doc, and why

1. **`crates/engine-core/src/capabilities.rs` was edited**, which is outside
   Track B's literal file boundary (`beatthis/`, `beat_queen_mary/`,
   `audio_minimap_composer/`, `energy_structure`). It is inside the round's
   SCOPE, and the instruction "DO extend the `BeatAnalyzing`-facing surface on
   the new path so the lead's swap is a constructor change, not a redesign"
   cannot be satisfied any other way: `BeatAnalysisResult` lives there, and
   engine-core cannot depend on engine-processors. The edit is purely additive
   (six new types, three `#[serde(default)]` fields) and surgical, so it should
   not collide with tracks E/F adding traits to the same file.
2. **`capability_impls.rs::map_beat_result`** gained three lines to carry the
   new fields. The hybrid `BeatAnalyzing` impl itself is untouched.
3. **`hybrid_beat_detector/mod.rs` gained four lines** — `..BeatAnalysisResult::empty()`
   plus a comment — purely to keep it compiling once the result type grew. No
   behaviour change, no rewiring; it publishes an empty lattice and an empty
   energy column, which is what upstream's Hybrid does.
4. **`crates/engine-test-fixtures/src/lib.rs`**: two `CANARY_ENTRIES` added for
   the DBN parity fixtures, per Track H's instruction.
5. **B.4's resampler replacement was not performed** — the path is already
   polyphase windowed-sinc. Doing the doc's literal instruction would have been
   a no-op at best.
6. **The chunk-plan tail-loss fix is new work not in the doc.** It is inside
   `beatthis/` and it materially affects the DBN (the sentinel tail is exactly
   the region the activation trim would silently discard), so it belongs in
   this track rather than being left for someone to find later.
7. **`quantize_beats` was rewritten in place**, not added alongside — see B.2.
8. **The minimap ruler prints no numbers** — see B.5.
9. **`testWorstBallroomHoldoutCaseKeepsItsBeats` not ported** — see B.2.
10. **`BeatLatticeSegment::downbeat_phase` is `Option<usize>`** where iOS has
    `Int?`. The value is a lattice index inside its own segment and is provably
    non-negative (downbeats are a subset of the emitted indices); the port
    clamps at 0 rather than widening the type.

## Things I could not run

- **`cargo test --workspace` and a clean `cargo clippy` over the whole crate.**
  Tracks C, D and G are mid-edit in `ferment-engine-processors`; over this
  session the crate was intermittently uncompilable for reasons in
  `video_scene_detector/`, `dinov2/`, `video_saliency/`, `encode_pipeline.rs`,
  `thumb_container.rs`, `parakeet/` and `gigaam/`. Everything below is scoped to
  my own modules and was run at moments the tree compiled. The three `gigaam::*`
  unit-test failures in the last full-crate run are Track D's, not mine.
- **`cargo deny check licenses`** still fails on the loro tree (`bitmaps`, `im`,
  `sized-chunks`, `xxhash-rust`), exactly as Track H reported. I added no
  dependency, so my delta there is zero.
- **The ORT/CoreML beat oracles** (`--test beat_analyzing_pipeline`,
  `coreml_processors`) need the model fleet, which is not on this machine; they
  compile (`cargo check --all-targets` clean for my changes) but are
  `#[ignore]`d and unrun. **The lead should re-run `just engine-oracle`** once
  the fleet is mounted — the chunk-plan fix changes the logits at the tail of
  every track, so `beatthis_coreml_logits_match_swift_python_fixture` and the
  behavioural dumps are the tests most likely to move.
- **No new `--test` target was added**, so nothing needed adding to
  `just engine-oracle`: every test here is a `--lib` unit test and already runs
  under `cargo test -p ferment-engine-processors` (Justfile line 140) and under
  CI's `cargo test --workspace`.

## Environment note for the lead

Eight orphaned CPU busy-loop shells (PPID 1, `while :; do :; done`) were left
running by Track H's LRU flake experiment — its `kill $LOADPIDS` never fires
because `jobs -p` is empty in a non-interactive zsh. They had been burning
~58 % of a core each for **1 h 57 m**, saturating the box for all six agents
(load average 41/57/60). I reaped them. Worth removing that pattern from the
Justfile/report before it is copied again.
