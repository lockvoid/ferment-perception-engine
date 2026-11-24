# Track F report — VideoSaliency: pure math + substitute ONNX backing

Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track F". Executed
2026-08-31 against `main @ 5b97e84c` in the shared working tree.

All of F.1, F.2 and F.3 are done. The pure math is a faithful port; both
backings are exported, qualified and oracled; the two Vision-calibrated
constants were re-measured and **one of them changed** — YuNet's detection
threshold, not the heat floor. Details, evidence and the numbers the lead has
to sign off are below.

---

## Zero-trust pass: every constant re-read from the Swift source

Read on 2026-08-31 from
`ferment-cuts-ios/Packages/FermentEngine/Sources/FermentEngine/Processors/VideoSaliencyAnalyzer/`
(455 LOC across 5 files) and `Protocols/VideoSaliencyAnalyzing.swift`.

| Doc claim | Source | Verdict |
|---|---|---|
| `samplesPerSecond 0.5`, `min 3`, `max 32`, `interior 0.1`, `maxLongEdge 512`, `heatFloor 0.3` | `VideoSaliencyAnalyzing.swift:7-19` | **confirmed** |
| tolerance `min(0.25, interior × duration)` | `VideoSaliencySamplePlan.swift:22` | **confirmed** |
| scenes `< 0.4 s` get one midpoint grab | `VideoSaliencySamplePlan.swift:45` + `:23-28` | **confirmed** |
| sample count `ceil(duration × rate)` clamped both ways | `VideoSaliencySamplePlan.swift:46-47` | **confirmed** |
| `confidenceWindow 0.15` | `VideoSaliencyMeasure.swift:19` | **confirmed** |
| pedestal subtraction → centroid → ±2σ extent clamped → mass share | `VideoSaliencyMeasure.swift:107-151` | **confirmed** |
| empty map → `(0.5, 0.5, 1, 1, 0)` | `VideoSaliencyMeasure.swift:124` | **confirmed** |
| `unitClipped` intersects ONLY on escape | `VideoSaliencyMeasure.swift:79-82` | **confirmed** |
| `dominantFace` = largest area | `VideoSaliencyMeasure.swift:87-89` | **confirmed** |
| lower median, `faceShare`, `spread`, all-failed scene | `VideoSaliencyReduce.swift:19-51` | **confirmed** |
| top-left origin, row 0 = top; face boxes y-flipped | `VideoSaliencyMeasure.swift:14, 48, 105-106` | **confirmed** |
| result `frameWidth/Height` = display-oriented source size | `VideoSaliencyAnalyzer.swift:29-32, 65-66` | **confirmed** |

**Nothing in the dispatch doc disagreed with the source.** Two details the doc
did not state, found by reading, both of which change the port:

1. **`max(by:)` keeps the FIRST maximum on a tie**, where Rust's
   `Iterator::max_by` keeps the last. `dominant_face` is written to keep the
   first and `equal_areas_keep_the_first_face` pins it — a detector that
   reorders equal-area boxes must not flip the verdict.
2. **The `count > 1` guard, not the 0.4 s guard, is what selects the midpoint
   branch.** With `minSamplesPerScene 3` the two are equivalent today, but the
   port keeps the guard where Swift keeps it so a config change behaves the
   same on both platforms.

---

## F.1 — the pure math

`crates/engine-processors/src/video_saliency/`, 1 673 LOC including tests.

| File | What |
|---|---|
| `config.rs` | `VideoSaliencyConfig` + the two named floors (see F.3) |
| `sample_plan.rs` | `plan` / `sample_count` — scenes → grabs, ascending |
| `measure.rs` | `heat_measurement`, `unit_clipped`, `dominant_face`, `measure` |
| `reduce.rs` | `verdicts` + lower `median` |
| `grabber.rs` | `display_size`, `maximum_size`, the sparse seek-and-decode grabber |
| `analyzer.rs` | `plan` → `measure` → `assemble`, plus the path-based `analyze` |
| `u2netp.rs`, `yunet.rs` | the two ONNX backings (F.2) |

Coordinates are normalized with a top-left origin throughout. Because clipping
to the unit square commutes with a y-flip, `unit_clipped` operates on the
already-flipped box and the Swift rect cases port verbatim.

### Deviations from Swift, and why

- **Trait sources are `async`.** Swift measures synchronously inside
  `runOnDSP`. Written synchronously here (the RIFE `block_on` pattern) it
  deadlocks: `analyze` is async for decode, so a `block_on` inside `measure`
  panics with `EnterError`. **This was found by running, not by reading** — the
  first end-to-end attempt failed on it. Both traits now use
  `#[async_trait(?Send)]`, matching every other capability trait in
  `engine-core`.
- **`unit_clipped` on a box entirely outside the frame** returns zero extent at
  the clamped corner. Swift's `CGRect.intersection` answers `.null` there; no
  real detection reaches that branch and a `null` rect has no useful Rust
  equivalent.
- **`CGRect.contains` treats an empty rect as not-contained.** The port's
  containment test is the plain inequality, so a zero-area box already inside
  comes back untouched instead of round-tripping through the intersection.
  Degenerate either way.
- **A mis-shaped heat map is an error, not a centred default.** Swift cannot hit
  this (Vision owns the buffer); a third-party `SaliencyHeatSource` can, and a
  silent `(0.5, 0.5, 1, 1, 0)` is indistinguishable from a real measurement.
- **Display orientation is applied by us.** iOS gets it free from
  `appliesPreferredTrackTransform`. `engine-media` exposes `display_matrix` but
  no decoder in this repo consumes it, so `grabber.rs` reads the quarter-turn
  and rotates the frame — reusing `crates/project/src/engine.rs`'s
  `rotation_degrees` convention so a video's poster and its saliency
  measurement agree on which way is up. Without this a portrait source would
  report portrait dimensions while measuring in landscape coordinates.
- **The grabber's seek rule.** `AVAssetImageGenerator` spends its tolerance to
  avoid a decode; the port seeks to the sync sample before
  `time − tolerance` and takes the first frame at or after it. Same observable
  behaviour (a grab up to `tolerance` early, never outside the scene), no
  generator. The end-to-end test pins the resulting positions, so a grabber
  that ignored the tolerance fails (mutation M15).
- **`mode` is reported by the heat source, not the config.** iOS's `mode` picks
  between two Vision requests; with one substitute backing a config knob would
  be a lie. `SaliencyHeatSource::mode()` declares what the source measures and
  the result carries it, so the lead's cook mapping cannot silently write
  `attention` for an objectness map. `the_result_reports_the_heat_sources_own_mode`
  pins it.
- **Not ported:** `wholeVideoScenes` (a caller convenience with no consumer
  here), `ProgressThrottler` wiring (the repo has its own `ProgressSink`).
  Cancellation is `ProgressSink::is_closed()` → `EngineError::Cancelled`,
  checked before each grab.

---

## F.2 — the two backings

### U²-Netp for the heat map — kept, as the doc proposed

- **Apache-2.0** (`xuebinqin/U-2-Net`, licence text fetched and read).
- Released ONNX published by **rembg** (MIT tooling) at a pinned URL; the
  download's md5 matches rembg's own pin `8e83ca70e441ab06c318d82300c84806`.
  Upstream's PyTorch checkpoint lives on Google Drive, which is not
  scriptable — a pinned release asset is the reproducible source.
- `export.py` trims it: seven side outputs → one, named `saliency`, input named
  `image`. **1055 → 1049 nodes.** Honest accounting: the six removed nodes are
  the redundant side-head sigmoids, *not* a compute win — d0 is the conv over
  the concatenation of all six heads, so nothing upstream of them is dead. The
  value is one named output instead of `"1959"` as a magic string.
- **opset 11**, static `[1,3,320,320] → [1,1,320,320]`, portability gate clean
  (no custom domains, no `DeformConv`, no 5-D `GridSample`).
- Pre-processing is upstream's `ToTensorLab(flag: 0)`, quirk included: the image
  is divided by **its own maximum**, not by 255, before the ImageNet shift.
  `normalization_divides_by_the_images_own_peak` asserts that a mid-grey field
  and a white field produce the *same* tensor, which is the only cheap way to
  catch a `/255` slip.

**Considered and rejected:** ISNet/DIS (Apache-2.0 but 176 MB), BiRefNet (MIT
but ~900 MB at 1024²), RMBG-1.4 (**non-commercial — disqualified**). No
fixation/attention model (TranSalNet, MSI-Net, UNISAL) ships a first-class ONNX
under a clean licence, so the substitute is an *objectness* map where iOS shipped
*attention*. That is the single most consequential fact in this port and it is
what makes `confidence` incomparable across platforms — see F.3.

### YuNet for the faces — kept, as the doc proposed

- **MIT** (Shiqi Yu, 2020; licence text fetched from the OpenCV org repo).
- `face_detection_yunet_2023mar.onnx`, taken byte-for-byte from the **official
  OpenCV Hugging Face org**. opset 11, static `[1,3,640,640]`, twelve outputs,
  portability gate clean.
- The decode is transcribed from OpenCV's own
  `modules/objdetect/src/face_detect.cpp` (fetched and read, not recalled):
  strides 8/16/32, `score = sqrt(clamp(cls) · clamp(obj))`,
  `cx = (col + Δ)·s`, `w = exp(Δ)·s`, greedy NMS at IoU 0.3, `top_k 5000`.
- Pre-processing is `blobFromImage`'s defaults: raw 0..255 floats in **BGR**
  order, no mean, no scale. `channels_reach_the_graph_in_bgr_order` pins it.
- The frame is scaled uniformly into 640×640 and padded at the **right and
  bottom**, matching `padWithDivisor`'s top-left anchoring; detections divide
  the scale back out.

**`face_detection_yunet_2026may.onnx` (dynamic H/W) was evaluated and rejected.**
It works (2 faces on the demo photo vs 3 for the 2023mar model) and would avoid
the 512→640 upscale, but a static shape is the safer artifact for the DirectML
and QNN qualification that locked decision #10 defers to the Windows box, and
the 2023mar file is the one the field has validated. Recorded here so the lead
can revisit.

### Model files

| File | sha256 | bytes | licence | source |
|---|---|---|---|---|
| `u2netp.saliency.onnx` | `57f3e6be217ac36f38a7913dc0233c1d038dec8eb63f8175627968e51f65d3be` | 4 574 211 | Apache-2.0 | `https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx` |
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | 232 589 | MIT | `https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx` |

Both are under the 10 MB rule, so both ship in-repo under
`scripts/exports/video_saliency/artifacts/` **and** are staged in
`dist/models-publish/` for the CDN. `ModelId::U2Netp` / `ModelId::YuNetFace` and
the two `fleet.rs` entries carry the licence + `source_url` that validation
requires. **`cargo deny check licenses` gains no finding from this track** — no
new Rust dependency was added. It still fails on the `loro` tree
(`bitmaps`/`im`/`sized-chunks` MPL-2.0, `xxhash-rust` BSL-1.0), which is the
ReplicaMan track's, exactly as Track H reported.

---

## F.3 — recalibration (the judgment-heavy part)

Two constants were calibrated against Vision. Both were measured. **One holds,
one had to move.**

### Constant 1 — `heatFloor`: measured, and it holds at 0.3

`scripts/exports/video_saliency/measure_pedestal.py`. Nine frames: the committed
reference frame plus eight extracted from the FermentEngine video fixtures at
the same 512-long-edge cap the shipping grabber applies. The background
estimator is the **10th percentile, not the median** — on a product shot the
subject fills more than half the grid and a median reads the subject, which
inflates the derived floor to 0.54 (that error is visible in the first run and
is why the estimator changed).

```
vision cut sits 0.2468 of the way from background to peak

frame                                   peak     mean       p10   bg/peak  equiv floor
--------------------------------------------------------------------------------------
selfie_320.png                        0.8470   0.0216  0.000013  0.000015       0.2468
dhh-interview-2.0.png                 1.0000   0.2816  0.000015  0.000015       0.2468
dhh-interview-12.0.png                1.0000   0.2880  0.000016  0.000016       0.2468
djs-walking-city-1.86.png             1.0000   0.2644  0.000011  0.000011       0.2468
cooking-chiken-rosemary-5.0.png       1.0000   0.4180  0.000028  0.000028       0.2469
hairdryer-product-placement-3.0.png   1.0000   0.4975  0.000003  0.000003       0.2468
electronic-music-festival-2.0.png     0.0200   0.0001  0.000005  0.000234       0.2470
formula1-onboard-4.0.png              0.9999   0.1773  0.000008  0.000008       0.2468
cloudy_9x16-6.0.png                   1.0000   0.1462  0.000007  0.000007       0.2468

equivalent floor: median 0.2468  min 0.2468  max 0.2470
```

**Vision's pedestal is 0.06 mean against a 0.85 peak — a ratio of 0.071.
U²-Netp's is 3e-6..3e-5 against ~1.0 — a ratio of ~1e-5, four orders of
magnitude lower.** The wide low pedestal that `heatFloor` exists to remove
**does not exist in this model**, because a salient-object segmenter answers
near-binary where a fixation model answers with a broad blob.

Deriving the equivalent: iOS's cut sits at
`(0.3·0.85 − 0.06) / (0.85 − 0.06) = 0.2468` of the way from background to
peak. With a background of ~0 that collapses to the cut position itself, so the
equivalent floor is **0.2468 ± 0.0002 across all nine frames**.

The sweep says the difference does not matter:

```
centroid / extent / confidence against the candidate floors  (cx cy w confidence)
frame                                     floor 0            floor 0.1         floor 0.2468        floor 0.3          floor 0.5
selfie_320.png                    0.665 0.761 0.289 0.920  0.673 …  0.969  0.684 … 0.994  0.688 0.741 0.188 0.997  0.696 … 1.000
dhh-interview-2.0.png             0.686 0.627 0.494 0.268  0.688 …  0.269  0.690 … 0.271  0.690 0.629 0.488 0.271  0.692 … 0.272
dhh-interview-12.0.png            0.718 0.622 0.449 0.295  0.720 …  0.299  0.723 … 0.303  0.724 0.621 0.428 0.304  0.728 … 0.308
djs-walking-city-1.86.png         0.349 0.434 0.766 0.239  0.348 …  0.241  0.348 … 0.242  0.348 0.437 0.765 0.243  0.348 … 0.246
cooking-chiken-rosemary-5.0.png   0.524 0.629 0.660 0.201  0.524 …  0.205  0.524 … 0.208  0.525 0.638 0.656 0.209  0.525 … 0.215
hairdryer-product-placement-3.0   0.506 0.656 0.818 0.181  0.507 …  0.182  0.507 … 0.183  0.507 0.657 0.811 0.183  0.507 … 0.185
electronic-music-festival-2.0     0.603 0.582 1.000 0.101  0.582 …  0.045  0.535 … 0.062  0.539 0.618 1.000 0.027  0.561 … 0.000
formula1-onboard-4.0.png          0.676 0.801 0.665 0.356  0.685 …  0.381  0.679 … 0.400  0.676 0.829 0.645 0.407  0.667 … 0.438
cloudy_9x16-6.0.png               0.303 0.547 0.531 0.609  0.302 …  0.621  0.301 … 0.627  0.301 0.546 0.507 0.629  0.300 … 0.637
```

From floor 0 to floor 0.5 the centroid moves **less than 0.005 on every frame
with a real subject**. The only frames that move are the two where the model
found nothing (`electronic-music-festival`, peak 0.02 — pure noise) and the
crowd photo, where "the subject" is a cluster of small blobs.

**Recommendation for the lead: keep 0.3.** Not because iOS says so — because
0.2468 and 0.3 are indistinguishable on this model, and one number across both
platforms is worth more than 0.05 of an inert constant. `config.rs` names both
(`VISION_HEAT_FLOOR`, `U2NETP_HEAT_FLOOR`) and carries the measurement in the
doc comment; `measure_pedestal.py` regenerates the table if the heat source is
ever swapped. **If you prefer the derived value, change `U2NETP_HEAT_FLOOR` to
`0.2468` — nothing else moves.**

### Constant 2 — YuNet's score threshold: **0.9 → 0.75**, and this one is load-bearing

iOS has no equivalent (Vision owns its threshold), so this constant is new and
had to be measured from scratch.
`scripts/exports/video_saliency/measure_face_threshold.py`, over fixture frames
whose contents were **checked by eye, not guessed** (the first run labelled
`hairdryer-product-placement` "no face"; it has a face, front and centre — the
label was wrong, not the detector):

```
frame                               actual content              0.5    0.6    0.7    0.8    0.9   0.95   scores
dhh-interview-2.0.png               1 frontal face                1      1      1      1      1      0   [0.915]
dhh-interview-8.0.png               1 frontal face                1      1      1      1      1      0   [0.935, 0.485]
dhh-interview-16.0.png              1 frontal face                1      1      1      1      1      0   [0.924]
djs-walking-city-1.86.png           2 faces, one nearer           2      2      2      2      1      0   [0.900, 0.900]
hairdryer-product-placement-3.0.png 1 frontal face                1      1      1      1      1      0   [0.929]
cloudy_9x16-6.0.png                 0 (head from behind)          1      1      0      0      0      0   [0.602]
formula1-onboard-4.0.png            0 (cockpit)                   0      0      0      0      0      0   []
```

True faces score **0.900–0.935**; the worst false positive (a DJ shot from
behind, verified by looking at the frame) scores **0.602**. OpenCV Zoo's shipped
0.9 therefore sits **exactly on two of the true positives** — and that is not
theoretical: at 0.9 the ported `two_faces_resolve_to_one_face_not_the_band_between_them`
**failed**, because the Rust resize kernel differs from PIL's by a fraction of a
pixel and pushed 0.900 below the line. The face vanished and the frame fell into
the saliency lane.

The shipped value is the midpoint of the measured gap:

```rust
const OPENCV_ZOO_SCORE_THRESHOLD: f64 = 0.9;    // upstream reference
const WORST_MEASURED_FALSE_POSITIVE: f64 = 0.6; // DJ's head from behind
pub const SCORE_THRESHOLD: f64 = (OPENCV_ZOO_SCORE_THRESHOLD + WORST_MEASURED_FALSE_POSITIVE) / 2.0;
```

**0.75** — 0.148 of margin above the worst false positive, 0.150 below the
weakest true face. The test passes with the dominant face at cx 0.5375, well
off the union midpoint of ~0.34. **The lead signs this one off too.**

### The finding the lead needs most: `confidence` is not iOS's number

iOS documents the saliency-lane `confidence` as "an isolated subject 0.77–0.92,
a busy landscape ~0.5, a featureless field ~0.97". Measured through this
backing at floor 0.3:

| Content | iOS (Vision, attention) | here (U²-Netp, objectness) |
|---|---|---|
| isolated synthetic subject | 0.77–0.92 | **1.00** |
| talking head | — | **0.27–0.30** |
| street scene, two subjects | — | **0.24** |
| product shot filling the frame | — | **0.18–0.21** |
| landscape | ~0.5 | **0.63** |
| model found nothing | **~0.97** | **0.03** |

The sign flips on the most important row. Vision answered a featureless field
with a *compact, centred* response, so a high `confidence` there meant nothing;
U²-Netp answers with noise, so a low `confidence` means "no subject". That is
arguably the better signal — but **an app-side `FramingResolver` carrying iOS's
thresholds across would invert its own logic**. The field's doc comment on
`VideoSaliencySample::confidence` states the measured ranges.

A related behavioural difference: iOS's `testUniformFieldWithNoSubjectReadsAsCentre`
does **not** hold here. A flat grey field measures `peak=0.0022, confidence=0.000,
cx=0.541, cy=0.776` — the diffuse branch the iOS doc allowed but Vision never
took. The no-subject signal is `confidence ≈ 0`, not a centre prior. That test
was deliberately not ported; porting it would have meant asserting behaviour the
substitute does not have.

---

## Test evidence

### Unit suites — 43 tests, ported from all 9 Swift files

`cargo test -p ferment-engine-processors --lib video_saliency`

```
test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 928 filtered out
```

| Swift file | Ported to | Cases |
|---|---|---|
| `VideoSaliencySamplePlanTests` (135 LOC) | `sample_plan::tests` | 7 — all 7 |
| `VideoSaliencyMeasureMathTests` (80) | `measure::heat_math_tests` | 5 — all 4 + a floor-scaling case |
| `VideoSaliencyDominantFaceTests` (83) | `measure::dominant_face_tests` + `ort_saliency` | 4 pure (incl. **both A13 escape-clip cases** and the tie rule) + the real-footage case as an oracle |
| `VideoSaliencyReduceTests` (123) | `reduce::tests` | 7 — all 7 |
| `VideoSaliencyOrientationTests` (86) | `grabber::tests` + `ort_saliency` | 3 pure geometry + the portrait clip as an oracle |
| `VideoSaliencyAnalyzerTests` (151) | `analyzer::tests` + `ort_saliency` | 5 with stand-in sources + the synthetic clip as an oracle |
| `VideoSaliencyFaceTests` (57) | `ort_saliency` | the dhh-interview case |
| `VideoSaliencyCancelTests` (52) | `analyzer::analyze` | cancellation is `ProgressSink::is_closed()`; **not** ported as a test — see gaps |
| `VideoSaliencyMeasureTests` (78) | superseded | Vision's centre-compression bounds are meaningless for a substitute; the synthetic-clip oracle measures the same property with far tighter bounds |

Plus `measure::source_contract_tests` (1) — a mis-shaped heat map must error.

### ORT oracles — 7 tests, all green

`ORT_DYLIB_PATH=… cargo test -p ferment-engine-processors --features ort --test ort_saliency -- --ignored --test-threads=1`

```
running 7 tests
test a_crowd_measures_as_one_dominant_face_not_the_union ... ok
test a_portrait_source_reports_a_portrait_frame_and_in_frame_coordinates ... ok
test a_synthetic_clip_measures_its_subject_end_to_end ... ok
test a_talking_head_is_measured_from_faces ... ok
test two_faces_resolve_to_one_face_not_the_band_between_them ... ok
test u2netp_heat_matches_the_python_origin ... ok
test yunet_faces_match_the_python_origin ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.43s
```

**`u2netp: observed maximum difference 0`** — bit-exact against the Python
origin over all 102 400 map cells. YuNet's 24 decoded faces match to better
than 2e-4 of the frame.

Reference frames are committed **at each graph's own input size** (320×320 and a
pre-letterboxed 640×640), so no resize runs on either side and the comparison is
exact rather than kernel-dependent — the same decode-path-isolation reasoning
Track E applies to its WAV fixtures.

Measured end to end over the committed synthetic clip (ground truth in
parentheses):

```
still scene:      cx 0.29974 (0.300)   cy 0.50193 (0.500)   w 0.1383 (0.135)   h 0.2460 (0.240)   spread 9.8e-6
travelling scene: cx 0.1924 / 0.4736 / 0.7568  (0.191 / 0.473 / 0.754 at time − tolerance)   spread 0.2304
dhh-interview:    faceShare 1.00 (iOS asserts ≥ 0.6), confidence 0.923, frame 640×360
cloudy_9x16:      frame 608×1080, 6 samples, every coordinate in frame
djs @ 1.86 s:     source=Face, cx 0.5375, w 0.105  (union midpoint ≈ 0.34, iOS bound w < 0.2)
```

The substitute is *more* accurate than Vision on the positional test: iOS
records Vision compressing a disc at true 0.25/0.50/0.75 to 0.332/0.504/0.670,
so its own test allowed ±0.08. U²-Netp lands within 0.001 of ground truth, and
the ported assertions are tightened to ±0.02 accordingly.

### Mutation testing — 16 mutations, 16 red

Every oracle was proven to fail when the thing it claims to check is wrong.
Each mutation was applied, run, and reverted; goldens were restored and their
sha256 re-verified.

| # | Mutation | Result |
|---|---|---|
| M1 | `unit_clipped` drops the untouched fast path (always intersect) | **red** — `face_box_is_clipped_to_the_unit_frame`. `0.2 + 0.5 − 0.2 = 0.49999999999999994`: the float dust iOS's comment warns about is real in Rust too |
| M2 | heat-map row 0 read as the BOTTOM of the frame | **red** — 2 tests |
| M3 | `dominant_face` ties keep the LAST face | **red** — `equal_areas_keep_the_first_face` |
| M4 | `median` takes the upper middle | **red** — `even_count_takes_the_lower_median` |
| M5 | seek tolerance ignores the scene's interior | **red** — 3 tests |
| M6 | the heat map wins over a detected face | **red** — `a_face_beats_the_heat_map` |
| M7 | `assemble` skips the time sort and hard-codes `objectness` | **red** — 2 tests |
| M8 | U²-Netp normalizes by 255 instead of the image's own peak | **red** — `normalization_divides_by_the_images_own_peak` |
| M9 | YuNet fed RGB instead of BGR | **red** — `channels_reach_the_graph_in_bgr_order` |
| M10 | YuNet score is the product, not the geometric mean | **red** — 2 tests |
| M11 | display size ignores the track transform | **red** — `display_size_applies_the_track_transform` |
| M12 | **u2netp golden** perturbed by 0.25 | **red** — `observed maximum difference 0.25`, exactly the injected delta |
| M13 | **yunet faces golden** nudged 0.01 on x | **red** — `x: 0.5523337639277001 vs golden 0.562333763986826` |
| M14 | extent is ±1σ instead of ±2σ | **red** — the end-to-end clip |
| M15 | the grabber ignores the plan's seek tolerance | **red** — the end-to-end clip |
| M16 | a mis-shaped heat map falls back to the centred default | **red** — `a_mis_shaped_heat_map_is_an_error_not_a_centred_default` |

M15 exists because the first version of the end-to-end assertions **survived**
it: `cx < 0.35` and `cx > 0.65` are satisfied whether or not the tolerance is
spent. The test now pins the three grab positions the tolerance rule produces.

### Gates

| Gate | Result |
|---|---|
| `cargo fmt --check` on every file I touched | **clean** |
| `cargo clippy -p ferment-engine-core -p ferment-engine-infer -p ferment-engine-models -p ferment-engine-test-fixtures --all-targets -D warnings` | **clean** |
| `cargo clippy -p ferment-engine-processors --features ort --all-targets -D warnings` | **zero findings in `video_saliency/` or `ort_saliency.rs`**; the 10 that remain are all in `beatthis/`, `energy_structure.rs`, `beat_queen_mary/` (Track B, in flight) |
| `cargo test -p ferment-engine-processors --lib` | **942 passed, 0 failed**, 28 ignored |
| `cargo test -p ferment-engine-test-fixtures` | **10 passed** (canary green with 3 new entries) |
| `cargo deny check licenses` | fails on the `loro` tree only — **no finding from this track** |
| root `pnpm test` | untouched — no JS/TS file changed |

---

## What I could not run

- **`cargo test -p ferment-engine-models` is red on one test that is not mine.**
  `default_manifest_is_complete_and_valid` panics with
  `unknown model id 'pyannote_segmentation'`: Track E added `ModelId` variants
  and `fleet.rs` entries but not the matching `all_model_ids()` entries. I added
  **only my two** (`U2Netp`, `YuNetFace`) rather than fix their gap in their
  file. **Proof my entries are sound:** with Track E's two ids added locally,
  `validate_complete_fleet` walks all 32 models clean and the only remaining
  assertion is Track D's stale resource count (3 vs 5, from `gigaam_vocab` +
  `ambernet_langs`). The other 14 tests in the crate pass. Two counts I did
  update to the truth as of this edit: `manifest.models.len()` 24 → **32** and
  `studio_downloads_in` 29 → **31**; expect a one-number merge with whoever
  lands last.
- **No GPU EP qualification.** CPU EP only, per locked decision #10. Both graphs
  are opset 11 with no disqualifying ops, so DirectML/CUDA should take them, but
  that is the Windows box's job.
- **No Linux run.** macOS arm64 only. The end-to-end and fixture-backed oracles
  need a platform decoder (`UnavailableVideoIo` on Linux) as well as ORT, so
  they are `#[ignore]`d and gated behind `just engine-oracle`; the 43 unit tests
  run everywhere.
- **The shared tree broke under me twice** while Track C was mid-edit
  (`video_scene_detector/shot_identity.rs`, then `slow_motion_renderer/`), and
  once for Track G (`engine-media/platform/macos.rs` still has
  `if false && is_inter_frame_codec`, which fails clippy). I waited each time
  and touched none of their files. Every number above was measured with the
  tree compiling.

## Known gaps, stated plainly

- **Cancellation has no test.** `analyze` checks `ProgressSink::is_closed()`
  before each grab and returns `EngineError::Cancelled`, mirroring
  `VideoSaliencyCancelTests`. Testing it needs a long clip and a racing
  canceller; I judged the fixture cost above the value for a three-line check
  and left it unpinned rather than write a flaky test. **Say the word and I will
  add it.**
- **Resize-kernel parity is measured nowhere.** The committed reference frames
  sit at each graph's input size specifically so the oracles do not depend on
  `image`'s Lanczos3 matching PIL's LANCZOS. The two do differ, and that
  difference is what pushed a 0.900 face under a 0.9 threshold — the
  consequence is handled by the recalibrated threshold, not by a parity number.
  If the lead wants that number, it is one script away.
- **`electronic-music-festival` is a real miss.** U²-Netp returns a peak of
  0.0200 on that frame — it finds nothing in a dark crowd shot. The measurement
  degrades gracefully (confidence 0.027), but it is worth knowing that this
  backing has content it cannot read.

---

## What the lead needs for wiring

### New capability traits — `crates/engine-core/src/capabilities.rs`

```rust
pub struct RgbFrameRef<'a> { pub data: &'a [u8], pub width: u32, pub height: u32 }
impl<'a> RgbFrameRef<'a> { pub fn new(data: &'a [u8], width: u32, height: u32) -> Option<Self> }

pub enum SaliencyMode { Attention, Objectness }              // serde: snake_case
pub struct SaliencyHeatMap { pub values: Vec<f32>, pub width: usize, pub height: usize }
pub struct FaceRect { pub x: f64, pub y: f64, pub width: f64, pub height: f64, pub confidence: f64 }

#[async_trait(?Send)]
pub trait SaliencyHeatSource: EngineService {
    fn mode(&self) -> SaliencyMode;
    async fn heat(&self, frame: RgbFrameRef<'_>) -> Result<SaliencyHeatMap>;
}

#[async_trait(?Send)]
pub trait FaceRectSource: EngineService {
    async fn faces(&self, frame: RgbFrameRef<'_>) -> Result<Vec<FaceRect>>;
}
```

`RgbFrameRef` is packed RGB24 (`data.len() == 3·width·height`). `FaceRect` is
normalized to the frame, **top-left origin**, unclipped — clipping is the
measurement layer's job.

### Result type — `ferment_engine_processors::video_saliency`

```rust
pub struct VideoSaliencyResult {
    pub frame_width: u32,          // display-oriented, NOT the capped grab size
    pub frame_height: u32,
    pub mode: SaliencyMode,        // objectness with the shipped backing
    pub scenes: Vec<VideoSaliencySceneVerdict>,  // one per input scene, in order
    pub samples: Vec<VideoSaliencySample>,       // ascending time
}
pub struct VideoSaliencySample {
    pub scene_index: usize, pub time: f64,
    pub cx: f64, pub cy: f64, pub w: f64, pub h: f64,
    pub confidence: f64, pub source: SampleSource,   // Face | Saliency
}
pub struct VideoSaliencySceneVerdict {
    pub index: usize, pub sample_count: usize,
    pub cx: f64, pub cy: f64, pub w: f64, pub h: f64,
    pub confidence: f64, pub face_share: f64, pub spread: f64,
}
```

Field-for-field the iOS `VideoSaliencyResult`, so `CookData.VideoSaliency.make`
ports directly — **except that `mode` must be written from
`result.mode` (`"objectness"`), never hard-coded to iOS's `"attention"`.**

### Entry points — `VideoSaliencyAnalyzer`

```rust
VideoSaliencyAnalyzer::new(heat: Arc<dyn SaliencyHeatSource>, faces: Arc<dyn FaceRectSource>)
    .with_video_backend(backend)                                   // optional; platform default otherwise

async fn analyze(&self, video_path: &Path, scenes: &[VideoScene], progress: ProgressSink)
    -> Result<VideoSaliencyResult>                                 // plan → sparse grab → measure → reduce

// or, for a caller that owns decode (the scene detector's convention):
fn plan(&self, scenes: &[VideoScene]) -> Vec<SampleEntry>
async fn measure(&self, entry: &SampleEntry, frame: RgbFrameRef<'_>) -> Result<VideoSaliencySample>
fn assemble(&self, frame_size: (u32, u32), scene_count: usize, samples: Vec<VideoSaliencySample>)
    -> VideoSaliencyResult
// with video_saliency::{display_size, maximum_size} to prepare frames identically
```

`scenes` is `ferment_engine_core::VideoScene`, so the scene detector's result
maps straight in. `analyze` errors on an empty scene list, logs and skips a
grab that fails to decode, and returns `EngineError::Cancelled` when the
progress sink is closed.

### Nothing was wired

No `crates/project` change, no cook kind, no operator key, no processor
version, no graph node. The seam is clean: construct the analyzer with the two
ORT-backed sources and call it.

### Also touched, for the record

- `ModelId::{U2Netp, YuNetFace}` + `name()` arms + the uniqueness-test array
  (`crates/engine-infer/src/model.rs`).
- Two `third_party_ort` entries + `SALIENCY_COREML_GAP`, `U2NETP_SOURCE`,
  `YUNET_SOURCE` (`crates/engine-models/src/fleet.rs`); two `all_model_ids`
  entries (`crates/engine-models/src/lib.rs`).
- `just engine-oracle` runs `--test ort_saliency` (a `--test` target nothing
  invokes is not a gate).
- Three `CANARY_ENTRIES` (`Video/{cloudy_9x16,dhh-interview,djs-walking-city}.mp4`).
- `scripts/exports/.gitignore`: the untrimmed U²-Netp download and the upstream
  demo photo, both re-fetchable.
- **`MODEL_FLEET_VERSION` was NOT bumped** — the dispatch doc says the
  last-landing track bumps it once, and B–G are still in flight.

### Files

**New:** `crates/engine-processors/src/video_saliency/{mod,config,sample_plan,measure,reduce,grabber,analyzer,u2netp,yunet}.rs`;
`crates/engine-processors/tests/ort_saliency.rs`;
`crates/engine-processors/tests/fixtures/saliency-two-scenes.mp4` (30 155 B,
`ac91822f3eccb54f8c3a530a56e82646193eebd735224e36e66e92ea798e1e5c`);
`scripts/exports/video_saliency/{export,gen_goldens,measure_pedestal,measure_face_threshold,make_e2e_clip}.py`
+ `artifacts/{u2netp.saliency.onnx,face_detection_yunet_2023mar.onnx,manifest.json,pedestal.json}`
+ `frames/{selfie_320.png,selfie_640.png}`;
`scripts/exports/rust_goldens/{u2netp_saliency_out.bin,yunet_faces.json}`.

**Changed:** `crates/engine-core/src/capabilities.rs`,
`crates/engine-infer/src/model.rs`, `crates/engine-models/src/{fleet,lib}.rs`,
`crates/engine-test-fixtures/src/lib.rs`,
`crates/engine-processors/src/lib.rs`, `Justfile`,
`scripts/exports/.gitignore`.

**Committed artifact shas:**

```
57f3e6be217ac36f38a7913dc0233c1d038dec8eb63f8175627968e51f65d3be  artifacts/u2netp.saliency.onnx
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  artifacts/face_detection_yunet_2023mar.onnx
5de3af3d6a0d6b77863100c2f0d494dadb0385832417eae792e635ee7aeb6f1a  frames/selfie_320.png
b4be74edd36b145c127f0e241e43f4bc0b30331448a4ebb7e3ff8f0ed9b5cb1c  frames/selfie_640.png
62e1c46aaa938838459004206efc56efe5eb3774a498dc1e8132e41fbeeda00e  rust_goldens/u2netp_saliency_out.bin
64105f53847ae2500b26c8ea515220b65088903abb9400a4fc7b531f302c4725  rust_goldens/yunet_faces.json
ac91822f3eccb54f8c3a530a56e82646193eebd735224e36e66e92ea798e1e5c  tests/fixtures/saliency-two-scenes.mp4
```

The two reference frames are derived from OpenCV Zoo's MIT-licensed
`largest_selfie.jpg` (`ab8413ad9bb4f53068f4fb63c6747e5989991dd02241c923d5595b614ecf2bf6`),
which `gen_goldens.py` fetches and sha-verifies; the source photo itself is not
committed. Fixture frames used for the pedestal and threshold tables come from
the FermentEngine video fixtures and are not committed either — the scripts
regenerate them, and the measurement tables (`artifacts/pedestal.json`) ship.

## Two decisions for the lead

1. **`heat_floor`: keep 0.3, or take the derived 0.2468?** My recommendation is
   keep 0.3; the measurement says the choice is inert on this model and one
   number across platforms is worth more. One-line change either way.
2. **`SCORE_THRESHOLD = 0.75`** replaces OpenCV Zoo's 0.9. This one is not
   optional — 0.9 demonstrably drops real faces — but the exact midpoint is a
   judgment call on seven fixture frames. If you want more evidence before
   signing, `measure_face_threshold.py` takes any `--fixtures` tree.
