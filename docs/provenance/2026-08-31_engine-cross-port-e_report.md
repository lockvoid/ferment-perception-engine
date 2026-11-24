# Track E report — Diarization: Pyannote + ReDimNet

Dispatch: `2026-08-31_engine-cross-port.md` § "Track E". Executed 2026-08-31
against `main @ 5b97e84c` in the shared working tree.

**Both parity bars are met with margin, measured not asserted:**

| Bar | Required | **Measured** |
|---|---|---|
| Pyannote frame decisions vs the reference export | ≥ 99.9% | **100.000%** — 1767/1767 frames over three cases, zero disagreements |
| ReDimNet cosine per window vs the torch goldens | ≥ 0.999 | **0.999999762** worst of 20 windows |
| ReDimNet window count | exact | exact (5 per clip × 4 clips) |
| ReDimNet window times | 1e-9 | exact to the assertion (1e-9) |

Everything below was run, not reasoned about. Every parity test was mutation-tested.

---

## Zero-trust: every constant re-read from source

All re-verified against `ferment-cuts-ios` at HEAD before any code was written.
**The doc is correct on every constant it states.** Four things the doc did not
say are recorded below; three of them changed what I wrote.

| Doc claim | Source | Verdict |
|---|---|---|
| `sampleRate 16000`, `windowSamples 160_000`, `windowFrames 589`, `frameStride 270`, `frameSamples 991`, `stepFrames 294` | `SegmentationWindows.swift:25,28,29,34,35,48` | exact |
| powerset table `[[],[0],[1],[2],[0,1],[0,2],[1,2]]` | `PowersetDecoder.swift:28` — and independently confirmed against the checkpoint's own `config.json` `id2label` (`NO_SPEAKER, SPEAKER_1..3, SPEAKERS_1_AND_2, _1_AND_3, _2_AND_3`) | exact |
| `minimumTurnSeconds 0.20`, `minimumGapSeconds 0.20` | `TurnAssembler.swift:30-31` | exact |
| gaps close FIRST, then short runs drop; leading/trailing never closed; overlap flag smoothed before the split | `TurnAssembler.swift:44-51,104` | exact — ported in that order |
| threshold **0.45**, average-linkage cosine | `PyannoteSpeakerSegmenter.swift:115`, `LocalSpeakerStitcher.swift:63-92` | exact |
| vectorless candidates attached by shared seconds | `LocalSpeakerStitcher.swift:98-113` | exact |
| per-frame marginal/coverage ≥ 0.5 | `LocalSpeakerStitcher.swift:158,199` | exact (both the activity vote and the crosstalk vote) |
| final fingerprint = ONE pool over ALL clean speech; two-phase (probe → pool) | `PyannoteSpeakerSegmenter.swift:200-233` | exact |
| ReDimNet `windowSamples 32_080` (2.005 s), `windowFrames 200`, `hopLength 160`, `embeddingDimension 192`, never padded | `ReDimNetSpeakerEncoder.swift:32,34,35,43,44` | exact |
| `windows(for:)` rounds not truncates, never crosses a segment boundary, sorts by offset | `ReDimNetSpeakerEncoder.swift:165-190` | exact |
| `b2-vox2-lm` MIT, PalabraAI/redimnet2 | repo licence MIT; release asset is **15,897,450 bytes**, matching the iOS `convert.py` comment byte-for-byte | exact |
| `stage.py` guards `[1,589,7]` | `Scripts/speaker_segmenter/stage.py:95-112` | exact — kept, see E.1 |

### Four findings the doc did not carry

**1. The ONNX graph really does emit log-probabilities — and the iOS
`parity.py` docstring is wrong about it.** `parity.py:21` says "The reference
emits raw logits"; the graph's terminal node is `LogSoftmax` and `exp(output)`
sums to 1.0 per frame (measured: 1.00000001…1.00000006). The doc's instruction
("the graph emits LOG-probabilities, exponentiate them") is right and the
comment in the iOS script is not.

**This is numerically harmless and I checked rather than assumed.**
`make_fixtures.py` applies a softmax to those log-probabilities before writing
`reference_probabilities.bin`. Softmax over a log-probability vector is the
identity (`softmax(log p)_i = p_i / Σp_j = p_i` because `Σp_j = 1`), so the
goldens are true probabilities either way. Measured on all three cases:
`|softmax(out) − exp(out)|` max **2.3e-7**, and `|exp(out) − golden|` max
**1.5e-7** — pure fp32 rounding. The goldens are sound; only the prose was off.

**2. The ONNX I/O contract differs from the CoreML one the Swift is written
against.** Input `input_values` (not `audio`), output `logits` (not
`segments`), and **both shapes are dynamic**: `[batch_size, num_channels,
num_samples] → [batch_size, num_frames, 7]`. Consequence for E.1's guard: the
`[1,589,7]` check **cannot be a schema read** the way `assert_segmentation_3`
does it on the CoreML spec — it has to be a check on a real prediction's output
shape. It is (`segmenter.rs::predict`), and `stage.py` qualifies by running a
window rather than by reading the graph.

**3. `preprocessor_config.json` on the ONNX mirror says `offset: 990` where the
iOS constant is `frameSamples = 991`.** Source wins: 991 is the SincNet
receptive field the Swift derives in its own comment
(`251 + 2·10 + 4·30 + 2·30 + 4·90 + 2·90 = 991`, arithmetic checked). The
difference is one sample — 62.5 µs on a turn's `end_time` — and it touches no
golden, because the committed fixtures are per-frame class/activity/probability
arrays, not times. Ported 991. Flagged so the lead knows the two numbers exist.

**4. The `Tests/Fixtures/Pyannote/` fixtures are NOT tracked by git in
`ferment-cuts-ios`** (`Packages/FermentEngine/.gitignore:20` ignores
`Tests/Fixtures/*`); only `SpeakerEncoder/` is force-added. Mutation testing
therefore backed them up to `/tmp` and restored from there rather than with
`git checkout`. Both trees verified byte-identical afterwards (26 files,
sha256).

---

## E.1 — Pyannote model: qualified, pinned, staged

No export. `onnx-community/pyannote-segmentation-3.0`, pinned **by repository
revision** rather than by `main`, because "the same weights" has to survive the
mirror being updated:

```
repo      onnx-community/pyannote-segmentation-3.0
revision  733a93b6473d019a773298e08cefa686894b1854   (2025-07-08)
file      onnx/model.onnx
licence   MIT  (HF card `license: mit`; upstream pyannote/segmentation-3.0 is MIT)
size      5,986,908 bytes
sha256    057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25
opset     17, ai.onnx only, producer pytorch 2.3.0
```

**This is the same artifact the iOS parity script scores against and the same
one the committed goldens were computed from** — I verified that directly
rather than trusting the provenance: running this file through onnxruntime on
`Pyannote/*/audio_int16.bin` reproduces `reference_classes.bin` **589/589 on all
three cases** and `reference_probabilities.bin` to **9.3e-8**.

`scripts/exports/pyannote/stage.py` is the repeatable qualification, and the
`[1,589,7]` guard is kept and strengthened. It refuses on:

- sha256 mismatch;
- opset > 20 or any non-`ai.onnx` domain;
- a terminal op that is not `LogSoftmax` (the decoder's exponentiation assumes it);
- an input schema that is not `{input_values}`;
- **an output shape that is not `(1, 589, 7)` on a real 160 000-sample window** —
  the check that separates segmentation-3.0 from `speaker-diarization-community-1`;
- `exp(output)` not summing to 1 per frame.

Proven, not asserted:

```
$ uv run python stage.py
  5,986,908 bytes  sha256 057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25
  opset 17, input {'input_values': ['batch_size', 'num_channels', 'num_samples']}, output (1, 589, 7)
  exp(output) sums to 1 per frame — the graph speaks in log-probabilities
  OK: this is segmentation-3.0, not community-1

$ # the impostor case, forced:
REFUSED as designed: unexpected output shape (1, 589, 7) — segmentation-3.0 emits [1, 589, 8];
an 8-class output means this is the community-1 checkpoint
```

The same guard exists at runtime in `pyannote::segmenter::predict` and has two
unit tests driven by a stub backend (`an_eight_class_checkpoint_is_refused_…`,
`a_different_frame_rate_is_refused_too`). Deleting the guard turns both red —
see the mutation table.

`inspect_onnx.py`: `portable_directml_webgpu_qnn: true`, no golden-rule
disqualifiers, no custom domains.

## E.2 — The pure logic

`crates/engine-processors/src/pyannote/` — five modules, free functions rather
than Swift's `enum`-as-namespace (Rust has no such idiom):

| Module | Swift origin | Notes |
|---|---|---|
| `windows.rs` | `SegmentationWindows.swift` | zero-pad the last chunk, drop pad frames **by count**; `end_time` is deliberately not the next `start_time` (61.9 ms of audio every 16.9 ms) and three tests say so |
| `powerset.rs` | `PowersetDecoder.swift` | 7-class table, argmax + marginals + per-frame confidence |
| `turns.rs` | `TurnAssembler.swift` | gaps close **first**, then short runs drop; leading/trailing never closed; overlap flag smoothed on its own before the split |
| `stitcher.rs` | `LocalSpeakerStitcher.swift` | average-linkage cosine @ 0.45, vectorless attach by shared seconds, chunk averaging with per-frame marginal/coverage ≥ 0.5 |
| `segmenter.rs` | `PyannoteSpeakerSegmenter.swift` | the orchestrator + the `[1,589,7]` guard + the `exp()` + `SpeakerDiarizing` |

The overlapping frame times were left alone — deliberately. `end_time(frame)`
is `(frame·270 + 991)/16000`, which is **past** `start_time(frame+1)`. Pinned by
`frames_overlap_each_other` and by `a_reply_runs_to_the_end_of_its_last_frames_audio`.

Two tie-breaking details in `cluster` are ported exactly rather than
idiomatically, because they decide how many people were in the room: the
best-pair search uses **strictly greater** (so a tie keeps the first pair in
`(left, right)` order), and `attach_vectorless` uses **strictly greater than
zero** (so a candidate sharing no seconds with anyone becomes its own voice
rather than joining the nearest). Both carry a comment saying why.

Swift's `sorted` is not stable; the two order-sensitive sorts (`orphans` by
first-turn start, groups by first-heard) use Rust's **stable** `sort_by`, so
equal-start ties resolve by candidate index. That is a strictly more
deterministic answer than the Swift, not a different one.

## E.3 — ReDimNet: exported by tracing the pinned upstream

`scripts/exports/redimnet/export.py` clones `PalabraAI/redimnet2` at
`cdc875670034dd7068013ca2ab21ec083a040ff8` and traces the live modules, so the
weight mapping is identity by construction and nothing re-implements a layer.
It re-checks 17 fields of the checkpoint's own `model_config` before exporting,
so a checkpoint swap that moved the mel parameters cannot convert silently.

```
weights   b2-vox2-lm.pt   15,897,450 bytes
          sha256 0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a
          3,677,760 params, 192-d, 200 frames = 32,080 samples = 2.005 s
master    redimnet_speaker.onnx   16,056,349 bytes
          sha256 11ce0e8a03f3b6fa911988c48cfadd2274e1d320e3b265d46de4898feeeaa135
          fp32 whole-graph, opset 17, fixed input audio[1, 32080]
licence   MIT
```

**Deterministic**: exported twice from scratch, identical sha256 both times.
`inspect_onnx.py`: `portable_directml_webgpu_qnn: true`, no `fft`/`stft`/complex
ops (the frontend is upstream's own conv-based `SpectralFeaturesTF`, which
`assert_no_fft` pins), no custom domains.

**Precision: whole-graph fp32, and it is a decision.** The iOS CoreML build is
fp16 with the mel frontend held at fp32 via
`ct.transform.FP16ComputePrecision(op_selector=…)` — const-aware, because the
`clip(1e-8, 1e8)` guard and the `+1e-8` before the log are separate `const` ops
and holding only their consumer still lets `1e-8` cast to 0. **ONNX has no
equivalent per-op precision policy**: the repo's `to_fp16.py` casts the whole
graph, so "fp16 with an fp32 mel frontend" would mean re-deriving coremltools'
op-selection boundary in ONNX and then proving it landed where it was meant to.
The doc's own instruction ("simplest correct choice is whole-graph fp32 unless
fp16-with-fp32-mel is genuinely easy") resolves this: it is not easy, so fp32.
The cost is 15.3 MB against ~8.4 MB — 7 MB of CDN for a model that runs once per
2 s of speech. The overflow the fp16 path would hit is real and unrelated to
whether we could split it: the mel tensor spans 7.96e-9…7.10e5 (8.9e13 of range
against fp16's 1.1e12) and reaches 1.39e5 on ordinary speech, past fp16's 65504.

**Windowing** (`redimnet/windows.rs`) takes `&[TimeRange]` rather than
`&[AudioEventSegment]`: the tiling reads only `start` and `duration`, and
`AudioEventSegment` would have forced the diarizer to invent an `event` and a
`confidence` it does not have. Same maths, no dead fields.

## E.4 — Goldens

All fixture reads go through `ferment-engine-test-fixtures`. Three sentinel
entries added to `CANARY_ENTRIES`:
`Pyannote/one-speaker/reference_probabilities.bin`,
`SpeakerEncoder/speaker-a-1.embedding.bin`, `SpeakerEncoder/speaker-a-1.wav`.
The canary passes against a configured root (18 entries, 0 missing).

WAV-only kept, and I now have a second reason for it beyond the iOS one: the
engine's own `decode_path` scales 16-bit PCM by **32767** while `soundfile`
(which wrote the goldens) uses **32768**. That 3.05e-5 relative difference is a
pure global scale, which ReDimNet's first layer (`NormalizeAudio`: zero-mean,
unit-std) removes exactly — confirmed by the measured cosine of 0.999999762
through the real decode path. An MP3 fixture would not have been so forgiving
(upstream measured 0.678 over 75 windows).

`tests/ort_diarization.rs` is registered in `just engine-oracle`, immediately
after `ort_enhance`.

---

## Test evidence

```
$ cargo test -p ferment-engine-processors --lib pyannote
test result: ok. 50 passed; 0 failed; 0 ignored

$ cargo test -p ferment-engine-processors --lib redimnet
test result: ok. 9 passed; 0 failed; 0 ignored

$ ORT_DYLIB_PATH=… cargo test -p ferment-engine-processors --features ort \
      --test ort_diarization -- --ignored --test-threads=1 --nocapture
running 7 tests
test audio_shorter_than_one_frame_is_no_voices ... ok
test different_speakers_separate_by_a_clear_margin ...
  redimnet identity: same [0.8101531, 0.8187548]
                     cross [0.024805548, 0.10766896, -0.04426478, 0.08350187]  margin 0.7025
ok
test one_person_reading_is_one_voice ... ok
test pyannote_frame_decisions_match_the_reference_export ...
  pyannote one-speaker: 589/589 frame decisions agree
  pyannote turn-taking: 589/589 frame decisions agree
  pyannote overlap:     589/589 frame decisions agree
  pyannote overall: 1767/1767 = 1.000000
ok
test redimnet_windows_match_the_torch_goldens ...
  redimnet speaker-a-1: 5 windows, worst cosine 0.999999821
  redimnet speaker-a-2: 5 windows, worst cosine 0.999999940
  redimnet speaker-b-1: 5 windows, worst cosine 0.999999762
  redimnet speaker-b-2: 5 windows, worst cosine 0.999999881
  redimnet overall worst cosine 0.999999762
ok
test silence_is_no_voices ... ok
test two_people_are_two_voices_and_the_seam_is_found ...
  diarization halves (before, after) per voice: [(8.7245625, 0.0), (0.0, 11.21075)]
ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

**66 tests total** (50 pyannote lib incl. 5 fixture-backed oracles, 9 redimnet
lib, 7 ORT). The two-speaker case is a *perfect* split — 8.72 s of voice 0
before the seam and 0.00 s after, 11.21 s of voice 1 after and 0.00 s before —
i.e. **0% confusion** against the 5% bar.

Also measured, from the export harness (three-way, torch vs onnxruntime vs the
iOS goldens, which were written by a different toolchain in a different repo):

```
fixture        windows  min cos ORT|torch  min cos ORT|golden  max |‖v‖-1|
  speaker-a-1        5           1.000000            1.000000     1.19e-07
  speaker-a-2        5           1.000000            1.000000     1.19e-07
  speaker-b-1        5           1.000000            1.000000     1.19e-07
  speaker-b-2        5           1.000000            1.000000     1.79e-07
```
Worst 1−cos over the 20 windows: **2.0e-12**.

### Mutation testing — every parity assertion proven to fail when the data is wrong

Run against prebuilt test binaries so a parallel track's mid-edit compile break
could not mask a result. Fixtures backed up first, restored after, and verified
byte-identical (26 files by sha256).

| # | Mutation | Result | Verdict |
|---|---|---|---|
| M1 | flip one byte of `Pyannote/one-speaker/reference_activity.bin` | `every_reference_frame_decodes_to_the_reference_speakers` **RED**; ORT `pyannote_frame_decisions_…` **RED** | real |
| M2 | `+1 mod 7` on one entry of `turn-taking/reference_classes.bin` | `confidence_is_the_winning_class_probability` **RED** | real |
| M3 | `+0.25` on one cell of `overlap/reference_probabilities.bin` | 2 oracles **RED** | real |
| M4 | `+0.05` on `speaker-a-1.embedding.bin` window 2, component 0 | **RED** — "worst window cosine 0.998754799 is under the 0.999 bar" | real, and the bar is the thing that catches it |
| M5 | claim 4 windows instead of 5 in `speaker-b-1.embedding.json` (+ truncate the bin) | **RED** — "window count disagrees with the torch reference" | real |
| M6 | shift `speaker-a-2` golden `start_times` by **one sample** (62.5 µs) | **RED** — "speaker-a-2 window 0 start" | the 1e-9 time bar is real |
| M7 | code: drop the `.exp()` in `predict`, pass log-probabilities through | `one_person_reading_is_one_voice` **RED** (0 voices), `two_people_…` **RED** (0 voices); frame-decision parity stays **GREEN** | see below |
| M8 | code: delete the `[1,589,7]` shape guard | both guard tests **RED** | real |
| M9 | code: delete the `score < threshold` stop in `cluster` | 4 stitcher unit tests **RED**; ORT `two_people_…` **RED** (1 voice, expected 2) | real |

**M7 is the interesting one and I am reporting it rather than hiding it.** The
frame-decision parity suite stays green when the exponentiation is removed,
because argmax is invariant under a monotone transform — so that suite alone
does **not** cover the "graph speaks in logs" contract. Three other things do:
the stub-backend unit test
`the_graphs_log_probabilities_come_back_exponentiated`, and both end-to-end
diarization tests (the marginal/coverage ≥ 0.5 vote can never fire on negative
log-probabilities, so every voice vanishes). Without the E2E cases in this
suite, that mutation would have survived.

### Gates

| Gate | Result |
|---|---|
| `rustfmt --check` on every file I touched | clean |
| `cargo clippy -p ferment-engine-core -p ferment-engine-infer --no-deps --all-targets -D warnings` | clean (0 findings) |
| `cargo clippy -p ferment-engine-processors --no-deps --all-targets --features ort` | **0 findings in `pyannote/`, `redimnet/`, `ort_diarization.rs`** |
| `cargo test -p ferment-engine-models` | 14 pass, 1 fails — **not mine**, see "Not mine" below |
| `cargo test -p ferment-engine-test-fixtures` | 10 passed, canary green with the 3 new entries |
| `cargo deny check licenses` | **FAILS on the loro tree** (`bitmaps`/`im`/`sized-chunks` MPL-2.0, `xxhash-rust` BSL-1.0, all under `ferment-doc`) — identical to what Track H reported. **I added zero Rust dependencies**, so nothing here is mine. |
| root `pnpm test` | untouched (no JS/TS changed) |

`cargo clippy` had to be run with `--no-deps`: a plain
`clippy -p ferment-engine-processors` also lints `ferment-engine-media`, which
is mid-edit by another track and currently has two `needless_borrow` errors in
`platform/macos.rs`.

---

## THE DIARIZATION RESULT SHAPE — for the lead's cook mapping

The capability, in `crates/engine-core/src/capabilities.rs`, shaped after
`SpeechTranscribing`:

```rust
#[async_trait(?Send)]
pub trait SpeakerDiarizing: EngineService {
    async fn diarize(&self, audio_path: &Path, progress: ProgressSink)
        -> Result<Vec<LocalVoice>>;
}

pub struct SpeakerTurn {   // Copy, Serialize, Deserialize
    pub start_time: f64,   // seconds in the source file
    pub end_time: f64,     // NOT start_time + a uniform step; see the note below
    pub overlapped: bool,  // another voice was heard over the WHOLE of this stretch
}
impl SpeakerTurn { pub fn duration(&self) -> f64 }

pub struct LocalVoice {    // Serialize, Deserialize
    pub turns: Vec<SpeakerTurn>,
    pub vector: Vec<f32>,  // 192-d, unit length — EMPTY when no whole 2.005 s window fit
    pub confidence: f64,   // [0,1], mean winning-class probability over the voice's frames
}
```

Implemented by `ferment_engine_processors::PyannoteSpeakerSegmenter`
(`::new(backend)`). The impl lives in `pyannote/segmenter.rs`, **not** in
`capability_impls.rs` — deliberately, to avoid editing a file three other tracks
are in. Move it if you prefer the convention; it is a 6-line block.

There is also a lower-level `PyannoteSpeakerSegmenter::diarize(frames:
AudioFrames, progress) -> Result<Vec<LocalVoice>>` that skips the decode, which
is what the oracle drives. Same shape.

**Six things the wiring pass needs to know:**

1. **Voices are ordered by when they are first heard.** Stable across runs for
   the same audio; the index is the natural `turnSpeakers` value.
2. **`vector` may be empty**, and that is an answer, not a failure — a voice
   whose clean speech never filled a whole 2.005 s window. iOS stores this as an
   **all-zero slot of width `dim`** rather than an absent one, so `vectors`
   stays a plain `dim` stride for every reader; zeros are unambiguous because the
   encoder emits unit vectors. Mirror that.
3. **A voice always has at least one turn.** Groups that average away to nothing
   are dropped inside `diarize`, because "a speaker with nothing to say" is what
   the store refuses by name.
4. **`end_time` is not the next turn's `start_time` minus a gap.** A frame is
   61.9 ms of audio taken every 16.9 ms, so a turn runs to the end of its last
   frame's receptive field. Do not "normalise" turn boundaries.
5. **Turns of one voice never overlap each other**; turns of *different* voices
   can (that is what `overlapped` records). iOS flattens all voices' turns into
   one time-ordered column family joined back by `turnSpeakers`, and the order is
   **non-decreasing**, not strictly increasing — two voices may start in the same
   instant.
6. **The cook payload iOS writes** (`CookData.SpeakerEmbeddings`, processor
   version 2) is: `kind`, `confidences: [Double]` (per voice), `dim: Int`,
   `encoderName`, `encoderHash`, `segmenterName`, `segmenterHash`,
   `turnStarts/turnEnds: [Double]`, `turnOverlapped: [Bool]`,
   `turnSpeakers: [Int]`, `vectors: String` (packed). **The four name/hash fields
   are not in the engine result** — on iOS they come from the models'
   `weights_sha256` metadata. Here the equivalent identity is the fleet
   manifest's `sha256` for `pyannote_segmentation` / `redimnet_speaker`
   (both recorded above). Two checkpoints produce incomparable embeddings, so
   whatever stores a vector has to store the encoder's identity beside it.

Progress fractions reported by `diarize` (absolute, monotonic): 0.05 loading →
0.10…0.60 "Finding voices…" → 0.60…0.75 "Telling voices apart…" → 0.75…0.95
"Pooling voices…" → 1.0 "Done". The zero-speech paths report 1.0 immediately.

---

## Model delivery

| ModelId | file | bytes | sha256 | licence | source_url |
|---|---|---|---|---|---|
| `PyannoteSegmentation` | `pyannote_segmentation.onnx` | 5,986,908 | `057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25` | MIT | `https://huggingface.co/onnx-community/pyannote-segmentation-3.0/blob/733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx` |
| `RedimnetSpeaker` | `redimnet_speaker.onnx` | 16,056,349 | `11ce0e8a03f3b6fa911988c48cfadd2274e1d320e3b265d46de4898feeeaa135` | MIT | `https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b2-vox2-lm.pt` |

Both staged at `dist/models-publish/` (sha-verified after the copy). Both are
raw Studio-CDN downloads with an explicit Core ML gap
(`DIARIZATION_COREML_GAP`: "ONNX is the qualified baseline for this family; no
Core ML export audited here" — locked #10). `pyannote_segmentation.onnx` is
5.7 MB so it ships in-repo under `scripts/exports/pyannote/artifacts/`;
`redimnet_speaker.onnx` is 15.3 MB, over the >10 MB rule, so it is gitignored
and regenerated byte-for-byte by `export.py`.

**`MODEL_FLEET_VERSION` NOT bumped** — per the doc it bumps once, in the
last-landing track that adds models, and C/D/F are also adding. Currently
`2026-07-18.1`.

**Licence attribution.** FluidAudio (Apache-2.0) rides along:
`crates/engine-processors/notices/THIRD_PARTY_NOTICES.md` +
`LICENSE-FluidAudio.txt` (verbatim upstream copy), mirroring the iOS package's
own notice and naming exactly which files the shape came from. Every module
that carries adapted logic also names it in its own header.

---

## Deviations from the dispatch, and why

1. **`stage.py` written, not just a curl.** E.1 said "no export work: qualify it,
   pin the sha, stage it". A one-off download is not a qualification, so the
   qualification is a script with six refusal conditions, and it runs.
2. **Whole-graph fp32 for ReDimNet** (~16 MB) rather than fp16-with-fp32-mel.
   Reasoned above; the doc explicitly permits it.
3. **The `SpeakerDiarizing` impl lives in `pyannote/segmenter.rs`**, not in
   `capability_impls.rs`, to avoid a shared-file edit. Purely a placement call.
4. **`windows(for:)` takes `TimeRange`, not `AudioEventSegment`** — the tiling
   uses only `start`/`duration`, and the diarizer has no honest `event` or
   `confidence` to supply.
5. **The dead pre-sort in `windows(for:)` is dropped.** iOS sorts the segments by
   `startTime` and then sorts the *result* by offset; since each segment's
   windows depend only on that segment, and equal offsets produce identical
   plans, the pre-sort cannot change the output. Both ordering tests are ported
   and pass. iOS's own comment says the pre-sort "is not enough", which is how it
   got there.
6. **`ort_inventory_deploys_twenty_one_raw_studio_downloads` renamed** to
   `ort_inventory_deploys_only_raw_studio_downloads`, with the count kept as an
   assertion inside. Same spirit as Track H.5: a number in a test's name goes
   stale every round and has to be updated in two places.
7. **`third_party_ort` gained a `coreml_gap` parameter.** Its hard-coded reason
   ("macOS enhancement backing is VideoToolbox") is true of the 016 fleet and
   false of diarization, and a wrong reason in a manifest is worse than none.
   Five call sites now pass `ENHANCEMENT_COREML_GAP`.
8. **Two tests added beyond the ported Swift suites**: the stub-backend shape
   guard (M8 showed the guard was otherwise untested) and
   `the_graphs_log_probabilities_come_back_exponentiated` (M7 showed the `exp()`
   was otherwise only covered end-to-end).
9. **The "two people" case asserts confusion, not DER.** iOS ports a 236-LOC
   `DiarizationScorer` with Hungarian assignment; with two voices and two
   reference spans the assignment is a 2×2 and the useful number — how much of a
   voice's speech lands in the other person's half — is four lines. Measured 0%.
   If the lead wants a real DER scorer later it belongs in a shared test-support
   crate, not in this suite.

## Not mine (currently red in the shared tree)

- **`ferment-engine-models`: `default_manifest_is_complete_and_valid` fails on
  `resources.len()` — 5 found, 3 asserted.** The two extra resources are Track
  D's (`gigaam_vocab`, `ambernet_langs`); their count bump has not landed. My
  half passes: `models.len() == 32` (which includes both of mine),
  `every_shipped_entry_records_provenance`,
  `third_party_masters_carry_upstream_provenance` and
  `ort_inventory_deploys_only_raw_studio_downloads` are all green. I did have to
  add both ids to `all_model_ids()` (30 → 32) or the manifest would not validate.
- **`cargo deny check licenses`** — the loro tree, as Track H reported.
- **`cargo clippy -p ferment-engine-processors` with deps** — two
  `needless_borrow` errors in `engine-media/src/platform/macos.rs`.
- Mid-session, `beatthis/lattice.rs` and `video_scene_detector/shot_identity.rs`
  each broke the crate's test build for a while. Both cleared; the mutation runs
  were done against prebuilt binaries so those windows could not corrupt a
  result.

## What I could not run

- **Non-CPU execution providers.** Locked #10 makes CPU-EP the qualified
  baseline and defers DirectML/CUDA to the Windows box; the suite honours
  `FERMENT_ENGINE_ORT_EP` so the box can replay it unchanged. Both graphs pass
  `inspect_onnx.py`'s portability gate (`portable_directml_webgpu_qnn: true`), so
  there is no known blocker — just no measurement.
- **A Core ML lane.** Neither model has an audited `.mlmodelc` here; the iOS
  twins are built by the Swift package's own scripts. Recorded in the manifest as
  an explicit gap rather than omitted.
- **The 5.7 MB pyannote master and 15.3 MB ReDimNet master were not pushed to
  the CDN** — that is Dimitri's step. Both are staged in `dist/models-publish/`
  with the shas above.
- **Linux.** Everything here was run on macOS/arm64. Nothing in these modules is
  platform-gated (no Accelerate, no platform `cfg`), so the standing "CI lints
  Linux" risk is low, but it is unmeasured.
- **The full `just engine-oracle`** — it needs `FERMENT_ENGINE_MODELS_DIR` (a
  Core ML fleet) which is not on this machine. My leg of it was run directly:
  `cargo test … --test ort_diarization -- --ignored`, green.

## Files

**New:** `crates/engine-processors/src/pyannote/{mod,windows,powerset,turns,stitcher,segmenter,oracle_tests}.rs`
(1,945 LOC incl. tests), `crates/engine-processors/src/redimnet/{mod,windows,encoder}.rs`
(361 LOC), `crates/engine-processors/tests/ort_diarization.rs` (415 LOC),
`crates/engine-processors/notices/{THIRD_PARTY_NOTICES.md,LICENSE-FluidAudio.txt}`,
`scripts/exports/pyannote/{stage.py,pyproject.toml,uv.lock,artifacts/pyannote_segmentation.onnx}`,
`scripts/exports/redimnet/{export.py,pyproject.toml,uv.lock}`.

**Changed:** `crates/engine-core/src/capabilities.rs` (`SpeakerTurn`,
`LocalVoice`, `SpeakerDiarizing`), `crates/engine-infer/src/model.rs` (two
`ModelId` variants), `crates/engine-models/src/{fleet.rs,lib.rs}` (two entries,
the `coreml_gap` parameter, `all_model_ids`),
`crates/engine-processors/src/lib.rs` (two modules + re-exports),
`crates/engine-test-fixtures/src/lib.rs` (three canary entries), `Justfile`
(`ort_diarization` in `engine-oracle`), `scripts/exports/.gitignore`.

**Housekeeping.** `scripts/exports/{pyannote,redimnet}/.venv` total ~1.0 GB
(torch is most of it), gitignored and matching the nine sibling families;
`uv sync` recreates either from its lock file, so
`rm -rf scripts/exports/{pyannote,redimnet}/.venv` reclaims that at any time.
Disk at the end of the session: 51 GB free.
