# Track D report — Speech system: AmberNet + GigaAM + routing

**Executed 2026-08-31 against `main @ 5b97e84c`** in the shared working tree
(tracks B, C, E, F, G live in the same checkout).
Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track D".

All six items are done. Both model families are exported, qualified against
their stock origins, and driven end to end through the real ONNX Runtime — the
Russian smoke test transcribes its fixture verbatim. The seam-merge refactor is
proven three ways, including a 200 000-case differential run against the
pre-refactor code itself.

Mutation testing found **two holes and one latent bug** and closed all three:
the contiguous merge tier was indistinguishable from the LCS tier under every
existing test, nothing proved the port fed the encoder its real `length`, and
the softmax's max-subtraction was folded from a zero floor.

---

## Zero-trust pass: every constant re-read from source before coding

| Doc claim | Source | Verdict |
|---|---|---|
| AmberNet 224 LOC | `AmberNetLanguageID.swift` — **224 lines at HEAD** (229 at `88ac816ec`) | **confirmed** |
| `audio[1,480000]` f32 + `num_frames[1]` i32, 107 logits | `convert.py:NGC_URL/NUM_CLASSES`, `AmberNetLanguageID.predict` | **confirmed** |
| `scoringWindowSeconds 30`, `hopLength 160`, `minimumFrames 2`, `windowSamples 480_000` | `AmberNetLanguageID.swift:33-49` | **confirmed** |
| GigaAM 10 files / 1205 LOC | `wc -l Processors/GigaAM/*.swift` -> **1205 total**, 10 files | **confirmed, exactly** |
| `maxChunkSamples 240_000`, `overlapFrames 50`, `samplesPerEncoderFrame 640`, `blankTokenId 1024`, `maxSymbolsPerStep 10`, `maxTokensPerChunk 300` | `GigaAMConfig.swift` | **confirmed, all six** |
| mel frames `= (n−320)/160 + 1`; encoder frames = two rounds of `(L−1)/2 + 1` | `GigaAMConfig.melFrames` / `.encoderFrames` | **confirmed** |
| the `length` input is load-bearing | `GigaAMConfig.swift:96-104` + `EncoderModule` docstring in `convert.py` | **confirmed and now measured** — see below |
| Parakeet's 25 codes | `Parakeet.swift:42-46` | **confirmed, exact list and order-independent set** |
| `iw→he`, `jw→jv` | `TranscriberRouting.swift:19` | **confirmed** |
| endsSentence guard | `CaptionSegmenter.swift:51-64` (`c7ab8d6b2`) | **confirmed, and cross-checked against the running Ruby twin** |
| the coverage gate is buried upstream | `c23c77d1e` commit body | **confirmed** |

### Where the doc and the source disagree — source wins

1. **`TranscriberRouting.swift` is not in `Packages/FermentEngine/`.** The doc
   lists it under the engine sources; it actually lives at
   `Sources/FermentCuts/Features/Project/Processors/TranscriberRouting.swift`
   (app layer, because it consults `AppleSpeechTranscriber.supportedLanguageCodes()`
   live). Nothing changes for us — locked #5 removes the Apple lane, which is
   exactly what made it app-layer — but the port target is a *host* file, not an
   engine one, and it is worth the lead knowing that when the wiring pass looks
   for it.

2. **The doc says "the 107 tags live in model metadata upstream".** True on iOS
   (`user_defined_metadata["languages"]`, read back off the compiled model). It
   is **not** available to us: the ORT backend surfaces no user-metadata channel,
   so the list is a committed resource as the doc instructs
   (`crates/engine-processors/src/ambernet/ambernet_langs.json`, written by the
   export script straight from `model.cfg.train_ds.labels`). The label count is
   re-checked against the graph's logit count at every prediction, so the two
   cannot silently drift.

3. **`Scripts/gigaam/convert.py` already emits ONNX.** The doc frames GigaAM as
   "check SberDevices first; else torch→ONNX mirroring `convert.py`". Reading the
   file: it exports the four ONNX graphs as the *intermediate* on its way to Core
   ML (`export_onnx()`, opset 17), with the IO names the Swift port pins. So the
   redshift exporter is that script with the Core ML half deleted — not a new
   conversion. See "Upstream check" below for what SberDevices actually offers.

4. **`melFrames` for a 161..=319-sample buffer.** iOS's own comment says Swift's
   truncating `/` answers 1 where upstream's `out_len` answers 0. Rust's `/`
   truncates identically, so the port inherits the same answer; pinned by test
   rather than left as folklore (`config.rs::a_sub_window_buffer_yields_no_frames`).

---

## Upstream check before converting anything (D.2's first instruction)

Researched 2026-08-31, not assumed:

- **`salute-developers/GigaAM` is MIT** ("2024/12 — MIT License", badge and
  README) and ships an **official `model.to_onnx()`** helper. It emits the
  library's own **two-graph** shape (encoder + fused decoder-joint) with a
  **dynamic-length encoder** — not the static 15 s, `length`-masked, four-graph
  split this port's decode loop is built around, and with no `token_id`/
  `token_prob` argmax folded into the joint.
- Community re-exports exist — `istupakov/gigaam-v3-onnx` and
  `cnonim/gigaam-v3-e2e-rnnt-onnx`, both MIT, both derived from
  `ai-sage/GigaAM-v3` — with the same shape mismatch plus provenance we do not
  control.
- **Verdict:** the checkpoint is official and MIT; the split is ours, mirrored
  1:1 from the iOS converter so the two fleets share one export decision.
  Parity is *asserted against the checkpoint's own torch modules* in the export
  script rather than assumed.
- **AmberNet:** `nvidia/langid_ambernet` is an **NGC** model (huggingface.co
  returns 401 for that path). NVIDIA states: "The license to use this model is
  covered by the NGC TERMS OF USE unless another License/Terms Of Use/EULA is
  clearly specified. By downloading the public and release version of the model,
  you accept the terms and conditions of the NGC TERMS OF USE." There is **no
  SPDX identifier**, so the manifest records
  `LicenseRef-NVIDIA-NGC-Terms-of-Use` rather than claiming one it does not have.

---

## D.1 — AmberNet

**Export** (`scripts/exports/ambernet/export_ambernet.py`, mirrors
`Scripts/language_id/convert.py`): pulls `ambernet.nemo` v1.12.0 from the NGC
public API (116 049 920 bytes, sha256 verified against the pinned digest),
re-checks ten checkpoint config values against the constants the graph bakes in,
then traces frontend + NeMo's own encoder/decoder to one ONNX graph.

`audio[1, 480000] f32` + `num_frames[1] i32` → `logits[1, 107]`,
`embedding[1, 512]`. Opset 17, fp32, 117.0 MB.

The `MEL_SCALE = 2^10` lift is kept even though this export is fp32, where it is
provably neutral (`log(S·mel + S·guard) = log(mel+guard) + log(S)`, and the
per-feature normalisation subtracts a per-bin mean). Keeping it means this graph
and the iOS Core ML graph are the same graph, and an fp16 rung stays available
without re-deriving the guard.

**Port** (`crates/engine-processors/src/ambernet/`): `build_window` and
`hypotheses` line for line, plus the driver. `LanguageIdResult { hypotheses:
[{code, probability}], speech_seconds }`. **No coverage threshold anywhere** —
an empty `Some([])` short-circuits before the model loads, and that is the only
gate.

**One deliberate improvement over iOS.** iOS computes `speechSeconds` in the app
layer as `min(30, Σ max(0, duration))` — the segments' *claim*. The port takes
the smaller of that and what the window actually gathered, so a segment running
past the end of the file cannot inflate the reported number
(`scored_seconds_cannot_exceed_what_was_actually_gathered`). All four of the iOS
test's cases still hold (30 / 2.5 / 4 / 0).

---

## D.2 — GigaAM

**Export** (`scripts/exports/gigaam/export_gigaam.py`): four graphs + the
1024-piece vocabulary from `ai-sage/GigaAM-v3` @ `e2e_rnnt`.

```
preprocessor  audio [1, 240000]              -> mel [1, 64, 1499]              0.45 MB
encoder       mel [1, 64, 1499], length [1]  -> encoded [1, 768, 375],
                                                encoded_len [1]              885 MB fp32
                                                                             443 MB fp16
decoder       x [1,1], h_in/c_in [1,1,320]   -> dec [1,1,320], h_out, c_out    4.6 MB
joint         enc [1,768,1], dec [1,320,1]   -> token_id, token_prob           2.7 MB
```

**Encoder precision — fp16 ships.** The doc's fp16 trigger (">2 GB") does not
fire, so fp32 was permissible; fp16 was chosen anyway because it is the rung
Parakeet already ships in this repo and it halves a 885 MB download. Measured
drift against the fp32 torch origin is **9.5e-3 through Python ORT 1.29 and
2.2e-3 through the vendored runtime**, on encoder states spanning −1.62..1.58 —
0.3% and 0.07% of the range respectively, tighter *in relative terms* than
Parakeet's shipped fp16 encoder (5.0e-4 on ±0.06 = 0.8%). The fp32 master stays
on disk as the golden reference and is gitignored; only fp16 is staged for the
CDN.

**Port** (`crates/engine-processors/src/gigaam/`): chunk planning, the RNN-T
greedy decode, the joint, and the postprocessing, all mirroring the Swift.

**The `length` input is load-bearing, and that is now measured, not asserted.**
Feeding the encoder `length = 1499` on a chunk with 499 real mel frames moves
the encoder states over the *shared* span by **2.08** — a number of the same
order as the states themselves. The oracle runs both lengths through one graph
and fails if they agree.

**Three deliberate deviations from iOS, all documented in the code:**

1. **Chunks decode in order, not three-at-a-time.** Upstream's
   `parallelChunkConcurrency = 3` buys wall time against shared Core ML handles;
   it cannot change output, because chunks decode from a zeroed LSTM state by
   design and the seam merge resolves the overlap at token level. The ORT
   backend serialises per session anyway.
2. **No ANE-aligned buffers, no `projectionDirty` flag, no reusable feature
   provider.** All three are Core ML allocator economics with no behavioural
   content; the Rust loop passes `Vec<f32>` and the decode is identical.
3. **`encoder_frames` asserts the layout instead of probing it.** Upstream's
   `precondition` reasoning ported verbatim: a probe would turn a mis-export into
   wrong numbers instead of a stop.

**The vocabulary is compiled into the crate** (`include_str!`), like Parakeet's,
so it is not a download and does not appear in `model_files`.

---

## D.3 — `token_seam_merge`, and the proof it did not move

`crates/engine-processors/src/token_seam_merge/` — the three-tier merge
(contiguous → LCS → midpoint) generic over `TokenSeamRow { token, timestamp }`,
with `sequence_matcher` moved under it (it had exactly one consumer and now has
two). Parakeet's `merge_chunks` and GigaAM's are both ~20-line adapters that zip
their own row shape, call the shared body, and unzip: Parakeet's TDT `duration`
lane and GigaAM's confidence lane ride through as opaque payload.

**The gate, three ways.**

1. `parakeet::oracle_tests::parakeet_merge_chunks_tests` (6 cases: contiguous
   dedup, LCS rewrite, midpoint cut, both empty sides, disjoint-in-time) are
   unchanged and green.
2. `parakeet_transcribes_speech_end_to_end` drives the refactored merge through
   the real masters and still produces "quick brown fox … lazy dog".
3. **A differential run against the pre-refactor code itself.** I extracted
   `git show HEAD:.../parakeet/postprocess.rs` into a scratch crate beside the
   new module and ran **200 000 random chunk pairs** through both — small token
   alphabets and tight timestamps so the matcher actually fires:

   ```
   200000 random chunk pairs: byte-identical to the pre-refactor merge
   (146963 with both sides non-empty, 123983 where the merge removed rows)
   ```

   Tokens, timestamps, **durations**, confidences and `removed_count` all agree
   on every case. That is the doc's "byte-identical" claim demonstrated rather
   than argued.

**A hole the existing goldens could not see, found and closed.** Mutation-testing
the refactor turned up that deleting the *contiguous* tier outright left all 13
merge tests green: on every input the suite had, the LCS tier picks the same
anchors, so the two tiers were indistinguishable and the first tier was
effectively untested. The tiers genuinely differ when the contiguous run is long
enough to win but shorter than the LCS — then the anchors end earlier and the
right-hand side's rewritten tokens are taken verbatim instead of the left's.
`a_long_enough_contiguous_run_outranks_a_longer_lcs` pins exactly that case, and
the same mutation now goes red with the two answers printed side by side.

**Deleted while in there:** `parakeet::postprocess::deduplicate_boundary` (a
suffix-prefix deduplicator from before the three-tier merge landed) — `pub` with
zero consumers anywhere in the workspace, and its five tests only exercised
itself. Its five tests go with it.

---

## D.4 — Routing

`crates/engine-processors/src/transcriber_routing.rs`, a pure function with no
model, no chip axis and no Apple lane:

```rust
pub enum Route { GigaAm, Parakeet, Unsupported { code: String } }
pub fn route_code(code: &str) -> Route;
pub fn route<'a>(hypothesis_codes: impl IntoIterator<Item = &'a str>) -> Route;
pub fn model_files(route: &Route) -> &'static [ModelId];
```

The transcribers declare their own knowledge, as upstream does:
`Parakeet::SUPPORTED_LANGUAGE_CODES` (the 25) and `GigaAm::SUPPORTED_LANGUAGE_CODES`
(`["ru"]`), plus `MODEL_FILES` on each. `route` walks the top 3 and takes the
first supported one; nothing supported in the top three returns the **top**
hypothesis as `Unsupported`, and an empty hypothesis list returns `und`.

`route` takes `&str`s rather than `LanguageHypothesis`es on purpose: the
probabilities play no part in the decision (the list arrives sorted), and this
keeps the routing module free of any dependency on the language-ID module.

**The acceptance case passes:** `route(["be", "ru", "uk"]) == GigaAm` — the live
2026-08-26 «be 0.885 / ru 0.113» Russian file transcribes.

---

## D.5 — Caption segmenter abbreviation guard

`CaptionSegmenter::ends_sentence` mirrors `c7ab8d6b2` exactly: a sentence-final
`.`/`!`/`?` does **not** end a caption when the word starts with a capital or a
digit, does not end in `..`, and has fewer than 3 characters outside
capitals/digits/periods.

**Cross-checked against the running Ruby twin, not just the Swift.** I found
`ferment-api/packs/project/app/lib/captions/segmenter.rb` and executed its
`sentence_end?` against my 15-case table:

```
OK   "friend."  ruby=true rust_expected=true
OK   "Hello."   ruby=true rust_expected=true
OK   "Word."    ruby=true rust_expected=true
OK   "Dr."      ruby=false rust_expected=false
OK   "U.S."     ruby=false rust_expected=false
OK   "9."       ruby=false rust_expected=false
OK   "No.9"     ruby=false rust_expected=false
OK   "Yes..."   ruby=true rust_expected=true
OK   "Ok.."     ruby=true rust_expected=true
OK   "Hey."     ruby=false rust_expected=false
OK   "word"     ruby=false rust_expected=false
OK   ""         ruby=false rust_expected=false
OK   "Wow!"     ruby=true rust_expected=true
OK   "Ok!"      ruby=false rust_expected=false
OK   "США."     ruby=false rust_expected=false
```

Three findings the port would have missed by transcribing the Swift alone:

- **The guard is not period-specific.** `"Ok!"` reads on, because the clause
  order tests the sentence-ender first and then applies the abbreviation rule to
  whatever it was. Both twins do this; both suites were silent about it. Pinned.
- **Short capitalised words read on.** `"Hey."` has two characters outside the
  capital/digit/period class, so it never ends a caption. Shared quirk, pinned.
- **One genuine divergence, stated rather than hidden:** Ruby's `[[:digit:]]` is
  Nd only; Swift's `Character.isNumber` and Rust's `char::is_numeric` are Nd|Nl|No.
  A word opening with a Roman numeral (`Ⅷ.`) reads on here and on iOS, and cuts
  in Ruby. No transcriber emits those. Recorded in the doc comment.

The Ruby suite's own three abbreviation cases (`U.S.`, `Dr.`, and the Cyrillic
«США.») are ported as caption-level tests, not just predicate tests.

---

## D.6 — INVESTIGATE AND REPORT: the coverage gate

**Finding: the gate is in `crates/project`, not in the engine. Per the doc, I
left it alone. It is the lead's to remove.**

- `crates/engine-processors/`, `crates/engine-core/`, `crates/engine-models/`
  and `crates/engine-infer/` contain **no** transcript coverage gate. The only
  `coverage` references in the engine are the AED's own
  `AudioEventSummary.speech.coverage` statistic (which is a *measurement*, and
  stays) and its tests.
- The gate is `crates/project/src/processors.rs:471`:

  ```rust
  let events = read_audio_events(&context)?;
  if speech_coverage(&events) < gate_threshold(&context)? {
      return Ok(Verdict::skipped());
  }
  ```

  fed by `CoverageGateOptions` (`crates/project/src/dto.rs`) and
  `speech_coverage()` (`processors.rs:986`).
- **What upstream did** (`c23c77d1e`, transcript v4): "The speech coverage gate
  is buried … speech is a property of EVENTS, not of the timeline's extent —
  segments exist → transcribe (a music video with speech on a third of its
  length gets captions; the no-speech case skips at languageId + join). Music
  lanes keep their `musical.coverage 0.5`."
- **So the lead's edit is:** delete the `speech_coverage(...) < gate_threshold`
  branch at `processors.rs:471` and leave the four `musical_coverage` gates at
  lines 519/565/598/627 exactly as they are. The skip moves to the languageId
  node: no speech segments → no hypotheses → `Unsupported` → no transcript.
  `speech_coverage()` becomes dead once that branch goes.

---

## Test evidence

All numbers below were produced by running the commands, not estimated.

| Suite | Command | Result |
|---|---|---|
| Track D unit suites | `cargo test -p ferment-engine-processors --lib -- gigaam ambernet token_seam transcriber_routing caption_segmenter parakeet` | **165 passed, 0 failed** |
| Seam-merge differential vs the pre-refactor code | scratch crate, 200 000 random chunk pairs | **byte-identical**, 123 983 of them non-trivial |
| ORT oracles (all families) | `cargo test -p ferment-engine-processors --features ort --test ort_semantic -- --ignored` | **16 passed, 0 failed** |
| `ferment-engine-models` | `cargo test -p ferment-engine-models --lib` | 14 passed, **1 failed — Track E's**, see below |
| `cargo fmt --check` on every file I touched | | clean (the one `fleet.rs` diff the checker reports is Track F's `YUNET_SOURCE` line) |
| `cargo clippy -p ferment-engine-models -p ferment-engine-infer --all-targets -D warnings` (1.95.0) | | clean |
| `cargo clippy -p ferment-engine-processors --all-targets --no-deps -D warnings` (other tracks' five in-flight lints `-A`'d) | | **clean** after fixing 5 findings in my files — see "Not run / not verified" |
| `cargo deny check licenses` | | **fails only on the four pre-existing loro crates**; my additions introduce no Rust dependency at all |

### The six new ORT oracles, with their measured deltas

```
running 6 tests
gigaam decoder dec [1, 1, 320]: observed maximum difference 0.0000013709068
gigaam decoder h_out [1, 1, 320]: observed maximum difference 0.0000013709068
gigaam decoder c_out [1, 1, 320]: observed maximum difference 0.0000016093254
gigaam joint token_prob: observed maximum difference 0
gigaam mel [1, 64, 1499]: observed maximum difference 0.00041675568
test gigaam_decoder_matches_torch_origin ... ok
test gigaam_preprocessor_matches_the_checkpoints_own_featurizer ... ok
test gigaam_joint_matches_torch_origin_exactly_on_the_token_id ... ok
ambernet logits [1, 107] at num_frames 500: observed maximum difference 0.0008146763
ambernet logits [1, 107] at num_frames 3000: observed maximum difference 0.000044345856
ambernet embedding [1, 512]: observed maximum difference 0.0007199049
test ambernet_matches_nemo_origin_and_honours_num_frames ... ok
gigaam encoded [1, 768, 375]: observed maximum difference 0.0021915436
gigaam encoded [768, 125] at length 499: observed maximum difference 0.0020923615
test gigaam_encoder_matches_torch_origin_and_masks_by_length ... ok
transcript: "быстрая коричневая лиса прыгает через ленивую собаку."
test gigaam_transcribes_russian_speech_end_to_end ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 10 filtered out; finished in 9.13s
```

The Russian E2E is the real production path: AAC in a container → AudioToolbox
decode → resample → chunk plan → preprocessor → encoder → RNN-T greedy decode →
seam merge → words. The fixture (`tests/fixtures/speech-ru.m4a`, 17 455 bytes)
is `say -v Milena` reading a fixed sentence, the same recipe as the existing
`speech-fox.m4a`. It comes back **verbatim, punctuation and all**.

### The full suite, refactor gate included

```
running 16 tests
test gigaam_joint_matches_torch_origin_exactly_on_the_token_id ... ok
test gigaam_decoder_matches_torch_origin ... ok
test parakeet_preprocessor_matches_python_origin ... ok
test music_tagger_heads_match_python_origin ... ok
test gigaam_preprocessor_matches_the_checkpoints_own_featurizer ... ok
test parakeet_joint_matches_python_origin ... ok
test parakeet_decoder_matches_python_origin ... ok
test dinov2_embedding_matches_python_origin ... ok
test dinov2_fp16_master_still_answers_the_origin ... ok
test dinov2_encoder_puts_light_nearer_than_content ... ok
test dasheng_embedding_matches_python_origin ... ok
test ambernet_matches_nemo_origin_and_honours_num_frames ... ok
test gigaam_encoder_matches_torch_origin_and_masks_by_length ... ok
test parakeet_encoder_matches_python_origin ... ok
test gigaam_transcribes_russian_speech_end_to_end ... ok
test parakeet_transcribes_speech_end_to_end ... ok

test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.04s
```

### Export-time parity, from the scripts themselves

```
[gigaam parity]
    frontend pt<->FeatureExtractor     max|delta| = 1.19209e-05
    preprocessor pt<->onnx             max|delta| = 2.38419e-05
    encoder pt<->onnx                  max|delta| = 4.64916e-06
    decoder pt<->onnx                  max|delta| = 5.96046e-08
    joint token pt<->onnx              max|delta| = 0
    joint prob pt<->onnx               max|delta| = 0

[gigaam qualification vs the torch origin, CPU EP]
    preprocessor (conv re-expression)            max|delta| = 0.000416756
    encoder gigaam_encoder.onnx                  max|delta| = 5.81145e-06
    encoder gigaam_encoder.onnx (length 499)     max|delta| = 3.45707e-06
    encoder gigaam_encoder.fp16.onnx             max|delta| = 0.00953028
    encoder gigaam_encoder.fp16.onnx (length 499) max|delta| = 0.00740629
    length sensitivity (the load-bearing input): 2.08247

[ambernet parity]
    5 s window: logits nemo<->graph          max|delta| = 0.00153542
    5 s window: logits graph<->onnx          max|delta| = 2.02656e-06
    full 30 s window: logits nemo<->graph    max|delta| = 5.60284e-06
    full 30 s window: logits graph<->onnx    max|delta| = 3.42727e-06
    num_frames 500: top-1 'la' logit 3.1244
    num_frames 3000: top-1 'yo' logit 3.2087
    num_frames sensitivity (the load-bearing input): 4.765
```

The 5 s `nemo<->graph` figure (1.5e-3) is the pre-emphasis tail iOS documents:
NeMo zeroes the pre-emphasised waveform past `seq_len` while this graph
pre-emphasises the already-zero-padded buffer. Top-1 agreement is asserted
exactly at both lengths in the export script and in the oracle.

---

## Mutation testing — every oracle proven to fail when the answer is wrong

| Mutation | Result | Verdict |
|---|---|---|
| the abbreviation guard deleted (`ends_sentence` returns `true` for any `.!?` ending) | **3 red**: `test_abbreviation_period_is_not_a_sentence_end`, `test_the_abbreviation_guard_is_unicode`, `test_ends_sentence_clause_by_clause` | real |
| `ambernet_logits_out.bin`, `gigaam_encoded_out.bin`, `gigaam_mel_out.bin` each perturbed by +0.25 | **3 oracles red**, each reporting the injected offset | real |
| the contiguous tier removed from `token_seam_merge` | **was 13 green → now 1 red** | **was garbage, now fixed** |
| the port always feeds the encoder `length = 1499` instead of the chunk's real mel frames | **the E2E stayed green**; a new unit test now goes red | **was garbage, now fixed** |

**Golden perturbation** — the deltas name the offset, so the oracle is reading
the file it claims to read:

```
######## MUTATION A: goldens +0.25 ########
test gigaam_decoder_matches_torch_origin ... ok
test gigaam_preprocessor_matches_the_checkpoints_own_featurizer ... FAILED
test gigaam_joint_matches_torch_origin_exactly_on_the_token_id ... ok
test ambernet_matches_nemo_origin_and_honours_num_frames ... FAILED
test gigaam_encoder_matches_torch_origin_and_masks_by_length ... FAILED
test gigaam_transcribes_russian_speech_end_to_end ... ok
gigaam mel [1, 64, 1499]: maximum difference 0.25041676 exceeds 0.005
ambernet logits [1, 107] at num_frames 500: maximum difference 0.25081468 exceeds 0.005
gigaam encoded [1, 768, 375]: maximum difference 0.4276017 exceeds 0.05
test result: FAILED. 3 passed; 3 failed
```

(The decoder, joint and E2E stay green because they read goldens I did not
touch — which is the control.)

**The contiguous tier** — 13 green before the new test existed, 1 red after,
with both tiers' answers printed:

```
######## MUTATION B2: the contiguous tier removed ########
test token_seam_merge::tests::a_long_enough_contiguous_run_outranks_a_longer_lcs ... FAILED
left: [10, 20, 30, 40]
right: [10, 20, 99, 40]
test result: FAILED. 13 passed; 1 failed
```

**The load-bearing `length`, and the second hole this pass found.** The ONNX
oracle proves the *graph* honours `length`; nothing proved the *port* hands it
the right value. Mutating `run_encoder` to always send the padded 1499 left the
Russian E2E **green** — on a short, clean, single-chunk clip the transcript is
identical either way:

```
######## MUTATION C: the port always feeds length = 1499 ########
transcript: "быстрая коричневая лиса прыгает через ленивую собаку."
test gigaam_transcribes_russian_speech_end_to_end ... ok
test result: ok. 1 passed; 0 failed
```

So the plumbing had no gate. `the_encoder_is_told_the_chunks_real_mel_frames_not_the_padded_window`
now drives `transcribe` against a recording backend and asserts the encoder was
told the chunk's own mel-frame count, with a second assertion that the test
cannot go vacuous by the chunk happening to fill the window. Re-running the same
mutation:

```
######## MUTATION C2: the port always feeds length = 1499 ########
test gigaam::processor::tests::the_encoder_is_told_the_chunks_real_mel_frames_not_the_padded_window ... FAILED
  left: [1499]
 right: [299]
test result: FAILED. 8 passed; 1 failed
```

### One latent bug a read-through turned up, fixed

Re-reading the port against the Swift while the tree was blocked:
`hypotheses` folded the softmax maximum from a `0.0` floor. Softmax is
shift-invariant, so on AmberNet's real ±20 logits that is indistinguishable from
the Swift's `logits.max()` — but on a set of large *negative* logits every
exponent underflows to zero, the total becomes zero, and all 107 probabilities
come back NaN. Now folds from `NEG_INFINITY` (the model's own maximum, as the
Swift does), with `all_negative_logits_do_not_underflow` pinning it alongside
the existing overflow test.

---

## Model files, shas and licences

Staged under `dist/models-publish/` for Dimitri's CDN upload.

| File | Bytes | sha256 | Licence | Source |
|---|---|---|---|---|
| `ambernet_langid.onnx` | 116 985 602 | `ba773727c51bffe590241a33cb5c1d7567c5fa8b2bd20a6d4859dd9a5c845953` | `LicenseRef-NVIDIA-NGC-Terms-of-Use` | NGC `langid_ambernet` v1.12.0 |
| `ambernet_langs.json` | 646 | `9e515c3c7932659fd1e6c3febc395529d0a8092328adb9f5e75185a04bb523d0` | same | same |
| `gigaam_preprocessor.onnx` | 454 608 | `6661deb17bfbd3a82bebc66cefeaf406c66d7398253c105384761cd4f4217967` | MIT | `ai-sage/GigaAM-v3` @ `e2e_rnnt` |
| `gigaam_encoder.fp16.onnx` | 442 853 782 | `8f95f8f608df482bc6b38787f1b0bc1dc50cca3b44064aceef44f7169ec987fd` | MIT | same |
| `gigaam_decoder.onnx` | 4 599 946 | `ab2dd54773da9b1ebfbb184191cf9eaab8afac5bf3e24b936fe6173359fd8185` | MIT | same |
| `gigaam_joint.onnx` | 2 713 887 | `6e215897f2a47d6982f1e35e459c5728c80acef695fdd991f4de0d8f95cb823a` | MIT | same |
| `gigaam_vocab.json` | 12 406 | `260c932355adc98a7440d49887f72a2fca42256971ad07a4e91265e93859ff69` | MIT | same |

Local-only reference, gitignored, **not** for the CDN:
`gigaam_encoder.onnx` (fp32, 885 369 291 bytes,
`0b47815698b18ea8c018baf11b21fab0e8264383b25d7e9899ee709eee70556d`) — the fp16
rung's qualification baseline.

The NGC checkpoint itself is pinned by digest in the export script:
`ambernet.nemo` v1.12.0, 116 049 920 bytes,
`2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681`. The script
refuses to proceed on a mismatch.

`MODEL_FLEET_VERSION` is **not** bumped — per the doc it bumps once, in the
last-landing model track, and Tracks C/E/F also add models.

---

## What the lead needs for the wiring pass

**New public surface** (all in `ferment-engine-processors`):

```rust
// crate::ambernet
pub struct AmberNetLanguageId;                   // ::new(Backend)
pub async fn identify(&self, source: &dyn AudioSampleSource,
                      speech_segments: Option<&[AudioEventSegment]>,
                      progress: ProgressSink) -> Result<LanguageIdResult>;
pub struct LanguageIdResult { hypotheses: Vec<LanguageHypothesis>, speech_seconds: f64 }
pub struct LanguageHypothesis { code: String, probability: f32 }  // Serialize/Deserialize

// crate::gigaam
pub struct GigaAm;                                // ::new(Backend)
pub async fn transcribe(&self, frames: AudioFrames,
                        speech_segments: Option<&[AudioEventSegment]>,
                        progress: ProgressSink) -> Result<Vec<TranscribedWord>>;
pub const SUPPORTED_LANGUAGE_CODES: [&str; 1];
pub const MODEL_FILES: [ModelId; 4];
pub const LANGUAGE: &str;                         // "ru"

// crate::transcriber_routing
pub enum Route { GigaAm, Parakeet, Unsupported { code: String } }
pub fn route_code(&str) -> Route;
pub fn route(impl IntoIterator<Item = &str>) -> Route;
pub fn model_files(&Route) -> &'static [ModelId];

// crate::parakeet::Parakeet — new associated consts, same shape as GigaAm's
pub const SUPPORTED_LANGUAGE_CODES: [&str; 25];
pub const MODEL_FILES: [ModelId; 4];
```

**Five new `ModelId` variants**: `GigaAmPreprocessor`, `GigaAmEncoder`,
`GigaAmDecoder`, `GigaAmJoint`, `AmberNetLangId`, all in `all_model_ids()` and
`fleet.rs` with licence + source URL.

**Three things I deliberately did not do, because they are yours:**

1. **`impl core::SpeechTranscribing for GigaAm`** is not written.
   `capability_impls.rs` was being edited by another track for most of this
   session and the trait impl is a wiring decision (which transcriber the facade
   hands back is exactly what the router decides). The adapter is mechanical:
   `GigaAm::transcribe` already returns `Vec<core::TranscribedWord>` in absolute
   time with `id::short()` ids, so the impl is
   `CaptionSegmenter::captions(&words)` + `words.join(" ")` +
   `language: words.is_empty().then_some(None).unwrap_or(Some("ru"))` — which is
   exactly what the Parakeet impl does at `capability_impls.rs:500-511`. Note
   iOS reports `language: nil` when the decode yields nothing, deliberately.
2. **No `LanguageIdentifying` capability trait in `engine-core`.** The Track D
   boundary in the dispatch doc does not list `engine-core`, and the trait's
   shape depends on how the languageId cook is keyed. `AmberNetLanguageId` is
   directly callable meanwhile.
3. **The `DownloadTranscriberModel` graph node and the languageId cook.**
   `model_files(&route)` gives you the `ModelId`s; the vocabularies are compiled
   in, so there is nothing else to fetch.

**Also worth your attention:**

- **`crates/project/src/processors.rs:471`** — the speech coverage gate to
  delete (D.6 above), leaving the four `musical_coverage` gates alone.
- **`capability_impls.rs` has a private copy of `speech_segment_coalescer::coalesce`**
  (`coalesce_segments`, lines 583+) that is character-for-character the same
  function. I did not touch it (shared file, live track); GigaAM calls the real
  one. Worth collapsing in the integration pass.
- **Two shared literals in `fleet.rs` that Tracks D and E both bump:**
  `ort_inventory_deploys_only_raw_studio_downloads` (now 29) and
  `third_party_masters_carry_upstream_provenance` (now 18). Whoever lands last
  re-runs and adjusts.

---

## Files

**New**
- `crates/engine-processors/src/ambernet/{mod.rs,langs.rs,ambernet_langs.json}`
- `crates/engine-processors/src/gigaam/{mod.rs,config.rs,decoder.rs,postprocess.rs,processor.rs,vocab.rs,vocab.json}`
- `crates/engine-processors/src/token_seam_merge/mod.rs`
  (+ `sequence_matcher.rs`, moved verbatim from `parakeet/`)
- `crates/engine-processors/src/transcriber_routing.rs`
- `crates/engine-processors/tests/fixtures/speech-ru.m4a`
- `scripts/exports/ambernet/{export_ambernet.py,gen_goldens.py,pyproject.toml,artifacts/}`
- `scripts/exports/gigaam/{export_gigaam.py,gen_goldens.py,pyproject.toml,artifacts/}`
- 17 goldens under `scripts/exports/rust_goldens/{ambernet_*,gigaam_*}.bin`

**Changed**
- `crates/engine-processors/src/lib.rs` (three module declarations + two re-exports)
- `crates/engine-processors/src/caption_segmenter.rs` (the guard + its tests)
- `crates/engine-processors/src/parakeet/{mod.rs,postprocess.rs,processor.rs,oracle_tests.rs}`
- `crates/engine-processors/tests/ort_semantic.rs` (six oracles appended)
- `crates/engine-processors/notices/THIRD_PARTY_NOTICES.md` (the seam-merge lineage)
- `crates/engine-infer/src/model.rs`, `crates/engine-models/src/{fleet.rs,lib.rs}`
- `.gitignore` (the three over-10 MB masters + the two checkpoint dirs)

---

## Shared-tree note (needs your attention)

Six tracks are live in this one checkout and it showed:

- **Another track committed my staged rename.** `git mv parakeet/sequence_matcher.rs
  token_seam_merge/sequence_matcher.rs` sat in the index; commit `7945bdc2`
  ("docs/fleet: mark R1's two open findings as superseded by R4") swept it in.
  No damage — the file is byte-identical and in the right place — but a
  `commit -a` in a six-agent tree picks up whatever is staged, and next time it
  could be a half-finished edit. Worth a word to the fleet.
- `crates/engine-processors` was **uncompilable for long stretches** while
  Tracks B, C, F and G were mid-edit (`beatthis/lattice.rs`,
  `video_scene_detector/shot_identity.rs`, `video_saliency/analyzer.rs`,
  `video_encoder.rs`). Every test run in this report had to be retried until the
  tree happened to be green; the crate-wide clippy gate is the one thing I could
  not land in a green window (see below).
- `cargo test -p ferment-engine-models --lib` is red on Track E's
  `pyannote_segmentation` fleet entry, not on anything of mine.

---

## Not run / not verified

- **`cargo test -p ferment-engine-models --lib` is red on one test that is not
  mine:** `default_manifest_is_complete_and_valid` fails with
  `Models("unknown model id 'pyannote_segmentation'")` — Track E has the fleet
  entry in but the `ModelId` variant is not yet reachable from that path. My
  five ids are all in `all_model_ids()`; the other 14 tests in that crate pass.
- **`cargo clippy -p ferment-engine-processors --all-targets -D warnings` cannot go
  green in this tree — but not because of Track D.** Plain `clippy -p` dies in
  `crates/engine-media/src/platform/macos.rs` (two `needless_borrow`s, Track G),
  so I ran it `--no-deps` with the other tracks' five in-flight lints allowed at
  the command line. That surfaced **five findings in my files, all fixed**:
  two `assertions_on_constants` in `gigaam/config.rs` (the chunk-geometry
  invariants are now `const _: () = assert!(…)` — a compile-time gate is
  strictly stronger than the test they were in), two `manual_is_multiple_of`
  in those same new const asserts, and one `unnecessary_cast` in the new spy
  test. Remaining findings all belong to other tracks:
  `beatthis/lattice.rs` (manual range contains), `energy_structure.rs`
  (int_plus_one), `audio_minimap_composer/layout.rs` (unused_mut),
  `beat_queen_mary/mod.rs` (neg_cmp_op_on_partial_ord),
  `video_saliency/{u2netp,yunet}.rs` (unused_variables, identity_op),
  `engine-media/platform/macos.rs` (needless_borrow). With exactly those five
  lint kinds allowed on the command line, `clippy --all-targets --no-deps
  -D warnings` over the whole crate comes back **clean** — so Track D itself
  passes the gate, and the gate as written needs one re-run after the other
  tracks land.
- **Windows / DirectML / CUDA**: not attempted. Per locked #10 these ship
  CPU-EP-qualified; the goldens are in `scripts/exports/rust_goldens/` and replay
  on the box unchanged.
- **Real Russian speech beyond the TTS fixture.** The E2E uses `say -v Milena`,
  which is clean studio-grade audio. iOS validated against FLEURS-ru (WER 4.38%);
  reproducing that here would need the corpus and is a qualification job, not a
  port gate.
- **`just engine-oracle` was not run end to end** (it demands
  `FERMENT_ENGINE_MODELS_DIR` + a full Core ML fleet). The new oracles live in
  the existing `--test ort_semantic` target, which that recipe already invokes,
  so no Justfile change was needed and none was made. No new fixtures go through
  `ferment-engine-test-fixtures`, so `CANARY_ENTRIES` is untouched.
