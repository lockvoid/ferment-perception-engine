# 2026-08-31 — Engine cross-port: integration pass (LEAD — not for dispatch)

**Owner: the lead.** Runs after the fleet tracks in
`2026-08-31_engine-cross-port.md` land and pass review. The fleet delivers
processors + protocols in the engine crates; this pass wires them into
`crates/project` (and, where needed, the desktop/web layers). Blocked per-item on
the corresponding fleet track.

## Authority

Server contract: `ferment-api/packs/project/app/lib/networks.rb` —
"Operator keys must be byte-identical to the iOS networks, or cooks
(`pmck/Element/<id>/<key>`, version-free) won't resolve across runtimes."
Server public vocabulary (verified 2026-08-31): video `original poster videoScenes
videoSaliency videoMinimap audioMinimap audioEvents speakerEmbeddings languageId
transcript`; audio `original audioMinimap beats musicHighlights musicTags
audioEvents speakerEmbeddings languageId transcript`; image `original poster`;
graphic `original`. Device-local operators (variants, waveforms, denoised lanes,
model nodes, `audioSemanticEmbeddings`, `colorAnalysis`) don't appear in the server
vocabulary — they are file-only and never sync; mirror them from the iOS network
files, not from `networks.rb`.
iOS graph shape: `Sources/FermentCuts/Features/Project/Networks/*.swift @ 3d1c2c180`.

## 1. Network graph sync (`crates/project/src/networks.rs`) — after fleet A-review

- **Video:** `aiVideo` + `proxyVideo` → one **`videoVariants`**
  (`EncodeVideoProcessor`, thumbstrip on); artifacts `<cookKey>.proxy|.ai|.thumbstrip`
  (the one bare-cookKey exception — keep it). Add **`videoSaliency`**
  (`videoVariants as video` w/ `sourceArtifact "ai"`, `videoScenes as scenes`).
  Add **`languageId` + `languageIdModel`** (`original as audio`, `originalBlob`,
  `audioEvents as audioEvents`, model). Add **`speakerEmbeddings` +
  `speakerEncodingModel` + `speakerSegmentationModel`** (`original as audio`,
  `originalBlob`, both models — deliberately NOT `audioEvents`). `videoScenes` reads
  the perception feed: `videoVariants as video` (`sourceArtifact "ai"`) + model;
  drops `original`/`originalBlob`. `transcript` gains `languageId as languageId`;
  `speechTranscribingModel` becomes the routed downloader (reads the languageId cook
  via the fleet's routing module). `videoMinimap` still wires `original as video` —
  mirror, don't improve.
- **Audio:** remove **`aiAudio`** (+ decide `audio_encoder.rs` fate: stays if
  encode_pipeline needs it internally, leaves the element graph regardless). Add
  `languageId`(+Model) and `speakerEmbeddings`(+2 models). `audioMinimap`:
  **`join: anySucceeded`** + bare **`.beats`** wire. `transcript` gains the
  `languageId` wire.
- **Image:** add `poster`; mirror `ImageElementNetwork.swift` exactly (check image
  `colorAnalysis` fate there).
- New wire input name `languageId` (existing: `source video audio scenes
  audioEvents audioEmbeddings`).
- Pin test: full per-kind operator-key lists asserted; server's 13 public keys ⊆ ours.

## 2. Kinds + processor versions (`crates/project/src/catalog.rs` + cook writers)

- Beats cook `data.kind` **`"beatGrid"` → `"beats"`**; new kinds `"languageId"`,
  `"speakerEmbeddings"`, `"videoSaliency"` — each verified against the iOS
  processor's `loadCookData`/`make(from:)` at HEAD.
- Mirror iOS processor versions: AnalyzeBeats **7**, TranscribeSpeech **4**,
  DetectVideoScenes **3**, EncodeVideo **4**, ComposeAudioMinimap **3**,
  DetectVideoSaliency **3**, IdentifyLanguage **1**, EmbedSpeakers **2**
  (re-verify each at wiring time; versions ride `Cook.version`, they re-key nothing).
- New `ProcessorKind` variants + registry entries: `IdentifyLanguageProcessor`,
  `EmbedSpeakersProcessor`, `DetectVideoSaliencyProcessor`; retire
  `EncodeAudioProcessor` from the graph.
- Beats payload gains `segments` + `energyPerBar` + `energyStructure` (fleet B's
  result types); scenes payload gains `embeddingDims` + flat `embeddings`
  (all-or-nothing); transcript coverage gate: D located it — delete the
  `speech_coverage(...) < gate_threshold` branch at
  `crates/project/src/processors.rs:471` (then `speech_coverage()` and
  `CoverageGateOptions` go dead); the four `musical_coverage` gates at
  519/565/598/627 STAY.

## 3. Versioned cook blobs (`crates/project` + storage read path) — independent

Upstream `c23c77d1e` (`ProcessorMan/Models/Operator.swift`,
`BlobManager{,+Writes}.swift`):
- `cookKey` = `pmck/Element/<id>/<key>` (row ref, version-free, unchanged);
  **`blobKey` = `<cookKey>@<processorVersion>`** (the on-disk file).
- Writers write the blobKey; a version bump misses last version's file by
  construction. On write, sweep retired siblings (bare `<key>` and any `<key>@M`).
- Readers resolve **newest `@N` by integer N, else bare** (bare stays valid for
  pre-versioning files and server-downloaded artifacts).
- Exception: `EncodeVideoProcessor` stays bare-cookKey + suffix.
- Home: `BlobKey::from_cook_ref` in `crates/project` + the storage reader. Port the
  iOS `VersionedCookBlobTests` scenarios.

## 4. Engine swap-ins (per landed fleet track)

- **B:** `engine.rs` drops `HybridBeatDetector` (construct the DBN+lattice path);
  delete `hybrid_beat_detector/` + its `BeatAnalyzing` impl + the unused raw
  `BeatThis`/`BeatQueenMary` impls if the new path obsoletes them.
- **C:** scene detector consumes the `ai` variant (geometry now 640-wide);
  cook mapping for embeddings.
- **D:** wire `IdentifyLanguage` + routed transcriber downloads; remove the
  project-side coverage gate (speech skip moves to languageId + `all_succeeded`
  pruning; **music lanes keep `musical.coverage 0.5`**).
- **E:** `EmbedSpeakers` cook mapping from the diarization result type. **No
  project-level speaker fold** — that arrives with the replica round.
- **F:** `DetectVideoSaliency` wiring + cook mapping. Constants SIGNED OFF by
  the lead (2026-08-31): `U2NETP_HEAT_FLOOR` stays **0.3** (measured inert —
  0.2468-equivalent vs 0.3 moves centroids < 0.005 on every real-subject frame;
  one number across platforms wins) and YuNet `SCORE_THRESHOLD` = **0.75**
  (0.9 measurably drops real faces; midpoint of the measured 0.60–0.90 gap).
  WIRING HAZARD: `confidence` is INVERTED vs iOS on the no-subject case
  (Vision ~0.97, U²-Netp ~0.03) — the cook payload's `mode` must be written
  from `result.mode` (`"objectness"`), and no iOS-side confidence threshold may
  be carried across. The field's doc comment carries the measured ranges.
- **G:** `videoVariants` operator + artifact suffix contract; thumbstrip artifact
  exposure to the library/preview layer.
- **B (ruler labels), decided:** no bundled font. The renderer has never drawn
  text (mel/dB labels are absent for the same reason); the tick set
  `{bar, x, labeled}` rides `LayoutGeometry::bar_ruler` — the JSON sidecar iOS
  itself says the numbers belong in — and the app layer draws labels in the
  webview, where text is crisp, styleable and i18n-able. Wiring: extend
  `core::AudioMinimapComposing::compose` with downbeats +
  `core::MinimapLayout` with the ticks (one change, per the B report).
- **C (identity margin), follow-up:** resampler drift is 0.0368 Euclidean on a
  0.15-wide decision band (gates 0.72/0.80). Strobe regression passes and the
  veto abstains safely when the model is absent, so shipping as-is; before the
  veto is trusted library-wide, re-measure the same/similar distances on
  Studio-labelled clips (constants: `EncoderShotIdentity::{SAME,SIMILAR}_SHOT_DISTANCE`).

## 5. Order

1. §3 (blob versioning) — independent, can land first.
2. §1 + §2 in one pass once fleet B/D/E/F/G are reviewed (the graph change is
   atomic: keys, wires, kinds, versions together).
3. §4 swap-ins ride the same pass per track.
4. After: joint QA against a live server project (staging), then update
   `migration/README.md` index + board notes.
