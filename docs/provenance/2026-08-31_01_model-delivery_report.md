# W3 executor report — mlmodelc folder+manifest delivery + the full CDN pack

**Executed 2026-08-31 against `main @ 1b93f5f1`.** Dispatch doc:
`2026-08-31_01_model-delivery.md`. Four staged commits, each green on the full
gate set:

```
1ecd310c  stage 1  the publisher stages a prefix, not an archive
7bba320f  stage 2  the fleet points at a prefix and pins the manifest
f50c1017  stage 3  one streamed transfer, two lanes, no archive
77fd5833  stage 4  the zip dependency goes, the fleet root moves
```

All four stages are done. The zip lane is gone from the publisher, the fleet,
the client and the dependency graph. `dist/models-publish/` is the complete
upload tree, staged and verified — **and verified again over real HTTP**, which
turned out to be the one thing the dispatch doc did not ask for and the one
thing that made the pre-upload evidence worth anything (§ "The gate that was
not asked for").

---

## Stage table

| Stage | What landed | Gates |
|---|---|---|
| 1 | `scripts/models-publish` rewritten: 31 flat ONNX + 17 mlmodelc folders + 17 manifests + 17 sidecars, `manifest-pins.json` emitted, upload plan printed | fmt/clippy/tests unchanged-green; staged tree passes its own verify gate |
| 2 | `ArtifactSource::ManifestPrefix`, `ModelFileManifest` + its refusals, `coreml_manifest()` with the 17 generated pins, `model_download_base_url()` deleted | fmt, clippy, `cargo test --workspace` (129 suites) |
| 3 | `ModelDownloader`'s manifest lane, one streamed+resumable transfer for both lanes, zip extractor deleted, Range-aware stub tests | same, plus 15 client tests |
| 4 | `zip` dropped from `ferment-project`, `MODEL_FLEET_VERSION` → `2026-08-31.2` | same, plus `cargo deny`, `pnpm test:unit` (736), local `--check` |

---

## Ruling 1: the bytes did not change

The assertion the doc asked to be made loudest, made three ways.

**In the publisher**, per folder, every run: the staged tree's deterministic
hash must equal the `sha256` the fleet pins, or the script refuses to write the
manifest. All 17 pass.

**In the fleet's tests**, `coreml_tree_digests_are_unchanged_by_the_manifest_lane`
pins four of them literally, so a future round that "refreshes" a model off the
drifted iOS CDN fails a test rather than someone's install.

**Against git**: the set of 17 Core ML tree digests at `1b93f5f1` and the set at
`77fd5833` are *identical as sets* — not merely the same count.

| model | files | bytes | tree sha256 (unchanged) | manifest pin |
|---|---|---|---|---|
| `autoshot_fp16.mlmodelc` | 5 | 16,688,431 | `f712c395e798843e…` | `4e514aceee193f56…` |
| `autoshot_fp32.mlmodelc` | 5 | 33,240,609 | `ca89cfb652f5ed4e…` | `96fdc04f840af4cf…` |
| `beat_this_fp16_1.mlmodelc` | 5 | 41,950,448 | `2d0dc63cdf653b96…` | `5305e01516ea96cf…` |
| `beat_this_fp16_2.mlmodelc` | 5 | 41,950,448 | `3c8b36a964e02744…` | `c886f4a4733b55f4…` |
| `beat_this_fp16_3.mlmodelc` | 5 | 41,950,444 | `57c31126a34422f5…` | `3e2f7f3c0a5b6eb1…` |
| `beat_this_fp32_1.mlmodelc` | 5 | 82,531,365 | `a7545871ad602006…` | `fce9f024af758465…` |
| `beat_this_fp32_2.mlmodelc` | 5 | 82,531,365 | `e8be2c7596980525…` | `5eb61ba3a011508e…` |
| `beat_this_fp32_3.mlmodelc` | 5 | 82,531,360 | `80e7648c91ca3abb…` | `c529215acdc21a80…` |
| `dasheng_base.mlmodelc` | 5 | 171,589,126 | `38bd31b58c6937c2…` | `f9381c6d87c7daf0…` |
| `dfnet.mlmodelc` | 5 | 10,196,112 | `78cd8b5e5fa71398…` | `cdf6d203390a0934…` |
| `fireredvad_aed.mlmodelc` | 5 | 4,200,859 | `bfe853b8d2a16d9b…` | `d0f6818a7c123796…` |
| `music_highlighter.mlmodelc` | 5 | 1,583,427 | `e508643366377508…` | `b0f3272cf23555d3…` |
| `music_tagger_heads.mlmodelc` | 5 | 2,483,854 | `79caec991f88832b…` | `58a2c02516222040…` |
| `parakeet_decoder.mlmodelc` | 5 | 23,622,326 | `cf13c5338e5de76e…` | `82ea03e58ea30797…` |
| `parakeet_encoder.mlmodelc` | 5 | 446,150,618 | `283298cf3efd50de…` | `2e896efef332f425…` |
| `parakeet_joint.mlmodelc` | 5 | 12,658,756 | `0a45a181800211c2…` | `d252c4142c729d53…` |
| `parakeet_preprocessor.mlmodelc` | 5 | 522,823 | `038ee0bfce85bc43…` | `0734e4e6d8e8c5e6…` |

Full pins: `dist/models-publish-cache/manifest-pins.json` (outside the upload
tree — ruling 7 says the CDN carries no index, and an index file sitting next to
the payload is an invitation for a client to read one).

Every model is 5 objects: `coremldata.bin`, `metadata.json`, `model.mil`,
`analytics/coremldata.bin`, `weights/weight.bin`. No symlinks, no empty
directories, no `.DS_Store` in any of the 17 archives — the three cases the
staging script treats as hard errors never fired, which is the answer worth
recording.

### The one place the two orderings differ, and why both are copied

The manifest's entry order and the tree hash's file order are **not the same
sort** and both are ported rather than invented:

- Manifest entries: sorted by the relative path string, as
  `make-manifests.py`'s `files.sort()` produces.
- Tree hash: sorted by path *components*, because that is `PathBuf: Ord` in
  Rust, and `sha256_path` is what the fleet's digests were taken with. A plain
  byte sort of the joined string disagrees whenever a file and a directory share
  a prefix (`a.bin` vs `a/b`: `.` sorts before `/`, but the component sort puts
  `a` before `a.bin`).

For these 17 trees the two orders happen to coincide, so the distinction cost
nothing this round — but the publisher implements both, and the 17 matching tree
hashes are the proof the component sort was reimplemented correctly in Python.

---

## The gate that was not asked for

The doc's stage-1 gate is "the staged tree passes its own verify gate", and the
report line for `--check` is "runs only after Dimitri uploads". Staging-side
verification proves the bytes on disk; it does not prove that what a *client*
does with them works — the client fetches `<prefix>.manifest.json`, checks it
against a compiled-in pin, then fetches `<prefix>/<path>` for every entry, and
none of those URL joins are exercised by hashing files locally.

So `--check` gained `--base URL`. Pointed at a `python3 -m http.server` over
`dist/models-publish/`, it runs the real client-side sequence against the real
staged bytes:

```
$ ./scripts/models-publish --check --base http://127.0.0.1:8731
…
48 verified, 0 failed, 0 skipped
```

31 ONNX by digest; 17 manifests against the pins the fleet now compiles in; 17
sidecars agreeing with the manifests beside them; and all 85 payload objects by
size and digest. That is everything the CDN check will do, minus the CDN. The
same flag is what Dimitri can point at a staging bucket before promoting it.

---

## Deviations from the doc

**1. The tree is 3.4 GB, not ~6.9 GB.** The doc's gate expects a `du` around
6.9 GB. Measured: 2.34 GB of ONNX + 1.02 GB of Core ML folders + 25 KB of
manifests/sidecars/JSON = **3.36 GB, 152 files** (31 + 2 + 85 + 17 + 17). The
count breakdown the doc gives is exactly right; the size estimate looks like it
double-counted the folders against the 977 MB of zips they replace. Nothing is
missing — the `--check` run above accounts for all 48 objects.

**2. `verify_download` moved to stage 3, not stage 2.** The doc puts the fleet
work in stage 2. Changing the verifier from `&[u8]` to a path is part of taking
the body out of RAM, which is stage 3's job; doing it in stage 2 would have
meant a client that buffers a file to disk to hand it back to a byte verifier.
Stage 2 is the fleet and the format; stage 3 is the transport.

**3. `crates/project/src/lib.rs` — one line, outside the boundary.** The doc
orders `model_download_base_url()` deleted and puts `crates/project` (other than
`models.rs` and `Cargo.toml`) out of bounds. The function is re-exported from
`lib.rs:27`; deleting it without removing that identifier does not compile. I
removed the identifier and nothing else in that file. Flagging it because the
boundary said to.

**4. Between stage 2 and stage 3, Core ML downloads do not work.** The fleet
points at manifest prefixes one commit before the client can fetch one; the arm
returns an explicit error in between. That is inherent in the doc's stage split
and lives for exactly one commit.

**5. No single-file-manifest special case.** iOS's `installPayload` treats a
one-entry manifest whose path equals the model's own name as a file rather than
a directory. Studio has no such artifact — the two JSON resources are `Bundled`
and never fetched — and every manifest-prefix artifact's fleet `sha256` is a
*tree* hash, so a file-shaped payload could not validate without the fleet
declaring it differently anyway. Adding the branch would have meant shipping an
untestable path. Noted here so whoever adds the first single-file manifest knows
it is the first.

**6. Staging is a tempdir, not durable.** Per the doc ("the staging tempdir").
Resume therefore covers a cut *within* an install, not across a killed process —
iOS's `Models.incoming` survives an app launch, ours does not. The trade is
recorded in a comment at the site: a killed Studio download costs the artifact,
not just its tail. The corollary is that iOS's prune step has nothing to do here
and is deliberately absent.

**7. Two additions to the manifest's refusals, both from Windows.** Backslash is
rejected in a manifest path (iOS does not need to: on Windows it is a second
separator, and `a\..\..\b` walks straight through a check that only knows `/`),
and the case-folded duplicate check's justification is NTFS as well as APFS.
`invalidSize` has no arm at all: sizes decode as `u64`, so a negative one is
unrepresentable rather than rejected — iOS needs the arm because Swift decodes
into `Int64`.

**8. `--force` and `--base` are new flags.** `--force` restages folders that
already verify (a full re-extract of all 17 is 23 s; skipping them makes a
republish of one changed model cheap). `--base` is § "The gate that was not
asked for".

---

## Transfer lane — test inventory

`crates/project/src/models.rs`, all against a Range-aware HTTP stub that records
every request path and its `Range` header, can cut a *named* object's body
mid-message, and can answer a ranged request with a 200:

| Test | What it pins |
|---|---|
| `manifest_prefix_installs_a_verified_tree` | manifest first, then exactly one request per entry; installs as a tree; `manifest.json` stays in staging |
| `a_manifest_that_fails_its_pin_refuses_before_any_payload_byte` | one request total, nothing installed — the pin's whole purpose |
| `a_traversing_manifest_is_refused_and_writes_nothing_outside` | a manifest that *hashes to its pin* is still refused; nothing written outside |
| `an_object_that_never_matches_its_digest_is_refetched_then_refused` | exactly `INTEGRITY_ATTEMPTS` hits, then refusal (tampered bytes are the same length, so it is the digest that refuses, not the size) |
| `an_object_shorter_than_its_manifest_is_refused` | the size check catches a short object before a digest is taken |
| `a_cut_mid_object_resumes_from_the_bytes_on_disk` | exactly one resumed request, `bytes=4-`, and the tree still hashes right |
| `a_cut_manifest_is_taken_again_from_the_start` | the manifest is retried and **never** resumed |
| `a_server_that_ignores_the_range_still_produces_the_right_file` | 200-on-Range truncates and restarts |
| `a_staged_tree_that_fails_the_fleet_digest_is_not_adopted` | every file proved and the set still refused |
| `progress_never_goes_backwards_and_reaches_one` | monotonic across a resume, ends at 1.0 |
| `raw_download_installs_a_single_verified_file` / `_resumes_after_a_cut` / `_rejects_digest_mismatch` | the ONNX lane on the same transfer, resume included |
| `live_cdn_raw_download_smoke` / `live_cdn_manifest_download_smoke` | `#[ignore]` — the post-upload proof |

`crates/engine-models/src/lib.rs` carries the format's own refusals:
`file_manifest_refuses_what_it_cannot_safely_write` (traversal, absolute, NUL,
backslash, empty, empty-segment, case-folded duplicate, file/dir conflict,
uppercase digest), `file_manifest_parses_the_ios_byte_format` (the literal CDN
bytes), and the two `manifest_prefix_*` validation tests.

The one test that had to be *rewritten* rather than ported found a real gap in
my first cut: the mid-object cut was hitting the manifest fetch instead, because
the stub's cut was "the next body" rather than "this object's body". Fixing the
stub is what produced `a_cut_manifest_is_taken_again_from_the_start`, which is
now the only thing proving the manifest is refetched fresh rather than resumed.

---

## Notes worth the lead's time

**`.timeout(30s)` on the blocking client is an inactivity timeout, not a
ceiling.** reqwest's blocking `Read` impl re-arms the deadline on every `read`
call, so ruling 5's "request timeout 30 s" behaves like iOS's
`timeoutIntervalForRequest` and does *not* cap a 443 MB body at 30 seconds. The
same constant passed to `Response::bytes()` would mean the opposite. That is
recorded at the constant, because it is the kind of thing that gets "fixed" into
a bug.

**The zip removal took a duplicate crate version with it.** `ferment-project`
was the workspace's only user of `zip` 4.x; `zip` 6.0.0 remains via `ndarray`'s
npy support, so the lockfile now has one `zip` instead of two. `cargo deny check
bans licenses sources` is green.

**Progress dips on a 200-answers-a-Range restart.** The bar reports bytes on
disk, and a restart truncates. iOS has the identical property (`startOffset` is
reset to 0 and `onBytes` follows it), so this is parity, not an oversight — and
the alternative, clamping progress to its high-water mark, would hide a restart
that actually happened.

---

## What could not be run

**The live CDN `--check`, and the two `#[ignore]`d smoke tests.** They need the
upload. The exact sequence, printed by the script itself:

```
scripts/models-publish                       # stage + verify (already done; idempotent)
# 1. the 17 Core ML prefixes — aws s3 sync <dir>/ --delete, immutable
# 2. the flat masters — *.onnx + the two JSON inputs, immutable, top level only
# 3. the manifests and their sidecars — LAST, max-age=60
```

Ordering is iOS's rule verbatim and it is not cosmetic: a client that reads a
new manifest must be able to fetch every object it names, so payload goes first
and immutable, manifests go last and short-cached. Step 2 excludes `*/*` so the
`metadata.json` inside each mlmodelc prefix is not re-uploaded by the flat-file
sync — the one filter interaction in the plan that is easy to get wrong.

After the upload:

```
scripts/models-publish --check                                    # all 48 objects, live
cargo test -p ferment-project -- --ignored live_cdn               # both smokes
```

**A macOS Core ML end-to-end install against the live CDN** — same blocker.
Everything up to the network is covered by the stub tests.

---

## Gates

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo test --workspace` (CI's exclude set) | 129 suites, 0 failures |
| `cargo deny check bans licenses sources` | ok, ok, ok |
| `pnpm test:unit` | 99 files, 736 tests, untouched-green |
| `scripts/models-publish` | 31 + 17 staged, 17/17 tree hashes match the fleet |
| `scripts/models-publish --check --base <local>` | 48 verified, 0 failed |
