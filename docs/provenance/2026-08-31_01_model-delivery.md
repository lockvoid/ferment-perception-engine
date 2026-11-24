# 2026-08-31_01 — W3: mlmodelc folder+manifest delivery + the full CDN pack (FLEET)

**One executor. Four staged commits, each leaving the workspace green.**
End by writing `docs/fleet/2026-08-31_01_model-delivery_report.md`.

Cuts iOS retired the zip lane (commit `0c6b29c31`: "a model is a CDN prefix
plus a manifest, fetched straight into place with Range resume; the zip lane
and ZIPFoundation are gone"). This track adopts the same delivery for Studio's
CoreML artifacts and rebuilds `dist/models-publish/` as the COMPLETE upload
tree (ONNX + mlmodelc folders + manifests) for Dimitri's CDN upload.

## Authority

- iOS scheme (byte-format authority):
  `ferment-cuts-ios/Sources/FermentCuts/Services/ModelManager/{ModelFileManifest,ModelInstall,ModelTransferLane}.swift`
  and `Scripts/cdn/{make-manifests.py,upload-model,verify-upload-against-zips.py}`
  @ HEAD.
- Redshift model plane: `crates/engine-models/src/{lib.rs,fleet.rs}`,
  `crates/project/src/models.rs`, `scripts/models-publish`.
- The 17 pinned zips: `dist/models-publish-cache/*.mlmodelc.zip` (byte-mirrors
  of the July iOS CDN; their extracted-tree digests are the `sha256` values in
  `fleet.rs`).

## LOCKED rulings (lead, 2026-08-31)

1. **Bytes do not change; the wrapper does.** The staged mlmodelc folders are
   built by unzipping the 17 cached zips — NOT re-fetched from the iOS CDN,
   whose object names have since drifted (`*_pal8_*`); Studio's CoreML
   families are qualified against these exact bytes. The `fleet.rs`
   installed-tree `sha256` values therefore DO NOT CHANGE — assert that.
2. **Manifest byte-format is iOS's, exactly**: compact JSON
   `{"files":[{"path","bytes","sha256"},…]}` + trailing `\n`; sidecar
   `<digest>  <name>.manifest.json\n`. Entries sorted by the same walk
   make-manifests.py uses; `.DS_Store` excluded; symlinks are a hard error;
   empty dirs unrepresentable (error, don't skip).
3. **Studio pins the manifest digest in the fleet, and does NOT fetch the
   sidecar.** New source variant in `engine-models`:
   `ArtifactSource::ManifestPrefix { prefix_url, manifest_sha256 }` — the
   client verifies fetched manifest bytes against the compiled-in pin (a
   stronger trust anchor than iOS's short-cached sidecar; the sidecar still
   ships on the CDN for parity/tooling). The existing artifact-level `sha256`
   (installed tree) stays as the final proof — three checks total: manifest
   pin → per-file sha/size → tree hash at install.
4. **ONNX stays raw single-file at its current flat URLs** — 21 of 31 masters
   may already be live; do not churn their addressing. The NEW transfer code
   (stream-to-disk + Range resume) is shared by both lanes, killing today's
   whole-body RAM buffering (443 MB encoder in a `Vec<u8>`).
5. **Manifest-lane client constants are iOS's**: per-file integrity attempts 2
   (delete + refetch on sha/size mismatch), transport attempts 4 re-entered
   only while the file grew, resume token = destination size (`Range:
   bytes=N-`; a 200 truncates and restarts; a 416 deletes the partial and
   retries), request timeout 30 s. Manifest refetched fresh on every attempt.
6. **The zip lane dies in the same track**: `extract_zip_safely`,
   `archive_root`, the zip mirror fallback in `models-publish`, and the `zip`
   dependency from `crates/project/Cargo.toml`. `archive_sha256` remains for
   ONNX raw downloads (where it equals `sha256`) — do not redesign that field.
7. **No fleet index file.** The manifest is per-object; Studio's catalog is
   the compiled-in fleet, as iOS's is `ModelManifest.all`.
8. `MODEL_FLEET_VERSION` bumps ONCE, in stage 4 (new on-disk root, clean
   re-download for CoreML users).

## Boundary

MAY edit: `crates/engine-models/**`, `crates/project/src/models.rs` +
`crates/project/Cargo.toml` (drop `zip`), `scripts/models-publish`,
`dist/models-publish/**` (regenerate). MUST NOT edit: `crates/project`
elsewhere (W1's tree — if `runtime.rs` needs a line, STOP and report),
`crates/processorman`, engine crates other than `engine-models`, `apps/**`,
CI, `deny.toml`, root `Cargo.toml`.

## Stage 1 — publisher: the full CDN tree (script only, no crate code)

Rewrite `scripts/models-publish` to stage:

```
dist/models-publish/
  <file>.onnx / <file>.fp16.onnx           31 masters, flat (unchanged)
  <name>.mlmodelc/<path>                   17 folders, unzipped from cache
  <name>.mlmodelc.manifest.json            compact + trailing \n
  <name>.mlmodelc.manifest.json.sha256     sidecar
  gigaam_vocab.json, ambernet_langs.json   already staged; inert for Studio
                                           (Bundled), leave them
```

- Folder staging: unzip `dist/models-publish-cache/<name>.mlmodelc.zip`,
  verify every member against the manifest entries it just computed, and
  verify the staged tree's deterministic hash equals the `fleet.rs` `sha256`
  for that artifact (the ruling-1 assertion). Port
  `verify-upload-against-zips.py`'s gate: identical path sets, identical
  bytes.
- Emit a `manifest-pins.json` (name → manifest_sha256) as stage 2's input, so
  the 17 pins are generated, not hand-typed.
- `--check` gains the manifest lane: fetch `<base>/<name>.mlmodelc.manifest.json`,
  verify against the pin, then per-file fetch+hash (full check, as today).
- Print the upload commands for the human step, iOS's ordering rule verbatim:
  payload first (`aws s3 sync <dir>/ --delete --cache-control "public,
  max-age=31536000, immutable"`), manifest + sidecar LAST with
  `max-age=60` — a stale manifest at the edge is the one failure this layout
  can produce.

**Deliverable of this stage: the complete `dist/models-publish/` tree, staged
and verified — Dimitri uploads it.**

## Stage 2 — fleet: the `ManifestPrefix` source

- `ArtifactSource::ManifestPrefix { prefix_url, manifest_sha256 }` in
  `engine-models` (serde `kind: "manifest_prefix"`), with validation arms:
  requires `sha256` (tree pin), forbids `compressed_bytes`, requires
  provenance like every non-Missing source.
- `coreml_download` becomes `coreml_manifest(...)`: `prefix_url =
  {STUDIO_MODEL_BASE_URL}/{file_name}` (objects at `<prefix>/<path>`,
  manifest at `<prefix>.manifest.json`), `manifest_sha256` from stage 1's
  pins file. Keep `source_url` provenance pointing at the iOS zip lineage.
- Update the two inventory tests (`coreml_fleet_downloads_seventeen…` asserts
  the new shape; ONNX test unchanged) and `examples/manifest_json` output
  consumers (`models-publish` from stage 1 reads the new kind).
- Delete the dead `model_download_base_url()` re-export
  (`crates/project/src/models.rs:393` — returns the WRONG base, zero callers).

## Stage 3 — client: the manifest lane in `ModelDownloader`

- New fetch path for `ManifestPrefix`: fetch manifest → verify pinned digest →
  validate entries (port `ModelFileManifest.swift:39-85`'s hardening: safe
  relative paths, no `..`/absolute/NUL, lowercase 64-hex, no case-folded
  duplicate paths, no file/dir conflicts) → serial per-file fetch into the
  staging tempdir (stream to disk, Range resume, ruling-5 constants; per-file
  size + sha) → `ModelStore::install_staged_directory` (UNCHANGED — it
  already re-verifies the tree hash and renames atomically).
- Progress: per-file bytes weighted by `manifest.totalBytes` within the
  existing 0.85 download span; steps 0.88/0.94/1.0 preserved.
- ONNX raw path moves onto the same streamed transfer (resume included);
  behaviour otherwise identical.
- Zip lane deleted per ruling 6.
- Tests: port the shape of iOS's `ModelDeliveryTests` — a local HTTP stub
  (Range-aware) serving a folder + manifest; cases: happy install, manifest
  digest mismatch refuses before any payload byte, per-file sha mismatch
  deletes + retries once then fails, resume after a cut mid-file, 200-on-Range
  restarts, traversal/duplicate-path manifests refused, staged tree failing
  the fleet `sha256` refuses adoption (the store's existing test may cover
  this — extend, don't duplicate).

## Stage 4 — cleanup + version

Drop the `zip` dependency; `MODEL_FLEET_VERSION` → next value; full gates;
re-run stage 1's `--check` expectations against the local tree.

## Gates (every stage)

`cargo fmt --all -- --check`; `cargo clippy --all-targets -- -D warnings`;
CI's `cargo test --workspace` set; root `pnpm test` untouched-green.
Stage 1 additionally: the staged tree passes its own verify gate, and
`du`/count summary in the report (expect ~6.9 GB, 31 ONNX + 17 folders + 17
manifests + 17 sidecars + 2 JSONs).

## Report

Stage table; the 17 manifest pins with the tree-hash assertion result per
model; transfer-lane test inventory; deviations; what could not be run (the
live CDN `--check` runs only after Dimitri uploads).
