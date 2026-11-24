# Third-party notices — `ferment-engine-processors`

Adapted third-party *source*. Model weights carry their own licence and source
URL in `crates/engine-models/src/fleet.rs`, which `validate_complete_fleet`
enforces; Rust dependencies are governed by `cargo deny check licenses`. This
file is for code whose shape came from somewhere else.

## FluidAudio

The speaker-diarization scaffolding around `pyannote/segmentation-3.0` —
sliding-window geometry, the 7-class powerset decode, agglomerative clustering
of chunk-local speakers, and cross-chunk reconstruction — is adapted from
FluidAudio:

- Source: <https://github.com/FluidInference/FluidAudio>
- Adapted files: `Sources/FluidAudio/Diarizer/Segmentation/*`,
  `Sources/FluidAudio/Diarizer/Offline/*`

FluidAudio is licensed under the Apache License, Version 2.0. A verbatim copy
of the upstream license is included as [`LICENSE-FluidAudio.txt`](LICENSE-FluidAudio.txt).

Ferment's port is a rewrite against this crate's own conventions, not a copy:
the chunk step is expressed in whole model frames so overlapping chunks average
by index with no interpolation, clustering is plain average-linkage cosine
rather than a `fastcluster` FFI, and the embedder is ReDimNet2 rather than
WeSpeaker. FluidAudio's PLDA, VBx, Sortformer, LS-EEND and streaming paths are
not used. Every module that carries adapted logic names it in its own header.
It reaches this repository through the iOS port at
`ferment-cuts-ios/Packages/FermentEngine/Sources/FermentEngine/Processors/Pyannote/`,
whose own notice this mirrors.

### `token_seam_merge`

The three-tier chunk-seam reconciliation both chunked speech ports use
(contiguous run → LCS → midpoint cut) is also FluidAudio's, from
`ChunkProcessor.mergeChunks` — the same Apache-2.0 grant and license copy cover
it. Ferment's version is generic over the row payload so Parakeet's TDT
`duration` lane and GigaAM's confidence lane ride through one body
(`src/token_seam_merge/`).
