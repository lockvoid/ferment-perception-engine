"""Origin goldens for dinov2_vits14: the upstream torch model (the same
`DinoIdentity` module the export traced) on a committed deterministic frame.

Consumed by `crates/engine-processors/tests/ort_semantic.rs`.

The input is the 224x126 frame `vimage_reference.swift` produced, so the golden
pins the GRAPH and nothing else — resampler agreement is a separate, measured
gate (`crates/engine-processors/src/dinov2/mod.rs::resampler_tests`).

    uv run --project scripts/exports/dinov2 python scripts/exports/dinov2/gen_goldens.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from export_dinov2 import H, W, load_identity

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLDENS = REPO / "scripts/exports/rust_goldens"


def main() -> None:
    frame = np.frombuffer(
        (HERE / "goldens/resample_vimage_224x126.rgb").read_bytes(), dtype=np.uint8
    ).reshape(H, W, 3)
    # [1, 3, H, W] in 0..255 — the surface the Rust encoder hands over.
    image = frame.transpose(2, 0, 1)[None, ...].astype(np.float32)

    model = load_identity()
    with torch.no_grad():
        embedding = model(torch.from_numpy(image)).numpy().astype(np.float32)

    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / "dinov2_image_in.bin").write_bytes(image.tobytes())
    (GOLDENS / "dinov2_embedding_out.bin").write_bytes(embedding.tobytes())
    print(
        f"dinov2 goldens: image {image.shape} -> embedding {embedding.shape} "
        f"(norm {float(np.linalg.norm(embedding)):.6f})"
    )


if __name__ == "__main__":
    main()
