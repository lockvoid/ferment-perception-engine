"""Fetch and qualify the two substitute backings for the video-saliency port.

Apple Vision has no off-Apple equivalent, so the heat map and the face boxes
come from third-party ONNX masters. Neither needs a torch export -- both ship
as released ONNX -- so this script pins them by sha256, does the one graph edit
U^2-Netp needs, and gates both through the migration-016 portability inspector.

  u2netp  Apache-2.0, xuebinqin/U-2-Net, released as ONNX by rembg (MIT tooling)
          [1,3,320,320] f32 -> seven side outputs, sigmoid applied in-graph.
          We keep only d0 and give the graph readable I/O names; the other six
          heads are training scaffolding and cost inference time for nothing.

  yunet   MIT, Shiqi Yu, OpenCV Zoo `face_detection_yunet_2026may` (the 2023mar weights, symbolic H/W)
          [1,3,640,640] f32 BGR 0..255 -> cls/obj/bbox/kps at strides 8/16/32.
          Used verbatim; the decode lives in Rust, transcribed from OpenCV.

Usage: python export.py [--output-dir artifacts]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent

# (filename, url, sha256 of the upstream download)
SOURCES = {
    "u2netp": (
        "u2netp.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    ),
    "yunet": (
        "face_detection_yunet_2026may.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
        "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    ),
}

# DirectML tops out at opset 20 and the whole fleet must stay under it.
MAX_OPSET = 20
PORTABLE_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}
# Never portable off desktop CUDA (migration 016 "golden rule").
CUDA_ONLY_OPS = {"DeformConv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, output_dir: Path) -> Path:
    filename, url, expected = SOURCES[name]
    target = output_dir / filename
    if not target.exists():
        print(f"fetching {url}")
        with urllib.request.urlopen(url) as response, target.open("wb") as handle:
            handle.write(response.read())
    actual = sha256(target)
    if actual != expected:
        raise SystemExit(f"{filename}: sha256 {actual} != pinned {expected}")
    print(f"{filename}: sha256 ok ({target.stat().st_size} bytes)")
    return target


def trim_u2netp(source: Path, target: Path) -> None:
    """Keep the d0 head, name the graph's I/O, drop the six training heads."""
    model = onnx.load(source)
    graph = model.graph
    fused = graph.output[0].name
    original_input = graph.input[0].name

    del graph.output[1:]
    graph.output[0].name = "saliency"
    graph.input[0].name = "image"
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == original_input:
                node.input[index] = "image"
        for index, value in enumerate(node.output):
            if value == fused:
                node.output[index] = "saliency"

    # onnx's extractor prunes everything the surviving output does not need.
    onnx.save(model, target)
    extracted = onnx.utils.Extractor(onnx.load(target)).extract_model(
        ["image"], ["saliency"]
    )
    onnx.checker.check_model(extracted)
    onnx.save(extracted, target)
    print(f"{target.name}: {len(extracted.graph.node)} nodes (was {len(graph.node)})")


def portability_report(path: Path) -> dict:
    model = onnx.load(path)
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    histogram = Counter(node.op_type for node in model.graph.node)
    disqualifiers = sorted(
        {node.op_type for node in model.graph.node if node.op_type in CUDA_ONLY_OPS}
        | {
            f"{node.domain}:{node.op_type}"
            for node in model.graph.node
            if node.domain and node.domain not in PORTABLE_DOMAINS
        }
    )
    over_ceiling = [
        f"{domain}:{version}"
        for domain, version in opsets.items()
        if domain in PORTABLE_DOMAINS and version > MAX_OPSET
    ]
    return {
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "opsets": opsets,
        "inputs": [
            {
                "name": value.name,
                "shape": [
                    d.dim_param or d.dim_value
                    for d in value.type.tensor_type.shape.dim
                ],
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "shape": [
                    d.dim_param or d.dim_value
                    for d in value.type.tensor_type.shape.dim
                ],
            }
            for value in model.graph.output
        ],
        "op_histogram": dict(sorted(histogram.items())),
        "disqualifiers": disqualifiers + over_ceiling,
        "portable": not disqualifiers and not over_ceiling,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_u2netp = fetch("u2netp", args.output_dir)
    yunet = fetch("yunet", args.output_dir)
    trimmed = args.output_dir / "u2netp.saliency.onnx"
    trim_u2netp(raw_u2netp, trimmed)

    manifest = {
        "sources": {
            name: {"url": url, "sha256": digest}
            for name, (_, url, digest) in SOURCES.items()
        },
        "masters": [portability_report(path) for path in (trimmed, yunet)],
    }
    for master in manifest["masters"]:
        if not master["portable"]:
            raise SystemExit(f"{master['file']}: {master['disqualifiers']}")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["masters"], indent=2))


if __name__ == "__main__":
    main()
