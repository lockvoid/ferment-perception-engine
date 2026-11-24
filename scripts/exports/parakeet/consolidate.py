"""Post-export packaging: every parakeet master becomes ONE self-contained
.onnx under scripts/exports/parakeet/artifacts/, the shape the studio
manifest's raw-download path ships.

- preprocessor / decoder / joint: fold any external data in, stay fp32.
- encoder: ~2.3 GB of fp32 weights exceeds the 2 GB protobuf cap, so the
  weights convert to fp16 (graph I/O stays fp32 via keep_io_types, matching
  the runtime contract) — the same treatment the 016 enhance masters ship
  with. Parity vs the fp32 origin goldens is gated in
  crates/engine-processors/tests/ort_semantic.rs.

    ./scripts/exports/parakeet/.venv/bin/python scripts/exports/parakeet/consolidate.py
"""

from __future__ import annotations

import pathlib

import onnx
# ORT's fork of the fp16 converter tracks value types node-by-node, so it
# stays correct on >2 GB models where full protobuf shape inference dies
# (onnxconverter-common left mixed fp16/fp32 inputs on pre_encode Mul nodes).
from onnxruntime.transformers.float16 import convert_float_to_float16

HERE = pathlib.Path(__file__).resolve().parent
REPO_DIR = HERE / "repo"
ARTIFACTS = HERE / "artifacts"


def consolidate(name: str, fp16: bool = False) -> None:
    source = REPO_DIR / name / "1" / "model.onnx"
    target = ARTIFACTS / f"{name}.onnx"
    model = onnx.load(source.as_posix())  # resolves sibling external-data files
    if fp16:
        # disable_shape_infer: onnx shape inference silently empties protos
        # over the 2 GB serialization cap, which is why fp16 exists here at all.
        model = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    onnx.save(model, target.as_posix())
    print(f"{target.name}: {target.stat().st_size:,} bytes{' (fp16 weights)' if fp16 else ''}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    consolidate("parakeet_preprocessor")
    consolidate("parakeet_encoder", fp16=True)
    consolidate("parakeet_decoder")
    consolidate("parakeet_joint")


if __name__ == "__main__":
    main()
