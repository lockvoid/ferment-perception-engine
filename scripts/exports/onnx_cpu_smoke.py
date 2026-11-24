"""Load an ONNX model on the CPU EP and run it once on random inputs sized to
the graph's own signature. Empirically proves the whole graph executes on the
portable CPU reference (including GridSample, which static shape inference can
leave rank-unresolved under dynamic H/W).

Usage: python onnx_cpu_smoke.py --onnx model.onnx --hw 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--hw", type=int, default=64, help="concrete size for dynamic H/W")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feeds = {}
    described = []
    for spec in session.get_inputs():
        dims = [args.hw if (isinstance(d, str) or d is None or d < 0) else d for d in spec.shape]
        # tiny scalar-ish timestep inputs often come through as [] or [1]
        dims = [d if d else 1 for d in dims] or [1]
        dtype = NP_DTYPE.get(spec.type, np.float32)
        array = np.random.default_rng(0).random(dims).astype(dtype)
        feeds[spec.name] = array
        described.append({"name": spec.name, "fed_shape": dims, "type": spec.type})

    outputs = session.run(None, feeds)
    result = {
        "onnx": str(args.onnx),
        "session_providers": session.get_providers(),
        "inputs": described,
        "outputs": [
            {"name": o.name, "shape": list(np.asarray(v).shape)}
            for o, v in zip(session.get_outputs(), outputs)
        ],
        "ran": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
