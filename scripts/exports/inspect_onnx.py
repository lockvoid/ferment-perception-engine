"""Static ONNX op-coverage inspector — the golden-rule gate for migration 016.

Answers, without a GPU: does this graph port to DirectML / WebGPU / QNN, or is it
CUDA-desktop only? Reports opset, an op histogram, I/O, GridSample dimensionality
(4-D sampling ports; 5-D does not on DirectML), and any disqualifying op
(DeformConv, custom domains, or ops beyond the DirectML opset-20 ceiling).

Usage: python inspect_onnx.py --onnx model.onnx
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx
from onnx import shape_inference

# DirectML supports up to ONNX opset 20, EXCEPT GridSample-20 5-D and DeformConv
# (migration 016 "golden rule"). These op types are hard-disqualifying anywhere
# but desktop CUDA/TensorRT.
CUDA_ONLY_OPS = {"DeformConv"}
# Custom/contrib domains never port to the portable EPs.
PORTABLE_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}


def rank_map(model: onnx.ModelProto) -> dict[str, int]:
    ranks: dict[str, int] = {}
    graph = model.graph
    for collection in (graph.input, graph.output, graph.value_info):
        for value in collection:
            ttype = value.type.tensor_type
            if ttype.HasField("shape"):
                ranks[value.name] = len(ttype.shape.dim)
    for init in graph.initializer:
        ranks[init.name] = len(init.dims)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as error:  # shape inference is best-effort
        print(f"(shape inference partial: {error})")
    ranks = rank_map(model)

    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    histogram = Counter(node.op_type for node in model.graph.node)

    grid_samples = []
    disqualifiers = []
    custom_domain_ops = []
    for node in model.graph.node:
        if node.domain and node.domain not in PORTABLE_DOMAINS:
            custom_domain_ops.append(f"{node.domain}:{node.op_type}")
        if node.op_type in CUDA_ONLY_OPS:
            disqualifiers.append(node.op_type)
        if node.op_type == "GridSample":
            x_rank = ranks.get(node.input[0])
            grid_rank = ranks.get(node.input[1]) if len(node.input) > 1 else None
            grid_samples.append({"x_rank": x_rank, "grid_rank": grid_rank})
            if x_rank is not None and x_rank != 4:
                disqualifiers.append(f"GridSample-{x_rank}D")

    def io(values):
        out = []
        for value in values:
            dims = [
                d.dim_param or (d.dim_value if d.HasField("dim_value") else "?")
                for d in value.type.tensor_type.shape.dim
            ]
            out.append({"name": value.name, "shape": dims})
        return out

    max_ai_onnx = max((v for k, v in opsets.items() if k == "ai.onnx"), default=0)
    portable = (
        not disqualifiers
        and not custom_domain_ops
        and max_ai_onnx <= 20
    )
    report = {
        "onnx": str(args.onnx),
        "ir_version": model.ir_version,
        "opset_imports": opsets,
        "inputs": io(model.graph.input),
        "outputs": io(model.graph.output),
        "num_nodes": len(model.graph.node),
        "op_histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "grid_samples": grid_samples,
        "custom_domain_ops": sorted(set(custom_domain_ops)),
        "golden_rule_disqualifiers": sorted(set(disqualifiers)),
        "portable_directml_webgpu_qnn": portable,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
