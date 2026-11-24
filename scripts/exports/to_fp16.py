"""Convert an fp32 ONNX model to fp16 (keep_io_types) with the Resize-scales fix.

Reusable across 016 models. The ORT fp16 converter retypes Resize roi/scales to
fp16 (invalid — they must stay fp32); force_resize_scales_fp32 repairs that while
leaving the data path fp16 (ORT Resize accepts fp16 data + fp32 scales).

Usage: python to_fp16.py --in model.onnx --out model.fp16.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.transformers import float16

FP16_OP_BLOCK_LIST = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []))


def force_resize_scales_fp32(model: onnx.ModelProto) -> None:
    graph = model.graph
    guarded = set()
    for node in graph.node:
        if node.op_type == "Resize":
            guarded.update(name for name in node.input[1:] if name)
    if not guarded:
        return
    for init in graph.initializer:
        if init.name in guarded and init.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in graph.node:
        if node.op_type != "Constant" or not any(o in guarded for o in node.output):
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT16:
                arr = numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))


def topologically_sort(model: onnx.ModelProto) -> None:
    graph = model.graph
    available = {v.name for v in graph.input}
    available.update(i.name for i in graph.initializer)
    remaining = list(graph.node)
    ordered = []
    while remaining:
        for index, node in enumerate(remaining):
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(name for name in node.output if name)
                remaining.pop(index)
                break
        else:
            stuck = ", ".join(n.name or n.op_type for n in remaining[:5])
            raise RuntimeError(f"could not topologically sort near {stuck}")
    del graph.node[:]
    graph.node.extend(ordered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", dest="output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.source)
    fp16 = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True, op_block_list=FP16_OP_BLOCK_LIST
    )
    force_resize_scales_fp32(fp16)
    topologically_sort(fp16)
    onnx.checker.check_model(fp16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"wrote {args.output} bytes={args.output.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
