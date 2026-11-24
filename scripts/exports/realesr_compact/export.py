"""Export a Real-ESRGAN Compact checkpoint to fp16 ONNX and CPU-golden it.

This is the harness every later 016 model reuses: torch reference -> fp32 ONNX
(opset 17, dynamic H/W) -> fp16 ONNX -> parity check on the CPU EP, emitting a
fixed golden input/output pair so the *same* tensors can be replayed on
DirectML / CUDA / WebGPU via ep_probe.py.

Usage:
  python export.py --weights realesr-general-x4v3.pth --config realesr-general-x4v3 \
                   --output-dir artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from onnxruntime.transformers import float16

from srvgg import CONFIGS, SRVGGNetCompact

# The nearest-neighbour residual Resize runs fine in fp16 (fp16 data + fp32
# scales), so it is NOT blocked — that would force its output to fp32 and clash
# with the fp16 conv body at the residual Add. Instead `force_resize_scales_fp32`
# fixes only the scales/roi the converter wrongly retyped to fp16.
FP16_OP_BLOCK_LIST = list(getattr(float16, "DEFAULT_OP_BLOCK_LIST", []))

OPSET = 17
GOLDEN_SHAPE = (1, 3, 64, 64)  # LR input; SR output is 4x on each spatial axis


def load_checkpoint(model: torch.nn.Module, weights: Path) -> None:
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if isinstance(blob, dict) and key in blob:
            blob = blob[key]
            break
    model.load_state_dict(blob, strict=True)


def force_resize_scales_fp32(model: onnx.ModelProto) -> None:
    """Resize requires fp32 roi/scales/sizes; the fp16 converter retypes the
    Constant feeding them without a cast. Retype those producers back to fp32
    (the data input may stay fp16 — ORT's Resize accepts a float16 X)."""
    graph = model.graph
    guarded = set()
    for node in graph.node:
        if node.op_type == "Resize":
            # inputs: X, roi, scales, sizes — everything past X must be fp32.
            guarded.update(name for name in node.input[1:] if name)
    if not guarded:
        return
    for initializer in graph.initializer:
        if initializer.name in guarded and initializer.data_type == onnx.TensorProto.FLOAT16:
            array = numpy_helper.to_array(initializer).astype(np.float32)
            initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        if not any(output in guarded for output in node.output):
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.t.data_type == onnx.TensorProto.FLOAT16:
                array = numpy_helper.to_array(attribute.t).astype(np.float32)
                attribute.t.CopyFrom(numpy_helper.from_array(array, attribute.t.name))


def topologically_sort(model: onnx.ModelProto) -> None:
    """Repair node order after the fp16 converter inserts I/O Cast nodes."""
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(init.name for init in graph.initializer)
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
            stuck = ", ".join(node.name or node.op_type for node in remaining[:5])
            raise RuntimeError(f"could not topologically sort ONNX graph near {stuck}")
    del graph.node[:]
    graph.node.extend(ordered)


def export_fp16(model: torch.nn.Module, sample: torch.Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fp32 = output.with_suffix(".fp32.onnx")
    torch.onnx.export(
        model,
        (sample,),
        fp32,
        input_names=["lr"],
        output_names=["sr"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes={"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}},
        dynamo=False,
    )
    try:
        loaded = onnx.load(fp32)
        fp16 = float16.convert_float_to_float16(
            loaded,
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=FP16_OP_BLOCK_LIST,
        )
        force_resize_scales_fp32(fp16)
        topologically_sort(fp16)
        onnx.checker.check_model(fp16)
        onnx.save(fp16, output)
    finally:
        fp32.unlink(missing_ok=True)


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = SRVGGNetCompact(**CONFIGS[args.config]).eval()
    load_checkpoint(model, args.weights)

    torch.manual_seed(0)
    sample = torch.rand(*GOLDEN_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        reference = model(sample).numpy()

    stem = args.config
    onnx_path = args.output_dir / f"{stem}.fp16.onnx"
    export_fp16(model, sample, onnx_path)

    # CPU-EP parity against the torch fp32 reference on the fixed golden input.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"lr": sample.numpy()})[0]

    max_diff = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
    golden_psnr = psnr(reference, actual)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    # Persist the golden pair so the box replays identical tensors on GPU EPs.
    np.save(args.output_dir / f"{stem}.input.npy", sample.numpy())
    np.save(args.output_dir / f"{stem}.reference.npy", reference)
    manifest = {
        "config": args.config,
        "onnx": onnx_path.name,
        "sha256": digest,
        "bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "golden_input_shape": list(sample.shape),
        "golden_output_shape": list(reference.shape),
        "cpu_max_abs_diff_vs_torch_fp32": max_diff,
        "cpu_psnr_db_vs_torch_fp32": golden_psnr,
    }
    (args.output_dir / f"{stem}.golden.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
