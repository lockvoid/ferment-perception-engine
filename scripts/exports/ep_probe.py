"""Replay a golden (input, reference) pair through one ONNX Runtime execution
provider and report parity. Reused across models and EPs (CPU / DirectML /
CUDA); run on the Mac for CPU and on the Windows box for DmlExecutionProvider /
CUDAExecutionProvider.

Fail-loud discipline mirrors the Rust `OrtBackend`: if the requested provider is
not actually the one ORT selected, this exits non-zero instead of silently
reporting CPU numbers under a GPU label.

Usage (box, single-input model):
  python ep_probe.py --onnx rife_v4.25.fp16.onnx --provider dml \
      --feed input=rife.input.npy --reference rife.reference.npy
Multi-input model:
  python ep_probe.py --onnx fastdvdnet.fp16.onnx --provider dml \
      --feed frames=fastdvdnet.input_frames.npy \
      --feed noise_map=fastdvdnet.input_noise.npy \
      --reference fastdvdnet.reference.npy
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
}


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)


def parse_feed(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--feed expects name=path, got {value!r}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--feed",
        type=parse_feed,
        action="append",
        required=True,
        help="input tensor as name=path.npy (repeatable for multi-input models)",
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    args = parser.parse_args()

    requested = PROVIDERS[args.provider]
    if requested == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        # ORT >= 1.21: load cudart/cublas/cufft/cudnn from the nvidia-*-cu12 pip
        # wheels (no system CUDA toolkit needed). Failures surface at session
        # creation with a clear provider error — fail-loud is preserved.
        ort.preload_dlls()
    available = ort.get_available_providers()
    if requested not in available:
        raise SystemExit(f"provider {requested} not available; have {available}")

    session = ort.InferenceSession(str(args.onnx), providers=[requested])
    used = session.get_providers()
    if used[0] != requested:
        raise SystemExit(f"ORT selected {used} but {requested} was requested (silent downgrade)")
    # The Python wrapper otherwise recreates the session on CPU after a runtime
    # EP_FAIL and silently retries — which would report CPU numbers under a GPU
    # label (observed with cuDNN-9-on-Pascal). Fail loudly instead.
    session.disable_fallback()

    feeds = {name: np.load(path) for name, path in args.feed}
    reference = np.load(args.reference)
    actual = session.run(None, feeds)[0]

    result = {
        "onnx": str(args.onnx),
        "provider": requested,
        "session_providers": used,
        "max_abs_diff_vs_torch_fp32": float(
            np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)))
        ),
        "psnr_db_vs_torch_fp32": psnr(reference, actual),
        "output_shape": list(actual.shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
