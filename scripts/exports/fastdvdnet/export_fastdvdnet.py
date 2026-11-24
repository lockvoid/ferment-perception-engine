"""Export FastDVDnet (MIT, Tassano) to fp16 ONNX + CPU golden.

FastDVDnet is a stateless 5-frame temporal denoiser: pure Conv/BN/ReLU +
PixelShuffle + residual, no grid_sample/deformconv — trivially golden-rule
portable. It is the off-macOS backing for `TemporalNoiseFiltering`.

Model input matches the upstream forward(x, noise_map):
  frames    [1, 5*3=15, H, W]  five RGB frames concatenated, [0,1]
  noise_map [1, 1, H, W]       per-pixel noise sigma/255
  ->        [1, 3, H, W]        denoised centre frame

Vendored arch + MIT LICENSE under ./vendor/ (byte-exact from m-tassano/fastdvdnet).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE.parent))  # for the shared fp16 tooling

from models import FastDVDnet  # noqa: E402  (vendored)
import to_fp16  # noqa: E402  (scripts/exports/to_fp16.py)

OPSET = 17
H = W = 64
NOISE_SIGMA = 20.0 / 255.0  # representative noise level


def load_model(weights: Path) -> torch.nn.Module:
    model = FastDVDnet(num_input_frames=5)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()  # eval() so BatchNorm uses running stats


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main() -> None:
    out_dir = HERE / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(HERE / "model.pth")

    torch.manual_seed(0)
    frames = torch.rand(1, 15, H, W, dtype=torch.float32)
    noise_map = torch.full((1, 1, H, W), NOISE_SIGMA, dtype=torch.float32)
    with torch.no_grad():
        reference = model(frames, noise_map).numpy()

    fp32 = out_dir / "fastdvdnet.fp32.onnx"
    onnx_path = out_dir / "fastdvdnet.fp16.onnx"
    torch.onnx.export(
        model,
        (frames, noise_map),
        fp32,
        input_names=["frames", "noise_map"],
        output_names=["denoised"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes={
            "frames": {2: "h", 3: "w"},
            "noise_map": {2: "h", 3: "w"},
            "denoised": {2: "h", 3: "w"},
        },
        dynamo=False,
    )
    fp16 = to_fp16.float16.convert_float_to_float16(
        onnx.load(fp32),
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=to_fp16.FP16_OP_BLOCK_LIST,
    )
    to_fp16.force_resize_scales_fp32(fp16)
    to_fp16.topologically_sort(fp16)
    onnx.checker.check_model(fp16)
    onnx.save(fp16, onnx_path)
    fp32.unlink(missing_ok=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None, {"frames": frames.numpy(), "noise_map": noise_map.numpy()}
    )[0]

    max_diff = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
    np.save(out_dir / "fastdvdnet.input_frames.npy", frames.numpy())
    np.save(out_dir / "fastdvdnet.input_noise.npy", noise_map.numpy())
    np.save(out_dir / "fastdvdnet.reference.npy", reference)
    manifest = {
        "model": "fastdvdnet",
        "license": "MIT",
        "onnx": onnx_path.name,
        "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "output_shape": list(reference.shape),
        "cpu_max_abs_diff_vs_torch_fp32": max_diff,
        "cpu_psnr_db_vs_torch_fp32": psnr(reference, actual),
    }
    (out_dir / "fastdvdnet.golden.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
