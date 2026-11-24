"""RIFE natural-frame goldens (migration 016 §6.3 — the shipping gate).

Extracts consecutive frame pairs from real fixture videos (diverse motion:
onboard racing, camera pan, kitchen close-up), builds the v2-interface input at
t=0.5, and writes:
  - <name>.input.npy / <name>.reference.npy  — fp32-CPU reference pair for
    per-EP replay via ep_probe.py (tolerance derivation);
  - rust_goldens/rife_nat_<name>_{in0,in1,out}.bin — u8 goldens for the Rust
    integration test (out = fp16 master on the CPU EP, quantized with the
    backing's exact math).

Frames are 256x256 center crops at native fps (real inter-frame motion).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
fixtures_dir = os.environ.get("FERMENT_ENGINE_FIXTURES_DIR")
if not fixtures_dir:
    raise SystemExit("Set FERMENT_ENGINE_FIXTURES_DIR to the reference fixture root")
FIXTURES = Path(fixtures_dir) / "Video"
OUT = HERE / "natural"
RUST_OUT = HERE.parent / "rust_goldens"
OUT.mkdir(exist_ok=True)
SIZE = 256

CLIPS = {
    "formula1": (FIXTURES / "formula1-onboard.mov", 3.0),
    "djreel": (FIXTURES / "dj-reel-1.mp4", 2.0),
    "cooking": (FIXTURES / "cooking-chiken-rosemary.mp4", 2.0),
}


def extract_pair(video: Path, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Two consecutive frames as u8 [H, W, 3] center crops."""
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(video),
            "-frames:v", "2",
            "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={SIZE}:{SIZE}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(2, SIZE, SIZE, 3)
    return frames[0], frames[1]


def planar(frame_u8: np.ndarray) -> np.ndarray:
    return frame_u8.astype(np.float32).transpose(2, 0, 1) / 255.0


def quantize(planar_f32: np.ndarray) -> np.ndarray:
    return (
        np.round(np.clip(planar_f32, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    import math

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


fp32 = ort.InferenceSession(
    str(HERE / "extracted/rife_v2/rife_v4.25.onnx"), providers=["CPUExecutionProvider"]
)
fp16 = ort.InferenceSession(
    str(HERE / "artifacts/rife_v4.25_v2.fp16.onnx"), providers=["CPUExecutionProvider"]
)

report = {}
for name, (video, seconds) in CLIPS.items():
    img0, img1 = extract_pair(video, seconds)
    x = np.concatenate(
        [planar(img0), planar(img1), np.full((1, SIZE, SIZE), 0.5, np.float32)]
    )[None]
    reference = fp32.run(None, {"input": x})[0]
    fp16_cpu = fp16.run(None, {"input": x})[0]

    np.save(OUT / f"rife_nat_{name}.input.npy", x)
    np.save(OUT / f"rife_nat_{name}.reference.npy", reference)
    (RUST_OUT / f"rife_nat_{name}_in0.bin").write_bytes(img0.tobytes())
    (RUST_OUT / f"rife_nat_{name}_in1.bin").write_bytes(img1.tobytes())
    (RUST_OUT / f"rife_nat_{name}_out.bin").write_bytes(quantize(fp16_cpu[0]).tobytes())

    report[name] = {
        "fp16_cpu_vs_fp32_max_abs": round(
            float(np.max(np.abs(fp16_cpu.astype(np.float64) - reference.astype(np.float64)))), 5
        ),
        "fp16_cpu_vs_fp32_psnr_db": round(psnr(fp16_cpu, reference), 2),
        "motion_mean_abs_u8": round(
            float(np.mean(np.abs(img1.astype(np.int16) - img0.astype(np.int16)))), 2
        ),
    }

(OUT / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
