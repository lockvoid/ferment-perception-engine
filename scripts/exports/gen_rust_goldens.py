"""Generate u8-space goldens for the Rust backing integration tests
(`crates/engine-processors/tests/ort_enhance.rs`).

The Rust backings consume u8 RGB24 frames and emit u8 RGB24 frames. This script
reproduces the exact same math (u8/255 planar pack -> fp16 ONNX on the CPU EP ->
clamp*255 round quantize) so the Rust test can compare byte-for-byte against a
small tolerance (cross-onnxruntime-version drift).

Deterministic inputs use pixel = (x*7 + y*13 + c*29 + seed) % 256 — mirrored in
the Rust test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
OUT = HERE / "rust_goldens"
OUT.mkdir(exist_ok=True)


def pattern_frame(width: int, height: int, seed: int) -> np.ndarray:
    """u8 interleaved RGB24 [H, W, 3]."""
    y, x, c = np.meshgrid(
        np.arange(height), np.arange(width), np.arange(3), indexing="ij"
    )
    return ((x * 7 + y * 13 + c * 29 + seed) % 256).astype(np.uint8)


def planar(frame_u8: np.ndarray) -> np.ndarray:
    """[H, W, 3] u8 -> [3, H, W] f32 in [0, 1]."""
    return frame_u8.astype(np.float32).transpose(2, 0, 1) / 255.0


def quantize(planar_f32: np.ndarray) -> np.ndarray:
    """[3, H, W] f32 -> interleaved u8 with the Rust backing's clamp/round."""
    clamped = np.clip(planar_f32, 0.0, 1.0) * 255.0
    return np.round(clamped).astype(np.uint8).transpose(1, 2, 0)


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


meta = {}

# --- Compact (realesr-general-x4v3): 32x24 -> 128x96 --------------------------
w, h = 32, 24
frame = pattern_frame(w, h, seed=3)
sess = session(HERE / "realesr_compact/artifacts/realesr-general-x4v3.fp16.onnx")
sr = sess.run(None, {"lr": planar(frame)[None]})[0][0]
(OUT / "compact_in.bin").write_bytes(frame.tobytes())
(OUT / "compact_out.bin").write_bytes(quantize(sr).tobytes())
meta["compact"] = {"in": [w, h], "out": [w * 4, h * 4], "seed": 3}

# --- FastDVDnet: five 32x24 frames, strength 0.5 ------------------------------
frames = [pattern_frame(w, h, seed=10 * i) for i in range(5)]
noise = np.full((1, 1, h, w), 0.5 * 55.0 / 255.0, dtype=np.float32)
stacked = np.concatenate([planar(f) for f in frames])[None]  # [1,15,H,W]
sess = session(HERE / "fastdvdnet/artifacts/fastdvdnet.fp16.onnx")
den = sess.run(None, {"frames": stacked, "noise_map": noise})[0][0]
for i, f in enumerate(frames):
    (OUT / f"fastdvd_in{i}.bin").write_bytes(f.tobytes())
(OUT / "fastdvd_out.bin").write_bytes(quantize(den).tobytes())
meta["fastdvd"] = {"in": [w, h], "strength": 0.5, "seeds": [0, 10, 20, 30, 40]}

# --- RIFE v2: two 48x48 frames (pads to 64), phase 0.5 ------------------------
size = 48
img0 = pattern_frame(size, size, seed=5)
img1 = pattern_frame(size, size, seed=90)
padded = 64


def planar_padded(frame_u8: np.ndarray) -> np.ndarray:
    p = planar(frame_u8)  # [3, 48, 48]
    return np.pad(p, ((0, 0), (0, padded - size), (0, padded - size)), mode="edge")


x = np.concatenate(
    [planar_padded(img0), planar_padded(img1), np.full((1, padded, padded), 0.5, np.float32)]
)[None]
sess = session(HERE / "rife/artifacts/rife_v4.25_v2.fp16.onnx")
mid = sess.run(None, {"input": x})[0][0][:, :size, :size]  # crop back
(OUT / "rife_in0.bin").write_bytes(img0.tobytes())
(OUT / "rife_in1.bin").write_bytes(img1.tobytes())
(OUT / "rife_out.bin").write_bytes(quantize(mid).tobytes())
meta["rife"] = {"in": [size, size], "phase": 0.5, "seeds": [5, 90]}

(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
print("goldens written to", OUT)
