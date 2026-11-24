"""The three DINOv2 numbers this port owes the reader, all measured rather than
argued:

1. **Resampler.** iOS prepares the encoder's input with vImage's
   `kvImageHighQualityResampling`; Rust uses `Lanczos3`. How far apart are the
   two EMBEDDINGS of the same source frame?
2. **fp16.** iOS ships fp16 for the ANE. What does fp16 cost this graph in
   accuracy and, on the CPU EP that is Studio's qualified baseline, in time?
3. **Normalisation.** iOS inherits CoreML's single-scalar `ImageType` scale
   (`mean(1/(255*std))`) instead of the exact per-channel gain. How much does
   staying bug-compatible with iOS actually change the vector?

Prerequisites:
    uv run --project scripts/exports/dinov2 python scripts/exports/dinov2/export_dinov2.py \
        --out scripts/exports/dinov2/artifacts/dinov2_vits14.onnx
    uv run --project scripts/exports/dinov2 python scripts/exports/to_fp16.py \
        --in  scripts/exports/dinov2/artifacts/dinov2_vits14.onnx \
        --out scripts/exports/dinov2/artifacts/dinov2_vits14.fp16.onnx
    cargo test -p ferment-engine-processors --lib dinov2::resampler -- --ignored

    uv run --project scripts/exports/dinov2 python scripts/exports/dinov2/measure_drift.py
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch

from export_dinov2 import H, W, load_identity

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
FP32 = HERE / "artifacts/dinov2_vits14.onnx"
FP16 = HERE / "artifacts/dinov2_vits14.fp16.onnx"
LANCZOS = REPO / "target/dinov2_lanczos3_224x126.rgb"


def planar(path: pathlib.Path) -> np.ndarray:
    frame = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(H, W, 3)
    return frame.transpose(2, 0, 1)[None, ...].astype(np.float32)


def session(path: pathlib.Path) -> ort.InferenceSession:
    return ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])


def embed(sess: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    return sess.run(None, {"image": image})[0].astype(np.float32).ravel()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def coreml_embeddings() -> tuple[np.ndarray, np.ndarray] | None:
    """The iOS artifact's own answer for the two resampled frames, via
    `coreml_reference.swift`. `None` where the iOS checkout is not present."""
    model = (
        REPO.parent
        / "ferment-cuts-ios/Packages/FermentEngine/LocalModels/dinov2_vits14_identity_fp32.mlmodelc"
    )
    if not model.is_dir():
        return None
    with tempfile.TemporaryDirectory() as scratch:
        binary = pathlib.Path(scratch) / "coreml_reference"
        build = subprocess.run(
            ["swiftc", "-O", (HERE / "coreml_reference.swift").as_posix(), "-o", binary.as_posix()],
            capture_output=True,
        )
        if build.returncode != 0:
            return None
        out = []
        for frame in (HERE / "goldens/resample_vimage_224x126.rgb", LANCZOS):
            dump = pathlib.Path(scratch) / f"{frame.stem}.f32"
            subprocess.run(
                [binary.as_posix(), model.as_posix(), frame.as_posix(), dump.as_posix()],
                check=True,
                capture_output=True,
            )
            out.append(np.frombuffer(dump.read_bytes(), dtype=np.float32))
        return out[0], out[1]


def main() -> None:
    vimage = planar(HERE / "goldens/resample_vimage_224x126.rgb")
    if not LANCZOS.exists():
        raise SystemExit(
            f"{LANCZOS} is missing — run:\n"
            "  cargo test -p ferment-engine-processors --lib dinov2::resampler -- --ignored"
        )
    lanczos = planar(LANCZOS)

    fp32 = session(FP32)
    a, b = embed(fp32, vimage), embed(fp32, lanczos)
    pixel = np.abs(vimage - lanczos)
    print("1. RESAMPLER — vImage kvImageHighQualityResampling vs image-rs Lanczos3")
    print(f"   pixel   mean |delta| {pixel.mean():.4f}/255   max {pixel.max():.0f}/255")
    print(f"   embedding cosine distance   {cosine_distance(a, b):.6f}")
    print(f"   euclidean (unit vectors)    {float(np.linalg.norm(a - b)):.6f}")
    print("   for scale: same shot across a flash <= 0.22 cosine, a real cut >= 0.34")

    print("\n2. FP16 vs FP32 on the CPU EP")
    if FP16.exists():
        fp16 = session(FP16)
        c = embed(fp16, vimage)
        print(f"   embedding cosine distance   {cosine_distance(a, c):.6f}")
        print(f"   max |delta| per component   {float(np.abs(a - c).max()):.3e}")
        for label, sess in (("fp32", fp32), ("fp16", fp16)):
            embed(sess, vimage)  # warm
            start = time.perf_counter()
            for _ in range(20):
                embed(sess, vimage)
            print(f"   {label} CPU EP  {(time.perf_counter() - start) / 20 * 1000:.2f} ms/frame")
    else:
        print(f"   SKIP: {FP16} not built")

    print("\n3. NORMALISATION — iOS CoreML scalar-mean gain vs exact per-channel")
    with tempfile.TemporaryDirectory() as scratch:
        exact_path = pathlib.Path(scratch) / "exact.onnx"
        subprocess.run(
            [
                "python",
                (HERE / "export_dinov2.py").as_posix(),
                "--out",
                exact_path.as_posix(),
                "--exact-imagenet-norm",
            ],
            check=True,
            capture_output=True,
        )
        exact = embed(session(exact_path), vimage)
    print(f"   embedding cosine distance   {cosine_distance(a, exact):.6f}")
    print(f"   max |delta| per component   {float(np.abs(a - exact).max()):.3e}")

    print("\n4. THE iOS REFERENCE — the ONNX master against the shipped .mlmodelc")
    coreml = coreml_embeddings()
    if coreml is None:
        print("   SKIP: no ../ferment-cuts-ios LocalModels checkout, or swiftc unavailable")
    else:
        cm_vimage, cm_lanczos = coreml
        print(f"   graph, identical pixels          {cosine_distance(a, cm_vimage):.3e}")
        print(f"   Studio end to end vs iOS         {cosine_distance(b, cm_vimage):.6f}")
        print(f"   resampler alone, in iOS's model  {cosine_distance(cm_lanczos, cm_vimage):.6f}")

    print("\n5. Sanity — the property every consumer rides")
    model = load_identity()
    rng = np.random.default_rng(2026)
    base = rng.integers(0, 256, size=(1, 3, H, W)).astype(np.float32)
    lit = np.clip(base + 60.0, 0.0, 255.0)
    other = rng.integers(0, 256, size=(1, 3, H, W)).astype(np.float32)
    with torch.no_grad():
        vectors = [
            model(torch.from_numpy(image)).numpy().ravel() for image in (base, lit, other)
        ]
    print(f"   same picture under light    {cosine_distance(vectors[0], vectors[1]):.4f}")
    print(f"   a different picture         {cosine_distance(vectors[0], vectors[2]):.4f}")


if __name__ == "__main__":
    main()
