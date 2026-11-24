"""Sweep YuNet's detection threshold against the FermentEngine video fixtures.

iOS delegates the decision to Vision, which has no exposed threshold. The
substitute needs one, and it is not free: a face wins over the heat map
outright, so a false positive moves a whole scene's verdict, while a miss drops
the scene into the (differently scaled) saliency lane. OpenCV Zoo ships 0.9;
this script shows what that buys on our own footage.

Frames are prepared exactly as the shipping Rust path prepares them: long edge
capped at 512, scaled uniformly into the model's 640x640 grid, padded at the
right and bottom, fed BGR 0..255.

Usage:
  python measure_face_threshold.py --fixtures <FermentEngine/Tests/Fixtures>
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent

MODEL_EDGE = 640
MAX_LONG_EDGE = 512
STRIDES = (8, 16, 32)
NMS_IOU = 0.3
SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
# Below this nothing is reported at all -- the sweep floor, not a candidate.
REPORT_FLOOR = 0.4

# (video, seconds, what the frame actually holds -- checked by eye, not guessed)
FIXTURE_FRAMES = [
    ("Video/dhh-interview.mp4", 2.0, "1 frontal face"),
    ("Video/dhh-interview.mp4", 8.0, "1 frontal face"),
    ("Video/dhh-interview.mp4", 16.0, "1 frontal face"),
    ("Video/djs-walking-city.mp4", 1.86, "2 faces, one nearer"),
    ("Video/hairdryer-product-placement.mp4", 3.0, "1 frontal face"),
    # A DJ shot from behind: a head, no frontal face. Anything above 0 here is
    # the false positive the threshold has to keep out.
    ("Video/cloudy_9x16.mp4", 6.0, "0 (head from behind)"),
    ("Video/formula1-onboard.mov", 4.0, "0 (cockpit)"),
]


def blob(image: Image.Image) -> tuple[np.ndarray, float]:
    long_edge = max(image.size)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    scale = MODEL_EDGE / max(image.size)
    fitted = image.convert("RGB").resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (MODEL_EDGE, MODEL_EDGE), (0, 0, 0))
    canvas.paste(fitted, (0, 0))
    array = np.asarray(canvas).astype(np.float32)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]), scale


def decode(outputs: dict, threshold: float) -> list[tuple[float, float, float, float, float]]:
    faces = []
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"][0, :, 0]
        obj = outputs[f"obj_{stride}"][0, :, 0]
        bbox = outputs[f"bbox_{stride}"][0]
        cells = MODEL_EDGE // stride
        for row in range(cells):
            for column in range(cells):
                index = row * cells + column
                score = float(
                    np.sqrt(np.clip(cls[index], 0, 1) * np.clip(obj[index], 0, 1))
                )
                if score < threshold:
                    continue
                cx = (column + bbox[index, 0]) * stride
                cy = (row + bbox[index, 1]) * stride
                w = float(np.exp(bbox[index, 2]) * stride)
                h = float(np.exp(bbox[index, 3]) * stride)
                faces.append((cx - w / 2, cy - h / 2, w, h, score))
    return nms(faces)


def nms(faces: list) -> list:
    kept: list = []
    for face in sorted(faces, key=lambda f: -f[4]):
        if all(iou(face, keeper) < NMS_IOU for keeper in kept):
            kept.append(face)
    return kept


def iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def extract(video: Path, seconds: float, into: Path) -> Path:
    target = into / f"{video.stem}-{seconds}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(seconds), "-i", str(video),
         "-frames:v", "1", str(target)],
        check=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=HERE / "artifacts/face_detection_yunet_2026may.onnx"
    )
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]

    header = (
        f"{'frame':<36}{'actual content':<24}"
        + "".join(f"{t:>7}" for t in SWEEP)
        + "   scores"
    )
    print(header)
    print("-" * len(header))
    with tempfile.TemporaryDirectory() as scratch:
        for relative, seconds, expectation in FIXTURE_FRAMES:
            video = args.fixtures / relative
            if not video.exists():
                print(f"SKIP {relative}: not in {args.fixtures}")
                continue
            frame = extract(video, seconds, Path(scratch))
            tensor, _ = blob(Image.open(frame))
            raw = session.run(names, {"input": tensor})
            outputs = dict(zip(names, raw))
            counts = "".join(f"{len(decode(outputs, t)):>7}" for t in SWEEP)
            scores = ", ".join(
                f"{face[4]:.3f}" for face in decode(outputs, REPORT_FLOOR)
            )
            print(f"{frame.name:<36}{expectation:<24}{counts}   [{scores}]")
    print(
        "\n(cells are the face count kept after NMS at that threshold; the bracket"
        f"\n lists every surviving score above {REPORT_FLOOR})"
    )


if __name__ == "__main__":
    main()
