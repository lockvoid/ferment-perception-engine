"""Reference inputs and outputs for the video-saliency ORT oracles.

Both backings ship as released ONNX, so the origin here is the graph itself run
through onnxruntime's CPU EP in Python -- the pyannote pattern. What the oracle
proves is that the Rust pre-processing, tensor plumbing and (for YuNet) the
decode agree with an independent Python implementation on committed pixels.

Reference frames are derived once, here, from OpenCV Zoo's MIT-licensed
`largest_selfie.jpg` and committed at exactly the size each graph eats. Feeding
the model's own grid keeps resize-kernel parity (PIL LANCZOS vs the `image`
crate's Lanczos3) out of the oracle -- that divergence is measured and recorded
in the track report, not gated here.

Outputs land in ../rust_goldens/ next to the rest of the fleet's goldens.

Usage: python gen_goldens.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "frames"
GOLDENS = HERE.parent / "rust_goldens"

# OpenCV Zoo's demo image, MIT-licensed with the rest of that directory. It has
# a crowd of faces at several scales -- exactly the "dominant face, not the
# union" case the port has to get right.
SELFIE_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/example_outputs/largest_selfie.jpg"
SELFIE_SHA = "ab8413ad9bb4f53068f4fb63c6747e5989991dd02241c923d5595b614ecf2bf6"

U2NETP_EDGE = 320
YUNET_EDGE = 640
STRIDES = (8, 16, 32)
# Must track `yunet::SCORE_THRESHOLD` -- the goldens are what ships.
SCORE_THRESHOLD = 0.75
NMS_IOU = 0.3

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def source_image() -> Image.Image:
    FRAMES.mkdir(parents=True, exist_ok=True)
    cached = FRAMES / "largest_selfie.jpg"
    if not cached.exists():
        print(f"fetching {SELFIE_URL}")
        with urllib.request.urlopen(SELFIE_URL) as response:
            cached.write_bytes(response.read())
    digest = hashlib.sha256(cached.read_bytes()).hexdigest()
    if digest != SELFIE_SHA:
        raise SystemExit(f"largest_selfie.jpg sha256 {digest} != pinned {SELFIE_SHA}")
    return Image.open(cached).convert("RGB")


def write_f32(name: str, values: np.ndarray) -> None:
    path = GOLDENS / name
    path.write_bytes(values.astype("<f4").tobytes())
    print(f"{name}: {values.size} f32")


def u2netp_frame(source: Image.Image) -> Image.Image:
    """The model's own grid, so the oracle needs no resize on either side."""
    return source.resize((U2NETP_EDGE, U2NETP_EDGE), Image.Resampling.LANCZOS)


def yunet_frame(source: Image.Image) -> Image.Image:
    """Uniformly scaled into 640x640 and padded at the right and bottom, the
    way OpenCV's `padWithDivisor` pads. Committed already letterboxed, so the
    Rust path's own letterbox is a no-op on it."""
    scale = YUNET_EDGE / max(source.size)
    fitted = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (YUNET_EDGE, YUNET_EDGE), (0, 0, 0))
    canvas.paste(fitted, (0, 0))
    return canvas


def u2netp_tensor(frame: Image.Image) -> np.ndarray:
    array = np.asarray(frame).astype(np.float64) / 255.0
    array = array / max(array.max(), 1e-6)
    for channel in range(3):
        array[:, :, channel] = (array[:, :, channel] - MEAN[channel]) / STD[channel]
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]).astype(np.float32)


def yunet_tensor(frame: Image.Image) -> np.ndarray:
    array = np.asarray(frame).astype(np.float32)[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def decode_faces(outputs: dict[str, np.ndarray]) -> list[dict]:
    """OpenCV `FaceDetectorYNImpl::postProcess`, transcribed."""
    candidates = []
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"][0, :, 0]
        obj = outputs[f"obj_{stride}"][0, :, 0]
        bbox = outputs[f"bbox_{stride}"][0]
        cells = YUNET_EDGE // stride
        for row in range(cells):
            for column in range(cells):
                index = row * cells + column
                score = float(
                    np.sqrt(np.clip(cls[index], 0, 1) * np.clip(obj[index], 0, 1))
                )
                if score < SCORE_THRESHOLD:
                    continue
                cx = (column + float(bbox[index, 0])) * stride
                cy = (row + float(bbox[index, 1])) * stride
                width = float(np.exp(bbox[index, 2]) * stride)
                height = float(np.exp(bbox[index, 3]) * stride)
                candidates.append(
                    {
                        "x": cx - width / 2,
                        "y": cy - height / 2,
                        "width": width,
                        "height": height,
                        "confidence": score,
                    }
                )
    kept: list[dict] = []
    for face in sorted(candidates, key=lambda f: -f["confidence"]):
        if all(iou(face, keeper) < NMS_IOU for keeper in kept):
            kept.append(face)
    # Normalized against the committed 640x640 frame, top-left origin -- the
    # same coordinates `FaceRectSource` returns.
    return [
        {key: value / YUNET_EDGE if key != "confidence" else value
         for key, value in face.items()}
        for face in kept
    ]


def iou(a: dict, b: dict) -> float:
    left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    source = source_image()

    saliency_frame = u2netp_frame(source)
    saliency_frame.save(FRAMES / "selfie_320.png")
    face_frame = yunet_frame(source)
    face_frame.save(FRAMES / "selfie_640.png")

    saliency = ort.InferenceSession(
        str(HERE / "artifacts/u2netp.saliency.onnx"), providers=["CPUExecutionProvider"]
    )
    heat = saliency.run(["saliency"], {"image": u2netp_tensor(saliency_frame)})[0]
    write_f32("u2netp_saliency_out.bin", heat.reshape(-1))
    print(f"  heat peak {heat.max():.6f} mean {heat.mean():.6f}")

    faces = ort.InferenceSession(
        str(HERE / "artifacts/face_detection_yunet_2026may.onnx"),
        providers=["CPUExecutionProvider"],
    )
    names = [output.name for output in faces.get_outputs()]
    outputs = dict(zip(names, faces.run(names, {"input": yunet_tensor(face_frame)})))
    # No raw-tensor golden for YuNet: the decoded boxes are pinned to 1e-6, and
    # any pre-processing drift (channel order, padding, scale) moves them.
    decoded = decode_faces(outputs)
    (GOLDENS / "yunet_faces.json").write_text(json.dumps(decoded, indent=2) + "\n")
    print(f"yunet_faces.json: {len(decoded)} faces above {SCORE_THRESHOLD}")
    for face in decoded:
        print("  " + json.dumps({k: round(v, 6) for k, v in face.items()}))


if __name__ == "__main__":
    main()
