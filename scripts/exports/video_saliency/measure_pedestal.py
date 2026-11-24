"""Measure U^2-Netp's heat pedestal and derive the equivalent `heatFloor`.

iOS calibrated `VideoSaliencyConfig.heatFloor = 0.3` against Apple Vision's
attention map, whose background sits at a mean of ~0.06 under a ~0.85 peak.
U^2-Netp is a salient-OBJECT segmenter: its background is a near-zero
probability, so 0.3 x peak is not the same cut at all. This script measures the
substitute's own pedestal on real frames and reports the floor that lands in
the same place relative to background and peak.

Frames come from the FermentEngine video fixtures (they are third-party
footage and stay out of git); `--frames-dir` uses a directory of stills
instead. Every frame is prepared exactly as the shipping Rust path prepares it:
long edge capped at 512, then stretched to the model's 320x320 grid, divided by
its own peak, ImageNet-shifted.

Usage:
  python measure_pedestal.py --fixtures ../../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_EDGE = 320
# `VideoSaliencyConfig.maxLongEdge` -- the cap the grabber applies before the
# frame ever reaches a heat source.
MAX_LONG_EDGE = 512

# iOS's calibration point, from `VideoSaliencyConfig.heatFloor`'s doc comment:
# "mean ~0.06 against a 0.85 peak on a talking head".
VISION_FLOOR = 0.3
VISION_PEDESTAL = 0.06
VISION_PEAK = 0.85

# Candidate floors compared side by side, so "the constant barely matters for
# this model" is a measurement rather than a claim.
FLOOR_SWEEP = (0.0, 0.1, 0.2468, 0.3, 0.5)

# (video, seconds) -- a spread of subject kinds, all from the engine fixtures.
FIXTURE_FRAMES = [
    ("Video/dhh-interview.mp4", 2.0),
    ("Video/dhh-interview.mp4", 12.0),
    ("Video/djs-walking-city.mp4", 1.86),
    ("Video/cooking-chiken-rosemary.mp4", 5.0),
    ("Video/hairdryer-product-placement.mp4", 3.0),
    ("Video/electronic-music-festival.mp4", 2.0),
    ("Video/formula1-onboard.mov", 4.0),
    ("Video/cloudy_9x16.mp4", 6.0),
]


def vision_cut_position() -> float:
    """Where iOS's floor sits between Vision's background and its peak."""
    return (VISION_FLOOR * VISION_PEAK - VISION_PEDESTAL) / (VISION_PEAK - VISION_PEDESTAL)


def equivalent_floor(background: float, peak: float, position: float) -> float:
    """The floor fraction that cuts at the same relative height."""
    if peak <= 0:
        return VISION_FLOOR
    return (background + position * (peak - background)) / peak


def prepare(image: Image.Image) -> np.ndarray:
    long_edge = max(image.size)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        capped = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(capped, Image.Resampling.LANCZOS)
    grid = image.convert("RGB").resize(
        (MODEL_EDGE, MODEL_EDGE), Image.Resampling.LANCZOS
    )
    array = np.asarray(grid).astype(np.float64) / 255.0
    array = array / max(array.max(), 1e-6)
    for channel in range(3):
        array[:, :, channel] = (array[:, :, channel] - MEAN[channel]) / STD[channel]
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]).astype(np.float32)


def heat_measurement(heat: np.ndarray, floor: float) -> dict:
    """The Rust `measure::heat_measurement`, for comparing candidate floors."""
    height, width = heat.shape
    cutoff = float(heat.max()) * floor
    values = np.clip(heat.astype(np.float64) - cutoff, 0.0, None)
    mass = values.sum()
    if mass <= 0:
        return {"cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0, "confidence": 0.0}
    xs = (np.arange(width) + 0.5) / width
    ys = (np.arange(height) + 0.5) / height
    cx = float((values.sum(axis=0) * xs).sum() / mass)
    cy = float((values.sum(axis=1) * ys).sum() / mass)
    sigma_x = float(np.sqrt((values.sum(axis=0) * (xs - cx) ** 2).sum() / mass))
    sigma_y = float(np.sqrt((values.sum(axis=1) * (ys - cy) ** 2).sum() / mass))
    window = np.outer(np.abs(ys - cy) <= 0.15, np.abs(xs - cx) <= 0.15)
    return {
        "cx": cx,
        "cy": cy,
        "w": min(1.0, cx + 2 * sigma_x) - max(0.0, cx - 2 * sigma_x),
        "h": min(1.0, cy + 2 * sigma_y) - max(0.0, cy - 2 * sigma_y),
        "confidence": float(values[window].sum() / mass),
    }


def extract(video: Path, seconds: float, into: Path) -> Path:
    target = into / f"{video.stem}-{seconds}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(seconds), "-i", str(video),
         "-frames:v", "1", str(target)],
        check=True,
    )
    return target


# The one frame that ships in-repo, so the table has a row anybody can
# reproduce without the FermentEngine fixture tree. Named explicitly rather
# than globbed: `frames/` also holds YuNet's letterboxed reference, whose black
# padding would report a pedestal that no real frame has.
COMMITTED_FRAMES = ("selfie_320.png",)


def collect(fixtures: Path | None, frames_dir: Path | None, scratch: Path) -> list[Path]:
    frames = [HERE / "frames" / name for name in COMMITTED_FRAMES]
    frames = [frame for frame in frames if frame.exists()]
    if frames_dir:
        frames += sorted(p for p in frames_dir.iterdir() if p.suffix in {".png", ".jpg"})
    if fixtures:
        for relative, seconds in FIXTURE_FRAMES:
            video = fixtures / relative
            if video.exists():
                frames.append(extract(video, seconds, scratch))
            else:
                print(f"SKIP {relative}: not in {fixtures}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=HERE / "artifacts/u2netp.saliency.onnx")
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "artifacts/pedestal.json")
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    position = vision_cut_position()
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        frames = collect(args.fixtures, args.frames_dir, Path(scratch))
        if not frames:
            raise SystemExit("no frames: pass --fixtures or --frames-dir")
        for frame in frames:
            heat = session.run(["saliency"], {"image": prepare(Image.open(frame))})[0]
            heat = heat[0, 0]
            peak = float(heat.max())
            # 10th percentile, not the median: on a frame whose subject fills
            # more than half the grid (a product shot) the median IS the
            # subject, and a background estimator that reads the subject
            # reports a floor twice too high.
            background = float(np.percentile(heat, 10))
            rows.append(
                {
                    "frame": frame.name,
                    "peak": peak,
                    "mean": float(heat.mean()),
                    "median": float(np.median(heat)),
                    "p10": background,
                    "background_over_peak": background / peak if peak else 0.0,
                    "equivalent_floor": equivalent_floor(background, peak, position),
                    "sweep": {
                        f"{floor:g}": heat_measurement(heat, floor)
                        for floor in FLOOR_SWEEP
                    },
                }
            )

    floors = [row["equivalent_floor"] for row in rows]
    report = {
        "vision_reference": {
            "floor": VISION_FLOOR,
            "pedestal": VISION_PEDESTAL,
            "peak": VISION_PEAK,
            "cut_position_between_background_and_peak": position,
        },
        "frames": rows,
        "equivalent_floor_median": float(np.median(floors)),
        "equivalent_floor_min": float(min(floors)),
        "equivalent_floor_max": float(max(floors)),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"vision cut sits {position:.4f} of the way from background to peak\n")
    header = (
        f"{'frame':<36}{'peak':>8}{'mean':>9}{'p10':>10}{'bg/peak':>10}{'equiv floor':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['frame']:<36}{row['peak']:>8.4f}{row['mean']:>9.4f}"
            f"{row['p10']:>10.6f}{row['background_over_peak']:>10.6f}"
            f"{row['equivalent_floor']:>13.4f}"
        )
    print(
        f"\nequivalent floor: median {report['equivalent_floor_median']:.4f}"
        f"  min {report['equivalent_floor_min']:.4f}"
        f"  max {report['equivalent_floor_max']:.4f}"
    )

    print("\ncentroid / extent / confidence against the candidate floors")
    sweep_header = f"{'frame':<36}" + "".join(
        f"{'floor ' + format(floor, 'g'):>26}" for floor in FLOOR_SWEEP
    )
    print(sweep_header)
    print("-" * len(sweep_header))
    for row in rows:
        cells = "".join(
            f"{m['cx']:>7.3f}{m['cy']:>6.3f}{m['w']:>6.3f}{m['confidence']:>7.3f}"
            for m in (row["sweep"][f"{floor:g}"] for floor in FLOOR_SWEEP)
        )
        print(f"{row['frame']:<36}{cells}")
    print("(each block: cx cy w confidence)")


if __name__ == "__main__":
    main()
