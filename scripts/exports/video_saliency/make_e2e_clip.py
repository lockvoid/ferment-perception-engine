"""Render the synthetic two-scene clip the saliency end-to-end test measures.

Scene 0 (0-4 s): a bright disc parked at (0.30, 0.50).
Scene 1 (4-10 s): the same disc crossing x 0.15 -> 0.85 at constant y.

Mirrors the Swift `SyntheticSubjectFixture` two-scene spec, so the ported
assertions ("a still scene gives a still verdict", "a travelling scene keeps
its curve") measure the same thing. A hard cut at frame 120 separates them.

Needs ffmpeg on PATH. The output is committed under
crates/engine-processors/tests/fixtures/.

Usage: python make_e2e_clip.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[2] / "crates/engine-processors/tests/fixtures/saliency-two-scenes.mp4"

WIDTH, HEIGHT, FPS = 512, 288, 30
STILL_FRAMES, TRAVEL_FRAMES = 120, 180
RADIUS = round(0.12 * HEIGHT)
# Flat, slightly cool background and a warm subject: enough separation for a
# salient-object model, no texture that could invent a second subject.
BACKGROUND = (58, 62, 72)
SUBJECT = (238, 232, 214)


def frame(x: float, y: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    cx, cy = x * WIDTH, y * HEIGHT
    draw.ellipse(
        [cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS], fill=SUBJECT
    )
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        index = 0
        for _ in range(STILL_FRAMES):
            frame(0.30, 0.50).save(Path(scratch) / f"{index:05d}.png")
            index += 1
        for step in range(TRAVEL_FRAMES):
            x = 0.15 + (0.85 - 0.15) * step / (TRAVEL_FRAMES - 1)
            frame(x, 0.50).save(Path(scratch) / f"{index:05d}.png")
            index += 1
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-framerate", str(FPS),
                "-i", str(Path(scratch) / "%05d.png"),
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "20",
                "-pix_fmt", "yuv420p",
                # Short GOP: every planned grab lands near a sync sample, so the
                # test measures the sampling plan and not the decoder's patience.
                "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
                str(OUTPUT),
            ],
            check=True,
        )
    print(f"{OUTPUT}: {OUTPUT.stat().st_size} bytes, {index} frames")


if __name__ == "__main__":
    main()
