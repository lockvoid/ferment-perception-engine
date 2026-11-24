"""Stage pyannote/segmentation-3.0 (MIT) as the ONNX master, and qualify it.

Produces:
  - artifacts/pyannote_segmentation.onnx   fp32, opset 17, 5.7 MB

    cd scripts/exports/pyannote && uv sync && uv run python stage.py

There is no export step. The weights are pyannote/segmentation-3.0 — PyanNet,
~1.5M parameters (SincNet + 2x BiLSTM + 2 dense + a 7-way classifier), 10 s /
16 kHz window, MIT. The original checkpoint on Hugging Face is GATED, and a
gated download is not a build step; `onnx-community/pyannote-segmentation-3.0`
is an ungated mirror of the same weights, pinned here by repository revision.
It is also the exact artifact the iOS port's
`Scripts/speaker_segmenter/make_fixtures.py` computed
`Tests/Fixtures/Pyannote/*/reference_*.bin` from, so the goldens the Rust
oracle replays and this file are two halves of one measurement.

What this script does instead of exporting is QUALIFY, and one check carries
the weight: the output must be `[1, 589, 7]`. `speaker-diarization-community-1`
ships in the same family under a similar name with an EIGHT-class powerset and
a different frame rate; taking it by accident would mis-decode every frame
while still producing plausible speaker turns. The check has to be a RUN rather
than a schema read, because this graph declares `num_frames` dynamic — only
feeding it a real window pins the frame count.
"""

import hashlib
import os
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "pyannote_segmentation.onnx")

REPO = "onnx-community/pyannote-segmentation-3.0"
# Pinned by revision, not by `main`: "the same weights" has to survive the
# mirror being updated.
REVISION = "733a93b6473d019a773298e08cefa686894b1854"
REMOTE_PATH = "onnx/model.onnx"
EXPECTED_SHA256 = "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25"

# The window and the class count are the model's, not a preference: SincNet's
# stride ladder turns 160_000 samples into exactly 589 frames, and powerset over
# <=3 speakers with <=2 simultaneous is exactly 7 classes.
WINDOW_SAMPLES = 160_000
FRAMES = 589
CLASSES = 7


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch() -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{REMOTE_PATH}"
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, ONNX_PATH)
    digest = sha256(ONNX_PATH)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"sha256 {digest}, expected {EXPECTED_SHA256}")
    print(f"{ONNX_PATH}\n  {os.path.getsize(ONNX_PATH):,} bytes  sha256 {digest}")
    return ONNX_PATH


def qualify(path: str) -> None:
    model = onnx.load(path)
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    if set(opsets) != {"ai.onnx"} or opsets["ai.onnx"] > 20:
        raise RuntimeError(f"opset imports {opsets} are not portable (need ai.onnx <= 20)")

    # The decoder exponentiates what comes back, and that is only correct
    # because the graph normalises inside itself. `logits` is the tensor's
    # NAME, not its content.
    if model.graph.node[-1].op_type != "LogSoftmax":
        raise RuntimeError(
            f"graph ends in {model.graph.node[-1].op_type}, not LogSoftmax — "
            "the Rust decoder assumes log-probabilities come back"
        )

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in session.get_inputs()}
    if list(inputs) != ["input_values"]:
        raise RuntimeError(f"unexpected input schema {inputs}")

    window = np.zeros((1, 1, WINDOW_SAMPLES), dtype=np.float32)
    output = session.run(None, {"input_values": window})[0]
    if output.shape != (1, FRAMES, CLASSES):
        raise RuntimeError(
            f"unexpected output shape {output.shape} — segmentation-3.0 emits "
            f"[1, {FRAMES}, {CLASSES}]; an 8-class output means this is the "
            "community-1 checkpoint"
        )

    total = np.exp(output.astype(np.float64)).sum(axis=-1)
    if not np.allclose(total, 1.0, atol=1e-5):
        raise RuntimeError(f"exp(output) does not sum to 1 (min {total.min()}, max {total.max()})")

    print(f"  opset {opsets['ai.onnx']}, input {inputs}, output {output.shape}")
    print("  exp(output) sums to 1 per frame — the graph speaks in log-probabilities")
    print("  OK: this is segmentation-3.0, not community-1")


if __name__ == "__main__":
    qualify(fetch())
