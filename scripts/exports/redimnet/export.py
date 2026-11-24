"""Export ReDimNet2-B2 (PalabraAI, MIT) -> ONNX with the mel frontend and the
L2 normalisation inside the graph.

Produces:
  - artifacts/redimnet_speaker.onnx        fp32, opset 17, fixed 2.005 s window
  - a parity report against torch AND against the committed iOS goldens

    cd scripts/exports/redimnet && uv sync && uv run python export.py

Source of truth
---------------
Weights are a GitHub release asset of github.com/PalabraAI/redimnet2 (MIT),
`b2-vox2-lm.pt` (3,677,760 params, 15,897,450 bytes). The architecture is that
repo's own Python, cloned at a pinned commit into `checkpoints/` and traced
live, so the weight mapping is identity by construction and nothing here
re-implements a layer. This mirrors the iOS port's
`Scripts/speaker_encoder/convert.py`, which produces the CoreML twin from the
same checkpoint and wrote the goldens this script validates against.

Architecture (from the checkpoint's own `model_config`):
    TFMelBanks (72 mels, 16 kHz, hamming, hop 160) -> ReDimNet2 backbone
    (C=20, F=72, 6 stages) -> ASTP attentive-statistics pooling -> BatchNorm
    -> Linear 2880->192

    audio[1, 32080]  ->  embedding[1, 192], ||v||2 = 1

Precision: whole-graph fp32, and that is a decision rather than a default
-------------------------------------------------------------------------
The iOS CoreML build is fp16 with the mel frontend held at fp32, because plain
fp16 OVERFLOWS on ordinary speech: `NormalizeAudio` rescales the waveform to
unit std, the power spectrum reaches 6.4e4 and the 256-bin filterbank sums past
fp16's 65504 ceiling; `log(inf)` then poisons that bin's mean into NaN. Across
the iOS fixtures the mel tensor spans 7.96e-9 to 7.10e5 -- 8.9e13 of dynamic
range against the 1.1e12 fp16 offers counting subnormals, i.e. 81x too wide, so
no scale factor fixes it.

ONNX has no per-op precision policy the way coremltools does: `to_fp16.py`
casts the whole graph. The whole graph at fp32 is 15.8 MB against the ~8.4 MB a
mixed build would be -- 7 MB of CDN for a model that is one prediction per 2 s
of speech. Paying it is cheaper than reproducing coremltools'
`FP16ComputePrecision(op_selector=...)` boundary in ONNX and then having to
prove the boundary landed where it was meant to.

Fixed window, never padded
--------------------------
The graph is exported at a FIXED [1, 32080] input. This network is
length-sensitive in four places (waveform mean/std normalisation, per-mel-bin
mean subtraction, softmax over time in each transformer block, ASTP pooling's
global context) and nothing upstream masks, so a padded window would dilute
every one of them. Spans are tiled into whole windows and the caller averages
the per-window vectors; a span shorter than one window scores nothing.
"""

import hashlib
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
REPO_DIR = os.path.join(CHECKPOINTS_DIR, "redimnet2")

# The iOS fixture tree, which carries both the LibriSpeech clips and the torch
# goldens `Scripts/speaker_encoder/convert.py` wrote from this same checkpoint.
FIXTURES_DIR = os.environ.get(
    "FERMENT_ENGINE_FIXTURES_DIR",
    os.path.join(
        SCRIPT_DIR,
        "../../../../ferment-cuts-ios/Packages/FermentEngine/Tests/Fixtures",
    ),
)

REPO_URL = "https://github.com/PalabraAI/redimnet2.git"
REPO_COMMIT = "cdc875670034dd7068013ca2ab21ec083a040ff8"

WEIGHTS_URL = (
    "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b2-vox2-lm.pt"
)
WEIGHTS_SHA256 = "0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a"
WEIGHTS_FILE = os.path.join(CHECKPOINTS_DIR, "b2-vox2-lm.pt")

MODEL_NAME = "redimnet_speaker"
ONNX_PATH = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.onnx")

# Graph constants, every one re-checked against the checkpoint's own
# `model_config` in `load_model` so a checkpoint swap cannot convert silently
# into a graph that no longer matches its weights.
SAMPLE_RATE = 16_000
HOP_LENGTH = 160
WIN_LENGTH = 400  # 25 ms
N_FFT = 512
N_MELS = 72
EMBED_DIM = 192

# The scoring window, in FRAMES first: the backbone truncates its input to a
# multiple of `time_stride` (4) before reshaping, so a window whose frame count
# is not a multiple of 4 silently discards its own tail.
#
# `SpectralFeaturesTF` frames with `conv1d(stride=HOP, padding=HOP//2)`:
#   frames = floor((samples + 2*(HOP//2) - WIN_LENGTH) / HOP) + 1
#          = floor((samples - 240) / 160) + 1
WINDOW_FRAMES = 200  # 4 | 200
WINDOW_SAMPLES = (WINDOW_FRAMES - 1) * HOP_LENGTH + 240  # 32_080, 2.005 s
TIME_STRIDE = 4

OPSET = 17

GOLDEN_STEMS = ["speaker-a-1", "speaker-a-2", "speaker-b-1", "speaker-b-2"]

# The bar the iOS port holds itself to and the Rust oracle re-asserts.
PARITY_BAR = 0.999


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights() -> str:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Fetching {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
    digest = sha256(WEIGHTS_FILE)
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"weights sha256 {digest}, expected {WEIGHTS_SHA256}")
    print(
        f"Checkpoint: b2-vox2-lm.pt ({os.path.getsize(WEIGHTS_FILE):,} bytes, sha256 ok)"
    )
    return WEIGHTS_FILE


def clone_architecture() -> None:
    """Vendor the upstream model code at a pinned commit. The wrapper below
    traces these live modules, so this IS the weight mapping."""
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print(f"Cloning {REPO_URL}")
        subprocess.run(
            ["git", "clone", REPO_URL, REPO_DIR], check=True, capture_output=True
        )
    subprocess.run(
        ["git", "-C", REPO_DIR, "checkout", "--quiet", REPO_COMMIT],
        check=True,
        capture_output=True,
    )
    sys.path.insert(0, REPO_DIR)


def load_model():
    from redimnet2.redimnet2 import ReDimNet2Wrap

    state = torch.load(download_weights(), map_location="cpu", weights_only=False)
    config = state["model_config"]

    model = ReDimNet2Wrap(**config)
    model.load_state_dict(state["state_dict"])
    model.eval()

    spec = model.spec.torchfbank[2]
    checks = [
        ("F (n_mels)", config["F"], N_MELS),
        ("hop_length", config["hop_length"], HOP_LENGTH),
        ("embed_dim", config["embed_dim"], EMBED_DIM),
        ("feat_type", config["feat_type"], "tf"),
        ("pooling_func", config["pooling_func"], "ASTP"),
        ("norm_signal", config["spec_params"]["norm_signal"], True),
        ("do_preemph", config["spec_params"]["do_preemph"], True),
        ("do_spec_aug", config["spec_params"]["do_spec_aug"], False),
        ("frontend frame_length", spec.length, WIN_LENGTH),
        ("frontend frame_step", spec.shift, HOP_LENGTH),
        ("frontend nfft", spec.nfft, N_FFT),
        ("frontend samplerate", spec.samplerate, SAMPLE_RATE),
        ("frontend num_bins", spec.num_bins, N_MELS),
        ("backbone time_stride", model.backbone.time_stride, TIME_STRIDE),
        ("pad_right_samples", model.pad_right_samples, None),
        ("before_pool_offset", model.before_pool_offset, None),
        ("emb_bn", model.bn2, None),
    ]
    for name, got, want in checks:
        if got != want:
            raise RuntimeError(f"checkpoint {name}={got!r}, script assumes {want!r}")

    frames = (WINDOW_SAMPLES + 2 * (HOP_LENGTH // 2) - WIN_LENGTH) // HOP_LENGTH + 1
    if frames != WINDOW_FRAMES or WINDOW_FRAMES % TIME_STRIDE:
        raise RuntimeError(
            f"window yields {frames} frames, want {WINDOW_FRAMES} "
            f"divisible by {TIME_STRIDE}"
        )

    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"Restored ReDimNet2-B2: {parameters:,} params, embedding {EMBED_DIM}-d, "
        f"window {WINDOW_FRAMES} frames ({WINDOW_SAMPLES} samples, "
        f"{WINDOW_SAMPLES / SAMPLE_RATE:.3f} s)"
    )
    return model


class SpeakerEncoderExport(nn.Module):
    """The upstream model plus the L2 normalisation.

    Normalising inside the graph makes unit length a property of the model
    rather than of whoever calls it: cosine similarity is then a dot product,
    with no second place that could forget to normalise."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        embedding = self.model(audio)
        return embedding / embedding.norm(dim=1, keepdim=True)


def assert_no_fft(path: str) -> None:
    """The frontend must stay two fixed real matmuls. `feat_type='tf'` gives
    that; `feat_type='pt'` would route through torchaudio's MelSpectrogram,
    i.e. a real STFT, whose portable-EP support is not something we want to
    discover on a user's machine."""
    import onnx

    model = onnx.load(path)
    banned = {"fft", "rfft", "irfft", "stft", "complex"}
    found = sorted(
        {
            node.op_type
            for node in model.graph.node
            if any(b in node.op_type.lower() for b in banned)
        }
    )
    if found:
        raise RuntimeError(
            f"graph contains spectral ops {found} — frontend is not conv-based; "
            f"check model_config['feat_type']"
        )
    domains = {node.domain for node in model.graph.node}
    if domains - {"", "ai.onnx"}:
        raise RuntimeError(f"graph uses non-portable domains {domains}")


def export(model) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    net = SpeakerEncoderExport(model).eval()
    example = torch.randn(1, WINDOW_SAMPLES) * 0.05

    with torch.no_grad():
        embedding = net(example)
    if tuple(embedding.shape) != (1, EMBED_DIM):
        raise RuntimeError(f"traced embedding shape {tuple(embedding.shape)}")

    print(f"Exporting ONNX opset {OPSET} (audio [1, {WINDOW_SAMPLES}] fp32)...")
    torch.onnx.export(
        net,
        (example,),
        ONNX_PATH,
        input_names=["audio"],
        output_names=["embedding"],
        opset_version=OPSET,
        dynamo=False,
    )
    assert_no_fft(ONNX_PATH)
    print(f"  {ONNX_PATH}  {os.path.getsize(ONNX_PATH):,} bytes")
    print(f"  sha256 {sha256(ONNX_PATH)}")


def torch_embedding(model, window: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        vector = model(torch.from_numpy(window).unsqueeze(0)).numpy().reshape(-1)
    return vector / np.linalg.norm(vector)


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"{path} is {rate} Hz")
    return np.ascontiguousarray(audio)


def window_offsets(sample_count: int) -> list:
    """Whole, non-overlapping windows from the head of a span — the same tiling
    the Rust `redimnet::windows` performs."""
    return [i * WINDOW_SAMPLES for i in range(sample_count // WINDOW_SAMPLES)]


def validate(model) -> None:
    """Three-way: ONNX-through-onnxruntime against torch, and both against the
    committed iOS goldens (which are torch outputs of this same checkpoint,
    written by a different toolchain in a different repository)."""
    import onnxruntime as ort

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print(f"\n=== parity: onnxruntime vs torch vs iOS goldens (bar {PARITY_BAR}) ===")
    print(f"{'fixture':<14} {'windows':>7} {'min cos ORT|torch':>18} "
          f"{'min cos ORT|golden':>19} {'max |‖v‖-1|':>12}")

    failures = []
    for stem in GOLDEN_STEMS:
        wav = os.path.join(FIXTURES_DIR, "SpeakerEncoder", f"{stem}.wav")
        binary = os.path.join(FIXTURES_DIR, "SpeakerEncoder", f"{stem}.embedding.bin")
        meta_path = os.path.join(FIXTURES_DIR, "SpeakerEncoder", f"{stem}.embedding.json")
        if not (os.path.exists(wav) and os.path.exists(binary)):
            print(f"  SKIP {stem} (fixture absent)")
            continue

        with open(meta_path) as handle:
            meta = json.load(handle)
        audio = read_wav(wav)
        offsets = window_offsets(len(audio))
        if offsets != meta["offsets"]:
            failures.append(f"{stem}: offsets {offsets} != golden {meta['offsets']}")
        golden = np.fromfile(binary, dtype="<f4").reshape(meta["windows"], EMBED_DIM)

        worst_torch, worst_golden, worst_norm = 1.0, 1.0, 0.0
        for index, offset in enumerate(offsets):
            window = np.ascontiguousarray(audio[offset : offset + WINDOW_SAMPLES])
            reference = torch_embedding(model, window)
            actual = np.asarray(
                session.run(None, {"audio": window.reshape(1, -1)})[0]
            ).reshape(-1)
            if not np.isfinite(actual).all():
                failures.append(f"{stem}@{offset}: non-finite")
                continue
            worst_norm = max(worst_norm, abs(float(np.linalg.norm(actual)) - 1.0))
            unit = actual / np.linalg.norm(actual)
            worst_torch = min(worst_torch, float(reference @ unit))
            worst_golden = min(worst_golden, float(golden[index] @ unit))

        if min(worst_torch, worst_golden) < PARITY_BAR:
            failures.append(
                f"{stem}: torch {worst_torch:.6f} golden {worst_golden:.6f}"
            )
        print(f"  {stem:<14} {len(offsets):>7} {worst_torch:>18.6f} "
              f"{worst_golden:>19.6f} {worst_norm:>12.2e}")

    if failures:
        raise RuntimeError("parity failures: " + "; ".join(failures))


def main() -> None:
    clone_architecture()
    model = load_model()
    export(model)
    validate(model)
    print(f"\nDone. {ONNX_PATH}")


if __name__ == "__main__":
    main()
