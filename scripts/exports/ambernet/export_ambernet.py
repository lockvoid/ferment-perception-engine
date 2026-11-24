"""NVIDIA AmberNet LangID (`langid_ambernet`, NGC v1.12.0) -> ONNX,
with the mel frontend baked into the graph.

Produces, in `artifacts/`:
  ambernet_langid.onnx     audio[1, 480000] f32, num_frames[1] i32
                             -> logits[1, 107] f32, embedding[1, 512] f32
  ambernet_langs.json      the 107 VoxLingua107 tags, in the checkpoint's
                           own label order (copied into the Rust crate as a
                           committed resource -- see `ambernet/langs.rs`)

Provenance
----------
Mirrors `ferment-cuts-ios/Packages/FermentEngine/Scripts/language_id/convert.py`
(commit 88ac816ec), which produces the same graph for Core ML. Everything
below -- the 107 labels, the mel parameters, the Hann window and the librosa
filterbank -- is read out of the checkpoint rather than transcribed from a
model card.

The checkpoint is NOT on HuggingFace: `nvidia/langid_ambernet` is an NGC model
(catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet) and
huggingface.co returns 401 for that path. This script pulls `ambernet.nemo`
(v1.12.0, 116,049,920 bytes) from the NGC public API and verifies its sha256.
License: NGC Terms of Use (recorded in `engine-models/fleet.rs`).

Architecture (from the checkpoint's own `model_config.yaml`):
    AudioToMelSpectrogramPreprocessor -> ConvASREncoder (5 Jasper blocks,
    depthwise-separable + squeeze-excite, all stride 1) -> SpeakerDecoder
    (x-vector stats pooling -> Linear 6144->512 -> BN -> ReLU -> Linear 512->107)

Why a `num_frames` input instead of a fixed full window: language ID runs on
whatever speech the caller found, which is usually less than the 30 s window.
Every length-sensitive step in this network -- the per-feature normalisation,
each squeeze-excite pooling, and the final x-vector stats pooling -- must see
the true frame count or a short clip gets its statistics diluted by the zero
padding. All strides are 1, so ONE frame count is correct for the whole graph.

Usage:
    cd scripts/exports/ambernet
    uv venv --python 3.11 .venv
    VIRTUAL_ENV=$PWD/.venv uv pip install "nemo_toolkit[asr]" "torch>=2.0,<2.8" \
        soundfile librosa "numpy<2" "onnx>=1.16,<1.20" "onnxruntime>=1.18"
    .venv/bin/python export_ambernet.py

Parity validation against NeMo eager is mandatory and always runs.
"""

import hashlib
import json
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

NGC_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/"
    "langid_ambernet/versions/1.12.0/files/ambernet.nemo"
)
NGC_SHA256 = "2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681"
NEMO_FILE = os.path.join(CHECKPOINTS_DIR, "ambernet.nemo")

SAMPLE_RATE = 16_000
WIN_LENGTH = 400        # 25 ms
HOP_LENGTH = 160        # 10 ms
N_FFT = 512
N_FREQ = N_FFT // 2 + 1  # 257
N_MELS = 80
PREEMPH = 0.97
LOG_ZERO_GUARD = 2.0 ** -24

# The mel energies and the log guard are scaled by this before the log. It is
# exactly neutral: log(S*mel + S*guard) = log(mel + guard) + log(S), and the
# per-feature normalisation that follows subtracts a per-bin mean, so the
# constant log(S) cancels to the last bit.
#
# It exists because NeMo's guard (2^-24 ~ 5.96e-8) is *precisely* the smallest
# fp16 subnormal, which an fp16 backend flushes to zero -> log(0) = -inf ->
# NaN. Scaling by 2^10 lifts it to the smallest fp16 *normal*. This export is
# fp32, where the scale is a no-op; it is kept so this graph and the iOS Core
# ML graph are the same graph, and so an fp16 conversion stays available.
MEL_SCALE = 2.0 ** 10
NORM_EPS = 1e-5         # NeMo's CONSTANT, added to the per-feature std
PAD_TO = 16             # NeMo pads the time axis to a multiple of this
NUM_CLASSES = 107
EMB_SIZE = 512

WINDOW_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS            # 480_000
# NeMo >=2 `get_seq_len`: floor((n + 2*(n_fft//2) - n_fft) / hop) = floor(n/hop).
MAX_FRAMES = N_SAMPLES // HOP_LENGTH                # 3_000
PADDED_FRAMES = MAX_FRAMES + (-MAX_FRAMES % PAD_TO)  # 3_008

MODEL_NAME = "ambernet_langid"


def download_checkpoint() -> str:
    """Fetch `ambernet.nemo` from the NGC public model registry."""
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    if os.path.exists(NEMO_FILE):
        print(f"Checkpoint present: {NEMO_FILE}")
        return NEMO_FILE
    print(f"Fetching {NGC_URL}")
    urllib.request.urlretrieve(NGC_URL, NEMO_FILE)
    digest = hashlib.sha256(open(NEMO_FILE, "rb").read()).hexdigest()
    if digest != NGC_SHA256:
        os.remove(NEMO_FILE)
        raise RuntimeError(f"checkpoint sha256 {digest}, expected {NGC_SHA256}")
    print(f"  wrote {NEMO_FILE} ({os.path.getsize(NEMO_FILE)} bytes, sha256 ok)")
    return NEMO_FILE


def _make_seq_mask_like(like, lengths, valid_ones=True, time_dim=-1):
    """Un-scripted twin of `nemo...tdnn_attention.make_seq_mask_like`.

    The original carries @torch.jit.script_if_tracing, which keeps it a
    scripted subgraph instead of being traced into plain ops. Same math,
    traced eagerly. Only `StatsPoolLayer` calls it.
    """
    mask = (
        torch.arange(like.shape[time_dim], device=like.device)
        .repeat(lengths.shape[0], 1)
        .lt(lengths.unsqueeze(-1))
    )
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    if time_dim != -1:
        mask = mask.transpose(time_dim, -1)
    if not valid_ones:
        mask = ~mask
    return mask


def load_nemo_model():
    """Restore the NeMo model and assert the config still matches every
    constant this script bakes into the graph."""
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules import tdnn_attention
    from nemo.core.classes.common import typecheck

    typecheck.set_typecheck_enabled(False)
    tdnn_attention.make_seq_mask_like = _make_seq_mask_like

    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
        download_checkpoint(), map_location="cpu"
    )
    model.eval()

    cfg = model.cfg.preprocessor
    checks = [
        ("sample_rate", cfg.sample_rate, SAMPLE_RATE),
        ("window_size", cfg.window_size, WIN_LENGTH / SAMPLE_RATE),
        ("window_stride", cfg.window_stride, HOP_LENGTH / SAMPLE_RATE),
        ("n_fft", cfg.n_fft, N_FFT),
        ("features", cfg.features, N_MELS),
        ("normalize", cfg.normalize, "per_feature"),
        ("window", cfg.window, "hann"),
        ("num_classes", model.cfg.decoder.num_classes, NUM_CLASSES),
        ("emb_sizes", model.cfg.decoder.emb_sizes, EMB_SIZE),
        ("pool_mode", model.cfg.decoder.pool_mode, "xvector"),
    ]
    for name, got, want in checks:
        if got != want:
            raise RuntimeError(f"checkpoint {name}={got!r}, script assumes {want!r}")

    labels = list(model.cfg.train_ds.labels)
    if len(labels) != NUM_CLASSES:
        raise RuntimeError(f"expected {NUM_CLASSES} labels, got {len(labels)}")
    return model, labels


class MelFrontend(nn.Module):
    """NeMo's `AudioToMelSpectrogramPreprocessor`, rebuilt out of exportable
    ops. The Hann window and the mel filterbank are the checkpoint's own
    buffers, not recomputed here."""

    def __init__(self, window: torch.Tensor, filterbank: torch.Tensor):
        super().__init__()
        # torch.stft zero-pads a short win_length up to n_fft *centred*.
        left = (N_FFT - WIN_LENGTH) // 2
        self.register_buffer(
            "window", F.pad(window, (left, N_FFT - WIN_LENGTH - left))
        )  # [512]

        n = torch.arange(N_FFT).float()
        k = torch.arange(N_FREQ).float()
        angle = 2.0 * np.pi * k[:, None] * n[None, :] / N_FFT
        self.register_buffer("cos_basis", torch.cos(angle))  # [257, 512]
        self.register_buffer("sin_basis", torch.sin(angle))
        self.register_buffer("mel_fb", filterbank[0] * MEL_SCALE)  # [80, 257]

    def forward(self, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # audio: [1, N_SAMPLES]; mask: [1, 1, PADDED_FRAMES], 1.0 = valid.

        # 1. Pre-emphasis on the waveform (NeMo applies it before the STFT).
        x = torch.cat(
            (audio[:, :1], audio[:, 1:] - PREEMPH * audio[:, :-1]), dim=1
        )

        # 2. Centre padding. NeMo >=2 passes pad_mode="constant", so zeros.
        x = F.pad(x, (N_FFT // 2, N_FFT // 2))

        # 3. Frame extraction via im2col.
        frames = F.unfold(
            x.unsqueeze(1).unsqueeze(2),
            kernel_size=(1, N_FFT),
            stride=(1, HOP_LENGTH),
        )                                                  # [1, 512, T]
        frames = frames * self.window[None, :, None]

        # 4. DFT as two fixed real matmuls.
        real = torch.matmul(self.cos_basis, frames)        # [1, 257, T]
        imag = -torch.matmul(self.sin_basis, frames)

        # 5. Magnitude then power. NeMo takes sqrt and squares it back
        #    (mag_power 2.0); keeping both steps preserves its rounding.
        magnitude = torch.sqrt(real * real + imag * imag)
        power = magnitude.pow(2.0)

        # 6. Mel projection + log. Both energies and guard carry MEL_SCALE;
        #    the resulting constant log offset cancels in step 8.
        mel = torch.matmul(self.mel_fb, power)             # [1, 80, T]
        log_mel = torch.log(mel + LOG_ZERO_GUARD * MEL_SCALE)

        # 7. Pad the time axis out to the graph's fixed width.
        log_mel = F.pad(log_mel, (0, PADDED_FRAMES - log_mel.shape[-1]))

        # 8. Per-feature normalisation over valid frames only. NeMo uses the
        #    unbiased (n-1) variance here.
        count = mask.sum(dim=-1, keepdim=True)             # [1, 1, 1]
        mean = (log_mel * mask).sum(dim=-1, keepdim=True) / count
        var = ((log_mel - mean) * mask).pow(2).sum(dim=-1, keepdim=True) / (count - 1.0)
        std = torch.sqrt(var)
        normalised = (log_mel - mean) / (std + NORM_EPS)

        # 9. Padded frames carry no signal into the encoder.
        return normalised * mask


class AmberNetExport(nn.Module):
    """Frontend + NeMo's own encoder/decoder, in one exportable graph."""

    def __init__(self, model):
        super().__init__()
        featurizer = model.preprocessor.featurizer
        self.frontend = MelFrontend(
            featurizer.window.detach().clone(), featurizer.fb.detach().clone()
        )
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, audio: torch.Tensor, num_frames: torch.Tensor):
        index = torch.arange(PADDED_FRAMES, device=audio.device).unsqueeze(0)
        mask = (index < num_frames.unsqueeze(1)).float().unsqueeze(1)
        feats = self.frontend(audio, mask)
        encoded, encoded_len = self.encoder(audio_signal=feats, length=num_frames)
        logits, embedding = self.decoder(encoder_output=encoded, length=encoded_len)
        return logits, embedding


def reference_logits(model, audio: torch.Tensor, num_frames: int):
    """NeMo eager, through its own preprocessor -- the parity oracle."""
    valid_samples = num_frames * HOP_LENGTH
    with torch.no_grad():
        length = torch.tensor([valid_samples], dtype=torch.int64)
        feats, feats_len = model.preprocessor(
            input_signal=audio[:, :valid_samples], length=length
        )
        encoded, encoded_len = model.encoder(audio_signal=feats, length=feats_len)
        logits, embedding = model.decoder(encoder_output=encoded, length=encoded_len)
    return logits, embedding


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model, labels = load_nemo_model()
    print(f"Loaded checkpoint: {len(labels)} labels, first 8 = {labels[:8]}")

    langs_path = os.path.join(ARTIFACTS_DIR, "ambernet_langs.json")
    with open(langs_path, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False)
    print(f"  langs -> {langs_path}")

    net = AmberNetExport(model).eval()

    torch.manual_seed(0)
    audio = torch.zeros(1, N_SAMPLES)
    audio[0, : SAMPLE_RATE * 5] = torch.randn(SAMPLE_RATE * 5) * 0.1
    frames = torch.tensor([SAMPLE_RATE * 5 // HOP_LENGTH], dtype=torch.int32)

    onnx_path = os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.onnx")
    print(f"Exporting ONNX (audio [1, {N_SAMPLES}], num_frames int32)...")
    with torch.no_grad():
        torch.onnx.export(
            net,
            (audio, frames),
            onnx_path,
            input_names=["audio", "num_frames"],
            output_names=["logits", "embedding"],
            opset_version=17,
            dynamo=False,
        )
    print(f"  onnx -> {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    print("\n[parity]")
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    parity = {}
    for label, valid_seconds in (("5 s window", 5), ("full 30 s window", 30)):
        clip = torch.zeros(1, N_SAMPLES)
        n_valid = SAMPLE_RATE * valid_seconds
        clip[0, :n_valid] = torch.randn(n_valid) * 0.1
        n_frames = n_valid // HOP_LENGTH
        frames_t = torch.tensor([n_frames], dtype=torch.int32)

        ref_logits, ref_emb = reference_logits(model, clip, n_frames)
        with torch.no_grad():
            pt_logits, pt_emb = net(clip, frames_t)
        onnx_logits, onnx_emb = session.run(
            None, {"audio": clip.numpy(), "num_frames": frames_t.numpy()}
        )

        def report(name, a, b):
            delta = float(np.abs(np.asarray(a, dtype=np.float64)
                                 - np.asarray(b, dtype=np.float64)).max())
            print(f"    {name:<40} max|delta| = {delta:.6g}")
            parity[name] = delta
            return delta

        report(f"{label}: logits nemo<->graph", ref_logits.numpy(), pt_logits.numpy())
        report(f"{label}: logits graph<->onnx", pt_logits.numpy(), onnx_logits)
        report(f"{label}: embedding graph<->onnx", pt_emb.numpy(), onnx_emb)
        assert int(np.argmax(ref_logits.numpy())) == int(np.argmax(onnx_logits)), \
            f"{label}: top-1 disagreement between NeMo and ONNX"

    with open(os.path.join(ARTIFACTS_DIR, "export_summary.json"), "w") as fh:
        json.dump({
            "source": NGC_URL,
            "sha256": NGC_SHA256,
            "labels": len(labels),
            "parity": parity,
            "geometry": {
                "window_samples": N_SAMPLES,
                "hop_length": HOP_LENGTH,
                "max_frames": MAX_FRAMES,
                "padded_frames": PADDED_FRAMES,
            },
        }, fh, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
