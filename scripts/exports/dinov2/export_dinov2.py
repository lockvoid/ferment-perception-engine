"""DINOv2 ViT-S/14 -> ONNX: the visual semantic encoder behind the scene
detector's flash veto and per-scene embeddings.

Mirrors the iOS converter (`ferment-cuts-ios`
`Packages/FermentEngine/Scripts/visual_semantic_encoder/convert.py`) decision
for decision, so the two platforms produce vectors in the SAME space and the
measured shot-identity distances (0.72 / 0.80) transfer:

  * the ViT-S/14 forward is written out statically, with the positional grid
    interpolated ONCE at export for the fixed token count -- no shape
    arithmetic in the graph;
  * the input is 224x126 (16x9 patches at /14 -- the analysis aspect). Squashing
    to a square cost a third of the measured same-shot/other-shot gap upstream,
    so portrait frames are rotated to landscape by the caller instead;
  * ImageNet normalisation rides INSIDE the graph, so the runtime hands over
    plain 0..255 RGB;
  * the output is the CLS token, L2-normalised (384 floats).

One deliberate quirk is inherited on purpose: CoreML's `ImageType` accepts a
single scalar `scale` with a per-channel `bias`, so iOS normalises with
`scale = mean(1/(255*std))` for all three channels rather than the exact
per-channel gain. Reproducing the exact per-channel gain here would put Studio
embeddings in a slightly different space from iOS ones, and both land in the
same server-side column. `--exact-imagenet-norm` exports the textbook variant
for the drift measurement in `measure_drift.py`; the shipping master does not
use it.

Weights: torch.hub `facebookresearch/dinov2`, Apache-2.0.

    uv run --project scripts/exports/dinov2 python \
        scripts/exports/dinov2/export_dinov2.py \
        --out scripts/exports/dinov2/artifacts/dinov2_vits14.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

H, W = 126, 224
PATCH = 14
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OPSET = 17


class DinoIdentity(torch.nn.Module):
    """The ViT-S/14 forward with the fixed grid: 0..255 RGB -> ImageNet
    normalisation -> patch conv -> cls + baked positional embedding -> 12
    pre-norm blocks (SDPA attention, layer scale, MLP) -> norm -> normalised
    CLS. Static shapes only."""

    def __init__(self, backbone: torch.nn.Module, exact_norm: bool):
        super().__init__()
        b = backbone
        self.patch = b.patch_embed.proj
        self.cls = torch.nn.Parameter(b.cls_token.detach().clone(), requires_grad=False)
        dim = b.pos_embed.shape[-1]
        tokens = 1 + (H // PATCH) * (W // PATCH)
        # The backbone's own interpolation for this grid, computed once.
        with torch.no_grad():
            pos = b.interpolate_pos_encoding(torch.zeros(1, tokens, dim), H, W)
        self.pos = torch.nn.Parameter(pos.detach().clone(), requires_grad=False)
        self.blocks = torch.nn.ModuleList(b.blocks)
        self.norm = b.norm
        self.heads = b.blocks[0].attn.num_heads
        self.dim = dim
        self.tokens = tokens

        per_channel = 1.0 / (255.0 * np.array(STD))
        scale = per_channel if exact_norm else np.full(3, float(per_channel.mean()))
        self.register_buffer(
            "in_scale", torch.tensor(scale, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "in_bias",
            torch.tensor(-np.array(MEAN) / np.array(STD), dtype=torch.float32).view(1, 3, 1, 1),
        )

    def attention(self, attn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qkv = (
            attn.qkv(x)
            .reshape(1, self.tokens, 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(1, self.tokens, self.dim)
        return attn.proj(out)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image * self.in_scale + self.in_bias
        x = self.patch(x).flatten(2).transpose(1, 2)  # [1, N, D]
        x = torch.cat([self.cls, x], dim=1) + self.pos
        for block in self.blocks:
            x = x + block.ls1(self.attention(block.attn, block.norm1(x)))
            x = x + block.ls2(block.mlp(block.norm2(x)))
        x = self.norm(x)
        cls = x[:, 0]
        return cls / cls.norm(dim=-1, keepdim=True)


# `torch.hub.load("facebookresearch/dinov2", ...)` pulls the repo through the
# GitHub API, which rate-limits (HTTP 403) from a shared address. The pinned
# clone under `repo/` is the same code, fetched over plain git; the weights
# still come from `dl.fbaipublicfiles.com` via the repo's own hubconf.
REPO = Path(__file__).resolve().parent / "repo"
REPO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def load_identity(exact_norm: bool = False) -> DinoIdentity:
    """The exported module, ready to run. Shared with `gen_goldens.py` and
    `measure_drift.py` so no second copy of the forward can drift."""
    if not REPO.is_dir():
        raise SystemExit(
            f"clone the weights repo first:\n"
            f"  git clone --depth 1 https://github.com/facebookresearch/dinov2 {REPO}\n"
            f"  (pinned at {REPO_COMMIT})"
        )
    torch.hub.set_dir(os.environ.get("TORCH_HUB", "/tmp/torchhub"))
    backbone = torch.hub.load(
        REPO.as_posix(), "dinov2_vits14", source="local", verbose=False
    ).eval()
    return DinoIdentity(backbone, exact_norm).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exact-imagenet-norm",
        action="store_true",
        help="per-channel gain instead of iOS's CoreML scalar-mean gain (drift measurement only)",
    )
    args = parser.parse_args()

    model = load_identity(args.exact_imagenet_norm)

    # 0..255 RGB, the surface the Rust side hands over.
    example = torch.rand(1, 3, H, W) * 255.0
    with torch.no_grad():
        reference = model(example).numpy()
    print("torch reference:", reference.shape, "norm", float(np.linalg.norm(reference)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.out.as_posix(),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=OPSET,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(args.out.as_posix(), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": example.numpy()})[0]
    drift = float(np.abs(onnx_out - reference).max())
    print(f"onnx parity max|delta| vs torch: {drift:.3e}")
    if drift > 1e-4:
        raise SystemExit(f"ONNX export drifted from torch by {drift:.3e}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
