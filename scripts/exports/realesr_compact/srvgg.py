"""SRVGGNetCompact — the Real-ESRGAN "Compact" super-resolution network.

Reimplemented here (BSD-3-Clause, matching upstream Real-ESRGAN) so the export
does not drag in the heavy, pinned `basicsr`/`realesrgan` dependency tree. The
architecture is deliberately byte-identical to xinntao/Real-ESRGAN's
`realesrgan/archs/srvgg_arch.py` so official checkpoints load with strict=True.

It is pure conv + PReLU + PixelShuffle + a nearest-neighbour residual — no
grid_sample, no deformable conv — i.e. it clears migration 016's "golden rule"
and ports to DirectML / CUDA / WebGPU / QNN.
"""

from __future__ import annotations

from torch import nn
from torch.nn import functional as F


def _activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"unsupported activation {act_type!r}")


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # first conv + activation
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(_activation(act_type, num_feat))
        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(_activation(act_type, num_feat))
        # last conv projects to (out_ch * upscale^2) channels for the shuffle
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # residual: add the nearest-neighbour upsample of the input
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# Official checkpoint configurations (xinntao/Real-ESRGAN inference scripts).
CONFIGS = {
    "realesr-general-x4v3": dict(num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
    # Denoise-folding companion checkpoint to general-x4v3 (same architecture).
    "realesr-general-wdn-x4v3": dict(num_feat=64, num_conv=32, upscale=4, act_type="prelu"),
    "realesr-animevideov3": dict(num_feat=64, num_conv=16, upscale=4, act_type="prelu"),
}
