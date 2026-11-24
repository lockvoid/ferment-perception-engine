"""Run the engine-rust parakeet export with two export shims:

1. Legacy TorchScript ONNX exporter forced (torch 2.9's dynamo exporter
   rejects NeMo's typed-method call convention).
2. Export-safe STFT: ONNX cannot express complex tensors, so the featurizer's
   `stft` (complex output) is replaced with a matmul real-DFT returning the
   stacked real/imag pair `[B, N, T, 2]`, and `torch.view_as_real` becomes a
   passthrough. Every other NeMo op (preemphasis, mel filterbank, log,
   per-feature normalization) runs unmodified, and the patched pipeline is
   parity-gated against the untouched NeMo preprocessor before export.
"""

import math
import pathlib
import sys

import torch

# ---- shim 1: legacy exporter ------------------------------------------------
_original_export = torch.onnx.export


def _legacy_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_export(*args, **kwargs)


torch.onnx.export = _legacy_export

# ---- shim 2: export-safe STFT ----------------------------------------------


class RealStft(torch.nn.Module):
    """torch.stft(center=True, pad_mode="constant", onesided, unnormalized)
    as a strided conv1d against windowed DFT bases, emitting `[B, n_bins, T, 2]`.

    conv1d rather than unfold+matmul: the TorchScript exporter cannot emit
    Unfold over a dynamic time axis ("input size not accessible"), while Conv
    with symmetric padding traces cleanly on every EP."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # torch.stft zero-pads a short window symmetrically to n_fft.
        padded_window = torch.zeros(n_fft, dtype=torch.float32)
        left = (n_fft - win_length) // 2
        padded_window[left : left + win_length] = window.to(torch.float32)
        bins = n_fft // 2 + 1
        position = torch.arange(n_fft, dtype=torch.float32)
        frequency = torch.arange(bins, dtype=torch.float32)
        angle = 2.0 * math.pi * frequency[:, None] * position[None, :] / n_fft
        # Fold the window into the kernels; conv1d(x, k)[b,f,t] is exactly the
        # windowed frame·basis sum. Kernel layout [bins, 1, n_fft].
        self.register_buffer("cos_kernel", (torch.cos(angle) * padded_window).unsqueeze(1))
        self.register_buffer("sin_kernel", (-torch.sin(angle) * padded_window).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.n_fft // 2
        x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="constant")  # [B,1,L+2p]
        real = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_length)
        imaginary = torch.nn.functional.conv1d(x, self.sin_kernel, stride=self.hop_length)
        # -> [B, bins, T, 2], the layout view_as_real(torch.stft(...)) yields.
        return torch.stack((real, imaginary), dim=-1)


def install_export_stft(featurizer) -> None:
    assert not featurizer.exact_pad, "exact_pad uses center=False; shim assumes center=True"
    # `stft` is a CLASS method on FilterbankFeatures; assigning an nn.Module to
    # `featurizer.stft` registers it in `_modules`, which class-attribute lookup
    # SHADOWS (the original method would keep winning, silently). Register the
    # module under a fresh name and shadow the method with a bound instance
    # attribute — functions are non-data descriptors, so the instance dict wins.
    import types

    featurizer.export_stft = RealStft(
        featurizer.n_fft,
        featurizer.hop_length,
        featurizer.win_length,
        featurizer.window,
    )
    featurizer.stft = types.MethodType(lambda self, x: self.export_stft(x), featurizer)
    # The featurizer's forward applies torch.view_as_real to the (formerly
    # complex) STFT output; our replacement already IS the real view.
    torch.view_as_real = lambda tensor: tensor


def parity_gate(model_loader) -> None:
    """Original complex-STFT pipeline vs the shimmed one, same real signal."""
    model = model_loader().eval()
    preprocessor = model.preprocessor
    torch.manual_seed(0)
    samples = 48_000
    tone = torch.sin(torch.linspace(0.0, 2_000.0, samples)) * 0.3
    noise = torch.randn(samples) * 0.05
    audio = (tone + noise).unsqueeze(0)
    length = torch.tensor([samples], dtype=torch.int32)

    with torch.no_grad():
        reference, reference_length = preprocessor(input_signal=audio, length=length)
    install_export_stft(preprocessor.featurizer)
    with torch.no_grad():
        shimmed, shimmed_length = preprocessor(input_signal=audio, length=length)
    assert torch.equal(reference_length, shimmed_length), "frame-length mismatch"
    worst = (reference - shimmed).abs().max().item()
    print(f"preprocessor parity (complex STFT vs export shim): max abs diff {worst:.3e}")
    assert worst < 1e-3, f"shimmed mel drifted: {worst}"


# ---- run the sibling-repo export with the shims -----------------------------
HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parents[3] / "ferment-engine-rust/scripts/export/parakeet.py"

from nemo.collections.asr.models import EncDecRNNTBPEModel  # noqa: E402

print("parity gate: loading model ...")
parity_gate(
    lambda: EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
)

# The sibling script loads the model itself; install the shim on that instance
# too by intercepting from_pretrained.
_original_from_pretrained = EncDecRNNTBPEModel.from_pretrained.__func__


def _patched_from_pretrained(cls, *args, **kwargs):
    model = _original_from_pretrained(cls, *args, **kwargs)
    install_export_stft(model.preprocessor.featurizer)
    return model


EncDecRNNTBPEModel.from_pretrained = classmethod(_patched_from_pretrained)

# ---- shim 3: NeMo typecheck off for tracing ---------------------------------
# The TorchScript tracer invokes forward positionally; NeMo's typecheck
# decorator rejects that ("kwargs only for typed methods"). NeMo's own export
# path disables typecheck the same way — the traced math is unchanged.
from nemo.core.classes.common import typecheck  # noqa: E402

typecheck.set_typecheck_enabled(enabled=False)

sys.argv = [str(SCRIPT), "--repo", str(HERE / "repo")]
code = compile(SCRIPT.read_text(), str(SCRIPT), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SCRIPT)})
