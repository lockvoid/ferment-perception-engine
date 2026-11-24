"""nvidia/nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1) -> three ONNX graphs
plus the two sidecars `crates/engine-processors/src/nemotron/` reads.

Produces, in `artifacts/`:

    nemotron_preprocessor.onnx   audio[1, N] f32, audio_length[1] i32
                                 -> mel[1, 128, T] f32, mel_length[1] i32
    nemotron_encoder.onnx        mel[1, 128, 65] f32, mel_length[1] i32,
                                 cache_channel[1, 24, L, 1024] f32,
                                 cache_time[1, 24, 1024, 8] f32,
                                 cache_len[1] i32, prompt_id[1] i32
                                 -> encoded[1, 1024, 7] f32, encoded_length[1] i32,
                                    cache_channel_out, cache_time_out, cache_len_out
    nemotron_decoder_joint.onnx  token[1, 1] i32, token_length[1] i32,
                                 h_in/c_in[2, 1, 640] f32, encoder[1, 1024, 1] f32
                                 -> logits[1, 1, 1, 13088] f32, h_out, c_out
    nemotron_vocab.json          {"<id>": "<piece>"}, 13 087 entries
    nemotron_metadata.json       the export's own account of its shapes, which
                                 `NemotronModels::load` checks before a chunk runs

Boundary
--------
The three graphs and every tensor NAME are iOS's
(`Processors/Nemotron/NemotronModels.swift`), because the Rust port is the
Swift processor's twin and the two must be able to disagree about numbers, not
about shapes. What iOS runs are FluidInference's Core ML conversions of the same
checkpoint; this is our own export of the checkpoint itself.

The 560 ms tier is `att_context_size` index 3 — `[56, 13]` in the checkpoint's
own list of four. `chunk_mel_frames` 56 / 8 = 7 encoder frames per chunk, and
13 encoder frames of right context are baked into the attention mask.

    ONE NUMBER IS NOT RECOVERABLE FROM THE CHECKPOINT
    iOS pins `cacheChannelShape = [1, 24, 42, 1024]` — a 42-frame left context,
    where this checkpoint's 560 ms row says 56. 42 is six chunks of seven; 56 is
    eight. That choice belongs to the FluidInference conversion iOS runs and is
    not in the weights, so it cannot be read off them. This script exports the
    CHECKPOINT's own 56 and writes it into the metadata sidecar; the Rust
    `config.rs` is pinned to iOS's 42 and will refuse the bundle until the lead
    rules which one both platforms take (report §7, ruling row N-1). Nothing
    here guesses.

Usage
-----
    uv run --project scripts/exports/nemotron python scripts/exports/nemotron/export.py \
        --nemo /path/to/nemotron-3.5-asr-streaming-0.6b.nemo
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tarfile
import types

import torch

# ---- shim 1: the legacy TorchScript exporter -------------------------------
# torch 2.7's dynamo exporter rejects NeMo's typed-method call convention, the
# same wall `scripts/exports/parakeet/run_export.py` hit.
_original_export = torch.onnx.export


def _legacy_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_export(*args, **kwargs)


torch.onnx.export = _legacy_export


# ---- shim 2: an export-safe STFT -------------------------------------------
# Verbatim from `scripts/exports/parakeet/run_export.py`: ONNX cannot express
# complex tensors, so the featurizer's `stft` (complex output) becomes a matmul
# real-DFT returning the stacked real/imag pair `[B, N, T, 2]`, and
# `torch.view_as_real` becomes a passthrough. Every other NeMo op runs
# unmodified, and the shim is parity-gated against the untouched preprocessor
# before anything is exported. Same featurizer class, same 512-point window;
# duplicated rather than imported because the two export environments are
# separate `uv` projects.


class RealStft(torch.nn.Module):
    """`torch.stft(center=True, pad_mode="constant", onesided, unnormalized)`
    as a strided conv1d against windowed DFT bases, emitting `[B, n_bins, T, 2]`."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, window: torch.Tensor):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        padded_window = torch.zeros(n_fft, dtype=torch.float32)
        left = (n_fft - win_length) // 2
        padded_window[left : left + win_length] = window.to(torch.float32)
        bins = n_fft // 2 + 1
        position = torch.arange(n_fft, dtype=torch.float32)
        frequency = torch.arange(bins, dtype=torch.float32)
        angle = 2.0 * math.pi * frequency[:, None] * position[None, :] / n_fft
        self.register_buffer("cos_kernel", (torch.cos(angle) * padded_window).unsqueeze(1))
        self.register_buffer("sin_kernel", (-torch.sin(angle) * padded_window).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.n_fft // 2
        x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="constant")
        real = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_length)
        imaginary = torch.nn.functional.conv1d(x, self.sin_kernel, stride=self.hop_length)
        return torch.stack((real, imaginary), dim=-1)


def install_export_stft(featurizer) -> None:
    assert not featurizer.exact_pad, "exact_pad uses center=False; shim assumes center=True"
    # `stft` is a CLASS method; assigning an nn.Module to `featurizer.stft`
    # registers it in `_modules`, which class-attribute lookup SHADOWS. Register
    # under a fresh name and shadow the method with a bound instance attribute.
    featurizer.export_stft = RealStft(
        featurizer.n_fft,
        featurizer.hop_length,
        featurizer.win_length,
        featurizer.window,
    )
    featurizer.stft = types.MethodType(lambda self, x: self.export_stft(x), featurizer)
    torch.view_as_real = lambda tensor: tensor


def parity_gate(model) -> None:
    """The untouched complex-STFT pipeline vs the shimmed one, same signal."""
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

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

# The 560 ms tier: index 2 of the checkpoint's `att_context_size_all`,
# `[56, 6]`. The tier is NOT recoverable from the cache shape — all four rows
# share a left context of 56 — it is recoverable from `valid_out_len`, which is
# 7 only for this row and is exactly `ENCODER_FRAMES_PER_CHUNK`. Asserted below
# rather than trusted.
TIER_INDEX = 2
TIER_CONTEXT = [56, 6]
CHUNK_MEL_FRAMES = 56
PRE_ENCODE_CACHE_FRAMES = 9
ENCODER_MEL_FRAMES = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE_FRAMES
SUBSAMPLING = 8
ENCODER_FRAMES_PER_CHUNK = CHUNK_MEL_FRAMES // SUBSAMPLING
PRE_ENCODE_DROP_FRAMES = -(-PRE_ENCODE_CACHE_FRAMES // SUBSAMPLING)
MEL_FEATURES = 128
ENCODER_DIM = 1024
DECODER_LAYERS = 2
DECODER_HIDDEN = 640
VOCAB_SIZE = 13_087


# --- the export-safe preprocessor -------------------------------------------


class Preprocessor(torch.nn.Module):
    """NeMo's `AudioToMelSpectrogramPreprocessor`, with `dither` off.

    Dither adds Gaussian noise at train time; leaving it on would make the
    export non-deterministic and every golden a lie.
    """

    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, audio: torch.Tensor, audio_length: torch.Tensor):
        mel, mel_length = self.preprocessor(
            input_signal=audio, length=audio_length.to(torch.int64)
        )
        return mel, mel_length.to(torch.int32)


class Encoder(torch.nn.Module):
    """One cache-aware chunk through the Conformer, caches in and out.

    `prompt_id` selects the language prompt embedding the multilingual model
    prepends; `auto` is the id the metadata sidecar publishes as
    `default_prompt_id`.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        # `EncDecRNNTBPEModelWithPrompt.forward` concatenates a one-hot
        # language vector to the ENCODED features and folds it back through
        # `prompt_kernel`; the prompt is not an encoder input at all. Carried
        # here verbatim so the graph answers what the model answers.
        self.concat = bool(getattr(model, "concat", False))
        self.num_prompts = int(getattr(model, "num_prompts", 0))
        self.prompt_kernel = getattr(model, "prompt_kernel", None)

    def forward(
        self,
        mel: torch.Tensor,
        mel_length: torch.Tensor,
        cache_channel: torch.Tensor,
        cache_time: torch.Tensor,
        cache_len: torch.Tensor,
        prompt_id: torch.Tensor,
    ):
        encoded, encoded_length, channel_out, time_out, len_out = (
            self.encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_length.to(torch.int64),
                cache_last_channel=cache_channel,
                cache_last_time=cache_time,
                cache_last_channel_len=cache_len.to(torch.int64),
                # Both are `cache_aware_stream_step`'s own defaults on this
                # tier — `drop_extra_pre_encoded` is NeMo's own
                # `1 + (pre_encode_cache - 1) // subsampling` = 2 — and are
                # passed only so the two numbers the host depends on are stated
                # where they are used. `load_model` asserts the tier produces
                # them.
                keep_all_outputs=True,
                drop_extra_pre_encoded=PRE_ENCODE_DROP_FRAMES,
            )
        )
        if self.concat:
            encoded = torch.transpose(encoded, 1, 2)  # B x D x T -> B x T x D
            time_steps = encoded.shape[1]
            prompt = torch.zeros(
                1, time_steps, self.num_prompts, dtype=encoded.dtype, device=encoded.device
            )
            prompt.scatter_(
                2,
                prompt_id.to(torch.int64).view(1, 1, 1).expand(-1, time_steps, -1),
                1.0,
            )
            encoded = self.prompt_kernel(torch.cat([encoded, prompt], dim=-1))
            encoded = torch.transpose(encoded, 1, 2)  # B x T x D -> B x D x T
        return (
            encoded,
            encoded_length.to(torch.int32),
            channel_out,
            time_out,
            len_out.to(torch.int32),
        )


class DecoderJoint(torch.nn.Module):
    """The RNN-T prediction network and the joint, fused into one step.

    iOS's boundary is one graph per SYMBOL, not per chunk: the greedy walk on
    the host side needs the logits before it can decide whether to advance the
    LSTM, so a fused decoder+joint is the only shape that keeps the walk out of
    the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.joint = model.joint

    def forward(
        self,
        token: torch.Tensor,
        token_length: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
        encoder: torch.Tensor,
    ):
        # `token_length` is iOS's input and is unused by the prediction network;
        # it is folded in as a no-op so the tracer keeps it on the graph and
        # the two boundaries stay the same shape.
        token = token + (token_length.to(token.dtype).view(1, 1) - token_length.to(token.dtype).view(1, 1))
        decoded, state = self.decoder.predict(
            y=token.to(torch.int64),
            state=[h_in, c_in],
            add_sos=False,
            batch_size=1,
        )
        # `joint` takes f as [B, T, H] and g as [B, U, H]; the encoder step
        # crosses the boundary as [1, 1024, 1] because that is iOS's shape.
        logits = self.joint.joint(encoder.transpose(1, 2), decoded)
        return logits, state[0], state[1]


def load_model(nemo_path: pathlib.Path):
    from nemo.collections.asr.models import ASRModel
    # ---- shim 3: NeMo typecheck off for tracing ----------------------------
    # The TorchScript tracer invokes forward positionally; NeMo's typecheck
    # decorator rejects that. NeMo's own export path disables it the same way —
    # the traced math is unchanged.
    from nemo.core.classes.common import typecheck

    model = ASRModel.restore_from(str(nemo_path), map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    parity_gate(model)
    typecheck.set_typecheck_enabled(enabled=False)

    # `setup_streaming_params(att_context_size=…)` CANNOT select a tier: it only
    # feeds `streaming_cfg`, while `forward` reads `self.att_context_size`,
    # which is set once to `att_context_size_all[0]`. The only setter is
    # `set_default_att_context_size`, which is what NeMo's own cache-aware
    # pipeline calls. Without it this exported `[56, 3]` — a 4-frame mask grid
    # fed 7 frames a step.
    model.encoder.set_default_att_context_size(TIER_CONTEXT)
    model.encoder.setup_streaming_params()
    if list(model.encoder.att_context_size) != TIER_CONTEXT:
        raise SystemExit(f"the encoder is on {model.encoder.att_context_size}, not {TIER_CONTEXT}")
    streaming = model.encoder.streaming_cfg
    # The invariant that catches a wrong tier on the FIRST run: this row's
    # valid_out_len is the encoder frames a chunk yields, and the host's
    # ENCODER_FRAMES_PER_CHUNK is the same number. On `[56, 3]` it is 4.
    if streaming.valid_out_len != ENCODER_FRAMES_PER_CHUNK:
        raise SystemExit(
            f"tier {TIER_CONTEXT} yields {streaming.valid_out_len} encoder frames a chunk, "
            f"but the host expects {ENCODER_FRAMES_PER_CHUNK}"
        )
    if streaming.shift_size[1] != CHUNK_MEL_FRAMES:
        raise SystemExit(
            f"tier {TIER_CONTEXT} shifts {streaming.shift_size[1]} mel frames, "
            f"but the host feeds {CHUNK_MEL_FRAMES}"
        )
    if streaming.pre_encode_cache_size[1] != PRE_ENCODE_CACHE_FRAMES:
        raise SystemExit(
            f"tier {TIER_CONTEXT} wants {streaming.pre_encode_cache_size[1]} mel frames of "
            f"pre-encode cache, but the host sends {PRE_ENCODE_CACHE_FRAMES}"
        )
    print(f"streaming tier {TIER_CONTEXT}: {streaming}")
    return model


def vocabulary(nemo_path: pathlib.Path) -> dict[str, str]:
    """`{"<id>": "<piece>"}` — the shape `NemotronModels::load` decodes.

    Read out of the checkpoint's own joint vocabulary rather than the tokenizer
    file, because the joint's `num_classes` is what the decoder indexes.
    """
    import yaml

    with tarfile.open(nemo_path) as tar:
        name = [n for n in tar.getnames() if n.endswith("model_config.yaml")][0]
        config = yaml.safe_load(tar.extractfile(name).read().decode())
    pieces = config["joint"]["vocabulary"]
    if len(pieces) != VOCAB_SIZE:
        raise SystemExit(f"joint vocabulary has {len(pieces)} pieces, expected {VOCAB_SIZE}")
    return {str(index): piece for index, piece in enumerate(pieces)}


def auto_prompt_id(nemo_path: pathlib.Path) -> int:
    """The prompt embedding under which the model picks the language itself.

    `model_defaults.prompt_dictionary["auto"]`, read off the checkpoint — 101
    here. Hard-coding 0 instead is `en-US`, and that is not a mild default: the
    Russian fixture transcribes to the EMPTY STRING under it while the English
    one is word-perfect, which is how this was found.
    """
    import yaml

    with tarfile.open(nemo_path) as tar:
        name = [n for n in tar.getnames() if n.endswith("model_config.yaml")][0]
        config = yaml.safe_load(tar.extractfile(name).read().decode())
    dictionary = config["model_defaults"]["prompt_dictionary"]
    if "auto" not in dictionary:
        raise SystemExit("the checkpoint's prompt_dictionary has no 'auto' entry")
    return int(dictionary["auto"])


def language_tag_token_ids(pieces: dict[str, str]) -> list[int]:
    """The `<xx-XX>` tokens the decoder emits before any text."""
    return sorted(
        int(index)
        for index, piece in pieces.items()
        if piece.startswith("<") and piece.endswith(">") and "-" in piece
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True, type=pathlib.Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model = load_model(args.nemo)
    cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(
        batch_size=1, dtype=torch.float32, device="cpu", max_dim=0
    )
    print(f"cache_channel {tuple(cache_channel.shape)}  cache_time {tuple(cache_time.shape)}")

    audio = torch.zeros(1, CHUNK_MEL_FRAMES * 160, dtype=torch.float32)
    audio_length = torch.tensor([audio.shape[1]], dtype=torch.int32)
    torch.onnx.export(
        Preprocessor(model),
        (audio, audio_length),
        str(ARTIFACTS / "nemotron_preprocessor.onnx"),
        input_names=["audio", "audio_length"],
        output_names=["mel", "mel_length"],
        dynamic_axes={"audio": {1: "samples"}, "mel": {2: "frames"}},
        opset_version=args.opset,
        dynamo=False,
    )

    mel = torch.zeros(1, MEL_FEATURES, ENCODER_MEL_FRAMES, dtype=torch.float32)
    mel_length = torch.tensor([ENCODER_MEL_FRAMES], dtype=torch.int32)
    prompt_id = torch.tensor([0], dtype=torch.int32)
    torch.onnx.export(
        Encoder(model),
        (mel, mel_length, cache_channel, cache_time, cache_len.to(torch.int32), prompt_id),
        str(ARTIFACTS / "nemotron_encoder.onnx"),
        input_names=[
            "mel",
            "mel_length",
            "cache_channel",
            "cache_time",
            "cache_len",
            "prompt_id",
        ],
        output_names=[
            "encoded",
            "encoded_length",
            "cache_channel_out",
            "cache_time_out",
            "cache_len_out",
        ],
        opset_version=args.opset,
        dynamo=False,
    )

    token = torch.zeros(1, 1, dtype=torch.int32)
    token_length = torch.tensor([1], dtype=torch.int32)
    hidden = torch.zeros(DECODER_LAYERS, 1, DECODER_HIDDEN, dtype=torch.float32)
    cell = torch.zeros(DECODER_LAYERS, 1, DECODER_HIDDEN, dtype=torch.float32)
    encoder_step = torch.zeros(1, ENCODER_DIM, 1, dtype=torch.float32)
    torch.onnx.export(
        DecoderJoint(model),
        (token, token_length, hidden, cell, encoder_step),
        str(ARTIFACTS / "nemotron_decoder_joint.onnx"),
        input_names=["token", "token_length", "h_in", "c_in", "encoder"],
        output_names=["logits", "h_out", "c_out"],
        opset_version=args.opset,
        dynamo=False,
    )

    pieces = vocabulary(args.nemo)
    (ARTIFACTS / "nemotron_vocab.json").write_text(json.dumps(pieces, ensure_ascii=False))
    metadata = {
        "chunk_mel_frames": CHUNK_MEL_FRAMES,
        "pre_encode_cache": PRE_ENCODE_CACHE_FRAMES,
        "vocab_size": VOCAB_SIZE,
        "blank_idx": VOCAB_SIZE,
        "cache_channel_shape": list(int(dim) for dim in cache_channel.shape),
        "cache_time_shape": list(int(dim) for dim in cache_time.shape),
        "default_prompt_id": auto_prompt_id(args.nemo),
        "lang_tag_token_ids": language_tag_token_ids(pieces),
        "att_context_size_index": TIER_INDEX,
        "att_context_size": TIER_CONTEXT,
        "encoder_frames_per_chunk": ENCODER_FRAMES_PER_CHUNK,
    }
    (ARTIFACTS / "nemotron_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "lang_tag_token_ids"}, indent=2))
    print(f"nemotron export: {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
