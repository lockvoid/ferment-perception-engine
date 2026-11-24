"""Origin goldens for the four Parakeet RNNT sub-models: stock NeMo torch
(complex STFT, no export shims) on deterministic inputs.

The decoder runs with `_rnnt_export = True` (step semantics, no SOS) and the
joint with loss/WER fusion off — the same inference-contract flags the export
uses, exercised through the same NeMo eager code the ONNX was traced from.

Consumed by crates/engine-processors/tests/ort_semantic.rs.

    ./scripts/exports/parakeet/.venv/bin/python scripts/exports/parakeet/gen_goldens.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
GOLDENS = REPO / "scripts/exports/rust_goldens"


def deterministic_audio(samples: int = 48_000) -> np.ndarray:
    # Tone sweep + seeded noise, the dasheng-golden recipe at 3 s / 16 kHz.
    rng = np.random.default_rng(2026)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    sweep = np.sin(2.0 * np.pi * (200.0 + 1800.0 * t) * t) * 0.3
    noise = rng.standard_normal(samples) * 0.02
    return (sweep + noise).astype(np.float32)[None, :]


def write(name: str, array: np.ndarray) -> None:
    (GOLDENS / name).write_bytes(np.ascontiguousarray(array).tobytes())
    print(f"{name} {array.dtype} {array.shape}")


def main() -> None:
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    model = EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    ).eval()
    GOLDENS.mkdir(parents=True, exist_ok=True)

    # ---- preprocessor: audio -> log-mel [1, 128, T] -------------------------
    audio = deterministic_audio()
    length = torch.tensor([audio.shape[1]], dtype=torch.int32)
    with torch.no_grad():
        mel, mel_length = model.preprocessor(
            input_signal=torch.from_numpy(audio), length=length
        )
    write("parakeet_audio_in.bin", audio)
    write("parakeet_mel_out.bin", mel.numpy().astype(np.float32))
    # NeMo's frame count (samples/hop = 300) trails the padded mel (301
    # frames); the encoder masks by this length, so persist the exact value.
    write("parakeet_mel_len.bin", mel_length.numpy().astype(np.int32))

    # ---- encoder: log-mel -> encoded [1, 1024, T'] --------------------------
    with torch.no_grad():
        encoded, encoded_length = model.encoder(audio_signal=mel, length=mel_length)
    write("parakeet_encoded_out.bin", encoded.numpy().astype(np.float32))
    print(f"mel frames {int(mel_length.item())} -> encoded frames {int(encoded_length.item())}")

    # ---- decoder: one token + LSTM state -> step [1, 640, 1] ----------------
    model.decoder._rnnt_export = True
    torch.manual_seed(2026)
    h_in = torch.randn(2, 1, 640)
    c_in = torch.randn(2, 1, 640)
    targets = torch.tensor([[42]], dtype=torch.int32)
    with torch.no_grad():
        g, _, states = model.decoder(
            targets=targets, target_length=torch.tensor([1], dtype=torch.int32),
            states=[h_in, c_in],
        )
    write("parakeet_dec_h_in.bin", h_in.numpy().astype(np.float32))
    write("parakeet_dec_c_in.bin", c_in.numpy().astype(np.float32))
    write("parakeet_dec_out.bin", g.numpy().astype(np.float32))
    write("parakeet_dec_h_out.bin", states[0].numpy().astype(np.float32))
    write("parakeet_dec_c_out.bin", states[1].numpy().astype(np.float32))

    # ---- joint: encoder/decoder steps -> (token_id, token_prob, duration) ---
    # Real pipeline tensors, not noise: random steps saturate the joint to
    # blank at prob 1.0, which would make the golden trivially weak.
    model.joint.set_fuse_loss_wer(False)
    encoder_step = encoded[:, :, 19:20].contiguous()
    decoder_step = g.contiguous()
    with torch.no_grad():
        logits = model.joint(encoder_outputs=encoder_step, decoder_outputs=decoder_step)
    logits = logits.squeeze(1).squeeze(1)
    n_durations = model.joint.num_extra_outputs
    token_logits = logits[:, : logits.shape[-1] - n_durations]
    duration_logits = logits[:, logits.shape[-1] - n_durations :]
    token_probs = torch.softmax(token_logits, dim=-1)
    token_id = token_probs.argmax(dim=-1)
    token_prob = token_probs.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
    duration = duration_logits.argmax(dim=-1)
    write("parakeet_joint_enc_in.bin", encoder_step.numpy().astype(np.float32))
    write("parakeet_joint_dec_in.bin", decoder_step.numpy().astype(np.float32))
    write("parakeet_joint_token_id_out.bin", token_id.numpy().astype(np.int64))
    write("parakeet_joint_token_prob_out.bin", token_prob.numpy().astype(np.float32))
    write("parakeet_joint_duration_out.bin", duration.numpy().astype(np.int64))
    print(
        f"joint: token_id {int(token_id.item())} "
        f"prob {float(token_prob.item()):.6f} duration {int(duration.item())}"
    )


if __name__ == "__main__":
    main()
