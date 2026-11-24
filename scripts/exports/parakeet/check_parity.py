"""ONNX-vs-origin parity for the four consolidated parakeet artifacts:
runs each artifact through onnxruntime CPU on the golden inputs and reports
max abs diff against the stock-NeMo goldens from gen_goldens.py.

    ./scripts/exports/parakeet/.venv/bin/python scripts/exports/parakeet/check_parity.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
GOLDENS = HERE.parents[2] / "scripts/exports/rust_goldens"


def golden(name: str, dtype, shape) -> np.ndarray:
    return np.frombuffer((GOLDENS / name).read_bytes(), dtype=dtype).reshape(shape)


def session(name: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        (ARTIFACTS / f"{name}.onnx").as_posix(), providers=["CPUExecutionProvider"]
    )


def report(label: str, actual: np.ndarray, expected: np.ndarray) -> float:
    diff = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max())
    print(f"{label}: max abs diff {diff:.3e}  (range {expected.min():.3f}..{expected.max():.3f})")
    return diff


def main() -> None:
    audio = golden("parakeet_audio_in.bin", np.float32, (1, 48_000))
    mel = golden("parakeet_mel_out.bin", np.float32, (1, 128, -1))

    sess = session("parakeet_preprocessor")
    (mel_onnx,) = sess.run(
        None,
        {"audio_signal": audio, "audio_length": np.array([48_000], dtype=np.int32)},
    )
    report("preprocessor mel", mel_onnx, mel)

    encoded = golden("parakeet_encoded_out.bin", np.float32, (1, 1024, -1))
    mel_len = golden("parakeet_mel_len.bin", np.int32, (1,))
    sess = session("parakeet_encoder")
    outputs = sess.run(None, {"audio_signal": mel, "length": mel_len})
    report("encoder outputs (fp16 weights)", outputs[0], encoded)

    sess = session("parakeet_decoder")
    # target_length was passthrough-only, so the tracer pruned it: the shipped
    # decoder contract is (targets, h_in, c_in).
    dec, h_out, c_out = sess.run(
        None,
        {
            "targets": np.array([[42]], dtype=np.int32),
            "h_in": golden("parakeet_dec_h_in.bin", np.float32, (2, 1, 640)),
            "c_in": golden("parakeet_dec_c_in.bin", np.float32, (2, 1, 640)),
        },
    )
    report("decoder step", dec, golden("parakeet_dec_out.bin", np.float32, (1, 640, 1)))
    report("decoder h_out", h_out, golden("parakeet_dec_h_out.bin", np.float32, (2, 1, 640)))
    report("decoder c_out", c_out, golden("parakeet_dec_c_out.bin", np.float32, (2, 1, 640)))

    sess = session("parakeet_joint")
    token_id, token_prob, duration = sess.run(
        None,
        {
            "encoder_step": golden("parakeet_joint_enc_in.bin", np.float32, (1, 1024, 1)),
            "decoder_step": golden("parakeet_joint_dec_in.bin", np.float32, (1, 640, 1)),
        },
    )
    want_id = golden("parakeet_joint_token_id_out.bin", np.int64, (1,))
    want_dur = golden("parakeet_joint_duration_out.bin", np.int64, (1,))
    assert token_id.tolist() == want_id.tolist(), (token_id, want_id)
    assert duration.tolist() == want_dur.tolist(), (duration, want_dur)
    report(
        "joint token_prob",
        token_prob,
        golden("parakeet_joint_token_prob_out.bin", np.float32, (1,)),
    )
    print(f"joint token_id {token_id.tolist()} duration {duration.tolist()} exact-match ok")


if __name__ == "__main__":
    main()
