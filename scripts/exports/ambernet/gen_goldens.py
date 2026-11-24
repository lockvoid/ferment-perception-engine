"""Origin goldens for AmberNet LangID: stock NeMo eager
(`AudioToMelSpectrogramPreprocessor` -> `ConvASREncoder` -> `SpeakerDecoder`)
on deterministic input, with no export shims.

Two cases from ONE committed clip, because `num_frames` is what makes the
graph length-aware:

  * `num_frames = 500`  — NeMo eager over the 5 s clip alone;
  * `num_frames = 3000` — NeMo eager over the same clip zero-padded to the
    full 30 s window, i.e. 5 s of speech and 25 s of digital silence.

The two answers differ by a lot; a graph that ignored `num_frames` would
return the second for both.

Only the 5 s clip is committed (320 KB); the Rust oracle zero-pads it to the
graph's 480 000-sample window, which is exactly what `build_window` does.

Consumed by crates/engine-processors/tests/ort_semantic.rs.

    ./scripts/exports/ambernet/.venv/bin/python scripts/exports/ambernet/gen_goldens.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort
import torch

import export_ambernet as ex

REPO = pathlib.Path(__file__).resolve().parents[3]
GOLDENS = REPO / "scripts/exports/rust_goldens"

CLIP_SECONDS = 5
CLIP_SAMPLES = ex.SAMPLE_RATE * CLIP_SECONDS      # 80_000
CLIP_FRAMES = CLIP_SAMPLES // ex.HOP_LENGTH       # 500


def deterministic_audio(samples: int) -> np.ndarray:
    """Tone sweep + seeded noise — the dasheng/parakeet golden recipe."""
    rng = np.random.default_rng(2026)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    sweep = np.sin(2.0 * np.pi * (200.0 + 1800.0 * t) * t) * 0.3
    noise = rng.standard_normal(samples) * 0.02
    return (sweep + noise).astype(np.float32)[None, :]


def write(name: str, array: np.ndarray) -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / name).write_bytes(np.ascontiguousarray(array).tobytes())
    print(f"  {name} {array.dtype} {tuple(array.shape)}")


def report(label: str, a: np.ndarray, b: np.ndarray) -> float:
    delta = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    print(f"    {label:<44} max|delta| = {delta:.6g}")
    return delta


def main() -> None:
    model, labels = ex.load_nemo_model()

    clip = deterministic_audio(CLIP_SAMPLES)
    window = np.zeros((1, ex.N_SAMPLES), dtype=np.float32)
    window[0, :CLIP_SAMPLES] = clip[0]

    print("\n[goldens]")
    write("ambernet_audio_in.bin", clip)

    logits = {}
    for frames, signal in ((CLIP_FRAMES, clip), (ex.MAX_FRAMES, window)):
        with torch.no_grad():
            length = torch.tensor([signal.shape[1]], dtype=torch.int64)
            feats, feats_len = model.preprocessor(
                input_signal=torch.from_numpy(signal), length=length
            )
            encoded, encoded_len = model.encoder(audio_signal=feats, length=feats_len)
            out, embedding = model.decoder(encoder_output=encoded, length=encoded_len)
        logits[frames] = out.numpy().astype(np.float32)
        assert int(feats_len[0]) == frames, (int(feats_len[0]), frames)
        if frames == CLIP_FRAMES:
            write("ambernet_logits_out.bin", logits[frames])
            write("ambernet_embedding_out.bin", embedding.numpy().astype(np.float32))
        else:
            write("ambernet_logits_full_out.bin", logits[frames])
        top = int(np.argmax(logits[frames]))
        print(f"    num_frames {frames}: top-1 {labels[top]!r} logit {logits[frames][0, top]:.4f}")

    print(
        "    num_frames sensitivity (the load-bearing input): "
        f"{float(np.abs(logits[CLIP_FRAMES] - logits[ex.MAX_FRAMES]).max()):.6g}"
    )

    print("\n[qualification: onnx vs the NeMo origin, CPU EP]")
    session = ort.InferenceSession(
        str(pathlib.Path(ex.ARTIFACTS_DIR) / f"{ex.MODEL_NAME}.onnx"),
        providers=["CPUExecutionProvider"],
    )
    for frames in (CLIP_FRAMES, ex.MAX_FRAMES):
        out = session.run(
            None,
            {"audio": window, "num_frames": np.array([frames], dtype=np.int32)},
        )
        report(f"logits at num_frames {frames}", logits[frames], out[0])
        assert int(np.argmax(out[0])) == int(np.argmax(logits[frames])), \
            f"top-1 disagreement at num_frames {frames}"
    print("\nDone.")


if __name__ == "__main__":
    main()
