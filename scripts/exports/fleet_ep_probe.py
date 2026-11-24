"""The behavioural gate's CUDA leg: each DirectML-gated family's published
master on one ORT execution provider, fed its door straight from
`ferment-models-export`'s goldens, the raw answers dumped for
`behavioral_from_dumps.rs` to post-process with the shipping pure functions —
the same decisions `ort_behavioral.rs` compares live.

Exists because the CUDA-build onnxruntime DLLs hang under the Rust crate's
load-dynamic path (016 notes); Python loads them fine. Fail-loud: provider
asserted, runtime fallback disabled.

Usage (box):
  python fleet_ep_probe.py --provider cuda --models /path/to/models \
      --export /path/to/exports/models --dump-dir ./ep-dumps

`--provider trt-rtx` is NVIDIA's TensorRT for RTX, an EP-ABI plugin
(`onnxruntime-ep-nv-tensorrt-rtx-cu12` beside `onnxruntime>=1.24`). It compiles
each model for the GPU at session creation, so `session_seconds` is reported
per family; nodes it cannot place fail the session instead of running on CPU.
`--repeat N` adds the median of N timed runs after the first (dumped) one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "trt-rtx": "trt-rtx",
}


def plugin_session(path: Path) -> ort.InferenceSession:
    import onnxruntime_ep_nv_tensorrt_rtx as plugin

    name = plugin.get_ep_name()
    if not any(device.ep_name == name for device in ort.get_ep_devices()):
        ort.register_execution_provider_library(name, plugin.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == name]
    if not devices:
        raise SystemExit(f"{name}: registered, but it offers no device on this machine")
    options = ort.SessionOptions()
    # Fail-loud: a node the EP cannot take is an error, not a silent CPU island.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(str(path), sess_options=options)
    used = session.get_providers()
    if used[0] != name:
        raise SystemExit(f"{path.name}: ORT selected {used}, wanted {name}")
    return session


def session_for(path: Path, provider: str) -> ort.InferenceSession:
    if provider == "trt-rtx":
        return plugin_session(path)
    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(str(path), providers=[provider])
    used = session.get_providers()
    if used[0] != provider:
        raise SystemExit(f"{path.name}: ORT selected {used}, wanted {provider}")
    session.disable_fallback()
    return session


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def max_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def positional_encoding(positions: int, dimension: int) -> np.ndarray:
    """`musichighlighter::pos_enc`: interleaved sin/cos, position 0 all zeros."""
    inverse = 1.0 / np.power(10_000.0, 2.0 * (np.arange(dimension) // 2) / dimension)
    encoding = np.zeros((positions, dimension), dtype=np.float32)
    for position in range(1, positions):
        angles = position * inverse
        encoding[position, 0::2] = np.sin(angles[0::2])
        encoding[position, 1::2] = np.cos(angles[1::2])
    return encoding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--models", type=Path, required=True, help="the published objects, flat")
    parser.add_argument("--export", type=Path, required=True, help="ferment-models-export/models")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--repeat", type=int, default=0, help="timed runs after the first")
    args = parser.parse_args()
    provider = PROVIDERS[args.provider]
    goldens = lambda model, name: args.export / model / "goldens" / name  # noqa: E731
    results: dict[str, dict] = {}
    timings: dict[str, dict] = {}

    def open_session(family: str, path: Path) -> ort.InferenceSession:
        started = time.perf_counter()
        session = session_for(path, provider)
        timings.setdefault(family, {})["session_seconds"] = round(time.perf_counter() - started, 3)
        return session

    def run(family: str, session: ort.InferenceSession, outputs: list[str], feeds: dict) -> list:
        answer = session.run(outputs, feeds)
        if args.repeat > 0 and "run_ms_median" not in timings.get(family, {}):
            samples = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                session.run(outputs, feeds)
                samples.append((time.perf_counter() - started) * 1000.0)
            timings.setdefault(family, {})["run_ms_median"] = round(statistics.median(samples), 2)
        return answer

    def dump(name: str, array: np.ndarray) -> None:
        if args.dump_dir:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            array.astype(np.float32).tofile(args.dump_dir / f"{name}.{args.provider}.f32")

    # --- beat_this_1: the golden mel cut to the door -> logits rows [beat, downbeat]
    mel = read_f32(goldens("beat_this", "beat_this_mel_out.f32")).reshape(128, -1)
    door = np.ascontiguousarray(mel[:, :1500])[None, None]
    sess = open_session("beat_this_1", args.models / "beat_this_1.onnx")
    logits = run("beat_this_1", sess, ["logits"], {"mel": door})[0].reshape(-1)
    dump("beat_this_1", logits)
    results["beat_this_1"] = {
        "max_abs_diff": max_diff(logits, read_f32(goldens("beat_this", "beat_this_1_logits_out.f32")))
    }

    # --- music_highlighter: the golden track's 37 chunks tiled over 200 slots
    chunks = read_f32(goldens("music_highlighter", "music_highlighter_track_chunks_in.f32")).reshape(-1, 129, 128)
    slots = np.arange(200) % len(chunks)
    encoding = positional_encoding(len(chunks), 256)
    sess = open_session("music_highlighter", args.models / "music_highlighter.onnx")
    logits = run(
        "music_highlighter", sess, ["logits"], {"chunks": chunks[slots], "pos_enc": encoding[slots][None]}
    )[0].reshape(-1)
    dump("music_highlighter", logits)
    results["music_highlighter"] = {
        "chunks": len(chunks),
        "max_abs_diff": max_diff(
            logits, read_f32(goldens("music_highlighter", "music_highlighter_track_logits_out.f32"))
        ),
    }

    # --- autoshot: the golden video window and its host histograms -> one_hot
    frames = np.fromfile(goldens("autoshot", "autoshot_video_frames_in.u8"), dtype=np.uint8).reshape(100, 27, 48, 3)
    histograms = read_f32(goldens("autoshot", "autoshot_video_color_hist_out.f32")).reshape(1, 100, 512)
    sess = open_session("autoshot", args.models / "autoshot.onnx")
    one_hot = run("autoshot", sess, ["one_hot"], {
        "frames": frames.astype(np.float32).transpose(3, 0, 1, 2)[None],
        "color_hist": histograms,
    })[0].reshape(-1)
    dump("autoshot", one_hot)
    results["autoshot"] = {
        "max_abs_diff": max_diff(one_hot, read_f32(goldens("autoshot", "autoshot_video_one_hot_out.f32")))
    }

    # --- fireredvad_aed: kaldi's golden fbank, the contract's CMVN, one masked window
    contract = json.loads((args.export / "fireredvad_aed" / "model.json").read_text())
    cmvn = contract["frontend"]["cmvn"]
    means = np.asarray(cmvn["means"], dtype=np.float32)
    inverse_std = np.asarray(cmvn["inverseStd"], dtype=np.float32)
    window = contract["frontend"]["windows"]["frames"]
    sess = open_session("fireredvad_aed", args.models / "fireredvad_aed.onnx")
    worst = 0.0
    for clip in ["hairdryer", "sec5"]:
        fbank = read_f32(goldens("fireredvad_aed", f"fireredvad_aed_{clip}_fbank_out.f32")).reshape(-1, 80)
        count = len(fbank)
        assert count <= window, f"{clip}: {count} frames is more than one window"
        feat = np.zeros((1, 80, 1, window), dtype=np.float32)
        feat[0, :, 0, :count] = ((fbank - means) * inverse_std).T
        mask = np.zeros((1, 1, 1, window), dtype=np.float32)
        mask[..., :count] = 1.0
        probs = run("fireredvad_aed", sess, ["probs"], {"feat": feat, "mask": mask})[0][0, :, 0, :count].T.reshape(-1)
        dump(f"fireredvad_{clip}", probs)
        expected = read_f32(goldens("fireredvad_aed", f"fireredvad_aed_{clip}_probs_out.f32"))
        worst = max(worst, max_diff(probs, expected))
    results["fireredvad_aed"] = {"max_abs_diff": worst}

    print(json.dumps({"provider": provider, "results": results, "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
