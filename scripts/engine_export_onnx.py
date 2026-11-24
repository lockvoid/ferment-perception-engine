#!/usr/bin/env python3
"""Export and verify every locally recoverable Engine model as FP16 ONNX.

The model implementations and oracle fixtures remain in the read-only Swift
checkout. Outputs are written to an external models directory, never the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def export_fp16(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".fp32.onnx")
    torch.onnx.export(
        model,
        inputs,
        temporary,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    try:
        convert_fp16_model(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def convert_fp16_model(source: Path, output: Path) -> None:
    fp32 = onnx.load(source)
    fp16 = convert_float_to_float16(fp32, keep_io_types=True, disable_shape_infer=True)
    topologically_sort(fp16)
    onnx.checker.check_model(fp16)
    onnx.save(fp16, output)


def topologically_sort(model: onnx.ModelProto) -> None:
    """Repair node order after ORT's FP16 converter adds I/O Cast nodes."""
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    remaining = list(graph.node)
    ordered = []
    while remaining:
        for index, node in enumerate(remaining):
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(name for name in node.output if name)
                remaining.pop(index)
                break
        else:
            unresolved = ", ".join(node.name or node.op_type for node in remaining[:5])
            raise RuntimeError(f"could not topologically sort ONNX graph near {unresolved}")
    del graph.node[:]
    graph.node.extend(ordered)


def ort_outputs(model: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)


def max_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def export_highlighter(swift: Path, checkpoints: Path, output: Path) -> float:
    directory = swift / "Scripts/music_highlghting"
    module = load_module("ferment_music_highlighter", directory / "convert.py")
    source = module.HighlighterModel()
    source.load_state_dict(
        torch.load(checkpoints / "music_highlighter.pt", map_location="cpu", weights_only=True)
    )
    source.train()
    model = module.HighlighterExportModel(source).eval()
    chunks = torch.zeros(module.MAX_CHUNKS, module.CHUNK_FRAMES, module.N_MELS)
    positions = torch.zeros(1, module.MAX_CHUNKS, module.DIM * 4)
    export_fp16(model, (chunks, positions), ["chunks", "pos_enc"], ["logits"], output)

    fixtures = swift / "Tests/Fixtures/MusicHighlighter"
    chunks_10 = np.fromfile(fixtures / "test_chunks.bin", dtype=np.float32).reshape(
        10, module.CHUNK_FRAMES, module.N_MELS
    )
    positions_10 = np.fromfile(fixtures / "test_pos_enc.bin", dtype=np.float32).reshape(
        1, 10, module.DIM * 4
    )
    tiled_chunks = np.tile(chunks_10, (module.MAX_CHUNKS // 10, 1, 1))
    tiled_positions = np.tile(positions_10, (1, module.MAX_CHUNKS // 10, 1))
    logits = ort_outputs(output, {"chunks": tiled_chunks, "pos_enc": tiled_positions})[0][:10]
    scores = np.exp(logits - np.max(logits))
    scores /= np.sum(scores)
    expected = np.fromfile(fixtures / "test_scores.bin", dtype=np.float32)
    return max_difference(scores, expected)


def export_autoshot(swift: Path, checkpoints: Path, output: Path) -> float:
    directory = swift / "Scripts/scene_detection"
    sys.path.insert(0, str(directory))
    module = load_module("ferment_autoshot", directory / "autoshot_model.py")
    model = module.AutoShotSupernet()
    module.load_autoshot_weights(model, str(checkpoints / "autoshot.pth"))
    model.eval()
    frames = torch.zeros(1, 3, module.N_FRAMES, module.H, module.W)
    histograms = torch.zeros(1, module.N_FRAMES, 512)
    export_fp16(
        model,
        (frames, histograms),
        ["frames", "color_hist"],
        ["one_hot", "many_hot"],
        output,
    )

    fixtures = swift / "Tests/Fixtures/VideoSceneDetector"
    frame_bytes = np.fromfile(fixtures / "test_frames.bin", dtype=np.uint8).reshape(
        module.N_FRAMES, module.H, module.W, 3
    )
    frame_input = frame_bytes.astype(np.float32).transpose(3, 0, 1, 2)[None]
    histograms = np.fromfile(fixtures / "test_color_hist.bin", dtype=np.float32).reshape(
        1, module.N_FRAMES, 512
    )
    one_hot = ort_outputs(output, {"frames": frame_input, "color_hist": histograms})[0].reshape(-1)
    expected = np.fromfile(fixtures / "test_one_hot.bin", dtype=np.float32)
    return max_difference(one_hot, expected)


def export_fireredvad(swift: Path, source: Path, output: Path) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    convert_fp16_model(source, output)
    maximum = 0.0
    for fixture_name in ["hairdryer", "sec5"]:
        fixtures = swift / f"Tests/Fixtures/AudioEventDetector/{fixture_name}"
        features = np.fromfile(fixtures / "feat_cmvn.bin", dtype=np.float32).reshape(
            1, -1, 80
        )
        probabilities = ort_outputs(output, {"feat": features})[0].reshape(-1)
        expected = np.fromfile(fixtures / "probs.bin", dtype=np.float32)
        maximum = max(maximum, max_difference(probabilities, expected))
    return maximum


def export_beatthis(swift: Path, source: Path, output: Path) -> float:
    fixtures = swift / "Tests/Fixtures"
    mel = np.fromfile(fixtures / "model_input_mel.bin", dtype=np.float32).reshape(
        1, 1500, 128
    )
    expected = ort_outputs(source, {"spectrogram": mel})[0]
    convert_fp16_model(source, output)
    actual = ort_outputs(output, {"spectrogram": mel})[0]
    return max_difference(actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=[
            "music_highlighter",
            "autoshot",
            "fireredvad_aed",
            "beat_this_1",
            "beat_this_2",
            "beat_this_3",
        ],
    )
    parser.add_argument("--swift", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-onnx", type=Path)
    arguments = parser.parse_args()

    output = arguments.output_dir / f"{arguments.model}.onnx"
    if arguments.model == "music_highlighter":
        difference = export_highlighter(arguments.swift, arguments.checkpoints, output)
        # FP16 ONNX measured 0.00146762 against the FP32 Swift/PyTorch score
        # fixture; retain a narrow per-model ceiling instead of a global one.
        tolerance = 0.0015
    elif arguments.model == "autoshot":
        difference = export_autoshot(arguments.swift, arguments.checkpoints, output)
        tolerance = 0.05
    elif arguments.model == "fireredvad_aed":
        if arguments.source_onnx is None:
            parser.error("fireredvad_aed requires --source-onnx")
        difference = export_fireredvad(arguments.swift, arguments.source_onnx, output)
        # The classifier-only FP16 conversion measured 0.00176954 across both
        # dynamic-length Swift fixtures.
        tolerance = 0.002
    else:
        if arguments.source_onnx is None:
            parser.error(f"{arguments.model} requires --source-onnx")
        difference = export_beatthis(arguments.swift, arguments.source_onnx, output)
        tolerance = 0.075
    if difference > tolerance:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"{arguments.model} maximum oracle difference {difference} exceeds {tolerance}"
        )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        f"verified {output} bytes={output.stat().st_size} sha256={digest} "
        f"max_difference={difference:.8g} tolerance={tolerance}"
    )


if __name__ == "__main__":
    main()
