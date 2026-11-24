"""EdgeTAM V1 (facebookresearch/EdgeTAM, Apache-2.0) -> the ONNX graphs
`crates/engine-processors/src/edgetam/` runs.

Produces, in `artifacts/`:

    edgetam_v1_image_encoder.onnx        normalized_image[1,3,1024,1024] f16
                                         -> vision_features[1,256,64,64],
                                            high_resolution_features_0[1,32,256,256],
                                            high_resolution_features_1[1,64,128,128]
    edgetam_v1_memory_encoder.onnx       raw_vision_features[1,256,64,64] f16,
                                         selected_low_resolution_mask[1,1,256,256] f16
                                         -> memory_features[1,512,64]
    edgetam_v1_prompt_mask_decoder.onnx  vision_features, high_resolution_features_0/1,
                                         point_coordinates[1,N,2], point_labels[1,N],
                                         add_no_memory_embedding[1],
                                         apply_object_score_mask_gate[1],
                                         mask_input[1,1,256,256], has_mask_input[1]
                                         -> candidate_masks[1,4,256,256],
                                            iou_predictions[1,4],
                                            candidate_object_pointers[1,4,256],
                                            object_score_logits[1,1]

Boundary
--------
The graph names, shapes and flags are iOS's `EdgeTAMModel.swift` /
`EdgeTAMTensor.swift`, because the Rust port is the Swift processor's twin. Two
deliberate differences, both recorded in the lane report:

* iOS's prompt/mask decoder is a MULTIFUNCTION Core ML package with sixteen
  specialized `points_N_{with,no}_mask` variants. ONNX has no multifunction
  container, so this is ONE graph with a dynamic token axis and the upstream
  SAM2 ONNX export's `has_mask_input` flag (report D1).
* Every tensor at the boundary is `float16` — iOS's contract — while the
  COMPUTE stays `float32`. Casting at the door rather than tracing a half model
  keeps every op on a kernel CPU export actually implements, and `float16` in,
  `float16` out is what the Rust `Tensor::from_f16` sends and reads.

What this script does NOT produce
---------------------------------
`edgetam_v1_memory_attention`, the last of the four. The image session needs
only the first two, so the still-image capability is complete; the video session
needs all four and is not.

The memory-attention graph is blocked on one thing, stated exactly:
`EdgeTAMTemporalState`'s fixed 7-spatial/16-pointer bank carries a VALIDITY
vector per bank, and upstream's `MemoryAttention` takes no key-padding mask —
it is handed a variable-length memory instead, built by
`_prepare_memory_conditioned_features`. Exporting the fixed contract therefore
means threading a mask through `MemoryAttention`/`RoPEAttention`, which changes
third-party model code rather than wrapping it. That is a ruling (report §7
N-4), not a guess.

(The OTHER half of N-4 dissolved on reading the config: `edgetam.yaml` sets
`add_tpos_enc_to_obj_ptrs: false`, so the object pointers carry NO temporal
positional embedding at all and the conditioning pointer's distance — the
number the fixed 16-slot bank has no room for — never enters the model.)

Usage
-----
    uv run --project scripts/exports/edgetam python scripts/exports/edgetam/export.py \
        --checkpoint /path/to/edgetam.pt --repo /path/to/EdgeTAM
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

IMAGE_SIZE = 1024
LOW_RESOLUTION = 256
# The token axis is DYNAMIC, so the graph takes the caller's real click count —
# 1…8 — exactly as iOS's sixteen specialized functions do.
#
# It was static at 8 for one export, because the traced `_embed_points` baked a
# broadcast against the example's count into a `Where` and any other count died
# with "Attempting to broadcast an axis by a dimension other than 1". That was
# the MASKED-ASSIGNMENT form; rewriting it as arithmetic (below) removed the
# scatter and the axis became dynamic. Padding to 8 was measured NOT to be
# neutral: the padding label is `-1`, whose embedding is the learned
# `not_a_point_embed`, so a one-click prompt reached the two-way transformer as
# nine sparse tokens against iOS's two and the IoUs moved (0.022 at one token,
# 0.053 at eight). This is the tracing example only.
PROMPT_TOKENS = 4


def install_export_embed_points(prompt_encoder) -> None:
    """Rewrite `PromptEncoder._embed_points`'s masked assignments as arithmetic.

    Upstream writes the five label branches as BOOLEAN-MASK INDEXED ASSIGNMENT
    (`point_embedding[labels == 1] += …`). The ONNX exporter turns each of those
    into a scatter whose shapes depend on how many entries the mask selects —
    data-dependent, so the traced graph carries the example's count and any
    other prompt fails shape inference at `sam_prompt_encoder/Where_6`
    ("Incompatible dimensions"). This is the rewrite every SAM2 ONNX export
    makes, and it is arithmetically identical: a label matches at most one
    branch, so summing the masked weights equals assigning the matching one.

    The parity of the two forms is asserted in `parity_gate` before anything is
    exported, on a prompt that exercises every label EdgeTAM emits (−1, 0, 1,
    2, 3).
    """
    import types

    def export_embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        selector = labels.unsqueeze(-1)
        not_a_point = (selector == -1).to(embedding.dtype)
        # `[labels == -1] = 0` then `+= not_a_point`: the zeroing is a MASKED
        # REPLACEMENT, not an addition, so it has to stay one.
        embedding = (1.0 - not_a_point) * embedding + not_a_point * self.not_a_point_embed.weight
        for label, table in enumerate(self.point_embeddings):
            embedding = embedding + (selector == label).to(embedding.dtype) * table.weight
        return embedding

    prompt_encoder._embed_points = types.MethodType(export_embed_points, prompt_encoder)


def parity_gate(model) -> None:
    """The untouched masked-assignment path vs the arithmetic one, same prompt."""
    encoder = model.sam_prompt_encoder
    torch.manual_seed(0)
    points = torch.rand(1, 5, 2) * IMAGE_SIZE
    # Every label EdgeTAM emits: background, foreground, both box corners, and
    # the not-a-point token.
    labels = torch.tensor([[0, 1, 2, 3, -1]], dtype=torch.int32)
    with torch.no_grad():
        reference = encoder._embed_points(points, labels, pad=True)
    install_export_embed_points(encoder)
    with torch.no_grad():
        shimmed = encoder._embed_points(points, labels, pad=True)
    worst = (reference - shimmed).abs().max().item()
    print(f"prompt-encoder parity (masked assignment vs arithmetic): max abs diff {worst:.3e}")
    assert worst == 0.0, f"the rewrite is not identical: {worst}"


class ImageEncoder(torch.nn.Module):
    """`forward_image` and the three FPN levels iOS's contract names.

    `no_mem_embed` is deliberately NOT added here. iOS adds it inside the
    DECODER, under `add_no_memory_embedding`, because the video path adds it
    only on a first conditioning frame while the image path always does — one
    encoder output, two policies.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, normalized_image: torch.Tensor):
        backbone = self.model.forward_image(normalized_image.float())
        fpn = backbone["backbone_fpn"]
        return (
            fpn[2].half(),
            fpn[0].half(),
            fpn[1].half(),
        )


class PromptMaskDecoder(torch.nn.Module):
    """`_forward_sam_heads` with its slicing undone and its policy exposed.

    iOS keeps ALL FOUR decoder candidates and selects host-side, so this returns
    `predict_masks`'s raw output rather than `MaskDecoder.forward`'s
    multimask-sliced view. `EdgeTAMPromptInference.selectedCandidateIndex` is
    the policy: candidate 0 for a box or several clicks, argmax over 1…3
    otherwise — which is exactly SAM2's own `mask_slice`.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        vision_features: torch.Tensor,
        high_resolution_features_0: torch.Tensor,
        high_resolution_features_1: torch.Tensor,
        point_coordinates: torch.Tensor,
        point_labels: torch.Tensor,
        add_no_memory_embedding: torch.Tensor,
        apply_object_score_mask_gate: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        model = self.model
        features = vision_features.float()
        add_no_mem = add_no_memory_embedding.float().view(1, 1, 1, 1)
        # `SAM2ImagePredictor.set_image` adds `no_mem_embed` once to the
        # lowest-resolution feature; the video path adds it only on a first
        # conditioning frame. The flag is the seam, so the graph carries both.
        batch, channels, height, width = features.shape
        flat = features.flatten(2).permute(2, 0, 1)
        flat = flat + add_no_mem.view(1, 1, 1) * model.no_mem_embed.squeeze(0)
        features = flat.permute(1, 2, 0).view(batch, channels, height, width)

        # The prompt encoder takes points; the mask comes in as a DENSE
        # embedding so the "no mask" case stays a flag rather than a second
        # graph — `no_mask_embed` is what SAM2 substitutes when `masks=None`.
        sparse, dense_no_mask = model.sam_prompt_encoder(
            points=(point_coordinates.float(), point_labels.to(torch.int32)),
            boxes=None,
            masks=None,
        )
        dense_with_mask = model.sam_prompt_encoder.mask_downscaling(mask_input.float())
        has_mask = has_mask_input.float().view(1, 1, 1, 1)
        dense = has_mask * dense_with_mask + (1.0 - has_mask) * dense_no_mask

        masks, ious, mask_tokens_out, object_score_logits = model.sam_mask_decoder.predict_masks(
            image_embeddings=features,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[
                high_resolution_features_0.float(),
                high_resolution_features_1.float(),
            ],
        )

        # The video path gates the masks on the object score; the image path
        # returns the raw logits (`SAM2ImagePredictor` does not gate).
        # `NO_OBJ_SCORE` is upstream's -1024.0.
        is_obj_appearing = (object_score_logits > 0).float()
        gate = apply_object_score_mask_gate.float().view(1, 1, 1, 1)
        keep = torch.clamp(is_obj_appearing.view(-1, 1, 1, 1) + (1.0 - gate), max=1.0)
        masks = keep * masks + (1.0 - keep) * (-1024.0)

        # One pointer per candidate, with `fixed_no_obj_ptr`'s occlusion
        # handling — iOS indexes this by the SAME candidate index it picked.
        pointers = model.obj_ptr_proj(mask_tokens_out)
        lambda_is_obj_appearing = is_obj_appearing.view(-1, 1, 1)
        pointers = lambda_is_obj_appearing * pointers
        pointers = pointers + (1.0 - lambda_is_obj_appearing) * model.no_obj_ptr

        return (
            masks.half(),
            ious.half(),
            pointers.half(),
            object_score_logits.half(),
        )


class MemoryEncoder(torch.nn.Module):
    """`_encode_new_memory`, with the two branches its flags fix.

    iOS's boundary takes the SELECTED 256x256 mask logits and the raw vision
    features, so the 256->1024 upsample the encoder wants lives inside the
    graph. Two upstream branches are pinned rather than exposed, because iOS's
    contract has no input for either:

    * `is_mask_from_pts` is FALSE — the sigmoid path, not the binarize one.
      `binarize_mask_from_pts_for_mem_enc` only ever applies to a mask that came
      straight from a click, and this graph is fed the DECODER's logits.
    * `non_overlap_masks_for_mem_enc` is a batch-wide constraint across several
      objects; EdgeTAM V1 tracks one object per session (`EdgeTAMSession` refuses
      a second), so there is nothing to make non-overlapping.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        raw_vision_features: torch.Tensor,
        selected_low_resolution_mask: torch.Tensor,
    ):
        model = self.model
        features = raw_vision_features.float()
        high_res = torch.nn.functional.interpolate(
            selected_low_resolution_mask.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        mask_for_mem = torch.sigmoid(high_res)
        if model.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * model.sigmoid_scale_for_mem_enc
        if model.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + model.sigmoid_bias_for_mem_enc
        out = model.memory_encoder(features, mask_for_mem, skip_mask_sigmoid=True)
        memory = out["vision_features"]
        pos = out["vision_pos_enc"][0]
        if model.spatial_perceiver is not None:
            memory, _ = model.spatial_perceiver(memory, pos)
        return memory.half()


def load_model(repo: pathlib.Path, checkpoint: pathlib.Path):
    sys.path.insert(0, str(repo))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from sam2.build_sam import _load_checkpoint  # noqa: F401  (kept for parity with upstream)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = repo / "sam2"
    initialize_config_dir(config_dir=str(config_dir), version_base="1.2")
    cfg = compose(config_name="edgetam.yaml")
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    # The two flags the fixed banks depend on, asserted rather than assumed.
    assert model.add_tpos_enc_to_obj_ptrs is False, (
        "the object pointers carry a temporal embedding in this checkpoint; the "
        "fixed 16-slot bank has no place for the conditioning pointer's distance "
        "(report §7 N-4)"
    )
    assert model.use_multimask_token_for_obj_ptr is True
    parity_gate(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model = load_model(args.repo, args.checkpoint)

    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16)
    torch.onnx.export(
        ImageEncoder(model),
        (image,),
        str(ARTIFACTS / "edgetam_v1_image_encoder.onnx"),
        input_names=["normalized_image"],
        output_names=[
            "vision_features",
            "high_resolution_features_0",
            "high_resolution_features_1",
        ],
        opset_version=args.opset,
        dynamo=False,
    )

    with torch.no_grad():
        vision, high0, high1 = ImageEncoder(model)(image)
    torch.onnx.export(
        PromptMaskDecoder(model),
        (
            vision,
            high0,
            high1,
            torch.zeros(1, PROMPT_TOKENS, 2, dtype=torch.float16),
            torch.zeros(1, PROMPT_TOKENS, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
            torch.zeros(1, 1, LOW_RESOLUTION, LOW_RESOLUTION, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float16),
        ),
        str(ARTIFACTS / "edgetam_v1_prompt_mask_decoder.onnx"),
        input_names=[
            "vision_features",
            "high_resolution_features_0",
            "high_resolution_features_1",
            "point_coordinates",
            "point_labels",
            "add_no_memory_embedding",
            "apply_object_score_mask_gate",
            "mask_input",
            "has_mask_input",
        ],
        output_names=[
            "candidate_masks",
            "iou_predictions",
            "candidate_object_pointers",
            "object_score_logits",
        ],
        dynamic_axes={
            "point_coordinates": {1: "tokens"},
            "point_labels": {1: "tokens"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    with torch.no_grad():
        mask = torch.zeros(1, 1, LOW_RESOLUTION, LOW_RESOLUTION, dtype=torch.float16)
    torch.onnx.export(
        MemoryEncoder(model),
        (vision, mask),
        str(ARTIFACTS / "edgetam_v1_memory_encoder.onnx"),
        input_names=["raw_vision_features", "selected_low_resolution_mask"],
        output_names=["memory_features"],
        opset_version=args.opset,
        dynamo=False,
    )

    (ARTIFACTS / "edgetam_metadata.json").write_text(
        json.dumps(
            {
                "upstream": "https://github.com/facebookresearch/EdgeTAM",
                "license": "Apache-2.0",
                "image_size": IMAGE_SIZE,
                "low_resolution_mask": LOW_RESOLUTION,
                "candidates": 4,
                "boundary_dtype": "float16",
                "graphs_exported": [
                    "edgetam_v1_image_encoder",
                    "edgetam_v1_prompt_mask_decoder",
                    "edgetam_v1_memory_encoder",
                ],
                "graphs_blocked": {
                    "edgetam_v1_memory_attention": "needs a key-padding mask threaded "
                    "through MemoryAttention for the fixed 7/16-slot bank (report §7 N-4)",
                },
                "memory_encoder_pinned_branches": {
                    "is_mask_from_pts": False,
                    "non_overlap_masks_for_mem_enc": False,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"edgetam export: {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
