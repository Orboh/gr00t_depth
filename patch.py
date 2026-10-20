"""Attach the depth encoder and fusion adapter to a GR00T N1.7 model.

    from gr00t_depth import DepthFusionConfig, attach_depth_fusion
    model = ...  # a loaded Gr00tN1d7, or its action_head
    attach_depth_fusion(model, DepthFusionConfig())

What it does, without editing NVIDIA's files:
  * registers `depth_encoder` and `depth_fusion` as submodules of the action
    head, so they move with `.to()`, appear in `.parameters()` and in the
    state dict;
  * wraps the action head's `forward` (training) and `_encode_features`
    (inference) so the depth tensor and `image_grid_thw` from the action
    inputs reach `process_backbone_output`;
  * extends `process_backbone_output`: after GR00T's own LayerNorm, the
    image tokens are refined with depth.

Depth reaches the action head as the input key `depth` with shape
(B, V, 4, H, W): GR00T passes the full input dict to the action head, so
anything the data pipeline puts in the batch arrives there. When a batch has
no `depth` key, the model behaves exactly like unmodified GR00T.
"""
from __future__ import annotations

import torch.nn as nn

from .adapter import DepthFusionAdapter
from .config import DepthFusionConfig
from .encoder import DepthEncoder
from .layout import image_token_layout

DEPTH_KEY = "depth"
_CTX_KEY = "_depth_ctx"


def get_action_head(model: nn.Module) -> nn.Module:
    return model.action_head if hasattr(model, "action_head") else model


def is_attached(model: nn.Module) -> bool:
    return getattr(get_action_head(model), "_depth_fusion_attached", False)


def attach_depth_fusion(model: nn.Module, cfg: DepthFusionConfig | None = None) -> nn.Module:
    cfg = cfg or DepthFusionConfig()
    head = get_action_head(model)
    if is_attached(head):
        raise RuntimeError("depth fusion is already attached to this model")
    bb_dim = head.config.backbone_embedding_dim
    if cfg.backbone_dim != bb_dim:
        raise ValueError(f"cfg.backbone_dim={cfg.backbone_dim} but the model's backbone_embedding_dim={bb_dim}")

    ref = next(head.parameters())
    head.depth_encoder = DepthEncoder(cfg).to(device=ref.device, dtype=ref.dtype)
    head.depth_fusion = DepthFusionAdapter(cfg).to(device=ref.device, dtype=ref.dtype)
    head.depth_fusion_config = cfg

    orig_forward = head.forward
    orig_encode = head._encode_features
    orig_process = head.process_backbone_output

    def _stash(backbone_output, action_input):
        depth = action_input.get(DEPTH_KEY) if hasattr(action_input, "get") else None
        if depth is not None:
            grid = action_input.get("image_grid_thw")
            if grid is None:
                raise KeyError("depth was given but `image_grid_thw` is missing from the inputs")
            backbone_output[_CTX_KEY] = (depth, grid)

    def forward(backbone_output, action_input):
        _stash(backbone_output, action_input)
        return orig_forward(backbone_output, action_input)

    def _encode_features(backbone_output, action_input):
        _stash(backbone_output, action_input)
        return orig_encode(backbone_output, action_input)

    def process_backbone_output(backbone_output):
        backbone_output = orig_process(backbone_output)       # GR00T's own LayerNorm
        ctx = backbone_output.pop(_CTX_KEY, None) if hasattr(backbone_output, "pop") else None
        if ctx is None:
            return backbone_output
        depth, grid = ctx
        feats = backbone_output["backbone_features"]
        image_xy, image_index = image_token_layout(backbone_output["image_mask"], grid, cfg.merge_size)
        tokens, depth_xy, depth_valid = head.depth_encoder(depth.to(feats.dtype))
        backbone_output["backbone_features"] = head.depth_fusion(
            feats, image_index, image_xy, tokens, depth_xy, depth_valid)
        return backbone_output

    head.forward = forward
    head._encode_features = _encode_features
    head.process_backbone_output = process_backbone_output
    head._depth_fusion_attached = True
    return model
