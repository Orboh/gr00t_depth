"""Save and load the depth modules separately from the GR00T checkpoint.

GR00T's own loader doesn't know about the new modules, so they live in a
side file next to the checkpoint:

    load GR00T normally -> attach_depth_fusion() -> load_depth_modules()
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import DepthFusionConfig
from .patch import attach_depth_fusion, get_action_head, is_attached

FILENAME = "depth_fusion.pt"


def save_depth_modules(model: nn.Module, path: str | Path) -> Path:
    head = get_action_head(model)
    if not is_attached(head):
        raise RuntimeError("model has no depth modules attached")
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        path = path / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": head.depth_fusion_config.to_dict(),
        "depth_encoder": head.depth_encoder.state_dict(),
        "depth_fusion": head.depth_fusion.state_dict(),
    }, path)
    return path


def load_depth_modules(model: nn.Module, path: str | Path) -> nn.Module:
    """Attach (if needed) and load the depth modules into a GR00T model."""
    path = Path(path)
    if path.is_dir():
        path = path / FILENAME
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = DepthFusionConfig.from_dict(blob["config"])
    if not is_attached(model):
        attach_depth_fusion(model, cfg)
    head = get_action_head(model)
    head.depth_encoder.load_state_dict(blob["depth_encoder"])
    head.depth_fusion.load_state_dict(blob["depth_fusion"])
    return model
