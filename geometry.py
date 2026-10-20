"""Turning raw depth images into the encoder's input.

The encoder takes a 4-channel "point map": for every pixel, its X, Y, Z
position in meters in the camera frame, plus a validity flag. Computing XYZ
at the camera's native resolution, before any crop or resize, means the
values stay metrically correct whatever augmentation is applied afterwards:
crops and resizes only choose which pixels survive. Apply the same geometric
transform as the RGB image, with nearest-neighbour sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def depth_to_points(depth_m: torch.Tensor, K: Intrinsics) -> torch.Tensor:
    """depth_m: (..., H, W) depth in meters -> (..., 3, H, W) XYZ in meters."""
    H, W = depth_m.shape[-2:]
    v, u = torch.meshgrid(
        torch.arange(H, dtype=depth_m.dtype, device=depth_m.device),
        torch.arange(W, dtype=depth_m.dtype, device=depth_m.device),
        indexing="ij",
    )
    z = depth_m
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    return torch.stack([x, y, z], dim=-3)


def make_depth_input(depth_raw: np.ndarray | torch.Tensor, K: Intrinsics, depth_scale: float = 0.001,
                     min_depth: float = 0.1, max_depth: float = 2.0) -> torch.Tensor:
    """Raw sensor depth -> (4, H, W) float32 point map [X, Y, Z, valid].

    depth_raw   : (H, W) integer or float depth, e.g. uint16 millimeters
    depth_scale : multiply raw values by this to get meters (0.001 for mm)
    min/max     : working range; pixels outside it, and zeros (sensor holes),
                  are marked invalid and their XYZ set to 0
    """
    d = torch.as_tensor(np.asarray(depth_raw), dtype=torch.float32) * depth_scale
    valid = (d > min_depth) & (d < max_depth) & torch.isfinite(d)
    d = torch.where(valid, d, torch.zeros_like(d))
    xyz = depth_to_points(d, K) * valid
    return torch.cat([xyz, valid[None].float()], dim=0)


def resize_depth_input(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a (..., 4, H, W) point map with nearest-neighbour sampling, so
    no pixel is an average across an object edge (which would invent points
    in mid-air)."""
    lead = x.shape[:-3]
    y = F.interpolate(x.reshape(-1, *x.shape[-3:]), size=size, mode="nearest")
    return y.reshape(*lead, *y.shape[-3:])
