"""Depth encoder: point map -> grid of depth tokens."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DepthFusionConfig


def _gn(c: int) -> nn.GroupNorm:
    # GroupNorm rather than BatchNorm: behaves the same with tiny batches and
    # in eval mode, and doesn't need running statistics.
    return nn.GroupNorm(min(32, c // 4), c)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.n1 = _gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.n2 = _gn(c_out)
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), _gn(c_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.conv1(x)))
        return F.gelu(self.n2(self.conv2(h)) + self.skip(x))


class DepthEncoder(nn.Module):
    """ResNet-style CNN over a point map [X, Y, Z, valid].

    Input : (B, V, 4, H, W), V = number of depth views
    Output: tokens (B, V, G*G, depth_dim), row-major over a G x G grid
            coords (G*G, 2), each token's (x, y) center in [0, 1]
            valid  (B, V, G*G), False where the token's area had no valid depth
    """

    def __init__(self, cfg: DepthFusionConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.encoder_width
        self.stem = nn.Sequential(nn.Conv2d(cfg.in_channels, w, 5, 2, 2, bias=False), _gn(w), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock(w, w, 1),
            ResBlock(w, 2 * w, 2),
            ResBlock(2 * w, 4 * w, 2),
            ResBlock(4 * w, 4 * w, 1),
        )
        self.proj = nn.Conv2d(4 * w, cfg.depth_dim, 1)
        G = cfg.depth_token_grid
        ys, xs = torch.meshgrid((torch.arange(G) + 0.5) / G, (torch.arange(G) + 0.5) / G, indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=-1).reshape(-1, 2), persistent=False)

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, V, C, H, W = depth.shape
        x = depth.reshape(B * V, C, H, W)
        valid = x[:, 3:4]
        xyz = x[:, :3] / self.cfg.xyz_scale * valid
        x = torch.cat([xyz, valid], dim=1)

        G = self.cfg.depth_token_grid
        f = self.stages(self.stem(x))
        f = F.adaptive_avg_pool2d(f, G)
        tokens = self.proj(f).flatten(2).transpose(1, 2)            # (B*V, G*G, D)
        token_valid = F.adaptive_avg_pool2d(valid.float(), G).flatten(1) > 0
        return (tokens.reshape(B, V, G * G, -1),
                self.coords.to(tokens.dtype),
                token_valid.reshape(B, V, G * G))
