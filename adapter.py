"""Depth fusion adapter: refines the VLM's image tokens with depth.

Each VLM image token cross-attends to the depth tokens of its paired depth
view, and the result is added back through a gate that starts at exactly
zero:

    token <- token + tanh(g_attn) * CrossAttention(token, depth tokens)
    token <- token + tanh(g_ff)   * FeedForward(token)

Properties the GR00T integration relies on:
  * same number and shape of tokens out as in, so the DiT is unchanged;
  * with the gates at 0 the output equals the input exactly, so training
    starts from pretrained GR00T behavior;
  * only image tokens with a paired depth view are touched; the instruction
    (text) tokens pass through unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DepthFusionConfig


class FourierPos2D(nn.Module):
    """(x, y) in [0, 1] -> learned embedding via sin/cos features."""

    def __init__(self, n_freq: int, dim: int):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.proj = nn.Linear(4 * n_freq, dim)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        ang = xy[..., None] * self.freqs.to(xy.dtype)                       # (..., 2, F)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)      # (..., 4F)
        return self.proj(feats)


class FusionLayer(nn.Module):
    def __init__(self, cfg: DepthFusionConfig):
        super().__init__()
        d, a, h = cfg.backbone_dim, cfg.attn_dim, cfg.n_heads
        if a % h:
            raise ValueError("attn_dim must be divisible by n_heads")
        self.n_heads, self.head_dim = h, a // h
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(cfg.depth_dim)
        self.q = nn.Linear(d, a)
        self.k = nn.Linear(cfg.depth_dim, a)
        self.v = nn.Linear(cfg.depth_dim, a)
        self.q_pos = FourierPos2D(cfg.pos_frequencies, a)
        self.k_pos = FourierPos2D(cfg.pos_frequencies, a)
        self.out = nn.Linear(a, d)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.ff_hidden, d))
        self.dropout = cfg.dropout
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ff = nn.Parameter(torch.zeros(1))
        self.log_sigma = (nn.Parameter(torch.full((h,), math.log(cfg.locality_sigma_init)))
                          if cfg.locality_sigma_init else None)

    def forward(self, x: torch.Tensor, q_xy: torch.Tensor, update: torch.Tensor,
                kv: torch.Tensor, kv_xy: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, S, d)  VLM tokens
        q_xy    : (B, S, 2)  location of each VLM token
        update  : (B, S, 1)  1 where this token is refined, else 0
        kv      : (B, K, depth_dim) depth tokens (all depth views concatenated)
        kv_xy   : (K, 2)     location of each depth token
        allowed : (B, S, K)  bool, which depth tokens each VLM token may attend to
        """
        B, S, _ = x.shape
        K = kv.shape[1]
        h, hd = self.n_heads, self.head_dim

        q = (self.q(self.q_norm(x)) + self.q_pos(q_xy)).view(B, S, h, hd).transpose(1, 2)
        kvn = self.kv_norm(kv)
        k = (self.k(kvn) + self.k_pos(kv_xy)).view(B, K, h, hd).transpose(1, 2)
        v = self.v(kvn).view(B, K, h, hd).transpose(1, 2)

        bias = torch.zeros(B, h, S, K, dtype=q.dtype, device=q.device)
        if self.log_sigma is not None:
            d2 = ((q_xy[:, :, None, :] - kv_xy[None, None, :, :]) ** 2).sum(-1)       # (B, S, K)
            sigma2 = (2 * self.log_sigma).exp().to(q.dtype)                           # (h,)
            bias = bias - d2[:, None] / (2 * sigma2[None, :, None, None])
        bias = bias.masked_fill(~allowed[:, None], float("-inf"))

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = self.out(attn.transpose(1, 2).reshape(B, S, h * hd))
        x = x + update * torch.tanh(self.gate_attn).to(x.dtype) * attn
        x = x + update * torch.tanh(self.gate_ff).to(x.dtype) * self.ff(self.ff_norm(x))
        return x


class DepthFusionAdapter(nn.Module):
    def __init__(self, cfg: DepthFusionConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(FusionLayer(cfg) for _ in range(cfg.n_layers))

    def gate_values(self) -> dict[str, float]:
        """tanh(gate) per layer: 0 = depth unused, larger = depth used more."""
        out = {}
        for i, layer in enumerate(self.layers):
            out[f"layer{i}.attn"] = float(torch.tanh(layer.gate_attn.detach()).float())
            out[f"layer{i}.ff"] = float(torch.tanh(layer.gate_ff.detach()).float())
        return out

    def image_to_view(self, n_images: int, n_views: int, device) -> torch.Tensor:
        m = self.cfg.image_to_depth_view
        if m is None:
            m = [i if i < n_views else -1 for i in range(n_images)]
        if len(m) < n_images:
            raise ValueError(f"image_to_depth_view has {len(m)} entries but samples have {n_images} images")
        if any(v >= n_views for v in m[:n_images]):
            raise ValueError(f"image_to_depth_view refers to a depth view >= {n_views}")
        return torch.tensor(m[:n_images] + [-1], dtype=torch.long, device=device)  # last entry: index -1

    def forward(self, vl_tokens: torch.Tensor, image_index: torch.Tensor, image_xy: torch.Tensor,
                depth_tokens: torch.Tensor, depth_xy: torch.Tensor, depth_valid: torch.Tensor) -> torch.Tensor:
        """
        vl_tokens    : (B, S, d) VLM tokens (after GR00T's LayerNorm)
        image_index  : (B, S) image index within the sample, -1 for non-image tokens
        image_xy     : (B, S, 2)
        depth_tokens : (B, V, N, depth_dim)
        depth_xy     : (N, 2)
        depth_valid  : (B, V, N) bool
        """
        B, V, N, _ = depth_tokens.shape
        n_images = int(image_index.max().item()) + 1 if (image_index >= 0).any() else 0
        if n_images == 0:
            return vl_tokens
        view_of_image = self.image_to_view(n_images, V, vl_tokens.device)
        q_view = view_of_image[image_index]                                     # (B, S), -1 = none

        kv = depth_tokens.reshape(B, V * N, -1).to(vl_tokens.dtype)
        kv_xy = depth_xy.repeat(V, 1).to(vl_tokens.dtype)
        k_view = torch.arange(V, device=vl_tokens.device).repeat_interleave(N)  # (V*N,)
        allowed = (q_view[:, :, None] == k_view[None, None, :]) & depth_valid.reshape(B, 1, V * N)
        has_keys = allowed.any(-1)                                              # (B, S)
        update = has_keys.unsqueeze(-1).to(vl_tokens.dtype)
        # Rows with nothing to attend to (text tokens, unpaired images, views
        # with no valid depth) get a dummy all-allowed row to keep softmax
        # finite; their result is discarded by `update`.
        allowed = allowed | ~has_keys[:, :, None]

        x = vl_tokens
        for layer in self.layers:
            x = layer(x, image_xy.to(x.dtype), update, kv, kv_xy, allowed)
        return x
