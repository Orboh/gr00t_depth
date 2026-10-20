"""Settings for the depth encoder and the depth fusion adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DepthFusionConfig:
    # Width of the VLM tokens the adapter refines. 2048 for GR00T N1.7
    # (Cosmos-Reason2-2B hidden size, `backbone_embedding_dim` in its config).
    backbone_dim: int = 2048

    # ---- depth input -------------------------------------------------------
    # Channels of the depth input: X, Y, Z (meters, camera frame) + valid mask.
    in_channels: int = 4
    # XYZ are divided by this before the encoder, so values sit roughly in [-1, 1].
    xyz_scale: float = 1.0

    # ---- depth encoder -----------------------------------------------------
    encoder_width: int = 64
    # Depth tokens per side; the encoder output is pooled to this grid.
    depth_token_grid: int = 16
    depth_dim: int = 512

    # ---- fusion adapter ----------------------------------------------------
    attn_dim: int = 512
    n_heads: int = 8
    n_layers: int = 2
    ff_hidden: int = 1024
    dropout: float = 0.0
    # Number of sin/cos frequencies per axis for 2D position encodings.
    pos_frequencies: int = 16
    # Bias attention toward depth tokens near the image token's location:
    # logit bias = -dist^2 / (2 sigma^2), sigma learned per head, starting here
    # (in normalized image coordinates, 0..1). None disables the bias.
    locality_sigma_init: float | None = 0.15

    # ---- mapping VLM image tokens to depth views ----------------------------
    # Qwen3-VL merges 2x2 vision patches into one VLM token.
    merge_size: int = 2
    # For each image in a sample (in the order the processor feeds them to the
    # VLM: frame-major, then camera view), which depth view it pairs with, or -1
    # for none. None means image i pairs with depth view i when that view exists.
    image_to_depth_view: list[int] | None = None

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DepthFusionConfig":
        return cls(**d)
