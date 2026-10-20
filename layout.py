"""Where each VLM image token sits in its image.

The Qwen3-VL backbone splits each image into patches, merges 2x2 patches into
one token, and lays an image's tokens out row-major. `image_grid_thw` gives
each image's (t, h, w) size in patches, in the order the images appear across
the whole batch; `image_mask` marks which sequence positions are image tokens.
From these we recover, for every image token, which image it came from and the
(x, y) center of its region, so it can be matched with depth from the same spot.
"""
from __future__ import annotations

import torch


def image_token_layout(image_mask: torch.Tensor, image_grid_thw: torch.Tensor,
                       merge_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    image_mask     : (B, S) bool, True at image tokens
    image_grid_thw : (N_images_in_batch, 3) long, patches per image (t, h, w)

    Returns
        coords      : (B, S, 2) float, (x, y) in [0, 1]; 0 at non-image tokens
        image_index : (B, S) long, index of the image within its sample; -1 at
                      non-image tokens
    """
    B, S = image_mask.shape
    device = image_mask.device
    coords = torch.zeros(B, S, 2, device=device)
    image_index = torch.full((B, S), -1, dtype=torch.long, device=device)

    grid = image_grid_thw.tolist()
    row = 0
    for b in range(B):
        positions = image_mask[b].nonzero(as_tuple=True)[0]
        need, start, k = positions.numel(), 0, 0
        while start < need:
            if row >= len(grid):
                raise ValueError("image_grid_thw has fewer images than image_mask has image tokens")
            t, h, w = grid[row]
            gh, gw = h // merge_size, w // merge_size
            n = t * gh * gw
            if start + n > need:
                raise ValueError(f"sample {b}: image {k} needs {n} tokens but only {need - start} remain; "
                                 "check merge_size and that image_grid_thw matches image_mask")
            ys, xs = torch.meshgrid((torch.arange(gh, device=device) + 0.5) / gh,
                                    (torch.arange(gw, device=device) + 0.5) / gw, indexing="ij")
            xy = torch.stack([xs, ys], dim=-1).reshape(-1, 2).repeat(t, 1)
            idx = positions[start:start + n]
            coords[b, idx] = xy
            image_index[b, idx] = k
            start, k, row = start + n, k + 1, row + 1
    if row != len(grid):
        raise ValueError(f"image_grid_thw has {len(grid)} images but image_mask only accounts for {row}")
    return coords, image_index
