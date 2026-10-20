"""Choose which parts of GR00T + depth train, and build optimizer groups.

Training options for the DiT (`dit=`):
    "none"             DiT frozen (cheapest)
    "image_cross_attn" only the DiT blocks that read image tokens
    "cross_attn"       all cross-attention blocks (image and text)
    "all"              the whole DiT
"""
from __future__ import annotations

import torch.nn as nn

from .patch import get_action_head, is_attached

EMBODIMENT_MODULES = ("state_encoder", "action_encoder", "action_decoder")
DIT_CHOICES = ("none", "image_cross_attn", "cross_attn", "all")


def _dit_blocks(head: nn.Module, which: str) -> list[nn.Module]:
    blocks = list(head.model.transformer_blocks)
    if which == "all":
        return [head.model]
    # Even-indexed blocks cross-attend to VLM tokens, odd ones self-attend.
    cross = [i for i in range(0, len(blocks), 2)]
    if which == "image_cross_attn":
        if not head.config.use_alternate_vl_dit:
            raise ValueError("image_cross_attn needs AlternateVLDiT (use_alternate_vl_dit=True)")
        n = head.config.attend_text_every_n_blocks
        # AlternateVLDiT: block i reads text tokens when i % (2n) == 0, image tokens otherwise.
        cross = [i for i in cross if i % (2 * n) != 0]
    return [blocks[i] for i in cross]


def configure_trainable(model: nn.Module, dit: str = "none", embodiment: bool = True,
                        depth: bool = True, vlln: bool = False) -> dict[str, int]:
    """Freeze everything, then unfreeze the chosen parts. Returns trainable
    parameter counts per part."""
    if dit not in DIT_CHOICES:
        raise ValueError(f"dit must be one of {DIT_CHOICES}")
    head = get_action_head(model)
    model.requires_grad_(False)
    counts: dict[str, int] = {}

    def enable(name: str, modules: list[nn.Module]):
        n = 0
        for m in modules:
            m.requires_grad_(True)
            n += sum(p.numel() for p in m.parameters())
        counts[name] = n

    if depth:
        if not is_attached(head):
            raise RuntimeError("attach_depth_fusion() first")
        enable("depth", [head.depth_encoder, head.depth_fusion])
    if embodiment:
        mods = [getattr(head, n) for n in EMBODIMENT_MODULES]
        if getattr(head.config, "add_pos_embed", False):
            mods.append(head.position_embedding)
        enable("embodiment", mods)
    if dit != "none":
        enable(f"dit_{dit}", _dit_blocks(head, dit))
    if vlln:
        enable("vlln", [head.vlln, head.vl_self_attention])

    # GR00T puts frozen modules in eval mode during training based on these
    # flags (dropout off); keep them consistent with what we froze.
    head.tune_projector = embodiment
    head.tune_diffusion_model = dit != "none"
    head.tune_vlln = vlln
    counts["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def param_groups(model: nn.Module, lr: float, depth_lr: float | None = None,
                 weight_decay: float = 1e-5) -> list[dict]:
    """Optimizer groups:
      * depth modules, optionally at their own (usually higher) learning rate,
        since they start from scratch;
      * per-embodiment layers with NO weight decay: their weights are stacked
        across all 32 robot slots, so decay would slowly shrink the other
        robots' slots even though only yours gets gradients;
      * everything else trainable.
    """
    head = get_action_head(model)
    depth_ids, emb_ids = set(), set()
    if is_attached(head):
        depth_ids = {id(p) for m in (head.depth_encoder, head.depth_fusion) for p in m.parameters()}
    emb_ids = {id(p) for n in EMBODIMENT_MODULES for p in getattr(head, n).parameters()}

    groups = {"depth": [], "embodiment": [], "other": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = "depth" if id(p) in depth_ids else "embodiment" if id(p) in emb_ids else "other"
        groups[key].append(p)
    out = []
    if groups["depth"]:
        out.append({"name": "depth", "params": groups["depth"], "lr": depth_lr or lr,
                    "weight_decay": weight_decay})
    if groups["embodiment"]:
        out.append({"name": "embodiment", "params": groups["embodiment"], "lr": lr, "weight_decay": 0.0})
    if groups["other"]:
        out.append({"name": "other", "params": groups["other"], "lr": lr, "weight_decay": weight_decay})
    return out
