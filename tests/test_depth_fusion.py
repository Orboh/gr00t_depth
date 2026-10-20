"""Tests for the depth encoder, fusion adapter and GR00T integration.

    GR00T_REPO=/path/to/Isaac-GR00T python -m pytest gr00t_depth/tests -q
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

gr00t_cfg = pytest.importorskip("gr00t.configs.model.gr00t_n1d7")
gr00t_model = pytest.importorskip("gr00t.model.gr00t_n1d7.gr00t_n1d7")
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from gr00t_depth import (DepthFusionConfig, Intrinsics, attach_depth_fusion, configure_trainable,  # noqa: E402
                         image_token_layout, load_depth_modules, make_depth_input, param_groups,
                         resize_depth_input, save_depth_modules)
from geometry import depth_to_points  # noqa: E402

BB = 64            # backbone_embedding_dim of the tiny model
GRID = (1, 8, 8)   # each image: 8x8 patches -> 4x4 = 16 VLM tokens after 2x2 merge
N_IMG_TOK = 16


# ----------------------------------------------------------------------------- fixtures
def tiny_gr00t_config():
    return gr00t_cfg.Gr00tN1d7Config(
        backbone_embedding_dim=BB, hidden_size=32, input_embedding_dim=48,
        max_state_dim=8, max_action_dim=6, action_horizon=4, max_num_embodiments=4,
        num_inference_timesteps=2,
        diffusion_model_cfg={"positional_embeddings": None, "num_layers": 4, "num_attention_heads": 2,
                             "attention_head_dim": 24, "norm_type": "ada_norm", "dropout": 0.0,
                             "final_dropout": True, "output_dim": 32, "interleave_self_attention": True})


def tiny_depth_config(**kw):
    base = dict(backbone_dim=BB, encoder_width=8, depth_token_grid=4, depth_dim=32, attn_dim=32,
                n_heads=4, n_layers=2, ff_hidden=64)
    base.update(kw)
    return DepthFusionConfig(**base)


def make_head(seed: int = 0):
    torch.manual_seed(seed)
    return gr00t_model.Gr00tN1d7ActionHead(tiny_gr00t_config()).eval()


def make_depth(B=2, V=2, H=64, W=64, seed=0):
    """A floor plane with a box on it, per view, as a realistic point map."""
    rng = np.random.default_rng(seed)
    K = Intrinsics(fx=60.0, fy=60.0, cx=W / 2, cy=H / 2)
    out = torch.zeros(B, V, 4, H, W)
    for b in range(B):
        for v in range(V):
            d = np.full((H, W), 900.0 + 50 * v)
            r, c = rng.integers(10, 40, size=2)
            d[r:r + 16, c:c + 16] -= 200 + 20 * b          # box closer to camera
            d[rng.random((H, W)) < 0.05] = 0               # sensor holes
            out[b, v] = make_depth_input(d.astype(np.uint16), K)
    return out


def layout_sample(n_text_before: int, n_text_between: int, n_text_after: int, n_pad: int):
    return ([0] * n_text_before + [1] * N_IMG_TOK + [0] * n_text_between + [1] * N_IMG_TOK
            + [0] * n_text_after, n_pad)


def make_inputs(seed: int = 0, with_depth: bool = True):
    g = torch.Generator().manual_seed(seed)
    s0, p0 = layout_sample(5, 3, 4, 0)
    s1, p1 = layout_sample(3, 2, 0, 0)
    S = max(len(s0) + p0, len(s1) + p1)
    image_mask = torch.zeros(2, S, dtype=torch.bool)
    attn = torch.zeros(2, S, dtype=torch.bool)
    image_mask[0, :len(s0)] = torch.tensor(s0, dtype=torch.bool)
    image_mask[1, :len(s1)] = torch.tensor(s1, dtype=torch.bool)
    attn[0, :len(s0)] = True
    attn[1, :len(s1)] = True
    backbone = BatchFeature(data={
        "backbone_features": torch.randn(2, S, BB, generator=g),
        "backbone_attention_mask": attn,
        "image_mask": image_mask,
    })
    action = {
        "state": torch.randn(2, 1, 8, generator=g),
        "action": torch.randn(2, 4, 6, generator=g),
        "action_mask": torch.ones(2, 4, 6),
        "embodiment_id": torch.tensor([1, 1]),
        "image_grid_thw": torch.tensor([GRID] * 4),
    }
    if with_depth:
        action["depth"] = make_depth(seed=seed)
    return backbone, BatchFeature(data=action)


def with_depth(head, **kw):
    h = copy.deepcopy(head)
    attach_depth_fusion(h, tiny_depth_config(**kw))
    return h.eval()


def open_gates(head, value=0.5):
    with torch.no_grad():
        for layer in head.depth_fusion.layers:
            layer.gate_attn.fill_(value)
            layer.gate_ff.fill_(value)


# ------------------------------------------------------------------------------- geometry
def test_point_map_geometry():
    K = Intrinsics(fx=100.0, fy=100.0, cx=2.0, cy=1.0)
    d = torch.full((3, 5), 2.0)
    p = depth_to_points(d, K)
    assert p.shape == (3, 3, 5)
    assert torch.allclose(p[:, 1, 2], torch.tensor([0.0, 0.0, 2.0]))     # principal point
    assert torch.allclose(p[:, 1, 4], torch.tensor([0.04, 0.0, 2.0]))    # 2 px right -> x = 2/100*2


def test_invalid_pixels_are_zeroed_and_flagged():
    raw = np.array([[0, 500, 5000]], dtype=np.uint16)   # hole, valid, beyond max_depth
    x = make_depth_input(raw, Intrinsics(1, 1, 0, 0), max_depth=2.0)
    assert x[3].tolist() == [[0.0, 1.0, 0.0]]
    assert x[:3, 0, 0].abs().sum() == 0 and x[:3, 0, 2].abs().sum() == 0
    y = resize_depth_input(x[None], (2, 6))
    assert set(y[0, 3].unique().tolist()) <= {0.0, 1.0}  # nearest: validity stays binary


# --------------------------------------------------------------------------------- layout
def test_image_token_layout():
    backbone, action = make_inputs()
    xy, idx = image_token_layout(backbone["image_mask"], action["image_grid_thw"], merge_size=2)
    assert (idx[0, :5] == -1).all() and (idx[0, 5:21] == 0).all() and (idx[0, 21:24] == -1).all()
    assert (idx[0, 24:40] == 1).all()
    assert torch.allclose(xy[0, 5], torch.tensor([0.125, 0.125]))     # top-left of a 4x4 grid
    assert torch.allclose(xy[0, 8], torch.tensor([0.875, 0.125]))     # end of first row
    assert torch.allclose(xy[0, 20], torch.tensor([0.875, 0.875]))    # bottom-right


def test_layout_mismatch_raises():
    backbone, action = make_inputs()
    with pytest.raises(ValueError):
        image_token_layout(backbone["image_mask"], action["image_grid_thw"][:3])


# ------------------------------------------------------------------------- the identity test
def test_identity_training_forward():
    """Gates at zero: training loss is bit-identical to unmodified GR00T."""
    ref = make_head()
    dep = with_depth(ref)
    torch.manual_seed(123)
    out_ref = ref(*make_inputs())
    torch.manual_seed(123)
    out_dep = dep(*make_inputs())
    assert torch.equal(out_ref["loss"], out_dep["loss"])
    assert torch.equal(out_ref["backbone_features"], out_dep["backbone_features"])


def test_identity_inference():
    """Gates at zero: predicted actions are bit-identical to unmodified GR00T."""
    ref = make_head()
    dep = with_depth(ref)
    b, a = make_inputs()
    a.pop("action")
    b2, a2 = make_inputs()
    a2.pop("action")
    torch.manual_seed(7)
    act_ref = ref.get_action(b, a)["action_pred"]
    torch.manual_seed(7)
    act_dep = dep.get_action(b2, a2)["action_pred"]
    assert torch.equal(act_ref, act_dep)


def test_no_depth_key_behaves_like_gr00t_even_with_open_gates():
    ref = make_head()
    dep = with_depth(ref)
    open_gates(dep)
    torch.manual_seed(1)
    l_ref = ref(*make_inputs(with_depth=False))["loss"]
    torch.manual_seed(1)
    l_dep = dep(*make_inputs(with_depth=False))["loss"]
    assert torch.equal(l_ref, l_dep)


# --------------------------------------------------------------------------- fusion behavior
def test_open_gates_change_only_image_tokens():
    dep = with_depth(make_head())
    open_gates(dep)
    b, a = make_inputs()
    before = b["backbone_features"].clone()
    dep._encode_features(b, a)  # runs the patched process_backbone_output
    after = b["backbone_features"]
    after_ln = dep.vlln(before)
    img = b["image_mask"]
    assert torch.equal(after[~img], after_ln[~img])        # text and padding untouched
    assert not torch.allclose(after[img], after_ln[img])   # image tokens refined


def test_unpaired_image_is_untouched():
    dep = with_depth(make_head(), image_to_depth_view=[0, -1])
    open_gates(dep)
    b, a = make_inputs()
    after_ln = dep.vlln(b["backbone_features"].clone())
    dep._encode_features(b, a)
    second = torch.zeros_like(b["image_mask"])
    second[0, 24:40] = True
    assert torch.equal(b["backbone_features"][second], after_ln[second])
    first = torch.zeros_like(b["image_mask"])
    first[0, 5:21] = True
    assert not torch.allclose(b["backbone_features"][first], after_ln[first])


def test_view_without_valid_depth_is_untouched():
    dep = with_depth(make_head())
    open_gates(dep)
    b, a = make_inputs()
    a["depth"][:, 1, 3] = 0           # view 1 has no valid pixels
    after_ln = dep.vlln(b["backbone_features"].clone())
    dep._encode_features(b, a)
    second = torch.zeros_like(b["image_mask"])
    second[0, 24:40] = True
    assert torch.equal(b["backbone_features"][second], after_ln[second])
    assert torch.isfinite(b["backbone_features"]).all()


def test_gradients_reach_gates_then_encoder():
    dep = with_depth(make_head())
    configure_trainable(dep, dit="none")
    dep.train()
    torch.manual_seed(0)
    dep(*make_inputs())["loss"].backward()
    gate_grads = [l.gate_attn.grad for l in dep.depth_fusion.layers]
    assert all(g is not None and g.abs().sum() > 0 for g in gate_grads), "gates must learn from step 0"

    dep.zero_grad()
    open_gates(dep, 0.3)
    torch.manual_seed(0)
    dep(*make_inputs())["loss"].backward()
    enc_grad = sum(p.grad.abs().sum() for p in dep.depth_encoder.parameters() if p.grad is not None)
    assert enc_grad > 0, "once a gate opens, the depth encoder must receive gradients"


# ------------------------------------------------------------------------- trainable setup
def test_configure_trainable():
    dep = with_depth(make_head())
    counts = configure_trainable(dep, dit="none")
    assert not any(p.requires_grad for p in dep.model.parameters())
    assert all(p.requires_grad for p in dep.depth_fusion.parameters())
    assert all(p.requires_grad for p in dep.state_encoder.parameters())
    assert not any(p.requires_grad for p in dep.vlln.parameters())
    assert counts["total_trainable"] == counts["depth"] + counts["embodiment"]

    configure_trainable(dep, dit="image_cross_attn")
    blocks = dep.model.transformer_blocks
    # 4 blocks, attend_text_every_n_blocks=2: 0 = text cross-attn, 1/3 = self-attn, 2 = image cross-attn
    assert [any(p.requires_grad for p in b.parameters()) for b in blocks] == [False, False, True, False]

    configure_trainable(dep, dit="cross_attn")
    assert [any(p.requires_grad for p in b.parameters()) for b in blocks] == [True, False, True, False]


def test_param_groups_no_decay_on_embodiment_slots():
    dep = with_depth(make_head())
    configure_trainable(dep, dit="none")
    groups = {g["name"]: g for g in param_groups(dep, lr=1e-4, depth_lr=1e-3, weight_decay=0.01)}
    assert groups["embodiment"]["weight_decay"] == 0.0
    assert groups["depth"]["lr"] == 1e-3
    n_trainable = sum(p.numel() for p in dep.parameters() if p.requires_grad)
    assert sum(p.numel() for g in groups.values() for p in g["params"]) == n_trainable


# ---------------------------------------------------------------------------- checkpointing
def test_checkpoint_roundtrip(tmp_path):
    dep = with_depth(make_head(), image_to_depth_view=[0, 1])
    open_gates(dep, 0.2)
    path = save_depth_modules(dep, tmp_path)
    fresh = make_head()
    load_depth_modules(fresh, path)
    fresh.eval()
    assert fresh.depth_fusion_config.image_to_depth_view == [0, 1]
    b1, a1 = make_inputs()
    b2, a2 = make_inputs()
    dep._encode_features(b1, a1)
    fresh._encode_features(b2, a2)
    assert torch.equal(b1["backbone_features"], b2["backbone_features"])


def test_attach_guards():
    dep = with_depth(make_head())
    with pytest.raises(RuntimeError):
        attach_depth_fusion(dep, tiny_depth_config())
    with pytest.raises(ValueError):
        attach_depth_fusion(make_head(), DepthFusionConfig(backbone_dim=2048))
