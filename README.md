# gr00t_depth: depth fusion for Isaac GR00T N1.7

Adds a depth encoder and a depth fusion adapter to GR00T N1.7's action head,
for better grasping. The VLM is untouched; the adapter replaces the spot where
GR00T applies a LayerNorm to the VLM tokens before the DiT.

```
RGB + instruction ─► VLM (frozen) ─► LayerNorm ─► [depth fusion adapter] ─► DiT ─► actions
                                                         ▲
depth ─► point map [X,Y,Z,valid] ─► depth encoder ─► depth tokens
```

## How it works

- **Depth encoder** (`encoder.py`, ~2.5M params): a ResNet-style CNN over a
  4-channel point map (XYZ in meters + validity), pooled to a 16×16 grid of
  tokens. GroupNorm instead of BatchNorm, so small batches and eval mode
  behave the same.
- **Fusion adapter** (`adapter.py`, ~13.8M params, 2 layers): each VLM
  *image* token cross-attends to the depth tokens of its paired camera and
  adds the result through a gate initialized to 0:
  `token + tanh(g) · CrossAttention(token, depth)`, then the same for a
  feed-forward. Tokens keep their count and shape, so the DiT is unchanged.
- **Location matching** (`layout.py`): from `image_mask` and
  `image_grid_thw`, every VLM image token gets the (x, y) center of the patch
  it covers. Image and depth tokens share 2D position encodings, plus a
  learned locality bias that starts each token looking at nearby depth.
- **Text tokens, unpaired images, and cameras with no valid depth** are left
  exactly as they were.
- **No depth in the batch → plain GR00T.**

## Guarantees (tested)

All tests run against NVIDIA's real `Gr00tN1d7ActionHead` with a tiny config on CPU:

| Test | What it proves |
|---|---|
| `test_identity_training_forward` | With gates at 0, the training loss is **bit-identical** to unmodified GR00T |
| `test_identity_inference` | With gates at 0, predicted actions are **bit-identical** |
| `test_no_depth_key_…` | Without a `depth` input, behavior is unmodified GR00T even with open gates |
| `test_open_gates_change_only_image_tokens` | Depth changes image tokens only; text and padding untouched |
| `test_unpaired_image_is_untouched`, `test_view_without_valid_depth_…` | Images without (valid) depth are untouched, no NaNs |
| `test_gradients_reach_gates_then_encoder` | Gates learn from step 0; once open, the encoder trains |
| `test_configure_trainable`, `test_param_groups…` | The right parts freeze/unfreeze |
| `test_checkpoint_roundtrip` | Saved depth modules reload to identical outputs |

```bash
GR00T_REPO=/path/to/Isaac-GR00T python -m pytest gr00t_depth/tests -q
```

## Usage

```python
from gr00t_depth import (DepthFusionConfig, attach_depth_fusion, configure_trainable,
                         param_groups, save_depth_modules, load_depth_modules)

model = ...  # GR00T N1.7 loaded the usual way (Gr00tN1d7)

attach_depth_fusion(model, DepthFusionConfig(
    image_to_depth_view=[0, -1],   # e.g. camera 0 has depth, camera 1 doesn't
))

# What trains: depth modules + per-robot embodiment layers; DiT frozen.
# dit= "none" | "image_cross_attn" | "cross_attn" | "all"
print(configure_trainable(model, dit="none"))
opt = torch.optim.AdamW(param_groups(model, lr=1e-4, depth_lr=1e-3))

# training batches need a "depth" key: (B, V, 4, H, W) point maps
# watch model.action_head.depth_fusion.gate_values(): growing gates = depth is being used

save_depth_modules(model, "checkpoints/run1")      # writes depth_fusion.pt
# later: load GR00T as usual, then
load_depth_modules(model, "checkpoints/run1")
```

Making the depth input from a sensor frame:

```python
from gr00t_depth import Intrinsics, make_depth_input
x = make_depth_input(depth_uint16_mm, Intrinsics(fx, fy, cx, cy), depth_scale=0.001,
                     min_depth=0.1, max_depth=2.0)    # (4, H, W)
```

Compute the point map at the camera's native resolution, then apply the
**same crop/resize as the RGB image, with nearest-neighbour sampling**
(`resize_depth_input`). XYZ stays metrically correct through any crop or
resize, because those only choose which pixels survive.

## Files

| File | |
|---|---|
| `config.py` | `DepthFusionConfig` |
| `geometry.py` | depth → point map, nearest resize |
| `encoder.py` | `DepthEncoder` |
| `layout.py` | VLM image token → (image index, x, y) |
| `adapter.py` | `DepthFusionAdapter` |
| `patch.py` | `attach_depth_fusion`: hooks into the action head without editing NVIDIA's files |
| `trainable.py` | `configure_trainable`, `param_groups` |
| `checkpoint.py` | save/load the depth modules as a side file |

## Size

Full-size config: about 16M trainable parameters (2.5M encoder + 13.8M
adapter), versus GR00T's 3B.

## Not done yet (next steps)

1. **Data pipeline:** carry depth through GR00T's processor and collator as
   the `depth` key, replaying the RGB crop/resize/rotation on it
   (`apply_with_replay`, as the mask-guided example does) and skipping
   color jitter.
2. **Training script:** a wrapper around GR00T's fine-tuning that attaches
   the modules, applies `configure_trainable`, uses `param_groups`, logs
   gate values, and saves `depth_fusion.pt` with each checkpoint.
3. **Policy server:** accept `depth` in observations and call
   `load_depth_modules` after loading the policy.
4. **Full-size check:** run the identity test once on the real 3B
   checkpoint (needs a 16 GB GPU, or a CPU machine with plenty of RAM).
