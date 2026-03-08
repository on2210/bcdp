# bcdp/intervention/mlp_weight_mask.py
from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import torch
from bcdp.model.hf_handle import HuggingFaceCausalLMHandle


def _as_set(xs: Sequence[int]) -> set[int]:
    return set(int(x) for x in xs)


@contextmanager
def mask_mlp_writers(
    *,
    handle: HuggingFaceCausalLMHandle,
    layer_to_neurons: Dict[int, Sequence[int]],
    mode: str,
):
    if mode not in {"ablate", "keep_only"}:
        raise ValueError(f"mode must be 'ablate' or 'keep_only', got {mode}")

    saved: List[Tuple[torch.nn.Module, torch.Tensor]] = []

    try:
        with torch.no_grad():
            for layer, neurons in layer_to_neurons.items():
                out_proj = handle.resolve_mlp_out_proj(int(layer))
                Wp = out_proj.weight  # Parameter [d_model, d_mlp]

                # Save original (on same device/dtype)
                saved.append((out_proj, Wp.detach().clone()))

                cols = sorted(_as_set(neurons))
                d_mlp = Wp.shape[1]
                device = Wp.device
                dtype = Wp.dtype

                if mode == "ablate":
                    if len(cols) > 0:
                        idx = torch.tensor(cols, device=device, dtype=torch.long)
                        # write via .data to avoid leaf-view in-place issues
                        Wp.data[:, idx] = 0
                else:  # keep_only
                    mask = torch.zeros(d_mlp, device=device, dtype=dtype)
                    if len(cols) > 0:
                        idx = torch.tensor(cols, device=device, dtype=torch.long)
                        mask[idx] = 1
                    # multiply columns by mask
                    Wp.data.mul_(mask.unsqueeze(0))

        yield

    finally:
        with torch.no_grad():
            for out_proj, W0 in saved:
                out_proj.weight.data.copy_(W0)