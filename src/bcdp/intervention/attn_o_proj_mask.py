# bcdp/intervention/attn_o_proj_mask.py
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle


@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int


def _group_heads(heads: Sequence[HeadRef]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for h in heads:
        out.setdefault(int(h.layer), []).append(int(h.head))
    return out


@contextmanager
def mask_attn_o_proj_heads(
    *,
    handle: HuggingFaceCausalLMHandle,
    heads: Sequence[HeadRef],
    mode: str,  # "ablate" | "keep_only"
):
    """
    HF-only head masking via attention o_proj.weight slicing.

    Works with Gemma-style shapes too:
      o_proj.weight: [d_model, attn_out_dim]
    where attn_out_dim is typically n_heads * head_dim (e.g. 2048),
    and d_model can be larger (e.g. 2304).

    We mask input columns of o_proj corresponding to the head slice:
      columns [h*d_head_attn : (h+1)*d_head_attn]
    where d_head_attn = attn_out_dim / n_heads.
    """
    if handle.n_heads is None:
        raise ValueError("handle.n_heads is required for head masking.")
    if mode not in {"ablate", "keep_only"}:
        raise ValueError(f"mode must be 'ablate' or 'keep_only', got {mode}")

    n_heads = int(handle.n_heads)
    by_layer = _group_heads(heads)

    saved: List[Tuple[nn.Module, torch.Tensor]] = []

    try:
        with torch.no_grad():
            for layer, hs in by_layer.items():
                o_proj = handle.resolve_attn_o_proj(int(layer))
                Wp = o_proj.weight  # [d_model, attn_out_dim]
                saved.append((o_proj, Wp.detach().clone()))

                attn_out_dim = int(Wp.shape[1])
                if attn_out_dim % n_heads != 0:
                    raise ValueError(
                        f"o_proj.in_features={attn_out_dim} not divisible by n_heads={n_heads}. "
                        "Cannot map heads to slices."
                    )
                d_head_attn = attn_out_dim // n_heads

                col_mask = torch.ones(attn_out_dim, device=Wp.device, dtype=Wp.dtype)

                if mode == "ablate":
                    for h in hs:
                        if 0 <= h < n_heads:
                            a, b = h * d_head_attn, (h + 1) * d_head_attn
                            col_mask[a:b] = 0
                else:  # keep_only
                    col_mask.zero_()
                    for h in hs:
                        if 0 <= h < n_heads:
                            a, b = h * d_head_attn, (h + 1) * d_head_attn
                            col_mask[a:b] = 1

                # apply on columns
                Wp.data.mul_(col_mask.unsqueeze(0))

        yield
    finally:
        with torch.no_grad():
            for mod, W0 in saved:
                mod.weight.data.copy_(W0)