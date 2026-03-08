# bcdp/ranking/mlp_writers_hf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle


@dataclass(frozen=True)
class MLPWriterScore:
    layer: int
    neuron: int  # column index in out-proj (d_mlp)
    score: float


@torch.no_grad()
def rank_mlp_writers_hf(
    *,
    handle: HuggingFaceCausalLMHandle,
    W: torch.Tensor,                      # [d_model, d_model] projection
    topk_per_layer: int = 50,
    normalize: bool = True,               # ratio vs absolute energy
    eps: float = 1e-8,
) -> List[MLPWriterScore]:
    """
    Ranks MLP writer neurons by how much their output-weight column lies in subspace W.

    Uses ONLY weights (no forward pass), so it's fast and HF-only.
    """
    device = handle.device
    W = W.to(device=device)

    out: List[MLPWriterScore] = []

    for layer in range(handle.n_layers):
        out_proj = handle.resolve_mlp_out_proj(layer)
        if not hasattr(out_proj, "weight"):
            raise RuntimeError(f"MLP out proj at layer {layer} has no .weight")

        # Linear(d_mlp -> d_model) has weight shape [d_model, d_mlp]
        weight = out_proj.weight.detach().to(device=device)  # [d_model, d_mlp]
        d_model, d_mlp = weight.shape

        # columns are writer vectors w_j in R^{d_model}
        # compute projected energy for all columns at once
        W_ = W.to(device=weight.device, dtype=weight.dtype)
        Ww = W_ @ weight  # [d_model, d_mlp]
        num = (Ww * Ww).sum(dim=0)  # [d_mlp]

        if normalize:
            den = (weight * weight).sum(dim=0) + eps
            score = num / den
        else:
            score = num

        k = min(int(topk_per_layer), int(d_mlp))
        vals, idx = torch.topk(score, k=k, largest=True)

        for v, j in zip(vals.tolist(), idx.tolist()):
            out.append(MLPWriterScore(layer=layer, neuron=int(j), score=float(v)))

    out.sort(key=lambda x: x.score, reverse=True)
    return out