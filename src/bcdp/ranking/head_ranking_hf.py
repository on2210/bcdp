# bcdp/ranking/head_ranking_hf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from bcdp.trace.site import Position
from bcdp.model.hf_handle import HuggingFaceCausalLMHandle


@dataclass(frozen=True)
class HeadScore:
    layer: int
    head: int
    score: float


def _pick_pos(batch: Dict[str, Any], position: Position, device: torch.device) -> torch.Tensor:
    pos_idx = batch["positions"][(position, None)]
    return pos_idx.to(device=device)


@torch.no_grad()
def rank_heads_hf(
    *,
    handle: HuggingFaceCausalLMHandle,
    dataloader: Iterable[Dict[str, Any]],
    W: torch.Tensor,
    position: Position = Position.LAST,
    max_batches: Optional[int] = None,
) -> List[HeadScore]:
    """
    HF-only head ranking via capturing the INPUT to o_proj (pre-hook).
    Works robustly for Llama/Gemma/Qwen-like architectures where:
      o_proj input is concatenated head outputs: [B,T, n_heads*head_dim].

    Score(head) = E[ ||W y||^2 / (||y||^2 + eps) ] where y is the head contribution to d_model.
    """

    device = handle.device
    cfg = getattr(handle.model, "config", None)
    n_heads = int(getattr(cfg, "num_attention_heads", 0))
    if n_heads <= 0:
        raise RuntimeError("Could not read num_attention_heads from model config.")

    # Keep W in fp32 for stable scoring; project y_model to fp32 too.
    W_f = W.to(device=device, dtype=torch.float32)

    scores: List[HeadScore] = []

    for layer in range(handle.n_layers):
        _, o_proj = handle.resolve_attention_projections(layer)

        # o_proj.weight: [d_model, attn_dim]
        d_model = o_proj.weight.shape[0]
        attn_dim = o_proj.weight.shape[1]
        if attn_dim % n_heads != 0:
            raise RuntimeError(f"Layer {layer}: attn_dim {attn_dim} not divisible by n_heads {n_heads}")
        head_dim = attn_dim // n_heads

        # Precompute per-head weight chunks in fp32 on device:
        # chunk_h: [d_model, head_dim]
        oW = o_proj.weight.to(device=device, dtype=torch.float32)
        oW_chunks = [oW[:, h * head_dim : (h + 1) * head_dim] for h in range(n_heads)]

        # Accumulators
        head_sum = torch.zeros(n_heads, device=device, dtype=torch.float32)
        head_cnt = 0

        # Hook to capture o_proj input
        captured: Dict[str, torch.Tensor] = {}

        def _pre_hook(mod, inp):
            # inp[0] expected [B,T,attn_dim]
            x = inp[0]
            if not torch.is_tensor(x):
                raise TypeError("o_proj pre-hook received non-tensor input.")
            captured["attn_in"] = x
            return None

        hook = o_proj.register_forward_pre_hook(_pre_hook)

        try:
            for bi, batch in enumerate(dataloader):
                if max_batches is not None and bi >= max_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                captured.clear()

                # We don't need attentions or hidden_states now.
                _ = handle.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                if "attn_in" not in captured:
                    raise RuntimeError(
                        f"Layer {layer}: did not capture o_proj input. "
                        "This may indicate an unexpected module structure."
                    )

                attn_in = captured["attn_in"]  # [B,T,attn_dim] (dtype bf16/fp16)
                layer_device = attn_in.device
                B, T, D = attn_in.shape
                if D != attn_dim:
                    raise RuntimeError(f"Layer {layer}: captured attn_in dim {D} != attn_dim {attn_dim}")

                # Split heads: [B,T,n_heads,head_dim]
                attn_in_f = attn_in.to(device=layer_device, dtype=torch.float32)
                heads = attn_in_f.view(B, T, n_heads, head_dim)

                pos_idx = _pick_pos(batch, position, layer_device)   # <-- device = layer_device
                ar = torch.arange(B, device=layer_device)            # <-- same
                h_vecs = heads[ar, pos_idx, :, :]

                # For each head, map to d_model using its o_proj chunk, then score vs W.
                for h in range(n_heads):
                    y_h = h_vecs[:, h, :]
                    y_model = (y_h @ (oW_chunks[h]).to(device=y_h.device).T ).to(device=W_f.device)
                    proj = y_model @ W_f.T

                    num = (proj * proj).sum(dim=-1)
                    den = (y_model * y_model).sum(dim=-1) + 1e-8
                    ratio = (num / den).mean()

                    head_sum[h] += ratio

                head_cnt += 1

            if head_cnt == 0:
                raise RuntimeError("No batches processed in head ranking (max_batches too small?)")

            head_mean = head_sum / head_cnt
            for h in range(n_heads):
                scores.append(HeadScore(layer=layer, head=h, score=float(head_mean[h].item())))

        finally:
            hook.remove()

    scores.sort(key=lambda x: x.score, reverse=True)
    return scores