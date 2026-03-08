# bcdp/intervention/interchange.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from bcdp.trace.site import Site


def _ensure_2d_h_at_positions(act: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    """
    Extract [B, d] from activation.
    - act: [B,T,d] or [B,d]
    - pos_idx: [B] long
    """
    if act.ndim == 2:
        return act
    if act.ndim != 3:
        raise ValueError(f"Expected act [B,T,d] or [B,d], got {tuple(act.shape)}")

    if pos_idx.dtype != torch.long:
        pos_idx = pos_idx.long()

    bsz = act.shape[0]
    ar = torch.arange(bsz, device=act.device)
    return act[ar, pos_idx, :]  # [B,d]


def _write_back_positions(act: torch.Tensor, pos_idx: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
    """
    Write h_new [B,d] back into act:
    - if act is [B,d] => return h_new
    - if act is [B,T,d] => replace at positions
    """
    if act.ndim == 2:
        return h_new
    if act.ndim != 3:
        raise ValueError(f"Expected act [B,T,d] or [B,d], got {tuple(act.shape)}")

    if pos_idx.dtype != torch.long:
        pos_idx = pos_idx.long()

    bsz = act.shape[0]
    ar = torch.arange(bsz, device=act.device)

    # Avoid in-place on views in case backend dislikes it
    out = act.clone()
    out[ar, pos_idx, :] = h_new
    return out


@dataclass(frozen=True)
class InterchangeSubspaceIntervention:
    """
    Apply interchange intervention at a single Site:

        h' = W h_donor + (I - W) h_orig

    where W is a projection matrix [d,d], and donor activations are provided per Site as [B,d].

    Notes:
    - Expects batch['positions'] to include key (site.position, site.index) -> LongTensor[B]
      (same convention as Tracer).
    - Works for hook activations shaped [B,T,d] or [B,d].
    """
    site: Site
    W: torch.Tensor                       # [d,d] on same device/dtype as activations at runtime
    donor_acts: Dict[Site, torch.Tensor]  # site -> [B,d] donor values

    def sites(self) -> Tuple[Site, ...]:
        return (self.site,)

    def hook_for(self, site: Site):
        if site != self.site:
            raise KeyError(f"InterchangeSubspaceIntervention has no hook for site={site}")

        def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> Optional[torch.Tensor]:
            # Expect runner to provide positions in ctx
            pos_map = ctx.get("positions", None)
            if pos_map is None:
                raise RuntimeError("Interchange hook missing ctx['positions']")

            key = (self.site.position, self.site.index)
            if key not in pos_map:
                raise KeyError(f"positions missing key={key} for site={self.site}")
            pos_idx = pos_map[key].to(device=act.device)  # [B]

            # Extract current h_orig [B,d]
            h_o = _ensure_2d_h_at_positions(act, pos_idx)

            # Fetch donor [B,d]
            if self.site not in self.donor_acts:
                raise KeyError(f"donor_acts missing site={self.site}")
            h_c = self.donor_acts[self.site].to(device=h_o.device, dtype=h_o.dtype)

            # Build (I - W) lazily on same device/dtype
            W = self.W.to(device=h_o.device, dtype=h_o.dtype)
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)

            # Apply interchange in [B,d]
            # (using right-multiply; both are fine if consistent)
            # h_new = (h_c @ W.T) + (h_o @ (I - W).T)
            h_new = (h_c @ W.T) + (h_o @ (I - W).T)

            # Write back into activation tensor
            patched = _write_back_positions(act, pos_idx, h_new)
            return patched

        return _hook
