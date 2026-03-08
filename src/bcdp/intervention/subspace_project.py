from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from bcdp.trace.site import Site

@dataclass(frozen=True)
class ProjectOutSubspaceIntervention:
    """
    h' = (I - W) h   at one Site
    W is [d,d] projection onto subspace.
    """
    site: Site
    W: torch.Tensor

    def sites(self) -> Tuple[Site, ...]:
        return (self.site,)

    def hook_for(self, site: Site):
        if site != self.site:
            raise KeyError(f"No hook for site={site}")

        def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> Optional[torch.Tensor]:
            pos_map = ctx["positions"]
            key = (self.site.position, self.site.index)
            pos_idx = pos_map[key].to(device=act.device)

            # act: [B,T,d] or [B,d]
            if act.ndim == 2:
                h = act
                write_back = lambda x: x
            else:
                bsz = act.shape[0]
                ar = torch.arange(bsz, device=act.device)
                h = act[ar, pos_idx, :]
                def write_back(h_new):
                    out = act.clone()
                    out[ar, pos_idx, :] = h_new
                    return out

            W = self.W.to(device=h.device, dtype=h.dtype)
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            h_new = h @ (I - W).T

            return write_back(h_new)

        return _hook
