# bcdp/intervention/project_out_plan.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from bcdp.trace.site import Site
from .base import InterventionPlan
from .interchange import _ensure_2d_h_at_positions, _write_back_positions


@dataclass(frozen=True)
class ProjectOutSubspaceIntervention:
    """
    Project-out intervention at a single Site:

        h' = (I - W) h

    where W is a projection matrix [d,d].

    Notes:
    - Expects ctx['positions'] with key (site.position, site.index) -> LongTensor[B]
      (same convention as Tracer/Interchange).
    - Works for act shaped [B,T,d] or [B,d].
    """
    site: Site
    W: torch.Tensor  # [d,d]

    def sites(self) -> Tuple[Site, ...]:
        return (self.site,)

    def hook_for(self, site: Site):
        if site != self.site:
            raise KeyError(f"ProjectOutSubspaceIntervention has no hook for site={site}")

        def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> Optional[torch.Tensor]:
            pos_map = ctx.get("positions", None)
            if pos_map is None:
                raise RuntimeError("ProjectOut hook missing ctx['positions']")

            key = (self.site.position, self.site.index)
            if key not in pos_map:
                raise KeyError(f"positions missing key={key} for site={self.site}")

            pos_idx = pos_map[key].to(device=act.device)  # [B]

            # Extract [B,d]
            h = _ensure_2d_h_at_positions(act, pos_idx)

            # Make (I - W) on the right device/dtype
            W = self.W.to(device=h.device, dtype=h.dtype)
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)

            # Project out: h_new = h @ (I - W).T
            h_new = h @ (I - W).T

            # Write back
            patched = _write_back_positions(act, pos_idx, h_new)
            return patched

        return _hook


@dataclass
class ProjectOutPlan(InterventionPlan):
    """
    Single-pass plan (no donor):
      apply ProjectOutSubspaceIntervention at `site` with projection matrix W.
    """
    site: Site
    W: torch.Tensor

    def donor_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def donor_sites(self) -> Tuple[Site, ...]:
        return tuple()

    def build_intervention(
        self,
        donor_acts: Dict[Site, torch.Tensor],
        batch: Dict[str, Any],
    ) -> ProjectOutSubspaceIntervention:
        return ProjectOutSubspaceIntervention(site=self.site, W=self.W)