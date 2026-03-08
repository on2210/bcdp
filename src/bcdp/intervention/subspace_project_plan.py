from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from bcdp.trace.site import Site
from bcdp.intervention.base import InterventionPlan
from .subspace_project import ProjectOutSubspaceIntervention

@dataclass
class ProjectOutPlan(InterventionPlan):
    site: Site
    W: torch.Tensor

    def donor_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def donor_sites(self) -> Tuple[Site, ...]:
        return tuple()

    def build_intervention(self, donor_acts: Dict[Site, torch.Tensor], batch: Dict[str, Any]):
        return ProjectOutSubspaceIntervention(site=self.site, W=self.W)
