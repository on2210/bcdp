# bcdp/intervention/interchange_plan.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import torch

from bcdp.trace.site import Site
from .base import InterventionPlan
from .interchange import InterchangeSubspaceIntervention


@dataclass
class InterchangePlan(InterventionPlan):
    """
    A simple 2-pass interchange plan.

    - donor_builder: function(batch) -> donor_batch
    - site: where to apply interchange
    - W: projection matrix [d,d]
    """

    donor_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    site: Site
    W: torch.Tensor

    # -----------------------------
    # Plan interface
    # -----------------------------

    def donor_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.donor_builder(batch)

    def donor_sites(self) -> Tuple[Site, ...]:
        return (self.site,)

    def build_intervention(
        self,
        donor_acts: Dict[Site, torch.Tensor],
        batch: Dict[str, Any],
    ) -> InterchangeSubspaceIntervention:

        return InterchangeSubspaceIntervention(
            site=self.site,
            W=self.W,
            donor_acts=donor_acts,
        )
