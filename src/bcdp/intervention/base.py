# bcdp/intervention/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import torch

from bcdp.trace.site import Site


# ----------------------------
# Core types
# ----------------------------

HookCtx = Dict[str, Any]
HookFn = Callable[[torch.Tensor, HookCtx], Optional[torch.Tensor]]


class Intervention(Protocol):
    """
    A local transformation applied via hooks at specific Sites.

    Convention:
      - hook returns None => leave activation unchanged
      - hook returns Tensor => replace activation with returned tensor
    """

    def sites(self) -> Tuple[Site, ...]: ...

    def hook_for(self, site: Site) -> HookFn: ...


class InterventionPlan(Protocol):
    """
    An end-to-end patching plan.

    The runner will:
      1) build donor batch (counterfactual)
      2) run donor forward and collect donor activations
      3) build Intervention using donor activations
      4) run original forward with patch hooks installed
    """

    def donor_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a donor/counterfactual batch (same shapes), or None if no donor pass is needed."""
        ...

    def donor_sites(self) -> Tuple[Site, ...]:
        """Which Sites to record from the donor pass."""
        ...

    def build_intervention(self, donor_acts: Dict[Site, torch.Tensor], batch: Dict[str, Any]) -> Intervention:
        """Construct the Intervention that will be applied on the original pass."""
        ...


# ----------------------------
# Result container (optional but handy)
# ----------------------------

@dataclass
class InterventionResult:
    outputs: Any
    donor_acts: Optional[Dict[Site, torch.Tensor]] = None
