# bcdp/trace/activation_cache.py
"""
ActivationCache is a typed container for traced model activations.

It represents a frozen experimental snapshot:
- activations collected at specific Sites
- aligned across the same set of examples (axis 0)
- accompanied by labels (also aligned to examples)

Downstream stages (subspace discovery, ranking, causal tests) should treat this
as read-only and rely only on the invariants enforced here.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

from .site import Site


class ActivationCache:
    """
    Minimal, strict activation container.

    Invariants:
    - Example axis is always axis 0.
    - labels is 1D with shape [N].
    - For every stored tensor X: X.shape[0] == N.
    """

    def __init__(
        self,
        activations: Dict[Site, torch.Tensor],
        labels: torch.Tensor,
        variant_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        if not isinstance(activations, dict):
            raise TypeError(f"activations must be a dict[Site, Tensor], got {type(activations)}")

        if not torch.is_tensor(labels):
            raise TypeError("labels must be a torch.Tensor")

        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D with shape [N], got shape {tuple(labels.shape)}")

        n = int(labels.shape[0])

        for site, x in activations.items():
            if not isinstance(site, Site):
                raise TypeError(f"ActivationCache keys must be Site, got {type(site)}")

            if not torch.is_tensor(x):
                raise TypeError(f"activations[{site}] must be a torch.Tensor, got {type(x)}")

            if x.ndim < 2:
                raise ValueError(
                    f"activations[{site}] must have at least 2 dims [N, ...], got shape {tuple(x.shape)}"
                )

            if int(x.shape[0]) != n:
                raise ValueError(
                    f"activations[{site}].shape[0] must equal N={n}, got {int(x.shape[0])} "
                    f"(shape {tuple(x.shape)})"
                )

        if variant_names is not None:
            if not isinstance(variant_names, tuple) or not all(isinstance(v, str) for v in variant_names):
                raise TypeError("variant_names must be a tuple[str, ...] or None")

        # Store internally to discourage casual mutation.
        self._activations: Dict[Site, torch.Tensor] = activations
        self._labels: torch.Tensor = labels
        self._variant_names: Optional[Tuple[str, ...]] = variant_names

    # ---- Read-only-ish accessors ----

    @property
    def activations(self) -> Dict[Site, torch.Tensor]:
        """Dictionary mapping Site -> Tensor. Treat as read-only."""
        return self._activations

    @property
    def labels(self) -> torch.Tensor:
        """1D tensor of labels aligned with examples (axis 0)."""
        return self._labels

    @property
    def variant_names(self) -> Optional[Tuple[str, ...]]:
        """Optional names for the variant axis (if present)."""
        return self._variant_names

    # ---- Minimal API ----

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def sites(self) -> Tuple[Site, ...]:
        """
        Return all recorded sites in a stable order.
        Python dict preserves insertion order, so we keep that.
        """
        return tuple(self._activations.keys())

    def has(self, site: Site) -> bool:
        return site in self._activations

    def get(self, site: Site) -> torch.Tensor:
        """
        Return activations tensor for a given site.
        Raises KeyError if missing.
        """
        return self._activations[site]

    def subset(self, mask: torch.Tensor) -> "ActivationCache":
        """
        Return a new ActivationCache filtered by a boolean mask over examples.

        mask:
          - shape [N]
          - dtype bool (or convertible to bool)
        """
        mask = self._normalize_mask(mask)

        if int(mask.sum().item()) == 0:
            raise ValueError("subset mask selects 0 examples")

        new_acts: Dict[Site, torch.Tensor] = {site: x[mask] for site, x in self._activations.items()}
        new_labels = self._labels[mask]

        return ActivationCache(
            activations=new_acts,
            labels=new_labels,
            variant_names=self._variant_names,
        )

    def mean(self, site: Site, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute mean activation at a site over examples (axis 0), optionally masked.

        - If mask is None: mean over all examples.
        - If mask provided: mean over selected examples.

        Note: This method reduces ONLY the example axis (axis 0).
        Any remaining axes (e.g., variants) are preserved.
        """
        x = self.get(site)

        if mask is None:
            return x.mean(dim=0)

        mask = self._normalize_mask(mask)
        if int(mask.sum().item()) == 0:
            raise ValueError("mean mask selects 0 examples")

        return x[mask].mean(dim=0)

    # ---- Helpers ----

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(mask):
            raise TypeError("mask must be a torch.Tensor")

        if mask.ndim != 1:
            raise ValueError(f"mask must be 1D with shape [N], got shape {tuple(mask.shape)}")

        n = len(self)
        if int(mask.shape[0]) != n:
            raise ValueError(f"mask length must match N={n}, got {int(mask.shape[0])}")

        # Allow int/long masks (0/1) but convert to bool.
        if mask.dtype != torch.bool:
            mask = mask != 0

        return mask
