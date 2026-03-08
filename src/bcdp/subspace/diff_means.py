# bcdp/subspace/diff_means.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from bcdp.trace.activation_cache import ActivationCache
from bcdp.trace.site import Site

from .subspace import Subspace


@dataclass(frozen=True)
class DiffMeans:
    """
    Diff-in-means subspace discovery (MVP).

    Produces a rank-1 subspace:
      v = mean(X[y==1]) - mean(X[y==0])
      basis = v / ||v||
    """
    name: str = "diff_means"
    assume_binary_labels: bool = True
    orthonormal_tol: float = 1e-3

    def fit(
        self,
        cache: ActivationCache,
        *,
        site: Site,
        tags: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> Subspace:
        if not cache.has(site):
            raise KeyError(f"ActivationCache is missing site={site}")

        y = cache.labels
        if y.ndim != 1:
            raise ValueError(f"cache.labels must be 1D, got {tuple(y.shape)}")

        # Normalize labels to bool mask for positive class.
        # Convention: y!=0 is positive.
        pos = (y != 0)
        neg = ~pos

        if int(pos.sum().item()) == 0 or int(neg.sum().item()) == 0:
            raise ValueError(
                f"DiffMeans requires both classes present; got pos={int(pos.sum().item())}, neg={int(neg.sum().item())}"
            )

        X = cache.get(site)  # [N, d] (or [N, ... , d] but in our tracing we store [N,d])
        if X.ndim != 2:
            raise ValueError(f"DiffMeans expects X at site to be 2D [N,d], got {tuple(X.shape)} for site={site}")

        mu_pos = X[pos].mean(dim=0)
        mu_neg = X[neg].mean(dim=0)

        v = (mu_pos - mu_neg)  # [d]
        v_norm = float(v.norm().item())
        if v_norm == 0.0:
            raise ValueError("DiffMeans got zero vector (means identical).")

        v = v / v_norm
        basis = v[:, None]  # [d,1]

        stats = {
            "v_norm_pre_normalize": v_norm,
            "n_pos": float(pos.sum().item()),
            "n_neg": float(neg.sum().item()),
        }

        t: Dict[str, Union[str, int, float, bool]] = {}
        if tags is not None:
            t.update(tags)

        return Subspace(
            basis=basis,
            method=self.name,
            anchor_sites=(site,),
            scope_sites=None,
            scope_stats={},
            stats=stats,
            tags=t,
            assume_orthonormal=True,
        )
