# bcdp/subspace/subspace.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from bcdp.trace.site import Site


def _as_tuple_sites(sites: Sequence[Site]) -> Tuple[Site, ...]:
    if not isinstance(sites, (list, tuple)):
        raise TypeError(f"sites must be a sequence of Site, got {type(sites)}")
    out: list[Site] = []
    for s in sites:
        if not isinstance(s, Site):
            raise TypeError(f"Expected Site in sites, got {type(s)}")
        out.append(s)
    if len(out) == 0:
        raise ValueError("sites must be non-empty")
    return tuple(out)


def _check_orthonormal_columns(basis: torch.Tensor, *, tol: float = 1e-3) -> float:
    """
    Returns max absolute deviation from identity of B^T B.
    """
    k = basis.shape[1]
    gram = basis.T @ basis  # [k,k]
    eye = torch.eye(k, device=gram.device, dtype=gram.dtype)
    err = (gram - eye).abs().max().item()
    if err > tol:
        raise ValueError(f"basis columns are not orthonormal within tol={tol}: max|B^T B - I|={err:.4e}")
    return float(err)


@dataclass(frozen=True)
class Subspace:
    """
    A low-dimensional subspace in an ambient vector space (typically residual stream d_model).

    Key idea:
      - basis defines the subspace (columns).
      - anchor_sites describe *where/how it was learned* (reproducibility).
      - scope_sites + scope_stats describe *where it appears stable / where we validated it*.

    Invariants:
      - basis is 2D, shape [d_model, k], k>=1
      - columns are (by default) orthonormal (enforced unless you disable checks)
      - anchor_sites is non-empty
      - scope_sites is optional; if provided, must be non-empty
    """

    basis: torch.Tensor  # [d_model, k], columns
    method: str
    anchor_sites: Tuple[Site, ...]
    scope_sites: Optional[Tuple[Site, ...]] = None

    # Optional per-site diagnostics (e.g., cosine to anchor, separation, proj energy ratio)
    scope_stats: Dict[Site, Dict[str, float]] = field(default_factory=dict)

    # Optional global stats/tags for logging & reproducibility
    stats: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    # Convention: we expect orthonormal columns unless explicitly stated otherwise
    assume_orthonormal: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Subspace.method must be a non-empty str")

        if not torch.is_tensor(self.basis):
            raise TypeError(f"Subspace.basis must be a torch.Tensor, got {type(self.basis)}")

        if self.basis.ndim != 2:
            raise ValueError(f"Subspace.basis must be 2D [d_model,k], got shape {tuple(self.basis.shape)}")

        d_model, k = self.basis.shape
        if d_model <= 0 or k <= 0:
            raise ValueError(f"Invalid basis shape: {tuple(self.basis.shape)} (need d_model>0,k>0)")

        # sites
        object.__setattr__(self, "anchor_sites", _as_tuple_sites(self.anchor_sites))
        if self.scope_sites is not None:
            object.__setattr__(self, "scope_sites", _as_tuple_sites(self.scope_sites))

        # scope_stats key types
        if not isinstance(self.scope_stats, dict):
            raise TypeError("scope_stats must be a dict[Site, dict[str,float]]")
        for s, d in self.scope_stats.items():
            if not isinstance(s, Site):
                raise TypeError(f"scope_stats keys must be Site, got {type(s)}")
            if not isinstance(d, dict):
                raise TypeError(f"scope_stats[{s}] must be a dict[str,float], got {type(d)}")
            for kk, vv in d.items():
                if not isinstance(kk, str):
                    raise TypeError(f"scope_stats[{s}] keys must be str, got {type(kk)}")
                if not isinstance(vv, (int, float)):
                    raise TypeError(f"scope_stats[{s}][{kk}] must be float-like, got {type(vv)}")

        if self.assume_orthonormal:
            # Enforce orthonormality (tolerant) for numerical stability.
            _check_orthonormal_columns(self.basis)

    # -----------------------
    # Geometry helpers
    # -----------------------

    @property
    def d_model(self) -> int:
        return int(self.basis.shape[0])

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "Subspace":
        b = self.basis.to(device=device, dtype=dtype) if dtype is not None else self.basis.to(device=device)
        # preserve immutability by returning a new object
        return Subspace(
            basis=b,
            method=self.method,
            anchor_sites=self.anchor_sites,
            scope_sites=self.scope_sites,
            scope_stats=dict(self.scope_stats),
            stats=dict(self.stats),
            tags=dict(self.tags),
            assume_orthonormal=self.assume_orthonormal,
        )

    def coords(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return coordinates of x in the subspace.

        x: [..., d_model]
        returns: [..., k]
        """
        x2 = self._require_last_dim(x)
        # [..., d] @ [d,k] -> [..., k]
        return x2 @ self.basis

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto the subspace.

        x: [..., d_model]
        returns: [..., d_model]
        """
        c = self.coords(x)            # [..., k]
        return c @ self.basis.T       # [..., d]

    def remove(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove (project out) the subspace component: x - proj(x).
        """
        return x - self.project(x)

    def steer(
        self,
        x: torch.Tensor,
        alpha: Union[float, torch.Tensor],
        *,
        direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add a subspace direction to x.

        - If direction is None:
            * if k==1: uses the single basis vector
            * else: uses all-ones direction in coord space (not recommended, but deterministic)
        - alpha can be float or tensor broadcastable to x[..., 1] or x[..., k].

        direction:
          - shape [k] (coord-space direction), or
          - shape [d_model] (ambient direction), or
          - shape [d_model, k] treated as a custom basis (rare; kept out for now)

        returns: x + delta in ambient space
        """
        x2 = self._require_last_dim(x)

        if direction is None:
            if self.k == 1:
                delta = self.basis[:, 0]  # [d]
                return x2 + self._scale(delta, alpha, target=x2)
            else:
                dir_k = torch.ones((self.k,), device=x2.device, dtype=x2.dtype)
                delta = self.basis @ dir_k  # [d]
                return x2 + self._scale(delta, alpha, target=x2)

        if not torch.is_tensor(direction):
            raise TypeError("direction must be a torch.Tensor or None")

        if direction.ndim == 1:
            if int(direction.shape[0]) == self.k:
                # coord-space direction
                dir_k = direction.to(device=x2.device, dtype=x2.dtype)
                delta = self.basis @ dir_k  # [d]
                return x2 + self._scale(delta, alpha, target=x2)
            if int(direction.shape[0]) == self.d_model:
                # ambient direction
                delta = direction.to(device=x2.device, dtype=x2.dtype)
                return x2 + self._scale(delta, alpha, target=x2)
            raise ValueError(
                f"direction 1D must have shape [k]={self.k} or [d_model]={self.d_model}, got {tuple(direction.shape)}"
            )

        raise ValueError(f"direction must be 1D tensor or None; got shape {tuple(direction.shape)}")

    def with_scope(
        self,
        scope_sites: Sequence[Site],
        *,
        scope_stats: Optional[Mapping[Site, Mapping[str, float]]] = None,
    ) -> "Subspace":
        """
        Return a copy with updated scope sites/stats.
        """
        ss = _as_tuple_sites(scope_sites)
        new_stats: Dict[Site, Dict[str, float]] = dict(self.scope_stats)
        if scope_stats is not None:
            for s, d in scope_stats.items():
                if not isinstance(s, Site):
                    raise TypeError(f"scope_stats keys must be Site, got {type(s)}")
                new_stats[s] = {str(k): float(v) for k, v in d.items()}
        return Subspace(
            basis=self.basis,
            method=self.method,
            anchor_sites=self.anchor_sites,
            scope_sites=ss,
            scope_stats=new_stats,
            stats=dict(self.stats),
            tags=dict(self.tags),
            assume_orthonormal=self.assume_orthonormal,
        )

    # -----------------------
    # Internals
    # -----------------------

    def _require_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
        if x.ndim < 1:
            raise ValueError("x must have at least 1 dimension")
        if int(x.shape[-1]) != self.d_model:
            raise ValueError(f"x last dim must be d_model={self.d_model}, got {int(x.shape[-1])} (shape {tuple(x.shape)})")
        return x

    def _scale(self, v: torch.Tensor, alpha: Union[float, torch.Tensor], *, target: torch.Tensor) -> torch.Tensor:
        """
        Scale a vector v:[d] by alpha, returning something broadcastable to target:[...,d].
        """
        v2 = v.to(device=target.device, dtype=target.dtype)
        if isinstance(alpha, (int, float)):
            return v2 * float(alpha)
        if torch.is_tensor(alpha):
            a = alpha.to(device=target.device, dtype=target.dtype)
            # Try to broadcast: if a is scalar or [...,1] that's fine; if it's [...], also fine.
            return v2 * a
        raise TypeError(f"alpha must be float or tensor, got {type(alpha)}")
