# bcdp/subspace/base.py
from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable, Union

from bcdp.trace.activation_cache import ActivationCache
from bcdp.trace.site import Site

from .subspace import Subspace


@runtime_checkable
class SubspaceDiscoveryMethod(Protocol):
    """
    Interface for methods that discover a Subspace from an ActivationCache.

    The method chooses how to use labels and which site(s) to consume.
    """

    name: str

    def fit(
        self,
        cache: ActivationCache,
        *,
        site: Site,
        tags: Dict[str, Union[str, int, float, bool]] | None = None,
    ) -> Subspace:
        ...
