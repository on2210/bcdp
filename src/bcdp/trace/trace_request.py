# bcdp/trace/trace_request.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import torch

from .site import Site, Stream  # enums defined in site.py as you requested


@dataclass(frozen=True)
class TraceRequest:
    """
    Specifies *what to record* during a forward pass.

    The Tracer should be able to satisfy this request without any binding-specific logic.
    All "semantic positions" (e.g., answer token) should be resolved by the batch/collate,
    typically via batch["positions"][<key>], if Site.pos is a string key.
    """

    # The set of Sites to capture. Each Site encodes (layer, stream, pos).
    sites: Sequence[Site]

    # If True, also return logits in the ActivationCache meta (or a separate field).
    # Keep False by default to reduce memory usage.
    return_logits: bool = False

    # Optional: request attention patterns (e.g., [B, n_heads, T, T]) if supported.
    # Kept as a flag so we can extend later without changing the core API.
    return_attn: bool = False

    # Optional: if True, keep tensors on GPU in cache. Default is CPU to avoid OOM.
    store_on_device: bool = False

    # Optional: whether to keep gradients (usually False). If False, Tracer detaches.
    require_grads: bool = False

    # Optional: arbitrary key/value metadata to tag this trace (e.g., experiment name).
    tags: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sites) == 0:
            raise ValueError("TraceRequest.sites must be non-empty.")

    @property
    def streams(self) -> Set[Stream]:
        """Convenience: which streams are requested."""
        return {s.stream for s in self.sites}

    @property
    def layers(self) -> Set[int]:
        """Convenience: which layers are requested."""
        return {s.layer for s in self.sites}

    def group_by_layer(self) -> Dict[int, List[Site]]:
        """Convenience: sites grouped by layer."""
        out: Dict[int, List[Site]] = {}
        for site in self.sites:
            out.setdefault(site.layer, []).append(site)
        return out

    def group_by_stream(self) -> Dict[Stream, List[Site]]:
        """Convenience: sites grouped by stream."""
        out: Dict[Stream, List[Site]] = {}
        for site in self.sites:
            out.setdefault(site.stream, []).append(site)
        return out
