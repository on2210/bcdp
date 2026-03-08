# bcdp/model/model_handle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable, Tuple

import torch

# NOTE: Stream enum is defined in bcdp/trace/sites.py (per your setup).
# We import it here so ModelHandle can expose "add_hook" by (layer, stream).
from bcdp.trace.site import Stream


@runtime_checkable
class HookHandle(Protocol):
    """A small handle that can remove a registered hook."""
    def remove(self) -> None: ...


@dataclass
class _TorchHookHandle:
    """Concrete HookHandle wrapper around a torch forward hook handle."""
    _handle: Any

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@runtime_checkable
class ModelHandle(Protocol):
    """
    Backend-agnostic interface used by Tracer and downstream pipeline stages.

    Responsibilities:
      1) Run forward passes.
      2) Provide a unified hook API keyed by (layer, stream).
      3) Expose basic model geometry (d_model, n_layers, etc.).

    Non-responsibilities:
      - No dataset/binding logic.
      - No subspace logic.
      - No evaluation metrics.
    """

    # --- Core forward ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Run a forward pass. Should accept HF-style kwargs and return a model output."""

    # --- Hooks (measurement) ---
    def add_hook(
        self,
        *,
        layer: int,
        stream: Stream,
        hook_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor | None],
    ) -> HookHandle:
        """
        Register a forward hook for the given (layer, stream).

        hook_fn signature: (activation_tensor, context_dict) -> None
        The context dict can include info like layer/stream, batch idx, etc.
        """
        ...

    def clear_hooks(self) -> None:
        """Remove all hooks registered via this handle."""
        ...

    # --- Device / dtype ---
    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def to(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> "ModelHandle":
        """Move model to device/dtype. Returns self for chaining."""
        ...

    # --- Model geometry ---
    @property
    def d_model(self) -> int: ...

    @property
    def n_layers(self) -> int: ...

    @property
    def n_heads(self) -> Optional[int]: ...

    @property
    def d_head(self) -> Optional[int]: ...

    @property
    def d_mlp(self) -> Optional[int]: ...
