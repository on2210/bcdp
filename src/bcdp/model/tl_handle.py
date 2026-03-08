# bcdp/model/tl_handle.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List

import torch

from bcdp.trace.site import Stream
from .model_handle import ModelHandle, HookHandle


class TransformerLensHandle(ModelHandle):
    """
    TransformerLens adapter (stub for now).

    This file exists so the architecture supports multiple backends.
    Implement when you're ready to use TL hookpoints.

    Expected TL mapping examples (later):
      layer L resid_post -> blocks.L.hook_resid_post
      layer L attn_out   -> blocks.L.attn.hook_z (or hook_result depending on TL)
      layer L mlp_out    -> blocks.L.mlp.hook_post
    """

    def __init__(self, tl_model: Any, tokenizer: Any = None) -> None:
        self.model = tl_model
        self.tokenizer = tokenizer
        self._active_hooks: List[HookHandle] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        # TL forward signature varies; keep it permissive.
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def add_hook(
        self,
        *,
        layer: int,
        stream: Stream,
        hook_fn: Callable[[torch.Tensor, Dict[str, Any]], None],
    ) -> HookHandle:
        raise NotImplementedError(
            "TransformerLensHandle.add_hook is not implemented yet. "
            "Once you commit to TL hookpoints, we'll map (layer, stream) -> hook name."
        )

    def clear_hooks(self) -> None:
        for h in self._active_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._active_hooks = []

    @property
    def device(self) -> torch.device:
        # TL models usually have .cfg.device or parameters
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except Exception:
            return torch.float32

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "TransformerLensHandle":
        # TL models typically have .to
        if dtype is None:
            self.model.to(device)
        else:
            self.model.to(device=device, dtype=dtype)
        return self

    @property
    def d_model(self) -> int:
        return int(getattr(self.model.cfg, "d_model"))

    @property
    def n_layers(self) -> int:
        return int(getattr(self.model.cfg, "n_layers"))

    @property
    def n_heads(self) -> Optional[int]:
        return int(getattr(self.model.cfg, "n_heads"))

    @property
    def d_head(self) -> Optional[int]:
        return int(getattr(self.model.cfg, "d_head", self.d_model // self.n_heads))

    @property
    def d_mlp(self) -> Optional[int]:
        return int(getattr(self.model.cfg, "d_mlp"))
