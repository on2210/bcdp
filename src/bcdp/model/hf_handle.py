# bcdp/model/hf_handle.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List, Sequence, Tuple

import torch
import torch.nn as nn

from bcdp.trace.site import Stream  # <- your Stream enum lives in trace/site.py
from .model_handle import ModelHandle, HookHandle, _TorchHookHandle

from typing import Literal, Tuple

HookKind = Literal["forward", "pre"]


@dataclass
class HFModelGeometry:
    d_model: int
    n_layers: int
    n_heads: Optional[int] = None
    d_head: Optional[int] = None
    d_mlp: Optional[int] = None


class HuggingFaceCausalLMHandle(ModelHandle):
    """
    Minimal HuggingFace CausalLM adapter with multi-architecture hook resolution.

    Supported families for hooks (ATTN_OUT / MLP_OUT):
      - GPT2-like:      model.transformer.h[L].attn / .mlp
      - GPT-NeoX (Pythia): model.gpt_neox.layers[L].attention / .mlp
      - Llama-like:     model.model.layers[L].self_attn / .mlp
      - Qwen-like:      model.model.layers[L].self_attn / .mlp   (Qwen2 is Llama-derivative)
      - Gemma-like:     model.model.layers[L].self_attn / .mlp

    Notes:
      - For resid stream tracing (RESID_PRE/MID/POST), prefer `output_hidden_states=True`
        and read outputs.hidden_states in the Tracer. We keep hooks focused on ATTN/MLP.
      - If a given HF model variant differs in module names, extend `_resolve_module`.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        *,
        geometry: Optional[HFModelGeometry] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._active_hooks: List[HookHandle] = []

        self._geometry = geometry or self._infer_geometry(model)
        self._arch = self._detect_arch(model)

        # Cache layer container for faster resolve
        self._layer_blocks = self._get_layer_blocks(model, self._arch)

    # -------------------------
    # Required ModelHandle API
    # -------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def add_hook(
        self,
        *,
        layer: int,
        stream: Stream,
        hook_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor | None],
    ) -> HookHandle:
        module, kind = self._resolve_module(layer=layer, stream=stream)

        ctx_base = {"layer": layer, "stream": stream, "arch": self._arch}

        if kind == "forward":
            def _wrapped_hook(mod: nn.Module, inp: tuple, out: Any) -> Any:
                act = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(act):
                    raise TypeError(
                        f"Hook output for (layer={layer}, stream={stream}) is not a tensor: {type(act)}"
                    )

                new_act = hook_fn(act, dict(ctx_base))

                # None -> keep original
                if new_act is None:
                    return None

                # preserve tuple outputs
                if isinstance(out, (tuple, list)):
                    out_list = list(out)
                    out_list[0] = new_act
                    return type(out)(out_list)

                return new_act

            h = module.register_forward_hook(_wrapped_hook)

        elif kind == "pre":
            def _wrapped_pre_hook(mod: nn.Module, inp: tuple) -> Any:
                if len(inp) == 0 or (not torch.is_tensor(inp[0])):
                    raise TypeError(
                        f"Pre-hook input for (layer={layer}, stream={stream}) is not a tensor tuple."
                    )
                act = inp[0]  # resid_pre tensor [B,T,d] in most blocks

                new_act = hook_fn(act, dict(ctx_base))

                # None -> keep original inputs
                if new_act is None:
                    return None

                # Replace first positional arg, keep the rest
                return (new_act,) + tuple(inp[1:])

            h = module.register_forward_pre_hook(_wrapped_pre_hook)

        else:
            raise ValueError(f"Unknown hook kind: {kind}")

        handle: HookHandle = _TorchHookHandle(h)
        self._active_hooks.append(handle)
        return handle
        

        def _wrapped_hook(mod: nn.Module, inp: tuple, out: Any) -> Any:
            act = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(act):
                raise TypeError(
                    f"Hook output for (layer={layer}, stream={stream}) is not a tensor: {type(act)}"
                )

            ctx = {"layer": layer, "stream": stream, "arch": self._arch}

            new_act = hook_fn(act, ctx)

            # If hook_fn returns None → keep original
            if new_act is None:
                return None

            # If module originally returned tuple, reconstruct it
            if isinstance(out, (tuple, list)):
                out_list = list(out)
                out_list[0] = new_act
                return type(out)(out_list)

            return new_act


        h = module.register_forward_hook(_wrapped_hook)
        handle: HookHandle = _TorchHookHandle(h)
        self._active_hooks.append(handle)
        return handle

    def clear_hooks(self) -> None:
        for h in self._active_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._active_hooks = []

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "HuggingFaceCausalLMHandle":
        if dtype is None:
            self.model.to(device)
        else:
            self.model.to(device=device, dtype=dtype)
        return self

    @property
    def d_model(self) -> int:
        return self._geometry.d_model

    @property
    def n_layers(self) -> int:
        return self._geometry.n_layers

    @property
    def n_heads(self) -> Optional[int]:
        return self._geometry.n_heads

    @property
    def d_head(self) -> Optional[int]:
        return self._geometry.d_head

    @property
    def d_mlp(self) -> Optional[int]:
        return self._geometry.d_mlp

    # -------------------------
    # Internals: arch + geometry
    # -------------------------

    def _detect_arch(self, model: nn.Module) -> str:
        """
        Prefer config.model_type when present (most reliable), else fallback to attr probes.
        """
        cfg = getattr(model, "config", None)
        model_type = getattr(cfg, "model_type", None) if cfg is not None else None
        if isinstance(model_type, str):
            mt = model_type.lower()
            if mt in {"gpt2"}:
                return "gpt2_like"
            if mt in {"gpt_neox"}:
                return "gpt_neox_like"  # Pythia
            if mt in {"llama"}:
                return "llama_like"
            if mt in {"qwen2", "qwen2_moe", "qwen", "qwen1_5"}:
                return "qwen_like"
            if mt in {"gemma", "gemma2"}:
                return "gemma_like"

        # Fallbacks (older or custom wrappers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return "gpt2_like"
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return "gpt_neox_like"
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Many decoder-only models expose this (Llama/Qwen/Gemma families)
            return "decoder_layers_like"

        return "unknown"

    def _infer_geometry(self, model: nn.Module) -> HFModelGeometry:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("HuggingFace model has no .config; please provide HFModelGeometry explicitly.")

        d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if d_model is None:
            raise ValueError("Could not infer d_model from config (hidden_size / n_embd).")

        n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        if n_layers is None:
            raise ValueError("Could not infer n_layers from config (num_hidden_layers / n_layer).")

        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        d_head = None
        if n_heads is not None and d_model % n_heads == 0:
            d_head = d_model // n_heads

        d_mlp = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None)

        return HFModelGeometry(
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads) if n_heads is not None else None,
            d_head=int(d_head) if d_head is not None else None,
            d_mlp=int(d_mlp) if d_mlp is not None else None,
        )

    # -------------------------
    # Internals: layer blocks + resolution
    # -------------------------

    def _get_layer_blocks(self, model: nn.Module, arch: str) -> Sequence[nn.Module]:
        """
        Return a sequence-like container of per-layer blocks.
        """
        if arch == "gpt2_like":
            return model.transformer.h  # type: ignore[attr-defined]
        if arch == "gpt_neox_like":
            return model.gpt_neox.layers  # type: ignore[attr-defined]

        if arch in {"llama_like", "qwen_like", "gemma_like", "decoder_layers_like"}:
            # Common HF structure for Llama/Qwen/Gemma decoder-only:
            # model.model.layers is a ModuleList of decoder layers.
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return model.model.layers  # type: ignore[attr-defined]

        raise NotImplementedError(
            f"Cannot locate layer blocks for arch={arch}. "
            "Extend _get_layer_blocks for this model."
        )


    def _resolve_module(self, *, layer: int, stream: Stream) -> Tuple[nn.Module, HookKind]:
        """
        Map (layer, stream) -> (module, kind)

        kind:
        - "forward": use register_forward_hook (patch module output)
        - "pre":     use register_forward_pre_hook (patch module input)
        """
        block = self._layer_blocks[layer]

        # -------- Residual streams (generic) --------
        # resid_pre: input to the block
        # resid_post: output of the block
        if stream == Stream.RESID_PRE:
            return block, "pre"
        if stream == Stream.RESID_POST:
            return block, "forward"

        # -------- Existing support: ATTN_OUT / MLP_OUT --------
        if stream not in {Stream.ATTN_OUT, Stream.MLP_OUT}:
            raise NotImplementedError(
                f"HF handle hooks do not support stream={stream} for arch={self._arch}. "
                "Extend _resolve_module if needed."
            )

        if self._arch == "gpt2_like":
            if stream == Stream.ATTN_OUT:
                return block.attn, "forward"  # type: ignore[attr-defined]
            if stream == Stream.MLP_OUT:
                return block.mlp, "forward"   # type: ignore[attr-defined]

        if self._arch == "gpt_neox_like":
            if stream == Stream.ATTN_OUT:
                return block.attention, "forward"  # type: ignore[attr-defined]
            if stream == Stream.MLP_OUT:
                return block.mlp, "forward"        # type: ignore[attr-defined]

        if self._arch in {"llama_like", "qwen_like", "gemma_like", "decoder_layers_like"}:
            if stream == Stream.ATTN_OUT:
                if hasattr(block, "self_attn"):
                    return block.self_attn, "forward"  # type: ignore[attr-defined]
                if hasattr(block, "attn"):
                    return block.attn, "forward"       # type: ignore[attr-defined]
                raise NotImplementedError(
                    f"Decoder block has no self_attn/attn for arch={self._arch} (layer={layer})."
                )

            if stream == Stream.MLP_OUT:
                if hasattr(block, "mlp"):
                    return block.mlp, "forward"  # type: ignore[attr-defined]
                if hasattr(block, "feed_forward"):
                    return block.feed_forward, "forward"  # type: ignore[attr-defined]
                raise NotImplementedError(
                    f"Decoder block has no mlp/feed_forward for arch={self._arch} (layer={layer})."
                )

        raise NotImplementedError(
            f"HF handle cannot resolve module for (arch={self._arch}, layer={layer}, stream={stream})."
        )
    # -------------------------
    # Head-level utilities (HF only)
    # -------------------------

    def resolve_attention_projections(self, layer: int):
        """
        Returns (v_proj, o_proj) modules for a given layer.

        Supports:
          - Llama/Qwen/Gemma-like
          - GPT2-like
          - GPT-NeoX-like
        """
        block = self._layer_blocks[layer]

        if self._arch == "gpt2_like":
            return block.attn.c_attn, block.attn.c_proj  # GPT2 packs qkv

        if self._arch == "gpt_neox_like":
            return block.attention.query_key_value, block.attention.dense

        if self._arch in {"llama_like", "qwen_like", "gemma_like", "decoder_layers_like"}:
            attn = block.self_attn if hasattr(block, "self_attn") else block.attn
            return attn.v_proj, attn.o_proj

        raise NotImplementedError(f"Head ranking not implemented for arch={self._arch}")

    # -------------------------
    # MLP writer utilities (HF only)
    # -------------------------

    def resolve_mlp_out_proj(self, layer: int) -> nn.Module:
        """
        Returns the MLP output projection module that maps d_mlp -> d_model.
        This matrix's columns are the 'writers'.
        """
        block = self._layer_blocks[layer]

        if self._arch == "gpt2_like":
            return block.mlp.c_proj  # type: ignore[attr-defined]

        if self._arch == "gpt_neox_like":
            return block.mlp.dense_4h_to_h  # type: ignore[attr-defined]

        if self._arch in {"llama_like", "qwen_like", "gemma_like", "decoder_layers_like"}:
            if hasattr(block, "mlp"):
                mlp = block.mlp
                if hasattr(mlp, "down_proj"):
                    return mlp.down_proj
            if hasattr(block, "feed_forward"):
                ff = block.feed_forward
                if hasattr(ff, "w2"):
                    return ff.w2

            raise NotImplementedError(
                f"Cannot resolve MLP out proj for arch={self._arch} layer={layer}"
            )

        raise NotImplementedError(
            f"MLP writer resolve not implemented for arch={self._arch}"
        )


    def resolve_attn_o_proj(self, layer: int) -> nn.Linear:
        """
        Return the attention output projection module (o_proj / out_proj) for a given layer.
        Used for HF-only head writer ablations via weight masking.
        """
        block = self._layer_blocks[int(layer)]

        if self._arch == "gpt2_like":
            # GPT2: block.attn.c_proj is output proj
            return block.attn.c_proj  # type: ignore[attr-defined]

        if self._arch == "gpt_neox_like":
            # GPT-NeoX: attention.dense is output proj
            return block.attention.dense  # type: ignore[attr-defined]

        # Llama/Qwen/Gemma families
        if hasattr(block, "self_attn"):
            sa = block.self_attn  # type: ignore[attr-defined]
            # common: o_proj
            if hasattr(sa, "o_proj"):
                return sa.o_proj  # type: ignore[attr-defined]
            # some models: out_proj
            if hasattr(sa, "out_proj"):
                return sa.out_proj  # type: ignore[attr-defined]

        raise NotImplementedError(
            f"Cannot resolve attn o_proj for arch={self._arch}. "
            "Extend resolve_attn_o_proj for this model."
        )