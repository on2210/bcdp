# bcdp/trace/trace.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from bcdp.model.model_handle import ModelHandle, HookHandle
from .site import Site, Stream, Position
from .activation_cache import ActivationCache
from .trace_request import TraceRequest

PosKey = tuple[Position, Optional[int]]

class Tracer:
    """
    Measurement engine.

    - Does NOT know binding semantics.
    - Does NOT discover subspaces.
    - Captures requested activations at Sites and returns an ActivationCache.

    Residual streams (RESID_PRE / RESID_POST) are captured via model forward outputs
    (hidden_states), because this is the most stable cross-architecture method in HF.

    Non-residual streams (ATTN_OUT / MLP_OUT) are captured via hooks (ModelHandle.add_hook).
    """


    def __init__(
        self,
        model: ModelHandle,
        *,
        device: Optional[torch.device] = None,
        store_device: Optional[torch.device] = None,
        detach: bool = True,
        use_amp: bool = False,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.device = device or model.device
        self.store_device = store_device or torch.device("cpu")
        self.detach = detach
        self.use_amp = use_amp
        self.dtype = dtype
        self.verbose = verbose

        self._active_hooks: List[HookHandle] = []
        self._buffer: Dict[Site, List[torch.Tensor]] = {}

        # These are set per-batch so hooks can slice positions without knowing semantics.
        self._current_positions: Optional[Dict[Position, torch.Tensor]] = None
        self._current_request: Optional[TraceRequest] = None

    # -----------------------
    # Public API
    # -----------------------

    def trace(self, dataloader: Iterable[Dict[str, Any]], request: TraceRequest) -> ActivationCache:
        """
        Run tracing over batches and return an ActivationCache.

        Expected batch format:
          - batch["input_ids"]: LongTensor [B, T]
          - batch["labels"]:   Tensor [B] (1D)
          - batch["positions"]: Dict[Position, LongTensor[B]] (position indices per example)

        Notes:
          - Tracer treats Position as a *key* only; Dataset/Collate must provide indices.
          - RESID_MID is intentionally not supported yet.
        """
        # Enforce "no MID for now"
        if any(s.stream == Stream.RESID_MID for s in request.sites):
            raise NotImplementedError("RESID_MID is not supported yet. Please use RESID_PRE/RESID_POST only.")

        self._reset_buffers(request)

        # Register hooks for non-residual streams (ATTN_OUT / MLP_OUT) if needed.
        self._register_non_resid_hooks_if_needed(request)

        all_labels: List[torch.Tensor] = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            labels = batch["labels"]
            if not torch.is_tensor(labels):
                raise TypeError("batch['labels'] must be a torch.Tensor")
            if labels.ndim != 1:
                raise ValueError(f"batch['labels'] must be 1D [B], got {tuple(labels.shape)}")

            # Store labels aligned with example axis 0
            if self.detach and (not request.require_grads):
                all_labels.append(labels.detach().to(self.store_device))
            else:
                all_labels.append(labels.to(self.store_device))

            positions = self._get_positions(batch)

            # Expose per-batch positions to hook callbacks
            self._current_positions = positions
            self._current_request = request

            need_hidden_states = any(
                s.stream in {Stream.RESID_PRE, Stream.RESID_POST}
                for s in request.sites
            )

            # Forward pass
            grad_ctx = torch.enable_grad() if request.require_grads else torch.no_grad()
            with grad_ctx:
                if self.use_amp and not request.require_grads:
                    autocast_dtype = self.dtype or self.model.dtype
                    with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                        outputs = self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=need_hidden_states,
                        )
                else:
                    outputs = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=need_hidden_states,
                    )

            # Capture residual streams from hidden_states (if requested)
            if need_hidden_states:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError(
                        "Model output has no hidden_states; ensure output_hidden_states=True is supported."
                    )

                self._record_residual_sites(
                    request=request,
                    hidden_states=hidden_states,
                    positions=positions,
                )

            # Hooks already wrote non-resid activations into self._buffer

            if request.return_logits:
                # If/when you want logits, decide storage: meta vs special site.
                # Keep it out for now to keep ActivationCache strict.
                pass

            # Clear per-batch hook context
            self._current_positions = None
            self._current_request = None

        labels_cat = torch.cat(all_labels, dim=0)

        activations: Dict[Site, torch.Tensor] = {}
        for site, chunks in self._buffer.items():
            if len(chunks) == 0:
                raise RuntimeError(f"No activations recorded for site={site}")
            activations[site] = torch.cat(chunks, dim=0)

        self.close()
        return ActivationCache(activations=activations, labels=labels_cat, variant_names=None)

    def close(self) -> None:
        """Remove any active hooks."""
        for h in self._active_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._active_hooks = []

    # -----------------------
    # Internals
    # -----------------------

    def _reset_buffers(self, request: TraceRequest) -> None:
        self._buffer = {site: [] for site in request.sites}

    def _get_positions(self, batch: Dict[str, Any]) -> Dict[PosKey, torch.Tensor]:
        pos = batch.get("positions", None)
        if pos is None or not isinstance(pos, dict):
            raise KeyError("batch must include batch['positions'] as dict[(Position, index|None), LongTensor[B]]")

        # Validate presence for any (Position,index) referenced by requested sites
        for site in self._buffer.keys():
            key: PosKey = (site.position, site.index)
            if key not in pos:
                raise KeyError(f"batch['positions'] missing key={key} (needed for site={site})")

        out: Dict[PosKey, torch.Tensor] = {}
        for k, v in pos.items():
            if (
                not isinstance(k, tuple)
                or len(k) != 2
                or not isinstance(k[0], Position)
                or (k[1] is not None and not isinstance(k[1], int))
            ):
                raise TypeError(f"positions keys must be (Position, int|None), got {k} of type {type(k)}")

            if not torch.is_tensor(v) or v.ndim != 1:
                raise TypeError(
                    f"positions[{k}] must be a 1D tensor [B], got {type(v)} with shape {getattr(v, 'shape', None)}"
                )
            out[k] = v.to(self.device)

        return out

    def _record_residual_sites(
        self,
        *,
        request: TraceRequest,
        hidden_states: Tuple[torch.Tensor, ...],
        positions: Dict[Position, torch.Tensor],
    ) -> None:
        """
        Record RESID_PRE / RESID_POST sites from hidden_states.

        HF convention:
          hidden_states[0]     = embedding output (pre layer 0)
          hidden_states[l + 1] = output after layer l

        Mapping:
          RESID_PRE  at layer l -> hidden_states[l]
          RESID_POST at layer l -> hidden_states[l + 1]
        """
        for site in request.sites:
            if site.stream not in {Stream.RESID_PRE, Stream.RESID_POST}:
                continue

            pos_idx = positions[(site.position, site.index)]
            x = self._extract_from_hidden_states(hidden_states, site=site, pos_idx=pos_idx)  # [B, d_model]

            if self.detach and (not request.require_grads):
                x = x.detach()

            if not request.store_on_device:
                x = x.to(self.store_device)

            self._buffer[site].append(x)

    def _extract_from_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        *,
        site: Site,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Extract [B, d_model] for RESID_PRE / RESID_POST from hidden_states."""
        if site.stream == Stream.RESID_POST:
            hs_index = site.layer + 1
        elif site.stream == Stream.RESID_PRE:
            hs_index = site.layer
        else:
            raise RuntimeError(f"Unexpected stream in _extract_from_hidden_states: {site.stream}")

        if hs_index < 0 or hs_index >= len(hidden_states):
            raise IndexError(
                f"hidden_states index out of range: hs_index={hs_index}, len(hidden_states)={len(hidden_states)} "
                f"for site={site}"
            )

        hs = hidden_states[hs_index]  # [B, T, d_model]
        if hs.ndim != 3:
            raise ValueError(f"Expected hidden_states[{hs_index}] to have shape [B,T,d], got {tuple(hs.shape)}")

        if pos_idx.dtype != torch.long:
            pos_idx = pos_idx.long()

        bsz = hs.shape[0]
        batch_arange = torch.arange(bsz, device=hs.device)
        return hs[batch_arange, pos_idx, :]  # [B, d_model]

    def _register_non_resid_hooks_if_needed(self, request: TraceRequest) -> None:
        """
        Register hooks for ATTN_OUT / MLP_OUT.

        We slice [B,T,d] -> [B,d] inside the hook using the *current batch positions*.
        This keeps Tracer independent of binding semantics, and avoids storing full [B,T,d] tensors.
        """
        for site in request.sites:
            if site.stream not in {Stream.ATTN_OUT, Stream.MLP_OUT}:
                continue

            def _make_hook(site_: Site):
                def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> None:
                    # act is typically [B, T, d_model] for attn/mlp modules
                    positions = self._current_positions
                    req = self._current_request
                    if positions is None or req is None:
                        # This should not happen unless hooks fire outside trace().
                        raise RuntimeError("Hook fired without current batch context (positions/request).")

                    pos_idx = positions[(site_.position, site_.index)]
                    if act.ndim == 3:
                        if pos_idx.dtype != torch.long:
                            pos_idx_ = pos_idx.long()
                        else:
                            pos_idx_ = pos_idx
                        bsz = act.shape[0]
                        batch_arange = torch.arange(bsz, device=act.device)
                        x = act[batch_arange, pos_idx_, :]  # [B, d]
                    elif act.ndim == 2:
                        # Some architectures might already output [B, d]
                        x = act
                    else:
                        raise ValueError(
                            f"Hook activation must have shape [B,T,d] or [B,d], got {tuple(act.shape)} "
                            f"for site={site_}, ctx={ctx}"
                        )

                    if self.detach and (not req.require_grads):
                        x = x.detach()

                    if not req.store_on_device:
                        x = x.to(self.store_device)

                    self._buffer[site_].append(x)
                return _hook

            handle = self.model.add_hook(layer=site.layer, stream=site.stream, hook_fn=_make_hook(site))
            self._active_hooks.append(handle)
