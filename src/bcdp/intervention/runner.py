# bcdp/intervention/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from bcdp.model.model_handle import ModelHandle, HookHandle
from bcdp.trace.site import Site, Stream, Position

from .base import InterventionPlan, InterventionResult


PosKey = Tuple[Position, Optional[int]]


def _get_positions_from_batch(batch: Dict[str, Any], device: torch.device) -> Dict[PosKey, torch.Tensor]:
    pos = batch.get("positions", None)
    if pos is None or not isinstance(pos, dict):
        raise KeyError("batch must include batch['positions'] as dict[(Position, index|None), LongTensor[B]]")

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
            raise TypeError(f"positions[{k}] must be 1D tensor [B], got {type(v)} shape={getattr(v,'shape',None)}")
        out[k] = v.to(device)
    return out


def _extract_at_positions(act: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    """
    act: [B,T,d] or [B,d] -> returns [B,d]
    """
    if act.ndim == 2:
        return act
    if act.ndim != 3:
        raise ValueError(f"Activation must be [B,T,d] or [B,d], got {tuple(act.shape)}")
    if pos_idx.dtype != torch.long:
        pos_idx = pos_idx.long()
    bsz = act.shape[0]
    ar = torch.arange(bsz, device=act.device)
    return act[ar, pos_idx, :]


def _record_resid_from_hidden_states(
    *,
    site: Site,
    hidden_states: Tuple[torch.Tensor, ...],
    positions: Dict[PosKey, torch.Tensor],
) -> torch.Tensor:
    """
    Match Tracer convention:
      RESID_PRE  at layer l -> hidden_states[l]
      RESID_POST at layer l -> hidden_states[l+1]
    Return [B,d] at site.position.
    """
    if site.stream == Stream.RESID_POST:
        hs_index = site.layer + 1
    elif site.stream == Stream.RESID_PRE:
        hs_index = site.layer
    else:
        raise ValueError(f"Not a resid stream: {site.stream}")

    if hs_index < 0 or hs_index >= len(hidden_states):
        raise IndexError(f"hidden_states index out of range: hs_index={hs_index}, len={len(hidden_states)} for site={site}")

    hs = hidden_states[hs_index]  # [B,T,d]
    if hs.ndim != 3:
        raise ValueError(f"Expected hidden_states[{hs_index}] [B,T,d], got {tuple(hs.shape)}")

    key = (site.position, site.index)
    if key not in positions:
        raise KeyError(f"positions missing key={key} for site={site}")
    pos_idx = positions[key].to(device=hs.device)   # [B]

    if pos_idx.dtype != torch.long:
        pos_idx = pos_idx.long()
    bsz = hs.shape[0]
    ar = torch.arange(bsz, device=hs.device)
    return hs[ar, pos_idx, :]


@dataclass
class InterventionRunner:
    model: ModelHandle
    device: Optional[torch.device] = None
    store_device: torch.device = torch.device("cpu")
    use_amp: bool = False
    dtype: Optional[torch.dtype] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = self.model.device

    def run(self, batch: Dict[str, Any], plan: InterventionPlan, *, require_grads: bool = False) -> InterventionResult:
        """
        End-to-end:
          donor pass -> collect donor activations -> build intervention -> original pass with patching
        """
        donor_batch = plan.donor_batch(batch)
        donor_acts: Optional[Dict[Site, torch.Tensor]] = None

        if donor_batch is not None:
            donor_acts = self._forward_collect_activations(
                batch=donor_batch,
                sites=plan.donor_sites(),
                apply_intervention=None,
                require_grads=False,  # donor is usually no-grad
            )

        intervention = plan.build_intervention(donor_acts or {}, batch)
        outputs = self._forward_with_intervention(
            batch=batch,
            intervention=intervention,
            require_grads=require_grads,
        )

        return InterventionResult(outputs=outputs, donor_acts=donor_acts)

    # -----------------------
    # Internals
    # -----------------------

    def _forward_collect_activations(
        self,
        *,
        batch: Dict[str, Any],
        sites: Tuple[Site, ...],
        apply_intervention: Optional[Any],
        require_grads: bool,
    ) -> Dict[Site, torch.Tensor]:
        """
        Forward pass collecting [B,d] at given sites.
        Supports RESID_PRE/POST via hidden_states, and other streams via hooks.
        """
        if len(sites) == 0:
            return {}

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        positions = _get_positions_from_batch(batch, device=self.device)

        # Decide if we need hidden_states
        need_hidden_states = any(s.stream in {Stream.RESID_PRE, Stream.RESID_POST} for s in sites)

        # Storage for non-resid hooks
        buffer: Dict[Site, torch.Tensor] = {}
        active_hooks: List[HookHandle] = []

        # Register hooks for non-resid streams
        for site in sites:
            if site.stream in {Stream.RESID_PRE, Stream.RESID_POST}:
                continue

            def _make_hook(site_: Site):
                def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> Optional[torch.Tensor]:
                    # record [B,d] at positions
                    key = (site_.position, site_.index)
                    if key not in positions:
                        raise KeyError(f"positions missing key={key} for site={site_}")
                    pos_idx = positions[key].to(act.device)

                    x = _extract_at_positions(act, pos_idx)
                    buffer[site_] = x.detach().to(self.store_device)
                    return None  # collection pass doesn't patch
                return _hook

            h = self.model.add_hook(layer=site.layer, stream=site.stream, hook_fn=_make_hook(site))
            active_hooks.append(h)

        # Forward
        grad_ctx = torch.enable_grad() if require_grads else torch.no_grad()
        try:
            with grad_ctx:
                if self.use_amp and (not require_grads):
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
        finally:
            # Cleanup hooks
            for h in active_hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        # Residual sites from hidden_states
        if need_hidden_states:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Model output has no hidden_states; ensure output_hidden_states=True is supported.")

            for site in sites:
                if site.stream not in {Stream.RESID_PRE, Stream.RESID_POST}:
                    continue
                x = _record_resid_from_hidden_states(site=site, hidden_states=hidden_states, positions=positions)
                buffer[site] = x.detach().to(self.store_device)

        # Validate all requested sites recorded
        for s in sites:
            if s not in buffer:
                raise RuntimeError(f"Failed to record activations for site={s}")

        return buffer

    def _forward_with_intervention(
        self,
        *,
        batch: Dict[str, Any],
        intervention: Any,
        require_grads: bool,
    ) -> Any:
        """
        Forward pass with patch hooks installed.
        The hook ctx includes positions so interventions can edit only the relevant token positions.
        """
        sites = intervention.sites()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        positions = _get_positions_from_batch(batch, device=self.device)
        need_hidden_states = any(s.stream in {Stream.RESID_PRE, Stream.RESID_POST} for s in sites)

        active_hooks: List[HookHandle] = []

        for site in sites:
            hook_fn = intervention.hook_for(site)

            def _make_wrapped(site_: Site, hook_fn_):
                def _hook(act: torch.Tensor, ctx: Dict[str, Any]) -> Optional[torch.Tensor]:
                    # enrich ctx so interventions can slice positions
                    ctx2 = dict(ctx)
                    ctx2["positions"] = positions
                    ctx2["site"] = site_
                    return hook_fn_(act, ctx2)
                return _hook

            h = self.model.add_hook(layer=site.layer, stream=site.stream, hook_fn=_make_wrapped(site, hook_fn))
            active_hooks.append(h)

        grad_ctx = torch.enable_grad() if require_grads else torch.no_grad()
        try:
            with grad_ctx:
                if self.use_amp and (not require_grads):
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
        finally:
            for h in active_hooks:
                try:
                    h.remove()
                except Exception:
                    pass

        return outputs
