# bcdp/eval/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from bcdp.model.model_handle import ModelHandle
from bcdp.trace.site import Position
from bcdp.eval.metrics import BatchMetrics, summarize_batch

# Optional dependency: intervention runner
try:
    from bcdp.intervention.base import InterventionPlan
    from bcdp.intervention.runner import InterventionRunner
except Exception:  # keep eval usable without intervention package in some contexts
    InterventionPlan = Any
    InterventionRunner = Any

PosKey = Tuple[Position, Optional[int]]


def _extract_logits(outputs: Any) -> torch.Tensor:
    """
    Normalize various model output formats to a logits tensor.
    Supports:
      - dict-like outputs with "logits"
      - HF ModelOutput with .logits
    """
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits  # HF-style
    raise TypeError(f"Cannot extract logits from outputs type={type(outputs)}")


def _positions_from_batch(batch: Dict[str, Any]) -> Optional[Dict[PosKey, torch.Tensor]]:
    pos = batch.get("positions", None)
    return pos if isinstance(pos, dict) else None


def _pick_eval_token_index(
    *,
    batch: Dict[str, Any],
    default: PosKey,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a LongTensor [B] token index per example.

    Priority:
      1) batch["positions"][default] if present
      2) "last non-pad" computed from attention_mask if available
      3) fallback to T-1
    """
    positions = _positions_from_batch(batch)
    if positions is not None and default in positions:
        idx = positions[default]
        if not torch.is_tensor(idx) or idx.ndim != 1:
            raise TypeError(f"positions[{default}] must be a 1D tensor [B], got {type(idx)} shape={getattr(idx,'shape',None)}")
        return idx.to(device).long()

    input_ids = batch["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise TypeError("batch['input_ids'] must be LongTensor [B,T]")
    B, T = input_ids.shape

    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        if not torch.is_tensor(attention_mask) or attention_mask.shape != (B, T):
            raise TypeError("batch['attention_mask'] must be Tensor [B,T] matching input_ids")
        lengths = attention_mask.long().sum(dim=1)  # [B]
        return torch.clamp(lengths - 1, min=0).to(device).long()

    return torch.full((B,), T - 1, dtype=torch.long, device=device)


def _slice_logits_at_positions(logits_btv: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    """
    logits_btv: [B,T,V]
    pos_idx:    [B]
    returns:    [B,V]
    """
    if logits_btv.ndim == 2:
        return logits_btv
    if logits_btv.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got {tuple(logits_btv.shape)}")
    if pos_idx.ndim != 1:
        raise ValueError(f"pos_idx must be [B], got {tuple(pos_idx.shape)}")
    if pos_idx.dtype != torch.long:
        pos_idx = pos_idx.long()

    B = logits_btv.shape[0]
    ar = torch.arange(B, device=logits_btv.device)
    return logits_btv[ar, pos_idx, :]


@dataclass(frozen=True)
class EvalConfig:
    """
    Default evaluation is next-token style:
      - take logits at a chosen token position
      - compare argmax to label vocab id
    """
    eval_pos: PosKey = (Position.LAST, None)
    return_per_example: bool = False
    use_amp: bool = False
    amp_dtype: Optional[torch.dtype] = None  # if None, use model dtype


@dataclass(frozen=True)
class EvalResult:
    """
    Aggregated over all batches.
    """
    n: int
    mean_correct_logit: float
    mean_correct_logprob: float
    mean_accuracy: float
    mean_margin_vs_max_other: float
    per_batch: Optional[List[BatchMetrics]] = None


class Evaluator:
    def __init__(
        self,
        model: ModelHandle,
        *,
        device: Optional[torch.device] = None,
        cfg: EvalConfig = EvalConfig(),
        runner: Optional[InterventionRunner] = None,
    ) -> None:
        self.model = model
        self.device = device or model.device
        self.cfg = cfg
        self.runner = runner  # optional; used only if a plan is provided

    def evaluate(
        self,
        dataloader: Iterable[Dict[str, Any]],
        *,
        intervention_plan: Optional[InterventionPlan] = None,
        require_grads: bool = False,
    ) -> EvalResult:
        """
        Evaluate over batches. If intervention_plan is provided, uses InterventionRunner to run
        donor->patch->original (or patch-only), and evaluates outputs from that run.

        Assumes batch contains:
          - input_ids: [B,T]
          - labels:    [B] (vocab_id mode is typical)  (see collate.py)
          - positions: dict[(Position, idx|None)] -> [B] (optional but recommended)
        """
        per_batch: List[BatchMetrics] = []
        totals = {
            "n": 0,
            "sum_correct_logit": 0.0,
            "sum_correct_logprob": 0.0,
            "sum_accuracy": 0.0,
            "sum_margin": 0.0,
        }

        grad_ctx = torch.enable_grad() if require_grads else torch.no_grad()

        for batch in dataloader:
            if "input_ids" not in batch or "labels" not in batch:
                raise KeyError("batch must contain 'input_ids' and 'labels'")

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            if labels.ndim != 1:
                raise ValueError(f"batch['labels'] must be [B], got {tuple(labels.shape)}")

            # Which token are we evaluating at?
            pos_idx = _pick_eval_token_index(batch=batch, default=self.cfg.eval_pos, device=self.device)

            with grad_ctx:
                if intervention_plan is None:
                    if self.cfg.use_amp and (not require_grads):
                        dtype = self.cfg.amp_dtype or self.model.dtype
                        with torch.autocast(device_type=self.device.type, dtype=dtype):
                            outputs = self.model.forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=False,
                            )
                    else:
                        outputs = self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                        )
                else:
                    if self.runner is None:
                        raise ValueError("Evaluator.evaluate: intervention_plan was provided but Evaluator.runner is None")
                    # Runner expects the original batch dict; make sure tensors are on cpu/host
                    # or leave as-is; runner will move to device internally.
                    outputs = self.runner.run(batch, intervention_plan, require_grads=require_grads).outputs

            logits = _extract_logits(outputs)  # [B,T,V] typically
            logits_bv = _slice_logits_at_positions(logits, pos_idx)  # [B,V]

            bm = summarize_batch(
                logits_bv,
                labels,
                return_per_example=self.cfg.return_per_example,
            )
            per_batch.append(bm)

            totals["n"] += bm.n
            totals["sum_correct_logit"] += bm.mean_correct_logit * bm.n
            totals["sum_correct_logprob"] += bm.mean_correct_logprob * bm.n
            totals["sum_accuracy"] += bm.mean_accuracy * bm.n
            totals["sum_margin"] += bm.mean_margin_vs_max_other * bm.n

        n = totals["n"]
        if n == 0:
            raise ValueError("Empty dataloader (no batches/examples)")

        return EvalResult(
            n=n,
            mean_correct_logit=totals["sum_correct_logit"] / n,
            mean_correct_logprob=totals["sum_correct_logprob"] / n,
            mean_accuracy=totals["sum_accuracy"] / n,
            mean_margin_vs_max_other=totals["sum_margin"] / n,
            per_batch=per_batch,
        )
