# bcdp/eval/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class BatchMetrics:
    """
    Per-batch metrics (scalars) plus optional per-example tensors.
    """
    n: int
    mean_correct_logit: float
    mean_correct_logprob: float
    mean_accuracy: float
    mean_margin_vs_max_other: float

    # Optional per-example fields (useful for debugging / stratification)
    correct_logit: Optional[torch.Tensor] = None        # [B]
    correct_logprob: Optional[torch.Tensor] = None      # [B]
    accuracy: Optional[torch.Tensor] = None             # [B]
    margin_vs_max_other: Optional[torch.Tensor] = None  # [B]


def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to [B, V] for next-token evaluation.
    Accepts [B, V] or [B, T, V] (caller should slice T).
    """
    if logits.ndim == 2:
        return logits
    raise ValueError(f"Expected logits [B,V], got {tuple(logits.shape)}")


def gather_correct_logits(logits_bv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits_bv: [B, V]
    labels:    [B] long
    returns:   [B]
    """
    logits_bv = _ensure_2d_logits(logits_bv)
    if labels.ndim != 1:
        raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")
    if labels.dtype != torch.long:
        labels = labels.long()

    return logits_bv.gather(1, labels.unsqueeze(1)).squeeze(1)


def correct_logprobs(logits_bv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Returns log softmax probability of the correct label for each example. [B]
    """
    logits_bv = _ensure_2d_logits(logits_bv)
    if labels.dtype != torch.long:
        labels = labels.long()
    logp = torch.log_softmax(logits_bv, dim=-1)
    return logp.gather(1, labels.unsqueeze(1)).squeeze(1)


def accuracy_from_logits(logits_bv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Returns per-example 0/1 accuracy. [B]
    """
    logits_bv = _ensure_2d_logits(logits_bv)
    pred = torch.argmax(logits_bv, dim=-1)
    if labels.dtype != torch.long:
        labels = labels.long()
    return (pred == labels).float()


def margin_vs_max_other(logits_bv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    margin = logit(correct) - max_{v != correct} logit(v)
    Returns [B].

    Uses topk(2) to avoid cloning logits.
    """
    logits_bv = _ensure_2d_logits(logits_bv)
    if labels.dtype != torch.long:
        labels = labels.long()

    correct = gather_correct_logits(logits_bv, labels)  # [B]

    # top2 values/indices: [B,2]
    top2_vals, top2_idx = torch.topk(logits_bv, k=2, dim=-1)

    top1_val = top2_vals[:, 0]
    top1_idx = top2_idx[:, 0]
    top2_val = top2_vals[:, 1]

    # If top1 is the correct label, "max other" is top2; else it's top1.
    max_other = torch.where(top1_idx == labels, top2_val, top1_val)

    return correct - max_other



def pairwise_logit_diff(
    logits_bv: torch.Tensor,
    *,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
) -> torch.Tensor:
    """
    For binary forced-choice evaluations:
      diff = logit(pos_id) - logit(neg_id)
    pos_ids, neg_ids: [B] long
    Returns [B].
    """
    logits_bv = _ensure_2d_logits(logits_bv)
    if pos_ids.dtype != torch.long:
        pos_ids = pos_ids.long()
    if neg_ids.dtype != torch.long:
        neg_ids = neg_ids.long()

    pos = logits_bv.gather(1, pos_ids.unsqueeze(1)).squeeze(1)
    neg = logits_bv.gather(1, neg_ids.unsqueeze(1)).squeeze(1)
    return pos - neg


def summarize_batch(
    logits_bv: torch.Tensor,
    labels: torch.Tensor,
    *,
    return_per_example: bool = False,
) -> BatchMetrics:
    """
    Compute the default bundle of metrics for a batch.
    """
    logits_bv = _ensure_2d_logits(logits_bv)

    cl = gather_correct_logits(logits_bv, labels)
    cp = correct_logprobs(logits_bv, labels)
    acc = accuracy_from_logits(logits_bv, labels)
    mar = margin_vs_max_other(logits_bv, labels)

    out = BatchMetrics(
        n=int(logits_bv.shape[0]),
        mean_correct_logit=float(cl.mean().item()),
        mean_correct_logprob=float(cp.mean().item()),
        mean_accuracy=float(acc.mean().item()),
        mean_margin_vs_max_other=float(mar.mean().item()),
        correct_logit=cl if return_per_example else None,
        correct_logprob=cp if return_per_example else None,
        accuracy=acc if return_per_example else None,
        margin_vs_max_other=mar if return_per_example else None,
    )
    return out
