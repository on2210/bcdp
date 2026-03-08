# bcdp/experiments/eval_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch


@dataclass
class EvalMetrics:
    n: int
    acc: float
    margin: float


@torch.no_grad()
def forward_logits_last(handle, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    out = handle.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
    )
    return out.logits[:, -1, :]  # [B,V]


@torch.no_grad()
def accuracy_from_logits(logits_last: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits_last.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def margin_correct_vs_max_other(logits_last: torch.Tensor, labels: torch.Tensor) -> float:
    correct = logits_last.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
    tmp = logits_last.clone()
    tmp.scatter_(1, labels.view(-1, 1), float("-inf"))
    other = tmp.max(dim=-1).values
    return float((correct - other).mean().item())


@torch.no_grad()
def evaluate_base(
    handle,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> EvalMetrics:
    n_total = 0
    acc_sum = 0.0
    margin_sum = 0.0

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        labels = batch["labels"].to(device)
        logits = forward_logits_last(handle, batch, device)
        bsz = labels.numel()

        acc_sum += accuracy_from_logits(logits, labels) * bsz
        margin_sum += margin_correct_vs_max_other(logits, labels) * bsz
        n_total += bsz

    if n_total == 0:
        raise ValueError("evaluate_base: no examples")

    return EvalMetrics(n=n_total, acc=acc_sum / n_total, margin=margin_sum / n_total)