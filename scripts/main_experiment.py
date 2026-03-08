# bcdp/experiments/main_experiment.py
# End-to-end "main experiment" runner with:
# - layerwise DBCM (LAST token / resid_post by default)
# - necessity (project-out) + random controls
# - sufficiency (interchange transfer) + random controls
# - head ranking + ablation necessity (+ random controls) + success rates
# - MLP writer ranking + ablation necessity (+ random controls) + success rates
# - split-half stability
# - merged subspaces (all/low/high) evaluation
# - JSON artifacts + publication-ready plots (titles + legends)

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from enum import Enum
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position
from bcdp.utils.linalg import svd_right_vectors

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.intervention.runner import InterventionRunner
from bcdp.subspace.dbcm import DBCM, DBCMResult
from bcdp.intervention.project_out_plan import ProjectOutPlan
from bcdp.intervention.interchange_plan import InterchangePlan

from bcdp.ranking.head_ranking_hf import rank_heads_hf, HeadScore
from bcdp.ranking.mlp_writers_hf import rank_mlp_writers_hf, MLPWriterScore
from bcdp.intervention.attn_o_proj_mask import mask_attn_o_proj_heads, HeadRef
from bcdp.intervention.mlp_weight_mask import mask_mlp_writers

from bcdp.experiments.eval_utils import (
    forward_logits_last,
    accuracy_from_logits,
    margin_correct_vs_max_other,
    evaluate_base,
)

import matplotlib.pyplot as plt

from bcdp.subspace.dbcm import DBCMResult
torch.serialization.add_safe_globals([DBCMResult])


# ----------------------------
# Config
# ----------------------------

@dataclass
class MainConfig:
    # model
    model_name: str = "google/gemma-2-2b"
    dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    device_map: str = "auto"

    # site
    stream: Stream = Stream.RESID_POST
    position: Position = Position.LAST

    # data
    dataset_seed: int = 123
    num_examples_total: int = 8192
    batch_size: int = 16
    max_length: int = 128

    # subspace training
    svd_rows: int = 512
    dbcm_steps: int = 200
    dbcm_epochs: int = 0          # if >0 uses fit_epoch over paired loaders
    dbcm_max_batches: int = 128   # for fit_epoch
    lambda_l1: float = 1e-3
    lr: float = 1e-2

    # eval
    eval_batches: int = 256

    # random controls
    n_random: int = 10

    # component tests
    topH_heads: int = 20
    topM_writers: int = 50
    writers_cap_per_layer: int = 10

    # stability
    do_split_half: bool = True

    # merges
    do_merges: bool = True
    merge_k_mode: str = "median"   # "median" | "min" | "fixed"
    merge_k_fixed: int = 32

    # output
    out_root: str = "runs"
    run_tag: str = "main"
    seed: int = 0


# ----------------------------
# IO + misc
# ----------------------------

def _torch_dtype(cfg: MainConfig) -> torch.dtype:
    if cfg.dtype == "bf16":
        return torch.bfloat16
    if cfg.dtype == "fp16":
        return torch.float16
    return torch.float32


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_jsonable(x):
    if isinstance(x, Enum):
        return x.value if hasattr(x, "value") else x.name
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.dtype):
        return str(x)
    return x

def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=to_jsonable)


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _random_projection(d_model: int, k: int, device: torch.device) -> torch.Tensor:
    Q = torch.randn(d_model, k, device=device)
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    return Q @ Q.T


# ----------------------------
# Plotting (titles + legends always)
# ----------------------------

def _plot_line_with_band(
    *,
    x: np.ndarray,
    y: np.ndarray,
    band_mean: Optional[np.ndarray],
    band_std: Optional[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    y_label_main: str = "BCDP / intervention",
    band_label_mean: str = "Random mean",
    band_label_band: str = "Random ±1σ",
):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, marker="o", label=y_label_main)

    if band_mean is not None and band_std is not None:
        lower = band_mean - band_std
        upper = band_mean + band_std
        plt.fill_between(x, lower, upper, alpha=0.2, label=band_label_band)
        plt.plot(x, band_mean, linestyle="--", label=band_label_mean)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_line(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    label: str = "BCDP",
):

    plt.figure()
    plt.plot(x, y, marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_bar(
    *,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
    label: str = "BCDP",
):
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values, label=label)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _summarize_random_band(rows: List[Dict[str, Any]], list_key: str, field: str) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for r in rows:
        lst = r.get(list_key, [])
        vals = []
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, dict) and field in item:
                    vals.append(float(item[field]))
        if not vals:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
    return np.array(means, dtype=np.float64), np.array(stds, dtype=np.float64)


# ----------------------------
# Success-rate metrics
# ----------------------------

@torch.no_grad()
def _forward_preds_last(runner: InterventionRunner, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    logits = forward_logits_last(runner.model, batch, device)
    return logits.argmax(dim=-1)


@torch.no_grad()
def evaluate_disruption_success(
    *,
    runner: InterventionRunner,
    loader: DataLoader,
    device: torch.device,
    forward_after: callable,
    max_batches: int,
) -> Dict[str, float]:
    """
    Generic "disruption success" for necessity-style interventions.

    Success definition:
        correct BEFORE AND incorrect AFTER

    Also reports flip_rate:
        predicted token changed (regardless of correctness)
    """
    n_total = 0
    n_base_correct = 0
    n_success = 0
    n_flip = 0

    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        labels = b["labels"].to(device)
        bsz = int(labels.numel())

        pred_before = _forward_preds_last(runner, b, device)
        base_correct = (pred_before == labels)

        pred_after = forward_after(b)

        n_total += bsz
        n_base_correct += int(base_correct.sum().item())
        n_success += int((base_correct & (pred_after != labels)).sum().item())
        n_flip += int((pred_after != pred_before).sum().item())

    denom = max(n_base_correct, 1)
    return {
        "n": float(n_total),
        "base_correct_n": float(n_base_correct),
        "success_rate": float(n_success) / float(denom),
        "flip_rate": float(n_flip) / float(max(n_total, 1)),
    }


@torch.no_grad()
def evaluate_project_out_metrics(
    *,
    runner: InterventionRunner,
    handle: HuggingFaceCausalLMHandle,
    loader: DataLoader,
    device: torch.device,
    plan: ProjectOutPlan,
    max_batches: int,
) -> Dict[str, float]:
    n_total = 0
    acc_sum = 0.0
    mar_sum = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        labels = b["labels"].to(device)
        bsz = int(labels.numel())

        out = runner.run(b, plan, require_grads=False)
        logits = out.outputs.logits[:, -1, :]
        acc_sum += accuracy_from_logits(logits, labels) * bsz
        mar_sum += margin_correct_vs_max_other(logits, labels) * bsz
        n_total += bsz

    if n_total == 0:
        raise ValueError("evaluate_project_out_metrics: no examples")
    return {"n": float(n_total), "acc": acc_sum / n_total, "margin": mar_sum / n_total}


@torch.no_grad()
def evaluate_transfer_metrics(
    *,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    site: Site,
    W: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> Dict[str, Any]:
    """
    recipient = swap
    donor = base

    We report recipient metrics + donor-label metrics on recipient logits (before/after),
    and deltas for donor metrics.
    """
    n_total = 0
    rec_acc_b = rec_acc_a = rec_mar_b = rec_mar_a = 0.0
    don_acc_b = don_acc_a = don_mar_b = don_mar_a = 0.0

    for bi, (b_base, b_swap) in enumerate(zip(loader_base, loader_swap)):
        if bi >= max_batches:
            break

        labels_rec = b_swap["labels"].to(device)
        labels_don = b_base["labels"].to(device)
        bsz = int(labels_rec.numel())

        logits_before = forward_logits_last(runner.model, b_swap, device)

        rec_acc_b += accuracy_from_logits(logits_before, labels_rec) * bsz
        rec_mar_b += margin_correct_vs_max_other(logits_before, labels_rec) * bsz
        don_acc_b += accuracy_from_logits(logits_before, labels_don) * bsz
        don_mar_b += margin_correct_vs_max_other(logits_before, labels_don) * bsz

        def donor_builder(_batch: Dict[str, Any]) -> Dict[str, Any]:
            return b_base

        plan = InterchangePlan(donor_builder=donor_builder, site=site, W=W)
        res = runner.run(b_swap, plan, require_grads=False)
        logits_after = res.outputs.logits[:, -1, :]

        rec_acc_a += accuracy_from_logits(logits_after, labels_rec) * bsz
        rec_mar_a += margin_correct_vs_max_other(logits_after, labels_rec) * bsz
        don_acc_a += accuracy_from_logits(logits_after, labels_don) * bsz
        don_mar_a += margin_correct_vs_max_other(logits_after, labels_don) * bsz

        n_total += bsz

    if n_total == 0:
        raise ValueError("evaluate_transfer_metrics: no examples")

    return {
        "n": float(n_total),
        "recipient_acc_before": rec_acc_b / n_total,
        "recipient_acc_after": rec_acc_a / n_total,
        "recipient_margin_before": rec_mar_b / n_total,
        "recipient_margin_after": rec_mar_a / n_total,
        "donor_acc_before": don_acc_b / n_total,
        "donor_acc_after": don_acc_a / n_total,
        "donor_margin_before": don_mar_b / n_total,
        "donor_margin_after": don_mar_a / n_total,
        "delta_donor_acc": (don_acc_a - don_acc_b) / n_total,
        "delta_donor_margin": (don_mar_a - don_mar_b) / n_total,
    }


@torch.no_grad()
def evaluate_transfer_success(
    *,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    site: Site,
    W: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """
    Sufficiency success:
      success_rate = P( pred_after == donor_label ) on recipient logits.
      flip_to_donor_rate = P( pred_before != donor_label AND pred_after == donor_label )
    """
    n_total = 0
    n_success = 0
    n_flip_to_donor = 0

    for bi, (b_base, b_swap) in enumerate(zip(loader_base, loader_swap)):
        if bi >= max_batches:
            break

        labels_don = b_base["labels"].to(device)
        bsz = int(labels_don.numel())

        logits_before = forward_logits_last(runner.model, b_swap, device)
        pred_before = logits_before.argmax(dim=-1)

        def donor_builder(_batch: Dict[str, Any]) -> Dict[str, Any]:
            return b_base

        plan = InterchangePlan(donor_builder=donor_builder, site=site, W=W)
        res = runner.run(b_swap, plan, require_grads=False)
        logits_after = res.outputs.logits[:, -1, :]
        pred_after = logits_after.argmax(dim=-1)

        n_total += bsz
        n_success += int((pred_after == labels_don).sum().item())
        n_flip_to_donor += int(((pred_before != labels_don) & (pred_after == labels_don)).sum().item())

    return {
        "n": float(n_total),
        "success_rate": float(n_success) / float(max(n_total, 1)),
        "flip_to_donor_rate": float(n_flip_to_donor) / float(max(n_total, 1)),
    }


# ----------------------------
# Ranking helpers
# ----------------------------

def _top_writers_to_dict(
    writers: List[MLPWriterScore],
    topk_total: int,
    cap_per_layer: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    total = 0
    for w in writers:
        if total >= topk_total:
            break
        layer = int(w.layer)
        out.setdefault(layer, [])
        if len(out[layer]) >= cap_per_layer:
            continue
        out[layer].append(int(w.neuron))
        total += 1
    return out


def _sample_random_heads(*, n_layers: int, n_heads: int, k: int, rng: random.Random) -> List[HeadRef]:
    return [HeadRef(layer=rng.randrange(n_layers), head=rng.randrange(n_heads)) for _ in range(k)]


def _sample_random_writers_like(
    *,
    handle: HuggingFaceCausalLMHandle,
    like: Dict[int, List[int]],
    rng: random.Random,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for layer, neurons in like.items():
        out_proj = handle.resolve_mlp_out_proj(int(layer))
        d_mlp = out_proj.weight.shape[1]
        out[layer] = [rng.randrange(d_mlp) for _ in range(len(neurons))]
    return out


# ----------------------------
# Merge helpers
# ----------------------------

def _choose_merge_k(cfg: MainConfig, ks: List[int]) -> int:
    ks = [int(x) for x in ks if int(x) > 0]
    if not ks:
        return cfg.merge_k_fixed
    if cfg.merge_k_mode == "min":
        return int(min(ks))
    if cfg.merge_k_mode == "fixed":
        return int(cfg.merge_k_fixed)
    ks_sorted = sorted(ks)
    return int(ks_sorted[len(ks_sorted) // 2])


def _merge_basis_to_projector(bases: List[torch.Tensor], k: int, device: torch.device) -> torch.Tensor:
    """
    bases: list of [d, k_i] (columns)
    Build a merged orthonormal basis via SVD on concatenated bases, then return rank-k projector.
    """
    if not bases:
        raise ValueError("merge: empty bases list")

    B = torch.cat([b.float().cpu() for b in bases if b is not None and b.numel() > 0], dim=1)
    if B.numel() == 0:
        raise ValueError("merge: concatenated basis is empty")

    U, _, _ = torch.linalg.svd(B, full_matrices=False)
    k = min(k, U.shape[1])
    U = U[:, :k].contiguous()
    return (U @ U.T).to(device)


# ----------------------------
# Per-layer experiment
# ----------------------------

def run_one_layer(
    *,
    layer: int,
    handle: HuggingFaceCausalLMHandle,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    device: torch.device,
    cfg: MainConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    site = Site(layer=layer, stream=cfg.stream, position=cfg.position, index=None)

    # ---- Build X for SVD basis V (from base activations)
    X_list: List[torch.Tensor] = []
    got = 0
    for b in loader_base:
        acts = runner._forward_collect_activations(
            batch=b,
            sites=(site,),
            apply_intervention=None,
            require_grads=False,
        )
        x = acts[site].detach().float().cpu()
        X_list.append(x)
        got += x.shape[0]
        if got >= cfg.svd_rows:
            break
    X = torch.cat(X_list, dim=0)[: cfg.svd_rows]  # [N,d]

    V = svd_right_vectors(X, center=True)  # [d,r] float32 CPU

    # ---- Fit DBCM
    dbcm = DBCM(
        runner=runner,
        site=site,
        lambda_l1=cfg.lambda_l1,
        lr=cfg.lr,
        steps=cfg.dbcm_steps,
        device=device,
    )

    if cfg.dbcm_epochs and cfg.dbcm_epochs > 0:
        res = dbcm.fit_epoch(
            V=V,
            loader_o=loader_base,
            loader_c=loader_swap,
            epochs=cfg.dbcm_epochs,
            max_batches=cfg.dbcm_max_batches,
            log_every=50,
        )
    else:
        batch_o = next(iter(loader_base))
        batch_c = next(iter(loader_swap))
        res = dbcm.fit(V=V, batch_o=batch_o, batch_c=batch_c)

    W = res.projection.to(device)
    k = int(res.mask.sum().item())

    # save candidate
    cand_path = out_dir / "candidates" / f"layer_{layer:02d}.pt"
    _ensure_dir(cand_path.parent)
    torch.save(res, cand_path)

    # ---- Baseline eval (base)
    base_metrics = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

    # ---- Necessity: project-out W + random
    nec_plan = ProjectOutPlan(site=site, W=W)
    nec_after = evaluate_project_out_metrics(
        runner=runner, handle=handle, loader=loader_base, device=device, plan=nec_plan, max_batches=cfg.eval_batches
    )

    def _after_project_out(batch: Dict[str, Any]) -> torch.Tensor:
        out = runner.run(batch, nec_plan, require_grads=False)
        return out.outputs.logits[:, -1, :].argmax(dim=-1)

    nec_success = evaluate_disruption_success(
        runner=runner,
        loader=loader_base,
        device=device,
        forward_after=_after_project_out,
        max_batches=cfg.eval_batches,
    )

    nec_metrics = {
        "n": nec_after["n"],
        "acc": nec_after["acc"],
        "margin": nec_after["margin"],
        "delta_acc": nec_after["acc"] - base_metrics.acc,
        "delta_margin": nec_after["margin"] - base_metrics.margin,
        "success_rate": nec_success["success_rate"],
        "flip_rate": nec_success["flip_rate"],
        "base_correct_n": nec_success["base_correct_n"],
    }

    rng = random.Random(cfg.seed + 10_000 + layer)
    rand_nec_list: List[Dict[str, Any]] = []
    for _ in range(cfg.n_random):
        W_rand = _random_projection(handle.d_model, max(k, 1), device=device)
        plan = ProjectOutPlan(site=site, W=W_rand)

        m = evaluate_project_out_metrics(
            runner=runner, handle=handle, loader=loader_base, device=device, plan=plan, max_batches=cfg.eval_batches
        )

        def _after_rand(batch: Dict[str, Any], _plan=plan) -> torch.Tensor:
            out = runner.run(batch, _plan, require_grads=False)
            return out.outputs.logits[:, -1, :].argmax(dim=-1)

        s = evaluate_disruption_success(
            runner=runner, loader=loader_base, device=device, forward_after=_after_rand, max_batches=cfg.eval_batches
        )

        rand_nec_list.append({
            "acc": m["acc"],
            "margin": m["margin"],
            "delta_acc": m["acc"] - base_metrics.acc,
            "delta_margin": m["margin"] - base_metrics.margin,
            "success_rate": s["success_rate"],
            "flip_rate": s["flip_rate"],
        })

    # ---- Sufficiency: transfer + random
    suf_metrics = evaluate_transfer_metrics(
        runner=runner,
        loader_base=loader_base,
        loader_swap=loader_swap,
        site=site,
        W=W,
        device=device,
        max_batches=cfg.eval_batches,
    )
    suf_success = evaluate_transfer_success(
        runner=runner,
        loader_base=loader_base,
        loader_swap=loader_swap,
        site=site,
        W=W,
        device=device,
        max_batches=cfg.eval_batches,
    )
    suf_metrics["success_rate"] = suf_success["success_rate"]
    suf_metrics["flip_to_donor_rate"] = suf_success["flip_to_donor_rate"]

    rand_suf_list: List[Dict[str, Any]] = []
    for _ in range(cfg.n_random):
        W_rand = _random_projection(handle.d_model, max(k, 1), device=device)

        rand_suf = evaluate_transfer_metrics(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=W_rand,
            device=device,
            max_batches=cfg.eval_batches,
        )
        rand_suc = evaluate_transfer_success(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=W_rand,
            device=device,
            max_batches=cfg.eval_batches,
        )
        rand_suf["success_rate"] = rand_suc["success_rate"]
        rand_suf["flip_to_donor_rate"] = rand_suc["flip_to_donor_rate"]
        rand_suf_list.append(rand_suf)

    # ---- Head ranking + necessity (+ success + random)
    head_scores: List[HeadScore] = rank_heads_hf(
        handle=handle,
        dataloader=loader_base,
        W=W,
        position=cfg.position,
        max_batches=cfg.eval_batches,
    )
    top_heads = head_scores[: cfg.topH_heads]
    top_head_refs = [HeadRef(layer=h.layer, head=h.head) for h in top_heads]

    with mask_attn_o_proj_heads(handle=handle, heads=top_head_refs, mode="ablate"):
        ab_metrics = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

        def _after_head_ablate(batch: Dict[str, Any]) -> torch.Tensor:
            # model state is already masked inside the context
            return _forward_preds_last(runner, batch, device)

        head_succ = evaluate_disruption_success(
            runner=runner,
            loader=loader_base,
            device=device,
            forward_after=_after_head_ablate,
            max_batches=cfg.eval_batches,
        )

    head_nec = {
        "delta_acc": ab_metrics.acc - base_metrics.acc,
        "delta_margin": ab_metrics.margin - base_metrics.margin,
        "success_rate": head_succ["success_rate"],
        "flip_rate": head_succ["flip_rate"],
        "base_correct_n": head_succ["base_correct_n"],
    }

    rand_head_nec: List[Dict[str, Any]] = []
    for _ in range(cfg.n_random):
        rnd = _sample_random_heads(
            n_layers=handle.n_layers,
            n_heads=int(handle.n_heads or 0),
            k=len(top_head_refs),
            rng=rng,
        )
        with mask_attn_o_proj_heads(handle=handle, heads=rnd, mode="ablate"):
            m = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

            def _after_rand_head(batch: Dict[str, Any]) -> torch.Tensor:
                return _forward_preds_last(runner, batch, device)

            s = evaluate_disruption_success(
                runner=runner,
                loader=loader_base,
                device=device,
                forward_after=_after_rand_head,
                max_batches=cfg.eval_batches,
            )

        rand_head_nec.append({
            "delta_acc": m.acc - base_metrics.acc,
            "delta_margin": m.margin - base_metrics.margin,
            "success_rate": s["success_rate"],
            "flip_rate": s["flip_rate"],
        })

    # ---- Writer ranking + necessity (+ success + random)
    writer_scores = rank_mlp_writers_hf(handle=handle, W=W, topk_per_layer=cfg.topM_writers, normalize=True)
    layer_to_neurons = _top_writers_to_dict(
        writer_scores, topk_total=cfg.topM_writers, cap_per_layer=cfg.writers_cap_per_layer
    )

    with mask_mlp_writers(handle=handle, layer_to_neurons=layer_to_neurons, mode="ablate"):
        w_ab = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

        def _after_writer_ablate(batch: Dict[str, Any]) -> torch.Tensor:
            return _forward_preds_last(runner, batch, device)

        w_succ = evaluate_disruption_success(
            runner=runner,
            loader=loader_base,
            device=device,
            forward_after=_after_writer_ablate,
            max_batches=cfg.eval_batches,
        )

    writer_nec = {
        "delta_acc": w_ab.acc - base_metrics.acc,
        "delta_margin": w_ab.margin - base_metrics.margin,
        "counts_per_layer": {str(k): len(v) for k, v in layer_to_neurons.items()},
        "success_rate": w_succ["success_rate"],
        "flip_rate": w_succ["flip_rate"],
        "base_correct_n": w_succ["base_correct_n"],
    }

    rand_writer_nec: List[Dict[str, Any]] = []
    for _ in range(cfg.n_random):
        rnd_map = _sample_random_writers_like(handle=handle, like=layer_to_neurons, rng=rng)
        with mask_mlp_writers(handle=handle, layer_to_neurons=rnd_map, mode="ablate"):
            m = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

            def _after_rand_writer(batch: Dict[str, Any]) -> torch.Tensor:
                return _forward_preds_last(runner, batch, device)

            s = evaluate_disruption_success(
                runner=runner,
                loader=loader_base,
                device=device,
                forward_after=_after_rand_writer,
                max_batches=cfg.eval_batches,
            )

        rand_writer_nec.append({
            "delta_acc": m.acc - base_metrics.acc,
            "delta_margin": m.margin - base_metrics.margin,
            "success_rate": s["success_rate"],
            "flip_rate": s["flip_rate"],
        })

    # ---- Stability split-half
    stability = None
    if cfg.do_split_half and X.shape[0] >= 4:
        half = X.shape[0] // 2
        V_A = svd_right_vectors(X[:half], center=True)
        V_B = svd_right_vectors(X[half:], center=True)

        if cfg.dbcm_epochs and cfg.dbcm_epochs > 0:
            res_A = dbcm.fit_epoch(V=V_A, loader_o=loader_base, loader_c=loader_swap, epochs=1, max_batches=cfg.dbcm_max_batches, log_every=0)
            res_B = dbcm.fit_epoch(V=V_B, loader_o=loader_base, loader_c=loader_swap, epochs=1, max_batches=cfg.dbcm_max_batches, log_every=0)
        else:
            batch_o = next(iter(loader_base))
            batch_c = next(iter(loader_swap))
            res_A = dbcm.fit(V=V_A, batch_o=batch_o, batch_c=batch_c)
            res_B = dbcm.fit(V=V_B, batch_o=batch_o, batch_c=batch_c)

        WA = res_A.projection.to(device)
        WB = res_B.projection.to(device)
        stability = float(torch.norm(WA @ WB).item() / (torch.norm(WA).item() + 1e-8))

    metrics = {
        "layer": layer,
        "k": k,
        "site": {"layer": layer, "stream": str(cfg.stream), "position": str(cfg.position)},
        "baseline_base": asdict(base_metrics),
        "subspace_necessity": nec_metrics,
        "subspace_necessity_random": rand_nec_list,
        "subspace_sufficiency_transfer": suf_metrics,
        "subspace_sufficiency_transfer_random": rand_suf_list,
        "head_ranking_top": [asdict(x) for x in head_scores[:50]],
        "head_necessity_topH": head_nec,
        "head_necessity_random": rand_head_nec,
        "writer_ranking_top": [asdict(x) for x in writer_scores[:50]],
        "writer_necessity_topM": writer_nec,
        "writer_necessity_random": rand_writer_nec,
        "stability_split_half": stability,
    }

    _save_json(out_dir / "metrics" / f"layer_{layer:02d}.json", metrics)

    print(
        f"[DONE] layer={layer:02d} k={k} "
        f"necΔm={nec_metrics['delta_margin']:+.3f} necSR={nec_metrics['success_rate']:.3f} "
        f"sufΔdonAcc={suf_metrics['delta_donor_acc']:+.3f} sufSR={suf_metrics['success_rate']:.3f} "
        f"headΔm={head_nec['delta_margin']:+.3f} headSR={head_nec['success_rate']:.3f} "
        f"writerΔm={writer_nec['delta_margin']:+.3f} writerSR={writer_nec['success_rate']:.3f} "
        f"stab={(stability if stability is not None else float('nan')):.3f}"
    )

    return metrics


# ----------------------------
# Merged subspace evaluation
# ----------------------------

def evaluate_merged_group(
    *,
    name: str,
    layers: List[int],
    layer_results: Dict[int, DBCMResult],
    handle: HuggingFaceCausalLMHandle,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    device: torch.device,
    cfg: MainConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    ks = [int(layer_results[l].mask.sum().item()) for l in layers]
    k_merge = _choose_merge_k(cfg, ks)

    bases = [layer_results[l].basis for l in layers if layer_results[l].basis is not None]
    Wm = _merge_basis_to_projector(bases, k=k_merge, device=device)

    base_metrics = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

    nec_deltas_acc, nec_deltas_mar, nec_succs = [], [], []
    suf_deltas_don_acc, suf_deltas_don_mar, suf_succs = [], [], []
    rand_nec_acc, rand_nec_mar, rand_nec_succ = [], [], []
    rand_suf_acc, rand_suf_mar, rand_suf_succ = [], [], []

    for layer in layers:
        site = Site(layer=layer, stream=cfg.stream, position=cfg.position, index=None)

        # necessity
        plan = ProjectOutPlan(site=site, W=Wm)
        nec_after = evaluate_project_out_metrics(
            runner=runner, handle=handle, loader=loader_base, device=device, plan=plan, max_batches=cfg.eval_batches
        )
        nec_deltas_acc.append(nec_after["acc"] - base_metrics.acc)
        nec_deltas_mar.append(nec_after["margin"] - base_metrics.margin)

        def _after_po(batch: Dict[str, Any], _plan=plan) -> torch.Tensor:
            out = runner.run(batch, _plan, require_grads=False)
            return out.outputs.logits[:, -1, :].argmax(dim=-1)

        s = evaluate_disruption_success(
            runner=runner, loader=loader_base, device=device, forward_after=_after_po, max_batches=cfg.eval_batches
        )
        nec_succs.append(float(s["success_rate"]))

        # sufficiency
        suf = evaluate_transfer_metrics(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=Wm,
            device=device,
            max_batches=cfg.eval_batches,
        )
        suf_deltas_don_acc.append(float(suf["delta_donor_acc"]))
        suf_deltas_don_mar.append(float(suf["delta_donor_margin"]))

        sucs = evaluate_transfer_success(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=Wm,
            device=device,
            max_batches=cfg.eval_batches,
        )
        suf_succs.append(float(sucs["success_rate"]))

        # random controls (one draw per layer)
        W_rand = _random_projection(handle.d_model, max(k_merge, 1), device=device)

        plan_r = ProjectOutPlan(site=site, W=W_rand)
        m = evaluate_project_out_metrics(
            runner=runner, handle=handle, loader=loader_base, device=device, plan=plan_r, max_batches=cfg.eval_batches
        )
        rand_nec_acc.append(m["acc"] - base_metrics.acc)
        rand_nec_mar.append(m["margin"] - base_metrics.margin)

        def _after_po_r(batch: Dict[str, Any], _plan=plan_r) -> torch.Tensor:
            out = runner.run(batch, _plan, require_grads=False)
            return out.outputs.logits[:, -1, :].argmax(dim=-1)

        rs = evaluate_disruption_success(
            runner=runner, loader=loader_base, device=device, forward_after=_after_po_r, max_batches=cfg.eval_batches
        )
        rand_nec_succ.append(float(rs["success_rate"]))

        suf_r = evaluate_transfer_metrics(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=W_rand,
            device=device,
            max_batches=cfg.eval_batches,
        )
        rand_suf_acc.append(float(suf_r["delta_donor_acc"]))
        rand_suf_mar.append(float(suf_r["delta_donor_margin"]))

        rsuf = evaluate_transfer_success(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=W_rand,
            device=device,
            max_batches=cfg.eval_batches,
        )
        rand_suf_succ.append(float(rsuf["success_rate"]))

    out = {
        "name": name,
        "layers": layers,
        "k_merge": k_merge,
        "baseline_base": asdict(base_metrics),
        "merged_necessity": {
            "mean_delta_acc": float(np.mean(nec_deltas_acc)),
            "mean_delta_margin": float(np.mean(nec_deltas_mar)),
            "mean_success_rate": float(np.mean(nec_succs)),
        },
        "merged_sufficiency": {
            "mean_delta_donor_acc": float(np.mean(suf_deltas_don_acc)),
            "mean_delta_donor_margin": float(np.mean(suf_deltas_don_mar)),
            "mean_success_rate": float(np.mean(suf_succs)),
        },
        "random_controls": {
            "nec_mean_delta_acc": float(np.mean(rand_nec_acc)),
            "nec_mean_delta_margin": float(np.mean(rand_nec_mar)),
            "nec_mean_success_rate": float(np.mean(rand_nec_succ)),
            "suf_mean_delta_donor_acc": float(np.mean(rand_suf_acc)),
            "suf_mean_delta_donor_margin": float(np.mean(rand_suf_mar)),
            "suf_mean_success_rate": float(np.mean(rand_suf_succ)),
        },
    }

    _save_json(out_dir / "metrics" / f"merge_{name}.json", out)
    torch.save({"W_merge": Wm.detach().cpu(), "k_merge": k_merge, "layers": layers}, out_dir / "candidates" / f"merge_{name}.pt")
    return out


# ----------------------------
# Reporting plots
# ----------------------------

def make_report_plots(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    layer_files = sorted(metrics_dir.glob("layer_*.json"))
    rows = []
    for fp in layer_files:
        with open(fp, "r") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: int(r["layer"]))
    if not rows:
        raise FileNotFoundError(f"No layer metrics under {metrics_dir}")

    layers = np.array([int(r["layer"]) for r in rows], dtype=np.int32)

    # k
    k = np.array([int(r.get("k", 0)) for r in rows], dtype=np.float64)
    _plot_line(
        x=layers, y=k,
        title="DBCM effective rank (k) by layer",
        xlabel="Layer", ylabel="k (mask sum)",
        out_path=plots_dir / "k_by_layer.png",
        label="BCDP (DBCM)",
    )

    # Subspace necessity Δmargin + band
    nec_delta_margin = np.array([float(r["subspace_necessity"]["delta_margin"]) for r in rows], dtype=np.float64)
    nec_rand_mean, nec_rand_std = _summarize_random_band(rows, "subspace_necessity_random", "delta_margin")
    _plot_line_with_band(
        x=layers, y=nec_delta_margin,
        band_mean=nec_rand_mean, band_std=nec_rand_std,
        title="Subspace necessity: Δmargin after project-out vs random subspaces",
        xlabel="Layer", ylabel="Δmargin",
        out_path=plots_dir / "subspace_necessity_delta_margin.png",
    )

    # Subspace necessity success rate + band
    nec_sr = np.array([float(r["subspace_necessity"]["success_rate"]) for r in rows], dtype=np.float64)
    nec_sr_rand_mean, nec_sr_rand_std = _summarize_random_band(rows, "subspace_necessity_random", "success_rate")
    _plot_line_with_band(
        x=layers, y=nec_sr,
        band_mean=nec_sr_rand_mean, band_std=nec_sr_rand_std,
        title="Subspace necessity: disruption success rate vs random subspaces",
        xlabel="Layer", ylabel="Success rate (correct→wrong | correct)",
        out_path=plots_dir / "subspace_necessity_success_rate.png",
        y_label_main="BCDP success rate",
    )

    # Subspace sufficiency Δdonor_acc + band
    suf_delta_donor_acc = np.array([float(r["subspace_sufficiency_transfer"]["delta_donor_acc"]) for r in rows], dtype=np.float64)
    suf_rand_mean, suf_rand_std = _summarize_random_band(rows, "subspace_sufficiency_transfer_random", "delta_donor_acc")
    _plot_line_with_band(
        x=layers, y=suf_delta_donor_acc,
        band_mean=suf_rand_mean, band_std=suf_rand_std,
        title="Subspace sufficiency (transfer): Δdonor_acc vs random subspaces",
        xlabel="Layer", ylabel="Δdonor_acc",
        out_path=plots_dir / "subspace_sufficiency_delta_donor_acc.png",
    )

    # Subspace sufficiency success rate + band
    suf_sr = np.array([float(r["subspace_sufficiency_transfer"]["success_rate"]) for r in rows], dtype=np.float64)
    suf_sr_rand_mean, suf_sr_rand_std = _summarize_random_band(rows, "subspace_sufficiency_transfer_random", "success_rate")
    _plot_line_with_band(
        x=layers, y=suf_sr,
        band_mean=suf_sr_rand_mean, band_std=suf_sr_rand_std,
        title="Subspace sufficiency (transfer): success rate vs random subspaces",
        xlabel="Layer", ylabel="Success rate (pred==donor_label)",
        out_path=plots_dir / "subspace_sufficiency_success_rate.png",
        y_label_main="BCDP success rate",
    )

    # Head necessity Δmargin + band
    head_nec_delta_margin = np.array([float(r["head_necessity_topH"]["delta_margin"]) for r in rows], dtype=np.float64)
    head_rand_mean, head_rand_std = _summarize_random_band(rows, "head_necessity_random", "delta_margin")
    _plot_line_with_band(
        x=layers, y=head_nec_delta_margin,
        band_mean=head_rand_mean, band_std=head_rand_std,
        title="Head necessity: Δmargin after ablating top-H heads vs random heads",
        xlabel="Layer", ylabel="Δmargin",
        out_path=plots_dir / "head_necessity_delta_margin.png",
        y_label_main="Ablate top-H (BCDP-ranked)",
    )

    # Head necessity success rate + band
    head_sr = np.array([float(r["head_necessity_topH"]["success_rate"]) for r in rows], dtype=np.float64)
    head_sr_rand_mean, head_sr_rand_std = _summarize_random_band(rows, "head_necessity_random", "success_rate")
    _plot_line_with_band(
        x=layers, y=head_sr,
        band_mean=head_sr_rand_mean, band_std=head_sr_rand_std,
        title="Head necessity: disruption success rate (ablate top-H) vs random heads",
        xlabel="Layer", ylabel="Success rate (correct→wrong | correct)",
        out_path=plots_dir / "head_necessity_success_rate.png",
        y_label_main="Ablate top-H (BCDP-ranked)",
    )

    # Writer necessity Δmargin + band
    writer_nec_delta_margin = np.array([float(r["writer_necessity_topM"]["delta_margin"]) for r in rows], dtype=np.float64)
    writer_rand_mean, writer_rand_std = _summarize_random_band(rows, "writer_necessity_random", "delta_margin")
    _plot_line_with_band(
        x=layers, y=writer_nec_delta_margin,
        band_mean=writer_rand_mean, band_std=writer_rand_std,
        title="MLP writer necessity: Δmargin after ablating top-M writers vs random writers",
        xlabel="Layer", ylabel="Δmargin",
        out_path=plots_dir / "writer_necessity_delta_margin.png",
        y_label_main="Ablate top-M (BCDP-ranked)",
    )

    # Writer necessity success rate + band
    writer_sr = np.array([float(r["writer_necessity_topM"]["success_rate"]) for r in rows], dtype=np.float64)
    writer_sr_rand_mean, writer_sr_rand_std = _summarize_random_band(rows, "writer_necessity_random", "success_rate")
    _plot_line_with_band(
        x=layers, y=writer_sr,
        band_mean=writer_sr_rand_mean, band_std=writer_sr_rand_std,
        title="MLP writer necessity: disruption success rate (ablate top-M) vs random writers",
        xlabel="Layer", ylabel="Success rate (correct→wrong | correct)",
        out_path=plots_dir / "writer_necessity_success_rate.png",
        y_label_main="Ablate top-M (BCDP-ranked)",
    )

    # Stability
    stab = np.array(
        [float(r["stability_split_half"]) if r.get("stability_split_half") is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    _plot_line(
        x=layers, y=stab,
        title="Subspace stability: split-half similarity",
        xlabel="Layer", ylabel="similarity",
        out_path=plots_dir / "stability_split_half.png",
        label="Split-half similarity",
    )

    # Merges summary bars
    merge_files = sorted(metrics_dir.glob("merge_*.json"))
    if merge_files:
        merges = []
        for fp in merge_files:
            with open(fp, "r") as f:
                merges.append(json.load(f))
        merges.sort(key=lambda x: x["name"])

        labels = [m["name"] for m in merges]
        vals_suf = [float(m["merged_sufficiency"]["mean_delta_donor_acc"]) for m in merges]
        vals_nec = [float(m["merged_necessity"]["mean_delta_margin"]) for m in merges]
        vals_suf_sr = [float(m["merged_sufficiency"]["mean_success_rate"]) for m in merges]
        vals_nec_sr = [float(m["merged_necessity"]["mean_success_rate"]) for m in merges]

        _plot_bar(
            labels=labels, values=vals_suf,
            title="Merged subspaces: mean sufficiency (Δdonor_acc)",
            ylabel="mean Δdonor_acc",
            out_path=plots_dir / "merge_sufficiency_mean_delta_donor_acc.png",
            label="Merged (BCDP)",
        )
        _plot_bar(
            labels=labels, values=vals_nec,
            title="Merged subspaces: mean necessity (Δmargin)",
            ylabel="mean Δmargin",
            out_path=plots_dir / "merge_necessity_mean_delta_margin.png",
            label="Merged (BCDP)",
        )
        _plot_bar(
            labels=labels, values=vals_suf_sr,
            title="Merged subspaces: mean sufficiency success rate",
            ylabel="mean success rate",
            out_path=plots_dir / "merge_sufficiency_mean_success_rate.png",
            label="Merged (BCDP)",
        )
        _plot_bar(
            labels=labels, values=vals_nec_sr,
            title="Merged subspaces: mean necessity success rate",
            ylabel="mean success rate",
            out_path=plots_dir / "merge_necessity_mean_success_rate.png",
            label="Merged (BCDP)",
        )

    print(f"[REPORT] wrote plots to: {plots_dir}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="Override output dir, else runs/<model>/<tag_seed>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_examples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--eval_batches", type=int, default=None)
    ap.add_argument("--svd_rows", type=int, default=None)
    ap.add_argument("--dbcm_steps", type=int, default=None)
    ap.add_argument("--dbcm_epochs", type=int, default=None)
    ap.add_argument("--dbcm_max_batches", type=int, default=None)
    ap.add_argument("--lambda_l1", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_random", type=int, default=None)
    ap.add_argument("--topH", type=int, default=None)
    ap.add_argument("--topM", type=int, default=None)
    args = ap.parse_args()

    cfg = MainConfig()
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_examples is not None:
        cfg.num_examples_total = args.num_examples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_batches is not None:
        cfg.eval_batches = args.eval_batches
    if args.svd_rows is not None:
        cfg.svd_rows = args.svd_rows
    if args.dbcm_steps is not None:
        cfg.dbcm_steps = args.dbcm_steps
    if args.dbcm_epochs is not None:
        cfg.dbcm_epochs = args.dbcm_epochs
    if args.dbcm_max_batches is not None:
        cfg.dbcm_max_batches = args.dbcm_max_batches
    if args.lambda_l1 is not None:
        cfg.lambda_l1 = args.lambda_l1
    if args.lr is not None:
        cfg.lr = args.lr
    if args.n_random is not None:
        cfg.n_random = args.n_random
    if args.topH is not None:
        cfg.topH_heads = args.topH
    if args.topM is not None:
        cfg.topM_writers = args.topM

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output dir
    safe_model = cfg.model_name.replace("/", "_")
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(cfg.out_root) / safe_model / f"{cfg.run_tag}_{cfg.seed}"
    _ensure_dir(out_dir)
    _save_json(out_dir / "config.json", asdict(cfg))

    # model
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_torch_dtype(cfg) if torch.cuda.is_available() else torch.float32,
        device_map=cfg.device_map,
    )
    model.eval()
    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # data pools (swap with your real pools if needed)
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )
    spec = SampledDatasetSpec(n=2, num_examples=cfg.num_examples_total, seed=cfg.dataset_seed, topic_id="main")
    ds = SampledExampleDataset(pools=pools, spec=spec)

    prompt_spec = PromptSpec(
        facts_prefix="Facts:",
        row_template="{S} is {A}.",
        row_sep="\n",
        question_template="Question: Who is {A}?\nAnswer:",
        track_spans=True,
    )

    base = identity_variant(n=2, name="base", rw="is")
    swap = swap_first_two(n=2, name="swap", rw="is")

    coll_cfg = CollateConfig(
        target_entity_index=0,
        label_mode="vocab_id",
        max_length=cfg.max_length,
        add_special_tokens=False,
    )

    collate_base = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=base, cfg=coll_cfg)
    collate_swap = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=swap, cfg=coll_cfg)

    loader_base = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_base)
    loader_swap = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_swap)

    runner = InterventionRunner(model=handle, device=device, store_device=device)

    # Run all layers
    layer_metrics: List[Dict[str, Any]] = []
    layer_results: Dict[int, DBCMResult] = {}

    print("loop begins")
    for layer in range(handle.n_layers):
        res_path = out_dir / "metrics" / f"layer_{layer:02d}.json"
        file_path = Path(res_path)

        if file_path.is_file():
            print("layer "+ str(layer) + " exists")
            try:
                # Open the file in read mode ('r')
                with open(res_path, 'r') as file:
                    # Load the JSON data into a Python dictionary
                    m = json.load(file)

            except FileNotFoundError:
                print("Error: The file 'data.json' was not found.")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON. {e}")
        else:
            print("layer "+ str(layer) + " does not exist")
            m = run_one_layer(
                layer=layer,
                handle=handle,
                runner=runner,
                loader_base=loader_base,
                loader_swap=loader_swap,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
            )
        
        layer_metrics.append(m)

        pt_path = out_dir / "candidates" / f"layer_{layer:02d}.pt"
        layer_results[layer] = torch.load(pt_path, map_location="cpu")

    # Merges
    if cfg.do_merges:
        all_layers = list(range(handle.n_layers))
        half = handle.n_layers // 2
        low_layers = list(range(0, half))
        high_layers = list(range(half, handle.n_layers))

        for name, group in [("all", all_layers), ("low", low_layers), ("high", high_layers)]:
            evaluate_merged_group(
                name=name,
                layers=group,
                layer_results=layer_results,
                handle=handle,
                runner=runner,
                loader_base=loader_base,
                loader_swap=loader_swap,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
            )

    # Plots + summary
    make_report_plots(out_dir)

    layers = np.array([int(r["layer"]) for r in layer_metrics], dtype=np.int32)
    suf = np.array([float(r["subspace_sufficiency_transfer"]["delta_donor_acc"]) for r in layer_metrics], dtype=np.float64)
    nec = np.array([float(r["subspace_necessity"]["delta_margin"]) for r in layer_metrics], dtype=np.float64)

    # robust argmax/argmin (avoid crash if all nan)
    best_suf_layer = int(layers[int(np.nanargmax(suf))]) if np.isfinite(suf).any() else int(layers[0])
    best_nec_layer = int(layers[int(np.nanargmin(nec))]) if np.isfinite(nec).any() else int(layers[0])

    summary = {
        "best_layer_by_sufficiency_delta_donor_acc": best_suf_layer,
        "best_layer_by_necessity_delta_margin": best_nec_layer,
        "run_dir": str(out_dir),
    }
    _save_json(out_dir / "summary.json", summary)
    print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()