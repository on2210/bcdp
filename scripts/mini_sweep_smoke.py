# bcdp/scripts/mini_sweep_smoke.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position
from bcdp.subspace.dbcm import DBCM
from bcdp.utils.linalg import svd_right_vectors

from bcdp.intervention.runner import InterventionRunner
from bcdp.intervention.interchange_plan import InterchangePlan
from bcdp.intervention.project_out_plan import ProjectOutPlan

from bcdp.experiments.eval_utils import (
    forward_logits_last,
    accuracy_from_logits,
    margin_correct_vs_max_other,
)

from bcdp.intervention.attn_o_proj_mask import mask_attn_o_proj_heads, HeadRef
from bcdp.intervention.mlp_weight_mask import mask_mlp_writers

# Use YOUR existing head ranking file
from bcdp.ranking.head_ranking_hf import rank_heads_hf, HeadScore

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig


@dataclass
class MiniCfg:
    model_name: str = "google/gemma-2-2b"
    seed: int = 0

    stream: Stream = Stream.RESID_POST
    position: Position = Position.LAST

    # moderate
    num_examples_total: int = 2048
    batch_size: int = 8
    max_length: int = 128

    svd_rows: int = 256
    dbcm_steps: int = 1
    lambda_l1: float = 1e-3
    lr: float = 1e-2

    eval_batches: int = 64          # for metrics
    head_rank_batches: int = 16     # cheaper head ranking
    topH: int = 20

    writer_sanity_neuron: int = 0


def _set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_loader(handle, loader, device, max_batches: int) -> Tuple[float, float]:
    n_total = 0
    acc_sum = 0.0
    mar_sum = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        labels = b["labels"].to(device)
        logits = forward_logits_last(handle, b, device)
        bsz = labels.numel()
        acc_sum += accuracy_from_logits(logits, labels) * bsz
        mar_sum += margin_correct_vs_max_other(logits, labels) * bsz
        n_total += bsz
    return acc_sum / n_total, mar_sum / n_total


@torch.no_grad()
def eval_project_out(runner: InterventionRunner, loader, plan, device, max_batches: int) -> Tuple[float, float]:
    n_total = 0
    acc_sum = 0.0
    mar_sum = 0.0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        labels = b["labels"].to(device)
        out = runner.run(b, plan, require_grads=False)
        logits = out.outputs.logits[:, -1, :]
        bsz = labels.numel()
        acc_sum += accuracy_from_logits(logits, labels) * bsz
        mar_sum += margin_correct_vs_max_other(logits, labels) * bsz
        n_total += bsz
    return acc_sum / n_total, mar_sum / n_total


@torch.no_grad()
def eval_transfer(
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    site: Site,
    W: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> float:
    """
    Returns Δ donor-acc measured on recipient logits:
      donor_acc_after - donor_acc_before
    """
    n_total = 0
    don_b = 0.0
    don_a = 0.0
    for bi, (b_base, b_swap) in enumerate(zip(loader_base, loader_swap)):
        if bi >= max_batches:
            break
        labels_don = b_base["labels"].to(device)
        bsz = labels_don.numel()

        logits_before = forward_logits_last(runner.model, b_swap, device)
        don_b += accuracy_from_logits(logits_before, labels_don) * bsz

        def donor_builder(_batch: Dict[str, Any]) -> Dict[str, Any]:
            return b_base

        plan = InterchangePlan(donor_builder=donor_builder, site=site, W=W)
        patched = runner.run(b_swap, plan, require_grads=False)
        logits_after = patched.outputs.logits[:, -1, :]

        don_a += accuracy_from_logits(logits_after, labels_don) * bsz
        n_total += bsz

    return (don_a - don_b) / n_total


def collect_svd_V(
    runner: InterventionRunner,
    loader_base: DataLoader,
    site: Site,
    svd_rows: int,
) -> torch.Tensor:
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
        if got >= svd_rows:
            break
    X = torch.cat(X_list, dim=0)[:svd_rows]
    V = svd_right_vectors(X, center=True)
    return V


def run_one_layer(
    *,
    layer: int,
    cfg: MiniCfg,
    handle: HuggingFaceCausalLMHandle,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    device: torch.device,
) -> Dict[str, float]:

    site = Site(layer=layer, stream=cfg.stream, position=cfg.position, index=None)

    # ---- DBCM
    V = collect_svd_V(runner, loader_base, site, cfg.svd_rows)
    batch_o = next(iter(loader_base))
    batch_c = next(iter(loader_swap))

    dbcm = DBCM(
        runner=runner,
        site=site,
        lambda_l1=cfg.lambda_l1,
        lr=cfg.lr,
        steps=cfg.dbcm_steps,
        device=device,
    )
    res = dbcm.fit_epoch(V=V, loader_o=loader_base, loader_c=loader_swap)
    W = res.projection.to(device)
    k = int(res.mask.sum().item())

    # ---- baseline
    acc0, mar0 = eval_loader(handle, loader_base, device, cfg.eval_batches)

    # ---- necessity: project-out
    nec_plan = ProjectOutPlan(site=site, W=W)
    acc1, mar1 = eval_project_out(runner, loader_base, nec_plan, device, cfg.eval_batches)
    nec_dacc = acc1 - acc0
    nec_dm = mar1 - mar0

    # ---- sufficiency transfer (Δ donor-acc)
    suf_d = eval_transfer(runner, loader_base, loader_swap, site, W, device, cfg.eval_batches)

    # ---- head ranking (cheap) + necessity sanity (top-1 ablation, just to test)
    head_scores: List[HeadScore] = rank_heads_hf(
        handle=handle,
        dataloader=loader_base,
        W=W,
        position=cfg.position,
        max_batches=cfg.head_rank_batches,
    )
    top = head_scores[0]
    b = next(iter(loader_base))
    logits_a = forward_logits_last(handle, b, device)
    with mask_attn_o_proj_heads(handle=handle, heads=[HeadRef(layer=top.layer, head=top.head)], mode="ablate"):
        logits_b = forward_logits_last(handle, b, device)
    head_delta_logits = float((logits_b - logits_a).abs().mean().item())

    # ---- writer mask sanity (one neuron)
    logits_a2 = forward_logits_last(handle, b, device)
    with mask_mlp_writers(handle=handle, layer_to_neurons={layer: [cfg.writer_sanity_neuron]}, mode="ablate"):
        logits_b2 = forward_logits_last(handle, b, device)
    writer_delta_logits = float((logits_b2 - logits_a2).abs().mean().item())

    return {
        "layer": float(layer),
        "k": float(k),
        "base_acc": float(acc0),
        "base_margin": float(mar0),
        "nec_delta_acc": float(nec_dacc),
        "nec_delta_margin": float(nec_dm),
        "suf_delta_donor_acc": float(suf_d),
        "head_top1_score": float(top.score),
        "head_top1_delta_logits": float(head_delta_logits),
        "writer_delta_logits": float(writer_delta_logits),
    }


def main():
    cfg = MiniCfg()
    _set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)
    runner = InterventionRunner(model=handle, device=device, store_device=device)

    n_layers = handle.n_layers
    layers = [n_layers - 10, n_layers - 6, n_layers - 2]

    # ---- data
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )
    spec = SampledDatasetSpec(n=2, num_examples=cfg.num_examples_total, seed=cfg.seed, topic_id="mini")
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

    print(f"[MINI] model={cfg.model_name} device={device} layers={layers}")
    print(f"[MINI] svd_rows={cfg.svd_rows} dbcm_steps={cfg.dbcm_steps} eval_batches={cfg.eval_batches} n_random=SKIPPED (mini)")
    print("--------------------------------------------------------------------------")
    print(f"{'layer':>5} | {'k':>3} | {'baseAcc':>7} | {'necΔm':>7} | {'sufΔdonAcc':>10} | {'headΔlog':>8} | {'wrΔlog':>7}")
    print("-" * 90)

    rows: List[Dict[str, float]] = []
    for L in layers:
        r = run_one_layer(
            layer=L,
            cfg=cfg,
            handle=handle,
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            device=device,
        )
        rows.append(r)
        print(
            f"{int(r['layer']):5d} | {int(r['k']):3d} | {r['base_acc']:.3f} | {r['nec_delta_margin']:+.3f} | "
            f"{r['suf_delta_donor_acc']:+.3f} | {r['head_top1_delta_logits']:.3f} | {r['writer_delta_logits']:.3f}"
        )

    print("-" * 90)
    print("[MINI] DONE ✅")


if __name__ == "__main__":
    main()