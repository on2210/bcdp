# bcdp/experiments/layer_sweep.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position
from bcdp.utils.linalg import svd_right_vectors

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.intervention.runner import InterventionRunner
from bcdp.subspace.dbcm import DBCM

from bcdp.intervention.project_out_plan import ProjectOutPlan
from bcdp.intervention.interchange_plan import InterchangePlan

from bcdp.ranking.head_ranking_hf import rank_heads_hf, HeadScore
from bcdp.ranking.mlp_writers_hf import rank_mlp_writers_hf, MLPWriterScore
from bcdp.intervention.attn_o_proj_mask import mask_attn_o_proj_heads, HeadRef
from bcdp.intervention.mlp_weight_mask import mask_mlp_writers

from bcdp.experiments.eval_utils import accuracy_from_logits, margin_correct_vs_max_other, _forward_logits_last, evaluate_base

# =========================
# Config
# =========================

@dataclass
class ExperimentConfig:
    model_name: str = "google/gemma-2-2b"

    stream: Stream = Stream.RESID_POST
    position: Position = Position.LAST

    # data
    dataset_seed: int = 123
    num_examples_total: int = 8192
    batch_size: int = 16
    max_length: int = 128

    # subspace training
    svd_rows: int = 512            # rows used for SVD basis V
    dbcm_steps: int = 1
    lambda_l1: float = 1e-3
    lr: float = 1e-2

    # eval
    eval_batches: int = 256        # 256 * batch_size examples

    # random controls
    n_random: int = 10

    # component tests
    topH_heads: int = 20
    topM_writers: int = 50
    writers_cap_per_layer: int = 10

    # stability
    do_split_half: bool = True

    seed: int = 0


# =========================
# IO helpers
# =========================

def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _random_subspace(d_model: int, k: int, device: torch.device) -> torch.Tensor:
    Q = torch.randn(d_model, k, device=device)
    Q, _ = torch.linalg.qr(Q)
    return Q @ Q.T


def _top_writers_to_dict(
    writers: List[MLPWriterScore],
    topk_total: int,
    cap_per_layer: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for w in writers:
        if sum(len(v) for v in out.values()) >= topk_total:
            break
        layer = int(w.layer)
        if layer not in out:
            out[layer] = []
        if len(out[layer]) >= cap_per_layer:
            continue
        out[layer].append(int(w.neuron))
    return out


def _sample_random_heads(
    *,
    n_layers: int,
    n_heads: int,
    k: int,
    rng: random.Random,
) -> List[HeadRef]:
    heads: List[HeadRef] = []
    for _ in range(k):
        layer = rng.randrange(n_layers)
        head = rng.randrange(n_heads)
        heads.append(HeadRef(layer=layer, head=head))
    return heads


def _sample_random_writers_like(
    *,
    handle: HuggingFaceCausalLMHandle,
    like: Dict[int, List[int]],   # layer -> [neurons...], we sample same counts per layer
    rng: random.Random,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for layer, neurons in like.items():
        out_proj = handle.resolve_mlp_out_proj(int(layer))
        d_mlp = out_proj.weight.shape[1]
        cnt = len(neurons)
        out[layer] = [rng.randrange(d_mlp) for _ in range(cnt)]
    return out


# =========================
# Transfer evaluation (zip base+swap)
# =========================

@torch.no_grad()
def evaluate_transfer(
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
    recipient = swap batch
    donor = base batch (paired by zip, assuming shuffle=False and same dataset order)

    returns:
      recipient_acc_before/after, recipient_margin_before/after
      donor_acc_before/after (measured against donor labels)
      donor_margin_before/after (measured against donor labels)
    """
    n_total = 0

    rec_acc_b = 0.0
    rec_acc_a = 0.0
    rec_mar_b = 0.0
    rec_mar_a = 0.0

    don_acc_b = 0.0
    don_acc_a = 0.0
    don_mar_b = 0.0
    don_mar_a = 0.0

    for bi, (b_base, b_swap) in enumerate(zip(loader_base, loader_swap)):
        if bi >= max_batches:
            break

        labels_rec = b_swap["labels"].to(device)
        labels_don = b_base["labels"].to(device)

        # recipient baseline (no patch)
        logits_rec = forward_logits_last(runner.model, b_swap, device)
        bsz = labels_rec.numel()

        rec_acc_b += accuracy_from_logits(logits_rec, labels_rec) * bsz
        rec_mar_b += margin_correct_vs_max_other(logits_rec, labels_rec) * bsz

        don_acc_b += accuracy_from_logits(logits_rec, labels_don) * bsz
        don_mar_b += margin_correct_vs_max_other(logits_rec, labels_don) * bsz

        # patch
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
        raise ValueError("evaluate_transfer: no examples")

    return {
        "n": n_total,
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


# =========================
# One layer experiment
# =========================

def run_one_layer(
    *,
    layer: int,
    handle: HuggingFaceCausalLMHandle,
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    out_dir: Path,
) -> None:
    site = Site(layer=layer, stream=cfg.stream, position=cfg.position, index=None)

    # --------
    # Build V (SVD) from base activations at site
    # --------
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
    V = svd_right_vectors(X, center=True)         # [d,r] float32 CPU

    # --------
    # DBCM fit
    # --------
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

    res = dbcm.fit(V=V, batch_o=batch_o, batch_c=batch_c)
    W = res.projection.to(device)
    k = int(res.mask.sum().item())

    # save candidate
    cand_path = out_dir / "candidates" / f"layer_{layer:02d}.pt"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, cand_path)

    # --------
    # Baseline eval (base)
    # --------
    base_metrics = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

    # --------
    # Subspace necessity (project-out W) + random
    # --------
    nec_plan = ProjectOutPlan(site=site, W=W)
    # use runner to apply plan
    nec_acc_sum = 0.0
    nec_mar_sum = 0.0
    n_total = 0

    for bi, b in enumerate(loader_base):
        if bi >= cfg.eval_batches:
            break
        labels = b["labels"].to(device)
        bsz = labels.numel()

        out = runner.run(b, nec_plan, require_grads=False)
        logits = out.outputs.logits[:, -1, :]

        nec_acc_sum += accuracy_from_logits(logits, labels) * bsz
        nec_mar_sum += margin_correct_vs_max_other(logits, labels) * bsz
        n_total += bsz

    nec_metrics = {
        "n": n_total,
        "acc": nec_acc_sum / n_total,
        "margin": nec_mar_sum / n_total,
        "delta_acc": (nec_acc_sum / n_total) - base_metrics.acc,
        "delta_margin": (nec_mar_sum / n_total) - base_metrics.margin,
    }

    rng = random.Random(cfg.seed + 1000 + layer)

    rand_nec_list: List[Dict[str, Any]] = []
    for i in range(cfg.n_random):
        W_rand = _random_subspace(handle.d_model, k, device=device)
        plan = ProjectOutPlan(site=site, W=W_rand)

        acc_sum = 0.0
        mar_sum = 0.0
        n2 = 0
        for bi, b in enumerate(loader_base):
            if bi >= cfg.eval_batches:
                break
            labels = b["labels"].to(device)
            bsz = labels.numel()
            out = runner.run(b, plan, require_grads=False)
            logits = out.outputs.logits[:, -1, :]
            acc_sum += accuracy_from_logits(logits, labels) * bsz
            mar_sum += margin_correct_vs_max_other(logits, labels) * bsz
            n2 += bsz

        rand_nec_list.append({
            "acc": acc_sum / n2,
            "margin": mar_sum / n2,
            "delta_acc": (acc_sum / n2) - base_metrics.acc,
            "delta_margin": (mar_sum / n2) - base_metrics.margin,
        })

    # --------
    # Subspace sufficiency transfer + random
    # --------
    suf_metrics = evaluate_transfer(
        runner=runner,
        loader_base=loader_base,
        loader_swap=loader_swap,
        site=site,
        W=W,
        device=device,
        max_batches=cfg.eval_batches,
    )

    rand_suf_list: List[Dict[str, Any]] = []
    for i in range(cfg.n_random):
        W_rand = _random_subspace(handle.d_model, k, device=device)
        rand_suf = evaluate_transfer(
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            site=site,
            W=W_rand,
            device=device,
            max_batches=cfg.eval_batches,
        )
        rand_suf_list.append(rand_suf)

    # --------
    # Head ranking + necessity (+ random)
    # --------
    # Head ranking (HF, capture o_proj input)
    head_scores: list[HeadScore] = rank_heads_hf(
        handle=handle,
        dataloader=loader_base,
        W=W,
        position=cfg.position,
        max_batches=cfg.eval_batches,
    )

    top_heads = head_scores[: cfg.topH_heads]
    top_head_refs = [HeadRef(layer=h.layer, head=h.head) for h in top_heads]

    # necessity: ablate top heads
    with mask_attn_o_proj_heads(handle=handle, heads=top_head_refs, mode="ablate"):
        ab_metrics = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

    head_nec = {
        "delta_acc": ab_metrics.acc - base_metrics.acc,
        "delta_margin": ab_metrics.margin - base_metrics.margin,
    }

    # random heads controls (same count)
    rand_head_nec: List[Dict[str, Any]] = []
    for i in range(cfg.n_random):
        rnd = _sample_random_heads(
            n_layers=handle.n_layers,
            n_heads=int(handle.n_heads or 0),
            k=len(top_head_refs),
            rng=rng,
        )
        with mask_attn_o_proj_heads(handle=handle, heads=rnd, mode="ablate"):
            m = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)
        rand_head_nec.append({
            "delta_acc": m.acc - base_metrics.acc,
            "delta_margin": m.margin - base_metrics.margin,
        })

    # --------
    # MLP writer ranking + necessity (+ random)
    # --------
    writer_scores = rank_mlp_writers_hf(handle=handle, W=W, topk_per_layer=cfg.topM_writers, normalize=True)
    layer_to_neurons = _top_writers_to_dict(
        writer_scores,
        topk_total=cfg.topM_writers,
        cap_per_layer=cfg.writers_cap_per_layer,
    )

    with mask_mlp_writers(handle=handle, layer_to_neurons=layer_to_neurons, mode="ablate"):
        w_ab = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)

    writer_nec = {
        "delta_acc": w_ab.acc - base_metrics.acc,
        "delta_margin": w_ab.margin - base_metrics.margin,
        "counts_per_layer": {str(k): len(v) for k, v in layer_to_neurons.items()},
    }

    rand_writer_nec: List[Dict[str, Any]] = []
    for i in range(cfg.n_random):
        rnd_map = _sample_random_writers_like(handle=handle, like=layer_to_neurons, rng=rng)
        with mask_mlp_writers(handle=handle, layer_to_neurons=rnd_map, mode="ablate"):
            m = evaluate_base(handle, loader_base, device, max_batches=cfg.eval_batches)
        rand_writer_nec.append({
            "delta_acc": m.acc - base_metrics.acc,
            "delta_margin": m.margin - base_metrics.margin,
        })

    # --------
    # Stability split-half (projection similarity)
    # --------
    stability = None
    if cfg.do_split_half and X.shape[0] >= 2:
        half = X.shape[0] // 2
        V_A = svd_right_vectors(X[:half], center=True)
        V_B = svd_right_vectors(X[half:], center=True)

        res_A = dbcm.fit(V=V_A, batch_o=batch_o, batch_c=batch_c)
        res_B = dbcm.fit(V=V_B, batch_o=batch_o, batch_c=batch_c)

        WA = res_A.projection.to(device)
        WB = res_B.projection.to(device)
        # simple Frobenius alignment
        stability = float(torch.norm(WA @ WB).item() / (torch.norm(WA).item() + 1e-8))

    # --------
    # Save all
    # --------
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
        f"necΔm={nec_metrics['delta_margin']:+.3f} "
        f"sufΔdonAcc={suf_metrics['delta_donor_acc']:+.3f} "
        f"headΔm={head_nec['delta_margin']:+.3f} "
        f"writerΔm={writer_nec['delta_margin']:+.3f} "
        f"stab={stability if stability is not None else float('nan'):.3f}"
    )


# =========================
# Main entrypoint
# =========================

def main():
    cfg = ExperimentConfig()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path("runs") / "gemma-2-2b" / f"main_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "config.json", asdict(cfg))

    # ---- model
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

    # ---- data (toy binding, but scaled)
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

    # ---- sweep layers
    for layer in range(handle.n_layers):
        run_one_layer(
            layer=layer,
            handle=handle,
            runner=runner,
            loader_base=loader_base,
            loader_swap=loader_swap,
            device=device,
            cfg=cfg,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()