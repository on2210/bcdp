# bcdp/experiments/smoke_test_experiment.py
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

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig


@dataclass
class SmokeCfg:
    model_name: str = "google/gemma-2-2b"
    layer: int = 22
    stream: Stream = Stream.RESID_POST
    position: Position = Position.LAST

    # small & fast
    num_examples_total: int = 256
    batch_size: int = 8
    max_length: int = 128

    svd_rows: int = 128
    dbcm_steps: int = 1
    lambda_l1: float = 1e-3
    lr: float = 1e-2

    eval_batches: int = 8  # 8 * batch_size examples

    seed: int = 0


def _set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval_one_loader(handle, loader, device, max_batches: int) -> Tuple[float, float]:
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
    return acc_sum / max(n_total, 1), mar_sum / max(n_total, 1)


@torch.no_grad()
def _eval_project_out(runner: InterventionRunner, loader, plan, device, max_batches: int) -> Tuple[float, float]:
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
    return acc_sum / max(n_total, 1), mar_sum / max(n_total, 1)


@torch.no_grad()
def _eval_transfer(
    runner: InterventionRunner,
    loader_base: DataLoader,
    loader_swap: DataLoader,
    site: Site,
    W: torch.Tensor,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    n_total = 0
    donor_acc_before = 0.0
    donor_acc_after = 0.0
    rec_acc_before = 0.0
    rec_acc_after = 0.0

    for bi, (b_base, b_swap) in enumerate(zip(loader_base, loader_swap)):
        if bi >= max_batches:
            break

        labels_don = b_base["labels"].to(device)
        labels_rec = b_swap["labels"].to(device)
        bsz = labels_rec.numel()

        # recipient baseline
        logits_rec = forward_logits_last(runner.model, b_swap, device)
        rec_acc_before += accuracy_from_logits(logits_rec, labels_rec) * bsz
        donor_acc_before += accuracy_from_logits(logits_rec, labels_don) * bsz

        # patch: donor = base, recipient = swap
        def donor_builder(_batch: Dict[str, Any]) -> Dict[str, Any]:
            return b_base

        plan = InterchangePlan(donor_builder=donor_builder, site=site, W=W)
        patched = runner.run(b_swap, plan, require_grads=False)
        logits_after = patched.outputs.logits[:, -1, :]

        rec_acc_after += accuracy_from_logits(logits_after, labels_rec) * bsz
        donor_acc_after += accuracy_from_logits(logits_after, labels_don) * bsz

        n_total += bsz

    if n_total == 0:
        return {}

    return {
        "recipient_acc_before": rec_acc_before / n_total,
        "recipient_acc_after": rec_acc_after / n_total,
        "donor_acc_before": donor_acc_before / n_total,
        "donor_acc_after": donor_acc_after / n_total,
        "delta_donor_acc": (donor_acc_after - donor_acc_before) / n_total,
    }


def _check_single_token_labels(tokenizer, loader_base: DataLoader, n_batches: int = 3):
    bad = 0
    checked = 0
    for bi, b in enumerate(loader_base):
        if bi >= n_batches:
            break
        # labels are vocab ids already; to check single-token constraint we need the label string.
        # If your collate stores label_text, use it; otherwise we approximate by decoding the id.
        if "label_text" in b:
            texts = b["label_text"]
        else:
            ids = b["labels"].tolist()
            texts = [tokenizer.decode([i]) for i in ids]

        for t in texts:
            toks = tokenizer.encode(t, add_special_tokens=False)
            checked += 1
            if len(toks) != 1:
                bad += 1

    print(f"[LABELS] single-token check: {checked-bad}/{checked} ok, {bad} not single-token")
    if bad > 0:
        print("        ⚠️  If this is large, consider changing Pools (names/attributes) to enforce 1-token labels.")


def main():
    cfg = SmokeCfg()
    _set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[SMOKE] model={cfg.model_name} layer={cfg.layer} device={device}")

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

    print(f"[MODEL] d_model={handle.d_model} n_layers={handle.n_layers} n_heads={handle.n_heads} d_head={handle.d_head}")

    # ---- data
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )
    spec = SampledDatasetSpec(n=2, num_examples=cfg.num_examples_total, seed=cfg.seed, topic_id="smoke")
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

    _check_single_token_labels(tok, loader_base, n_batches=3)

    runner = InterventionRunner(model=handle, device=device, store_device=device)

    site = Site(layer=cfg.layer, stream=cfg.stream, position=cfg.position, index=None)

    # ---- collect activations for SVD
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
    X = torch.cat(X_list, dim=0)[: cfg.svd_rows]
    V = svd_right_vectors(X, center=True)

    # ---- DBCM
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
    print(f"[DBCM] k={k} | basis={tuple(res.basis.shape)} | W={tuple(W.shape)}")

    # ---- baseline eval
    acc0, mar0 = _eval_one_loader(handle, loader_base, device, cfg.eval_batches)
    print(f"[BASE] acc={acc0:.3f} margin={mar0:.3f}")

    # ---- necessity: project-out
    nec_plan = ProjectOutPlan(site=site, W=W)
    acc1, mar1 = _eval_project_out(runner, loader_base, nec_plan, device, cfg.eval_batches)
    print(f"[NEC]  acc={acc1:.3f} margin={mar1:.3f} | Δacc={acc1-acc0:+.3f} Δm={mar1-mar0:+.3f}")

    # ---- sufficiency transfer
    tr = _eval_transfer(runner, loader_base, loader_swap, site, W, device, cfg.eval_batches)
    if tr:
        print(
            f"[SUF] recipient acc {tr['recipient_acc_before']:.3f}->{tr['recipient_acc_after']:.3f} | "
            f"donor acc {tr['donor_acc_before']:.3f}->{tr['donor_acc_after']:.3f} (Δ {tr['delta_donor_acc']:+.3f})"
        )
    else:
        print("[SUF] skipped (no examples)")

    # ---- head masking sanity (single head)
    can_head_slice = (handle.n_heads is not None and handle.d_head is not None and handle.n_heads * handle.d_head == handle.d_model)
    if can_head_slice:
        # pick layer=cfg.layer head=0
        b = next(iter(loader_base))
        logits_a = forward_logits_last(handle, b, device)
        with mask_attn_o_proj_heads(handle=handle, heads=[HeadRef(layer=cfg.layer, head=0)], mode="ablate"):
            logits_b = forward_logits_last(handle, b, device)
        diff = float((logits_b - logits_a).abs().mean().item())
        print(f"[HEAD SANITY] ablate (layer={cfg.layer}, head=0) | mean|Δlogits|={diff:.6f}")
        if diff == 0.0:
            print("              ⚠️  No change detected — check resolve_attn_o_proj or slicing.")
    else:
        print("[HEAD SANITY] skipped (n_heads*d_head != d_model) — likely GQA mapping needed.")

    # ---- writer masking sanity (one neuron in this layer)
    # We just ablate neuron 0 in the MLP down_proj of this layer.
    b = next(iter(loader_base))
    logits_a = forward_logits_last(handle, b, device)
    with mask_mlp_writers(handle=handle, layer_to_neurons={cfg.layer: [0]}, mode="ablate"):
        logits_b = forward_logits_last(handle, b, device)
    diff = float((logits_b - logits_a).abs().mean().item())
    print(f"[WRITER SANITY] ablate (layer={cfg.layer}, neuron=0) | mean|Δlogits|={diff:.6f}")
    if diff == 0.0:
        print("               ⚠️  No change detected — check resolve_mlp_out_proj.")

    print("[SMOKE] DONE ✅")


if __name__ == "__main__":
    main()