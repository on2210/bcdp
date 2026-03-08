import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position
from bcdp.utils.linalg import svd_right_vectors

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two, swap_attr_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.intervention.runner import InterventionRunner
from bcdp.intervention.interchange_plan import InterchangePlan
from bcdp.subspace.dbcm import DBCM

# NEW: necessity + sufficiency utilities you added
from bcdp.intervention.subspace_project_plan import ProjectOutPlan
from bcdp.data.paired_collate import make_paired_collate_fn, make_donor_builder_from_paired_batch

import torch.nn.functional as F

from bcdp.ranking.head_ranking_hf import rank_heads_hf
from bcdp.ranking.mlp_writers_hf import rank_mlp_writers_hf
from bcdp.intervention.mlp_weight_mask import mask_mlp_writers

def top_writers_to_dict(writers, topk_total: int) -> dict[int, list[int]]:
    d: dict[int, list[int]] = {}
    for w in writers[:topk_total]:
        d.setdefault(w.layer, []).append(w.neuron)
    return d

def debug_batch_predictions(handle, tokenizer, batch, *, pos_key, topk=10, n_show=3):
    device = handle.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    out = handle.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B,T,V]

    # pick positions exactly like your evaluator
    pos_idx = batch["positions"][pos_key].to(device)  # [B]
    ar = torch.arange(input_ids.shape[0], device=device)
    logits_bv = logits[ar, pos_idx, :]  # [B,V]

    labels = batch["labels"].to(device)

    print("\n===== DEBUG BATCH =====")
    for i in range(min(n_show, input_ids.shape[0])):
        print(f"\n--- example {i} ---")
        # show tail of prompt tokens
        text = tokenizer.decode(input_ids[i].tolist())

        lab = labels[i].item()
        print("label id:", lab, "label tok:", repr(tokenizer.decode([lab])))

        # top-k
        probs = F.softmax(logits_bv[i], dim=-1)
        p, ids = torch.topk(probs, k=topk, dim=-1)
        print(f"top-{topk}:")
        for j in range(topk):
            tid = ids[j].item()
            print(f"  {j+1:2d}. {repr(tokenizer.decode([tid])):16s} p={p[j].item():.4f}")

        # show if label matches any of topk
        in_topk = (ids == lab).any().item()
        print("label_in_topk:", bool(in_topk))

    print("\n======================\n")


@torch.no_grad()
def _forward_logits_last(handle: HuggingFaceCausalLMHandle, batch, device):
    """Baseline forward (no intervention). Returns logits [B,V] for LAST token."""
    out = handle.forward(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch.get("attention_mask", None).to(device) if batch.get("attention_mask", None) is not None else None,
        output_hidden_states=False,
    )
    logits = out.logits  # [B,T,V]
    return logits[:, -1, :].detach()


@torch.no_grad()
def _run_plan_logits_last(runner: InterventionRunner, batch, plan, device):
    """Intervention forward via runner. Returns logits [B,V] for LAST token."""
    res = runner.run(batch, plan, require_grads=False)
    logits = res.outputs.logits  # [B,T,V]
    return logits[:, -1, :].detach()


def _accuracy_from_logits(logits_bv: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits_bv.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _margin_correct_vs_max_other(logits_bv: torch.Tensor, labels: torch.Tensor) -> float:
    # margin = logit(correct) - max_{v!=correct} logit(v)
    B, V = logits_bv.shape
    ar = torch.arange(B, device=logits_bv.device)
    correct = logits_bv[ar, labels]

    top2_vals, top2_idx = torch.topk(logits_bv, k=2, dim=-1)
    top1_val, top1_idx = top2_vals[:, 0], top2_idx[:, 0]
    top2_val = top2_vals[:, 1]
    max_other = torch.where(top1_idx == labels, top2_val, top1_val)

    return (correct - max_other).mean().item()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1) Model (NOTE: your original is huge; consider swapping to a tiny model for CI)
    # ----------------------------
    model_name = "EleutherAI/pythia-70m"  # as in your script :contentReference[oaicite:1]{index=1}
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager",   # <-- זה ה-key
    )
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # ----------------------------
    # 2) Tiny dataset (base vs swap)
    # ----------------------------
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["pie", "cheese", "wine", "bread", "apples", "pears", "beef", "eggs"],
    )

    spec = SampledDatasetSpec(n=2, num_examples=256, seed=123, topic_id="toy")
    ds = SampledExampleDataset(pools=pools, spec=spec)

    prompt_spec = PromptSpec(
        facts_prefix="Facts:",
        row_template="{S} loves {A}.",
        row_sep="\n",
        question_template="Question: Who loves {A}?\nAnswer:",
        track_spans=True,
    )

    base = identity_variant(n=2, name="base", rw="loves")
    swap = swap_attr_first_two(n=2, name="swap", rw="loves")

    cfg = CollateConfig(
        target_entity_index=0,
        label_mode="vocab_id",
        max_length=128,
        add_special_tokens=False,
    )

    collate_base = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=base, cfg=cfg)
    collate_swap = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=swap, cfg=cfg)

    loader_base = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_base)
    loader_swap = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_swap)

    # ----------------------------
    # 3) Choose a site
    # ----------------------------
    site = Site(layer=22, stream=Stream.RESID_POST, position=Position.LAST, index=None)
    print(f"Testing site={site}")

    # ----------------------------
    # 4) Runner
    # ----------------------------
    runner = InterventionRunner(
        model=handle,
        device=device,
        store_device=device,
    )

    # ----------------------------
    # 5) Build V from cached X (SVD)
    # ----------------------------
    X_list = []
    max_rows = 256
    for b in loader_base:
        acts = runner._forward_collect_activations(
            batch=b,
            sites=(site,),
            apply_intervention=None,
            require_grads=False,
        )
        X_list.append(acts[site].detach().float().cpu())
        if sum(x.shape[0] for x in X_list) >= max_rows:
            break

    X = torch.cat(X_list, dim=0)  # [N,d]
    V = svd_right_vectors(X, center=True)  # [d, r] float32 CPU

    # ----------------------------
    # 6) One paired batch for DBCM training (orig + corrupt)
    # ----------------------------
    batch_o = next(iter(loader_base))
    batch_c = next(iter(loader_swap))

    # ----------------------------
    # 7) Train DBCM (short)
    # ----------------------------
    dbcm = DBCM(
        runner=runner,
        site=site,
        lambda_l1=1e-3,
        lr=1e-2,
        steps=1,
        device=device,
    )

    print("Training DBCM (toy) ...")
    #res = dbcm.fit(V=V, batch_o=batch_o, batch_c=batch_c)
    res = dbcm.fit_epoch(V=V, loader_o=loader_base, loader_c=loader_swap)

    mask_sum = float(res.mask.sum().item())
    print("\n=== Toy DBCM sanity ===")
    print(f"mask_sum (binary): {mask_sum:.1f}")
    print(f"basis shape: {tuple(res.basis.shape)}")
    print(f"projection shape: {tuple(res.projection.shape)}")

    assert torch.isfinite(res.projection).all(), "projection has NaNs/Infs"
    assert res.projection.ndim == 2 and res.projection.shape[0] == res.projection.shape[1], "projection not square"
    assert (res.mask == 0).logical_or(res.mask == 1).all(), "mask is not binary"
    print("DONE ✅ (toy functional test passed)")

    # ============================================================
    # 8) NECESSITY test: project-out W on BASE and check behavior changes
    # ============================================================
    print("\n=== NECESSITY: project-out subspace ===")
    W = res.projection  # [d,d], torch on (likely) device; plan will move as needed
    plan_nec = ProjectOutPlan(site=site, W=W)

    # evaluate on a couple of base batches
    deltas = []
    for i, b in enumerate(loader_base):
        if i >= 10:
            break
        labels = b["labels"].to(device)

        logits_base = _forward_logits_last(handle, b, device)
        acc_base = _accuracy_from_logits(logits_base, labels)
        mar_base = _margin_correct_vs_max_other(logits_base, labels)

        logits_nec = _run_plan_logits_last(runner, b, plan_nec, device)
        acc_nec = _accuracy_from_logits(logits_nec, labels)
        mar_nec = _margin_correct_vs_max_other(logits_nec, labels)

        d_acc = acc_nec - acc_base
        d_mar = mar_nec - mar_base
        deltas.append((d_acc, d_mar))

        print(f"batch {i}: acc {acc_base:.3f} -> {acc_nec:.3f} (Δ {d_acc:+.3f}), "
              f"margin {mar_base:.3f} -> {mar_nec:.3f} (Δ {d_mar:+.3f})")

        # sanity: intervention should not be a no-op numerically
        assert torch.isfinite(logits_nec).all(), "necessity logits have NaNs/Infs"

    # weak assertion: at least one batch changes margin meaningfully
    max_abs_dmar = max(abs(d[1]) for d in deltas) if deltas else 0.0
    assert max_abs_dmar > 1e-5, "project-out seems to have no effect (check site/W wiring)"

    # ============================================================
    # 9) SUFFICIENCY test: interchange donor=BASE into recipient=SWAP within subspace W
    # ============================================================
    print("\n=== SUFFICIENCY: interchange donor(base) -> recipient(swap) within subspace ===")

    paired_collate = make_paired_collate_fn(
        tokenizer=tok,
        prompt_spec=prompt_spec,
        recipient_variant=swap,  # recipient is the corrupted binding
        donor_variant=base,      # donor is correct binding
        cfg=cfg,
    )
    loader_paired = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=paired_collate)

    donor_builder = make_donor_builder_from_paired_batch()
    plan_suf = InterchangePlan(donor_builder=donor_builder, site=site, W=W)

    suf_deltas = []
    for i, b in enumerate(loader_paired):
        if i >= 10:
            break

        # recipient labels live in b["labels"] (swap-rendered)
        labels = b["labels"].to(device)

        # baseline on recipient (swap) WITHOUT intervention:
        # We can compute this by running handle.forward on recipient input_ids
        logits_rec = _forward_logits_last(handle, b, device)
        acc_rec = _accuracy_from_logits(logits_rec, labels)
        mar_rec = _margin_correct_vs_max_other(logits_rec, labels)

        # with interchange plan (donor inside batch)
        logits_suf = _run_plan_logits_last(runner, b, plan_suf, device)
        acc_suf = _accuracy_from_logits(logits_suf, labels)
        mar_suf = _margin_correct_vs_max_other(logits_suf, labels)

        d_acc = acc_suf - acc_rec
        d_mar = mar_suf - mar_rec
        suf_deltas.append((d_acc, d_mar))

        print(f"batch {i}: acc {acc_rec:.3f} -> {acc_suf:.3f} (Δ {d_acc:+.3f}), "
              f"margin {mar_rec:.3f} -> {mar_suf:.3f} (Δ {d_mar:+.3f})")

        assert torch.isfinite(logits_suf).all(), "sufficiency logits have NaNs/Infs"

    # weak assertion: interchange should change something (not necessarily always improve on this toy setup)
    max_abs_dmar_suf = max(abs(d[1]) for d in suf_deltas) if suf_deltas else 0.0
    assert max_abs_dmar_suf > 1e-5, "interchange seems to have no effect (check donor_builder/site/W)"

    pos_key = (Position.LAST, None)  # או מה שה-eval שלך משתמש
    debug_batch_predictions(handle, tok, batch_o, pos_key=pos_key, topk=10, n_show=3)

    print("\n=== HEAD RANKING ===")
    head_scores = rank_heads_hf(
        handle=handle,
        dataloader=loader_base,
        W=W,
        position=Position.LAST,
        max_batches=3,
    )
    print("Top 10 heads:")
    for hs in head_scores[:10]:
        print(hs)

    assert len(head_scores) == handle.n_layers * handle.n_heads
    assert all(torch.isfinite(torch.tensor([h.score for h in head_scores])))

    print("\n=== MLP WRITER RANKING ===")
    writers = rank_mlp_writers_hf(handle=handle, W=W, topk_per_layer=20, normalize=True)

    layer_to_neurons = top_writers_to_dict(writers, topk_total=20)  # choose 20 global top writers
    print("Top writers dict:", {k: len(v) for k, v in layer_to_neurons.items()})

    assert len(writers) > 0
    assert torch.isfinite(torch.tensor([x.score for x in writers])).all()

    print("\n=== MLP WRITERS NECESSITY: ablate top writers ===")
    deltas = []
    for i, b in enumerate(loader_base):
        if i >= 3:
            break
        labels = b["labels"].to(device)

        logits_base = _forward_logits_last(handle, b, device)
        acc_base = _accuracy_from_logits(logits_base, labels)
        mar_base = _margin_correct_vs_max_other(logits_base, labels)

        with mask_mlp_writers(handle=handle, layer_to_neurons=layer_to_neurons, mode="ablate"):
            logits_ab = _forward_logits_last(handle, b, device)

        acc_ab = _accuracy_from_logits(logits_ab, labels)
        mar_ab = _margin_correct_vs_max_other(logits_ab, labels)

        d_acc = acc_ab - acc_base
        d_mar = mar_ab - mar_base
        deltas.append((d_acc, d_mar))

        print(f"batch {i}: acc {acc_base:.3f} -> {acc_ab:.3f} (Δ {d_acc:+.3f}), "
            f"margin {mar_base:.3f} -> {mar_ab:.3f} (Δ {d_mar:+.3f})")

    # sanity assertion: should have *some* effect
    assert max(abs(dm) for _, dm in deltas) > 1e-5, "MLP writer ablation seems to have no effect"

    print("\nALL DONE ✅ (DBCM + necessity + sufficiency tests passed)")


if __name__ == "__main__":
    main()
