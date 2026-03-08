# scripts/sanity_interchange.py
#
# End-to-end sanity for the 2-pass interchange pipeline:
#   donor (counterfactual) forward -> cache donor acts
#   original forward w/ interchange hook -> patched acts + logits
#
# This script checks two things:
#  1) The activation at the target Site actually changes after patching
#  2) Logits change (not guaranteed huge, but should generally differ)
#
# Run:
#   python scripts/sanity_interchange.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position

from bcdp.utils.linalg import orthonormalize, projection_from_basis

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.intervention.runner import InterventionRunner
from bcdp.intervention.interchange_plan import InterchangePlan
from bcdp.intervention.interchange import InterchangeSubspaceIntervention


# ----------------------------
# A tiny wrapper intervention that *records* the patched activation
# ----------------------------
class RecordingWrapperIntervention:
    def __init__(self, inner: InterchangeSubspaceIntervention):
        self.inner = inner
        self.record = {}  # Site -> Tensor[B,d] (patched, at positions)

    def sites(self):
        return self.inner.sites()

    def hook_for(self, site: Site):
        inner_hook = self.inner.hook_for(site)

        def hook(act, ctx):
            out = inner_hook(act, ctx)
            if out is None:
                return None

            # record [B,d] at the site.position
            pos_map = ctx["positions"]
            key = (site.position, site.index)
            pos_idx = pos_map[key].long()

            if out.ndim == 2:
                h = out
            elif out.ndim == 3:
                bsz = out.shape[0]
                ar = torch.arange(bsz, device=out.device)
                h = out[ar, pos_idx, :]
            else:
                raise ValueError(f"Unexpected activation shape in record: {tuple(out.shape)}")

            self.record[site] = h.detach().cpu()
            return out

        return hook


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 0) Load tiny model
    # ----------------------------
    model_name = "EleutherAI/pythia-70m-deduped"
    print(f"Loading {model_name} on {device} ...")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # ----------------------------
    # 1) Build a tiny dataset + two variants (base vs swap)
    # IMPORTANT: pools without leading spaces (you already discovered why)
    # ----------------------------
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )
    spec = SampledDatasetSpec(n=2, num_examples=32, seed=123, topic_id="toy")
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

    cfg = CollateConfig(
        target_entity_index=0,
        label_mode="vocab_id",
        max_length=128,
        add_special_tokens=False,
    )

    collate_base = make_collate_fn(
        tokenizer=tok,
        prompt_spec=prompt_spec,
        variant=base,
        cfg=cfg,
    )

    collate_swap = make_collate_fn(
        tokenizer=tok,
        prompt_spec=prompt_spec,
        variant=swap,
        cfg=cfg,
    )


    loader_base = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_base)
    loader_swap = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_swap)

    # take first paired batch (same dataset order => aligned)
    batch_o = next(iter(loader_base))
    batch_c = next(iter(loader_swap))

    # make sure shapes align
    assert batch_o["input_ids"].shape == batch_c["input_ids"].shape
    assert set(batch_o["positions"].keys()) == set(batch_c["positions"].keys())

    # ----------------------------
    # 2) Pick a site to patch
    # ----------------------------
    site = Site(layer=0, stream=Stream.RESID_POST, position=Position.QA, index=None)

    # ----------------------------
    # 3) Build a random small subspace + projection W
    # We'll infer d_model by grabbing resid from a donor collection pass.
    # ----------------------------
    runner = InterventionRunner(model=handle, device=device, store_device=torch.device("cpu"))

    donor_acts = runner._forward_collect_activations(  # ok for a sanity script
        batch=batch_c,
        sites=(site,),
        apply_intervention=None,
        require_grads=False,
    )
    Xc = donor_acts[site]          # [B,d] on CPU
    d_model = Xc.shape[1]
    k = 3

    B = orthonormalize(torch.randn(d_model, k))
    W = projection_from_basis(B)   # [d,d]

    print(f"Using site={site}")
    print(f"d_model={d_model}, k={k}")

    # ----------------------------
    # 4) Create an InterchangePlan
    # donor_builder returns *this* donor batch (paired)
    # ----------------------------
    def donor_builder(_batch):
        return batch_c

    # Build plan, but we'll wrap intervention to record patched activations.
    class RecordingPlan(InterchangePlan):
        def build_intervention(self, donor_acts, batch):
            inner = super().build_intervention(donor_acts, batch)
            self._wrapped = RecordingWrapperIntervention(inner)  # store for access after run
            return self._wrapped

    plan = RecordingPlan(donor_builder=donor_builder, site=site, W=W)

    # ----------------------------
    # 5) Run end-to-end patching
    # ----------------------------
    res = runner.run(batch_o, plan, require_grads=False)
    outputs = res.outputs

    # Collect original (unpatched) activations for comparison
    orig_acts = runner._forward_collect_activations(
        batch=batch_o,
        sites=(site,),
        apply_intervention=None,
        require_grads=False,
    )
    Xo = orig_acts[site]  # [B,d] CPU

    # Patched activations recorded by wrapper
    Xp = plan._wrapped.record[site]  # [B,d] CPU

    # ----------------------------
    # 6) Diagnostics
    # ----------------------------
    delta = (Xp - Xo)
    delta_norm = delta.norm(dim=1).mean().item()
    xo_norm = Xo.norm(dim=1).mean().item()
    xp_norm = Xp.norm(dim=1).mean().item()

    # How much of the change lies in the projected subspace?
    # (delta projected energy / delta energy)
    # Note: W is [d,d] on CPU; delta is [B,d]
    W_cpu = W.detach().cpu().float()
    delta_cpu = delta.detach().cpu().float()
    delta_proj = delta_cpu @ W_cpu.T
    frac = (delta_proj.norm(dim=1) / (delta.norm(dim=1) + 1e-9)).mean().item()

    print("\n=== Activation sanity ===")
    print(f"mean ||Xo||: {xo_norm:.4f}")
    print(f"mean ||Xp||: {xp_norm:.4f}")
    print(f"mean ||Xp - Xo||: {delta_norm:.6f}")
    print(f"mean frac of delta in subspace: {frac:.4f}  (should be >0, often sizable)")

    # Logits sanity
    logits = outputs.logits.detach().cpu()  # [B,T,V]
    # compare to unpatched logits
    out_o = handle.forward(
        input_ids=batch_o["input_ids"].to(device),
        attention_mask=batch_o.get("attention_mask", None).to(device) if batch_o.get("attention_mask", None) is not None else None,
        output_hidden_states=False,
    )
    logits_o = out_o.logits.detach().cpu()

    # compare only last token logits to keep things small
    lo = logits_o[:, -1, :]
    lp = logits[:, -1, :]
    logit_diff = (lp - lo).abs().mean().item()

    print("\n=== Logits sanity (last token) ===")
    print(f"mean |logits_patched - logits_orig|: {logit_diff:.6f}")
    print("\nDONE ✅")


if __name__ == "__main__":
    main()
