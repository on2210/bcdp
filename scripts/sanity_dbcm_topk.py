# scripts/sanity_dbcm_topk.py

import torch
from torch.utils.data import DataLoader

from bcdp.subspace.dbcm_topk import dbcm_topk_from_activations
from bcdp.intervention.interchange_plan import InterchangePlan
from bcdp.intervention.runner import InterventionRunner

# השתמש באותו setup כמו sanity_interchange שלך
# (model, dataset, collate וכו')

def evaluate_layer(site, loader_base, loader_swap, runner, k):

    batch_o = next(iter(loader_base))
    batch_c = next(iter(loader_swap))

    # Collect activations from original
    acts = runner._forward_collect_activations(
        batch=batch_o,
        sites=(site,),
        apply_intervention=None,
        require_grads=False,
    )
    X = acts[site]  # [B,d]

    subspace = dbcm_topk_from_activations(X, k=k)

    def donor_builder(_):
        return batch_c

    plan = InterchangePlan(
        donor_builder=donor_builder,
        site=site,
        W=subspace.projection,
    )

    res = runner.run(batch_o, plan)
    logits_p = res.outputs.logits[:, -1, :].detach().cpu()

    out_o = runner.model.forward(
        input_ids=batch_o["input_ids"].to(runner.device),
        attention_mask=batch_o.get("attention_mask", None),
    )
    logits_o = out_o.logits[:, -1, :].detach().cpu()

    diff = (logits_p - logits_o).abs().mean().item()

    print(f"Layer {site.layer}, k={k}: mean |Δlogits| = {diff:.6f}")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1) Model
    # ----------------------------
    model_name = "EleutherAI/pythia-70m-deduped"
    print(f"Loading {model_name} on {device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from bcdp.model.hf_handle import HuggingFaceCausalLMHandle

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # ----------------------------
    # 2) Dataset
    # ----------------------------
    from bcdp.data.pools import Pools
    from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
    from bcdp.data.prompt_spec import PromptSpec
    from bcdp.data.variant import identity_variant, swap_first_two
    from bcdp.data.collate import make_collate_fn, CollateConfig

    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )

    spec = SampledDatasetSpec(
        n=2,
        num_examples=64,
        seed=123,
        topic_id="toy",
    )

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

    from torch.utils.data import DataLoader

    loader_base = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_base)
    loader_swap = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_swap)

    # ----------------------------
    # 3) Runner
    # ----------------------------
    from bcdp.intervention.runner import InterventionRunner

    runner = InterventionRunner(
        model=handle,
        device=device,
        store_device=torch.device("cpu"),
    )

    # ----------------------------
    # 4) Choose site
    # ----------------------------
    from bcdp.trace.site import Site, Stream, Position

    site = Site(
        layer=4,
        stream=Stream.RESID_POST,
        position=Position.LAST,
        index=None,
    )

    print(f"Testing site={site}")

    # ----------------------------
    # 5) Run evaluation for several k
    # ----------------------------
    for k in [0, 4, 8, 16, 32, model.cfg.d_model]:
        evaluate_layer(site, loader_base, loader_swap, runner, k)



if __name__ == "__main__":
    main()
