#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledDatasetSpec, SampledExampleDataset
from bcdp.data.sampler import SamplingRules
from bcdp.data.variant import identity_variant
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.trace.site import Site, Stream, Position
from bcdp.trace.trace_request import TraceRequest
from bcdp.trace.trace import Tracer

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--num-examples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    # ---- Model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(args.model)
    hf_model.to(args.device)
    hf_model.eval()

    model = HuggingFaceCausalLMHandle(hf_model, tokenizer=tok)  # <-- ModelHandle


    # ---- Data
    names_pool = [" Bob", " Ali", " Maya", " Noam", " Dana", " Amir", " Leah", " Yossi"]
    entities_pool = [" wise", " kind", " funny", " brave", " calm", " creative", " honest", " patient"]

    pools = Pools(names=names_pool, entities=entities_pool)
    spec = SampledDatasetSpec(
        n=args.n,
        num_examples=args.num_examples,
        seed=args.seed,
        topic_id=None,
        rules=SamplingRules(),
    )
    ds = SampledExampleDataset(pools, spec)

    prompt_spec = PromptSpec(
        facts_prefix="Facts:",
        row_template="{S} {RW} {A}.",
        row_sep="\n",
        question_template="Question: Who is {A}?\nAnswer:",
        track_spans=True,
    )
    variant = identity_variant(n=args.n, name="base", rw="is")

    collate_cfg = CollateConfig(
        target_entity_index=0,
        label_mode="subject_index",   # not used in tracing, but fine
        add_special_tokens=False,
        return_attention_mask=True,
    )
    collate_fn = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=variant, cfg=collate_cfg)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))
    for k in ["input_ids", "attention_mask", "labels", "positions"]:
        assert k in batch, f"Missing {k} in batch"

    # ---- TraceRequest (small set of sites)
    L = args.layer
    sites = (
        Site(layer=L, stream=Stream.RESID_POST, position=Position.QA, index=None),
        Site(layer=L, stream=Stream.RESID_POST, position=Position.LAST, index=None),
        Site(layer=L, stream=Stream.ATTN_OUT, position=Position.QA, index=None),
        Site(layer=L, stream=Stream.MLP_OUT, position=Position.QA, index=None),
    )
    req = TraceRequest(sites=sites, require_grads=False, store_on_device=False)

    # ---- Trace
    tracer = Tracer(model=model, device=torch.device(args.device))
    cache = tracer.trace([batch], request=req)  # pass an iterable of batches


    sites = cache.sites()

    # ---- Inspect cache
    print("\n=== Tracer sanity results ===")
    print(f"Batch input_ids: {tuple(batch['input_ids'].shape)} on {batch['input_ids'].device}")
    print(f"Cache sites: {len(sites)}")

    print(f"Cache sites: {len(sites)}")
    for s in sites:
        x = cache.get(s)
        print(f"- {s}: {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")

    print("\n[DONE] Tracer sanity passed (cache populated).")


if __name__ == "__main__":
    main()
