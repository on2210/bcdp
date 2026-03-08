# run_eval.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# ---- data ----
from bcdp.data.pools import Pools, load_lines
from bcdp.data.dataset import SampledDatasetSpec, SampledExampleDataset
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two, VariantSpec
from bcdp.data.collate import make_collate_fn, CollateConfig

# ---- model / runner / eval ----
from bcdp.eval.evaluator import Evaluator, EvalConfig
from bcdp.intervention.runner import InterventionRunner  # only if you want interventions

# pick ONE backend handle (hf / tl) depending on what you use in your project.
# These imports assume your package layout matches the uploaded files.
from bcdp.model.hf_handle import HFModelHandle
# from bcdp.model.tl_handle import TLModelHandle


def build_variant(kind: str, n: int, rw: str, seed: int) -> VariantSpec:
    if kind == "base":
        return identity_variant(n, name="base", rw=rw)
    if kind == "swap":
        return swap_first_two(n, name="swap", rw=rw)
    raise ValueError(f"Unknown variant kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name, e.g. gpt2 or EleutherAI/pythia-70m")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--names", type=str, required=True, help="Path to names pool txt")
    ap.add_argument("--entities", type=str, required=True, help="Path to entities pool txt")

    ap.add_argument("--n", type=int, default=2, help="N (num bindings per example)")
    ap.add_argument("--num_examples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topic_id", type=str, default=None)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--variant", type=str, default="base", choices=["base", "swap"])
    ap.add_argument("--rw", type=str, default="is")

    ap.add_argument("--target_entity_index", type=int, default=0)
    ap.add_argument("--eval_pos", type=str, default="LAST", choices=["LAST", "QA"])  # minimal choices
    ap.add_argument("--max_length", type=int, default=None)

    # If you want to evaluate under an intervention later, keep this flag.
    ap.add_argument("--with_runner", action="store_true", help="Create InterventionRunner (needed for intervention eval)")

    args = ap.parse_args()

    device = torch.device(args.device)

    # -------------------
    # 1) Pools + Dataset
    # -------------------
    pools = Pools(
        names=load_lines(args.names),
        entities=load_lines(args.entities),
    )

    ds_spec = SampledDatasetSpec(
        n=args.n,
        num_examples=args.num_examples,
        seed=args.seed,
        topic_id=args.topic_id,
    )
    dataset = SampledExampleDataset(pools=pools, spec=ds_spec)

    # -------------------
    # 2) Prompt + Variant
    # -------------------
    prompt_spec = PromptSpec(track_spans=True)
    variant = build_variant(args.variant, n=args.n, rw=args.rw, seed=args.seed)

    # -------------------
    # 3) Model + Tokenizer
    # -------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Important for some models (e.g., GPT-2) that have no pad token:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HFModelHandle.from_pretrained(
        args.model,
        device=device,
    )

    # -------------------
    # 4) Collate + Loader
    # -------------------
    collate_cfg = CollateConfig(
        target_entity_index=args.target_entity_index,
        label_mode="vocab_id",
        max_length=args.max_length,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        variant=variant,
        cfg=collate_cfg,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # -------------------
    # 5) Evaluator (+ optional Runner)
    # -------------------
    if args.eval_pos == "LAST":
        eval_pos = ("LAST", None)
    else:
        eval_pos = ("QA", None)

    # Your Evaluator expects PosKey = (Position, Optional[int]).
    # Here we pass the real enum objects to match evaluator.py.
    from bcdp.trace.site import Position
    pos_key = (Position.LAST, None) if args.eval_pos == "LAST" else (Position.QA, None)

    runner: Optional[InterventionRunner] = None
    if args.with_runner:
        runner = InterventionRunner(model=model)

    evaluator = Evaluator(
        model=model,
        device=device,
        cfg=EvalConfig(eval_pos=pos_key, return_per_example=False),
        runner=runner,
    )

    # -------------------
    # 6) Run eval (no intervention)
    # -------------------
    res = evaluator.evaluate(loader, intervention_plan=None, require_grads=False)

    print("=== EVAL RESULT ===")
    print(f"n = {res.n}")
    print(f"mean_accuracy            = {res.mean_accuracy:.4f}")
    print(f"mean_correct_logit       = {res.mean_correct_logit:.4f}")
    print(f"mean_correct_logprob     = {res.mean_correct_logprob:.4f}")
    print(f"mean_margin_vs_max_other = {res.mean_margin_vs_max_other:.4f}")
    print()
    print("=== DATASET SPEC ===")
    print(asdict(ds_spec))
    print("=== VARIANT ===")
    print(variant)


if __name__ == "__main__":
    main()
