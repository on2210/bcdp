#!/usr/bin/env python3
"""
Sanity checks for the Data + Prompt layer.

Tests:
  (1) Reproducibility: Sampled dataset generated twice with same seed is identical.
  (2) Collate contract: batch has input_ids/labels/positions with valid token indices
      and includes keys required by a typical TraceRequest (SUBJECT/ENTITY indexed, QA/LAST unindexed).

This script does NOT require running the model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader

# ---- Your project imports (adjust only if you changed paths) ----
from bcdp.data.example import Example
from bcdp.data.pools import Pools, load_lines
from bcdp.data.sampler import ExampleSampler, SamplingRules
from bcdp.data.dataset import ExampleDataset, SampledDatasetSpec, SampledExampleDataset

from bcdp.data.variant import VariantSpec, identity_variant
from bcdp.data.prompt_spec import PromptSpec

from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.trace.site import Position, Stream, Site
from bcdp.trace.trace_request import TraceRequest


PosKey = Tuple[Position, Optional[int]]


def _assert_examples_equal(a: Example, b: Example) -> None:
    assert a.topic_id == b.topic_id
    assert a.names == b.names
    assert a.entities == b.entities


def test_reproducibility(
    *,
    names_pool: list[str],
    entities_pool: list[str],
    n: int,
    num_examples: int,
    seed: int,
) -> None:
    pools = Pools(names=names_pool, entities=entities_pool)
    spec = SampledDatasetSpec(n=n, num_examples=num_examples, seed=seed, topic_id=None, rules=SamplingRules())

    ds1 = SampledExampleDataset(pools, spec)
    ds2 = SampledExampleDataset(pools, spec)

    assert len(ds1) == len(ds2) == num_examples
    for i in range(num_examples):
        _assert_examples_equal(ds1[i], ds2[i])

    print(f"[OK] Reproducibility: generated {num_examples} examples deterministically (seed={seed}, N={n}).")


def _validate_positions(batch: Dict[str, Any]) -> None:
    assert "positions" in batch and isinstance(batch["positions"], dict)
    positions: Dict[PosKey, torch.Tensor] = batch["positions"]

    input_ids = batch["input_ids"]
    B, T = input_ids.shape

    for k, v in positions.items():
        assert isinstance(k, tuple) and len(k) == 2, f"Bad position key: {k}"
        pos, idx = k
        assert isinstance(pos, Position)
        assert (idx is None) or isinstance(idx, int)
        assert torch.is_tensor(v) and v.shape == (B,), f"positions[{k}] must be [B], got {getattr(v, 'shape', None)}"

        # index validity
        assert torch.all(v >= 0).item(), f"positions[{k}] has negative indices"
        assert torch.all(v < T).item(), f"positions[{k}] has indices >= seq_len"

    print(f"[OK] Positions: all keys map to valid token indices (B={B}, T={T}).")


def _build_typical_trace_request(n: int, layer: int = 0) -> TraceRequest:
    """
    A typical request we'd use for early binding experiments:
      - resid_post at QA and LAST
      - resid_post at SUBJECT(i) and ENTITY(i) for i in [0..N-1]
    """
    sites = []
    for i in range(n):
        sites.append(Site(layer=layer, stream=Stream.RESID_POST, position=Position.SUBJECT, index=i))
        sites.append(Site(layer=layer, stream=Stream.RESID_POST, position=Position.ENTITY, index=i))

    sites.append(Site(layer=layer, stream=Stream.RESID_POST, position=Position.QA, index=None))
    sites.append(Site(layer=layer, stream=Stream.RESID_POST, position=Position.LAST, index=None))

    return TraceRequest(sites=tuple(sites), require_grads=False, store_on_device=False)


def test_collate_contract(
    *,
    tokenizer_name: str,
    names_pool: list[str],
    entities_pool: list[str],
    n: int,
    num_examples: int,
    seed: int,
    batch_size: int,
) -> None:
    # Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        # common for decoder-only models
        tok.pad_token = tok.eos_token

    pools = Pools(names=names_pool, entities=entities_pool)
    spec = SampledDatasetSpec(n=n, num_examples=num_examples, seed=seed, topic_id=None, rules=SamplingRules())
    ds = SampledExampleDataset(pools, spec)

    # PromptSpec + Variant
    prompt_spec = PromptSpec(
        facts_prefix="Facts:",
        row_template="{S} {RW} {A}.",
        row_sep="\n",
        question_template="Question: Who is {A}?\nAnswer:",
        track_spans=True,
    )
    variant = identity_variant(n=n, name="base", rw="is")

    collate_cfg = CollateConfig(
        target_entity_index=0,           # dataset semantics; OK for MVP
        label_mode="subject_index",      # simplest to validate here
        max_length=None,
        pad_to_multiple_of=None,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    collate_fn = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=variant, cfg=collate_cfg)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    assert "input_ids" in batch and torch.is_tensor(batch["input_ids"])
    assert "labels" in batch and torch.is_tensor(batch["labels"])
    assert batch["labels"].shape == (batch_size,)

    _validate_positions(batch)

    # Ensure batch positions include keys required by a typical TraceRequest
    req = _build_typical_trace_request(n=n, layer=0)
    required_keys = {(s.position, s.index) for s in req.sites}
    present_keys = set(batch["positions"].keys())

    missing = required_keys - present_keys
    assert not missing, f"Batch missing required position keys: {missing}"

    print("[OK] Collate contract: batch has required keys for TraceRequest + valid indices.")

    # Optional: show one rendered prompt for eyeballing
    ex0 = ds[0]
    inst0 = prompt_spec.render(ex0, variant, target_entity_index=collate_cfg.target_entity_index)
    print("\n--- Example prompt (first instance) ---")
    print(inst0.text)
    print("--------------------------------------\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="gpt2", help="Tokenizer name/path (HF). Default: gpt2")
    ap.add_argument("--names", default=None, help="Path to names pool .txt (one per line)")
    ap.add_argument("--entities", default=None, help="Path to entities pool .txt (one per line)")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # If no pool files provided, use tiny built-in pools (still deterministic)
    if args.names is None:
        names_pool = [" Bob", " Ali", " Maya", " Noam", " Dana", " Amir", " Leah", " Yossi"]
    else:
        names_pool = load_lines(args.names)

    if args.entities is None:
        entities_pool = [" wise", " kind", " funny", " brave", " calm", " creative", " honest", " patient"]
    else:
        entities_pool = load_lines(args.entities)

    test_reproducibility(
        names_pool=names_pool,
        entities_pool=entities_pool,
        n=args.n,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    test_collate_contract(
        tokenizer_name=args.tokenizer,
        names_pool=names_pool,
        entities_pool=entities_pool,
        n=args.n,
        num_examples=args.num_examples,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("[DONE] Data + Prompt layer looks ready.")


if __name__ == "__main__":
    main()
