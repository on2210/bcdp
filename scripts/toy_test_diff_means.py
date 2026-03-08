# run_toy_pythia70m.py
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle
from bcdp.trace.site import Site, Stream, Position
from bcdp.trace.trace_request import TraceRequest
from bcdp.trace.trace import Tracer

from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn, CollateConfig

from bcdp.subspace.diff_means import DiffMeans


def main():
    # -----------------------
    # 0) Setup
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model_name = "EleutherAI/pythia-70m-deduped"  # or "EleutherAI/pythia-70m"
    print(f"Loading model: {model_name} on {device}")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("is_fast:", getattr(tok, "is_fast", False))
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Need a *fast* tokenizer for return_offsets_mapping=True. Got a slow tokenizer.")

    if tok.pad_token_id is None:
        # Pythia tokenizers sometimes have no pad token configured
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # -----------------------
    # 1) Dataset (synthetic pools)
    # -----------------------
    # Keep pools small but sufficient; ensure N <= pool size.
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

    # -----------------------
    # 2) Prompt + variants
    # -----------------------
    prompt_spec = PromptSpec(
        facts_prefix="Facts:",
        row_template="{S} is {A}.",
        row_sep="\n",
        question_template="Question: Who is {A}?\nAnswer:",
        track_spans=True,
    )

    base = identity_variant(n=2, name="base", rw="is")
    swap = swap_first_two(n=2, name="swap", rw="is")

    # We'll trace ONLY one variant at a time here.
    # Label convention for DiffMeans needs binary labels.
    # Simple trick: run two dataloaders (base and swap) and concatenate them,
    # with labels 0 for base and 1 for swap.

    def make_loader(variant, label_value: int):
        collate = make_collate_fn(
            tokenizer=tok,
            prompt_spec=prompt_spec,
            variant=variant,
            cfg=CollateConfig(
                target_entity_index=0,
                label_mode="vocab_id",  # doesn't matter here; we override labels below
                max_length=128,
                add_special_tokens=False,
            ),
        )

        def collate_with_fixed_label(examples):
            batch = collate(examples)
            B = batch["input_ids"].shape[0]
            batch["labels"] = torch.full((B,), label_value, dtype=torch.long)
            return batch

        return DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_with_fixed_label)

    loader_base = make_loader(base, 0)
    loader_swap = make_loader(swap, 1)

    # Chain loaders
    def chain_loaders(*loaders):
        for ld in loaders:
            for batch in ld:
                yield batch

    loader = chain_loaders(loader_base, loader_swap)

    # -----------------------
    # 3) Trace request
    # -----------------------
    # Choose a single anchor site for DiffMeans MVP:
    # resid_post at QA position, layer 0 (you can sweep later).
    anchor_site = Site(layer=0, stream=Stream.RESID_POST, position=Position.QA, index=None)

    request = TraceRequest(
        sites=[anchor_site],
        store_on_device=False,   # keep cache on CPU
        require_grads=False,
    )

    tracer = Tracer(model=handle, device=device, store_device=torch.device("cpu"), detach=True)

    # -----------------------
    # 4) Run trace
    # -----------------------
    print("Tracing...")
    cache = tracer.trace(loader, request)
    X = cache.get(anchor_site)
    y = cache.labels
    print(f"Cache size: N={len(cache)}")
    print(f"X shape @ {anchor_site}: {tuple(X.shape)}  dtype={X.dtype}  device={X.device}")
    print(f"labels: {y.unique(sorted=True).tolist()}  counts: "
          f"{int((y==0).sum())} / {int((y!=0).sum())}")

    # -----------------------
    # 5) DiffMeans → Subspace
    # -----------------------
    dm = DiffMeans()
    sub = dm.fit(cache, site=anchor_site, tags={"model": model_name, "note": "toy run"})
    v = sub.basis[:, 0]

    # quick diagnostics
    proj = sub.coords(X).squeeze(-1)  # [N]
    mu0 = proj[y == 0].mean().item()
    mu1 = proj[y != 0].mean().item()
    std0 = proj[y == 0].std().item()
    std1 = proj[y != 0].std().item()

    print("\nSubspace learned:")
    print(f"  method: {sub.method}")
    print(f"  basis shape: {tuple(sub.basis.shape)}")
    print(f"  v_norm (should be 1): {float(v.norm().item()):.6f}")
    print("Projection stats (coords along v):")
    print(f"  class0 mean/std: {mu0:+.4f} / {std0:.4f}")
    print(f"  class1 mean/std: {mu1:+.4f} / {std1:.4f}")
    print(f"  separation (mean1-mean0): {(mu1-mu0):+.4f}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
