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


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 1) Tiny model (toy)
    # ----------------------------

    model_name = "google/gemma-2-9b"  # אם זה השם אצלך ב-HF
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",   # או "cuda" אם יש GPU אחד גדול
    )
    model.eval()


    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)

    # ----------------------------
    # 2) Tiny dataset (base vs swap)
    # ----------------------------
    pools = Pools(
        names=["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"],
        entities=["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"],
    )

    spec = SampledDatasetSpec(n=2, num_examples=512, seed=123, topic_id="toy")
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

    collate_base = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=base, cfg=cfg)
    collate_swap = make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=swap, cfg=cfg)

    loader_base = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_base)
    loader_swap = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_swap)

    # ----------------------------
    # 3) Choose a site
    # ----------------------------
    site = Site(layer=9, stream=Stream.RESID_POST, position=Position.LAST, index=None)
    print(f"Testing site={site}")

    # ----------------------------
    # 4) Runner (IMPORTANT: store_device=device for training)
    # ----------------------------
    runner = InterventionRunner(
        model=handle,
        device=device,
        store_device=device,  # <-- keep tensors on device (training-friendly)
    )

    # ----------------------------
    # 5) Build V from a cached X (paper-style: SVD on many samples)
    # ----------------------------
    # collect X from a few batches
    X_list = []
    max_rows = 500  
    for b in loader_base:
        acts = runner._forward_collect_activations(
            batch=b,
            sites=(site,),
            apply_intervention=None,
            require_grads=False,
        )
        X_list.append(acts[site].detach().float().cpu())  # CPU float32 for SVD
        if sum(x.shape[0] for x in X_list) >= max_rows:
            break

    X = torch.cat(X_list, dim=0)  # [N,d]
    V = svd_right_vectors(X, center=True)  # [d, r] (float32 CPU)
    # Use only first r=min(N,d) already; for tiny model this is fine.

    # ----------------------------
    # 6) Prepare one paired batch (orig + donor)
    # ----------------------------
    batch_o = next(iter(loader_base))
    batch_c = next(iter(loader_swap))

    # ----------------------------
    # 7) Run paper-faithful DBCM training (short)
    # ----------------------------
    dbcm = DBCM(
        runner=runner,
        site=site,
        lambda_l1=1e-2,
        lr=1e-2,
        steps=50,   
        device=device,
    )

    print("Training DBCM (toy) ...")
    res = dbcm.fit(V=V, batch_o=batch_o, batch_c=batch_c)

    # ----------------------------
    # 8) Assertions / sanity prints
    # ----------------------------
    mask_sum = float(res.mask.sum().item())
    print("\n=== Toy DBCM sanity ===")
    print(f"mask_sum (binary): {mask_sum:.1f}")
    print(f"basis shape: {tuple(res.basis.shape)}")
    print(f"projection shape: {tuple(res.projection.shape)}")

    # Core functional checks:
    assert torch.isfinite(res.projection).all(), "projection has NaNs/Infs"
    assert res.projection.ndim == 2 and res.projection.shape[0] == res.projection.shape[1], "projection not square"
    assert (res.mask == 0).logical_or(res.mask == 1).all(), "mask is not binary"
    print("DONE ✅ (toy functional test passed)")


if __name__ == "__main__":
    main()
