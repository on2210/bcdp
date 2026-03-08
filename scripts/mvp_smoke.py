# scripts/mvp_smoke.py
from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- bcdp imports ----
from bcdp.data.pools import Pools
from bcdp.data.dataset import SampledExampleDataset, SampledDatasetSpec
from bcdp.data.prompt_spec import PromptSpec
from bcdp.data.variant import identity_variant, swap_first_two
from bcdp.data.collate import make_collate_fn

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle

from bcdp.trace.site import Site, Stream, Position
from bcdp.trace.activation_cache import ActivationCache

from bcdp.intervention.runner import InterventionRunner

from bcdp.subspace.diff_means import DiffMeans
from bcdp.subspace.dbcm import DBCM
from bcdp.utils.linalg import svd_right_vectors


def main():
    # -----------------------------
    # 0) Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MVP] device={device}")

    # -----------------------------
    # 1) Tiny pools (no files)
    # -----------------------------
    # NOTE: adjust these if your prompts assume leading spaces etc.
    pools = Pools(
        names=["Bob", "Ali", "Zack", "Maya", "Noa", "Dana", "Tom", "Lia"],
        entities=["wise", "kind", "rich", "poor", "tall", "short", "happy", "sad"],
    )

    # -----------------------------
    # 2) Dataset: attribute binding-style
    # -----------------------------
    # n=2 => two names + two attributes per example
    spec = SampledDatasetSpec(n=2, num_examples=64, seed=0, topic_id="people")
    ds = SampledExampleDataset(pools=pools, spec=spec)

    # -----------------------------
    # 3) Prompt spec + variants
    # -----------------------------
    prompt_spec = PromptSpec()  # uses your default template
    v_base = identity_variant(n=2, name="base", rw="is")
    v_swap = swap_first_two(n=2, name="swap", rw="is")

    # -----------------------------
    # 4) Model + tokenizer
    # -----------------------------
    model_name = os.environ.get("BCDP_MODEL", "gpt2")
    print(f"[MVP] model={model_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if device.type == "cuda" else None),
    )
    model.eval()

    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok).to(device)
    runner = InterventionRunner(model=handle, device=device, use_amp=(device.type == "cuda"))

    # -----------------------------
    # 5) Dataloaders for base vs counterfactual
    #    IMPORTANT: same dataset, no shuffle, same batch_size => aligned
    # -----------------------------
    bs = 16
    dl_base = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=v_base),
    )
    dl_swap = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer=tok, prompt_spec=prompt_spec, variant=v_swap),
    )

    batch_o = next(iter(dl_base))
    batch_c = next(iter(dl_swap))
    print(f"[MVP] batch_o keys={list(batch_o.keys())}")
    print(f"[MVP] input_ids={tuple(batch_o['input_ids'].shape)} labels={tuple(batch_o['labels'].shape)}")

    # -----------------------------
    # 6) Choose a Site to record (resid_post @ QA, layer 0)
    #    You can move layer around once this runs.
    # -----------------------------
    site = Site(layer=0, stream=Stream.RESID_POST, position=Position.QA, index=None)
    print(f"[MVP] site={site}")

    # -----------------------------
    # 7) Record activations for BOTH conditions (base vs swap)
    # -----------------------------
    with torch.no_grad():
        acts_o = runner._forward_collect_activations(
            batch=batch_o,
            sites=(site,),
            apply_intervention=None,
            require_grads=False,
        )
        acts_c = runner._forward_collect_activations(
            batch=batch_c,
            sites=(site,),
            apply_intervention=None,
            require_grads=False,
        )

    Xo = acts_o[site].detach().to("cpu")  # [B,d]
    Xc = acts_c[site].detach().to("cpu")  # [B,d]

    # Concatenate and build binary labels:
    # 1 = original/base, 0 = counterfactual/swap
    X = torch.cat([Xo, Xc], dim=0)  # [2B,d]
    y = torch.cat(
        [torch.ones(Xo.shape[0]), torch.zeros(Xc.shape[0])],
        dim=0
    ).long()  # [2B]

    print("[MVP] cache label counts:", torch.unique(y, return_counts=True))

    cache = ActivationCache(
        activations={site: X},
        labels=y,
        variant_names=None,
    )


    # -----------------------------
    # 8) Subspace discovery: DiffMeans (rank-1)
    # -----------------------------
    dm = DiffMeans()
    subspace = dm.fit(cache, site=site, tags={"mvp": True})
    print(f"[MVP] DiffMeans basis: {tuple(subspace.basis.shape)}")

    # -----------------------------
    # 9) Prepare V for DBCM (top-r SVD of X)
    # -----------------------------
    r = 8
    V = svd_right_vectors(cache.get(site), center=True, k=r)  # [d,r]
    print(f"[MVP] V (SVD right vectors): {tuple(V.shape)}")

    # -----------------------------
    # 10) Run a short DBCM optimization as an end-to-end patching test
    # -----------------------------
    dbcm = DBCM(
        runner=runner,
        site=site,
        lambda_l1=1e-3,
        lr=5e-2,
        steps=40,  # keep small for smoke
        device=device,
    )

    result = dbcm.fit(V=V, batch_o=batch_o, batch_c=batch_c)
    print(f"[MVP] DBCM mask shape: {tuple(result.mask.shape)}  mask_sum={result.mask.sum().item():.0f}")
    print(f"[MVP] DBCM basis shape: {tuple(result.basis.shape)}  projection shape: {tuple(result.projection.shape)}")

    # -----------------------------
    # 11) Sanity forward (no intervention)
    # -----------------------------
    with torch.no_grad():
        out = handle.forward(
            input_ids=batch_o["input_ids"].to(device),
            attention_mask=batch_o.get("attention_mask", None).to(device) if batch_o.get("attention_mask", None) is not None else None,
        )
    print(f"[MVP] forward logits: {tuple(out.logits.shape)}")

    print("\n[MVP] ✅ Smoke test finished OK")


if __name__ == "__main__":
    main()
