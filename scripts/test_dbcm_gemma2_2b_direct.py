import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Config
# -------------------------
MODEL_NAME = "google/gemma-2-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

LAYER = 10                # choose a mid layer
POS_MODE = "last"         # "last" only in this script (simple & robust)
K_SUBSPACE = 8            # subspace dim (can be 1, 4, 8...)
BATCH = 8
N_BATCHES = 6
SEED = 0

# prompt templates (simple)
def make_prompt(A: str, B: str, a: str, b: str, swap: bool) -> str:
    if not swap:
        facts = f"Facts:\n{A} is {a}.\n{B} is {b}.\n"
        q = f"Question: Who is {a}?\nAnswer:"
    else:
        facts = f"Facts:\n{A} is {b}.\n{B} is {a}.\n"
        q = f"Question: Who is {a}?\nAnswer:"
    return facts + q


# -------------------------
# Utilities: labels derived from context
# -------------------------
@torch.no_grad()
def derive_next_token_label_id(tokenizer, prompt: str, answer_text: str) -> int:
    """
    Robust label: first token added when encoding(prompt + answer_text) vs encoding(prompt).
    Tries a few common separators.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    L = len(prompt_ids)

    candidates = [
        answer_text,
        " " + answer_text,
        "\n" + answer_text,
    ]
    for suffix in candidates:
        full_ids = tokenizer(prompt + suffix, add_special_tokens=False)["input_ids"]
        if len(full_ids) > L:
            return int(full_ids[L])

    raise RuntimeError(f"Could not derive label from context. answer_text={answer_text!r}")


def pick_last_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B,T] with 1s for tokens
    # returns pos indices [B]
    return attention_mask.long().sum(dim=1) - 1


def projection_from_basis(B: torch.Tensor) -> torch.Tensor:
    """
    B: [d,k] (assumed orthonormal-ish). Returns W: [d,d] projection.
    """
    return B @ B.T


def orthonormalize(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # QR orthonormalization
    Q, R = torch.linalg.qr(B)
    return Q


# -------------------------
# Hooking into Gemma blocks
# -------------------------
def get_block(model, layer: int):
    """
    HF Gemma typically has model.model.layers as list of blocks.
    This function handles both common layouts.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer]
    raise AttributeError("Cannot find Gemma layers in model. Inspect model structure.")


@dataclass
class HookContext:
    W: torch.Tensor                       # [d,d]
    donor_h: Optional[torch.Tensor] = None  # [B,d] for selected positions


def make_project_out_hook(ctx: HookContext, pos_idx: torch.Tensor):
    """
    Modify block output h -> (I-W)h at selected positions.
    Hook signature gets (module, inputs, output). output can be tuple or tensor.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        # h: [B,T,d]
        Bsz, T, d = h.shape
        ar = torch.arange(Bsz, device=h.device)
        W = ctx.W.to(device=h.device, dtype=h.dtype)
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        P = (I - W)  # [d,d]

        h_new = h.clone()
        h_sel = h[ar, pos_idx, :]          # [B,d]
        h_new[ar, pos_idx, :] = h_sel @ P.T

        if rest is None:
            return h_new
        return (h_new, *rest)

    return hook


def make_interchange_hook(ctx: HookContext, pos_idx: torch.Tensor):
    """
    Interchange in subspace:
      h' = W*h_donor + (I-W)*h_rec
    Requires ctx.donor_h set to [B,d].
    """
    def hook(module, inputs, output):
        if ctx.donor_h is None:
            raise RuntimeError("donor_h is None: run donor pass capture first.")

        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        Bsz, T, d = h.shape
        ar = torch.arange(Bsz, device=h.device)
        W = ctx.W.to(device=h.device, dtype=h.dtype)
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        Pn = (I - W)

        h_new = h.clone()
        h_rec = h[ar, pos_idx, :]                         # [B,d]
        h_don = ctx.donor_h.to(device=h.device, dtype=h.dtype)  # [B,d]

        h_mix = (h_don @ W.T) + (h_rec @ Pn.T)
        h_new[ar, pos_idx, :] = h_mix

        if rest is None:
            return h_new
        return (h_new, *rest)

    return hook


@torch.no_grad()
def capture_donor_hidden_at_layer(model, input_ids, attention_mask, layer: int, pos_idx: torch.Tensor) -> torch.Tensor:
    """
    Runs forward with a hook that captures block output at selected positions.
    Returns donor_h: [B,d].
    """
    block = get_block(model, layer)
    captured = {}

    def cap_hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        Bsz = h.shape[0]
        ar = torch.arange(Bsz, device=h.device)
        pos_idx_ = pos_idx.to(h.device)
        ar = torch.arange(h.shape[0], device=h.device)
        captured["h"] = h[ar, pos_idx_, :].detach()

        return output  # do not modify

    handle = block.register_forward_hook(cap_hook)
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return captured["h"]
    finally:
        handle.remove()


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def next_token_logits(model, input_ids, attention_mask) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits[:, -1, :]  # [B,V]


def accuracy_from_logits(logits_bv: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits_bv.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def mean_margin(logits_bv: torch.Tensor, labels: torch.Tensor) -> float:
    Bsz, V = logits_bv.shape
    ar = torch.arange(Bsz, device=logits_bv.device)
    correct = logits_bv[ar, labels]

    top2_vals, top2_idx = torch.topk(logits_bv, k=2, dim=-1)
    top1_val, top1_idx = top2_vals[:, 0], top2_idx[:, 0]
    top2_val = top2_vals[:, 1]
    max_other = torch.where(top1_idx == labels, top2_val, top1_val)
    return (correct - max_other).mean().item()


@torch.no_grad()
def print_topk(prompt: str, tokenizer, model, k=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    logits = model(**inputs, use_cache=False).logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    p, ids = torch.topk(probs, k=k, dim=-1)
    print("\nPROMPT:\n", prompt)
    print(f"\nTop-{k} next tokens:")
    for i in range(k):
        tid = ids[0, i].item()
        s = tokenizer.decode([tid])
        print(f"{i+1:2d}. {repr(s):16s} prob={p[0,i].item():.4f}")


# -------------------------
# Main experiment
# -------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    # small toy pools
    names = ["Bob", "Ali", "Sara", "Dana", "Tom", "Maya", "Ron", "Lea"]
    attrs = ["wise", "kind", "funny", "brave", "calm", "fast", "tall", "young"]

    # quick single-prompt sanity (prints top-k)
    p0 = make_prompt("Bob", "Ali", "wise", "kind", swap=False)
    print_topk(p0, tokenizer, model, k=10)

    # build batches
    prompts_base, prompts_swap = [], []
    labels_base, labels_swap = [], []

    for _ in range(BATCH * N_BATCHES):
        A, B = random.sample(names, 2)
        a, b = random.sample(attrs, 2)

        pb = make_prompt(A, B, a, b, swap=False)
        ps = make_prompt(A, B, a, b, swap=True)

        # label = first generated token for correct answer (A) given the prompt
        lb = derive_next_token_label_id(tokenizer, pb, A)
        ls = derive_next_token_label_id(tokenizer, ps, A)

        prompts_base.append(pb)
        prompts_swap.append(ps)
        labels_base.append(lb)
        labels_swap.append(ls)

    # tokenize
    enc_b = tokenizer(prompts_base, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    enc_s = tokenizer(prompts_swap, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

    input_ids_b = enc_b["input_ids"].to(next(model.parameters()).device)
    attn_b = enc_b["attention_mask"].to(next(model.parameters()).device)

    input_ids_s = enc_s["input_ids"].to(next(model.parameters()).device)
    attn_s = enc_s["attention_mask"].to(next(model.parameters()).device)

    labels_b = torch.tensor(labels_base, dtype=torch.long, device=next(model.parameters()).device)
    labels_s = torch.tensor(labels_swap, dtype=torch.long, device=next(model.parameters()).device)

    # positions (last token)
    pos_b = pick_last_positions(attn_b)
    pos_s = pick_last_positions(attn_s)

    # -------------------------
    # 1) Baseline eval (next token at end of prompt)
    # -------------------------
    logits_b = next_token_logits(model, input_ids_b, attn_b)
    logits_s = next_token_logits(model, input_ids_s, attn_s)

    acc_b = accuracy_from_logits(logits_b, labels_b)
    acc_s = accuracy_from_logits(logits_s, labels_s)
    mar_b = mean_margin(logits_b, labels_b)
    mar_s = mean_margin(logits_s, labels_s)

    print("\n=== BASELINE ===")
    print(f"base: acc={acc_b:.3f} margin={mar_b:.3f}")
    print(f"swap: acc={acc_s:.3f} margin={mar_s:.3f}")

    # -------------------------
    # 2) Build a simple subspace from diff-means at a chosen layer (no grads)
    #    We capture hidden states at that layer output (resid_post-like).
    # -------------------------
    donor_h_b = capture_donor_hidden_at_layer(model, input_ids_b, attn_b, layer=LAYER, pos_idx=pos_b)  # [N,d]
    donor_h_s = capture_donor_hidden_at_layer(model, input_ids_s, attn_s, layer=LAYER, pos_idx=pos_s)  # [N,d]

    delta = (donor_h_b.float().mean(dim=0) - donor_h_s.float().mean(dim=0))  # [d]
    delta = delta / (delta.norm() + 1e-8)

    # Make a k-dim basis: first vector is delta, rest random orthonormal (for now)
    d = delta.numel()
    B = torch.randn(d, K_SUBSPACE, device=delta.device, dtype=torch.float32)
    B[:, 0] = delta
    B = orthonormalize(B)
    W = projection_from_basis(B).to(device=next(model.parameters()).device, dtype=DTYPE)

    print("\nSubspace:")
    print(f"layer={LAYER}, k={K_SUBSPACE}, d_model={d}")

    # -------------------------
    # 3) NECESSITY: project-out W at that layer, measure base performance drop/change
    # -------------------------
    ctx = HookContext(W=W)
    block = get_block(model, LAYER)
    hook_handle = block.register_forward_hook(make_project_out_hook(ctx, pos_b))

    try:
        logits_b_nec = next_token_logits(model, input_ids_b, attn_b)
    finally:
        hook_handle.remove()

    acc_b_nec = accuracy_from_logits(logits_b_nec, labels_b)
    mar_b_nec = mean_margin(logits_b_nec, labels_b)

    print("\n=== NECESSITY (project-out) on BASE ===")
    print(f"base: acc {acc_b:.3f} -> {acc_b_nec:.3f} (Δ {acc_b_nec-acc_b:+.3f})")
    print(f"base: mar {mar_b:.3f} -> {mar_b_nec:.3f} (Δ {mar_b_nec-mar_b:+.3f})")

    # -------------------------
    # 4) SUFFICIENCY: donor(base) -> recipient(swap) within W at that layer
    # -------------------------
    # capture donor hidden for base at chosen layer/pos
    donor_h = capture_donor_hidden_at_layer(model, input_ids_b, attn_b, layer=LAYER, pos_idx=pos_b)  # [N,d]
    ctx2 = HookContext(W=W, donor_h=donor_h)

    block = get_block(model, LAYER)
    hook_handle = block.register_forward_hook(make_interchange_hook(ctx2, pos_s))  # apply on swap positions

    try:
        logits_s_suf = next_token_logits(model, input_ids_s, attn_s)
    finally:
        hook_handle.remove()

    acc_s_suf = accuracy_from_logits(logits_s_suf, labels_s)
    mar_s_suf = mean_margin(logits_s_suf, labels_s)

    print("\n=== SUFFICIENCY (interchange base->swap in subspace) on SWAP ===")
    print(f"swap: acc {acc_s:.3f} -> {acc_s_suf:.3f} (Δ {acc_s_suf-acc_s:+.3f})")
    print(f"swap: mar {mar_s:.3f} -> {mar_s_suf:.3f} (Δ {mar_s_suf-mar_s:+.3f})")

    # extra debug: show decoded label token for first example
    print("\nLabel token (example 0):")
    print("base label:", repr(tokenizer.decode([labels_b[0].item()])))
    print("swap label:", repr(tokenizer.decode([labels_s[0].item()])))


if __name__ == "__main__":
    main()
