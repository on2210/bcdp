# bcdp/scripts/sanity_completion_handle.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from bcdp.model.hf_handle import HuggingFaceCausalLMHandle


def load_handle(model_name="google/gemma-2-2b"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    handle = HuggingFaceCausalLMHandle(model=model, tokenizer=tok)
    return tok, handle


@torch.no_grad()
def print_topk(prompt: str, tok, handle: HuggingFaceCausalLMHandle, top_k=10):
    # NOTE: with device_map="auto", moving inputs to handle.device usually works fine
    # (same pattern you used before), and HF dispatches across GPUs.
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(handle.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(handle.device)

    out = handle.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]  # [1,V]
    probs = F.softmax(logits, dim=-1)

    topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)

    print("\n=== PROMPT ===")
    print(prompt)
    print(f"\nTop-{top_k} next tokens:")
    for i in range(top_k):
        tid = topk_ids[0, i].item()
        s = tok.decode([tid])
        print(f"{i+1:2d}. {repr(s):16s} prob={topk_probs[0, i].item():.4f}")


@torch.no_grad()
def print_greedy_completion(prompt: str, tok, handle: HuggingFaceCausalLMHandle, max_new_tokens=10):
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(handle.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(handle.device)

    out_ids = handle.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    print("\n=== GREEDY COMPLETION ===")
    print(tok.decode(out_ids[0]))


@torch.no_grad()
def inspect_specific_tokens(prompt: str, tok, handle: HuggingFaceCausalLMHandle, tokens_to_check):
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(handle.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(handle.device)

    out = handle.forward(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]  # [1,V]
    probs = F.softmax(logits, dim=-1)

    print("\n=== SPECIFIC TOKEN SCORES ===")
    for t in tokens_to_check:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) != 1:
            print(f"{repr(t)} -> not single token (ids={ids})")
            continue
        tid = ids[0]
        print(
            f"{repr(t):10s} | logit={logits[0, tid].item():.3f} | prob={probs[0, tid].item():.4f}"
        )


if __name__ == "__main__":
    tok, handle = load_handle("google/gemma-2-2b")

    prompt = (
        "Facts:\n"
        "Bob is wise.\n"
        "Ali is kind.\n"
        "Question: Who is wise?\n"
        "Answer:"
    )

    print_topk(prompt, tok, handle, top_k=10)
    print_greedy_completion(prompt, tok, handle, max_new_tokens=5)
    inspect_specific_tokens(prompt, tok, handle, tokens_to_check=[" Bob", " Ali"])
