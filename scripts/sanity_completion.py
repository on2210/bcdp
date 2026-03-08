import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name="google/gemma-2-2b", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def print_topk(prompt, tokenizer, model, top_k=10):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # next-token logits

    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k=top_k, dim=-1)

    print("\n=== PROMPT ===")
    print(prompt)
    print("\n=== TOP-K NEXT TOKENS ===")

    for i in range(top_k):
        tok_id = topk_ids[0, i].item()
        tok = tokenizer.decode([tok_id])
        print(f"{i+1:2d}. {repr(tok):15s}  prob={topk_probs[0,i].item():.4f}")


@torch.no_grad()
def print_greedy_completion(prompt, tokenizer, model, max_new_tokens=10):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    print("\n=== GREEDY COMPLETION ===")
    print(tokenizer.decode(out_ids[0]))


@torch.no_grad()
def inspect_specific_tokens(prompt, tokenizer, model, tokens_to_check):
    """
    Useful for binding sanity:
    Check logit / prob for specific candidate answers.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    print("\n=== SPECIFIC TOKEN SCORES ===")

    for tok in tokens_to_check:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)
        if len(tok_id) != 1:
            print(f"{repr(tok)} -> not single token (ids={tok_id})")
            continue

        tid = tok_id[0]
        print(
            f"{repr(tok):10s} | "
            f"logit={logits[0, tid].item():.3f} | "
            f"prob={probs[0, tid].item():.4f}"
        )


if __name__ == "__main__":
    tokenizer, model = load_model()

    prompt = (
        "Facts:\n"
        "Bob is wise.\n"
        "Ali is kind.\n"
        "Question: Who is wise?\n"
        "Answer:"
    )

    print_topk(prompt, tokenizer, model, top_k=10)
    print_greedy_completion(prompt, tokenizer, model, max_new_tokens=5)

    # חשוב: לרוב עם רווח מוביל
    inspect_specific_tokens(
        prompt,
        tokenizer,
        model,
        tokens_to_check=[" Bob", " Ali"]
    )
