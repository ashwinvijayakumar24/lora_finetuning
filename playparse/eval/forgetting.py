"""T7: general-knowledge retention on a fixed MMLU sample (docs/phases/T7.md).

Each question is asked 0-shot in one user message (Llama 3.2 chat template, pinned
date, no system prompt) ending in "Answer:". The prediction is the letter A-D whose
token the model scores highest as the first token of its reply. Comparing that
accuracy for the base model and the base + adapter measures forgetting.

    python -m playparse.eval.forgetting --adapter none --out results/t7/base
    python -m playparse.eval.forgetting --adapter runs/sweep/r5_full/best --out results/t7/r5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from playparse.paths import REPO_ROOT, WEIGHTS
from playparse.prompt import CHAT_DATE_STRING

LETTERS = "ABCD"
DEFAULT_DATA = REPO_ROOT / "data" / "mmlu_1k.jsonl"


def format_question(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return f"{q['question']}\n{opts}\nAnswer:"


def render(tokenizer, q: dict) -> list[int]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": format_question(q)}],
                                         tokenize=False, add_generation_prompt=True,
                                         date_string=CHAT_DATE_STRING)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def letter_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


@torch.no_grad()
def score(model, tokenizer, questions: list[dict], batch_size: int = 16) -> np.ndarray:
    """Predicted letter index (0-3) for each question."""
    device = next(model.parameters()).device
    cand = torch.tensor(letter_ids(tokenizer), device=device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    preds = []
    for i in range(0, len(questions), batch_size):
        seqs = [render(tokenizer, q) for q in questions[i:i + batch_size]]
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, s in enumerate(seqs):  # left padding: the last position is every row's next token
            ids[j, n - len(s):] = torch.tensor(s)
            mask[j, n - len(s):] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, -1, :]
        preds.append(logits.index_select(-1, cand).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_model(weights: str, adapter: str | None, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=dtype).to(device).eval()
    if adapter:
        from playparse.lora import load_adapter, merge_lora

        load_adapter(model, adapter, adapter_dtype=torch.float32)
        model = merge_lora(model)
    return model, tok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir, or 'none' for the base model")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    adapter = None if args.adapter == "none" else args.adapter
    dtype = torch.float32 if args.device == "cpu" else (torch.float16 if args.device == "mps" else torch.bfloat16)
    qs = [json.loads(l) for l in open(args.data)][: args.limit]
    t0 = time.time()
    model, tok = load_model(args.weights, adapter, args.device, dtype)
    preds = score(model, tok, qs, args.batch_size)
    correct = (preds == np.array([q["answer"] for q in qs])).astype(float)
    lo, hi = bootstrap_ci(correct)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(
        json.dumps({"id": q["id"], "pred": int(p), "answer": q["answer"]}) + "\n" for q, p in zip(qs, preds)))
    res = {"adapter": adapter, "n": len(qs), "accuracy": float(correct.mean()), "ci95": [lo, hi],
           "device": args.device, "dtype": str(dtype), "seconds": round(time.time() - t0, 1)}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
