"""R1-R3: Llama 3.2 1B Instruct through Hugging Face transformers.

- R1 zero-shot: the shared system prompt plus the play.
- R2 fixed few-shot: the same 8 committed train examples before every play
  (`fewshot_r2.json`).
- R3 retrieved few-shot: the 8 most similar train plays for each play (`retrieval.py`).

Every rung builds its turns with `playparse.prompt.build_messages`, so the system
prompt and user-turn format are exactly what a fine-tuned adapter will later see.
Few-shot examples are inserted as prior user/assistant turns between the system
prompt and the real play.

Decoding is greedy (PRD §6). Two details matter for reproducibility, and both are
pinned here (see docs/issues/p1-eval-chat-template-date.md and
p1-eval-generation-config-sampling.md):
- The Llama 3.2 chat template stamps today's date into the system prompt unless
  `date_string` is passed, so the prompt would change every day.
- The checkpoint's generation_config.json defaults to sampling (temperature 0.6), so
  greedy decoding has to be requested explicitly.

`prompt_style` ("full" by default, "minimal" = no system message, or "minimal_v2" =
minimal plus the line of scrimmage, whose outputs are schema-v2 field spots; see
`playparse.prompt.PROMPT_STYLES`) must match the style an adapter was trained with.
A schema-v2 output is converted to the v1 label it stands for before the harness
sees it (`HFPredictor._to_v1`), so metrics keep their v1 meaning.
A non-default style is recorded in `config()`, so it changes the config hash; the
"full" config is unchanged from before the option existed, so earlier results and
their resumable prediction caches keep their hashes.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from playparse.eval.harness import Prediction
from playparse.prompt import (
    CHAT_DATE_STRING,
    DEFAULT_PROMPT_STYLE,
    SYSTEM_PROMPT,
    build_messages,
    check_prompt_style,
    schema_of_style,
)

FEWSHOT_R2_PATH = Path(__file__).with_name("fewshot_r2.json")

# Gold labels in train seasons 2015-2022 are p99 88, max 105 tokens compact (see
# docs/benchmarks/p1-local-throughput.md); base models pretty-print with spaces and
# newlines, roughly doubling that. 256 leaves headroom without letting a rambling
# generation run for long.
DEFAULT_MAX_NEW_TOKENS = 256

Example = dict  # {"posteam", "desc", "label"} - label is PlayLabel.to_json()


def load_fewshot(path: str | Path = FEWSHOT_R2_PATH) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def build_fewshot_messages(
    record: dict, examples: Sequence[Example], system: str = SYSTEM_PROMPT,
    style: str = DEFAULT_PROMPT_STYLE,
) -> list[dict[str, str]]:
    """System prompt (style "full" only), then (user play, assistant label) pairs,
    then the real play."""
    query = build_messages(record.get("posteam"), record["desc"], system, style, los=_los(record, style))
    msgs = query[:-1]  # the system message, or nothing for style "minimal"
    for ex in examples:
        msgs.append(build_messages(ex.get("posteam"), ex["desc"], system, style, los=_los(ex, style))[-1])
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append(query[-1])
    return msgs


def _los(record: dict, style: str) -> str | None:
    """The record's line of scrimmage; a schema-v2 style refuses a record without one."""
    if schema_of_style(style) == "v2" and "los" not in record:
        raise KeyError(f"prompt style {style!r} needs records with a 'los' field: score a v2 data file "
                       "(data/processed_v2/*.jsonl)")
    return record.get("los")


def render_chat(tokenizer, msgs: list[dict[str, str]]) -> str:
    """Chat-template text ending with the assistant header, with the date pinned."""
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE_STRING
    )


def render_record(
    tokenizer, record: dict, examples: Sequence[Example] = (), system: str = SYSTEM_PROMPT,
    style: str = DEFAULT_PROMPT_STYLE,
) -> str:
    """The prompt text the harness sends for one record (no model needed).

    With no examples this is exactly the training prompt: tokenized with
    add_special_tokens=False it gives the same ids as
    `playparse.train.collate.encode_prompt` (tests/test_prompt_parity.py).
    """
    return render_chat(tokenizer, build_fewshot_messages(record, examples, system, style))


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFPredictor:
    """Batched greedy generation with a chat-templated prompt.

    `examples_fn(record) -> list[Example]` supplies the few-shot examples per record:
    None for R1, a constant list for R2, a retriever for R3.
    """

    prompt_style = DEFAULT_PROMPT_STYLE

    def __init__(
        self,
        weights: str | Path,
        *,
        name: str = "hf-zero-shot",
        examples_fn: Callable[[dict], Sequence[Example]] | None = None,
        examples_id: str | None = None,
        batch_size: int = 8,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
        dtype: str | None = None,
        system: str = SYSTEM_PROMPT,
        prompt_style: str = DEFAULT_PROMPT_STYLE,
        base_quant: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from playparse.train.build import normalize_quant, quantized_load_kwargs

        self.name = name
        # None: the base in `dtype`. "nf4": the base in 4-bit NF4 (bitsandbytes, CUDA
        # only), as a QLoRA adapter (R6) was trained; compute stays in `dtype`.
        self.base_quant = normalize_quant(base_quant)
        self.weights = str(weights)
        self.examples_fn = examples_fn
        self.examples_id = examples_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.system = system
        self.prompt_style = check_prompt_style(prompt_style)
        self.device = pick_device(device)
        if dtype is None:
            dtype = "float32" if self.device == "cpu" else ("bfloat16" if self.device == "cuda" else "float16")
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.weights)
        self.tokenizer.padding_side = "left"  # decoder-only batching: pad on the left
        if self.tokenizer.pad_token is None:
            # Llama 3.2 ships no pad token. Its reserved fine-tune pad token is never
            # generated, so padding with it cannot be confused with a real <|eot_id|>.
            pad = "<|finetune_right_pad_id|>"
            self.tokenizer.pad_token = pad if pad in self.tokenizer.get_vocab() else self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights, dtype=getattr(torch, dtype), **quantized_load_kwargs(self.base_quant, self.device))
        if self.base_quant is None:
            self.model.to(self.device)  # a 4-bit model is already on its GPU (device_map)
        self.model.eval()
        gc = self.model.generation_config
        self.eos_ids = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]

    @property
    def schema(self) -> str:
        """"v2" for prompt style minimal_v2: outputs are field spots, converted to v1 text."""
        return schema_of_style(self.prompt_style)

    # ------------------------------------------------------------------ harness API

    def config(self) -> dict[str, Any]:
        cfg = {
            "weights": Path(self.weights).name,
            "examples": self.examples_id,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
            # Batched fp16 greedy decoding is not bit-identical across batch sizes
            # (padding changes the kernels' reduction order), so batch size is part of
            # the config. See docs/issues/p1-eval-batch-size-changes-greedy-outputs.md.
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "device": self.device,
            "chat_date_string": CHAT_DATE_STRING,
            "system_prompt_sha": _short_sha(self.system),
        }
        if self.prompt_style != DEFAULT_PROMPT_STYLE:
            # Only non-default styles add keys, so every "full" config (and its
            # hash) is byte-identical to the ones recorded before this option.
            cfg["prompt_style"] = self.prompt_style
            cfg["system_prompt_sha"] = None  # no system message is sent
        if self.schema != "v1":
            cfg["schema"] = self.schema
        if getattr(self, "base_quant", None) is not None:
            # Only a quantized base adds the key, so unquantized configs keep their hashes.
            cfg["base_quant"] = self.base_quant
        return cfg

    def render(self, record: dict) -> str:
        examples = self.examples_fn(record) if self.examples_fn else ()
        return render_record(self.tokenizer, record, examples, self.system, self.prompt_style)

    def predict_batch(self, records: Sequence[dict]) -> list[Prediction]:
        import torch

        texts = [self.render(r) for r in records]
        # The chat template already starts with <|begin_of_text|>; letting the
        # tokenizer add another would duplicate BOS.
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
            )
        prompt_len = enc["input_ids"].shape[1]
        gen = out[:, prompt_len:].cpu()
        in_counts = enc["attention_mask"].sum(dim=1).cpu().tolist()
        results = []
        for i, row in enumerate(gen.tolist()):
            n_out = len(row)
            for j, tok in enumerate(row):
                if tok in self.eos_ids or tok == self.tokenizer.pad_token_id:
                    n_out = j + (1 if tok in self.eos_ids else 0)
                    break
            text = self.tokenizer.decode(row[:n_out], skip_special_tokens=True)
            usage = {
                "input_tokens": int(in_counts[i]),
                "output_tokens": int(n_out),
                "hit_max_new_tokens": int(n_out >= self.max_new_tokens),
            }
            if self.schema == "v2":
                text, usage = self._to_v1(records[i], text, usage)
            results.append(Prediction(text, usage))
        return results

    @staticmethod
    def _to_v1(record: dict, raw: str, usage: dict) -> tuple[str, dict]:
        """A schema-v2 output as the v1 text the unchanged harness scores.

        The model's own text is kept in usage["raw_v2"] (non-numeric, so it is stored
        in predictions.jsonl but never summed). An output that is not valid v2 becomes
        text with no JSON object, so it scores as invalid, never as a lucky v1 label.
        """
        from playparse.ffscore.schema_v2 import v2_text_to_v1_text

        text, err = v2_text_to_v1_text(raw, record.get("los"), record.get("posteam"))
        usage = {**usage, "raw_v2": raw}
        if err is not None:
            usage["v2_error"] = err
        return text, usage


def _short_sha(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- factories


def make_r1(weights: str | Path, **kw) -> HFPredictor:
    return HFPredictor(weights, name="r1-zero-shot", examples_fn=None, examples_id=None, **kw)


def make_r2(weights: str | Path, fewshot_path: str | Path = FEWSHOT_R2_PATH, **kw) -> HFPredictor:
    examples = load_fewshot(fewshot_path)
    blob = json.dumps(examples, sort_keys=True)
    return HFPredictor(
        weights,
        name="r2-fixed-few-shot",
        examples_fn=lambda _rec: examples,
        examples_id=f"fixed:{Path(fewshot_path).name}:{_short_sha(blob)}",
        **kw,
    )


def make_r3(weights: str | Path, retriever, **kw) -> HFPredictor:
    return HFPredictor(
        weights,
        name="r3-retrieved-few-shot",
        examples_fn=retriever.examples_for,
        examples_id=f"retrieved:{retriever.config_id()}",
        **kw,
    )
