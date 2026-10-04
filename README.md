# llm_finetuning — PlayParse (planning stage)

The third layer of the `custom_llm` stack:

```
llm_inference_engine   → runs Llama 3.2 1B, one request at a time   (built)
llm_serving_layer      → serves it: paged KV, batching, prefix cache (built)
llm_finetuning         → specializes it, then serves many specializations at once (this)
```

**PlayParse** fine-tunes Llama 3.2 1B to read raw NFL play-by-play text and extract
fantasy-relevant stats. A deterministic scorer turns those stats into exact fantasy
points. nflverse data provides exact ground truth for free, so every result is an
exact match, with no LLM judge involved.

The project measures two things:

1. **When fine-tuning is worth it.** A decision ladder compares regex, prompting,
   retrieval, a frontier model, LoRA, QLoRA, full fine-tuning, and distillation.
2. **How to serve it.** LoRA is added to the from-scratch engine and serving layer,
   including many adapters batched together on one base model.

## Docs

- [`PRD.md`](PRD.md): the full spec covering task, data, eval, ladder, training,
  serving, claims, and phases.
- [`LEARNING_MAP.md`](LEARNING_MAP.md): every item on the LoRA learning list, mapped
  to the phase and artifact that teaches it.

## Status

Planning. The next step is **P0**: a hand-written `LoRALinear`, verified against
Hugging Face PEFT.
