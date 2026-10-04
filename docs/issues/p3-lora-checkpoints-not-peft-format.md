# P3: training with playparse.lora saved checkpoints nobody downstream could load

## What happened

With `lora.impl` resolving to `playparse` (the hand-written LoRA from P0), the
training loop's checkpoints held a generic `trainable.safetensors`: every
`requires_grad` tensor under its in-model name. That file is not PEFT format. The
serving layer (`playparse.serving.adapter.load_peft_adapter`), vLLM, and
`PeftModel.from_pretrained` all expect `adapter_config.json` plus
`adapter_model.safetensors` with `base_model.model.` key prefixes.

## Why it matters

A trained adapter is only useful if serving (P5a/P5b), the registry (P6), and the
vLLM comparison can load it. With the old default, every run would have needed a
conversion step that did not exist. The rank and alpha were also not stored with
the weights, so a loader would have had to guess the scaling factor.

## How it was found

P2's "open items" listed it, and reading `build.apply_lora` confirmed the
fallback to `save_trainable_safetensors`.

## Root cause

P2 was written before P0 merged, so it used a generic writer as a placeholder.

## Fix

`playparse` is now the default implementation. `build.playparse_adapter_io`
provides `save_fn = playparse.lora.io.save_adapter` (PEFT format, fp32 tensors)
and a `load_fn` that fills the already-injected layers on resume. The load
function first checks that the saved rank and alpha match the run, so a
checkpoint from a different configuration fails loudly instead of loading at the
wrong scale.

Field names were checked against both consumers: `LoRAConfig.to_peft_dict`
writes only keys that the serving validator lists as benign (it rejects unknown
truthy keys), with `lora_alpha` as an int when it is whole and `r` as an int.

## Guarding tests

`tests/test_p3_adapter_io.py` trains a tiny run, then checks:

- the checkpoint directory holds exactly the two PEFT files, with PEFT key names;
- `playparse.lora.load_adapter` and `PeftModel.from_pretrained` both reproduce
  the trained model's logits;
- `load_peft_adapter` (serving) reads the same A and B matrices and scale;
- a mismatched rank is rejected;
- resume with the new IO is bit-exact.

`tests/test_train_cli.py` runs the CLI with the default implementation and checks
for `adapter_config.json` in the checkpoint.
