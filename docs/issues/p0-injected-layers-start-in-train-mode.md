# P0: injected LoRA layers started in train mode inside an eval() model

## What happened

`test_lora_state_dict_roundtrip` failed. It injects adapters into two copies of a
tiny model, copies the adapter weights from one to the other, and expects the two
models to give identical logits. They differed, even though every weight was equal.

## How it was found

By the round-trip test itself. The test helper `tiny_llama()` returns the model
already in `.eval()` mode, and the config used the default `dropout=0.05`. Two
forward passes through the *same* model also disagreed, which pointed at
randomness rather than at the weights.

## Root cause

`nn.Module.__init__` sets `self.training = True`. `inject_lora` builds a new
`LoRALinear` for each target and attaches it to a model that is in eval mode, but
attaching a module does not change its mode. So every adapter's dropout was active
while the rest of the model was in eval mode. Calling `model.eval()` *after*
injection would have hidden the problem. Calling it *before* injection, which is
the natural order when you load a model for evaluation, did not.

PEFT avoids the issue differently: `get_peft_model` leaves the whole model in
train mode, so there is no mismatch, but also no eval behaviour until you call
`.eval()` yourself.

## Why it matters

Silent dropout at evaluation time makes eval numbers noisy and slightly wrong. It
would also break the P5a claim that merged and unmerged adapters give
bit-identical output, because the unmerged path would be random.

## Fix

`inject_lora` now calls `wrapped.train(base.training)`, so each adapter inherits
the train/eval mode of the layer it replaces.

## Guarding test

`tests/test_lora_inject.py::test_inject_into_eval_model_keeps_dropout_off`
injects with `dropout=0.5` into an eval model and asserts that every adapter is in
eval mode and that two forward passes are identical.
