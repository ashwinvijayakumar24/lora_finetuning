# P2: a tiny random model with LoRA could not memorize 8 examples

## What happened

The overfit test (train a tiny model with LoRA on 8 examples until the loss is
near zero) stalled. The loss dropped from about 3.9 to 3.574 and stayed there
for 200 steps. The gradient norm fell to exactly 0.00. Changing the learning
rate did not help.

## How it was found

`tests/test_train_loop.py::test_overfits_eight_examples` failed on its first run.

## Root cause

The test model is a random-init Llama with hidden size 32 and tied embeddings
(the input embedding matrix is reused as the output head). The default init
gives the embeddings a standard deviation of 0.02. The final RMSNorm (a
normalization layer that rescales each hidden vector to a fixed size) makes the
last hidden vector have length about sqrt(32) ≈ 5.7. Each logit is the dot product
of that vector with one embedding row of length about 0.11, so every logit sits
roughly within ±0.6.

LoRA only adds low-rank updates to the linear layers inside the transformer. It
cannot change the embeddings, the output head, or the norm weights. So no matter
how well the adapter learns, the model cannot make one token much more likely
than the others. The best it can do is match the overall token frequencies,
which is the 3.57 plateau (close to the entropy of the completion tokens).

Two checks confirmed this:

- Full fine-tuning of the same model memorized the data (loss 0.002).
- LoRA on the same model with embeddings rescaled to std 0.5 also memorized it
  (loss 0.002 by step 150).

## Why it matters

This is not a bug in the loop, but it is a trap when testing a training loop on
toy models: a test can "fail to learn" for reasons that have nothing to do with
the code under test. A real pretrained model has large, meaningful embeddings, so
it does not have this floor.

## Fix

`tests/p2_fixtures.py::tiny_llama` takes an `embed_std` argument. The overfit
and generation tests use `embed_std=0.5`.

## Guarding test

`tests/test_train_loop.py::test_overfits_eight_examples` (loss must fall below
0.05) and `tests/test_generate.py::test_trained_model_reproduces_completions_and_stops_on_eot`.

## A related detail

PEFT initializes LoRA's B matrix to zero, so the update starts at exactly zero.
A side effect is that the gradient of A is exactly zero on the first step (A's
gradient is multiplied by B). The gradient-equivalence tests would then compare
zeros to zeros for half the parameters. The test fixture gives B small random
values so every adapter parameter has a non-zero gradient.
