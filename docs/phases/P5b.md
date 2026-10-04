# P5b — Many LoRA adapters in one batch, on the serving layer

**Status:** built and verified locally (Apple M4). Correctness is settled on CPU
and on the real 1B model on MPS. Performance claims L3 (batch-32 arm), L4 and L5
need one H100 run, and L7 needs vLLM on that node; both Slurm jobs are ready.

| Claim | Local result | Status |
|---|---|---|
| L3 (batch-32 arm) unmerged overhead small at low rank | measured end to end at batch 32 through the scheduler; laptop noise (another agent training on the same GPU) exceeds the effect | pending H100 |
| L4 multi-LoRA scales with N | kernel arm: v1 cost grows 10x from 1 to 32 distinct adapters in a batch, v2's is flat; the full goodput grid runs end to end but machine drift makes cells incomparable | pending H100 |
| L5 adapters beat one merged model per tenant | 80 GB GPU, weights only: 31 merged copies vs 3,319 resident r=16 adapters; with 2,048 KV tokens per tenant, 30 vs 834 (from measured bytes) | analytic, from measured bytes; H100 run re-measures |
| L6 prefix cache is adapter-safe | zero cross-adapter hits, outputs equal solo references; the fault-injected token-only key is caught by the same test | earned locally |
| L7 competitive with vLLM multi-LoRA | script written against the same workload module, not run (no CUDA, no vLLM locally) | pending H100 |
| Correctness gate | mixed-adapter batches are token-identical to each request run alone: tiny model (CPU) with chunked prefill, staggered arrival, pool smaller than the adapter set and both preemption policies; real 1B on MPS (16 layers, 12 tokens, v1 and v2) | earned locally; FlashInfer arm pending GPU |

## 1. Why many adapters on one base model matters

A LoRA adapter is small. For Llama 3.2 1B with rank 16 on all seven projections
it is about 22 MB in fp16, against about 2.5 GB for the model itself. If every
tenant (a customer, a task, a registry version) gets its own fine-tune, there
are two ways to serve them:

- **One merged model per tenant.** Fold each adapter into a full copy of the
  weights. Each copy is a normal model with zero extra latency (claim L1), but it
  costs 2.5 GB of GPU memory, and requests for different tenants can never share
  a batch. An 80 GB GPU holds about 30 copies before any KV cache.
- **One base model plus many adapters.** Keep a single copy of the base weights
  and apply each request's adapter on the fly (unmerged mode, P5a). The extra
  memory per tenant is the 22 MB adapter, and requests for different tenants
  can share one batch, so the GPU stays busy even when each tenant's own
  traffic is light.

The second option is what S-LoRA and Punica made practical, and it changes the
economics: on the numbers in §7, an 80 GB GPU can keep about 3,300 rank-16
adapters resident instead of about 30 merged copies (before KV cache, which
narrows the gap; see §7). The price is a per-row low-rank matmul in
every projection, and the engineering question is how cheaply a batch that
mixes many adapters can compute it. That is what this phase builds and measures.

## 2. What was built

All code is in `playparse/serving/`. Neither `llm_serving_layer` nor
`llm_inference_engine` was edited; P5b subclasses and wraps them.

| File | Role |
|---|---|
| `_serving_path.py` | Imports the serving layer's `serving` and `bench` packages by file location (never via `sys.path`, see [issue](../issues/p5b-serving-root-shadows-tests.md)), and checks that exactly one `engine` with the batched seam is imported. |
| `adapter_pool.py` | `AdapterPool`: fixed device slots, host registry, LRU eviction, refcounts, pins, stats. The v1/v2 kernels and the per-forward `BatchPlan`. `synthetic_adapter` for scale tests. |
| `multi_lora.py` | `PooledLoRALinear` (the weight-dict wrapper), the `linear()` hook, and `MultiLoRAModelGPU`, whose `forward_varlen` requires an explicit per-row selection. |
| `adapter_scheduler.py` | `AdapterRequest` (`adapter_id`) and `AdapterScheduler`: the serving layer's continuous-batching scheduler with adapter-aware admission, slot lifetime and per-step routing. |
| `adapter_radix.py` | `AdapterRadixCache`: the serving layer's radix prefix cache with adapter-keyed namespaces and a cross-adapter audit. |
| `p5b_workload.py` | The multi-tenant workload (prompts, lengths, arrivals, tenant per request with uniform or Zipf popularity) and goodput scoring, shared by our benchmark and the vLLM one. |
| `_provenance.py` | Git SHAs of all three repositories and hardware for every artifact. |

### Public API

```python
from playparse.serving.adapter_pool import AdapterPool
from playparse.serving.multi_lora import MultiLoRAModelGPU
from playparse.serving.adapter_scheduler import AdapterScheduler, AdapterRequest
from playparse.serving.adapter_radix import AdapterRadixCache
from serving.memory.allocator import BlockAllocator           # the serving layer's own
from serving.backends.paged_torch import PagedTorchBackend    # or FlashInferBackend
from serving.scheduler.scheduler import SchedulerConfig

pool = AdapterPool(config, n_slots=32, max_rank=64, device="cuda:0", kernel="v2")
pool.register(load_peft_adapter("runs/playparse-r16/adapter", base_config=config), "playparse-v3")
pool.pin("playparse-v3")                              # optional: never evicted

model = MultiLoRAModelGPU(weights, config, pool, device="cuda:0")
alloc = BlockAllocator(num_blocks, 16)
sched = AdapterScheduler(model, PagedTorchBackend(...), alloc, SchedulerConfig(max_batch_size=32),
                         prefix_cache=AdapterRadixCache(alloc, block_copy=backend.copy_block))
sched.add_request(AdapterRequest(request_id="r1", prompt_ids=ids, max_tokens=64,
                                 adapter_id="playparse-v3"))     # None = base model
sched.run_until_idle()          # or sched.step() from the server's event loop
```

## 3. Request-level routing: how an adapter reaches a matmul

Each request carries `adapter_id`. The scheduler's policy (admission order,
chunked prefill, preemption, batch composition) is unchanged and knows nothing
about adapters; batches simply mix them. Three things connect a request's
adapter to the rows of the packed batch:

1. **Admission takes a slot.** When the scheduler admits a request, the request
   takes a reference on its adapter's pool slot, loading the adapter if it is not
   resident. The reference is held while the request runs (and while it is
   swapped out under preemption) and released when it finishes, is cancelled,
   or is recompute-preempted back to the queue.
2. **Per-sequence slots become per-row slots.** The serving layer packs all
   sequences' new tokens along one axis, and `BatchMeta.batch_indices[t]` says
   which sequence owns token row `t`. So the row slots are one gather:

   ```
   seq_slots     = [3, -1, 7]           # one per sequence; -1 = base model
   query_lens    = [4,  1, 1]           # a 4-token prefill chunk, then two decodes
   batch_indices = [0,0,0,0, 1, 2]
   row_slots     = seq_slots[batch_indices] = [3,3,3,3, -1, 7]
   ```

3. **The selection is passed explicitly on every forward call.**
   `MultiLoRAModelGPU.forward_varlen(tokens, meta, backend, adapter=selection)`
   has no default for `adapter`. Inside the call the selection reaches the
   engine's `linear()` through P5a's context variable (the engine's `linear(x, w)`
   has no parameter for it), set and reset around exactly that one forward pass.

The upstream `step()` builds the batch and calls the model in one method, so
there is no clean place to hand it per-sequence data. P5b records each
sequence's slot in `_tokens_for` (called once per scheduled sequence, in batch
order) and checks the record against `BatchMeta` on every step; a mismatch
raises instead of misrouting. The one-method upstream change that removes this
is in [p5b-scheduler-forward-hook](../issues/p5b-scheduler-forward-hook.md).

## 4. The adapter pool, and why it is paged KV for adapters

A server may know 256 tenants but afford GPU memory for 32 of their adapters.
The pool solves the same problem the paged KV cache solves for attention state:

| | paged KV cache (serving layer) | adapter pool (P5b) |
|---|---|---|
| fixed resource | physical KV blocks | device adapter slots |
| who holds it | a running sequence holds refs on its blocks | a running request holds a ref on its slot |
| reusable when | refcount is 0 | refcount is 0 and not pinned |
| none free | admission waits (or preempts) | admission waits |
| eviction | LRU over cached prefix blocks | LRU over idle slots |
| backing store | swap to host memory | every adapter always has a host copy |

**Requests wait, they are not rejected.** If a request's adapter is not resident
and every slot is in use, admission returns "not yet" exactly as it does when KV
blocks are short; the upstream head-of-line rule keeps it at the front of the
queue until a running request finishes and frees a slot. An adapter that was
never registered is a client error and is rejected at `add_request`.

**Refcounts make eviction safe.** A slot is only overwritten when no admitted
request references it. Without that, a long-running request could have its
adapter swapped for another tenant's in the middle of generation, and its text
would silently change. As a second line of defence, every step checks that each
request's slot still holds the adapter it asked for.

**Every load zero-fills the slot first.** An adapter that targets only `q_proj`
and `v_proj`, loaded into a slot whose previous tenant targeted all seven
projections, would otherwise inherit that tenant's MLP matrices.

**Statistics:** hits, misses, evictions, loads, load time and bytes, and the
number of scheduler steps in which a request waited for a slot.

## 5. The kernels: v1 loop, v2 BGMV + segmented prefill

For one projection and a batch of `T` rows, the low-rank term is
`delta[t] = (x[t] @ A[s].T) @ B[s].T` with `s = row_slots[t]`.

**The stacked layout.** For each projection the pool holds
`A: (n_slots + 1, max_rank, in)` and `B: (n_slots + 1, out, max_rank)`, the
same `(r, in)` / `(out, r)` layout PEFT uses, stacked by slot. Ranks below
`max_rank` are zero-padded (zeros add nothing), the LoRA scale is folded into
`B`, and the extra last slot is all zeros so base-model rows can go through the
same code and come out as exact zeros.

**v1: loop over distinct adapters.** For each adapter present in the batch,
gather its rows, do two small matmuls, scatter back. Simple, identical to P5a's
math (bit for bit, tested), and the correctness reference. Its cost grows
linearly with the number of distinct adapters in the batch: 32 distinct adapters
means 64 small matmuls per projection.

**v2: BGMV for decode rows, one matmul per prefill segment.** BGMV ("batched
gather matrix-vector", from Punica) handles a decode batch, where every row may
use a different adapter, in a fixed number of operations:

```
a  = A[row_slots]               # (T, R, in)    gather each row's A
b  = B[row_slots]               # (T, out, R)   gather each row's B
xa = bmm(x[:, None, :], a^T)    # (T, 1, R)     shrink
y  = bmm(xa, b^T)               # (T, 1, out)   expand
```

Its cost does not depend on how many distinct adapters there are. It does
depend on rows: it copies `R x (in + out)` weights per row. For a decode batch
that is the price of mixing adapters. For a prefill chunk, whose hundreds of
rows all share one adapter, it would copy the same weights hundreds of times,
so v2 handles each prefill chunk with one ordinary matmul pair instead (the
segmented-GMV idea, SGMV, that Punica uses for prefill). Rows are processed in
chunks that keep the gathered buffer under 64 MB.

**The batch plan.** Which rows are decode rows, which are prefill segments, and
which rows belong to which adapter is known to the scheduler in plain Python.
`BatchPlan` turns it into index tensors once per forward pass, so neither kernel
synchronises with the host inside the 7-per-layer projection loop. Discovering
this structure on the device in every projection is what made the first
version of both kernels slow
([issue](../issues/p5b-kernel-costs-not-about-adapters.md)).

**v3 (stretch, not built):** a fused Triton/CUDA BGMV kernel that reads each
row's `A`/`B` directly from the stacks without materialising the gather.

## 6. Why the prefix cache must be keyed by adapter (L6)

The serving layer's radix cache reuses KV blocks between requests that share a
prompt prefix, keyed by token ids. With adapters that key is wrong. The K and
V vectors stored for a token depend on everything the adapter changes: directly
through `k_proj` and `v_proj`, and through every hidden state from layer 1
onward even if the adapter only touches the MLP. So "same tokens" does not mean
"same KV" across adapters, or between an adapter and the base model.

If tenant B were handed the blocks tenant A computed for the same system
prompt, B's output would be fluent and plausible and wrong. Nothing would raise,
no metric would move, and the prefix cache would even report a better hit rate.

`AdapterRadixCache` gives each adapter its own key namespace. The trie compares
token ids and never uses them for anything else, so token `t` under adapter
number `c` is keyed as `t + c * 2**32`: disjoint ranges per adapter, raw ids for
the base model (so base-only traffic behaves exactly like the upstream cache),
and one global LRU over all namespaces because they share one block allocator.
The scheduler's three cache calls run inside `cache.scope(request.adapter_id)`,
and a call outside any scope raises rather than guessing.

Independently of the key, the cache records which adapter produced each cached
block and checks every block it hands out against the requesting adapter. That
audit counter is the L6 measurement.

**The evidence:** same prompt under two adapters gives no cross hit; same adapter
hits all 4 full blocks; base vs adapter gives no cross hit; interleaved traffic
over shared prompts gives cache-on == cache-off == solo outputs with real hits
and zero cross-adapter hits. And the **fault injection**: the same workload with
the token-only key (the bug) makes the output comparison fail for the second
adapter and the audit counter go positive. A test that could not fail on the bug
would not be evidence that the bug is absent.

## 7. Measured locally vs pending GPU

Full local write-up with tables:
[docs/benchmarks/p5b-local.md](../benchmarks/p5b-local.md).

Local runs used a 1B-shaped random model with 4 of 16 layers on a shared
laptop GPU (another agent was training throughout). They are indicative only.

- **Correctness (earned locally):** batched mixed-adapter greedy output equals
  each request run alone, on the tiny model (19 scheduler scenarios, both
  kernels, both preemption policies) and on the real 1B on MPS (6 requests, 3
  adapters at r = 8/16/64 plus base, 12 tokens, v1 and v2:
  `results/p5b/gate_real_mps_L16.json`).
- **Kernel arm (L4 mechanism, clear):** at batch 32, r=16, v1 grows from
  11.9 ms to 120 ms as the batch goes from 1 to 32 distinct adapters; v2 stays at
  33 ms. Crossover between 4 and 16 distinct adapters. v2 costs more with
  `max_rank` (90 ms at r=64) because it materialises gathered weights.
- **decode32 (L3 batch-32 arm, unresolved):** one shared adapter with v1 is
  within ±10% of base, below this machine's resolution; v2 is 15–97% slower in
  that case. With 32 distinct adapters v2 is 25% faster per step than v1.
- **Goodput grid (L4, runs end to end, not interpretable):** machine speed
  drifted about 2x during the run. Pool hit rates behave as expected (uniform
  N >= 64: 0–2%; Zipf: 42–52%). One cell (N=256, uniform, v1) collapsed while
  v2 met the SLO on the same requests, a single-sample lead consistent with the
  PRD's prediction.
- **Capacity (L5, from measured bytes):** on an 80 GB GPU, 31 merged copies vs
  3,319 resident r=16 adapters (weights only); with 2,048 KV tokens per tenant,
  30 vs 834.

**Pending GPU (one allocation each, templates ready):**

- `sbatch scripts/slurm/p5b_bench.sbatch` — the real-1B gate on CUDA with
  PagedTorch and FlashInfer backends (`REQUIRE_GPU=1`), then the full benchmark:
  kernel arm, decode32, the N = 1/4/16/64/256 x uniform/Zipf x v1/v2 goodput grid
  with 512 requests per cell and a 32-slot pool, and capacity re-measured on
  the H100.
- `sbatch scripts/slurm/p5b_vllm.sbatch` — our goodput arm, then
  `scripts/p5b_vllm_bench.py` (vLLM `enable_lora`, same requests, same adapter
  tensors, same arrival times, same frozen SLO and offered rate).
- Thresholds stated before those runs, in `scripts/p5b_bench.py`: L3 overhead at
  r=16, batch 32 at most 15%; L4 v2 goodput at N=256 (uniform) at least 50% of
  N=1; L7 our goodput within 2x of vLLM's.

## 8. Tests

| Suite | Count | What it proves |
|---|---|---|
| `test_p5b_pool.py` (fast) | 27 | v2 == v1 == formula on mixed ranks and base rows, fp32 and fp16, any chunking; pool v1 bit-identical to P5a's `LoRALinear`; planned kernels == unplanned; LRU, refcount, pin, wait, zero-fill, unregister, memory accounting; model-level mixed batch == P5a per request; explicit selection required; misaligned rows raise; wrapper composition with P5a. |
| `test_p5b_scheduler.py` (fast) | 19 | The gate: batched == solo for both kernels, with a pool smaller than the adapter set, chunked prefill, staggered arrivals, RECOMPUTE and SWAP preemption; positive checks that waits, evictions and preemptions happened; leak checks; misrouting raises. |
| `test_p5b_radix.py` (fast) | 9 | L6 at cache level and end to end, including the fault injection. |
| `test_p5b_workload.py` (fast) | 7 | Workload is seeded and Zipf-skewed; SLO refuses impossible anchors; goodput counts only requests within both bounds; exported PEFT adapters are the benchmark's tensors. |
| `test_p5b_real.py` | 3 slow + 2 `gpu` | Real 1B: batched == solo for v1 and v2; v2 logits within fp16 noise of v1; CUDA arm with PagedTorch and FlashInfer. |

```bash
pytest                                                   # default suite, fast
pytest -m slow tests/test_p5b_real.py -v                 # real 1B (MPS if available)
PLAYPARSE_P5B_LAYERS=4 pytest -m slow tests/test_p5b_real.py   # low-memory mode
REQUIRE_GPU=1 pytest -m "slow or gpu" tests/test_p5b_real.py   # CUDA node
```

## 9. Serving-layer limitations met on the way

- **No forward hook in `Scheduler.step()`** — worked around with a checked
  record; [upstream patch proposed](../issues/p5b-scheduler-forward-hook.md).
- **Head-of-line admission.** A request waiting for an adapter slot blocks the
  queue behind it, as a request waiting for KV blocks does. That is the
  upstream fairness rule and P5b keeps it; letting resident-adapter requests pass
  a blocked one would raise utilisation and needs a starvation bound.
- **`PagedTorchBackend.attend` loops over sequences in Python.** On a laptop
  that loop, not LoRA, is a large share of every decode step; FlashInfer removes
  it on GPU.
- **The engine's `linear()` has no module identity**
  ([P5a issue](../issues/p5a-linear-no-module-identity.md)), so adapters ride on
  wrapper objects and wrappers stack; one stacking bug was found and fixed
  ([issue](../issues/p5b-linear-wrapper-cycle.md)).

## 10. Follow-ups

- **Adaptive kernel choice.** The kernel arm shows v1 is cheaper with few
  distinct adapters in a batch and v2 with many (crossover 4–16 locally). The
  `BatchPlan` already knows the count, so picking per forward pass is a few
  lines; measure the crossover on the H100 first.
- **v3: a fused BGMV kernel** (Triton or CUDA) that indexes the stacks in place,
  removing v2's gather cost and its growth with `max_rank`.
- **Rank buckets.** One pool padded to r=64 makes every r=8 adapter pay r=64
  in v2. Separate pools (or stacks) per rank bucket avoid that.
- **Admission that does not block behind a cold adapter**, with a starvation
  bound (§9).
- **Prefetch on arrival.** Start the host-to-device copy when a request enters
  the queue rather than when it is admitted.
- **Upstream hooks:** `Scheduler._forward(batch, tokens, meta)` in the serving
  layer and a named `linear()` hook in the engine would remove both workarounds.

## 11. Issues found

- [Two linear() wrappers re-installed in turn called each other forever](../issues/p5b-linear-wrapper-cycle.md)
- [Putting the serving layer on sys.path would hijack `import tests`](../issues/p5b-serving-root-shadows-tests.md)
- [The scheduler has no hook between batch and forward pass](../issues/p5b-scheduler-forward-hook.md)
- [The first kernels were slow for reasons unrelated to adapters](../issues/p5b-kernel-costs-not-about-adapters.md)
- [The first SLO calibration measured a negative TTFT](../issues/p5b-calibration-negative-ttft.md)
