"""A hand-written supervised fine-tuning loop.

What one optimizer step does (with grad_accum_steps = G micro-batches):

    1. Take the next micro_batch_size * G examples from a deterministic stream.
    2. Count N = the total number of target (completion) tokens across all G
       micro-batches, *before* any forward pass.
    3. For each micro-batch: loss_sum = summed cross-entropy over its target
       tokens; backward(loss_sum / N). Gradients add up across micro-batches.
    4. Clip the gradient norm, AdamW step, scheduler step, zero the gradients.

Step 3 is why the gradient equals that of one big batch: the big batch's loss is
(sum of all token losses) / N, and the sum splits cleanly across micro-batches.
The common bug, averaging each micro-batch on its own and then dividing by G,
weights a token in a short micro-batch more than a token in a long one. See
docs/phases/P2.md for a worked example and tests/test_train_loop.py for the test.

The loop is generic over any nn.Module whose forward takes (input_ids,
attention_mask) and returns logits (or an object with .logits). The trainable
parameters are exactly those with requires_grad=True. Saving and loading the
adapter are injectable callables, so the hand-written LoRA (playparse.lora) and
PEFT can both be plugged in.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from playparse.train.collate import IGNORE_INDEX, EncodedExample, collate, count_target_tokens

SaveFn = Callable[[nn.Module, Path], None]
LoadFn = Callable[[nn.Module, Path], None]
ValCallback = Callable[[nn.Module, int], dict]
StopHook = Callable[[int, dict], bool]

CHECKPOINT_DIR = "checkpoints"
TRAINER_STATE = "trainer_state.pt"
ADAPTER_DIR = "adapter"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Every knob of a training run. Load with TrainConfig.from_file(path)."""

    output_dir: str = "runs/default"
    seed: int = 0

    # optimizer
    lr: float = 2e-4
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    max_grad_norm: float | None = 1.0

    # length of training: max_steps (optimizer steps) wins; otherwise epochs
    max_steps: int | None = None
    epochs: float | None = 1.0

    # schedule: linear warmup to lr, cosine decay to lr * min_lr_ratio
    warmup_steps: int = 0
    warmup_ratio: float = 0.0  # used only when warmup_steps == 0
    min_lr_ratio: float = 0.0

    # batching: effective batch = micro_batch_size * grad_accum_steps examples
    micro_batch_size: int = 4
    grad_accum_steps: int = 1
    shuffle: bool = True
    pad_to_multiple_of: int | None = None

    # evaluation and logging (all in optimizer steps; None disables)
    log_every: int = 1
    eval_every: int | None = None
    eval_batch_size: int | None = None  # defaults to micro_batch_size
    eval_max_examples: int | None = None
    gen_every: int | None = None  # how often to call val_callback

    # checkpoints
    save_every: int | None = None
    keep_last_checkpoints: int | None = 2
    save_final: bool = True
    save_best: bool = False  # adapter only, to <output_dir>/best

    # early stopping on a logged eval metric
    early_stop_metric: str = "val_loss"
    early_stop_mode: str = "min"  # "min" or "max"
    early_stop_patience: int | None = None  # evaluations without improvement

    # hardware
    device: str = "auto"  # auto | cuda | mps | cpu
    autocast: str = "auto"  # auto (bf16 on cuda, off elsewhere) | bf16 | fp16 | none
    deterministic: bool = False  # torch.use_deterministic_algorithms on CUDA

    @property
    def examples_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["adam_betas"] = list(self.adam_betas)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        d = dict(d)
        if "adam_betas" in d:
            d["adam_betas"] = tuple(d["adam_betas"])
        return cls(**d)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "TrainConfig":
        """Load from .json, or .yaml/.yml (needs pyyaml). A top-level "train" key is
        accepted so one file can also hold model/LoRA/data sections."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        if "train" in d and isinstance(d["train"], dict):
            d = d["train"]
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Small helpers: device, autocast, seeding, memory
# ---------------------------------------------------------------------------


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, mode: str = "auto"):
    """bf16 autocast on CUDA by default. On MPS and CPU the default is no autocast:
    load the frozen base in bf16 and keep the adapter in fp32 instead (PEFT and
    playparse.lora both run the adapter in its own dtype)."""
    if mode == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else None
    elif mode == "bf16":
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    elif mode == "none":
        dtype = None
    else:
        raise ValueError(f"unknown autocast mode {mode!r}")
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class MemoryTracker:
    """Peak memory, measured the best way each backend allows.

    cuda: torch.cuda.max_memory_allocated (exact peak of tensor allocations).
    mps:  max of torch.mps.driver_allocated_memory() sampled before and after
          every backward (MPS has no peak counter; this is total memory the
          Metal driver holds for the process, including its cache). The max of
          current_allocated_memory() (live tensors only) is kept as peak_alloc.
    cpu:  process peak resident set size (ru_maxrss).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._peak = 0
        self._peak_alloc = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    @property
    def kind(self) -> str:
        return {"cuda": "cuda_max_allocated", "mps": "mps_driver_allocated_max"}.get(self.device.type, "cpu_max_rss")

    def sample(self) -> None:
        if self.device.type == "mps":
            self._peak = max(self._peak, int(torch.mps.driver_allocated_memory()))
            self._peak_alloc = max(self._peak_alloc, int(torch.mps.current_allocated_memory()))

    def peak_alloc_bytes(self) -> int | None:
        """Peak of live tensor memory (cuda exact, mps sampled, cpu unavailable)."""
        if self.device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        if self.device.type == "mps":
            return self._peak_alloc
        return None

    def peak_bytes(self) -> int:
        if self.device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        if self.device.type == "mps":
            self.sample()
            return self._peak
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)


# ---------------------------------------------------------------------------
# Loss and gradient accumulation
# ---------------------------------------------------------------------------


def model_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    if isinstance(out, torch.Tensor):
        return out
    return out.logits


def token_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Summed next-token cross-entropy over target positions.

    Logits at position i predict the token at i + 1, so logits drop the last
    position and labels drop the first. Only positions whose (shifted) label is
    not -100 are scored. Selecting them before upcasting to fp32 keeps the fp32
    copy small: (n_targets x vocab) instead of (batch x seq x vocab).
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(logits.device)
    mask = shift_labels != IGNORE_INDEX
    n = int(mask.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    selected = shift_logits[mask].float()
    return F.cross_entropy(selected, shift_labels[mask], reduction="sum"), n


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def accumulate_gradients(
    model: nn.Module,
    micro_batches: Sequence[dict[str, torch.Tensor]],
    device: torch.device,
    autocast: str = "auto",
    on_micro_batch: Callable[[], None] | None = None,
) -> tuple[float, int]:
    """Forward and backward every micro-batch of one optimizer step.

    Every micro-batch's summed loss is divided by N, the total target-token count
    of the *whole step*, so the accumulated gradient equals the gradient of the
    mean token loss over all G micro-batches together. Returns (mean loss, N).
    Gradients are added to whatever .grad already holds; the caller zeroes them.
    """
    n_total = sum(count_target_tokens(b["labels"]) for b in micro_batches)
    if n_total == 0:
        raise ValueError("optimizer step has no target tokens (all labels are -100)")
    loss_total = 0.0
    for b in micro_batches:
        b = _to_device(b, device)
        with autocast_context(device, autocast):
            logits = model_logits(model, b["input_ids"], b["attention_mask"])
        loss_sum, _ = token_loss_sum(logits, b["labels"])
        if on_micro_batch is not None:
            on_micro_batch()  # activations are at their peak here, just before backward
        (loss_sum / n_total).backward()
        loss_total += float(loss_sum.detach())
        del logits, loss_sum
        if on_micro_batch is not None:
            on_micro_batch()
    return loss_total / n_total, n_total


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    examples: Sequence[EncodedExample],
    device: torch.device,
    batch_size: int,
    pad_id: int,
    autocast: str = "auto",
    pad_to_multiple_of: int | None = None,
) -> tuple[float, int]:
    """Token-weighted mean loss over examples (the same quantity as the train loss)."""
    was_training = model.training
    model.eval()
    total, n_total = 0.0, 0
    try:
        for i in range(0, len(examples), batch_size):
            b = _to_device(collate(examples[i : i + batch_size], pad_id, pad_to_multiple_of), device)
            with autocast_context(device, autocast):
                logits = model_logits(model, b["input_ids"], b["attention_mask"])
            loss_sum, n = token_loss_sum(logits, b["labels"])
            total += float(loss_sum)
            n_total += n
    finally:
        model.train(was_training)
    return (total / n_total if n_total else float("nan")), n_total


# ---------------------------------------------------------------------------
# Data order, schedule, early stopping
# ---------------------------------------------------------------------------


class DataOrder:
    """A deterministic, resumable stream of example indices.

    Epoch e is a permutation drawn from a generator seeded with (seed + e). The
    examples for optimizer step s are positions [s*k, (s+1)*k) of the endless
    concatenation of epochs. Because the order is a pure function of (seed, step),
    resuming at step s needs no saved sampler state.
    """

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        if n <= 0:
            raise ValueError("empty training set")
        self.n, self.seed, self.shuffle = n, seed, shuffle
        self._perms: dict[int, list[int]] = {}

    def _perm(self, epoch: int) -> list[int]:
        if epoch not in self._perms:
            if self.shuffle:
                g = torch.Generator().manual_seed(self.seed + epoch)
                self._perms[epoch] = torch.randperm(self.n, generator=g).tolist()
            else:
                self._perms[epoch] = list(range(self.n))
            for old in [e for e in self._perms if e < epoch - 1]:
                del self._perms[old]
        return self._perms[epoch]

    def indices(self, start: int, count: int) -> list[int]:
        return [self._perm(p // self.n)[p % self.n] for p in range(start, start + count)]


def lr_lambda_factory(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    """Multiplier on the base lr at scheduler step s (s = optimizer steps done).

    Linear warmup from lr/warmup to lr over warmup_steps, then a half cosine from
    lr down to lr * min_lr_ratio at total_steps.
    """

    def f(s: int) -> float:
        if warmup_steps > 0 and s < warmup_steps:
            return (s + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (s - warmup_steps) / decay_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


@dataclass
class EarlyStopping:
    metric: str = "val_loss"
    mode: str = "min"
    patience: int | None = None
    best: float | None = None
    best_step: int | None = None
    bad_evals: int = 0

    def update(self, step: int, metrics: dict) -> tuple[bool, bool]:
        """Returns (improved, should_stop). Ignores metric dicts without our metric."""
        if self.metric not in metrics:
            return False, False
        v = float(metrics[self.metric])
        better = self.best is None or (v < self.best if self.mode == "min" else v > self.best)
        if better:
            self.best, self.best_step, self.bad_evals = v, step, 0
        else:
            self.bad_evals += 1
        stop = self.patience is not None and self.bad_evals >= self.patience
        return better, stop

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def save_trainable_safetensors(model: nn.Module, path: Path) -> None:
    """Default adapter saver: every requires_grad parameter, by its module name."""
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    tensors = {n: p.detach().to("cpu").contiguous() for n, p in trainable_named_parameters(model)}
    save_file(tensors, str(path / "trainable.safetensors"))


def load_trainable_safetensors(model: nn.Module, path: Path) -> None:
    """Default adapter loader: the keys must match the trainable parameters exactly."""
    from safetensors.torch import load_file

    tensors = load_file(str(path / "trainable.safetensors"))
    params = dict(trainable_named_parameters(model))
    if set(tensors) != set(params):
        missing, extra = set(params) - set(tensors), set(tensors) - set(params)
        raise KeyError(f"checkpoint/model mismatch: missing {sorted(missing)[:5]}, unexpected {sorted(extra)[:5]}")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(tensors[n].to(p.device, p.dtype))


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / CHECKPOINT_DIR / f"step_{step:07d}"


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir) / CHECKPOINT_DIR
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("step_") and (p / TRAINER_STATE).exists())


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    extra: dict,
    save_fn: SaveFn,
) -> Path:
    """Write everything needed to continue *exactly*: adapter weights, optimizer
    moments, scheduler position, every RNG state, and the step counter. The write
    goes to a temp directory first and is renamed into place, so a job killed
    mid-save never leaves a half checkpoint that looks complete."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_fn(model, tmp / ADAPTER_DIR)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": get_rng_state(),
            **extra,
        },
        tmp / TRAINER_STATE,
    )
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return path


def load_checkpoint(path: Path, model: nn.Module, load_fn: LoadFn) -> dict:
    """Load adapter weights into model; return the trainer state for the caller to apply."""
    load_fn(model, path / ADAPTER_DIR)
    # weights_only=False: our own file, and it holds Python/numpy RNG state objects.
    return torch.load(path / TRAINER_STATE, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    step: int
    train_loss: float | None
    val_loss: float | None
    best_metric: float | None
    best_step: int | None
    stopped_early: bool
    last_checkpoint: str | None
    history: list[dict] = field(default_factory=list)


class _MetricsLog:
    def __init__(self, path: Path | None):
        self.path = path
        self.records: list[dict] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        record = {"time": round(time.time(), 3), **record}
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")


def train(
    model: nn.Module,
    train_data: Sequence[EncodedExample],
    val_data: Sequence[EncodedExample] | None,
    cfg: TrainConfig,
    *,
    pad_id: int,
    save_fn: SaveFn = save_trainable_safetensors,
    load_fn: LoadFn = load_trainable_safetensors,
    val_callback: ValCallback | None = None,
    should_stop: StopHook | None = None,
    resume_from: str | os.PathLike | None = None,
    stop_after_step: int | None = None,
    log_to_stdout: bool = False,
) -> TrainResult:
    """Train model on train_data and return a TrainResult.

    model         any nn.Module; parameters with requires_grad are trained
    train_data    encoded examples (playparse.train.collate.encode_records)
    val_data      optional; its loss is computed every cfg.eval_every steps
    pad_id        token id used for padding (value is irrelevant, it is masked)
    save_fn/load_fn  adapter (de)serializers used in checkpoints
    val_callback  called as val_callback(model, step) every cfg.gen_every steps,
                  model in eval mode under no_grad; returns metrics to log, e.g.
                  {"val_exact_match": 0.83}. Those metrics feed early stopping.
    should_stop   extra hook: should_stop(step, metrics) -> True ends training.
    resume_from   a checkpoint directory, or "latest" (newest under output_dir)
    stop_after_step  end the run after this step as if interrupted (the schedule
                  still targets the full length). Used by the resume test.

    Writes <output_dir>/metrics.jsonl and checkpoints under <output_dir>/checkpoints.
    """
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)
    log = _MetricsLog(out / "metrics.jsonl" if cfg.output_dir else None)

    seed_everything(cfg.seed, cfg.deterministic)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no parameters with requires_grad=True")
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, betas=tuple(cfg.adam_betas), eps=cfg.adam_eps, weight_decay=cfg.weight_decay
    )

    per_step = cfg.examples_per_step
    if cfg.max_steps is not None:
        total_steps = cfg.max_steps
    elif cfg.epochs is not None:
        total_steps = math.ceil(cfg.epochs * len(train_data) / per_step)
    else:
        raise ValueError("set max_steps or epochs")
    warmup = cfg.warmup_steps or int(round(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(warmup, total_steps, cfg.min_lr_ratio))
    order = DataOrder(len(train_data), cfg.seed, cfg.shuffle)
    early = EarlyStopping(cfg.early_stop_metric, cfg.early_stop_mode, cfg.early_stop_patience)
    eval_bs = cfg.eval_batch_size or cfg.micro_batch_size
    val_subset = list(val_data[: cfg.eval_max_examples] if cfg.eval_max_examples else val_data) if val_data else None

    step = 0
    tokens_seen = 0
    if resume_from is not None:
        ckpt = Path(resume_from)
        if str(resume_from) == "latest":
            found = list_checkpoints(out)
            if not found:
                raise FileNotFoundError(f"no checkpoints under {out / CHECKPOINT_DIR}")
            ckpt = found[-1]
        state = load_checkpoint(ckpt, model, load_fn)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        early.load_state_dict(state.get("early_stopping", {}))
        tokens_seen = state.get("tokens_seen", 0)
        step = int(state["step"])
        set_rng_state(state["rng"])
        log.write({"event": "resume", "step": step, "from": str(ckpt)})

    log.write({"event": "start", "step": step, "total_steps": total_steps, "warmup_steps": warmup,
               "n_train": len(train_data), "n_val": len(val_subset or []),
               "n_trainable_params": sum(p.numel() for p in params), "device": str(device),
               "config": cfg.to_dict()})

    mem = MemoryTracker(device)
    last_train_loss: float | None = None
    last_val_loss: float | None = None
    last_ckpt: str | None = None
    stopped_early = False
    win_loss, win_tokens, win_inputs, win_time, win_steps = 0.0, 0, 0, 0.0, 0
    end_step = total_steps if stop_after_step is None else min(total_steps, stop_after_step)

    def do_save() -> Path:
        p = save_checkpoint(
            checkpoint_dir(out, step), model, optimizer, scheduler, step,
            {"early_stopping": early.state_dict(), "tokens_seen": tokens_seen, "config": cfg.to_dict()},
            save_fn,
        )
        if cfg.keep_last_checkpoints:
            for old in list_checkpoints(out)[: -cfg.keep_last_checkpoints]:
                shutil.rmtree(old)
        log.write({"event": "checkpoint", "step": step, "path": str(p)})
        return p

    while step < end_step:
        t0 = time.perf_counter()
        idx = order.indices(step * per_step, per_step)
        micro = [
            collate([train_data[i] for i in idx[j : j + cfg.micro_batch_size]], pad_id, cfg.pad_to_multiple_of)
            for j in range(0, per_step, cfg.micro_batch_size)
        ]
        n_inputs = sum(int(b["attention_mask"].sum()) for b in micro)
        loss, n_targets = accumulate_gradients(model, micro, device, cfg.autocast, on_micro_batch=mem.sample)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params, cfg.max_grad_norm if cfg.max_grad_norm is not None else float("inf")
        )
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        dt = time.perf_counter() - t0

        step += 1
        tokens_seen += n_inputs
        last_train_loss = loss
        win_loss += loss * n_targets
        win_tokens += n_targets
        win_inputs += n_inputs
        win_time += dt
        win_steps += 1

        if step % cfg.log_every == 0 or step == end_step:
            rec = {
                "event": "train",
                "step": step,
                "epoch": step * per_step / len(train_data),
                "train_loss": win_loss / win_tokens,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "target_tokens": win_tokens,
                "tokens_per_sec": win_inputs / win_time,
                "target_tokens_per_sec": win_tokens / win_time,
                "step_time_s": win_time / win_steps,
                "tokens_seen": tokens_seen,
                "peak_mem_bytes": mem.peak_bytes(),
                "peak_alloc_bytes": mem.peak_alloc_bytes(),
                "mem_kind": mem.kind,
            }
            log.write(rec)
            if log_to_stdout:
                print(f"step {step}/{total_steps} loss {rec['train_loss']:.4f} lr {lr:.2e} "
                      f"tok/s {rec['tokens_per_sec']:.0f} mem {rec['peak_mem_bytes'] / 2**30:.2f}GiB", flush=True)
            win_loss, win_tokens, win_inputs, win_time, win_steps = 0.0, 0, 0, 0.0, 0

        metrics: dict = {}
        if val_subset and cfg.eval_every and (step % cfg.eval_every == 0 or step == total_steps):
            t_eval = time.perf_counter()
            last_val_loss, n_val_tokens = evaluate_loss(
                model, val_subset, device, eval_bs, pad_id, cfg.autocast, cfg.pad_to_multiple_of
            )
            metrics.update({"val_loss": last_val_loss, "val_tokens": n_val_tokens})
            log.write({"event": "eval", "step": step, "val_loss": last_val_loss, "val_tokens": n_val_tokens,
                       "eval_time_s": time.perf_counter() - t_eval})
            if log_to_stdout:
                print(f"step {step} val_loss {last_val_loss:.4f}", flush=True)
        if val_callback is not None and cfg.gen_every and (step % cfg.gen_every == 0 or step == total_steps):
            was_training = model.training
            model.eval()
            with torch.no_grad():
                cb_metrics = dict(val_callback(model, step) or {})
            model.train(was_training)
            metrics.update(cb_metrics)
            log.write({"event": "val_callback", "step": step, **cb_metrics})

        if metrics:
            improved, stop = early.update(step, metrics)
            if improved and cfg.save_best and cfg.output_dir:
                save_fn(model, out / "best")
                log.write({"event": "best", "step": step, cfg.early_stop_metric: early.best})
            if stop:
                stopped_early = True
                log.write({"event": "early_stop", "step": step, "best": early.best, "best_step": early.best_step})
        if should_stop is not None and should_stop(step, metrics):
            stopped_early = True
            log.write({"event": "early_stop", "step": step, "reason": "should_stop hook"})

        is_last = step == end_step or stopped_early
        if cfg.output_dir and ((cfg.save_every and step % cfg.save_every == 0) or (is_last and cfg.save_final)):
            last_ckpt = str(do_save())
        if stopped_early:
            break

    log.write({"event": "end", "step": step, "stopped_early": stopped_early})
    return TrainResult(
        step=step,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        best_metric=early.best,
        best_step=early.best_step,
        stopped_early=stopped_early,
        last_checkpoint=last_ckpt,
        history=log.records,
    )
