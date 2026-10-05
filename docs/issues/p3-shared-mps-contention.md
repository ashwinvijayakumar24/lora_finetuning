# P3: two agents' real-model jobs on one 16 GB Mac slowed the pilot up to 20×

## What happened

The pilot training run started at 15:46 after checking that no other heavy
Python process was running. Within the same minute, another agent started a
serving benchmark (`scripts/p5b_bench.py --quick`), and later a second benchmark
and a slow real-model test suite. While they overlapped, macOS swap rose to
9–10 GB of 10 GB, the training process sat in uninterruptible wait, and logged
throughput fell from about 190–270 input tokens/s to 11–40 tokens/s.

## Why it matters

- **Timing numbers are contaminated both ways.** The pilot's tokens/s and the
  other agent's latency benchmarks were each measured while the other job held
  unified memory. Neither is a clean measurement for those windows.
- **Wall time.** The training run took 2.1 hours instead of about 1.3.

## How it was found

The training log showed step 5 → step 10 taking over 5 minutes instead of about
1, and `ps` / `sysctl vm.swapusage` showed the other jobs and a nearly full swap.

## Root cause

Both agents check `ps` before launching, but a check followed by a launch is not
atomic: two jobs that check within the same minute both see an idle machine.
There is no shared lock for the GPU.

## What was done

- The pilot run was paused with `SIGSTOP` for 3 minutes so a benchmark could
  finish, then resumed with `SIGCONT`. The run was resumable from checkpoints
  every 25 steps in any case.
- A watcher reported every new Python process during the remaining runs; all
  pilot evals ran after the other jobs had finished.
- The benchmark doc reports the median over logging windows (233 tokens/s) and
  states the contention explicitly.

## Suggested fix (not implemented; it touches shared tooling)

A simple advisory lock file for real-model jobs on the laptop, for example
`flock ~/.playparse_gpu.lock python ...`, used by every agent's long jobs. A
check-then-launch convention cannot prevent this race.

## Guarding test

None possible in the test suite; this is an operating-procedure issue.
