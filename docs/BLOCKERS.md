# Blockers — things that need Ashwin's input

Each pin names what is blocked, what was built anyway, and exactly what is needed to
unblock it. Work continues on everything not downstream of a pin. A pin becomes a
*real* blocker only when nothing else is left to do without it.

| ID | Status | Needs | Blocks | Built anyway |
|---|---|---|---|---|
| B1 | **Resolved** 2026-10-04 | GitHub re-auth (`gh auth login`) | Pushing commits | — |
| B2 | Pinned | An `ANTHROPIC_API_KEY` exported in the shell that runs jobs | R4 (frontier rung), teacher-model choice and 1k-play pricing pilot, R8–R9 distillation, claim T2, T6 | Frontier client with mocked tests; rejection filter; distillation pipeline tested on synthetic teacher outputs |
| B3 | Pinned | PACE Phoenix GPU access details: Slurm account, QOS/partition, and where to put the repo and weights on the cluster | Full-scale training (R5 on all ~300k plays), sweeps at scale, QLoRA (bitsandbytes needs CUDA), full fine-tune R7, authoritative serving benchmarks (L1, L3–L5, L7) | Every script and Slurm template; local MPS pilot runs at reduced scale, clearly labelled as indicative |

## How to unblock

- **B2:** run `export ANTHROPIC_API_KEY=...` in the terminal before resuming, or put it
  in `llm_finetuning/.env` (gitignored). Nothing is ever committed with the key.
- **B3:** fill in the placeholders in `scripts/slurm/*.sbatch` (account, QOS) or tell
  me the values, and confirm I can reach the cluster from this machine (for example,
  `ssh <user>@login-phoenix.pace.gatech.edu` works non-interactively).
