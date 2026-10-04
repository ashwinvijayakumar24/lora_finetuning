#!/bin/bash
# P5a — authoritative LoRA serving benchmark (claims L1, L3) and the GPU oracle (L2).
#
# TEMPLATE: fill in the three <PLACEHOLDER> lines (account, QOS, partition/gres)
# before submitting. Everything else is ready to run.
#
# ONE ALLOCATION FOR EVERYTHING. Base, merged and unmerged arms run in the same
# process on the same GPU, interleaved, because a comparison spanning two
# allocations measures node assignment rather than LoRA (the serving layer saw
# the engine move ~79 -> ~60 tok/s across nodes for identical code).
#
# Order:
#   1. GPU oracle tests (tests/test_p5a_real_oracle.py, the -m gpu arm): merged ==
#      unmerged greedy tokens on CUDA fp16, both close to HF fp32 + PEFT.
#   2. Benchmark: base / base_repeat / merged_r16 / base_patched / unmerged_r{8,16,64},
#      batch 1, plus the batch 1/32 linear microbenchmark.
# The L3 threshold (15% decode overhead at r=16, batch 1) is fixed in
# scripts/p5a_bench.py BEFORE this runs; do not change it after seeing results.
#
#SBATCH --job-name=p5a_bench
#SBATCH --account=<PLACEHOLDER_ACCOUNT>
#SBATCH --qos=<PLACEHOLDER_QOS>
#SBATCH --partition=<PLACEHOLDER_PARTITION e.g. gpu-h100>
#SBATCH --gres=gpu:h100:1
# PACE enforces a max 8:1 CPU:GPU ratio on gpu-h100.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/p5a_bench_%j.out
#SBATCH --error=logs/p5a_bench_%j.out

set -euo pipefail
mkdir -p logs results/p5a

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS:-?} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate "${CONDA_ENV:-playparse}"

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
# Both must point at shared copies on the cluster (paths below are examples).
export PLAYPARSE_WEIGHTS="${PLAYPARSE_WEIGHTS:-$HOME/<PLACEHOLDER_STORAGE>/llm_inference_engine/weights}"
export PLAYPARSE_ENGINE_DIR="${PLAYPARSE_ENGINE_DIR:-$HOME/<PLACEHOLDER_STORAGE>/llm_inference_engine}"

python3 - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA")
print(f"  {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}")
PREFLIGHT

echo "############ 1: oracle (CPU fp32 + CUDA fp16) ############"
PLAYPARSE_RECORD=1 python3 -m pytest -m "slow or gpu" tests/test_p5a_real_oracle.py -v
[ -f results/p5a/oracle_local_L16.json ] && mv results/p5a/oracle_local_L16.json "results/p5a/oracle_cuda_${SLURM_JOB_ID}.json"

echo "############ 2: benchmark ############"
python3 scripts/p5a_bench.py --device cuda --max-tokens 128 --n-warmup 1 --n-runs 5 --micro

echo "=== artifacts ==="
ls -1 results/p5a/ | tail -20
echo "=== done ==="
