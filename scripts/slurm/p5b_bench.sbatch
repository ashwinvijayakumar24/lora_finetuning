#!/bin/bash
# P5b — multi-LoRA serving: correctness gate on CUDA (incl. FlashInfer) and the
# authoritative benchmark for claims L3 (batch-32 arm), L4 and L5.
#
# TEMPLATE: fill in the <PLACEHOLDER> lines (account, QOS, partition, storage)
# before submitting. Everything else is ready to run.
#
# ONE ALLOCATION FOR EVERYTHING. All arms (base, v1, v2, every N and popularity)
# run in the same process on the same GPU, because a comparison spanning two
# allocations measures node assignment rather than the kernel (the serving layer
# saw the engine move ~79 -> ~60 tok/s across nodes for identical code).
#
# Order:
#   1. Gate: tests/test_p5b_real.py -m "slow or gpu" — batched mixed-adapter greedy
#      tokens == each request alone, real 1B, PagedTorch + FlashInfer, v1 + v2.
#      REQUIRE_GPU=1 makes an unusable GPU a hard failure, not a silent skip.
#   2. Benchmark: scripts/p5b_bench.py --model real, all four arms, full settings.
# Thresholds (L3 15% at r=16 batch 32; L4 v2 retention >= 50% at N=256 uniform;
# L7 factor 2) are fixed in scripts/p5b_bench.py BEFORE this runs.
#
#SBATCH --job-name=p5b_bench
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/p5b_bench_%j.out
#SBATCH --error=logs/p5b_bench_%j.out

set -euo pipefail
mkdir -p logs results/p5b

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID}  qos: ${SLURM_JOB_QOS:-?} ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

module load cuda/12.9.1
module load anaconda3
conda activate "${CONDA_ENV:-llm}"

cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export PLAYPARSE_WEIGHTS="${PLAYPARSE_WEIGHTS:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PLAYPARSE_ENGINE_DIR="${PLAYPARSE_ENGINE_DIR:-$HOME/ps-simpliearn-0/llm_inference_engine}"
export PLAYPARSE_SERVING_DIR="${PLAYPARSE_SERVING_DIR:-$HOME/ps-simpliearn-0/llm_serving_layer}"
export PLAYPARSE_TEST_DEVICE=cuda:0

python3 - <<'PREFLIGHT'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA")
print(f"  {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}")
try:
    import flashinfer
    print(f"  flashinfer {flashinfer.__version__}")
except ImportError:
    print("  WARNING: flashinfer not installed; the FlashInfer arm of the gate will FAIL under REQUIRE_GPU=1")
PREFLIGHT

echo "############ 1: gate (real 1B, CUDA, PagedTorch + FlashInfer) ############"
REQUIRE_GPU=1 PLAYPARSE_RECORD=1 python3 -m pytest -m "slow or gpu" tests/test_p5b_real.py -v

echo "############ 2: benchmark (L3 batch-32, L4, L5) ############"
python3 scripts/p5b_bench.py --device cuda --model real \
    --arms kernel,decode32,goodput,capacity \
    --ns 1,4,16,64,256 --popularities uniform,zipf --kernels v1,v2 --ranks 8,16,64 \
    --n-slots 32 --rank 16 --n-requests 512 --max-batch 32 --kv-blocks 8192 \
    --iters 50 --decode-steps 50 --saturation

echo "=== artifacts ==="
ls -1 results/p5b/ | tail -20
echo "=== done ==="
