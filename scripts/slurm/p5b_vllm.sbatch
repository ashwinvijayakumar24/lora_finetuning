#!/bin/bash
# P5b — claim L7: our multi-LoRA serving vs vLLM --enable-lora, same workload, ONE allocation.
#
# Runs our goodput arm first (playparse env), then vLLM (vllm env) on the same
# GPU, reusing our frozen SLO thresholds and offered rate via --from-artifact, so
# both systems face identical requests, adapters, arrival times and SLO bounds.
# L7 is earned if our goodput is within L7_FACTOR = 2.0 of vLLM's per cell
# (stated in scripts/p5b_bench.py before this runs).
#
# The vllm env is separate because vLLM pins its own torch:
#   conda create -n vllm python=3.11 -y && conda activate vllm
#   pip install "vllm>=0.6" safetensors numpy && pip install -e . --no-deps
#
#SBATCH --job-name=p5b_vllm
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/p5b_vllm_%j.out
#SBATCH --error=logs/p5b_vllm_%j.out

set -euo pipefail
mkdir -p logs results/p5b
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

module load cuda/12.9.1
module load anaconda3
cd "${SLURM_SUBMIT_DIR}"
export HF_HUB_OFFLINE=1
export PLAYPARSE_WEIGHTS="${PLAYPARSE_WEIGHTS:-$HOME/ps-simpliearn-0/llm_inference_engine/weights}"
export PLAYPARSE_ENGINE_DIR="${PLAYPARSE_ENGINE_DIR:-$HOME/ps-simpliearn-0/llm_inference_engine}"
export PLAYPARSE_SERVING_DIR="${PLAYPARSE_SERVING_DIR:-$HOME/ps-simpliearn-0/llm_serving_layer}"

echo "############ 1: ours (goodput arm only) ############"
conda activate "${CONDA_ENV:-llm}"
python3 scripts/p5b_bench.py --device cuda --model real --arms goodput \
    --ns 1,4,16,64,256 --popularities uniform,zipf --kernels v2 \
    --n-slots 32 --rank 16 --n-requests 512 --max-batch 32 --kv-blocks 8192 --saturation
OURS=$(ls -t results/p5b/bench_cuda_real_*.json | head -1)
echo "ours: ${OURS}"

echo "############ 2: vLLM, same workload, same SLO, same rate ############"
# Use an existing vLLM env read-only; our code is found via PYTHONPATH, never installed into it.
conda activate "${VLLM_ENV:-$HOME/ps-simpliearn-0/envs/vllm}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
python3 scripts/p5b_vllm_bench.py --model-path "${PLAYPARSE_WEIGHTS}" \
    --from-artifact "${OURS}" --ns 1,4,16,64,256 --n-requests 512 --saturation

ls -1 results/p5b/ | tail -10
echo "=== done ==="
