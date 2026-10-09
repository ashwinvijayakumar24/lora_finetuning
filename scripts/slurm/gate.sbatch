#!/bin/bash
# GPU gate: run before any real job (others depend on it with --dependency=afterok).
#   1. the CUDA-only and slow test suites
#   2. a 40-step training smoke on real data (minimal prompt) to measure throughput
#SBATCH --job-name=playparse-gate
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-h200,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source scripts/slurm/_env.sh
mkdir -p results/gate
python -m pytest -q -m "gpu or slow" -rfEs -p no:cacheprovider 2>&1 | tee "results/gate/pytest_${SLURM_JOB_ID}.txt"
python scripts/train.py --config configs/train_default.yaml \
    --set 'train.output_dir="runs/gate_smoke"' train.device=cuda train.max_steps=40 train.epochs=null \
          train.eval_every=1000 train.gen_every=1000 train.save_every=1000 data.train_sample=5000 \
    2>&1 | tee "results/gate/smoke_${SLURM_JOB_ID}.txt"
python -m playparse.eval.run --rung lora --adapter "$(ls -d runs/gate_smoke/checkpoints/step_*/adapter | sort | tail -1)" --prompt-style minimal --data data/processed/eval_lite.jsonl --limit 64 \
    --out "results/gate/eval_smoke_${SLURM_JOB_ID}" --device cuda --batch-size 32 --no-resume
cp runs/gate_smoke/metrics.jsonl "results/gate/smoke_metrics_${SLURM_JOB_ID}.jsonl"
echo GATE_OK
