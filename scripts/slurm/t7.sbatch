#!/bin/bash
# T7 forgetting check (docs/phases/T7.md): base vs base + R5 vs base + T3b on data/mmlu_1k.jsonl.
#   sbatch scripts/slurm/t7.sbatch
#SBATCH --job-name=playparse-t7
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-h200,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
source scripts/slurm/_env.sh
python -m playparse.eval.forgetting --adapter none --out results/t7/base --batch-size 32
python -m playparse.eval.forgetting --adapter runs/sweep/r5_full/best --out results/t7/r5 --batch-size 32
python -m playparse.eval.forgetting --adapter runs/t3b_spots/best --out results/t7/t3b --batch-size 32
