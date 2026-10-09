#!/bin/bash
# Score one trained adapter (or a full fine-tuned model) with the eval harness.
#   sbatch scripts/slurm/eval.sbatch ADAPTER_DIR DATA_JSONL OUT_DIR [extra eval.run args...]
# Example:
#   sbatch scripts/slurm/eval.sbatch runs/r5/best data/processed/eval_lite.jsonl results/r5_eval_lite
# A run trained with impl=full is scored with: --rung r1 --weights <dir> (pass as extra args
# after setting ADAPTER_DIR to "-").
# T3b (schema v2, field spots): score the v2 file and pass the style and schema, e.g.
#   sbatch scripts/slurm/eval.sbatch runs/t3b_spots/best data/processed_v2/test.jsonl results/t3b_test \
#       --prompt-style minimal_v2 --schema v2
# and the R0 + line-of-scrimmage arm (CPU-only, but runs fine here):
#   sbatch scripts/slurm/eval.sbatch - data/processed_v2/test.jsonl results/r0los_test --rung r0los
#SBATCH --job-name=playparse-eval
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-h200,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
set -euo pipefail
ADAPTER=${1:?adapter dir or -}; DATA=${2:?data jsonl}; OUT=${3:?out dir}; shift 3
source scripts/slurm/_env.sh
if [[ "$ADAPTER" == "-" ]]; then
    python -m playparse.eval.run --data "$DATA" --out "$OUT" --device cuda --batch-size 32 "$@"
else
    # The eval CLI refuses a prompt-style mismatch: pass --prompt-style minimal for runs
    # trained with the minimal prompt (all GPU configs).
    python -m playparse.eval.run --rung lora --adapter "$ADAPTER" --data "$DATA" --out "$OUT" \
        --device cuda --batch-size 32 "$@"
fi
