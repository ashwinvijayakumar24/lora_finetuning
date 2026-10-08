#!/bin/bash
# ---------------------------------------------------------------------------
# PlayParse: one training run on a single GPU (Georgia Tech PACE Phoenix).
#
# QOS embers is free but preemptible; the job resumes from its latest checkpoint
# when Slurm requeues it (preemption or the 2-hour limit), so a run longer than
# one allocation simply spans several. Any of H100 / H200 / A100 / L40S will do
# for training (quality does not depend on the GPU; the GPU is logged).
# A short --time makes the job a backfill candidate on a busy cluster.
#
# Submit:
#   sbatch scripts/slurm/train_h100.sbatch configs/train_default.yaml runs/r5 [k=v ...]
# ---------------------------------------------------------------------------
#SBATCH --job-name=playparse-train
#SBATCH --account=paceship-simpliearn
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100,gpu-h200,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300                   # warn 5 min before the time limit
#SBATCH --requeue                             # allow requeue (embers preemption)

set -euo pipefail

CONFIG=${1:?usage: sbatch train_h100.sbatch CONFIG OUTPUT_DIR [--set k=v ...]}
OUTPUT_DIR=${2:?usage: sbatch train_h100.sbatch CONFIG OUTPUT_DIR [--set k=v ...]}
shift 2
EXTRA=("$@")

STORE=$HOME/ps-simpliearn-0
REPO=${REPO:-$STORE/llm_finetuning}
export PLAYPARSE_WEIGHTS=${PLAYPARSE_WEIGHTS:-$STORE/llm_inference_engine/weights}
export PLAYPARSE_PROCESSED_DIR=${PLAYPARSE_PROCESSED_DIR:-$REPO/data/processed}
export PLAYPARSE_ENGINE_DIR=${PLAYPARSE_ENGINE_DIR:-$STORE/llm_inference_engine}

# Keep Hugging Face from reaching the network or filling $HOME on compute nodes.
export HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-$STORE/.hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

module load cuda/12.9.1
module load anaconda3
conda activate "${CONDA_ENV:-llm}"

cd "$REPO"

echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
echo "git sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

RESUME=()
if compgen -G "$OUTPUT_DIR/checkpoints/step_*/trainer_state.pt" > /dev/null; then
    echo "found checkpoints in $OUTPUT_DIR, resuming from the latest"
    RESUME=(--resume latest)
fi

# On the time-limit warning, requeue the job; it resumes from the last checkpoint.
# Anything after the last save_every boundary is recomputed, so keep save_every modest.
trap 'echo "USR1: requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

python scripts/train.py --config "$CONFIG" "${RESUME[@]}" \
    --set "train.output_dir=\"$OUTPUT_DIR\"" train.device=cuda ${EXTRA[@]+"${EXTRA[@]}"} &
wait $!
