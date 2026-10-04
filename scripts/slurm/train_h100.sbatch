#!/bin/bash
# ---------------------------------------------------------------------------
# PlayParse: one LoRA training run on a single H100 (Georgia Tech PACE Phoenix).
#
# TEMPLATE. Before the first submit, replace every line marked  >>> FILL IN <<<.
#   - account:  your PACE charge account (e.g. gts-<pi-username>);  `pace-quota` lists them
#   - qos:      "inferno" (paid, preemptible=no) or "embers" (free, preemptible) on Phoenix
#   - paths:    REPO / VENV / PLAYPARSE_WEIGHTS / PLAYPARSE_PROCESSED_DIR
# Check partition and GPU names against current PACE docs: `sinfo -o "%P %G"`.
#
# Submit:
#   sbatch scripts/slurm/train_h100.sbatch configs/train_default.yaml runs/r5 [--set k=v ...]
# Resume after preemption or timeout (same command; it resumes if checkpoints exist):
#   sbatch scripts/slurm/train_h100.sbatch configs/train_default.yaml runs/r5
# ---------------------------------------------------------------------------
#SBATCH --job-name=playparse-train
#SBATCH --account=REPLACE_ME_ACCOUNT          # >>> FILL IN <<<
#SBATCH --qos=REPLACE_ME_QOS                  # >>> FILL IN <<< (inferno | embers)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1                     # >>> CHECK <<< GPU type string on Phoenix
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:USR1@300                   # warn 5 min before the time limit
#SBATCH --requeue                             # allow requeue (embers preemption)

set -euo pipefail

CONFIG=${1:?usage: sbatch train_h100.sbatch CONFIG OUTPUT_DIR [--set k=v ...]}
OUTPUT_DIR=${2:?usage: sbatch train_h100.sbatch CONFIG OUTPUT_DIR [--set k=v ...]}
shift 2
EXTRA=("$@")

REPO=${REPO:-$HOME/llm_finetuning}                                    # >>> FILL IN <<<
VENV=${VENV:-$REPO/.venv}                                             # >>> FILL IN <<<
export PLAYPARSE_WEIGHTS=${PLAYPARSE_WEIGHTS:-$HOME/scratch/llama-3.2-1b-instruct}   # >>> FILL IN <<<
export PLAYPARSE_PROCESSED_DIR=${PLAYPARSE_PROCESSED_DIR:-$REPO/data/processed}     # >>> FILL IN <<<

# Keep Hugging Face from reaching the network or filling $HOME on compute nodes.
export HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/scratch/hf_home}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# module load cuda/12.x    # >>> CHECK <<< usually unnecessary with pip-installed torch wheels

cd "$REPO"
source "$VENV/bin/activate"

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
    --set "train.output_dir=\"$OUTPUT_DIR\"" train.device=cuda "${EXTRA[@]}" &
wait $!
