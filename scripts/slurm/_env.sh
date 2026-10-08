# Shared PACE environment for PlayParse jobs (sourced by the sbatch scripts).
STORE=$HOME/ps-simpliearn-0
export REPO=${REPO:-$STORE/llm_finetuning}
export PLAYPARSE_WEIGHTS=${PLAYPARSE_WEIGHTS:-$STORE/llm_inference_engine/weights}
export PLAYPARSE_PROCESSED_DIR=${PLAYPARSE_PROCESSED_DIR:-$REPO/data/processed}
export PLAYPARSE_ENGINE_DIR=${PLAYPARSE_ENGINE_DIR:-$STORE/llm_inference_engine}
export PLAYPARSE_SERVING_DIR=${PLAYPARSE_SERVING_DIR:-$STORE/llm_serving_layer}
export HF_HUB_OFFLINE=1 HF_HOME=${HF_HOME:-$STORE/.hf_cache} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
module load cuda/12.9.1
module load anaconda3
conda activate "${CONDA_ENV:-llm}"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is); git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
