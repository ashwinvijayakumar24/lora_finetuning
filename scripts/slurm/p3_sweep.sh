#!/bin/bash
# ---------------------------------------------------------------------------
# P3: submit the PRD section 8.4 sweep, one knob at a time, on PACE Phoenix.
#
# Each run = configs/p3_sweep_50k.yaml (the defaults on train_50k) + overrides,
# submitted through scripts/slurm/train_h100.sbatch (fill in its >>> FILL IN <<<
# lines first). Prints the sbatch commands; set SUBMIT=1 to submit them.
#
#   bash scripts/slurm/p3_sweep.sh                 # dry run: print everything
#   SUBMIT=1 bash scripts/slurm/p3_sweep.sh rank   # submit one group
#   SUBMIT=1 bash scripts/slurm/p3_sweep.sh all
#
# Groups: r5 rank alpha targets dropout lr prompt datasize all
# All runs use prompt_style: minimal (see docs/benchmarks/p3-prompt-ablation.md)
# except the one full-prompt control in the 'prompt' group.
# Resubmitting the same command resumes any run that already has checkpoints.
# After training, score the best adapter of each run with the eval harness:
#   python -m playparse.eval.run --rung lora --adapter runs/sweep/<name>/best \
#       --prompt-style minimal --data data/processed/eval_lite.jsonl --out results/p3_sweep/<name>
# (the frozen test set is scored once, for the final R5 only).
# ---------------------------------------------------------------------------
set -euo pipefail

GROUP=${1:-all}
BASE=configs/p3_sweep_50k.yaml
SBATCH=scripts/slurm/train_h100.sbatch
OUT=${OUT:-runs/sweep}

run() {  # run NAME CONFIG [overrides...]
    # DEPENDENCY=afterok:<gate> makes the run wait for the GPU gate.
    # EVAL=1 also queues an eval of the run's best checkpoint on eval_lite once
    # training succeeds (afterok on the training job).
    local name=$1 config=$2
    shift 2
    local dep=()
    [[ -n "${DEPENDENCY:-}" ]] && dep=(--dependency "$DEPENDENCY")
    local cmd=(sbatch --parsable --job-name "pp-$name" ${dep[@]+"${dep[@]}"} "$SBATCH" "$config" "$OUT/$name" "$@")
    printf "%q " "${cmd[@]}"; echo   # copy-pasteable (quotes preserved)
    if [[ "${SUBMIT:-0}" == "1" ]]; then
        local jid
        jid=$("${cmd[@]}")
        echo "  -> train job $jid"
        if [[ "${EVAL:-0}" == "1" ]]; then
            local ejid
            ejid=$(sbatch --parsable --job-name "pp-eval-$name" --dependency "afterok:$jid" \
                scripts/slurm/eval.sbatch "$OUT/$name/best" data/processed/eval_lite.jsonl "results/p3_sweep/$name")
            echo "  -> eval job $ejid"
        fi
    fi
}

want() { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }

# R5: the default config on the full train split (2 epochs, early stop on val EM).
if want r5; then
    run r5_full configs/train_default.yaml
fi

# Rank, alpha = 2r.
if want rank; then
    for r in 2 4 8 16 64; do
        run "rank_r${r}" "$BASE" "lora.r=$r" "lora.alpha=$((2 * r))"
    done
fi

# Alpha: alpha = r and alpha = 2r at r=16; fixed alpha = 16 across ranks.
if want alpha; then
    run alpha_r16_a16 "$BASE" lora.r=16 lora.alpha=16
    for r in 2 4 8 64; do
        run "alpha_fixed16_r${r}" "$BASE" "lora.r=$r" lora.alpha=16
    done
fi

# Targets at r=16, then at a matched trainable-parameter budget (all-linear r=16
# = 11.27M params; per layer a rank-r adapter costs r*(in+out): q,o 4096r; k,v
# 2560r; gate,up,down 10240r).
if want targets; then
    run targets_qv_r16 "$BASE" 'lora.targets=["q_proj","v_proj"]'
    run targets_qkvo_r16 "$BASE" 'lora.targets=["q_proj","k_proj","v_proj","o_proj"]'
    # all-linear r=16 is the base run (rank_r16)
    run targets_qv_matched_r106 "$BASE" 'lora.targets=["q_proj","v_proj"]' lora.r=106 lora.alpha=212
    run targets_qkvo_matched_r53 "$BASE" 'lora.targets=["q_proj","k_proj","v_proj","o_proj"]' lora.r=53 lora.alpha=106
fi

if want dropout; then
    for d in 0.0 0.1; do  # 0.05 is the base run
        run "dropout_${d}" "$BASE" "lora.dropout=$d"
    done
fi

# Full-prompt control: the minimal-prompt finding came from one seed at 1% of
# the data, so the sweep re-checks it at 50k.
if want prompt; then
    run prompt_full_r16 "$BASE" data.prompt_style=full
fi

if want lr; then
    for lr in 5.0e-5 1.0e-4 5.0e-4 1.0e-3; do  # 2e-4 is the base run
        run "lr_${lr}" "$BASE" "train.lr=$lr"
    done
fi

# Data-size curve on the full train split, nested seeded samples (1k inside 5k ...).
# The full-data point is r5_full. Small sets get more epochs' worth of eval cadence.
if want datasize; then
    for n in 1000 5000 20000 100000; do
        every=$(( n < 20000 ? 20 : 100 ))
        run "datasize_${n}" configs/train_default.yaml "data.train_sample=$n" \
            "train.eval_every=$every" "train.gen_every=$((2 * every))" "train.save_every=$((2 * every))"
    done
fi
