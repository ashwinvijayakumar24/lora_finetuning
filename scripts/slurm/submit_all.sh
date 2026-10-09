#!/bin/bash
# Submit the whole GPU plan behind one gate job. Run on the PACE login node with
#   nohup bash scripts/slurm/submit_all.sh > logs/submit_all.log 2>&1 &
# so a dropped SSH connection cannot cut a submission short. Ends with SUBMIT_DONE.
set -u
cd "$(dirname "$0")/../.."
mkdir -p logs
G=$(sbatch --parsable scripts/slurm/gate.sbatch); echo "gate $G"
DEPENDENCY=afterok:$G SUBMIT=1 bash scripts/slurm/p3_sweep.sh all | grep -E "^  ->|^sbatch" | sed 's/^sbatch.*--job-name \(pp-[^ ]*\).*/\1/' | paste - - 
echo "r7 $(sbatch --parsable --dependency afterok:$G --job-name pp-r7_full_ft scripts/slurm/train_h100.sbatch configs/r7_full_ft.yaml runs/r7_full_ft)"
echo "p5a $(sbatch --parsable --dependency afterok:$G scripts/slurm/p5a_bench.sbatch)"
echo "p5b $(sbatch --parsable --dependency afterok:$G scripts/slurm/p5b_bench.sbatch)"
echo SUBMIT_DONE
