#!/bin/bash
# Pull finished results from PACE. A result directory is copied only once its
# result.json exists, so a running eval's partial predictions (which double as its
# resume cache) are never committed and then pulled back over the live file.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-pace:ps-simpliearn-0/llm_finetuning}
ssh -o BatchMode=yes pace 'cd ~/ps-simpliearn-0/llm_finetuning && find results -name result.json -printf "%h\n"; find results -maxdepth 2 -type f ! -path "*/eval_lite/*" ! -path "*/test/*" ! -path "*/t3b_test*" ! -path "*/r0los_test*" -printf "%p\n"' > /tmp/pp_done.txt
rsync -az --files-from=<(grep -v '^results/eval_lite\|^results/test\|t3b_test\|r0los_test' /tmp/pp_done.txt | sed 's#^results/##') "$REMOTE/results/" results/ 2>/dev/null || true
grep -E '^results/(eval_lite|test|t3b_test|r0los_test)' /tmp/pp_done.txt | sort -u | while read -r d; do
    mkdir -p "$d"; rsync -az "$REMOTE/$d/" "$d/"
done
rsync -az --include='metrics.jsonl' --include='run_spec.json' --include='*/' --exclude='*' "$REMOTE/runs/sweep/" results/sweep_runs/
echo synced
