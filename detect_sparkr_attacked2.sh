#!/bin/bash
# ============================================================================
# detect_sparkr_attacked.sh
#
# Detect every attacked jsonl in outputs/attacked/sparkr/ and print the same
# metric block spanwm_detect_v7.py prints (mean z / mean p / AUROC / TPR@FPR),
# plus a one-line-per-attack summary table at the end.
#
#   positive class = attacked_text   (the attacked watermarked text)
#   negative class = unwatermarked_text  (override with NEGATIVE=natural)
#
# MUST run on a GPU node: SpARKR's greenlist is torch.randperm(..., generator)
# on self.device, and CPU vs CUDA generators give different permutations from
# the same seed -> detecting on CPU what was embedded on GPU silently yields
# AUROC ~ 0.5. The script srun's itself into your standing allocation.
#
# Usage:
#   bash detect_sparkr_attacked.sh              # auto-picks your RUNNING jobid
#   JOBID=1893349 bash detect_sparkr_attacked.sh
#   NEGATIVE=natural bash detect_sparkr_attacked.sh
# ============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=/home/ssgyejin/miniconda3/envs/spanwm/bin/python
ALGORITHM=Adaptive
CONFIG=config/Adaptive.json
COLUMN=attacked_text
# The attack was applied to the FULL watermarked_text (prompt included), so the
# attacked text no longer starts with the prompt and char-stripping would fall
# back to a word-count rule that eats into the generation at high ratios.
# Score raw text on both classes -- same protocol as spanwm_detect_v7.py.
TRUNCATION=none
NEGATIVE="${NEGATIVE:-unwatermarked}"

IN_DIR=outputs/attacked/atw
LOG_DIR="$IN_DIR/logs"

# ---- get onto a GPU ---------------------------------------------------------
if [[ "${SPARKR_ON_GPU:-0}" != "1" ]]; then
  if $PYTHON -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    export SPARKR_ON_GPU=1
  else
    JOBID="${JOBID:-$(squeue -u "$USER" -h -t RUNNING -o '%A' | head -1)}"
    if [[ -z "$JOBID" ]]; then
      echo "no RUNNING slurm job found and no GPU here -- start an allocation" >&2
      exit 1
    fi
    echo "no local GPU; attaching to slurm job $JOBID ..."
    exec srun --jobid="$JOBID" --overlap --ntasks=1 --cpus-per-task=8 \
         --chdir="$PWD" --export=ALL,SPARKR_ON_GPU=1,NEGATIVE="$NEGATIVE" \
         bash "$0"
  fi
fi

mkdir -p "$LOG_DIR"

# ---- run --------------------------------------------------------------------
shopt -s nullglob
files=("$IN_DIR"/*.jsonl)
if (( ${#files[@]} == 0 )); then
  echo "no jsonl under $IN_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  name=$(basename "$f" .jsonl)
  echo
  echo "### $name"
  $PYTHON baseline_detect.py \
      --input "$f" \
      --algorithm "$ALGORITHM" \
      --config "$CONFIG" \
      --column "$COLUMN" \
      --truncation "$TRUNCATION" \
      --negative "$NEGATIVE" \
      2>&1 | tee "$LOG_DIR/$name.$NEGATIVE.log"
done

# ---- summary ----------------------------------------------------------------
echo
echo "======================================================================================"
printf "%-38s %8s %8s %8s %8s %8s\n" "attack" "AUROC" "T@10%" "T@5%" "T@1%" "T@0.1%"
echo "--------------------------------------------------------------------------------------"
for f in "${files[@]}"; do
  name=$(basename "$f" .jsonl)
  log="$LOG_DIR/$name.$NEGATIVE.log"
  [[ -f "$log" ]] || continue
  auroc=$(grep -m1 '^AUROC  ' "$log" | awk '{print $NF}')
  t10=$(grep -m1 'TPR@FPR= 10.0%' "$log" | awk '{print $NF}')
  t5=$(grep -m1 'TPR@FPR=  5.0%' "$log" | awk '{print $NF}')
  t1=$(grep -m1 'TPR@FPR=  1.0%' "$log" | awk '{print $NF}')
  t01=$(grep -m1 'TPR@FPR=  0.1%' "$log" | awk '{print $NF}')
  printf "%-38s %8s %8s %8s %8s %8s\n" \
    "${name#sparkr_softfix_c4_n200_}" "$auroc" "$t10" "$t5" "$t1" "$t01"
done
echo "======================================================================================"
echo "logs        -> $LOG_DIR/"
echo "per-sample  -> $IN_DIR/*.${COLUMN}.${NEGATIVE}.scores.json"
