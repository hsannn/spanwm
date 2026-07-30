#!/bin/bash
# Paraphrase-robustness detection driven by a manifest, so a new batch of
# attacked files is one appended line rather than a new script.
#
#   sbatch scripts/detect_paraphrase_manifest.sbatch <manifest.tsv>
#
# manifest columns (tab separated, '#' comments ignored):
#   relative_path <TAB> ALGORITHM <TAB> config_path <TAB> model_hf_dir
#
# Every run scores `attacked_text` with --truncation none: a paraphrase keeps
# the prompt only sometimes (0-81 of 200 across the attackers we have), so
# stripping would score the attackers on inconsistent footings. --vocab_size
# model keys the greenlist on exactly what generation used.
#SBATCH -J para_detect
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
set -u
MANIFEST=${1:?manifest tsv}
export HF_HOME=/scratch/hyojun/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/spanwm"
PY=$HOME/anaconda3/envs/wmattack/bin/python
cd "$HOME/spanwm" || exit 1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "PREFLIGHT FAIL: CUDA unavailable on $(hostname)"; exit 1; }

grep -v '^\s*#' "$MANIFEST" | grep -v '^\s*$' | while IFS=$'\t' read -r F ALG CFG HFID; do
  M=$(ls -d /scratch/hyojun/hf_cache/hub/${HFID}/snapshots/*/ 2>/dev/null | head -1)
  if [ ! -f "$F" ]; then echo "-- missing file: $F"; continue; fi
  if [ ! -f "$CFG" ]; then echo "-- missing config: $CFG"; continue; fi
  if [ -z "$M" ]; then echo "-- model not cached: $HFID"; continue; fi
  ATK=$("$PY" -c "import json,sys; print(json.loads(open(sys.argv[1]).readline()).get('attack','?'))" "$F")
  echo "################ $(basename "$F") ################"
  echo "==== $ALG | $(basename "$CFG") | attack=$ATK ===="
  "$PY" baseline_detect.py --input "$F" --algorithm "$ALG" --config "$CFG" \
        --model "$M" --negative unwatermarked --column attacked_text \
        --truncation none \
        --scores_out "${F%.jsonl}.attacked.scores.json"
done
echo "== manifest done $(date) =="
