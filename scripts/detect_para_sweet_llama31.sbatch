#!/bin/bash
# Paraphrase robustness for SWEET / Llama-3.1-8B / C4 (clean baseline: 0.9973).
#
# Positives are `attacked_text`. --truncation none keeps positives and negatives
# on the same footing: a paraphrase retains the prompt only sometimes (measured
# 3-81 of 200 depending on the paraphraser), so stripping would treat the three
# attacks inconsistently. This matches the earlier Qwen paraphrase batch.
#SBATCH -J para_sweet_l31
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
set -u
export HF_HOME=/scratch/hyojun/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/spanwm"
PY=$HOME/anaconda3/envs/wmattack/bin/python
cd "$HOME/spanwm" || exit 1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "PREFLIGHT FAIL: CUDA unavailable on $(hostname)"; exit 1; }
M=$(ls -d /scratch/hyojun/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/*/ | head -1)
CFG=config/generated/SWEET_llama3.1-8b_c4.json
[ -f "$CFG" ] || { echo "MISSING $CFG"; exit 1; }
echo "config: $(cat $CFG)"
D=outputs/paraphrase/c4_sweet_llama31
for ATK in GPTParaphrase Paraphrase_gemma-4-12B-it Paraphrase_gpt-oss-20b; do
  F="$D/sweet_tau_llama3.1-8b_c4_n200_${ATK}.jsonl"
  [ -f "$F" ] || { echo "-- missing $F"; continue; }
  echo "################ attack: $ATK ################"
  echo "==== sweet / SWEET / c4 / $ATK ===="
  "$PY" baseline_detect.py --input "$F" --algorithm SWEET --config "$CFG" \
        --model "$M" --negative unwatermarked --column attacked_text \
        --truncation none \
        --scores_out "${F%.jsonl}.attacked.scores.json"
done
echo "== done $(date) =="
