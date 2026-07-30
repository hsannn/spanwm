#!/bin/bash
# Detection on paraphrase-attacked Qwen3-4B outputs.
#
# Positives are `attacked_text` (the rewritten text), not the generation, so
# every run passes --column attacked_text and --truncation none: a paraphrase
# no longer begins with the prompt (measured: 0/35 for CNN-DM, 15/200 for C4),
# and the word-level fallback would chop its opening words as if they were.
#
# vocab_size: baseline_embed.py keys the greenlist on model.config.vocab_size,
# which on Qwen (151936) differs from len(tokenizer) (151669) -- the historical
# detection default. These runs pass so detection keys on
# exactly what generation used.
#
# SpanWM v8 files were generated with Qwen/Qwen3-4B (instruct) while only
# Qwen3-4B-Base is cached here. Detection for SpanWM is model-free (tokenizer +
# vocab only) and the two share a tokenizer and vocab_size (151936, recorded in
# the files themselves), so the Base snapshot supplies an equivalent tokenizer.
#SBATCH -J para_detect
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
set -u
export HF_HOME=/scratch/hyojun/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HOME/spanwm"
PY=$HOME/anaconda3/envs/wmattack/bin/python
cd "$HOME/spanwm" || exit 1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "PREFLIGHT FAIL: CUDA unavailable on $(hostname)"; exit 1; }
Q=$(ls -d /scratch/hyojun/hf_cache/hub/models--Qwen--Qwen3-4B-Base/snapshots/*/ | head -1)
P=outputs/paraphrase

for ATK in GPTParaphrase Paraphrase_gemma-4-12B-it Paraphrase_gpt-oss-20b; do
  echo "################ attack: $ATK ################"

  for SPEC in "sparkr:SpARKR:config/SpARKR_softfix.json:$P/daily_mail/sparkr/sparkr_softfix_qwen3-4b_cnn_dailymail_n200_${ATK}.jsonl" \
              "sweet:SWEET:config/generated/SWEET_qwen3-4b_cnn_dailymail.json:$P/daily_mail/sweet/sweet_tau_qwen3-4b_cnn_dailymail_n200_${ATK}.jsonl"; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; ALG=${REST%%:*}; REST=${REST#*:}
    CFG=${REST%%:*}; F=${REST#*:}
    [ -f "$F" ] || { echo "-- missing $F"; continue; }
    echo "==== $NAME / $ALG / cnn_dailymail / $ATK ===="
    "$PY" baseline_detect.py --input "$F" --algorithm "$ALG" --config "$CFG" \
          --model "$Q" --negative unwatermarked --column attacked_text \
          --truncation none \
          --scores_out "${F%.jsonl}.attacked.scores.json"
  done

  for DS in c4 wmt16; do
    case $DS in c4) F=$P/c4/spanwm/spanwm_v8_Qwen3-4B_c4_n200_${ATK}.jsonl ;;
                wmt16) F=$P/wm16/spanwm/spanwm_v8_Qwen3-4B_wmt16_n200_${ATK}.jsonl ;; esac
    [ -f "$F" ] || { echo "-- missing $F"; continue; }
    echo "==== spanwm_v8 / $DS / $ATK ===="
    "$PY" spanwm_detect_v8.py --input "$F" --model "$Q" \
          --config config/SpanWM_v8.json --column attacked_text \
          --negative unwatermarked
  done
done
echo "== paraphrase detection done $(date) =="
